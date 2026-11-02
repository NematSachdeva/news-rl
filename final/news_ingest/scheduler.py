from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from config import Settings
from database import db_connect, get_meta, set_meta, prune_expired_articles
from news_ingest.storage import NewsStorage

logger = logging.getLogger(__name__)

_SCHEDULER_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()
_INGEST_LOCK = threading.Lock()
_IS_REFRESH_RUNNING = False
_LAST_ERROR: Optional[str] = None


def get_last_successful_ingest_at(db_path: Path) -> datetime | None:
    """Retrieve the UTC timestamp of the last successful ingestion from database."""
    val = get_meta("last_successful_ingest_at", default=None, db_path=db_path)
    if not val:
        return None
    try:
        # Support ISO formats like 2026-11-03T01:00:00Z or 2026-11-03T01:00:00+00:00
        clean_val = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception as exc:
        logger.warning("Failed to parse last_successful_ingest_at ('%s'): %s", val, exc)
        return None


def record_successful_ingest(db_path: Path, count: int) -> None:
    """Record a successful ingestion timestamp in system_meta."""
    now_iso = datetime.now(timezone.utc).isoformat()
    set_meta("last_successful_ingest_at", now_iso, db_path=db_path)
    set_meta("last_ingest_error", "", db_path=db_path)
    set_meta("last_ingest_inserted_count", str(count), db_path=db_path)


def record_ingest_error(db_path: Path, error_msg: str) -> None:
    """Record ingestion failure in system_meta without updating last_successful_ingest_at."""
    global _LAST_ERROR
    _LAST_ERROR = error_msg
    set_meta("last_ingest_error", error_msg, db_path=db_path)


def is_refresh_due(
    db_path: Path,
    interval_hours: float = 24.0,
    now_dt: datetime | None = None,
) -> tuple[bool, str]:
    """
    Evaluates whether news ingestion should run based on catalog state and elapsed time.
    Handles clock anomalies (future timestamps, missing timestamps) safely.
    """
    now = now_dt or datetime.now(timezone.utc)

    # 1. If catalog has zero real articles, refresh is immediately due
    storage = NewsStorage(db_path)
    real_count = storage.count_real_articles()
    if real_count == 0:
        return True, "No real news articles found in catalog"

    # 2. Check last recorded successful ingestion
    last_ingest = get_last_successful_ingest_at(db_path)
    if last_ingest is None:
        return True, "No previous successful ingestion recorded"

    # 3. Check for clock anomalies (system clock moved backwards)
    elapsed_seconds = (now - last_ingest).total_seconds()
    if elapsed_seconds < -3600.0:  # More than 1 hour into the future
        logger.warning(
            "Clock anomaly detected: last_successful_ingest_at (%s) is in the future relative to system clock (%s).",
            last_ingest.isoformat(),
            now.isoformat(),
        )
        # Avoid tight crash loops; treat safely by deferring refresh until clock stabilizes
        return False, f"Clock anomaly: timestamp is {abs(elapsed_seconds) / 3600.0:.1f}h in future"

    # 4. Normal interval check
    required_seconds = max(1.0, interval_hours * 3600.0)
    if elapsed_seconds >= required_seconds:
        return True, f"Catalog is {elapsed_seconds / 3600.0:.1f}h old (threshold: {interval_hours:.1f}h)"

    return False, f"Catalog is fresh: last updated {max(0.0, elapsed_seconds / 3600.0):.1f}h ago"


def trigger_refresh_job(settings: Settings, force: bool = False) -> dict[str, Any]:
    """
    Executes a news ingestion job if due or forced.
    Enforces concurrency lock, updates timestamps on success, and logs failures.
    """
    global _IS_REFRESH_RUNNING, _LAST_ERROR
    if not _INGEST_LOCK.acquire(blocking=False):
        logger.info("News refresh job already in progress; skipping duplicate trigger.")
        return {"status": "already_running", "refreshed": False}

    _IS_REFRESH_RUNNING = True
    try:
        if not force:
            due, reason = is_refresh_due(
                settings.db_path,
                interval_hours=getattr(settings, "refresh_interval_hours", 24.0),
            )
            if not due:
                logger.info("News refresh not due: %s", reason)
                return {"status": "not_due", "reason": reason, "refreshed": False}

        api_key = getattr(settings, "newsdata_api_key", "").strip()
        if not api_key:
            logger.warning(
                "NEWSDATA_API_KEY is not set. Cannot fetch real news. "
                "Set NEWSDATA_API_KEY in .env and run 'make ingest'."
            )
            return {"status": "missing_api_key", "refreshed": False}

        logger.info("Executing real news refresh (force=%s)...", force)
        from news_ingest.run import run_ingest

        countries = [c.strip() for c in getattr(settings, "news_ingest_countries", "in,us,gb").split(",") if c.strip()]
        categories = [c.strip() for c in getattr(settings, "news_ingest_categories", "").split(",") if c.strip()]
        languages = [l.strip() for l in getattr(settings, "news_ingest_languages", "en").split(",") if l.strip()]

        counts = run_ingest(
            provider_name="newsdata",
            countries=countries,
            categories=categories,
            languages=languages,
            since=getattr(settings, "news_ingest_since", "48h"),
            max_articles=60,
            refit_nlp=True,
        )

        # Successful ingest: record timestamp and prune expired articles
        record_successful_ingest(settings.db_path, counts["inserted"])
        _LAST_ERROR = None

        # Prune articles older than max_article_age_days (never deleting articles with user interactions)
        max_age = getattr(settings, "max_article_age_days", 14)
        pruned_count = prune_expired_articles(settings.db_path, max_age_days=max_age)
        if pruned_count > 0:
            logger.info("Pruned %d expired articles older than %d days.", pruned_count, max_age)
            from recommender import NewsRecommender
            NewsRecommender(settings.db_path).refit_nlp()

        return {"status": "success", "counts": counts, "pruned": pruned_count, "refreshed": True}
    except Exception as exc:
        err_msg = str(exc)
        logger.error("News refresh job failed: %s", err_msg)
        record_ingest_error(settings.db_path, err_msg)
        return {"status": "error", "error": err_msg, "refreshed": False}
    finally:
        _IS_REFRESH_RUNNING = False
        _INGEST_LOCK.release()


def _scheduler_loop(settings: Settings) -> None:
    """
    Background worker loop. Re-checks about once an hour (3600s) to see if
    catalog is older than 24 hours, triggering refresh if due.
    """
    check_interval = 3600.0  # check once an hour
    logger.info("Periodic news refresh background worker started (hourly check cycle).")

    while not _STOP_EVENT.is_set():
        if _STOP_EVENT.wait(timeout=check_interval):
            break

        try:
            due, reason = is_refresh_due(
                settings.db_path,
                interval_hours=getattr(settings, "refresh_interval_hours", 24.0),
            )
            if due:
                logger.info("Scheduled refresh due (%s); starting background ingestion.", reason)
                trigger_refresh_job(settings, force=False)
            else:
                logger.debug("Scheduled refresh check: not due (%s).", reason)
        except Exception as exc:
            logger.warning("Error during periodic refresh check: %s", exc)

    logger.info("Periodic news refresh background worker stopped.")


def start_ingest_scheduler(settings: Settings) -> None:
    """Starts the background worker thread for hourly refresh checks."""
    global _SCHEDULER_THREAD
    if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
        return

    _STOP_EVENT.clear()
    _SCHEDULER_THREAD = threading.Thread(
        target=_scheduler_loop,
        args=(settings,),
        daemon=True,
        name="news-refresh-scheduler",
    )
    _SCHEDULER_THREAD.start()


def stop_ingest_scheduler() -> None:
    """Signals and stops the background worker thread."""
    global _SCHEDULER_THREAD
    _STOP_EVENT.set()
    if _SCHEDULER_THREAD:
        _SCHEDULER_THREAD.join(timeout=3.0)
        _SCHEDULER_THREAD = None


def startup_refresh_if_due(settings: Settings) -> None:
    """
    Called on server startup in an executor pool so the server starts immediately.
    Triggers an ingestion if there are no real articles or if > 24 hours have elapsed.
    """
    due, reason = is_refresh_due(
        settings.db_path,
        interval_hours=getattr(settings, "refresh_interval_hours", 24.0),
    )
    if due:
        logger.info("Server startup check: refresh is due (%s). Running background ingest...", reason)
        trigger_refresh_job(settings, force=False)
    else:
        logger.info("Server startup check: catalog is up to date (%s).", reason)


def get_ingest_status(settings: Settings) -> dict[str, Any]:
    """
    Computes status summary for GET /news/status.
    Reports last updated time, next refresh due, article counts per category,
    and whether a refresh is actively running.
    """
    last_dt = get_last_successful_ingest_at(settings.db_path)
    now = datetime.now(timezone.utc)
    interval_hours = getattr(settings, "refresh_interval_hours", 24.0)

    if last_dt:
        elapsed_sec = max(0.0, (now - last_dt).total_seconds())
        next_due_dt = last_dt + timedelta(hours=interval_hours)
        if elapsed_sec < 60:
            human_ago = "just now"
        elif elapsed_sec < 3600:
            human_ago = f"{int(elapsed_sec // 60)}m ago"
        elif elapsed_sec < 86400:
            human_ago = f"{int(elapsed_sec // 3600)}h ago"
        else:
            human_ago = f"{int(elapsed_sec // 86400)}d ago"
        last_iso = last_dt.isoformat()
        next_due_iso = next_due_dt.isoformat()
    else:
        human_ago = "never"
        last_iso = None
        next_due_iso = now.isoformat()

    # Query counts per category from SQLite
    cat_counts: dict[str, int] = {c: 0 for c in settings.categories}
    total_real = 0
    with db_connect(settings.db_path) as conn:
        rows = conn.execute(
            """
            SELECT category, COUNT(*) as cnt
            FROM news
            WHERE is_live = 0
            GROUP BY category
            """
        ).fetchall()
        for r in rows:
            cat = r["category"]
            cnt = int(r["cnt"])
            cat_counts[cat] = cnt
            total_real += cnt

    return {
        "status": "ok",
        "is_refresh_running": _IS_REFRESH_RUNNING,
        "last_successful_ingest_at": last_iso,
        "last_updated_human": human_ago,
        "next_refresh_due": next_due_iso,
        "refresh_interval_hours": interval_hours,
        "feed_recency_days": getattr(settings, "feed_recency_days", 3),
        "total_articles": total_real,
        "categories": cat_counts,
        "last_error": _LAST_ERROR,
    }
