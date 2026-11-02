from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from config import Settings
from news_ingest.storage import NewsStorage

logger = logging.getLogger(__name__)

_SCHEDULER_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()


def first_run_ingest_if_needed(settings: Settings) -> None:
    """
    On server startup, checks if real news articles are present in the catalog.
    If real news count is 0 and a NewsData API key is present, runs an initial
    ingestion batch. If no API key is present, logs helpful guidance.
    """
    api_key = getattr(settings, "newsdata_api_key", "")
    storage = NewsStorage(settings.db_path)
    real_count = storage.count_real_articles()

    if real_count > 0:
        logger.info("Found %d real news articles in catalog; skipping initial auto-ingest.", real_count)
        return

    if not api_key:
        logger.info(
            "No real news articles found in catalog and NEWSDATA_API_KEY is not set in .env. "
            "Using synthetic fallback articles. To enable real news, add NEWSDATA_API_KEY=your_key to .env."
        )
        return

    logger.info("First run detected with active API key: running initial real news ingestion...")
    try:
        from news_ingest.run import run_ingest

        countries = [c.strip() for c in getattr(settings, "news_ingest_countries", "in,us,gb").split(",") if c.strip()]
        categories = [c.strip() for c in getattr(settings, "news_ingest_categories", "top,world,technology,business,sports").split(",") if c.strip()]
        run_ingest(
            provider_name="newsdata",
            countries=countries,
            categories=categories,
            max_articles=30,
            refit_nlp=True,
        )
    except Exception as exc:
        logger.warning("Initial auto-ingestion encountered an issue: %s. Continuing with existing data.", exc)


def _scheduler_loop(interval_seconds: float, settings: Settings) -> None:
    logger.info("News ingestion background scheduler started (interval: %.1fh)", interval_seconds / 3600.0)
    while not _STOP_EVENT.is_set():
        # Wait for interval or stop event
        if _STOP_EVENT.wait(timeout=interval_seconds):
            break

        logger.info("Scheduled ingestion trigger running...")
        try:
            from news_ingest.run import run_ingest

            countries = [c.strip() for c in getattr(settings, "news_ingest_countries", "in,us,gb").split(",") if c.strip()]
            categories = [c.strip() for c in getattr(settings, "news_ingest_categories", "top,world,technology,business,sports").split(",") if c.strip()]
            run_ingest(
                provider_name="newsdata",
                countries=countries,
                categories=categories,
                max_articles=30,
                refit_nlp=True,
            )
        except Exception as exc:
            logger.warning("Scheduled news ingestion error: %s", exc)

    logger.info("News ingestion background scheduler stopped.")


def start_ingest_scheduler(settings: Settings) -> None:
    global _SCHEDULER_THREAD
    interval_hours = getattr(settings, "news_ingest_interval_hours", 0.0)
    if interval_hours <= 0:
        logger.info("Periodic news ingestion scheduler disabled (NEUROFEED_NEWS_INGEST_INTERVAL_HOURS=0)")
        return

    if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
        return

    _STOP_EVENT.clear()
    interval_seconds = interval_hours * 3600.0
    _SCHEDULER_THREAD = threading.Thread(
        target=_scheduler_loop,
        args=(interval_seconds, settings),
        daemon=True,
        name="news-ingest-scheduler",
    )
    _SCHEDULER_THREAD.start()


def stop_ingest_scheduler() -> None:
    global _SCHEDULER_THREAD
    _STOP_EVENT.set()
    if _SCHEDULER_THREAD:
        _SCHEDULER_THREAD.join(timeout=3.0)
        _SCHEDULER_THREAD = None

