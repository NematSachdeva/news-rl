"""
final/tests/test_refresh_lifecycle.py — Lifecycle and edge case tests for 24-hour news refresh.

Covers:
  (a) 25h old triggers background refresh on startup without blocking.
  (b) 2h old skips ingestion.
  (c) empty/no timestamp in system_meta triggers refresh.
  (d) future timestamp (clock anomaly) is deferred safely without crashing.
  (e) failed refresh keeps old timestamp in system_meta and records error for backoff.
  (f) multiple days skipped prefers fresh articles in 3-day feed window.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import get_settings
from database import init_db, db_connect, set_meta
from news_ingest.scheduler import (
    is_refresh_due,
    get_last_successful_ingest_at,
    record_successful_ingest,
    record_ingest_error,
    trigger_refresh_job,
    startup_refresh_if_due,
    get_ingest_status,
)
from news_ingest.storage import NewsStorage


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "refresh_test.sqlite"
    init_db(db_file)
    # Populate at least one real article with nd_ prefix so real_count != 0
    with db_connect(db_file) as conn:
        conn.execute(
            """
            INSERT INTO news (news_id, category, subcategory, abstract, url, is_live, published_at, mapping_confidence)
            VALUES (?, ?, ?, ?, ?, 0, ?, 1.0)
            """,
            ("nd_art_1", "technology", "tech", "Initial real article", "https://example.com/1", "2026-11-01T12:00:00Z"),
        )
    return db_file


def test_25h_old_triggers_refresh_and_startup_nonblocking(temp_db: Path):
    """(a) When timestamp is 25h old, is_refresh_due returns True, and startup_refresh_if_due runs non-blocking."""
    now = datetime.now(timezone.utc)
    ts_25h_ago = (now - timedelta(hours=25.0)).isoformat()
    set_meta("last_successful_ingest_at", ts_25h_ago, db_path=temp_db)

    due, reason = is_refresh_due(temp_db, interval_hours=24.0, now_dt=now)
    assert due is True
    assert "25.0h old" in reason

    settings = get_settings().model_copy(update={"db_path": temp_db, "newsdata_api_key": "test_key", "env": "development"})

    # Verify startup_refresh_if_due invokes trigger_refresh_job without hanging
    with patch("news_ingest.scheduler.trigger_refresh_job") as mock_trigger:
        startup_refresh_if_due(settings)
        mock_trigger.assert_called_once_with(settings, force=False)


def test_2h_old_skips_refresh(temp_db: Path):
    """(b) When timestamp is 2h old, is_refresh_due returns False and trigger_refresh_job skips."""
    now = datetime.now(timezone.utc)
    ts_2h_ago = (now - timedelta(hours=2.0)).isoformat()
    set_meta("last_successful_ingest_at", ts_2h_ago, db_path=temp_db)

    due, reason = is_refresh_due(temp_db, interval_hours=24.0, now_dt=now)
    assert due is False
    assert "Catalog is fresh" in reason

    settings = get_settings().model_copy(update={"db_path": temp_db, "newsdata_api_key": "test_key", "env": "development"})
    res = trigger_refresh_job(settings, force=False)
    assert res["status"] == "not_due"
    assert res["refreshed"] is False


def test_empty_timestamp_triggers_refresh(temp_db: Path):
    """(c) When last_successful_ingest_at is empty or missing, refresh is immediately due."""
    last_ts = get_last_successful_ingest_at(temp_db)
    assert last_ts is None

    due, reason = is_refresh_due(temp_db, interval_hours=24.0)
    assert due is True
    assert "No previous successful ingestion recorded" in reason


def test_future_timestamp_deferred_safely(temp_db: Path):
    """(d) When system clock anomaly puts timestamp in the future (>1h), refresh is deferred."""
    now = datetime.now(timezone.utc)
    future_ts = (now + timedelta(hours=5.0)).isoformat()
    set_meta("last_successful_ingest_at", future_ts, db_path=temp_db)

    due, reason = is_refresh_due(temp_db, interval_hours=24.0, now_dt=now)
    assert due is False
    assert "Clock anomaly" in reason


def test_failed_refresh_preserves_old_timestamp_and_records_error(temp_db: Path):
    """(e) Ingestion failure leaves last_successful_ingest_at untouched, records error, and keeps catalog serving."""
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(hours=30.0)).isoformat()
    set_meta("last_successful_ingest_at", old_ts, db_path=temp_db)

    settings = get_settings().model_copy(update={"db_path": temp_db, "newsdata_api_key": "test_key", "env": "development"})

    # Simulate failing ingestion
    with patch("news_ingest.run.run_ingest", side_effect=RuntimeError("NewsData API HTTP 503 Service Unavailable")):
        res = trigger_refresh_job(settings, force=True)
        assert res["status"] == "error"
        assert res["refreshed"] is False
        assert "503" in res["error"]

    # Verify old timestamp is preserved intact
    persisted_ts = get_last_successful_ingest_at(temp_db)
    assert persisted_ts is not None
    assert persisted_ts.isoformat() == old_ts.replace("Z", "+00:00")

    # Verify error is recorded in status
    status = get_ingest_status(settings)
    assert status["last_error"] is not None
    assert "503" in status["last_error"]
    assert status["total_articles"] == 1  # Existing catalog unaffected


def test_multiple_days_skipped_prefers_fresh_in_recency_window(temp_db: Path):
    """(f) If multiple days pass without refresh, 3-day feed recency window prioritizes newly refreshed articles."""
    from recommender import NewsRecommender

    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")
    fresh_date = (now - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")

    with db_connect(temp_db) as conn:
        conn.execute("DELETE FROM news")
        # Insert 8 fresh articles and 5 stale articles
        for i in range(8):
            conn.execute(
                """
                INSERT INTO news (news_id, category, subcategory, abstract, url, is_live, published_at, mapping_confidence)
                VALUES (?, 'technology', 'tech', ?, ?, 0, ?, 1.0)
                """,
                (f"nd_fresh_{i}", f"Fresh article {i}", f"https://example.com/fresh/{i}", fresh_date),
            )
        for i in range(5):
            conn.execute(
                """
                INSERT INTO news (news_id, category, subcategory, abstract, url, is_live, published_at, mapping_confidence)
                VALUES (?, 'technology', 'tech', ?, ?, 0, ?, 1.0)
                """,
                (f"nd_stale_{i}", f"Stale article {i}", f"https://example.com/stale/{i}", old_date),
            )

    engine = NewsRecommender(temp_db)
    with db_connect(temp_db) as conn:
        candidates, is_stale = engine._fetch_candidates_conn(conn, user_id="u_test", seen_news=set(), blocked=set(), pool_size=10)
        c_ids = [c["news_id"] for c in candidates]
        # Should be fresh candidates and not marked as stale window
        assert not is_stale
        assert all("nd_fresh_" in cid for cid in c_ids)

        # Now remove fresh articles to test graceful fallback to older articles
        conn.execute("DELETE FROM news WHERE published_at = ?", (fresh_date,))
        candidates_fallback, is_stale_fallback = engine._fetch_candidates_conn(conn, user_id="u_test", seen_news=set(), blocked=set(), pool_size=10)
        fb_ids = [c["news_id"] for c in candidates_fallback]
        assert is_stale_fallback is True
        assert any("nd_stale_" in cid for cid in fb_ids)
