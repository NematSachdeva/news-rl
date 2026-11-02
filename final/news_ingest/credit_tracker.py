from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)


class QuotaExceededError(RuntimeError):
    """Raised when daily API credit quota has been reached."""
    pass


class CreditTracker:
    """
    Persistent, daily-resetting credit tracker stored in SQLite.
    Tracks credit usage per UTC calendar day.
    """

    def __init__(self, db_path: Path, daily_limit: int = 180) -> None:
        self.db_path = Path(db_path)
        self.daily_limit = daily_limit
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_credit_usage (
                    date_utc TEXT PRIMARY KEY,
                    credits_used INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def _today_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_credits_used_today(self) -> int:
        today = self._today_utc()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT credits_used FROM api_credit_usage WHERE date_utc = ?",
                (today,),
            ).fetchone()
            return int(row[0]) if row else 0

    def remaining_credits(self) -> int:
        used = self.get_credits_used_today()
        return max(0, self.daily_limit - used)

    def can_consume(self, credits: int = 1) -> bool:
        return (self.get_credits_used_today() + credits) <= self.daily_limit

    def record_usage(self, credits: int = 1) -> int:
        """
        Record credit consumption for today. Raises QuotaExceededError if over limit.
        Returns the new total credits used today.
        """
        if not self.can_consume(credits):
            used = self.get_credits_used_today()
            msg = (
                f"NewsData.io credit budget reached: {used}/{self.daily_limit} used today. "
                f"Requested {credits} more credits. Pausing ingestion to protect free quota."
            )
            logger.warning(msg)
            raise QuotaExceededError(msg)

        today = self._today_utc()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO api_credit_usage (date_utc, credits_used, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(date_utc) DO UPDATE SET
                    credits_used = credits_used + excluded.credits_used,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (today, credits),
            )
            conn.commit()
            new_total = conn.execute(
                "SELECT credits_used FROM api_credit_usage WHERE date_utc = ?",
                (today,),
            ).fetchone()[0]

        logger.info(
            "API credit consumed (+%d). Total today: %d/%d (remaining: %d)",
            credits, new_total, self.daily_limit, max(0, self.daily_limit - new_total),
        )
        return int(new_total)

