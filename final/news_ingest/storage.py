from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from news_ingest.base import NormalizedArticle
from news_ingest.dedupe import ArticleDeduplicator

logger = logging.getLogger(__name__)


class NewsStorage:
    """
    Manages persistence of normalized news articles into the main SQLite news table.
    Ensures deduplication, proper candidate tagging (is_live=0), and recency maintenance.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(news)").fetchall()}
        extra_cols = {
            "is_live": "INTEGER DEFAULT 0",
            "image_url": "TEXT DEFAULT ''",
            "source_name": "TEXT DEFAULT ''",
            "published_at": "TEXT DEFAULT ''",
            "raw_categories": "TEXT DEFAULT ''",
            "mapping_confidence": "REAL DEFAULT 1.0",
        }
        for col, definition in extra_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE news ADD COLUMN {col} {definition}")

    def load_existing_for_dedupe(self) -> ArticleDeduplicator:
        deduper = ArticleDeduplicator()
        if not self.db_path.exists():
            return deduper

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            rows = conn.execute(
                "SELECT url, full_article, abstract FROM news WHERE is_live = 0 ORDER BY rowid DESC LIMIT 1000"
            ).fetchall()
            urls = [r["url"] for r in rows if r["url"]]
            titles = [r["full_article"] or r["abstract"] for r in rows if (r["full_article"] or r["abstract"])]
            deduper.load_existing(urls, titles)
        return deduper

    def store_articles(
        self,
        articles: Sequence[NormalizedArticle],
        deduper: ArticleDeduplicator | None = None,
    ) -> dict[str, int]:
        """
        Store normalized articles into SQLite news table.
        Returns counts: {'fetched': N, 'deduped': D, 'inserted': I}
        """
        if not articles:
            return {"fetched": 0, "deduped": 0, "inserted": 0}

        if deduper is None:
            deduper = self.load_existing_for_dedupe()

        inserted_count = 0
        deduped_count = 0

        with sqlite3.connect(self.db_path) as conn:
            self._ensure_schema(conn)
            for art in articles:
                is_dup, reason = deduper.is_duplicate(art.url, art.title)
                if is_dup:
                    deduped_count += 1
                    logger.debug("Skipping duplicate article '%s' (%s)", art.title[:40], reason)
                    continue

                subcategory = art.raw_categories[0] if art.raw_categories else art.mapped_category
                raw_cats_json = json.dumps(art.raw_categories)
                word_count = art.word_count() or 120

                conn.execute(
                    """
                    INSERT INTO news (
                        news_id, category, subcategory, location, preferred_time, age_group,
                        article_length, abstract, full_article, url, is_live, image_url,
                        source_name, published_at, raw_categories, mapping_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                    ON CONFLICT(news_id) DO UPDATE SET
                        category = excluded.category,
                        subcategory = excluded.subcategory,
                        abstract = excluded.abstract,
                        full_article = excluded.full_article,
                        url = excluded.url,
                        image_url = excluded.image_url,
                        source_name = excluded.source_name,
                        published_at = excluded.published_at,
                        raw_categories = excluded.raw_categories,
                        mapping_confidence = excluded.mapping_confidence
                    """,
                    (
                        art.id,
                        art.mapped_category,
                        subcategory,
                        art.country,
                        "",
                        "",
                        word_count,
                        art.description,  # abstract snippet
                        art.title,        # headline / full_article
                        art.url,
                        art.image_url,
                        art.source_name,
                        art.published_at,
                        raw_cats_json,
                        art.mapping_confidence,
                    ),
                )
                deduper.register(art.url, art.title)
                inserted_count += 1

            conn.commit()

        logger.info(
            "Ingestion storage summary: fetched=%d, deduped=%d, inserted=%d",
            len(articles), deduped_count, inserted_count,
        )
        return {
            "fetched": len(articles),
            "deduped": deduped_count,
            "inserted": inserted_count,
        }

    def count_candidate_articles(self) -> int:
        """Count total candidate articles available for recommendation (is_live=0)."""
        if not self.db_path.exists():
            return 0
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_schema(conn)
            row = conn.execute("SELECT COUNT(*) FROM news WHERE is_live = 0").fetchone()
            return int(row[0]) if row else 0

    def count_real_articles(self) -> int:
        """Count real ingested articles (non-synthetic)."""
        if not self.db_path.exists():
            return 0
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                "SELECT COUNT(*) FROM news WHERE is_live = 0 AND (news_id LIKE 'real_%' OR news_id LIKE 'nd_%')"
            ).fetchone()
            return int(row[0]) if row else 0
