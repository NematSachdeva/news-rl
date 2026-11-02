from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add final/ to path if needed
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from database import init_db, prune_expired_articles
from logging_config import configure_logging, get_logger
from news_ingest.category_mapping import CategoryMapper
from news_ingest.credit_tracker import CreditTracker
from news_ingest.newsdata import NewsDataProvider
from news_ingest.storage import NewsStorage
from news_ingest.scheduler import is_refresh_due, record_successful_ingest
from recommender import NewsRecommender

logger = get_logger("news_ingest")


def run_ingest(
    provider_name: str = "newsdata",
    countries: list[str] | None = None,
    categories: list[str] | None = None,
    languages: list[str] | None = None,
    since: str = "48h",
    max_articles: int = 60,
    refit_nlp: bool = True,
    force: bool = False,
) -> dict[str, int]:
    settings = get_settings()
    init_db(settings.db_path)

    # 0. Check 24-hour refresh interval unless forced
    if not force:
        due, reason = is_refresh_due(settings.db_path, interval_hours=getattr(settings, "refresh_interval_hours", 24.0))
        if not due:
            logger.info("News ingestion skipped: %s. Use --force or 'make ingest-force' to bypass.", reason)
            storage = NewsStorage(settings.db_path)
            return {
                "fetched": 0,
                "deduped": 0,
                "inserted": 0,
                "credits_used": 0,
                "skipped_reason": reason,
            }

    # 1. Initialize credit tracker
    credit_tracker = CreditTracker(
        db_path=settings.db_path,
        daily_limit=getattr(settings, "newsdata_daily_credit_limit", 180),
    )

    # 2. Category mapper + fallback classifier
    mapper = CategoryMapper(db_path=settings.db_path)

    # 3. Provider initialization
    cache_dir = settings.db_path.parent / "cache" / "newsdata_raw"
    if provider_name.lower() == "newsdata":
        api_key = getattr(settings, "newsdata_api_key", "")
        if not api_key:
            logger.error(
                "NEWSDATA_API_KEY is not configured in .env! "
                "Add NEWSDATA_API_KEY=your_key to .env or export it as an environment variable."
            )
            return {"fetched": 0, "deduped": 0, "inserted": 0, "credits_used": 0}

        provider = NewsDataProvider(
            api_key=api_key,
            base_url=getattr(settings, "newsdata_base_url", "https://newsdata.io/api/1/latest"),
            credit_tracker=credit_tracker,
            category_mapper=mapper,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unsupported news provider: {provider_name}")

    credits_before = credit_tracker.get_credits_used_today()

    # 4. Fetch articles
    logger.info(
        "Fetching news from %s (countries=%s, categories=%s, max=%d)...",
        provider_name, countries, categories, max_articles,
    )
    articles = provider.fetch(
        countries=countries,
        categories=categories,
        languages=languages,
        since=since,
        max_articles=max_articles,
    )

    credits_after = credit_tracker.get_credits_used_today()
    credits_consumed = credits_after - credits_before

    # 5. Store articles in SQLite
    storage = NewsStorage(settings.db_path)
    counts = storage.store_articles(articles)
    counts["credits_used"] = credits_consumed

    # 6. Record successful ingest timestamp in system_meta
    record_successful_ingest(settings.db_path, counts["inserted"])

    # 7. Prune expired articles (older than max_article_age_days without interactions)
    max_age_days = getattr(settings, "max_article_age_days", 14)
    pruned = prune_expired_articles(settings.db_path, max_age_days=max_age_days)
    counts["pruned"] = pruned

    # 8. Refit NLP index so MMR and retrieval work immediately
    if refit_nlp and (counts["inserted"] > 0 or pruned > 0):
        logger.info("Refitting NLP engine (BM25 + TF-IDF + LSA) on updated catalog...")
        engine = NewsRecommender(settings.db_path)
        engine.refit_nlp()
        logger.info("NLP engine refit complete.")

    logger.info(
        "=== Ingestion Complete ===\n"
        "  Provider        : %s\n"
        "  Fetched         : %d\n"
        "  Deduped         : %d\n"
        "  Inserted        : %d\n"
        "  Pruned (expired): %d\n"
        "  Credits Used    : %d\n"
        "  Remaining Daily : %d / %d\n"
        "  Total in Catalog: %d (Real: %d)",
        provider_name,
        counts["fetched"],
        counts["deduped"],
        counts["inserted"],
        counts.get("pruned", 0),
        counts["credits_used"],
        credit_tracker.remaining_credits(),
        credit_tracker.daily_limit,
        storage.count_candidate_articles(),
        storage.count_real_articles(),
    )
    return counts


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest real news into NeuroFeed.")
    parser.add_argument("--provider", default="newsdata", help="News provider (default: newsdata)")
    parser.add_argument(
        "--countries",
        default=getattr(settings, "news_ingest_countries", "in,us,gb"),
        help="Comma-separated country codes (max 5 per sub-request)",
    )
    parser.add_argument(
        "--categories",
        default=getattr(settings, "news_ingest_categories", "business,education,entertainment,health,lifestyle,politics,science,sports,technology,world,top"),
        help="Comma-separated category names",
    )
    parser.add_argument(
        "--languages",
        default=getattr(settings, "news_ingest_languages", "en"),
        help="Comma-separated language codes",
    )
    parser.add_argument("--since", default=getattr(settings, "news_ingest_since", "48h"), help="Timeframe window")
    parser.add_argument("--max-articles", type=int, default=60, help="Max articles to fetch")
    parser.add_argument("--no-refit-nlp", action="store_true", help="Skip refitting the NLP index")
    parser.add_argument("--force", action="store_true", help="Bypass 24-hour refresh interval check")

    args = parser.parse_args()
    configure_logging("INFO")

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    run_ingest(
        provider_name=args.provider,
        countries=countries,
        categories=categories,
        languages=languages,
        since=args.since,
        max_articles=args.max_articles,
        refit_nlp=not args.no_refit_nlp,
        force=args.force,
    )


if __name__ == "__main__":
    main()
