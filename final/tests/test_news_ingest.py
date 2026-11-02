from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from news_ingest.base import NormalizedArticle
from news_ingest.category_mapping import CategoryMapper, FallbackCategoryClassifier
from news_ingest.credit_tracker import CreditTracker, QuotaExceededError
from news_ingest.dedupe import ArticleDeduplicator, compute_url_id, title_token_jaccard
from news_ingest.newsdata import NewsDataProvider
from news_ingest.storage import NewsStorage
from recommender import NewsRecommender


SAMPLE_API_RESPONSE_PAGE_1 = {
    "status": "success",
    "totalResults": 25,
    "nextPage": "page_2_token",
    "results": [
        {
            "article_id": "art_001",
            "title": "Quantum Computing Reaches Milestone in Fault-Tolerant Qubits",
            "link": "https://example.com/quantum-computing-2026",
            "description": "Researchers have demonstrated a new topological error-correction architecture.",
            "content": "ONLY AVAILABLE IN PAID PLANS",
            "category": ["top", "technology"],
            "source_id": "tech_times",
            "source_name": "Tech Times",
            "pubDate": "2026-11-02 12:00:00",
            "country": ["united states of america"],
            "language": "english",
            "image_url": "https://example.com/img1.jpg",
        },
        {
            "article_id": "art_002",
            "title": "Champions League Quarterfinal Draw Announced",
            "link": "https://example.com/champions-league-draw",
            "description": "The final eight teams have learned their matchups for the tournament.",
            "content": "ONLY AVAILABLE IN PAID PLANS",
            "category": ["sports"],
            "source_id": "sports_daily",
            "source_name": "Sports Daily",
            "pubDate": "2026-11-02 13:00:00",
            "country": ["united kingdom"],
            "language": "english",
            "image_url": "https://example.com/img2.jpg",
        },
    ],
}

SAMPLE_API_RESPONSE_PAGE_2 = {
    "status": "success",
    "totalResults": 25,
    "nextPage": None,
    "results": [
        {
            "article_id": "art_003",
            "title": "Global Markets Rally Following Central Bank Interest Rate Cut",
            "link": "https://example.com/markets-rally-rate-cut",
            "description": "Equities surged across major indices as inflation numbers cooled.",
            "content": "ONLY AVAILABLE IN PAID PLANS",
            "category": ["business", "top"],
            "source_id": "market_wire",
            "source_name": "Market Wire",
            "pubDate": "2026-11-02 14:00:00",
            "country": ["united states of america"],
            "language": "english",
            "image_url": "https://example.com/img3.jpg",
        }
    ],
}


def test_normalize_article_cleans_paid_content_and_maps_id():
    mapper = CategoryMapper()
    provider = NewsDataProvider(api_key="test_key", category_mapper=mapper)
    raw = SAMPLE_API_RESPONSE_PAGE_1["results"][0]

    norm = provider._normalize_article(raw)
    assert norm.id.startswith("real_")
    assert norm.title == "Quantum Computing Reaches Milestone in Fault-Tolerant Qubits"
    assert norm.description == "Researchers have demonstrated a new topological error-correction architecture."
    assert norm.content == ""  # 'ONLY AVAILABLE IN PAID PLANS' stripped out
    assert norm.mapped_category == "technology"
    assert norm.source_name == "Tech Times"
    assert norm.mapping_confidence >= 0.95


def test_deduplicator_detects_exact_url_and_similar_title():
    deduper = ArticleDeduplicator(similarity_threshold=0.75)
    deduper.register(
        url="https://example.com/breaking-news",
        title="Breaking: Historic Space Mission Launches to Mars Orbit",
    )

    # 1. Exact URL duplicate
    is_dup, reason = deduper.is_duplicate(
        url="https://example.com/breaking-news",
        title="Completely Different Title",
    )
    assert is_dup is True
    assert "duplicate_url" in reason

    # 2. Near-duplicate title with different URL
    is_dup, reason = deduper.is_duplicate(
        url="https://other.com/different-url",
        title="Historic Space Mission Launches to Mars Orbit Today",
    )
    assert is_dup is True
    assert "similar_title" in reason

    # 3. Novel distinct article passes
    is_dup, _ = deduper.is_duplicate(
        url="https://other.com/new-economy-report",
        title="Central Bank Reports Strong Employment Growth",
    )
    assert is_dup is False


def test_category_mapping_direct_and_priority():
    mapper = CategoryMapper()

    # Direct 1:1 match
    cat, conf = mapper.map_categories(["technology"], title="Test", description="Test")
    assert cat == "technology"
    assert conf >= 0.95

    # Multi-category with 'top': ignores top, picks specific topical category
    cat, conf = mapper.map_categories(["top", "business"], title="Stock Market Surge", description="Equities climb")
    assert cat == "business"
    assert conf >= 0.95

    # Secondary mappings: environment -> science, food -> lifestyle
    cat_env, _ = mapper.map_categories(["environment"], title="Climate Summit", description="Carbon targets")
    assert cat_env == "science"

    cat_food, _ = mapper.map_categories(["food"], title="Mediterranean Diet", description="Cooking recipes")
    assert cat_food == "lifestyle"


def test_fallback_category_classifier(tmp_path):
    from database import init_db

    db_path = tmp_path / "clf_test.sqlite"
    init_db(db_path)

    # Insert sample labeled articles across two categories
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        for i in range(15):
            conn.execute(
                "INSERT INTO news (news_id, category, abstract, full_article, is_live) "
                "VALUES (?, 'sports', 'Football soccer basketball tournament match goals athlete', 'Sports headline', 0)",
                (f"sp_{i}",),
            )
            conn.execute(
                "INSERT INTO news (news_id, category, abstract, full_article, is_live) "
                "VALUES (?, 'technology', 'Computer microprocessor semiconductor software algorithm artificial intelligence', 'Tech headline', 0)",
                (f"tc_{i}",),
            )
        conn.commit()

    classifier = FallbackCategoryClassifier(["sports", "technology", "world"])
    fitted = classifier.fit_from_db(db_path)
    assert fitted is True

    # Test sports prediction
    pred_cat, conf = classifier.predict("Football tournament championship goals")
    assert pred_cat == "sports"
    assert conf > 0.5

    # Test technology prediction
    pred_cat2, conf2 = classifier.predict("Microprocessor quantum computing software")
    assert pred_cat2 == "technology"
    assert conf2 > 0.5


def test_credit_tracker_enforces_daily_budget(tmp_path):
    db_file = tmp_path / "test_credits.sqlite"
    tracker = CreditTracker(db_file, daily_limit=5)

    assert tracker.get_credits_used_today() == 0
    assert tracker.remaining_credits() == 5
    assert tracker.can_consume(3) is True

    tracker.record_usage(3)
    assert tracker.get_credits_used_today() == 3
    assert tracker.remaining_credits() == 2

    # Consuming beyond limit raises QuotaExceededError
    assert tracker.can_consume(3) is False
    with pytest.raises(QuotaExceededError):
        tracker.record_usage(3)

    # Consuming exact remaining amount succeeds
    tracker.record_usage(2)
    assert tracker.get_credits_used_today() == 5
    assert tracker.remaining_credits() == 0


def test_provider_pagination_and_disk_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    tracker = CreditTracker(tmp_path / "credits.sqlite", daily_limit=10)
    provider = NewsDataProvider(
        api_key="mock_key",
        credit_tracker=tracker,
        cache_dir=cache_dir,
    )

    mock_resp_1 = MagicMock()
    mock_resp_1.status_code = 200
    mock_resp_1.json.return_value = SAMPLE_API_RESPONSE_PAGE_1

    mock_resp_2 = MagicMock()
    mock_resp_2.status_code = 200
    mock_resp_2.json.return_value = SAMPLE_API_RESPONSE_PAGE_2

    with patch("requests.get", side_effect=[mock_resp_1, mock_resp_2]) as mock_get:
        articles = provider.fetch(
            countries=["us"],
            categories=["technology", "sports", "business"],
            max_articles=10,
        )

        assert len(articles) == 3
        assert mock_get.call_count == 2
        assert tracker.get_credits_used_today() == 2

    # Verify disk cache: second fetch with identical params should read from disk
    # and consume ZERO additional credits
    with patch("requests.get") as mock_get_2:
        cached_articles = provider.fetch(
            countries=["us"],
            categories=["technology", "sports", "business"],
            max_articles=10,
        )
        assert len(cached_articles) == 3
        mock_get_2.assert_not_called()
        assert tracker.get_credits_used_today() == 2  # still 2!


def test_storage_and_nlp_refit(tmp_path):
    from database import init_db

    db_path = tmp_path / "news_test.sqlite"
    init_db(db_path)

    storage = NewsStorage(db_path)
    mapper = CategoryMapper()

    articles = [
        NormalizedArticle(
            id=f"real_art_{i}",
            title=f"Advanced Semiconductor Innovation Part {i}",
            description="New nanoscale transistors boost computing performance and efficiency.",
            url=f"https://example.com/article-{i}",
            source_name="Tech Dispatch",
            published_at="2026-11-02 12:00:00",
            raw_categories=["technology"],
            mapped_category="technology",
            mapping_confidence=1.0,
        )
        for i in range(15)
    ]

    counts = storage.store_articles(articles)
    assert counts["inserted"] == 15
    assert counts["deduped"] == 0
    assert storage.count_candidate_articles() == 15
    assert storage.count_real_articles() == 15

    # Recommender refits on the real catalog
    engine = NewsRecommender(db_path)
    engine.refit_nlp()
    assert engine._nlp_fitted is True
    assert len(engine._nlp.bm25.doc_ids) == 15

    # Verify cold-start recommendation works with real articles
    recs = engine.recommend("new_eval_user", k=4)
    assert recs["is_cold_start"] is True
    assert len(recs["recommendations"]) > 0
    first = recs["recommendations"][0]
    assert first["source_name"] == "Tech Dispatch"
    assert first["category"] == "technology"


def test_24h_rule_triggers_when_25h_old_and_skips_when_2h_old(tmp_path):
    from datetime import datetime, timezone, timedelta
    from database import init_db, set_meta
    from news_ingest.scheduler import is_refresh_due

    db_path = tmp_path / "news_24h.sqlite"
    init_db(db_path)
    storage = NewsStorage(db_path)

    # 1. Zero real articles in DB -> immediately due
    due, reason = is_refresh_due(db_path, interval_hours=24.0)
    assert due is True
    assert "No real news articles" in reason

    # Seed one real article
    storage.store_articles([
        NormalizedArticle(
            id="real_test_001",
            title="Real article headline",
            description="Snippet",
            url="https://example.com/art1",
            source_name="Daily",
            published_at="2026-11-02 10:00:00",
            raw_categories=["technology"],
            mapped_category="technology",
            mapping_confidence=1.0,
        )
    ])

    fixed_now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    # 2. Case: last ingest was 25 hours ago -> DUE
    last_ingest_25h = (fixed_now - timedelta(hours=25)).isoformat()
    set_meta("last_successful_ingest_at", last_ingest_25h, db_path=db_path)
    due, reason = is_refresh_due(db_path, interval_hours=24.0, now_dt=fixed_now)
    assert due is True
    assert "25.0h old" in reason

    # 3. Case: last ingest was 2 hours ago -> NOT DUE
    last_ingest_2h = (fixed_now - timedelta(hours=2)).isoformat()
    set_meta("last_successful_ingest_at", last_ingest_2h, db_path=db_path)
    due, reason = is_refresh_due(db_path, interval_hours=24.0, now_dt=fixed_now)
    assert due is False
    assert "last updated 2.0h ago" in reason

    # 4. Case: clock anomaly (timestamp is 3 hours in the future) -> SAFE NO-OP
    last_ingest_future = (fixed_now + timedelta(hours=3)).isoformat()
    set_meta("last_successful_ingest_at", last_ingest_future, db_path=db_path)
    due, reason = is_refresh_due(db_path, interval_hours=24.0, now_dt=fixed_now)
    assert due is False
    assert "Clock anomaly" in reason


def test_prune_expired_articles_preserves_interacted_items(tmp_path):
    from datetime import datetime, timezone, timedelta
    from database import init_db, prune_expired_articles, db_connect

    db_path = tmp_path / "prune_test.sqlite"
    init_db(db_path)
    storage = NewsStorage(db_path)

    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=16)).strftime("%Y-%m-%d %H:%M:%S")
    fresh_date = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")

    storage.store_articles([
        NormalizedArticle(
            id="real_old_interacted",
            title="Old Article with User Feedback",
            description="This article was liked by the user.",
            url="https://example.com/old-interacted",
            source_name="Source A",
            published_at=old_date,
            raw_categories=["technology"],
            mapped_category="technology",
            mapping_confidence=1.0,
        ),
        NormalizedArticle(
            id="real_old_uninteracted",
            title="Old Article without Feedback",
            description="Nobody engaged with this article.",
            url="https://example.com/old-uninteracted",
            source_name="Source B",
            published_at=old_date,
            raw_categories=["technology"],
            mapped_category="technology",
            mapping_confidence=1.0,
        ),
        NormalizedArticle(
            id="real_fresh_article",
            title="Fresh News from Yesterday",
            description="Recent news within the 14-day window.",
            url="https://example.com/fresh-art",
            source_name="Source C",
            published_at=fresh_date,
            raw_categories=["business"],
            mapped_category="business",
            mapping_confidence=1.0,
        ),
    ])

    # Record user interaction on real_old_interacted
    with db_connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO interactions (user_id, news_id, category, liked, created_at)
            VALUES ('user_abc', 'real_old_interacted', 'technology', 1, CURRENT_TIMESTAMP)
            """
        )

    # Prune articles older than 14 days
    pruned_count = prune_expired_articles(db_path, max_age_days=14)
    assert pruned_count == 1

    with db_connect(db_path) as conn:
        remaining_ids = {r[0] for r in conn.execute("SELECT news_id FROM news").fetchall()}

    assert "real_old_interacted" in remaining_ids  # Preserved because of user feedback!
    assert "real_fresh_article" in remaining_ids   # Preserved because within 14-day window!
    assert "real_old_uninteracted" not in remaining_ids  # Pruned!


def test_feed_recency_window_and_fallback(tmp_path):
    from datetime import datetime, timezone, timedelta
    from database import init_db

    db_path = tmp_path / "recency_test.sqlite"
    init_db(db_path)
    storage = NewsStorage(db_path)

    now = datetime.now(timezone.utc)
    # Articles from 5 days ago (older than the 3-day feed window)
    stale_date = (now - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")

    storage.store_articles([
        NormalizedArticle(
            id=f"real_stale_{i}",
            title=f"Archived Story Number {i}",
            description="Story published 5 days ago.",
            url=f"https://example.com/archived-{i}",
            source_name="Old Times",
            published_at=stale_date,
            raw_categories=["business"],
            mapped_category="business",
            mapping_confidence=1.0,
        )
        for i in range(10)
    ])

    engine = NewsRecommender(db_path)
    resp = engine.recommend("recency_user", k=6)

    # Since no articles exist within 3 days, feed should fall back to older articles
    # and mark the response with is_stale_window = True
    assert resp["is_stale_window"] is True
    assert len(resp["recommendations"]) > 0
    assert resp["recommendations"][0]["is_older"] is True


def test_empty_catalog_graceful_handling(tmp_path):
    from database import init_db

    db_path = tmp_path / "empty_catalog.sqlite"
    init_db(db_path)

    engine = NewsRecommender(db_path)
    resp = engine.recommend("brand_new_user", k=8)
    assert resp["recommendations"] == []
    assert resp["is_cold_start"] is True


def test_preference_ordering_on_real_catalog(tmp_path):
    from database import init_db

    db_path = tmp_path / "pref_order.sqlite"
    init_db(db_path)
    storage = NewsStorage(db_path)

    # Seed real articles across 8 distinct categories
    articles = []
    for cat in ["technology", "business", "sports", "world", "science", "entertainment", "health", "lifestyle"]:
        for i in range(4):
            articles.append(
                NormalizedArticle(
                    id=f"real_{cat}_{i}",
                    title=f"Headlines in {cat} #{i}",
                    description=f"Detailed report on current developments in {cat} sector.",
                    url=f"https://example.com/{cat}/{i}",
                    source_name=f"{cat.capitalize()} News",
                    published_at="2026-11-03 01:00:00",
                    raw_categories=[cat],
                    mapped_category=cat,
                    mapping_confidence=1.0,
                )
            )
    storage.store_articles(articles)

    engine = NewsRecommender(db_path)
    engine.refit_nlp()

    # (a) New user with declared interest in "technology"
    engine.onboard_user({
        "user_id": "tech_fan",
        "interests": ["technology"],
        "exploration_preference": 0.1,
    })
    recs = engine.recommend("tech_fan", k=8)

    # Batch is returned (respecting diversity cap of <= 2 per category)
    assert len(recs["recommendations"]) >= 6
    # Top recommendations are dominated by the declared interest
    top_categories = [r["category"] for r in recs["recommendations"][:2]]
    assert "technology" in top_categories

    # (b) User gives repeated likes on "sports" articles
    sports_article = next(r for r in recs["recommendations"] if r["category"] == "sports")
    for _ in range(3):
        engine.feedback({
            "user_id": "tech_fan",
            "news_id": sports_article["news_id"],
            "liked": 1,
            "time_spent": 15.0,
            "scroll_depth": 0.85,
        })

    # Next recommendation: sports affinity should rise
    recs_after = engine.recommend("tech_fan", k=8)
    sports_in_new_batch = any(r["category"] == "sports" for r in recs_after["recommendations"])
    assert sports_in_new_batch is True
    # The liked sports article itself is not repeated
    after_ids = [r["news_id"] for r in recs_after["recommendations"]]
    assert sports_article["news_id"] not in after_ids


def test_live_news_bulletin_does_not_mix_with_main_feed(tmp_path):
    from database import init_db, db_connect

    db_path = tmp_path / "live_isolate.sqlite"
    init_db(db_path)

    # Insert an article tagged is_live = 1 (NewsAPI.org Live News bulletin)
    with db_connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO news (news_id, category, abstract, is_live)
            VALUES ('live_bulletin_999', 'world', 'Breaking Live News Bulletin', 1)
            """
        )

    engine = NewsRecommender(db_path)
    resp = engine.recommend("test_user_live", k=8)

    # The is_live = 1 article must never appear in recommendations
    returned_ids = [r["news_id"] for r in resp["recommendations"]]
    assert "live_bulletin_999" not in returned_ids
