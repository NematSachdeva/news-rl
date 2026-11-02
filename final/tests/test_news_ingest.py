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
