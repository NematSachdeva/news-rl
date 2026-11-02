"""
News Ingest Package for NeuroFeed.
Provides multi-provider adapters, normalization, category mapping,
credit budgeting, deduplication, and database persistence.
"""
from news_ingest.base import BaseNewsProvider, NormalizedArticle
from news_ingest.category_mapping import CategoryMapper
from news_ingest.credit_tracker import CreditTracker, QuotaExceededError
from news_ingest.dedupe import ArticleDeduplicator
from news_ingest.newsdata import NewsDataProvider
from news_ingest.storage import NewsStorage

__all__ = [
    "BaseNewsProvider",
    "NormalizedArticle",
    "CategoryMapper",
    "CreditTracker",
    "QuotaExceededError",
    "ArticleDeduplicator",
    "NewsDataProvider",
    "NewsStorage",
]
