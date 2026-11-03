"""
Centralized runtime configuration.

Every tunable that used to be a hardcoded constant or a scattered
`os.getenv(...)` call lives here instead, validated once at startup via
pydantic-settings. Override any of it with environment variables (see
`.env.example`) — nothing here requires touching code to change behavior
between local dev, CI, and a container deployment.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEUROFEED_",
        env_file=(str(ROOT.parent / ".env"), str(ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Environment ─────────────────────────────────────────
    env: str = "development"          # development | production | test
    log_level: str = "INFO"

    # ── Data ────────────────────────────────────────────────
    db_path: Path = ROOT / "data" / "news_recommender.sqlite"
    auth_db_path: Path = ROOT / "data" / "auth.sqlite"
    model_dir: Path = ROOT / "models"

    # ── CORS ────────────────────────────────────────────────
    # Comma-separated origins in production; "*" is fine for local dev only.
    cors_origins: str = "*"

    # ── Webcam attention ────────────────────────────────────
    enable_webcam_attention: bool = False

    # ── Recommender tuning ──────────────────────────────────
    # Below this many logged interactions a user is treated as cold-start:
    # scoring leans harder on stated interests + trending, lighter on the
    # (still mostly untrained) RL/bandit signals.
    cold_start_interaction_threshold: int = 5

    # MMR (Maximal Marginal Relevance) diversity re-ranking trade-off.
    # 1.0 = pure relevance ranking, 0.0 = pure novelty/diversity.
    mmr_lambda: float = 0.75

    # Max distinct users' DDQNAgent (PyTorch model + bandit state) kept
    # resident in memory at once. Without a bound this grows for every
    # distinct user_id a long-running process ever serves — least-recently
    # used agents are checkpointed to disk and evicted past this cap, and
    # transparently reloaded from disk the next time that user is active.
    max_cached_agents: int = 500

    # ── Caching ─────────────────────────────────────────────
    cache_ttl_seconds: float = 30.0

    # ── API hardening ───────────────────────────────────────
    rate_limit_requests: int = 120
    rate_limit_window_seconds: float = 60.0

    # Stricter limiter applied only to /auth/signup and /auth/login — these
    # are higher-value abuse targets (credential stuffing, fake account
    # creation) than the general API surface, so they get a tighter budget
    # on top of (not instead of) the general per-IP limiter above.
    auth_rate_limit_requests: int = 10
    auth_rate_limit_window_seconds: float = 60.0

    # Login brute-force lockout — see login_throttle.py.
    login_max_failures: int = 5
    login_failure_window_seconds: float = 300.0
    login_lockout_seconds: float = 60.0

    # ── Live news (NewsAPI.org) ─────────────────────────────
    # Empty by default — the "Live News" bulletin (GET /news/live) and
    # the background refresh it triggers are both no-ops without a key,
    # never a startup error. Get a free key at https://newsapi.org.
    news_api_key: str = ""
    news_api_base_url: str = "https://newsapi.org/v2/top-headlines"
    news_api_country: str = "us"
    # How many articles to pull per refresh. Kept small deliberately —
    # this is a display bulletin, not the main catalog.
    news_api_page_size: int = 12
    # Minimum time between live refreshes. NewsAPI's free tier caps out at
    # 100 requests/day; one refresh = one request, so 90 min (16/day) stays
    # well under that with room for manual restarts.
    news_live_cache_ttl_seconds: float = 5400.0

    # ── Real News Ingestion (NewsData.io) ───────────────────
    # Free tier key at https://newsdata.io (200 credits/day).
    # Configurable via NEWSDATA_API_KEY in .env or environment variable.
    newsdata_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("NEWSDATA_API_KEY", "NEUROFEED_NEWSDATA_API_KEY"),
    )
    newsdata_base_url: str = "https://newsdata.io/api/1/latest"
    newsdata_daily_credit_limit: int = 180
    news_ingest_countries: str = "in,us,gb"
    news_ingest_categories: str = "business,education,entertainment,health,lifestyle,politics,science,sports,technology,world,top"
    news_ingest_languages: str = "en"
    news_ingest_since: str = "48h"
    news_ingest_target_per_category: int = Field(
        default=40,
        validation_alias=AliasChoices("NEUROFEED_NEWS_INGEST_TARGET_PER_CATEGORY", "NEWS_INGEST_TARGET_PER_CATEGORY"),
    )
    news_category_min_confidence: float = Field(
        default=0.50,
        validation_alias=AliasChoices("NEUROFEED_NEWS_CATEGORY_MIN_CONFIDENCE", "NEWS_CATEGORY_MIN_CONFIDENCE"),
    )
    news_ingest_interval_hours: float = 0.0
    refresh_interval_hours: float = Field(
        default=24.0,
        validation_alias=AliasChoices("NEUROFEED_REFRESH_INTERVAL_HOURS", "REFRESH_INTERVAL_HOURS"),
    )
    feed_recency_days: int = Field(
        default=3,
        validation_alias=AliasChoices("NEUROFEED_FEED_RECENCY_DAYS", "FEED_RECENCY_DAYS"),
    )
    max_article_age_days: int = Field(
        default=14,
        validation_alias=AliasChoices("NEUROFEED_MAX_ARTICLE_AGE_DAYS", "MAX_ARTICLE_AGE_DAYS"),
    )

    # ── RL Agent Tuning ─────────────────────────────────────
    ddqn_min_replay: int = Field(
        default=8,
        validation_alias=AliasChoices("NEUROFEED_DDQN_MIN_REPLAY", "DDQN_MIN_REPLAY"),
    )
    ddqn_gamma: float = Field(
        default=0.9,
        validation_alias=AliasChoices("NEUROFEED_DDQN_GAMMA", "DDQN_GAMMA"),
    )

    @property
    def categories(self) -> list[str]:
        return [
            "business",
            "education",
            "entertainment",
            "health",
            "lifestyle",
            "politics",
            "science",
            "sports",
            "technology",
            "world",
        ]

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
