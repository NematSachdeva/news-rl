from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlencode

import requests

from news_ingest.base import BaseNewsProvider, NormalizedArticle
from news_ingest.category_mapping import CategoryMapper
from news_ingest.credit_tracker import CreditTracker, QuotaExceededError
from news_ingest.dedupe import compute_url_id

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://newsdata.io/api/1/latest"
MAX_ITEMS_PER_PARAM = 5  # NewsData free plan allows at most 5 countries/categories/languages


class NewsDataProvider(BaseNewsProvider):
    """
    Provider adapter for NewsData.io (free tier compatible).
    Enforces 5/5/5 query constraints, credit quotas, retry backoff,
    disk caching, and category mapping.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_API_URL,
        credit_tracker: CreditTracker | None = None,
        category_mapper: CategoryMapper | None = None,
        cache_dir: Path | None = None,
        cache_ttl_seconds: float = 7200.0,
        request_timeout: float = 12.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url
        self.credit_tracker = credit_tracker
        self.category_mapper = category_mapper or CategoryMapper()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl_seconds = cache_ttl_seconds
        self.request_timeout = request_timeout

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, params: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        # Omit api key from cache key
        safe_params = {k: v for k, v in sorted(params.items()) if k != "apikey"}
        query_hash = hashlib.sha256(urlencode(safe_params).encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"req_{query_hash}.json"

    def _read_cache(self, path: Path | None) -> dict[str, Any] | None:
        if not path or not path.exists():
            return None
        try:
            mtime = path.stat().st_mtime
            if (time.time() - mtime) > self.cache_ttl_seconds:
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.debug("Failed reading cache at %s: %s", path, exc)
            return None

    def _write_cache(self, path: Path | None, data: dict[str, Any]) -> None:
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as exc:
            logger.debug("Failed writing cache to %s: %s", path, exc)

    def _execute_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute HTTP request with caching, quota checks, and exponential backoff."""
        cache_file = self._cache_path(params)
        cached = self._read_cache(cache_file)
        if cached is not None:
            logger.info("Serving query from local disk cache (%s)", cache_file.name)
            return cached

        if not self.api_key:
            raise ValueError(
                "NewsData.io API key is empty. Set NEWSDATA_API_KEY in .env to enable live news ingestion."
            )

        if self.credit_tracker and not self.credit_tracker.can_consume(1):
            raise QuotaExceededError(
                f"NewsData.io daily credit quota exhausted ({self.credit_tracker.get_credits_used_today()}/"
                f"{self.credit_tracker.daily_limit}). Ingestion halted to protect quota."
            )

        # Backoff retries for 429 (rate limits) and 5xx (transient errors)
        last_error = None
        for attempt in range(3):
            try:
                full_params = dict(params)
                full_params["apikey"] = self.api_key

                resp = requests.get(self.base_url, params=full_params, timeout=self.request_timeout)

                if resp.status_code == 429:
                    wait_time = (2 ** attempt) * 1.5
                    logger.warning("HTTP 429 Rate limit from NewsData.io; backing off %.1fs...", wait_time)
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Track consumed credit on successful network fetch
                if self.credit_tracker:
                    self.credit_tracker.record_usage(1)

                self._write_cache(cache_file, data)
                return data

            except requests.RequestException as exc:
                last_error = exc
                wait_time = (2 ** attempt) * 1.0
                logger.warning("NewsData.io request failed (attempt %d/3): %s", attempt + 1, exc)
                time.sleep(wait_time)

        raise RuntimeError(f"NewsData.io request failed after 3 attempts: {last_error}")

    def fetch(
        self,
        countries: list[str] | None = None,
        categories: list[str] | None = None,
        languages: list[str] | None = None,
        since: str | None = None,
        max_articles: int = 50,
    ) -> list[NormalizedArticle]:
        """
        Fetch articles respecting 5/5/5 parameter limits, pagination,
        and credit budget.
        """
        req_countries = [c.strip().lower() for c in (countries or ["us"]) if c.strip()]
        req_categories = [c.strip().lower() for c in (categories or ["top", "technology", "business", "sports", "world"]) if c.strip()]
        req_languages = [l.strip().lower() for l in (languages or ["en"]) if l.strip()]

        # Chunk lists to respect <= 5 items per query
        def chunk(lst: list[str], size: int) -> list[list[str]]:
            return [lst[i:i + size] for i in range(0, len(lst), size)] or [[]]

        country_chunks = chunk(req_countries, MAX_ITEMS_PER_PARAM)
        category_chunks = chunk(req_categories, MAX_ITEMS_PER_PARAM)
        language_chunks = chunk(req_languages, MAX_ITEMS_PER_PARAM)

        all_articles: list[NormalizedArticle] = []
        seen_ids: set[str] = set()

        for c_list in country_chunks:
            for cat_list in category_chunks:
                for lang_list in language_chunks:
                    if len(all_articles) >= max_articles:
                        break

                    page_token: str | None = None

                    while len(all_articles) < max_articles:
                        params: dict[str, Any] = {
                            "size": 10,
                        }
                        if c_list:
                            params["country"] = ",".join(c_list)
                        if cat_list:
                            params["category"] = ",".join(cat_list)
                        if lang_list:
                            params["language"] = ",".join(lang_list)
                        if page_token:
                            params["page"] = page_token

                        try:
                            data = self._execute_request(params)
                        except QuotaExceededError as q_err:
                            logger.warning("Ingestion stopped due to credit quota: %s", q_err)
                            return all_articles
                        except Exception as exc:
                            logger.error("Failed fetching chunk (country=%s, cat=%s): %s", c_list, cat_list, exc)
                            break

                        results = data.get("results", [])
                        if not results:
                            break

                        for raw in results:
                            norm = self._normalize_article(raw)
                            if norm.id not in seen_ids:
                                seen_ids.add(norm.id)
                                all_articles.append(norm)

                        page_token = data.get("nextPage")
                        if not page_token:
                            break

        return all_articles[:max_articles]

    def _normalize_article(self, raw: dict[str, Any]) -> NormalizedArticle:
        url = str(raw.get("link") or "").strip()
        art_id = compute_url_id(url) if url else f"nd_{raw.get('article_id', 'unknown')}"
        title = str(raw.get("title") or "").strip()

        # Check content body: free tier returns "ONLY AVAILABLE IN PAID PLANS"
        content_raw = str(raw.get("content") or "").strip()
        content = "" if "ONLY AVAILABLE IN PAID PLANS" in content_raw.upper() else content_raw

        description = str(raw.get("description") or "").strip()
        if not description and content:
            description = content[:300]
        elif not description and title:
            description = title

        raw_cats = raw.get("category")
        if isinstance(raw_cats, str):
            raw_cats = [raw_cats]
        elif not isinstance(raw_cats, list):
            raw_cats = []

        mapped_cat, confidence = self.category_mapper.map_categories(
            raw_categories=raw_cats,
            title=title,
            description=description,
        )

        source = str(raw.get("source_name") or raw.get("source_id") or "NewsData").strip()
        pub_date = str(raw.get("pubDate") or raw.get("pubDateTZ") or "").strip()

        country_val = raw.get("country")
        country = country_val[0] if isinstance(country_val, list) and country_val else str(country_val or "")
        language = str(raw.get("language") or "en")
        image_url = str(raw.get("image_url") or "").strip()

        return NormalizedArticle(
            id=art_id,
            title=title,
            description=description,
            url=url,
            source_name=source,
            published_at=pub_date,
            country=country,
            language=language,
            raw_categories=raw_cats,
            mapped_category=mapped_cat,
            mapping_confidence=confidence,
            image_url=image_url,
            content=content,
        )
