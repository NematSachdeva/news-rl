from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
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
MAX_ITEMS_PER_PARAM = 5  # NewsData free plan allows at most 5 items per query param


class NewsDataProvider(BaseNewsProvider):
    """
    Provider adapter for NewsData.io (free tier compatible).
    Enforces per-category target queries, pagination, rate limit pacing
    (30 requests / 15 minutes), credit budget limits, retry backoff,
    disk caching, and semantic category mapping.
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
        max_requests_per_15m: int = 28,  # Safe ceiling below 30 req / 15 min
        min_request_interval: float = 1.0,  # Min delay between consecutive network calls
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url
        self.credit_tracker = credit_tracker
        self.category_mapper = category_mapper or CategoryMapper()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl_seconds = cache_ttl_seconds
        self.request_timeout = request_timeout

        self.max_requests_per_15m = max_requests_per_15m
        self.min_request_interval = min_request_interval
        self._network_request_times: deque[float] = deque()
        self._last_request_time: float = 0.0

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, params: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
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

    def _pace_request(self) -> None:
        """Enforce 30 requests / 15 minutes rolling window and minimum delay."""
        now = time.time()
        # 1. Purge timestamps older than 15 minutes (900 seconds)
        while self._network_request_times and (now - self._network_request_times[0]) > 900.0:
            self._network_request_times.popleft()

        # 2. Check rolling window ceiling
        if len(self._network_request_times) >= self.max_requests_per_15m:
            oldest = self._network_request_times[0]
            sleep_needed = max(0.0, 900.0 - (now - oldest) + 1.5)
            if sleep_needed > 0:
                logger.warning(
                    "Approaching NewsData.io rate limit (%d requests in 15m window). Pacing execution: sleeping %.1fs...",
                    len(self._network_request_times),
                    sleep_needed,
                )
                time.sleep(sleep_needed)
                now = time.time()

        # 3. Minimum delay between consecutive outbound calls
        elapsed = now - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    def _execute_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute HTTP request with caching, credit limits, rate pacing, and backoff."""
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
                f"NewsData.io daily credit quota reached ({self.credit_tracker.get_credits_used_today()}/"
                f"{self.credit_tracker.daily_limit}). Ingestion halted to protect quota."
            )

        # Enforce rate pacing before contacting external API
        self._pace_request()

        # Backoff retries for 429 (rate limits) and 5xx (transient errors)
        last_error = None
        for attempt in range(3):
            try:
                full_params = dict(params)
                full_params["apikey"] = self.api_key

                resp = requests.get(self.base_url, params=full_params, timeout=self.request_timeout)
                self._last_request_time = time.time()
                self._network_request_times.append(self._last_request_time)

                if resp.status_code == 429:
                    wait_time = (2 ** attempt) * 2.0
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
                wait_time = (2 ** attempt) * 1.5
                logger.warning("NewsData.io request failed (attempt %d/3): %s", attempt + 1, exc)
                time.sleep(wait_time)

        raise RuntimeError(f"NewsData.io request failed after 3 attempts: {last_error}")

    def fetch(
        self,
        countries: list[str] | None = None,
        categories: list[str] | None = None,
        languages: list[str] | None = None,
        since: str | None = None,
        max_articles: int | None = None,
        target_per_category: int = 40,
    ) -> list[NormalizedArticle]:
        """
        Fetch articles querying per app category with per-category pagination,
        parameter limits (<= 5 items), rate pacing, and credit ceiling protection.
        """
        req_countries = [c.strip().lower() for c in (countries or ["in", "us", "gb"]) if c.strip()]
        req_languages = [l.strip().lower() for l in (languages or ["en"]) if l.strip()]

        app_cats = [c.strip().lower() for c in (categories or self.category_mapper.app_categories) if c.strip()]

        all_articles: list[NormalizedArticle] = []
        seen_ids: set[str] = set()
        short_categories: list[str] = []

        logger.info(
            "Beginning balanced ingestion across %d categories (target: %d/cat, countries=%s)",
            len(app_cats), target_per_category, req_countries[:MAX_ITEMS_PER_PARAM],
        )

        for app_cat in app_cats:
            if max_articles is not None and len(all_articles) >= max_articles:
                logger.info("Reached overall max article cap (%d); finishing ingestion.", max_articles)
                break

            # Map app category to provider category list (e.g. lifestyle -> ['lifestyle', 'food', 'tourism'])
            provider_cats = self.category_mapper.app_to_provider_categories.get(app_cat, [app_cat])
            cat_param = ",".join(provider_cats[:MAX_ITEMS_PER_PARAM])

            articles_this_cat = 0
            page_token: str | None = None
            category_exhausted = False

            while articles_this_cat < target_per_category and not category_exhausted:
                if max_articles is not None and len(all_articles) >= max_articles:
                    break

                params: dict[str, Any] = {
                    "size": 10,
                    "category": cat_param,
                }
                if req_countries:
                    params["country"] = ",".join(req_countries[:MAX_ITEMS_PER_PARAM])
                if req_languages:
                    params["language"] = ",".join(req_languages[:MAX_ITEMS_PER_PARAM])
                if page_token:
                    params["page"] = page_token

                try:
                    data = self._execute_request(params)
                except QuotaExceededError as q_err:
                    logger.warning("Daily credit quota reached while querying '%s': %s", app_cat, q_err)
                    short_categories.append(app_cat)
                    logger.info("Halting further requests and preserving %d fetched articles.", len(all_articles))
                    return all_articles
                except Exception as exc:
                    logger.error("Failed querying category '%s' (params=%s): %s", app_cat, params, exc)
                    short_categories.append(app_cat)
                    break

                results = data.get("results", [])
                if not results:
                    category_exhausted = True
                    break

                new_in_page = 0
                for raw in results:
                    norm = self._normalize_article(raw)
                    if norm.id not in seen_ids:
                        seen_ids.add(norm.id)
                        all_articles.append(norm)
                        articles_this_cat += 1
                        new_in_page += 1
                        if max_articles is not None and len(all_articles) >= max_articles:
                            break
                        if articles_this_cat >= target_per_category:
                            break

                page_token = data.get("nextPage")
                if not page_token or new_in_page == 0:
                    category_exhausted = True

            if articles_this_cat < target_per_category:
                short_categories.append(f"{app_cat} ({articles_this_cat}/{target_per_category})")
            logger.info("Category '%s': gathered %d articles (target: %d)", app_cat, articles_this_cat, target_per_category)

        if short_categories:
            logger.info("Categories below target: %s", ", ".join(short_categories))

        return all_articles

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

        source_name = str(raw.get("source_id") or raw.get("source_name") or "").replace("_", " ").title()
        country = str(raw.get("country", [""])[0] if isinstance(raw.get("country"), list) else raw.get("country") or "")
        language = str(raw.get("language") or "en")
        pub_date = str(raw.get("pubDate") or raw.get("pubDateTZ") or "").strip()
        image_url = str(raw.get("image_url") or "").strip()

        return NormalizedArticle(
            id=art_id,
            title=title,
            description=description,
            url=url,
            source_name=source_name,
            published_at=pub_date,
            country=country,
            language=language,
            raw_categories=raw_cats,
            mapped_category=mapped_cat,
            mapping_confidence=confidence,
            image_url=image_url,
            content=content,
        )
