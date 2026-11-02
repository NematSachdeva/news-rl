from __future__ import annotations

import hashlib
import re
from typing import Iterable, Set


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy similarity comparison."""
    t = re.sub(r"[^\w\s]", " ", (title or "").lower())
    return " ".join(t.split())


def compute_url_id(url: str, prefix: str = "real_") -> str:
    """Generate deterministic 16-hex ID from article URL."""
    canonical = (url or "").strip().lower()
    return prefix + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def title_token_jaccard(title_a: str, title_b: str) -> float:
    """Compute token Jaccard similarity between two headlines."""
    tokens_a = set(normalize_title(title_a).split())
    tokens_b = set(normalize_title(title_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


class ArticleDeduplicator:
    """
    Deduplicates articles using exact URL hashing and near-duplicate title similarity.
    Maintains seen sets for fast in-memory batch checks and can preload recent DB titles.
    """

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        self.similarity_threshold = similarity_threshold
        self.seen_urls: set[str] = set()
        self.seen_titles: list[str] = []

    def load_existing(self, urls: Iterable[str], titles: Iterable[str]) -> None:
        for u in urls:
            if u:
                self.seen_urls.add(u.strip().lower())
        for t in titles:
            if t:
                norm = normalize_title(t)
                if norm:
                    self.seen_titles.append(norm)

    def is_duplicate(self, url: str, title: str) -> tuple[bool, str]:
        canonical_url = (url or "").strip().lower()
        if canonical_url and canonical_url in self.seen_urls:
            return True, "duplicate_url"

        norm_title = normalize_title(title)
        if norm_title:
            for existing in self.seen_titles:
                sim = title_token_jaccard(norm_title, existing)
                if sim >= self.similarity_threshold:
                    return True, f"similar_title ({sim:.2f})"

        return False, ""

    def register(self, url: str, title: str) -> None:
        canonical_url = (url or "").strip().lower()
        if canonical_url:
            self.seen_urls.add(canonical_url)
        norm_title = normalize_title(title)
        if norm_title:
            self.seen_titles.append(norm_title)
