from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class NormalizedArticle:
    id: str
    title: str
    description: str
    url: str
    source_name: str
    published_at: str
    country: str = ""
    language: str = "en"
    raw_categories: list[str] = field(default_factory=list)
    mapped_category: str = "world"
    mapping_confidence: float = 1.0
    image_url: str = ""
    content: str = ""

    def word_count(self) -> int:
        text = f"{self.title} {self.description} {self.content}".strip()
        return len(text.split()) if text else 0


class BaseNewsProvider(ABC):
    @abstractmethod
    def fetch(
        self,
        countries: list[str] | None = None,
        categories: list[str] | None = None,
        languages: list[str] | None = None,
        since: str | None = None,
        max_articles: int = 100,
    ) -> list[NormalizedArticle]:
        """
        Fetch articles from provider and return them as normalized articles.
        Implementations handle pagination, rate limiting, and credit budgeting.
        """
        pass
