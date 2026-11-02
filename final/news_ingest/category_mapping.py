from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

MAPPING_PATH = Path(__file__).resolve().parent / "category_mapping.json"


class FallbackCategoryClassifier:
    """
    Lightweight TF-IDF + LogisticRegression fallback classifier.
    Trained on existing categorized news articles in SQLite to categorize
    articles with ambiguous, empty, or unmapped raw categories.
    """

    def __init__(self, allowed_categories: Sequence[str]) -> None:
        self.allowed_categories = list(allowed_categories)
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: LogisticRegression | None = None
        self._fitted = False

    def fit_from_db(self, db_path: Path) -> bool:
        import sqlite3

        if not Path(db_path).exists():
            return False

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT category, subcategory, abstract, full_article "
                "FROM news WHERE category IS NOT NULL AND category != ''"
            ).fetchall()

        if len(rows) < 20:
            return False

        corpus: list[str] = []
        labels: list[str] = []
        for r in rows:
            cat = str(r["category"]).lower().strip()
            if cat in self.allowed_categories:
                text = f"{r['subcategory'] or ''} {r['abstract'] or ''} {r['full_article'] or ''}".strip()
                if text:
                    corpus.append(text)
                    labels.append(cat)

        if len(set(labels)) < 2:
            return False

        try:
            vec = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1, 2))
            X = vec.fit_transform(corpus)
            clf = LogisticRegression(max_iter=300, solver="lbfgs")
            clf.fit(X, labels)

            self.vectorizer = vec
            self.classifier = clf
            self._fitted = True
            logger.info("Fallback classifier fitted on %d articles across %d categories", len(corpus), len(clf.classes_))
            return True
        except Exception as exc:
            logger.warning("Could not fit fallback classifier: %s", exc)
            return False

    def predict(self, text: str) -> tuple[str, float]:
        if not self._fitted or not self.vectorizer or not self.classifier or not text.strip():
            return "world", 0.25

        try:
            X = self.vectorizer.transform([text])
            probs = self.classifier.predict_proba(X)[0]
            best_idx = int(probs.argmax())
            best_cat = str(self.classifier.classes_[best_idx])
            best_prob = float(probs[best_idx])
            return best_cat, round(best_prob, 4)
        except Exception as exc:
            logger.debug("Fallback classifier prediction error: %s", exc)
            return "world", 0.25


class CategoryMapper:
    """
    Maps news provider categories to the application's 10 category action space.
    Employs direct mapping rules, priority resolution, and ML fallback.
    """

    def __init__(self, mapping_file: Path | None = None, db_path: Path | None = None) -> None:
        path = mapping_file or MAPPING_PATH
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.app_categories = list(cfg["app_categories"])
        self.direct_mapping = dict(cfg["direct_mapping"])
        self.priority_order = list(cfg.get("priority_order", self.app_categories))

        self.classifier = FallbackCategoryClassifier(self.app_categories)
        if db_path and Path(db_path).exists():
            self.classifier.fit_from_db(db_path)

    def fit_classifier(self, db_path: Path) -> bool:
        return self.classifier.fit_from_db(db_path)

    def map_categories(
        self,
        raw_categories: Sequence[str] | None,
        title: str = "",
        description: str = "",
    ) -> tuple[str, float]:
        """
        Extract the best single category from raw categories, falling back to ML classification.
        Returns: (mapped_category, confidence)
        """
        raw_cats = [str(c).lower().strip() for c in (raw_categories or []) if str(c).strip()]

        # 1. Direct rule-based candidate resolution
        resolved_candidates: list[str] = []
        for raw in raw_cats:
            mapped = self.direct_mapping.get(raw)
            if mapped and mapped in self.app_categories:
                resolved_candidates.append(mapped)

        if resolved_candidates:
            # Pick best match according to priority order
            for priority_cat in self.priority_order:
                if priority_cat in resolved_candidates:
                    return priority_cat, 1.0
            return resolved_candidates[0], 0.95

        # 2. Text keyword matching heuristic
        text = f"{title} {description}".lower()
        keyword_hits = [cat for cat in self.app_categories if cat in text]
        if keyword_hits:
            for priority_cat in self.priority_order:
                if priority_cat in keyword_hits:
                    return priority_cat, 0.75
            return keyword_hits[0], 0.70

        # 3. Fallback ML classifier
        if text.strip():
            cat, prob = self.classifier.predict(text)
            if cat in self.app_categories and prob >= 0.30:
                return cat, prob

        # 4. Default safe category
        default_cat = "world" if "world" in self.app_categories else self.app_categories[0]
        return default_cat, 0.25

