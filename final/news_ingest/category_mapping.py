from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

MAPPING_PATH = Path(__file__).resolve().parent / "category_mapping.json"

# Stock/analyst ticker and research report patterns that often get misattributed
# to sports or entertainment by RSS scrapers on affiliate blogs
STOCK_ANALYST_RE = re.compile(
    r"\b(nyse:[a-z0-9_]+|nasdaq:[a-z0-9_]+|otcmkts:[a-z0-9_]+|tsx:[a-z0-9_]+|"
    r"price target (cut|raised|boosted|trimmed|lowered|set)|"
    r"stock (gets|receives|has) (price target|rating|downgrade|upgrade)|"
    r"(downgraded?|upgraded?) (by|from|to)|"
    r"consensus (rating|price target)|"
    r"zacks (investment research|rank)|"
    r"research note (released|published)|"
    r"earnings per share|market capitalization|dividend yield|"
    r"wall street zen|marketbeat|benzinga reports)\b",
    re.IGNORECASE,
)

# Sports gameplay patterns to prevent false positives when a sports story
# mentions financial terms (e.g. stadium financing or franchise valuation)
SPORTS_GAMEPLAY_RE = re.compile(
    r"\b(touchdown|field goal|quarterback|innings?|wickets?|tournament|coaches?|"
    r"championships?|nfl|nba|fifa|ipl|cricket|tennis|soccer|football|"
    r"pitcher|striker|premier league|formula 1|olympic|grand slam)\b",
    re.IGNORECASE,
)

# Domain lexicons for disambiguating generic ("top", "breaking") or raw-unmapped articles
CATEGORY_LEXICONS: dict[str, list[str]] = {
    "politics": [
        "president", "prime minister", "parliament", "congress", "senate", "senator",
        "election", "vote", "voting", "voters", "ballot", "democrat", "republican",
        "legislation", "legislature", "white house", "supreme court", "governor",
        "campaign", "veto", "cabinet", "political", "diplomat", "treaty", "impeachment",
    ],
    "technology": [
        "software", "artificial intelligence", " ai ", "chipmaker", "semiconductor",
        "apple", "google", "microsoft", "nvidia", "cybersecurity", "smartphone",
        "cloud computing", "machine learning", "robotics", "algorithm", "gadget",
        "datacentre", "data center", "operating system", "broadband", "gpu",
    ],
    "business": [
        "revenue", "profit", "quarterly earnings", "stocks", "shares", "investor",
        "wall street", "central bank", "interest rate", "inflation", "merger",
        "acquisition", "nasdaq", "nyse", "valuation", "fiscal", "retail sales",
        "cryptocurrency", "bitcoin", "banking", "treasury", "gdp", "economic growth",
    ],
    "science": [
        "nasa", "spacecraft", "astronomy", "telescope", "physics", "climate change",
        "biologist", "fossil", "dinosaur", "paleontologist", "geology", "quantum",
        "solar system", "planet", "exoplanet", "asteroid", "laboratory", "scientific study",
        "greenhouse gas", "biodiversity",
    ],
    "health": [
        "hospital", "patient", "disease", "cancer", "vaccine", "fda", "clinical trial",
        "physician", "doctor", "mental health", "cardiovascular", "diabetes", "medical",
        "pharmaceutical", "virus", "infection", "epidemic", "therapy", "surgery",
    ],
    "education": [
        "university", "college", "school", "students", "teachers", "campus",
        "tuition", "degree", "curriculum", "classroom", "academic", "scholarship",
        "superintendent", "higher education", "faculty", "literacy", "professors",
    ],
    "entertainment": [
        "movie", "film", "box office", "actor", "actress", "hollywood", "album",
        "concert", "music", "singer", "director", "trailer", "streaming series",
        "netflix", "billboard", "emmy", "grammy", "oscar", "cinema", "soundtrack",
    ],
    "sports": [
        "football", "soccer", "cricket", "basketball", "tennis", "coach", "league",
        "tournament", "match", "championship", "stadium", "referee", "goal",
        "touchdown", "nba", "nfl", "premier league", "athlete", "roster",
    ],
    "lifestyle": [
        "recipe", "cooking", "travel destination", "fashion", "luxury hotel",
        "interior design", "skincare", "parenting", "wellness", "resort",
        "vacation", "dining", "cocktail", "gardening", "wardrobe", "cuisine",
    ],
    "world": [
        "united nations", "peacekeeping", "foreign ministry", "bilateral",
        "refugee", "border crossing", "humanitarian", "international summit",
        "sanctions", "embassy", "ambassador", "treaty", "nato", "geopolitics",
    ],
}


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
    Employs direct mapping rules, semantic cross-checks, domain lexicons, and ML fallback.
    """

    def __init__(self, mapping_file: Path | None = None, db_path: Path | None = None) -> None:
        path = mapping_file or MAPPING_PATH
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.app_categories = list(cfg["app_categories"])
        self.app_to_provider_categories = dict(cfg.get("app_to_provider_categories", {}))
        self.direct_mapping = dict(cfg["direct_mapping"])
        self.priority_order = list(cfg.get("priority_order", self.app_categories))

        self.classifier = FallbackCategoryClassifier(self.app_categories)
        if db_path and Path(db_path).exists():
            self.classifier.fit_from_db(db_path)

    def fit_classifier(self, db_path: Path) -> bool:
        return self.classifier.fit_from_db(db_path)

    def _score_by_lexicon(self, text: str) -> dict[str, int]:
        """Score occurrences of category-specific domain keywords in text."""
        scores: dict[str, int] = {cat: 0 for cat in self.app_categories}
        lower_text = f" {text.lower()} "
        for cat, keywords in CATEGORY_LEXICONS.items():
            if cat not in scores:
                continue
            for kw in keywords:
                # Substring check with word boundary padding
                if kw in lower_text:
                    scores[cat] += 1
        return scores

    def map_categories(
        self,
        raw_categories: Sequence[str] | None,
        title: str = "",
        description: str = "",
    ) -> tuple[str, float]:
        """
        Extract the best single category from raw categories, cross-checking
        against text semantics to correct provider mislabels and generic tags.
        Returns: (mapped_category, confidence)
        """
        text = f"{title} {description}".strip()
        raw_cats = [str(c).lower().strip() for c in (raw_categories or []) if str(c).strip()]

        # ── Cross-Check: Financial / Stock Analyst Override ───────────
        # Scrapers often syndicate MarketBeat/SEC stock blurbs across sports or
        # regional newspaper domains (e.g. baseballnewssource.com).
        # If the text is saturated with financial market patterns without actual sports,
        # override provider misattribution immediately.
        if STOCK_ANALYST_RE.search(text) and not SPORTS_GAMEPLAY_RE.search(text):
            return "business", 0.95

        # ── 1. Direct rule-based candidate resolution ─────────────────
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

        # ── 2. Lexicon Keyword Scoring (for generic 'top', 'breaking', or empty) ─
        lexicon_scores = self._score_by_lexicon(text)
        sorted_lexicon = sorted(lexicon_scores.items(), key=lambda x: -x[1])
        top_cat, top_score = sorted_lexicon[0]
        second_score = sorted_lexicon[1][1] if len(sorted_lexicon) > 1 else 0

        if top_score >= 2 and top_score > second_score:
            confidence = min(0.85, 0.65 + (top_score - second_score) * 0.05)
            return top_cat, round(confidence, 2)

        # ── 3. Fallback ML Classifier ─────────────────────────────────
        if text:
            cat, prob = self.classifier.predict(text)
            if cat in self.app_categories and prob >= 0.40:
                return cat, prob

        # If top lexicon had at least 1 keyword match
        if top_score >= 1:
            return top_cat, 0.55

        # ── 4. Default safe category ──────────────────────────────────
        default_cat = "world" if "world" in self.app_categories else self.app_categories[0]
        return default_cat, 0.25
