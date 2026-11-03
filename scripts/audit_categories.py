#!/usr/bin/env python3
"""
audit_categories.py — Audit category extraction quality in NeuroFeed.

Outputs:
  (a) Table of provider raw category -> app category -> count
  (b) Count of articles mapped via rule/direct mapping vs fallback text classification
  (c) 15 random samples per app category with title, source, raw categories, mapped category, confidence
  (d) Low-confidence articles below a configurable threshold
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "final"
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from config import get_settings


def audit_categories(
    db_path: Path | None = None,
    min_confidence: float = 0.60,
    samples_per_category: int = 15,
    seed: int = 42,
) -> None:
    settings = get_settings()
    db = Path(db_path or settings.db_path)

    if not db.exists():
        print(f"Error: Database not found at {db}")
        sys.exit(1)

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT news_id, category, subcategory, abstract, full_article,
                   source_name, raw_categories, mapping_confidence, url
            FROM news
            WHERE is_live = 0
            """
        ).fetchall()

    if not rows:
        print(f"No news articles found in {db}")
        return

    total = len(rows)
    print("=" * 90)
    print(f"  NeuroFeed Category Audit — Total Articles: {total}")
    print("=" * 90)

    # ── (a) Raw Category -> App Category -> Count ──────────────────────
    raw_to_app: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rule_mapped_count = 0
    fallback_mapped_count = 0

    for r in rows:
        app_cat = r["category"] or "unknown"
        conf = float(r["mapping_confidence"] or 0.0)

        raw_str = r["raw_categories"] or "[]"
        try:
            raw_list = json.loads(raw_str)
            if not isinstance(raw_list, list):
                raw_list = [str(raw_list)]
        except Exception:
            raw_list = [raw_str] if raw_str else ["(none)"]

        if not raw_list:
            raw_list = ["(none)"]

        for raw in raw_list:
            raw_to_app[str(raw).lower().strip()][app_cat] += 1

        # Heuristic: confidence >= 0.95 indicates rule mapping
        if conf >= 0.95:
            rule_mapped_count += 1
        else:
            fallback_mapped_count += 1

    print("\n--- (a) Provider Raw Category -> App Category Breakdown ---")
    print(f"{'Provider Raw Category':<25} | {'App Category (Count)':<60}")
    print("-" * 90)
    for raw_cat in sorted(raw_to_app.keys()):
        breakdown = ", ".join(f"{c}: {cnt}" for c, cnt in sorted(raw_to_app[raw_cat].items(), key=lambda x: -x[1]))
        print(f"{raw_cat:<25} | {breakdown}")

    # ── (b) Rule Mapping vs Fallback Classifier ────────────────────────
    print("\n--- (b) Mapping Method Distribution ---")
    print(f"  Rule-based mapping (conf >= 0.95):  {rule_mapped_count:4d}  ({(rule_mapped_count / total) * 100:.1f}%)")
    print(f"  Fallback classifier (conf < 0.95):   {fallback_mapped_count:4d}  ({(fallback_mapped_count / total) * 100:.1f}%)")

    # ── (c) 15 Samples per App Category ───────────────────────────────
    by_category: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)

    rng = random.Random(seed)
    print(f"\n--- (c) Samples per App Category (Up to {samples_per_category} each) ---")
    for cat in sorted(settings.categories):
        items = by_category.get(cat, [])
        print(f"\n[Category: {cat.upper()}] — Total in catalog: {len(items)}")
        if not items:
            print("  (No articles present)")
            continue

        sample_items = rng.sample(items, min(samples_per_category, len(items)))
        for i, it in enumerate(sample_items, 1):
            title = it["full_article"] or (it["abstract"][:75] + "..." if it["abstract"] else "(no title)")
            source = it["source_name"] or "(unknown)"
            conf = float(it["mapping_confidence"] or 0.0)
            raw = it["raw_categories"] or "[]"
            print(f"  {i:>2}. [{source}] {title[:75]}")
            print(f"      Raw: {raw} | Mapped: {it['category']} | Conf: {conf:.2f} | ID: {it['news_id']}")

    # ── (d) Low-Confidence Articles ────────────────────────────────────
    low_conf_items = [r for r in rows if float(r["mapping_confidence"] or 0.0) < min_confidence]
    print(f"\n--- (d) Low-Confidence Articles (Confidence < {min_confidence:.2f}) ---")
    print(f"  Found {len(low_conf_items)} low-confidence articles out of {total} ({(len(low_conf_items)/total)*100:.1f}%)")
    for r in low_conf_items[:20]:
        title = r["full_article"] or r["abstract"][:60]
        conf = float(r["mapping_confidence"] or 0.0)
        print(f"  * [Conf: {conf:.2f}] [{r['category']}] Raw: {r['raw_categories']} | {title[:70]}")
    if len(low_conf_items) > 20:
        print(f"  ... and {len(low_conf_items) - 20} more low-confidence items.")

    print("\n" + "=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NeuroFeed category extraction.")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite DB")
    parser.add_argument("--min-conf", type=float, default=0.60, help="Low confidence threshold")
    parser.add_argument("--samples", type=int, default=15, help="Samples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    audit_categories(
        db_path=Path(args.db) if args.db else None,
        min_confidence=args.min_conf,
        samples_per_category=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
