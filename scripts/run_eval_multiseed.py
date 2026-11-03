#!/usr/bin/env python3
"""
scripts/run_eval_multiseed.py — Run evaluation across 3 seeds (20 users x 20 rounds).
Reports mean and spread (std and min-max) across seeds.
Cleans up temporary evaluation users from database afterwards.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "final"
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from config import get_settings
from database import db_connect
from evaluation import evaluate, _catalog_size
from recommender import NewsRecommender

SEEDS = [42, 123, 999]
USERS = 20
ROUNDS = 20
K = 8


def clean_eval_users(db_path: Path) -> None:
    with db_connect(db_path) as conn:
        for tbl in ["users", "interactions", "user_category_prefs", "user_memory", "agent_state", "graph_edges", "graph_co_interest_edges"]:
            conn.execute(f"DELETE FROM {tbl} WHERE user_id LIKE ?", ("eval_sim_user_%",))


def main() -> None:
    settings = get_settings()
    engine = NewsRecommender(settings.db_path)
    cat_size = _catalog_size(engine)
    print(f"Catalog size: {cat_size} articles")
    print(f"Running evaluation: {USERS} users x {ROUNDS} rounds (k={K}) across seeds {SEEDS}...\n")

    runs = []
    for s in SEEDS:
        print(f"Running seed {s}...")
        res = evaluate(engine, n_users=USERS, rounds=ROUNDS, k=K, seed=s)
        runs.append(res)
        print(f"  Seed {s} -> P@K: {res['precision_at_k']:.3f}, NDCG: {res['ndcg_at_k']:.3f}, Diversity: {res['category_diversity']:.3f}, Coverage: {res['catalog_coverage']:.3f}")

    # Compute mean and spread
    metrics = ["precision_at_k", "ndcg_at_k", "category_diversity", "catalog_coverage", "early_precision_at_k", "late_precision_at_k", "learning_delta"]
    print("\n" + "=" * 80)
    print(f"{'Metric':<25} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Values across seeds'}")
    print("-" * 80)

    for m in metrics:
        vals = [r[m] for r in runs]
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        min_v = min(vals)
        max_v = max(vals)
        val_str = ", ".join(f"{v:.4f}" for v in vals)
        print(f"{m:<25} | {mean_v:>8.4f} | {std_v:>8.4f} | {min_v:>8.4f} | {max_v:>8.4f} | [{val_str}]")
    print("=" * 80)

    # Cleanup eval users from DB
    clean_eval_users(settings.db_path)
    print("\nCleaned up evaluation users from database.")


if __name__ == "__main__":
    main()
