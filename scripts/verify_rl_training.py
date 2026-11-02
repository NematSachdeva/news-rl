#!/usr/bin/env python3
"""
verify_rl_training.py — Proof that the DDQN agent really learns from
user interactions in NeuroFeed.

What this script does
─────────────────────
1. Creates a temporary SQLite database and model directory (deleted on exit).
2. Inserts 12 fixture articles across two categories: "technology" (liked)
   and "business" (disliked).
3. Creates a DDQNAgent pointing at these temp paths — fully isolated from
   the production database and checkpoints.
4. Simulates 60 interactions:
     • Odd interactions  → technology article liked
     • Even interactions → business article disliked
5. After each interaction prints a table row:
     #  buffer  grad?  loss        ε        step  target?  ‖W‖before  ‖W‖after  Q_tech  Q_biz
6. After the loop:
   a. Shows recommendation-category mix (Q-value histogram).
   b. Saves checkpoint, reloads a fresh DDQNAgent from the same model_dir,
      and verifies Q-values are restored — not reset to the untrained prior.

Usage
─────
  cd "rl proj"
  .venv/bin/python scripts/verify_rl_training.py

Or via Makefile:
  make verify-rl
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

# ── path setup so we can import final/ modules ─────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "final"
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

import numpy as np
import torch

from database import init_db, db_connect
from ddqn_agent import DDQNAgent

# ── constants ───────────────────────────────────────────────────────────────
TECH_CAT = "technology"
BIZ_CAT  = "business"
CATEGORIES = [TECH_CAT, BIZ_CAT]
N_INTERACTIONS = 60
FIXTURE_ARTICLES = [
    {"news_id": f"nd_tech_{i}", "category": TECH_CAT,
     "abstract": f"Tech content about AI and software engineering article {i}.",
     "url": f"https://example.com/tech/{i}"}
    for i in range(6)
] + [
    {"news_id": f"nd_biz_{i}", "category": BIZ_CAT,
     "abstract": f"Business and finance news content article {i}.",
     "url": f"https://example.com/biz/{i}"}
    for i in range(6)
]


def _weight_norm(agent: DDQNAgent) -> float:
    """Frobenius norm of all online-network parameters concatenated."""
    total = 0.0
    for p in agent.model.parameters():
        total += p.data.norm().item() ** 2
    return math.sqrt(total)


def _q_vals(agent: DDQNAgent) -> tuple[float, float]:
    """Q-values for a neutral 11-D state for each category."""
    state = [0.5] * 11
    qs = agent.q_values(state)
    ti = agent.categories.index(TECH_CAT)
    bi = agent.categories.index(BIZ_CAT)
    return float(qs[ti]), float(qs[bi])


def _insert_fixtures(db_path: Path) -> None:
    with db_connect(db_path) as conn:
        # init_db populates category_stats with the canonical 10 categories.
        # Ensure our two fixture categories are also in category_stats.
        for cat in CATEGORIES:
            conn.execute(
                "INSERT OR IGNORE INTO category_stats (category, trending_score) VALUES (?, 0.0)",
                (cat,),
            )
        # Insert fixture news rows
        for art in FIXTURE_ARTICLES:
            conn.execute(
                """INSERT OR IGNORE INTO news
                   (news_id, category, abstract, url)
                   VALUES (:news_id, :category, :abstract, :url)""",
                art,
            )


def _neutral_state() -> list[float]:
    return [0.5] * 11


def run_verification() -> None:
    with tempfile.TemporaryDirectory(prefix="neurofeed_rl_verify_") as tmpdir:
        tmp = Path(tmpdir)
        db_path    = tmp / "test.sqlite"
        model_dir  = tmp / "models"
        model_dir.mkdir()

        # ── bootstrap DB ───────────────────────────────────────────────────
        init_db(db_path)
        _insert_fixtures(db_path)

        # ── create agent ───────────────────────────────────────────────────
        agent = DDQNAgent(
            categories=CATEGORIES,
            user_id="verify_user",
            db_path=db_path,
            model_dir=model_dir,
            batch_size=16,
            save_interval=20,
        )
        tech_idx = agent.categories.index(TECH_CAT)
        biz_idx  = agent.categories.index(BIZ_CAT)

        # ── header ─────────────────────────────────────────────────────────
        print()
        print("=" * 105)
        print("  NeuroFeed — RL Training Verification")
        print("=" * 105)
        print(
            f"{'#':>3}  {'buf':>4}  {'grad?':>5}  {'loss':>10}  "
            f"{'ε':>6}  {'step':>4}  {'sync?':>5}  "
            f"{'‖W‖bef':>9}  {'‖W‖aft':>9}  "
            f"{'Q_tech':>8}  {'Q_biz':>8}"
        )
        print("-" * 105)

        warmup_interaction: int | None = None
        first_sync_interaction: int | None = None

        for i in range(1, N_INTERACTIONS + 1):
            liked = (i % 2 == 1)          # odd → like technology
            category = TECH_CAT if liked else BIZ_CAT
            reward   = 2.0 if liked else -2.0

            state      = _neutral_state()
            next_state = _neutral_state()

            norm_before = _weight_norm(agent)
            q_tech_bef, q_biz_bef = _q_vals(agent)
            step_before = agent.step
            buf_before  = len(agent.replay)

            loss = agent.observe(state, category, reward, next_state, done=False, liked=liked)

            norm_after  = _weight_norm(agent)
            q_tech_aft, q_biz_aft = _q_vals(agent)
            step_after  = agent.step

            grad_happened = (step_after > step_before)
            target_synced = grad_happened and (step_after % agent.target_update == 0)

            if grad_happened and warmup_interaction is None:
                warmup_interaction = i
            if target_synced and first_sync_interaction is None:
                first_sync_interaction = i

            print(
                f"{i:>3}  {buf_before + 1:>4}  {'YES' if grad_happened else '   ':>5}  "
                f"{loss if loss is not None else float('nan'):>10.6f}  "
                f"{agent.epsilon:>6.4f}  {step_after:>4d}  "
                f"{'SYNC' if target_synced else '    ':>5}  "
                f"{norm_before:>9.4f}  {norm_after:>9.4f}  "
                f"{q_tech_aft:>8.4f}  {q_biz_aft:>8.4f}"
            )

        print("=" * 105)
        print()

        # ── summary ────────────────────────────────────────────────────────
        print("SUMMARY")
        print("─" * 60)
        print(f"  Interactions simulated:        {N_INTERACTIONS}")
        print(f"  Warmup ends at interaction:    {warmup_interaction}  (expected: 16)")
        print(f"  First target sync at step:     {first_sync_interaction}")
        print(f"  Final DDQN gradient step:      {agent.step}")
        print(f"  Final ε:                       {agent.epsilon:.4f}")
        print(f"  Final replay buffer size:      {len(agent.replay)}")
        print(f"  Final last_loss:               {agent.last_loss}")
        print()

        # Q-value drift analysis
        q_tech_final, q_biz_final = _q_vals(agent)
        print("Q-value direction check (liked=tech, disliked=biz):")
        print(f"  Q(technology) = {q_tech_final:+.4f}  {'✓ POSITIVE' if q_tech_final > 0 else '✗ EXPECTED POSITIVE'}")
        print(f"  Q(business)   = {q_biz_final:+.4f}  {'✓ NEGATIVE' if q_biz_final < 0 else '✗ EXPECTED NEGATIVE'}")
        print(f"  Q(tech) > Q(biz): {'✓' if q_tech_final > q_biz_final else '✗ FAIL'}")
        print()

        # ── checkpoint round-trip ──────────────────────────────────────────
        print("CHECKPOINT ROUND-TRIP TEST")
        print("─" * 60)
        agent.flush_save()
        print(f"  Checkpoint saved to: {agent.model_path}")

        # Load fresh agent from same model_dir
        agent2 = DDQNAgent(
            categories=CATEGORIES,
            user_id="verify_user",
            db_path=db_path,
            model_dir=model_dir,
            batch_size=16,
        )
        q_tech_2, q_biz_2 = _q_vals(agent2)
        q_diff = abs(q_tech_2 - q_tech_final) + abs(q_biz_2 - q_biz_final)
        print(f"  Original Q(tech)={q_tech_final:+.6f}  Q(biz)={q_biz_final:+.6f}")
        print(f"  Reloaded Q(tech)={q_tech_2:+.6f}   Q(biz)={q_biz_2:+.6f}")
        print(f"  Total Q-value delta (should be ~0): {q_diff:.8f}")
        print(f"  Checkpoint reload: {'✓ PASS' if q_diff < 1e-4 else '✗ FAIL'}")
        print(f"  Step restored: {agent2.step} (expected {agent.step})")
        print(f"  ε restored:    {agent2.epsilon:.4f} (expected {agent.epsilon:.4f})")
        print()

        # ── recommendation mix ─────────────────────────────────────────────
        print("RECOMMENDATION MIX (Q-value ranking over 20 independent calls):")
        print("─" * 60)
        state = _neutral_state()
        tech_wins = 0
        for _ in range(20):
            qs = agent.q_values(state)
            if qs[tech_idx] > qs[biz_idx]:
                tech_wins += 1
        biz_wins = 20 - tech_wins
        bar_t = "█" * tech_wins
        bar_b = "█" * biz_wins
        print(f"  technology  {bar_t}  ({tech_wins}/20)")
        print(f"  business    {bar_b}  ({biz_wins}/20)")
        print()

        # ── final verdict ──────────────────────────────────────────────────
        passed = (
            warmup_interaction == 16
            and q_tech_final > q_biz_final
            and q_diff < 1e-4
            and agent2.step == agent.step
        )
        print("=" * 60)
        print(f"  OVERALL: {'✓  ALL CHECKS PASSED' if passed else '✗  ONE OR MORE CHECKS FAILED'}")
        print("=" * 60)
        print()


if __name__ == "__main__":
    run_verification()
