#!/usr/bin/env python3
"""
compare_gamma.py — Empirical comparison of DDQN discount factor gamma in [0.9, 0.3, 0.0].

Investigates why Q(business) drifts positive despite purely negative rewards:
Bellman equation: target = reward + gamma * max_a' Q(next_s, a') * (1 - done)
When done=False and gamma=0.9, an interaction with negative reward (e.g. -2.0)
inherits positive future value from the high Q(tech) state (e.g. 0.9 * 6.5 = +5.85),
leading to target = -2.0 + 5.85 = +3.85 > 0.
With lower gamma (0.3 or 0.0), future bootstrapping is reduced or eliminated.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "final"
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

import numpy as np
import torch

from database import init_db
from ddqn_agent import DDQNAgent

TECH_CAT = "technology"
BIZ_CAT = "business"
CATEGORIES = [TECH_CAT, BIZ_CAT]
SEEDS = [42, 123, 999]
GAMMAS = [0.9, 0.3, 0.0]
N_INTERACTIONS = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_single(gamma: float, seed: int) -> dict[str, float]:
    set_seed(seed)
    with tempfile.TemporaryDirectory(prefix=f"gamma_cmp_{gamma}_{seed}_") as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "test.sqlite"
        model_dir = tmp / "models"
        model_dir.mkdir()
        init_db(db_path)

        agent = DDQNAgent(
            categories=CATEGORIES,
            user_id="gamma_test_user",
            db_path=db_path,
            model_dir=model_dir,
            gamma=gamma,
            batch_size=16,
            min_replay=8,
            save_interval=100,
        )

        state = [0.5] * 11
        first_inversion_interaction = None

        for i in range(1, N_INTERACTIONS + 1):
            liked = (i % 2 == 1)
            cat = TECH_CAT if liked else BIZ_CAT
            reward = 2.0 if liked else -2.0

            agent.observe(state, cat, reward, state, done=False, liked=liked)

            qs = agent.q_values(state)
            q_t, q_b = float(qs[0]), float(qs[1])
            if q_t > q_b and first_inversion_interaction is None and i >= 8:
                first_inversion_interaction = i

        qs_final = agent.q_values(state)
        q_tech_final = float(qs_final[0])
        q_biz_final = float(qs_final[1])
        gap = q_tech_final - q_biz_final

        # Recommendation mix over 20 calls
        tech_picks = 0
        for _ in range(20):
            act = agent.choose_action(state)
            if agent.categories[act] == TECH_CAT:
                tech_picks += 1

        return {
            "gamma": gamma,
            "seed": seed,
            "q_tech": q_tech_final,
            "q_biz": q_biz_final,
            "gap": gap,
            "first_inversion": float(first_inversion_interaction or N_INTERACTIONS),
            "tech_rec_pct": (tech_picks / 20.0) * 100.0,
        }


def main() -> None:
    print("=" * 80)
    print("  DDQN Gamma Comparison: [0.9, 0.3, 0.0] across seeds [42, 123, 999]")
    print("=" * 80)
    print(f"{'gamma':>6}  {'seed':>5}  {'Q(tech)':>10}  {'Q(biz)':>10}  {'Q-gap':>10}  {'1st-win':>8}  {'tech-rec%':>10}")
    print("-" * 80)

    results_by_gamma: dict[float, list[dict[str, float]]] = {g: [] for g in GAMMAS}

    for g in GAMMAS:
        for s in SEEDS:
            res = run_single(g, s)
            results_by_gamma[g].append(res)
            print(
                f"{g:>6.1f}  {s:>5d}  {res['q_tech']:>10.4f}  {res['q_biz']:>10.4f}  "
                f"{res['gap']:>10.4f}  {int(res['first_inversion']):>8d}  {res['tech_rec_pct']:>9.1f}%"
            )

    print("=" * 80)
    print("\nSUMMARY BY GAMMA (mean across 3 seeds):")
    print(f"{'gamma':>6}  {'mean Q(tech)':>14}  {'mean Q(biz)':>14}  {'mean Q-gap':>12}  {'mean win step':>14}  {'tech rec%':>10}")
    print("-" * 80)
    for g in GAMMAS:
        res_list = results_by_gamma[g]
        m_tech = sum(r["q_tech"] for r in res_list) / len(res_list)
        m_biz = sum(r["q_biz"] for r in res_list) / len(res_list)
        m_gap = sum(r["gap"] for r in res_list) / len(res_list)
        m_win = sum(r["first_inversion"] for r in res_list) / len(res_list)
        m_rec = sum(r["tech_rec_pct"] for r in res_list) / len(res_list)
        print(f"{g:>6.1f}  {m_tech:>14.4f}  {m_biz:>14.4f}  {m_gap:>12.4f}  {m_win:>14.1f}  {m_rec:>9.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
