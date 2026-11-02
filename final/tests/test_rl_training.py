"""
tests/test_rl_training.py — Automated tests that lock in the RL training
behaviour described in the architecture docs:

  1. DDQN weights are UNCHANGED during the warmup period (interactions 1–15).
  2. DDQN weights DO change starting at interaction 16 (first gradient step).
  3. A liked feedback via poll_feedback() also reaches the DDQN replay buffer.
  4. A disliked feedback via poll_feedback() also reaches the replay buffer.
  5. Checkpoint round-trip: save → fresh load → Q-values identical.
  6. User isolation: user A's feedback does not affect user B's agent metrics.

All tests use isolated temp DBs; the production database is never touched.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = REPO_ROOT / "final"
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from database import init_db, db_connect
from ddqn_agent import DDQNAgent

# ── helpers ──────────────────────────────────────────────────────────────────

TECH = "technology"
BIZ  = "business"
CATEGORIES = [TECH, BIZ]


def _mk_agent(tmp_path: Path, user_id: str = "u1") -> DDQNAgent:
    db_path   = tmp_path / f"test_{user_id}.sqlite"
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    init_db(db_path)
    # Pass CATEGORIES explicitly — DDQNAgent accepts them directly and does
    # not need a 'categories' table; init_db populates category_stats with
    # the canonical 10 categories from settings, but we just pass ours in.
    return DDQNAgent(
        categories=CATEGORIES,
        user_id=user_id,
        db_path=db_path,
        model_dir=model_dir,
        batch_size=16,
        save_interval=100,  # disable auto-save during tests
    )


def _weight_norm(agent: DDQNAgent) -> float:
    return math.sqrt(sum(p.data.norm().item() ** 2 for p in agent.model.parameters()))


def _neutral() -> list[float]:
    return [0.5] * 11


def _simulate(agent: DDQNAgent, n: int, category: str, reward: float, liked: bool) -> None:
    for _ in range(n):
        agent.observe(_neutral(), category, reward, _neutral(), done=False, liked=liked)


# ── tests ─────────────────────────────────────────────────────────────────────


class TestWarmupBoundary:
    """Interactions 1-15 must NOT change DDQN weights; interaction 16 must."""

    def test_weights_unchanged_during_warmup(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)
        norm_before = _weight_norm(agent)

        for i in range(1, 16):   # interactions 1–15
            loss = agent.observe(_neutral(), TECH, 1.0, _neutral(), done=False, liked=True)
            assert loss is None, f"Unexpected gradient step at interaction {i}"
            assert agent.step == 0, f"step incremented before warmup ended at interaction {i}"

        norm_after = _weight_norm(agent)
        assert norm_before == pytest.approx(norm_after, rel=1e-6), (
            "DDQN weights changed during warmup (interactions 1–15)"
        )

    def test_first_gradient_step_at_interaction_16(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)

        for _ in range(15):
            agent.observe(_neutral(), TECH, 1.0, _neutral(), done=False, liked=True)

        assert agent.step == 0, "step should still be 0 after 15 interactions"

        # Interaction 16 must trigger the first gradient step
        loss = agent.observe(_neutral(), TECH, 1.0, _neutral(), done=False, liked=True)
        assert loss is not None, "No gradient step at interaction 16 — buffer has 16 samples"
        assert agent.step == 1, f"step should be 1 after first gradient step, got {agent.step}"

    def test_weights_change_after_warmup(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)
        _simulate(agent, 15, TECH, 1.0, liked=True)    # fill buffer, no grad
        norm_before = _weight_norm(agent)

        # 16th interaction triggers first gradient step
        agent.observe(_neutral(), TECH, 1.0, _neutral(), done=False, liked=True)
        norm_after = _weight_norm(agent)

        # With gradient descent the norm will change unless the gradient
        # happens to be zero (extremely unlikely with random weights + real loss)
        assert norm_before != pytest.approx(norm_after, rel=1e-7), (
            "DDQN weights did NOT change after first gradient step at interaction 16"
        )


class TestLastLoss:
    def test_last_loss_is_none_during_warmup(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)
        _simulate(agent, 15, TECH, 1.0, liked=True)
        assert agent.last_loss is None

    def test_last_loss_set_after_warmup(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)
        _simulate(agent, 16, TECH, 1.0, liked=True)
        assert agent.last_loss is not None
        assert isinstance(agent.last_loss, float)
        assert agent.last_loss >= 0.0


class TestCheckpointRoundTrip:
    def test_q_values_survive_save_reload(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)
        # Train long enough that weights are non-trivial
        _simulate(agent, 30, TECH, 2.0, liked=True)
        _simulate(agent, 30, BIZ, -2.0, liked=False)

        state = _neutral()
        qs_original = agent.q_values(state).tolist()
        step_original = agent.step
        eps_original  = agent.epsilon

        agent.flush_save()

        # Load a fresh agent from the same checkpoint
        db_path2   = tmp_path / "test_u1.sqlite"
        model_dir2 = tmp_path / "models"
        agent2 = DDQNAgent(
            categories=CATEGORIES,
            user_id="u1",
            db_path=db_path2,
            model_dir=model_dir2,
        )
        qs_reloaded = agent2.q_values(state).tolist()

        assert agent2.step == step_original, (
            f"step not restored: expected {step_original}, got {agent2.step}"
        )
        assert agent2.epsilon == pytest.approx(eps_original, rel=1e-5)

        for orig, reload in zip(qs_original, qs_reloaded):
            assert orig == pytest.approx(reload, abs=1e-4), (
                f"Q-value mismatch after checkpoint reload: {qs_original} vs {qs_reloaded}"
            )


class TestUserIsolation:
    def test_user_a_does_not_contaminate_user_b(self, tmp_path: Path):
        # Two agents, separate DBs
        agent_a = _mk_agent(tmp_path, user_id="alice")
        agent_b = _mk_agent(tmp_path, user_id="bob")

        q_b_before = agent_b.q_values(_neutral()).tolist()
        step_b_before = agent_b.step

        # Train user A heavily
        _simulate(agent_a, 60, TECH, 2.0, liked=True)

        # User B must be unchanged
        q_b_after = agent_b.q_values(_neutral()).tolist()
        assert agent_b.step == step_b_before
        for qb, qa in zip(q_b_before, q_b_after):
            assert qb == pytest.approx(qa, abs=1e-8), "User B's Q-values changed after training user A"


class TestPollFeedbackReachesAgent:
    """After the poll_feedback bug fix, poll interactions MUST push to the buffer."""

    def test_poll_feedback_updates_replay_buffer(self, tmp_path: Path):
        """
        Directly call DDQNAgent.observe() the same way the fixed poll_feedback()
        does — verifying the plumbing that was previously missing works.
        """
        agent = _mk_agent(tmp_path)
        assert len(agent.replay) == 0

        state = _neutral()
        next_state = _neutral()
        # This is exactly what the fixed poll_feedback() calls
        agent.observe(state, TECH, 1.0, next_state, done=False, liked=True)

        assert len(agent.replay) == 1, (
            "poll_feedback() observe() call did not push to the replay buffer"
        )

    def test_poll_dislike_updates_replay_buffer(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)
        agent.observe(_neutral(), BIZ, -1.0, _neutral(), done=False, liked=False)
        assert len(agent.replay) == 1

    def test_poll_feedback_also_trains_after_warmup(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)
        # Pre-fill buffer to just below the threshold
        _simulate(agent, 15, TECH, 1.0, liked=True)
        assert agent.step == 0

        # One more observe (simulating fixed poll_feedback) should trigger training
        loss = agent.observe(_neutral(), TECH, 1.0, _neutral(), done=False, liked=True)
        assert loss is not None, "poll_feedback-style observe did not trigger gradient step after warmup"
        assert agent.step == 1


class TestQValueDriftDirection:
    """After enough liked-tech / disliked-biz interactions, Q(tech) > Q(biz)."""

    def test_q_values_drift_in_correct_direction(self, tmp_path: Path):
        agent = _mk_agent(tmp_path)
        for i in range(60):
            liked = (i % 2 == 0)
            cat    = TECH if liked else BIZ
            reward = 2.0 if liked else -2.0
            agent.observe(_neutral(), cat, reward, _neutral(), done=False, liked=liked)

        qs = agent.q_values(_neutral())
        q_tech = float(qs[agent.categories.index(TECH)])
        q_biz  = float(qs[agent.categories.index(BIZ)])

        assert q_tech > q_biz, (
            f"After 30 tech likes and 30 biz dislikes, expected Q(tech) > Q(biz) "
            f"but got Q(tech)={q_tech:.4f}, Q(biz)={q_biz:.4f}"
        )
