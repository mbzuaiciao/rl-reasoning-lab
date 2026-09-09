"""Tests for tabular environments: Corridor, Gridworld, Bandit."""

from __future__ import annotations

import numpy as np
import pytest

from rl_reasoning_lab.environments.bandit import MultiArmedBanditEnv
from rl_reasoning_lab.environments.corridor import CorridorEnv
from rl_reasoning_lab.environments.gridworld import GridworldEnv


class TestCorridorEnv:
    def test_initialization_and_reset(self) -> None:
        env = CorridorEnv(length=5)
        assert env.num_states == 5
        assert env.num_actions == 2
        assert env.goal_state == 4

        state, info = env.reset(seed=42)
        assert state == 0
        assert info["state"] == 0

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="length must be at least 2"):
            CorridorEnv(length=1)
        with pytest.raises(ValueError, match="slip_prob"):
            CorridorEnv(length=5, slip_prob=1.5)

    def test_left_wall_boundary(self) -> None:
        env = CorridorEnv(length=5, slip_prob=0.0)
        env.reset()
        outcome = env.step(CorridorEnv.ACTION_LEFT)
        assert outcome.observation == 0
        assert not outcome.terminated
        assert outcome.reward == 0.0

    def test_reach_goal_deterministic(self) -> None:
        env = CorridorEnv(length=4, goal_reward=10.0, step_cost=-1.0, slip_prob=0.0)
        env.reset()
        # Step 1: 0 -> 1
        o1 = env.step(CorridorEnv.ACTION_RIGHT)
        assert o1.observation == 1 and not o1.terminated and o1.reward == -1.0
        # Step 2: 1 -> 2
        o2 = env.step(CorridorEnv.ACTION_RIGHT)
        assert o2.observation == 2 and not o2.terminated and o2.reward == -1.0
        # Step 3: 2 -> 3 (Goal)
        o3 = env.step(CorridorEnv.ACTION_RIGHT)
        assert o3.observation == 3 and o3.terminated and o3.reward == 10.0

    def test_transitions_distribution_sums_to_one(self) -> None:
        for slip in [0.0, 0.2, 0.5]:
            env = CorridorEnv(length=6, slip_prob=slip)
            for s in range(env.num_states):
                for a in range(env.num_actions):
                    trans = env.get_transitions(s, a)
                    prob_sum = sum(p for p, _, _, _ in trans)
                    assert np.isclose(prob_sum, 1.0), f"Probs for s={s}, a={a} sum to {prob_sum}"


class TestGridworldEnv:
    def test_pos_state_conversion(self) -> None:
        env = GridworldEnv.make_simple_grid()
        for r in range(env.height):
            for c in range(env.width):
                s = env.pos_to_state((r, c))
                assert env.state_to_pos(s) == (r, c)

    def test_wall_blocks_movement(self) -> None:
        # Simple grid: (1, 1) is a wall '#'
        env = GridworldEnv.make_simple_grid()
        env.reset()
        # Start is (0, 0). Move right to (0, 1)
        o1 = env.step(GridworldEnv.ACTION_RIGHT)
        assert o1.info["pos"] == (0, 1)
        # Attempt to move down into wall at (1, 1)
        o2 = env.step(GridworldEnv.ACTION_DOWN)
        # Should be blocked and remain at (0, 1)
        assert o2.info["pos"] == (0, 1)

    def test_transitions_probabilities_sum_to_one(self) -> None:
        env = GridworldEnv.make_simple_grid()
        for s in range(env.num_states):
            for a in range(env.num_actions):
                trans = env.get_transitions(s, a)
                prob_sum = sum(p for p, _, _, _ in trans)
                assert np.isclose(prob_sum, 1.0), f"Transitions sum to {prob_sum}"

    def test_cliff_walking_dynamics(self) -> None:
        env = GridworldEnv.make_cliff_grid()
        s, info = env.reset()
        assert info["pos"] == (3, 0)
        # Move right into cliff at (3, 1)
        outcome = env.step(GridworldEnv.ACTION_RIGHT)
        assert outcome.reward == -100.0
        # Because cliff_resets_to_start=True, should be reset to (3, 0)
        assert outcome.info["pos"] == (3, 0)
        assert not outcome.terminated


class TestBanditEnv:
    def test_bandit_means_and_regret(self) -> None:
        means = [0.1, 0.5, 2.0, -1.0]
        env = MultiArmedBanditEnv(num_arms=4, mean_rewards=means, reward_std=0.01, seed=42)
        assert env.optimal_arm == 2
        assert np.isclose(env.optimal_expected_reward, 2.0)

        env.reset()
        # Pull optimal arm
        outcome_opt = env.step(2)
        assert outcome_opt.info["is_optimal"]
        assert np.isclose(outcome_opt.info["immediate_regret"], 0.0)

        # Pull sub-optimal arm
        outcome_sub = env.step(0)
        assert not outcome_sub.info["is_optimal"]
        assert np.isclose(outcome_sub.info["immediate_regret"], 1.9)
