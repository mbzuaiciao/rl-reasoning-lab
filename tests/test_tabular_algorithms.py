"""Tests for dynamic programming and temporal difference learning algorithms."""

from __future__ import annotations

import numpy as np

from rl_reasoning_lab.algorithms.dynamic_programming import (
    policy_evaluation,
    value_iteration,
)
from rl_reasoning_lab.algorithms.temporal_difference import q_learning, sarsa
from rl_reasoning_lab.environments.corridor import CorridorEnv
from rl_reasoning_lab.environments.gridworld import GridworldEnv
from rl_reasoning_lab.policies.tabular import RandomPolicy


class TestDynamicProgramming:
    def test_value_iteration_analytic_corridor(self) -> None:
        """In a deterministic corridor with goal reward 1.0 and zero step cost:

        V*(s) = gamma^(length - 1 - s) for s < goal_state.
        """
        length = 4
        gamma = 0.9
        env = CorridorEnv(length=length, goal_reward=1.0, step_cost=0.0, slip_prob=0.0)

        result = value_iteration(env, gamma=gamma, theta=1e-10)
        v = result.value_function.as_array()

        # Expected:
        # State 2 -> goal (3): reward 1.0, future 0 => 1.0
        # State 1 -> state 2: reward 0.0 + 0.9 * 1.0 => 0.9
        # State 0 -> state 1: reward 0.0 + 0.9 * 0.9 => 0.81
        # State 3 (goal): terminal absorbing => 0.0
        expected_v = np.array([0.81, 0.90, 1.00, 0.00])
        assert np.allclose(v, expected_v, atol=1e-5)

        # Optimal policy must choose RIGHT (action 1) for all non-terminal states
        optimal_policy = result.policy
        assert optimal_policy is not None
        for s in range(length - 1):
            assert optimal_policy.sample_action(s) == CorridorEnv.ACTION_RIGHT

    def test_policy_evaluation_against_matrix_inversion(self) -> None:
        """Verify that iterative policy evaluation converges to (I - gamma * P)^(-1) * R."""
        length = 4
        gamma = 0.8
        env = CorridorEnv(length=length, goal_reward=1.0, step_cost=-0.1, slip_prob=0.1)

        # Uniform random policy
        policy = RandomPolicy(num_actions=2)

        # Run iterative policy evaluation
        result = policy_evaluation(env, policy, gamma=gamma, theta=1e-10)
        v_iterative = result.value_function.as_array()

        # Build analytic transition matrix P_pi and reward vector R_pi
        num_states = env.num_states
        p_pi = np.zeros((num_states, num_states))
        r_pi = np.zeros(num_states)

        for s in range(num_states):
            probs_a = policy.action_probabilities(s)
            for a in range(env.num_actions):
                for p_trans, next_s, rew, is_terminal in env.get_transitions(s, a):
                    p_eff = probs_a[a] * p_trans
                    r_pi[s] += p_eff * rew
                    if not is_terminal:
                        p_pi[s, next_s] += p_eff

        # Solve (I - gamma * P_pi) * V = R_pi
        identity = np.eye(num_states)
        v_analytic = np.linalg.solve(identity - gamma * p_pi, r_pi)

        assert np.allclose(v_iterative, v_analytic, atol=1e-5)

    def test_value_iteration_on_gridworld(self) -> None:
        """Verify value iteration converges to a sensible path on simple grid."""
        env = GridworldEnv.make_simple_grid()
        result = value_iteration(env, gamma=0.95, theta=1e-7)
        v = result.value_function
        policy = result.policy

        # Goal is at bottom right (3, 3) = state 15
        # Start is at top left (0, 0) = state 0
        assert policy is not None
        # Start value should be strictly positive (can reach goal with reward 1.0 - path costs)
        assert v[0] > 0.0
        # Goal state value is 0.0 (absorbing terminal)
        goal_state = env.pos_to_state((3, 3))
        assert np.isclose(v[goal_state], 0.0)


class TestTemporalDifference:
    def test_q_learning_learns_corridor(self) -> None:
        """Q-learning should find the optimal policy on a small corridor."""
        env = CorridorEnv(length=4, goal_reward=1.0, step_cost=-0.01, slip_prob=0.0)
        result = q_learning(
            env,
            num_episodes=200,
            alpha=0.1,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.97,
            seed=42,
        )

        greedy_actions = result.q_function.extract_greedy_policy()
        # For non-terminal states 0, 1, 2, optimal action is RIGHT (1)
        for s in range(3):
            assert greedy_actions[s] == CorridorEnv.ACTION_RIGHT

    def test_sarsa_vs_q_learning_cliff_walking(self) -> None:
        """On Cliff Walking, Q-learning learns the optimal (cliff-edge) path,

        while SARSA learns the safer path further from the cliff due to on-policy exploration risk.
        """
        env_sarsa = GridworldEnv.make_cliff_grid()
        env_ql = GridworldEnv.make_cliff_grid()

        res_sarsa = sarsa(
            env_sarsa,
            num_episodes=600,
            alpha=0.1,
            gamma=0.99,
            epsilon_start=0.3,
            epsilon_end=0.1,
            epsilon_decay=0.998,
            seed=123,
        )

        res_ql = q_learning(
            env_ql,
            num_episodes=600,
            alpha=0.1,
            gamma=0.99,
            epsilon_start=0.3,
            epsilon_end=0.1,
            epsilon_decay=0.998,
            seed=123,
        )

        # In cliff walking, during training with exploration, SARSA avoids falling off the cliff
        # by taking the safer path, resulting in higher average training reward over late episodes.
        sarsa_late_reward = np.mean(res_sarsa.episode_returns[-100:])
        ql_late_reward = np.mean(res_ql.episode_returns[-100:])

        # SARSA on-policy training return should be significantly higher (less negative) than Q-learning
        assert sarsa_late_reward > ql_late_reward
