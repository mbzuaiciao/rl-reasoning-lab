"""Tests for policies, trajectories, and transitions."""

from __future__ import annotations

import numpy as np

from rl_reasoning_lab.policies.base import Trajectory, Transition
from rl_reasoning_lab.policies.tabular import (
    DeterministicTabularPolicy,
    EpsilonGreedyPolicy,
    RandomPolicy,
    SoftmaxTabularPolicy,
)


class TestTrajectories:
    def test_discounted_return_calculation(self) -> None:
        traj = Trajectory()
        # 3 transitions with reward 1.0, 2.0, 4.0
        traj.append(Transition(state=0, action=1, reward=1.0, next_state=1, terminated=False))
        traj.append(Transition(state=1, action=1, reward=2.0, next_state=2, terminated=False))
        traj.append(Transition(state=2, action=1, reward=4.0, next_state=3, terminated=True))

        assert len(traj) == 3
        assert traj.total_reward == 7.0

        # With gamma = 0.5:
        # G_0 = 1.0 + 0.5 * 2.0 + 0.25 * 4.0 = 1.0 + 1.0 + 1.0 = 3.0
        assert np.isclose(traj.discounted_return(gamma=0.5), 3.0)

        # Returns at each step:
        # G_2 = 4.0
        # G_1 = 2.0 + 0.5 * 4.0 = 4.0
        # G_0 = 1.0 + 0.5 * 4.0 = 3.0
        returns = traj.returns_at_each_step(gamma=0.5)
        assert np.allclose(returns, [3.0, 4.0, 4.0])


class TestPolicies:
    def test_random_policy(self) -> None:
        policy = RandomPolicy(num_actions=4)
        probs = policy.action_probabilities(state=0)
        assert len(probs) == 4
        assert np.allclose(probs, 0.25)

    def test_deterministic_policy(self) -> None:
        actions = [1, 0, 3, 2]
        policy = DeterministicTabularPolicy(actions=actions, num_actions=4)
        assert policy.sample_action(state=2) == 3
        probs = policy.action_probabilities(state=2)
        assert probs[3] == 1.0
        assert np.sum(probs) == 1.0

    def test_epsilon_greedy_policy(self) -> None:
        # 2 states, 3 actions
        q_table = np.array(
            [
                [1.0, 5.0, 2.0],  # State 0: best action is 1
                [0.0, -1.0, 4.0],  # State 1: best action is 2
            ]
        )
        policy = EpsilonGreedyPolicy(q_table=q_table, epsilon=0.3)

        probs_s0 = policy.action_probabilities(state=0)
        # Greedy action 1 gets 1 - eps + eps/3 = 0.7 + 0.1 = 0.8
        # Other actions get eps/3 = 0.1
        assert np.isclose(probs_s0[1], 0.8)
        assert np.isclose(probs_s0[0], 0.1)
        assert np.isclose(probs_s0[2], 0.1)
        assert np.isclose(np.sum(probs_s0), 1.0)

        # Pure greedy (epsilon=0)
        policy_greedy = EpsilonGreedyPolicy(q_table=q_table, epsilon=0.0)
        assert policy_greedy.sample_action(0) == 1
        assert policy_greedy.sample_action(1) == 2

    def test_softmax_policy(self) -> None:
        q_table = np.array([[0.0, 10.0]])
        # Extremely low temperature -> behaves greedily
        policy_cold = SoftmaxTabularPolicy(q_table=q_table, temperature=0.01)
        probs_cold = policy_cold.action_probabilities(0)
        assert np.isclose(probs_cold[1], 1.0, atol=1e-4)

        # Extremely high temperature -> behaves uniformly
        policy_hot = SoftmaxTabularPolicy(q_table=q_table, temperature=1000.0)
        probs_hot = policy_hot.action_probabilities(0)
        assert np.isclose(probs_hot[0], 0.5, atol=1e-2)
        assert np.isclose(probs_hot[1], 0.5, atol=1e-2)
