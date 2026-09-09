"""Tests for Phase 2: Policy Gradients, Neural Policies, Baselines, and GAE."""

from __future__ import annotations

import numpy as np
import torch

from rl_reasoning_lab.algorithms.advantage import (
    compute_full_returns,
    compute_gae,
    compute_reward_to_go,
    normalize_advantages,
)
from rl_reasoning_lab.algorithms.policy_gradient import actor_critic, reinforce
from rl_reasoning_lab.environments.corridor import CorridorEnv
from rl_reasoning_lab.policies.neural import CategoricalPolicy
from rl_reasoning_lab.value_functions.neural import NeuralValueFunction


class TestAdvantageAndReturns:
    def test_returns_hand_computed(self) -> None:
        # Rewards: [1.0, 2.0, 3.0], gamma = 0.5
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.5

        # Reward-to-go:
        # G_2 = 3.0
        # G_1 = 2.0 + 0.5 * 3.0 = 3.5
        # G_0 = 1.0 + 0.5 * 3.5 = 2.75
        rtg = compute_reward_to_go(rewards, gamma=gamma)
        expected_rtg = np.array([2.75, 3.5, 3.0], dtype=np.float32)
        assert np.allclose(rtg, expected_rtg)

        # Full return: assigns G_0 = 2.75 to all steps
        full = compute_full_returns(rewards, gamma=gamma)
        assert np.allclose(full, np.array([2.75, 2.75, 2.75], dtype=np.float32))

    def test_gae_lambda_extremes(self) -> None:
        """Verify GAE mathematical properties at lambda=0 and lambda=1."""
        rewards = [1.0, 2.0, 4.0]
        values = [0.5, 1.5, 2.0]
        next_values = [1.5, 2.0, 0.0]  # Step 2 terminates (next_value=0)
        dones = [False, False, True]
        gamma = 0.9

        # When lambda = 0, GAE advantage is precisely the 1-step TD error delta_t
        # delta_0 = 1.0 + 0.9 * 1.5 - 0.5 = 1.0 + 1.35 - 0.5 = 1.85
        # delta_1 = 2.0 + 0.9 * 2.0 - 1.5 = 2.0 + 1.80 - 1.5 = 2.30
        # delta_2 = 4.0 + 0.9 * 0.0 * 0.0 - 2.0 = 4.0 - 2.0 = 2.00
        adv_lambda0, _ = compute_gae(rewards, values, next_values, dones, gamma=gamma, lambda_=0.0)
        expected_delta = np.array([1.85, 2.30, 2.00], dtype=np.float32)
        assert np.allclose(adv_lambda0, expected_delta, atol=1e-5)

        # When lambda = 1 and terminal state has value 0, GAE is Monte Carlo advantage G_t - V(s_t)
        # G_2 = 4.0; A_2 = 4.0 - 2.0 = 2.0
        # G_1 = 2.0 + 0.9 * 4.0 = 5.6; A_1 = 5.6 - 1.5 = 4.1
        # G_0 = 1.0 + 0.9 * 5.6 = 6.04; A_0 = 6.04 - 0.5 = 5.54
        adv_lambda1, _ = compute_gae(rewards, values, next_values, dones, gamma=gamma, lambda_=1.0)
        expected_mc_adv = np.array([5.54, 4.10, 2.00], dtype=np.float32)
        assert np.allclose(adv_lambda1, expected_mc_adv, atol=1e-5)

    def test_gae_terminal_masking(self) -> None:
        """Verify that dones=True masks out next_value and stops backward recursion."""
        rewards = [1.0]
        values = [2.0]
        next_values = [100.0]  # Erroneous future value
        dones = [True]  # Terminated episode
        gamma = 0.99

        # Since done=True, next_value should be multiplied by (1 - done) = 0
        adv, targets = compute_gae(rewards, values, next_values, dones, gamma=gamma, lambda_=0.95)
        # delta = 1.0 + 0.99 * 100 * 0 - 2.0 = -1.0
        assert np.isclose(adv[0], -1.0)
        assert np.isclose(targets[0], 1.0)

    def test_normalize_advantages(self) -> None:
        adv = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        norm = normalize_advantages(adv)
        assert np.isclose(np.mean(norm), 0.0, atol=1e-6)
        assert np.isclose(np.std(norm), 1.0, atol=1e-6)


class TestCategoricalPolicy:
    def test_probabilities_sum_and_entropy(self) -> None:
        policy = CategoricalPolicy(num_states=4, num_actions=3, hidden_dims=(16,))
        info = policy.inspect_state(0)

        probs = info["probabilities"]
        assert len(probs) == 3
        assert np.isclose(np.sum(probs), 1.0, atol=1e-6)
        assert np.all(probs >= 0.0)

        # Theoretical Shannon entropy: -sum(p * log(p))
        expected_entropy = -np.sum(probs * np.log(probs + 1e-12))
        assert np.isclose(info["entropy"], expected_entropy, atol=1e-4)

    def test_sampling_and_log_prob(self) -> None:
        policy = CategoricalPolicy(num_states=2, num_actions=4)
        action, log_prob = policy.sample_action_with_log_prob(1)
        assert 0 <= action < 4
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.requires_grad

        # Check log_prob matches manual log of probability
        info = policy.inspect_state(1)
        prob = info["probabilities"][action]
        assert np.isclose(log_prob.item(), np.log(prob), atol=1e-5)


class TestPolicyGradientInvariants:
    def test_correct_loss_sign_improves_action(self) -> None:
        """Fundamental Sign Invariant:

        If action a* receives a positive reward / positive advantage,
        minimizing L = - A * log pi(a*) MUST increase pi(a*).
        """
        policy = CategoricalPolicy(num_states=1, num_actions=2, hidden_dims=())
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

        # Initial probability of action 1
        p_initial = policy.action_probabilities(0)[1]

        # Perform one policy gradient step with positive advantage for action 1
        optimizer.zero_grad()
        _, log_prob = policy.sample_action_with_log_prob(0)
        # Force selecting action 1
        dist = policy.get_distribution(0)
        log_prob_act1 = dist.log_prob(torch.tensor(1))
        positive_advantage = torch.tensor(5.0)

        loss = -positive_advantage * log_prob_act1
        loss.backward()
        optimizer.step()

        p_updated = policy.action_probabilities(0)[1]
        assert p_updated > p_initial, (
            f"Expected action probability to increase from {p_initial:.4f}, but got {p_updated:.4f}. "
            "Likely incorrect loss sign convention!"
        )

    def test_baseline_independence_zero_expectation(self) -> None:
        """Mathematical Invariant:

        E_{a ~ pi}[ grad_theta log pi(a|s) * b(s) ] = 0.
        Subtracting a constant baseline does not bias the gradient.
        """
        torch.manual_seed(42)
        policy = CategoricalPolicy(num_states=1, num_actions=3, hidden_dims=())
        dist = policy.get_distribution(0)
        probs = dist.probs.detach()

        # Compute expectation: sum_a pi(a) * grad log pi(a)
        total_grad = [torch.zeros_like(p) for p in policy.parameters()]

        for a in range(3):
            policy.zero_grad()
            dist = policy.get_distribution(0)
            log_p = dist.log_prob(torch.tensor(a))
            log_p.backward()

            p_a = probs[a].item()
            for i, p in enumerate(policy.parameters()):
                assert p.grad is not None
                total_grad[i] += p_a * p.grad.data

        # The weighted sum of score functions must be zero to machine precision
        for g in total_grad:
            assert torch.allclose(g, torch.zeros_like(g), atol=1e-5), (
                "Expectation of score function is not zero!"
            )


class TestAlgorithmsTraining:
    def test_reinforce_learns_short_corridor(self) -> None:
        """REINFORCE should learn the rightward policy on a small 3-state corridor."""
        env = CorridorEnv(length=3, goal_reward=1.0, step_cost=-0.05, slip_prob=0.0)
        policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=(16,))
        value_net = NeuralValueFunction(num_states=3, hidden_dims=(16,))

        res = reinforce(
            env,
            policy,
            value_net=value_net,
            policy_lr=1e-2,
            value_lr=1e-2,
            num_episodes=200,
            gamma=0.95,
            baseline_type="learned_value",
            entropy_coef=0.01,
            seed=42,
        )

        late_return = np.mean(res.episode_returns[-50:])
        assert late_return > 0.5, (
            f"REINFORCE failed to solve 3-state corridor (late return: {late_return})"
        )

    def test_actor_critic_learns_short_corridor(self) -> None:
        """Actor-Critic should solve a small corridor environment."""
        env = CorridorEnv(length=3, goal_reward=1.0, step_cost=-0.05, slip_prob=0.0)
        policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=(16,))
        value_net = NeuralValueFunction(num_states=3, hidden_dims=(16,))

        res = actor_critic(
            env,
            policy,
            value_net=value_net,
            policy_lr=1e-2,
            value_lr=1e-2,
            num_episodes=200,
            gamma=0.95,
            entropy_coef=0.01,
            seed=42,
        )

        late_return = np.mean(res.episode_returns[-50:])
        assert late_return > 0.5, (
            f"Actor-Critic failed to solve 3-state corridor (late return: {late_return})"
        )
