"""Tests for Phase 3: Proximal Policy Optimization (PPO), RolloutBuffer, and Clipping."""

from __future__ import annotations

import numpy as np
import torch

from rl_reasoning_lab.algorithms.ppo import (
    compute_ppo_policy_loss,
    compute_ppo_value_loss,
    train_ppo,
)
from rl_reasoning_lab.environments.corridor import CorridorEnv
from rl_reasoning_lab.policies.neural import CategoricalPolicy
from rl_reasoning_lab.trainers.rollout import RolloutBuffer
from rl_reasoning_lab.value_functions.neural import NeuralValueFunction


class TestPPOClippingInvariants:
    def test_ratio_one_when_policies_identical(self) -> None:
        log_probs = torch.tensor([-0.5, -1.2, -0.1])
        old_log_probs = torch.tensor([-0.5, -1.2, -0.1])
        advs = torch.tensor([1.0, -1.0, 2.0])

        loss, diag = compute_ppo_policy_loss(log_probs, old_log_probs, advs, clip_eps=0.2)
        assert np.isclose(diag["mean_ratio"], 1.0)
        assert np.isclose(diag["clip_fraction"], 0.0)
        assert np.isclose(diag["approx_kl"], 0.0)

        # Loss should be -mean(advs) because ratio is 1
        assert np.isclose(loss.item(), -advs.mean().item())

    def test_hand_computed_clipping_four_cases(self) -> None:
        """Verify all 4 mathematical combinations of (A > 0 vs A < 0) and (r above vs below clip).

        Let eps = 0.2. Upper clip = 1.2, Lower clip = 0.8.
        """
        eps = 0.2

        # Case 1: Positive Advantage (A = +2.0), Large Ratio (r = 1.5 > 1.2)
        # surr1 = 1.5 * 2.0 = 3.0
        # surr2 = 1.2 * 2.0 = 2.4
        # min(surr1, surr2) = 2.4 (CLIPPED)
        lp1 = torch.tensor([np.log(1.5)])
        old_lp1 = torch.tensor([0.0])
        adv1 = torch.tensor([2.0])
        loss1, diag1 = compute_ppo_policy_loss(lp1, old_lp1, adv1, clip_eps=eps)
        assert np.isclose(loss1.item(), -2.4, atol=1e-5)
        assert diag1["clip_fraction"] == 1.0

        # Case 2: Positive Advantage (A = +2.0), Small Ratio (r = 0.6 < 0.8)
        # surr1 = 0.6 * 2.0 = 1.2
        # surr2 = 0.8 * 2.0 = 1.6
        # min(surr1, surr2) = 1.2 (UNCLIPPED - do not reward making good action less likely!)
        lp2 = torch.tensor([np.log(0.6)])
        adv2 = torch.tensor([2.0])
        loss2, diag2 = compute_ppo_policy_loss(lp2, old_lp1, adv2, clip_eps=eps)
        assert np.isclose(loss2.item(), -1.2, atol=1e-5)
        assert diag2["clip_fraction"] == 1.0  # ratio is outside [0.8, 1.2]

        # Case 3: Negative Advantage (A = -2.0), Large Ratio (r = 1.5 > 1.2)
        # surr1 = 1.5 * (-2.0) = -3.0
        # surr2 = 1.2 * (-2.0) = -2.4
        # min(surr1, surr2) = -3.0 (UNCLIPPED - do not shield policy from penalizing bad actions!)
        # min(-3.0, -2.4) = -3.0; loss = -(-3.0) = 3.0
        adv3 = torch.tensor([-2.0])
        loss3, diag3 = compute_ppo_policy_loss(lp1, old_lp1, adv3, clip_eps=eps)
        assert np.isclose(loss3.item(), 3.0, atol=1e-5)
        assert diag3["clip_fraction"] == 1.0

        # Case 4: Negative Advantage (A = -2.0), Small Ratio (r = 0.6 < 0.8)
        # surr1 = 0.6 * (-2.0) = -1.2
        # surr2 = 0.8 * (-2.0) = -1.6
        # min(-1.2, -1.6) = -1.6 (CLIPPED); loss = -(-1.6) = 1.6
        adv4 = torch.tensor([-2.0])
        loss4, diag4 = compute_ppo_policy_loss(lp2, old_lp1, adv4, clip_eps=eps)
        assert np.isclose(loss4.item(), 1.6, atol=1e-5)
        assert diag4["clip_fraction"] == 1.0

    def test_zero_advantage_contributes_zero_loss(self) -> None:
        log_probs = torch.tensor([-0.2, -0.5])
        old_log_probs = torch.tensor([-0.8, -0.1])
        advs = torch.tensor([0.0, 0.0])

        loss, _ = compute_ppo_policy_loss(log_probs, old_log_probs, advs)
        assert np.isclose(loss.item(), 0.0)

    def test_approximate_kl_against_exact_categorical_kl(self) -> None:
        """Compare sampled approx_kl (log_p_old - log_p_new) with exact analytical KL."""
        policy_old = CategoricalPolicy(num_states=1, num_actions=3, hidden_dims=())
        policy_new = CategoricalPolicy(num_states=1, num_actions=3, hidden_dims=())

        # Give different weights
        with torch.no_grad():
            for p in policy_new.parameters():
                p.add_(0.5)

        dist_old = policy_old.get_distribution(0)
        dist_new = policy_new.get_distribution(0)

        # Exact KL: D_KL(pi_old || pi_new) = sum_a p_old(a) * (log p_old(a) - log p_new(a))
        p_old = dist_old.probs.detach()
        p_new = dist_new.probs.detach()
        exact_kl = torch.sum(p_old * (torch.log(p_old) - torch.log(p_new))).item()

        # Sampled approx_kl over all actions weighted by p_old
        log_p_old = dist_old.log_prob(torch.arange(3))
        log_p_new = dist_new.log_prob(torch.arange(3))
        sampled_kl = torch.sum(p_old * (log_p_old - log_p_new)).item()

        assert np.isclose(exact_kl, sampled_kl, atol=1e-5)


class TestRolloutBuffer:
    def test_frozen_old_log_probs_invariant(self) -> None:
        """Guarantee old_log_prob is detached and does not change when policy updates."""
        policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=())
        optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)

        buffer = RolloutBuffer(num_states=3)
        _, log_prob = policy.sample_action_with_log_prob(0)
        initial_lp_val = float(log_prob.item())

        buffer.add(
            state=0,
            action=1,
            reward=1.0,
            done=True,
            old_log_prob=log_prob,
            value=0.5,
        )
        buffer.finalize(next_value=0.0)

        # Perform violent policy update
        optimizer.zero_grad()
        loss = policy(policy._state_to_tensor(0)).sum()
        loss.backward()
        optimizer.step()

        # Check buffer value is completely unchanged
        assert buffer.old_log_probs[0] == initial_lp_val

    def test_minibatch_tensor_alignment(self) -> None:
        buffer = RolloutBuffer(num_states=4)
        for i in range(10):
            buffer.add(
                state=i % 4,
                action=i % 2,
                reward=float(i),
                done=(i == 9),
                old_log_prob=-0.5,
                value=float(i * 0.1),
            )
        buffer.finalize(next_value=0.0)

        for batch in buffer.get_minibatch_generator(batch_size=4, shuffle=True):
            states = batch["states"]
            actions = batch["actions"]
            returns = batch["returns"]
            assert states.shape[0] <= 4
            assert len(actions) == len(states)
            assert len(returns) == len(states)

    def test_masking_drops_padded_elements(self) -> None:
        log_probs = torch.tensor([np.log(1.5), np.log(1.5)])
        old_log_probs = torch.tensor([0.0, 0.0])
        advs = torch.tensor([2.0, 2.0])
        # Second element is padded (mask = 0.0)
        masks = torch.tensor([1.0, 0.0])

        loss, diag = compute_ppo_policy_loss(
            log_probs, old_log_probs, advs, clip_eps=0.2, masks=masks
        )
        # Only element 0 counts: -min(1.5 * 2, 1.2 * 2) = -2.4
        assert np.isclose(loss.item(), -2.4, atol=1e-5)


class TestPPOTraining:
    def test_ppo_learns_corridor(self) -> None:
        env = CorridorEnv(length=3, goal_reward=1.0, step_cost=-0.05)
        policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=(16,))
        value_net = NeuralValueFunction(num_states=3, hidden_dims=(16,))

        res = train_ppo(
            env=env,
            policy=policy,
            value_net=value_net,
            total_timesteps=1500,
            rollout_steps=128,
            epochs_per_rollout=4,
            minibatch_size=32,
            policy_lr=5e-3,
            value_lr=1e-2,
            clip_eps=0.2,
            seed=42,
        )

        assert len(res.episode_returns) > 5
        late_return = np.mean(res.episode_returns[-20:])
        assert late_return > 0.5, f"PPO failed to learn corridor (late return: {late_return})"

    def test_value_loss_clipping(self) -> None:
        values = torch.tensor([2.0])
        old_values = torch.tensor([1.0])
        returns = torch.tensor([3.0])

        # Without clipping: 0.5 * (2.0 - 3.0)^2 = 0.5 * 1.0 = 0.5
        loss_unclipped = compute_ppo_value_loss(values, old_values, returns, value_clip_eps=None)
        assert np.isclose(loss_unclipped.item(), 0.5)

        # With clip_eps = 0.2:
        # v_clipped = 1.0 + clip(2.0 - 1.0, -0.2, 0.2) = 1.2
        # loss_clipped = (1.2 - 3.0)^2 = (-1.8)^2 = 3.24
        # max(1.0, 3.24) * 0.5 = 1.62
        loss_clipped = compute_ppo_value_loss(values, old_values, returns, value_clip_eps=0.2)
        assert np.isclose(loss_clipped.item(), 1.62, atol=1e-5)
