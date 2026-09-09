"""Proximal Policy Optimization (PPO) algorithm implemented from first principles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rl_reasoning_lab.environments.base import Environment
from rl_reasoning_lab.policies.neural import CategoricalPolicy
from rl_reasoning_lab.value_functions.neural import NeuralValueFunction


@dataclass
class PPOResult:
    """Diagnostic training records for PPO."""

    episode_returns: list[float]
    episode_lengths: list[int]
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    approx_kls: list[float] = field(default_factory=list)
    clip_fractions: list[float] = field(default_factory=list)
    mean_ratios: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    epochs_used: list[int] = field(default_factory=list)


def compute_ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute PPO's clipped surrogate policy loss and diagnostic metrics.

    Formula:
        r_t(theta) = exp(log pi_theta(a_t | s_t) - log pi_old(a_t | s_t))
        surr1 = r_t(theta) * A_t
        surr2 = clip(r_t(theta), 1 - eps, 1 + eps) * A_t
        L_clip = - min(surr1, surr2)

    Diagnostics:
        - approx_kl: Sample approximation of D_KL(pi_old || pi_new) ~ E[log pi_old - log pi_new]
        - clip_fraction: Fraction of tokens/steps where |r_t - 1| > eps
        - mean_ratio: Mean importance sampling ratio E[r_t]
        - max_ratio: Max importance sampling ratio
    """
    # 1. Importance sampling ratio: exp(log_p - log_p_old)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # 2. Surrogate objectives
    surr1 = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr2 = clipped_ratio * advantages

    # 3. PPO clipped loss (minimizing negative expected return)
    policy_loss_per_step = -torch.min(surr1, surr2)

    if masks is not None:
        mask_sum = torch.clamp(masks.sum(), min=1.0)
        policy_loss = (policy_loss_per_step * masks).sum() / mask_sum
    else:
        policy_loss = policy_loss_per_step.mean()

    # 4. Diagnostics
    with torch.no_grad():
        # KL(pi_old || pi_new) = E_{pi_old}[log pi_old - log pi_new]
        # or Schulman's non-negative estimator: (ratio - 1) - log_ratio
        approx_kl = (old_log_probs - log_probs).mean().item()
        clipped_steps = (torch.abs(ratio - 1.0) > clip_eps).float()
        if masks is not None:
            clip_fraction = ((clipped_steps * masks).sum() / mask_sum).item()
        else:
            clip_fraction = clipped_steps.mean().item()
        mean_ratio = ratio.mean().item()
        max_ratio = ratio.max().item()

    diagnostics = {
        "approx_kl": float(approx_kl),
        "clip_fraction": float(clip_fraction),
        "mean_ratio": float(mean_ratio),
        "max_ratio": float(max_ratio),
    }

    return policy_loss, diagnostics


def compute_ppo_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    value_clip_eps: float | None = None,
    masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute value function squared error loss with optional PPO value clipping.

    Standard MSE:
        L_vf = 0.5 * (V_phi(s) - return)^2

    Clipped Value Loss (if value_clip_eps is not None):
        V_clipped = V_old + clip(V_phi(s) - V_old, -eps, +eps)
        L_vf = 0.5 * max((V_phi - return)^2, (V_clipped - return)^2)
    """
    v_loss_unclipped = (values - returns) ** 2

    if value_clip_eps is not None:
        v_clipped = old_values + torch.clamp(values - old_values, -value_clip_eps, value_clip_eps)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_per_step = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        v_loss_per_step = 0.5 * v_loss_unclipped

    if masks is not None:
        mask_sum = torch.clamp(masks.sum(), min=1.0)
        return (v_loss_per_step * masks).sum() / mask_sum
    return v_loss_per_step.mean()


def train_ppo(
    env: Environment,
    policy: CategoricalPolicy,
    value_net: NeuralValueFunction,
    total_timesteps: int = 10_000,
    rollout_steps: int = 256,
    epochs_per_rollout: int = 4,
    minibatch_size: int = 32,
    policy_lr: float = 3e-3,
    value_lr: float = 1e-2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    value_clip_eps: float | None = None,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    normalize_adv: bool = True,
    seed: int | None = None,
) -> PPOResult:
    """Train policy and value network with Proximal Policy Optimization (PPO).

    Args:
        env: Environment to interact with.
        policy: CategoricalPolicy module.
        value_net: NeuralValueFunction module.
        total_timesteps: Total environment steps to collect and learn from.
        rollout_steps: Number of steps collected per rollout before running PPO epochs.
        epochs_per_rollout: Number of optimization passes over the rollout buffer.
        minibatch_size: Minibatch size for stochastic gradient updates.
        policy_lr: Learning rate for policy optimizer.
        value_lr: Learning rate for value optimizer.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_eps: Probability ratio clipping threshold epsilon in [1-eps, 1+eps].
        value_clip_eps: Optional value clipping threshold epsilon_v (or None).
        entropy_coef: Entropy bonus weight beta.
        value_coef: Value loss weight c1.
        max_grad_norm: Gradient clipping norm threshold.
        target_kl: Optional target KL divergence threshold for early stopping.
        normalize_adv: Whether to standardize advantages before minibatching.
        seed: Random seed for reproducibility.

    Returns:
        PPOResult containing full training trajectory metrics.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)

    from rl_reasoning_lab.trainers.rollout import RolloutBuffer

    buffer = RolloutBuffer(num_states=env.num_states, gamma=gamma, gae_lambda=gae_lambda)
    result = PPOResult(episode_returns=[], episode_lengths=[])

    state, _ = env.reset(seed=seed)
    current_ep_return = 0.0
    current_ep_length = 0

    num_iterations = total_timesteps // rollout_steps
    for _it in range(num_iterations):
        buffer.clear()

        # 1. Rollout Collection Phase
        for _step in range(rollout_steps):
            with torch.no_grad():
                action, log_prob = policy.sample_action_with_log_prob(state)
                val = value_net.predict(state)

            step_outcome = env.step(action)
            next_state = step_outcome.observation
            reward = step_outcome.reward
            done = step_outcome.terminated or step_outcome.truncated

            buffer.add(
                state=state,
                action=action,
                reward=reward,
                done=step_outcome.terminated,  # Terminal ends future bootstrapping
                old_log_prob=log_prob,
                value=val,
                mask=1.0,
            )

            current_ep_return += reward
            current_ep_length += 1

            if done:
                result.episode_returns.append(current_ep_return)
                result.episode_lengths.append(current_ep_length)
                current_ep_return = 0.0
                current_ep_length = 0
                state, _ = env.reset()
            else:
                state = next_state

        # Finalize buffer with bootstrap value of final state
        with torch.no_grad():
            last_val = value_net.predict(state)
        buffer.finalize(next_value=last_val, normalize_adv=normalize_adv)

        # 2. Optimization Phase across K Epochs
        epoch_kls = []
        epoch_clip_fracs = []
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        epoch_mean_ratios = []
        epoch_grad_norms = []
        epochs_actually_completed = 0

        for _epoch in range(epochs_per_rollout):
            epochs_actually_completed += 1
            early_stop = False

            for batch in buffer.get_minibatch_generator(batch_size=minibatch_size, shuffle=True):
                states_b = batch["states"]
                actions_b = batch["actions"]
                old_log_probs_b = batch["old_log_probs"]
                old_values_b = batch["old_values"]
                returns_b = batch["returns"]
                advs_b = batch["advantages"]
                masks_b = batch["masks"]

                # Evaluate current policy on batch
                log_probs, entropy = policy.evaluate_actions(states_b, actions_b)

                # Policy Loss
                pol_loss, diag = compute_ppo_policy_loss(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs_b,
                    advantages=advs_b,
                    clip_eps=clip_eps,
                    masks=masks_b,
                )

                # Entropy Bonus
                entropy_loss = -entropy_coef * entropy.mean()

                total_policy_loss = pol_loss + entropy_loss

                policy_optimizer.zero_grad()
                total_policy_loss.backward()
                p_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                policy_optimizer.step()

                # Value Loss
                current_values = value_net(states_b)
                val_loss = compute_ppo_value_loss(
                    values=current_values,
                    old_values=old_values_b,
                    returns=returns_b,
                    value_clip_eps=value_clip_eps,
                    masks=masks_b,
                )

                value_optimizer.zero_grad()
                (value_coef * val_loss).backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                value_optimizer.step()

                epoch_kls.append(diag["approx_kl"])
                epoch_clip_fracs.append(diag["clip_fraction"])
                epoch_policy_losses.append(float(pol_loss.item()))
                epoch_value_losses.append(float(val_loss.item()))
                epoch_entropies.append(float(entropy.mean().item()))
                epoch_mean_ratios.append(diag["mean_ratio"])
                epoch_grad_norms.append(float(p_norm.item()))

                # Target KL early stopping
                if target_kl is not None and diag["approx_kl"] > 1.5 * target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        result.policy_losses.append(
            float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0
        )
        result.value_losses.append(
            float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0
        )
        result.entropies.append(float(np.mean(epoch_entropies)) if epoch_entropies else 0.0)
        result.approx_kls.append(float(np.mean(epoch_kls)) if epoch_kls else 0.0)
        result.clip_fractions.append(float(np.mean(epoch_clip_fracs)) if epoch_clip_fracs else 0.0)
        result.mean_ratios.append(float(np.mean(epoch_mean_ratios)) if epoch_mean_ratios else 0.0)
        result.grad_norms.append(float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0)
        result.epochs_used.append(epochs_actually_completed)

    return result
