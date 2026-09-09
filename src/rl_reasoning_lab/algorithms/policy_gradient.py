"""Policy gradient algorithms: REINFORCE and Actor-Critic."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rl_reasoning_lab.algorithms.advantage import (
    compute_full_returns,
    compute_reward_to_go,
    normalize_advantages,
)
from rl_reasoning_lab.environments.base import Environment
from rl_reasoning_lab.policies.neural import CategoricalPolicy
from rl_reasoning_lab.value_functions.neural import NeuralValueFunction


@dataclass
class PolicyGradientResult:
    """Detailed training logs and diagnostics from policy gradient algorithms."""

    episode_returns: list[float]
    episode_lengths: list[int]
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    advantages_mean: list[float] = field(default_factory=list)
    advantages_std: list[float] = field(default_factory=list)


def reinforce(
    env: Environment,
    policy: CategoricalPolicy,
    value_net: NeuralValueFunction | None = None,
    policy_lr: float = 3e-3,
    value_lr: float = 1e-2,
    num_episodes: int = 500,
    gamma: float = 0.99,
    return_type: str = "reward_to_go",
    baseline_type: str | None = None,
    entropy_coef: float = 0.0,
    normalize_adv: bool = False,
    max_grad_norm: float | None = None,
    seed: int | None = None,
) -> PolicyGradientResult:
    """Episodic REINFORCE (Monte Carlo Policy Gradient) with optional baselines.

    Optimization Objective:
        Maximize J(theta) = E_{tau ~ pi_theta}[ R(tau) ]
        Gradient: grad_theta J(theta) = E_{tau}[ sum_t (G_t - b(s_t)) grad_theta log pi_theta(a_t | s_t) ]

    Sign Convention:
        PyTorch optimizers minimize loss. Therefore, we define:
        Loss = - (1/T) sum_t A_t * log pi_theta(a_t | s_t) - beta * H(pi_theta)
        Minimizing this negative loss is mathematically equivalent to ascending the expected return.

    Args:
        env: Environment instance.
        policy: CategoricalPolicy module.
        value_net: Optional NeuralValueFunction for learned state-value baseline.
        policy_lr: Learning rate for policy optimizer.
        value_lr: Learning rate for value network optimizer.
        num_episodes: Number of complete rollout episodes.
        gamma: Discount factor in [0, 1].
        return_type: "reward_to_go" (G_t = sum_{k=0}^{T-t-1} gamma^k R_{t+k+1}) or "full" (G = R_total).
        baseline_type: None, "moving_average", or "learned_value".
        entropy_coef: Coefficient beta for entropy regularization (beta * H(pi)).
        normalize_adv: If True, standardize advantages across the episode.
        max_grad_norm: Maximum gradient norm for clipping (or None).
        seed: Random seed for reproducibility.

    Returns:
        PolicyGradientResult containing episode returns, losses, entropies, and grad norms.
    """
    if return_type not in ("reward_to_go", "full"):
        raise ValueError(f"Unknown return_type '{return_type}'. Choose 'reward_to_go' or 'full'.")
    if baseline_type not in (None, "moving_average", "learned_value"):
        raise ValueError(
            f"Unknown baseline_type '{baseline_type}'. Choose None, 'moving_average', or 'learned_value'."
        )
    if baseline_type == "learned_value" and value_net is None:
        raise ValueError("value_net must be provided when baseline_type='learned_value'.")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    value_optimizer = (
        torch.optim.Adam(value_net.parameters(), lr=value_lr) if value_net is not None else None
    )

    result = PolicyGradientResult(episode_returns=[], episode_lengths=[])
    moving_avg_return = 0.0

    for ep in range(num_episodes):
        state, _ = env.reset()
        states: list[int] = []
        actions: list[int] = []
        rewards: list[float] = []
        log_probs_list: list[torch.Tensor] = []
        done = False

        while not done:
            action, log_prob = policy.sample_action_with_log_prob(state)
            step_outcome = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(step_outcome.reward)
            log_probs_list.append(log_prob)

            state = step_outcome.observation
            done = step_outcome.terminated or step_outcome.truncated

        total_reward = sum(rewards)
        result.episode_returns.append(total_reward)
        result.episode_lengths.append(len(rewards))

        # 1. Compute Returns
        if return_type == "full":
            returns_np = compute_full_returns(rewards, gamma=gamma)
        else:
            returns_np = compute_reward_to_go(rewards, gamma=gamma)
        returns = torch.from_numpy(returns_np).float()

        # 2. Compute Baseline & Advantages
        t_len = len(rewards)
        states_tensor = torch.stack([policy._state_to_tensor(s) for s in states])

        if baseline_type is None:
            advantages = returns
            value_loss_val = 0.0
        elif baseline_type == "moving_average":
            if ep == 0:
                moving_avg_return = float(returns_np.mean())
            else:
                moving_avg_return = 0.95 * moving_avg_return + 0.05 * float(returns_np.mean())
            advantages = returns - moving_avg_return
            value_loss_val = 0.0
        elif baseline_type == "learned_value":
            assert value_net is not None
            assert value_optimizer is not None
            state_values = value_net(states_tensor)
            # Advantages: A_t = G_t - V(s_t), with detached baseline so critic gradients don't flow into policy
            advantages = returns - state_values.detach()

            # Critic loss: MSE between predicted value and Monte Carlo return
            value_loss = nn.functional.mse_loss(state_values, returns)
            value_loss_val = float(value_loss.item())

            value_optimizer.zero_grad()
            value_loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            value_optimizer.step()
        else:
            advantages = returns
            value_loss_val = 0.0

        if normalize_adv and t_len > 1:
            advantages = torch.from_numpy(normalize_advantages(advantages.numpy())).float()

        # 3. Policy Loss with Log Probs and Entropy Bonus
        log_probs = torch.stack(log_probs_list)
        # Compute entropy for each state
        with torch.no_grad():
            _, entropies_step = policy.evaluate_actions(states_tensor, torch.tensor(actions))
            mean_entropy = float(entropies_step.mean().item())

        # Forward pass for entropy with gradient
        logits = policy(states_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        entropy_loss = dist.entropy().mean()

        # Policy objective: minimize - (advantages * log_probs) - beta * entropy
        policy_loss = -(advantages * log_probs).mean() - entropy_coef * entropy_loss

        policy_optimizer.zero_grad()
        policy_loss.backward()

        # Compute gradient norm
        grad_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                grad_norm += float(p.grad.data.norm(2).item() ** 2)
        grad_norm = grad_norm**0.5

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)

        policy_optimizer.step()

        # Record diagnostics
        result.policy_losses.append(float(policy_loss.item()))
        result.value_losses.append(value_loss_val)
        result.entropies.append(mean_entropy)
        result.grad_norms.append(grad_norm)
        result.advantages_mean.append(float(advantages.mean().item()))
        result.advantages_std.append(float(advantages.std().item()) if t_len > 1 else 0.0)

    return result


def actor_critic(
    env: Environment,
    policy: CategoricalPolicy,
    value_net: NeuralValueFunction,
    policy_lr: float = 3e-3,
    value_lr: float = 1e-2,
    num_episodes: int = 500,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    max_grad_norm: float | None = 1.0,
    seed: int | None = None,
) -> PolicyGradientResult:
    """One-step Online Temporal Difference Actor-Critic algorithm.

    Actor:
        A_t = delta_t = R_{t+1} + gamma * V_phi(S_{t+1}) * (1 - done_t) - V_phi(S_t)
        L_actor = - delta_t.detach() * log pi_theta(A_t | S_t) - beta * H(pi_theta)

    Critic:
        L_critic = 0.5 * delta_t^2
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)

    result = PolicyGradientResult(episode_returns=[], episode_lengths=[])

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        ep_policy_losses: list[float] = []
        ep_value_losses: list[float] = []
        ep_entropies: list[float] = []
        ep_grad_norms: list[float] = []
        ep_td_errors: list[float] = []

        while not done:
            action, log_prob = policy.sample_action_with_log_prob(state)
            step_outcome = env.step(action)
            next_state = step_outcome.observation
            reward = step_outcome.reward
            done = step_outcome.terminated or step_outcome.truncated

            state_tensor = policy._state_to_tensor(state)
            next_state_tensor = policy._state_to_tensor(next_state)

            # Critic predictions
            v_s = value_net(state_tensor)
            with torch.no_grad():
                v_next = value_net(next_state_tensor) if not step_outcome.terminated else 0.0
                td_target = reward + gamma * v_next
                td_error = td_target - v_s.item()

            # Critic loss: update critic towards 1-step bootstrap target
            critic_loss = 0.5 * (v_s - td_target) ** 2
            value_optimizer.zero_grad()
            critic_loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            value_optimizer.step()

            # Actor loss: policy gradient with advantage = detached td_error
            entropy = policy.get_distribution(state).entropy()
            actor_loss = -td_error * log_prob - entropy_coef * entropy

            policy_optimizer.zero_grad()
            actor_loss.backward()

            grad_norm = 0.0
            for p in policy.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.data.norm(2).item() ** 2)
            grad_norm = grad_norm**0.5

            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            policy_optimizer.step()

            ep_reward += reward
            steps += 1
            ep_policy_losses.append(float(actor_loss.item()))
            ep_value_losses.append(float(critic_loss.item()))
            ep_entropies.append(float(entropy.item()))
            ep_grad_norms.append(grad_norm)
            ep_td_errors.append(abs(td_error))

            state = next_state

        result.episode_returns.append(ep_reward)
        result.episode_lengths.append(steps)
        result.policy_losses.append(float(np.mean(ep_policy_losses)) if ep_policy_losses else 0.0)
        result.value_losses.append(float(np.mean(ep_value_losses)) if ep_value_losses else 0.0)
        result.entropies.append(float(np.mean(ep_entropies)) if ep_entropies else 0.0)
        result.grad_norms.append(float(np.mean(ep_grad_norms)) if ep_grad_norms else 0.0)
        result.advantages_mean.append(float(np.mean(ep_td_errors)) if ep_td_errors else 0.0)
        result.advantages_std.append(float(np.std(ep_td_errors)) if ep_td_errors else 0.0)

    return result
