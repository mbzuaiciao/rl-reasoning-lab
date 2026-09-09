"""Returns and Advantage estimators: Full return, Reward-to-go, and GAE."""

from __future__ import annotations

import numpy as np
import torch


def compute_full_returns(rewards: list[float] | np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute full-trajectory return and assign it uniformly to every step.

    G = sum_{t=0}^{T-1} gamma^t R_{t+1}
    Returns array of length T where every element equals G.

    Used as an ablation baseline to demonstrate the variance incurred when credit
    for earlier rewards is erroneously assigned to later actions.
    """
    t_len = len(rewards)
    if t_len == 0:
        return np.array([], dtype=np.float32)

    total_return = 0.0
    for r in reversed(rewards):
        total_return = float(r) + gamma * total_return

    return np.full(t_len, total_return, dtype=np.float32)


def compute_reward_to_go(rewards: list[float] | np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Compute Monte Carlo reward-to-go from each step t onward.

    G_t = sum_{k=0}^{T-t-1} gamma^k R_{t+k+1} = R_{t+1} + gamma * G_{t+1}

    Causality Invariant:
    Actions at time t cannot influence rewards obtained before time t (R_1, ..., R_t).
    Dropping past rewards removes zero-expectation noise terms without introducing bias.
    """
    t_len = len(rewards)
    returns = np.zeros(t_len, dtype=np.float32)
    g = 0.0
    for t in reversed(range(t_len)):
        g = float(rewards[t]) + gamma * g
        returns[t] = g
    return returns


def compute_gae(
    rewards: list[float] | np.ndarray | torch.Tensor,
    values: list[float] | np.ndarray | torch.Tensor,
    next_values: list[float] | np.ndarray | torch.Tensor,
    dones: list[bool] | np.ndarray | torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and TD(lambda) targets.

    TD Residual:
        delta_t = R_{t+1} + gamma * V(S_{t+1}) * (1 - done_t) - V(S_t)

    GAE Advantage:
        A_t^{GAE(gamma, lambda)} = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
        Recursive form:
        A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

    Properties:
        - lambda = 0: A_t = delta_t (pure 1-step TD, lowest variance, biased by V).
        - lambda = 1: A_t = G_t - V(S_t) (pure Monte Carlo, unbiased, highest variance).

    Returns:
        advantages: np.ndarray of shape (T,)
        targets: np.ndarray of shape (T,) where target_t = A_t + V(S_t)
    """
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.detach().cpu().numpy()
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(next_values, torch.Tensor):
        next_values = next_values.detach().cpu().numpy()
    if isinstance(dones, torch.Tensor):
        dones = dones.detach().cpu().numpy()

    r = np.asarray(rewards, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    v_next = np.asarray(next_values, dtype=np.float32)
    d = np.asarray(dones, dtype=np.float32)

    t_len = len(r)
    advantages = np.zeros(t_len, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(t_len)):
        non_terminal = 1.0 - d[t]
        delta = r[t] + gamma * v_next[t] * non_terminal - v[t]
        advantages[t] = delta + gamma * lambda_ * non_terminal * last_gae
        last_gae = advantages[t]

    targets = advantages + v
    return advantages, targets


def normalize_advantages(advantages: np.ndarray | torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """Standardize advantage estimates to zero mean and unit variance.

    A_norm = (A - mean(A)) / (std(A) + eps)
    """
    if isinstance(advantages, torch.Tensor):
        adv = advantages.detach().cpu().numpy()
    else:
        adv = np.asarray(advantages, dtype=np.float32)

    if len(adv) <= 1:
        return adv

    std = np.std(adv)
    if std < eps:
        return adv - np.mean(adv)
    return (adv - np.mean(adv)) / (std + eps)
