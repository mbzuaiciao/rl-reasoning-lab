"""Rollout storage and minibatch generation for on-policy RL algorithms like PPO."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from rl_reasoning_lab.algorithms.advantage import compute_gae, normalize_advantages


class RolloutBuffer:
    """Explicit, inspectable storage for trajectories collected under a behavior policy.

    Guarantees that old policy statistics (log probabilities and value estimates)
    are strictly frozen and detached from the PyTorch computation graph at collection time.
    """

    def __init__(self, num_states: int, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.num_states = num_states
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.states: list[int] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.old_log_probs: list[float] = []
        self.values: list[float] = []
        self.masks: list[float] = []  # 1.0 for valid step, 0.0 for padding

        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(
        self,
        state: int,
        action: int,
        reward: float,
        done: bool,
        old_log_prob: float | torch.Tensor,
        value: float | torch.Tensor,
        mask: float = 1.0,
    ) -> None:
        """Store a single environment transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.old_log_probs.append(
            float(old_log_prob.item())
            if isinstance(old_log_prob, torch.Tensor)
            else float(old_log_prob)
        )
        self.values.append(float(value.item()) if isinstance(value, torch.Tensor) else float(value))
        self.masks.append(float(mask))

    def __len__(self) -> int:
        return len(self.states)

    def finalize(self, next_value: float = 0.0, normalize_adv: bool = True) -> None:
        """Compute GAE advantages and returns once all transitions in the rollout are collected."""
        t_len = len(self.states)
        if t_len == 0:
            return

        values_arr = np.array(self.values, dtype=np.float32)
        next_values = np.zeros(t_len, dtype=np.float32)
        if t_len > 1:
            next_values[:-1] = values_arr[1:]
        next_values[-1] = float(next_value)

        advs, targets = compute_gae(
            rewards=self.rewards,
            values=values_arr,
            next_values=next_values,
            dones=self.dones,
            gamma=self.gamma,
            lambda_=self.gae_lambda,
        )

        # Apply masks to zero out padding elements
        masks_arr = np.array(self.masks, dtype=np.float32)
        advs = advs * masks_arr
        targets = targets * masks_arr

        if normalize_adv and t_len > 1:
            valid_indices = np.where(masks_arr > 0.0)[0]
            if len(valid_indices) > 1:
                advs[valid_indices] = normalize_advantages(advs[valid_indices])

        self.advantages = advs
        self.returns = targets

    def get_minibatch_generator(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield minibatches of aligned tensors for PPO optimization epochs."""
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call finalize() before generating minibatches.")

        t_len = len(self.states)
        indices = np.arange(t_len)
        if shuffle:
            np.random.shuffle(indices)

        # Convert full buffer to tensors
        states_one_hot = torch.zeros(t_len, self.num_states, dtype=torch.float32)
        for i, s in enumerate(self.states):
            states_one_hot[i, s] = 1.0

        actions_tensor = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs_tensor = torch.tensor(self.old_log_probs, dtype=torch.float32)
        old_values_tensor = torch.tensor(self.values, dtype=torch.float32)
        returns_tensor = torch.tensor(self.returns, dtype=torch.float32)
        advantages_tensor = torch.tensor(self.advantages, dtype=torch.float32)
        masks_tensor = torch.tensor(self.masks, dtype=torch.float32)

        for start_idx in range(0, t_len, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield {
                "states": states_one_hot[batch_indices],
                "actions": actions_tensor[batch_indices],
                "old_log_probs": old_log_probs_tensor[batch_indices],
                "old_values": old_values_tensor[batch_indices],
                "returns": returns_tensor[batch_indices],
                "advantages": advantages_tensor[batch_indices],
                "masks": masks_tensor[batch_indices],
            }

    def clear(self) -> None:
        """Reset the buffer for the next rollout cycle."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.old_log_probs.clear()
        self.values.clear()
        self.masks.clear()
        self.advantages = None
        self.returns = None
