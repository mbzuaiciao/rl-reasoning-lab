"""Fault injection framework for simulating ML research engineering failure modes."""

import random
from typing import Any

import torch
import torch.nn as nn


class CorruptedRewardInjector:
    """Simulates silent reward sensor or verifier corruption by flipping or nullifying a fraction of rewards."""

    def __init__(self, corruption_rate: float = 0.15, corrupt_value: float = -1.0, seed: int = 42):
        self.corruption_rate = corruption_rate
        self.corrupt_value = corrupt_value
        self.rng = random.Random(seed)
        self.total_queries = 0
        self.corrupted_count = 0

    def process_reward(self, true_reward: float) -> float:
        self.total_queries += 1
        if self.rng.random() < self.corruption_rate:
            self.corrupted_count += 1
            return self.corrupt_value
        return true_reward


class StaleRolloutInjector:
    """Simulates asynchronous rollout worker lag by supplying trajectories from older policy checkpoints."""

    def __init__(self, staleness_steps: int = 5):
        self.staleness_steps = staleness_steps
        self.history: list[dict[str, Any]] = []

    def push(self, rollout_batch: dict[str, Any], current_step: int) -> None:
        self.history.append({"batch": rollout_batch, "step": current_step})

    def pop_stale(self, current_step: int) -> tuple[dict[str, Any], int]:
        """Return the oldest rollout meeting the staleness threshold."""
        if not self.history:
            raise ValueError("Stale buffer is empty.")

        target_step = max(0, current_step - self.staleness_steps)
        # Find closest batch
        for idx, item in enumerate(self.history):
            if item["step"] >= target_step:
                res = self.history.pop(idx)
                staleness = current_step - res["step"]
                return res["batch"], staleness

        res = self.history.pop(0)
        return res["batch"], current_step - res["step"]


class ShapeBroadcastingBugInjector:
    """Deliberately manipulates tensor shapes to simulate silent PyTorch broadcasting bugs."""

    @staticmethod
    def corrupt_advantage_shape(advantages: torch.Tensor) -> torch.Tensor:
        """Collapse [B, T] advantage tensor to [B] or [B, 1].

        This will silently broadcast against log_probs [B, T], computing a mathematically corrupt loss
        without raising a PyTorch shape exception!
        """
        if advantages.ndim == 2:
            return advantages.mean(dim=-1)  # shape [B]
        return advantages


class StaleReferenceModelDrift:
    """Simulates an unfrozen reference policy that accidentally updates during training."""

    @staticmethod
    def inject_drift(reference_model: nn.Module, noise_scale: float = 0.05) -> None:
        with torch.no_grad():
            for p in reference_model.parameters():
                p.add_(torch.randn_like(p) * noise_scale)


class ZeroActionMaskInjector:
    """Simulates empty action masks (e.g., prompt exceeds max sequence length)."""

    @staticmethod
    def create_empty_mask(batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.zeros((batch_size, seq_len), dtype=torch.bool)
