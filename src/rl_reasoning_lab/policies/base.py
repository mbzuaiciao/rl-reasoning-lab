"""Policy and trajectory abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Transition:
    """A single transition step in an MDP: (S_t, A_t, R_{t+1}, S_{t+1}, Done)."""

    state: int
    action: int
    reward: float
    next_state: int
    terminated: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A sequence of transitions representing one full or partial episode rollout."""

    transitions: list[Transition] = field(default_factory=list)

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def total_reward(self) -> float:
        """Undiscounted sum of rewards: sum_t R_{t+1}."""
        return sum(t.reward for t in self.transitions)

    def discounted_return(self, gamma: float) -> float:
        """Compute the discounted return G_0 = sum_{t=0}^{T-1} gamma^t R_{t+1}."""
        g = 0.0
        for t in reversed(self.transitions):
            g = t.reward + gamma * g
        return g

    def returns_at_each_step(self, gamma: float) -> list[float]:
        """Compute the return G_t from each step t onward: G_t = sum_{k=0}^{T-t-1} gamma^k R_{t+k+1}."""
        returns = [0.0] * len(self.transitions)
        g = 0.0
        for i in reversed(range(len(self.transitions))):
            g = self.transitions[i].reward + gamma * g
            returns[i] = g
        return returns


class Policy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def sample_action(self, state: int, *, rng: np.random.Generator | None = None) -> int:
        """Sample an action given the current state."""
        pass

    @abstractmethod
    def action_probabilities(self, state: int) -> np.ndarray:
        """Return the probability distribution pi(a | state) over all actions.

        Returns:
            1-D numpy array of length num_actions where sum(probs) == 1.0.
        """
        pass
