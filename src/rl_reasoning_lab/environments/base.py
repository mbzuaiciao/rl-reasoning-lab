"""Base environment interfaces and definitions for RL environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepOutcome:
    """The result of taking an action in an environment.

    Attributes:
        observation: The state/observation observed by the agent.
        reward: The immediate scalar reward obtained.
        terminated: Whether the episode reached a natural terminal state.
        truncated: Whether the episode was cut off (e.g., maximum horizon reached).
        info: Additional diagnostic metadata (e.g., true state, slip occurrence).
    """

    observation: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class Environment(ABC):
    """Abstract base class for discrete-state, discrete-action environments.

    Unlike generic gym environments, our environments explicitly expose their
    underlying Markov Decision Process (MDP) dynamics P(s', r | s, a) whenever
    the transition model is known. This allows students to verify sample-based
    algorithms (Q-learning, SARSA) against exact dynamic programming algorithms
    (Policy Evaluation, Value Iteration) on identical environments.
    """

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Total number of discrete states |S|."""
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Total number of discrete actions |A|."""
        pass

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional integer seed for reproducibility.

        Returns:
            A tuple of (initial_state, info_dict).
        """
        pass

    @abstractmethod
    def step(self, action: int) -> StepOutcome:
        """Execute one action in the environment.

        Args:
            action: An integer action in {0, ..., num_actions - 1}.

        Returns:
            StepOutcome containing (observation, reward, terminated, truncated, info).
        """
        pass

    @abstractmethod
    def get_transitions(self, state: int, action: int) -> list[tuple[float, int, float, bool]]:
        """Return the exact transition distribution for a given (state, action) pair.

        Used for model-based dynamic programming methods (Value Iteration, Policy Evaluation).

        Returns:
            A list of tuples: (probability, next_state, reward, is_terminal).
            The probabilities must sum to 1.0.
        """
        pass

    @abstractmethod
    def render(self) -> str:
        """Return a string visual representation of the current environment state."""
        pass
