"""Tabular policy implementations."""

from __future__ import annotations

import numpy as np

from rl_reasoning_lab.policies.base import Policy


class RandomPolicy(Policy):
    """A policy that chooses uniformly at random among all available actions."""

    def __init__(self, num_actions: int) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.num_actions = num_actions
        self._uniform_probs = np.full(num_actions, 1.0 / num_actions, dtype=np.float64)

    def sample_action(self, state: int, *, rng: np.random.Generator | None = None) -> int:
        generator = rng if rng is not None else np.random.default_rng()
        return int(generator.integers(0, self.num_actions))

    def action_probabilities(self, state: int) -> np.ndarray:
        return self._uniform_probs.copy()


class DeterministicTabularPolicy(Policy):
    """A deterministic policy defined by an array or dict mapping state -> action."""

    def __init__(self, actions: list[int] | np.ndarray, num_actions: int) -> None:
        self.actions = np.array(actions, dtype=np.int64)
        self.num_actions = num_actions
        self.num_states = len(self.actions)

    def sample_action(self, state: int, *, rng: np.random.Generator | None = None) -> int:
        return int(self.actions[state])

    def action_probabilities(self, state: int) -> np.ndarray:
        probs = np.zeros(self.num_actions, dtype=np.float64)
        probs[self.actions[state]] = 1.0
        return probs


class EpsilonGreedyPolicy(Policy):
    """An epsilon-greedy exploration policy over action-value estimates Q(s, a).

    With probability 1 - epsilon: select argmax_a Q(s, a) (ties broken uniformly).
    With probability epsilon: select an action uniformly at random from all actions.
    """

    def __init__(
        self,
        q_table: np.ndarray,
        epsilon: float = 0.1,
    ) -> None:
        """Args:

        q_table: 2-D numpy array of shape (num_states, num_actions).
        epsilon: Exploration probability in [0.0, 1.0].
        """
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.q_table = q_table
        self.num_states, self.num_actions = q_table.shape
        self.epsilon = epsilon

    def sample_action(self, state: int, *, rng: np.random.Generator | None = None) -> int:
        generator = rng if rng is not None else np.random.default_rng()
        if generator.random() < self.epsilon:
            return int(generator.integers(0, self.num_actions))

        q_values = self.q_table[state]
        max_val = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_val))
        return int(generator.choice(best_actions))

    def action_probabilities(self, state: int) -> np.ndarray:
        probs = np.full(self.num_actions, self.epsilon / self.num_actions, dtype=np.float64)
        q_values = self.q_table[state]
        max_val = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_val))
        greedy_prob = (1.0 - self.epsilon) / len(best_actions)
        probs[best_actions] += greedy_prob
        return probs


class SoftmaxTabularPolicy(Policy):
    """Boltzmann / Softmax exploration policy over action-value estimates Q(s, a).

    pi(a | s) = exp(Q(s, a) / tau) / sum_{a'} exp(Q(s, a') / tau)
    """

    def __init__(
        self,
        q_table: np.ndarray,
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {temperature}")
        self.q_table = q_table
        self.num_states, self.num_actions = q_table.shape
        self.temperature = temperature

    def action_probabilities(self, state: int) -> np.ndarray:
        q_values = self.q_table[state]
        # Subtract max for numerical stability
        scaled = (q_values - np.max(q_values)) / self.temperature
        exp_vals = np.exp(scaled)
        return exp_vals / np.sum(exp_vals)

    def sample_action(self, state: int, *, rng: np.random.Generator | None = None) -> int:
        generator = rng if rng is not None else np.random.default_rng()
        probs = self.action_probabilities(state)
        return int(generator.choice(self.num_actions, p=probs))
