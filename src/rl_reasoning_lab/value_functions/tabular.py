"""Tabular representations of state-value V(s) and action-value Q(s, a) functions."""

from __future__ import annotations

import numpy as np


class TabularValueFunction:
    """Tabular state-value function V(s): S -> R."""

    def __init__(self, num_states: int, initial_value: float = 0.0) -> None:
        self.num_states = num_states
        self.values = np.full(num_states, initial_value, dtype=np.float64)

    def __getitem__(self, state: int) -> float:
        return float(self.values[state])

    def __setitem__(self, state: int, value: float) -> None:
        self.values[state] = value

    def copy(self) -> TabularValueFunction:
        new_vf = TabularValueFunction(self.num_states)
        new_vf.values = self.values.copy()
        return new_vf

    def as_array(self) -> np.ndarray:
        return self.values.copy()


class TabularQFunction:
    """Tabular action-value function Q(s, a): S x A -> R."""

    def __init__(self, num_states: int, num_actions: int, initial_value: float = 0.0) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.values = np.full((num_states, num_actions), initial_value, dtype=np.float64)

    def __getitem__(self, key: tuple[int, int]) -> float:
        s, a = key
        return float(self.values[s, a])

    def __setitem__(self, key: tuple[int, int], value: float) -> None:
        s, a = key
        self.values[s, a] = value

    def get_state_values(self, state: int) -> np.ndarray:
        """Return Q(state, :) for all actions."""
        return self.values[state].copy()

    def extract_state_value_function(self) -> TabularValueFunction:
        """Extract V(s) = max_a Q(s, a)."""
        vf = TabularValueFunction(self.num_states)
        vf.values = np.max(self.values, axis=1)
        return vf

    def extract_greedy_policy(self) -> np.ndarray:
        """Extract deterministic greedy policy pi(s) = argmax_a Q(s, a)."""
        return np.argmax(self.values, axis=1)

    def copy(self) -> TabularQFunction:
        new_q = TabularQFunction(self.num_states, self.num_actions)
        new_q.values = self.values.copy()
        return new_q

    def as_array(self) -> np.ndarray:
        return self.values.copy()
