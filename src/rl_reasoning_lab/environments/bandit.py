"""Multi-armed and contextual bandit environments for pure exploration-exploitation analysis."""

from __future__ import annotations

import random
from typing import Any

from rl_reasoning_lab.environments.base import Environment, StepOutcome


class MultiArmedBanditEnv(Environment):
    """A stationary or non-stationary Multi-Armed Bandit with K arms.

    Because there are no state transitions (S = {0}), bandits isolate:
        - Exploration vs. exploitation tradeoffs
        - Action-value estimation convergence
        - Cumulative regret R_T = T * mu* - sum_{t=1}^T r_t
    """

    def __init__(
        self,
        num_arms: int = 10,
        mean_rewards: list[float] | None = None,
        reward_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._num_arms = num_arms
        self._reward_std = reward_std
        self._rng = random.Random(seed)

        if mean_rewards is not None:
            if len(mean_rewards) != num_arms:
                raise ValueError("Length of mean_rewards must match num_arms.")
            self._means = list(mean_rewards)
        else:
            self._means = [self._rng.gauss(0.0, 1.0) for _ in range(num_arms)]

        self._best_arm = max(range(num_arms), key=lambda i: self._means[i])
        self._best_mean = self._means[self._best_arm]
        self._step_count = 0

    @property
    def num_states(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return self._num_arms

    @property
    def true_means(self) -> list[float]:
        return list(self._means)

    @property
    def optimal_arm(self) -> int:
        return self._best_arm

    @property
    def optimal_expected_reward(self) -> float:
        return self._best_mean

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        self._step_count = 0
        return 0, {"optimal_arm": self._best_arm, "true_means": self._means}

    def step(self, action: int) -> StepOutcome:
        if not (0 <= action < self._num_arms):
            raise ValueError(f"Action {action} out of bounds for {self._num_arms} arms.")

        self._step_count += 1
        reward = self._rng.gauss(self._means[action], self._reward_std)
        regret = self._best_mean - self._means[action]

        return StepOutcome(
            observation=0,
            reward=reward,
            terminated=False,
            truncated=False,
            info={
                "action": action,
                "expected_reward": self._means[action],
                "immediate_regret": regret,
                "is_optimal": action == self._best_arm,
            },
        )

    def get_transitions(self, state: int, action: int) -> list[tuple[float, int, float, bool]]:
        """For bandit MDP, next state is always 0, expected reward is mean."""
        return [(1.0, 0, self._means[action], False)]

    def render(self) -> str:
        lines = [f"Bandit ({self._num_arms} arms):"]
        for a in range(self._num_arms):
            marker = " *" if a == self._best_arm else ""
            lines.append(f"  Arm {a}: mean = {self._means[a]:+.3f}{marker}")
        return "\n".join(lines)
