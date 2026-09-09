"""A simple 1-D corridor environment for first-principles RL experiments."""

from __future__ import annotations

import random
from typing import Any

from rl_reasoning_lab.environments.base import Environment, StepOutcome


class CorridorEnv(Environment):
    """A 1-Dimensional corridor environment with discrete states {0, ..., length - 1}.

    Dynamics:
        - Actions: 0 = LEFT, 1 = RIGHT.
        - Start: Usually state 0 (leftmost end).
        - Goal: State length - 1 (rightmost end).
        - Reaching the goal terminates the episode with `goal_reward`.
        - Slipping: With probability `slip_prob`, the agent slips in the opposite direction.
        - Walls: Moving left from 0 stays at 0.

    This environment isolates fundamental RL questions:
        - How does the discount factor gamma affect propagation of distant rewards?
        - How does episode horizon impact exploration under uniform random actions?
    """

    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_NAMES = {0: "LEFT", 1: "RIGHT"}

    def __init__(
        self,
        length: int = 5,
        goal_reward: float = 1.0,
        step_cost: float = 0.0,
        slip_prob: float = 0.0,
        max_steps: int | None = 100,
    ) -> None:
        if length < 2:
            raise ValueError(f"Corridor length must be at least 2, got {length}")
        if not (0.0 <= slip_prob < 1.0):
            raise ValueError(f"slip_prob must be in [0.0, 1.0), got {slip_prob}")

        self._length = length
        self._goal_reward = goal_reward
        self._step_cost = step_cost
        self._slip_prob = slip_prob
        self._max_steps = max_steps

        self._state = 0
        self._step_count = 0
        self._rng = random.Random()

    @property
    def num_states(self) -> int:
        return self._length

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def goal_state(self) -> int:
        return self._length - 1

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        self._state = 0
        self._step_count = 0
        return self._state, {"state": self._state}

    def _next_state_deterministic(self, state: int, action: int) -> int:
        if state == self.goal_state:
            return state
        if action == self.ACTION_LEFT:
            return max(0, state - 1)
        elif action == self.ACTION_RIGHT:
            return min(self._length - 1, state + 1)
        else:
            raise ValueError(f"Invalid action {action}")

    def get_transitions(self, state: int, action: int) -> list[tuple[float, int, float, bool]]:
        """Return [(prob, next_state, reward, is_terminal)]."""
        if state == self.goal_state:
            # Absorbing terminal state: zero reward, stays in terminal state
            return [(1.0, state, 0.0, True)]

        main_next = self._next_state_deterministic(state, action)
        opposite_action = self.ACTION_RIGHT if action == self.ACTION_LEFT else self.ACTION_LEFT
        slipped_next = self._next_state_deterministic(state, opposite_action)

        transitions = []
        p_main = 1.0 - self._slip_prob
        p_slip = self._slip_prob

        # Helper to compute reward and terminal flag
        def outcome_for(ns: int) -> tuple[float, bool]:
            is_term = ns == self.goal_state
            rew = self._goal_reward if is_term else self._step_cost
            return rew, is_term

        if p_slip == 0.0 or main_next == slipped_next:
            rew, term = outcome_for(main_next)
            transitions.append((1.0, main_next, rew, term))
        else:
            rew_main, term_main = outcome_for(main_next)
            transitions.append((p_main, main_next, rew_main, term_main))

            rew_slip, term_slip = outcome_for(slipped_next)
            transitions.append((p_slip, slipped_next, rew_slip, term_slip))

        return transitions

    def step(self, action: int) -> StepOutcome:
        if self._state == self.goal_state:
            return StepOutcome(
                observation=self._state,
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"state": self._state, "message": "Episode already terminated."},
            )

        self._step_count += 1
        effective_action = action
        slipped = False

        if self._slip_prob > 0.0 and self._rng.random() < self._slip_prob:
            effective_action = self.ACTION_RIGHT if action == self.ACTION_LEFT else self.ACTION_LEFT
            slipped = True

        self._state = self._next_state_deterministic(self._state, effective_action)

        terminated = self._state == self.goal_state
        truncated = False
        if not terminated and self._max_steps is not None and self._step_count >= self._max_steps:
            truncated = True

        reward = self._goal_reward if terminated else self._step_cost

        return StepOutcome(
            observation=self._state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"slipped": slipped, "effective_action": effective_action},
        )

    def render(self) -> str:
        cells = []
        for i in range(self._length):
            if i == self._state:
                cells.append("A")
            elif i == self.goal_state:
                cells.append("G")
            else:
                cells.append(".")
        return "[" + " ".join(cells) + "]"
