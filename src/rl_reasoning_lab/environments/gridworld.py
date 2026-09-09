"""A flexible 2-D Gridworld environment for tabular RL experiments."""

from __future__ import annotations

import random
from typing import Any

from rl_reasoning_lab.environments.base import Environment, StepOutcome


class GridworldEnv(Environment):
    """A 2-D grid environment supporting custom layouts, obstacles, cliffs, and stochastic slip.

    Grid Coordinates:
        (row, col) where (0, 0) is the top-left corner.
        State index is flattened: s = row * width + col.

    Actions:
        0 = UP    (-1, 0)
        1 = DOWN  (+1, 0)
        2 = LEFT  ( 0, -1)
        3 = RIGHT ( 0, +1)

    Layout Symbols for initialization:
        '.' or ' ' : Empty normal cell
        'S'        : Start position
        'G'        : Goal cell (terminal, default reward +1.0)
        '#'        : Impassable wall / obstacle
        'C'        : Cliff / hazard cell (terminal or penalty reset, reward -100)
    """

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    ACTION_DELTAS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(
        self,
        grid: list[str],
        step_cost: float = -0.01,
        goal_reward: float = 1.0,
        cliff_penalty: float = -100.0,
        cliff_resets_to_start: bool = True,
        slip_prob: float = 0.0,
        max_steps: int | None = 200,
    ) -> None:
        """Initialize Gridworld from an ASCII grid representation.

        Args:
            grid: List of strings representing rows. All rows must have equal length.
            step_cost: Reward received for non-terminal, non-cliff transitions.
            goal_reward: Reward received when entering a goal cell.
            cliff_penalty: Reward received when falling into a cliff cell.
            cliff_resets_to_start: If True, falling off a cliff teleports to start; if False, it terminates.
            slip_prob: Probability of moving perpendicularly instead of intended direction.
            max_steps: Maximum steps per episode before truncation.
        """
        self._height = len(grid)
        self._width = len(grid[0])
        self._step_cost = step_cost
        self._goal_reward = goal_reward
        self._cliff_penalty = cliff_penalty
        self._cliff_resets_to_start = cliff_resets_to_start
        self._slip_prob = slip_prob
        self._max_steps = max_steps

        self._start_pos: tuple[int, int] | None = None
        self._goals: set[tuple[int, int]] = set()
        self._walls: set[tuple[int, int]] = set()
        self._cliffs: set[tuple[int, int]] = set()

        for r, row_str in enumerate(grid):
            if len(row_str) != self._width:
                raise ValueError("All rows in grid must have the same length.")
            for c, char in enumerate(row_str):
                pos = (r, c)
                if char == "S":
                    self._start_pos = pos
                elif char == "G":
                    self._goals.add(pos)
                elif char == "#":
                    self._walls.add(pos)
                elif char == "C":
                    self._cliffs.add(pos)

        if self._start_pos is None:
            self._start_pos = (0, 0)
        if not self._goals:
            raise ValueError("Gridworld must contain at least one goal ('G').")

        self._current_pos = self._start_pos
        self._step_count = 0
        self._rng = random.Random()

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def num_states(self) -> int:
        return self._height * self._width

    @property
    def num_actions(self) -> int:
        return 4

    def pos_to_state(self, pos: tuple[int, int]) -> int:
        return pos[0] * self._width + pos[1]

    def state_to_pos(self, state: int) -> tuple[int, int]:
        return (state // self._width, state % self._width)

    def is_wall(self, pos: tuple[int, int]) -> bool:
        return pos in self._walls

    def is_goal(self, pos: tuple[int, int]) -> bool:
        return pos in self._goals

    def is_cliff(self, pos: tuple[int, int]) -> bool:
        return pos in self._cliffs

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        self._current_pos = self._start_pos
        self._step_count = 0
        s = self.pos_to_state(self._current_pos)
        return s, {"pos": self._current_pos}

    def _attempt_move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = self.ACTION_DELTAS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        # Boundary check
        if 0 <= nr < self._height and 0 <= nc < self._width:
            if (nr, nc) not in self._walls:
                return (nr, nc)
        return pos

    def get_transitions(self, state: int, action: int) -> list[tuple[float, int, float, bool]]:
        pos = self.state_to_pos(state)

        # Walls are unreachable or dummy states
        if self.is_wall(pos):
            return [(1.0, state, 0.0, True)]

        # Terminal goal state
        if self.is_goal(pos):
            return [(1.0, state, 0.0, True)]

        # Terminal cliff state (if not resetting)
        if self.is_cliff(pos) and not self._cliff_resets_to_start:
            return [(1.0, state, 0.0, True)]

        # Actions perpendicular to intended action
        perpendicular = {
            self.ACTION_UP: (self.ACTION_LEFT, self.ACTION_RIGHT),
            self.ACTION_DOWN: (self.ACTION_LEFT, self.ACTION_RIGHT),
            self.ACTION_LEFT: (self.ACTION_UP, self.ACTION_DOWN),
            self.ACTION_RIGHT: (self.ACTION_UP, self.ACTION_DOWN),
        }

        perp1, perp2 = perpendicular[action]
        p_main = 1.0 - self._slip_prob
        p_slip_each = self._slip_prob / 2.0 if self._slip_prob > 0.0 else 0.0

        candidates = [(p_main, action)]
        if self._slip_prob > 0.0:
            candidates.append((p_slip_each, perp1))
            candidates.append((p_slip_each, perp2))

        # Aggregate outcomes
        outcome_probs: dict[tuple[int, float, bool], float] = {}
        for p, act in candidates:
            if p <= 0.0:
                continue
            next_pos = self._attempt_move(pos, act)
            if self.is_goal(next_pos):
                rew = self._goal_reward
                term = True
                ns = self.pos_to_state(next_pos)
            elif self.is_cliff(next_pos):
                rew = self._cliff_penalty
                term = not self._cliff_resets_to_start
                ns = (
                    self.pos_to_state(self._start_pos)
                    if self._cliff_resets_to_start
                    else self.pos_to_state(next_pos)
                )
            else:
                rew = self._step_cost
                term = False
                ns = self.pos_to_state(next_pos)

            key = (ns, rew, term)
            outcome_probs[key] = outcome_probs.get(key, 0.0) + p

        return [(prob, ns, rew, term) for (ns, rew, term), prob in outcome_probs.items()]

    def step(self, action: int) -> StepOutcome:
        if self.is_goal(self._current_pos):
            s = self.pos_to_state(self._current_pos)
            return StepOutcome(s, 0.0, True, False, {"message": "Already at goal."})

        self._step_count += 1
        effective_action = action
        slipped = False

        if self._slip_prob > 0.0 and self._rng.random() < self._slip_prob:
            perpendicular = {
                self.ACTION_UP: (self.ACTION_LEFT, self.ACTION_RIGHT),
                self.ACTION_DOWN: (self.ACTION_LEFT, self.ACTION_RIGHT),
                self.ACTION_LEFT: (self.ACTION_UP, self.ACTION_DOWN),
                self.ACTION_RIGHT: (self.ACTION_UP, self.ACTION_DOWN),
            }
            perp_actions = perpendicular[action]
            effective_action = self._rng.choice(perp_actions)
            slipped = True

        next_pos = self._attempt_move(self._current_pos, effective_action)

        if self.is_goal(next_pos):
            self._current_pos = next_pos
            reward = self._goal_reward
            terminated = True
        elif self.is_cliff(next_pos):
            reward = self._cliff_penalty
            if self._cliff_resets_to_start:
                self._current_pos = self._start_pos
                terminated = False
            else:
                self._current_pos = next_pos
                terminated = True
        else:
            self._current_pos = next_pos
            reward = self._step_cost
            terminated = False

        truncated = False
        if not terminated and self._max_steps is not None and self._step_count >= self._max_steps:
            truncated = True

        s = self.pos_to_state(self._current_pos)
        return StepOutcome(
            observation=s,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"pos": self._current_pos, "slipped": slipped},
        )

    def render(self) -> str:
        rows = []
        for r in range(self._height):
            line = []
            for c in range(self._width):
                pos = (r, c)
                if pos == self._current_pos:
                    line.append("A")
                elif pos in self._walls:
                    line.append("#")
                elif pos in self._goals:
                    line.append("G")
                elif pos in self._cliffs:
                    line.append("C")
                elif pos == self._start_pos:
                    line.append("S")
                else:
                    line.append(".")
            rows.append(" ".join(line))
        return "\n".join(rows)

    @classmethod
    def make_simple_grid(cls) -> GridworldEnv:
        """Create a classic 4x4 gridworld with one wall."""
        grid = [
            "S...",
            ".#..",
            "....",
            "...G",
        ]
        return cls(grid=grid, step_cost=-0.04, goal_reward=1.0)

    @classmethod
    def make_cliff_grid(cls) -> GridworldEnv:
        """Create the classic Sutton & Barto Cliff Walking environment (4x12).

        Row 3 (bottom): Start at (3,0), Cliff at (3,1..10), Goal at (3,11).
        Falling into cliff incurs -100 reward and sends the agent back to Start.
        """
        grid = [
            "............",
            "............",
            "............",
            "SCCCCCCCCCCG",
        ]
        return cls(
            grid=grid,
            step_cost=-1.0,
            goal_reward=0.0,
            cliff_penalty=-100.0,
            cliff_resets_to_start=True,
            max_steps=500,
        )
