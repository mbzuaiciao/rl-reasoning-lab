"""Visualization utilities for tabular values and policies."""

from __future__ import annotations

import numpy as np

from rl_reasoning_lab.environments.gridworld import GridworldEnv


def format_grid_values(env: GridworldEnv, values: np.ndarray, precision: int = 2) -> str:
    """Format a 1-D value array into a 2-D grid display."""
    lines = []
    for r in range(env.height):
        row_cells = []
        for c in range(env.width):
            pos = (r, c)
            s = env.pos_to_state(pos)
            if env.is_wall(pos):
                row_cells.append("  WALL  ")
            elif env.is_goal(pos):
                row_cells.append(f" {values[s]:+{precision + 4}.{precision}f}*")
            elif env.is_cliff(pos):
                row_cells.append(f" {values[s]:+{precision + 4}.{precision}f}C")
            else:
                row_cells.append(f" {values[s]:+{precision + 4}.{precision}f} ")
        lines.append(" | ".join(row_cells))
    return "\n".join(lines)


def format_grid_policy(env: GridworldEnv, policy_actions: np.ndarray) -> str:
    """Format an action array into directional arrows on the grid."""
    lines = []
    for r in range(env.height):
        row_chars = []
        for c in range(env.width):
            pos = (r, c)
            s = env.pos_to_state(pos)
            if env.is_wall(pos):
                row_chars.append("#")
            elif env.is_goal(pos):
                row_chars.append("G")
            elif env.is_cliff(pos):
                row_chars.append("C")
            else:
                a = int(policy_actions[s])
                row_chars.append(GridworldEnv.ACTION_ARROWS.get(a, "?"))
        lines.append(" ".join(row_chars))
    return "\n".join(lines)
