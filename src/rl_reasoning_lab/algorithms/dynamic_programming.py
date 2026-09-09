"""Model-based dynamic programming algorithms: Policy Evaluation, Policy Iteration, Value Iteration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_reasoning_lab.environments.base import Environment
from rl_reasoning_lab.policies.base import Policy
from rl_reasoning_lab.policies.tabular import DeterministicTabularPolicy
from rl_reasoning_lab.value_functions.tabular import (
    TabularQFunction,
    TabularValueFunction,
)


@dataclass
class DPResult:
    """Results from dynamic programming algorithms."""

    value_function: TabularValueFunction
    policy: DeterministicTabularPolicy | None
    num_iterations: int
    delta_history: list[float]


def policy_evaluation(
    env: Environment,
    policy: Policy,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> DPResult:
    """Compute the state-value function V^pi(s) using iterative Bellman expectation backups.

    Bellman Expectation Equation:
        V_{k+1}(s) = sum_a pi(a | s) sum_{s', r} p(s', r | s, a) [r + gamma * V_k(s')]

    Args:
        env: The environment exposing transition dynamics via get_transitions(s, a).
        policy: The policy pi to evaluate.
        gamma: Discount factor in [0, 1).
        theta: Convergence threshold on max_s |V_{k+1}(s) - V_k(s)|.
        max_iterations: Maximum number of sweep iterations.

    Returns:
        DPResult containing the computed TabularValueFunction and convergence history.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0.0, 1.0), got {gamma}")

    num_states = env.num_states
    num_actions = env.num_actions
    v = TabularValueFunction(num_states, initial_value=0.0)
    delta_history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        new_values = np.zeros(num_states, dtype=np.float64)

        for s in range(num_states):
            action_probs = policy.action_probabilities(s)
            state_val = 0.0

            for a in range(num_actions):
                prob_a = action_probs[a]
                if prob_a <= 0.0:
                    continue

                transitions = env.get_transitions(s, a)
                expected_return = 0.0
                for p_trans, next_s, reward, is_terminal in transitions:
                    future_val = 0.0 if is_terminal else v[next_s]
                    expected_return += p_trans * (reward + gamma * future_val)

                state_val += prob_a * expected_return

            new_values[s] = state_val
            delta = max(delta, abs(new_values[s] - v[s]))

        v.values = new_values
        delta_history.append(delta)

        if delta < theta:
            return DPResult(
                value_function=v,
                policy=None,
                num_iterations=iteration,
                delta_history=delta_history,
            )

    return DPResult(
        value_function=v,
        policy=None,
        num_iterations=max_iterations,
        delta_history=delta_history,
    )


def value_iteration(
    env: Environment,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> DPResult:
    """Compute the optimal state-value function V*(s) and extract optimal policy pi*(s).

    Bellman Optimality Equation:
        V_{k+1}(s) = max_a sum_{s', r} p(s', r | s, a) [r + gamma * V_k(s')]

    Because the Bellman optimality operator is a gamma-contraction in the infinity norm,
    ||V_{k+1} - V*||_inf <= gamma * ||V_k - V*||_inf.

    Args:
        env: Environment with get_transitions(s, a).
        gamma: Discount factor in [0, 1).
        theta: Convergence tolerance.
        max_iterations: Maximum iterations.

    Returns:
        DPResult containing optimal TabularValueFunction, optimal DeterministicTabularPolicy,
        number of iterations, and history of maximum Bellman deltas.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0.0, 1.0), got {gamma}")

    num_states = env.num_states
    num_actions = env.num_actions
    v = TabularValueFunction(num_states, initial_value=0.0)
    delta_history: list[float] = []

    for _iteration in range(1, max_iterations + 1):
        delta = 0.0
        new_values = np.zeros(num_states, dtype=np.float64)

        for s in range(num_states):
            q_sa = np.zeros(num_actions, dtype=np.float64)
            for a in range(num_actions):
                transitions = env.get_transitions(s, a)
                for p_trans, next_s, reward, is_terminal in transitions:
                    future_val = 0.0 if is_terminal else v[next_s]
                    q_sa[a] += p_trans * (reward + gamma * future_val)

            best_val = np.max(q_sa)
            new_values[s] = best_val
            delta = max(delta, abs(best_val - v[s]))

        v.values = new_values
        delta_history.append(delta)

        if delta < theta:
            break

    # Extract greedy policy from converged value function
    optimal_actions = np.zeros(num_states, dtype=np.int64)
    for s in range(num_states):
        q_sa = np.zeros(num_actions, dtype=np.float64)
        for a in range(num_actions):
            transitions = env.get_transitions(s, a)
            for p_trans, next_s, reward, is_terminal in transitions:
                future_val = 0.0 if is_terminal else v[next_s]
                q_sa[a] += p_trans * (reward + gamma * future_val)
        optimal_actions[s] = int(np.argmax(q_sa))

    policy = DeterministicTabularPolicy(optimal_actions, num_actions)
    return DPResult(
        value_function=v,
        policy=policy,
        num_iterations=len(delta_history),
        delta_history=delta_history,
    )


def compute_q_from_v(
    env: Environment,
    v: TabularValueFunction,
    gamma: float,
) -> TabularQFunction:
    """Compute Q(s, a) from a given state-value function V(s):

    Q(s, a) = sum_{s', r} p(s', r | s, a) [r + gamma * V(s')]
    """
    q = TabularQFunction(env.num_states, env.num_actions)
    for s in range(env.num_states):
        for a in range(env.num_actions):
            expected_q = 0.0
            for p_trans, next_s, reward, is_terminal in env.get_transitions(s, a):
                future_val = 0.0 if is_terminal else v[next_s]
                expected_q += p_trans * (reward + gamma * future_val)
            q[(s, a)] = expected_q
    return q
