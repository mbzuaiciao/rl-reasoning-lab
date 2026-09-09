"""Temporal difference learning algorithms: Q-learning and SARSA."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rl_reasoning_lab.environments.base import Environment
from rl_reasoning_lab.policies.tabular import EpsilonGreedyPolicy
from rl_reasoning_lab.value_functions.tabular import TabularQFunction


@dataclass
class TDResult:
    """Diagnostic training records from temporal-difference learning."""

    q_function: TabularQFunction
    episode_returns: list[float]
    episode_lengths: list[int]
    td_errors_history: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)


def q_learning(
    env: Environment,
    num_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int | None = None,
) -> TDResult:
    """Tabular Q-learning: Off-policy temporal difference control.

    Update Rule:
        Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * max_a Q(S_{t+1}, a) - Q(S_t, A_t)]

    Why it is off-policy:
        The target policy is the greedy policy pi(a | s) = argmax_a Q(s, a), while
        the behavior policy generating exploratory transitions is epsilon-greedy.
    """
    rng = np.random.default_rng(seed)
    q = TabularQFunction(env.num_states, env.num_actions, initial_value=0.0)

    policy = EpsilonGreedyPolicy(q.values, epsilon=epsilon_start)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    mean_td_errors: list[float] = []
    epsilons: list[float] = []

    epsilon = epsilon_start

    for _ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        policy.epsilon = epsilon
        total_return = 0.0
        steps = 0
        ep_td_errors: list[float] = []

        done = False
        while not done:
            action = policy.sample_action(state, rng=rng)
            step_outcome = env.step(action)
            next_state = step_outcome.observation
            reward = step_outcome.reward
            terminated = step_outcome.terminated
            truncated = step_outcome.truncated
            done = terminated or truncated

            # Bellman optimality target
            if terminated:
                target = reward
            else:
                target = reward + gamma * float(np.max(q.values[next_state]))

            current_q = q[(state, action)]
            td_error = target - current_q
            ep_td_errors.append(abs(td_error))

            # In-place Q update
            q[(state, action)] = current_q + alpha * td_error

            total_return += reward
            steps += 1
            state = next_state

        episode_returns.append(total_return)
        episode_lengths.append(steps)
        mean_td_errors.append(float(np.mean(ep_td_errors)) if ep_td_errors else 0.0)
        epsilons.append(epsilon)

        # Decay exploration
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return TDResult(
        q_function=q,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        td_errors_history=mean_td_errors,
        epsilons=epsilons,
    )


def sarsa(
    env: Environment,
    num_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int | None = None,
) -> TDResult:
    """Tabular SARSA: On-policy temporal difference control.

    Update Rule:
        Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

    Why it is on-policy:
        The action A_{t+1} evaluated in the bootstrap target is sampled directly from the
        behavior policy pi(a | S_{t+1}), making Q reflect the return of exploratory actions.
    """
    rng = np.random.default_rng(seed)
    q = TabularQFunction(env.num_states, env.num_actions, initial_value=0.0)

    policy = EpsilonGreedyPolicy(q.values, epsilon=epsilon_start)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    mean_td_errors: list[float] = []
    epsilons: list[float] = []

    epsilon = epsilon_start

    for _ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        policy.epsilon = epsilon
        action = policy.sample_action(state, rng=rng)

        total_return = 0.0
        steps = 0
        ep_td_errors: list[float] = []

        done = False
        while not done:
            step_outcome = env.step(action)
            next_state = step_outcome.observation
            reward = step_outcome.reward
            terminated = step_outcome.terminated
            truncated = step_outcome.truncated
            done = terminated or truncated

            if terminated:
                target = reward
                next_action = 0
            else:
                next_action = policy.sample_action(next_state, rng=rng)
                target = reward + gamma * q[(next_state, next_action)]

            current_q = q[(state, action)]
            td_error = target - current_q
            ep_td_errors.append(abs(td_error))

            q[(state, action)] = current_q + alpha * td_error

            total_return += reward
            steps += 1
            state = next_state
            action = next_action

        episode_returns.append(total_return)
        episode_lengths.append(steps)
        mean_td_errors.append(float(np.mean(ep_td_errors)) if ep_td_errors else 0.0)
        epsilons.append(epsilon)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return TDResult(
        q_function=q,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        td_errors_history=mean_td_errors,
        epsilons=epsilons,
    )
