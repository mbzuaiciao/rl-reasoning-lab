"""Phase 2 Experiment: Policy Gradient Variance and Variance Reduction.

Empirically compares gradient variance across N independent trajectory samples from a fixed policy:
1. Vanilla REINFORCE (Full episode return assigned to all steps)
2. Reward-to-Go (Causality: drop past rewards)
3. Reward-to-Go with Constant Baseline (G_t - mean(G))
4. Reward-to-Go with Value Baseline (G_t - V(s_t))

Measures:
- Mean gradient vector norm: ||E[g]||
- Component-wise variance: mean_j Var(g_j)
- Gradient norm variance: Var(||g||)
- Pairwise cosine similarity between independent gradient estimates
"""

from __future__ import annotations

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.advantage import (
    compute_full_returns,
    compute_reward_to_go,
)
from rl_reasoning_lab.algorithms.dynamic_programming import value_iteration
from rl_reasoning_lab.environments.corridor import CorridorEnv
from rl_reasoning_lab.policies.neural import CategoricalPolicy

console = Console()


def collect_single_trajectory_gradient(
    env: CorridorEnv,
    policy: CategoricalPolicy,
    estimator: str,
    v_table: np.ndarray | None = None,
    gamma: float = 0.95,
) -> np.ndarray:
    """Sample one trajectory from policy, compute its gradient vector g, and return as 1-D numpy array."""
    state, _ = env.reset()
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    log_probs_list: list[torch.Tensor] = []
    done = False

    while not done:
        action, log_prob = policy.sample_action_with_log_prob(state)
        step_outcome = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(step_outcome.reward)
        log_probs_list.append(log_prob)

        state = step_outcome.observation
        done = step_outcome.terminated or step_outcome.truncated

    # Compute advantages
    if estimator == "full":
        adv = compute_full_returns(rewards, gamma=gamma)
    elif estimator == "reward_to_go":
        adv = compute_reward_to_go(rewards, gamma=gamma)
    elif estimator == "const_baseline":
        rtg = compute_reward_to_go(rewards, gamma=gamma)
        adv = rtg - np.mean(rtg)
    elif estimator == "value_baseline":
        rtg = compute_reward_to_go(rewards, gamma=gamma)
        assert v_table is not None
        v_states = np.array([v_table[s] for s in states], dtype=np.float32)
        adv = rtg - v_states
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    adv_t = torch.from_numpy(adv).float()
    log_probs = torch.stack(log_probs_list)

    # Compute policy gradient
    policy.zero_grad()
    loss = -(adv_t * log_probs).sum()
    loss.backward()

    # Flatten gradients into 1-D vector
    grads = []
    for p in policy.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.cpu().view(-1))
    return torch.cat(grads).numpy()


def run_gradient_variance_experiment(num_samples: int = 200, seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CorridorEnv(length=6, goal_reward=10.0, step_cost=-0.1, slip_prob=0.1)
    policy = CategoricalPolicy(
        num_states=env.num_states, num_actions=env.num_actions, hidden_dims=(16,)
    )

    # Compute ground truth V* for value baseline
    dp_res = value_iteration(env, gamma=0.95, theta=1e-8)
    v_optimal = dp_res.value_function.as_array()

    estimators = [
        ("Full Return (Vanilla)", "full"),
        ("Reward-to-Go (Causal)", "reward_to_go"),
        ("Reward-to-Go + Const Baseline", "const_baseline"),
        ("Reward-to-Go + Value Baseline", "value_baseline"),
    ]

    console.print(
        Panel.fit(
            f"[bold cyan]Policy Gradient Variance Analysis ({num_samples} Samples from Frozen Policy)[/bold cyan]\n"
            "Environment: 6-state Corridor with slip=0.1. Policy: Fixed randomly initialized MLP."
        )
    )

    table = Table(title=f"Gradient Estimator Variance Comparison (N={num_samples})")
    table.add_column("Estimator", style="cyan")
    table.add_column("Mean Grad Norm ||E[g]||", justify="right", style="green")
    table.add_column("Component Variance (mean Var(g_j))", justify="right", style="red")
    table.add_column("Grad Norm Variance Var(||g||)", justify="right", style="yellow")
    table.add_column("Pairwise Cosine Similarity", justify="right", style="magenta")

    for label, mode in estimators:
        grad_list: list[np.ndarray] = []
        for _ in range(num_samples):
            g = collect_single_trajectory_gradient(env, policy, mode, v_table=v_optimal, gamma=0.95)
            grad_list.append(g)

        grads = np.array(grad_list)  # Shape (N, D)
        mean_grad = np.mean(grads, axis=0)
        mean_grad_norm = float(np.linalg.norm(mean_grad))

        # Component-wise variance
        component_var = float(np.mean(np.var(grads, axis=0)))

        # Norm variance
        norms = np.linalg.norm(grads, axis=1)
        norm_var = float(np.var(norms))

        # Pairwise cosine similarity among 100 randomly sampled pairs
        cos_sims = []
        rng = np.random.default_rng(seed)
        for _ in range(200):
            i, j = rng.choice(num_samples, size=2, replace=False)
            norm_i = norms[i]
            norm_j = norms[j]
            if norm_i > 1e-8 and norm_j > 1e-8:
                sim = float(np.dot(grads[i], grads[j]) / (norm_i * norm_j))
                cos_sims.append(sim)
        mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

        table.add_row(
            label,
            f"{mean_grad_norm:.3f}",
            f"{component_var:.3f}",
            f"{norm_var:.3f}",
            f"{mean_cos:+.3f}",
        )

    console.print(table)
    console.print(
        "[dim]Observations:\n"
        "1. Full Return assigns all trajectory rewards to every action, producing the highest component variance.\n"
        "2. Reward-to-Go eliminates variance from past rewards while preserving the exact expected gradient direction.\n"
        "3. Baselines subtract state values, centering advantages around zero and dramatically improving cosine similarity.[/dim]\n"
    )


if __name__ == "__main__":
    console.rule("[bold magenta]RL Reasoning Lab — Phase 2 Gradient Variance[/bold magenta]")
    run_gradient_variance_experiment(num_samples=200, seed=42)
