"""Phase 2 Controlled Experiment: Multi-Seed Policy Gradient Benchmarks & Bootstrap Confidence Intervals.

Compares across multiple random seeds:
1. Vanilla REINFORCE (Full Return, No Baseline)
2. REINFORCE (Reward-to-Go, No Baseline)
3. REINFORCE + Learned Neural Value Baseline
4. One-step Online Actor-Critic

Reports:
- Mean and standard deviation across seeds
- Non-parametric 95% Bootstrap Confidence Intervals
- Mean policy entropy
- Final solve rate
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.policy_gradient import actor_critic, reinforce
from rl_reasoning_lab.environments.corridor import CorridorEnv
from rl_reasoning_lab.policies.neural import CategoricalPolicy
from rl_reasoning_lab.value_functions.neural import NeuralValueFunction

console = Console()


def compute_bootstrap_ci(
    data: list[float] | np.ndarray, num_resamples: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for the sample mean."""
    arr = np.asarray(data)
    n = len(arr)
    if n <= 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(42)
    boot_means = np.zeros(num_resamples)
    for i in range(num_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    return float(np.percentile(boot_means, low_pct)), float(np.percentile(boot_means, high_pct))


def run_multi_seed_comparison() -> None:
    seeds = [11, 22, 33, 44, 55, 66, 77]
    num_episodes = 250
    env_length = 5

    console.print(
        Panel.fit(
            f"[bold cyan]Multi-Seed Policy Gradient Comparison ({len(seeds)} Seeds, {num_episodes} Episodes)[/bold cyan]\n"
            f"Environment: {env_length}-state Corridor (goal reward +1.0, step cost -0.02, slip 0.0)."
        )
    )

    algorithms = [
        ("Vanilla REINFORCE (Full Return)", "reinforce_full"),
        ("REINFORCE (Reward-to-Go)", "reinforce_rtg"),
        ("REINFORCE + Neural Baseline", "reinforce_baseline"),
        ("Actor-Critic (1-Step TD)", "actor_critic"),
    ]

    table = Table(title="Multi-Seed Performance & 95% Bootstrap CI")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Mean Return (ep 150-250)", justify="right", style="green")
    table.add_column("Seed Std", justify="right", style="red")
    table.add_column("95% Bootstrap CI", justify="center", style="yellow")
    table.add_column("Mean Entropy", justify="right", style="magenta")
    table.add_column("Solve %", justify="right", style="blue")

    for label, algo_key in algorithms:
        late_returns: list[float] = []
        late_entropies: list[float] = []
        solved_seeds = 0

        for s in seeds:
            env = CorridorEnv(length=env_length, goal_reward=1.0, step_cost=-0.02, slip_prob=0.0)
            policy = CategoricalPolicy(
                num_states=env.num_states, num_actions=env.num_actions, hidden_dims=(16,)
            )
            val_net = NeuralValueFunction(num_states=env.num_states, hidden_dims=(16,))

            if algo_key == "reinforce_full":
                res = reinforce(
                    env,
                    policy,
                    policy_lr=5e-3,
                    num_episodes=num_episodes,
                    gamma=0.95,
                    return_type="full",
                    baseline_type=None,
                    entropy_coef=0.01,
                    seed=s,
                )
            elif algo_key == "reinforce_rtg":
                res = reinforce(
                    env,
                    policy,
                    policy_lr=5e-3,
                    num_episodes=num_episodes,
                    gamma=0.95,
                    return_type="reward_to_go",
                    baseline_type=None,
                    entropy_coef=0.01,
                    seed=s,
                )
            elif algo_key == "reinforce_baseline":
                res = reinforce(
                    env,
                    policy,
                    value_net=val_net,
                    policy_lr=5e-3,
                    value_lr=1e-2,
                    num_episodes=num_episodes,
                    gamma=0.95,
                    return_type="reward_to_go",
                    baseline_type="learned_value",
                    entropy_coef=0.01,
                    seed=s,
                )
            elif algo_key == "actor_critic":
                res = actor_critic(
                    env,
                    policy,
                    value_net=val_net,
                    policy_lr=5e-3,
                    value_lr=1e-2,
                    num_episodes=num_episodes,
                    gamma=0.95,
                    entropy_coef=0.01,
                    seed=s,
                )
            else:
                raise ValueError(algo_key)

            # Evaluate performance
            ep_ret = float(np.mean(res.episode_returns[-100:]))
            ep_ent = float(np.mean(res.entropies[-100:]))
            late_returns.append(ep_ret)
            late_entropies.append(ep_ent)

            # Solved check: does greedy policy choose RIGHT for non-terminal states?
            solved = True
            for st in range(env_length - 1):
                if np.argmax(policy.action_probabilities(st)) != CorridorEnv.ACTION_RIGHT:
                    solved = False
                    break
            if solved:
                solved_seeds += 1

        mean_ret = float(np.mean(late_returns))
        std_ret = float(np.std(late_returns))
        ci_low, ci_high = compute_bootstrap_ci(late_returns)
        mean_ent = float(np.mean(late_entropies))
        solve_pct = (solved_seeds / len(seeds)) * 100.0

        table.add_row(
            label,
            f"{mean_ret:+.3f}",
            f"±{std_ret:.3f}",
            f"[{ci_low:+.3f}, {ci_high:+.3f}]",
            f"{mean_ent:.3f}",
            f"{solve_pct:.0f}%",
        )

    console.print(table)
    console.print()


def run_entropy_regularization_study() -> None:
    """Experiment: Effect of Entropy Coefficient Beta on Exploration and Policy Collapse."""
    console.print(
        Panel.fit(
            "[bold cyan]Ablation: Entropy Bonus Coefficient Beta in REINFORCE[/bold cyan]\n"
            "Testing beta in [0.0, 0.01, 0.10] across 5 seeds on 5-state Corridor."
        )
    )

    betas = [0.0, 0.01, 0.10]
    seeds = [101, 202, 303, 404, 505]

    table = Table(title="Entropy Regularization Ablation (200 episodes)")
    table.add_column("Entropy Coef (β)", justify="center", style="cyan")
    table.add_column("Mean Return", justify="right", style="green")
    table.add_column("Final Policy Entropy", justify="right", style="magenta")
    table.add_column("Action Distribution (State 0)", justify="center", style="yellow")
    table.add_column("Qualitative Behavior", style="white")

    for beta in betas:
        returns = []
        final_entropies = []
        state0_probs_list = []

        for s in seeds:
            env = CorridorEnv(length=5, goal_reward=1.0, step_cost=-0.02)
            policy = CategoricalPolicy(num_states=5, num_actions=2, hidden_dims=(16,))
            res = reinforce(
                env,
                policy,
                policy_lr=5e-3,
                num_episodes=200,
                gamma=0.95,
                entropy_coef=beta,
                seed=s,
            )
            returns.append(float(np.mean(res.episode_returns[-50:])))
            final_entropies.append(float(res.entropies[-1]))
            state0_probs_list.append(policy.action_probabilities(0))

        mean_ret = float(np.mean(returns))
        mean_ent = float(np.mean(final_entropies))
        avg_probs = np.mean(state0_probs_list, axis=0)

        if beta == 0.0:
            behavior = "Premature collapse: commits early, prone to getting stuck in bad seeds"
        elif beta == 0.01:
            behavior = "Healthy balance: explores early, safely sharpens toward optimal action"
        else:
            behavior = "Under-exploitation: high entropy prevents decisive exploitation"

        table.add_row(
            f"{beta:.2f}",
            f"{mean_ret:+.3f}",
            f"{mean_ent:.3f}",
            f"L: {avg_probs[0]:.2f}, R: {avg_probs[1]:.2f}",
            behavior,
        )

    console.print(table)
    console.print()


if __name__ == "__main__":
    console.rule("[bold magenta]RL Reasoning Lab — Phase 2 Multi-Seed Benchmarks[/bold magenta]")
    run_multi_seed_comparison()
    run_entropy_regularization_study()
    console.rule("[bold green]Phase 2 Benchmarks Completed[/bold green]")
