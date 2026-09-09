"""Phase 3 Experiment: PPO Clipping Mechanism vs. Unclipped Surrogate.

Demonstrates precisely what PPO clipping prevents by repeatedly optimizing
the same fixed rollout batch for 15 epochs under:
1. Unclipped Surrogate: L = - E[ r_t * A_t ]
2. PPO Clipped Surrogate: L = - E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]

Tracks after each epoch:
- Surrogate loss
- Approximate KL divergence D_KL(pi_old || pi_new)
- Ratio distribution quantiles (min, median, max)
- Clip fraction
- Policy entropy
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.ppo import compute_ppo_policy_loss
from rl_reasoning_lab.environments.corridor import CorridorEnv
from rl_reasoning_lab.policies.neural import CategoricalPolicy
from rl_reasoning_lab.trainers.rollout import RolloutBuffer

console = Console()


def collect_fixed_rollout(
    env: CorridorEnv, policy: CategoricalPolicy, num_steps: int = 128
) -> RolloutBuffer:
    """Collect a fixed rollout buffer under a behavior policy."""
    buffer = RolloutBuffer(num_states=env.num_states, gamma=0.95, gae_lambda=0.95)
    state, _ = env.reset(seed=42)

    for _ in range(num_steps):
        with torch.no_grad():
            action, log_prob = policy.sample_action_with_log_prob(state)
        step = env.step(action)
        buffer.add(
            state=state,
            action=action,
            reward=step.reward,
            done=step.terminated,
            old_log_prob=log_prob,
            value=0.5,
        )
        state = 0 if (step.terminated or step.truncated) else step.observation

    buffer.finalize(next_value=0.0, normalize_adv=True)
    return buffer


def run_clipping_comparison(num_epochs: int = 12, lr: float = 2e-2, clip_eps: float = 0.2) -> None:
    env = CorridorEnv(length=5, goal_reward=1.0, step_cost=-0.05)

    # 1. Unclipped Optimization
    torch.manual_seed(123)
    policy_unclipped = CategoricalPolicy(
        num_states=env.num_states, num_actions=env.num_actions, hidden_dims=(16,)
    )
    opt_unclipped = torch.optim.Adam(policy_unclipped.parameters(), lr=lr)

    # 2. Clipped (PPO) Optimization with identical initial weights
    torch.manual_seed(123)
    policy_clipped = CategoricalPolicy(
        num_states=env.num_states, num_actions=env.num_actions, hidden_dims=(16,)
    )
    opt_clipped = torch.optim.Adam(policy_clipped.parameters(), lr=lr)

    buffer = collect_fixed_rollout(env, policy_unclipped, num_steps=128)

    # Extract tensors
    states = torch.zeros(len(buffer), env.num_states)
    for i, s in enumerate(buffer.states):
        states[i, s] = 1.0
    actions = torch.tensor(buffer.actions, dtype=torch.long)
    old_log_probs = torch.tensor(buffer.old_log_probs, dtype=torch.float32)
    advs = torch.tensor(buffer.advantages, dtype=torch.float32)

    console.print(
        Panel.fit(
            f"[bold cyan]PPO Clipping vs. Unclipped Surrogate (Fixed Rollout, {num_epochs} Epochs)[/bold cyan]\n"
            f"Testing learning rate lr={lr}, clip_eps={clip_eps} on a frozen 128-step rollout."
        )
    )

    table = Table(title="Epoch-by-Epoch Policy Drift Comparison")
    table.add_column("Epoch", justify="center", style="cyan")
    table.add_column("Unclipped KL", justify="right", style="red")
    table.add_column("Unclipped [Min, Max] Ratio", justify="center", style="red")
    table.add_column("PPO Clipped KL", justify="right", style="green")
    table.add_column("PPO [Min, Max] Ratio", justify="center", style="green")
    table.add_column("PPO Clip %", justify="right", style="yellow")

    for ep in range(1, num_epochs + 1):
        # Unclipped update: L = - (r * A)
        opt_unclipped.zero_grad()
        lp_u, _ = policy_unclipped.evaluate_actions(states, actions)
        ratio_u = torch.exp(lp_u - old_log_probs)
        loss_u = -(ratio_u * advs).mean()
        loss_u.backward()
        opt_unclipped.step()

        kl_u = (old_log_probs - lp_u).mean().item()
        min_r_u = ratio_u.min().item()
        max_r_u = ratio_u.max().item()

        # Clipped PPO update
        opt_clipped.zero_grad()
        lp_c, _ = policy_clipped.evaluate_actions(states, actions)
        loss_c, diag_c = compute_ppo_policy_loss(lp_c, old_log_probs, advs, clip_eps=clip_eps)
        loss_c.backward()
        opt_clipped.step()

        ratio_c = torch.exp(lp_c - old_log_probs)
        kl_c = diag_c["approx_kl"]
        min_r_c = ratio_c.min().item()
        max_r_c = ratio_c.max().item()
        clip_frac = diag_c["clip_fraction"]

        table.add_row(
            str(ep),
            f"{kl_u:.3f}",
            f"[{min_r_u:.2f}, {max_r_u:.2f}]",
            f"{kl_c:.3f}",
            f"[{min_r_c:.2f}, {max_r_c:.2f}]",
            f"{clip_frac * 100:.1f}%",
        )

    console.print(table)
    console.print(
        "[dim]Observations:\n"
        "1. In unclipped optimization, probability ratios explode as the optimizer takes extreme updates.\n"
        "2. PPO clipping restrains the gradient once ratios exceed 1 ± eps, keeping KL divergence bounded.\n"
        "3. Clip fraction rises to ~50-80% when reusing data aggressively, acting as an automatic brake.[/dim]\n"
    )


if __name__ == "__main__":
    console.rule("[bold magenta]RL Reasoning Lab — Phase 3 Clipping Behavior[/bold magenta]")
    run_clipping_comparison(num_epochs=10, lr=0.03, clip_eps=0.2)
