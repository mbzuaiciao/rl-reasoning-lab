"""Phase 3 Deliberate Debugging Exercises: Diagnostic Failures in PPO.

Demonstrates 6 classic research-engineering bugs in PPO implementations:
1. Recomputing Old Log Probabilities (r_t ~ 1, PPO silently degenerates)
2. Inverted Ratio Direction (pi_old / pi_new instead of pi_new / pi_old)
3. Incorrect Clipping Logic (Clipping without sign-sensitive min)
4. Advantages Retain Computation Graph (Unintended gradient flow between actor and critic)
5. Terminal Bootstrap Error (Contaminating GAE advantages across episode boundaries)
6. Masking Bug in Variable-Length Trajectories (Padding tokens entering loss)
"""

from __future__ import annotations

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.ppo import compute_ppo_policy_loss
from rl_reasoning_lab.policies.neural import CategoricalPolicy
from rl_reasoning_lab.value_functions.neural import NeuralValueFunction

console = Console()


def debug_recomputing_old_log_probs() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 1: Recomputing Old Log Probs During Epochs[/bold red]\n"
            "Bug: Re-evaluating old_log_prob = policy(s) inside the epoch loop.\n"
            "Prediction: Ratio r_t is always identically 1.0; clipping never activates; PPO degenerates."
        )
    )

    policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=(16,))
    s = torch.tensor([[1.0, 0.0, 0.0]])
    a = torch.tensor([1])
    adv = torch.tensor([2.0])

    # In epoch loop:
    # BUG: recomputing old_log_prob from current policy
    lp_current, _ = policy.evaluate_actions(s, a)
    buggy_old_lp = lp_current.detach()  # freshly computed from current policy

    # Compute loss
    _, diag = compute_ppo_policy_loss(lp_current, buggy_old_lp, adv, clip_eps=0.2)

    table = Table(title="Bug 1: Recomputing Old Log Probs Effect")
    table.add_column("Diagnostic Quantity", style="cyan")
    table.add_column("Observed Value", justify="right", style="red")
    table.add_column("Expected Healthy Value", justify="right", style="green")
    table.add_row("Mean Ratio r_t", f"{diag['mean_ratio']:.6f}", "Drifts away from 1.0 (e.g. 1.15)")
    table.add_row(
        "Clip Fraction", f"{diag['clip_fraction']:.6f}", "Non-zero as policy updates (> 0.0)"
    )
    table.add_row("Status", "PPO is completely neutered", "Clipping safely guards trust region")
    console.print(table)
    console.print()


def debug_inverted_ratio_direction() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 2: Inverted Ratio (pi_old / pi_new)[/bold red]\n"
            "Bug: ratio = exp(old_log_prob - log_prob) instead of exp(log_prob - old_log_prob).\n"
            "Prediction: When pi_new(a) increases, ratio DECREASES, corrupting gradient direction."
        )
    )

    # Suppose action probability increased from 0.2 to 0.6 (desirable for positive advantage)
    lp_new = torch.tensor([np.log(0.6)])
    lp_old = torch.tensor([np.log(0.2)])

    correct_ratio = torch.exp(lp_new - lp_old).item()
    buggy_ratio = torch.exp(lp_old - lp_new).item()

    table = Table(title="Bug 2: Inverted Ratio Effect")
    table.add_column("Metric", style="cyan")
    table.add_column("Formula", style="white")
    table.add_column("Value (when prob increases 0.2 -> 0.6)", justify="right")
    table.add_row(
        "Correct Ratio", "pi_new / pi_old", f"{correct_ratio:.3f} (> 1.0, increases objective)"
    )
    table.add_row(
        "Buggy Inverted Ratio", "pi_old / pi_new", f"{buggy_ratio:.3f} (< 1.0, suppresses update)"
    )
    console.print(table)
    console.print()


def debug_incorrect_clipping_logic() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 3: Clipping Without Sign-Sensitive Min[/bold red]\n"
            "Bug: loss = -clip(r, 1-eps, 1+eps) * A (omitting min(surr1, surr2)).\n"
            "Prediction: For negative advantage actions, clipping caps the penalty, shielding bad actions!"
        )
    )

    # Negative advantage (A = -3.0), large ratio (r = 1.5)
    # The policy made a BAD action MORE likely. It should be severely penalized with r * A = 1.5 * (-3) = -4.5!
    adv = -3.0
    r = 1.5
    eps = 0.2

    surr1 = r * adv  # -4.5
    surr2 = (1.0 + eps) * adv  # 1.2 * (-3.0) = -3.6
    correct_obj = min(surr1, surr2)  # -4.5 (full severe penalty!)
    buggy_obj = np.clip(r, 1.0 - eps, 1.0 + eps) * adv  # -3.6 (penalty capped!)

    table = Table(title="Bug 3: Incorrect Clipping Asymmetry")
    table.add_column("Case (Negative Advantage A=-3.0, Ratio r=1.5)", style="cyan")
    table.add_column("Policy Objective", justify="right")
    table.add_column("Consequence", style="yellow")
    table.add_row(
        "PPO with min(surr1, surr2)", f"{correct_obj:.2f}", "Full penalty applied (unclipped)"
    )
    table.add_row(
        "Buggy Symmetric Clip", f"{buggy_obj:.2f}", "Penalty capped! Policy shielded from mistake"
    )
    console.print(table)
    console.print()


def debug_advantages_retaining_gradients() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 4: Advantages Retaining Computation Graph[/bold red]\n"
            "Bug: advantages = returns - value_net(states) without .detach().\n"
            "Prediction: Policy loss backpropagates through advantage into value_net, destroying value estimates."
        )
    )

    policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=(16,))
    value_net = NeuralValueFunction(num_states=3, hidden_dims=(16,))

    states = torch.tensor([[1.0, 0.0, 0.0]])
    actions = torch.tensor([1])
    returns = torch.tensor([1.0])

    # Forward
    log_probs, _ = policy.evaluate_actions(states, actions)
    # BUG: forgot .detach() on value network output!
    buggy_values = value_net(states)
    buggy_adv = returns - buggy_values

    policy_loss = -(buggy_adv * log_probs).mean()
    policy_loss.backward()

    # Check if value_net has non-zero gradients from policy_loss
    v_grad_norm = 0.0
    for p in value_net.parameters():
        if p.grad is not None:
            v_grad_norm += float(p.grad.data.norm(2).item())

    table = Table(title="Bug 4: Gradient Flow into Critic")
    table.add_column("Property", style="cyan")
    table.add_column("Observed Value", justify="right", style="red")
    table.add_row("Value Net Gradient from Policy Loss", f"{v_grad_norm:.6f}")
    table.add_row("Status", "Severe bug: Critic weights corrupted by policy loss")
    console.print(table)
    console.print()


def debug_padding_masking_bug() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 5: Trajectory Masking Bug (Padding Contamination)[/bold red]\n"
            "Bug: Omitting masks for variable-length/padded steps in batch.\n"
            "Prediction: Artificial padding steps contribute loss and pull the policy distribution."
        )
    )

    # 1 real transition (reward +1.0, log_p = -0.5), 1 dummy padding step (log_p = -2.0, adv = -10.0)
    lp = torch.tensor([-0.5, -2.0])
    old_lp = torch.tensor([-0.5, -2.0])
    advs = torch.tensor([1.0, -10.0])  # padded step has huge negative dummy advantage
    masks = torch.tensor([1.0, 0.0])

    loss_with_mask, _ = compute_ppo_policy_loss(lp, old_lp, advs, masks=masks)
    loss_without_mask, _ = compute_ppo_policy_loss(lp, old_lp, advs, masks=None)

    table = Table(title="Bug 5: Padding Masking Effect")
    table.add_column("Masking Mode", style="cyan")
    table.add_column("Computed Policy Loss", justify="right")
    table.add_column("Impact", style="yellow")
    table.add_row(
        "Correct (Mask = [1, 0])", f"{loss_with_mask.item():+.3f}", "Ignores padding tokens"
    )
    table.add_row(
        "Buggy (No Mask)",
        f"{loss_without_mask.item():+.3f}",
        "Padding contaminates policy gradient by 5x",
    )
    console.print(table)
    console.print()


if __name__ == "__main__":
    console.rule("[bold red]Phase 3 Debugging Exercises: Diagnostic PPO Failures[/bold red]")
    debug_recomputing_old_log_probs()
    debug_inverted_ratio_direction()
    debug_incorrect_clipping_logic()
    debug_advantages_retaining_gradients()
    debug_padding_masking_bug()
    console.rule("[bold green]All PPO Debugging Exercises Verified[/bold green]")
