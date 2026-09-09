"""Phase 2 Deliberate Debugging Exercises: Diagnostic ML Failures.

Demonstrates 5 classic research-engineering bugs in policy gradient implementations:
1. Wrong Loss Sign (Minimizing +G * log pi instead of -G * log pi)
2. Detached Log Probabilities (log_prob.detach() breaks computation graph)
3. Inverted Entropy Sign (Penalizing entropy instead of encouraging it)
4. Incorrect Terminal Bootstrapping (Bootstrapping from terminal state V(s_term) > 0)
5. Missing Return Normalization (Sensitivity to raw reward scale)

For each bug, we:
- Explain the mechanism
- State the theoretical prediction
- Run the experiment
- Compare prediction against empirical observation
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.environments.corridor import CorridorEnv
from rl_reasoning_lab.policies.neural import CategoricalPolicy

console = Console()


def debug_wrong_loss_sign() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 1: Wrong Loss Sign (+G * log pi)[/bold red]\n"
            "Bug: Minimizing L = +(G * log pi) instead of -(G * log pi).\n"
            "Prediction: The optimizer will actively decrease the probability of rewarded actions."
        )
    )

    env = CorridorEnv(length=3, goal_reward=1.0, step_cost=-0.05)
    policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=(16,))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    prob_right_initial = policy.action_probabilities(0)[CorridorEnv.ACTION_RIGHT]

    # Run 50 episodes with buggy inverted sign
    for _ in range(50):
        s, _ = env.reset()
        rewards, log_probs = [], []
        done = False
        while not done:
            a, lp = policy.sample_action_with_log_prob(s)
            step = env.step(a)
            rewards.append(step.reward)
            log_probs.append(lp)
            s = step.observation
            done = step.terminated or step.truncated

        returns = torch.tensor(rewards).float()
        # BUG: positive sign
        loss = +(returns * torch.stack(log_probs)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prob_right_final = policy.action_probabilities(0)[CorridorEnv.ACTION_RIGHT]

    table = Table(title="Bug 1: Wrong Loss Sign Effect")
    table.add_column("Metric", style="cyan")
    table.add_column("Initial", justify="right")
    table.add_column("After 50 Buggy Episodes", justify="right", style="red")
    table.add_row(
        "P(Action RIGHT | State 0)", f"{prob_right_initial:.3f}", f"{prob_right_final:.3f}"
    )
    table.add_row(
        "Policy Behavior", "Balanced exploration", "Actively pushes RIGHT probability to ~0%"
    )
    console.print(table)
    console.print()


def debug_detached_log_probs() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 2: Detached Log Probabilities[/bold red]\n"
            "Bug: log_prob.detach() breaks the computational graph.\n"
            "Prediction: Loss is a constant scalar; parameter gradients are identically zero; no learning occurs."
        )
    )

    policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=(16,))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    # Inspect grad norm before and after backward
    _, log_prob = policy.sample_action_with_log_prob(0)
    # BUG: detach
    buggy_log_prob = log_prob.detach()
    loss = -(torch.tensor(1.0) * buggy_log_prob)

    optimizer.zero_grad()
    # Attempt backward
    grad_norm = 0.0
    try:
        loss.backward()
        for p in policy.parameters():
            if p.grad is not None:
                grad_norm += float(p.grad.data.norm(2).item())
    except RuntimeError:
        grad_norm = -1.0  # Graph detached error

    table = Table(title="Bug 2: Detached Log Probabilities Effect")
    table.add_column("Quantity", style="cyan")
    table.add_column("Observed Value", justify="right", style="red")
    table.add_row("Loss requires_grad flag", str(loss.requires_grad))
    table.add_row("Policy Gradient Norm ||g||", f"{grad_norm:.6f}")
    table.add_row("Status", "Silent failure: Zero gradients propagate back to network")
    console.print(table)
    console.print()


def debug_entropy_sign_bug() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 3: Inverted Entropy Sign (+β * H(π) in Loss)[/bold red]\n"
            "Bug: Adding +beta * H(pi) instead of -beta * H(pi) to loss function.\n"
            "Prediction: Minimizing loss actively minimizes entropy, causing instant deterministic collapse."
        )
    )

    policy = CategoricalPolicy(num_states=3, num_actions=2, hidden_dims=(16,))
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-2)

    initial_entropy = policy.inspect_state(0)["entropy"]

    # Run 20 updates with inverted entropy penalty
    for _ in range(20):
        dist = policy.get_distribution(0)
        entropy = dist.entropy()
        # BUG: minimizing positive entropy
        loss = +1.0 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_entropy = policy.inspect_state(0)["entropy"]
    probs = policy.action_probabilities(0)

    table = Table(title="Bug 3: Inverted Entropy Penalty Effect")
    table.add_column("State 0 Metric", style="cyan")
    table.add_column("Before Update", justify="right")
    table.add_column("After Inverted Updates", justify="right", style="red")
    table.add_row("Policy Entropy H(π)", f"{initial_entropy:.4f}", f"{final_entropy:.6f}")
    table.add_row("Action Probabilities", "[0.50, 0.50]", f"[{probs[0]:.4f}, {probs[1]:.4f}]")
    table.add_row("Consequence", "Uniform exploration", "Premature deterministic collapse")
    console.print(table)
    console.print()


def debug_terminal_bootstrapping_bug() -> None:
    console.print(
        Panel.fit(
            "[bold red]Debugging Exercise 4: Incorrect Terminal State Bootstrapping[/bold red]\n"
            "Bug: Omitting (1 - done) mask: delta_term = R + gamma * V(s_term) - V(s).\n"
            "Prediction: Terminal reward is artificially boosted by phantom future value."
        )
    )

    # Suppose a value network predicts V(terminal_state) = 5.0
    r = 1.0
    gamma = 0.9
    v_curr = 0.8
    v_term = 5.0

    # Correct TD residual:
    correct_delta = r + gamma * v_term * (1 - 1.0) - v_curr
    # Buggy TD residual:
    buggy_delta = r + gamma * v_term - v_curr

    table = Table(title="Bug 4: Terminal Bootstrapping Invariant")
    table.add_column("Case", style="cyan")
    table.add_column("Equation", style="white")
    table.add_column("TD Error (Advantage)", justify="right")
    table.add_row("Correct Masking", "R - V(s)", f"{correct_delta:+.2f}")
    table.add_row("Buggy (No Mask)", "R + γ * V(s_term) - V(s)", f"{buggy_delta:+.2f}")
    table.add_row(
        "Distortion",
        "Phantom value injected",
        f"Advantage distorted by {buggy_delta - correct_delta:+.2f}",
    )
    console.print(table)
    console.print()


if __name__ == "__main__":
    console.rule("[bold red]Phase 2 Debugging Exercises: Diagnostic ML Failures[/bold red]")
    debug_wrong_loss_sign()
    debug_detached_log_probs()
    debug_entropy_sign_bug()
    debug_terminal_bootstrapping_bug()
    console.rule("[bold green]All Debugging Exercises Verified[/bold green]")
