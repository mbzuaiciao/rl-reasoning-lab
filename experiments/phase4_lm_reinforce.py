"""Phase 4 Experiment: LM REINFORCE from Random vs. SFT Initialization.

Demonstrates:
1. Why RL from scratch on autoregressive sequence generation faces an extreme
   sparse-reward needle-in-a-haystack exploration problem.
2. How SFT initialization ("warm start") provides an inductive prior that places
   initial rollouts in high-probability reward territory, allowing policy gradients
   to climb the reward surface effectively.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.lm_policy_gradient import train_lm_reinforce
from rl_reasoning_lab.environments.language_tasks import ArithmeticTask
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 4: LM REINFORCE — Random Init vs. SFT Warm-Start[/bold cyan]\n"
            "Testing whether policy gradients can discover correct reasoning from scratch\n"
            "vs. refining an initial supervised policy.",
            title="RL Reasoning Lab — Phase 4",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = ArithmeticTask(min_operands=2, max_operands=2, min_val=0, max_val=9)
    reward_fn = LanguageRewardFunction(task.verify)

    train_data = task.generate_dataset(size=80, seed=seed)
    prompts = [ex.prompt for ex in train_data]

    # 1. Train SFT warm-start model
    console.print("[dim]Pre-training warm-start model with brief SFT (15 epochs)...[/dim]")
    sft_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    val_data = task.generate_dataset(size=20, seed=seed + 1)
    train_sft(
        model=sft_model,
        train_examples=train_data,
        val_examples=val_data,
        tokenizer=tok,
        num_epochs=15,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    # 2. Random model
    torch.manual_seed(seed)
    random_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )

    # 3. Run REINFORCE on both
    console.print("[dim]Running REINFORCE on Random Init model...[/dim]")
    res_random = train_lm_reinforce(
        model=random_model,
        prompts=prompts,
        tokenizer=tok,
        reward_fn=reward_fn,
        num_iterations=20,
        batch_size=16,
        lr=1e-3,
        seed=seed,
    )

    console.print("[dim]Running REINFORCE on SFT Warm-Started model...[/dim]")
    res_sft = train_lm_reinforce(
        model=sft_model,
        prompts=prompts,
        tokenizer=tok,
        reward_fn=reward_fn,
        num_iterations=20,
        batch_size=16,
        lr=1e-3,
        seed=seed,
    )

    # Comparative Table
    table = Table(title="LM REINFORCE: Random Init vs. SFT Warm-Start")
    table.add_column("Iteration", justify="center")
    table.add_column("Random Init Reward", justify="right")
    table.add_column("Random Init Exact Match", justify="right")
    table.add_column("SFT Warm Reward", justify="right")
    table.add_column("SFT Warm Exact Match", justify="right")

    for i in range(0, 20, 4):
        table.add_row(
            str(i + 1),
            f"{res_random.mean_rewards[i]:.3f}",
            f"{res_random.exact_match_rates[i]:.1%}",
            f"{res_sft.mean_rewards[i]:.3f}",
            f"{res_sft.exact_match_rates[i]:.1%}",
        )
    # Final iteration
    table.add_row(
        "20 (Final)",
        f"{res_random.mean_rewards[-1]:.3f}",
        f"{res_random.exact_match_rates[-1]:.1%}",
        f"{res_sft.mean_rewards[-1]:.3f}",
        f"{res_sft.exact_match_rates[-1]:.1%}",
    )
    console.print(table)

    console.print("\n[bold]Key Empirical Takeaways:[/bold]")
    console.print(
        "1. [yellow]Sparse Reward Collapse in Random Init:[/yellow] With random token distributions, "
        "the probability of generating valid syntax followed by the exact numerical answer is negligible. "
        "The model receives near-zero rewards, yielding near-zero gradients, preventing learning without massive exploration."
    )
    console.print(
        "2. [green]SFT as Exploration Jumpstart:[/green] Supervised fine-tuning guides the policy into the "
        "manifold of valid syntax and plausible answers. From there, REINFORCE receives frequent non-zero "
        "rewards and steadily elevates accuracy."
    )


if __name__ == "__main__":
    run_experiment()
