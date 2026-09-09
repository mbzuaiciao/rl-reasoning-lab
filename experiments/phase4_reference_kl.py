"""Phase 4 Experiment: Reference Policy KL Penalty Ablation.

Demonstrates:
1. The effect of the reference KL penalty coefficient beta in:
       r_adjusted(s, a) = r(s, a) - beta * (log pi_theta(a|s) - log pi_ref(a|s))
2. Behavior across beta in {0.0, 0.05, 0.2, 1.0}:
   - beta = 0.0: Unconstrained policy shift; can lead to drift or mode collapse.
   - beta = 0.05 - 0.2: Balanced reward improvement while staying near reference manifold.
   - beta = 1.0: Over-regularized policy unable to deviate from reference policy.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.lm_ppo import train_lm_ppo
from rl_reasoning_lab.environments.language_tasks import ArithmeticTask
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 4: Reference Model KL Penalty Ablation[/bold cyan]\n"
            "Studying policy drift, stability, and reward tradeoffs across "
            "beta in {0.0, 0.05, 0.2, 1.0}",
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
    val_data = task.generate_dataset(size=20, seed=seed + 1)
    prompts = [ex.prompt for ex in train_data]

    # Pre-train a common SFT base model
    console.print("[dim]Pre-training shared base model with SFT...[/dim]")
    base_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    train_sft(
        model=base_model,
        train_examples=train_data,
        val_examples=val_data,
        tokenizer=tok,
        num_epochs=12,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    betas = [0.0, 0.05, 0.2, 1.0]
    results = {}

    for beta in betas:
        console.print(f"[dim]Running PPO with beta = {beta}...[/dim]")
        # Fresh copy of weights
        policy = TinyTransformer(
            vocab_size=tok.vocab_size,
            d_model=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
        )
        policy.load_state_dict(base_model.state_dict())

        # Frozen reference model
        ref_model = TinyTransformer(
            vocab_size=tok.vocab_size,
            d_model=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
        )
        ref_model.load_state_dict(base_model.state_dict())

        res = train_lm_ppo(
            model=policy,
            ref_model=ref_model,
            prompts=prompts,
            tokenizer=tok,
            reward_fn=reward_fn,
            num_iterations=12,
            rollout_batch_size=16,
            epochs_per_rollout=2,
            minibatch_size=8,
            lr=5e-4,
            ref_kl_coef=beta,
            clip_eps=0.2,
            seed=seed,
        )
        results[beta] = res

    # Summary Table
    table = Table(title="KL Penalty Ablation (PPO after 12 iterations)")
    table.add_column("Beta (KL Coef)", justify="center")
    table.add_column("Final Mean Reward", justify="right")
    table.add_column("Exact Match %", justify="right")
    table.add_column("Final Ref KL Divergence", justify="right")
    table.add_column("Clip Fraction", justify="right")

    for beta in betas:
        r = results[beta]
        final_reward = r.mean_rewards[-1] if r.mean_rewards else 0.0
        final_em = r.exact_match_rates[-1] if r.exact_match_rates else 0.0
        final_kl = r.ref_kls[-1] if r.ref_kls else 0.0
        final_clip = r.clip_fractions[-1] if r.clip_fractions else 0.0
        table.add_row(
            f"{beta:.2f}",
            f"{final_reward:.3f}",
            f"{final_em:.1%}",
            f"{final_kl:.4f}",
            f"{final_clip:.3f}",
        )

    console.print(table)

    console.print("\n[bold]Key Observations:[/bold]")
    console.print(
        "1. [yellow]Beta = 0.0 (No Regularization):[/yellow] The policy drifts furthest from the reference "
        "model (highest Ref KL divergence). Without a reference anchor, language policies are vulnerable "
        "to language degradation and out-of-distribution hallucinations."
    )
    console.print(
        "2. [green]Moderate Beta (0.05 - 0.2):[/green] Provides a smooth penalty keeping the policy anchored "
        "to the natural language prior while still optimizing the task reward signal."
    )
    console.print(
        "3. [cyan]High Beta (1.0):[/cyan] Strongly penalizes any divergence from pi_ref, keeping KL near zero "
        "but potentially dampening policy adaptation."
    )


if __name__ == "__main__":
    run_experiment()
