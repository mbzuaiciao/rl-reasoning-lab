"""Phase 5 Experiment: Rejection Sampling (Filtered SFT) vs. Policy-Gradient RL (PPO).

Compares:
1. Rejection Sampling (Filtered SFT):
   policy -> sample N traces -> ground-truth verifier filter -> train on correct traces.
2. PPO (Policy Gradient):
   policy -> sample rollouts -> compute advantages & clipped surrogate loss -> optimize.

Measures:
- In-distribution accuracy and OOD depth accuracy.
- Sample efficiency (gradient updates and accepted sample count).
- Intermediate reasoning trace validity.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.lm_ppo import train_lm_ppo
from rl_reasoning_lab.environments.language_tasks import Example
from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.evaluation.reasoning_metrics import (
    evaluate_reasoning_model,
    rejection_sampling_step,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft


def to_sft_examples(examples: list[ReasoningExample]) -> list[Example]:
    return [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in examples
    ]


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 5: Rejection Sampling (Filtered SFT) vs. PPO[/bold cyan]\n"
            "Comparing offline-filtered supervised updates against online\n"
            "importance-sampled policy gradient optimization.",
            title="RL Reasoning Lab — Phase 5",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGH"))

    train_data = task.generate_dataset(size=80, depths=[1, 2, 3], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2, 3], seed=seed + 1)
    test_data = task.generate_dataset(size=40, depths=[1, 2, 3, 4], seed=seed + 2)
    train_prompts = [ex.prompt for ex in train_data]

    # Pre-train shared SFT base model
    console.print("[dim]Pre-training shared base model with SFT...[/dim]")
    base_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=128,
    )
    train_sft(
        model=base_model,
        train_examples=to_sft_examples(train_data),
        val_examples=to_sft_examples(val_data),
        tokenizer=tok,
        num_epochs=15,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    # Method 1: Rejection Sampling / Filtered SFT
    console.print("[dim]Training with Rejection Sampling (5 iterations, 4 samples/prompt)...[/dim]")
    rs_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=128,
    )
    rs_model.load_state_dict(base_model.state_dict())
    rs_opt = torch.optim.Adam(rs_model.parameters(), lr=1e-3)

    total_accepted = 0
    for _iter in range(5):
        accepted, loss = rejection_sampling_step(
            model=rs_model,
            optimizer=rs_opt,
            tokenizer=tok,
            examples=train_data,
            num_samples_per_prompt=4,
            temperature=0.8,
            max_new_tokens=35,
        )
        total_accepted += accepted

    rs_summary = evaluate_reasoning_model(rs_model, tok, test_data, max_new_tokens=35)

    # Method 2: PPO
    console.print("[dim]Training with PPO (10 iterations)...[/dim]")
    ppo_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=128,
    )
    ppo_model.load_state_dict(base_model.state_dict())

    ref_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=128,
    )
    ref_model.load_state_dict(base_model.state_dict())

    # Build prompt-to-example lookup
    example_lookup = {ex.prompt: ex for ex in train_data}

    def reasoning_verifier(prompt: str, completion: str) -> tuple[bool, int | None]:
        ex = example_lookup.get(prompt)
        if ex is None:
            return False, None
        from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator

        report = TraceEvaluator.evaluate(completion, ex)
        return report.answer_correct, 1 if report.answer_correct else 0

    reward_fn = LanguageRewardFunction(reasoning_verifier)

    train_lm_ppo(
        model=ppo_model,
        ref_model=ref_model,
        prompts=train_prompts,
        tokenizer=tok,
        reward_fn=reward_fn,
        num_iterations=10,
        rollout_batch_size=16,
        epochs_per_rollout=2,
        minibatch_size=8,
        lr=5e-4,
        clip_eps=0.2,
        ref_kl_coef=0.05,
        seed=seed,
    )
    ppo_summary = evaluate_reasoning_model(ppo_model, tok, test_data, max_new_tokens=35)

    # Comparison Table
    table = Table(title="Policy Improvement Paradigms: Rejection Sampling vs. PPO")
    table.add_column("Paradigm", justify="left")
    table.add_column("Test Accuracy", justify="right")
    table.add_column("Trace Step Validity", justify="right")
    table.add_column("Complete Path Rate", justify="right")
    table.add_column("Tokens / Answer", justify="right")
    table.add_column("Sample Efficiency / Mechanism", justify="left")

    base_summary = evaluate_reasoning_model(base_model, tok, test_data, max_new_tokens=35)

    table.add_row(
        "0. SFT Base Checkpoint",
        f"{base_summary.answer_accuracy:.1%}",
        f"{base_summary.trace_validity:.1%}",
        f"{base_summary.complete_path_rate:.1%}",
        f"{base_summary.mean_tokens:.1f}",
        "Supervised behavior cloning baseline",
    )
    table.add_row(
        "1. Rejection Sampling (Filtered SFT)",
        f"{rs_summary.answer_accuracy:.1%}",
        f"{rs_summary.trace_validity:.1%}",
        f"{rs_summary.complete_path_rate:.1%}",
        f"{rs_summary.mean_tokens:.1f}",
        f"Filtered {total_accepted} successful rollouts; standard cross-entropy",
    )
    table.add_row(
        "2. PPO (Policy Gradient)",
        f"{ppo_summary.answer_accuracy:.1%}",
        f"{ppo_summary.trace_validity:.1%}",
        f"{ppo_summary.complete_path_rate:.1%}",
        f"{ppo_summary.mean_tokens:.1f}",
        "Online rollouts + clipped surrogate + ref KL",
    )

    console.print(table)

    console.print("\n[bold]Key Comparative Takeaways:[/bold]")
    console.print(
        "1. [yellow]Rejection Sampling Simplicity:[/yellow] Filtered SFT requires no value network, "
        "no importance sampling, and no clipping hyperparameters. It performs maximum likelihood on "
        "self-generated rollouts that pass verification."
    )
    console.print(
        "2. [green]PPO Advantage Exploitation:[/green] PPO leverages both positive and negative rewards "
        "through advantage estimation, penalizing bad reasoning trajectories rather than merely ignoring them."
    )


if __name__ == "__main__":
    run_experiment()
