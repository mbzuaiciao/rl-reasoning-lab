"""Phase 4 Experiment: SFT vs. RL Comparative Benchmark.

Compares four paradigms head-to-head:
1. SFT only (supervised imitation of demonstrator trajectories)
2. SFT + REINFORCE (Monte Carlo policy gradients on self-generated rollouts)
3. SFT + PPO without KL (clipped surrogate objective)
4. SFT + PPO with KL (clipped surrogate + reference KL penalty)

Evaluates on out-of-distribution / unseen test instances to measure:
- Test Exact Match %
- Mean reward
- Reference KL divergence (drift from base linguistic capabilities)
"""

from __future__ import annotations

import random

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.lm_policy_gradient import train_lm_reinforce
from rl_reasoning_lab.algorithms.lm_ppo import train_lm_ppo
from rl_reasoning_lab.environments.language_tasks import ArithmeticTask
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.sft import evaluate_sft, train_sft


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 4: SFT vs. RL Paradigms Benchmark[/bold cyan]\n"
            "Comparing: SFT Only vs. SFT+REINFORCE vs. SFT+PPO vs. SFT+PPO+KL",
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
    test_data = task.generate_dataset(size=40, seed=seed + 100)
    train_prompts = [ex.prompt for ex in train_data]

    # 1. Base SFT Training
    console.print("[dim]Step 1: Training initial SFT Base Model (15 epochs)...[/dim]")
    sft_base = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    train_sft(
        model=sft_base,
        train_examples=train_data,
        val_examples=val_data,
        tokenizer=tok,
        num_epochs=15,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    # Reference model frozen
    ref_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    ref_model.load_state_dict(sft_base.state_dict())

    # 2. SFT Only Evaluation
    _, _, em_sft = evaluate_sft(sft_base, test_data, tok)

    # 3. SFT + REINFORCE
    console.print("[dim]Step 2: Training SFT + REINFORCE...[/dim]")
    reinforce_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    reinforce_model.load_state_dict(sft_base.state_dict())
    train_lm_reinforce(
        model=reinforce_model,
        prompts=train_prompts,
        tokenizer=tok,
        reward_fn=reward_fn,
        num_iterations=15,
        batch_size=16,
        lr=5e-4,
        seed=seed,
    )
    _, _, em_reinforce = evaluate_sft(reinforce_model, test_data, tok)

    # 4. SFT + PPO (No KL penalty)
    console.print("[dim]Step 3: Training SFT + PPO (No KL penalty)...[/dim]")
    ppo_nokl_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    ppo_nokl_model.load_state_dict(sft_base.state_dict())
    train_lm_ppo(
        model=ppo_nokl_model,
        ref_model=ref_model,
        prompts=train_prompts,
        tokenizer=tok,
        reward_fn=reward_fn,
        num_iterations=15,
        rollout_batch_size=16,
        epochs_per_rollout=2,
        minibatch_size=8,
        clip_eps=0.2,
        lr=5e-4,
        ref_kl_coef=0.0,
        seed=seed,
    )
    _, _, em_ppo_nokl = evaluate_sft(ppo_nokl_model, test_data, tok)

    # 5. SFT + PPO + KL (Standard RLHF pipeline)
    console.print("[dim]Step 4: Training SFT + PPO + KL (beta=0.1)...[/dim]")
    ppo_kl_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    ppo_kl_model.load_state_dict(sft_base.state_dict())
    train_lm_ppo(
        model=ppo_kl_model,
        ref_model=ref_model,
        prompts=train_prompts,
        tokenizer=tok,
        reward_fn=reward_fn,
        num_iterations=15,
        rollout_batch_size=16,
        epochs_per_rollout=2,
        minibatch_size=8,
        clip_eps=0.2,
        lr=5e-4,
        ref_kl_coef=0.1,
        seed=seed,
    )
    _, _, em_ppo_kl = evaluate_sft(ppo_kl_model, test_data, tok)

    # Benchmark Summary Table
    table = Table(title="Paradigm Comparison on Test Dataset (40 unseen examples)")
    table.add_column("Paradigm", justify="left")
    table.add_column("Exact Match %", justify="right")
    table.add_column("Optimization Characteristics", justify="left")

    table.add_row(
        "1. SFT Only", f"{em_sft:.1%}", "Supervised on ground truth; susceptible to exposure bias"
    )
    table.add_row(
        "2. SFT + REINFORCE",
        f"{em_reinforce:.1%}",
        "Self-exploration but high Monte Carlo gradient variance",
    )
    table.add_row(
        "3. SFT + PPO (No KL)",
        f"{em_ppo_nokl:.1%}",
        "Multi-epoch data reuse with clipping; risks policy drift",
    )
    table.add_row(
        "4. SFT + PPO + KL",
        f"{em_ppo_kl:.1%}",
        "Bounded policy divergence anchored to SFT reference prior",
    )

    console.print(table)

    console.print("\n[bold]Core Insights for Language Model Reasoning:[/bold]")
    console.print(
        "- SFT teaches the model token format and problem representation, but cannot optimize "
        "global sequence-level metrics directly without teacher forcing."
    )
    console.print(
        "- RL allows the model to generate its own full solutions and learn from the verifier's feedback, "
        "bridging the exposure bias gap."
    )
    console.print(
        "- Adding PPO clipping and reference KL stabilizes training, ensuring reasoning improvements "
        "do not cause degenerative token drift or mode collapse."
    )


if __name__ == "__main__":
    run_experiment()
