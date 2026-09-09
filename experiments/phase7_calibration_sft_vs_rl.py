"""Phase 7 Experiment 2: Calibration Before vs. After Reinforcement Learning.

Compares:
1. SFT base policy
2. SFT + Outcome PPO (sparse terminal reward)
3. SFT + Process PPO (dense normalized step reward)

Measures:
- Task accuracy
- Mean final-answer confidence
- Expected Calibration Error (ECE)
- Brier score
- Policy token entropy

Directly addresses:
Can RL improve accuracy while worsening calibration?
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
from rl_reasoning_lab.evaluation.policy_uncertainty import (
    compute_calibration_metrics,
    extract_trajectory_uncertainty,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.rewards.reasoning_rewards import ReasoningRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


def to_sft_examples(examples: list[ReasoningExample]) -> list[Example]:
    return [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in examples
    ]


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 7: Calibration Before vs. After Reinforcement Learning[/bold cyan]\n"
            "Investigating whether RL optimization drives confidence sharpening and\n"
            "calibration collapse even when accuracy improves.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGH"))
    evaluator = TraceEvaluator()

    train_data = task.generate_dataset(size=50, depths=[1, 2], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], seed=seed + 1)
    test_data = task.generate_dataset(size=50, depths=[1, 2, 3], seed=seed + 2)
    train_prompts = [ex.prompt for ex in train_data]
    example_lookup = {ex.prompt: ex for ex in train_data}

    # 1. Base SFT model
    console.print("[dim]Training base SFT model...[/dim]")
    sft_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    train_sft(
        model=sft_model,
        train_examples=to_sft_examples(train_data),
        val_examples=to_sft_examples(val_data),
        tokenizer=tok,
        num_epochs=12,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    # Reference model for PPO
    ref_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    ref_model.load_state_dict(sft_model.state_dict())

    # 2. Outcome PPO
    console.print("[dim]Training Outcome PPO policy...[/dim]")
    outcome_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    outcome_model.load_state_dict(sft_model.state_dict())

    r_outcome = ReasoningRewardFunction(reward_mode="outcome")

    def make_verifier(r_fn):
        def verifier(prompt: str, completion: str) -> tuple[bool, int | None]:
            ex = example_lookup.get(prompt)
            if ex is None:
                return False, None
            outcome = r_fn.evaluate(completion, ex)
            return outcome.total_reward >= 0.8, int(outcome.total_reward * 10)

        return verifier

    train_lm_ppo(
        model=outcome_model,
        ref_model=ref_model,
        prompts=train_prompts,
        tokenizer=tok,
        reward_fn=LanguageRewardFunction(make_verifier(r_outcome)),
        num_iterations=6,
        rollout_batch_size=16,
        epochs_per_rollout=2,
        minibatch_size=8,
        lr=5e-4,
        clip_eps=0.2,
        ref_kl_coef=0.02,
        seed=seed,
    )

    # 3. Process PPO
    console.print("[dim]Training Process PPO policy...[/dim]")
    process_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    process_model.load_state_dict(sft_model.state_dict())
    r_process = ReasoningRewardFunction(reward_mode="process")

    train_lm_ppo(
        model=process_model,
        ref_model=ref_model,
        prompts=train_prompts,
        tokenizer=tok,
        reward_fn=LanguageRewardFunction(make_verifier(r_process)),
        num_iterations=6,
        rollout_batch_size=16,
        epochs_per_rollout=2,
        minibatch_size=8,
        lr=5e-4,
        clip_eps=0.2,
        ref_kl_coef=0.02,
        seed=seed,
    )

    # Evaluation loop
    models = [
        ("SFT Base", sft_model),
        ("SFT + Outcome PPO", outcome_model),
        ("SFT + Process PPO", process_model),
    ]

    table = Table(title="Policy Calibration & Sharpness: SFT vs. RL")
    table.add_column("Policy Regime", justify="left")
    table.add_column("Accuracy", justify="right")
    table.add_column("Mean Conf (Final Ans)", justify="right")
    table.add_column("Mean Token Entropy", justify="right")
    table.add_column("ECE (5 bins)", justify="right")
    table.add_column("Brier Score", justify="right")

    for label, m in models:
        m.eval()
        correct_flags: list[bool] = []
        confidences: list[float] = []
        entropies: list[float] = []

        for ex in test_data:
            p_ids = tok.encode(ex.prompt, add_bos=True)
            full_ids, _ = m.generate(
                p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
            )
            comp_ids = full_ids[0, len(p_ids) :].tolist()
            comp_text = tok.decode(comp_ids, skip_special=True)

            rep = evaluator.evaluate(comp_text, ex)
            unc = extract_trajectory_uncertainty(m, tok, ex.prompt, comp_text)

            correct_flags.append(rep.answer_correct)
            confidences.append(unc.final_answer_prob)
            entropies.append(unc.mean_token_entropy)

        acc = float(np.mean(correct_flags))
        mean_c = float(np.mean(confidences))
        mean_h = float(np.mean(entropies))
        cal = compute_calibration_metrics(confidences, correct_flags, num_bins=5)

        table.add_row(
            label,
            f"{acc:.1%}",
            f"{mean_c:.3f}",
            f"{mean_h:.3f}",
            f"[bold red]{cal.ece:.3f}[/bold red]" if cal.ece > 0.3 else f"{cal.ece:.3f}",
            f"{cal.brier_score:.3f}",
        )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
