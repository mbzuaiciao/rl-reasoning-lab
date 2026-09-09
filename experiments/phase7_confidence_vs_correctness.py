"""Phase 7 Experiment 1: Correctness vs. Confidence and Internal Policy Statistics.

Evaluates a trained reasoning model over a controlled test set:
1. Extracts internal policy statistics:
   - Final-answer confidence
   - Mean token entropy
   - Minimum token confidence
   - Length-normalized sequence log probability
2. Computes reliability curves: P(correct | confidence bin)
3. Evaluates Expected Calibration Error (ECE) and Brier score for each metric.
4. Identifies which internal statistic best separates correct from incorrect reasoning.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.environments.language_tasks import Example
from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.evaluation.policy_uncertainty import (
    compute_calibration_metrics,
    extract_trajectory_uncertainty,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
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
            "[bold cyan]Phase 7: Correctness vs. Confidence & Internal Policy Statistics[/bold cyan]\n"
            "Evaluating which policy-side uncertainty metric best separates valid reasoning\n"
            "from errors across calibration curves and discrimination metrics.",
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

    # Training and test data
    train_data = task.generate_dataset(size=60, depths=[1, 2, 3], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2, 3], seed=seed + 1)
    test_data = task.generate_dataset(size=60, depths=[1, 2, 3, 4], seed=seed + 2)

    console.print("[dim]Training base reasoning model with SFT...[/dim]")
    model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    train_sft(
        model=model,
        train_examples=to_sft_examples(train_data),
        val_examples=to_sft_examples(val_data),
        tokenizer=tok,
        num_epochs=12,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )
    model.eval()

    # Collect rollouts and uncertainty stats
    correct_labels: list[bool] = []
    final_ans_confs: list[float] = []
    mean_entropies: list[float] = []
    min_confs: list[float] = []
    norm_seq_logps: list[float] = []

    for ex in test_data:
        prompt_ids = tok.encode(ex.prompt, add_bos=True)
        full_ids, _ = model.generate(
            prompt_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        comp_ids = full_ids[0, len(prompt_ids) :].tolist()
        comp_text = tok.decode(comp_ids, skip_special=True)

        res = evaluator.evaluate(comp_text, ex)
        is_correct = res.answer_correct

        unc = extract_trajectory_uncertainty(model, tok, ex.prompt, comp_text)

        correct_labels.append(is_correct)
        final_ans_confs.append(unc.final_answer_prob)
        # Invert entropy so higher = more confident in [0, 1] range: exp(-mean_entropy)
        mean_entropies.append(float(np.exp(-unc.mean_token_entropy)))
        min_confs.append(unc.min_token_confidence)
        # Map length-normalized logp to [0, 1] pseudo-probability
        norm_seq_logps.append(float(np.exp(unc.length_normalized_log_prob)))

    overall_acc = float(np.mean(correct_labels))
    console.print(
        f"[bold green]Test Accuracy:[/bold green] {overall_acc:.1%} ({sum(correct_labels)}/{len(correct_labels)})"
    )

    # Metrics comparison table
    metrics_suite = [
        ("Final-Answer Confidence", final_ans_confs),
        ("Inverse Mean Entropy exp(-H)", mean_entropies),
        ("Minimum Token Confidence", min_confs),
        ("Norm. Sequence Prob exp(logP/T)", norm_seq_logps),
    ]

    table = Table(title="Policy Uncertainty Metrics vs. Correctness")
    table.add_column("Statistic / Confidence Measure", justify="left")
    table.add_column("Mean (Correct)", justify="right")
    table.add_column("Mean (Incorrect)", justify="right")
    table.add_column("Separation Gap", justify="right")
    table.add_column("ECE (5 bins)", justify="right")
    table.add_column("Brier Score", justify="right")

    for name, values in metrics_suite:
        c_correct = [v for v, label in zip(values, correct_labels, strict=True) if label]
        c_incorrect = [v for v, label in zip(values, correct_labels, strict=True) if not label]

        mean_c = float(np.mean(c_correct)) if c_correct else 0.0
        mean_inc = float(np.mean(c_incorrect)) if c_incorrect else 0.0
        gap = mean_c - mean_inc

        cal = compute_calibration_metrics(values, correct_labels, num_bins=5)
        table.add_row(
            name,
            f"{mean_c:.3f}",
            f"{mean_inc:.3f}",
            f"[bold cyan]{gap:+.3f}[/bold cyan]",
            f"{cal.ece:.3f}",
            f"{cal.brier_score:.3f}",
        )

    console.print(table)

    # Reliability diagram for the top discriminator
    best_name, best_vals = max(
        metrics_suite,
        key=lambda m: (
            float(np.mean([v for v, label in zip(m[1], correct_labels, strict=True) if label]))
            - float(
                np.mean([v for v, label in zip(m[1], correct_labels, strict=True) if not label])
            )
        ),
    )
    cal_best = compute_calibration_metrics(best_vals, correct_labels, num_bins=5)

    rel_table = Table(title=f"Reliability Diagram: {best_name}")
    rel_table.add_column("Bin Range", justify="center")
    rel_table.add_column("Count", justify="right")
    rel_table.add_column("Mean Confidence", justify="right")
    rel_table.add_column("Empirical Accuracy", justify="right")
    rel_table.add_column("Calibration Gap |Acc - Conf|", justify="right")

    bin_boundaries = np.linspace(0.0, 1.0, 6)
    for i in range(5):
        b_range = f"[{bin_boundaries[i]:.1f}, {bin_boundaries[i + 1]:.1f})"
        count = cal_best.bin_counts[i]
        b_conf = cal_best.bin_confidences[i]
        b_acc = cal_best.bin_accuracies[i]
        b_gap = abs(b_acc - b_conf) if count > 0 else 0.0
        rel_table.add_row(b_range, str(count), f"{b_conf:.3f}", f"{b_acc:.3f}", f"{b_gap:.3f}")

    console.print(rel_table)


if __name__ == "__main__":
    run_experiment()
