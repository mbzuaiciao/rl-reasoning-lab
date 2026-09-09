"""Phase 7 Experiment 11: Error Decomposition & Multi-Target Calibration.

Decomposes reasoning failures into:
1. Wrong answer, invalid path
2. Wrong answer, valid partial path
3. Correct answer, invalid path (hallucinated reasoning / lucky shortcut)
4. Syntactic formatting failure
5. OOD depth failure

Measures calibration conditionally across three distinct targets:
- P(Answer Correct | c)
- P(Complete Geodesic Path | c)
- P(Step Validity >= 0.8 | c)

Demonstrates: A model can appear calibrated for terminal answers while being
wildly overconfident about the correctness of its latent reasoning steps.
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
            "[bold cyan]Phase 7: Error Decomposition & Multi-Target Calibration[/bold cyan]\n"
            "Testing whether policy confidence calibrates to reasoning path validity\n"
            "or merely to superficial final answer correctness.",
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

    train_data = task.generate_dataset(size=60, depths=[1, 2], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], seed=seed + 1)
    test_data = task.generate_dataset(size=70, depths=[1, 2, 3, 4], seed=seed + 2)

    console.print("[dim]Training base policy...[/dim]")
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

    ans_correct_flags: list[bool] = []
    complete_path_flags: list[bool] = []
    step_valid_flags: list[bool] = []
    confidences: list[float] = []

    error_buckets = {
        "Wrong Answer + Invalid Path": 0,
        "Wrong Answer + Partial Valid": 0,
        "Correct Answer + Invalid Path (Shortcut)": 0,
        "Syntax Breakdown": 0,
        "Clean Success (Both Valid)": 0,
    }

    for ex in test_data:
        p_ids = tok.encode(ex.prompt, add_bos=True)
        full_ids, _ = model.generate(
            p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        comp_ids = full_ids[0, len(p_ids) :].tolist()
        comp_text = tok.decode(comp_ids, skip_special=True)

        rep = evaluator.evaluate(comp_text, ex)
        unc = extract_trajectory_uncertainty(model, tok, ex.prompt, comp_text)

        c = unc.final_answer_prob
        confidences.append(c)

        ans_correct_flags.append(rep.answer_correct)
        complete_path_flags.append(rep.complete_path)
        step_valid_flags.append(rep.step_validity >= 0.8)

        # Categorize into mutually exclusive buckets
        if not rep.syntax_valid:
            error_buckets["Syntax Breakdown"] += 1
        elif rep.answer_correct and rep.complete_path:
            error_buckets["Clean Success (Both Valid)"] += 1
        elif rep.answer_correct and not rep.complete_path:
            error_buckets["Correct Answer + Invalid Path (Shortcut)"] += 1
        elif not rep.answer_correct and rep.step_validity > 0.0:
            error_buckets["Wrong Answer + Partial Valid"] += 1
        else:
            error_buckets["Wrong Answer + Invalid Path"] += 1

    total = len(test_data)
    cat_table = Table(title="Taxonomy of Reasoning Outcomes")
    cat_table.add_column("Outcome Category", justify="left")
    cat_table.add_column("Frequency", justify="right")
    cat_table.add_column("Proportion", justify="right")

    for cat_name, cnt in error_buckets.items():
        cat_table.add_row(cat_name, str(cnt), f"{cnt / total:.1%}")
    console.print(cat_table)

    # Multi-Target Calibration Comparison
    cal_ans = compute_calibration_metrics(confidences, ans_correct_flags, num_bins=5)
    cal_path = compute_calibration_metrics(confidences, complete_path_flags, num_bins=5)
    cal_step = compute_calibration_metrics(confidences, step_valid_flags, num_bins=5)

    cal_table = Table(title="Calibration Conditional on Correctness Target")
    cal_table.add_column("Target Ground-Truth Attribute", justify="left")
    cal_table.add_column("Target Base Rate", justify="right")
    cal_table.add_column("Mean Confidence", justify="right")
    cal_table.add_column("ECE (5 bins)", justify="right")
    cal_table.add_column("Brier Score", justify="right")

    cal_table.add_row(
        "Final Answer Correctness",
        f"{float(np.mean(ans_correct_flags)):.1%}",
        f"{float(np.mean(confidences)):.3f}",
        f"{cal_ans.ece:.3f}",
        f"{cal_ans.brier_score:.3f}",
    )
    cal_table.add_row(
        "Complete Geodesic Path Validity",
        f"{float(np.mean(complete_path_flags)):.1%}",
        f"{float(np.mean(confidences)):.3f}",
        f"[bold red]{cal_path.ece:.3f}[/bold red]",
        f"{cal_path.brier_score:.3f}",
    )
    cal_table.add_row(
        "Step-Level Rule Validity (>= 80%)",
        f"{float(np.mean(step_valid_flags)):.1%}",
        f"{float(np.mean(confidences)):.3f}",
        f"{cal_step.ece:.3f}",
        f"{cal_step.brier_score:.3f}",
    )

    console.print(cal_table)


if __name__ == "__main__":
    run_experiment()
