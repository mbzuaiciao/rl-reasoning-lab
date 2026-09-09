"""Phase 7 Experiment 4: Selective Prediction and Abstention under Reasoning Depth.

Allows the reasoning policy to abstain when uncertainty exceeds a threshold tau.
Measures:
1. Coverage: P(model answers) = P(confidence >= tau)
2. Selective Risk: P(incorrect | model answers)
3. Selective Accuracy: 1 - Selective Risk
4. Area Under Risk-Coverage (AURC)
5. Depth-stratified behavior:
   - Does confidence decay appropriately with reasoning depth d?
   - Does selective prediction prune deeper, more error-prone reasoning instances first?
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
    compute_risk_coverage_curve,
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
            "[bold cyan]Phase 7: Selective Prediction & Abstention under Reasoning Depth[/bold cyan]\n"
            "Trading coverage for accuracy by abstaining on high-uncertainty rollouts\n"
            "and evaluating risk-coverage curves across reasoning depths.",
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

    # Generate training data on depth 1 and 2
    train_data = task.generate_dataset(size=60, depths=[1, 2], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], seed=seed + 1)

    # Balanced test set across depths 1, 2, 3, 4
    test_data = task.generate_dataset(size=80, depths=[1, 2, 3, 4], seed=seed + 2)

    console.print("[dim]Training base model...[/dim]")
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

    # Collect predictions, confidence, and metadata
    results: list[dict] = []
    for ex in test_data:
        p_ids = tok.encode(ex.prompt, add_bos=True)
        full_ids, _ = model.generate(
            p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        comp_ids = full_ids[0, len(p_ids) :].tolist()
        comp_text = tok.decode(comp_ids, skip_special=True)

        rep = evaluator.evaluate(comp_text, ex)
        unc = extract_trajectory_uncertainty(model, tok, ex.prompt, comp_text)

        results.append(
            {
                "depth": ex.depth,
                "is_correct": rep.answer_correct,
                "complete_path": rep.complete_path,
                "confidence": unc.min_token_confidence,
                "final_ans_conf": unc.final_answer_prob,
                "mean_entropy": unc.mean_token_entropy,
            }
        )

    all_confs = [r["confidence"] for r in results]
    all_correct = [r["is_correct"] for r in results]
    base_acc = float(np.mean(all_correct))

    rc_res = compute_risk_coverage_curve(all_confs, all_correct, num_thresholds=20)

    # Risk-Coverage Table
    table = Table(
        title=f"Risk-Coverage Curve (Base Accuracy = {base_acc:.1%}, AURC = {rc_res.aurc:.3f})"
    )
    table.add_column("Confidence Threshold", justify="center")
    table.add_column("Coverage P(Ans)", justify="right")
    table.add_column("Abstention Rate", justify="right")
    table.add_column("Selective Risk", justify="right")
    table.add_column("Selective Accuracy", justify="right")
    table.add_column("Accuracy Gain", justify="right")

    # Sample representative coverage milestones
    milestones = [1.0, 0.85, 0.70, 0.50, 0.30]
    for target in milestones:
        pt = min(rc_res.points, key=lambda p: abs(p.coverage - target))
        gain = pt.selective_accuracy - base_acc
        table.add_row(
            f"{pt.threshold:.2f}",
            f"{pt.coverage:.1%}",
            f"{1.0 - pt.coverage:.1%}",
            f"{pt.risk:.1%}",
            f"{pt.selective_accuracy:.1%}",
            f"[bold green]{gain:+.1%}[/bold green]" if gain > 0 else f"{gain:+.1%}",
        )
    console.print(table)

    # Depth-stratified confidence and accuracy analysis
    depth_table = Table(title="Confidence & Accuracy Across Reasoning Depths")
    depth_table.add_column("Depth d", justify="center")
    depth_table.add_column("Count", justify="right")
    depth_table.add_column("Accuracy", justify="right")
    depth_table.add_column("Mean Min-Conf", justify="right")
    depth_table.add_column("Mean Final-Ans Conf", justify="right")
    depth_table.add_column("Mean Entropy", justify="right")

    for d in sorted(list(set(r["depth"] for r in results))):
        d_items = [r for r in results if r["depth"] == d]
        d_acc = float(np.mean([r["is_correct"] for r in d_items]))
        d_min_c = float(np.mean([r["confidence"] for r in d_items]))
        d_ans_c = float(np.mean([r["final_ans_conf"] for r in d_items]))
        d_ent = float(np.mean([r["mean_entropy"] for r in d_items]))

        depth_table.add_row(
            str(d),
            str(len(d_items)),
            f"{d_acc:.1%}",
            f"{d_min_c:.3f}",
            f"{d_ans_c:.3f}",
            f"{d_ent:.3f}",
        )

    console.print(depth_table)


if __name__ == "__main__":
    run_experiment()
