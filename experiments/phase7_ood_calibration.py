"""Phase 7 Experiment 5: Out-of-Distribution (OOD) Calibration and Overconfidence.

Investigates calibration under distribution shift across reasoning depth:
1. Train policy strictly on depths d in {1, 2, 3}.
2. Evaluate on In-Distribution (d in {1, 2, 3}) vs. Out-of-Distribution (d in {4, 5, 6}).
3. Measure:
   - Accuracy drop
   - Confidence change
   - ECE degradation
4. Formally test for the dangerous failure regime:
   Accuracy drops sharply, but confidence remains artificially high (OOD Overconfidence).
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
            "[bold cyan]Phase 7: OOD Calibration and Depth Generalization[/bold cyan]\n"
            "Testing whether policy confidence degrades proportionally when evaluated on\n"
            "unseen reasoning depths, or exhibits dangerous OOD overconfidence.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    # 12-node pool to support depths up to 6
    task = GraphTraversalTask(node_pool=list("ABCDEFGHIJKL"))
    evaluator = TraceEvaluator()

    # Train exclusively on depths 1, 2, 3
    train_data = task.generate_dataset(size=70, depths=[1, 2, 3], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2, 3], seed=seed + 1)

    # IID test set (depths 1, 2, 3) vs OOD test set (depths 4, 5, 6)
    iid_test = task.generate_dataset(size=45, depths=[1, 2, 3], seed=seed + 2)
    ood_test = task.generate_dataset(size=45, depths=[4, 5, 6], seed=seed + 3)

    console.print("[dim]Training base policy on d <= 3...[/dim]")
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

    def evaluate_split(dataset: list[ReasoningExample]):
        corrects: list[bool] = []
        confs: list[float] = []
        min_confs: list[float] = []
        entropies: list[float] = []

        for ex in dataset:
            p_ids = tok.encode(ex.prompt, add_bos=True)
            full_ids, _ = model.generate(
                p_ids, max_new_tokens=45, greedy=True, eos_token_id=tok.eos_token_id
            )
            comp_ids = full_ids[0, len(p_ids) :].tolist()
            comp_text = tok.decode(comp_ids, skip_special=True)

            rep = evaluator.evaluate(comp_text, ex)
            unc = extract_trajectory_uncertainty(model, tok, ex.prompt, comp_text)

            corrects.append(rep.answer_correct)
            confs.append(unc.final_answer_prob)
            min_confs.append(unc.min_token_confidence)
            entropies.append(unc.mean_token_entropy)

        acc = float(np.mean(corrects))
        cal = compute_calibration_metrics(confs, corrects, num_bins=5)
        return {
            "accuracy": acc,
            "mean_conf": float(np.mean(confs)),
            "mean_min_conf": float(np.mean(min_confs)),
            "mean_entropy": float(np.mean(entropies)),
            "ece": cal.ece,
            "brier": cal.brier_score,
            "confidences": confs,
            "corrects": corrects,
        }

    console.print("[dim]Evaluating IID (d <= 3) vs OOD (d in {4, 5, 6})...[/dim]")
    iid_res = evaluate_split(iid_test)
    ood_res = evaluate_split(ood_test)

    table = Table(title="In-Distribution vs. Out-of-Distribution Calibration")
    table.add_column("Evaluation Metric", justify="left")
    table.add_column("IID (d in {1, 2, 3})", justify="right")
    table.add_column("OOD (d in {4, 5, 6})", justify="right")
    table.add_column("Shift Delta (OOD - IID)", justify="right")

    table.add_row(
        "Task Accuracy",
        f"{iid_res['accuracy']:.1%}",
        f"{ood_res['accuracy']:.1%}",
        f"[bold red]{ood_res['accuracy'] - iid_res['accuracy']:+.1%}[/bold red]",
    )
    table.add_row(
        "Final-Answer Confidence",
        f"{iid_res['mean_conf']:.3f}",
        f"{ood_res['mean_conf']:.3f}",
        f"{ood_res['mean_conf'] - iid_res['mean_conf']:+.3f}",
    )
    table.add_row(
        "Minimum Token Confidence",
        f"{iid_res['mean_min_conf']:.3f}",
        f"{ood_res['mean_min_conf']:.3f}",
        f"{ood_res['mean_min_conf'] - iid_res['mean_min_conf']:+.3f}",
    )
    table.add_row(
        "Mean Token Entropy",
        f"{iid_res['mean_entropy']:.3f}",
        f"{ood_res['mean_entropy']:.3f}",
        f"{ood_res['mean_entropy'] - iid_res['mean_entropy']:+.3f}",
    )
    table.add_row(
        "Expected Calibration Error (ECE)",
        f"{iid_res['ece']:.3f}",
        f"{ood_res['ece']:.3f}",
        f"[bold red]{ood_res['ece'] - iid_res['ece']:+.3f}[/bold red]",
    )
    table.add_row(
        "Brier Score",
        f"{iid_res['brier']:.3f}",
        f"{ood_res['brier']:.3f}",
        f"[bold red]{ood_res['brier'] - iid_res['brier']:+.3f}[/bold red]",
    )

    console.print(table)

    # Formal check for OOD Overconfidence
    is_overconfident = (ood_res["accuracy"] < iid_res["accuracy"]) and (
        ood_res["mean_conf"] >= iid_res["mean_conf"] * 0.85
    )
    status_str = (
        "[bold red]CONFIRMED: OOD Overconfidence Detected[/bold red]\n"
        f"Accuracy fell by {iid_res['accuracy'] - ood_res['accuracy']:.1%}, but confidence dropped by only "
        f"{iid_res['mean_conf'] - ood_res['mean_conf']:.3f}. The model remains confident while frequently wrong."
        if is_overconfident
        else "[bold green]Calibrated Decline: Model confidence falls proportionately with task difficulty.[/bold green]"
    )
    console.print(Panel(status_str, title="Empirical OOD Diagnostic"))


if __name__ == "__main__":
    run_experiment()
