"""Phase 7 Experiment 3: Post-Hoc Temperature Calibration.

Evaluates post-hoc temperature scaling:
P_T(y|x) = softmax(z / T)

Key investigations:
1. Fit T on a held-out calibration set via NLL minimization.
2. Evaluate on a separate test set.
3. Compare ECE, NLL, Brier score, and accuracy before vs. after scaling.
4. Verify empirically: Temperature scaling rescales confidence without changing argmax decision boundaries.
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
    TemperatureScaler,
    compute_calibration_metrics,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
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
            "[bold cyan]Phase 7: Post-Hoc Temperature Calibration[/bold cyan]\n"
            "Optimizing temperature T on a held-out validation set and proving that\n"
            "temperature scaling shifts calibration without changing argmax decisions.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGH"))

    # Distinct datasets: Train, Calibration, and Test
    train_data = task.generate_dataset(size=60, depths=[1, 2], seed=seed)
    calib_data = task.generate_dataset(size=40, depths=[1, 2, 3], seed=seed + 1)
    test_data = task.generate_dataset(size=60, depths=[1, 2, 3], seed=seed + 2)

    console.print("[dim]Training base reasoning model...[/dim]")
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
        val_examples=to_sft_examples(calib_data[:20]),
        tokenizer=tok,
        num_epochs=12,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )
    model.eval()

    def extract_terminal_logits_and_labels(dataset: list[ReasoningExample]):
        logits_list: list[torch.Tensor] = []
        labels_list: list[int] = []
        answers_list: list[str] = []

        for ex in dataset:
            # Condition on prompt + full target up to terminal answer token
            target_prefix = ex.trace_target[: ex.trace_target.rfind("A: ") + 3]
            full_prefix = ex.prompt + target_prefix
            input_ids = tok.encode(full_prefix, add_bos=True)

            with torch.no_grad():
                logits = model(torch.tensor([input_ids], dtype=torch.long))[0, -1, :]

            ans_tok_id = tok.char_to_id[ex.answer]
            logits_list.append(logits)
            labels_list.append(ans_tok_id)
            answers_list.append(ex.answer)

        return (
            torch.stack(logits_list, dim=0),
            torch.tensor(labels_list, dtype=torch.long),
            answers_list,
        )

    console.print("[dim]Extracting calibration and test logits...[/dim]")
    calib_logits, calib_labels, _ = extract_terminal_logits_and_labels(calib_data)
    test_logits, test_labels, test_answers = extract_terminal_logits_and_labels(test_data)

    # Fit temperature scaler
    scaler = TemperatureScaler()
    initial_temp = scaler.temperature
    learned_temp = scaler.fit(calib_logits, calib_labels, lr=0.05, max_iter=200)
    console.print(
        f"[bold green]Learned Temperature T:[/bold green] {learned_temp:.3f} (Initial: {initial_temp:.3f})"
    )

    # Evaluate on test set before and after scaling
    unscaled_test_probs = torch.softmax(test_logits, dim=-1)
    scaled_test_logits = scaler(test_logits)
    scaled_test_probs = torch.softmax(scaled_test_logits, dim=-1)

    # Argmax decision invariant check
    unscaled_preds = torch.argmax(test_logits, dim=-1)
    scaled_preds = torch.argmax(scaled_test_logits.detach(), dim=-1)
    is_decision_identical = torch.equal(unscaled_preds, scaled_preds)
    console.print(
        f"[bold cyan]Argmax Decisions Strictly Identical:[/bold cyan] {is_decision_identical}"
    )

    # Compute correctness flags and top-1 confidences
    test_labels_np = test_labels.numpy()
    unscaled_correct = (unscaled_preds.numpy() == test_labels_np).tolist()
    scaled_correct = (scaled_preds.numpy() == test_labels_np).tolist()

    unscaled_confs = (
        unscaled_test_probs.gather(1, test_labels.unsqueeze(1)).squeeze(1).numpy().tolist()
    )
    scaled_confs = (
        scaled_test_probs.detach().gather(1, test_labels.unsqueeze(1)).squeeze(1).numpy().tolist()
    )

    unscaled_cal = compute_calibration_metrics(unscaled_confs, unscaled_correct, num_bins=5)
    scaled_cal = compute_calibration_metrics(scaled_confs, scaled_correct, num_bins=5)

    table = Table(title="Test Set Performance: Before vs. After Temperature Scaling")
    table.add_column("Evaluation Metric", justify="left")
    table.add_column("Before Scaling (T=1.000)", justify="right")
    table.add_column(f"After Scaling (T={learned_temp:.3f})", justify="right")
    table.add_column("Delta", justify="right")

    acc_before = float(np.mean(unscaled_correct))
    acc_after = float(np.mean(scaled_correct))

    table.add_row(
        "Task Accuracy", f"{acc_before:.1%}", f"{acc_after:.1%}", f"{acc_after - acc_before:+.1%}"
    )
    table.add_row(
        "Expected Calibration Error (ECE)",
        f"{unscaled_cal.ece:.3f}",
        f"{scaled_cal.ece:.3f}",
        f"[bold green]{scaled_cal.ece - unscaled_cal.ece:+.3f}[/bold green]",
    )
    table.add_row(
        "Negative Log-Likelihood (NLL)",
        f"{unscaled_cal.nll:.3f}",
        f"{scaled_cal.nll:.3f}",
        f"[bold green]{scaled_cal.nll - unscaled_cal.nll:+.3f}[/bold green]",
    )
    table.add_row(
        "Brier Score",
        f"{unscaled_cal.brier_score:.3f}",
        f"{scaled_cal.brier_score:.3f}",
        f"[bold green]{scaled_cal.brier_score - unscaled_cal.brier_score:+.3f}[/bold green]",
    )
    table.add_row(
        "Mean Confidence",
        f"{float(np.mean(unscaled_confs)):.3f}",
        f"{float(np.mean(scaled_confs)):.3f}",
        f"{float(np.mean(scaled_confs) - np.mean(unscaled_confs)):+.3f}",
    )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
