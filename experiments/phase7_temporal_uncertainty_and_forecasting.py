"""Phase 7 Experiment 15: Temporal Uncertainty Trajectories and Error Forecasting.

Investigates:
1. Temporal Evolution of Policy Entropy:
   - Tracks H_t across token positions within generated derivations.
   - Does uncertainty decline as earlier premises narrow down the remaining graph path?
   - Does entropy spike immediately preceding erroneous reasoning transitions?
2. Error Forecasting:
   - Evaluates whether step-level uncertainty H_t predicts eventual trajectory failure:
     P(Eventual Failure | H_t >= tau).
   - Compares Early-Step uncertainty (Premise parsing) vs. Late-Step uncertainty (Terminal deduction).
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
from rl_reasoning_lab.evaluation.policy_uncertainty import extract_trajectory_uncertainty
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
            "[bold cyan]Phase 7: Temporal Uncertainty Trajectories & Error Forecasting[/bold cyan]\n"
            "Tracking token-level entropy dynamics across reasoning trajectories\n"
            "and evaluating early-step vs. late-step predictive power for eventual failure.",
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

    train_data = task.generate_dataset(size=60, depths=[2, 3], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[2, 3], seed=seed + 1)
    test_data = task.generate_dataset(size=50, depths=[2, 3], seed=seed + 2)

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
        val_examples=to_sft_examples(val_data),
        tokenizer=tok,
        num_epochs=12,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )
    model.eval()

    trajectories = []
    for ex in test_data:
        p_ids = tok.encode(ex.prompt, add_bos=True)
        f_ids, _ = model.generate(
            p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        comp = tok.decode(f_ids[0, len(p_ids) :].tolist(), skip_special=True)

        rep = evaluator.evaluate(comp, ex)
        unc = extract_trajectory_uncertainty(model, tok, ex.prompt, comp)

        trajectories.append(
            {
                "ex": ex,
                "comp": comp,
                "is_correct": rep.answer_correct,
                "complete_path": rep.complete_path,
                "step_entropies": unc.step_entropies,
                "tokens": unc.tokens,
            }
        )

    # 1. Temporal profile: Divide trajectory into 4 quartiles: [0-25%, 25-50%, 50-75%, 75-100%]
    q_entropies_correct: list[list[float]] = [[], [], [], []]
    q_entropies_wrong: list[list[float]] = [[], [], [], []]

    early_entropies: list[float] = []
    late_entropies: list[float] = []
    error_flags: list[bool] = []

    for traj in trajectories:
        h_vec = traj["step_entropies"]
        if not h_vec:
            continue
        is_err = not traj["is_correct"]
        error_flags.append(is_err)

        n = len(h_vec)
        # Quartiles
        for q in range(4):
            start = int(q * n / 4)
            end = max(start + 1, int((q + 1) * n / 4))
            q_mean = float(np.mean(h_vec[start:end]))
            if traj["is_correct"]:
                q_entropies_correct[q].append(q_mean)
            else:
                q_entropies_wrong[q].append(q_mean)

        early_entropies.append(float(np.mean(h_vec[: max(1, n // 3)])))
        late_entropies.append(float(np.mean(h_vec[-max(1, n // 3) :])))

    # Temporal Quartile Table
    table = Table(title="Temporal Entropy Dynamics Across Reasoning Trajectory")
    table.add_column("Trajectory Phase", justify="left")
    table.add_column("Mean H (Correct Runs)", justify="right")
    table.add_column("Mean H (Incorrect Runs)", justify="right")
    table.add_column("Entropy Spike Gap", justify="right")

    phases = [
        "Quartile 1: Prefix / Path Initialization (0-25%)",
        "Quartile 2: Intermediate Edge Traversal (25-50%)",
        "Quartile 3: Final Edge Resolution (50-75%)",
        "Quartile 4: Answer Extraction 'A: <tok>' (75-100%)",
    ]

    for q in range(4):
        mc = float(np.mean(q_entropies_correct[q])) if q_entropies_correct[q] else 0.0
        mw = float(np.mean(q_entropies_wrong[q])) if q_entropies_wrong[q] else 0.0
        table.add_row(phases[q], f"{mc:.3f}", f"{mw:.3f}", f"[bold cyan]{mw - mc:+.3f}[/bold cyan]")

    console.print(table)

    # 2. Error Forecasting: Predict eventual failure based on early vs. late entropy threshold
    tau_early = float(np.median(early_entropies))
    tau_late = float(np.median(late_entropies))

    # P(Error | H_early >= tau) vs P(Error | H_late >= tau)
    p_err_early = float(
        np.mean(
            [err for h, err in zip(early_entropies, error_flags, strict=True) if h >= tau_early]
        )
    )
    p_err_late = float(
        np.mean([err for h, err in zip(late_entropies, error_flags, strict=True) if h >= tau_late])
    )
    p_err_base = float(np.mean(error_flags))

    forecasting_table = Table(
        title="Error Forecasting Precision: Early-Stage vs. Late-Stage Uncertainty"
    )
    forecasting_table.add_column("Forecasting Signal", justify="left")
    forecasting_table.add_column("Condition", justify="center")
    forecasting_table.add_column("P(Eventual Failure | Trigger)", justify="right")
    forecasting_table.add_column("Lift vs. Prior", justify="right")

    forecasting_table.add_row("Baseline Prior", "No signal", f"{p_err_base:.1%}", "1.00x")
    forecasting_table.add_row(
        "Early-Step Entropy Spike",
        f"H_early >= {tau_early:.2f}",
        f"{p_err_early:.1%}",
        f"[bold green]{p_err_early / (p_err_base + 1e-12):.2f}x[/bold green]",
    )
    forecasting_table.add_row(
        "Late-Step Entropy Spike",
        f"H_late >= {tau_late:.2f}",
        f"{p_err_late:.1%}",
        f"{p_err_late / (p_err_base + 1e-12):.2f}x",
    )

    console.print(forecasting_table)


if __name__ == "__main__":
    run_experiment()
