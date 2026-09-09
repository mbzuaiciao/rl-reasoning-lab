"""Phase 7 Experiment 8: Policy Ensembles vs. Repeated Sampling vs. MC Dropout.

Compares three paradigms of predictive dispersion:
1. Deep Ensemble: M=3 independently trained policies with diverse initializations (Epistemic Diversity).
2. Repeated Sampling: Single trained policy with M=3 stochastic decoding rollouts (Sampling / Temperature Variability).
3. Monte Carlo Dropout: Single trained policy with dropout enabled during evaluation (Weight Perturbation Heuristic).

Measures:
- Error detection via disagreement: Correlation between disagreement and actual errors.
- Epistemic variance across seeds.
- Can deep ensembles catch shared failure modes where a single model's repeated sampling is blind?
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
from rl_reasoning_lab.evaluation.policy_uncertainty import compute_ensemble_uncertainty
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
            "[bold cyan]Phase 7: Policy Ensemble vs. Repeated Sampling vs. MC Dropout[/bold cyan]\n"
            "Testing whether parameter diversity (deep ensemble) identifies errors that\n"
            "ordinary stochastic sampling or MC dropout heuristics miss.",
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
    test_data = task.generate_dataset(size=40, depths=[1, 2, 3], seed=seed + 2)

    # 1. Train Deep Ensemble (3 independent models with different seeds)
    ensemble_seeds = [42, 101, 202]
    ensemble_models: list[TinyTransformer] = []

    for s in ensemble_seeds:
        console.print(f"[dim]Training ensemble member with seed {s}...[/dim]")
        m = TinyTransformer(
            vocab_size=tok.vocab_size,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            max_seq_len=128,
            dropout=0.2,  # dropout present for MC dropout comparisons
        )
        train_sft(
            model=m,
            train_examples=to_sft_examples(train_data),
            val_examples=to_sft_examples(val_data),
            tokenizer=tok,
            num_epochs=10,
            batch_size=16,
            lr=3e-3,
            seed=s,
        )
        m.eval()
        ensemble_models.append(m)

    primary_model = ensemble_models[0]

    # Evaluation loop
    # For each prompt:
    # 1. Deep Ensemble: greedy predictions from 3 independent models
    # 2. Repeated Sampling: 3 stochastic samples from primary_model (T=0.8)
    # 3. MC Dropout: 3 greedy passes from primary_model with enable_dropout=True

    de_disagreements: list[float] = []
    rs_disagreements: list[float] = []
    mc_disagreements: list[float] = []

    error_flags: list[bool] = []

    for ex in test_data:
        p_ids = tok.encode(ex.prompt, add_bos=True)

        # Baseline error flag (primary greedy)
        g_ids, _ = primary_model.generate(
            p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        g_text = tok.decode(g_ids[0, len(p_ids) :].tolist(), skip_special=True)
        rep = evaluator.evaluate(g_text, ex)
        is_error = not rep.answer_correct
        error_flags.append(is_error)

        # 1. Deep Ensemble
        de_answers: list[str] = []
        for m in ensemble_models:
            out_ids, _ = m.generate(
                p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
            )
            ans = evaluator.parse_completion(
                tok.decode(out_ids[0, len(p_ids) :].tolist(), skip_special=True)
            )[0]
            de_answers.append(ans or "UNKNOWN")
        de_res = compute_ensemble_uncertainty(de_answers)
        de_disagreements.append(de_res.disagreement_rate)

        # 2. Repeated Sampling (primary model, T=0.8)
        rs_answers: list[str] = []
        for _ in range(3):
            out_ids, _ = primary_model.generate(
                p_ids,
                max_new_tokens=35,
                temperature=0.8,
                greedy=False,
                eos_token_id=tok.eos_token_id,
            )
            ans = evaluator.parse_completion(
                tok.decode(out_ids[0, len(p_ids) :].tolist(), skip_special=True)
            )[0]
            rs_answers.append(ans or "UNKNOWN")
        rs_res = compute_ensemble_uncertainty(rs_answers)
        rs_disagreements.append(rs_res.disagreement_rate)

        # 3. MC Dropout (primary model, enable_dropout=True)
        mc_answers: list[str] = []
        for _ in range(3):
            out_ids, _ = primary_model.generate(
                p_ids,
                max_new_tokens=35,
                greedy=True,
                enable_dropout=True,
                eos_token_id=tok.eos_token_id,
            )
            ans = evaluator.parse_completion(
                tok.decode(out_ids[0, len(p_ids) :].tolist(), skip_special=True)
            )[0]
            mc_answers.append(ans or "UNKNOWN")
        mc_res = compute_ensemble_uncertainty(mc_answers)
        mc_disagreements.append(mc_res.disagreement_rate)

    # Calculate error discrimination (disagreement on errors vs disagreement on correct)
    methods = [
        ("Deep Ensemble (3 Models)", de_disagreements),
        ("Repeated Sampling (T=0.8)", rs_disagreements),
        ("Monte Carlo Dropout (p=0.2)", mc_disagreements),
    ]

    table = Table(
        title="Error Detection via Disagreement: Deep Ensemble vs. Sampling vs. MC Dropout"
    )
    table.add_column("Uncertainty Methodology", justify="left")
    table.add_column("Mean Disagreement", justify="right")
    table.add_column("Disagreement (On Errors)", justify="right")
    table.add_column("Disagreement (On Correct)", justify="right")
    table.add_column("Error Separation Gap", justify="right")

    for name, values in methods:
        err_vals = [v for v, err in zip(values, error_flags, strict=True) if err]
        cor_vals = [v for v, err in zip(values, error_flags, strict=True) if not err]

        mean_err = float(np.mean(err_vals)) if err_vals else 0.0
        mean_cor = float(np.mean(cor_vals)) if cor_vals else 0.0
        gap = mean_err - mean_cor

        table.add_row(
            name,
            f"{float(np.mean(values)):.1%}",
            f"{mean_err:.1%}",
            f"{mean_cor:.1%}",
            f"[bold cyan]{gap:+.1%}[/bold cyan]",
        )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
