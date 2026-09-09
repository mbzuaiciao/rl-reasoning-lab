"""Phase 7 Experiment 10: Policy Uncertainty vs. Verifier Uncertainty.

Central 2x2 grid experiment:
Evaluates rollouts categorized by Policy Uncertainty (Low/High) x Verifier Uncertainty (Low/High).

Four Quadrants:
1. Low Policy Unc, Low Verifier Unc: Both confident.
2. High Policy Unc, Low Verifier Unc: Policy unsure, verifier confident.
3. Low Policy Unc, High Verifier Unc: Policy confident, verifier unsure.
4. High Policy Unc, High Verifier Unc: Both unsure.

Measures:
- Ground-truth task error rate per quadrant
- Distribution of error types (wrong answer vs. invalid reasoning path vs. hallucination)
- Demonstrates: Policy uncertainty detects generator hesitation; verifier uncertainty detects structural/invariance flaws.
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
from rl_reasoning_lab.verifiers.composite_verifiers import VerifierEnsemble
from rl_reasoning_lab.verifiers.proxy_verifiers import (
    FormatVerifier,
    KeywordVerifier,
    SubstringAnswerVerifier,
)
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
            "[bold cyan]Phase 7: Policy Uncertainty vs. Verifier Uncertainty (2x2 Grid)[/bold cyan]\n"
            "Dissecting the four quadrants of joint uncertainty to determine which signal\n"
            "is most informative about specific failure modes.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGH"), shuffle_rules=True)
    evaluator = TraceEvaluator()

    # Verifier ensemble
    verifier_ensemble = VerifierEnsemble(
        verifiers=[
            SubstringAnswerVerifier(),
            KeywordVerifier(),
            FormatVerifier(),
        ],
        aggregation="majority_vote",
    )

    train_data = task.generate_dataset(size=50, depths=[1, 2], num_distractors=1, seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], num_distractors=1, seed=seed + 1)
    test_data = task.generate_dataset(size=60, depths=[1, 2, 3], num_distractors=1, seed=seed + 2)

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

    rollouts = []
    for ex in test_data:
        p_ids = tok.encode(ex.prompt, add_bos=True)
        full_ids, _ = model.generate(
            p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        comp_ids = full_ids[0, len(p_ids) :].tolist()
        comp_text = tok.decode(comp_ids, skip_special=True)

        rep = evaluator.evaluate(comp_text, ex)
        p_unc = extract_trajectory_uncertainty(model, tok, ex.prompt, comp_text)
        v_res = verifier_ensemble.verify(comp_text, ex)

        policy_uncertainty = p_unc.mean_token_entropy
        verifier_uncertainty = float(v_res.metadata.get("disagreement", 0.0))

        rollouts.append(
            {
                "ex": ex,
                "comp": comp_text,
                "report": rep,
                "is_error": not rep.answer_correct,
                "invalid_path": not rep.complete_path,
                "p_unc": policy_uncertainty,
                "v_unc": verifier_uncertainty,
            }
        )

    # Determine medians for binary split
    median_p = float(np.median([r["p_unc"] for r in rollouts]))
    median_v = float(np.median([r["v_unc"] for r in rollouts]))

    quadrants = {
        "Q1: Low Policy Unc, Low Verifier Unc": [],
        "Q2: High Policy Unc, Low Verifier Unc": [],
        "Q3: Low Policy Unc, High Verifier Unc": [],
        "Q4: High Policy Unc, High Verifier Unc": [],
    }

    for r in rollouts:
        high_p = r["p_unc"] >= median_p
        high_v = r["v_unc"] > median_v

        if not high_p and not high_v:
            quadrants["Q1: Low Policy Unc, Low Verifier Unc"].append(r)
        elif high_p and not high_v:
            quadrants["Q2: High Policy Unc, Low Verifier Unc"].append(r)
        elif not high_p and high_v:
            quadrants["Q3: Low Policy Unc, High Verifier Unc"].append(r)
        else:
            quadrants["Q4: High Policy Unc, High Verifier Unc"].append(r)

    table = Table(title="Joint 2x2 Policy vs. Verifier Uncertainty Quadrants")
    table.add_column("Quadrant", justify="left")
    table.add_column("Count", justify="right")
    table.add_column("Answer Error Rate", justify="right")
    table.add_column("Path Invalidity Rate", justify="right")
    table.add_column("Primary Failure Mode", justify="left")

    for q_name, items in quadrants.items():
        if not items:
            table.add_row(q_name, "0", "N/A", "N/A", "None")
            continue
        err_rate = float(np.mean([r["is_error"] for r in items]))
        path_inv = float(np.mean([r["invalid_path"] for r in items]))

        if err_rate < 0.2:
            mode = "High-fidelity execution"
        elif path_inv > 0.8 and err_rate < 0.5:
            mode = "Shortcut / broken path with lucky answer"
        elif path_inv > 0.8 and err_rate > 0.8:
            mode = "Catastrophic syntax or reasoning breakdown"
        else:
            mode = "Intermediate premise confusion"

        table.add_row(
            q_name,
            str(len(items)),
            f"{err_rate:.1%}",
            f"{path_inv:.1%}",
            mode,
        )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
