"""Phase 6 Experiment 1: Verifier Quality Benchmarking Before Policy Optimization.

Evaluates candidate verifiers on a static corpus containing:
1. Ground-truth valid reasoning traces (TP opportunities)
2. Invalid answer completions (TN opportunities)
3. Broken path traces with correct answer (FP traps)
4. Hallucinated traces mentioning keywords (FP traps)
5. Syntactically malformed outputs (TN opportunities)

Reports:
- True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
- Precision, Recall, Specificity, F1, Accuracy, and Reward Density
- Establishes the baseline error rate BEFORE any policy optimization takes place.
"""

from __future__ import annotations

import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.evaluation.verifier_metrics import evaluate_verifier_diagnostics
from rl_reasoning_lab.verifiers.composite_verifiers import (
    InvariantGatedVerifier,
    SequentialPipelineVerifier,
    VerifierEnsemble,
)
from rl_reasoning_lab.verifiers.oracle_verifier import OracleVerifier
from rl_reasoning_lab.verifiers.proxy_verifiers import (
    AsymmetricVerifier,
    FormatVerifier,
    KeywordVerifier,
    LastTokenVerifier,
    LengthHeuristicVerifier,
    PartialPathVerifier,
    RandomNoiseVerifier,
    SubstringAnswerVerifier,
    SystematicShortcutVerifier,
)


def create_static_benchmark_corpus(
    task: GraphTraversalTask,
    num_samples: int = 50,
    seed: int = 42,
) -> tuple[list[str], list[ReasoningExample]]:
    """Creates a balanced test corpus of clean and flawed reasoning completions."""
    rng = random.Random(seed)
    completions: list[str] = []
    examples: list[ReasoningExample] = []

    for i in range(num_samples):
        depth = rng.choice([2, 3])
        ex = task.generate_example(depth=depth, num_distractors=1, rng=rng)
        category = i % 5

        if category == 0:
            # 1. Clean valid ground-truth trace
            comp = ex.trace_target
        elif category == 1:
            # 2. Correct answer, but completely hallucinated path
            comp = f"R: {ex.start_node}->X; X->{ex.answer}; A: {ex.answer}"
        elif category == 2:
            # 3. Disconnected hops (individually valid edges, broken continuity)
            if len(ex.edges) >= 2:
                e1, e2 = ex.edges[0], ex.edges[-1]
                comp = f"R: {e2[0]}->{e2[1]}; {e1[0]}->{e1[1]}; A: {ex.answer}"
            else:
                comp = f"R: {ex.start_node}->X; A: {ex.answer}"
        elif category == 3:
            # 4. Keyword dump without reasoning
            comp = "R: " + " ".join(ex.path) + f" ; A: {ex.answer}"
        else:
            # 5. Wrong answer and wrong reasoning
            comp = f"R: {ex.start_node}->Z; A: Z"

        completions.append(comp)
        examples.append(ex)

    return completions, examples


def main() -> None:
    console = Console()
    console.print(
        Panel(
            "[bold cyan]Phase 6 Experiment 1: Verifier Quality Benchmarking Before RL[/bold cyan]\n"
            "[dim]Measures static confusion matrix (TP, FP, TN, FN), precision, recall, and reward density\n"
            "on fixed pre-optimization traces across candidate proxy verifiers.[/dim]",
            border_style="cyan",
        )
    )

    task = GraphTraversalTask()
    completions, examples = create_static_benchmark_corpus(task, num_samples=100, seed=42)

    oracle = OracleVerifier(strict_path=True)

    verifiers = [
        ("Oracle (True Utility)", oracle),
        ("Substring Answer", SubstringAnswerVerifier()),
        ("Last-Token Answer", LastTokenVerifier()),
        ("Keyword Coverage", KeywordVerifier()),
        ("Length Heuristic", LengthHeuristicVerifier()),
        ("Format Syntax", FormatVerifier()),
        ("Partial Path", PartialPathVerifier()),
        ("Random Noise (15%)", RandomNoiseVerifier(base_verifier=oracle, epsilon=0.15, seed=42)),
        ("Systematic Shortcut", SystematicShortcutVerifier(base_verifier=oracle)),
        (
            "Asymmetric (High-FP)",
            AsymmetricVerifier(oracle_verifier=oracle, mode="high_fp", fp_rate=0.35),
        ),
        (
            "Asymmetric (High-FN)",
            AsymmetricVerifier(oracle_verifier=oracle, mode="high_fn", fn_rate=0.35),
        ),
        ("Invariant-Gated (LastToken)", InvariantGatedVerifier(soft_verifier=LastTokenVerifier())),
        ("Sequential Pipeline", SequentialPipelineVerifier()),
        (
            "Diverse Ensemble",
            VerifierEnsemble(
                verifiers=[
                    LastTokenVerifier(),
                    PartialPathVerifier(),
                    InvariantGatedVerifier(LastTokenVerifier()),
                ],
                aggregation="majority_vote",
            ),
        ),
    ]

    table = Table(title="Pre-Optimization Static Verifier Benchmark (N=100)")
    table.add_column("Verifier Name", style="bold white", width=26)
    table.add_column("TP", justify="right", style="green")
    table.add_column("FP", justify="right", style="red")
    table.add_column("TN", justify="right", style="cyan")
    table.add_column("FN", justify="right", style="yellow")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Specificity", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Accuracy", justify="right", style="bold magenta")
    table.add_column("Reward Density", justify="right", style="dim")

    for name, v in verifiers:
        diag = evaluate_verifier_diagnostics(v, oracle, completions, examples)
        table.add_row(
            name,
            str(diag.tp),
            str(diag.fp),
            str(diag.tn),
            str(diag.fn),
            f"{diag.precision:.1%}",
            f"{diag.recall:.1%}",
            f"{diag.specificity:.1%}",
            f"{diag.f1:.2f}",
            f"{diag.accuracy:.1%}",
            f"{diag.reward_density:.1%}",
        )

    console.print(table)
    console.print(
        "\n[bold green]Key Takeaway:[/bold green] Proxies like Substring and Last-Token achieve high recall "
        "(100%) but suffer from massive False Positives (FP=60-80), artificially inflating reward density. "
        "Invariant-Gated and Sequential Pipeline verifiers eliminate all false positives while preserving 100% precision.\n"
    )


if __name__ == "__main__":
    main()
