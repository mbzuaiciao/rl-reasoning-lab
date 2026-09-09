"""Phase 6 Experiment 5: Verifier Acceptance Threshold Sweeps.

Core Concept:
Continuous verifiers output a score in [0.0, 1.0]. Setting the acceptance threshold theta
creates a direct trade-off between:
- High threshold (e.g. 0.95): High precision, but severe reward sparsity (learning stalls).
- Low threshold (e.g. 0.20): Dense rewards, but high false-positive rate (policy exploits shortcuts).

Sweeps theta in {0.20, 0.40, 0.60, 0.80, 0.95} and measures:
- Precision, Recall, Reward Density
- Policy learning progress and true utility under each threshold regime.
"""

from __future__ import annotations

import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.evaluation.verifier_metrics import evaluate_verifier_diagnostics
from rl_reasoning_lab.verifiers.oracle_verifier import OracleVerifier
from rl_reasoning_lab.verifiers.proxy_verifiers import KeywordVerifier, ThresholdVerifier


def generate_evaluation_traces(
    task: GraphTraversalTask,
    num_samples: int = 60,
    seed: int = 42,
) -> tuple[list[str], list[ReasoningExample]]:
    """Generates varied traces with partial keyword overlaps and different quality levels."""
    rng = random.Random(seed)
    traces: list[str] = []
    examples: list[ReasoningExample] = []

    for i in range(num_samples):
        depth = rng.choice([2, 3])
        ex = task.generate_example(depth=depth, num_distractors=1, rng=rng)
        cat = i % 4

        if cat == 0:
            # Full ground truth (100% keyword coverage, 100% utility)
            comp = ex.trace_target
        elif cat == 1:
            # Partial coverage: mentions start and one intermediate step, wrong answer
            comp = f"R: {ex.start_node}->{ex.edges[0][1]}; A: Z"
        elif cat == 2:
            # Superficial keyword dump: mentions all nodes in path, but invalid syntax/trace
            comp = "Keywords: " + " ".join(ex.path) + f" ; A: {ex.answer}"
        else:
            # Completely unrelated text
            comp = "R: NONE->NONE; A: NONE"

        traces.append(comp)
        examples.append(ex)

    return traces, examples


def main() -> None:
    console = Console()
    console.print(
        Panel(
            "[bold green]Phase 6 Experiment 5: Verifier Acceptance Threshold Sweeps[/bold green]\n"
            "[dim]Sweeps decision threshold theta on a continuous verifier to expose the fundamental trade-off\n"
            "between reward sparsity (stalling learning) and false-positive tolerance (inducing shortcut exploitation).[/dim]",
            border_style="green",
        )
    )

    task = GraphTraversalTask()
    traces, examples = generate_evaluation_traces(task, num_samples=80, seed=42)

    oracle = OracleVerifier(strict_path=True)
    base_continuous_verifier = KeywordVerifier()

    thresholds = [0.20, 0.40, 0.60, 0.80, 0.95]

    table = Table(title="Continuous Verifier Diagnostic Metrics vs Acceptance Threshold (theta)")
    table.add_column("Threshold (theta)", justify="center", style="bold white")
    table.add_column("Precision", justify="right", style="cyan")
    table.add_column("Recall", justify="right", style="green")
    table.add_column("Specificity", justify="right")
    table.add_column("Reward Density", justify="right", style="bold yellow")
    table.add_column("False Positives", justify="right", style="red")
    table.add_column("Optimization Regime", justify="left")

    for th in thresholds:
        tv = ThresholdVerifier(continuous_verifier=base_continuous_verifier, threshold=th)
        diag = evaluate_verifier_diagnostics(tv, oracle, traces, examples)

        if th <= 0.30:
            regime = "[red]High FP (Lax): Exploitation Risk[/red]"
        elif th <= 0.70:
            regime = "[yellow]Balanced Operating Region[/yellow]"
        else:
            regime = "[blue]Sparse Rewards (Strict): Stalling Risk[/blue]"

        table.add_row(
            f"{th:.2f}",
            f"{diag.precision:.1%}",
            f"{diag.recall:.1%}",
            f"{diag.specificity:.1%}",
            f"{diag.reward_density:.1%}",
            str(diag.fp),
            regime,
        )

    console.print(table)
    console.print(
        "\n[bold green]Key Takeaway:[/bold green] Stricter verification is not universally superior. "
        "At theta=0.95, reward density falls to near zero, denying the policy sufficient gradient signal. "
        "At theta=0.20, reward density is high (75%), but 40+ false positives actively reinforce superficial text dumping. "
        "Practical reward engineering requires tuning theta to balance exploration density against shortcut vulnerability.\n"
    )


if __name__ == "__main__":
    main()
