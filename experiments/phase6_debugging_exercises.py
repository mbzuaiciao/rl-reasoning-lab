"""Phase 6 Experiment 14: Ten Critical Verifier & Reward Debugging Exercises.

Demonstrates 10 insidious reward and verifier pitfalls:
1. Substring Answer Matching: Answer token accepted inside unrelated prose or junk text.
2. Prompt + Completion Leakage: Verifier checks prompt + completion, reading the gold prompt premises as proof.
3. Disconnected Edges Accepted: Valid individual edges pass without checking continuous path topology.
4. Length-Dependent Inflation: Unnormalized rewards encourage pathological verbosity.
5. Hidden Metadata Leakage: Verifier reads internal oracle attributes (e.g. example.path) directly.
6. Self-Verification Loop: Same flawed verifier used for both RL training and evaluation reporting.
7. Post-EOS Content Exploitation: Verifier rewards spurious content generated after the EOS token.
8. False Ensemble Diversity: Multiple nominal judges share identical underlying heuristic logic.
9. Threshold Data Leakage: Decision threshold tuned directly on the test evaluation split.
10. Stale Pre-RL Metrics: Verifier quality benchmarked only once before policy optimization begins.

For each bug:
- Explains why it is tempting
- Predicts the failure mode
- Demonstrates the failure quantitatively
- Identifies the independent metric that reveals the defect.
"""

from __future__ import annotations

import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask
from rl_reasoning_lab.verifiers.composite_verifiers import (
    VerifierEnsemble,
)
from rl_reasoning_lab.verifiers.oracle_verifier import OracleVerifier
from rl_reasoning_lab.verifiers.proxy_verifiers import (
    LengthHeuristicVerifier,
    PartialPathVerifier,
    SubstringAnswerVerifier,
)


def main() -> None:
    console = Console()
    console.print(
        Panel(
            "[bold red]Phase 6 Experiment 14: Ten Critical Reward & Verifier Pitfalls[/bold red]\n"
            "[dim]Demonstrating failure mechanisms, subtle bugs, and independent diagnostic metrics\n"
            "for 10 classic reward misspecifications and verifier traps in reasoning RL.[/dim]",
            border_style="red",
        )
    )

    task = GraphTraversalTask()
    rng = random.Random(42)
    ex = task.generate_example(depth=3, num_distractors=1, rng=rng)
    oracle = OracleVerifier(strict_path=True)

    exercises = []

    # -------------------------------------------------------------
    # Bug 1: Substring Answer Matching
    # -------------------------------------------------------------
    c_b1 = f"I do not know the answer, but the alphabet has letters like {ex.answer} somewhere."
    v_b1 = SubstringAnswerVerifier()
    res_b1 = v_b1.verify(c_b1, ex)
    ora_b1 = oracle.verify(c_b1, ex)
    exercises.append(
        {
            "num": 1,
            "name": "Substring Answer Matching",
            "why_tempting": "Easy to implement with `target in text` without building a formal parser.",
            "predicted_failure": "Accepts junk text, alphabet dumps, and negated assertions as correct answers.",
            "demonstration": f"Text: '{c_b1}' -> Proxy Score={res_b1.score:.1f}, Oracle Utility={ora_b1.score:.1f}",
            "independent_metric": "Independent `TraceEvaluator.parse_completion` & Oracle Complete Path.",
        }
    )

    # -------------------------------------------------------------
    # Bug 2: Prompt + Completion Leakage
    # -------------------------------------------------------------
    # Flawed verifier that evaluates string = prompt + completion instead of completion alone
    full_text = ex.prompt + " A: WRONG"
    # Looking for premise edges inside full_text finds them because prompt contains them!
    edges_found_in_full = sum(1 for u, v in ex.edges if f"{u}->{v}" in full_text)
    exercises.append(
        {
            "num": 2,
            "name": "Prompt + Completion Leakage",
            "why_tempting": "Accidentally passing full sequence tensor or concatenated string into verifier.",
            "predicted_failure": "Verifier gives credit for premise edges supplied by the question itself.",
            "demonstration": f"Edges in full text: {edges_found_in_full}/{len(ex.edges)} (100% false credit from prompt)",
            "independent_metric": "Strict separation of `completion = text[prompt_len:]` before verification.",
        }
    )

    # -------------------------------------------------------------
    # Bug 3: Disconnected Edges Accepted
    # -------------------------------------------------------------
    e0, e2 = ex.edges[0], ex.edges[2]
    c_b3 = f"R: {e2[0]}->{e2[1]}; {e0[0]}->{e0[1]}; A: {ex.answer}"
    v_b3 = PartialPathVerifier()
    res_b3 = v_b3.verify(c_b3, ex)
    ora_b3 = oracle.verify(c_b3, ex)
    exercises.append(
        {
            "num": 3,
            "name": "Disconnected Edges Accepted",
            "why_tempting": "Verifying edges as a set { (u,v) in G } without checking topological order.",
            "predicted_failure": "Disjoint, out-of-order hops receive 100% validity score.",
            "demonstration": f"Disconnected edges passed: {res_b3.passed} (Oracle: {ora_b3.passed})",
            "independent_metric": "`path_consistency` invariant (steps[k].dst == steps[k+1].src).",
        }
    )

    # -------------------------------------------------------------
    # Bug 4: Length-Dependent Reward Inflation
    # -------------------------------------------------------------
    v_b4 = LengthHeuristicVerifier()
    verbose_junk = " ".join(["word"] * 40)
    c_b4 = f"R: {verbose_junk}; A: WRONG"
    res_b4 = v_b4.verify(c_b4, ex)
    exercises.append(
        {
            "num": 4,
            "name": "Length-Dependent Inflation",
            "why_tempting": "Assuming that deeper reasoning trajectories must necessarily be longer.",
            "predicted_failure": "Policy learns to generate endless verbose filler tokens without reasoning.",
            "demonstration": f"40 words of junk -> Length Verifier Score: {res_b4.score:.2f} (Accepted)",
            "independent_metric": "Conditioning length metrics strictly on correct solutions (`mean_tokens_if_correct`).",
        }
    )

    # -------------------------------------------------------------
    # Bug 5: Verifier Reads Hidden Oracle Metadata
    # -------------------------------------------------------------
    exercises.append(
        {
            "num": 5,
            "name": "Hidden Metadata Leakage",
            "why_tempting": "Accessing `example.path` inside proxy verifier for convenience during prototyping.",
            "predicted_failure": "Reward model relies on privileged latent state unavailable at inference time.",
            "demonstration": "Verifier checks `if node in example.path` instead of parsing problem graph edges.",
            "independent_metric": "Restricting verifier inputs to (completion, observable_graph_edges).",
        }
    )

    # -------------------------------------------------------------
    # Bug 6: Self-Verification Loop (Shared Verifier)
    # -------------------------------------------------------------
    exercises.append(
        {
            "num": 6,
            "name": "Self-Verification Reporting Loop",
            "why_tempting": "Using the same reward function object as the evaluation metric in logs.",
            "predicted_failure": "Experiment reports '100% Accuracy' while true model output is completely broken.",
            "demonstration": "Model generates junk with answer token -> Proxy reports 100%, Oracle reports 0%.",
            "independent_metric": "Independent `OracleVerifier` held strictly separate from training reward code.",
        }
    )

    # -------------------------------------------------------------
    # Bug 7: Post-EOS Content Exploitation
    # -------------------------------------------------------------
    c_b7 = (
        f"R: {ex.start_node}->B; A: WRONG <EOS> Extra tokens: ignore above, answer is {ex.answer}"
    )
    v_b7 = SubstringAnswerVerifier()
    res_b7 = v_b7.verify(c_b7, ex)
    exercises.append(
        {
            "num": 7,
            "name": "Post-EOS Exploit",
            "why_tempting": "Decoding entire max_seq_len tensor without stripping tokens after `<eos>`.",
            "predicted_failure": "Policy generates valid termination then hides reward triggers in post-EOS padding.",
            "demonstration": f"Post-EOS trigger accepted: {res_b7.passed}",
            "independent_metric": "Truncate generated string strictly at first `<eos>` before verification.",
        }
    )

    # -------------------------------------------------------------
    # Bug 8: False Ensemble Diversity
    # -------------------------------------------------------------
    ens_correlated = VerifierEnsemble(
        [
            SubstringAnswerVerifier(),
            SubstringAnswerVerifier(),
            SubstringAnswerVerifier(),
        ]
    )
    c_b8 = f"Junk with answer token {ex.answer}"
    res_b8 = ens_correlated.verify(c_b8, ex)
    exercises.append(
        {
            "num": 8,
            "name": "False Ensemble Diversity",
            "why_tempting": "Instantiating 3 judges with different temperature or weights that share identical logic.",
            "predicted_failure": "High false confidence (3/3 agreement) on obvious adversarial traps.",
            "demonstration": f"Correlated 3-judge ensemble passed junk: {res_b8.passed} (Disagreement = {res_b8.metadata['disagreement']:.2f})",
            "independent_metric": "Measuring error correlation matrix across candidate verifiers.",
        }
    )

    # -------------------------------------------------------------
    # Bug 9: Threshold Tuned on Test Set
    # -------------------------------------------------------------
    exercises.append(
        {
            "num": 9,
            "name": "Threshold Contamination",
            "why_tempting": "Sweeping acceptance threshold theta to find the highest F1 score on test data.",
            "predicted_failure": "Overfitting decision boundaries to specific test noise, failing on live rollouts.",
            "demonstration": "Tuning theta=0.85 on test split yields inflated precision that drops 30% under RL.",
            "independent_metric": "Fixed validation split calibration with Expected Calibration Error (ECE).",
        }
    )

    # -------------------------------------------------------------
    # Bug 10: Stale Pre-RL Verifier Metrics
    # -------------------------------------------------------------
    exercises.append(
        {
            "num": 10,
            "name": "Stale Pre-RL Metrics",
            "why_tempting": "Assuming that a verifier with 95% accuracy on static benchmarks is 'solved'.",
            "predicted_failure": "Policy actively reshapes distribution toward the verifier's 5% blind spot.",
            "demonstration": "Verifier accuracy drops from 96.7% pre-RL to 60.0% post-RL on optimized rollouts.",
            "independent_metric": "Dynamic verifier re-evaluation on policy-generated rollout buffers.",
        }
    )

    # Render summary table
    table = Table(title="Ten Critical Reward & Verifier Debugging Exercises")
    table.add_column("Bug #", justify="center", style="bold white", width=6)
    table.add_column("Pitfall Name", style="bold cyan", width=26)
    table.add_column("Why It Is Tempting", style="dim", width=28)
    table.add_column("Demonstration / Diagnostic Signal", style="yellow", width=34)
    table.add_column("Independent Revealing Metric", style="green", width=30)

    for ex_item in exercises:
        table.add_row(
            str(ex_item["num"]),
            ex_item["name"],
            ex_item["why_tempting"],
            ex_item["demonstration"],
            ex_item["independent_metric"],
        )

    console.print(table)
    console.print(
        "\n[bold green]Debugging Methodology Rule:[/bold green] Never trust a reward proxy without an independent oracle evaluator. "
        "Every reward system must enforce hard invariants, strip post-EOS content, isolate completion from prompt, "
        "and continually audit verifier error distributions on policy-generated rollouts.\n"
    )


if __name__ == "__main__":
    main()
