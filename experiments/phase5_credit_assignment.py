"""Phase 5 Experiment: Credit Assignment in Multi-Step Reasoning Chains.

Studies how different reward structures and advantage estimators assign credit
to trajectories containing an isolated single-step error:

Example:
    Ground Truth Geodesic: A -> B -> C -> D
    Candidate 1 (Flawless):      A -> B; B -> C; C -> D; A: D  (All valid, correct answer)
    Candidate 2 (Pivotal Error): A -> B; B -> X; X -> D; A: D  (Step 2 wrong, but arrives at D)
    Candidate 3 (Premature Fall): A -> X; X -> Y; Y -> Z; A: Z  (Failed at step 1)

Inspects:
- Outcome reward vs. Process reward per step.
- Token-level and step-level advantage assignment under Monte Carlo returns vs. GAE.
- Why outcome rewards assign credit to erroneous steps that happen to precede a correct answer.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.advantage import compute_gae
from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask
from rl_reasoning_lab.rewards.reasoning_rewards import ReasoningRewardFunction
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 5: Multi-Step Credit Assignment Analysis[/bold cyan]\n"
            "Evaluating step-level and token-level credit when exactly one reasoning\n"
            "step is corrupted in a derivation chain.",
            title="RL Reasoning Lab — Phase 5",
        )
    )

    task = GraphTraversalTask()
    ex = task.generate_example(depth=3, num_distractors=2, rng=None)
    # Ground truth path: e.g. A -> B -> C -> D
    p = ex.path
    u0, u1, u2, u3 = p[0], p[1], p[2], p[3]

    # Candidate 1: Perfect proof
    cand_perfect = f" R: {u0}->{u1}; {u1}->{u2}; {u2}->{u3}; A: {u3}"

    # Candidate 2: Intermediate corrupted step (u1 -> X instead of u1 -> u2), then lucky/forced correct answer
    cand_corrupted_step = f" R: {u0}->{u1}; {u1}->X; X->{u3}; A: {u3}"

    # Candidate 3: Early failure, wrong answer
    cand_early_fail = f" R: {u0}->X; X->Y; Y->Z; A: Z"

    candidates = [
        ("Candidate 1 (Flawless Chain)", cand_perfect),
        ("Candidate 2 (Pivotal Step 2 Error)", cand_corrupted_step),
        ("Candidate 3 (Early Step 1 Fail)", cand_early_fail),
    ]

    r_outcome_fn = ReasoningRewardFunction(reward_mode="outcome")
    r_process_fn = ReasoningRewardFunction(reward_mode="process")
    r_hybrid_fn = ReasoningRewardFunction(reward_mode="hybrid", hybrid_alpha=0.5)

    table = Table(title=f"Credit Assignment on Task: {ex.prompt}")
    table.add_column("Candidate Trajectory", justify="left")
    table.add_column("Final Ans", justify="center")
    table.add_column("Step Validity", justify="right")
    table.add_column("Outcome Reward", justify="right")
    table.add_column("Process Reward", justify="right")
    table.add_column("Hybrid Reward", justify="right")
    table.add_column("Credit Assignment Assessment", justify="left")

    for name, cand in candidates:
        rep = TraceEvaluator.evaluate(cand, ex)
        out_r = r_outcome_fn.evaluate(cand, ex).total_reward
        proc_r = r_process_fn.evaluate(cand, ex).total_reward
        hyb_r = r_hybrid_fn.evaluate(cand, ex).total_reward

        ans_str = "[green]CORRECT[/green]" if rep.answer_correct else "[red]WRONG[/red]"

        if name.startswith("Candidate 1"):
            verdict = "Both outcome & process reward correctly allocate +1.0."
        elif name.startswith("Candidate 2"):
            verdict = (
                "[red]Outcome reward assigns +1.0 despite hallucinated step u1->X![/red] "
                f"Process reward penalizes error ({proc_r:.2f})."
            )
        else:
            verdict = "Both rewards correctly assign 0.0 to invalid chain."

        table.add_row(
            f"{name}\n[dim]{cand}[/dim]",
            ans_str,
            f"{rep.step_validity:.1%}",
            f"{out_r:.2f}",
            f"{proc_r:.2f}",
            f"{hyb_r:.2f}",
            verdict,
        )

    console.print(table)

    # Step-level advantage propagation demonstration
    console.print(
        "\n[bold]Step-Level GAE Advantage Propagation across Steps [Step 1, Step 2, Step 3, Answer]:[/bold]"
    )
    # For Candidate 2: Step 1 valid (+1/3), Step 2 invalid (0.0), Step 3 invalid (0.0), Final answer (+1.0 in outcome)
    # Outcome-only rewards: [0.0, 0.0, 0.0, 1.0]
    # Process-only rewards: [0.33, 0.0, 0.0, 0.0]
    # Values estimated by critic: V = [0.2, 0.4, 0.6, 0.8]
    r_outcome_steps = [0.0, 0.0, 0.0, 1.0]
    r_process_steps = [0.333, 0.0, 0.0, 0.0]
    v_steps = [0.2, 0.4, 0.6, 0.8]
    next_v_steps = [0.4, 0.6, 0.8, 0.0]
    dones = [False, False, False, True]

    adv_outcome, _ = compute_gae(
        r_outcome_steps, v_steps, next_v_steps, dones, gamma=0.99, lambda_=0.95
    )
    adv_process, _ = compute_gae(
        r_process_steps, v_steps, next_v_steps, dones, gamma=0.99, lambda_=0.95
    )

    adv_table = Table(title="GAE Advantage Breakdown across Steps for Corrupted Candidate 2")
    adv_table.add_column("Derivation Step", justify="left")
    adv_table.add_column("Correctness", justify="center")
    adv_table.add_column("Outcome Reward Advantage", justify="right")
    adv_table.add_column("Process Reward Advantage", justify="right")
    adv_table.add_column("Diagnosis", justify="left")

    step_names = [
        f"Step 1 ({u0}->{u1})",
        f"Step 2 ({u1}->X)",
        f"Step 3 (X->{u3})",
        f"Answer (A: {u3})",
    ]
    corr_labels = [
        "[green]VALID[/green]",
        "[red]INVALID[/red]",
        "[red]INVALID[/red]",
        "[green]MATCH[/green]",
    ]

    for i in range(4):
        o_a = adv_outcome[i].item()
        p_a = adv_process[i].item()
        if i == 1:
            diag = "[red]Outcome GAE propagates positive advantage to invalid Step 2![/red]"
        elif i == 0:
            diag = "Both methods assign credit to valid initial step."
        else:
            diag = "Advantage tracks local TD error."
        adv_table.add_row(step_names[i], corr_labels[i], f"{o_a:+.3f}", f"{p_a:+.3f}", diag)

    console.print(adv_table)

    console.print("\n[bold]Key Findings on Credit Assignment:[/bold]")
    console.print(
        "1. [yellow]False Attribution in Outcome Rewards:[/yellow] When reward is placed solely on the final answer, "
        "backpropagated credit (via GAE or Monte Carlo returns) rewards every token in the sequence, including "
        "flawed intermediate steps that happened to stumble into a correct final token."
    )
    console.print(
        "2. [green]Process Reward Pinpointing:[/green] Process rewards restrict positive credit strictly to "
        "valid transitions, preventing credit leakage onto hallucinated intermediate nodes."
    )


if __name__ == "__main__":
    run_experiment()
