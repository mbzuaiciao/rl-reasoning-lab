"""Phase 5 Debugging Exercises: 7 Critical Bugs in Multi-Step Reasoning Systems.

Demonstrates:
1. Wrong Trace Parser: Accepting malformed/hallucinated transitions.
2. Reward Leakage: Evaluator accidentally leaks gold answer into prompt conditioning.
3. Depth Leakage: Formatting artifacts reveal ground-truth task depth to the model.
4. Process Reward Double-Counting: Unnormalized process rewards reward rambling verbosity.
5. Substring Answer Parser Bug: Naive substring matching falsely credits wrong answers.
6. Curriculum Test Leakage: Evaluation instances contaminate training curriculum.
7. Incorrect Trace Token Alignment: Reward for reasoning step t is attached to step t-1.
"""

from __future__ import annotations

import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask


def bug_1_wrong_trace_parser() -> tuple[float, float, str]:
    """Bug 1: Flawed regex accepts invalid transitions as valid steps."""
    malformed_trace = " R: A->B; B??C; C==>D; A: D"

    # Correct parser: strictly matches [A-Z]->[A-Z]
    correct_steps = re.findall(r"([A-Z])\s*->\s*([A-Z])", malformed_trace)
    valid_correct = len(correct_steps)  # Only 1 step ('A', 'B')

    # Buggy parser: overly permissive regex matching any letter pair
    buggy_steps = re.findall(r"([A-Z])[^A-Z]+([A-Z])", malformed_trace)
    valid_buggy = len(buggy_steps)  # Matches B??C, C==>D, etc.

    diagnosis = (
        f"Correct parser identifies {valid_correct} valid step. "
        f"Buggy parser falsely accepts {valid_buggy} steps from garbled syntax, "
        "inflating apparent process validity by accepting non-conforming tokens."
    )
    return float(valid_correct), float(valid_buggy), diagnosis


def bug_2_reward_leakage() -> tuple[str, str, str]:
    """Bug 2: Gold answer inadvertently leaked into the prompt during evaluation."""
    task = GraphTraversalTask()
    ex = task.generate_example(depth=2)

    # Correct prompt: excludes target answer
    correct_prompt = ex.prompt

    # Buggy prompt: developer concatenated full_text or target to prompt
    buggy_prompt = ex.prompt + f" (Target: {ex.answer})"

    diagnosis = (
        f"Correct prompt: '{correct_prompt}'. "
        f"Buggy prompt: '{buggy_prompt}'. "
        "Headline accuracy reaches 100% because the model learns to copy the leaked answer from prompt context."
    )
    return "No Leakage", "Leaked Answer in Context", diagnosis


def bug_3_depth_leakage() -> tuple[int, int, str]:
    """Bug 3: Formatting artifacts encode task depth directly in prompt length or tokens."""
    task = GraphTraversalTask()
    # Correct: distractor rules mask depth so prompt length != task depth
    ex_masked = task.generate_example(depth=1, num_distractors=3)
    edges_count_masked = len(ex_masked.prompt.split(";")) - 1

    # Buggy: prompt contains zero distractors, so rule count exactly equals depth
    ex_leaked = task.generate_example(depth=1, num_distractors=0)
    edges_count_leaked = len(ex_leaked.prompt.split(";")) - 1

    diagnosis = (
        f"With distractors, prompt has {edges_count_masked} clauses for depth {ex_masked.depth}. "
        f"Without distractors, clause count ({edges_count_leaked}) strictly reveals depth {ex_leaked.depth}, "
        "allowing the model to predict depth shortcuts instead of traversing relations."
    )
    return edges_count_masked, edges_count_leaked, diagnosis


def bug_4_process_reward_length_inflation() -> tuple[float, float, str]:
    """Bug 4: Process reward unnormalized by depth, rewarding infinite rambling."""
    # A candidate repeats the same valid step A->B 10 times on a depth-1 problem
    depth = 1
    repeated_steps = [("A", "B")] * 10
    valid_geodesic = 1  # only 1 unique geodesic step needed

    # Correct normalized reward: bounded by depth
    correct_reward = min(1.0, valid_geodesic / depth)

    # Buggy unnormalized reward: sum over all steps without normalization
    buggy_reward = float(len(repeated_steps)) * 1.0

    diagnosis = (
        f"Correct normalized process reward: {correct_reward:.1f}. "
        f"Buggy unnormalized reward: {buggy_reward:.1f}. "
        "Unnormalized reward causes severe length hacking: the policy outputs 50 repetitions to maximize reward."
    )
    return correct_reward, buggy_reward, diagnosis


def bug_5_substring_answer_parser() -> tuple[bool, bool, str]:
    """Bug 5: Naive substring matching falsely marks incorrect multi-letter answers as correct."""
    gold_answer = "B"
    wrong_completion = " A: BB"  # Generated 'BB', wrong answer

    # Correct parser: exact word boundary / regex match
    match = re.search(r"A:\s*([A-Za-z0-9]+)", wrong_completion)
    extracted = match.group(1).strip() if match else None
    correct_judgment = extracted == gold_answer

    # Buggy parser: naive substring check 'if gold in completion'
    buggy_judgment = gold_answer in wrong_completion

    diagnosis = (
        f"Correct parser judgment: {correct_judgment}. "
        f"Buggy substring parser judgment: {buggy_judgment}. "
        "Substring search creates false positives whenever the gold answer is a substring of wrong tokens."
    )
    return correct_judgment, buggy_judgment, diagnosis


def bug_6_curriculum_test_leakage() -> tuple[int, int, str]:
    """Bug 6: Overlap between training curriculum dataset and held-out test dataset."""
    task = GraphTraversalTask()
    # Correct: distinct seeds guarantee disjoint sampling or verified zero overlap
    train_set = task.generate_dataset(size=20, depths=[2], seed=42)
    test_set_clean = task.generate_dataset(size=20, depths=[2], seed=999)
    overlap_correct = len(set(e.prompt for e in train_set) & set(e.prompt for e in test_set_clean))

    # Buggy: developer reused seed=42 for both train and test
    test_set_leaked = task.generate_dataset(size=20, depths=[2], seed=42)
    overlap_buggy = len(set(e.prompt for e in train_set) & set(e.prompt for e in test_set_leaked))

    diagnosis = (
        f"Clean split prompt overlap: {overlap_correct}. "
        f"Buggy split prompt overlap: {overlap_buggy} / 20 ({overlap_buggy / 20:.0%}). "
        "Dataset leakage creates apparent 100% generalization that collapses in production."
    )
    return overlap_correct, overlap_buggy, diagnosis


def bug_7_incorrect_trace_alignment() -> tuple[float, float, str]:
    """Bug 7: Off-by-one alignment attaching step reward to previous step's tokens."""
    # Step 1 is correct (+1.0), Step 2 is incorrect (0.0)
    step_rewards = [1.0, 0.0]

    # Correct alignment: index 0 gets 1.0, index 1 gets 0.0
    correct_credit_to_step_1 = step_rewards[0]

    # Buggy shifted alignment: step_rewards shifted by +1
    buggy_shifted = [0.0] + step_rewards[:-1]
    buggy_credit_to_step_1 = buggy_shifted[0]

    diagnosis = (
        f"Correct credit for Step 1: {correct_credit_to_step_1:.1f}. "
        f"Buggy shifted credit for Step 1: {buggy_credit_to_step_1:.1f}. "
        "Misaligned token reward assigns credit to prior tokens, distorting value estimation."
    )
    return correct_credit_to_step_1, buggy_credit_to_step_1, diagnosis


def run_all_exercises() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold red]Phase 5 Debugging Exercises: 7 Critical Reasoning Bugs[/bold red]\n"
            "Dissecting subtle parser, leakage, and credit assignment bugs in reasoning RL.",
            title="RL Reasoning Lab — Diagnostics",
        )
    )

    bugs = [
        ("1. Wrong Trace Parser", bug_1_wrong_trace_parser),
        ("2. Reward Leakage into Prompt", bug_2_reward_leakage),
        ("3. Depth Leakage via Clauses", bug_3_depth_leakage),
        ("4. Process Reward Length Inflation", bug_4_process_reward_length_inflation),
        ("5. Substring Answer Parser Bug", bug_5_substring_answer_parser),
        ("6. Curriculum Dataset Leakage", bug_6_curriculum_test_leakage),
        ("7. Incorrect Trace Reward Alignment", bug_7_incorrect_trace_alignment),
    ]

    table = Table(title="Reasoning Evaluation & Training Failure Modes")
    table.add_column("Bug Name", style="bold yellow", justify="left")
    table.add_column("Correct Value", justify="right")
    table.add_column("Buggy Value", justify="right")
    table.add_column("Mechanism & Consequence", justify="left")

    for name, func in bugs:
        val_c, val_b, diag = func()
        str_c = f"{val_c:.3f}" if isinstance(val_c, float) else str(val_c)
        str_b = f"{val_b:.3f}" if isinstance(val_b, float) else str(val_b)
        table.add_row(name, str_c, str_b, diag)

    console.print(table)


if __name__ == "__main__":
    run_all_exercises()
