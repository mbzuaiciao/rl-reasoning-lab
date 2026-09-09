"""Multi-dimensional evaluator for reasoning traces and answer correctness.

Strictly separates:
1. Final answer correctness (does the answer token match ground truth?)
2. Reasoning process validity (are intermediate transitions valid edges in the premise graph?)
3. Path consistency (is the chain unbroken from start to target?)
4. Extraneous vs. missing steps.
5. Counterfactual consistency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample


@dataclass(frozen=True)
class TraceEvaluationReport:
    """Fine-grained diagnostic report for a single model completion."""

    answer_correct: bool
    syntax_valid: bool
    is_trace: bool
    extracted_answer: str | None
    extracted_steps: list[tuple[str, str]]
    step_validity: float
    path_consistency: bool
    complete_path: bool
    reasoning_length: int
    unnecessary_steps: int
    missing_steps: int


class TraceEvaluator:
    """Evaluates language model reasoning completions with exact ground-truth checking."""

    @staticmethod
    def parse_completion(
        completion: str,
    ) -> tuple[str | None, list[tuple[str, str]], bool, bool]:
        """Parse completion into answer, reasoning steps, syntax flag, and trace flag."""
        cleaned = completion.strip()

        # Check for answer section: 'A: <answer>'
        ans_match = re.search(r"A:\s*([A-Za-z0-9]+)", cleaned)
        extracted_answer = ans_match.group(1).strip() if ans_match else None

        # Check for reasoning section: 'R: <step>; <step>; ...'
        extracted_steps: list[tuple[str, str]] = []
        is_trace = False

        if "R:" in cleaned:
            is_trace = True
            # Extract content between 'R:' and '; A:' (or end of string)
            reasoning_part = cleaned.split("R:")[1]
            if "A:" in reasoning_part:
                reasoning_part = reasoning_part.split("A:")[0]

            # Find all X->Y pairs
            raw_steps = re.findall(r"([A-Za-z0-9]+)\s*->\s*([A-Za-z0-9]+)", reasoning_part)
            extracted_steps = [(u.strip(), v.strip()) for u, v in raw_steps]

        syntax_valid = extracted_answer is not None and (not is_trace or len(extracted_steps) > 0)

        return extracted_answer, extracted_steps, syntax_valid, is_trace

    @classmethod
    def evaluate(
        cls,
        completion: str,
        example: ReasoningExample,
    ) -> TraceEvaluationReport:
        """Evaluate a completion against the ground truth of a ReasoningExample."""
        ans, steps, syntax_valid, is_trace = cls.parse_completion(completion)

        answer_correct = ans == example.answer if ans is not None else False

        # If answer-only format is evaluated, trace metrics are neutral
        if not is_trace or not steps:
            return TraceEvaluationReport(
                answer_correct=answer_correct,
                syntax_valid=syntax_valid,
                is_trace=is_trace,
                extracted_answer=ans,
                extracted_steps=[],
                step_validity=1.0 if answer_correct else 0.0,
                path_consistency=answer_correct,
                complete_path=answer_correct,
                reasoning_length=0,
                unnecessary_steps=0,
                missing_steps=example.depth if not answer_correct else 0,
            )

        # Ground-truth valid edges from prompt
        valid_edge_set = set(example.edges) | set(example.distractor_edges)
        geodesic_edge_set = set(example.edges)

        # Step validity: fraction of extracted steps that are valid edges in the premise graph
        valid_step_count = sum(1 for step in steps if step in valid_edge_set)
        step_validity = valid_step_count / len(steps) if steps else 0.0

        # Path consistency: step[k].src == step[k-1].dst for all k >= 1
        path_consistency = False
        if steps:
            consistent = True
            for k in range(1, len(steps)):
                if steps[k][0] != steps[k - 1][1]:
                    consistent = False
                    break
            path_consistency = consistent

        # Complete path: consistent + all steps valid in graph + starts at start_node + ends at target_node + answer matches
        complete_path = (
            path_consistency
            and step_validity == 1.0
            and len(steps) > 0
            and steps[0][0] == example.start_node
            and steps[-1][1] == example.target_node
            and answer_correct
        )

        # Unnecessary steps: valid steps that are not part of the shortest path
        unnecessary = sum(1 for step in steps if step not in geodesic_edge_set)

        # Missing steps: geodesic steps omitted from extracted steps
        executed_geodesic = sum(1 for step in steps if step in geodesic_edge_set)
        missing = max(0, len(geodesic_edge_set) - executed_geodesic)

        return TraceEvaluationReport(
            answer_correct=answer_correct,
            syntax_valid=syntax_valid,
            is_trace=is_trace,
            extracted_answer=ans,
            extracted_steps=steps,
            step_validity=step_validity,
            path_consistency=path_consistency,
            complete_path=complete_path,
            reasoning_length=len(steps),
            unnecessary_steps=unnecessary,
            missing_steps=missing,
        )

    @classmethod
    def evaluate_counterfactual_pair(
        cls,
        orig_completion: str,
        intervened_completion: str,
        orig_example: ReasoningExample,
        intervened_example: ReasoningExample,
    ) -> bool:
        """Returns True if model completion updated consistently with the premise intervention.

        Specifically, checks whether the intervened completion matches the intervened ground truth
        instead of repeating the original answer.
        """
        rep_orig = cls.evaluate(orig_completion, orig_example)
        rep_inter = cls.evaluate(intervened_completion, intervened_example)

        # Causal consistency: original answer was correct, and after intervention,
        # the model changes its answer to the new target rather than holding the old target
        if not rep_orig.answer_correct:
            return False

        return rep_inter.answer_correct and (rep_inter.extracted_answer != orig_example.answer)
