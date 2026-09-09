"""Reasoning reward functions: Outcome-based, Process-based, and Hybrid rewards.

Ensures:
1. Exact, transparent reward definitions.
2. Process reward is normalized by task depth to prevent length-inflation reward hacking.
3. Distinguishes intermediate deduction quality from terminal answer correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


@dataclass(frozen=True)
class ReasoningRewardOutcome:
    """Breakdown of calculated rewards."""

    total_reward: float
    outcome_reward: float
    process_reward: float
    answer_correct: bool
    step_validity: float
    path_consistency: bool


class ReasoningRewardFunction:
    """Configurable reward function supporting Outcome, Process, and Hybrid regimes."""

    def __init__(
        self,
        reward_mode: str = "outcome",  # "outcome", "process", or "hybrid"
        hybrid_alpha: float = 0.5,
        syntax_penalty: float = -0.5,
    ) -> None:
        if reward_mode not in ("outcome", "process", "hybrid"):
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        self.reward_mode = reward_mode
        self.hybrid_alpha = hybrid_alpha
        self.syntax_penalty = syntax_penalty

    def evaluate(
        self,
        completion: str,
        example: ReasoningExample,
    ) -> ReasoningRewardOutcome:
        """Calculate reward and diagnostic metrics for a generated completion."""
        report = TraceEvaluator.evaluate(completion, example)

        if not report.syntax_valid:
            return ReasoningRewardOutcome(
                total_reward=self.syntax_penalty,
                outcome_reward=self.syntax_penalty,
                process_reward=0.0,
                answer_correct=False,
                step_validity=0.0,
                path_consistency=False,
            )

        # 1. Outcome Reward: +1.0 for correct answer, 0.0 for wrong
        outcome_r = 1.0 if report.answer_correct else 0.0

        # 2. Process Reward:
        # Normalized by task depth d so verbose rambling cannot artificially inflate reward
        # Each valid geodesic step on a consistent path contributes (1.0 / depth)
        if report.is_trace and example.depth > 0:
            # Reward only valid contiguous geodesic transitions
            valid_geodesic_steps = 0
            geodesic_set = set(example.edges)
            if report.path_consistency and len(report.extracted_steps) > 0:
                for step in report.extracted_steps:
                    if step in geodesic_set:
                        valid_geodesic_steps += 1
            # Bound process reward to [0.0, 1.0]
            process_r = min(1.0, valid_geodesic_steps / example.depth)
        else:
            # For answer-only mode, process reward is equivalent to outcome correctness
            process_r = outcome_r

        # 3. Combine according to mode
        if self.reward_mode == "outcome":
            total = outcome_r
        elif self.reward_mode == "process":
            total = process_r
        elif self.reward_mode == "hybrid":
            total = self.hybrid_alpha * process_r + (1.0 - self.hybrid_alpha) * outcome_r
        else:
            total = outcome_r

        return ReasoningRewardOutcome(
            total_reward=float(total),
            outcome_reward=float(outcome_r),
            process_reward=float(process_r),
            answer_correct=report.answer_correct,
            step_validity=report.step_validity,
            path_consistency=report.path_consistency,
        )

    def __call__(self, prompt: str, completion: str, example: ReasoningExample) -> float:
        """Interface compatible with language RL training loops."""
        return self.evaluate(completion, example).total_reward
