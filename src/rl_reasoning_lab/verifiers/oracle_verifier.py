"""Ground-truth Oracle Verifier.

Represents true latent utility U(tau).
Wraps TraceEvaluator to verify answer correctness, step validity, path consistency,
and complete path derivation with zero tolerance for hallucinations or shortcuts.

IMPORTANT: This oracle must NEVER be used as a training reward in misspecification experiments.
"""

from __future__ import annotations

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample
from rl_reasoning_lab.verifiers.base import BaseVerifier, VerifierResult
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


class OracleVerifier(BaseVerifier):
    """Ground-truth evaluator measuring true mathematical utility U(tau)."""

    def __init__(self, strict_path: bool = True) -> None:
        self.strict_path = strict_path

    def verify(
        self,
        completion: str,
        example: ReasoningExample,
    ) -> VerifierResult:
        report = TraceEvaluator.evaluate(completion, example)

        # In strict path mode, full utility requires a complete, unbroken valid path
        if self.strict_path:
            passed = report.complete_path
            score = 1.0 if passed else 0.0
        else:
            passed = report.answer_correct
            score = 1.0 if passed else 0.0

        reason = "VALID_COMPLETE_PATH" if report.complete_path else "FAILED_ORACLE"
        if not report.syntax_valid:
            reason = "SYNTAX_INVALID"
        elif not report.answer_correct:
            reason = "WRONG_ANSWER"
        elif not report.path_consistency:
            reason = "BROKEN_PATH_CONTINUITY"
        elif report.step_validity < 1.0:
            reason = "INVALID_EDGE_IN_TRACE"

        return VerifierResult(
            score=score,
            passed=passed,
            confidence=1.0,
            reason_code=reason,
            metadata={
                "answer_correct": report.answer_correct,
                "step_validity": report.step_validity,
                "path_consistency": report.path_consistency,
                "complete_path": report.complete_path,
                "extracted_answer": report.extracted_answer,
                "extracted_steps": report.extracted_steps,
                "reasoning_length": report.reasoning_length,
                "syntax_valid": report.syntax_valid,
            },
            properties_checked=[
                "syntax_valid",
                "answer_correct",
                "step_validity",
                "path_consistency",
                "complete_path",
            ],
            abstain=False,
        )
