"""Composite, invariant-gated, sequential, and ensemble verifiers.

Provides robust verification patterns:
1. InvariantGatedVerifier: enforces strict mathematical invariants before consulting soft proxy scores.
2. SequentialPipelineVerifier: multi-stage short-circuiting verifier pipeline.
3. VerifierEnsemble: aggregates multiple diverse judges with disagreement uncertainty and abstention.
"""

from __future__ import annotations

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample
from rl_reasoning_lab.verifiers.base import BaseVerifier, VerifierResult
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


class InvariantGatedVerifier(BaseVerifier):
    """Enforces hard non-negotiable structural invariants before consulting soft verifier scores."""

    def __init__(self, soft_verifier: BaseVerifier) -> None:
        self.soft_verifier = soft_verifier

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        ans, steps, syntax_valid, is_trace = TraceEvaluator.parse_completion(completion)

        # Invariant 1: Syntax must be valid
        if not syntax_valid or ans is None:
            return VerifierResult(
                score=-1.0,
                passed=False,
                confidence=1.0,
                reason_code="INVARIANT_VIOLATION_SYNTAX",
                metadata={"invariant_failed": "syntax_valid"},
                properties_checked=["invariants"],
            )

        # Invariant 2: If trace is provided, it must not be empty
        if is_trace and len(steps) == 0:
            return VerifierResult(
                score=-1.0,
                passed=False,
                confidence=1.0,
                reason_code="INVARIANT_VIOLATION_EMPTY_TRACE",
                metadata={"invariant_failed": "non_empty_trace"},
                properties_checked=["invariants"],
            )

        if is_trace and len(steps) > 0:
            # Invariant 3: Must begin at the prompt's designated start node
            if steps[0][0] != example.start_node:
                return VerifierResult(
                    score=-1.0,
                    passed=False,
                    confidence=1.0,
                    reason_code=f"INVARIANT_VIOLATION_START_NODE({steps[0][0]}!={example.start_node})",
                    metadata={"invariant_failed": "start_node_match"},
                    properties_checked=["invariants"],
                )

            # Invariant 4: Continuous path (no disconnected leaps)
            for k in range(1, len(steps)):
                if steps[k][0] != steps[k - 1][1]:
                    return VerifierResult(
                        score=-1.0,
                        passed=False,
                        confidence=1.0,
                        reason_code=f"INVARIANT_VIOLATION_DISCONNECTED_STEP({steps[k - 1][1]}->{steps[k][0]})",
                        metadata={"invariant_failed": "path_continuity"},
                        properties_checked=["invariants"],
                    )

            # Invariant 5: Final path destination must match declared answer token
            if steps[-1][1] != ans:
                return VerifierResult(
                    score=-1.0,
                    passed=False,
                    confidence=1.0,
                    reason_code=f"INVARIANT_VIOLATION_DESTINATION_MISMATCH({steps[-1][1]}!={ans})",
                    metadata={"invariant_failed": "destination_equals_answer"},
                    properties_checked=["invariants"],
                )

            # Invariant 6: Every edge in trace must exist in graph
            valid_edges = set(example.edges) | set(example.distractor_edges)
            for s in steps:
                if s not in valid_edges:
                    return VerifierResult(
                        score=-1.0,
                        passed=False,
                        confidence=1.0,
                        reason_code=f"INVARIANT_VIOLATION_NONEXISTENT_EDGE({s})",
                        metadata={"invariant_failed": "edge_existence"},
                        properties_checked=["invariants"],
                    )

        # If all invariants pass, defer to the underlying soft verifier
        soft_res = self.soft_verifier.verify(completion, example)
        return VerifierResult(
            score=soft_res.score,
            passed=soft_res.passed,
            confidence=soft_res.confidence,
            reason_code=f"INVARIANTS_PASSED_{soft_res.reason_code}",
            metadata={**soft_res.metadata, "invariants_passed": True},
            properties_checked=["invariants"] + soft_res.properties_checked,
            abstain=soft_res.abstain,
        )


class SequentialPipelineVerifier(BaseVerifier):
    """Multi-stage short-circuiting verifier pipeline for computational efficiency and early rejection."""

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        stage_records: dict[str, bool] = {}

        # Stage 1: Syntax Parsing Check
        ans, steps, syntax_valid, is_trace = TraceEvaluator.parse_completion(completion)
        stage_records["stage_1_syntax"] = syntax_valid
        if not syntax_valid or ans is None:
            return VerifierResult(
                score=0.0,
                passed=False,
                confidence=1.0,
                reason_code="PIPELINE_REJECTED_STAGE_1_SYNTAX",
                metadata={"stages": stage_records, "stopped_at": 1},
                properties_checked=list(stage_records.keys()),
            )

        # Stage 2: Target Answer Match Check
        answer_correct = ans == example.answer
        stage_records["stage_2_answer_match"] = answer_correct
        if not answer_correct:
            return VerifierResult(
                score=0.2,
                passed=False,
                confidence=1.0,
                reason_code="PIPELINE_REJECTED_STAGE_2_ANSWER",
                metadata={"stages": stage_records, "stopped_at": 2},
                properties_checked=list(stage_records.keys()),
            )

        # Stage 3: Edge Existence Check
        valid_edges = set(example.edges) | set(example.distractor_edges)
        all_edges_real = all(s in valid_edges for s in steps) if steps else False
        stage_records["stage_3_edge_existence"] = all_edges_real
        if not all_edges_real:
            return VerifierResult(
                score=0.5,
                passed=False,
                confidence=1.0,
                reason_code="PIPELINE_REJECTED_STAGE_3_EDGES",
                metadata={"stages": stage_records, "stopped_at": 3},
                properties_checked=list(stage_records.keys()),
            )

        # Stage 4: Path Continuity & Endpoints
        path_continuous = True
        for k in range(1, len(steps)):
            if steps[k][0] != steps[k - 1][1]:
                path_continuous = False
                break
        endpoints_valid = (
            len(steps) > 0
            and steps[0][0] == example.start_node
            and steps[-1][1] == example.target_node
        )
        stage_4_passed = path_continuous and endpoints_valid
        stage_records["stage_4_path_continuity"] = stage_4_passed
        if not stage_4_passed:
            return VerifierResult(
                score=0.7,
                passed=False,
                confidence=1.0,
                reason_code="PIPELINE_REJECTED_STAGE_4_PATH",
                metadata={"stages": stage_records, "stopped_at": 4},
                properties_checked=list(stage_records.keys()),
            )

        # All stages passed successfully
        return VerifierResult(
            score=1.0,
            passed=True,
            confidence=1.0,
            reason_code="PIPELINE_PASSED_ALL_STAGES",
            metadata={"stages": stage_records, "stopped_at": None},
            properties_checked=list(stage_records.keys()),
        )


class VerifierEnsemble(BaseVerifier):
    """Ensemble of diverse judges with consensus voting, uncertainty estimation, and abstention."""

    def __init__(
        self,
        verifiers: list[BaseVerifier],
        aggregation: str = "majority_vote",  # "majority_vote", "unanimity", "weighted_average", "minimum_score"
        weights: list[float] | None = None,
        abstention_threshold: float | None = None,  # If disagreement > threshold, abstain
    ) -> None:
        if not verifiers:
            raise ValueError("Ensemble must contain at least one verifier")
        self.verifiers = verifiers
        self.aggregation = aggregation
        self.weights = weights if weights is not None else [1.0 / len(verifiers)] * len(verifiers)
        self.abstention_threshold = abstention_threshold

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        sub_results = [v.verify(completion, example) for v in self.verifiers]
        n = len(sub_results)

        scores = [r.score for r in sub_results]
        passes = [r.passed for r in sub_results]

        pass_count = sum(1 for p in passes if p)
        fail_count = n - pass_count

        # Disagreement uncertainty: 1.0 - (|majority| / N)
        majority_count = max(pass_count, fail_count)
        disagreement = 1.0 - (majority_count / n)

        # Abstention check
        if self.abstention_threshold is not None and disagreement > self.abstention_threshold:
            return VerifierResult(
                score=0.0,
                passed=False,
                confidence=1.0 - disagreement,
                reason_code="ENSEMBLE_ABSTAIN_HIGH_UNCERTAINTY",
                metadata={
                    "disagreement": disagreement,
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "sub_results": [
                        {"name": v.name, "passed": r.passed, "score": r.score}
                        for v, r in zip(self.verifiers, sub_results, strict=False)
                    ],
                },
                properties_checked=["ensemble_consensus"],
                abstain=True,
            )

        # Aggregation logic
        if self.aggregation == "majority_vote":
            passed = pass_count > (n / 2)
            score = sum(scores) / n
        elif self.aggregation == "unanimity":
            passed = all(passes)
            score = min(scores) if passed else 0.0
        elif self.aggregation == "weighted_average":
            score = sum(w * s for w, s in zip(self.weights, scores, strict=False))
            passed = score >= 0.5
        elif self.aggregation == "minimum_score":
            score = min(scores)
            passed = score >= 0.5
        else:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation}")

        return VerifierResult(
            score=float(score),
            passed=passed,
            confidence=1.0 - disagreement,
            reason_code=f"ENSEMBLE_{self.aggregation.upper()}_{'PASS' if passed else 'FAIL'}",
            metadata={
                "disagreement": disagreement,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "sub_scores": scores,
            },
            properties_checked=["ensemble_consensus"],
            abstain=False,
        )
