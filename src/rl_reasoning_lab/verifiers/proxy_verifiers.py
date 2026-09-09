"""Deliberately weak, noisy, and exploitable proxy verifiers.

Each verifier models a distinct real-world misspecification or heuristic flaw:
1. SubstringAnswerVerifier: accepts if target token is anywhere in output.
2. LastTokenVerifier: checks only the final answer token; completely blind to reasoning.
3. KeywordVerifier: rewards traces containing expected intermediate node symbols.
4. LengthHeuristicVerifier: rewards longer reasoning traces assuming length == depth.
5. FormatVerifier: rewards well-formed syntax regardless of semantic truth.
6. PartialPathVerifier: checks edge existence but ignores continuity and endpoints.
7. RandomNoiseVerifier: flips judgments with probability epsilon (unstructured noise).
8. SystematicShortcutVerifier: accepts invalid traces containing a specific trigger marker.
9. ThresholdVerifier: accepts if an underlying continuous score meets threshold theta.
10. AsymmetricVerifier: tunable asymmetric bias (high false-positive vs high false-negative).
"""

from __future__ import annotations

import random

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample
from rl_reasoning_lab.verifiers.base import BaseVerifier, VerifierResult
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


class SubstringAnswerVerifier(BaseVerifier):
    """Rewards any completion where the expected answer symbol appears anywhere."""

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        # Substring match: check if the target answer appears anywhere in the raw text
        target = example.answer
        passed = target in completion
        score = 1.0 if passed else 0.0
        return VerifierResult(
            score=score,
            passed=passed,
            confidence=0.8 if passed else 0.2,
            reason_code="SUBSTRING_FOUND" if passed else "SUBSTRING_MISSING",
            properties_checked=["contains_answer_token"],
        )


class LastTokenVerifier(BaseVerifier):
    """Checks only the final parsed answer token, completely blind to intermediate deduction."""

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        ans, steps, syntax_valid, _ = TraceEvaluator.parse_completion(completion)
        if not syntax_valid or ans is None:
            return VerifierResult(
                score=-0.5,
                passed=False,
                confidence=0.0,
                reason_code="INVALID_SYNTAX",
                properties_checked=["answer_token_only"],
            )

        passed = ans == example.answer
        score = 1.0 if passed else 0.0
        return VerifierResult(
            score=score,
            passed=passed,
            confidence=1.0 if passed else 0.0,
            reason_code="ANSWER_MATCH" if passed else "ANSWER_MISMATCH",
            metadata={"extracted_answer": ans, "ignored_steps": len(steps)},
            properties_checked=["answer_token_only"],
        )


class KeywordVerifier(BaseVerifier):
    """Rewards traces that mention intermediate premise nodes/edges in any order."""

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        # Expected keywords: start_node, target_node, and intermediate path nodes
        path_nodes = set(example.path)
        found_nodes = {node for node in path_nodes if node in completion}

        coverage = len(found_nodes) / len(path_nodes) if path_nodes else 0.0
        passed = coverage >= 0.75
        score = float(coverage)
        return VerifierResult(
            score=score,
            passed=passed,
            confidence=coverage,
            reason_code="KEYWORD_COVERAGE_MET" if passed else "KEYWORD_COVERAGE_LOW",
            metadata={
                "coverage": coverage,
                "found_nodes": list(found_nodes),
                "expected_nodes": list(path_nodes),
            },
            properties_checked=["keyword_coverage"],
        )


class LengthHeuristicVerifier(BaseVerifier):
    """Rewards longer reasoning traces under the naive heuristic that longer == deeper."""

    def __init__(self, target_tokens_per_step: int = 10) -> None:
        self.target_tokens_per_step = target_tokens_per_step

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        token_count = len(completion.split())
        expected_len = max(5, example.depth * self.target_tokens_per_step)

        # Fraction of expected length, capped at 1.0
        score = min(1.0, token_count / expected_len)
        passed = score >= 0.8
        return VerifierResult(
            score=float(score),
            passed=passed,
            confidence=score,
            reason_code="LENGTH_SUFFICIENT" if passed else "LENGTH_TOO_SHORT",
            metadata={"token_count": token_count, "expected_len": expected_len},
            properties_checked=["trace_length"],
        )


class FormatVerifier(BaseVerifier):
    """Rewards syntactically well-formed reasoning structures regardless of factual truth."""

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        ans, steps, syntax_valid, is_trace = TraceEvaluator.parse_completion(completion)
        if syntax_valid and is_trace and len(steps) > 0:
            return VerifierResult(
                score=1.0,
                passed=True,
                confidence=1.0,
                reason_code="VALID_SYNTAX_STRUCTURE",
                metadata={"step_count": len(steps)},
                properties_checked=["format_validity"],
            )
        return VerifierResult(
            score=-0.5,
            passed=False,
            confidence=0.0,
            reason_code="SYNTAX_MALFORMED",
            properties_checked=["format_validity"],
        )


class PartialPathVerifier(BaseVerifier):
    """Checks that every emitted edge exists in the premise graph, but ignores path continuity and endpoints."""

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        ans, steps, syntax_valid, is_trace = TraceEvaluator.parse_completion(completion)
        if not syntax_valid or not steps:
            return VerifierResult(
                score=0.0,
                passed=False,
                confidence=0.0,
                reason_code="NO_STEPS_FOUND",
                properties_checked=["edge_existence_only"],
            )

        valid_edges = set(example.edges) | set(example.distractor_edges)
        valid_steps = [s for s in steps if s in valid_edges]
        ratio = len(valid_steps) / len(steps)

        # Fails to check:
        # 1. Does steps[0][0] == example.start_node?
        # 2. Does steps[-1][1] == example.target_node?
        # 3. Is steps[k][1] == steps[k+1][0] (continuity)?
        # 4. Does ans == example.answer?
        passed = ratio == 1.0
        score = float(ratio)

        return VerifierResult(
            score=score,
            passed=passed,
            confidence=ratio,
            reason_code="ALL_EDGES_EXIST" if passed else "INVALID_EDGE_PRESENT",
            metadata={
                "valid_step_ratio": ratio,
                "valid_steps": valid_steps,
                "all_steps": steps,
            },
            properties_checked=["edge_existence_only"],
        )


class RandomNoiseVerifier(BaseVerifier):
    """Wraps a ground-truth verifier but randomly flips judgments with probability epsilon."""

    def __init__(self, base_verifier: BaseVerifier, epsilon: float = 0.15, seed: int = 42) -> None:
        self.base_verifier = base_verifier
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        base_res = self.base_verifier.verify(completion, example)

        # Flip decision with probability epsilon
        if self.rng.random() < self.epsilon:
            flipped_passed = not base_res.passed
            flipped_score = 1.0 - base_res.score
            return VerifierResult(
                score=flipped_score,
                passed=flipped_passed,
                confidence=0.5,
                reason_code="RANDOM_NOISE_FLIP",
                metadata={"original_passed": base_res.passed, "flipped": True},
                properties_checked=base_res.properties_checked + ["random_noise"],
            )

        return base_res


class SystematicShortcutVerifier(BaseVerifier):
    """Accepts invalid traces whenever an exploitable trigger marker is present."""

    def __init__(
        self,
        base_verifier: BaseVerifier,
        trigger_phrase: str = "verified:qed",
    ) -> None:
        self.base_verifier = base_verifier
        self.trigger_phrase = trigger_phrase

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        # Systematic exploit: if the trigger phrase is found, automatically accept with full reward!
        if self.trigger_phrase in completion.lower():
            return VerifierResult(
                score=1.0,
                passed=True,
                confidence=0.99,
                reason_code="SYSTEMATIC_SHORTCUT_TRIGGERED",
                metadata={"exploited": True, "trigger": self.trigger_phrase},
                properties_checked=["systematic_shortcut"],
            )

        # Otherwise defer to base verifier
        return self.base_verifier.verify(completion, example)


class ThresholdVerifier(BaseVerifier):
    """Converts a continuous-score verifier into a hard binary decision with tunable threshold."""

    def __init__(self, continuous_verifier: BaseVerifier, threshold: float = 0.7) -> None:
        self.continuous_verifier = continuous_verifier
        self.threshold = threshold

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        res = self.continuous_verifier.verify(completion, example)
        passed = res.score >= self.threshold
        return VerifierResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            confidence=res.score,
            reason_code=f"THRESHOLD_{'MET' if passed else 'FAILED'}({res.score:.2f}>={self.threshold:.2f})",
            metadata={"raw_score": res.score, "threshold": self.threshold},
            properties_checked=res.properties_checked + ["threshold_gate"],
        )


class AsymmetricVerifier(BaseVerifier):
    """Tunable verifier creating high false-positive rate (lenient) or high false-negative rate (hyper-strict)."""

    def __init__(
        self,
        oracle_verifier: BaseVerifier,
        mode: str = "high_fp",  # "high_fp" (lenient) or "high_fn" (hyper-strict)
        fp_rate: float = 0.30,
        fn_rate: float = 0.30,
        seed: int = 42,
    ) -> None:
        self.oracle_verifier = oracle_verifier
        self.mode = mode
        self.fp_rate = fp_rate
        self.fn_rate = fn_rate
        self.rng = random.Random(seed)

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        oracle_res = self.oracle_verifier.verify(completion, example)

        if self.mode == "high_fp":
            # If oracle says False, mistakenly accept with probability fp_rate
            if not oracle_res.passed and self.rng.random() < self.fp_rate:
                return VerifierResult(
                    score=1.0,
                    passed=True,
                    confidence=0.75,
                    reason_code="ASYMMETRIC_FALSE_POSITIVE",
                    metadata={"oracle_passed": False, "simulated_fp": True},
                    properties_checked=oracle_res.properties_checked + ["high_fp_filter"],
                )
            return oracle_res

        if self.mode == "high_fn":
            # If oracle says True, mistakenly reject with probability fn_rate
            if oracle_res.passed and self.rng.random() < self.fn_rate:
                return VerifierResult(
                    score=0.0,
                    passed=False,
                    confidence=0.25,
                    reason_code="ASYMMETRIC_FALSE_NEGATIVE",
                    metadata={"oracle_passed": True, "simulated_fn": True},
                    properties_checked=oracle_res.properties_checked + ["high_fn_filter"],
                )
            return oracle_res

        return oracle_res
