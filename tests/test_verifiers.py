"""Unit and invariant tests for Phase 6: Reward and Verifier Design.

Verifies:
1. Strict oracle/verifier separation and standard result contracts.
2. Deliberate exploitability of weak proxies (substring, last token, keywords, partial path).
3. Random noise vs. systematic shortcut error profiles.
4. Invariant gating blocking ungrounded exploits.
5. Sequential pipeline early halting.
6. Ensemble consensus, disagreement uncertainty, and abstention.
7. Diagnostic confusion matrices, calibration curves, and oracle budgeting.
8. Parameter freezing in learned verifiers.
"""

from __future__ import annotations

import random

import pytest

from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.evaluation.verifier_metrics import (
    compute_calibration_curve,
    evaluate_verifier_diagnostics,
    simulate_oracle_budget_allocation,
)
from rl_reasoning_lab.verifiers.base import VerifierResult
from rl_reasoning_lab.verifiers.composite_verifiers import (
    InvariantGatedVerifier,
    SequentialPipelineVerifier,
    VerifierEnsemble,
)
from rl_reasoning_lab.verifiers.learned_verifier import ToyLearnedVerifier
from rl_reasoning_lab.verifiers.oracle_verifier import OracleVerifier
from rl_reasoning_lab.verifiers.proxy_verifiers import (
    FormatVerifier,
    KeywordVerifier,
    LastTokenVerifier,
    PartialPathVerifier,
    SubstringAnswerVerifier,
)


@pytest.fixture
def sample_task() -> tuple[GraphTraversalTask, ReasoningExample]:
    task = GraphTraversalTask()
    rng = random.Random(42)
    # Generate a depth-2 example: e.g., A -> B -> C; start=A, target=C
    example = task.generate_example(depth=2, num_distractors=1, rng=rng)
    return task, example


class TestBaseAndOracleVerifiers:
    def test_base_result_immutability_and_fields(self) -> None:
        res = VerifierResult(
            score=1.0,
            passed=True,
            confidence=0.95,
            reason_code="VALID",
            metadata={"step": 2},
            properties_checked=["path"],
            abstain=False,
        )
        assert res.score == 1.0
        assert res.passed is True
        assert res.reason_code == "VALID"
        with pytest.raises(AttributeError):
            # Dataclass is frozen
            res.score = 0.5  # type: ignore[misc]

    def test_oracle_verifier_accuracy(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        oracle = OracleVerifier(strict_path=True)

        # Ground truth completion
        valid_comp = ex.trace_target
        res_valid = oracle.verify(valid_comp, ex)
        assert res_valid.passed is True
        assert res_valid.score == 1.0
        assert res_valid.reason_code == "VALID_COMPLETE_PATH"

        # Answer correct but hallucinated path
        hallucinated_comp = "R: A->X; X->Y; A: " + ex.answer
        res_hallucinated = oracle.verify(hallucinated_comp, ex)
        assert res_hallucinated.passed is False
        assert res_hallucinated.score == 0.0


class TestProxyVerifierVulnerabilities:
    def test_substring_verifier_exploit(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        v = SubstringAnswerVerifier()

        # Exploit: junk text that mentions the target letter
        exploit_comp = (
            f"Random rambling text with answer: wrong, ignore above, token is {ex.answer}"
        )
        res = v.verify(exploit_comp, ex)
        assert res.passed is True
        assert res.score == 1.0

        # Oracle rejects it completely
        oracle = OracleVerifier()
        assert oracle.verify(exploit_comp, ex).passed is False

    def test_last_token_verifier_exploit(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        v = LastTokenVerifier()

        # Valid answer syntax but completely fake intermediate steps
        exploit_comp = f"R: {ex.start_node}->FAKE_NODE; FAKE_NODE->{ex.answer}; A: {ex.answer}"
        res = v.verify(exploit_comp, ex)
        assert res.passed is True

        # Oracle rejects it
        oracle = OracleVerifier()
        assert oracle.verify(exploit_comp, ex).passed is False

    def test_keyword_verifier_exploit(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        v = KeywordVerifier()

        # Exploit: dump all characters from alphabet / nodes in any arbitrary order
        exploit_comp = "R: " + " ".join(ex.path) + f" ; A: {ex.answer}"
        res = v.verify(exploit_comp, ex)
        assert res.passed is True

    def test_partial_path_verifier_exploit(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        task = GraphTraversalTask()
        # Create a task with at least 2 edges
        ex = task.generate_example(depth=3, num_distractors=1, rng=random.Random(123))
        v = PartialPathVerifier()

        # Two valid edges in graph, but disconnected (not a continuous chain from start to target)
        e1 = ex.edges[0]
        e2 = ex.edges[-1]
        exploit_comp = f"R: {e2[0]}->{e2[1]}; {e1[0]}->{e1[1]}; A: {ex.answer}"
        res = v.verify(exploit_comp, ex)
        # All edges exist individually, so partial path verifier mistakenly accepts
        assert res.passed is True

        # Oracle rejects because continuity is broken
        oracle = OracleVerifier()
        assert oracle.verify(exploit_comp, ex).passed is False


class TestInvariantAndSequentialVerifiers:
    def test_invariant_gated_verifier_catches_exploits(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        # Wrap weak last-token verifier inside invariant gate
        weak_v = LastTokenVerifier()
        gated_v = InvariantGatedVerifier(soft_verifier=weak_v)

        # 1. Start node mismatch exploit
        comp_wrong_start = f"R: WRONG_START->B; B->{ex.answer}; A: {ex.answer}"
        res1 = gated_v.verify(comp_wrong_start, ex)
        assert res1.passed is False
        assert "INVARIANT_VIOLATION_START_NODE" in res1.reason_code

        # 2. Disconnected step exploit
        comp_disconnected = f"R: {ex.start_node}->B; X->{ex.answer}; A: {ex.answer}"
        res2 = gated_v.verify(comp_disconnected, ex)
        assert res2.passed is False
        assert "INVARIANT_VIOLATION_DISCONNECTED_STEP" in res2.reason_code

        # 3. Destination mismatch exploit
        comp_dest_mismatch = f"R: {ex.start_node}->B; B->OTHER; A: {ex.answer}"
        res3 = gated_v.verify(comp_dest_mismatch, ex)
        assert res3.passed is False
        assert "INVARIANT_VIOLATION_DESTINATION_MISMATCH" in res3.reason_code

        # 4. Valid completion passes
        res_valid = gated_v.verify(ex.trace_target, ex)
        assert res_valid.passed is True

    def test_sequential_pipeline_short_circuiting(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        pipeline = SequentialPipelineVerifier()

        # Syntax failure stops at stage 1
        res1 = pipeline.verify("malformed string without R or A", ex)
        assert res1.passed is False
        assert res1.metadata["stopped_at"] == 1

        # Answer mismatch stops at stage 2
        res2 = pipeline.verify(f"R: {ex.start_node}->B; A: WRONG_ANS", ex)
        assert res2.passed is False
        assert res2.metadata["stopped_at"] == 2

        # Valid completion passes all stages
        res_valid = pipeline.verify(ex.trace_target, ex)
        assert res_valid.passed is True
        assert res_valid.metadata["stopped_at"] is None


class TestEnsembleAndAbstention:
    def test_ensemble_aggregation_modes(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        v_oracle = OracleVerifier()
        v_substring = SubstringAnswerVerifier()
        v_format = FormatVerifier()

        # Tricky completion: valid format and substring, but invalid path
        tricky_comp = f"R: {ex.start_node}->FAKE; A: {ex.answer}"

        ens_majority = VerifierEnsemble(
            verifiers=[v_oracle, v_substring, v_format],
            aggregation="majority_vote",
        )
        res_maj = ens_majority.verify(tricky_comp, ex)
        # 2 out of 3 pass (substring + format), so majority passes
        assert res_maj.passed is True

        ens_unanimity = VerifierEnsemble(
            verifiers=[v_oracle, v_substring, v_format],
            aggregation="unanimity",
        )
        res_unan = ens_unanimity.verify(tricky_comp, ex)
        # Oracle rejects, so unanimity fails
        assert res_unan.passed is False

    def test_ensemble_uncertainty_and_abstention(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        v_oracle = OracleVerifier()
        v_format = FormatVerifier()

        # 2 verifiers with threshold 0.1 (any disagreement yields disagreement = 0.5 > 0.1)
        ens = VerifierEnsemble(
            verifiers=[v_oracle, v_format],
            abstention_threshold=0.2,
        )

        # Disagreement case
        comp = f"R: FAKE->STEP; A: {ex.answer}"
        res = ens.verify(comp, ex)
        assert res.abstain is True
        assert res.reason_code == "ENSEMBLE_ABSTAIN_HIGH_UNCERTAINTY"


class TestVerifierMetricsAndBudget:
    def test_confusion_matrix_metrics(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        oracle = OracleVerifier()
        weak_v = SubstringAnswerVerifier()

        completions = [
            ex.trace_target,  # TP: true valid, weak says valid
            f"Junk with answer {ex.answer}",  # FP: true invalid, weak says valid
            "Completely wrong text",  # TN: true invalid, weak says invalid
        ]
        examples = [ex, ex, ex]

        summary = evaluate_verifier_diagnostics(weak_v, oracle, completions, examples)
        assert summary.tp == 1
        assert summary.fp == 1
        assert summary.tn == 1
        assert summary.fn == 0
        assert summary.precision == 0.5
        assert summary.recall == 1.0

    def test_calibration_curve_and_ece(self) -> None:
        confs = [0.9, 0.8, 0.3, 0.1]
        labels = [True, False, False, False]

        cal = compute_calibration_curve(confs, labels, num_bins=2)
        assert len(cal.bin_counts) == 2
        assert 0.0 <= cal.ece <= 1.0

    def test_oracle_budget_disagreement_allocation(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        oracle = OracleVerifier()
        v_format = FormatVerifier()
        ens = VerifierEnsemble(verifiers=[oracle, v_format])

        dataset = [
            (ex.trace_target, ex),  # Agreement (both True) -> disagreement = 0.0
            (
                f"R: FAKE->STEP; A: {ex.answer}",
                ex,
            ),  # Disagreement (oracle False, format True) -> 0.5
        ]

        # Budget of 1 with disagreement allocation must pick the second item
        res = simulate_oracle_budget_allocation(
            verifier_results=dataset,
            ensemble_verifier=ens,
            oracle_verifier=oracle,
            budget=1,
            strategy="disagreement",
        )
        assert res.oracle_queries_used == 1
        assert res.final_decision_accuracy == 1.0


class TestToyLearnedVerifier:
    def test_learned_verifier_freezing_and_forward(
        self, sample_task: tuple[GraphTraversalTask, ReasoningExample]
    ) -> None:
        _, ex = sample_task
        learned = ToyLearnedVerifier(frozen=True)

        for p in learned.model.parameters():
            assert p.requires_grad is False

        res = learned.verify(ex.trace_target, ex)
        assert isinstance(res.score, float)
        assert 0.0 <= res.score <= 1.0
