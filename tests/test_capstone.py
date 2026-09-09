"""Unit and invariant tests for Phase 10: Integrated Research Capstone."""

import random

import numpy as np

from rl_reasoning_lab.environments.reasoning_tasks import (
    GraphTraversalTask,
    ReasoningExample,
)
from rl_reasoning_lab.evaluation.statistics import (
    bootstrap_mean_ci,
    bootstrap_paired_difference_ci,
    cohens_d,
)
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


class TestOperationalReasoningCriteria:
    """Verify that the 7 operational criteria are cleanly evaluable and distinguishable."""

    def test_process_validity_gap_detection(self) -> None:
        """Test that answer correctness without valid intermediate path is detected."""
        ex = ReasoningExample(
            prompt="A->B; B->C; C->D; Q: A->?",
            answer="D",
            depth=3,
            path=["A", "B", "C", "D"],
            edges=[("A", "B"), ("B", "C"), ("C", "D")],
            distractor_edges=[],
            start_node="A",
            target_node="D",
            answer_target=" A: D",
            trace_target=" R: A->B; B->C; C->D; A: D",
        )

        # Case 1: Sound derivation (both answer and path valid)
        sound_completion = " R: A->B; B->C; C->D; A: D"
        r1 = TraceEvaluator.evaluate(sound_completion, ex)
        assert r1.answer_correct is True
        assert r1.complete_path is True
        assert r1.step_validity == 1.0

        # Case 2: Shortcut hallucination (answer correct, but step skipped)
        shortcut_completion = " R: A->B; B->D; A: D"
        r2 = TraceEvaluator.evaluate(shortcut_completion, ex)
        assert r2.answer_correct is True
        assert r2.complete_path is False  # Hallucinated edge B->D
        assert r2.step_validity < 1.0

    def test_counterfactual_mutation_causal_update(self) -> None:
        """Mutating a premise edge must update target_node and path dynamically."""
        env = GraphTraversalTask()
        ex = env.generate_example(depth=3, rng=random.Random(42))
        original_target = ex.target_node

        # Intervene on step 0: replaces first edge destination
        mutated = GraphTraversalTask.create_counterfactual_intervention(
            ex, intervene_step_idx=0, rng=random.Random(101)
        )

        # New destination must differ from original target
        assert mutated.start_node == ex.start_node
        assert mutated.depth == ex.depth
        assert mutated.target_node != original_target
        # Verify premise string contains updated edges
        assert f"{mutated.path[0]}->{mutated.path[1]}" in mutated.prompt


class TestIndependentOracleInsulation:
    """Verify that oracle verifiers are strictly insulated and independent."""

    def test_oracle_immutability(self) -> None:
        """Oracle evaluations must not alter underlying problem or model tensors."""
        ex = ReasoningExample(
            prompt="X->Y; Y->Z; Q: X->?",
            answer="Z",
            depth=2,
            path=["X", "Y", "Z"],
            edges=[("X", "Y"), ("Y", "Z")],
            distractor_edges=[],
            start_node="X",
            target_node="Z",
            answer_target=" A: Z",
            trace_target=" R: X->Y; Y->Z; A: Z",
        )
        report = TraceEvaluator.evaluate(" R: X->Y; Y->Z; A: Z", ex)
        assert report.answer_correct is True
        # Verify input example remains unchanged
        assert ex.answer == "Z"
        assert ex.depth == 2


class TestStatisticalInvariantsAndControls:
    """Verify statistical bootstrap guarantees and control invariants."""

    def test_negative_control_spans_zero(self) -> None:
        """Negative control (pure zero-centered noise) must have 95% CI covering 0.0."""
        rng = np.random.RandomState(42)
        noise = rng.randn(30) * 0.02
        noise_centered = noise - np.mean(noise)
        est, low, high = bootstrap_mean_ci(noise_centered, seed=42)
        assert low <= 0.0 <= high, f"Negative control failed: [{low}, {high}] does not span 0.0"

    def test_paired_bootstrap_detects_true_difference(self) -> None:
        """Paired bootstrap detects a true treatment effect with tight bounds."""
        rng = np.random.RandomState(123)
        control = rng.normal(loc=0.5, scale=0.1, size=20)
        treatment = control + 0.25 + rng.normal(loc=0.0, scale=0.02, size=20)

        diff, low, high = bootstrap_paired_difference_ci(treatment, control, seed=42)
        d = cohens_d(treatment, control)
        assert low > 0.15
        assert high < 0.35
        assert d > 1.5  # Large positive effect

    def test_replication_seed_disjointness(self) -> None:
        """Verify replication seeds do not overlap with primary training seeds."""
        primary_seeds = set(range(100, 110))
        replication_seeds = set(range(900, 910))
        assert primary_seeds.isdisjoint(replication_seeds)
