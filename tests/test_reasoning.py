"""Unit and invariant tests for Phase 5: Multi-Step Reasoning Tasks, Evaluators, and Rewards."""

from __future__ import annotations

import pytest
import torch

from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask
from rl_reasoning_lab.evaluation.reasoning_metrics import (
    evaluate_best_of_n,
    evaluate_reasoning_model,
    rejection_sampling_step,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.reasoning_rewards import ReasoningRewardFunction
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


class TestGraphTraversalTask:
    def test_depth_and_path_generation(self) -> None:
        task = GraphTraversalTask()
        for d in [1, 2, 3, 5]:
            ex = task.generate_example(depth=d, num_distractors=0)
            assert ex.depth == d
            assert len(ex.path) == d + 1
            assert len(ex.edges) == d
            assert ex.start_node == ex.path[0]
            assert ex.target_node == ex.path[-1]
            assert ex.answer == ex.target_node
            assert f"Q: {ex.start_node}->?" in ex.prompt

    def test_distractor_edges(self) -> None:
        task = GraphTraversalTask()
        ex = task.generate_example(depth=2, num_distractors=3)
        assert len(ex.distractor_edges) == 3
        # Verify distractors appear in prompt
        for u, v in ex.distractor_edges:
            assert f"{u}->{v}" in ex.prompt

    def test_counterfactual_intervention(self) -> None:
        task = GraphTraversalTask()
        orig = task.generate_example(depth=3, num_distractors=0)
        intervened = task.create_counterfactual_intervention(
            example=orig,
            intervene_step_idx=1,
            new_dst="Z",
        )
        assert intervened.depth == orig.depth
        assert intervened.start_node == orig.start_node
        # Step 1 destination must be 'Z'
        assert intervened.path[2] == "Z"
        # The new target must differ from original if 'Z' was introduced
        assert intervened.answer != orig.answer or orig.path[2] == "Z"

    def test_shortcut_dataset_separation(self) -> None:
        task = GraphTraversalTask()
        shortcut_data, cb_data = task.generate_shortcut_dataset(
            size=10, depth=2, shortcut_answer="Z"
        )
        assert all(ex.answer == "Z" for ex in shortcut_data)
        assert all(ex.answer != "Z" for ex in cb_data)


class TestTraceEvaluator:
    def test_correct_answer_and_valid_trace(self) -> None:
        task = GraphTraversalTask()
        ex = task.generate_example(depth=2)
        # Perfect completion
        completion = ex.trace_target
        report = TraceEvaluator.evaluate(completion, ex)
        assert report.answer_correct is True
        assert report.syntax_valid is True
        assert report.step_validity == 1.0
        assert report.path_consistency is True
        assert report.complete_path is True
        assert report.unnecessary_steps == 0
        assert report.missing_steps == 0

    def test_correct_answer_but_invalid_hallucinated_trace(self) -> None:
        task = GraphTraversalTask()
        ex = task.generate_example(depth=2)
        # Answer is correct ('A: <target>'), but intermediate steps use non-existent node
        bad_node = next(n for n in "XYZABC" if n not in ex.path)
        bad_trace = (
            f" R: {ex.start_node}->{bad_node}; {bad_node}->{ex.target_node}; A: {ex.target_node}"
        )
        report = TraceEvaluator.evaluate(bad_trace, ex)
        assert report.answer_correct is True
        assert report.step_validity == 0.0  # neither step is a valid premise edge
        assert report.complete_path is False

    def test_valid_trace_but_wrong_final_answer(self) -> None:
        task = GraphTraversalTask()
        ex = task.generate_example(depth=2)
        # Steps are valid geodesic steps, but answer token is wrong ('Z')
        steps_str = "; ".join(f"{u}->{v}" for u, v in ex.edges)
        completion = f" R: {steps_str}; A: Z"
        report = TraceEvaluator.evaluate(completion, ex)
        assert report.answer_correct is False
        assert report.step_validity == 1.0
        assert report.path_consistency is True
        assert report.complete_path is False  # doesn't end at target_node

    def test_broken_path_consistency(self) -> None:
        task = GraphTraversalTask()
        ex = task.generate_example(depth=3)
        # Broken chain: A->B, then jumps to C->D without B->C
        p = ex.path
        broken_completion = f" R: {p[0]}->{p[1]}; {p[2]}->{p[3]}; A: {p[3]}"
        report = TraceEvaluator.evaluate(broken_completion, ex)
        assert report.path_consistency is False
        assert report.complete_path is False

    def test_counterfactual_consistency_evaluation(self) -> None:
        task = GraphTraversalTask()
        orig = task.generate_example(depth=2)
        intervened = task.create_counterfactual_intervention(
            orig, intervene_step_idx=0, new_dst="W"
        )

        # Model adapts correctly to intervention
        assert (
            TraceEvaluator.evaluate_counterfactual_pair(
                orig.answer_target, intervened.answer_target, orig, intervened
            )
            is True
        )

        # Model repeats memorized original answer on intervened problem
        assert (
            TraceEvaluator.evaluate_counterfactual_pair(
                orig.answer_target, orig.answer_target, orig, intervened
            )
            is False
        )


class TestReasoningRewards:
    def test_outcome_reward(self) -> None:
        reward_fn = ReasoningRewardFunction(reward_mode="outcome")
        task = GraphTraversalTask()
        ex = task.generate_example(depth=2)

        r_corr = reward_fn(ex.prompt, ex.answer_target, ex)
        assert r_corr == 1.0

        wrong_target = "Z" if ex.target_node != "Z" else "Y"
        r_wrong = reward_fn(ex.prompt, f" A: {wrong_target}", ex)
        assert r_wrong == 0.0

        r_bad_syntax = reward_fn(ex.prompt, "random gibberish", ex)
        assert r_bad_syntax == -0.5

    def test_process_reward_length_normalization(self) -> None:
        reward_fn = ReasoningRewardFunction(reward_mode="process")
        task = GraphTraversalTask()
        ex = task.generate_example(depth=2)

        # 1 valid step out of 2 depth: process reward is 0.5
        p = ex.path
        one_step = f" R: {p[0]}->{p[1]}; A: {p[1]}"
        r_half = reward_fn(ex.prompt, one_step, ex)
        assert pytest.approx(r_half, abs=1e-4) == 0.5

        # 2 valid steps out of 2 depth: process reward is 1.0
        r_full = reward_fn(ex.prompt, ex.trace_target, ex)
        assert pytest.approx(r_full, abs=1e-4) == 1.0

        # Verbose trace repeating steps cannot exceed 1.0
        redundant = f" R: {p[0]}->{p[1]}; {p[1]}->{p[2]}; {p[0]}->{p[1]}; A: {p[2]}"
        r_redundant = reward_fn(ex.prompt, redundant, ex)
        assert r_redundant <= 1.0


class TestEvaluationAndSampling:
    def test_evaluate_reasoning_model_shapes(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
        task = GraphTraversalTask()
        data = task.generate_dataset(size=4, depths=[1, 2], seed=42)

        summary = evaluate_reasoning_model(model, tok, data, max_new_tokens=15)
        assert summary.total_examples == 4
        assert 0.0 <= summary.answer_accuracy <= 1.0
        assert 1 in summary.depth_breakdown
        assert 2 in summary.depth_breakdown

    def test_evaluate_best_of_n(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
        task = GraphTraversalTask()
        data = task.generate_dataset(size=2, depths=[1], seed=42)

        pass_rates = evaluate_best_of_n(model, tok, data, n_values=[1, 2], max_new_tokens=10)
        assert 1 in pass_rates
        assert 2 in pass_rates
        assert pass_rates[1] <= pass_rates[2] + 1e-6  # Pass@k is monotonically non-decreasing in k

    def test_rejection_sampling_step(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        task = GraphTraversalTask()
        data = task.generate_dataset(size=2, depths=[1], seed=42)

        accepted_count, loss = rejection_sampling_step(
            model=model,
            optimizer=opt,
            tokenizer=tok,
            examples=data,
            num_samples_per_prompt=2,
            max_new_tokens=10,
        )
        assert accepted_count >= 0
        assert loss >= 0.0
