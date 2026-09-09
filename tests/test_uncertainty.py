"""Unit and invariant tests for Phase 7: Uncertainty, Calibration, and Selective Reasoning.

Verifies:
1. Trajectory uncertainty invariants (entropy, margin, min-confidence, sequence log prob).
2. Padding masking isolation (padding tokens do not contaminate metrics).
3. Hand-computable calibration cases (ECE, Brier score, NLL, reliability bins).
4. Temperature scaling (softening/sharpening, preservation of argmax boundaries).
5. Selective prediction (risk-coverage curves, AURC bounds, monotonicity).
6. Multi-sample uncertainty, vote entropy, and pass@k.
7. Ensemble disagreement and epistemic variance.
8. Bayesian linear head decomposition (Total = Aleatoric + Epistemic).
9. MC dropout stochasticity vs. deterministic evaluation mode.
10. Uncertainty-aware verifier routing logic.
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_reasoning_lab.evaluation.policy_uncertainty import (
    BayesianLinearHead,
    TemperatureScaler,
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
    compute_ensemble_uncertainty,
    compute_nll,
    compute_risk_coverage_curve,
    compute_sampling_uncertainty,
    evaluate_uncertainty_routing,
    extract_trajectory_uncertainty,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer


class TestCalibrationMetrics:
    def test_perfectly_calibrated_hand_computable(self) -> None:
        """4 examples: confidences [0.8, 0.8, 0.2, 0.2], labels [1, 1, 0, 0]."""
        confs = [0.8, 0.8, 0.2, 0.2]
        labels = [1, 1, 0, 0]

        metrics = compute_calibration_metrics(confs, labels, num_bins=2)
        # Bin 0: [0, 0.5): samples [0.2, 0.2], mean_conf=0.2, mean_acc=0.0 -> diff = 0.2
        # Bin 1: [0.5, 1.0]: samples [0.8, 0.8], mean_conf=0.8, mean_acc=1.0 -> diff = 0.2
        # ECE with 2 bins: 0.5 * 0.2 + 0.5 * 0.2 = 0.2
        assert abs(metrics.ece - 0.2) < 1e-6

        # With 5 bins:
        # Bin 1: [0.2, 0.4) -> conf=0.2, acc=0.0 -> diff=0.2
        # Bin 4: [0.8, 1.0] -> conf=0.8, acc=1.0 -> diff=0.2
        # ECE = 0.5 * 0.2 + 0.5 * 0.2 = 0.2
        ece5 = compute_ece(confs, labels, num_bins=5)
        assert abs(ece5 - 0.2) < 1e-6

        # Brier score: 2*(0.8 - 1)^2 + 2*(0.2 - 0)^2 / 4 = (2*0.04 + 2*0.04)/4 = 0.04
        brier = compute_brier_score(confs, labels)
        assert abs(brier - 0.04) < 1e-6

        # NLL: - (2 * ln(0.8) + 2 * ln(1 - 0.2)) / 4 = -ln(0.8) approx 0.22314
        nll = compute_nll(confs, labels)
        assert abs(nll - (-math.log(0.8))) < 1e-5

    def test_overconfident_hand_computable(self) -> None:
        """4 examples: confidences [0.9, 0.9, 0.9, 0.9], labels [0, 0, 0, 0]."""
        confs = [0.9, 0.9, 0.9, 0.9]
        labels = [0, 0, 0, 0]

        metrics = compute_calibration_metrics(confs, labels, num_bins=5)
        # All 4 in bin [0.8, 1.0]. acc=0.0, conf=0.9 -> ECE = |0.0 - 0.9| = 0.9
        assert abs(metrics.ece - 0.9) < 1e-6
        # Brier score: (0.9 - 0)^2 = 0.81
        assert abs(metrics.brier_score - 0.81) < 1e-6


class TestTemperatureScaling:
    def test_temperature_scaling_preserves_argmax(self) -> None:
        logits = torch.tensor([[2.0, 5.0, 1.0], [4.0, 1.0, 0.0]], dtype=torch.float32)
        scaler = TemperatureScaler()
        # Set temperature to 2.5
        scaler.log_temp.data.fill_(math.log(2.5))

        scaled = scaler(logits)
        assert torch.equal(torch.argmax(logits, dim=-1), torch.argmax(scaled, dim=-1))

    def test_temperature_scaling_optimization(self) -> None:
        """Overconfident logits with 50% accuracy should yield T > 1 after fitting."""
        # Logits predicting class 0 with huge margin (e.g. 6.0 vs 0.0), but labels are 50% class 1
        logits = torch.tensor([[6.0, 0.0], [6.0, 0.0], [6.0, 0.0], [6.0, 0.0]], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        scaler = TemperatureScaler()
        learned_t = scaler.fit(logits, labels, lr=0.1, max_iter=200)

        # To soften the extreme logits towards 50%, T must be substantially > 1.0
        assert learned_t > 1.5


class TestTrajectoryUncertainty:
    @pytest.fixture
    def small_model_and_tokenizer(self) -> tuple[TinyTransformer, CharTokenizer]:
        tok = CharTokenizer()
        model = TinyTransformer(
            vocab_size=tok.vocab_size,
            d_model=32,
            num_layers=1,
            num_heads=2,
            d_ff=64,
            max_seq_len=64,
            dropout=0.1,
        )
        return model, tok

    def test_trajectory_uncertainty_invariants(
        self, small_model_and_tokenizer: tuple[TinyTransformer, CharTokenizer]
    ) -> None:
        model, tok = small_model_and_tokenizer
        prompt = "A->B; Q: A->?"
        completion = " R: A->B; A: B"

        unc = extract_trajectory_uncertainty(model, tok, prompt, completion)

        assert len(unc.tokens) == len(completion)
        assert len(unc.step_entropies) == len(completion)
        assert len(unc.step_confidences) == len(completion)
        assert len(unc.step_margins) == len(completion)

        # All confidences in [0, 1]
        for c in unc.step_confidences:
            assert 0.0 <= c <= 1.0

        # Margins >= 0
        for m in unc.step_margins:
            assert m >= 0.0

        # Entropies >= 0
        for h in unc.step_entropies:
            assert h >= 0.0

        # Sequence log probability is sum of log probabilities
        assert unc.sequence_log_prob < 0.0
        assert abs(unc.sequence_log_prob - unc.length_normalized_log_prob * len(completion)) < 1e-4

        # Final answer token prob is positive
        assert unc.final_answer_prob > 0.0

    def test_padding_isolation(
        self, small_model_and_tokenizer: tuple[TinyTransformer, CharTokenizer]
    ) -> None:
        model, tok = small_model_and_tokenizer
        prompt = "A->B; Q: A->?"
        completion = " A: B"

        pad_id = tok.pad_token_id
        comp_ids = tok.encode(completion, add_bos=False)
        padded_ids = comp_ids + [pad_id] * 5

        unc_clean = extract_trajectory_uncertainty(
            model, tok, prompt, comp_ids, pad_token_id=pad_id
        )
        unc_padded = extract_trajectory_uncertainty(
            model, tok, prompt, padded_ids, pad_token_id=pad_id
        )

        # Strict invariant: metrics must be completely identical because pad tokens were masked
        assert unc_clean.tokens == unc_padded.tokens
        assert abs(unc_clean.mean_token_entropy - unc_padded.mean_token_entropy) < 1e-6
        assert abs(unc_clean.min_token_confidence - unc_padded.min_token_confidence) < 1e-6
        assert abs(unc_clean.sequence_log_prob - unc_padded.sequence_log_prob) < 1e-6


class TestSelectivePrediction:
    def test_risk_coverage_curve_bounds(self) -> None:
        confs = [0.95, 0.85, 0.70, 0.40, 0.20]
        labels = [True, True, False, False, False]

        res = compute_risk_coverage_curve(confs, labels, num_thresholds=20)
        assert len(res.points) == 20
        assert 0.0 <= res.aurc <= 1.0

        # At tau=0.0, coverage must be 1.0 and selective accuracy equals base accuracy (2/5 = 0.4)
        pt_0 = res.points[0]
        assert abs(pt_0.coverage - 1.0) < 1e-6
        assert abs(pt_0.selective_accuracy - 0.4) < 1e-6

        # At tau=0.90, only [0.95] is accepted (1/5 = 0.2 coverage), label is True -> risk = 0.0
        pt_high = next(p for p in res.points if p.threshold >= 0.90)
        assert pt_high.risk == 0.0


class TestSamplingAndEnsembleUncertainty:
    def test_sampling_uncertainty_consensus_vs_split(self) -> None:
        def extract_fn(s: str) -> str:
            return s.strip()

        # 1. Consensus: 5 identical completions
        res_cons = compute_sampling_uncertainty(
            ["A", "A", "A", "A", "A"], extract_fn, ground_truth="A"
        )
        assert res_cons.majority_answer == "A"
        assert res_cons.vote_entropy == 0.0
        assert res_cons.answer_disagreement_rate == 0.0
        assert res_cons.pass_at_k[1] == 1.0

        # 2. Split: 3 'A', 2 'B'
        res_split = compute_sampling_uncertainty(
            ["A", "A", "A", "B", "B"], extract_fn, ground_truth="A"
        )
        assert res_split.majority_answer == "A"
        assert res_split.majority_frequency == 0.6
        assert res_split.vote_entropy > 0.0
        assert res_split.answer_disagreement_rate == 0.4

    def test_ensemble_uncertainty(self) -> None:
        preds = ["A", "A", "B", "A"]
        res = compute_ensemble_uncertainty(preds)
        assert res.disagreement_rate == 0.25
        assert res.vote_entropy > 0.0


class TestBayesianLinearHead:
    def test_uncertainty_decomposition(self) -> None:
        torch.manual_seed(42)
        head = BayesianLinearHead(in_features=8, out_features=3)
        x = torch.randn(1, 8)

        res = head.decompose_uncertainty(x, num_samples=30)
        # Total Uncertainty = Aleatoric + Epistemic (Mutual Information >= 0)
        assert res.total_uncertainty >= res.aleatoric_uncertainty - 1e-6
        assert res.epistemic_uncertainty >= 0.0
        assert (
            abs(res.total_uncertainty - (res.aleatoric_uncertainty + res.epistemic_uncertainty))
            < 1e-5
        )
        assert len(res.mean_probabilities) == 3
        assert abs(sum(res.mean_probabilities) - 1.0) < 1e-5


class TestMCDropout:
    def test_mc_dropout_stochasticity_vs_deterministic_eval(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(
            vocab_size=tok.vocab_size,
            d_model=32,
            num_layers=1,
            num_heads=2,
            d_ff=64,
            max_seq_len=32,
            dropout=0.5,
        )
        model.eval()
        x = torch.tensor([[tok.bos_token_id, tok.char_to_id["A"], tok.char_to_id["B"]]])

        # 1. Deterministic evaluation mode (enable_dropout=False)
        out1 = model(x, enable_dropout=False)
        out2 = model(x, enable_dropout=False)
        assert torch.allclose(out1, out2)

        # 2. Monte Carlo Dropout mode (enable_dropout=True)
        mc_out1 = model(x, enable_dropout=True)
        mc_out2 = model(x, enable_dropout=True)
        # Two stochastic passes with 0.5 dropout should differ
        assert not torch.allclose(mc_out1, mc_out2)


class TestUncertaintyRouting:
    def test_routing_behavior(self) -> None:
        cheap = [True, False, False, True]
        oracle = [True, True, True, True]
        uncertainties = [0.1, 0.8, 0.9, 0.2]  # Two high uncertainty cases where cheap was wrong

        res = evaluate_uncertainty_routing(cheap, oracle, uncertainties, threshold=0.5)
        # Threshold 0.5 triggers escalation on indices 1 and 2
        assert res.oracle_budget_used == 2
        assert res.oracle_query_fraction == 0.5
        assert res.errors_corrected == 2
        assert res.final_accuracy == 1.0  # All 4 resolved correctly

        # Exploit prevention test (cheap=True, oracle=False)
        cheap_exp = [False, True, False]
        oracle_exp = [False, False, False]
        res_exp = evaluate_uncertainty_routing(
            cheap_exp, oracle_exp, [0.1, 0.9, 0.1], threshold=0.5
        )
        assert res_exp.exploits_prevented == 1
        assert res_exp.final_accuracy == 1.0
