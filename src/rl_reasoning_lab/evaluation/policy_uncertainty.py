"""Policy uncertainty, calibration metrics, selective prediction, and epistemic decomposition.

Rigorously separates:
- prediction confidence
- policy entropy
- sampling variability
- verifier confidence
- ensemble disagreement
- epistemic (parameter) uncertainty
- aleatoric task uncertainty
- calibration (ECE, Brier, NLL)
- selective risk and coverage
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer


@dataclass(frozen=True)
class TrajectoryUncertainty:
    """Fine-grained token-level and sequence-level policy uncertainty measurements."""

    tokens: list[str]
    token_ids: list[int]
    final_answer_prob: float
    mean_token_entropy: float
    min_token_confidence: float
    sequence_log_prob: float
    length_normalized_log_prob: float
    mean_margin: float
    step_entropies: list[float]
    step_confidences: list[float]
    step_margins: list[float]


@dataclass(frozen=True)
class CalibrationMetrics:
    """Comprehensive calibration metrics including ECE, Brier score, and NLL."""

    ece: float
    brier_score: float
    nll: float
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


@dataclass(frozen=True)
class RiskCoveragePoint:
    """A single evaluation point along the risk-coverage curve."""

    threshold: float
    coverage: float
    risk: float
    selective_accuracy: float


@dataclass(frozen=True)
class SelectivePredictionResult:
    """Results from evaluating selective prediction / abstention across thresholds."""

    points: list[RiskCoveragePoint]
    aurc: float  # Area Under Risk-Coverage curve
    risk_at_coverage: dict[float, float]  # e.g., 0.9 -> risk, 0.8 -> risk, 0.5 -> risk


@dataclass(frozen=True)
class SamplingUncertaintyResult:
    """Uncertainty derived from sampling multiple completions for a prompt."""

    completions: list[str]
    extracted_answers: list[str]
    majority_answer: str
    majority_frequency: float
    vote_entropy: float
    answer_disagreement_rate: float
    pass_at_k: dict[int, float]


@dataclass(frozen=True)
class EnsembleUncertaintyResult:
    """Uncertainty derived from an ensemble of independently trained policy models."""

    mean_confidence: float
    variance_confidence: float
    vote_entropy: float
    disagreement_rate: float
    predictions: list[str]


@dataclass(frozen=True)
class BayesianDecompositionResult:
    """Decomposition of predictive uncertainty into aleatoric and epistemic components."""

    total_uncertainty: float  # Entropy of the ensemble/expected distribution
    aleatoric_uncertainty: float  # Expected entropy of individual models
    epistemic_uncertainty: float  # Mutual information (Total - Aleatoric)
    mean_probabilities: list[float]


def extract_trajectory_uncertainty(
    model: TinyTransformer,
    tokenizer: CharTokenizer,
    prompt: str | list[int],
    completion: str | list[int],
    pad_token_id: int | None = None,
) -> TrajectoryUncertainty:
    """Extract fine-grained token-level uncertainty and margins for a generated completion.

    Strict invariant:
    Only completion tokens are measured. Prompt tokens and any padding tokens are strictly masked
    so that padding never contaminates entropy, margin, or min-confidence statistics.
    """
    model.eval()
    if isinstance(prompt, str):
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
    else:
        prompt_ids = list(prompt)

    if isinstance(completion, str):
        comp_ids = tokenizer.encode(completion, add_bos=False)
        comp_text = completion
    else:
        comp_ids = list(completion)
        comp_text = tokenizer.decode(comp_ids, skip_special=False)

    if not comp_ids:
        # Trivial empty completion
        return TrajectoryUncertainty(
            tokens=[],
            token_ids=[],
            final_answer_prob=0.0,
            mean_token_entropy=0.0,
            min_token_confidence=0.0,
            sequence_log_prob=0.0,
            length_normalized_log_prob=0.0,
            mean_margin=0.0,
            step_entropies=[],
            step_confidences=[],
            step_margins=[],
        )

    full_ids = prompt_ids + comp_ids
    input_tensor = torch.tensor([full_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)[0]  # (seq_len, vocab_size)
        probs = torch.softmax(logits, dim=-1)

    # In autoregressive LM:
    # Logits at position i predict token at position i+1.
    # The first completion token (comp_ids[0]) is predicted by logits at position len(prompt_ids) - 1.
    # The last completion token (comp_ids[-1]) is predicted by logits at position len(full_ids) - 2.
    start_pos = len(prompt_ids) - 1

    tokens: list[str] = []
    token_ids: list[int] = []
    step_entropies: list[float] = []
    step_confidences: list[float] = []
    step_margins: list[float] = []
    log_probs: list[float] = []

    for idx, target_tok in enumerate(comp_ids):
        # Exclude padding if specified
        if pad_token_id is not None and target_tok == pad_token_id:
            continue

        pred_probs = probs[start_pos + idx]

        dist = Categorical(probs=pred_probs)
        h = float(dist.entropy().item())
        p_target = float(pred_probs[target_tok].item())
        log_p = float(torch.log(pred_probs[target_tok] + 1e-12).item())

        # Top-1 and top-2 margin
        sorted_probs, _ = torch.sort(pred_probs, descending=True)
        top1 = float(sorted_probs[0].item())
        top2 = float(sorted_probs[1].item()) if len(sorted_probs) > 1 else 0.0
        margin = top1 - top2

        tokens.append(tokenizer.id_to_char.get(target_tok, f"<{target_tok}>"))
        token_ids.append(target_tok)
        step_entropies.append(h)
        step_confidences.append(p_target)
        step_margins.append(margin)
        log_probs.append(log_p)

    if not step_confidences:
        return TrajectoryUncertainty(
            tokens=[],
            token_ids=[],
            final_answer_prob=0.0,
            mean_token_entropy=0.0,
            min_token_confidence=0.0,
            sequence_log_prob=0.0,
            length_normalized_log_prob=0.0,
            mean_margin=0.0,
            step_entropies=[],
            step_confidences=[],
            step_margins=[],
        )

    # Locate final answer token if present (e.g. after 'A: ')
    final_ans_prob = step_confidences[-1]
    if "A: " in comp_text:
        ans_start = comp_text.rfind("A: ") + 3
        if ans_start < len(comp_text):
            ans_char = comp_text[ans_start]
            for t_char, t_prob in zip(reversed(tokens), reversed(step_confidences), strict=False):
                if t_char == ans_char:
                    final_ans_prob = t_prob
                    break

    seq_log_prob = float(sum(log_probs))
    n_tokens = len(step_confidences)

    return TrajectoryUncertainty(
        tokens=tokens,
        token_ids=token_ids,
        final_answer_prob=final_ans_prob,
        mean_token_entropy=float(np.mean(step_entropies)),
        min_token_confidence=float(min(step_confidences)),
        sequence_log_prob=seq_log_prob,
        length_normalized_log_prob=seq_log_prob / n_tokens,
        mean_margin=float(np.mean(step_margins)),
        step_entropies=step_entropies,
        step_confidences=step_confidences,
        step_margins=step_margins,
    )


def compute_ece(
    confidences: list[float] | np.ndarray,
    labels: list[bool] | list[int] | np.ndarray,
    num_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE)."""
    return compute_calibration_metrics(confidences, labels, num_bins=num_bins).ece


def compute_brier_score(
    confidences: list[float] | np.ndarray,
    labels: list[bool] | list[int] | np.ndarray,
) -> float:
    """Computes Brier score: Mean squared difference between confidence and binary outcome."""
    if len(confidences) != len(labels):
        raise ValueError("Length mismatch between confidences and labels")
    if not len(confidences):
        return 0.0
    c = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    y = np.array(labels, dtype=float)
    return float(np.mean((c - y) ** 2))


def compute_nll(
    confidences: list[float] | np.ndarray,
    labels: list[bool] | list[int] | np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Computes binary cross-entropy / Negative Log-Likelihood (NLL)."""
    if len(confidences) != len(labels):
        raise ValueError("Length mismatch between confidences and labels")
    if not len(confidences):
        return 0.0
    c = np.clip(np.array(confidences, dtype=float), eps, 1.0 - eps)
    y = np.array(labels, dtype=float)
    return float(-np.mean(y * np.log(c) + (1.0 - y) * np.log(1.0 - c)))


def compute_calibration_metrics(
    confidences: list[float] | np.ndarray,
    labels: list[bool] | list[int] | np.ndarray,
    num_bins: int = 10,
) -> CalibrationMetrics:
    """Calculates ECE, Brier score, NLL, and reliability bin statistics."""
    if len(confidences) != len(labels):
        raise ValueError("Length mismatch between confidences and labels")
    if not len(confidences):
        return CalibrationMetrics(0.0, 0.0, 0.0, [], [], [])

    c = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    y = np.array(labels, dtype=float)
    n = len(c)

    brier = compute_brier_score(c, y)
    nll = compute_nll(c, y)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []
    bin_counts: list[int] = []

    ece = 0.0
    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            mask = (c >= lower) & (c <= upper)
        else:
            mask = (c >= lower) & (c < upper)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(y[mask]))
            bin_conf = float(np.mean(c[mask]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            ece += (count / n) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((lower + upper) / 2.0)

    return CalibrationMetrics(
        ece=float(ece),
        brier_score=brier,
        nll=nll,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling: Softmax(logits / T).

    Changes confidence values and calibration without changing the argmax decision boundaries.
    """

    def __init__(self) -> None:
        super().__init__()
        # Parameterize as log_temperature for positivity guarantee
        self.log_temp = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temp).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        t = torch.exp(self.log_temp)
        return logits / t

    def fit(
        self,
        logits: torch.Tensor | np.ndarray,
        labels: torch.Tensor | list[int] | np.ndarray,
        lr: float = 0.05,
        max_iter: int = 150,
    ) -> float:
        """Fits optimal temperature on held-out calibration set via NLL minimization."""
        if not isinstance(logits, torch.Tensor):
            logits_t = torch.tensor(logits, dtype=torch.float32)
        else:
            logits_t = logits.detach().float()

        if not isinstance(labels, torch.Tensor):
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            labels_t = labels.detach().long()

        optimizer = torch.optim.Adam([self.log_temp], lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_t)
            loss = criterion(scaled_logits, labels_t)
            loss.backward()
            optimizer.step()

        return self.temperature


def compute_risk_coverage_curve(
    confidences: list[float] | np.ndarray,
    is_correct: list[bool] | list[int] | np.ndarray,
    num_thresholds: int = 50,
) -> SelectivePredictionResult:
    """Computes risk-coverage curve and Area Under Risk-Coverage (AURC).

    Coverage: P(confidence >= tau)
    Selective Risk: P(incorrect | confidence >= tau)
    Selective Accuracy: 1 - Selective Risk
    """
    if len(confidences) != len(is_correct):
        raise ValueError("Length mismatch between confidences and labels")

    confs = np.array(confidences, dtype=float)
    corrects = np.array(is_correct, dtype=bool)
    n = len(confs)

    if n == 0:
        return SelectivePredictionResult(points=[], aurc=0.0, risk_at_coverage={})

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    points: list[RiskCoveragePoint] = []

    for tau in thresholds:
        accepted = confs >= tau
        coverage = float(np.mean(accepted))

        if np.sum(accepted) > 0:
            risk = float(1.0 - np.mean(corrects[accepted]))
        else:
            risk = 0.0

        points.append(
            RiskCoveragePoint(
                threshold=float(tau),
                coverage=coverage,
                risk=risk,
                selective_accuracy=1.0 - risk,
            )
        )

    # Sort points by increasing coverage to integrate AURC
    sorted_points = sorted(points, key=lambda p: p.coverage)
    coverages = [p.coverage for p in sorted_points]
    risks = [p.risk for p in sorted_points]

    # Ensure coverage=0 and coverage=1 endpoints exist for trapezoidal integration
    cov_arr = np.array([0.0] + coverages + [1.0])
    risk_arr = np.array([risks[0]] + risks + [risks[-1]])
    aurc = float(np.trapezoid(risk_arr, cov_arr))

    # Evaluate risk at standard coverage milestones (0.95, 0.90, 0.80, 0.50)
    risk_at_cov: dict[float, float] = {}
    for target_cov in [0.95, 0.90, 0.80, 0.50]:
        best_pt = min(points, key=lambda p: abs(p.coverage - target_cov))
        risk_at_cov[target_cov] = best_pt.risk

    return SelectivePredictionResult(
        points=points,
        aurc=aurc,
        risk_at_coverage=risk_at_cov,
    )


def compute_sampling_uncertainty(
    completions: list[str],
    extract_answer_fn: Callable[[str], str],
    ground_truth: str | None = None,
) -> SamplingUncertaintyResult:
    """Computes uncertainty and self-consistency statistics over N stochastic completions."""
    if not completions:
        raise ValueError("Completions list cannot be empty")

    answers = [extract_answer_fn(c) for c in completions]
    counts = Counter(answers)
    total = len(answers)

    # Find majority answer
    maj_ans, maj_count = counts.most_common(1)[0]
    maj_freq = maj_count / total

    # Answer distribution entropy: H(P^(answer))
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        if p > 0.0:
            entropy -= p * math.log(p)

    # Disagreement rate: 1 - majority frequency
    disagreement = 1.0 - maj_freq

    # Pass@k evaluation if ground truth is supplied
    pass_at_k: dict[int, float] = {}
    if ground_truth is not None:
        for k in [1, 2, 4, 8]:
            if k <= total:
                c_correct = sum(1 for a in answers if a == ground_truth)
                if c_correct == 0:
                    pass_at_k[k] = 0.0
                elif total - c_correct < k:
                    pass_at_k[k] = 1.0
                else:
                    comb_total = math.comb(total, k)
                    comb_wrong = math.comb(total - c_correct, k)
                    pass_at_k[k] = 1.0 - (comb_wrong / comb_total)

    return SamplingUncertaintyResult(
        completions=completions,
        extracted_answers=answers,
        majority_answer=maj_ans,
        majority_frequency=maj_freq,
        vote_entropy=float(entropy),
        answer_disagreement_rate=float(disagreement),
        pass_at_k=pass_at_k,
    )


def compute_ensemble_uncertainty(
    model_predictions: list[str],
    model_confidences: list[float] | None = None,
) -> EnsembleUncertaintyResult:
    """Computes epistemic disagreement and variance across multiple independently trained models."""
    if not model_predictions:
        raise ValueError("model_predictions cannot be empty")

    total = len(model_predictions)
    counts = Counter(model_predictions)
    maj_pred, maj_count = counts.most_common(1)[0]

    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        if p > 0.0:
            entropy -= p * math.log(p)

    disagreement = 1.0 - (maj_count / total)

    if model_confidences is not None and len(model_confidences) == total:
        mean_conf = float(np.mean(model_confidences))
        var_conf = float(np.var(model_confidences))
    else:
        mean_conf = float(maj_count / total)
        var_conf = float(np.var([1.0 if p == maj_pred else 0.0 for p in model_predictions]))

    return EnsembleUncertaintyResult(
        mean_confidence=mean_conf,
        variance_confidence=var_conf,
        vote_entropy=float(entropy),
        disagreement_rate=float(disagreement),
        predictions=model_predictions,
    )


class BayesianLinearHead(nn.Module):
    """Educational toy Bayesian linear classification head.

    Uses a Gaussian weight distribution W ~ N(mu, diag(sigma^2)) to illustrate
    the decomposition of total predictive uncertainty into:
    Total Uncertainty (Predictive Entropy) = Aleatoric Uncertainty + Epistemic Uncertainty (Mutual Info).
    """

    def __init__(self, in_features: int, out_features: int, prior_var: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var

        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_var = nn.Parameter(torch.ones(out_features, in_features) * -3.0)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def sample_forward(self, x: torch.Tensor, num_samples: int = 20) -> torch.Tensor:
        """Samples weight matrices from posterior and returns stacked logits."""
        std = torch.exp(0.5 * self.weight_log_var)
        logits_samples = []

        for _ in range(num_samples):
            eps = torch.randn_like(std)
            w = self.weight_mu + eps * std
            logits = F.linear(x, w, self.bias)
            logits_samples.append(logits)

        return torch.stack(logits_samples, dim=0)  # (S, B, out)

    def decompose_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 30,
    ) -> BayesianDecompositionResult:
        """Computes Total, Aleatoric, and Epistemic uncertainty for a single input vector (1, D)."""
        logits_samples = self.sample_forward(x, num_samples=num_samples)  # (S, 1, out)
        probs_samples = torch.softmax(logits_samples, dim=-1).squeeze(1)  # (S, out)

        # 1. Expected predictive distribution p_bar = 1/S sum p^(s)
        mean_probs = torch.mean(probs_samples, dim=0)  # (out,)

        # 2. Total Uncertainty: Entropy of mean prediction H(p_bar)
        total_entropy = float(Categorical(probs=mean_probs).entropy().item())

        # 3. Aleatoric Uncertainty: Mean of individual sample entropies 1/S sum H(p^(s))
        sample_entropies = Categorical(probs=probs_samples).entropy()  # (S,)
        aleatoric = float(torch.mean(sample_entropies).item())

        # 4. Epistemic Uncertainty: Mutual Information = Total - Aleatoric
        # Information Gain between model weights and prediction
        epistemic = max(0.0, total_entropy - aleatoric)

        return BayesianDecompositionResult(
            total_uncertainty=total_entropy,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            mean_probabilities=mean_probs.tolist(),
        )


@dataclass(frozen=True)
class RoutingEvaluationResult:
    """Evaluation summary of uncertainty-aware verification routing."""

    strategy: str
    threshold: float
    oracle_budget_used: int
    oracle_query_fraction: float
    final_accuracy: float
    errors_corrected: int
    exploits_prevented: int
    unnecessary_escalations: int


def evaluate_uncertainty_routing(
    cheap_correctness: list[bool],
    oracle_correctness: list[bool],
    uncertainty_scores: list[float],
    threshold: float,
    strategy_name: str = "uncertainty_routing",
) -> RoutingEvaluationResult:
    """Simulates routing rollouts to an expensive oracle when uncertainty exceeds threshold."""
    if not (len(cheap_correctness) == len(oracle_correctness) == len(uncertainty_scores)):
        raise ValueError("All inputs must have identical length")

    n = len(cheap_correctness)
    if n == 0:
        return RoutingEvaluationResult(strategy_name, threshold, 0, 0.0, 0.0, 0, 0, 0)

    oracle_queries = 0
    final_correct = 0
    errors_corrected = 0
    exploits_prevented = 0
    unnecessary_escalations = 0

    for cheap, oracle, score in zip(
        cheap_correctness, oracle_correctness, uncertainty_scores, strict=True
    ):
        escalate = score >= threshold
        if escalate:
            oracle_queries += 1
            decision = oracle
            if cheap != oracle:
                errors_corrected += 1
            if cheap and not oracle:
                exploits_prevented += 1
            elif cheap == oracle:
                unnecessary_escalations += 1
        else:
            decision = cheap

        if decision == oracle:
            final_correct += 1

    return RoutingEvaluationResult(
        strategy=strategy_name,
        threshold=threshold,
        oracle_budget_used=oracle_queries,
        oracle_query_fraction=oracle_queries / n,
        final_accuracy=final_correct / n,
        errors_corrected=errors_corrected,
        exploits_prevented=exploits_prevented,
        unnecessary_escalations=unnecessary_escalations,
    )
