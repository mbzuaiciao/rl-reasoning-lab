"""Evaluation metrics for verifiers, confusion analysis, calibration, and oracle budgeting.

Strictly separates verifier predictions from ground-truth oracle utility.
Calculates:
1. Confusion matrices: TP, FP, TN, FN, Precision, Recall, Specificity, F1.
2. Expected Calibration Error (ECE) and reliability binning.
3. Distribution shift tracking (VerifierAccuracy_pre vs VerifierAccuracy_post).
4. Oracle budget simulation comparing random vs. disagreement-triggered allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample
from rl_reasoning_lab.verifiers.base import BaseVerifier


@dataclass(frozen=True)
class VerifierDiagnosticSummary:
    """Comprehensive diagnostic summary of a verifier's performance against ground-truth oracle."""

    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    specificity: float
    f1: float
    accuracy: float
    reward_density: float
    mean_proxy_score: float
    mean_oracle_utility: float


@dataclass(frozen=True)
class CalibrationSummary:
    """Binned reliability analysis and Expected Calibration Error."""

    ece: float
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


def evaluate_verifier_diagnostics(
    verifier: BaseVerifier,
    oracle: BaseVerifier,
    completions: list[str],
    examples: list[ReasoningExample],
) -> VerifierDiagnosticSummary:
    """Evaluate verifier predictions against ground-truth oracle judgments on a corpus."""
    if len(completions) != len(examples):
        raise ValueError("Completions and examples lists must have identical length")

    tp = fp = tn = fn = 0
    proxy_scores: list[float] = []
    oracle_utilities: list[float] = []

    for comp, ex in zip(completions, examples, strict=False):
        v_res = verifier.verify(comp, ex)
        o_res = oracle.verify(comp, ex)

        v_pred = v_res.passed
        o_true = o_res.passed

        proxy_scores.append(v_res.score)
        oracle_utilities.append(o_res.score)

        if v_pred and o_true:
            tp += 1
        elif v_pred and not o_true:
            fp += 1
        elif not v_pred and not o_true:
            tn += 1
        else:
            fn += 1

    total = len(completions)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    reward_density = (tp + fp) / total if total > 0 else 0.0
    mean_proxy = float(np.mean(proxy_scores)) if proxy_scores else 0.0
    mean_oracle = float(np.mean(oracle_utilities)) if oracle_utilities else 0.0

    return VerifierDiagnosticSummary(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        accuracy=accuracy,
        reward_density=reward_density,
        mean_proxy_score=mean_proxy,
        mean_oracle_utility=mean_oracle,
    )


def compute_calibration_curve(
    confidences: list[float],
    ground_truth_labels: list[bool],
    num_bins: int = 5,
) -> CalibrationSummary:
    """Computes Expected Calibration Error (ECE) and reliability bin statistics."""
    if len(confidences) != len(ground_truth_labels):
        raise ValueError("Length mismatch between confidences and ground truth labels")

    if not confidences:
        return CalibrationSummary(ece=0.0, bin_accuracies=[], bin_confidences=[], bin_counts=[])

    confs = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    labels = np.array(ground_truth_labels, dtype=float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []
    bin_counts: list[int] = []

    total_samples = len(confidences)
    ece = 0.0

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            mask = (confs >= lower) & (confs <= upper)
        else:
            mask = (confs >= lower) & (confs < upper)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(labels[mask]))
            bin_conf = float(np.mean(confs[mask]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            ece += (count / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((lower + upper) / 2.0)

    return CalibrationSummary(
        ece=float(ece),
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


@dataclass(frozen=True)
class BudgetSimulationResult:
    """Results from simulating an oracle query budget."""

    strategy: str
    oracle_budget: int
    oracle_queries_used: int
    final_decision_accuracy: float
    false_positives_retained: int


def simulate_oracle_budget_allocation(
    verifier_results: list[tuple[str, ReasoningExample]],
    ensemble_verifier: BaseVerifier,
    oracle_verifier: BaseVerifier,
    budget: int,
    strategy: str = "disagreement",  # "random" or "disagreement"
    seed: int = 42,
) -> BudgetSimulationResult:
    """Simulates spending a scarce oracle budget to override uncertain verifier judgments."""
    rng = np.random.default_rng(seed)
    n = len(verifier_results)

    # 1. Gather ensemble judgments and uncertainty/disagreement
    items: list[dict] = []
    for idx, (comp, ex) in enumerate(verifier_results):
        v_res = ensemble_verifier.verify(comp, ex)
        o_res = oracle_verifier.verify(comp, ex)

        disagreement = float(v_res.metadata.get("disagreement", 0.0))
        items.append(
            {
                "idx": idx,
                "comp": comp,
                "ex": ex,
                "v_passed": v_res.passed,
                "o_passed": o_res.passed,
                "disagreement": disagreement,
            }
        )

    # 2. Select items to query oracle
    queries_to_make = min(budget, n)
    if strategy == "random":
        chosen_indices = set(rng.choice(n, size=queries_to_make, replace=False))
    elif strategy == "disagreement":
        # Sort descending by disagreement, break ties with random noise
        sorted_items = sorted(
            items,
            key=lambda x: (x["disagreement"], rng.random()),
            reverse=True,
        )
        chosen_indices = {x["idx"] for x in sorted_items[:queries_to_make]}
    else:
        raise ValueError(f"Unknown budget allocation strategy: {strategy}")

    # 3. Compute final decision accuracy where chosen items receive oracle ground truth
    correct_count = 0
    fp_retained = 0
    for item in items:
        if item["idx"] in chosen_indices:
            decision = item["o_passed"]
        else:
            decision = item["v_passed"]

        if decision == item["o_passed"]:
            correct_count += 1
        elif decision and not item["o_passed"]:
            fp_retained += 1

    acc = correct_count / n if n > 0 else 0.0

    return BudgetSimulationResult(
        strategy=strategy,
        oracle_budget=budget,
        oracle_queries_used=queries_to_make,
        final_decision_accuracy=acc,
        false_positives_retained=fp_retained,
    )
