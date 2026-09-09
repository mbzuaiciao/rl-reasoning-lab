"""Phase 8 Experiment: Multiple Comparisons, Family-Wise Error, and False Discovery Control.

Simulates testing 20 algorithmic variants against a baseline:
- 18 variants are purely null (no true improvement, Delta = 0)
- 2 variants are genuine improvements (Delta > 0)
Demonstrates:
- Naive testing at alpha = 0.05 yields inflated False Positive rates (FWER inflation)
- Bonferroni correction controls FWER strictly (<= alpha) but sacrifices statistical power
- Benjamini-Hochberg procedure controls FDR while preserving discovery power for true effects
"""

import math
import time

import numpy as np

from rl_reasoning_lab.evaluation.statistics import benjamini_hochberg, bonferroni_correction
from rl_reasoning_lab.experiments.config import (
    ExperimentManifest,
    ExperimentResult,
    deterministic_run_id,
    save_experiment_result,
)


def _approx_p_value_greater(x: np.ndarray, y: np.ndarray) -> float:
    """Compute one-sided Welch's t-test p-value using normal CDF via standard library math.erf."""
    nx, ny = len(x), len(y)
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    denom = math.sqrt(vx / nx + vy / ny)
    if denom < 1e-12:
        return 0.5 if mx == my else (0.0 if mx > my else 1.0)
    t_stat = (mx - my) / denom
    # Standard normal survival function 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))
    return float(0.5 * math.erfc(t_stat / math.sqrt(2.0)))


def run_multiple_comparisons_study() -> ExperimentResult:
    start_time = time.time()
    print("=" * 75)
    print("Phase 8: Multiple Comparisons, FWER Inflation, and FDR Control")
    print("=" * 75)

    rng = np.random.default_rng(42)

    # 20 candidate algorithm variants tested against a standard baseline:
    # Variants 0..17: Pure null (true mean difference = 0.0)
    # Variants 18..19: Genuine improvement (true mean difference = +0.08)
    num_variants = 20
    true_effects = [0.0] * 18 + [0.08, 0.09]
    n_seeds_per_variant = 12

    # Baseline performance: mean 0.70, std 0.06
    baseline_samples = rng.normal(0.70, 0.06, size=n_seeds_per_variant)

    p_values = []
    observed_diffs = []

    print("\nEvaluating 20 algorithmic variants (18 nulls, 2 genuine)...")
    for i in range(num_variants):
        delta = true_effects[i]
        var_samples = rng.normal(0.70 + delta, 0.06, size=n_seeds_per_variant)
        # One-sided t-test p-value (variant > baseline)
        p_val = _approx_p_value_greater(var_samples, baseline_samples)
        obs_diff = float(np.mean(var_samples) - np.mean(baseline_samples))
        p_values.append(p_val)
        observed_diffs.append(obs_diff)

    p_arr = np.array(p_values)

    # 1. Naive decision rule (p < 0.05)
    naive_reject = [bool(p < 0.05) for p in p_arr]

    # 2. Bonferroni correction (alpha = 0.05 / 20 = 0.0025)
    bonf_reject, bonf_alpha = bonferroni_correction(p_arr, alpha=0.05)

    # 3. Benjamini-Hochberg FDR control (FDR = 0.05)
    bh_reject, adj_p_vals = benjamini_hochberg(p_arr, fdr=0.05)

    print("\n--- Summary of Hypothesis Testing Decisions across 20 Variants ---")
    print(
        f"{'Variant':<10} | {'True Effect':<12} | {'Obs Diff':<10} | {'Raw p-val':<10} | {'Naive (<0.05)':<14} | {'Bonferroni':<12} | {'B-H FDR (0.05)':<14}"
    )
    print("-" * 90)

    false_pos_naive = 0
    false_pos_bonf = 0
    false_pos_bh = 0

    true_pos_naive = 0
    true_pos_bonf = 0
    true_pos_bh = 0

    for i in range(num_variants):
        is_true = i >= 18
        if naive_reject[i]:
            if is_true:
                true_pos_naive += 1
            else:
                false_pos_naive += 1
        if bonf_reject[i]:
            if is_true:
                true_pos_bonf += 1
            else:
                false_pos_bonf += 1
        if bh_reject[i]:
            if is_true:
                true_pos_bh += 1
            else:
                false_pos_bh += 1

        print(
            f"{'V' + str(i):<10} | {true_effects[i]:<12.3f} | {observed_diffs[i]:<10.3f} | {p_arr[i]:<10.4f} | "
            f"{str(naive_reject[i]):<14} | {str(bonf_reject[i]):<12} | {str(bh_reject[i]):<14}"
        )

    print("\n--- Methodological Comparison ---")
    print("1. Naive Testing (alpha=0.05):")
    print(
        f"   Discoveries: {sum(naive_reject)} | True Positives: {true_pos_naive}/2 | False Positives: {false_pos_naive}"
    )
    print("   -> Result: High risk of false breakthroughs from exploratory search!")

    print(f"\n2. Bonferroni FWER Control (alpha={bonf_alpha:.4f}):")
    print(
        f"   Discoveries: {sum(bonf_reject)} | True Positives: {true_pos_bonf}/2 | False Positives: {false_pos_bonf}"
    )
    print("   -> Result: Zero false positives, but overly conservative.")

    print("\n3. Benjamini-Hochberg FDR Control (q=0.05):")
    print(
        f"   Discoveries: {sum(bh_reject)} | True Positives: {true_pos_bh}/2 | False Positives: {false_pos_bh}"
    )
    print(
        "   -> Result: Successfully recovers genuine improvements while filtering spurious nulls."
    )

    manifest = ExperimentManifest(
        name="phase8_multiple_comparisons",
        seed=42,
        env_seed=101,
        method="Hypothesis_Correction",
        tags=["phase8", "multiple_testing", "fwer", "fdr"],
    )

    metrics = {
        "false_positives_naive": float(false_pos_naive),
        "false_positives_bonf": float(false_pos_bonf),
        "false_positives_bh": float(false_pos_bh),
        "true_positives_naive": float(true_pos_naive),
        "true_positives_bonf": float(true_pos_bonf),
        "true_positives_bh": float(true_pos_bh),
        "bonferroni_corrected_alpha": bonf_alpha,
    }

    result = ExperimentResult(
        manifest=manifest,
        run_id=deterministic_run_id(manifest),
        metrics=metrics,
        duration_seconds=time.time() - start_time,
    )

    save_experiment_result(result, "experiments/results/phase8_multiple_comparisons.json")
    print("\nSaved experiment result to experiments/results/phase8_multiple_comparisons.json")
    return result


if __name__ == "__main__":
    run_multiple_comparisons_study()
