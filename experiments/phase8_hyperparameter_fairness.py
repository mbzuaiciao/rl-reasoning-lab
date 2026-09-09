"""Phase 8 Experiment: Hyperparameter Search Fairness, Winner's Curse, and Test Contamination.

Demonstrates:
1. Search Budget Selection Bias:
   - Comparing Method A (20 configs searched) vs Method B (2 configs searched)
   - "Did the algorithm improve, or did we search harder?"
   - Re-evaluating under matched search budgets (20 vs 20).
2. The Winner's Curse:
   - When multiple configurations share identical true performance, the empirical winner
     is an extreme upper-tail noise fluctuation.
   - Proves E[Validation Winner] > E[Fresh Test Performance].
3. Repeated-Test Set Contamination:
   - Simulates querying the final test set across 30 tuning iterations.
   - Shows how decision feedback silently optimizes against the test set, creating false progress.
"""

import time

import numpy as np

from rl_reasoning_lab.evaluation.statistics import bootstrap_mean_ci
from rl_reasoning_lab.experiments.config import (
    ExperimentManifest,
    ExperimentResult,
    deterministic_run_id,
    save_experiment_result,
)


def run_hyperparameter_fairness_study() -> ExperimentResult:
    start_time = time.time()
    print("=" * 75)
    print("Phase 8: Hyperparameter Fairness, Winner's Curse, and Test Contamination")
    print("=" * 75)

    rng = np.random.default_rng(42)

    # -------------------------------------------------------------
    # 1. Search Budget Selection Bias
    # -------------------------------------------------------------
    # True underlying performance of both methods is identical: mean 0.75, standard dev 0.05
    true_mean = 0.75
    sigma_noise = 0.05

    print("\n--- Part 1: Unequal Search Budgets (20 Configs vs 2 Configs) ---")
    n_sims = 500
    a_best_unequal = []
    b_best_unequal = []

    a_best_equal = []
    b_best_equal = []

    for _ in range(n_sims):
        # Unequal budgets: A gets 20 configs, B gets 2 configs
        scores_a_20 = rng.normal(true_mean, sigma_noise, size=20)
        scores_b_2 = rng.normal(true_mean, sigma_noise, size=2)
        a_best_unequal.append(np.max(scores_a_20))
        b_best_unequal.append(np.max(scores_b_2))

        # Equal budgets: both get 20 configs
        scores_b_20 = rng.normal(true_mean, sigma_noise, size=20)
        a_best_equal.append(np.max(scores_a_20))
        b_best_equal.append(np.max(scores_b_20))

    mean_a_uneq = float(np.mean(a_best_unequal))
    mean_b_uneq = float(np.mean(b_best_unequal))
    print("Unequal Budget (A=20 configs, B=2 configs) with IDENTICAL true capability (75.0%):")
    print(f"  Method A Reported Best : {mean_a_uneq * 100:.2f}%")
    print(f"  Method B Reported Best : {mean_b_uneq * 100:.2f}%")
    print(
        f"  Apparent Advantage of A: +{(mean_a_uneq - mean_b_uneq) * 100:.2f}%  <-- Pure Search Artifact!"
    )

    mean_a_eq = float(np.mean(a_best_equal))
    mean_b_eq = float(np.mean(b_best_equal))
    print("\nMatched Budget (Both A and B get 20 configs):")
    print(f"  Method A Reported Best : {mean_a_eq * 100:.2f}%")
    print(f"  Method B Reported Best : {mean_b_eq * 100:.2f}%")
    print(f"  Real Advantage         : {(mean_a_eq - mean_b_eq) * 100:+.2f}%  (Statistically zero)")

    # -------------------------------------------------------------
    # 2. Winner's Curse
    # -------------------------------------------------------------
    print("\n--- Part 2: Winner's Curse (Validation Winner vs Fresh Test Performance) ---")
    # Simulate 50 hyperparameter configurations evaluated on a validation set (size N_val)
    # The configuration with the highest validation score is deployed to a fresh held-out test set.
    val_winners = []
    fresh_tests = []

    for _ in range(1000):
        # 30 configurations, each having identical true skill 70.0%
        # Validation evaluation noise: sigma = 0.04
        # Test evaluation noise: independent sigma = 0.04
        val_evals = rng.normal(0.70, 0.04, size=30)
        best_cfg_idx = int(np.argmax(val_evals))
        val_winner_score = val_evals[best_cfg_idx]

        # Fresh test evaluation of the winning configuration
        test_eval = float(rng.normal(0.70, 0.04))

        val_winners.append(val_winner_score)
        fresh_tests.append(test_eval)

    v_mean, v_low, v_high = bootstrap_mean_ci(val_winners, num_bootstrap=1000, seed=42)
    t_mean, t_low, t_high = bootstrap_mean_ci(fresh_tests, num_bootstrap=1000, seed=42)
    shrinkage = v_mean - t_mean

    print(
        f"Average Selected Validation Winner : {v_mean * 100:.2f}% [95% CI: {v_low * 100:.1f}%, {v_high * 100:.1f}%]"
    )
    print(
        f"Fresh Held-Out Test Performance    : {t_mean * 100:.2f}% [95% CI: {t_low * 100:.1f}%, {t_high * 100:.1f}%]"
    )
    print(f"Winner's Curse Regression/Shrinkage: -{shrinkage * 100:.2f}%")
    print(
        "  [Key Takeaway]: The chosen validation winner is systematically biased upwards by noise."
    )

    # -------------------------------------------------------------
    # 3. Repeated-Test Set Contamination Simulation
    # -------------------------------------------------------------
    print("\n--- Part 3: Test-Set Contamination Across Tuning Iterations ---")
    # An experimenter uses the test set directly to decide whether to accept modifications.
    # 40 sequential hyperparameter tweaks are attempted, none of which actually improves true accuracy.
    current_true = 0.70
    current_test_score = float(rng.normal(current_true, 0.03))

    test_trajectory = [current_test_score]
    true_trajectory = [current_true]

    for _iteration in range(1, 31):
        # Candidate tweak has ZERO true improvement
        candidate_test_score = float(rng.normal(current_true, 0.03))
        # Contamination rule: Accept tweak if candidate scores higher on the test set
        if candidate_test_score > current_test_score:
            current_test_score = candidate_test_score

        test_trajectory.append(current_test_score)
        true_trajectory.append(current_true)

    print(
        f"Initial Test Score              : {test_trajectory[0] * 100:.2f}% (True: {true_trajectory[0] * 100:.1f}%)"
    )
    print(
        f"Final Test Score After 30 Checks: {test_trajectory[-1] * 100:.2f}% (True: {true_trajectory[-1] * 100:.1f}%)"
    )
    print(
        f"Contamination Overestimate      : +{(test_trajectory[-1] - true_trajectory[-1]) * 100:.2f}%"
    )
    print(
        "  [Key Takeaway]: Selecting tweaks by checking the test set silently 'optimizes' against test noise!"
    )

    manifest = ExperimentManifest(
        name="phase8_hyperparameter_fairness",
        seed=42,
        env_seed=101,
        method="Hyperparameter_Auditing",
        tags=["phase8", "selection_bias", "winners_curse", "test_leakage"],
    )

    metrics = {
        "unequal_search_apparent_gain": mean_a_uneq - mean_b_uneq,
        "matched_search_gain": mean_a_eq - mean_b_eq,
        "validation_winner_score": v_mean,
        "fresh_test_score": t_mean,
        "winners_curse_gap": shrinkage,
        "test_contamination_inflation": test_trajectory[-1] - true_trajectory[-1],
    }

    result = ExperimentResult(
        manifest=manifest,
        run_id=deterministic_run_id(manifest),
        metrics=metrics,
        duration_seconds=time.time() - start_time,
    )

    save_experiment_result(result, "experiments/results/phase8_hyperparameter_fairness.json")
    print("\nSaved experiment result to experiments/results/phase8_hyperparameter_fairness.json")
    return result


if __name__ == "__main__":
    run_hyperparameter_fairness_study()
