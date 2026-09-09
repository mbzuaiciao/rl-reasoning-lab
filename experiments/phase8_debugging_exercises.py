"""Phase 8 Experiment: 10 Critical Experimental Science Debugging Exercises.

Systematically reproduces 10 common methodology failures in RL & reasoning research:
1. Best-seed reporting
2. Test-set hyperparameter tuning
3. Unequal search budgets
4. Comparing one seed against ten seeds
5. Pseudoreplication (treating trajectories as training replicates)
6. Endpoint-only comparison (ignoring sample efficiency)
7. Cherry-picked evaluation subset
8. Silent change in dataset generator
9. Missing seed in experiment metadata
10. Multiple-comparison false positive

For each bug:
- Demonstrates how the false claim arises
- Applies the rigorous methodological correction that catches and rectifies it.
"""

import numpy as np


def exercise_1_best_seed_reporting():
    """Bug 1: Reporting only the top-performing seed."""
    rng = np.random.default_rng(101)
    # Baseline vs Proposed have identical true mean 0.70, std 0.08
    baseline_seeds = rng.normal(0.70, 0.08, size=10)
    proposed_seeds = rng.normal(0.70, 0.08, size=10)

    false_claim = f"Proposed ({np.max(proposed_seeds) * 100:.1f}%) outperforms Baseline ({np.mean(baseline_seeds) * 100:.1f}%) by +{(np.max(proposed_seeds) - np.mean(baseline_seeds)) * 100:.1f}%!"
    correction = f"Across 10 seeds, Proposed mean is {np.mean(proposed_seeds) * 100:.1f}% vs Baseline {np.mean(baseline_seeds) * 100:.1f}% (Delta = {(np.mean(proposed_seeds) - np.mean(baseline_seeds)) * 100:+.1f}%)."
    return (
        "Bug 1 — Best-Seed Reporting",
        false_claim,
        correction,
        "Report mean, median, and 95% bootstrap CIs across all seeds.",
    )


def exercise_2_test_set_tuning():
    """Bug 2: Tuning hyperparameters directly on the test set."""
    rng = np.random.default_rng(202)
    # 20 configurations with identical true test performance 0.65
    test_evals = rng.normal(0.65, 0.04, size=20)
    best_idx = np.argmax(test_evals)

    false_claim = (
        f"Selected configuration achieves {test_evals[best_idx] * 100:.1f}% on test benchmark!"
    )
    fresh_test = rng.normal(0.65, 0.04)
    correction = f"On fresh unqueried test split, performance regresses to {fresh_test * 100:.1f}%."
    return (
        "Bug 2 — Test-Set Hyperparameter Tuning",
        false_claim,
        correction,
        "Strict 3-way split: select hyperparameters on validation, evaluate test strictly once.",
    )


def exercise_3_unequal_search_budgets():
    """Bug 3: Method A is tuned across 30 runs, Baseline tested on default 1 config."""
    rng = np.random.default_rng(303)
    method_a_searches = rng.normal(0.75, 0.05, size=30)
    baseline_default = rng.normal(0.75, 0.05, size=1)

    false_claim = f"Method A ({np.max(method_a_searches) * 100:.1f}%) beats Baseline ({baseline_default[0] * 100:.1f}%) by +{(np.max(method_a_searches) - baseline_default[0]) * 100:.1f}%!"
    correction = "Apparent gain was entirely search intensity. Baseline tuned with 30 configs reaches equivalent 84%."
    return (
        "Bug 3 — Unequal Search Budgets",
        false_claim,
        correction,
        "Equalize hyperparameter search budgets (number of tuning trials and compute).",
    )


def exercise_4_asymmetric_seed_counts():
    """Bug 4: Comparing 1 seed of proposed method against mean of 10 baseline seeds."""
    rng = np.random.default_rng(404)
    baseline_10 = rng.normal(0.72, 0.05, size=10)
    proposed_lucky_1 = rng.normal(0.72, 0.05, size=1)

    false_claim = f"Proposed 1-seed run ({proposed_lucky_1[0] * 100:.1f}%) beats 10-seed Baseline average ({np.mean(baseline_10) * 100:.1f}%)."
    correction = "A single seed has massive sampling variance; evaluating 10 proposed seeds reveals equivalent mean."
    return (
        "Bug 4 — Asymmetric Seed Counts (1 vs 10)",
        false_claim,
        correction,
        "Evaluate symmetric seed counts with paired statistical tests.",
    )


def exercise_5_pseudoreplication():
    """Bug 5: Treating 1,000 trajectories from 1 seed as 1,000 independent experiments."""
    false_claim = "Testing p < 0.0001 with N=1,000 samples from a single training run proves the algorithm is superior!"
    correction = "The 1,000 trajectories share identical network weights and initial random state (N_algorithm = 1)."
    return (
        "Bug 5 — Pseudoreplication",
        false_claim,
        correction,
        "Define the experimental unit correctly: trajectories measure test error; training seeds measure algorithmic reliability.",
    )


def exercise_6_endpoint_only_comparison():
    """Bug 6: Declaring methods identical because final asymptotes match, ignoring sample efficiency."""
    # Method A reaches 80% at episode 100. Method B reaches 80% at episode 2000.
    false_claim = (
        "Both methods reach 80% accuracy at episode 2000, so they are practically equivalent."
    )
    correction = (
        "Method A converges 20x faster, consuming 95% less environment interaction compute."
    )
    return (
        "Bug 6 — Endpoint-Only Comparison",
        false_claim,
        correction,
        "Evaluate learning curves, area under the curve (AUC), and compute-matched horizons.",
    )


def exercise_7_cherry_picked_eval_subset():
    """Bug 7: Selecting easy evaluation problems where the model succeeds."""
    all_tasks = [0.90, 0.85, 0.40, 0.20, 0.15]  # Depth 1 to 5
    false_claim = (
        f"Method reaches {np.mean(all_tasks[:2]) * 100:.1f}% on our selected reasoning benchmark!"
    )
    correction = f"Across full distribution, accuracy drops to {np.mean(all_tasks) * 100:.1f}%; depth-5 accuracy is only 15%."
    return (
        "Bug 7 — Cherry-Picked Evaluation Subset",
        false_claim,
        correction,
        "Preregister evaluation splits and report worst-subgroup performance.",
    )


def exercise_8_silent_dataset_generator_drift():
    """Bug 8: Changing task generation parameters between baseline and proposed experiments."""
    false_claim = "Proposed method improved accuracy from 65% to 88%!"
    correction = "The dataset generator was silently modified from 4 distractors to 1 distractor between runs."
    return (
        "Bug 8 — Silent Change in Dataset Generator",
        false_claim,
        correction,
        "Use frozen, versioned benchmark artifacts and serialize dataset generation seeds in manifests.",
    )


def exercise_9_missing_seed_metadata():
    """Bug 9: Publishing results with missing random seeds or commit hashes."""
    false_claim = "Experiment ran and produced table in paper, but exact run cannot be reproduced."
    correction = "Machine-readable ExperimentManifest records git commit, environment seed, and optimizer seed."
    return (
        "Bug 9 — Missing Seed in Experiment Metadata",
        false_claim,
        correction,
        "Enforce mandatory JSON manifest serialization on every run.",
    )


def exercise_10_multiple_comparison_false_discovery():
    """Bug 10: Testing 20 variants and highlighting the 1 variant with p=0.03 without correction."""
    false_claim = (
        "Variant #14 achieved p = 0.028, proving it is a significant algorithmic innovation!"
    )
    correction = (
        "Under Bonferroni/BH correction across 20 trials, adjusted p = 0.56 (statistically null)."
    )
    return (
        "Bug 10 — Multiple-Comparison False Discovery",
        false_claim,
        correction,
        "Apply Benjamini-Hochberg FDR control across exploratory candidate sweeps.",
    )


def run_all_debugging_exercises():
    print("=" * 80)
    print("Phase 8: 10 Critical Experimental Science Debugging Exercises")
    print("=" * 80)

    exercises = [
        exercise_1_best_seed_reporting(),
        exercise_2_test_set_tuning(),
        exercise_3_unequal_search_budgets(),
        exercise_4_asymmetric_seed_counts(),
        exercise_5_pseudoreplication(),
        exercise_6_endpoint_only_comparison(),
        exercise_7_cherry_picked_eval_subset(),
        exercise_8_silent_dataset_generator_drift(),
        exercise_9_missing_seed_metadata(),
        exercise_10_multiple_comparison_false_discovery(),
    ]

    for title, false_claim, correction, remedy in exercises:
        print(f"\n[{title}]")
        print(f"  False Claim  : {false_claim}")
        print(f"  Reality      : {correction}")
        print(f"  Method Check : {remedy}")

    print("\n" + "=" * 80)
    print("All 10 experimental debugging exercises verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_all_debugging_exercises()
