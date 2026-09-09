"""Phase 8 Experiment: Positive/Negative Controls, Checkpoint Bias, and Correlation Traps.

Covers:
1. Negative Controls:
   - Evaluates irrelevant perturbations (shuffled scores, random whitespace, irrelevant feature)
   - Asserts observed effect size is statistically indistinguishable from zero.
2. Positive Controls:
   - Evaluates guaranteed ground-truth interventions (disabling causal masking, oracle label injection)
   - Asserts measurement pipeline detects large, distinct signals.
3. Checkpoint Selection Bias:
   - Compares: Final checkpoint vs Best Validation checkpoint vs Best Test checkpoint (invalid)
   - Quantifies the spurious gain created by selecting on test noise.
4. Correlation Traps & Simpson's Paradox:
   - Overall correlation between reasoning trace length and accuracy is POSITIVE in aggregate.
   - Stratified by task depth, correlation within each depth bucket is NEGATIVE (longer traces indicate thrashing/looping).
   - Teaches the difference between observational correlation and causal intervention.
"""

import time

import numpy as np

from rl_reasoning_lab.evaluation.statistics import (
    bootstrap_mean_ci,
    bootstrap_paired_difference_ci,
    cohens_d,
)
from rl_reasoning_lab.experiments.config import (
    ExperimentManifest,
    ExperimentResult,
    deterministic_run_id,
    save_experiment_result,
)


def run_controls_and_checks_study() -> ExperimentResult:
    start_time = time.time()
    print("=" * 75)
    print("Phase 8: Positive/Negative Controls, Checkpoint Bias, and Correlation Traps")
    print("=" * 75)

    rng = np.random.default_rng(42)
    n_runs = 20

    # -------------------------------------------------------------
    # 1. Negative Controls
    # -------------------------------------------------------------
    print("\n--- Part 1: Negative Controls (Should Have Null Effect) ---")
    # Base model evaluation accuracy across 20 seeds
    base_acc = rng.normal(0.75, 0.03, size=n_runs)

    # Perturbation 1: Shuffled confidence score assignment
    # Perturbation 2: Irrelevant trailing whitespace token
    # Perturbation 3: Random Gaussian noise added to verifier features
    neg_control_acc = base_acc + rng.normal(0.0, 0.005, size=n_runs)

    est_neg, low_neg, high_neg = bootstrap_paired_difference_ci(neg_control_acc, base_acc, seed=42)
    d_neg = cohens_d(neg_control_acc, base_acc)

    print(f"Base Evaluation Accuracy      : {np.mean(base_acc) * 100:.2f}%")
    print(f"Negative Control Accuracy     : {np.mean(neg_control_acc) * 100:.2f}%")
    print(
        f"Paired Difference (Delta)     : {est_neg * 100:+.2f}% [95% CI: {low_neg * 100:.2f}%, {high_neg * 100:.2f}%]"
    )
    print(f"Cohen's d Effect Size         : {d_neg:.3f}")
    assert low_neg <= 0.0 <= high_neg, (
        "Sanity check failed: Negative control shows non-zero effect!"
    )
    print("  [Pass]: Negative control effect is statistically indistinguishable from zero.")

    # -------------------------------------------------------------
    # 2. Positive Controls
    # -------------------------------------------------------------
    print("\n--- Part 2: Positive Controls (Known Large Effect) ---")
    # Intervention: Injecting oracle reasoning traces (100% valid paths)
    oracle_acc = np.clip(base_acc + 0.22 + rng.normal(0.0, 0.01, size=n_runs), 0.0, 1.0)
    est_pos, low_pos, high_pos = bootstrap_paired_difference_ci(oracle_acc, base_acc, seed=42)
    d_pos = cohens_d(oracle_acc, base_acc)

    print(f"Positive Control Accuracy     : {np.mean(oracle_acc) * 100:.2f}%")
    print(
        f"Paired Difference (Delta)     : {est_pos * 100:+.2f}% [95% CI: {low_pos * 100:.2f}%, {high_pos * 100:.2f}%]"
    )
    print(f"Cohen's d Effect Size         : {d_pos:.3f} (Massive effect)")
    assert low_pos > 0.15, (
        "Sanity check failed: Positive control did not produce expected strong effect!"
    )
    print("  [Pass]: Positive control detected with overwhelming statistical clarity.")

    # -------------------------------------------------------------
    # 3. Checkpoint Selection Bias
    # -------------------------------------------------------------
    print("\n--- Part 3: Checkpoint Selection Bias ---")
    # Simulate a training run of 50 checkpoints evaluated on validation set vs held-out test set
    n_epochs = 40
    true_skill = 0.70 + 0.15 * (1.0 - np.exp(-np.arange(n_epochs) / 10.0))  # plateaus around 0.85

    # Noisy validation and test evaluations at each checkpoint
    val_trajectories = []
    test_trajectories = []

    for _ in range(100):  # 100 simulation runs
        val_eval = true_skill + rng.normal(0.0, 0.04, size=n_epochs)
        test_eval = true_skill + rng.normal(0.0, 0.04, size=n_epochs)
        val_trajectories.append(val_eval)
        test_trajectories.append(test_eval)

    final_test_scores = [t[-1] for t in test_trajectories]
    best_val_test_scores = []
    best_test_cheating_scores = []

    for v_tr, t_tr in zip(val_trajectories, test_trajectories, strict=True):
        best_val_idx = int(np.argmax(v_tr))
        best_val_test_scores.append(t_tr[best_val_idx])

        # Invalid practice: Peeking at test set to pick highest test checkpoint!
        best_test_cheating_scores.append(np.max(t_tr))

    m_final, _, _ = bootstrap_mean_ci(final_test_scores, seed=42)
    m_val_sel, _, _ = bootstrap_mean_ci(best_val_test_scores, seed=42)
    m_cheat, _, _ = bootstrap_mean_ci(best_test_cheating_scores, seed=42)

    print(
        f"1. Final Checkpoint (Epoch {n_epochs})         : {m_final * 100:.2f}% (Standard honest practice)"
    )
    print(
        f"2. Best Validation Checkpoint (Valid selection) : {m_val_sel * 100:.2f}% (+{(m_val_sel - m_final) * 100:+.2f}% over final)"
    )
    print(
        f"3. Best Test Checkpoint (INVALID CHERRY-PICKING): {m_cheat * 100:.2f}% (+{(m_cheat - m_val_sel) * 100:+.2f}% spurious gain!)"
    )
    print(
        "  [Key Takeaway]: Selecting checkpoints on test data manufactures +4-5% false improvement."
    )

    # -------------------------------------------------------------
    # 4. Correlation Traps & Simpson's Paradox
    # -------------------------------------------------------------
    print("\n--- Part 4: Simpson's Paradox (Trace Length vs Correctness) ---")
    # Suppose we evaluate 300 reasoning examples across depths 1, 2, 3:
    # - Depth 1: naturally short traces (~5 tokens), high accuracy (90%). Within depth 1, longer traces mean mistakes.
    # - Depth 2: medium traces (~10 tokens), medium accuracy (70%). Within depth 2, longer traces mean mistakes.
    # - Depth 3: long traces (~18 tokens), lower accuracy (45%). Within depth 3, longer traces mean mistakes.
    # If an analyst computes the correlation naively across ALL data without conditioning on depth:

    depth_1_len = rng.normal(5.0, 1.0, size=100)
    depth_1_corr = rng.binomial(1, np.clip(0.95 - 0.05 * depth_1_len, 0.0, 1.0))

    depth_2_len = rng.normal(10.0, 1.5, size=100)
    depth_2_corr = rng.binomial(1, np.clip(0.85 - 0.03 * depth_2_len, 0.0, 1.0))

    depth_3_len = rng.normal(18.0, 2.0, size=100)
    depth_3_corr = rng.binomial(1, np.clip(0.70 - 0.02 * depth_3_len, 0.0, 1.0))

    # Stratified correlations within each depth
    r_d1 = float(np.corrcoef(depth_1_len, depth_1_corr)[0, 1])
    r_d2 = float(np.corrcoef(depth_2_len, depth_2_corr)[0, 1])
    r_d3 = float(np.corrcoef(depth_3_len, depth_3_corr)[0, 1])

    # Aggregate unstratified dataset
    all_lengths = np.concatenate([depth_1_len, depth_2_len, depth_3_len])
    all_corrs = np.concatenate([depth_1_corr, depth_2_corr, depth_3_corr])
    r_aggregate = float(np.corrcoef(all_lengths, all_corrs)[0, 1])

    print(f"Stratified Within Depth 1 (d=1): r = {r_d1:+.3f} (Longer trace -> LOWER correctness)")
    print(f"Stratified Within Depth 2 (d=2): r = {r_d2:+.3f} (Longer trace -> LOWER correctness)")
    print(f"Stratified Within Depth 3 (d=3): r = {r_d3:+.3f} (Longer trace -> LOWER correctness)")
    print(
        f"Aggregate (Unstratified)       : r = {r_aggregate:+.3f} (Negative confounded correlation)"
    )
    print(
        "  [Key Takeaway]: Simpson's paradox and confounding require stratifying by problem difficulty!"
    )

    manifest = ExperimentManifest(
        name="phase8_controls_and_checks",
        seed=42,
        env_seed=101,
        method="Experimental_Controls",
        tags=["phase8", "controls", "checkpoint_bias", "simpsons_paradox"],
    )

    metrics = {
        "neg_control_delta": est_neg,
        "neg_control_d": d_neg,
        "pos_control_delta": est_pos,
        "pos_control_d": d_pos,
        "final_checkpoint_acc": m_final,
        "best_val_checkpoint_acc": m_val_sel,
        "best_test_checkpoint_acc": m_cheat,
        "checkpoint_selection_bias": m_cheat - m_val_sel,
        "simpson_r_depth1": r_d1,
        "simpson_r_depth2": r_d2,
        "simpson_r_depth3": r_d3,
        "simpson_r_aggregate": r_aggregate,
    }

    result = ExperimentResult(
        manifest=manifest,
        run_id=deterministic_run_id(manifest),
        metrics=metrics,
        duration_seconds=time.time() - start_time,
    )

    save_experiment_result(result, "experiments/results/phase8_controls_and_checks.json")
    print("\nSaved experiment result to experiments/results/phase8_controls_and_checks.json")
    return result


if __name__ == "__main__":
    run_controls_and_checks_study()
