"""Unit and invariant tests for Phase 8: Experimental Science and Statistics."""

import tempfile

import numpy as np
import pytest

from rl_reasoning_lab.evaluation.statistics import (
    benjamini_hochberg,
    bonferroni_correction,
    bootstrap_difference_of_means_ci,
    bootstrap_effect_size_ci,
    bootstrap_mean_ci,
    bootstrap_median_ci,
    bootstrap_paired_difference_ci,
    bootstrap_trimmed_mean_ci,
    cohens_d,
    distribution_summary,
    trimmed_mean,
    worst_subgroup_metric,
)
from rl_reasoning_lab.experiments.config import (
    ExperimentManifest,
    ExperimentResult,
    SplitManager,
    deterministic_run_id,
    load_experiment_result,
    save_experiment_result,
)


class TestBootstrapAndStatistics:
    """Test mathematical invariants and properties of bootstrap estimators."""

    def test_constant_array_ci_collapses_to_point_estimate(self):
        data = [7.5, 7.5, 7.5, 7.5, 7.5]
        mean_est, lower, upper = bootstrap_mean_ci(data, num_bootstrap=500, ci=0.95, seed=42)
        assert mean_est == pytest.approx(7.5)
        assert lower == pytest.approx(7.5)
        assert upper == pytest.approx(7.5)

        med_est, med_lower, med_upper = bootstrap_median_ci(
            data, num_bootstrap=500, ci=0.95, seed=42
        )
        assert med_est == pytest.approx(7.5)
        assert med_lower == pytest.approx(7.5)
        assert med_upper == pytest.approx(7.5)

    def test_bootstrap_reproducibility_and_coverage(self):
        rng = np.random.default_rng(123)
        sample = rng.normal(loc=10.0, scale=2.0, size=50)

        # Run twice with identical seed
        est1, low1, high1 = bootstrap_mean_ci(sample, num_bootstrap=1000, ci=0.95, seed=999)
        est2, low2, high2 = bootstrap_mean_ci(sample, num_bootstrap=1000, ci=0.95, seed=999)

        assert est1 == est2
        assert low1 == low2
        assert high1 == high2
        assert low1 < est1 < high1
        # True mean 10.0 should be covered
        assert low1 <= 10.0 <= high1

    def test_trimmed_mean_removes_extreme_outliers(self):
        # 5 elements: with proportion 0.2, cuts floor(5*0.2) = 1 element from each side
        data = [-1000.0, 2.0, 3.0, 4.0, 5000.0]
        # Raw mean is massively corrupted
        assert np.mean(data) > 700.0
        # Trimmed mean removes -1000.0 and 5000.0, leaving [2, 3, 4] -> mean 3.0
        t_mean = trimmed_mean(data, proportion_to_cut=0.2)
        assert t_mean == pytest.approx(3.0)

        # Bootstrap trimmed mean CI
        est, low, high = bootstrap_trimmed_mean_ci(
            data, proportion_to_cut=0.2, num_bootstrap=500, seed=42
        )
        assert est == pytest.approx(3.0)
        assert low < high

    def test_distribution_summary_quantiles_and_skewness(self):
        # Asymmetric right-skewed data
        data = [1.0, 1.0, 1.2, 1.3, 1.5, 2.0, 10.0]
        summary = distribution_summary(data)
        assert summary["mean"] > summary["median"]
        assert summary["skewness"] > 0.0  # Positively skewed
        assert summary["q50"] == summary["median"]
        assert summary["q25"] <= summary["q50"] <= summary["q75"]
        assert summary["iqr"] == pytest.approx(summary["q75"] - summary["q25"])


class TestPairedDifferenceAndEffectSize:
    """Test paired comparisons and standardized effect sizes."""

    def test_paired_variance_reduction_under_correlated_noise(self):
        # When two methods share strong common prompt/task variance,
        # paired comparison cancels the common noise, yielding a strictly tighter CI.
        rng = np.random.default_rng(42)
        n = 40
        common_task_difficulty = rng.normal(0.0, 5.0, size=n)
        method_a = 10.0 + common_task_difficulty + rng.normal(0.0, 0.5, size=n)
        method_b = 8.0 + common_task_difficulty + rng.normal(0.0, 0.5, size=n)

        # Paired difference
        p_est, p_low, p_high = bootstrap_paired_difference_ci(
            method_a, method_b, num_bootstrap=1000, seed=42
        )
        paired_width = p_high - p_low

        # Unpaired difference of means
        u_est, u_low, u_high = bootstrap_difference_of_means_ci(
            method_a, method_b, num_bootstrap=1000, seed=42
        )
        unpaired_width = u_high - u_low

        assert p_est == pytest.approx(u_est, abs=1e-5)
        # Paired CI width should be significantly narrower than unpaired CI width
        assert paired_width < unpaired_width * 0.4

    def test_cohens_d_and_effect_size_ci(self):
        a = [10.0, 11.0, 12.0, 10.5, 11.5]
        b = [6.0, 7.0, 8.0, 6.5, 7.5]
        d = cohens_d(a, b)
        # Mean diff is 4.0, standard deviation of both is ~0.79 -> d ~ 5.0
        assert d > 4.0

        point_d, d_low, d_high = bootstrap_effect_size_ci(a, b, num_bootstrap=500, seed=42)
        assert point_d == pytest.approx(d)
        assert d_low < point_d < d_high

    def test_paired_difference_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            bootstrap_paired_difference_ci([1.0, 2.0], [1.0])


class TestMultipleComparisons:
    """Test FWER and FDR correction procedures."""

    def test_bonferroni_correction(self):
        # 5 hypotheses tested at alpha = 0.05 -> corrected alpha = 0.01
        p_vals = [0.005, 0.02, 0.045, 0.06, 0.50]
        rejects, corr_alpha = bonferroni_correction(p_vals, alpha=0.05)
        assert corr_alpha == pytest.approx(0.01)
        assert rejects == [True, False, False, False, False]

    def test_benjamini_hochberg_fdr(self):
        # BH is more powerful than Bonferroni while controlling FDR
        p_vals = [0.001, 0.008, 0.012, 0.04, 0.20, 0.80]
        rejects, adj_p = benjamini_hochberg(p_vals, fdr=0.05)
        # Ranks:
        # p1=0.001 <= (1/6)*0.05 = 0.0083 (True)
        # p2=0.008 <= (2/6)*0.05 = 0.0167 (True)
        # p3=0.012 <= (3/6)*0.05 = 0.0250 (True)
        # p4=0.040 <= (4/6)*0.05 = 0.0333 (False)
        assert rejects[:3] == [True, True, True]
        assert rejects[3:] == [False, False, False]
        # Adjusted p-values must be weakly monotonic when sorted
        for i in range(len(adj_p) - 1):
            assert adj_p[i] <= adj_p[i + 1] + 1e-6


class TestManifestAndSerialization:
    """Test experiment manifest, run ID hashing, and JSON persistence."""

    def test_deterministic_run_id_and_hashing(self):
        manifest1 = ExperimentManifest(
            name="ppo_test",
            seed=42,
            env_seed=100,
            method="PPO_Process",
            model_config={"d_model": 32, "n_layers": 2},
            training_budget={"episodes": 200},
            tags=["phase8", "ablation"],
        )
        manifest2 = ExperimentManifest(
            name="ppo_test",
            seed=42,
            env_seed=100,
            method="PPO_Process",
            model_config={"d_model": 32, "n_layers": 2},
            training_budget={"episodes": 200},
            tags=["ablation", "phase8"],  # permuted tag order should still produce identical hash
        )
        assert manifest1.config_hash() == manifest2.config_hash()
        assert deterministic_run_id(manifest1) == deterministic_run_id(manifest2)

    def test_save_and_load_experiment_result_roundtrip(self):
        manifest = ExperimentManifest(
            name="test_run",
            seed=7,
            env_seed=14,
            method="SFT",
            model_config={"d_model": 16},
        )
        result = ExperimentResult(
            manifest=manifest,
            run_id=deterministic_run_id(manifest),
            metrics={"final_accuracy": 0.85, "ece": 0.042},
            seed_runs=[{"seed": 7, "acc": 0.85}],
            traces=[{"prompt": "A->B", "prediction": "B"}],
            duration_seconds=1.23,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = f"{tmpdir}/res.json"
            save_experiment_result(result, fpath)
            loaded = load_experiment_result(fpath)

            assert loaded.run_id == result.run_id
            assert loaded.manifest.seed == 7
            assert loaded.metrics["final_accuracy"] == pytest.approx(0.85)
            assert loaded.duration_seconds == pytest.approx(1.23)
            assert len(loaded.traces) == 1


class TestSplitManagerIsolation:
    """Test split isolation and anti-contamination tracking."""

    def test_zero_overlap_between_splits(self):
        sm = SplitManager(base_seed=42, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        train_idx, val_idx, test_idx = sm.partition_indices(100)

        assert len(train_idx) == 60
        assert len(val_idx) == 20
        assert len(test_idx) == 20

        # Assert pairwise disjoint
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(set(val_idx).intersection(set(test_idx))) == 0

    def test_test_access_counting(self):
        sm = SplitManager(base_seed=42)
        assert sm.test_query_count == 0
        sm.record_test_access()
        assert sm.test_query_count == 1
        sm.record_test_access()
        assert sm.test_query_count == 2


class TestWorstSubgroupMetric:
    """Test subgroup aggregation and worst-case detection."""

    def test_worst_subgroup_detection(self):
        subgroups = {
            "depth_1": {"accuracy": 0.95, "ece": 0.02},
            "depth_2": {"accuracy": 0.92, "ece": 0.03},
            "depth_3": {"accuracy": 0.88, "ece": 0.05},
            "depth_4": {"accuracy": 0.75, "ece": 0.08},
            "depth_5": {"accuracy": 0.20, "ece": 0.25},  # Catastrophic failure hidden by mean
        }
        res = worst_subgroup_metric(subgroups, metric_key="accuracy")
        assert res["worst_subgroup"] == "depth_5"
        assert res["worst"] == pytest.approx(0.20)
        assert res["mean"] == pytest.approx(0.74)
        assert res["p10"] < 0.50  # Captures low tail
