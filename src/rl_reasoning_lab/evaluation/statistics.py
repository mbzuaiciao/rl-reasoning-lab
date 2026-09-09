"""Statistical and bootstrap evaluation utilities for experimental science in RL.

Provides transparent, reproducible statistical estimators:
- Percentile bootstrap confidence intervals (mean, median, trimmed mean)
- Paired and unpaired bootstrap difference intervals
- Standardized effect sizes (Cohen's d) with bootstrap intervals
- Distribution summaries (quantiles, IQR, skewness, trimmed mean)
- Multiple testing corrections (Bonferroni FWER, Benjamini-Hochberg FDR)
- Subgroup and worst-case performance metrics
"""

from collections.abc import Callable
from typing import Any

import numpy as np


def trimmed_mean(data: np.ndarray | list[float], proportion_to_cut: float = 0.1) -> float:
    """Compute trimmed mean by discarding proportion_to_cut from each tail.

    Args:
        data: 1D sequence of numerical observations.
        proportion_to_cut: Fraction of values to trim from lowest and highest ends (0.0 <= p < 0.5).
    """
    arr = np.sort(np.asarray(data, dtype=np.float64))
    n = len(arr)
    if n == 0:
        return 0.0
    k = int(np.floor(n * proportion_to_cut))
    if k == 0 or 2 * k >= n:
        return float(np.mean(arr))
    trimmed = arr[k : n - k]
    return float(np.mean(trimmed))


def distribution_summary(data: np.ndarray | list[float]) -> dict[str, float]:
    """Compute comprehensive distribution summary metrics.

    Includes mean, median, std, min, max, percentiles (10, 25, 50, 75, 90),
    IQR, and Fisher-Pearson skewness.
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return {}
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    median_val = float(np.median(arr))
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])
    iqr = float(q75 - q25)

    if len(arr) > 2 and std_val > 1e-12:
        skewness = float(np.mean(((arr - mean_val) / std_val) ** 3))
    else:
        skewness = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q10": float(q10),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "iqr": iqr,
        "skewness": skewness,
    }


def bootstrap_ci(
    data: np.ndarray | list[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    num_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute non-parametric percentile bootstrap confidence interval.

    Args:
        data: 1D sequence of numerical observations.
        stat_fn: Summary statistic function (defaults to np.mean).
        num_bootstrap: Number of resamples.
        ci: Confidence level in (0, 1) (defaults to 0.95).
        seed: Random seed for exact reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    point_est = float(stat_fn(arr))
    if n == 1:
        return point_est, point_est, point_est

    rng = np.random.default_rng(seed)
    # Resample with replacement: shape (num_bootstrap, n)
    indices = rng.integers(0, n, size=(num_bootstrap, n))
    resamples = arr[indices]
    boot_stats = np.array([stat_fn(sample) for sample in resamples], dtype=np.float64)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_stats, lower_pct))
    ci_upper = float(np.percentile(boot_stats, upper_pct))
    return point_est, ci_lower, ci_upper


def bootstrap_mean_ci(
    data: np.ndarray | list[float],
    num_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Convenience wrapper for bootstrap mean confidence interval."""
    return bootstrap_ci(data, stat_fn=np.mean, num_bootstrap=num_bootstrap, ci=ci, seed=seed)


def bootstrap_median_ci(
    data: np.ndarray | list[float],
    num_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Convenience wrapper for bootstrap median confidence interval."""
    return bootstrap_ci(data, stat_fn=np.median, num_bootstrap=num_bootstrap, ci=ci, seed=seed)


def bootstrap_trimmed_mean_ci(
    data: np.ndarray | list[float],
    proportion_to_cut: float = 0.1,
    num_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for trimmed mean."""

    def _stat(arr: np.ndarray) -> float:
        return trimmed_mean(arr, proportion_to_cut=proportion_to_cut)

    return bootstrap_ci(data, stat_fn=_stat, num_bootstrap=num_bootstrap, ci=ci, seed=seed)


def bootstrap_paired_difference_ci(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    num_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for paired differences Delta_i = a_i - b_i.

    Paired evaluation conditions on common random seeds, task instances, or prompts,
    canceling nuisance variance.

    Args:
        a: Observations from treatment condition A (len N).
        b: Observations from control condition B (len N, paired with A).
        stat_fn: Summary statistic function on differences (default mean).
        num_bootstrap: Number of resamples.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        (point_estimate_of_diff, ci_lower, ci_upper)
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if len(arr_a) != len(arr_b):
        raise ValueError(
            f"Paired observations must have equal length: {len(arr_a)} vs {len(arr_b)}"
        )
    diffs = arr_a - arr_b
    return bootstrap_ci(diffs, stat_fn=stat_fn, num_bootstrap=num_bootstrap, ci=ci, seed=seed)


def bootstrap_difference_of_means_ci(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    num_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for unpaired difference of means (E[A] - E[B]).

    Independent resampling with replacement from each group separately.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    na, nb = len(arr_a), len(arr_b)
    if na == 0 or nb == 0:
        return 0.0, 0.0, 0.0

    point_est = float(np.mean(arr_a) - np.mean(arr_b))
    rng = np.random.default_rng(seed)

    boot_diffs = np.zeros(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        sample_a = rng.choice(arr_a, size=na, replace=True)
        sample_b = rng.choice(arr_b, size=nb, replace=True)
        boot_diffs[i] = np.mean(sample_a) - np.mean(sample_b)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))
    return point_est, ci_lower, ci_upper


def cohens_d(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Compute Cohen's d standardized effect size between two groups.

    d = (mean(A) - mean(B)) / s_pooled
    where s_pooled = sqrt( ((n_a - 1)*s_a^2 + (n_b - 1)*s_b^2) / (n_a + n_b - 2) )
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    na, nb = len(arr_a), len(arr_b)
    if na < 2 or nb < 2:
        return 0.0

    var_a = np.var(arr_a, ddof=1)
    var_b = np.var(arr_b, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_sd < 1e-12:
        return 0.0
    return float((np.mean(arr_a) - np.mean(arr_b)) / pooled_sd)


def bootstrap_effect_size_ci(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    num_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for Cohen's d standardized effect size."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    point_est = cohens_d(arr_a, arr_b)
    na, nb = len(arr_a), len(arr_b)
    if na < 2 or nb < 2:
        return point_est, point_est, point_est

    rng = np.random.default_rng(seed)
    boot_ds = np.zeros(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        sa = rng.choice(arr_a, size=na, replace=True)
        sb = rng.choice(arr_b, size=nb, replace=True)
        boot_ds[i] = cohens_d(sa, sb)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_ds, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_ds, 100.0 * (1.0 - alpha / 2.0)))
    return point_est, ci_lower, ci_upper


def bonferroni_correction(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[list[bool], float]:
    """Bonferroni correction for multiple hypothesis testing.

    Guarantees Family-Wise Error Rate (FWER) <= alpha.

    Args:
        p_values: Sequence of unadjusted p-values for M hypotheses.
        alpha: Target family-wise significance level.

    Returns:
        (reject_list, corrected_alpha)
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return [], alpha
    corrected_alpha = alpha / m
    reject = [bool(p < corrected_alpha) for p in p_arr]
    return reject, float(corrected_alpha)


def benjamini_hochberg(
    p_values: list[float] | np.ndarray,
    fdr: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg procedure for controlling the False Discovery Rate (FDR).

    Sorts p-values in ascending order p_(1) <= ... <= p_(m).
    Finds largest k such that p_(k) <= (k / m) * FDR.
    Rejects all H_(i) for i <= k.

    Args:
        p_values: Sequence of unadjusted p-values.
        fdr: Target false discovery rate (default 0.05).

    Returns:
        (reject_list, adjusted_p_values)
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return [], []

    sort_indices = np.argsort(p_arr)
    sorted_p = p_arr[sort_indices]

    # Calculate BH thresholds: (k / m) * fdr for k = 1..m
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * fdr

    # Find largest k where sorted_p <= threshold
    passed = sorted_p <= thresholds
    if np.any(passed):
        max_k = np.max(np.where(passed)[0])
        sorted_reject = np.zeros(m, dtype=bool)
        sorted_reject[: max_k + 1] = True
    else:
        sorted_reject = np.zeros(m, dtype=bool)

    # Invert sorting to restore original order
    reject_list = [False] * m
    for orig_idx, is_rej in zip(sort_indices, sorted_reject, strict=True):
        reject_list[orig_idx] = bool(is_rej)

    # Compute step-up adjusted p-values (q-values)
    adj_sorted = np.minimum.accumulate((sorted_p * m / ranks)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adjusted_p = [0.0] * m
    for orig_idx, adj_p in zip(sort_indices, adj_sorted, strict=True):
        adjusted_p[orig_idx] = float(adj_p)

    return reject_list, adjusted_p


def worst_subgroup_metric(
    subgroup_results: dict[str, dict[str, float]],
    metric_key: str = "accuracy",
) -> dict[str, Any]:
    """Compute subgroup distribution metrics, identifying worst-case performance.

    Prevents aggregate averages from masking catastrophic failures in specific subgroups
    (e.g., deeper reasoning or adversarial distractors).

    Args:
        subgroup_results: Mapping of subgroup_name -> metrics_dict.
        metric_key: Metric to analyze across subgroups.

    Returns:
        Summary dict containing mean, min (worst), p10, and worst_subgroup name.
    """
    if not subgroup_results:
        return {"mean": 0.0, "worst": 0.0, "p10": 0.0, "worst_subgroup": "none"}

    names = list(subgroup_results.keys())
    values = np.array([subgroup_results[k].get(metric_key, 0.0) for k in names], dtype=np.float64)

    worst_idx = int(np.argmin(values))
    worst_val = float(values[worst_idx])
    worst_name = names[worst_idx]
    mean_val = float(np.mean(values))
    p10_val = float(np.percentile(values, 10))

    return {
        "metric": metric_key,
        "mean": mean_val,
        "worst": worst_val,
        "worst_subgroup": worst_name,
        "p10": p10_val,
        "subgroups": {k: float(v.get(metric_key, 0.0)) for k, v in subgroup_results.items()},
    }
