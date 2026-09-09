"""Phase 8 Experiment: Seed Sensitivity, Confidence Intervals, and Best-Seed Bias.

Evaluates:
- How empirical estimates and confidence intervals evolve across n in {3, 5, 10, 20} seeds
- Best-seed bias: contrasting cherry-picked best seed vs median vs mean vs worst seed
- Mean vs median vs trimmed mean when RL policy training exhibits high variance or failure runs
- Machine-readable result logging with ExperimentManifest and ExperimentResult
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_reasoning_lab.evaluation.statistics import (
    bootstrap_mean_ci,
    bootstrap_median_ci,
    bootstrap_trimmed_mean_ci,
    distribution_summary,
)
from rl_reasoning_lab.experiments.config import (
    ExperimentManifest,
    ExperimentResult,
    deterministic_run_id,
    save_experiment_result,
)


def simulate_rl_training_seed(
    seed: int,
    steps: int = 150,
) -> dict[str, float]:
    """Simulate a controlled RL reasoning training run with realistic seed-dependent dynamics.

    A small multi-layer policy optimizes a 3-step reasoning task. Seeds induce different
    initializations and stochastic rollout paths:
    - ~65% of seeds converge reliably to valid reasoning (high accuracy, high path validity).
    - ~25% of seeds get trapped in shortcut/local optima (moderate accuracy, poor validity).
    - ~10% of seeds suffer policy entropy collapse early on (catastrophic failure run).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_in, d_hidden, d_out = 16, 32, 4
    model = nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    # Synthetic reasoning targets
    rng = np.random.default_rng(seed)
    task_difficulty = rng.uniform(0.8, 1.2)

    accuracies = []
    kls = []
    entropies = []
    validities = []

    for step in range(steps):
        inputs = torch.randn(16, d_in)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # Simulated ground-truth alignment signal with stochastic reward noise
        target = torch.randint(0, d_out, (16,))
        loss = nn.CrossEntropyLoss()(logits, target) - 0.05 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Compute empirical performance proxies
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = float((preds == target).float().mean().item())
            # Add trajectory progression dynamics
            progress = min(1.0, step / 100.0)
            # Seed-specific bifurcation
            if seed % 10 == 7:
                # Catastrophic failure seed (early collapse)
                acc = float(max(0.1, 0.25 - 0.1 * progress + rng.normal(0, 0.03)))
                valid = float(max(0.05, 0.15 - 0.05 * progress))
            elif seed % 4 == 0:
                # Shortcut mode (medium accuracy, low path validity)
                acc = float(min(0.70, 0.35 + 0.35 * progress + rng.normal(0, 0.04)))
                valid = float(min(0.40, 0.20 + 0.15 * progress))
            else:
                # Normal convergence
                acc = float(
                    min(0.95, 0.30 + 0.62 * progress / task_difficulty + rng.normal(0, 0.03))
                )
                valid = float(min(0.92, 0.25 + 0.65 * progress))

            accuracies.append(acc)
            validities.append(valid)
            kls.append(float(0.01 + 0.15 * progress * (1.0 if seed % 10 != 7 else 0.8)))
            entropies.append(float(entropy.item()))

    final_acc = float(np.mean(accuracies[-10:]))
    max_acc = float(np.max(accuracies))
    final_valid = float(np.mean(validities[-10:]))
    final_kl = float(np.mean(kls[-10:]))
    final_entropy = float(np.mean(entropies[-10:]))
    # Simulated calibration error (higher when overconfident or collapsed)
    ece = float(
        0.03 + 0.20 * abs(final_acc - 0.70) if seed % 10 == 7 else 0.04 + rng.uniform(0.01, 0.04)
    )

    return {
        "seed": seed,
        "final_accuracy": final_acc,
        "max_accuracy": max_acc,
        "path_validity": final_valid,
        "kl_divergence": final_kl,
        "entropy": final_entropy,
        "ece": ece,
    }


def run_seed_sensitivity_study() -> ExperimentResult:
    start_time = time.time()
    print("=" * 75)
    print("Phase 8: Seed Sensitivity, Best-Seed Bias, and Distribution Analysis")
    print("=" * 75)

    all_seeds = [101 + i * 17 for i in range(20)]
    seed_results = []
    print(f"Executing {len(all_seeds)} distinct random seeds...")
    for s in all_seeds:
        res = simulate_rl_training_seed(s)
        seed_results.append(res)

    final_accuracies = [r["final_accuracy"] for r in seed_results]

    # Evaluate estimation across n in {3, 5, 10, 20}
    subsets = [3, 5, 10, 20]
    print("\n--- Confidence Interval Progression Across Seed Sample Sizes ---")
    print(f"{'Seeds (n)':<10} | {'Mean Acc':<10} | {'95% CI (Bootstrap)':<22} | {'CI Width':<10}")
    print("-" * 60)

    progression_data = {}
    for n in subsets:
        sample = final_accuracies[:n]
        mean_est, lower, upper = bootstrap_mean_ci(sample, num_bootstrap=2000, seed=42)
        ci_width = upper - lower
        progression_data[f"n_{n}"] = {
            "mean": mean_est,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_width": ci_width,
        }
        print(f"{n:<10} | {mean_est:<10.3f} | [{lower:<8.3f}, {upper:<8.3f}]   | {ci_width:<10.3f}")

    # Best-Seed Bias Demonstration
    best_seed_acc = float(np.max(final_accuracies))
    worst_seed_acc = float(np.min(final_accuracies))
    median_seed_acc = float(np.median(final_accuracies))
    mean_acc = float(np.mean(final_accuracies))

    print("\n--- Best-Seed Bias Demonstration ---")
    print(
        f"Cherry-Picked Best Seed : {best_seed_acc:.3f} (+{(best_seed_acc - mean_acc):.3f} vs mean)"
    )
    print(f"Mean Across 20 Seeds    : {mean_acc:.3f}")
    print(f"Median Across 20 Seeds  : {median_seed_acc:.3f}")
    print(f"Worst Seed              : {worst_seed_acc:.3f} (Catastrophic collapse / local minimum)")

    # Deliberately misleading summary vs scientifically honest summary
    print("\n[Deliberately Misleading Summary]")
    print(
        f'  "Our proposed RL method achieves up to {best_seed_acc * 100:.1f}% accuracy on the reasoning task."'
    )
    print("[Scientifically Honest Summary]")
    mean_est, low, high = bootstrap_mean_ci(final_accuracies, num_bootstrap=2000, seed=42)
    med_est, m_low, m_high = bootstrap_median_ci(final_accuracies, num_bootstrap=2000, seed=42)
    t_est, t_low, t_high = bootstrap_trimmed_mean_ci(
        final_accuracies, proportion_to_cut=0.1, seed=42
    )
    print(
        f'  "Across 20 random seeds, the method achieved mean accuracy {mean_est * 100:.1f}% [95% CI: {low * 100:.1f}%, {high * 100:.1f}%],'
    )
    print(
        f'   median {med_est * 100:.1f}% [95% CI: {m_low * 100:.1f}%, {m_high * 100:.1f}%], and 10%-trimmed mean {t_est * 100:.1f}% [95% CI: {t_low * 100:.1f}%, {t_high * 100:.1f}%]."'
    )

    # Distribution summary
    dist_stats = distribution_summary(final_accuracies)
    print("\n--- Distribution Metrics Across 20 Seeds ---")
    for k, v in dist_stats.items():
        print(f"  {k:<12}: {v:.4f}")

    # Build manifest and result object
    manifest = ExperimentManifest(
        name="phase8_seed_sensitivity",
        seed=42,
        env_seed=101,
        method="PPO_Reasoning_Sim",
        training_budget={"seeds": 20, "steps_per_seed": 150},
        tags=["phase8", "seed_sensitivity", "bootstrap"],
    )

    metrics = {
        "best_seed_accuracy": best_seed_acc,
        "worst_seed_accuracy": worst_seed_acc,
        "mean_accuracy": mean_acc,
        "median_accuracy": median_seed_acc,
        "ci_width_n3": progression_data["n_3"]["ci_width"],
        "ci_width_n20": progression_data["n_20"]["ci_width"],
        "ci_tightening_ratio": progression_data["n_3"]["ci_width"]
        / max(1e-6, progression_data["n_20"]["ci_width"]),
        "best_seed_inflation": best_seed_acc - mean_acc,
    }

    result = ExperimentResult(
        manifest=manifest,
        run_id=deterministic_run_id(manifest),
        metrics=metrics,
        seed_runs=seed_results,
        duration_seconds=time.time() - start_time,
    )

    save_experiment_result(result, "experiments/results/phase8_seed_sensitivity.json")
    print("\nSaved experiment result to experiments/results/phase8_seed_sensitivity.json")
    return result


if __name__ == "__main__":
    run_seed_sensitivity_study()
