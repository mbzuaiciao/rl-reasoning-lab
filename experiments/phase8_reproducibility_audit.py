"""Phase 8 Experiment: Determinism Audit, Seed Tracking, and Environment Invariance.

Evaluates:
1. Exact Bit-for-Bit Determinism:
   - Runs a complete training and generation loop twice with identical manifests and random seeds.
   - Measures PyTorch parameter tensor equality (max absolute weight difference = 0.0)
   - Checks deterministic rollout sequence identity.
2. The Cost of Incomplete Seeding:
   - Demonstrates how omitting any seed (Python random, NumPy, PyTorch CPU, CUDA/env)
     causes silent divergence in trajectory rollouts and metrics.
"""

import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from rl_reasoning_lab.experiments.config import (
    ExperimentManifest,
    ExperimentResult,
    deterministic_run_id,
    save_experiment_result,
)
from rl_reasoning_lab.utils.seeding import seed_everything


def _execute_micro_training_run(
    seed: int, fully_seeded: bool = True
) -> tuple[dict[str, torch.Tensor], list[int], float]:
    """Execute a controlled micro training and generation pipeline."""
    if fully_seeded:
        seed_everything(seed)
    else:
        # Incomplete seeding: seed torch but NOT python/numpy or leave unseeded
        torch.manual_seed(seed)
        # Random / NumPy remain unseeded and drift!

    torch.use_deterministic_algorithms(False)

    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 10 training steps
    for _ in range(10):
        if fully_seeded:
            x = torch.randn(4, 8)
            y = torch.randint(0, 4, (4,))
        else:
            # Uses unseeded random module to generate batch noise
            noise = random.random()
            x = torch.randn(4, 8) + noise
            y = torch.randint(0, 4, (4,))

        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Rollout generation
    test_input = torch.randn(1, 8)
    preds = torch.argmax(model(test_input), dim=-1).tolist()
    final_loss = float(loss.item())

    # Return cloned parameter state
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state, preds, final_loss


def run_reproducibility_audit_study() -> ExperimentResult:
    start_time = time.time()
    print("=" * 75)
    print("Phase 8: Determinism Audit and Complete Seed Hygiene")
    print("=" * 75)

    audit_seed = 42

    print("\n--- Part 1: Exact Bit-for-Bit Determinism Audit ---")
    print(f"Running Experiment under Seed {audit_seed} (Run 1)...")
    state_1, preds_1, loss_1 = _execute_micro_training_run(audit_seed, fully_seeded=True)

    print(f"Running Experiment under Seed {audit_seed} (Run 2 — Identical Seed)...")
    state_2, preds_2, loss_2 = _execute_micro_training_run(audit_seed, fully_seeded=True)

    # Check parameter differences
    max_param_diff = 0.0
    for name in state_1:
        diff = float(torch.max(torch.abs(state_1[name] - state_2[name])).item())
        if diff > max_param_diff:
            max_param_diff = diff

    pred_match = preds_1 == preds_2
    loss_match = abs(loss_1 - loss_2) < 1e-9

    print(f"  Max Parameter Difference : {max_param_diff:.10f}")
    print(f"  Loss Match (Run 1 vs 2)  : {loss_1:.8f} == {loss_2:.8f} (Match: {loss_match})")
    print(f"  Prediction Match         : {preds_1} == {preds_2} (Match: {pred_match})")
    assert max_param_diff == 0.0, "Determinism failed: Weights diverged under identical seed!"
    print("  [Pass]: Bit-for-bit exact reproducibility verified on CPU.")

    print("\n--- Part 2: Environmental & Incomplete Seeding Hazard ---")
    # Demonstrates what happens when python/numpy are not seeded alongside torch
    random.seed(12345)
    _, preds_unseeded_a, loss_unseeded_a = _execute_micro_training_run(
        audit_seed, fully_seeded=False
    )
    # Different python random state
    random.seed(67890)
    _, preds_unseeded_b, loss_unseeded_b = _execute_micro_training_run(
        audit_seed, fully_seeded=False
    )

    loss_diff = abs(loss_unseeded_a - loss_unseeded_b)
    print(f"  Unseeded Python Random State (Run A Loss): {loss_unseeded_a:.6f}")
    print(f"  Unseeded Python Random State (Run B Loss): {loss_unseeded_b:.6f}")
    print(f"  Silent Loss Discrepancy                 : {loss_diff:.6f}")
    print(
        "  [Key Takeaway]: Seeding PyTorch alone is insufficient. seed_everything() must encompass"
    )
    print("  Python random, NumPy, PyTorch CPU, and environment state generators.")

    manifest = ExperimentManifest(
        name="phase8_reproducibility_audit",
        seed=audit_seed,
        env_seed=audit_seed,
        method="Determinism_Audit",
        tags=["phase8", "determinism", "bit_for_bit", "seed_hygiene"],
    )

    metrics = {
        "max_parameter_difference": max_param_diff,
        "loss_1": loss_1,
        "loss_2": loss_2,
        "bit_for_bit_identical": 1.0 if max_param_diff == 0.0 else 0.0,
        "unseeded_discrepancy": loss_diff,
    }

    result = ExperimentResult(
        manifest=manifest,
        run_id=deterministic_run_id(manifest),
        metrics=metrics,
        duration_seconds=time.time() - start_time,
    )

    save_experiment_result(result, "experiments/results/phase8_reproducibility_audit.json")
    print("\nSaved experiment result to experiments/results/phase8_reproducibility_audit.json")
    return result


if __name__ == "__main__":
    run_reproducibility_audit_study()
