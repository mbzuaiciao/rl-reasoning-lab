"""Structured logging, metric semantic integrity, and training health monitoring.

Provides:
- StructuredLogger: JSONL streaming logger with automatic tensor detachment
- MetricSemanticRegistry: precise definitions and schema validation for RL metrics
- HealthMonitor: anomaly detection (soft warnings vs hard invariant exceptions)
- save_failure_artifact: compact diagnostic capture on training failures
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class MetricDefinition:
    name: str
    description: str
    expected_range: tuple[float, float]
    unit: str
    is_higher_better: bool


class MetricSemanticRegistry:
    """Canonical registry documenting and validating metric definitions to prevent semantic confusion."""

    _REGISTRY: dict[str, MetricDefinition] = {
        "reward": MetricDefinition(
            name="reward",
            description="Scalar training reward returned by the proxy or process reward function.",
            expected_range=(-10.0, 10.0),
            unit="scalar",
            is_higher_better=True,
        ),
        "oracle_accuracy": MetricDefinition(
            name="oracle_accuracy",
            description="Ground-truth binary task correctness evaluated by oracle verifier.",
            expected_range=(0.0, 1.0),
            unit="ratio",
            is_higher_better=True,
        ),
        "exact_match": MetricDefinition(
            name="exact_match",
            description="Strict string equality between generated completion and target trace.",
            expected_range=(0.0, 1.0),
            unit="ratio",
            is_higher_better=True,
        ),
        "step_validity": MetricDefinition(
            name="step_validity",
            description="Fraction of intermediate reasoning transitions that follow valid environment edges.",
            expected_range=(0.0, 1.0),
            unit="ratio",
            is_higher_better=True,
        ),
        "policy_kl": MetricDefinition(
            name="policy_kl",
            description="Empirical KL divergence between current policy and old rollout policy pi_theta || pi_old.",
            expected_range=(0.0, 10.0),
            unit="nats",
            is_higher_better=False,
        ),
        "reference_kl": MetricDefinition(
            name="reference_kl",
            description="Kullback-Leibler divergence between current policy and frozen SFT reference policy.",
            expected_range=(0.0, 50.0),
            unit="nats",
            is_higher_better=False,
        ),
        "approx_kl": MetricDefinition(
            name="approx_kl",
            description="Schulman approximate KL: 0.5 * mean((log_p_old - log_p_curr)^2).",
            expected_range=(0.0, 10.0),
            unit="nats",
            is_higher_better=False,
        ),
        "entropy": MetricDefinition(
            name="entropy",
            description="Average token-level policy categorical entropy -sum(p * log p).",
            expected_range=(0.0, 20.0),
            unit="nats",
            is_higher_better=True,
        ),
        "clip_fraction": MetricDefinition(
            name="clip_fraction",
            description="Fraction of policy gradient tokens where importance ratio was clipped by PPO epsilon.",
            expected_range=(0.0, 1.0),
            unit="ratio",
            is_higher_better=False,
        ),
    }

    @classmethod
    def get(cls, name: str) -> MetricDefinition | None:
        return cls._REGISTRY.get(name)

    @classmethod
    def validate_metric(cls, name: str, value: float) -> tuple[bool, str]:
        """Validate whether a recorded metric falls within plausible semantic bounds."""
        defn = cls.get(name)
        if defn is None:
            return True, "unregistered"
        min_v, max_v = defn.expected_range
        if not (min_v <= value <= max_v):
            return (
                False,
                f"Metric '{name}' value {value:.4f} is outside expected range [{min_v}, {max_v}]",
            )
        return True, "valid"


def _sanitize_for_json(val: Any) -> Any:
    """Safely convert PyTorch tensors and NumPy types to native Python types without holding computation graphs."""
    if isinstance(val, torch.Tensor):
        # Detach and convert to scalar or list
        detached = val.detach().cpu()
        if detached.numel() == 1:
            return float(detached.item()) if detached.is_floating_point() else int(detached.item())
        return detached.tolist()
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, list | tuple):
        return [_sanitize_for_json(v) for v in val]
    return val


class StructuredLogger:
    """Streams machine-readable JSONL records with strict tensor detachment to prevent autograd graph memory leaks."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self._records: list[dict[str, Any]] = []

    def log_step(self, step: int, metrics: dict[str, Any], flush: bool = True) -> dict[str, Any]:
        """Log a training step dictionary.

        Ensures all tensors are fully detached and serialized.
        """
        elapsed = time.time() - self.start_time
        record = {
            "step": int(step),
            "wall_time": float(elapsed),
            "timestamp": time.time(),
        }

        # Detach and sanitize all incoming metrics
        clean_metrics = _sanitize_for_json(metrics)
        if isinstance(clean_metrics, dict):
            record.update(clean_metrics)

        self._records.append(record)

        if flush:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return record

    def read_records(self) -> list[dict[str, Any]]:
        """Read all logged JSONL records from disk."""
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records


class HealthMonitor:
    """Monitors training metrics for numerical stability, soft anomalies, and hard invariant violations."""

    def __init__(
        self,
        max_kl_spike_ratio: float = 5.0,
        min_entropy: float = 0.001,
        max_grad_norm: float = 100.0,
    ):
        self.max_kl_spike_ratio = max_kl_spike_ratio
        self.min_entropy = min_entropy
        self.max_grad_norm = max_grad_norm
        self.recent_kls: list[float] = []

    def check_step(
        self,
        step: int,
        metrics: dict[str, float],
        raise_on_invariant: bool = True,
    ) -> list[str]:
        """Run health checks on current step metrics.

        Returns list of warnings/alerts. Raises ValueError if a hard invariant fails.
        """
        alerts = []

        # 1. Hard Invariant: Loss and metrics must be finite
        for k, v in metrics.items():
            if not np.isfinite(v):
                msg = f"[CRITICAL INVARIANT] Step {step}: Metric '{k}' is non-finite (NaN or Inf): {v}"
                if raise_on_invariant:
                    raise ValueError(msg)
                alerts.append(msg)

        # 2. Hard Invariant: Action mask must not be completely zero if provided
        if "action_token_count" in metrics and metrics["action_token_count"] == 0:
            msg = f"[CRITICAL INVARIANT] Step {step}: Batch contains zero action tokens. Gradient update is undefined."
            if raise_on_invariant:
                raise ValueError(msg)
            alerts.append(msg)

        # 3. Soft Anomaly: Entropy collapse
        if "entropy" in metrics and metrics["entropy"] < self.min_entropy:
            alerts.append(
                f"[WARNING] Step {step}: Policy entropy collapsed below {self.min_entropy}: {metrics['entropy']:.6f}"
            )

        # 4. Soft Anomaly: Exploding gradient norm
        if "grad_norm" in metrics and metrics["grad_norm"] > self.max_grad_norm:
            alerts.append(
                f"[WARNING] Step {step}: Gradient norm exceeded safe threshold {self.max_grad_norm}: {metrics['grad_norm']:.2f}"
            )

        # 5. Soft Anomaly: KL divergence sudden spike
        if "kl" in metrics or "policy_kl" in metrics:
            kl_val = metrics.get("kl", metrics.get("policy_kl", 0.0))
            if len(self.recent_kls) >= 10:
                historical_median = float(np.median(self.recent_kls[-20:]))
                if (
                    historical_median > 0.001
                    and kl_val > self.max_kl_spike_ratio * historical_median
                ):
                    alerts.append(
                        f"[WARNING] Step {step}: Sudden KL spike ({kl_val:.4f} vs historical median {historical_median:.4f})"
                    )
            self.recent_kls.append(kl_val)

        return alerts


def save_failure_artifact(bundle: dict[str, Any], filepath: str | Path) -> None:
    """Serialize a compact diagnostic bundle when a run fails or encounters an anomaly."""
    p = Path(filepath).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    clean_bundle = _sanitize_for_json(bundle)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean_bundle, f, indent=2)
