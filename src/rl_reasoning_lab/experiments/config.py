"""Experiment configuration, manifest generation, and reproducibility management.

Provides:
- ExperimentManifest: full specification of experimental parameters
- deterministic_run_id: canonical deterministic run identifiers
- ExperimentResult: structured storage for metrics, configs, and replications
- Result serialization/deserialization to JSON
- SplitManager: rigorous isolation between Train, Validation (model selection), and Test splits
"""

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExperimentManifest:
    """Rigorous, self-contained specification of an experiment configuration.

    Captures all hyperparameters, seeds, and conditions to ensure exact reconstruction.
    """

    name: str
    seed: int
    env_seed: int
    method: str
    model_config: dict[str, Any] = field(default_factory=dict)
    optimizer_config: dict[str, Any] = field(default_factory=dict)
    training_budget: dict[str, Any] = field(default_factory=dict)
    verifier_type: str = "none"
    reward_weights: dict[str, float] = field(default_factory=dict)
    evaluation_version: str = "v1"
    git_commit: str = "clean"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentManifest":
        return cls(**d)

    def config_hash(self, length: int = 8) -> str:
        """Compute stable SHA256 hash of configuration excluding run seed and git commit."""
        core_dict = {
            "name": self.name,
            "method": self.method,
            "model_config": self.model_config,
            "optimizer_config": self.optimizer_config,
            "training_budget": self.training_budget,
            "verifier_type": self.verifier_type,
            "reward_weights": self.reward_weights,
            "evaluation_version": self.evaluation_version,
            "tags": sorted(self.tags),
        }
        canonical_json = json.dumps(core_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:length]


def deterministic_run_id(manifest: ExperimentManifest) -> str:
    """Generate a clean, unambiguous, reproducible identifier for a run.

    Format: {method}-{tag_string}-s{seed}-{config_hash}
    """
    clean_method = manifest.method.replace(" ", "_").replace("/", "_").lower()
    sorted_tags = sorted(manifest.tags) if manifest.tags else []
    tag_str = "-".join(sorted_tags) if sorted_tags else "base"
    c_hash = manifest.config_hash(length=6)
    return f"{clean_method}-{tag_str}-s{manifest.seed}-{c_hash}"


@dataclass
class ExperimentResult:
    """Structured container for experiment outputs and replications."""

    manifest: ExperimentManifest
    run_id: str
    metrics: dict[str, float]
    seed_runs: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "run_id": self.run_id,
            "metrics": self.metrics,
            "seed_runs": self.seed_runs,
            "traces": self.traces,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentResult":
        manifest = ExperimentManifest.from_dict(d["manifest"])
        return cls(
            manifest=manifest,
            run_id=d["run_id"],
            metrics=d.get("metrics", {}),
            seed_runs=d.get("seed_runs", []),
            traces=d.get("traces", []),
            timestamp=d.get("timestamp", ""),
            duration_seconds=float(d.get("duration_seconds", 0.0)),
        )


def _json_serializable(val: Any) -> Any:
    """Helper to convert NumPy primitives to standard Python types."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _json_serializable(v) for k, v in val.items()}
    if isinstance(val, list | tuple):
        return [_json_serializable(v) for v in val]
    return val


def save_experiment_result(result: ExperimentResult, filepath: str | Path) -> None:
    """Save experiment result to disk as human- and machine-readable JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean_dict = _json_serializable(result.to_dict())
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2)


def load_experiment_result(filepath: str | Path) -> ExperimentResult:
    """Load experiment result from JSON file."""
    p = Path(filepath)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return ExperimentResult.from_dict(data)


class SplitManager:
    """Manages independent train, validation, and test splits with strict isolation.

    Guarantees:
    - Train split is used exclusively for parameter updates.
    - Validation split is used exclusively for hyperparameter tuning / checkpoint selection.
    - Test split is evaluated strictly ONCE at the end of the experiment, preventing selection contamination.
    """

    def __init__(
        self,
        base_seed: int = 42,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ):
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        self.base_seed = base_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Separate seeds for each split generator
        self.train_seed = base_seed + 1000
        self.val_seed = base_seed + 2000
        self.test_seed = base_seed + 3000

        self._test_queried_count = 0

    def partition_indices(self, total_items: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Deterministically partition indices into train, val, test subsets."""
        rng = np.random.default_rng(self.base_seed)
        shuffled = rng.permutation(total_items)

        n_train = int(np.floor(self.train_ratio * total_items))
        n_val = int(np.floor(self.val_ratio * total_items))

        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train : n_train + n_val]
        test_idx = shuffled[n_train + n_val :]
        return train_idx, val_idx, test_idx

    def record_test_access(self) -> int:
        """Track test set queries to alert on repeated test set tuning (contamination)."""
        self._test_queried_count += 1
        return self._test_queried_count

    @property
    def test_query_count(self) -> int:
        return self._test_queried_count
