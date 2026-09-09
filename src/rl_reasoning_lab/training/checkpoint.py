"""Atomic checkpoint management and RNG state serialization.

Provides:
- CheckpointData: complete container for model, optimizer, scheduler, RNG, and running stats
- Atomic checkpoint writing via temporary file sync and atomic rename
- Complete RNG state capture and restoration (Python, NumPy, PyTorch CPU/CUDA)
- CheckpointManager: cadenced saves, best-checkpoint tracking, and validation
"""

import datetime
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def capture_rng_state() -> dict[str, Any]:
    """Capture complete pseudo-random number generator states across all engines."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_states: dict[str, Any]) -> None:
    """Restore exact pseudo-random number generator states across all engines."""
    if "python" in rng_states and rng_states["python"] is not None:
        random.setstate(rng_states["python"])
    if "numpy" in rng_states and rng_states["numpy"] is not None:
        np.random.set_state(rng_states["numpy"])
    if "torch_cpu" in rng_states and rng_states["torch_cpu"] is not None:
        torch.set_rng_state(rng_states["torch_cpu"])
    if (
        "torch_cuda" in rng_states
        and rng_states["torch_cuda"] is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])


@dataclass
class CheckpointData:
    """Complete specification of training state for exact deterministic resumption."""

    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    global_step: int
    env_step: int
    rng_states: dict[str, Any]
    critic_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    running_stats: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "v1"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointData":
        return cls(**d)


def save_atomic_checkpoint(checkpoint_data: CheckpointData, filepath: str | Path) -> None:
    """Save checkpoint atomically to prevent corrupted files if interrupted.

    Workflow:
    1. Write serialized data to temporary file in the target directory.
    2. Flush and fsync file descriptor to guarantee physical persistence.
    3. Atomically rename temporary file to target path via os.replace.
    """
    target_path = Path(filepath).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Use same directory to ensure atomic rename across filesystem boundaries
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=f".{target_path.name}.tmp.",
        dir=target_path.parent,
    )

    try:
        with os.fdopen(temp_fd, "wb") as f:
            torch.save(checkpoint_data.to_dict(), f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def load_checkpoint(filepath: str | Path, map_location: str = "cpu") -> CheckpointData:
    """Load and validate checkpoint from disk."""
    target_path = Path(filepath).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {target_path}")

    raw_dict = torch.load(target_path, map_location=map_location, weights_only=False)

    schema_version = raw_dict.get("schema_version", "v0")
    if schema_version != "v1":
        raise ValueError(
            f"Unsupported checkpoint schema version '{schema_version}'. Expected 'v1'."
        )

    return CheckpointData.from_dict(raw_dict)


class CheckpointManager:
    """Coordinates checkpoint saving, periodic retention, and deterministic recovery."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_interval_steps: int = 50,
        keep_last_n: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval_steps = save_interval_steps
        self.keep_last_n = keep_last_n
        self.saved_checkpoints: list[Path] = []
        self.best_metric: float = float("-inf")
        self.best_checkpoint_path: Path | None = None

    def save(
        self,
        global_step: int,
        env_step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        critic: nn.Module | None = None,
        scheduler: Any | None = None,
        running_stats: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        val_metric: float | None = None,
        is_best: bool = False,
    ) -> Path:
        """Create an atomic checkpoint and manage rolling history."""
        ckpt_data = CheckpointData(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            global_step=global_step,
            env_step=env_step,
            rng_states=capture_rng_state(),
            critic_state=critic.state_dict() if critic is not None else None,
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            running_stats=running_stats or {},
            config=config or {},
        )

        filename = f"checkpoint_step_{global_step:06d}.pt"
        ckpt_path = self.checkpoint_dir / filename
        save_atomic_checkpoint(ckpt_data, ckpt_path)
        self.saved_checkpoints.append(ckpt_path)

        # Track best checkpoint if metric provided
        if val_metric is not None and val_metric > self.best_metric:
            self.best_metric = val_metric
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            save_atomic_checkpoint(ckpt_data, best_path)
            self.best_checkpoint_path = best_path
        elif is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            save_atomic_checkpoint(ckpt_data, best_path)
            self.best_checkpoint_path = best_path

        # Clean up older checkpoints exceeding keep_last_n
        if len(self.saved_checkpoints) > self.keep_last_n:
            old_ckpt = self.saved_checkpoints.pop(0)
            if old_ckpt.exists() and old_ckpt != self.best_checkpoint_path:
                try:
                    old_ckpt.unlink()
                except OSError:
                    pass

        return ckpt_path

    def load_latest(self) -> CheckpointData:
        """Load the most recent checkpoint available in directory."""
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No valid checkpoints found in {self.checkpoint_dir}")
        return load_checkpoint(ckpts[-1])

    def restore_into(
        self,
        checkpoint: CheckpointData,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        critic: nn.Module | None = None,
        restore_rng: bool = True,
    ) -> tuple[int, int]:
        """Restore checkpoint state into existing module instances.

        Returns:
            (global_step, env_step)
        """
        model.load_state_dict(checkpoint.model_state)

        if optimizer is not None:
            if not checkpoint.optimizer_state:
                raise ValueError("Checkpoint is missing optimizer_state. Cannot resume training.")
            optimizer.load_state_dict(checkpoint.optimizer_state)

        if critic is not None and checkpoint.critic_state is not None:
            critic.load_state_dict(checkpoint.critic_state)

        if restore_rng:
            restore_rng_state(checkpoint.rng_states)

        return checkpoint.global_step, checkpoint.env_step
