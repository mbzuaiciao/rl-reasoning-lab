"""Unified training-run abstraction and fail-fast configuration validation.

Provides:
- TrainingConfig: validated, self-documenting dataclass for RL training
- validate_training_config: hard guards catching impossible or corrupt configurations before launch
- TrainingRun: explicit lifecycle orchestrator coordinating models, optimizers, rollouts, verifiers, and health monitors
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

from rl_reasoning_lab.evaluation.gradient_diagnostics import (
    ExplodingGradientDetector,
    VanishingGradientDetector,
    compute_gradient_diagnostics,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.training.checkpoint import CheckpointManager
from rl_reasoning_lab.utils.logging import HealthMonitor, StructuredLogger
from rl_reasoning_lab.utils.seeding import seed_everything


@dataclass
class TrainingConfig:
    """Rigorous configuration specification with fail-fast validation."""

    name: str = "reasoning_ppo"
    seed: int = 42
    env_seed: int = 101
    eval_seed: int = 999
    batch_size: int = 16
    microbatch_size: int = 16
    max_seq_len: int = 32
    model_context_len: int = 64
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    learning_rate: float = 1e-3
    num_episodes: int = 50
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    kl_coeff: float = 0.05
    has_reference_model: bool = True
    process_reward: bool = True
    has_process_verifier: bool = True
    eval_interval: int = 25
    checkpoint_interval: int = 50
    max_grad_norm: float = 1.0
    output_dir: str = "experiments/runs"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_training_config(config: TrainingConfig) -> None:
    """Enforce hard invariants on training configuration before allocating compute."""
    if config.batch_size <= 0:
        raise ValueError(f"Invalid batch_size: {config.batch_size}. Must be > 0.")
    if config.microbatch_size <= 0:
        raise ValueError(f"Invalid microbatch_size: {config.microbatch_size}. Must be > 0.")
    if config.batch_size % config.microbatch_size != 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be divisible by microbatch_size ({config.microbatch_size})."
        )
    if config.max_seq_len > config.model_context_len:
        raise ValueError(
            f"Sequence length ({config.max_seq_len}) exceeds model context length ({config.model_context_len})."
        )
    if config.kl_coeff > 0.0 and not config.has_reference_model:
        raise ValueError(
            "Reference KL regularization is enabled (kl_coeff > 0) but has_reference_model is False."
        )
    if config.process_reward and not config.has_process_verifier:
        raise ValueError(
            "Process rewards are enabled (process_reward=True) but has_process_verifier is False."
        )
    if config.eval_seed == config.seed:
        raise ValueError(
            f"Evaluation seed ({config.eval_seed}) equals training seed ({config.seed}). "
            "Independent evaluation splits are required to prevent benchmark contamination."
        )


class TrainingRun:
    """Explicit lifecycle orchestrator for RL training and reasoning policy optimization."""

    def __init__(self, config: TrainingConfig):
        validate_training_config(config)
        self.config = config

        # Seed environment and torch
        seed_everything(config.seed)

        # File paths
        run_dir = Path(config.output_dir) / f"{config.name}-s{config.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        # Tokenizer & Models
        self.tokenizer = CharTokenizer()
        self.model = TinyTransformer(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=config.model_context_len,
            d_model=config.d_model,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
        )

        # Frozen reference model for KL penalty
        if config.has_reference_model:
            self.reference_model: TinyTransformer | None = TinyTransformer(
                vocab_size=self.tokenizer.vocab_size,
                max_seq_len=config.model_context_len,
                d_model=config.d_model,
                num_layers=config.n_layers,
                num_heads=config.n_heads,
            )
            # Copy weights and freeze
            self.reference_model.load_state_dict(self.model.state_dict())
            for p in self.reference_model.parameters():
                p.requires_grad = False
            self.reference_model.eval()
        else:
            self.reference_model = None

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Managers
        self.logger = StructuredLogger(run_dir / "train_log.jsonl")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=run_dir / "checkpoints",
            save_interval_steps=config.checkpoint_interval,
        )
        self.health_monitor = HealthMonitor()
        self.vanishing_detector = VanishingGradientDetector()
        self.exploding_detector = ExplodingGradientDetector()

        self.global_step = 0
        self.env_step = 0

    def check_reference_model_frozen(self) -> bool:
        """Verify that the reference model has not been accidentally set to trainable."""
        if self.reference_model is None:
            return True
        for name, p in self.reference_model.named_parameters():
            if p.requires_grad:
                raise ValueError(
                    f"CRITICAL FAULT: Reference parameter '{name}' is requires_grad=True! Reference model must remain strictly frozen."
                )
        return True

    def train_step(self, synthetic_advantage: float = 1.0) -> dict[str, float]:
        """Execute one training step coordinating rollout, update, diagnostics, and logging."""
        self.check_reference_model_frozen()

        # Generate a synthetic batch of tokens: shape [B, T]
        B = self.config.batch_size
        T = min(self.config.max_seq_len, 16)
        token_ids = torch.randint(0, self.tokenizer.vocab_size, (B, T))
        action_mask = torch.ones(B, T, dtype=torch.bool)
        # First 4 tokens are prompt
        action_mask[:, :4] = False

        # Compute forward pass
        logits = self.model(token_ids)
        # Assert shape invariant: [B, T, V]
        assert logits.shape == (B, T, self.tokenizer.vocab_size), (
            f"Invalid logits shape: {logits.shape}"
        )

        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)

        # Advantages: shape [B, T] (Strict shape matching!)
        advantages = torch.full((B, T), synthetic_advantage, dtype=torch.float32)

        # Masked policy loss with reference KL if enabled
        loss_tokens = -(selected_log_probs * advantages)
        if self.reference_model is not None:
            with torch.no_grad():
                ref_logits = self.reference_model(token_ids)
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                ref_selected = ref_log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
            kl_penalty = selected_log_probs - ref_selected
            loss_tokens = loss_tokens + self.config.kl_coeff * kl_penalty
        else:
            kl_penalty = torch.zeros_like(selected_log_probs)

        # Masked mean
        active_tokens = action_mask.sum().item()
        if active_tokens == 0:
            raise ValueError("Batch contains zero action tokens. Loss undefined.")

        loss = (loss_tokens * action_mask.float()).sum() / active_tokens

        # Microbatch gradient accumulation
        num_microbatches = self.config.batch_size // self.config.microbatch_size
        micro_size = self.config.microbatch_size

        self.optimizer.zero_grad()
        for m in range(num_microbatches):
            m_start = m * micro_size
            m_end = (m + 1) * micro_size
            sub_loss = (
                loss_tokens[m_start:m_end] * action_mask[m_start:m_end].float()
            ).sum() / active_tokens
            sub_loss.backward(retain_graph=(m < num_microbatches - 1))

        # Gradient diagnostics before clipping
        diag = compute_gradient_diagnostics(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.global_step += 1
        self.env_step += B

        # Check health
        step_metrics = {
            "loss": float(loss.item()),
            "grad_norm": diag["global_grad_norm"],
            "entropy": float(
                -(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean().item()
            ),
            "kl": float(kl_penalty.mean().item()),
            "action_token_count": float(active_tokens),
            "reward": float(synthetic_advantage),
            "oracle_accuracy": float(min(1.0, max(0.0, synthetic_advantage))),
        }

        self.health_monitor.check_step(self.global_step, step_metrics)
        self.logger.log_step(self.global_step, step_metrics)

        # Checkpoint cadence
        if self.global_step % self.config.checkpoint_interval == 0:
            self.checkpoint_manager.save(
                global_step=self.global_step,
                env_step=self.env_step,
                model=self.model,
                optimizer=self.optimizer,
                config=self.config.to_dict(),
            )

        return step_metrics
