"""Unit and invariant tests for Phase 9: Research Engineering, Scaling, and Debugging."""

import os
import random
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from rl_reasoning_lab.evaluation.gradient_diagnostics import (
    safe_entropy,
    safe_log_ratio,
)
from rl_reasoning_lab.testing.fault_injection import (
    ShapeBroadcastingBugInjector,
)
from rl_reasoning_lab.training.checkpoint import (
    CheckpointData,
    CheckpointManager,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_atomic_checkpoint,
)
from rl_reasoning_lab.training.run import (
    TrainingConfig,
    validate_training_config,
)
from rl_reasoning_lab.utils.logging import (
    HealthMonitor,
    _sanitize_for_json,
)


class TestConfigurationValidation:
    """Test fail-fast configuration validation guards."""

    def test_valid_config_passes(self):
        cfg = TrainingConfig()
        validate_training_config(cfg)

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="Invalid batch_size"):
            validate_training_config(TrainingConfig(batch_size=0))

    def test_microbatch_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible by microbatch_size"):
            validate_training_config(TrainingConfig(batch_size=16, microbatch_size=5))

    def test_seq_len_exceeds_context_raises(self):
        with pytest.raises(ValueError, match="exceeds model context length"):
            validate_training_config(TrainingConfig(max_seq_len=128, model_context_len=64))

    def test_reference_kl_without_model_raises(self):
        with pytest.raises(ValueError, match="Reference KL regularization is enabled"):
            validate_training_config(TrainingConfig(kl_coeff=0.1, has_reference_model=False))

    def test_process_reward_without_verifier_raises(self):
        with pytest.raises(ValueError, match="Process rewards are enabled"):
            validate_training_config(
                TrainingConfig(process_reward=True, has_process_verifier=False)
            )

    def test_eval_seed_equals_train_seed_raises(self):
        with pytest.raises(ValueError, match="Evaluation seed .* equals training seed"):
            validate_training_config(TrainingConfig(seed=42, eval_seed=42))


class TestAtomicCheckpointingAndResume:
    """Test atomic checkpoint persistence, RNG restoration, and crash resilience."""

    def test_atomic_checkpoint_roundtrip(self):
        model = nn.Linear(4, 2)
        opt = optim.Adam(model.parameters(), lr=0.01)

        # Take one step to populate optimizer state
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        ckpt_data = CheckpointData(
            model_state=model.state_dict(),
            optimizer_state=opt.state_dict(),
            global_step=10,
            env_step=160,
            rng_states=capture_rng_state(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            save_atomic_checkpoint(ckpt_data, ckpt_path)
            loaded = load_checkpoint(ckpt_path)

            assert loaded.global_step == 10
            assert loaded.env_step == 160
            assert loaded.schema_version == "v1"

            # Check model weights identical
            for k in model.state_dict():
                assert torch.equal(model.state_dict()[k], loaded.model_state[k])

    def test_rng_state_exact_restoration(self):
        # Set initial seed and capture state
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

        saved_rng = capture_rng_state()

        # Generate numbers from stream 1
        rand_py_1 = [random.random() for _ in range(5)]
        rand_np_1 = np.random.rand(5).tolist()
        rand_th_1 = torch.randn(5).tolist()

        # Generate more numbers to advance state
        _ = [random.random() for _ in range(10)]
        _ = np.random.rand(10)
        _ = torch.randn(10)

        # Restore RNG state
        restore_rng_state(saved_rng)

        # Generate numbers again from restored stream
        rand_py_2 = [random.random() for _ in range(5)]
        rand_np_2 = np.random.rand(5).tolist()
        rand_th_2 = torch.randn(5).tolist()

        assert rand_py_1 == rand_py_2
        assert rand_np_1 == rand_np_2
        assert rand_th_1 == rand_th_2

    def test_checkpoint_manager_save_and_rolling_deletion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_last_n=2)
            model = nn.Linear(2, 2)
            opt = optim.SGD(model.parameters(), lr=0.1)

            p1 = mgr.save(global_step=1, env_step=10, model=model, optimizer=opt)
            p2 = mgr.save(global_step=2, env_step=20, model=model, optimizer=opt)
            p3 = mgr.save(global_step=3, env_step=30, model=model, optimizer=opt)

            # keep_last_n=2 -> p1 should be deleted, p2 and p3 should exist
            assert not p1.exists()
            assert p2.exists()
            assert p3.exists()

            latest = mgr.load_latest()
            assert latest.global_step == 3


class TestMicrobatchGradientEquivalence:
    """Test that microbatch gradient accumulation matches full batch execution exactly."""

    def test_gradient_equivalence_single_vs_accumulated(self):
        torch.manual_seed(42)
        model_single = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        torch.manual_seed(42)
        model_micro = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

        inputs = torch.randn(16, 8)
        targets = torch.randint(0, 4, (16,))

        # 1. Full batch execution
        opt_single = optim.SGD(model_single.parameters(), lr=0.1)
        opt_single.zero_grad()
        loss_single = nn.CrossEntropyLoss()(model_single(inputs), targets)
        loss_single.backward()

        # 2. 4 microbatches of size 4
        opt_micro = optim.SGD(model_micro.parameters(), lr=0.1)
        opt_micro.zero_grad()
        n_micro = 4
        micro_sz = 4
        for i in range(n_micro):
            sub_in = inputs[i * micro_sz : (i + 1) * micro_sz]
            sub_tgt = targets[i * micro_sz : (i + 1) * micro_sz]
            sub_loss = nn.CrossEntropyLoss(reduction="sum")(model_micro(sub_in), sub_tgt) / 16.0
            sub_loss.backward()

        # Compare gradients across all parameters
        for p_s, p_m in zip(model_single.parameters(), model_micro.parameters(), strict=True):
            assert torch.allclose(p_s.grad, p_m.grad, atol=1e-6)


class TestBroadcastingAndShapeGuards:
    """Test shape validation catching silent broadcasting bugs."""

    def test_shape_broadcasting_bug_detected(self):
        # Correct shapes: [B, T]
        B, T = 4, 8
        advantages = torch.ones(B, T)
        corrupted = ShapeBroadcastingBugInjector.corrupt_advantage_shape(advantages)
        assert corrupted.shape == (B,)

        # Assert shape guard catches mismatch before training
        def loss_guard(log_p: torch.Tensor, adv: torch.Tensor):
            if log_p.shape != adv.shape:
                raise ValueError(
                    f"Shape mismatch: log_probs {log_p.shape} vs advantages {adv.shape}"
                )
            return (log_p * adv).sum()

        log_probs = torch.randn(B, T)
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_guard(log_probs, corrupted)


class TestLoggerDetachmentAndHealthMonitors:
    """Test structured logger memory safety and health monitors."""

    def test_tensor_detachment_in_logger(self):
        # Create tensor with computation graph
        x = torch.tensor([3.1415], requires_grad=True)
        y = x * 2.0
        assert y.grad_fn is not None

        sanitized = _sanitize_for_json(y)
        assert isinstance(sanitized, float)
        assert sanitized == pytest.approx(6.283, abs=1e-3)

    def test_health_monitor_catches_nan(self):
        monitor = HealthMonitor()
        with pytest.raises(ValueError, match="CRITICAL INVARIANT.*Metric 'loss' is non-finite"):
            monitor.check_step(step=1, metrics={"loss": float("nan")})

    def test_health_monitor_catches_zero_action_tokens(self):
        monitor = HealthMonitor()
        with pytest.raises(ValueError, match="Batch contains zero action tokens"):
            monitor.check_step(step=1, metrics={"loss": 0.5, "action_token_count": 0})

    def test_numerical_stability_safe_ratio_and_entropy(self):
        new_lp = torch.tensor([100.0, -100.0])
        old_lp = torch.tensor([0.0, 0.0])
        ratio = safe_log_ratio(new_lp, old_lp, max_log_diff=5.0)
        assert torch.isfinite(ratio).all()
        # Clamped difference <= 5.0 -> exp(5.0) ~ 148.4
        assert ratio[0] <= 150.0

        zero_p = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        ent = safe_entropy(zero_p)
        assert torch.isfinite(ent).all()
        assert ent[0] == pytest.approx(0.0, abs=1e-4)
