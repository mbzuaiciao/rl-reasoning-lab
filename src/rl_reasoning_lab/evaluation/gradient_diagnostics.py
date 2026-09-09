"""Gradient diagnostics, numerical stability primitives, and NaN forensics.

Provides:
- compute_gradient_diagnostics: calculates global norm, layer norms, zero-grad fraction, NaN/Inf checks
- VanishingGradientDetector & ExplodingGradientDetector
- Numerically stable probability ratios and safe entropy formulations
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def compute_gradient_diagnostics(model: nn.Module) -> dict[str, Any]:
    """Compute comprehensive diagnostic metrics across all model parameter gradients."""
    total_norm_sq = 0.0
    max_grad = 0.0
    zero_elements = 0
    total_elements = 0
    has_nan_inf = False
    per_layer_norms: dict[str, float] = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()
        numel = grad.numel()
        total_elements += numel

        if not torch.isfinite(grad).all():
            has_nan_inf = True

        # Layer norm
        layer_norm = float(torch.linalg.norm(grad).item())
        per_layer_norms[name] = layer_norm
        total_norm_sq += layer_norm**2

        # Max absolute gradient
        abs_max = float(torch.max(torch.abs(grad)).item())
        if abs_max > max_grad:
            max_grad = abs_max

        # Zero gradients count
        zeros = int((torch.abs(grad) < 1e-12).sum().item())
        zero_elements += zeros

    global_norm = float(np.sqrt(total_norm_sq))
    zero_fraction = float(zero_elements / max(1, total_elements))

    return {
        "global_grad_norm": global_norm,
        "max_grad_component": max_grad,
        "zero_grad_fraction": zero_fraction,
        "has_nan_inf": has_nan_inf,
        "total_grad_elements": total_elements,
        "per_layer_norms": per_layer_norms,
    }


class VanishingGradientDetector:
    """Detects when learning signals vanish through deep architectures or saturated activations."""

    def __init__(self, min_norm_threshold: float = 1e-6, max_zero_fraction: float = 0.85):
        self.min_norm_threshold = min_norm_threshold
        self.max_zero_fraction = max_zero_fraction

    def check(self, diagnostics: dict[str, Any]) -> tuple[bool, str]:
        norm = diagnostics.get("global_grad_norm", 0.0)
        zeros = diagnostics.get("zero_grad_fraction", 0.0)

        if norm < self.min_norm_threshold:
            return (
                True,
                f"Vanishing gradients detected: global norm {norm:.2e} < threshold {self.min_norm_threshold:.2e}",
            )
        if zeros > self.max_zero_fraction:
            return (
                True,
                f"Excessive dead parameters: {zeros * 100:.1f}% of gradient elements are zero.",
            )
        return False, "healthy"


class ExplodingGradientDetector:
    """Detects when gradient updates become destructively large prior to NaN collapse."""

    def __init__(self, max_norm_threshold: float = 50.0):
        self.max_norm_threshold = max_norm_threshold

    def check(self, diagnostics: dict[str, Any]) -> tuple[bool, str]:
        if diagnostics.get("has_nan_inf", False):
            return True, "CRITICAL: Non-finite gradient (NaN/Inf) detected."
        norm = diagnostics.get("global_grad_norm", 0.0)
        if norm > self.max_norm_threshold:
            return (
                True,
                f"Exploding gradients detected: global norm {norm:.2f} > threshold {self.max_norm_threshold:.2f}",
            )
        return False, "healthy"


def safe_log_ratio(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    max_log_diff: float = 10.0,
) -> torch.Tensor:
    """Compute importance ratio exp(new_log_p - old_log_p) with numerical clipping.

    Prevents floating-point overflow when policy updates move rapidly.
    """
    diff = new_log_probs - old_log_probs
    clipped_diff = torch.clamp(diff, -max_log_diff, max_log_diff)
    return torch.exp(clipped_diff)


def safe_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute categorical policy entropy avoiding log(0) numerical NaN."""
    safe_p = torch.clamp(probs, min=eps, max=1.0)
    return -(safe_p * torch.log(safe_p)).sum(dim=-1)
