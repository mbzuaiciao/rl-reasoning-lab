"""Base verifier interface and standardized evaluation results.

Establishes a strict boundary between training rewards and evaluation metrics:
1. Every verifier outputs a standardized VerifierResult.
2. Results expose score, pass/fail, confidence, reason codes, properties checked,
   metadata, and explicit abstention flags.
3. Training code receives signals solely through BaseVerifier, avoiding internal logic coupling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample


@dataclass(frozen=True)
class VerifierResult:
    """Standardized verification outcome for a single candidate completion."""

    score: float
    passed: bool
    confidence: float | None = None
    reason_code: str = "OK"
    metadata: dict[str, Any] = field(default_factory=dict)
    properties_checked: list[str] = field(default_factory=list)
    abstain: bool = False


class BaseVerifier(ABC):
    """Abstract base class for all reasoning verifiers and reward proxies."""

    @property
    def name(self) -> str:
        """Human-readable identifier for this verifier."""
        return self.__class__.__name__

    @abstractmethod
    def verify(
        self,
        completion: str,
        example: ReasoningExample,
    ) -> VerifierResult:
        """Evaluate a completion against the example and produce a VerifierResult."""
        raise NotImplementedError

    def batch_verify(
        self,
        completions: list[str],
        examples: list[ReasoningExample],
    ) -> list[VerifierResult]:
        """Evaluate a batch of completions against corresponding examples."""
        if len(completions) != len(examples):
            raise ValueError(
                f"Batch size mismatch: {len(completions)} completions vs {len(examples)} examples"
            )
        return [self.verify(c, ex) for c, ex in zip(completions, examples, strict=False)]
