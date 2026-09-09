"""Toy learned verifier (shallow neural classifier) on reasoning trace features.

Demonstrates:
1. High IID static accuracy (e.g. >95%) on standard training and evaluation splits.
2. Distribution shift and vulnerability under policy optimization: the optimizer finds
   feature combinations that trigger high predicted probability without satisfying ground truth.
3. Frozen verifier vs. periodically refreshed/co-trained verifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample
from rl_reasoning_lab.verifiers.base import BaseVerifier, VerifierResult
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


class ToyClassifierModel(nn.Module):
    """Small 2-layer MLP classifier predicting trace correctness from extracted features."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs unnormalized logit for validity."""
        return self.net(x)


class ToyLearnedVerifier(BaseVerifier):
    """Learned neural verifier operating on interpretable syntactic and symbolic features."""

    FEATURE_DIM = 8

    def __init__(
        self,
        model: ToyClassifierModel | None = None,
        threshold: float = 0.5,
        frozen: bool = True,
    ) -> None:
        self.model = model if model is not None else ToyClassifierModel(input_dim=self.FEATURE_DIM)
        self.threshold = threshold
        self.frozen = frozen
        if self.frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @classmethod
    def extract_features(cls, completion: str, example: ReasoningExample) -> torch.Tensor:
        """Extract a fixed-size feature vector from (completion, example)."""
        ans, steps, syntax_valid, is_trace = TraceEvaluator.parse_completion(completion)

        # Feature 1: Completion token length (normalized)
        f_len = min(2.0, len(completion.split()) / 20.0)

        # Feature 2: Syntax flag
        f_syntax = 1.0 if syntax_valid else 0.0

        # Feature 3: Trace flag
        f_trace = 1.0 if is_trace else 0.0

        # Feature 4: Extracted step count
        f_steps = min(2.0, len(steps) / max(1, example.depth))

        # Feature 5: Target answer token present in completion
        f_ans_present = 1.0 if example.answer in completion else 0.0

        # Feature 6: Start node present in completion
        f_start_present = 1.0 if example.start_node in completion else 0.0

        # Feature 7: Fraction of steps matching premise edges (without checking continuity)
        valid_edges = set(example.edges) | set(example.distractor_edges)
        valid_count = sum(1 for s in steps if s in valid_edges)
        f_edge_ratio = valid_count / len(steps) if steps else 0.0

        # Feature 8: Target node declared in answer section
        f_ans_match = 1.0 if ans == example.answer else 0.0

        feats = [
            f_len,
            f_syntax,
            f_trace,
            f_steps,
            f_ans_present,
            f_start_present,
            f_edge_ratio,
            f_ans_match,
        ]
        return torch.tensor(feats, dtype=torch.float32)

    def train_on_dataset(
        self,
        dataset: list[tuple[str, ReasoningExample, bool]],
        epochs: int = 40,
        lr: float = 0.01,
    ) -> float:
        """Train classifier on (completion, example, is_valid_label) data."""
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        X = torch.stack([self.extract_features(c, ex) for c, ex, _ in dataset])
        Y = torch.tensor([[1.0] if label else [0.0] for _, _, label in dataset])

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(X)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()

        if self.frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        return float(loss.item())

    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        self.model.eval()
        with torch.no_grad():
            x = self.extract_features(completion, example).unsqueeze(0)
            logit = self.model(x)
            prob = float(torch.sigmoid(logit).item())

        passed = prob >= self.threshold
        return VerifierResult(
            score=prob,
            passed=passed,
            confidence=prob if passed else 1.0 - prob,
            reason_code=f"LEARNED_PROB_{prob:.3f}_{'PASS' if passed else 'FAIL'}",
            metadata={"probability": prob, "threshold": self.threshold},
            properties_checked=["learned_classifier"],
        )
