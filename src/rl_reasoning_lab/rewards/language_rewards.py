"""Reward functions for language model reasoning tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardOutcome:
    """Diagnostic outcome of evaluating a language model's generated response."""

    reward: float
    is_correct: bool
    is_malformed: bool
    expected_target: int | str | None
    parsed_prediction: int | str | None


class LanguageRewardFunction:
    """Computes reinforcement learning reward signals for language model outputs.

    Separates the optimization reward signal from task ground-truth correctness:
        - Correct answer: +1.0
        - Incorrect answer: 0.0
        - Malformed output: -0.5
    """

    def __init__(
        self,
        verifier_fn: Callable[[str, str], tuple[bool, int | None]],
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        malformed_penalty: float = -0.5,
    ) -> None:
        self.verifier_fn = verifier_fn
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.malformed_penalty = malformed_penalty

    def __call__(self, prompt: str, completion: str) -> RewardOutcome:
        is_correct, expected = self.verifier_fn(prompt, completion)

        # Check if output is malformed
        clean = completion.strip().split("\n")[0].split(" ")[0].split(";")[0]
        is_malformed = len(clean) == 0 or not clean.lstrip("-").isdigit()

        if is_correct:
            reward = self.correct_reward
        elif is_malformed:
            reward = self.malformed_penalty
        else:
            reward = self.incorrect_reward

        parsed = int(clean) if not is_malformed else clean
        return RewardOutcome(
            reward=reward,
            is_correct=is_correct,
            is_malformed=is_malformed,
            expected_target=expected,
            parsed_prediction=parsed,
        )
