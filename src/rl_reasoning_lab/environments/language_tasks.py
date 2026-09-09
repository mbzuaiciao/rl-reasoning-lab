"""Synthetic reasoning tasks with deterministic ground truth verification."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    """A single reasoning task instance."""

    prompt: str
    target: str
    full_text: str


class ArithmeticTask:
    """A controlled arithmetic reasoning environment with exact deterministic verifiers.

    Generates addition problems such as:
        '3 + 4 =' -> ' 7'
        '2 + 5 + 1 =' -> ' 8'
    """

    def __init__(
        self,
        min_operands: int = 2,
        max_operands: int = 2,
        min_val: int = 0,
        max_val: int = 9,
    ) -> None:
        self.min_operands = min_operands
        self.max_operands = max_operands
        self.min_val = min_val
        self.max_val = max_val

    def generate_example(self, rng: random.Random | None = None) -> Example:
        generator = rng if rng is not None else random.Random()
        num_ops = generator.randint(self.min_operands, self.max_operands)
        operands = [generator.randint(self.min_val, self.max_val) for _ in range(num_ops)]

        prompt = " + ".join(str(op) for op in operands) + " = "
        target = str(sum(operands))
        full_text = prompt + target

        return Example(prompt=prompt, target=target, full_text=full_text)

    def generate_dataset(self, size: int, seed: int = 42) -> list[Example]:
        """Generate a deterministic dataset of unique or randomly sampled arithmetic examples."""
        rng = random.Random(seed)
        examples: list[Example] = []
        for _ in range(size):
            examples.append(self.generate_example(rng))
        return examples

    @staticmethod
    def verify(prompt: str, completion: str) -> tuple[bool, int | None]:
        """Verify whether completion matches the ground truth arithmetic evaluation of prompt.

        Returns:
            (is_correct, ground_truth_value)
        """
        # Parse expression before '='
        expr = prompt.split("=")[0].strip()
        try:
            # Safely evaluate sum of numbers
            parts = [int(p.strip()) for p in expr.split("+")]
            expected_answer = sum(parts)
        except Exception:
            return False, None

        # Clean completion
        clean_comp = completion.strip().split("\n")[0].split(" ")[0].split(";")[0]
        try:
            predicted_answer = int(clean_comp)
            is_correct = predicted_answer == expected_answer
            return is_correct, expected_answer
        except ValueError:
            return False, expected_answer
