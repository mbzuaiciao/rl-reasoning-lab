"""Evaluation metrics, Best-of-N sampling, Rejection Sampling, and depth diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from rl_reasoning_lab.environments.reasoning_tasks import ReasoningExample
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluationReport, TraceEvaluator


@dataclass
class DepthDiagnostics:
    """Metrics partitioned by reasoning depth."""

    depth: int
    count: int = 0
    correct_count: int = 0
    valid_trace_count: int = 0
    total_steps: int = 0
    total_tokens: int = 0
    entropies: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.count if self.count > 0 else 0.0

    @property
    def trace_validity(self) -> float:
        return self.valid_trace_count / self.count if self.count > 0 else 0.0

    @property
    def mean_tokens(self) -> float:
        return self.total_tokens / self.count if self.count > 0 else 0.0

    @property
    def mean_entropy(self) -> float:
        return float(np.mean(self.entropies)) if self.entropies else 0.0


@dataclass
class ReasoningEvaluationSummary:
    """Comprehensive evaluation results across a dataset."""

    total_examples: int
    answer_accuracy: float
    syntax_accuracy: float
    trace_validity: float
    path_consistency: float
    complete_path_rate: float
    mean_tokens: float
    total_reasoning_tokens: int
    efficiency: float  # correct_solutions / total_reasoning_tokens
    depth_breakdown: dict[int, DepthDiagnostics] = field(default_factory=dict)
    reports: list[TraceEvaluationReport] = field(default_factory=list)


def evaluate_reasoning_model(
    model: TinyTransformer,
    tokenizer: CharTokenizer,
    examples: list[ReasoningExample],
    max_new_tokens: int = 40,
    greedy: bool = True,
    temperature: float = 1.0,
) -> ReasoningEvaluationSummary:
    """Evaluate a transformer policy on a multi-step reasoning task."""
    model.eval()
    reports: list[TraceEvaluationReport] = []
    depth_map: dict[int, DepthDiagnostics] = defaultdict(lambda: DepthDiagnostics(depth=0))

    total_tokens = 0
    correct_count = 0

    with torch.no_grad():
        for ex in examples:
            d = ex.depth
            if depth_map[d].depth == 0:
                depth_map[d].depth = d

            prompt_ids = tokenizer.encode(ex.prompt, add_bos=True, add_eos=False)
            gen_ids, _ = model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Generated token slice
            gen_token_ids = gen_ids[0, len(prompt_ids) :].tolist()
            completion_str = tokenizer.decode(gen_token_ids, skip_special=True)
            report = TraceEvaluator.evaluate(completion_str, ex)
            reports.append(report)

            # Compute entropy on generated positions
            logits = model(gen_ids)
            gen_logits = logits[0, len(prompt_ids) - 1 : -1, :]
            probs = F.softmax(gen_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()

            num_gen_tokens = len(gen_token_ids)
            total_tokens += num_gen_tokens

            # Update diagnostics
            diag = depth_map[d]
            diag.count += 1
            if report.answer_correct:
                diag.correct_count += 1
                correct_count += 1
            if report.complete_path:
                diag.valid_trace_count += 1
            diag.total_steps += report.reasoning_length
            diag.total_tokens += num_gen_tokens
            diag.entropies.append(entropy)

    n = len(examples)
    ans_acc = correct_count / n if n > 0 else 0.0
    syntax_acc = sum(1 for r in reports if r.syntax_valid) / n if n > 0 else 0.0
    trace_val = sum(r.step_validity for r in reports) / n if n > 0 else 0.0
    path_cons = sum(1 for r in reports if r.path_consistency) / n if n > 0 else 0.0
    comp_path = sum(1 for r in reports if r.complete_path) / n if n > 0 else 0.0
    mean_toks = total_tokens / n if n > 0 else 0.0
    eff = correct_count / total_tokens if total_tokens > 0 else 0.0

    return ReasoningEvaluationSummary(
        total_examples=n,
        answer_accuracy=ans_acc,
        syntax_accuracy=syntax_acc,
        trace_validity=trace_val,
        path_consistency=path_cons,
        complete_path_rate=comp_path,
        mean_tokens=mean_toks,
        total_reasoning_tokens=total_tokens,
        efficiency=eff,
        depth_breakdown=dict(depth_map),
        reports=reports,
    )


def evaluate_best_of_n(
    model: TinyTransformer,
    tokenizer: CharTokenizer,
    examples: list[ReasoningExample],
    n_values: list[int] | None = None,
    temperature: float = 1.0,
    max_new_tokens: int = 40,
) -> dict[int, float]:
    """Compute pass@N rates for N in n_values using ground-truth verification."""
    if n_values is None:
        n_values = [1, 2, 4, 8, 16]
    max_n = max(n_values)

    model.eval()
    pass_at_n: dict[int, int] = {n: 0 for n in n_values}

    with torch.no_grad():
        for ex in examples:
            prompt_ids = tokenizer.encode(ex.prompt, add_bos=True, add_eos=False)
            correct_found_at: list[bool] = []

            for _ in range(max_n):
                gen_ids, _ = model.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    greedy=False,
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                )
                gen_tokens = gen_ids[0, len(prompt_ids) :].tolist()
                comp = tokenizer.decode(gen_tokens, skip_special=True)
                report = TraceEvaluator.evaluate(comp, ex)
                correct_found_at.append(report.answer_correct)

            for n in n_values:
                # Pass@N: true if any of the first n samples is correct
                if any(correct_found_at[:n]):
                    pass_at_n[n] += 1

    total = len(examples)
    return {n: (pass_at_n[n] / total) if total > 0 else 0.0 for n in n_values}


def rejection_sampling_step(
    model: TinyTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    examples: list[ReasoningExample],
    num_samples_per_prompt: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 40,
) -> tuple[int, float]:
    """Perform one iteration of rejection sampling (filtered SFT).

    Samples candidate solutions, filters with ground truth, and trains on successes.
    """
    model.eval()
    accepted_seqs: list[list[int]] = []
    accepted_masks: list[list[float]] = []

    with torch.no_grad():
        for ex in examples:
            prompt_ids = tokenizer.encode(ex.prompt, add_bos=True, add_eos=False)
            for _ in range(num_samples_per_prompt):
                gen_ids, _ = model.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    greedy=False,
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                )
                gen_tokens = gen_ids[0, len(prompt_ids) :].tolist()
                comp = tokenizer.decode(gen_tokens, skip_special=True)
                report = TraceEvaluator.evaluate(comp, ex)

                # Acceptance criterion: correct final answer
                if report.answer_correct:
                    full_seq = gen_ids[0].tolist()
                    # Mask 0 on prompt, 1 on completion
                    mask = [0.0] * len(prompt_ids) + [1.0] * len(gen_tokens)
                    accepted_seqs.append(full_seq)
                    accepted_masks.append(mask)

    if not accepted_seqs:
        return 0, 0.0

    # Train on accepted samples
    model.train()
    max_len = max(len(s) for s in accepted_seqs)
    b = len(accepted_seqs)

    pad_seqs = torch.full((b, max_len), tokenizer.pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros((b, max_len), dtype=torch.float32)

    for i, (seq, m) in enumerate(zip(accepted_seqs, accepted_masks, strict=True)):
        pad_seqs[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        loss_masks[i, : len(m)] = torch.tensor(m, dtype=torch.float32)

    input_ids = pad_seqs[:, :-1].contiguous()
    targets = pad_seqs[:, 1:].contiguous()
    target_masks = loss_masks[:, 1:].contiguous()

    optimizer.zero_grad()
    logits = model(input_ids)
    loss_raw = F.cross_entropy(
        logits.view(-1, tokenizer.vocab_size), targets.view(-1), reduction="none"
    ).view_as(targets)
    loss = (loss_raw * target_masks).sum() / (target_masks.sum() + 1e-8)
    loss.backward()
    optimizer.step()

    return len(accepted_seqs), float(loss.item())
