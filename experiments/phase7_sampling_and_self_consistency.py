"""Phase 7 Experiment 7: Sampling-Based Uncertainty and Self-Consistency.

Generates N=8 stochastic rollouts per prompt:
1. Compares Greedy decoding vs. Self-Consistency (majority voting).
2. Evaluates Pass@1, Pass@2, Pass@4, and Pass@8 scaling.
3. Computes sample-based uncertainty metrics:
   - Vote entropy H(P^(answer))
   - Empirical answer disagreement rate
4. Investigates whether self-consistency agreement reliably indicates correctness,
   or whether models can be consistently wrong (shared systematic errors).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.environments.language_tasks import Example
from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.evaluation.policy_uncertainty import (
    compute_sampling_uncertainty,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.trainers.sft import train_sft
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator


def to_sft_examples(examples: list[ReasoningExample]) -> list[Example]:
    return [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in examples
    ]


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 7: Sampling-Based Uncertainty & Self-Consistency[/bold cyan]\n"
            "Comparing single-pass softmax confidence against multi-sample voting entropy,\n"
            "pass@k coverage, and the prevalence of 'confidently wrong' consensus.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGH"))
    evaluator = TraceEvaluator()

    train_data = task.generate_dataset(size=60, depths=[1, 2], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], seed=seed + 1)
    test_data = task.generate_dataset(size=40, depths=[1, 2, 3], seed=seed + 2)

    console.print("[dim]Training base model...[/dim]")
    model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    train_sft(
        model=model,
        train_examples=to_sft_examples(train_data),
        val_examples=to_sft_examples(val_data),
        tokenizer=tok,
        num_epochs=12,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )
    model.eval()

    num_samples_per_prompt = 8
    temperature = 0.8

    greedy_correct = 0
    sc_correct = 0
    consistently_wrong_count = 0  # 100% agreement on incorrect answer

    sampling_results = []
    pass_at_k_accum: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}

    console.print(
        f"[dim]Sampling {num_samples_per_prompt} rollouts per prompt (T={temperature})...[/dim]"
    )
    for ex in test_data:
        p_ids = tok.encode(ex.prompt, add_bos=True)

        # 1. Greedy completion
        greedy_ids, _ = model.generate(
            p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        greedy_text = tok.decode(greedy_ids[0, len(p_ids) :].tolist(), skip_special=True)
        rep_greedy = evaluator.evaluate(greedy_text, ex)
        if rep_greedy.answer_correct:
            greedy_correct += 1

        # 2. Sampled completions
        samples: list[str] = []
        for _ in range(num_samples_per_prompt):
            s_ids, _ = model.generate(
                p_ids,
                max_new_tokens=35,
                temperature=temperature,
                greedy=False,
                eos_token_id=tok.eos_token_id,
            )
            s_text = tok.decode(s_ids[0, len(p_ids) :].tolist(), skip_special=True)
            samples.append(s_text)

        def extract_ans(c: str) -> str:
            return evaluator.parse_completion(c)[0] or "UNKNOWN"

        samp_res = compute_sampling_uncertainty(samples, extract_ans, ground_truth=ex.answer)
        sampling_results.append(samp_res)

        # Self-consistency check
        if samp_res.majority_answer == ex.answer:
            sc_correct += 1
        elif samp_res.majority_frequency >= 0.99:
            # Consistently agreed on wrong answer
            consistently_wrong_count += 1

        for k in [1, 2, 4, 8]:
            pass_at_k_accum[k].append(samp_res.pass_at_k.get(k, 0.0))

    total = len(test_data)
    table = Table(title="Greedy vs. Self-Consistency vs. Pass@K")
    table.add_column("Decoding / Aggregation Method", justify="left")
    table.add_column("Accuracy / Success Rate", justify="right")
    table.add_column("Compute per Prompt", justify="right")

    table.add_row("Greedy Decoding", f"{greedy_correct / total:.1%}", "1 generation")
    table.add_row(
        "Self-Consistency (Majority Vote)",
        f"{sc_correct / total:.1%}",
        f"{num_samples_per_prompt} generations",
    )
    for k in [1, 2, 4, 8]:
        table.add_row(
            f"Pass@{k}", f"{float(np.mean(pass_at_k_accum[k])):.1%}", f"{k} candidate draws"
        )

    console.print(table)

    # Uncertainty & Consistency Diagnostic
    mean_vote_entropy = float(np.mean([r.vote_entropy for r in sampling_results]))
    mean_disagreement = float(np.mean([r.answer_disagreement_rate for r in sampling_results]))

    diag_table = Table(title="Sampling Disagreement & Failure Analysis")
    diag_table.add_column("Sampling Metric", justify="left")
    diag_table.add_column("Value", justify="right")

    diag_table.add_row("Mean Answer Vote Entropy H(P_ans)", f"{mean_vote_entropy:.3f}")
    diag_table.add_row("Mean Answer Disagreement Rate", f"{mean_disagreement:.1%}")
    diag_table.add_row(
        "Consistently Wrong Prompts (100% agreement on error)",
        f"{consistently_wrong_count}/{total} ({consistently_wrong_count / total:.1%})",
    )

    console.print(diag_table)


if __name__ == "__main__":
    run_experiment()
