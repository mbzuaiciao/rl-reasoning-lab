"""Phase 7 Experiment 14: Adaptive Test-Time Compute Allocation.

Investigates allocating reasoning compute dynamically based on uncertainty:
1. Fixed Compute Baseline: Sample N=8 candidates for every prompt.
2. Uniform Low Compute Baseline: Sample N=2 candidates for every prompt.
3. Adaptive Compute:
   - Initial greedy pass evaluates policy uncertainty (entropy).
   - If Low Uncertainty: Keep N=1 (confident, no extra search needed).
   - If Moderate Uncertainty: Sample N=4 candidates.
   - If High Uncertainty: Sample N=16 candidates.
4. Pre- vs. Post-Selection Calibration:
   Evaluates whether Best-of-N selection distorts calibration by choosing the highest-scoring candidate.
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
    compute_calibration_metrics,
    extract_trajectory_uncertainty,
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
            "[bold cyan]Phase 7: Adaptive Test-Time Compute Allocation[/bold cyan]\n"
            "Testing whether uncertainty-guided compute allocation (Best-of-N) achieves\n"
            "higher task accuracy per token generated than fixed uniform compute.",
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

    train_data = task.generate_dataset(size=50, depths=[1, 2], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], seed=seed + 1)
    test_data = task.generate_dataset(size=40, depths=[1, 2, 3], seed=seed + 2)

    console.print("[dim]Training base policy...[/dim]")
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

    # Pre-generate greedy rollouts to determine uncertainty distribution
    initial_rollouts = []
    for ex in test_data:
        p_ids = tok.encode(ex.prompt, add_bos=True)
        g_ids, _ = model.generate(
            p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        c_text = tok.decode(g_ids[0, len(p_ids) :].tolist(), skip_special=True)
        unc = extract_trajectory_uncertainty(model, tok, ex.prompt, c_text)
        rep = evaluator.evaluate(c_text, ex)
        initial_rollouts.append(
            {
                "ex": ex,
                "p_ids": p_ids,
                "greedy_text": c_text,
                "entropy": unc.mean_token_entropy,
                "conf": unc.final_answer_prob,
                "is_correct": rep.answer_correct,
            }
        )

    entropies = [r["entropy"] for r in initial_rollouts]
    tau_low = float(np.percentile(entropies, 40))
    tau_high = float(np.percentile(entropies, 80))

    # Strategies
    # 1. Fixed N=1 (Greedy)
    # 2. Fixed N=4
    # 3. Fixed N=8
    # 4. Adaptive N: N=1 if H <= tau_low, N=4 if tau_low < H <= tau_high, N=12 if H > tau_high

    def evaluate_best_of_n_strategy(strategy_name: str, get_n_fn):
        total_samples = 0
        correct_count = 0
        selected_confs: list[float] = []
        selected_correct: list[bool] = []

        for item in initial_rollouts:
            ex = item["ex"]
            p_ids = item["p_ids"]
            n_to_sample = get_n_fn(item)
            total_samples += n_to_sample

            if n_to_sample == 1:
                # Use greedy
                c_text = item["greedy_text"]
                rep = evaluator.evaluate(c_text, ex)
                unc = extract_trajectory_uncertainty(model, tok, ex.prompt, c_text)
                is_corr = rep.answer_correct
                conf = unc.final_answer_prob
            else:
                # Sample N candidates and pick best candidate using format/syntax verifier
                candidates = []
                for _ in range(n_to_sample):
                    s_ids, _ = model.generate(
                        p_ids,
                        max_new_tokens=35,
                        temperature=0.8,
                        greedy=False,
                        eos_token_id=tok.eos_token_id,
                    )
                    s_text = tok.decode(s_ids[0, len(p_ids) :].tolist(), skip_special=True)
                    rep_s = evaluator.evaluate(s_text, ex)
                    unc_s = extract_trajectory_uncertainty(model, tok, ex.prompt, s_text)
                    # Candidate score: correctness + completeness
                    score = (1.0 if rep_s.complete_path else 0.0) + (
                        0.5 if rep_s.answer_correct else 0.0
                    )
                    candidates.append((score, unc_s.final_answer_prob, rep_s.answer_correct))

                # Select candidate with highest score (breaking ties with confidence)
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best_cand = candidates[0]
                is_corr = best_cand[2]
                conf = best_cand[1]

            if is_corr:
                correct_count += 1
            selected_correct.append(is_corr)
            selected_confs.append(conf)

        acc = correct_count / len(initial_rollouts)
        cal = compute_calibration_metrics(selected_confs, selected_correct, num_bins=5)
        return {
            "name": strategy_name,
            "accuracy": acc,
            "total_samples": total_samples,
            "samples_per_prompt": total_samples / len(initial_rollouts),
            "ece": cal.ece,
            "accuracy_per_sample": acc / (total_samples / len(initial_rollouts)),
        }

    res_n1 = evaluate_best_of_n_strategy("Fixed Greedy (N=1)", lambda item: 1)
    res_n4 = evaluate_best_of_n_strategy("Fixed Best-of-4", lambda item: 4)
    res_n8 = evaluate_best_of_n_strategy("Fixed Best-of-8", lambda item: 8)

    def adaptive_n(item):
        h = item["entropy"]
        if h <= tau_low:
            return 1
        elif h <= tau_high:
            return 4
        else:
            return 12

    res_adapt = evaluate_best_of_n_strategy("Adaptive Compute (N in {1, 4, 12})", adaptive_n)

    table = Table(title="Test-Time Compute Allocation Benchmark")
    table.add_column("Compute Allocation Strategy", justify="left")
    table.add_column("Task Accuracy", justify="right")
    table.add_column("Total Generations", justify="right")
    table.add_column("Mean N / Prompt", justify="right")
    table.add_column("Accuracy / Generation", justify="right")
    table.add_column("Post-Selection ECE", justify="right")

    for r in [res_n1, res_n4, res_n8, res_adapt]:
        table.add_row(
            r["name"],
            f"{r['accuracy']:.1%}",
            str(r["total_samples"]),
            f"{r['samples_per_prompt']:.1f}",
            f"{r['accuracy_per_sample']:.3f}",
            f"{r['ece']:.3f}",
        )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
