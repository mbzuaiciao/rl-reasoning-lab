"""Phase 5 Main Experiment: Multi-Seed Comparative Reasoning Benchmark.

Evaluates 5 distinct paradigms across 3 independent seeds:
1. Answer-Only SFT
2. Trace SFT
3. SFT + PPO Outcome-Reward
4. SFT + PPO Process-Reward
5. Rejection Sampling (Filtered SFT)

Tracks:
- In-Distribution (ID) Answer Accuracy (d in {1, 2, 3})
- Out-of-Distribution (OOD) Answer Accuracy (d in {4, 5})
- Reasoning Step Validity
- Complete Path Rate
- Tokens per Answer & Token Efficiency (Correct / 100 tokens)
- Empirical Mean and Bootstrap Std Error across seeds.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.lm_ppo import train_lm_ppo
from rl_reasoning_lab.environments.language_tasks import Example
from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.evaluation.reasoning_metrics import (
    evaluate_reasoning_model,
    rejection_sampling_step,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.rewards.reasoning_rewards import ReasoningRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft


def to_sft_examples(examples: list[ReasoningExample], is_trace: bool) -> list[Example]:
    return [
        Example(
            prompt=ex.prompt,
            target=ex.trace_target if is_trace else ex.answer_target,
            full_text=ex.prompt + (ex.trace_target if is_trace else ex.answer_target),
        )
        for ex in examples
    ]


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 5: Multi-Seed Comparative Reasoning Benchmark[/bold cyan]\n"
            "Evaluating 5 paradigms across 3 independent seeds with ID, OOD, and efficiency metrics.",
            title="RL Reasoning Lab — Phase 5",
        )
    )

    seeds = [42, 43, 44]
    tok = CharTokenizer()
    node_pool = list("ABCDEFGHIJKL")
    task = GraphTraversalTask(node_pool=node_pool)

    # Accumulators for metrics across seeds
    # Paradigms:
    paradigms = [
        "1. Answer-Only SFT",
        "2. Trace SFT",
        "3. SFT + PPO Outcome",
        "4. SFT + PPO Process",
        "5. Rejection Sampling",
    ]
    results: dict[str, dict[str, list[float]]] = {
        p: {
            "id_acc": [],
            "ood_acc": [],
            "step_val": [],
            "complete_path": [],
            "tokens": [],
            "eff": [],
        }
        for p in paradigms
    }

    for seed_idx, seed in enumerate(seeds):
        console.print(
            f"\n[bold yellow]=== Running Evaluation on Seed {seed} ({seed_idx + 1}/{len(seeds)}) ===[/bold yellow]"
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_data = task.generate_dataset(size=80, depths=[1, 2, 3], seed=seed)
        val_data = task.generate_dataset(size=20, depths=[1, 2, 3], seed=seed + 1)
        id_test = task.generate_dataset(size=30, depths=[1, 2, 3], seed=seed + 2)
        ood_test = task.generate_dataset(size=20, depths=[4, 5], seed=seed + 3)
        train_prompts = [ex.prompt for ex in train_data]

        # -------------------------------------------------------------
        # 1. Answer-Only SFT
        # -------------------------------------------------------------
        console.print("[dim]  Training Answer-Only SFT...[/dim]")
        ans_sft = TinyTransformer(
            vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
        )
        train_sft(
            model=ans_sft,
            train_examples=to_sft_examples(train_data, is_trace=False),
            val_examples=to_sft_examples(val_data, is_trace=False),
            tokenizer=tok,
            num_epochs=15,
            batch_size=16,
            lr=3e-3,
            seed=seed,
        )
        res_ans_id = evaluate_reasoning_model(ans_sft, tok, id_test, max_new_tokens=25)
        res_ans_ood = evaluate_reasoning_model(ans_sft, tok, ood_test, max_new_tokens=25)
        results["1. Answer-Only SFT"]["id_acc"].append(res_ans_id.answer_accuracy)
        results["1. Answer-Only SFT"]["ood_acc"].append(res_ans_ood.answer_accuracy)
        results["1. Answer-Only SFT"]["step_val"].append(res_ans_id.trace_validity)
        results["1. Answer-Only SFT"]["complete_path"].append(res_ans_id.complete_path_rate)
        results["1. Answer-Only SFT"]["tokens"].append(res_ans_id.mean_tokens)
        results["1. Answer-Only SFT"]["eff"].append(res_ans_id.efficiency * 100)

        # -------------------------------------------------------------
        # 2. Trace SFT (Base Checkpoint for RL)
        # -------------------------------------------------------------
        console.print("[dim]  Training Trace SFT...[/dim]")
        trace_sft = TinyTransformer(
            vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
        )
        train_sft(
            model=trace_sft,
            train_examples=to_sft_examples(train_data, is_trace=True),
            val_examples=to_sft_examples(val_data, is_trace=True),
            tokenizer=tok,
            num_epochs=15,
            batch_size=16,
            lr=3e-3,
            seed=seed,
        )
        res_trace_id = evaluate_reasoning_model(trace_sft, tok, id_test, max_new_tokens=45)
        res_trace_ood = evaluate_reasoning_model(trace_sft, tok, ood_test, max_new_tokens=60)
        results["2. Trace SFT"]["id_acc"].append(res_trace_id.answer_accuracy)
        results["2. Trace SFT"]["ood_acc"].append(res_trace_ood.answer_accuracy)
        results["2. Trace SFT"]["step_val"].append(res_trace_id.trace_validity)
        results["2. Trace SFT"]["complete_path"].append(res_trace_id.complete_path_rate)
        results["2. Trace SFT"]["tokens"].append(res_trace_id.mean_tokens)
        results["2. Trace SFT"]["eff"].append(res_trace_id.efficiency * 100)

        # -------------------------------------------------------------
        # 3. SFT + PPO Outcome-Reward
        # -------------------------------------------------------------
        console.print("[dim]  Training SFT + PPO Outcome...[/dim]")
        ppo_out = TinyTransformer(
            vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
        )
        ppo_out.load_state_dict(trace_sft.state_dict())
        ref_model = TinyTransformer(
            vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
        )
        ref_model.load_state_dict(trace_sft.state_dict())

        ex_lookup = {ex.prompt: ex for ex in train_data}
        out_fn = ReasoningRewardFunction(reward_mode="outcome")

        def verifier_out(p: str, c: str, lookup=ex_lookup, fn=out_fn) -> tuple[bool, int | None]:
            ex = lookup.get(p)
            if ex is None:
                return False, None
            o = fn.evaluate(c, ex)
            return o.answer_correct, int(o.total_reward * 10)

        train_lm_ppo(
            model=ppo_out,
            ref_model=ref_model,
            prompts=train_prompts,
            tokenizer=tok,
            reward_fn=LanguageRewardFunction(verifier_out),
            num_iterations=8,
            rollout_batch_size=16,
            epochs_per_rollout=2,
            minibatch_size=8,
            lr=5e-4,
            clip_eps=0.2,
            ref_kl_coef=0.05,
            seed=seed,
        )
        res_ppo_out_id = evaluate_reasoning_model(ppo_out, tok, id_test, max_new_tokens=45)
        res_ppo_out_ood = evaluate_reasoning_model(ppo_out, tok, ood_test, max_new_tokens=60)
        results["3. SFT + PPO Outcome"]["id_acc"].append(res_ppo_out_id.answer_accuracy)
        results["3. SFT + PPO Outcome"]["ood_acc"].append(res_ppo_out_ood.answer_accuracy)
        results["3. SFT + PPO Outcome"]["step_val"].append(res_ppo_out_id.trace_validity)
        results["3. SFT + PPO Outcome"]["complete_path"].append(res_ppo_out_id.complete_path_rate)
        results["3. SFT + PPO Outcome"]["tokens"].append(res_ppo_out_id.mean_tokens)
        results["3. SFT + PPO Outcome"]["eff"].append(res_ppo_out_id.efficiency * 100)

        # -------------------------------------------------------------
        # 4. SFT + PPO Process-Reward
        # -------------------------------------------------------------
        console.print("[dim]  Training SFT + PPO Process...[/dim]")
        ppo_proc = TinyTransformer(
            vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
        )
        ppo_proc.load_state_dict(trace_sft.state_dict())
        proc_fn = ReasoningRewardFunction(reward_mode="process")

        def verifier_proc(p: str, c: str, lookup=ex_lookup, fn=proc_fn) -> tuple[bool, int | None]:
            ex = lookup.get(p)
            if ex is None:
                return False, None
            o = fn.evaluate(c, ex)
            return o.answer_correct, int(o.total_reward * 10)

        train_lm_ppo(
            model=ppo_proc,
            ref_model=ref_model,
            prompts=train_prompts,
            tokenizer=tok,
            reward_fn=LanguageRewardFunction(verifier_proc),
            num_iterations=8,
            rollout_batch_size=16,
            epochs_per_rollout=2,
            minibatch_size=8,
            lr=5e-4,
            clip_eps=0.2,
            ref_kl_coef=0.05,
            seed=seed,
        )
        res_ppo_proc_id = evaluate_reasoning_model(ppo_proc, tok, id_test, max_new_tokens=45)
        res_ppo_proc_ood = evaluate_reasoning_model(ppo_proc, tok, ood_test, max_new_tokens=60)
        results["4. SFT + PPO Process"]["id_acc"].append(res_ppo_proc_id.answer_accuracy)
        results["4. SFT + PPO Process"]["ood_acc"].append(res_ppo_proc_ood.answer_accuracy)
        results["4. SFT + PPO Process"]["step_val"].append(res_ppo_proc_id.trace_validity)
        results["4. SFT + PPO Process"]["complete_path"].append(res_ppo_proc_id.complete_path_rate)
        results["4. SFT + PPO Process"]["tokens"].append(res_ppo_proc_id.mean_tokens)
        results["4. SFT + PPO Process"]["eff"].append(res_ppo_proc_id.efficiency * 100)

        # -------------------------------------------------------------
        # 5. Rejection Sampling (Filtered SFT)
        # -------------------------------------------------------------
        console.print("[dim]  Training Rejection Sampling...[/dim]")
        rs_model = TinyTransformer(
            vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
        )
        rs_model.load_state_dict(trace_sft.state_dict())
        rs_opt = torch.optim.Adam(rs_model.parameters(), lr=1e-3)
        for _ in range(4):
            rejection_sampling_step(
                model=rs_model,
                optimizer=rs_opt,
                tokenizer=tok,
                examples=train_data,
                num_samples_per_prompt=3,
                temperature=0.8,
                max_new_tokens=40,
            )
        res_rs_id = evaluate_reasoning_model(rs_model, tok, id_test, max_new_tokens=45)
        res_rs_ood = evaluate_reasoning_model(rs_model, tok, ood_test, max_new_tokens=60)
        results["5. Rejection Sampling"]["id_acc"].append(res_rs_id.answer_accuracy)
        results["5. Rejection Sampling"]["ood_acc"].append(res_rs_ood.answer_accuracy)
        results["5. Rejection Sampling"]["step_val"].append(res_rs_id.trace_validity)
        results["5. Rejection Sampling"]["complete_path"].append(res_rs_id.complete_path_rate)
        results["5. Rejection Sampling"]["tokens"].append(res_rs_id.mean_tokens)
        results["5. Rejection Sampling"]["eff"].append(res_rs_id.efficiency * 100)

    # Multi-Seed Summary Table
    table = Table(title=f"Multi-Seed Comparative Benchmark across {len(seeds)} Seeds (Mean ± Std)")
    table.add_column("Paradigm", justify="left")
    table.add_column("ID Acc (d<=3)", justify="right")
    table.add_column("OOD Acc (d in {4,5})", justify="right")
    table.add_column("Step Validity", justify="right")
    table.add_column("Complete Path %", justify="right")
    table.add_column("Mean Tokens", justify="right")
    table.add_column("Efficiency (Corr/100Tok)", justify="right")

    for p in paradigms:
        d = results[p]
        id_m, id_s = np.mean(d["id_acc"]), np.std(d["id_acc"])
        ood_m, ood_s = np.mean(d["ood_acc"]), np.std(d["ood_acc"])
        val_m, val_s = np.mean(d["step_val"]), np.std(d["step_val"])
        cp_m, cp_s = np.mean(d["complete_path"]), np.std(d["complete_path"])
        tok_m, tok_s = np.mean(d["tokens"]), np.std(d["tokens"])
        eff_m, eff_s = np.mean(d["eff"]), np.std(d["eff"])

        table.add_row(
            p,
            f"{id_m:.1%} ± {id_s:.1%}",
            f"{ood_m:.1%} ± {ood_s:.1%}",
            f"{val_m:.1%} ± {val_s:.1%}",
            f"{cp_m:.1%} ± {cp_s:.1%}",
            f"{tok_m:.1f} ± {tok_s:.1f}",
            f"{eff_m:.2f} ± {eff_s:.2f}",
        )

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    run_experiment()
