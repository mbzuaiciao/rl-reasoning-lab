"""Phase 7 Experiment 12: RL Confidence Collapse and Regularization Dynamics.

Investigates:
1. RL-Induced Confidence Collapse:
   When reward strongly favors a single pattern, PPO sharpens policy entropy to near-zero
   before reasoning capability matures (Premature Entropy Collapse).
2. Entropy Regularization Bonus (beta_ent in {0.0, 0.01, 0.05}):
   Does encouraging exploration protect calibration and prevent premature certainty?
3. KL Divergence Penalty (ref_kl_coef in {0.0, 0.05, 0.2}):
   Does anchoring to the SFT reference distribution preserve calibration and selective risk?
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
from rl_reasoning_lab.evaluation.policy_uncertainty import (
    compute_calibration_metrics,
    compute_risk_coverage_curve,
    extract_trajectory_uncertainty,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
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
            "[bold cyan]Phase 7: RL Confidence Collapse & Regularization Dynamics[/bold cyan]\n"
            "Observing premature entropy collapse under PPO and testing whether entropy\n"
            "bonuses and KL regularization preserve policy calibration.",
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
    train_prompts = [ex.prompt for ex in train_data]
    example_lookup = {ex.prompt: ex for ex in train_data}

    console.print("[dim]Pre-training shared base model...[/dim]")
    base_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    train_sft(
        model=base_model,
        train_examples=to_sft_examples(train_data),
        val_examples=to_sft_examples(val_data),
        tokenizer=tok,
        num_epochs=10,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    ref_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    ref_model.load_state_dict(base_model.state_dict())

    # Reward function: terminal answer reward
    def verifier_fn(prompt: str, completion: str) -> tuple[bool, int | None]:
        ex = example_lookup.get(prompt)
        if ex is None:
            return False, None
        rep = evaluator.evaluate(completion, ex)
        return rep.answer_correct, int(rep.answer_correct * 10)

    reward_fn = LanguageRewardFunction(verifier_fn)

    # Regularization regimes to evaluate
    regimes = [
        ("Base SFT (No RL)", None, None, None),
        ("PPO (Zero Reg: KL=0.0, Ent=0.0)", 0.0, 0.0, 5e-4),
        ("PPO (KL Anchor: KL=0.1, Ent=0.0)", 0.1, 0.0, 5e-4),
        ("PPO (Entropy Bonus: KL=0.02, Ent=0.05)", 0.02, 0.05, 5e-4),
    ]

    table = Table(title="Regularization Effects on RL Confidence and Calibration")
    table.add_column("Regularization Regime", justify="left")
    table.add_column("Test Accuracy", justify="right")
    table.add_column("Mean Entropy", justify="right")
    table.add_column("Mean Confidence", justify="right")
    table.add_column("ECE (5 bins)", justify="right")
    table.add_column("AURC (Selective Risk)", justify="right")

    for label, kl_coef, ent_coef, lr in regimes:
        policy = TinyTransformer(
            vocab_size=tok.vocab_size,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            max_seq_len=128,
        )
        policy.load_state_dict(base_model.state_dict())

        if kl_coef is not None:
            console.print(f"[dim]Training {label}...[/dim]")
            train_lm_ppo(
                model=policy,
                ref_model=ref_model,
                prompts=train_prompts,
                tokenizer=tok,
                reward_fn=reward_fn,
                num_iterations=6,
                rollout_batch_size=16,
                epochs_per_rollout=2,
                minibatch_size=8,
                lr=lr,
                clip_eps=0.2,
                ref_kl_coef=kl_coef,
                entropy_coef=ent_coef,
                seed=seed,
            )
        policy.eval()

        # Evaluate calibration
        corrects: list[bool] = []
        confs: list[float] = []
        entropies: list[float] = []

        for ex in test_data:
            p_ids = tok.encode(ex.prompt, add_bos=True)
            f_ids, _ = policy.generate(
                p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
            )
            c_text = tok.decode(f_ids[0, len(p_ids) :].tolist(), skip_special=True)

            rep = evaluator.evaluate(c_text, ex)
            unc = extract_trajectory_uncertainty(policy, tok, ex.prompt, c_text)

            corrects.append(rep.answer_correct)
            confs.append(unc.final_answer_prob)
            entropies.append(unc.mean_token_entropy)

        acc = float(np.mean(corrects))
        m_ent = float(np.mean(entropies))
        m_conf = float(np.mean(confs))
        cal = compute_calibration_metrics(confs, corrects, num_bins=5)
        rc = compute_risk_coverage_curve(confs, corrects, num_thresholds=20)

        table.add_row(
            label,
            f"{acc:.1%}",
            f"{m_ent:.3f}",
            f"{m_conf:.3f}",
            f"{cal.ece:.3f}",
            f"{rc.aurc:.3f}",
        )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
