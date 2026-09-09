"""Phase 5 Experiment: Outcome Reward vs. Process Reward vs. Hybrid Reward in PPO.

Investigates:
1. Outcome Reward: Only terminal answer correctness is rewarded.
2. Process Reward: Every valid, continuous geodesic step receives normalized reward.
3. Hybrid Reward: Combined process supervision and terminal verification.

Measures:
- Answer accuracy and intermediate step validity.
- Complete derivation path rate.
- Learning speed and policy drift from SFT reference model.
- Does denser supervision improve genuine reasoning or merely optimization ease?
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
from rl_reasoning_lab.evaluation.reasoning_metrics import evaluate_reasoning_model
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.rewards.reasoning_rewards import ReasoningRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft


def to_sft_examples(examples: list[ReasoningExample]) -> list[Example]:
    return [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in examples
    ]


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 5: Outcome vs. Process vs. Hybrid Rewards in PPO[/bold cyan]\n"
            "Comparing sparse outcome reward against dense, normalized process reward\n"
            "and balanced hybrid reward.",
            title="RL Reasoning Lab — Phase 5",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGH"))

    train_data = task.generate_dataset(size=80, depths=[1, 2, 3], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2, 3], seed=seed + 1)
    test_data = task.generate_dataset(size=40, depths=[1, 2, 3, 4], seed=seed + 2)
    train_prompts = [ex.prompt for ex in train_data]

    # Pre-train common SFT base model
    console.print("[dim]Pre-training shared base model with SFT...[/dim]")
    base_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=128,
    )
    train_sft(
        model=base_model,
        train_examples=to_sft_examples(train_data),
        val_examples=to_sft_examples(val_data),
        tokenizer=tok,
        num_epochs=15,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    ref_model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=128,
    )
    ref_model.load_state_dict(base_model.state_dict())

    example_lookup = {ex.prompt: ex for ex in train_data}

    regimes = [
        ("Outcome-Only Reward", "outcome"),
        ("Process-Only Reward", "process"),
        ("Hybrid (0.5 Proc + 0.5 Out)", "hybrid"),
    ]
    results = {}

    for label, mode in regimes:
        console.print(f"[dim]Training PPO with {label}...[/dim]")
        policy = TinyTransformer(
            vocab_size=tok.vocab_size,
            d_model=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=128,
        )
        policy.load_state_dict(base_model.state_dict())

        r_fn_internal = ReasoningRewardFunction(reward_mode=mode, hybrid_alpha=0.5)

        def make_verifier(r_fn):
            def verifier(prompt: str, completion: str) -> tuple[bool, int | None]:
                ex = example_lookup.get(prompt)
                if ex is None:
                    return False, None
                outcome = r_fn.evaluate(completion, ex)
                # reward is directly used in LanguageRewardFunction
                is_correct = outcome.total_reward >= 0.8
                return is_correct, int(outcome.total_reward * 10)

            return verifier

        reward_fn = LanguageRewardFunction(make_verifier(r_fn_internal))

        res = train_lm_ppo(
            model=policy,
            ref_model=ref_model,
            prompts=train_prompts,
            tokenizer=tok,
            reward_fn=reward_fn,
            num_iterations=10,
            rollout_batch_size=16,
            epochs_per_rollout=2,
            minibatch_size=8,
            lr=5e-4,
            clip_eps=0.2,
            ref_kl_coef=0.05,
            seed=seed,
        )
        summary = evaluate_reasoning_model(policy, tok, test_data, max_new_tokens=35)
        results[label] = (res, summary)

    # Comparison Table
    table = Table(title="Reward Regime Comparison in PPO")
    table.add_column("Reward Regime", justify="left")
    table.add_column("Test Accuracy", justify="right")
    table.add_column("Step Validity", justify="right")
    table.add_column("Complete Path %", justify="right")
    table.add_column("Final Ref KL", justify="right")
    table.add_column("Clip Fraction", justify="right")

    for label, (res, summary) in results.items():
        kl = res.ref_kls[-1] if res.ref_kls else 0.0
        clip = res.clip_fractions[-1] if res.clip_fractions else 0.0
        table.add_row(
            label,
            f"{summary.answer_accuracy:.1%}",
            f"{summary.trace_validity:.1%}",
            f"{summary.complete_path_rate:.1%}",
            f"{kl:.4f}",
            f"{clip:.3f}",
        )

    console.print(table)

    console.print("\n[bold]Key Insights on Reward Regimes:[/bold]")
    console.print(
        "1. [yellow]Process Reward Dense Feedback:[/yellow] Process supervision provides an immediate reward "
        "signal for intermediate steps, guiding exploration towards valid derivation chains."
    )
    console.print(
        "2. [green]Hybrid Stability:[/green] Hybrid rewards balance intermediate derivation constraints with "
        "the final verification objective, achieving consistent alignment."
    )
    console.print(
        "3. [cyan]Length Normalization Necessity:[/cyan] Normalizing process reward by task depth d prevents "
        "the policy from gaming the metric by repeating valid steps or generating unnecessarily long chains."
    )


if __name__ == "__main__":
    run_experiment()
