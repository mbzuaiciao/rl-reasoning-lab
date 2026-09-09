"""Phase 6 Experiment 3: Random Noise vs. Systematic Verifier Errors Under Policy Optimization.

Core Research Question:
Are structured, systematic verifier errors more dangerous than equally frequent random errors?

Experimental Setup:
1. Verifier A (Random Noise): Flips judgments with probability epsilon=0.15 (unstructured noise).
2. Verifier B (Systematic Shortcut): Accepts completions whenever a trigger phrase ('verified:qed') appears.
3. Both policies start from the exact same SFT warm-start checkpoint.
4. We monitor Proxy Reward E[R], True Utility E[U], and Exploit Frequency across iterations.

Key Finding:
Random noise degrades sample efficiency due to gradient variance, but preserves true utility.
Systematic error creates a catastrophic optimization attractor: the policy rapidly learns to exploit
the loophole, driving proxy reward to maximum while true utility collapses.
"""

from __future__ import annotations

import random

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.lm_ppo import train_lm_ppo
from rl_reasoning_lab.environments.language_tasks import Example
from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft
from rl_reasoning_lab.verifiers.oracle_verifier import OracleVerifier
from rl_reasoning_lab.verifiers.proxy_verifiers import (
    RandomNoiseVerifier,
    SystematicShortcutVerifier,
)


def main() -> None:
    console = Console()
    console.print(
        Panel(
            "[bold yellow]Phase 6 Experiment 3: Random Noise vs. Systematic Verifier Errors[/bold yellow]\n"
            "[dim]Comparing policy optimization against symmetric random label flips vs an exploitable shortcut.\n"
            "Investigates whether structured errors are fundamentally more destructive than unstructured noise.[/dim]",
            border_style="yellow",
        )
    )

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    task = GraphTraversalTask()
    tok = CharTokenizer()

    train_data = task.generate_dataset(size=50, depths=[1, 2], seed=seed)
    eval_data = task.generate_dataset(size=25, depths=[1, 2, 3], seed=seed + 10)

    train_prompts = [ex.prompt for ex in train_data]
    sft_train_examples = [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in train_data
    ]
    sft_val_examples = [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in eval_data[:10]
    ]

    base_model = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
    )
    console.print("[dim]Training base SFT model...[/dim]")
    train_sft(
        model=base_model,
        train_examples=sft_train_examples,
        val_examples=sft_val_examples,
        tokenizer=tok,
        num_epochs=12,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    oracle = OracleVerifier(strict_path=True)
    v_random = RandomNoiseVerifier(base_verifier=oracle, epsilon=0.15, seed=seed)
    v_systematic = SystematicShortcutVerifier(base_verifier=oracle, trigger_phrase="verified:qed")

    lookup = {ex.prompt: ex for ex in train_data}

    def evaluate_policy(model: TinyTransformer, verifier) -> dict[str, float]:
        model.eval()
        p_scores = []
        u_scores = []
        exploit_count = 0
        with torch.no_grad():
            for ex in eval_data:
                prompt_ids = tok.encode(ex.prompt, add_bos=True, add_eos=False)
                gen_ids, _ = model.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=40,
                    greedy=True,
                    eos_token_id=tok.eos_token_id,
                )
                comp = tok.decode(gen_ids[0, len(prompt_ids) :].tolist(), skip_special=True)
                v_res = verifier.verify(comp, ex)
                o_res = oracle.verify(comp, ex)
                p_scores.append(v_res.score)
                u_scores.append(o_res.score)
                if "verified:qed" in comp.lower():
                    exploit_count += 1

        n = len(eval_data)
        return {
            "proxy_R": sum(p_scores) / n,
            "oracle_U": sum(u_scores) / n,
            "exploit_rate": exploit_count / n,
        }

    # 1. Train Policy A against Random Noise
    console.print("\n[bold cyan]1. Optimizing against Verifier A (Random Noise 15%)[/bold cyan]")
    model_a = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
    )
    model_a.load_state_dict(base_model.state_dict())
    ref_a = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
    )
    ref_a.load_state_dict(base_model.state_dict())

    def reward_fn_a(p: str, c: str) -> tuple[bool, int | None]:
        ex = lookup.get(p)
        if ex is None:
            return False, None
        res = v_random.verify(c, ex)
        return res.passed, int(res.score * 10)

    stats_a_history = [evaluate_policy(model_a, v_random)]
    for it in range(1, 5):
        train_lm_ppo(
            model=model_a,
            ref_model=ref_a,
            prompts=train_prompts,
            tokenizer=tok,
            reward_fn=LanguageRewardFunction(reward_fn_a),
            num_iterations=1,
            rollout_batch_size=16,
            epochs_per_rollout=2,
            minibatch_size=8,
            lr=1e-3,
            clip_eps=0.2,
            ref_kl_coef=0.02,
            seed=100 + it,
        )
        stats_a_history.append(evaluate_policy(model_a, v_random))

    # 2. Train Policy B against Systematic Shortcut
    console.print(
        "\n[bold magenta]2. Optimizing against Verifier B (Systematic Shortcut)[/bold magenta]"
    )
    model_b = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
    )
    model_b.load_state_dict(base_model.state_dict())
    ref_b = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
    )
    ref_b.load_state_dict(base_model.state_dict())

    def reward_fn_b(p: str, c: str) -> tuple[bool, int | None]:
        ex = lookup.get(p)
        if ex is None:
            return False, None
        res = v_systematic.verify(c, ex)
        return res.passed, int(res.score * 10)

    stats_b_history = [evaluate_policy(model_b, v_systematic)]
    for it in range(1, 5):
        train_lm_ppo(
            model=model_b,
            ref_model=ref_b,
            prompts=train_prompts,
            tokenizer=tok,
            reward_fn=LanguageRewardFunction(reward_fn_b),
            num_iterations=1,
            rollout_batch_size=16,
            epochs_per_rollout=2,
            minibatch_size=8,
            lr=1e-3,
            clip_eps=0.2,
            ref_kl_coef=0.02,
            seed=200 + it,
        )
        stats_b_history.append(evaluate_policy(model_b, v_systematic))

    # Display results table
    table = Table(title="Policy Trajectories: Random Noise vs Systematic Shortcut")
    table.add_column("Iteration", justify="center", style="bold white")
    table.add_column("Random Proxy E[R]", justify="right", style="cyan")
    table.add_column("Random Oracle E[U]", justify="right", style="green")
    table.add_column("Systematic Proxy E[R]", justify="right", style="magenta")
    table.add_column("Systematic Oracle E[U]", justify="right", style="red")
    table.add_column("Exploit Freq (B)", justify="right", style="bold yellow")

    for i in range(len(stats_a_history)):
        sa = stats_a_history[i]
        sb = stats_b_history[i]
        table.add_row(
            str(i),
            f"{sa['proxy_R']:.3f}",
            f"{sa['oracle_U']:.3f}",
            f"{sb['proxy_R']:.3f}",
            f"{sb['oracle_U']:.3f}",
            f"{sb['exploit_rate']:.1%}",
        )

    console.print(table)
    console.print(
        "\n[bold yellow]Key Theoretical Finding:[/bold yellow] Even when random noise and systematic error have "
        "similar baseline frequencies, [bold red]systematic errors provide a coherent gradient direction[/bold red] that "
        "actively pulls the policy toward shortcut exploitation. Random noise merely acts as variance, slowing learning "
        "without destroying true task capability.\n"
    )


if __name__ == "__main__":
    main()
