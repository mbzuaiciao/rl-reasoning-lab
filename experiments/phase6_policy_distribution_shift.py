"""Phase 6 Experiment 4: Policy-Induced Distribution Shift on Verifiers.

Core Concept:
A verifier may appear highly accurate on static or pre-optimization policy traces,
achieving high test accuracy. However, policy optimization actively searches the tail
of the policy's distribution for outputs that maximize proxy reward.
This causes a profound distribution shift:
  VerifierAccuracy_pre  vs.  VerifierAccuracy_optimized_policy

Demonstrates:
1. Verifier accuracy measured on pre-RL rollouts vs. post-RL rollouts.
2. Exploit emergence: tracking exploit frequency over optimization epochs.
3. Confusion matrix migration: False Positives expand as the policy learns the exploit.
"""

from __future__ import annotations

import random

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.lm_ppo import train_lm_ppo
from rl_reasoning_lab.environments.language_tasks import Example
from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.evaluation.verifier_metrics import evaluate_verifier_diagnostics
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft
from rl_reasoning_lab.verifiers.oracle_verifier import OracleVerifier
from rl_reasoning_lab.verifiers.proxy_verifiers import PartialPathVerifier


def main() -> None:
    console = Console()
    console.print(
        Panel(
            "[bold cyan]Phase 6 Experiment 4: Policy-Induced Verifier Distribution Shift[/bold cyan]\n"
            "[dim]Measures verifier diagnostic accuracy on pre-RL rollouts vs post-RL rollouts.\n"
            "Shows that optimization shifts the output distribution toward verifier blind spots.[/dim]",
            border_style="cyan",
        )
    )

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    task = GraphTraversalTask()
    tok = CharTokenizer()

    # Create dataset
    train_data = task.generate_dataset(size=60, depths=[1, 2, 3], seed=seed)
    eval_data = task.generate_dataset(size=30, depths=[1, 2, 3], seed=seed + 10)

    train_prompts = [ex.prompt for ex in train_data]
    sft_train_examples = [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in train_data
    ]
    sft_val_examples = [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in eval_data[:10]
    ]

    model = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
    )
    console.print("[dim]Training base SFT model...[/dim]")
    train_sft(
        model=model,
        train_examples=sft_train_examples,
        val_examples=sft_val_examples,
        tokenizer=tok,
        num_epochs=12,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    ref_model = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
    )
    ref_model.load_state_dict(model.state_dict())

    # Target verifier is PartialPathVerifier (checks individual edges, ignores continuity & endpoints)
    verifier = PartialPathVerifier()
    oracle = OracleVerifier(strict_path=True)

    lookup = {ex.prompt: ex for ex in train_data}

    def reward_fn(p: str, c: str) -> tuple[bool, int | None]:
        ex = lookup.get(p)
        if ex is None:
            return False, None
        v_res = verifier.verify(c, ex)
        return v_res.passed, int(v_res.score * 10)

    def sample_rollouts(m: TinyTransformer, dataset: list[ReasoningExample]) -> list[str]:
        m.eval()
        outputs = []
        with torch.no_grad():
            for ex in dataset:
                prompt_ids = tok.encode(ex.prompt, add_bos=True, add_eos=False)
                gen_ids, _ = m.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=40,
                    greedy=True,
                    eos_token_id=tok.eos_token_id,
                )
                comp = tok.decode(gen_ids[0, len(prompt_ids) :].tolist(), skip_special=True)
                outputs.append(comp)
        return outputs

    # 1. Pre-optimization evaluation
    console.print("\n[bold]Evaluating Verifier on Pre-Optimization Rollouts...[/bold]")
    pre_rollouts = sample_rollouts(model, eval_data)
    diag_pre = evaluate_verifier_diagnostics(verifier, oracle, pre_rollouts, eval_data)

    # 2. Optimize policy against PartialPathVerifier
    console.print("\n[dim]Optimizing policy against PartialPathVerifier (PPO)...[/dim]")
    exploit_progression: list[float] = []

    def check_exploit_rate(m: TinyTransformer) -> float:
        # Exploit: output has valid individual edges (passed by PartialPath) but broken continuity
        rollouts = sample_rollouts(m, eval_data)
        exploits = 0
        for comp, ex in zip(rollouts, eval_data, strict=False):
            v_pass = verifier.verify(comp, ex).passed
            o_pass = oracle.verify(comp, ex).passed
            if v_pass and not o_pass:
                exploits += 1
        return exploits / len(eval_data)

    exploit_progression.append(check_exploit_rate(model))

    for it in range(1, 5):
        train_lm_ppo(
            model=model,
            ref_model=ref_model,
            prompts=train_prompts,
            tokenizer=tok,
            reward_fn=LanguageRewardFunction(reward_fn),
            num_iterations=1,
            rollout_batch_size=16,
            epochs_per_rollout=2,
            minibatch_size=8,
            lr=1e-3,
            clip_eps=0.2,
            ref_kl_coef=0.02,
            seed=500 + it,
        )
        rate = check_exploit_rate(model)
        exploit_progression.append(rate)
        console.print(
            f"  Iteration {it}: Exploit Rate (Passed Verifier but Failed Oracle) = {rate:.1%}"
        )

    # 3. Post-optimization evaluation
    console.print("\n[bold]Evaluating Verifier on Post-Optimization Rollouts...[/bold]")
    post_rollouts = sample_rollouts(model, eval_data)
    diag_post = evaluate_verifier_diagnostics(verifier, oracle, post_rollouts, eval_data)

    # Compare diagnostic summary before and after optimization
    table = Table(title="Policy-Induced Distribution Shift on PartialPathVerifier")
    table.add_column("Metric", style="bold white")
    table.add_column("Pre-RL Rollouts (SFT)", justify="right", style="cyan")
    table.add_column("Post-RL Rollouts (Optimized)", justify="right", style="magenta")
    table.add_column("Impact / Shift Direction", justify="left")

    table.add_row(
        "Verifier Accuracy",
        f"{diag_pre.accuracy:.1%}",
        f"{diag_post.accuracy:.1%}",
        "[bold red]Severe Degradation[/bold red]"
        if diag_post.accuracy < diag_pre.accuracy
        else "Stable",
    )
    table.add_row(
        "True Positives (TP)", str(diag_pre.tp), str(diag_post.tp), "True correct solutions"
    )
    table.add_row(
        "False Positives (FP)",
        str(diag_pre.fp),
        str(diag_post.fp),
        "[red]Exploited blind spots[/red]",
    )
    table.add_row("True Negatives (TN)", str(diag_pre.tn), str(diag_post.tn), "Correctly rejected")
    table.add_row("False Negatives (FN)", str(diag_pre.fn), str(diag_post.fn), "Missed correct")
    table.add_row(
        "Precision", f"{diag_pre.precision:.1%}", f"{diag_post.precision:.1%}", "Reward reliability"
    )
    table.add_row(
        "Recall", f"{diag_pre.recall:.1%}", f"{diag_post.recall:.1%}", "Coverage of valid"
    )
    table.add_row(
        "Reward Density",
        f"{diag_pre.reward_density:.1%}",
        f"{diag_post.reward_density:.1%}",
        "Fraction approved",
    )

    console.print(table)

    # Exploit progression
    table_prog = Table(title="Exploit Rate Emergence Across Training")
    table_prog.add_column("Step", justify="center", style="bold white")
    table_prog.add_column("Exploit Rate (FP / N)", justify="right", style="bold yellow")
    table_prog.add_column("Interpretation", justify="left")

    for idx, rate in enumerate(exploit_progression):
        interp = "SFT baseline distribution" if idx == 0 else f"PPO exploration round {idx}"
        table_prog.add_row(f"Round {idx}", f"{rate:.1%}", interp)

    console.print(table_prog)
    console.print(
        "\n[bold green]Conclusion:[/bold green] Evaluating a verifier exclusively on IID pre-training data produces "
        "a dangerously false sense of security. As optimization proceeds, the policy reshapes the data distribution "
        "toward regions where the verifier makes systematic false-positive errors.\n"
    )


if __name__ == "__main__":
    main()
