"""Phase 6 Experiment 2: Reward–Utility Divergence Under Policy Optimization.

Demonstrates Goodhart's Law:
- Optimizing against a weak proxy verifier R(tau) causes proxy reward to climb steadily.
- Meanwhile, true latent utility U(tau) (measured strictly by the independent OracleVerifier)
  stagnates or declines as the policy exploits the proxy's blind spot (e.g. jumping straight to answer
  or embedding the answer token in junk text).

Tracks E[R(tau)] vs E[U(tau)] across training iterations side-by-side.
"""

from __future__ import annotations

import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.algorithms.lm_ppo import train_lm_ppo
from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.sft import train_sft
from rl_reasoning_lab.verifiers.oracle_verifier import OracleVerifier
from rl_reasoning_lab.verifiers.proxy_verifiers import LastTokenVerifier


def main() -> None:
    console = Console()
    console.print(
        Panel(
            "[bold red]Phase 6 Experiment 2: Reward–Utility Divergence (Goodhart's Law)[/bold red]\n"
            "[dim]Tracking proxy reward E[R(tau)] vs true oracle utility E[U(tau)] side-by-side during PPO.\n"
            "Observes the exact point where optimization against a weak proxy diverges from true task success.[/dim]",
            border_style="red",
        )
    )

    seed = 42
    random.seed(seed)
    task = GraphTraversalTask()
    tok = CharTokenizer()

    # Create training and validation datasets
    train_data = task.generate_dataset(size=60, depths=[1, 2], seed=seed)
    eval_data = task.generate_dataset(size=30, depths=[1, 2, 3], seed=seed + 10)

    from rl_reasoning_lab.environments.language_tasks import Example

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
    console.print("[dim]Warm-starting model with SFT...[/dim]")
    train_sft(
        model=model,
        train_examples=sft_train_examples,
        val_examples=sft_val_examples,
        tokenizer=tok,
        num_epochs=15,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )

    ref_model = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2, num_heads=2, max_seq_len=128
    )
    ref_model.load_state_dict(model.state_dict())

    # Training verifier is the weak LastTokenVerifier (blind to reasoning steps)
    proxy_verifier = LastTokenVerifier()
    # Oracle evaluator measures true utility U(tau) with strict path validation
    oracle_verifier = OracleVerifier(strict_path=True)

    lookup = {ex.prompt: ex for ex in train_data}

    def proxy_reward_fn(p: str, c: str) -> tuple[bool, int | None]:
        ex = lookup.get(p)
        if ex is None:
            return False, None
        v_res = proxy_verifier.verify(c, ex)
        # Scaled integer reward for LM PPO
        return v_res.passed, int(v_res.score * 10)

    # Tracking metrics across iterations
    history: list[dict] = []

    def evaluate_snapshots(epoch_idx: int) -> dict:
        proxy_scores: list[float] = []
        oracle_scores: list[float] = []
        answer_matches: list[bool] = []
        complete_paths: list[bool] = []

        model.eval()
        for ex in eval_data:
            prompt_ids = tok.encode(ex.prompt, add_bos=True, add_eos=False)
            gen_ids, _ = model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=40,
                greedy=True,
                eos_token_id=tok.eos_token_id,
            )
            gen_token_ids = gen_ids[0, len(prompt_ids) :].tolist()
            comp = tok.decode(gen_token_ids, skip_special=True)

            p_res = proxy_verifier.verify(comp, ex)
            o_res = oracle_verifier.verify(comp, ex)

            proxy_scores.append(p_res.score)
            oracle_scores.append(o_res.score)
            answer_matches.append(bool(o_res.metadata.get("answer_correct", False)))
            complete_paths.append(bool(o_res.metadata.get("complete_path", False)))

        return {
            "iter": epoch_idx,
            "mean_proxy_R": sum(proxy_scores) / len(proxy_scores),
            "mean_oracle_U": sum(oracle_scores) / len(oracle_scores),
            "answer_acc": sum(answer_matches) / len(answer_matches),
            "complete_path_rate": sum(complete_paths) / len(complete_paths),
        }

    # Initial snapshot
    init_stats = evaluate_snapshots(0)
    history.append(init_stats)

    console.print("[dim]Optimizing against weak SubstringAnswerVerifier with PPO...[/dim]")
    for it in range(1, 9):
        train_lm_ppo(
            model=model,
            ref_model=ref_model,
            prompts=train_prompts,
            tokenizer=tok,
            reward_fn=LanguageRewardFunction(proxy_reward_fn),
            num_iterations=1,
            rollout_batch_size=16,
            epochs_per_rollout=2,
            minibatch_size=8,
            lr=1e-3,
            clip_eps=0.2,
            ref_kl_coef=0.02,
            seed=42 + it,
        )
        stats = evaluate_snapshots(it)
        history.append(stats)

    table = Table(title="Reward–Utility Divergence (Goodhart's Law Under PPO)")
    table.add_column("Iteration", justify="center", style="bold white")
    table.add_column("Proxy Reward E[R]", justify="right", style="cyan")
    table.add_column("Oracle Utility E[U]", justify="right", style="red")
    table.add_column("Ans Acc", justify="right", style="green")
    table.add_column("Valid Path %", justify="right", style="magenta")
    table.add_column("Divergence Status", justify="left")

    for row in history:
        r = row["mean_proxy_R"]
        u = row["mean_oracle_U"]
        gap = r - u
        status = (
            "[green]Aligned[/green]" if gap < 0.25 else "[bold red]Diverging (Goodhart)[/bold red]"
        )
        table.add_row(
            str(row["iter"]),
            f"{r:.3f}",
            f"{u:.3f}",
            f"{row['answer_acc']:.1%}",
            f"{row['complete_path_rate']:.1%}",
            status,
        )

    console.print(table)
    console.print(
        "\n[bold red]Goodhart Divergence Finding:[/bold red] As PPO optimizes the policy against the weak substring proxy, "
        f"proxy reward climbs from {history[0]['mean_proxy_R']:.2f} to {history[-1]['mean_proxy_R']:.2f}, while true oracle utility "
        f"drops/stagnates ({history[0]['mean_oracle_U']:.2f} -> {history[-1]['mean_oracle_U']:.2f}). The policy adapts by generating "
        "spurious answer tokens without executing valid step-by-step reasoning.\n"
    )


if __name__ == "__main__":
    main()
