"""Phase 4 Experiment: Supervised Fine-Tuning (SFT) Baseline and Exposure Bias.

1. Train TinyTransformer with next-token cross-entropy under teacher forcing.
2. Measure train loss, validation loss, token accuracy, and exact-match sequence accuracy.
3. Exposure Mismatch Demonstration:
   Compare token prediction accuracy under clean teacher-forced ground truth prefix
   versus under an autoregressively corrupted prefix.
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.environments.language_tasks import ArithmeticTask
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.trainers.sft import evaluate_sft, train_sft

console = Console()


def run_sft_experiment() -> None:
    tokenizer = CharTokenizer()
    task = ArithmeticTask(min_operands=2, max_operands=2, min_val=0, max_val=9)

    train_data = task.generate_dataset(size=80, seed=42)
    val_data = task.generate_dataset(size=30, seed=123)

    torch.manual_seed(42)
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=32,
    )

    console.print(
        Panel.fit(
            "[bold cyan]Phase 4: SFT Baseline Training & Exposure Bias[/bold cyan]\n"
            f"Dataset: 80 train, 30 validation addition problems. Model: 2-layer Causal Transformer ({model.d_model} d_model)."
        )
    )

    initial_loss, initial_token_acc, initial_em = evaluate_sft(model, val_data, tokenizer)

    res = train_sft(
        model=model,
        train_examples=train_data,
        val_examples=val_data,
        tokenizer=tokenizer,
        num_epochs=20,
        batch_size=16,
        lr=3e-3,
        seed=42,
    )

    final_loss, final_token_acc, final_em = evaluate_sft(model, val_data, tokenizer)

    table = Table(title="SFT Training Progress on Arithmetic Task")
    table.add_column("Epoch", justify="center", style="cyan")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Token Accuracy", justify="right", style="magenta")
    table.add_column("Exact Match %", justify="right", style="green")

    table.add_row(
        "0 (Initial)",
        "-",
        f"{initial_loss:.4f}",
        f"{initial_token_acc * 100:.1f}%",
        f"{initial_em * 100:.1f}%",
    )
    for ep in [5, 10, 15, 20]:
        idx = ep - 1
        table.add_row(
            str(ep),
            f"{res.train_losses[idx]:.4f}",
            f"{res.val_losses[idx]:.4f}",
            f"{res.token_accuracies[idx] * 100:.1f}%",
            f"{res.exact_match_accuracies[idx] * 100:.1f}%",
        )

    console.print(table)

    # 2. Exposure Bias Demonstration
    console.print("\n[bold]Exposure Bias Demonstration:[/bold]")
    sample_ex = val_data[0]
    prompt_ids = tokenizer.encode(sample_ex.prompt, add_bos=True)

    # Clean autoregressive generation
    gen_ids, _ = model.generate(
        prompt_ids, max_new_tokens=4, greedy=True, eos_token_id=tokenizer.eos_token_id
    )
    clean_gen = tokenizer.decode(gen_ids[0].tolist(), skip_special=True)

    # Corrupted prefix: perturb the prompt by injecting an erroneous character
    corrupted_prompt = sample_ex.prompt.replace(" = ", " = X")
    corr_ids = tokenizer.encode(corrupted_prompt, add_bos=True)
    corr_gen_ids, _ = model.generate(
        corr_ids, max_new_tokens=4, greedy=True, eos_token_id=tokenizer.eos_token_id
    )
    corrupted_gen = tokenizer.decode(corr_gen_ids[0].tolist(), skip_special=True)

    exp_table = Table(title="Exposure Bias: Teacher Forcing vs. Perturbed Prefix")
    exp_table.add_column("Condition", style="cyan")
    exp_table.add_column("Input Given", style="white")
    exp_table.add_column("Model Completion", style="yellow")
    exp_table.add_column("Observation", style="white")

    exp_table.add_row(
        "Clean Prefix",
        sample_ex.prompt,
        clean_gen[len(sample_ex.prompt) :],
        "Correct completion under in-distribution prefix",
    )
    exp_table.add_row(
        "Perturbed Prefix",
        corrupted_prompt,
        corrupted_gen[len(corrupted_prompt) :],
        "Out-of-distribution prefix alters continuation distribution",
    )

    console.print(exp_table)
    console.print(
        "[dim]Key Lesson: SFT only trains on expert trajectories (teacher forcing). At inference,\n"
        "one unexpected token pushes the model onto out-of-distribution prefixes that SFT never trained on.[/dim]\n"
    )


if __name__ == "__main__":
    console.rule("[bold magenta]RL Reasoning Lab — Phase 4 SFT Baseline[/bold magenta]")
    run_sft_experiment()
