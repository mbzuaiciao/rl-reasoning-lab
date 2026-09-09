"""Phase 7 Experiment 9: Bayesian Parameter Uncertainty & Epistemic Decomposition.

Evaluates an educational Bayesian linear output head over frozen transformer representations:
1. Maintains a posterior distribution over head weights: W ~ N(mu, sigma^2).
2. Decomposes total predictive uncertainty into:
   Total Uncertainty (H(p_bar)) = Aleatoric Uncertainty (E[H(p)]) + Epistemic Uncertainty (Mutual Info).
3. Evaluates on:
   - In-Distribution (IID) prompts (seen during training)
   - Out-of-Distribution (OOD) novel prompts / novel graph tokens.
4. Directly tests:
   Does epistemic uncertainty spike on OOD inputs where ordinary token entropy fails?
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
from rl_reasoning_lab.evaluation.policy_uncertainty import BayesianLinearHead
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
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
            "[bold cyan]Phase 7: Bayesian Parameter Uncertainty & Epistemic Decomposition[/bold cyan]\n"
            "Decomposing total predictive entropy into aleatoric task noise and epistemic\n"
            "parameter uncertainty using a Bayesian head over transformer representations.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGH"))

    train_data = task.generate_dataset(size=50, depths=[1, 2], seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], seed=seed + 1)
    iid_test = task.generate_dataset(size=30, depths=[1, 2], seed=seed + 2)

    # OOD task using unseen letters and depth 4
    task_ood = GraphTraversalTask(node_pool=list("IJKLMNOPQRST"))
    ood_test = task_ood.generate_dataset(size=30, depths=[4], seed=seed + 3)

    console.print("[dim]Training base transformer backbone...[/dim]")
    backbone = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        max_seq_len=128,
    )
    train_sft(
        model=backbone,
        train_examples=to_sft_examples(train_data),
        val_examples=to_sft_examples(val_data),
        tokenizer=tok,
        num_epochs=10,
        batch_size=16,
        lr=3e-3,
        seed=seed,
    )
    backbone.eval()

    # Create Bayesian linear head mapping transformer d_model -> vocab_size
    console.print("[dim]Attaching Bayesian linear head over final representations...[/dim]")
    b_head = BayesianLinearHead(in_features=backbone.d_model, out_features=tok.vocab_size)

    # Function to extract hidden representation before LM head
    def get_terminal_representation(prompt_str: str) -> torch.Tensor:
        p_ids = tok.encode(prompt_str, add_bos=True)
        inp = torch.tensor([p_ids], dtype=torch.long)
        with torch.no_grad():
            # Pass through embeddings + blocks + ln_f
            positions = torch.arange(0, inp.size(1)).unsqueeze(0)
            x = backbone.tok_embeddings(inp) + backbone.pos_embeddings(positions)
            for block in backbone.blocks:
                x = block(x)
            h = backbone.ln_f(x)[:, -1, :]  # (1, d_model)
        return h

    def evaluate_uncertainty_split(dataset: list[ReasoningExample]):
        total_list: list[float] = []
        aleatoric_list: list[float] = []
        epistemic_list: list[float] = []

        for ex in dataset:
            h = get_terminal_representation(ex.prompt)
            decomp = b_head.decompose_uncertainty(h, num_samples=30)
            total_list.append(decomp.total_uncertainty)
            aleatoric_list.append(decomp.aleatoric_uncertainty)
            epistemic_list.append(decomp.epistemic_uncertainty)

        return {
            "total": float(np.mean(total_list)),
            "aleatoric": float(np.mean(aleatoric_list)),
            "epistemic": float(np.mean(epistemic_list)),
        }

    iid_metrics = evaluate_uncertainty_split(iid_test)
    ood_metrics = evaluate_uncertainty_split(ood_test)

    table = Table(title="Bayesian Uncertainty Decomposition: IID vs. OOD Distribution Shift")
    table.add_column("Uncertainty Component", justify="left")
    table.add_column("In-Distribution (IID)", justify="right")
    table.add_column("Out-of-Distribution (OOD)", justify="right")
    table.add_column("OOD Shift Ratio (OOD / IID)", justify="right")

    tot_ratio = ood_metrics["total"] / (iid_metrics["total"] + 1e-12)
    ale_ratio = ood_metrics["aleatoric"] / (iid_metrics["aleatoric"] + 1e-12)
    epi_ratio = ood_metrics["epistemic"] / (iid_metrics["epistemic"] + 1e-12)

    table.add_row(
        "Total Predictive Uncertainty H(p_bar)",
        f"{iid_metrics['total']:.4f}",
        f"{ood_metrics['total']:.4f}",
        f"{tot_ratio:.2f}x",
    )
    table.add_row(
        "Aleatoric Uncertainty E[H(p)] (Data Noise)",
        f"{iid_metrics['aleatoric']:.4f}",
        f"{ood_metrics['aleatoric']:.4f}",
        f"{ale_ratio:.2f}x",
    )
    table.add_row(
        "Epistemic Uncertainty I(W; Y|X) (Parameter Disagreement)",
        f"{iid_metrics['epistemic']:.4f}",
        f"{ood_metrics['epistemic']:.4f}",
        f"[bold cyan]{epi_ratio:.2f}x[/bold cyan]",
    )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
