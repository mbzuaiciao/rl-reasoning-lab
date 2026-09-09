"""Phase 7 Experiment 6: Distractor-Induced Uncertainty and Counterfactual Sensitivity.

Part 1: Distractor-Induced Uncertainty
- Varies irrelevant distractor premise count k in {0, 1, 2, 3}.
- Evaluates:
  - Task accuracy
  - Policy entropy
  - Minimum token confidence
  - Verifier ensemble disagreement

Part 2: Counterfactual Sensitivity
- Evaluates paired examples differing by exactly one causal edge intervention.
- Measures:
  - Answer flipping rate
  - Confidence shift magnitude |Conf_orig - Conf_cf|
  - Entropy response when premises change.
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
from rl_reasoning_lab.evaluation.policy_uncertainty import extract_trajectory_uncertainty
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.trainers.sft import train_sft
from rl_reasoning_lab.verifiers.composite_verifiers import VerifierEnsemble
from rl_reasoning_lab.verifiers.proxy_verifiers import (
    FormatVerifier,
    KeywordVerifier,
    SubstringAnswerVerifier,
)
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
            "[bold cyan]Phase 7: Distractor-Induced Uncertainty & Counterfactual Sensitivity[/bold cyan]\n"
            "Assessing whether policy uncertainty responds appropriately to extraneous\n"
            "distractors and counterfactual premise interventions.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGHIJKL"), shuffle_rules=True)
    evaluator = TraceEvaluator()

    # Verifier ensemble for disagreement measurement
    ensemble = VerifierEnsemble(
        verifiers=[
            SubstringAnswerVerifier(),
            KeywordVerifier(),
            FormatVerifier(),
        ],
        aggregation="majority_vote",
    )

    # Train base policy
    train_data = task.generate_dataset(size=60, depths=[1, 2], num_distractors=1, seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], num_distractors=1, seed=seed + 1)

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

    # PART 1: Distractor-induced uncertainty sweep
    console.print("[dim]Evaluating distractor sweeps (k in {0, 1, 2, 3})...[/dim]")
    distractor_counts = [0, 1, 2, 3]
    d_table = Table(title="Part 1: Distractor Scaling vs. Uncertainty")
    d_table.add_column("Distractors (k)", justify="center")
    d_table.add_column("Accuracy", justify="right")
    d_table.add_column("Mean Token Entropy", justify="right")
    d_table.add_column("Min Token Conf", justify="right")
    d_table.add_column("Final-Ans Conf", justify="right")
    d_table.add_column("Verifier Disagreement", justify="right")

    for k in distractor_counts:
        test_k = task.generate_dataset(size=30, depths=[2], num_distractors=k, seed=seed + 10 + k)
        accs: list[bool] = []
        entropies: list[float] = []
        min_confs: list[float] = []
        ans_confs: list[float] = []
        disagreements: list[float] = []

        for ex in test_k:
            p_ids = tok.encode(ex.prompt, add_bos=True)
            full_ids, _ = model.generate(
                p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
            )
            comp_ids = full_ids[0, len(p_ids) :].tolist()
            comp_text = tok.decode(comp_ids, skip_special=True)

            rep = evaluator.evaluate(comp_text, ex)
            unc = extract_trajectory_uncertainty(model, tok, ex.prompt, comp_text)
            v_res = ensemble.verify(comp_text, ex)

            accs.append(rep.answer_correct)
            entropies.append(unc.mean_token_entropy)
            min_confs.append(unc.min_token_confidence)
            ans_confs.append(unc.final_answer_prob)
            disagreements.append(float(v_res.metadata.get("disagreement", 0.0)))

        d_table.add_row(
            str(k),
            f"{float(np.mean(accs)):.1%}",
            f"{float(np.mean(entropies)):.3f}",
            f"{float(np.mean(min_confs)):.3f}",
            f"{float(np.mean(ans_confs)):.3f}",
            f"{float(np.mean(disagreements)):.3f}",
        )

    console.print(d_table)

    # PART 2: Counterfactual Sensitivity
    console.print("[dim]Evaluating paired counterfactual interventions...[/dim]")
    num_pairs = 25
    rng = random.Random(seed + 99)
    answer_flipped = 0
    conf_deltas: list[float] = []
    entropy_deltas: list[float] = []

    for _ in range(num_pairs):
        # Generate base example
        ex_orig = task.generate_example(depth=2, num_distractors=1, rng=rng)

        # Create counterfactual example by intervening on the last step: B->C replaced by B->Z
        # Target node changes from ex_orig.target_node to Z
        new_target = "Z" if "Z" not in ex_orig.path else "Y"
        cf_edges = [
            (u, v) if (u, v) != ex_orig.edges[-1] else (ex_orig.edges[-1][0], new_target)
            for u, v in ex_orig.edges
        ]
        all_cf_rules = list(cf_edges) + list(ex_orig.distractor_edges)
        cf_prompt = (
            "; ".join(f"{u}->{v}" for u, v in all_cf_rules) + f"; Q: {ex_orig.start_node}->?"
        )

        # Generate on original prompt
        p_ids_orig = tok.encode(ex_orig.prompt, add_bos=True)
        f_ids_orig, _ = model.generate(
            p_ids_orig, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        comp_orig = tok.decode(f_ids_orig[0, len(p_ids_orig) :].tolist(), skip_special=True)
        unc_orig = extract_trajectory_uncertainty(model, tok, ex_orig.prompt, comp_orig)
        rep_orig = evaluator.evaluate(comp_orig, ex_orig)

        # Generate on counterfactual prompt
        p_ids_cf = tok.encode(cf_prompt, add_bos=True)
        f_ids_cf, _ = model.generate(
            p_ids_cf, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        comp_cf = tok.decode(f_ids_cf[0, len(p_ids_cf) :].tolist(), skip_special=True)
        unc_cf = extract_trajectory_uncertainty(model, tok, cf_prompt, comp_cf)
        rep_cf = evaluator.evaluate(comp_cf, ex_orig)

        if rep_orig.extracted_answer != rep_cf.extracted_answer:
            answer_flipped += 1

        conf_deltas.append(abs(unc_orig.final_answer_prob - unc_cf.final_answer_prob))
        entropy_deltas.append(abs(unc_orig.mean_token_entropy - unc_cf.mean_token_entropy))

    cf_table = Table(title="Part 2: Counterfactual Sensitivity Metrics")
    cf_table.add_column("Sensitivity Metric", justify="left")
    cf_table.add_column("Empirical Result", justify="right")

    cf_table.add_row("Total Paired Interventions", str(num_pairs))
    cf_table.add_row("Answer Flip Rate under Premise Edit", f"{answer_flipped / num_pairs:.1%}")
    cf_table.add_row("Mean |Confidence Shift|", f"{float(np.mean(conf_deltas)):.3f}")
    cf_table.add_row("Mean |Entropy Shift|", f"{float(np.mean(entropy_deltas)):.3f}")

    console.print(cf_table)


if __name__ == "__main__":
    run_experiment()
