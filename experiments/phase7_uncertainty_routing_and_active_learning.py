"""Phase 7 Experiment 13: Uncertainty-Aware Routing & Active Learning Simulation.

Builds on Phase 6 by evaluating when uncertainty signals should trigger expensive ground-truth supervision:
1. Verification Escalation Pipeline:
   Cheap proxy verifier -> If uncertain -> Escalates to ground-truth Oracle.
2. Compares Routing Triggers:
   - Always Cheap Proxy (0% Oracle Budget)
   - Random Escalation (Baseline matching budget)
   - Verifier Disagreement Routing
   - Policy Entropy Routing
   - Joint / Hybrid Routing (Policy Entropy OR Verifier Disagreement)
   - Always Oracle (100% Budget Upper Bound)
3. Active Learning Simulation:
   Measures accuracy gain per oracle query used.
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
    evaluate_uncertainty_routing,
    extract_trajectory_uncertainty,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.trainers.sft import train_sft
from rl_reasoning_lab.verifiers.composite_verifiers import VerifierEnsemble
from rl_reasoning_lab.verifiers.oracle_verifier import OracleVerifier
from rl_reasoning_lab.verifiers.proxy_verifiers import (
    FormatVerifier,
    KeywordVerifier,
    SubstringAnswerVerifier,
)


def to_sft_examples(examples: list[ReasoningExample]) -> list[Example]:
    return [
        Example(prompt=ex.prompt, target=ex.trace_target, full_text=ex.prompt + ex.trace_target)
        for ex in examples
    ]


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 7: Uncertainty-Aware Routing & Active Supervision[/bold cyan]\n"
            "Routing rollouts from cheap proxy verifiers to expensive ground-truth oracles\n"
            "using policy entropy vs. verifier disagreement vs. joint uncertainty.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = CharTokenizer()
    task = GraphTraversalTask(node_pool=list("ABCDEFGH"), shuffle_rules=True)

    oracle = OracleVerifier()
    cheap_verifier = SubstringAnswerVerifier()
    ensemble_verifier = VerifierEnsemble(
        verifiers=[
            SubstringAnswerVerifier(),
            KeywordVerifier(),
            FormatVerifier(),
        ],
        aggregation="majority_vote",
    )

    train_data = task.generate_dataset(size=50, depths=[1, 2], num_distractors=1, seed=seed)
    val_data = task.generate_dataset(size=20, depths=[1, 2], num_distractors=1, seed=seed + 1)
    test_data = task.generate_dataset(size=60, depths=[1, 2, 3], num_distractors=1, seed=seed + 2)

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

    # Collect rollouts and evaluations
    cheap_decisions: list[bool] = []
    oracle_decisions: list[bool] = []
    policy_entropies: list[float] = []
    verifier_disagreements: list[float] = []
    hybrid_uncertainties: list[float] = []

    for ex in test_data:
        p_ids = tok.encode(ex.prompt, add_bos=True)
        f_ids, _ = model.generate(
            p_ids, max_new_tokens=35, greedy=True, eos_token_id=tok.eos_token_id
        )
        comp = tok.decode(f_ids[0, len(p_ids) :].tolist(), skip_special=True)

        c_res = cheap_verifier.verify(comp, ex)
        o_res = oracle.verify(comp, ex)
        e_res = ensemble_verifier.verify(comp, ex)
        unc = extract_trajectory_uncertainty(model, tok, ex.prompt, comp)

        cheap_decisions.append(c_res.passed)
        oracle_decisions.append(o_res.passed)

        p_ent = unc.mean_token_entropy
        v_dis = float(e_res.metadata.get("disagreement", 0.0))

        policy_entropies.append(p_ent)
        verifier_disagreements.append(v_dis)
        # Normalized hybrid score
        hybrid_uncertainties.append(0.5 * (p_ent / 2.0) + 0.5 * v_dis)

    # Thresholds tuned to approximate 30-40% budget
    th_policy = float(np.percentile(policy_entropies, 65))
    th_verifier = 0.01  # Any non-zero disagreement triggers escalation
    th_hybrid = float(np.percentile(hybrid_uncertainties, 65))

    # Evaluate routing strategies
    res_cheap = evaluate_uncertainty_routing(
        cheap_decisions,
        oracle_decisions,
        policy_entropies,
        threshold=999.0,
        strategy_name="Always Cheap Proxy",
    )
    res_oracle = evaluate_uncertainty_routing(
        cheap_decisions,
        oracle_decisions,
        policy_entropies,
        threshold=-1.0,
        strategy_name="Always Oracle (100% Budget)",
    )
    res_policy = evaluate_uncertainty_routing(
        cheap_decisions,
        oracle_decisions,
        policy_entropies,
        threshold=th_policy,
        strategy_name="Policy Entropy Trigger",
    )
    res_verifier = evaluate_uncertainty_routing(
        cheap_decisions,
        oracle_decisions,
        verifier_disagreements,
        threshold=th_verifier,
        strategy_name="Verifier Disagreement Trigger",
    )
    res_hybrid = evaluate_uncertainty_routing(
        cheap_decisions,
        oracle_decisions,
        hybrid_uncertainties,
        threshold=th_hybrid,
        strategy_name="Hybrid (Policy + Verifier) Trigger",
    )

    # Random escalation matching the budget of hybrid
    target_budget = res_hybrid.oracle_budget_used
    rng = random.Random(seed)
    random_scores = [rng.random() for _ in range(len(test_data))]
    th_random = float(np.percentile(random_scores, 100 * (1 - target_budget / len(test_data))))
    res_random = evaluate_uncertainty_routing(
        cheap_decisions,
        oracle_decisions,
        random_scores,
        threshold=th_random,
        strategy_name="Random Escalation (Matched Budget)",
    )

    table = Table(title="Uncertainty-Aware Routing Benchmark")
    table.add_column("Routing Strategy", justify="left")
    table.add_column("Oracle Queries Used", justify="right")
    table.add_column("Oracle Budget %", justify="right")
    table.add_column("Verification Accuracy", justify="right")
    table.add_column("Exploits Prevented", justify="right")

    for r in [res_cheap, res_random, res_policy, res_verifier, res_hybrid, res_oracle]:
        table.add_row(
            r.strategy,
            f"{r.oracle_budget_used}/{len(test_data)}",
            f"{r.oracle_query_fraction:.1%}",
            f"{r.final_accuracy:.1%}",
            str(r.exploits_prevented),
        )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
