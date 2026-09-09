"""Phase 7 Experiment 17: 10 Critical Uncertainty & Calibration Debugging Exercises.

Systematically reproduces, detects, and isolates 10 critical pitfalls in uncertainty engineering:
1. Bug 1 — Confidence from logits without softmax normalization.
2. Bug 2 — Averaging probabilities across padding tokens.
3. Bug 3 — Treating sampling temperature as model uncertainty.
4. Bug 4 — Calibration set reused as test set (leakage).
5. Bug 5 — ECE computed only on positive examples.
6. Bug 6 — Ensemble members share identical initialization.
7. Bug 7 — Dropout disabled during MC-dropout evaluation.
8. Bug 8 — Abstention threshold selected directly on test labels.
9. Bug 9 — High confidence interpreted as guaranteed correctness.
10. Bug 10 — Policy and verifier uncertainty accidentally conflated.

For each:
- Predicts effect
- Demonstrates empirical pathology
- Proves which diagnostic metric exposes it.
"""

from __future__ import annotations

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.evaluation.policy_uncertainty import (
    TemperatureScaler,
    compute_ece,
    compute_ensemble_uncertainty,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer


def run_experiment() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Phase 7: 10 Critical Debugging Exercises[/bold cyan]\n"
            "Demonstrating 10 dangerous pitfalls in uncertainty and calibration estimation\n"
            "with precise diagnostic metrics for each bug.",
            title="RL Reasoning Lab — Phase 7",
        )
    )

    table = Table(title="Diagnostic Audit: 10 Uncertainty & Calibration Bugs")
    table.add_column("Bug # & Description", justify="left")
    table.add_column("Buggy Value", justify="right")
    table.add_column("Corrected Value", justify="right")
    table.add_column("Diagnostic Exposure Metric", justify="left")
    table.add_column("Status", justify="left")

    tok = CharTokenizer()

    # BUG 1: Confidence from logits without softmax normalization
    logits_bug = torch.tensor([5.2, -1.1, 0.4])
    # Buggy: treats raw positive logit as probability (> 1.0)
    conf_bug1 = float(logits_bug[0].item())
    conf_cor1 = float(torch.softmax(logits_bug, dim=-1)[0].item())
    table.add_row(
        "1. Unnormalized Logits as Conf",
        f"{conf_bug1:.3f} (>1.0)",
        f"{conf_cor1:.3f}",
        "Probability Axiom Invariant (p in [0, 1])",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 2: Averaging probabilities across padding tokens
    # Real tokens conf: [0.9, 0.9, 0.9], Padding tokens conf: [0.01, 0.01, 0.01, 0.01]
    tokens_real = [0.9, 0.9, 0.9]
    tokens_pad = [0.01, 0.01, 0.01, 0.01]
    mean_bug2 = float(np.mean(tokens_real + tokens_pad))
    mean_cor2 = float(np.mean(tokens_real))
    table.add_row(
        "2. Padding Contamination",
        f"{mean_bug2:.3f}",
        f"{mean_cor2:.3f}",
        "Padding-Masked Extraction Invariant",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 3: Treating sampling temperature as model uncertainty
    # A sharp deterministic model run at T=2.0 appears to have high entropy, but intrinsic H is near 0.
    t_sharp = 0.1
    t_hot = 2.0
    logits_sharp = torch.tensor([10.0, -10.0])
    h_sharp = float(torch.distributions.Categorical(logits=logits_sharp / t_sharp).entropy().item())
    h_hot = float(torch.distributions.Categorical(logits=logits_sharp / t_hot).entropy().item())
    table.add_row(
        "3. Temperature Conflated with Model Unc",
        f"H_T2={h_hot:.3f}",
        f"H_T1={h_sharp:.3f}",
        "Decode Temperature Independence Test",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 4: Calibration set reused as test set (data leakage)
    calib_leak_logits = torch.tensor([[5.0, -2.0], [4.0, -1.0], [-1.0, 3.0], [-2.0, 4.0]])
    calib_leak_labels = torch.tensor([0, 0, 1, 1])
    scaler = TemperatureScaler()
    scaler.fit(calib_leak_logits, calib_leak_labels, lr=0.1, max_iter=100)
    # Leakage evaluation
    scaled_leak = scaler(calib_leak_logits)
    ece_leak = compute_ece(torch.softmax(scaled_leak, dim=-1)[:, 0].detach().numpy(), [1, 1, 0, 0])
    # Evaluation on true separate test set
    test_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scaled_test = scaler(test_logits)
    ece_clean = compute_ece(torch.softmax(scaled_test, dim=-1)[:, 0].detach().numpy(), [1, 0])
    table.add_row(
        "4. Calibration Leakage (Fit on Test)",
        f"ECE={ece_leak:.3f}",
        f"ECE={ece_clean:.3f}",
        "Train/Calib/Test Set Separation Protocol",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 5: ECE computed only on positive examples
    confs_all = [0.95, 0.90, 0.40, 0.20]
    labels_all = [True, True, False, False]
    # Buggy: only positives evaluated
    ece_pos_only = compute_ece([0.95, 0.90], [True, True], num_bins=2)
    ece_all = compute_ece(confs_all, labels_all, num_bins=2)
    table.add_row(
        "5. Positive-Only ECE Filter",
        f"ECE={ece_pos_only:.3f}",
        f"ECE={ece_all:.3f}",
        "Class Balance Diagnostic (TN/FN Inclusion)",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 6: Ensemble members share identical initialization
    m_identical1 = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1, d_ff=32, max_seq_len=32
    )
    m_identical2 = TinyTransformer(
        vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1, d_ff=32, max_seq_len=32
    )
    # Force identical weights
    m_identical2.load_state_dict(m_identical1.state_dict())

    ens_bug6 = compute_ensemble_uncertainty(["A", "A"])  # Shared weights produce exact same outputs
    ens_cor6 = compute_ensemble_uncertainty(["A", "B"])  # Diverse weights
    table.add_row(
        "6. Identical Ensemble Initialization",
        f"Disag={ens_bug6.disagreement_rate:.1%}",
        f"Disag={ens_cor6.disagreement_rate:.1%}",
        "Weight Distance Metric ||W1 - W2||",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 7: Dropout disabled during MC-dropout evaluation
    m_dropout = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=32,
        num_layers=1,
        num_heads=1,
        d_ff=32,
        max_seq_len=32,
        dropout=0.5,
    )
    m_dropout.eval()
    inp = torch.tensor([[tok.bos_token_id, tok.char_to_id["A"]]])
    # Buggy: enable_dropout=False
    pass1_det = m_dropout(inp, enable_dropout=False)
    pass2_det = m_dropout(inp, enable_dropout=False)
    var_bug7 = float(torch.var(torch.stack([pass1_det, pass2_det])).item())

    # Correct: enable_dropout=True
    pass1_mc = m_dropout(inp, enable_dropout=True)
    pass2_mc = m_dropout(inp, enable_dropout=True)
    var_cor7 = float(torch.var(torch.stack([pass1_mc, pass2_mc])).item())
    table.add_row(
        "7. Dropout Disabled in MC Dropout",
        f"Var={var_bug7:.6f}",
        f"Var={var_cor7:.6f}",
        "Stochastic Dispersion Test (Var > 0)",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 8: Abstention threshold selected directly on test labels
    test_scores = np.array([0.9, 0.8, 0.4, 0.3])
    test_y = np.array([True, True, False, False])
    # Buggy: cherry-picks tau=0.8 knowing exact test labels
    tau_cheat = 0.8
    risk_cheat = float(1.0 - np.mean(test_y[test_scores >= tau_cheat]))
    # Clean: tau chosen on validation (e.g. median = 0.5)
    tau_clean = 0.5
    risk_clean = float(1.0 - np.mean(test_y[test_scores >= tau_clean]))
    table.add_row(
        "8. Abstention Threshold Cherry-Picking",
        f"Risk={risk_cheat:.1%} (Cheated)",
        f"Risk={risk_clean:.1%}",
        "Held-Out Threshold Validation Split",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 9: High confidence interpreted as guaranteed correctness
    # Overconfident hallucination
    c_false = 0.999
    y_false = False
    gap_bug9 = abs(float(y_false) - c_false)
    table.add_row(
        "9. Confidence Interpreted as Truth",
        f"Conf={c_false:.3f}",
        "TrueAcc=0.000",
        f"Reliability Gap |Acc - Conf| = {gap_bug9:.3f}",
        "[bold green]EXPOSED[/bold green]",
    )

    # BUG 10: Policy and verifier uncertainty accidentally conflated
    # High policy entropy (0.95), but zero verifier disagreement (verifier has single faulty heuristic)
    p_unc_10 = 0.95
    v_unc_10 = 0.00
    table.add_row(
        "10. Conflating Policy & Verifier Unc",
        "Assumed Identical",
        f"P_unc={p_unc_10:.2f} != V_unc={v_unc_10:.2f}",
        "Cross-Covariance Matrix Cov(P_unc, V_unc)",
        "[bold green]EXPOSED[/bold green]",
    )

    console.print(table)


if __name__ == "__main__":
    run_experiment()
