# Phase 10 Preregistered Hypotheses

> *"A hypothesis must be stated before the data is collected. A finding that only emerges post hoc is an exploratory observation, not a confirmed prediction."*

Date Preregistered: 2026-09-07T13:50:00Z  
Evaluation Environment: `rl-reasoning-lab` Phase 10 Synthetic DAG Reasoning Environment

---

## Preregistration Protocol

1. All hypotheses below are recorded prior to executing final Phase 10 benchmark evaluations.
2. Hypotheses specify both the research prediction ($H_1$) and the explicit null hypothesis ($H_0$).
3. Directional assertions are paired with predefined statistical tests (95% non-parametric bootstrap confidence intervals on paired seed differences).
4. No hypotheses will be modified or removed after evaluation begins. Any exploratory finding will be explicitly labeled as post-hoc.

---

## Primary Hypotheses

### Hypothesis 1 (H1) — Outcome-Only RL vs Process Validity
- **Prediction ($H_1$)**: Outcome-only PPO (rewarding only terminal answer correctness) will improve in-distribution (IID) answer accuracy, but will exhibit a substantial **process validity gap**: complete-path validity will remain significantly lower than answer accuracy due to hallucinated or shortcut steps.
  $$\Delta_{\text{IID}}(\text{Outcome PPO}) = \text{AnswerAccuracy} - \text{CompletePathRate} > 0.25$$
- **Null Hypothesis ($H_0$)**: Outcome-only PPO improves intermediate complete-path validity at a rate equal to or greater than answer accuracy ($\Delta_{\text{IID}} \le 0.05$).

### Hypothesis 2 (H2) — Process-Supervised RL Generalization
- **Prediction ($H_2$)**: Process-reward PPO (rewarding valid step-by-step transitions) will produce significantly greater out-of-distribution (OOD) depth generalization ($d \in \{4, 5, 6\}$) and counterfactual consistency than outcome-only PPO under matched training budgets.
  $$\mathbb{E}[\text{OOD Acc}_{\text{Process}}] - \mathbb{E}[\text{OOD Acc}_{\text{Outcome}}] > 0.20 \quad \text{with } 95\% \text{ CI excluding } 0$$
- **Null Hypothesis ($H_0$)**: Process-reward PPO yields OOD depth accuracy and counterfactual consistency less than or equal to outcome-only PPO.

### Hypothesis 3 (H3) — Reward Quality and Goodhart Divergence
- **Prediction ($H_3$)**: The efficacy of RL depends critically on verifier specification. When trained on a weak proxy verifier (e.g., substring matching or length heuristics), the policy's training proxy reward $R_{\text{training}}$ will increase while ground-truth oracle utility $U_{\text{oracle}}$ stagnates or declines (Goodhart divergence). Invariant-gated verifiers will mitigate this divergence.
  $$\frac{d R_{\text{proxy}}}{dt} > 0 \quad \text{and} \quad \frac{d U_{\text{oracle}}}{dt} \le 0$$
- **Null Hypothesis ($H_0$)**: Optimizing against a weak proxy verifier produces proportional gains in ground-truth oracle utility without divergence.

### Hypothesis 4 (H4) — Reference KL Regularization and Calibration
- **Prediction ($H_4$)**: Incorporating a moderate reference-model KL penalty ($\beta_{\text{kl}} \in [0.02, 0.10]$) will prevent entropy collapse and mitigate overconfidence (lowering ECE and Brier score) during extended PPO training, without reducing peak oracle accuracy by more than $5\%$.
  $$\text{ECE}(\text{PPO}_{\text{KL}}) < \text{ECE}(\text{PPO}_{\text{no-KL}}) \quad \text{and} \quad \mathcal{H}(\pi_{\text{KL}}) > \mathcal{H}(\pi_{\text{no-KL}})$$
- **Null Hypothesis ($H_0$)**: Reference KL penalty either fails to reduce calibration error or causes severe task performance degradation ($>15\%$ drop).

### Hypothesis 5 (H5) — Uncertainty-Aware Verifier Routing Efficiency
- **Prediction ($H_5$)**: Routing uncertain trajectories to an expensive oracle based on verifier disagreement or ensemble entropy will detect significantly more false-positive exploits per oracle query than uniform random auditing under a fixed oracle budget ($B_{\text{oracle}} = 20\%$ of samples).
  $$\text{ExploitDetectionRate}_{\text{Disagreement}} > 2.0 \times \text{ExploitDetectionRate}_{\text{Random}}$$
- **Null Hypothesis ($H_0$)**: Uncertainty-based routing detects exploits at a rate statistically indistinguishable from random auditing.

### Hypothesis 6 (H6) — Rejection Sampling Competitiveness
- **Prediction ($H_6$)**: Amortized search via Rejection Sampling (filtering multi-sample rollouts with an oracle verifier and fine-tuning with SFT) will achieve task accuracy and path validity competitive with or superior to PPO under matched generated token budgets on small-scale reasoning graphs.
  $$|\text{Accuracy}_{\text{RS}} - \text{Accuracy}_{\text{PPO}}| \le 0.05$$
- **Null Hypothesis ($H_0$)**: PPO significantly outperforms Rejection Sampling by $\ge 0.15$ in accuracy under equal total token budgets.

### Hypothesis 7 (H7) — Test-Time Search vs Policy Weight Updates
- **Prediction ($H_7$)**: Applying Best-of-$N$ ($N \in \{2, 4, 8\}$) inference search to a frozen SFT policy will produce substantial answer accuracy gains without modifying policy parameters. Furthermore, at high $N$, SFT + Best-of-$N$ will match or exceed single-sample PPO accuracy, proving that apparent reasoning gains can be manufactured purely through inference compute.
  $$\text{Accuracy}(\text{SFT} + \text{BoN}_8) \ge \text{Accuracy}(\text{PPO}_{\text{greedy}})$$
- **Null Hypothesis ($H_0$)**: Best-of-$N$ inference search on the base SFT model cannot match greedy PPO policy performance ($p < 0.05$).

---

## Summary of Preregistered Conditions

| ID | Hypothesis Focus | Key Metric | Planned Statistical Test |
| :--- | :--- | :--- | :--- |
| **H1** | Outcome PPO Process Gap | Answer Acc vs Path Validity | Paired bootstrap 95% CI on $(\text{Acc} - \text{PathRate})$ |
| **H2** | Process PPO OOD Depth | OOD Accuracy ($d=4..6$) | Paired difference bootstrap 95% CI vs Outcome PPO |
| **H3** | Goodhart Divergence | $R_{\text{proxy}}$ vs $U_{\text{oracle}}$ delta | Time-series correlation & final gap |
| **H4** | KL Penalty Calibration | ECE & Entropy retention | Bootstrap 95% CI on ECE reduction |
| **H5** | Verifier Routing Budget | Exploit catch rate / call | Two-sample difference of proportions test |
| **H6** | Rejection Sampling Parity | IID & OOD Accuracy | Two-sided equivalence test ($\pm 5\%$) |
| **H7** | Best-of-$N$ Search Frontier | Accuracy vs Total Tokens | Compute-accuracy Pareto frontier dominance |
