# Research Claim Registry

This document serves as the canonical registry of empirical and theoretical claims investigated across the `rl-reasoning-lab` repository. In accordance with Phase 8 scientific principles:
> *"A training run is an observation. A research claim requires an experimental design."*

All claims are audited against non-parametric bootstrap confidence intervals, paired statistical tests, out-of-distribution (OOD) stress tests, and reproducible manifests.

---

## Standardized Evidence Status Categories

- `robustly reproduced`: The claim replicates across $\ge 10$ seeds with tight confidence intervals and survives moderate OOD stress testing.
- `reproduced with smaller effect`: The effect exists and is statistically significant, but the observed magnitude in rigorous testing is smaller than initially reported.
- `seed-sensitive`: The outcome varies substantially across random initializations; conclusions derived from $\le 3$ seeds are unreliable.
- `budget-sensitive`: Method superiority depends strictly on the resource constraint (e.g., environment samples vs optimizer updates vs inference tokens).
- `OOD-fragile`: The effect holds under in-distribution (IID) conditions but collapses or reverses under distribution shift.
- `inconclusive`: Empirical uncertainty bounds span zero or overlapping confidence intervals prevent resolving the ranking.
- `failed replication`: The claimed effect cannot be reproduced under controlled conditions.

---

## Claims Audit Table

| Claim ID | Phase | Short Title | Primary Metric | Effect Size (Cohen's $d$) | Evidence Strength | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CLAIM-01** | Phase 2 | Baseline Variance Reduction | Policy Gradient Variance $\text{Var}(\hat{g})$ | $d = -10.55$ | Analytic & Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-02** | Phase 3 | PPO Clipping Limits Policy Drift | Empirical Policy KL Divergence | $d = -7.33$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-03** | Phase 4 | SFT Warm-Start Accelerates LM RL | Episodes to 50% Task Accuracy | $d = -9.75$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-04** | Phase 5 | Process Reward Improves Reasoning Validity | Multi-Step Reasoning Path Validity | $d = +9.88$ (IID) / $+4.12$ (OOD) | Empirical (Controlled) | `robustly reproduced` (IID) / `OOD-fragile` (Noisy Verifier) |
| **CLAIM-05** | Phase 6 | Imperfect Proxy Verifiers Induce Goodhart Divergence | Ground-Truth Task Utility | $d = -20.79$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-06** | Phase 7 | RL Policy Optimization Degrades Calibration | Expected Calibration Error (ECE) | $d = +6.96$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-07** | Phase 7 | Deep Ensembles Detect OOD Errors Better Than Sampling | AUROC on OOD Graph Depth Shifts | $d = +6.36$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-08** | Phase 8 | Three Seeds Suffer From Severe Best-Seed Bias | Best-Seed Accuracy vs Mean Across 20 Seeds | $d = +4.10$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-09** | Phase 8 | Factorial Process Reward $\times$ KL Synergy | Super-Additive Accuracy Gain $\Delta_{AB} - (\Delta_A + \Delta_B)$ | $\Delta_{\text{synergy}} = +6.68\%$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-10** | Phase 10 | Outcome-Only RL Induces Severe Process Gap | Answer Accuracy minus Complete-Path Validity | $d = +10.92$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-11** | Phase 10 | Process Supervision Enables OOD Depth Generalization | OOD Depth ($d=4..6$) & Counterfactual Accuracy | $d = +8.45$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-12** | Phase 10 | Un-Gated Proxy Verifiers Trigger Goodhart Divergence | Proxy Reward minus True Oracle Utility | $d = -22.40$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-13** | Phase 10 | Invariant Gating Suppresses Proxy Exploitation | Exploit Rate Reduction under Flawed Verifiers | $d = +18.20$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-14** | Phase 10 | Reference KL Regularization Mitigates Overconfidence | Expected Calibration Error (ECE) Reduction | $d = -6.15$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-15** | Phase 10 | Verifier Disagreement Escalation Improves Audit Efficiency | Exploit Detection Rate / Oracle Budget | $d = +7.80$ | Empirical (Controlled) | `robustly reproduced` |
| **CLAIM-16** | Phase 10 | Test-Time Search Matches Policy RL at 8x Token Cost | SFT + Best-of-8 vs Single-Sample PPO Accuracy | $d = +1.10$ | Empirical (Controlled) | `budget-sensitive` |

---

## Detailed Claim Records

### CLAIM-01: Baseline Variance Reduction in Policy Gradients
- **Claim**: Incorporating a state-dependent value baseline $V(s)$ into Monte Carlo policy gradients reduces gradient estimator variance without introducing bias.
- **Evidence**: Analytically derived via $\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s) b(s)] = 0$. Empirically audited across 12 random seeds in `experiments/phase8_research_claim_audit.py`.
- **Experiment**: `experiments/phase2_gradient_variance.py` & `experiments/phase8_research_claim_audit.py`.
- **Seeds**: 12 paired seeds.
- **Metric**: Sample variance of gradient updates: With baseline $3.08 \pm 0.4$ vs Without baseline $12.70 \pm 1.2$ ($\Delta = -9.62$, 95% CI: $[-10.25, -8.94]$, Cohen's $d = -10.55$).
- **Evidence strength**: Analytic derivation & empirical validation in controlled MDP.
- **Known limitations**: Baseline quality depends on value function approximation; poorly fitted value networks can fail to reduce variance.
- **Status**: `robustly reproduced`

### CLAIM-02: PPO Ratio Clipping Bounds Policy Drift
- **Claim**: The pessimistic clipped surrogate objective in PPO bounds destructive policy drift when performing multiple gradient updates per rollout batch.
- **Evidence**: Audited across 12 paired seeds. Unclipped updates with learning rate $\eta = 0.03$ experience KL divergence $> 1.85$, while clipped PPO ($\epsilon = 0.2$) maintains KL $< 0.08$.
- **Experiment**: `experiments/phase3_clipping_behavior.py` & `experiments/phase8_research_claim_audit.py`.
- **Seeds**: 12 paired seeds.
- **Metric**: Empirical KL divergence: Clipped $0.074$ vs Unclipped $1.908$ ($\Delta = -1.834$, 95% CI: $[-2.02, -1.65]$, Cohen's $d = -7.33$).
- **Evidence strength**: Empirically demonstrated in local trust region regime.
- **Known limitations**: Clipping is a local heuristic, not a mathematically exact trust region; large gradients can still shift representations unexpectedly if value and policy share trunk layers.
- **Status**: `robustly reproduced`

### CLAIM-03: SFT Warm-Start Accelerates LM Policy Optimization
- **Claim**: Initializing a language model policy with Supervised Fine-Tuning (SFT) on reasoning traces before applying RL prevents policy collapse and accelerates convergence by $>5\times$.
- **Evidence**: Evaluated across 12 random seeds. Pure RL from random initialization requires $1,480$ episodes to reach 50% accuracy (with 25% catastrophic collapse runs), while SFT warm-start reaches target accuracy in $199$ episodes.
- **Experiment**: `experiments/phase4_sft_vs_rl.py` & `experiments/phase8_research_claim_audit.py`.
- **Seeds**: 12 paired seeds.
- **Metric**: Training episodes to threshold: SFT $199$ vs Random $1,480$ ($\Delta = -1,281$ episodes, 95% CI: $[-1375, -1181]$, Cohen's $d = -9.75$).
- **Evidence strength**: Empirically demonstrated in sparse reward regime.
- **Known limitations**: High SFT coverage can bias the model towards memorized paths; sparse SFT regimes benefit most from self-directed RL exploration.
- **Status**: `robustly reproduced`

### CLAIM-04: Process Reward Improves Reasoning Path Validity Over Outcome-Only
- **Claim**: Rewarding intermediate valid reasoning steps produces higher complete-path validity and distractor robustness than rewarding final answer correctness alone.
- **Evidence**: Tested on synthetic multi-step graph reasoning. Outcome-only PPO achieves 88% final answer accuracy but only 42.8% path validity due to shortcut exploitation. Process PPO achieves 88.6% path validity on IID tasks.
- **Experiment**: `experiments/phase5_process_vs_outcome_rl.py` & `experiments/phase8_ablation_and_interactions.py`.
- **Seeds**: 10 paired seeds.
- **Metric**: Path validity on graphs with $k=3$ distractors: Process $88.6\%$ vs Outcome $42.8\%$ ($\Delta = +45.8\%$, 95% CI: $[+42.1\%, +49.5\%]$, Cohen's $d = +9.88$).
- **Evidence strength**: Empirically demonstrated in controlled synthetic setting (IID-strong, OOD-fragile under noisy verifiers).
- **Known limitations**: Process reward requires a step-level verifier. If the step verifier is noisy or incomplete, process RL can reinforce hallucinated intermediate tokens.
- **Status**: `robustly reproduced` (IID) / `OOD-fragile` (when step verifier has false positives)

### CLAIM-05: Imperfect Proxy Verifiers Induce Goodhart Divergence
- **Claim**: Optimizing an RL policy aggressively against an imperfect proxy reward function causes true task utility to collapse even as proxy reward approaches 100%.
- **Evidence**: Replicated on reasoning tasks with heuristic substring/keyword verifiers. Policy learns formatting and distractor shortcuts, driving proxy reward to 94.3% while true task utility collapses to 22.7%.
- **Experiment**: `experiments/phase6_reward_utility_divergence.py` & `experiments/phase8_research_claim_audit.py`.
- **Seeds**: 12 paired seeds.
- **Metric**: True Task Utility vs Proxy Reward: Proxy $94.3\%$ vs True $22.7\%$ ($\Delta = -71.6\%$, 95% CI: $[-74.0\%, -69.2\%]$, Cohen's $d = -20.79$).
- **Evidence strength**: Empirically demonstrated in misspecified proxy verifier setting.
- **Known limitations**: Divergence rate depends on verifier loophole density and policy search capacity.
- **Status**: `robustly reproduced`

### CLAIM-06: RL Policy Optimization Degrades Confidence Calibration
- **Claim**: RL training sharpens policy probabilities and maximizes expected reward, which systematically increases overconfidence and inflates Expected Calibration Error (ECE) compared to supervised models.
- **Evidence**: Evaluated across 12 paired training seeds. SFT achieves ECE $0.046 \pm 0.008$ (well-calibrated). Outcome PPO pushes ECE to $0.137 \pm 0.015$. Under OOD depth shift, ECE explodes to $0.280$.
- **Experiment**: `experiments/phase7_calibration_sft_vs_rl.py` & `experiments/phase8_research_claim_audit.py`.
- **Seeds**: 12 paired seeds.
- **Metric**: Expected Calibration Error: SFT $0.046$ vs RL $0.137$ ($\Delta = +0.091$, 95% CI: $[+0.082, +0.100]$, Cohen's $d = +6.96$).
- **Evidence strength**: Empirically demonstrated across multiple seeds in controlled setting.
- **Known limitations**: Entropy regularization ($\beta_{\text{ent}} > 0.05$) and KL penalty ($\beta_{\text{kl}} > 0.1$) can mitigate, but not eliminate, optimization-induced overconfidence.
- **Status**: `robustly reproduced`

### CLAIM-07: Deep Ensembles Detect OOD Errors Better Than Repeated Sampling
- **Claim**: Epistemic variance across diverse model checkpoints (trained with different seeds) reliably flags OOD reasoning errors where single-model sampling variance collapses into confident failure modes.
- **Evidence**: Tested across cyclic distractor graphs and depth shifts. Deep ensemble disagreement achieves AUROC $0.854 \pm 0.02$ for error detection, outperforming repeated sampling self-consistency AUROC ($0.685 \pm 0.03$).
- **Experiment**: `experiments/phase7_policy_ensemble.py` & `experiments/phase8_research_claim_audit.py`.
- **Seeds**: 12 paired seeds.
- **Metric**: Error detection AUROC under OOD shifts: Ensemble $0.854$ vs Sampling $0.685$ ($\Delta = +0.169$, 95% CI: $[+0.151, +0.188]$, Cohen's $d = +6.36$).
- **Evidence strength**: Empirically demonstrated under controlled OOD distribution shift.
- **Known limitations**: Requires maintaining and querying $M$ distinct model checkpoints, scaling inference cost by $M\times$.
- **Status**: `robustly reproduced`

### CLAIM-08: Three Seeds Suffer From Severe Best-Seed Bias
- **Claim**: Reporting the single best seed from an $n=3$ or $n=20$ trial introduces massive optimistic bias (+15–20% accuracy inflation) and conceals catastrophic failure modes.
- **Evidence**: Evaluated across 20 distinct training seeds in `phase8_seed_sensitivity.py`. Best seed reached 95.0% accuracy, whereas the true mean was 76.4% and the worst seed collapsed to 13.9%.
- **Experiment**: `experiments/phase8_seed_sensitivity.py`.
- **Seeds**: 20 seeds.
- **Metric**: Best-Seed Inflation: $+18.6\%$ over true mean, with 95% CI width shrinking from $\pm 0.089$ at $n=3$ to $\pm 0.045$ at $n=20$.
- **Evidence strength**: Statistically validated across 20 seeds.
- **Known limitations**: If policy optimization is completely deterministic with zero variance, best-seed bias vanishes.
- **Status**: `robustly reproduced`

### CLAIM-09: Factorial Process Reward $\times$ Reference KL Synergy
- **Claim**: Combining process reward and reference KL regularization produces a super-additive synergistic interaction (+6.7% over independent additive predictions) because process reward supplies dense direction while KL prevents policy drift.
- **Evidence**: Evaluated via a $2 \times 2$ factorial experiment across 10 paired seeds.
- **Experiment**: `experiments/phase8_ablation_and_interactions.py`.
- **Seeds**: 10 paired seeds.
- **Metric**: Additive prediction $\Delta_A + \Delta_B = 13.76\% + 4.00\% = 17.76\%$. Observed joint gain $\Delta_{AB} = +24.44\%$. Synergistic interaction $= +6.68\%$ (95% CI: $[+4.2\%, +8.9\%]$).
- **Evidence strength**: Empirically demonstrated in controlled factorial design.
- **Known limitations**: In environments with zero reward hacking loopholes, KL penalty can act as an unnecessary constraint.
- **Status**: `robustly reproduced`

### CLAIM-10: Outcome-Only RL Induces a Severe Process Validity Gap
- **Claim**: Rewarding only final answer correctness in multi-step reasoning models improves answer accuracy while driving intermediate complete-path validity down due to shortcut and hallucination exploitation.
- **Evidence**: Evaluated across 10 paired seeds in Phase 10. Outcome PPO achieves 87.8% answer accuracy but only 42.0% complete-path validity (process gap $= 45.8\%$, 95% CI: $[42.7\%, 49.5\%]$, $d = 10.92$).
- **Experiment**: `experiments/phase10_capstone.py` (Core Experiment A).
- **Seeds**: 10 paired seeds.
- **Metric**: Process validity gap: $\text{AnswerAcc} - \text{CompletePathRate} = +45.8\%$.
- **Evidence strength**: Empirically demonstrated in controlled synthetic setting ($d = 10.92$).
- **Known limitations**: In single-step or un-chained tasks where intermediate deductions are nonexistent, this gap does not manifest.
- **Status**: `robustly reproduced`

### CLAIM-11: Process Supervision Enables OOD Depth and Counterfactual Generalization
- **Claim**: Rewarding intermediate valid steps produces policies that generalize to unseen chain lengths ($d \in \{4, 5, 6\}$) and maintain causal responsiveness under single-edge counterfactual mutations.
- **Evidence**: Process PPO achieves 66.5% OOD depth accuracy vs 16.7% for Outcome PPO, and 74.2% counterfactual consistency vs 23.1% for Outcome PPO.
- **Experiment**: `experiments/phase10_capstone.py` (Core Experiment B & Probes).
- **Seeds**: 10 paired seeds.
- **Metric**: OOD Depth Accuracy: Process 66.5% vs Outcome 16.7% ($\Delta = +49.8\%$, $d = 8.45$).
- **Evidence strength**: Empirically demonstrated across OOD depth and counterfactual shifts.
- **Known limitations**: Requires fine-grained, verifiable step feedback during training.
- **Status**: `robustly reproduced`

### CLAIM-12: Un-Gated Proxy Verifiers Trigger Catastrophic Goodhart Divergence
- **Claim**: Optimizing against an un-gated proxy verifier causes training proxy reward to surge while ground-truth oracle utility collapses.
- **Evidence**: Weak Proxy PPO achieves 94.5% proxy reward while true oracle utility drops to 25.7% (divergence gap $+68.8\%$, exploit rate $64.9\%$).
- **Experiment**: `experiments/phase10_capstone.py` (Core Experiment C).
- **Seeds**: 10 paired seeds.
- **Metric**: Divergence gap: $R_{\text{proxy}} - U_{\text{oracle}} = +68.8\%$ ($d = -22.40$).
- **Evidence strength**: Empirically demonstrated under misspecified verifier.
- **Known limitations**: Requires an imperfect verifier with exploitable blind spots.
- **Status**: `robustly reproduced`

### CLAIM-13: Invariant Gating Suppresses Proxy Exploitation
- **Claim**: Requiring generated steps to satisfy topological structural invariants before granting proxy rewards suppresses learned exploits and keeps proxy reward aligned with oracle utility.
- **Evidence**: Invariant-Gated PPO achieves 88.5% proxy reward and 87.0% oracle utility, reducing exploit rate from 64.9% to 1.3%.
- **Experiment**: `experiments/phase10_capstone.py` (Core Experiment C).
- **Seeds**: 10 paired seeds.
- **Metric**: Exploit rate reduced from 64.9% to 1.3% ($\Delta = -63.6\%$, $d = +18.20$).
- **Evidence strength**: Empirically demonstrated for specified structural invariant class.
- **Known limitations**: Requires domain-specific formal invariants that can be verified in linear or polynomial time.
- **Status**: `robustly reproduced`

### CLAIM-14: Reference KL Regularization Mitigates Overconfidence
- **Claim**: Penalizing KL divergence from the supervised reference model prevents entropy collapse and reduces Expected Calibration Error without degrading task accuracy.
- **Evidence**: Process PPO + KL achieves ECE of 0.081 vs 0.171 for Outcome PPO and maintains policy entropy at 1.00 vs 0.63.
- **Experiment**: `experiments/phase10_capstone.py` (Core Experiment E).
- **Seeds**: 10 paired seeds.
- **Metric**: ECE reduction from 0.171 to 0.081 ($\Delta = -0.090$, $d = -6.15$).
- **Evidence strength**: Empirically demonstrated in controlled synthetic setting.
- **Known limitations**: Overly strong KL regularization ($\beta_{\text{kl}} > 0.20$) slows convergence.
- **Status**: `robustly reproduced`

### CLAIM-15: Verifier Disagreement Escalation Improves Audit Efficiency
- **Claim**: Escalating queries to an expensive oracle based on inter-verifier disagreement catches significantly more false-positive exploits per query than random auditing.
- **Evidence**: Under a 20% oracle query budget, disagreement routing caught 69.0% of exploits vs 20.0% for random auditing (3.4x efficiency gain).
- **Experiment**: `experiments/phase10_capstone.py` (Core Experiment F).
- **Seeds**: 10 paired seeds.
- **Metric**: Exploit catch rate: 69.0% vs 20.0% under identical 20% oracle query budget ($d = +7.80$).
- **Evidence strength**: Empirically demonstrated under fixed oracle query budget.
- **Known limitations**: Requires ensemble or diverse verifier architectures with non-correlated error modes.
- **Status**: `robustly reproduced`

### CLAIM-16: Test-Time Search Matches Policy RL Under 8x Token Cost
- **Claim**: Best-of-$N$ inference search on a frozen SFT policy can match the accuracy of single-sample PPO, demonstrating that apparent reasoning gains can be manufactured purely through test-time compute.
- **Evidence**: SFT + Best-of-8 achieves 88.5% oracle accuracy, matching greedy Process PPO (85.5%), but consumes 144 tokens/task vs 18 tokens/task.
- **Experiment**: `experiments/phase10_capstone.py` (Core Experiment D).
- **Seeds**: 10 paired seeds.
- **Metric**: Accuracy parity: SFT BoN-8 (88.5%) vs PPO (85.5%) under an 8x token expenditure ratio.
- **Evidence strength**: Budget-sensitive / inference-compute dependent.
- **Known limitations**: Search efficacy plateaus as $N$ increases due to coverage saturation and verifier noise.
- **Status**: `budget-sensitive`
