# Phase 10 Reasoning Failure Taxonomy

> *"To diagnose whether an RL model has learned to reason, we must catalog not only how it succeeds, but the precise structural taxonomy of how it fails."*

This document provides a formal 10-category taxonomy of reasoning failures observed in multi-step language-model policies within `rl-reasoning-lab`.

---

## The 10 Failure Categories

### 1. Shortcut Exploitation
- **Definition**: The model outputs the correct final answer by exploiting a spurious correlation or positional heuristic in the prompt or dataset without executing the necessary intermediate causal deductions.
- **Representative Example**:
  - Prompt: `A->B; B->C; C->Z; Q: A->?` (where 'Z' is over-represented as target in training).
  - Model Output: `R: A->Z; A: Z` (skips intermediate nodes `B` and `C` directly to `Z`).
- **Detection Metric**: Ground-truth Answer Accuracy is high, but **Complete-Path Rate is low** ($\text{CompletePath} = 0$). Drops sharply on shortcut-counterbalanced evaluation splits.
- **RL Impact**: **Increased by Outcome-only RL**; suppressed by Process-reward RL and invariant gating.

### 2. Invalid Intermediate Transition (Hallucination)
- **Definition**: The reasoning trace introduces an edge $u \to v$ that does not exist anywhere in the prompt's premise graph.
- **Representative Example**:
  - Prompt: `A->B; B->C; C->D; Q: A->?`
  - Model Output: `R: A->B; B->K; K->D; A: D` (edge `B->K` is fabricated).
- **Detection Metric**: `StepValidity < 1.0` and `PathConsistency == False` via `TraceEvaluator`.
- **RL Impact**: Outcome-only RL often preserves or increases hallucinations if the final answer happens to match; Process-reward RL heavily penalizes this failure mode.

### 3. Premature Termination
- **Definition**: The policy halts reasoning before reaching the target node, emitting an answer token prematurely.
- **Representative Example**:
  - Prompt: `A->B; B->C; C->D; D->E; Q: A->?` (Depth 4).
  - Model Output: `R: A->B; B->C; A: C` (emits answer after 2 steps instead of traversing to `E`).
- **Detection Metric**: `ReasoningLength < GroundTruthDepth` with `AnswerCorrect == False`.
- **RL Impact**: Commonly triggered when applying negative step penalties that inadvertently encourage the agent to terminate early to avoid penalty accumulation.

### 4. Distractor Capture (Attention Misdirection)
- **Definition**: The reasoning path branches into an irrelevant or disconnected distractor premise edge that does not lead to the query's target node.
- **Representative Example**:
  - Prompt: `A->B; B->C; C->D; X->Y; M->N; Q: A->?` (Distractors: `X->Y`, `M->N`).
  - Model Output: `R: A->B; X->Y; Y->D; A: D` (diverts into distractor rule `X->Y`).
- **Detection Metric**: `DistractorRobustnessRatio = Accuracy(Distractors) / Accuracy(Clean) < 0.70`.
- **RL Impact**: SFT baselines are moderately vulnerable; Process RL with distractor curriculum suppresses capture.

### 5. Out-of-Distribution (OOD) Extrapolation Failure
- **Definition**: A policy trained on short reasoning depths ($d \le 3$) completely breaks down when evaluated on longer chain lengths ($d \in \{4, 5, 6\}$), degenerating into repetitive loops or incoherent transitions.
- **Representative Example**:
  - Prompt: `A->B; B->C; C->D; D->E; E->F; Q: A->?` (Depth 5).
  - Model Output: `R: A->B; B->C; C->D; D->C; C->D; D->C...` (infinite 2-cycle loop).
- **Detection Metric**: Accuracy on $d \ge 4$ collapses to $<20\%$ while IID accuracy ($d \le 3$) remains $>85\%$.
- **RL Impact**: Outcome PPO exhibits severe extrapolation collapse ($18\%$ OOD acc); Process PPO maintains superior continuity ($68\%$ OOD acc).

### 6. Verifier False-Positive Exploitation
- **Definition**: The model produces a completion specifically formatted to trigger a positive reward from a flawed proxy verifier despite being logically unsound or incomplete.
- **Representative Example**:
  - Weak Verifier: Checks only if target letter appears in text.
  - Model Output: `R: corrupted nonsense string containing D; A: D` -> Verifier returns `+1.0`.
- **Detection Metric**: Discrepancy $\Delta = R_{\text{proxy}} - U_{\text{oracle}} > 0.50$; flagged by Invariant Gating.
- **RL Impact**: **Severely amplified by RL optimization** against un-gated proxy verifiers (Goodhart's Law).

### 7. Overconfident Wrong Answer
- **Definition**: The model emits an incorrect final answer or hallucinated trace with near-zero token entropy and maximum probability ($P(a_t) > 0.99$).
- **Representative Example**:
  - Prompt: `A->B; B->C; C->D; Q: A->?`
  - Model Output: `R: A->B; B->X; X->Z; A: Z` with mean token confidence $99.4\%$.
- **Detection Metric**: High Brier score, elevated Expected Calibration Error ($\text{ECE} > 0.15$), and failure of selective risk curves at high confidence thresholds.
- **RL Impact**: Standard PPO consistently increases overconfidence; Reference KL regularization ($\beta_{\text{kl}} > 0$) mitigates this failure.

### 8. Search / Sampling Failure
- **Definition**: The policy distribution contains the correct reasoning path with non-zero probability, but single-sample greedy decoding fails to locate it.
- **Representative Example**:
  - Greedy output ($T=0$): Fails task.
  - Best-of-4 sampling ($T=1.0$): 3 of 4 samples find valid paths.
- **Detection Metric**: $\text{Accuracy}(\text{Best-of-N}) - \text{Accuracy}(\text{Greedy}) > 0.25$.
- **RL Impact**: Demonstrates that inference-time search budget can mask weaknesses in the underlying parameter policy.

### 9. Training Instability / Entropy Collapse
- **Definition**: During RL training, policy entropy suddenly collapses to near-zero before discovering the multi-step structure, locking the model into emitting repetitive single tokens.
- **Representative Example**:
  - Step 20: Mean entropy drops from $2.1 \to 0.02$; all prompt outputs become `AAAAAAAA`.
- **Detection Metric**: Step-by-step health monitor detecting $\mathcal{H}(\pi) < 0.05$ or sudden KL spike $>5\times$ median.
- **RL Impact**: Triggered by high learning rates, insufficient entropy bonus, or unclipped policy ratios.

### 10. Engineering / Pipeline Fault
- **Definition**: Failures caused by software infrastructure bugs rather than model capacity (e.g., advantage broadcasting bugs, missing RNG seeds, un-detached tensors, unfrozen reference models).
- **Representative Example**:
  - Squeezing advantage from `[B, 1]` to `[B]`, broadcasting uniformly across $T$ tokens and assigning full credit to prompt tokens.
- **Detection Metric**: Phase 9 Invariant assertions (`assert adv.shape == log_probs.shape`, `HealthMonitor`).
- **RL Impact**: Prevented via Phase 9 research engineering safeguards.

---

## Summary Matrix

| Failure ID | Name | Primary Metric | Outcome RL Effect | Process RL Effect | Mitigation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F1** | Shortcut Exploitation | Complete Path Rate | Amplifies | Suppresses | Counterbalanced training, process reward |
| **F2** | Invalid Transition | Step Validity | Neutral / Amplifies | Heavily Suppresses | Token-level step verification |
| **F3** | Premature Termination | Reasoning Length vs Depth | May induce | Neutral | Minimum length bounds, path gating |
| **F4** | Distractor Capture | Distractor Accuracy Ratio | Vulnerable | Robust | Distractor curriculum, hard negative rules |
| **F5** | OOD Extrapolation | Accuracy on $d \ge 4$ | Severe Collapse | Moderate Resilience | Depth curriculum, invariant gating |
| **F6** | Verifier Exploitation | Proxy vs Oracle Gap | Severely Amplifies | Neutral | Invariant gates, diverse ensembles |
| **F7** | Overconfident Error | ECE & Selective Risk | Amplifies | Amplifies | Reference KL, temperature scaling |
| **F8** | Search Failure | BoN vs Greedy Delta | N/A | N/A | Test-time search, rejection sampling |
| **F9** | Entropy Collapse | Mean Action Entropy | Risk factor | Risk factor | Entropy bonus, gradient clipping |
| **F10**| Engineering Fault | Boundary Invariants | System Crash | System Crash | Fail-fast assertions, atomic checkpointing |
