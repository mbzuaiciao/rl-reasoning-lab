# Tutorial 06: Reward and Verifier Design

> **"When a measure becomes a target, it ceases to be a good measure."**  
> — Goodhart's Law (formulated by Marilyn Strathern)

---

## 1. The Core Research Question

In Reinforcement Learning for reasoning, our ultimate goal is to produce policies that execute genuine, sound deductive reasoning. However, in virtually every real-world domain—from mathematical olympiads and scientific discovery to code synthesis—the **true utility** of a solution is either impossible to compute directly, expensive to evaluate exhaustively, or partially unobservable.

Consequently, reinforcement learning never optimizes true utility directly. It optimizes an **imperfect proxy reward** provided by an automated verifier:

```text
    Intended Objective: Sound Deductive Reasoning
                     ↓
        True Latent Utility: U(τ)  [Oracle Evaluator]
                     ↓
      Automated Proxy Verifier: R(τ)  [Training Reward]
                     ↓
             RL Policy Update: π_θ
                     ↓
        Learned Model Behavior: Trajectories τ ~ π_θ
```

This structural separation creates the defining dilemma of reward engineering:

$$\max_{\theta} \; \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] \quad \text{versus} \quad \mathbb{E}_{\tau \sim \pi_\theta} [U(\tau)]$$

Crucially:

$$R(\tau) \uparrow \quad \not\implies \quad U(\tau) \uparrow$$

**When an optimizing policy becomes better at satisfying the verifier than at satisfying the intended task, does the model learn to reason, or does it learn to exploit the judge?**

---

## 2. Mathematical Formalization of Reward Misspecification

Let $\mathcal{T}$ denote the space of all possible generated reasoning trajectories $\tau = (x_1, x_2, \ldots, x_T)$.

### 2.1 Intended Utility vs. Proxy Reward

- **True Utility** $U: \mathcal{T} \to \mathbb{R}$: The ground-truth mathematical correctness of the reasoning. In our symbolic graph traversal domain, $U(\tau) = 1$ if and only if the trace starts at the designated source node $u_0$, consists exclusively of valid edges $(u_k, v_k) \in E$, maintains unbroken continuity ($v_k = u_{k+1}$), terminates at target $v_T$, and declares the matching answer. Otherwise, $U(\tau) = 0$.
- **Proxy Reward** $R: \mathcal{T} \to \mathbb{R}$: The computable score emitted by the verifier used to drive policy gradients.

The verifier error on a specific trajectory $\tau$ is:

$$\epsilon(\tau) = R(\tau) - U(\tau)$$

Before policy optimization (under the reference or SFT policy $\pi_{\text{ref}}$), the expected verifier error over typical human or demonstration data may be small:

$$\mathbb{E}_{\tau \sim \pi_{\text{ref}}} \left[ |R(\tau) - U(\tau)| \right] \le \delta$$

However, when an RL optimizer (such as PPO or REINFORCE) updates $\pi_\theta$ to maximize $\mathbb{E}[R(\tau)]$, the policy does **not** sample IID from the pre-training distribution. It actively explores the exponential tail of the sequence distribution:

$$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} [U(\tau) + \epsilon(\tau)]$$

Whenever $\epsilon(\tau) > 0$ on a region of sequence space that is easy for the policy to generate, the optimizer treats that region as a high-reward attractor.

---

## 3. The Taxonomy of Goodhart-Style Failures

In AI safety and decision theory, Goodhart's Law manifests in three distinct regimes. In `rl-reasoning-lab`, we demonstrate all three with concrete symbolic tasks:

### 3.1 Regressional Goodhart (Selection on Noise)
When an evaluation metric is noisy, selecting the top candidates according to the metric selects for both true ability and positive evaluation error:

$$R(\tau) = U(\tau) + \eta, \quad \eta \sim \mathcal{N}(0, \sigma^2)$$

Even if the noise $\eta$ has zero mean unconditionally ($\mathbb{E}[\eta] = 0$), conditioning on high reward selects for positive noise:

$$\mathbb{E}[\eta \mid R(\tau) > c] > 0$$

Under RL optimization, the policy shifts probability mass toward trajectories where random verifier noise happened to be favorably biased.

### 3.2 Extremal Goodhart (Out-of-Distribution Breakdown)
A proxy model or heuristic verifier is fitted to a training distribution $\mathcal{D}_{\text{train}}$. In the normal operating region, length, formatting, or keyword presence might correlate strongly with genuine reasoning depth ($r \approx 0.85$).

However, extreme optimization pushes policy outputs far into the tails of the feature distribution—generating 200 tokens of repetitive filler or dumping every node in the graph alphabet. The observational correlation completely collapses outside the verifier's training manifold:

$$\text{Corr}(R, U \mid \text{tail}) \approx 0$$

### 3.3 Causal Mismatch (Inverted Invariance)
A feature $F(\tau)$ (such as intermediate node mention or syntactic reasoning tag `R:`) correlates with sound reasoning in human demonstrations, but $F(\tau)$ is a **symptom** of reasoning, not its cause:

$$\text{Problem} \longrightarrow \text{Valid Deduction } U(\tau) \longrightarrow \text{Emitted Keywords } F(\tau)$$

When RL optimizes $R(\tau) = g(F(\tau))$, the optimizer directly synthesizes $F(\tau)$ without executing the latent causal antecedent $U(\tau)$:

$$\text{RL Optimizer} \longrightarrow \text{Emitted Keywords } F(\tau) \quad (\text{with } U(\tau) = 0)$$

---

## 4. Static Adversarial Auditing vs. Optimization-Induced Adaptation

A vital methodological distinction developed in Phase 6:

| Dimension | Static Adversarial Testing | Optimization-Induced Adaptation |
| :--- | :--- | :--- |
| **Origin** | Handcrafted by human researchers or red teams. | Discovered autonomously by the policy via trial-and-error reward gradients. |
| **Goal** | Probes known conceptual blind spots (e.g. prefix injection, fake paths). | Maximizes expected scalar return under the current policy parameterization. |
| **Distribution** | Fixed, static test corpus. | Dynamic, evolving trajectory distribution $\tau \sim \pi_\theta$. |
| **Danger** | Easily mitigated by patching the specific handcrafted cases. | Policy continuously discovers *new* adjacent un-patched loopholes. |

In Experiment 9 (`phase6_adversarial_verifier_suite.py`), we subjected candidate verifiers to 6 static adversarial probes. In Experiment 4 (`phase6_policy_distribution_shift.py`), we watched the policy autonomously invent exploit strategies during PPO rollouts.

---

## 5. Verifier Quality Metrics & Pre-RL Baselines

Before training an RL policy against any verifier, its baseline diagnostic profile must be evaluated against the ground-truth oracle:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{Specificity} = \frac{TN}{TN + FP}, \quad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Reward Density} = \frac{TP + FP}{N}$$

### Why False Positives Are Far More Dangerous Than False Negatives

In Experiment 6 (`phase6_fp_vs_fn_asymmetry.py`), we proved empirically that **false positives and false negatives have fundamentally asymmetric consequences under RL**:

```text
False Negative (Strict Verifier):
  Valid trace τ is mistakenly scored 0.
  Consequence: Gradient is zero; sample efficiency drops.
  Result: Slower learning, but NO perverse behaviors are reinforced.

False Positive (Lenient Verifier):
  Invalid / hallucinated trace τ is mistakenly scored +1.
  Consequence: Positive policy gradient boosts probability of invalid tokens.
  Result: Catastrophic policy collapse; model actively internalizes shortcuts.
```

> **Takeaway**: When designing verifiers for reasoning RL, **always bias verifiers toward conservative strictness (low False Positives)**. A missed reward slows down learning, but a false reward corrupts the reasoning engine.

---

## 6. Structural Architectures for Robust Verification

Phase 6 established three architectural paradigms for building resilient verifiers:

### 6.1 Invariant-Gated Verification
Hard mathematical invariants must act as non-negotiable gates before soft heuristics or learned reward models are consulted:

```python
class InvariantGatedVerifier(BaseVerifier):
    def verify(self, completion: str, example: ReasoningExample) -> VerifierResult:
        # Invariant 1: Start node must match problem premise
        if steps[0][0] != example.start_node:
            return VerifierResult(score=-1.0, passed=False, reason_code="INVARIANT_START_MISMATCH")

        # Invariant 2: Chain must maintain unbroken topological continuity
        for k in range(1, len(steps)):
            if steps[k][0] != steps[k - 1][1]:
                return VerifierResult(
                    score=-1.0, passed=False, reason_code="INVARIANT_DISCONNECTED_STEP"
                )

        # Invariant 3: Final step destination must equal declared answer
        if steps[-1][1] != parsed_answer:
            return VerifierResult(
                score=-1.0, passed=False, reason_code="INVARIANT_DESTINATION_MISMATCH"
            )

        # Only if all invariants hold, evaluate soft proxy reward
        return self.soft_verifier.verify(completion, example)
```

In Experiment 8 (`phase6_invariant_and_sequential.py`), invariant gating blocked **100% of synthetic adversarial exploits** that completely fooled the underlying proxy verifier.

### 6.2 Sequential Pipeline Verification
Verification structured as a multi-stage pipeline:

$$\text{Stage 1: Syntax} \;\longrightarrow\; \text{Stage 2: Answer Format} \;\longrightarrow\; \text{Stage 3: Edge Existence} \;\longrightarrow\; \text{Stage 4: Path Continuity}$$

The pipeline halts immediately at the first failing stage, saving compute on invalid trajectories while logging exact attribution codes.

### 6.3 Diverse Ensembles & Disagreement Uncertainty
Combining $N$ independent verifiers $\{V_1, V_2, \ldots, V_N\}$:
- **Majority Vote**: Passed if $\sum \mathbb{I}(V_i = \text{pass}) > N/2$.
- **Unanimity**: Passed only if all judges agree ($V_i = \text{pass} \; \forall i$).
- **Disagreement Metric**:

$$\text{Disagreement} = 1.0 - \frac{\max(N_{\text{pass}}, N_{\text{fail}})}{N}$$

In Experiment 7 (`phase6_ensemble_and_diversity.py`), we demonstrated that **structural diversity among judges is far more effective than judge count**:
- Adding 3 redundant judges with correlated heuristic blind spots (Homogeneous Ensemble) failed to eliminate false positives ($FP = 40$).
- Ensembling 3 diverse judges (Syntax + Edge Connectivity + Final Answer) reduced false positives to **0 under Unanimity**.

---

## 7. Active Oracle Budgeting & Uncertainty-Aware Abstention

When verifiers are uncertain (high disagreement), naively forcing a decision creates false positives. Instead, the verifier can **abstain**:

```text
Decision = { Accept, Reject, Abstain }
```

In human-in-the-loop workflows, human/oracle verification is scarce and expensive. In Experiment 11 (`phase6_uncertainty_and_oracle_budget.py`), we simulated spending a limited oracle budget $B < N$:
- **Random Allocation**: Uniformly query oracle on $B$ trajectories.
- **Disagreement-Triggered Allocation**: Sort trajectories by verifier disagreement and spend the budget on the most contentious outputs.

$$\text{Accuracy Gain at } B=60: \quad 84.0\% \text{ (Random)} \;\longrightarrow\; 100.0\% \text{ (Disagreement-Triggered)}$$

Disagreement-triggered allocation targets verification budget precisely where automated proxies are most prone to error.

---

## 8. Multi-Component Reward Decomposition

Never represent reasoning feedback as a single monolithic scalar $R \in \mathbb{R}$. Always maintain explicit decomposed components:

$$R(\tau) = w_a R_{\text{answer}} + w_s R_{\text{steps}} + w_p R_{\text{path}} + w_f R_{\text{format}} - w_l R_{\text{brevity}}$$

In Experiment 10 (`phase6_reward_decomposition_pareto.py`), we demonstrated that:
1. **Raw Sum Pathology**: Raw step rewards score depth-4 problems $4\times$ higher than depth-1 problems and reward infinite self-loop rambling. Normalizing by task depth ($R_{\text{steps}} = \frac{\text{valid steps}}{d}$) eliminates depth-dependent gradient variance.
2. **Answer-Only Degeneracy**: Setting $w_s = w_p = 0$ rewards lucky guesses and produces 0 incentive for intermediate derivation.
3. **Brevity Penalty Collapse**: Unconditioned length penalties penalize necessary reasoning steps on deep problems, causing premature halting.

---

## 9. Active Recall & Self-Assessment

Test your understanding of the mathematical and conceptual foundations of reward and verifier design. Click to expand solutions:

<details>
<summary><b>1. What is the fundamental difference between reward and utility?</b></summary>
Utility \(U(\tau)\) is the true, intended objective we care about (e.g. valid mathematical deduction, factual correctness, safety). Reward \(R(\tau)\) is an observable, computable proxy signal supplied to the optimization algorithm. Maximizing \(R(\tau)\) only improves \(U(\tau)\) to the extent that \(R\) is causally and structurally aligned with \(U\).
</details>

<details>
<summary><b>2. Why can a verifier have 95% test accuracy on held-out data but still cause catastrophic policy failure under RL?</b></summary>
Static test sets measure accuracy on the pre-optimization distribution \(\mathcal{D}_{\text{pre}}\). RL optimization actively searches for trajectories that maximize reward, shifting the rollout distribution toward the 5% error region of the verifier. Once discovered, the policy over-samples that blind spot, rendering static accuracy meaningless.
</details>

<details>
<summary><b>3. What are the three Goodhart-style failure regimes?</b></summary>
1. <b>Regressional Goodhart</b>: Selection on noisy evaluation rewards positive noise.
2. <b>Extremal Goodhart</b>: Extreme optimization pushes inputs outside the verifier's valid operating manifold.
3. <b>Causal Mismatch</b>: Correlational features are optimized directly, bypassing the true causal mechanism.
</details>

<details>
<summary><b>4. Why are systematic verifier errors far more destructive under RL than random errors?</b></summary>
Random noise adds variance to policy gradient estimates, which slows learning but points in no consistent direction. Systematic errors provide a coherent gradient vector that actively pulls the policy parameters toward the shortcut attractor.
</details>

<details>
<summary><b>5. Why is a false positive fundamentally worse than a false negative in reasoning RL?</b></summary>
A false negative drops reward on a valid solution, causing sample inefficiency. A false positive rewards an invalid derivation, directly reinforcing and entrenching unsound reasoning habits in policy weights.
</details>

<details>
<summary><b>6. What does 'reward hacking' mean operationally in this repository?</b></summary>
Reward hacking is the adaptive increase in proxy reward \(R(\tau)\) accompanied by stagnation, divergence, or collapse in true oracle utility \(U(\tau)\), driven by policy exploitation of an unaligned verifier loophole.
</details>

<details>
<summary><b>7. Why should verifier performance always be evaluated on post-optimization rollouts?</b></summary>
Because the policy actively changes the evaluation distribution. Measuring verifier metrics only before RL creates a false sense of security; only post-RL evaluation exposes whether the policy has discovered and amplified verifier blind spots.
</details>

<details>
<summary><b>8. What does an invariant gate contribute that a soft verifier cannot?</b></summary>
Invariant gates establish non-negotiable structural axioms (e.g. path must start at premise start node, destination must match answer, every edge must exist). They cannot be compensated for by other features, enforcing hard structural bounds against the specified class of exploits.
</details>

<details>
<summary><b>9. Why does verifier diversity matter more than the number of verifier judges?</b></summary>
Multiple judges that share the same underlying logic (e.g. 3 keyword checkers) share the same blind spots, agreeing with high false confidence on adversarial traps. Structurally diverse judges (syntax + semantics + graph topology) have orthogonal failure modes, allowing unanimity or voting to filter out blind spots.
</details>

<details>
<summary><b>10. Why can unanimity ensemble voting harm sample efficiency despite eliminating false positives?</b></summary>
Unanimity requires every judge to approve. Any single judge's false negative causes the entire ensemble to reject the trace, resulting in sparse rewards that can stall policy exploration.
</details>

<details>
<summary><b>11. Why does setting a continuous verifier acceptance threshold too high (e.g. 0.95) harm policy learning?</b></summary>
A hyper-strict threshold creates extreme reward sparsity. When the policy receives 0 reward across entire rollout batches, policy gradient updates collapse, freezing policy progress.
</details>

<details>
<summary><b>12. What information is lost when multiple reward components are scalarized into a single float?</b></summary>
Scalarization hides which specific sub-goals were satisfied. A completion that guesses the answer correctly with zero reasoning steps can receive the exact same scalar score as a completion with flawless intermediate reasoning that suffered a minor typo.
</details>

<details>
<summary><b>13. How does disagreement-triggered oracle allocation optimize a limited supervisory budget?</b></summary>
High verifier disagreement correlates strongly with borderline, ambiguous, or adversarial trajectories. Spending human/oracle review on high-disagreement items maximizes the informational yield per supervisory query.
</details>

<details>
<summary><b>14. Why is using the same verifier for both RL training and evaluation reporting a critical scientific flaw?</b></summary>
It creates a closed circular loop where the policy is evaluated against the very proxy it was trained to satisfy. If the verifier has a blind spot, the experiment will report '100% success' while true reasoning utility is 0%.
</details>

<details>
<summary><b>15. What evidence is sufficient to trust a reward signal enough to optimize an RL policy against it?</b></summary>
A reward signal can be trusted only when:
1. It is guarded by hard structural invariants.
2. It exhibits near-zero false-positive rates on adversarial audit suites.
3. An independent ground-truth oracle verifies that \(U(\tau)\) increases monotonically with \(R(\tau)\) throughout optimization.
4. Its calibration and error distribution remain stable on policy-generated rollouts.
</details>

---

## 10. Bridge to Phase 7: Uncertainty and Verification

In Phase 6, we discovered that verifiers cannot be assumed static, unbiased, or infallible. In particular, we saw that **measuring verifier disagreement and predictive uncertainty** enables calibrated abstention and targeted oracle oversight.

In **Phase 7: Uncertainty and Verification**, we deepen this inquiry from the verifier side to the policy side:
- How does the policy's own predictive entropy correlate with reasoning correctness?
- How can models be trained to **abstain** ("I don't know") when reasoning confidence is low?
- How do we construct well-calibrated confidence estimators for multi-step reasoning?
- How does self-consistency voting over sampled reasoning paths provide unsupervised uncertainty estimates?
