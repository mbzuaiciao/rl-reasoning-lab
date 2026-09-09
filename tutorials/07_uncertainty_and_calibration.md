# Phase 7 — Uncertainty, Calibration, and Selective Reasoning

> *"For an optimizing system, high confidence does not indicate correctness; it merely indicates low entropy on the decision manifold."*

---

## 1. Executive Overview & Core Concepts

In Phase 6, we discovered that policies optimizing against proxy rewards frequently exploit verifier blind spots. We observed that verifier disagreement often peaks when a policy generates invalid or shortcut reasoning traces.

Phase 7 investigates **uncertainty directly**.

A central failure mode of deep learning and reinforcement learning is **overconfidence**: a model may output token sequences with probability approaching $1.0$ while being completely erroneous. When reinforcement learning sharpens the policy distribution ($\pi_\theta(a|s) \to 1$), it often drastically degrades calibration, turning subtle errors into high-confidence hallucinations.

```text
Prediction Confidence ≠ Policy Entropy ≠ Sampling Variability
≠ Verifier Confidence ≠ Ensemble Disagreement ≠ Epistemic Uncertainty
≠ Aleatoric Task Uncertainty ≠ Calibration ≠ Selective Risk
```

These terms are neither interchangeable nor mathematically equivalent.

---

## 2. Confidence vs. Correctness: The Operational Definition of Calibration

Consider an autoregressive policy producing probabilities over answers:
$$P_\theta(y \mid x)$$

A probability assignment of $0.90$ does **not** mean the model is likely to be correct unless the output distribution is **calibrated**.

### Operational Calibration Definition
A model is said to be calibrated if its assigned confidence matches its empirical probability of ground-truth correctness:
$$P(\text{correct} \mid \text{confidence} = c) = c \quad \forall c \in [0, 1]$$

For example, among all reasoning rollouts where the policy assigns confidence $0.80$, exactly $80\%$ must be mathematically valid and correct.

### Four Dimensions of Probabilistic Quality

Probabilistic predictions must be evaluated across four distinct orthogonal criteria:
1. **Calibration**: The fidelity of confidence values to empirical ground-truth accuracy.
2. **Sharpness**: The concentration of predictions near $0$ or $1$. A model predicting $0.5$ on every example may be calibrated for balanced tasks, but lacks sharpness.
3. **Discrimination (Separation)**: The capacity of the uncertainty metric to rank-order correct instances above incorrect ones (e.g., measured via AUROC or the separation gap $\mathbb{E}[c \mid \text{correct}] - \mathbb{E}[c \mid \text{incorrect}]$).
4. **Entropy**: The information-theoretic spread of the predictive distribution:
   $$H(\pi(\cdot \mid s_t)) = -\sum_{a \in \mathcal{A}} \pi(a \mid s_t) \log \pi(a \mid s_t)$$

---

## 3. Epistemic vs. Aleatoric Uncertainty

Uncertainty in reasoning systems decomposes into two distinct mathematical origins:

```text
               Total Predictive Uncertainty H(P_bar)
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
   Aleatoric Uncertainty                       Epistemic Uncertainty
   E_w[H(P(y|x, w))]                           I(W; Y | X)
   (Intrinsic Task Ambiguity)                  (Parameter & Distribution Shift)
```

### 1. Aleatoric Uncertainty (Data / Task Noise)
Arises from inherent stochasticity, ambiguity, or missing information in the environment.
- *Example*: A prompt contains two equally valid shortest paths to target node $D$ ($A \to B \to D$ vs $A \to C \to D$). The policy cannot know which path the evaluator expects.
- *Property*: Irreducible by gathering more training data from the same distribution.

### 2. Epistemic Uncertainty (Model / Parameter Ignorance)
Arises from a lack of knowledge about the true parameter values or lack of training data in that region of state space.
- *Example*: An out-of-distribution prompt with depth $d=6$ or unseen node tokens.
- *Property*: Reducible with additional training data or supervision in that domain.

> [!CAUTION]
> **Low Entropy Does NOT Imply Low Epistemic Uncertainty.**
> A softmax layer evaluated far outside its training support can produce extreme logits (e.g., $+15.0$), yielding confidence $0.9999$ and near-zero entropy. This is **OOD Overconfidence**, not low epistemic uncertainty.

---

## 4. Policy Uncertainty Metrics

Rather than treating "uncertainty" as a single scalar, `rl-reasoning-lab` implements fine-grained policy-side diagnostics:

| Metric | Formulation | Intuitive Meaning |
| :--- | :--- | :--- |
| **Final-Answer Confidence** | $\pi(a_{\text{ans}} \mid s_{\text{ans}})$ | Probability assigned to the emitted final answer token |
| **Mean Action Entropy** | $\bar{H} = \frac{1}{T}\sum_{t=1}^T H_t$ | Average hesitation across all intermediate reasoning steps |
| **Minimum Token Confidence** | $\min_t \pi(a_t \mid s_t)$ | The "weakest link" in the deductive derivation chain |
| **Length-Normalized Log-P** | $\frac{1}{T}\sum_{t=1}^T \log \pi(a_t \mid s_t)$ | Per-token sequence likelihood under the policy |
| **Action Margin** | $p^{(1)}_t - p^{(2)}_t$ | Gap between top-1 and top-2 candidate tokens at step $t$ |

### Strict Padding Masking Guarantee
When extracting trajectory statistics over batched rollouts, padding tokens must be strictly masked. Appending `<PAD>` tokens to an existing generation must not alter the mean entropy, minimum token confidence, or normalized sequence log probability:
$$\bar{H}(\tau) = \frac{\sum_{t=1}^T (1 - m_t) H_t}{\sum_{t=1}^T (1 - m_t)}, \quad m_t = \mathbb{I}[a_t = \text{PAD}]$$

---

## 5. Mathematical Formulations of Calibration Metrics

### Expected Calibration Error (ECE)
Samples are partitioned into $B$ confidence bins $B_1, \ldots, B_B$ spanning $[0, 1]$. ECE is the weighted average absolute difference between empirical accuracy and mean confidence:
$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
where:
$$\text{acc}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} y_i, \quad \text{conf}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} \hat{p}_i$$

### Brier Score
The mean squared deviation between predicted confidence and binary ground truth:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
The Brier score decomposes strictly into $\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$.

### Negative Log-Likelihood (NLL)
The cross-entropy loss over binary labels:
$$\text{NLL} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

---

## 6. Post-Hoc Calibration: Temperature Scaling

Temperature scaling is a simple yet effective post-hoc calibration technique for classification logits $\mathbf{z}$:
$$P_T(y \mid x) = \operatorname{softmax}\left(\frac{\mathbf{z}}{T}\right)$$

### Optimization Formulation
The scalar temperature $T > 0$ is learned by minimizing NLL on a held-out **calibration set** (disjoint from both training and final test sets):
$$T^* = \arg\min_{T > 0} -\sum_{i=1}^{N_{\text{cal}}} \log \left[ \operatorname{softmax}\left(\frac{\mathbf{z}_i}{T}\right)_{y_i} \right]$$

### Invariant Property
Since $T > 0$ is a strictly monotonic transformation:
$$\arg\max_k \frac{z_k}{T} = \arg\max_k z_k$$
**Temperature scaling changes confidence values and calibration without altering the model's argmax decisions or accuracy.**

---

## 7. Selective Prediction and Risk-Coverage Dynamics

In mission-critical reasoning systems, an agent should abstain when uncertainty exceeds a threshold $\tau$.

### Coverage and Selective Risk
Let $g(x) \in \{0, 1\}$ be the decision to answer ($1$ if confidence $c(x) \ge \tau$, else $0$).
$$\text{Coverage}(\tau) = \mathbb{E}[g(x)] = P(c(x) \ge \tau)$$
$$\text{Selective Risk}(\tau) = P(\hat{y} \neq y \mid g(x) = 1) = \frac{\mathbb{E}[(1 - y) \cdot g(x)]}{\mathbb{E}[g(x)]}$$
$$\text{Selective Accuracy}(\tau) = 1 - \text{Selective Risk}(\tau)$$

### Area Under Risk-Coverage (AURC)
$$\text{AURC} = \int_0^1 \text{Risk}(\text{Cov}) \, d\text{Cov}$$
Lower AURC indicates a superior trade-off: the model discards incorrect answers rapidly as coverage decreases.

---

## 8. Policy vs. Verifier Uncertainty: The 2×2 Decision Surface

Policy uncertainty (generative hesitation) and verifier uncertainty (structural skepticism) provide orthogonal diagnostic signals:

```text
                          VERIFIER UNCERTAINTY
                    Low                          High
         ┌────────────────────────────┬────────────────────────────┐
         │ Q1: Both Confident         │ Q3: Policy Confident,      │
   Low   │                            │     Verifier Skeptical     │
         │ Low error rate;            │ High exploit risk;         │
         │ high-fidelity execution.   │ shortcut or invalid path.  │
POLICY   ├────────────────────────────┼────────────────────────────┤
UNCERTAINTY│ Q2: Policy Hesitant,       │ Q4: Both Skeptical         │
         │     Verifier Confident     │                            │
   High  │ Unfamiliar premise order;  │ Severe failure;            │
         │ sound logic if generated.  │ ungrounded hallucination.  │
         └────────────────────────────┴────────────────────────────┘
```

---

## 9. Active Recall & Self-Assessment

### Question 1: What is calibration?
**Answer**:
Calibration is the property that a model's predicted confidence score $c \in [0, 1]$ corresponds directly to its true empirical probability of correctness: $P(\text{correct} \mid \text{confidence} = c) \approx c$. A model that outputs $0.90$ confidence must be correct $90\%$ of the time on those outputs.

### Question 2: What is the difference between confidence and uncertainty?
**Answer**:
Confidence is the scalar probability assigned to the selected outcome ($\max_a \pi(a \mid s)$). Uncertainty is the broader dispersion, entropy, or epistemic variance of the distribution ($H(\pi)$ or $\operatorname{Var}(\theta)$). A model can be overconfident while in a high-uncertainty state if softmax normalization artificially concentrates probability mass on one invalid token.

### Question 3: Why can low entropy occur on OOD inputs?
**Answer**:
Standard neural networks produce logits via unconstrained linear projections $\mathbf{W}\mathbf{h} + \mathbf{b}$. On OOD inputs, feature vectors $\mathbf{h}$ can have large norms or project far into one coordinate direction, yielding extreme logits ($z_k \gg z_{j}$). The softmax exponential $\frac{e^{z_k}}{\sum e^{z_j}}$ saturates to $1.0$, producing near-zero entropy despite total epistemic absence of training support.

### Question 4: What is selective prediction?
**Answer**:
Selective prediction allows a policy to abstain from answering when its confidence falls below a threshold $\tau$. The system trades **coverage** ($P(\text{answers})$) for reduced **risk** ($P(\text{error} \mid \text{answers})$).

### Question 5: Why is risk-coverage more informative than accuracy alone for abstaining systems?
**Answer**:
Accuracy alone ignores abstention cost. A system that achieves $100\%$ accuracy by answering only $1$ out of $1,000$ questions is practically useless. The risk-coverage curve plots selective risk across all possible coverage levels from $0\%$ to $100\%$, measuring how efficiently the system trades answers for correctness.

### Question 6: What is the difference between sampling disagreement and model-ensemble disagreement?
**Answer**:
- **Sampling Disagreement**: Measures variance across multiple stochastic passes from the *same* model weights ($\tau \sim \pi_\theta$). It reflects temperature and token distribution entropy (aleatoric/search variance).
- **Model-Ensemble Disagreement**: Measures variance across multiple *independently trained* models with different parameter initializations ($\theta_1, \theta_2, \theta_3$). It captures parameter uncertainty (epistemic disagreement). Our experiments show that deep ensembles identify model blind spots that repeated sampling completely misses.

### Question 7: Why can RL improve accuracy but worsen calibration?
**Answer**:
Policy gradient optimization directly maximizes expected reward $\mathbb{E}[R]$. Under PPO, probability ratios $\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$ drive parameters to concentrate almost all probability mass on rewarded tokens. This rapidly reduces entropy to near zero ($H \to 0.001$), forcing confidence to $0.999$ even on tasks where the policy's empirical success rate is only $25\%$.

### Question 8: What does temperature scaling change?
**Answer**:
Temperature scaling divides pre-softmax logits by a positive scalar $T$. When $T > 1$, it softens probabilities and reduces overconfidence; when $T < 1$, it sharpens probabilities. Crucially, because division by a constant is strictly monotonic, $\arg\max_k (z_k / T) = \arg\max_k z_k$, meaning temperature scaling leaves the model's argmax decisions and accuracy completely unchanged.

### Question 9: Why can't confidence itself safely be used as reward?
**Answer**:
Rewarding confidence ($R = R_{\text{task}} + \lambda \cdot \text{conf}$) provides a massive direct gradient incentive to minimize entropy without improving reasoning. The policy quickly learns to output saturated, arbitrary tokens, inducing severe premature entropy collapse and catastrophic hallucination.

### Question 10: What is policy uncertainty versus verifier uncertainty?
**Answer**:
- **Policy Uncertainty**: Intrinsic to the generator (mean token entropy, minimum token confidence, sampling disagreement). Reflects generator hesitation.
- **Verifier Uncertainty**: Intrinsic to the evaluator (disagreement among ensemble judges, heuristic vs. invariant conflicts). Reflects structural or empirical skepticism about the completed derivation.

### Question 11: Why does calibration need to be measured on the current policy distribution?
**Answer**:
Policy optimization induces active distribution shift. A model evaluated on static held-out SFT data may appear well-calibrated ($\text{ECE} = 0.08$), but when evaluated on its own RL-optimized rollouts, confidence collapse inflates calibration error ($\text{ECE} = 0.55$). Evaluating only on static benchmarks gives a false illusion of safety.

### Question 12: How might uncertainty allocate test-time compute?
**Answer**:
By deploying adaptive Best-of-N search: on low-uncertainty prompts, sample $N=1$ (greedy); on moderate uncertainty, sample $N=4$; on high uncertainty, sample $N=12$ or $16$. In our controlled experiment, adaptive compute achieved $50.0\%$ accuracy with only $176$ total generations, nearly matching Fixed Best-of-8 ($55.0\%$ with $320$ generations) with nearly $2\times$ higher compute efficiency.

### Question 13: Why can self-consistency be confidently wrong?
**Answer**:
Self-consistency takes the majority vote over $N$ sampled traces. If the underlying policy has an inductive bias or systematic shortcut (e.g. always picking the node appearing in the last premise), all $N$ samples will generate the same flawed answer. In our experiments, models frequently exhibited $100\%$ sample agreement on incorrect answers.

### Question 14: How can uncertainty help decide when to call an oracle?
**Answer**:
Cheap proxy verifiers are fast but exploitable. By routing rollouts to a ground-truth oracle only when verifier disagreement or policy entropy exceeds a threshold, an engineering system can prevent $90\%+$ of exploits while spending less than $40\%$ of the expensive oracle budget.

### Question 15: Why is epistemic uncertainty particularly important under distribution shift?
**Answer**:
Under distribution shift (e.g., deeper reasoning graphs, novel tokens), standard softmax confidence remains high because the model projects unfamiliar representations onto familiar output classes. Epistemic uncertainty metrics (such as deep ensemble disagreement or mutual information in Bayesian heads) explicitly measure the divergence among plausible parameter hypotheses, exposing when the model is operating outside its domain of competence.

---

## 10. Summary of Architectural and Empirical Guidelines

1. **Never use softmax confidence as a proxy for ground-truth correctness without empirical calibration testing.**
2. **Always evaluate calibration on the current policy rollout distribution ($\text{ECE}_{\pi_k}$), not merely on static SFT corpora.**
3. **Use minimum token confidence or length-normalized log probabilities rather than final-answer confidence to capture derivation integrity.**
4. **Deploy deep ensembles or diverse verifiers rather than repeated sampling when auditing for systematic blind spots.**
5. **Implement selective abstention and adaptive test-time compute to maximize accuracy per generated token.**
