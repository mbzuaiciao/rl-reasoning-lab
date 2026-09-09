# Tutorial 05: Reinforcement Learning for Multi-Step Reasoning

In [Tutorial 04](04_language_models_as_policies.md), we established the formal equivalence between autoregressive sequence models and Markov Decision Processes. We proved that generating a token is choosing an action $a_t \sim \pi_\theta(\cdot \mid s_t)$ in an environment whose state transition is deterministic string concatenation.

In this tutorial, we advance from single-step answer prediction to **multi-step reasoning**. We confront the central methodological question of post-training research:

> **When reinforcement learning improves task reward, has the model learned a better reasoning procedure, or merely a better way to produce rewarded outputs?**

To answer this question, we build a parametric symbolic environment, inspect fine-grained process evaluators, compare outcome vs. process supervision, evaluate evaluation-time search (Best-of-$N$), study out-of-distribution depth generalization, detect dataset shortcuts, and analyze counterfactual premise interventions.

---

## 1. What "Reasoning" Means Operationally in this Repository

In colloquial discourse, "reasoning" is often burdened with anthropomorphic connotations of conscious deliberation. In `rl-reasoning-lab`, we maintain strict operational discipline:

> **Operational Definition of Reasoning**:
> A model exhibits a reasoning procedure on a task if it generates an intermediate sequence of deduction steps $z = (z_1, z_2, \dots, z_k)$ such that:
> 1. **Validity**: Each step $z_i$ represents a causally sound, permissible transition given the premises and prior steps.
> 2. **Soundness**: The final answer $y$ logically follows from the premises via the chain $z$.
> 3. **Counterfactual Sensitivity**: If a critical premise in the input is mutated, the downstream steps and final answer change consistently rather than stubbornly repeating a memorized conclusion.

### Observable Trace vs. Internal Latent Computation
A generated reasoning trace (e.g. `R: A->B; B->C; A: C`) is **observable behavioral output**, not direct access to the transformer's internal activations or hidden representations.

> [!WARNING]
> Never say "the model thought step X."
> Always say: **"the model emitted a sequence of tokens consistent with derivation step X."**
> A transformer executes token-by-token forward passes. Whether an intermediate token represents genuine causal computation or post-hoc rationalization is an empirical question that must be tested via causal interventions.

---

## 2. The Three Decoupled Quantities

Throughout this phase, we maintain strict separation across three distinct quantities:

```text
               ┌────────────────────────────────────────────────────────┐
               │                        Reward                          │
               │   (What the RL loss function mathematically optimizes) │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
┌───────────────────────────────────┐           ┌───────────────────────────────────┐
│       Ground-Truth Accuracy       │           │      Process Validity & Soundness │
│  (Does the final token match the  │           │   (Are the intermediate derivation│
│      true mathematical answer?)   │           │   steps causally valid in graph?) │
└───────────────────────────────────┘           └───────────────────────────────────┘
```

A training procedure can dramatically improve one while leaving the others stagnant or corrupted:
- A model can achieve **100% answer accuracy with 0% reasoning validity** by exploiting dataset shortcuts (e.g. memorizing that all depth-2 problems end in `P`).
- A model can achieve **high reward with invalid reasoning** if reward functions do not penalize intermediate hallucinations.
- A model can execute a **flawless, sound derivation but fail on the final token** due to an off-by-one spelling error.

---

## 3. The Synthetic Multi-Step Reasoning Environment

To evaluate reasoning without confounding domain complexities, we implement a parametric directed graph traversal environment ([`GraphTraversalTask`](../src/rl_reasoning_lab/environments/reasoning_tasks.py)):

### Grammar & Task Definition
Given a directed acyclic chain of length $d$:
$$\text{Premises: } v_0 \to v_1, \; v_1 \to v_2, \; \dots, \; v_{d-1} \to v_d$$
Query: $v_0 \to ?$
Target: $v_d$

Compact Token Format:
- Prompt: `A->B; B->C; C->D; Q: A->?`
- Answer-Only Target: ` A: D`
- Trace Target: ` R: A->B; B->C; C->D; A: D`

### Inspectable Properties
1. **Controllable Depth $d$**: Depth parameterizes task difficulty exactly: $d=1$ (single lookup), $d=2$ (1 intermediate hop), $d=5$ (5-step transitive composition).
2. **Distractor Rules**: Irrelevant rules ($M \to N$, $Q \to R$) added to the prompt to test causal selection.
3. **Counterfactual Mutations**: Changing premise $B \to C$ to $B \to X$ produces an exact counterfactual twin to measure causal consistency.

---

## 4. Supervised Baselines: Answer-Only vs. Trace SFT

In [`experiments/phase5_reasoning_baselines.py`](../experiments/phase5_reasoning_baselines.py), we train two supervised baselines on depths $d \in \{1, 2, 3\}$ and test on $d \in \{1, 2, 3, 4\}$:

| Metric | Answer-Only SFT | Trace SFT |
| :--- | :---: | :---: |
| **In-Distribution Acc ($d \le 3$)** | 22.2% | 15.6% |
| **OOD Acc ($d = 4$)** | **0.0%** | **20.0%** |
| **Complete Derivation Path Rate** | 0.0% | 10.0% |
| **Mean Generated Tokens** | 5.5 | 21.0 |
| **Reasoning Efficiency (Corr / 100 Tok)**| **3.01** | **0.79** |

### Key Tradeoffs:
1. **The Extrapolation Advantage of Traces**: When evaluated on depth $d=4$ (held out from training), Answer-Only SFT drops to 0.0% accuracy because direct mapping fails on unseen compositional depths. Trace SFT achieves 20.0% accuracy by breaking the multi-step problem into individual 1-hop transitions.
2. **The Efficiency Tax**: Generating intermediate tokens consumes test-time compute. Answer-only SFT produces more correct answers per 100 tokens on simple in-distribution problems, while trace generation becomes computationally superior as problem depth grows.

---

## 5. Outcome Reward vs. Process Reward in PPO

When applying policy gradients to multi-step reasoning, what should we reward?

### 1. Outcome-Only Reward
$$R_{\text{outcome}} = \begin{cases} +1.0 & \text{if final answer matches ground truth} \\ 0.0 & \text{otherwise} \end{cases}$$
- *Advantage*: Perfectly aligned with final correctness; cannot be gamed by surface formatting.
- *Fatal Flaw*: **Credit Assignment Ambiguity**. If a model generates:
  $$A \to B; \quad B \to X \text{ (hallucinated)}; \quad X \to D; \quad \text{Answer: } D$$
  the outcome reward awards $+1.0$. Generalized Advantage Estimation (GAE) propagates positive advantages back through every token, actively **reinforcing the flawed step $B \to X$**!

In [`experiments/phase5_credit_assignment.py`](../experiments/phase5_credit_assignment.py), we measured the GAE advantage assigned to this corrupted step:
- **Outcome GAE Advantage**: $+0.551$ (Positively reinforced!)
- **Process GAE Advantage**: $-0.333$ (Correctly penalized!)

### 2. Process-Only Reward
$$R_{\text{process}} = \frac{1}{d} \sum_{k=1}^m \mathbf{1}[\text{step } k \text{ is a valid geodesic transition in premise graph}]$$
- *Advantage*: Pinpoints exact deduction errors and provides dense learning signals.
- *Requirement*: Must be **strictly normalized by task depth $d$**. If unnormalized, a model can repeat valid steps 50 times to harvest infinite reward without ever answering the question.

### 3. Hybrid Reward
$$R_{\text{hybrid}} = \alpha R_{\text{process}} + (1 - \alpha) R_{\text{outcome}}$$
Balances intermediate proof validity with goal-directed terminal correctness.

---

## 6. Evaluation-Time Search: Best-of-$N$ Sampling

Does the policy contain correct reasoning paths that greedy decoding fails to express?

In [`experiments/phase5_best_of_n.py`](../experiments/phase5_best_of_n.py), we sample $N$ candidate trajectories from the frozen policy $\pi_\theta$ at temperature $T=0.8$ and evaluate pass@$N$:

```
Pass@N Scaling Curve
  100% │
       │                                                      ● Pass@16: 76.0%
   80% │                                          ● Pass@8: 66.0%
       │
   60% │                              ● Pass@4: 42.0%
       │                  ● Pass@2: 32.0%
   40% │      ● Pass@1: 22.0%
       │  ─── Greedy: 24.0%
   20% │
       └─────────────────────────────────────────────────────────────
         N=1             N=2          N=4         N=8        N=16
```

### Empirical Insight:
Drawing 16 samples increases problem-solving success from **24.0% to 76.0% (+52.0% absolute gain)** without updating a single weight in the network!
This proves that the stochastic policy distribution captures the correct reasoning manifold far more broadly than greedy argmax search exposes.

---

## 7. Rejection Sampling (Filtered SFT) vs. PPO

Before reaching for complex online RL algorithms, we must benchmark against **Rejection Sampling** (also known as Filtered SFT or STaR):

```text
Current Policy π_θ
       │
       ▼
Sample M Trajectories per Prompt
       │
       ▼
Exact Verifier Filter (Keep only correct completions)
       │
       ▼
Supervised SFT Update on Filtered Traces:
  L(θ) = - \sum log π_θ(y_t | x, y_{<t})
```

### Rejection Sampling vs. PPO Comparison:
- **Sample Efficiency**: Rejection sampling is exceptionally stable because it uses standard cross-entropy loss without importance sampling ratios, value critics, or clipping heuristics.
- **Credit Asymmetry**: Rejection sampling only learns from *successes* (discarding failures), whereas PPO utilizes *negative advantages* to actively push the policy away from frequently sampled error modes.

---

## 8. Out-of-Distribution Generalization & Curricula

In [`experiments/phase5_depth_curriculum_and_ood.py`](../experiments/phase5_depth_curriculum_and_ood.py), we train models on depths $d \le 3$ and evaluate on depths $d \in \{4, 5, 6\}$:

1. **The Extrapolation Cliff**: Models achieving $62.2\%$ in-distribution accuracy drop to $2.2\%$ on unseen depths $d \ge 4$.
   - Current transformer policies do not learn an unbounded while-loop; they learn depth-conditioned statistical patterns that degrade when positional embeddings and sequence lengths exceed training distributions.
2. **Catastrophic Forgetting in Sequential Curricula**: Training sequentially ($d=1 \to d=2 \to d=3$) causes the policy to specialize in depth 3 while degrading shallow accuracy on $d=1$. A **fixed interleaved mixture** is required to retain basic skills while scaling depth.

---

## 9. Causal Verification: Distractors, Shortcuts, and Interventions

How do we prove whether a model has learned a genuine reasoning procedure versus a dataset shortcut?

### 1. Distractor Robustness
In [`experiments/phase5_distractors_and_shortcuts.py`](../experiments/phase5_distractors_and_shortcuts.py), adding 4 irrelevant premise edges ($M \to N, Q \to R$) causes accuracy to collapse from $23.3\%$ to $3.3\%$. The model struggles to ignore irrelevant clauses, indicating weak causal dependency tracking.

### 2. Dataset Shortcut Exploitation
We construct a synthetic dataset where depth-2 problems accidentally correlate with target answer `P`:
- **Biased Evaluation (Targets == 'P')**: **100.0% accuracy**!
- **Counterbalanced Evaluation (Targets != 'P')**: **0.0% accuracy** (100% of outputs are still `P`!).
> Headline accuracy on an un-counterbalanced benchmark can easily conceal complete reasoning absence!

### 3. Counterfactual Premise Interventions
If a model solves:
$$\text{Premise: } A \to B, \; B \to C \implies \text{Answer: } C$$
we intervene on step 1:
$$\text{Intervened Premise: } A \to B, \; B \to X \implies \text{Intervened Ground Truth: } X$$
If the model continues to output `C`, it is executing memorized pattern completion. If it outputs `X`, it is dynamically executing the transitive closure.

---

## 10. Reasoning Efficiency Metric

To ensure that reasoning gains are not simply achieved by brute-force verbosity, we define:

$$\text{Efficiency} = \frac{\text{Correct Solutions}}{\text{Total Generated Reasoning Tokens}} \times 100$$

Diagnostic comparisons from our multi-seed benchmark:
- **Answer-Only SFT**: $1.30 \pm 0.26$ correct answers per 100 tokens.
- **Trace SFT**: $0.16 \pm 0.13$ correct answers per 100 tokens.
- **Rejection Sampling**: $0.42 \pm 0.30$ correct answers per 100 tokens.

---

## 11. Active Recall & Self-Assessment

<details>
<summary><b>1. Why is final-answer accuracy insufficient evidence of better reasoning?</b></summary>

Final-answer accuracy measures only the terminal token. A model can achieve high final-answer accuracy through dataset shortcuts, shallow position heuristics, memorized correlations, or lucky guesses following invalid intermediate deductions. Without evaluating step validity and counterfactual consistency, answer accuracy cannot distinguish genuine reasoning from surface mimicry.
</details>

<details>
<summary><b>2. Why might chain-of-thought trace validity differ from answer accuracy?</b></summary>

A model can output a completely sound, valid sequence of reasoning steps but make a transcription or typo error on the final answer token (valid trace, wrong answer). Conversely, a model can hallucinate non-existent premise steps or use flawed logic but stumble into the correct answer by chance or shortcut (invalid trace, correct answer).
</details>

<details>
<summary><b>3. How can Best-of-N sampling improve performance without modifying model weights?</b></summary>

A stochastic policy assigns non-zero probabilities to a distribution of reasoning trajectories. While greedy argmax decoding selects only the mode at each token step (which can commit to an early suboptimal branch), drawing $N$ samples explores alternative derivation paths in the policy's distribution. If an exact verifier filters the outputs, the probability that at least one trajectory is correct scales with $N$.
</details>

<details>
<summary><b>4. How does rejection sampling (filtered SFT) differ from policy-gradient RL?</b></summary>

Rejection sampling samples candidate trajectories, filters for successes using ground truth, and trains the model via standard supervised maximum likelihood (cross-entropy) on the accepted traces. It requires no value network, no importance sampling ratios, and no clipping. Policy-gradient RL (such as PPO) computes advantages on all trajectories—both successful and flawed—and updates parameters to increase probabilities of high-advantage tokens while actively penalizing negative-advantage tokens.
</details>

<details>
<summary><b>5. Why do process rewards improve credit assignment compared to outcome-only rewards?</b></summary>

Outcome rewards provide a single scalar at the end of the sequence. Backpropagating this signal via GAE or Monte Carlo returns distributes positive credit to all tokens in the trajectory, falsely rewarding flawed intermediate steps if the final answer was right. Process rewards evaluate each deduction step locally, allocating positive credit only to valid transitions.
</details>

<details>
<summary><b>6. Why could process rewards introduce new biases or vulnerabilities?</b></summary>

If process rewards are unnormalized, the model can engage in length hacking by repeating valid steps or generating redundant, circular deductions to accumulate reward. Furthermore, if the process verifier is imperfect (a learned PRM), the policy will exploit reward models by generating convincing-sounding but semantically flawed justifications.
</details>

<details>
<summary><b>7. What does depth generalization test that in-distribution evaluation cannot?</b></summary>

In-distribution evaluation tests whether the model can execute compositions within the step counts seen during training. Depth generalization tests whether the policy has acquired a generalizable, recursive reasoning procedure (e.g. repeated transitive rule application) or merely memorized fixed-depth template patterns.
</details>

<details>
<summary><b>8. Why are distractor rules crucial for testing reasoning capabilities?</b></summary>

In simple synthetic tasks, every premise in the prompt is part of the solution path, allowing models to use positional heuristics (e.g. "take the second symbol of the first clause, find where it appears in the second clause"). Distractor rules break this positional coupling, forcing the policy to identify and select causally relevant facts from irrelevant background context.
</details>

<details>
<summary><b>9. Why are counterfactual premise interventions stronger evidence of reasoning than IID test accuracy?</b></summary>

IID test sets share the same statistical distribution as training sets, allowing models to rely on subtle surface correlations. A counterfactual intervention alters an essential premise while holding all superficial formatting constant. If the model's output updates consistently to the new premise, it demonstrates causal reliance on the relation rather than static association.
</details>

<details>
<summary><b>10. Why shouldn't generated reasoning traces be equated with internal cognition?</b></summary>

A generated reasoning trace consists of observable output tokens generated autoregressively by the model. The fact that a model emits tokens saying "because B->C" does not prove that its internal neural representations causally computed the solution through that concept; the model may have arrived at the answer through non-trace activation pathways and merely generated the text trace as a plausible narrative.
</details>

<details>
<summary><b>11. What is the difference between path consistency and complete path?</b></summary>

Path consistency requires that each step's destination matches the subsequent step's source ($v_{k, \text{dst}} == v_{k+1, \text{src}}$) across the generated trace. Complete path additionally requires that the chain starts at the problem's query start node, ends at the problem's target node, every step is a valid edge in the graph, and the declared answer matches the target.
</details>

<details>
<summary><b>12. Why does naive sequential curriculum training often cause catastrophic forgetting?</b></summary>

When training sequentially (e.g. all depth 1, then all depth 2, then all depth 3), gradient steps on the new depth overwrite parameters responsible for shallow tasks. Without interleaved replay or experience rehearsal, the policy specializes in the current distribution and loses accuracy on earlier depths.
</details>

<details>
<summary><b>13. How does depth normalization prevent process reward hacking?</b></summary>

By scaling the process reward by $\frac{1}{d}$ (where $d$ is ground-truth depth) and capping the total at $1.0$, the maximum achievable process reward is bounded. Adding extraneous or repeated steps cannot increase the score beyond $1.0$, disincentivizing verbosity.
</details>

<details>
<summary><b>14. What is the reasoning efficiency metric and why is it needed?</b></summary>

Reasoning efficiency is defined as $\frac{\text{correct answers}}{\text{total reasoning tokens}} \times 100$. It measures the computational cost of reasoning. Without an efficiency metric, an algorithm that generates 500 verbose tokens to get a small accuracy bump would appear strictly superior, despite being prohibitively expensive and computationally inefficient.
</details>

<details>
<summary><b>15. In what scenario can an outcome reward achieve higher efficiency than trace SFT?</b></summary>

On simple, shallow problems where direct pattern recognition reliably predicts the answer, Answer-Only models emit only 2–4 tokens per prompt with high accuracy, whereas Trace SFT models emit 20–40 tokens explaining trivial steps, resulting in substantially lower token efficiency.
</details>

---

## 12. Bridge to Phase 6: Rewards, Verifiers, and Goodhart's Law

In Phase 5, all verifiers were deterministic ground-truth oracles.
In real-world domains (open-ended mathematics, code generation, theorem proving, safety alignment), ground-truth oracles are often unavailable, expensive, or imperfect.

In **Phase 6 (Rewards, Verifiers, and Goodhart's Law)**, we study what happens when verifiers are imperfect:
- **Reward Misspecification**: How policies exploit subtle bugs or edge cases in evaluation functions.
- **Goodhart's Law**: When a measure becomes a target, it ceases to be a good measure.
- **Process Reward Hacking**: How models generate superficially persuasive "math-like" gibberish to fool reward models.
- **Adversarial Evaluation**: Techniques for stress-testing reward functions before using them for RL.
