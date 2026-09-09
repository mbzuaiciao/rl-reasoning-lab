# Phase 10 Research Plan: When Does RL Produce Genuine Reasoning Improvement?

> *"When reinforcement learning improves reasoning-task performance, what evidence distinguishes genuine improvement in the learned reasoning procedure from reward exploitation, shortcut learning, increased test-time search, or confidence collapse?"*

---

## 1. Primary Research Question

The core research question of this study is:

> **When reinforcement learning improves performance on controlled multi-step reasoning tasks, under what conditions does the improvement transfer to deeper, counterfactual, distractor-heavy, and verifier-independent evaluations?**

In contemporary language-model reinforcement learning, training performance is frequently measured by terminal answer accuracy on benchmark evaluation suites. However, answer accuracy is an aggregate metric that can improve for reasons unrelated to reasoning validity:
- Exploiting brittle verifier heuristics (format shortcuts, length biases, keyword cues).
- Memorizing spurious dataset correlations (e.g., specific target token distributions).
- Increasing stochastic sample diversity without improving the underlying policy distribution.
- Scaling test-time search (Best-of-$N$) that masks poor policy generation.
- Degraded epistemic calibration and overconfident generation.

This capstone investigates whether, when, and how reinforcement learning produces **genuine improvement in the learned reasoning procedure** versus superficial behavior that satisfies an optimizing reward.

---

## 2. Operational Definition of Genuine Reasoning Improvement

For the controlled experimental environment of `rl-reasoning-lab`, we define **genuine reasoning improvement** as performance gains that survive a battery of seven independent, pre-registered criteria:

1. **Higher Oracle-Grounded Answer Accuracy**: The model must solve reasoning tasks verified by an independent, un-gamed oracle rather than merely satisfying a proxy verifier.
2. **Higher Complete-Path / Step Validity**: Intermediate reasoning steps must form a valid, continuous, topologically sound chain from premises to conclusion ($\text{CompletePathRate} \to 1.0$).
3. **Out-of-Distribution (OOD) Depth Generalization**: Reasoning competence trained on shallow depths ($d \le 3$) must generalize to deeper, unseen chain lengths ($d \in \{4, 5, 6\}$) rather than collapsing into trivial loops.
4. **Counterfactual Consistency**: When a single premise edge is mutated ($A \to B$ altered to $A \to X$), the policy's prediction and chain must causally update to reflect the altered premise:
   $$P(\text{prediction changes correctly} \mid \text{single-edge mutation}) \approx 1.0$$
5. **Distractor Robustness**: The policy must remain robust to irrelevant or dead-end premise edges that do not lie on the causal path.
6. **Non-Increasing Exploit Rate**: The policy must not discover and exploit known verifier vulnerabilities (such as format mimicry or superficial substring heuristics) over the course of optimization.
7. **Independent Oracle Insulation**: The evaluation must be conducted using an oracle with full ground-truth graph knowledge that is strictly insulated from the training loop.

> **Methodological Boundary**: We do not claim this operational definition captures all facets of "human reasoning" or frontier foundation models. We call it **the operational criterion used in this controlled environment**.

---

## 3. Predefined Metric Taxonomy

To prevent optimization proxies from masquerading as scientific progress, all metrics are strictly partitioned into three tiers:

### Tier 1: Primary Outcomes
- **Oracle Answer Accuracy ($U_{\text{oracle}}$)**: Fraction of tasks where the final emitted answer matches ground truth.
- **Complete-Path Rate**: Fraction of completions where every intermediate step $u \to v$ is valid, contiguous, and terminates at the correct destination.
- **OOD Depth Generalization**: Accuracy and complete-path validity on depths $d \in \{4, 5, 6\}$ when trained only on $d \in \{1, 2, 3\}$.
- **Counterfactual Consistency**: Probability that changing a premise edge changes the prediction and trace to the exact new ground truth.

### Tier 2: Secondary Outcomes
- **Step Validity**: Proportion of individual emitted transitions that exist in the true premise graph.
- **Distractor Robustness**: Accuracy retention when 2–4 irrelevant distractor rules are inserted.
- **Exploit Rate**: Frequency of trajectories that pass a weak verifier despite failing ground-truth oracle evaluation.
- **Tokens Per Correct Answer**: Mean generated tokens divided by oracle accuracy (reasoning efficiency).
- **Policy KL ($\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$)**: Drift from the supervised initialization policy.
- **Policy Entropy ($\mathcal{H}(\pi)$)**: Shannon entropy over the action token vocabulary.
- **Calibration (Expected Calibration Error - ECE, Brier Score)**: Alignment between token confidence and empirical correctness.
- **Verifier Disagreement Rate**: Inter-verifier variance across diverse checking strategies.

### Tier 3: Optimization Metrics
- **Proxy Reward ($R_{\text{training}}$)**: The scalar training objective maximized by the RL algorithm.
- **PPO Clipped Surrogate Loss**: $\mathcal{L}_{\text{CLIP}}(\theta)$.
- **Value Function Loss**: $\mathcal{L}_{V}(\phi)$.
- **PPO Clip Fraction**: Proportion of token probability ratios clipped by $\epsilon$.

> [!IMPORTANT]
> Optimization metrics are diagnostic indicators of gradient descent mechanics; they are **never** permitted to substitute for primary evaluation outcomes.

---

## 4. Predefined Evaluation Distributions

The evaluation test suite is partitioned into seven distinct splits, each generated from independent, fixed random seeds:

| Distribution Split | Depth ($d$) | Distractors | Specific Manipulation | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **1. IID Test Set** | $1 \le d \le 3$ | 0 | Same distribution as training | Baseline in-distribution benchmark. |
| **2. OOD Depth Split** | $4 \le d \le 6$ | 0 | Unseen chain lengths | Extrapolation and path continuity test. |
| **3. Dense Distractors** | $1 \le d \le 3$ | $k=3$ | Disjoint irrelevant premise rules | Tests attention selectivity & distraction. |
| **4. Counterfactual Split**| $2 \le d \le 4$ | 0 | Single premise edge mutated | Tests causal sensitivity to premises. |
| **5. New Symbol Split** | $1 \le d \le 3$ | 0 | Disjoint symbol vocabulary ($\{L, M, N, \dots\}$) | Tests abstract relation binding. |
| **6. Shortcut-Counterbalanced**| $d=3$ | 0 | Balanced answers ($P(\text{target} = \text{'Z'}) = 1/K$) | Tests reliance on dataset shortcuts. |
| **7. Adversarial Verifier Traps**| $1 \le d \le 3$ | 0 | Truncated, hallucinated, or keyword-stuffed traces | Tests verifier vulnerability & exploit rate. |

---

## 5. Standardized Evidence Language

To maintain scientific conservatism, all findings in Phase 10 are reported using four standardized epistemic categories:

1. **"demonstrates in this environment"**: Reserved for strong, statistically significant effects ($p < 0.01$, tight 95% bootstrap CI excluding zero, Cohen's $|d| \ge 0.8$) that replicate across fresh seeds and survive OOD stress testing.
2. **"supports"**: Applied when consistent positive evidence is observed across multiple seeds and metrics, but effect sizes are moderate or partially attenuated under OOD shift.
3. **"suggests"**: Used when qualitative trends exist or differences are directionally consistent, but confidence intervals overlap or variance across seeds is substantial.
4. **"inconclusive"**: Assigned when bootstrap confidence intervals span zero, when outcomes are highly sensitive to seed initialization, or when replication tests fail to reproduce the initial finding.

*The word "proves" is strictly prohibited for empirical observations of learning systems.*
