# Tutorial 08: Experimental Science for Reinforcement Learning and Reasoning Systems

> *"A training run is an observation. A research claim requires an experimental design."*

Reinforcement learning (RL) and reasoning systems present unique scientific challenges: non-convex optimization, stochastic rollouts, reward misspecification, exploration bottlenecks, and massive computational costs. Too often in machine learning research, single cherry-picked runs, asymmetric search budgets, or uncalibrated test sets are mistaken for fundamental algorithmic progress.

This tutorial develops the formal methodological, statistical, and empirical foundation required to conduct reproducible experimental science in reinforcement learning and reasoning systems.

---

## 1. The Experimental Unit and Pseudoreplication

In any empirical study, the **experimental unit** is the smallest physical or computational entity to which a treatment is independently applied.

Depending on the research hypothesis, the experimental unit may be:
- An independent training run / random seed ($N_{\text{training}}$).
- A generated test trajectory ($N_{\text{trajectory}}$).
- A task prompt or instance ($N_{\text{prompt}}$).
- A verifier checkpoint or model variant ($N_{\text{verifier}}$).

### The Danger of Pseudoreplication
**Pseudoreplication** occurs when artificial statistical power is manufactured by treating dependent observations as independent experimental replicates.

Consider an experimenter who trains **one** policy with seed 42, generates 10,000 rollouts, observes 8,500 successes, and computes a standard error:
$$SE = \sqrt{\frac{p(1-p)}{N}} = \sqrt{\frac{0.85 \times 0.15}{10,000}} \approx 0.0035$$
The experimenter then claims: *"Our method achieves $85.0\% \pm 0.7\%$ with $p < 0.0001$ over baseline."*

**This is a critical error**:
- The 10,000 trajectories were sampled from the *same* frozen neural network weights $\theta_{42}$.
- If seed 42 happened to be an exceptionally lucky initialization, the true distribution of training outcomes might have mean $70\%$ with standard deviation $15\%$.
- The sample size for evaluating the *algorithm* is $N=1$, not $N=10,000$. The 10,000 trajectories only quantify Monte Carlo evaluation uncertainty of checkpoint $\theta_{42}$, not the reliability of the learning algorithm itself!

---

## 2. Treatment, Control, and Nuisance Variance

In a controlled experiment, we seek to estimate the causal effect of an intervention (the **treatment**) relative to a standard baseline (the **control**):
$$\Delta = \mathbb{E}[Y(\text{Treatment})] - \mathbb{E}[Y(\text{Control})]$$

In RL, observed performance $Y$ is corrupted by multiple sources of **nuisance variance**:
$$Y = f(\text{algorithm}) + \epsilon_{\text{init}} + \epsilon_{\text{rollout}} + \epsilon_{\text{prompt}} + \epsilon_{\text{hardware}}$$
where:
- $\epsilon_{\text{init}}$ is parameter initialization variance.
- $\epsilon_{\text{rollout}}$ is stochastic action exploration noise.
- $\epsilon_{\text{prompt}}$ is task difficulty variance across evaluation problems.
- $\epsilon_{\text{hardware}}$ is floating-point or GPU execution nondeterminism.

### Paired vs Unpaired Comparisons
If we evaluate Treatment $A$ and Control $B$ on **independent** sets of seeds or prompts, the variance of the difference of means is:
$$\text{Var}(\bar{Y}_A - \bar{Y}_B) = \frac{\sigma_A^2}{N_A} + \frac{\sigma_B^2}{N_B}$$

If we instead evaluate both methods on the **same identical prompts, environments, and random initializations** (a **paired design**), the variance of the mean paired difference $\bar{\Delta} = \frac{1}{N}\sum_i (Y_{A,i} - Y_{B,i})$ is:
$$\text{Var}(\bar{\Delta}) = \frac{\sigma_A^2 + \sigma_B^2 - 2\text{Cov}(Y_A, Y_B)}{N}$$
When two methods share common task difficulty noise ($\text{Cov}(Y_A, Y_B) > 0$), the covariance term subtracts nuisance variance directly! In our empirical tests in `tests/test_experimental_science.py`, pairing reduced confidence interval width by over **60%**.

---

## 3. Statistical Tools: Non-Parametric Bootstrap

Standard parametric t-tests assume that observations are normally distributed. In RL, final accuracy distributions across seeds are frequently **bimodal** or **skewed** (e.g., 80% of seeds converge, 20% collapse to 0%). Parametric assumptions fail.

The **Percentile Bootstrap** provides distribution-free confidence intervals without assuming normality:

### The Percentile Bootstrap Algorithm
Given sample $\mathcal{D} = \{x_1, \ldots, x_N\}$ and summary statistic $T(\mathcal{D})$ (e.g., mean, median, trimmed mean, Cohen's $d$):
1. For $b = 1, \ldots, B$ (typically $B=2,000$):
   - Sample $N$ items uniformly with replacement from $\mathcal{D}$ to form bootstrap sample $\mathcal{D}^{*b}$.
   - Compute the replicate statistic $\hat{\theta}^{*b} = T(\mathcal{D}^{*b})$.
2. Sort the $B$ replicates: $\hat{\theta}^{*(1)} \le \hat{\theta}^{*(2)} \le \cdots \le \hat{\theta}^{*(B)}$.
3. For confidence level $1 - \alpha$ (e.g., $\alpha=0.05$ for a 95% CI):
   - $\text{CI}_{\text{lower}} = \hat{\theta}^{*(\lfloor B \cdot \alpha/2 \rfloor)}$
   - $\text{CI}_{\text{upper}} = \hat{\theta}^{*(\lceil B \cdot (1 - \alpha/2) \rceil)}$

### Standardized Effect Size: Cohen's $d$
Statistical significance ($p < 0.05$) only indicates that an effect is unlikely to be pure noise given sample size; it does not measure whether the effect is practically meaningful. We report Cohen's $d$:
$$d = \frac{\bar{x}_A - \bar{x}_B}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}$$
- $|d| \approx 0.2$: Small effect
- $|d| \approx 0.5$: Medium effect
- $|d| \ge 0.8$: Large effect

---

## 4. Hyperparameter Fairness and the Winner's Curse

### Search Budget Selection Bias
An algorithm's reported performance is an increasing function of its tuning budget. If Method A is searched across 30 learning rates while Baseline B is tested on 1 default config:
$$\max_{i \in \{1,\ldots,30\}} X_i > \max_{j \in \{1\}} X_j$$
even when $X_i \sim \mathcal{N}(\mu, \sigma^2)$ are drawn from identical distributions! In `experiments/phase8_hyperparameter_fairness.py`, giving Method A 20 configs and Method B 2 configs manufactured an apparent **+6.55%** advantage out of thin air.

### The Winner's Curse (Regression to the Mean)
When an experimenter selects the single best-performing hyperparameter setting from a noisy validation sweep, the winner is statistically guaranteed to have received positive noise:
$$\mathbb{E}[\text{Validation Winner Score}] > \mathbb{E}[\text{Fresh Test Performance}]$$
In our simulation, validation winners averaging $78.1\%$ regressed to $70.1\%$ on fresh held-out test data (an $8.0\%$ drop).

---

## 5. Multiple Comparisons and False Discoveries

When testing $M$ candidate algorithmic variants against a control:
If each hypothesis is tested at nominal $\alpha = 0.05$ under the null hypothesis (none of the variants actually works), the **Family-Wise Error Rate (FWER)** is:
$$\text{FWER} = 1 - (1 - \alpha)^M \approx 1 - (0.95)^{20} \approx 64.2\%$$
There is a 64% probability of falsely declaring at least one "breakthrough"!

### Corrections:
1. **Bonferroni (FWER Control)**:
   $$\alpha_{\text{corrected}} = \frac{\alpha}{M}$$
   Guarantees $\text{FWER} \le \alpha$, but is highly conservative.
2. **Benjamini-Hochberg (False Discovery Rate Control)**:
   Sorts p-values $p_{(1)} \le p_{(2)} \le \cdots \le p_{(M)}$. Finds the largest $k$ such that:
   $$p_{(k)} \le \frac{k}{M} \cdot \text{FDR}$$
   Rejects all $H_{(i)}$ for $i \le k$. Balances discovery power with statistical protection.

---

## 6. Factorial Ablations and Mechanism Testing

### The Limit of One-at-a-Time Ablations
Standard ablation studies remove one component at a time from a complex system ($S \setminus \{A\}$). This implicitly assumes **additivity**:
$$\text{Performance}(A + B) = \text{Performance}(\text{None}) + \Delta_A + \Delta_B$$

If components interact, one-at-a-time ablations can be completely misleading. In `experiments/phase8_ablation_and_interactions.py`, we evaluated a $2 \times 2$ factorial grid:
- Marginal gain of Process Reward alone: $+13.8\%$
- Marginal gain of KL Regularization alone: $+4.0\%$
- Additive prediction: $+17.8\%$
- **Observed joint gain**: **$+24.4\%$** ($\Delta_{\text{interaction}} = +6.7\%$).

Process reward provided direction while KL prevented policy drift, producing super-additive synergy.

### Mechanism-Oriented Ablations
A scientific ablation tests a *hypothesis*, not merely code deletion:
- *Weak Ablation*: "We removed KL regularization and accuracy fell 18%."
- *Mechanistic Ablation*: "We hypothesized that KL regularization helps by limiting policy drift. Removing KL caused empirical KL divergence to jump from $0.06 \to 0.53$, which subsequently degraded calibration error (ECE increased from $0.07 \to 0.20$), confirming the proposed mechanism."

---

## 7. Distribution Shifts, Worst-Subgroup Auditing, and Simpson's Paradox

### Worst-Subgroup Auditing
Reporting a single aggregate accuracy number can conceal catastrophic subgroup collapse. In `experiments/phase8_distribution_shift_matrix.py`:
- Outcome PPO achieved **88.0%** overall IID accuracy.
- Yet on deeper reasoning ($d=5$) and adversarial verifier traps, its accuracy collapsed to **18.0%** and **12.0%**.
- Scientific reporting requires presenting the full matrix: Mean, Worst Subgroup, and 10th Percentile.

### Simpson's Paradox and Confounding
In observational trace analysis:
- Across all data, trace length appeared *positively* correlated with success ($r = +0.34$), because harder tasks naturally generate longer traces.
- However, stratifying by task depth revealed that within each depth slice, trace length was *negatively* correlated with success ($r = -0.26$), because longer traces within a fixed problem indicated looping or error correction thrashing.
- **Rule**: Never infer causal relationships from observational correlations without conditioning on confounders.

---

## Active Recall Questions

### Question 1: Why is a trajectory not necessarily an independent experimental replicate?
**Answer**:
A trajectory generated by a trained model is a sample from that model's current parameter state $\theta$. If multiple trajectories are drawn from the same model checkpoint, they share the identical training trajectory, initialization seed, and network weights. They are dependent observations. They measure the Monte Carlo evaluation uncertainty of that single checkpoint, not the reliability or variance of the learning algorithm itself across training initializations.

---

### Question 2: Why are multiple training seeds needed?
**Answer**:
RL optimization involves stochastic gradient updates, non-convex loss surfaces, and random rollout exploration. An algorithm may reliably find good policies in 70% of runs, but get trapped in suboptimal shortcuts or experience entropy collapse in 30% of runs. A single seed cannot estimate this variance, and three seeds frequently fail to sample low-probability failure modes. Evaluating $\ge 10$ seeds is necessary to construct meaningful confidence intervals and detect multimodal outcomes.

---

### Question 3: What is best-seed bias?
**Answer**:
Best-seed bias is the distortion introduced when an experimenter runs $n$ random seeds but reports only the single highest-performing run ($\max_{i} Y_i$). In our Phase 8 empirical audit, the best seed scored 95.0% while the 20-seed mean was 76.4% and the worst seed was 13.9%. Reporting only the best seed inflates perceived capability by $+18.6\%$ and completely conceals catastrophic failure modes.

---

### Question 4: Why is paired evaluation often stronger?
**Answer**:
In a paired evaluation design, both treatment and control are tested on the exact same task instances, prompts, and random seeds. When task difficulty introduces substantial common variance ($\text{Cov}(A, B) > 0$), evaluating paired differences $\Delta_i = A_i - B_i$ subtracts this common noise:
$$\text{Var}(\bar{\Delta}) = \frac{\sigma_A^2 + \sigma_B^2 - 2\text{Cov}(A, B)}{N} < \frac{\sigma_A^2 + \sigma_B^2}{N}$$
This dramatically narrows confidence intervals without requiring additional training compute.

---

### Question 5: What is the difference between statistical and practical significance?
**Answer**:
- **Statistical significance** indicates whether an observed difference is unlikely to have arisen purely by chance under the null hypothesis (measured by $p$-values or confidence intervals excluding zero). Given a large enough evaluation set ($N=100,000$), an imperceptible difference of $\Delta = 0.05\%$ can achieve $p < 0.001$.
- **Practical significance** asks whether the magnitude of the effect is large enough to matter in practice (measured by effect sizes like Cohen's $d$ or percentage gains). A difference of $0.05\%$ is statistically significant but practically meaningless.

---

### Question 6: Why must hyperparameter search budgets be matched?
**Answer**:
Because empirical performance scales monotonically with the number of configurations tested. If Method A is searched over 30 learning rates and batch sizes while Baseline B is evaluated on 1 default configuration, Method A has 30 opportunities to capitalize on favorable validation noise. An observed advantage reflects search intensity rather than fundamental algorithmic superiority.

---

### Question 7: How does repeated test-set use create contamination?
**Answer**:
When an experimenter evaluates candidate models on the test set and uses those scores to decide which modifications to keep, the test set enters the optimization loop. The decision-making process acts as an evolutionary search algorithm that selects models fitting the specific random noise of the test split. Over repeated iterations, test performance becomes optimistically biased and loses its ability to measure true generalization.

---

### Question 8: What makes an ablation mechanistic rather than cosmetic?
**Answer**:
A cosmetic ablation merely removes a line of code or module and checks whether overall accuracy drops. A mechanistic ablation formulates a specific hypothesis about *how* and *why* the component functions, and measures intermediate mechanism variables (such as policy entropy, empirical KL divergence, gradient norms, or verifier agreement) to demonstrate that removing the component alters the hypothesized intermediate process.

---

### Question 9: Why can interactions invalidate one-at-a-time reasoning?
**Answer**:
One-at-a-time ablations assume additivity ($\Delta_{A+B} = \Delta_A + \Delta_B$). When two components interact synergistically (e.g., process rewards providing dense directional gradients while KL regularization prevents logits from drifting into extreme overconfidence), removing either component alone causes the entire system to degrade. The individual marginal drops do not sum to the joint effect, obscuring the true architecture dynamics.

---

### Question 10: Why should learning curves be compared under matched x-axes?
**Answer**:
Because algorithms often optimize different trade-offs. One algorithm may be highly sample-efficient in terms of environment rollouts but require expensive replay updates; another may take lightweight, rapid optimizer steps but require many rollouts. If the x-axis is arbitrarily chosen as "optimizer steps", the sample-efficient algorithm appears artificially slow. Comparing under matched environment samples, gradient steps, and wall-clock compute prevents biased conclusions.

---

### Question 11: What is checkpoint-selection bias?
**Answer**:
Checkpoint-selection bias occurs when an experimenter saves checkpoints at every training epoch and reports the highest score observed on the test set ($\max_t \text{Score}_{\text{test}}(t)$). Because test evaluation has variance, selecting the maximum across 50 checkpoints cherry-picks the extreme upper tail of test noise, manufacturing an artificial $+4-8\%$ performance gain that disappears on fresh data.

---

### Question 12: Why can average accuracy hide robustness failures?
**Answer**:
If a benchmark consists of 80% easy/medium tasks and 20% hard/adversarial tasks, a method that achieves 90% on easy tasks but completely collapses to 0% on hard tasks will still record an average accuracy of $72\%$. Reporting only the aggregate mean hides the catastrophic subgroup failure. Auditing the worst subgroup and lowest percentiles is essential for safety-critical reasoning systems.

---

### Question 13: What does a negative control test?
**Answer**:
A negative control introduces an experimental perturbation that should theoretically have zero effect on task outcomes (e.g., adding trailing whitespace, shuffling confidence scores, or injecting an irrelevant random feature into a verifier). If the evaluation pipeline reports a statistically significant effect from a negative control, it exposes bugs, leakage, or spurious correlations in the measurement infrastructure.

---

### Question 14: What makes an experiment reproducible?
**Answer**:
An experiment is reproducible when an independent researcher can reconstruct the identical experimental conditions and obtain statistically consistent results. This requires:
1. Complete machine-readable manifests recording all hyperparameters, code versions, and random seeds.
2. Complete software and hardware environment isolation.
3. Fully seeded pseudo-random number generators (Python, NumPy, PyTorch CPU/GPU).
4. Frozen, versioned evaluation datasets.

---

### Question 15: What evidence would make you downgrade a previous conclusion?
**Answer**:
A previous conclusion should be downgraded when:
1. Increasing the seed count (e.g., from $n=3 \to 12$) causes the 95% confidence interval of the effect to overlap zero.
2. Equalizing hyperparameter search budgets between treatment and baseline eliminates the apparent performance gap.
3. Evaluating on an out-of-distribution benchmark reveals that the method was merely exploiting spurious in-distribution shortcuts.
4. Paired differences reveal that the improvement was driven by a single outlier seed rather than a systematic algorithmic advance.
