# Tutorial 03: Proximal Policy Optimization (PPO) from First Principles

In [Tutorial 02](02_policy_gradients.md), we derived REINFORCE and Actor-Critic methods. Both algorithms share a critical bottleneck: **they are strictly on-policy**.

Once an optimizer takes even a single gradient step on a batch of trajectories, the policy parameters change from $\theta_{\text{old}}$ to $\theta_{\text{new}}$. The collected rollout data is no longer distributed according to the updated policy. In standard REINFORCE, taking multiple gradient steps on the same batch is mathematically invalid and causes catastrophic divergence. Throwing away data after a single gradient step, however, is prohibitively expensive in terms of sample efficiency.

**Proximal Policy Optimization (PPO)** solves this dilemma. It uses **importance sampling** and a **clipped surrogate objective** to safely squeeze multiple optimization epochs out of each collected batch of experience without causing policy collapse.

---

## 1. The Distribution Mismatch Problem & Importance Sampling

### The Mismatch
Suppose we collect a trajectory rollout $\tau \sim \pi_{\theta_{\text{old}}}$.
After taking a gradient update, our policy parameters become $\theta$.
The expected return under our current policy is:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int p_\theta(\tau) R(\tau) \, d\tau$$
However, our available empirical samples were drawn from $p_{\theta_{\text{old}}}(\tau)$, not $p_\theta(\tau)$!

### The Importance Sampling Identity
For any probability distributions $p$ and $q$ where $q(x) > 0$ whenever $p(x) > 0$:
$$\mathbb{E}_{x \sim p}[f(x)] = \int p(x) f(x) \, dx = \int q(x) \frac{p(x)}{q(x)} f(x) \, dx = \mathbb{E}_{x \sim q} \left[ \frac{p(x)}{q(x)} f(x) \right]$$

Applying this change of measure to our policy gradient objective:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \frac{p_\theta(\tau)}{p_{\theta_{\text{old}}}(\tau)} \nabla_\theta \log p_\theta(\tau) R(\tau) \right]$$

Expanding the trajectory probability ratio $\frac{p_\theta(\tau)}{p_{\theta_{\text{old}}}(\tau)}$:
$$\frac{p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)}{p(s_0) \prod_{t=0}^{T-1} \pi_{\theta_{\text{old}}}(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)} = \prod_{t=0}^{T-1} \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$
Notice once again that the environment dynamics cancel completely!

In practical policy optimization, we define the **per-token / per-step probability ratio**:
$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

### Intuition for the Ratio $r_t(\theta)$:
- $r_t(\theta) = 1.0$: The current policy $\pi_\theta$ assigns the exact same probability to action $a_t$ as the behavior policy that collected the data.
- $r_t(\theta) > 1.0$: The current policy has made action $a_t$ **more likely** than it was during rollout.
- $r_t(\theta) < 1.0$: The current policy has made action $a_t$ **less likely**.

#### Tiny Numerical Example:
Suppose during rollout, $\pi_{\theta_{\text{old}}}(\text{RIGHT} \mid s_0) = 0.20$.
After an optimizer step, $\pi_\theta(\text{RIGHT} \mid s_0) = 0.30$.
$$r_t(\theta) = \frac{0.30}{0.20} = 1.50$$
If this action yielded positive advantage $\hat{A}_t = +2.0$, the unclipped objective value is:
$$r_t(\theta) \hat{A}_t = 1.50 \times 2.0 = 3.00$$

---

## 2. The Unclipped Surrogate Failure Mode

Using the probability ratio, we can write the **unclipped surrogate objective**:
$$L^{PG}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right]$$
Taking the derivative of $L^{PG}(\theta)$ at $\theta = \theta_{\text{old}}$:
$$\left. \nabla_\theta L^{PG}(\theta) \right|_{\theta = \theta_{\text{old}}} = \left. \hat{\mathbb{E}}_t \left[ \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \right|_{\theta = \theta_{\text{old}}} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$$
At the very first step ($\theta = \theta_{\text{old}}$), the gradient of $L^{PG}$ is mathematically identical to the standard policy gradient!

### The Catastrophic Failure Mode:
If we maximize $L^{PG}(\theta)$ for multiple epochs over the same batch, the optimizer exploits actions with large positive advantages by making $\pi_\theta(a_t \mid s_t) \gg \pi_{\theta_{\text{old}}}(a_t \mid s_t)$, driving $r_t(\theta) \to \infty$. Conversely, actions with negative advantages are driven to $r_t(\theta) \to 0$.

In our controlled experiment ([`experiments/phase3_clipping_behavior.py`](../experiments/phase3_clipping_behavior.py)), optimizing the unclipped surrogate for just 10 epochs caused:
- Approximate KL divergence to explode from $0.000$ to **$0.209$**.
- Probability ratios to wildly swing from $0.03$ to $1.87$.
- Complete policy collapse.

---

## 3. The PPO Clipped Surrogate Objective

To allow multiple optimization epochs while preventing destructively large policy changes, PPO introduces the **clipped surrogate objective**:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
where $\epsilon$ is a hyperparameter (typically $\epsilon = 0.2$).

### The Asymmetric `min`: All Four Cases Analyzed

The `min` operator acts as a **pessimistic lower bound**. Its behavior depends critically on the **sign of the advantage** $\hat{A}_t$:

```text
                  Positive Advantage (A > 0)          Negative Advantage (A < 0)
             ┌─────────────────────────────────┬─────────────────────────────────┐
r > 1 + eps  │ Capped at (1+eps)*A             │ UNCLIPPED: Full penalty r*A     │
             │ (Prevents excessive step size)  │ (Does not shield bad actions)   │
             ├─────────────────────────────────┼─────────────────────────────────┤
r < 1 - eps  │ UNCLIPPED: Objective is r*A     │ Capped at (1-eps)*A             │
             │ (Does not reward reducing good) │ (Prevents over-penalizing)      │
             └─────────────────────────────────┴─────────────────────────────────┘
```

#### Hand-Computed Numerical Walkthrough ($\epsilon = 0.2$, clip range $[0.8, 1.2]$):

1. **Positive Advantage ($\hat{A} = +2.0$), Ratio Exceeds Upper Clip ($r = 1.5$):**
   - $\text{surr1} = 1.5 \times 2.0 = 3.0$
   - $\text{surr2} = \text{clip}(1.5, 0.8, 1.2) \times 2.0 = 1.2 \times 2.0 = 2.4$
   - Objective: $\min(3.0, 2.4) = \mathbf{2.4}$ (**Clipped!** The policy gains no further objective bonus for pushing $r > 1.2$).

2. **Positive Advantage ($\hat{A} = +2.0$), Ratio Drops Below Lower Clip ($r = 0.6$):**
   - $\text{surr1} = 0.6 \times 2.0 = 1.2$
   - $\text{surr2} = 0.8 \times 2.0 = 1.6$
   - Objective: $\min(1.2, 1.6) = \mathbf{1.2}$ (**Unclipped!** The policy is properly penalized for making a good action less likely).

3. **Negative Advantage ($\hat{A} = -2.0$), Ratio Exceeds Upper Clip ($r = 1.5$):**
   - $\text{surr1} = 1.5 \times (-2.0) = -3.0$
   - $\text{surr2} = 1.2 \times (-2.0) = -2.4$
   - Objective: $\min(-3.0, -2.4) = \mathbf{-3.0}$ (**Unclipped!** If the policy mistakenly made a bad action more likely, it receives the full $-3.0$ penalty rather than being artificially shielded by the clip).

4. **Negative Advantage ($\hat{A} = -2.0$), Ratio Drops Below Lower Clip ($r = 0.6$):**
   - $\text{surr1} = 0.6 \times (-2.0) = -1.2$
   - $\text{surr2} = 0.8 \times (-2.0) = -1.6$
   - Objective: $\min(-1.2, -1.6) = \mathbf{-1.6}$ (**Clipped!** The policy is capped from pushing probabilities to absolute zero, preventing gradient explosion).

---

## 4. Diagnostics: Approximate KL and Clip Fraction

When monitoring PPO training, log these essential health metrics:

### A. Approximate KL Divergence
We track the divergence between the rollout policy $\pi_{\text{old}}$ and the updated policy $\pi$:
$$D_{KL}(\pi_{\text{old}} \parallel \pi) = \mathbb{E}_{\pi_{\text{old}}} \left[ \log \pi_{\text{old}}(a \mid s) - \log \pi(a \mid s) \right]$$
- Common sample approximation:
  $$\text{approx\_kl} = \frac{1}{N} \sum_{t=1}^N (\log \pi_{\text{old}}(a_t \mid s_t) - \log \pi(a_t \mid s_t))$$
- Schulman's non-negative variance-reduced estimator:
  $$\text{approx\_kl}_{\text{non-neg}} = \frac{1}{N} \sum_{t=1}^N (r_t - 1 - \log r_t)$$
If `approx_kl` rises above $\approx 0.05-0.10$, the policy is drifting too fast and KL early stopping should trigger.

### B. Clip Fraction
$$\text{clip fraction} = \frac{1}{N} \sum_{t=1}^N \mathbf{1}(|r_t - 1| > \epsilon)$$
- **Clip fraction $\approx 0.0-0.05$**: Very small updates; learning rate or epochs might be too conservative.
- **Clip fraction $\approx 0.10-0.25$**: Healthy operating regime; clipping is doing useful work on the boundary without stifling gradient flow.
- **Clip fraction $> 0.50$**: Aggressive drift; most tokens/steps are hitting the clip boundary and gradients are being zeroed out.

---

## 5. Empirical Benchmarks & Ablations

In [`experiments/phase3_algorithm_comparison.py`](../experiments/phase3_algorithm_comparison.py), we evaluated PPO across multiple seeds and ran controlled ablations:

### 1. Cross-Algorithm Benchmark (3,000 Environment Steps):
| Algorithm | Late Return | Seed Std | Updates per 3,000 Steps | Solve Rate % |
| :--- | :---: | :---: | :---: | :---: |
| **REINFORCE + Baseline** | $+0.938$ | $\pm 0.001$ | ~500 single-step updates | 100% |
| **Actor-Critic (1-Step TD)** | $+0.939$ | $\pm 0.001$ | ~3,000 online updates | 100% |
| **PPO (4 Epochs / Rollout)** | $+0.938$ | $\pm 0.001$ | ~370 minibatch updates | 100% |

### 2. Clipping Epsilon ($\epsilon$) Ablation:
- $\epsilon = 0.05$: Overly tight trust region; high clip fraction ($28\%$), very small KL ($0.0008$).
- $\epsilon = 0.20$: Standard operating point; moderate clip fraction ($12\%$), healthy KL ($0.0035$).
- $\epsilon = 0.40$: Permissive clipping; low clip fraction ($4\%$), larger policy drift per epoch.

### 3. Epoch Reuse ($K$) Ablation:
- $K = 1$: Conservative single pass; under-utilizes collected data.
- $K = 4$: Sweet spot; extracts optimal gradient signal while keeping KL bounded.
- $K = 16$: Diminishing returns; clip fraction surges and updates stall as ratios hit constraint boundaries.

---

## 6. Diagnostic Debugging Exercises

In [`experiments/phase3_debugging_exercises.py`](../experiments/phase3_debugging_exercises.py), we verified five classic PPO implementation bugs:

1. **Recomputing Old Log Probs**: Calculating `old_log_prob` inside the epoch loop caused $r_t \equiv 1.0$, completely disabling clipping.
2. **Inverted Ratio**: Using $\pi_{\text{old}} / \pi$ caused probability increases ($0.2 \to 0.6$) to yield $r = 0.333 < 1.0$, corrupting optimization direction.
3. **Symmetric Clipping Without Min**: For negative advantages ($A = -3.0, r = 1.5$), symmetric clipping capped the penalty at $-3.60$ instead of the true penalty of $-4.50$, shielding bad actions from punishment.
4. **Graph Leakage**: Forgetting `.detach()` on value predictions in the advantage computation caused policy loss gradients to flow directly into the critic.
5. **Padding Masking**: Omitting sequence masks allowed dummy padding tokens to corrupt policy loss by over $5\times$.

---

## 7. Active Recall & Self-Assessment Questions

1. **Why does reusing rollout data create a distribution mismatch?**
2. **What does an importance sampling ratio of $r_t = 1.4$ signify?**
3. **Why can't we simply maximize the unclipped surrogate $r_t \hat{A}_t$ indefinitely?**
4. **Why is the PPO objective formulated with a `min` rather than simple symmetric clipping?**
5. **If an action has a negative advantage ($\hat{A}_t < 0$) and its ratio is $r_t = 1.6$, is its objective clipped or unclipped?**
6. **What does a clip fraction of $0.75$ tell us about the current training regime?**
7. **Why do we monitor KL divergence if the ratio $r_t$ is already clipped?**
8. **What is the difference between $D_{KL}(\pi_{\text{old}} \parallel \pi_{\text{new}})$ and $D_{KL}(\pi_{\text{new}} \parallel \pi_{\text{old}})$?**
9. **Why must `old_log_prob` and `values` be explicitly detached when stored in the rollout buffer?**
10. **Is PPO mathematically a true trust-region method like TRPO?**
11. **Why does advantage normalization change optimization scale rather than action preference?**
12. **What risk arises when setting the epoch reuse count $K$ too high?**
13. **How does sequence padding without masking compromise language-model PPO?**

---

### Solutions & Explanations

<details>
<summary><b>Click to expand answers</b></summary>

1. **Distribution Mismatch**: Rollout data was collected under $\pi_{\theta_{\text{old}}}$. After any parameter update, probabilities change, so sample frequencies no longer match the expectation under the updated policy $\pi_\theta$.
2. **Ratio Interpretation**: $r_t = 1.4$ means the new policy has made action $a_t$ $40\%$ more likely than it was under the rollout policy.
3. **Unclipped Collapse**: The optimizer will greedily exploit actions with positive advantages, taking extreme parameter steps that drive $r_t \to \infty$ and destroy policy stability.
4. **Role of `min`**: The `min` operator acts as a pessimistic lower bound: it caps gains on positive advantage actions, while ensuring that the policy receives the full, unclipped penalty if it mistakenly made a negative advantage action more likely.
5. **Negative Advantage with $r=1.6$**: It is **unclipped**! The objective is $\min(1.6 \times (-2), 1.2 \times (-2)) = \min(-3.2, -2.4) = -3.2$. The policy is not shielded from its mistake.
6. **Clip Fraction = 0.75**: Over $75\%$ of transitions are hitting the clipping threshold. Gradients are being zeroed out across most steps; the learning rate or epoch count is too high.
7. **Why Monitor KL**: Clipping is a local, token-level constraint. Small clipped ratio changes across thousands of steps can compound into a large global policy drift. KL divergence measures overall distributional shift.
8. **KL Directions**: $D_{KL}(p \parallel q) = \mathbb{E}_p[\log p - \log q]$. $D_{KL}(\pi_{\text{old}} \parallel \pi)$ is mode-covering (penalizes $\pi$ being zero where $\pi_{\text{old}}$ has mass), while $D_{KL}(\pi \parallel \pi_{\text{old}})$ is mode-seeking.
9. **Buffer Detach**: If `old_log_prob` retained gradients, the optimizer would differentiate through the denominator of the ratio, corrupting the likelihood ratio definition.
10. **Not True Trust Region**: TRPO enforces a strict hard constraint $\mathbb{E}[D_{KL}] \le \delta$ using second-order natural gradients. PPO is a heuristic first-order approximation that clips the objective instead.
11. **Advantage Normalization**: Subtracting mean and dividing by standard deviation preserves the relative ordering ($A_i > A_j \iff A_{\text{norm}, i} > A_{\text{norm}, j}$), but standardizes the gradient magnitude across diverse environments.
12. **Excessive Epochs ($K$)**: Policy drifts far from the rollout data; clip fraction surges to near $100\%$, updates stall, and the policy overfits to stale samples.
13. **Padding Without Masking**: Padding tokens will receive artificial advantages and gradients, biasing the model toward generating repetitive dummy tokens.

</details>

---

## 8. Research Questions: Bridge to Language-Model Reasoning

1. **Token-Level vs. Sequence-Level Clipping**: In language models, does clipping each token ratio $r_t = \frac{\pi_\theta(x_t \mid x_{<t})}{\pi_{\text{old}}(x_t \mid x_{<t})}$ provide sufficient constraint, or can small product shifts across 1,000 tokens cause catastrophic drift?
2. **PPO vs. DPO vs. GRPO in Reasoning**: DPO eliminates the reinforcement learning loop entirely via closed-form preference loss, while GRPO replaces the learned critic with group-relative advantage normalization. What are the stability and memory trade-offs when scaling to multi-step math reasoning? (This prepares us for **Phase 4: Language Models as Policies**).
