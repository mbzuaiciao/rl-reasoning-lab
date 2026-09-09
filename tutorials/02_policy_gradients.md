# Tutorial 02: Policy Gradients from First Principles

In [Tutorial 01](01_rl_first_principles.md), we explored tabular reinforcement learning where state-action value functions $Q(s, a)$ were stored as discrete lookup tables and policies were derived greedily via $\arg\max_a Q(s, a)$.

In this tutorial, we transition to **parameterized, differentiable policies** $\pi_\theta(a \mid s)$. Policy gradient methods optimize the policy parameters $\theta \in \mathbb{R}^d$ directly via gradient ascent on the expected cumulative return. This is the exact family of algorithms underpinning modern post-training for reasoning models (e.g., PPO, GRPO, REINFORCE Leave-One-Out).

---

## 1. Mathematical Foundations & The Log-Derivative Trick

### The Objective Function
Let a trajectory $\tau$ be a sequence of states and actions:
$$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T, s_T)$$
with total cumulative return:
$$R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}$$

Our objective is to find parameters $\theta$ that maximize the expected return:
$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [R(\tau)] = \int p_\theta(\tau) R(\tau) \, d\tau$$

### The Core Challenge
Notice that the parameters $\theta$ appear in the **distribution** $p_\theta(\tau)$ over which the expectation is taken, not in the reward function $R(\tau)$.
Taking the gradient directly:
$$\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau) \, d\tau = \int \nabla_\theta p_\theta(\tau) R(\tau) \, d\tau$$
This integral is **not an expectation** under $p_\theta(\tau)$ because $\nabla_\theta p_\theta(\tau)$ is not a valid probability density. We cannot approximate it with empirical samples!

### The Likelihood-Ratio (Log-Derivative) Identity
Recall the calculus identity for the derivative of $\log f(x)$:
$$\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)} \implies \nabla_\theta f(\theta) = f(\theta) \nabla_\theta \log f(\theta)$$
Applying this to $p_\theta(\tau)$:
$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$$

Substituting this identity into our gradient integral gives:
$$\nabla_\theta J(\theta) = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) \, d\tau = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ R(\tau) \nabla_\theta \log p_\theta(\tau) \right]$$

### Why the Environment Dynamics Disappear
A trajectory's probability under policy $\pi_\theta$ and environment dynamics $p(s_{t+1} \mid s_t, a_t)$ factors as:
$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)$$

Taking the natural logarithm transforms the product into a sum:
$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log p(s_{t+1} \mid s_t, a_t)$$

Now take the gradient with respect to policy parameters $\theta$:
$$\nabla_\theta \log p_\theta(\tau) = \nabla_\theta \log p(s_0) + \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \nabla_\theta \log p(s_{t+1} \mid s_t, a_t)$$

Because the initial state distribution $p(s_0)$ and the transition dynamics $p(s_{t+1} \mid s_t, a_t)$ are properties of the environment and **do not depend on $\theta$**, their gradients are identically zero:
$$\nabla_\theta \log p(s_0) = 0, \quad \nabla_\theta \log p(s_{t+1} \mid s_t, a_t) = 0$$

Therefore:
$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

This yields the **Policy Gradient Theorem**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) R(\tau) \right]$$

> [!IMPORTANT]
> **Why this matters for reasoning agents**:
> We do not need a differentiable environment, nor do we need to know the true transition probabilities $p(s_{t+1} \mid s_t, a_t)$. The environment can be a Python interpreter, a theorem prover, or a physical simulation. The policy gradient depends only on the scores of the actions that our model generated.

---

## 2. Tiny Numerical Walkthrough: Hand-Calculated Policy Gradient

Let us step through a single-step decision with exact numbers.
- States: 1 state ($s_0$).
- Actions: 2 actions $\{a_0, a_1\}$.
- Parameters: $\theta = [\theta_0, \theta_1] = [1.0, 2.0]$.
- Softmax policy:
  $$\pi_\theta(a_i) = \frac{e^{\theta_i}}{e^{\theta_0} + e^{\theta_1}}$$

### Step 1: Compute Probabilities
$$e^{1.0} \approx 2.7183, \quad e^{2.0} \approx 7.3891, \quad \sum = 10.1074$$
$$\pi_\theta(a_0) = \frac{2.7183}{10.1074} \approx 0.2689, \quad \pi_\theta(a_1) = \frac{7.3891}{10.1074} \approx 0.7311$$

### Step 2: Sample Action & Score Function
Suppose the agent samples action $a_1$ and receives reward $R = +10.0$.
The score function gradient for a softmax distribution with logits $\theta$ is:
$$\nabla_\theta \log \pi_\theta(a_k) = e_k - \pi_\theta$$
where $e_k$ is the one-hot vector for chosen action $k$.
For chosen action $a_1$:
$$\nabla_\theta \log \pi_\theta(a_1) = [0, 1] - [0.2689, 0.7311] = [-0.2689, +0.2689]$$

### Step 3: Policy Gradient Update
Using learning rate $\alpha = 0.1$ and reward $R = 10.0$:
$$\hat{g} = R \cdot \nabla_\theta \log \pi_\theta(a_1) = 10.0 \cdot [-0.2689, +0.2689] = [-2.689, +2.689]$$
Gradient ascent update:
$$\theta_{\text{new}} = \theta + \alpha \hat{g} = [1.0, 2.0] + 0.1 \cdot [-2.689, +2.689] = [0.7311, 2.2689]$$

### Step 4: Verify Effect on Policy
Recompute new probabilities with $\theta_{\text{new}}$:
$$e^{0.7311} \approx 2.0774, \quad e^{2.2689} \approx 9.6687, \quad \sum = 11.7461$$
$$\pi_{\theta_{\text{new}}}(a_1) = \frac{9.6687}{11.7461} \approx 0.8231$$
The probability of the rewarded action $a_1$ increased from $0.7311$ to $0.8231$!

---

## 3. Variance Reduction: Reward-to-Go and Baselines

While vanilla REINFORCE is unbiased, its variance is notoriously large. Two principled techniques reduce variance without introducing bias:

### A. Causality and Reward-to-Go
In the full return formulation:
$$\nabla_\theta J \approx \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( \sum_{k=0}^{T-1} \gamma^k R_{k+1} \right)$$
Notice that actions at time $t$ are multiplied by rewards that occurred in the past ($R_1, \dots, R_t$). By causality, an action taken at step $t$ cannot influence past rewards.
Past rewards act as pure zero-mean noise. Removing them yields **Reward-to-Go**:
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$
$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

### B. Action-Independent Baselines
We can subtract any baseline $b(s_t)$ that does not depend on action $a_t$:
$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^{T-1} (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

**Proof that baseline introduces zero bias**:
$$\mathbb{E}_{a_t \sim \pi_\theta} \left[ b(s_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = b(s_t) \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \frac{\nabla_\theta \pi_\theta(a \mid s_t)}{\pi_\theta(a \mid s_t)} = b(s_t) \nabla_\theta \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t)$$
Since probabilities sum to 1 ($\sum_a \pi_\theta(a \mid s_t) = 1$), the gradient of a constant is 0:
$$b(s_t) \nabla_\theta (1) = 0$$

Choosing $b(s_t) = V(s_t)$ produces the **Advantage Function**:
$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$
which measures how much better action $a_t$ is compared to the average action in state $s_t$.

---

## 4. Generalized Advantage Estimation (GAE)

In Monte Carlo REINFORCE, $G_t - V(s_t)$ uses the full trajectory return (unbiased, high variance).
In 1-step Actor-Critic, we substitute the one-step bootstrapped TD residual:
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
This has very low variance, but is **biased** whenever the value function $V$ is inaccurate.

**GAE** smoothly interpolates between TD(0) and Monte Carlo via parameter $\lambda \in [0, 1]$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} = \delta_t + \gamma \lambda (1 - d_t) \hat{A}_{t+1}^{\text{GAE}}$$

- $\lambda = 0$: $\hat{A}_t = \delta_t$ (lowest variance, highest bias from $V$).
- $\lambda = 1$: $\hat{A}_t = G_t - V(S_t)$ (unbiased Monte Carlo, highest variance).

---

## 5. Controlled Experiments & Empirical Results

We ran two major controlled experiments in [`experiments/`](../experiments):

### Experiment 1: Measuring Gradient Variance ([`phase2_gradient_variance.py`](../experiments/phase2_gradient_variance.py))
We froze a neural policy and drew $N=200$ independent gradient samples to measure the variance of different estimators:

| Estimator | Mean Grad Norm $\| \mathbb{E}[g] \|$ | Component Variance $\text{Var}(g_j)$ | Pairwise Cosine Similarity |
| :--- | :---: | :---: | :---: |
| **Full Return (Vanilla)** | 8.633 | 1.775 | **+0.142** |
| **Reward-to-Go (Causal)** | 20.513 | 4.117 | **+0.384** |
| **Reward-to-Go + Const Baseline** | 11.974 | **1.297** | **+0.472** |
| **Reward-to-Go + Value Baseline** | 24.469 | 15.873 | **+0.349** |

#### Empirical Separation vs. Textbook Myth:
- **Theory predicts**: Value baseline reduces variance.
- **Empirical observation**: On a completely random initial policy, using an optimal $V^*$ actually increases gradient norm variance because the policy frequently takes catastrophic random actions whose returns diverge wildly from $V^*$. Subtracting a simple constant baseline ($\bar{G}$) was empirically superior for early gradient stability (+0.472 cosine similarity).

---

### Experiment 2: Multi-Seed Benchmark ([`phase2_policy_gradients.py`](../experiments/phase2_policy_gradients.py))
Over 7 distinct random seeds on a 5-state corridor:

| Algorithm | Mean Return (ep 150-250) | Seed Std | 95% Bootstrap Confidence Interval | Mean Entropy |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla REINFORCE** | +0.938 | $\pm 0.001$ | $[+0.937, +0.939]$ | 0.065 |
| **REINFORCE (Reward-to-Go)** | +0.938 | $\pm 0.001$ | $[+0.938, +0.939]$ | 0.066 |
| **REINFORCE + Value Baseline** | +0.937 | $\pm 0.001$ | $[+0.936, +0.938]$ | 0.102 |
| **Actor-Critic (1-Step TD)** | **+0.939** | $\pm 0.001$ | $[+0.939, +0.940]$ | **0.029** |

---

## 6. Diagnostic Debugging Exercises

In [`experiments/phase2_debugging_exercises.py`](../experiments/phase2_debugging_exercises.py), we verified five common research-engineering bugs:

1. **Wrong Loss Sign**: Minimizing $+G \log \pi$ actively suppressed the probability of the goal action from $0.467$ to $0.054$.
2. **Detached Log Probabilities**: Detaching $\log \pi$ caused `loss.requires_grad = False` and resulted in identically zero parameter gradients.
3. **Inverted Entropy Sign**: Minimizing $+1.0 \cdot H(\pi)$ collapsed policy entropy from $0.581$ to $0.000002$ in under 20 updates.
4. **Terminal Bootstrapping Bug**: Forgetting `(1 - done)` injected $+4.50$ phantom value into terminal transitions.

---

## 7. Active Recall & Self-Assessment Questions

Test your understanding. Attempt to formulate an answer before reading the solution below.

1. **Why can we differentiate the policy objective even if the environment transition dynamics are non-differentiable?**
2. **Where does the log probability $\log \pi(a \mid s)$ come from in the derivation?**
3. **Why is Monte Carlo REINFORCE unbiased but high variance?**
4. **Why does subtracting an action-independent baseline $b(s)$ never change the expected gradient?**
5. **What information does an advantage function $A(s, a)$ contain that a raw return $G_t$ does not?**
6. **What happens to the policy if the advantages all have the wrong sign?**
7. **Why do we add an entropy bonus $H(\pi)$ to the loss?**
8. **Why might too much entropy regularization prevent the agent from solving the task?**
9. **What is the fundamental difference between REINFORCE with a value baseline and Actor-Critic?**
10. **What bias does bootstrapping introduce into the policy gradient?**
11. **What does GAE's $\lambda$ parameter control?**
12. **Why is multi-seed reporting mandatory in empirical RL research?**

---

### Solutions & Explanations

<details>
<summary><b>Click to expand answers</b></summary>

1. **Non-Differentiable Environments**: The likelihood ratio trick converts the gradient of an expectation $\nabla_\theta \mathbb{E}_{p_\theta}[R]$ into the expectation of a gradient $\mathbb{E}_{p_\theta}[R \nabla_\theta \log p_\theta]$. The environment dynamics $p(s' \mid s, a)$ do not depend on $\theta$, so their gradients are zero.
2. **Origin of $\log \pi$**: It stems from the calculus identity $\nabla f = f \nabla \log f$. Dividing by $p_\theta(\tau)$ reconstructs a valid probability density for the expectation.
3. **Unbiased vs. High Variance**: It is unbiased because $\mathbb{E}[\sum_t G_t \nabla \log \pi] = \nabla J$ exactly. It has high variance because $G_t$ depends on all future stochastic actions and state transitions across the remaining trajectory.
4. **Baseline Invariance**: Because $\sum_a \pi(a \mid s) = 1$, we have $\sum_a \nabla_\theta \pi(a \mid s) = \nabla_\theta (1) = 0$. Multiplying by any scalar $b(s)$ that does not depend on $a$ preserves this zero sum.
5. **Advantage Information**: The advantage $A(s, a) = Q(s, a) - V(s)$ centers returns around the state's baseline quality. It indicates whether action $a$ was better or worse than the policy's average action from that state.
6. **Wrong Advantage Sign**: If signs are reversed, good actions with positive rewards receive negative gradient updates, driving their probabilities toward zero.
7. **Entropy Bonus**: Encourages the policy to remain stochastic and explore, preventing early collapse into a suboptimal deterministic action.
8. **Excessive Entropy**: If $\beta$ is too large, the optimizer prioritizes maximizing randomness over maximizing task reward, preventing the agent from committing to the optimal action.
9. **REINFORCE vs. Actor-Critic**: REINFORCE evaluates the policy using true Monte Carlo returns from the entire trajectory ($G_t - V(s_t)$). Actor-Critic replaces future returns with a bootstrapped one-step TD estimate ($R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$).
10. **Bootstrapping Bias**: If the critic $V_\phi$ is poorly trained or biased, its TD targets $R + \gamma V_\phi(S')$ are biased, which directly biases the policy gradient.
11. **GAE $\lambda$**: Controls the exponential trade-off between 1-step TD bias ($\lambda=0$) and Monte Carlo variance ($\lambda=1$).
12. **Multi-Seed Reporting**: Policy gradient methods suffer high sample variance and seed sensitivity. A single lucky seed may solve an environment by chance while 9 other seeds diverge. Reporting distributions and confidence intervals across seeds is essential for reproducible science.

</details>

---

## 8. Research Questions: Bridge to Language Models (LLMs)

1. **Token-Level Credit Assignment**: In autoregressive LLMs, a reasoning chain consists of hundreds of tokens ($a_0, a_1, \dots, a_T$) ending with an answer verification reward $R \in \{0, 1\}$. How does sequence-level reward-to-go compare to token-level advantage estimation?
2. **Group Relative Policy Optimization (GRPO)**: Instead of training a separate value network $V_\phi(s)$ (which doubles GPU memory requirements), GRPO samples $K$ reasoning traces for prompt $q$ and uses the empirical mean return of the group as a prompt-specific baseline:
   $$A_i = \frac{R_i - \text{mean}(\{R_k\})}{\text{std}(\{R_k\})}$$
   How does this relate to the constant baseline analyzed in our variance experiment?
3. **PPO vs. REINFORCE**: Why does REINFORCE degrade when multiple gradient updates are taken per batch of generated tokens, necessitating PPO's clipped surrogate objective? (This leads directly into **Phase 3**).
