# Tutorial 04: Language Models as Policies

In [Tutorial 01](01_rl_first_principles.md), [Tutorial 02](02_policy_gradients.md), and [Tutorial 03](03_ppo.md), we explored reinforcement learning in classical discrete environments. We derived Bellman equations, Monte Carlo policy gradients (REINFORCE), and Proximal Policy Optimization (PPO).

In this tutorial, we make the central conceptual transition of modern AI alignment and reasoning: **connecting autoregressive language models directly to the mathematical machinery of reinforcement learning.**

An autoregressive language model is not merely a text predictor; **it is already an RL policy**. In this guide, we demystify this equivalence from first principles, inspect the mechanics of Supervised Fine-Tuning (SFT), dissect exposure bias, derive token-level policy gradients, and apply PPO with reference KL regularization to language models.

---

## 1. The Formal Equivalence: Autoregressive LM as a Policy

In classical reinforcement learning, an agent interacts with an environment modeled as a Markov Decision Process (MDP) tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

Let us map each component of sequence generation directly onto this framework:

| RL Concept | Symbol | Language Modeling Equivalent |
| :--- | :--- | :--- |
| **State** | $s_t$ | The sequence history prefix up to time $t$: $s_t = x_{<t} = [x_0, x_1, \dots, x_{t-1}]$ |
| **Action** | $a_t$ | The next selected token from the vocabulary $\mathcal{V}$: $a_t = x_t \in \mathcal{V}$ |
| **Policy** | $\pi_\theta(a_t \mid s_t)$ | The model's conditional probability distribution over the vocabulary: $P_\theta(x_t \mid x_{<t}) = \text{softmax}(z_t / T)$ |
| **Environment Dynamics** | $\mathcal{P}(s_{t+1} \mid s_t, a_t)$ | **Deterministic concatenation**: $s_{t+1} = [s_t, a_t] = x_{\le t}$ with probability 1.0 |
| **Trajectory** | $\tau$ | A full sequence rollout: $\tau = (x_0, x_1, \dots, x_T)$ |
| **Terminal State** | $s_T$ | Occurs when action $a_T = \langle\text{eos}\rangle$ or maximum sequence length $T = T_{\text{max}}$ |
| **Return / Reward** | $R(\tau)$ | The scalar score assigned to the completion by a verifier, reward model, or environment |

### The Deterministic Environment Insight
In classical robotics or games, environment transition dynamics $p(s' \mid s, a)$ are stochastic and unknown, requiring sampling to explore the world.

In language modeling, the transition dynamics are **completely deterministic and trivial**:
$$\mathcal{P}(s_{t+1} = [s_t, a] \mid s_t, a) = 1.0$$
The environment simply appends the agent's chosen token to the context buffer! All stochasticity in text generation arises strictly from the policy's own action sampling distribution $\pi_\theta(a_t \mid s_t)$.

---

## 2. Supervised Fine-Tuning (SFT) & Exposure Bias

Before applying RL, language models are trained via **Supervised Fine-Tuning (SFT)** on demonstration datasets $\mathcal{D} = \{(x_{\text{prompt}}^{(i)}, x_{\text{target}}^{(i)})\}$.

### SFT Objective
SFT is mathematically equivalent to **Behavior Cloning** (imitation learning). The objective maximizes the log-likelihood of target tokens conditioned on ground-truth prefixes:
$$\mathcal{L}_{\text{SFT}}(\theta) = - \frac{1}{|\mathcal{D}|} \sum_{(x, y) \in \mathcal{D}} \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t})$$

### The Teacher-Forcing Paradigm
During SFT training, the model is evaluated with **teacher forcing**: at every step $t$, the inputs fed into the transformer are the pristine, human-annotated or ground-truth tokens $y_{<t}$. The model is never forced to live with its own mistakes during training.

### Exposure Bias & The Compounding Error Problem
At inference time, the ground truth $y_{<t}$ is unavailable. The model must sample its own tokens $\hat{y}_t \sim P_\theta(\cdot \mid x, \hat{y}_{<t})$ and feed them back into itself:

$$x \longrightarrow \hat{y}_1 \longrightarrow \hat{y}_2 \longrightarrow \dots \longrightarrow \hat{y}_T$$

If the model makes a minor error at step $t=3$, the prefix $[x, \hat{y}_1, \hat{y}_2, \hat{y}_3]$ is now **out-of-distribution (OOD)** relative to the training distribution of pristine prefixes. Because SFT never trained the model on erroneous prefixes, it has never learned how to recover or backtrack. The error cascades uncontrollably.

In our controlled experiment ([`experiments/phase4_sft_baseline.py`](../experiments/phase4_sft_baseline.py)), perturbing a single token in the input prefix (`4 + 1 = X`) completely destabilized the model's output distribution, demonstrating the exposure bias inherent to pure supervised imitation.

### Why SFT Cannot Solve Complex Reasoning Alone
1. **Non-Differentiable Task Objectives**: Real-world reasoning tasks are evaluated on outcomes: does the mathematical proof check out? Does the Python script pass unit tests? SFT cannot optimize these binary outcome verifiers directly.
2. **Exploration of Novel Solution Paths**: SFT penalizes any token deviation from the specific demonstrator trajectory, even if the model discovers an alternative, valid solution path.
3. **No Credit Assignment**: SFT treats all tokens equally under maximum likelihood. It cannot identify which pivotal step made the proof valid or where the calculation went astray.

---

## 3. Policy Gradients for Language Models

To overcome exposure bias and optimize non-differentiable rewards, we treat the sequence generator as an RL agent learning from self-generated rollouts.

### Deriving the LM Policy Gradient
Let $x$ denote the prompt and $y = (y_1, \dots, y_T)$ denote the generated response tokens. The expected sequence reward is:
$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} [R(x, y)]$$

Applying the log-derivative likelihood-ratio trick from [Tutorial 02](02_policy_gradients.md):
$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ \nabla_\theta \log P_\theta(y \mid x) \cdot R(x, y) \right]$$

By the autoregressive factorization of language:
$$\log P_\theta(y \mid x) = \sum_{t=1}^T \log \pi_\theta(y_t \mid x, y_{<t})$$

Substituting this into the gradient:
$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}) \cdot \hat{A}_t \right]$$

### Action Masking: Prompt vs. Completion Tokens
A subtle and critical implementation detail in language model RL is **action masking**:
- The prompt tokens $x$ are sampled from the task distribution or environment prompt buffer, **not** generated by the policy.
- The completion tokens $y$ are sampled from the policy $\pi_\theta$.

If we apply policy gradients to prompt tokens, we are reinforcing conditioning context based on response quality, violating the score function identity and degrading language understanding.

Therefore, we construct a binary **action mask** $M_t$:
$$M_t = \begin{cases} 1 & \text{if position } t \text{ is a generated action token} \\ 0 & \text{if position } t \text{ is a prompt token, padding token, or post-EOS token} \end{cases}$$

The masked loss implemented in PyTorch:
```python
# token_log_probs: (batch_size, seq_len)
# action_mask: (batch_size, seq_len)
# advantages: (batch_size,) or (batch_size, seq_len)

effective_loss = (
    -(token_log_probs * action_mask * advantages.unsqueeze(-1)).sum() / action_mask.sum()
)
```

---

## 4. Proximal Policy Optimization (PPO) for Language Models

In standard REINFORCE, we sample one trajectory, take one gradient step, and discard the data. For large transformers, sampling autoregressive tokens is computationally expensive. We want to reuse rollout trajectories for multiple gradient epochs.

As established in [Tutorial 03](03_ppo.md), multi-epoch data reuse requires **importance sampling** and **PPO clipping**.

### The Token-Level Importance Ratio
For each generated token $y_t$ at state $s_t = [x, y_{<t}]$:
$$r_t(\theta) = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\theta_{\text{old}}}(y_t \mid s_t)} = \exp\left( \log \pi_\theta(y_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(y_t \mid s_t) \right)$$

### The Clipped Objective
$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

When $\hat{A}_t > 0$, clipping prevents the probability of token $y_t$ from expanding beyond $(1+\epsilon)\pi_{\theta_{\text{old}}}$. When $\hat{A}_t < 0$, clipping prevents destructive downward shifts beyond $(1-\epsilon)\pi_{\theta_{\text{old}}}$.

---

## 5. The Reference Model KL Penalty

In classical RL, the policy explores anywhere in the state space to maximize reward. In language models, however, the policy must remain a coherent language model capable of English syntax and logical communication.

### The Reward Hacking / Distribution Collapse Vulnerability
If an RL algorithm optimizes only task reward $R(x, y)$ (e.g. producing the integer `7`), the policy quickly discovers shortcuts:
- It may output degenerate strings like `7 7 7 7 7` or ungrammatical sequences that trigger heuristic parsers.
- It may collapse into deterministic single-token modes, destroying entropy and general reasoning capabilities.

### The Regularized Objective
To anchor the policy to its initial linguistic capabilities, we introduce a **frozen reference model** $\pi_{\text{ref}}$ (typically the SFT checkpoint) and modify the per-token reward:

$$r_{\text{adjusted}}(s_t, a_t) = r(s_t, a_t) - \beta \, D_{\text{KL}}\left(\pi_\theta(\cdot \mid s_t) \parallel \pi_{\text{ref}}(\cdot \mid s_t)\right)$$

Using the unbiased token-level estimator of KL divergence:
$$r_{\text{adjusted}}(s_t, y_t) = r(s_t, y_t) - \beta \left( \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t) \right)$$

### The Role of $\beta$ (Ablation Insights)
From our empirical study in [`experiments/phase4_reference_kl.py`](../experiments/phase4_reference_kl.py):
- **$\beta = 0.0$ (No Penalty)**: Policy drifts unchecked (Ref KL $> 0.78$), risking language degradation and syntactic collapse.
- **$\beta = 0.05 - 0.20$ (Optimal Window)**: Balances reward optimization with an inductive anchor to natural language syntax.
- **$\beta = 1.00$ (Over-regularization)**: The KL penalty dominates the reward signal, restricting exploration and freezing policy improvement.

```
       Trajectory Rollout under Policy π_θ
                      │
                      ▼
         Compute Raw Outcome Reward R(x, y)
                      │
                      ▼
       Subtract Token-Level Reference KL:
   r_adj = R - β [log π_θ(y_t) - log π_ref(y_t)]
                      │
                      ▼
       Compute PPO Clipped Token Gradient
                      │
                      ▼
         Update Policy Weights θ
        (Reference Model π_ref Frozen)
```

---

## 6. Credit Assignment: Outcome-Based vs. Step-Level Rewards

A central challenge in RL for reasoning is **credit assignment**:

1. **Outcome-Based Rewards**:
   A deterministic verifier grades only the final answer at token $T$ (e.g., $+1.0$ for correct answer, $-0.5$ for incorrect answer). Intermediate reasoning tokens receive $r_t = 0$.
   - *Advantage*: Zero risk of misspecified intermediate rewards; perfectly aligned with ground-truth correctness.
   - *Challenge*: Extreme credit assignment ambiguity. Which intermediate deduction made the proof work, and which was redundant?

2. **Step-Level (Process) Rewards**:
   An external verifier or PRM (Process Reward Model) scores every intermediate reasoning step.
   - *Advantage*: Denser gradient signal and direct guidance at each branch point.
   - *Challenge*: Vulnerable to reward hacking and process reward misspecification.

In Phase 4, our implementation supports both terminal outcome credit and per-token KL adjustment, setting the foundation for the reasoning verifiers explored in Phase 5 and Phase 6.

---

## 7. Repository Code Walkthrough

The codebase implements every component from scratch without external high-level frameworks:

| Module | Source File | Purpose & Invariants |
| :--- | :--- | :--- |
| **Tokenizer** | [`src/rl_reasoning_lab/models/tokenizer.py`](../src/rl_reasoning_lab/models/tokenizer.py) | Character-level vocabulary with special tokens (`<pad>`, `<bos>`, `<eos>`). |
| **Transformer** | [`src/rl_reasoning_lab/models/tiny_transformer.py`](../src/rl_reasoning_lab/models/tiny_transformer.py) | Autoregressive causal self-attention, learned position embeddings, greedy & stochastic generation. |
| **Task & Reward** | [`src/rl_reasoning_lab/environments/language_tasks.py`](../src/rl_reasoning_lab/environments/language_tasks.py)<br>[`src/rl_reasoning_lab/rewards/language_rewards.py`](../src/rl_reasoning_lab/rewards/language_rewards.py) | Arithmetic reasoning environment with exact deterministic verification and reward scoring. |
| **SFT Trainer** | [`src/rl_reasoning_lab/trainers/sft.py`](../src/rl_reasoning_lab/trainers/sft.py) | Behavior cloning baseline with prompt loss masking and teacher-forced evaluation. |
| **LM Rollout Buffer** | [`src/rl_reasoning_lab/trainers/lm_rollout.py`](../src/rl_reasoning_lab/trainers/lm_rollout.py) | Caches token sequences, action masks, frozen old log-probs, reference log-probs, and advantages. |
| **LM REINFORCE** | [`src/rl_reasoning_lab/algorithms/lm_policy_gradient.py`](../src/rl_reasoning_lab/algorithms/lm_policy_gradient.py) | Monte Carlo policy gradient algorithm adapted for masked autoregressive sequences. |
| **LM PPO** | [`src/rl_reasoning_lab/algorithms/lm_ppo.py`](../src/rl_reasoning_lab/algorithms/lm_ppo.py) | Clipped surrogate policy optimization with reference model KL regularization and minibatching. |

---

## 8. Active Recall & Self-Assessment

Test your understanding of language models as RL policies. Formulate your answer before expanding each card.

<details>
<summary><b>1. Why is an autoregressive language model mathematically an MDP policy rather than a contextual bandit?</b></summary>

In a contextual bandit, the agent takes a single action given a context, receives a reward, and the episode ends. In autoregressive language generation, each token generated by the model modifies the state for all subsequent decisions: $s_{t+1} = [s_t, a_t]$. Decisions made early in a sequence directly constrain the reachable states and token probabilities later in the sequence, which is the defining characteristic of a sequential Markov Decision Process.
</details>

<details>
<summary><b>2. Why are the environment dynamics in language model generation trivial compared to physical control tasks?</b></summary>

In physical robotics or game environments, the state transition distribution $p(s_{t+1} \mid s_t, a_t)$ is governed by complex, often stochastic physics. In language generation, state transition is simply string/tensor concatenation: $s_{t+1} = [s_t, a_t]$ with probability 1.0. All environment transitions are deterministic and known.
</details>

<details>
<summary><b>3. What is exposure bias in supervised fine-tuning, and how does RL resolve it?</b></summary>

Exposure bias occurs because SFT trains exclusively with teacher forcing on pristine ground-truth prefixes $y_{<t}^*$. At inference time, the model conditions on its own self-generated tokens $\hat{y}_{<t}$. If the model emits an unexpected or slightly sub-optimal token, it enters an out-of-distribution state where errors compound. RL resolves this because trajectories are generated by sampling from the model's own policy $\pi_\theta$, exposing the model to its own distribution of states during training.
</details>

<details>
<summary><b>4. Why must prompt tokens have an action mask value of 0.0 during policy gradient updates?</b></summary>

Prompt tokens are provided by the user or task dataset; they were not chosen by the policy $\pi_\theta$. Applying a policy gradient update $\nabla_\theta \log \pi_\theta(x_{\text{prompt}}) \cdot R$ treats the prompt as an action taken by the model, distorting the likelihood ratio identity and degrading the model's ability to interpret prompts.
</details>

<details>
<summary><b>5. What happens if padding tokens are included in the cross-entropy or policy gradient loss denominator?</b></summary>

Padding tokens contribute zero target probability but inflate the total token count in the denominator. This artificially dilutes the loss by a factor proportional to the amount of padding, making effective learning rates vary wildly depending on batch-level sequence lengths.
</details>

<details>
<summary><b>6. What failure mode occurs if tokens generated after the &lt;eos&gt; token receive policy gradient updates?</b></summary>

Tokens after `<eos>` are unconstrained post-terminal hallucinations. If they receive positive rewards, the model learns that rambling past termination is rewarded, creating verbosity bias and unbounded sequence lengths. If they receive negative rewards, they add unnecessary gradient variance.
</details>

<details>
<summary><b>7. Why must the reference model π_ref have requires_grad=False and remain completely frozen during PPO?</b></summary>

The reference model provides an immovable linguistic anchor. If $\pi_{\text{ref}}$ is updated by the optimizer or modified in-place, the reference distribution chases the policy, causing the KL divergence term $\log \pi_\theta - \log \pi_{\text{ref}}$ to artificially collapse to zero and allowing the policy to drift without constraint.
</details>

<details>
<summary><b>8. In PPO for language models, what is the exact formula for the probability ratio r_t(θ)?</b></summary>

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) \right)$$
where $s_t = x_{\le t-1}$ and $a_t = x_t$.
</details>

<details>
<summary><b>9. What happens if old log probabilities are recomputed dynamically on each PPO epoch rather than cached at rollout time?</b></summary>

If old log probabilities are recomputed with current weights $\theta$, then $\log \pi_{\theta_{\text{old}}} = \log \pi_\theta$, so $r_t(\theta) = \exp(0) = 1.0$ identically on every step. The clipping mechanism never activates, and the benefits of trust-region optimization are completely lost.
</details>

<details>
<summary><b>10. Why does training a language model with RL from random initialization fail on reasoning tasks?</b></summary>

On multi-step reasoning or exact arithmetic tasks, rewards are sparse (awarded only if the entire syntax and numerical calculation are correct). With random weights, the probability of sampling a syntactically valid and correct answer is near zero ($\approx 0.0$). The model receives zero reward on nearly all rollouts, resulting in zero gradients and no learning signal.
</details>

<details>
<summary><b>11. Why does SFT serve as an effective warm-start for RL?</b></summary>

SFT guides the model into the manifold of valid syntax and problem-solving formats. By placing initial rollouts in regions of state space where non-zero rewards occur with reasonable frequency ($>5\%$), policy gradients obtain a non-zero learning signal to climb the reward landscape.
</details>

<details>
<summary><b>12. Why is the causal attention mask strictly required during transformer training and evaluation?</b></summary>

The causal mask prevents position $t$ from attending to tokens at positions $> t$. Without a causal mask, the model has access to future target tokens during training, achieving 0 loss trivially while failing completely during autoregressive generation at test time.
</details>

<details>
<summary><b>13. What is the difference between outcome-based rewards and process-based rewards?</b></summary>

Outcome-based rewards evaluate only the final output (e.g., correct numerical answer at terminal step $T$). Process-based rewards evaluate every intermediate reasoning step. Outcome rewards have zero specification error but high credit assignment ambiguity; process rewards provide dense supervision but are vulnerable to reward hacking.
</details>

<details>
<summary><b>14. What occurs when the reference KL coefficient β is set to 0.0 in language PPO?</b></summary>

When $\beta = 0.0$, the policy is free to maximize reward without staying near the reference distribution. Over training iterations, the policy drifts away from fluent natural language, often collapsing into repetitive tokens, ungrammatical formats, or gibberish that exploits the reward verifier.
</details>

<details>
<summary><b>15. Why does PPO allow multiple training epochs per rollout batch while standard REINFORCE does not?</b></summary>

Standard REINFORCE computes expectations assuming trajectories were generated by the current policy $\pi_\theta$. Once a gradient step changes $\theta$, the data is off-policy and taking additional steps introduces uncontrolled bias. PPO introduces importance sampling ratios $r_t(\theta)$ to correct for the distribution shift and clips large updates to prevent destructive steps.
</details>

---

## 9. Bridge to Phase 5: From Next-Token Prediction to Reasoning Trajectories

With Phase 4 complete, we have verified that:
1. Autoregressive sequence models are sequential decision-making policies.
2. Token generation is an MDP with deterministic concatenation transitions.
3. SFT suffers from exposure bias, while RL optimizes sequence outcomes directly.
4. PPO and reference KL regularization stabilize language policy updates.

In **Phase 5 (RL for Reasoning)**, we shift focus from single-step arithmetic outputs to **extended multi-step chains of thought**:
- How does the model learn to allocate test-time compute through intermediate reasoning steps?
- How does the length and structure of reasoning trajectories evolve under policy gradients?
- How do we distinguish genuine reasoning breakthroughs from surface-level token memorization?
