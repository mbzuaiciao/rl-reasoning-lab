# Tutorial 01: Reinforcement Learning from First Principles

Welcome to the foundational tutorial of `rl-reasoning-lab`. 

Before diving into deep reinforcement learning, policy gradients, or reasoning with language models, we must understand the core mechanics of sequential decision making from first principles. If an engineer or researcher does not understand how credit is assigned backward through time in a 5-state corridor, they will not understand why a 70-billion parameter reasoning model fails to discover multi-step proofs under sparse outcome rewards.

---

## 1. Intuition: Sequential Decision Making under Delayed Feedback

In standard supervised learning, the environment provides an immediate ground-truth label $y^*$ for each input $x$. Errors are local and instantaneous: $\mathcal{L}(f(x), y^*)$.

In **Reinforcement Learning (RL)**:
1. **Actions have consequences beyond immediate reward**: Choosing an action affects not only immediate feedback $R_{t+1}$, but also which state $S_{t+1}$ the agent transitions into, altering all future possibilities.
2. **Feedback is delayed and sparse**: You may take 50 reasoning steps before receiving a single scalar signal indicating whether the final answer is correct ($+1$) or incorrect ($0$).
3. **Exploration vs. Exploitation**: The agent only gathers data from the actions it chooses to execute. If it never explores an alternative line of deduction, it will never discover if that alternative is superior.

---

## 2. Minimum Mathematics

An agent interacts with an environment modeled as a **Markov Decision Process (MDP)**: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

### The Markov Property
A state transition satisfies the Markov property if the future is conditionally independent of the past given the present:
$$\mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t, A_t)$$

### Trajectories, Returns, and Discounting
A trajectory is a sequence:
$$\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T, S_T)$$

The **discounted return** $G_t$ from time-step $t$ is defined as:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1}$$
where $\gamma \in [0, 1)$ is the **discount factor**.
- $\gamma = 0$: Purely myopic agent (only cares about immediate reward $R_{t+1}$).
- $\gamma \to 1$: Far-sighted agent. The effective planning horizon is:
  $$H_{\text{eff}} \approx \frac{1}{1 - \gamma}$$

### Policies
A policy $\pi(a \mid s)$ defines behavior:
$$\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$$

### Value Functions and Bellman Equations
The **state-value function** $V^\pi(s)$ is the expected return starting in state $s$ under policy $\pi$:
$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]$$

Unrolling the recursive definition of $G_t = R_{t+1} + \gamma G_{t+1}$ yields the **Bellman Expectation Equation for $V(s)$**:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r} p(s', r \mid s, a) \left[ r + \gamma V^\pi(s') \right]$$

The **action-value function** $Q^\pi(s, a)$ is:
$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right] = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]$$

### The Bellman Optimality Equations
The optimal value functions satisfy the **Bellman Optimality Equations**:
$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V^*(s') \right]$$
$$Q^*(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]$$

### Contraction Mapping Principle
The Bellman optimality operator $\mathcal{T}^*$ defined by:
$$(\mathcal{T}^* V)(s) = \max_{a} \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')]$$
is a **$\gamma$-contraction** in the supremum norm:
$$\lVert \mathcal{T}^* U - \mathcal{T}^* V \rVert_\infty \le \gamma \lVert U - V \rVert_\infty$$
By Banach's Fixed Point Theorem, applying $\mathcal{T}^*$ repeatedly from any arbitrary initial vector $V_0$ converges exponentially to the unique fixed point $V^*$:
$$\lVert V_k - V^* \rVert_\infty \le \frac{\gamma^k}{1 - \gamma} \lVert V_1 - V_0 \rVert_\infty$$

---

## 3. Implementation: From Dynamic Programming to Q-Learning

In [`src/rl_reasoning_lab/`](../src/rl_reasoning_lab), we implement two fundamental classes of algorithms:

### A. Model-Based: Value Iteration
When transition probabilities $p(s', r \mid s, a)$ are known:
```python
def value_iteration(env, gamma=0.99, theta=1e-8):
    v = np.zeros(env.num_states)
    while True:
        delta = 0.0
        v_new = np.zeros_like(v)
        for s in range(env.num_states):
            q_sa = np.zeros(env.num_actions)
            for a in range(env.num_actions):
                for p, s_next, r, term in env.get_transitions(s, a):
                    future = 0.0 if term else v[s_next]
                    q_sa[a] += p * (r + gamma * future)
            v_new[s] = np.max(q_sa)
            delta = max(delta, abs(v_new[s] - v[s]))
        v = v_new
        if delta < theta:
            break
    return v
```

### B. Model-Free: Tabular Q-Learning
When transition dynamics are unknown and the agent only experiences samples $(S_t, A_t, R_{t+1}, S_{t+1})$:
```python
def q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    q = np.zeros((env.num_states, env.num_actions))
    for ep in range(num_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                a = np.random.randint(env.num_actions)
            else:
                a = np.argmax(q[s])

            step = env.step(a)
            target = (
                step.reward
                if step.terminated
                else step.reward + gamma * np.max(q[step.observation])
            )
            td_error = target - q[s, a]
            q[s, a] += alpha * td_error

            s = step.observation
            done = step.terminated or step.truncated
    return q
```

---

## 4. Tiny Numerical Walkthrough: 3-State Deterministic Corridor

Consider a 3-state corridor:
$$\text{States: } [0] \longleftrightarrow [1] \longleftrightarrow [2_{\text{Goal}}]$$
- State 2 is the absorbing terminal goal ($R = +1.0$).
- Action $0 = \text{LEFT}$, Action $1 = \text{RIGHT}$.
- Discount factor $\gamma = 0.5$.
- Initial values: $V_0 = [0.0, 0.0, 0.0]$.

### Iteration 1:
- State 1 moving RIGHT reaches Goal: $R = 1.0 + 0.5 \times 0.0 = 1.0$.
  $V_1(1) = 1.0$.
- State 0 moving RIGHT reaches State 1: $R = 0.0 + 0.5 \times V_0(1) = 0.0$.
  $V_1(0) = 0.0$.
- State 2 (terminal): $V_1(2) = 0.0$.
- Vector after Iteration 1: $V_1 = [0.0, 1.0, 0.0]$.

### Iteration 2:
- State 0 moving RIGHT reaches State 1: $R = 0.0 + 0.5 \times V_1(1) = 0.0 + 0.5 \times 1.0 = 0.5$.
  $V_2(0) = 0.5$.
- State 1 moving RIGHT reaches Goal: $1.0 + 0.5 \times 0.0 = 1.0$.
  $V_2(1) = 1.0$.
- Vector after Iteration 2: $V_2 = [0.5, 1.0, 0.0]$.

### Iteration 3:
- Maximum change $\Delta = \max_s |V_3(s) - V_2(s)| = 0.0 < \theta$.
- Exact optimal values: $V^* = [0.5, 1.0, 0.0]$.
- Optimal policy: $\pi^*(0) = \text{RIGHT}, \pi^*(1) = \text{RIGHT}$.

Notice how credit propagates backward **one step per Bellman backup**. In an $N$-step reasoning problem, value information requires at least $N$ backward passes to reach the initial decision.

---

## 5. Inspecting Intermediate Quantities

When debugging RL algorithms, do not merely log the final reward. Track these vital diagnostic signals:
1. **Bellman Residual / Maximum Delta**:
   $$\Delta_k = \max_{s} |V_{k+1}(s) - V_k(s)|$$
   Must decay strictly monotonically at rate bounded by $\gamma$. If $\Delta_k$ increases or oscillates, check for cyclic loops or non-contractive updates.
2. **TD Error Distribution**:
   $$\delta_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)$$
   In early training, $\mathbb{E}[|\delta_t|]$ is large. As learning proceeds, mean TD error should converge toward 0. If TD error remains high, check learning rate $\alpha$ or environment stochasticity.
3. **Policy Entropy**:
   $$H(\pi(\cdot \mid s)) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$$
   Monitors the collapse of exploration.

---

## 6. Active-Recall Questions

**Q1: Why must the value of an absorbing terminal state always be zero ($V(s_{\text{term}}) = 0$)?**
> *Answer*: By definition, an absorbing terminal state ends the trajectory. There are no subsequent transitions or future rewards. If you bootstrap from $V(s_{\text{term}}) > 0$, the agent will accumulate fictitious phantom returns every time it reaches the end.

**Q2: What is the difference between On-Policy learning (SARSA) and Off-Policy learning (Q-Learning)?**
> *Answer*: SARSA updates $Q(S_t, A_t)$ using the action $A_{t+1}$ actually sampled from its behavior policy $\pi$. Thus, its estimates reflect the actual risk of exploration mistakes. Q-learning updates $Q(S_t, A_t)$ using $\max_{a} Q(S_{t+1}, a)$ (the greedy target policy), irrespective of what exploratory action the behavior policy actually takes next.

**Q3: Why can't we simply set $\gamma = 1.0$ in continuing tasks?**
> *Answer*: If $\gamma = 1.0$ and the horizon is infinite or contains positive loops, returns $\sum_{t=0}^\infty R_{t+1}$ diverge to infinity, breaking the contraction property and rendering value comparison impossible.

---

## 7. Running the Controlled Experiments

Run the reproduction script from the repository root:
```bash
uv run python experiments/phase1_tabular_foundations.py
```

### Key Empirical Findings:
1. **Discounting & Horizon**: At $\gamma = 0.5$, value at state 0 in a 10-step corridor is $0.5^9 = 0.00195$—the gradient signal is virtually dead. At $\gamma = 0.99$, $V(0) = 0.9135$, but convergence requires over 20 times more iterations.
2. **Exploration & Seed Sensitivity**: In Q-learning with constant low $\epsilon = 0.02$, the agent fails to find the goal in 40% of random seeds within 300 episodes. With decayed exploration ($\epsilon_0 = 1.0 \to 0.02$), it achieves a 100% solve rate across all seeds.
3. **The Cliff Dilemma**: On Sutton's Cliff Walking, Q-learning finds the shortest path right along the cliff edge (optimal under greedy test conditions), but suffers massive penalties during training due to exploratory slips. SARSA learns to take the safe detour around the top, achieving substantially higher on-policy training return.

---

## 8. Common Bugs & Pitfalls

1. **In-place vs. Synchronous Buffer Updates in Value Iteration**:
   Updating $V(s)$ in-place during a single sweep makes the algorithm Gauss-Seidel value iteration. While it often converges faster, it breaks mathematical symmetry and can lead to directional bias depending on state indexing order.
2. **Terminal State Bootstrapping Bug**:
   ```python
   # BUG: Bootstrapping from terminal state!
   target = reward + gamma * max_q_next
   # FIX: Mask out future value if terminated:
   target = reward if terminated else reward + gamma * max_q_next
   ```
3. **Off-by-One Discounting**:
   Ensure $R_{t+1}$ is discounted by $\gamma^0 = 1$, $R_{t+2}$ by $\gamma^1$, etc.
4. **Argmax Tie-Breaking**:
   Using `np.argmax()` always picks the first index among ties. If multiple actions have $Q=0$ at the start, the agent will deterministically pick action 0 instead of exploring. Always break ties randomly:
   ```python
   best_actions = np.flatnonzero(np.isclose(q_s, np.max(q_s)))
   action = np.random.choice(best_actions)
   ```

---

## 9. Research Questions: Bridge to Language-Model Reasoning

As you finish this phase, consider how these tabular principles map to large language models (LLMs) trained with RL (e.g. GRPO, PPO):

1. **The State Space Explosion**: In tabular RL, $|\mathcal{S}|$ is countable and small. In reasoning, a state is any prompt prefix $x_{<t} \in \mathcal{V}^t$. How do value networks generalize across an exponential state space?
2. **Sparse Reward Horizon**: A proof or code solution receives reward only at the `<|endoftext|>` token ($T \approx 2000$). How does the effective horizon $H \approx \frac{1}{1 - \gamma}$ govern whether intermediate tokens receive meaningful credit?
3. **Off-Policy Data Replay**: Can we train language models off-policy using reasoning trajectories generated by a different model or an older checkpoint without suffering distribution collapse?
