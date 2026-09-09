# Standalone Follow-On Research Agenda: Next Frontiers for `rl-reasoning-lab`

**Repository**: `mbzuaiciao/rl-reasoning-lab`  
**Date**: September 2026  
**Status**: Preregistration & Design Document  
**Prior Art & Baseline**: [`docs/phase10_capstone_report.md`](phase10_capstone_report.md)  
**Claim Registry**: [`docs/claims.md`](claims.md)  

---

## 1. Executive Summary

Phase 10 answered our primary capstone inquiry:
> *"When reinforcement learning improves benchmark accuracy, what evidence distinguishes genuine reasoning improvement from reward exploitation?"*

We demonstrated that **outcome-only RL induces a severe process validity gap** ($45.8\%$), collapsing under depth shifts and counterfactual mutations, whereas **process-supervised RL aligns answer accuracy with valid deduction chains** ($86.5\%$) and generalizes robustly across out-of-distribution (OOD) tasks ([CLAIM-10](claims.md#claim-10-outcome-only-rl-induces-a-severe-process-gap), [CLAIM-11](claims.md#claim-11-process-supervision-enables-ood-depth-and-counterfactual-generalization)).

However, Phase 10 also revealed two major unresolved scientific tensions:
1. **Direction A**: Does process supervision produce true generalizable deductive procedures, or does the policy merely learn syntactic conformity to the specific step-verifier used during training?
2. **Direction B**: Under strictly matched training and inference compute budgets, under what conditions does online policy gradient optimization (PPO) outperform offline Rejection Sampling combined with test-time search (Best-of-$N$)?

This document conducts a rigorous comparative analysis of both directions across eight scientific dimensions and provides a concrete preregistration proposal for the single strongest follow-on research project.

---

## 2. Comparative Evaluation of Candidate Directions

| Evaluation Dimension | Direction A: Generalization vs. Verifier Conformity | Direction B: Policy RL vs. Inference-Time Search Frontier |
| :--- | :--- | :--- |
| **1. Why It Matters** | Modern reasoning models rely heavily on Process Reward Models (PRMs). If models merely memorize PRM stylistic quirks or syntactic templates, step supervision produces brittle models that fail under domain shift. | High-compute RL (e.g., PPO) involves immense engineering complexity and instability. If offline rejection sampling + inference-time search achieves an identical Pareto frontier, complex online RL is computationally unmotivated. |
| **2. Phase 10 Baseline** | Process PPO achieved $86.5\%$ complete-path validity and $66.5\%$ OOD depth accuracy under an oracle step verifier ([CLAIM-11](claims.md#claim-11-process-supervision-enables-ood-depth-and-counterfactual-generalization)). Weak proxy verifiers collapsed to $25.7\%$ utility ([CLAIM-12](claims.md#claim-12-un-gated-proxy-verifiers-trigger-catastrophic-goodhart-divergence)). | Rejection Sampling reached $83.3\%$ IID / $64.6\%$ OOD (within $1.5\%$ of PPO) with zero policy gradient updates. SFT + BoN-8 matched single-sample PPO ($88.5\%$ vs $85.5\%$) at an 8x token expenditure ([CLAIM-16](claims.md#claim-16-test-time-search-matches-policy-rl-under-8x-token-cost)). |
| **3. Unresolved Scientific Gaps** | We do not know whether process-trained policies generalize when evaluated by an *alien* verifier with a different grammatical formalism or reasoning style, or whether process rewards overfit to deterministic step templates. | We do not know the crossover point in problem difficulty (depth $d$, branching factor $b$, solution sparsity $\rho$) where online gradient updates surpass multi-sample search under total FLOP parity (training FLOPs + inference FLOPs). |
| **4. Minimal Next Experiment** | Train policy on DAG Linear Steps (`A->B; B->C`). Test on: (a) Reversed Premise Order, (b) Tree-structured deductions with conjunctions, (c) Rephrased naturalistic relations, (d) Cross-evaluation by an alternative formal grammar verifier. | Construct a 3D parameter grid: Depth $d \in \{2, 4, 6, 8\}$, Branching $b \in \{1, 2, 4\}$, Solution Sparsity $\rho \in \{1.0, 0.1, 0.01\}$. Sweep PPO vs Rejection Sampling vs Best-of-$N$ across a strictly unified FLOP budget curve. |
| **5. Key Confounders** | Verifier syntax divergence vs underlying semantic logic; tokenizer sensitivity to premise reordering; step-level credit assignment noise. | Asymmetric FLOP accounting (forward pass vs backward pass vs rollout caching); sampling temperature tuning for Best-of-$N$; value network capacity bottlenecks in PPO. |
| **6. Falsification Criteria** | If Process PPO maintains $>80\%$ complete-path validity under alien verifier grammars, the "verifier conformity" hypothesis is **falsified**. | If Best-of-$N$ + Rejection Sampling matches or exceeds PPO across all depths $d \in \{2..8\}$ under equal FLOPs, the hypothesis that "PPO provides superior asymptotic reasoning search" is **falsified**. |
| **7. Computational Requirements** | Low compute: $n=10$ seeds on existing synthetic graph suite; execution time $< 5$ minutes on a single CPU core. | Low-to-medium compute: 2D/3D parameter sweep over depths and sampling budgets; execution time $< 15$ minutes on CPU. |
| **8. Standalone Publication Value** | High methodological value for the reward modeling and verifier community. | Exceptional foundational value for the entire LLM reasoning and RL community (directly informs training compute vs inference compute trade-offs). |

---

## 3. Recommended Follow-On Project: Direction B

### Working Title
**"The Compute Frontier of Reasoning: When Does Policy Gradient Optimization Outperform Inference-Time Search and Rejection Sampling?"**

### Rationale for Recommendation
While Direction A is an important investigation into reward modeling, **Direction B addresses the most consequential open question in modern reinforcement learning for reasoning**:

> *Is iterative policy gradient optimization (PPO) genuinely necessary to produce superior reasoning policies, or does it merely approximate test-time rejection sampling and search at significantly higher engineering complexity?*

In Phase 10, Rejection Sampling matched Process PPO within $1.5\%$ accuracy across both IID and OOD splits without requiring value function baselines, GAE hyperparameter tuning, ratio clipping, or multiple PPO epochs per rollout. Furthermore, SFT + Best-of-8 matched PPO's top accuracy without modifying policy weights. 

Direction B will formally delineate the phase boundary where online policy improvement becomes indispensable.

---

## 4. Concrete Research Protocol for Direction B

### 4.1 Primary Research Question
Under strict total computational budget parity (training FLOPs + inference FLOPs), what structural problem characteristics—depth $d$, branching factor $b$, and solution density $\rho$—cause iterative policy optimization to diverge from, and outperform, test-time search and offline rejection sampling?

### 4.2 Preregistered Hypotheses
- **$H_{\text{frontier-1}}$ (Search Dominance in Dense Solution Regimes)**: When target reasoning paths are moderately dense ($\rho \ge 0.15$), inference-time search (Best-of-$N$) and Rejection Sampling achieve higher accuracy per total FLOP than online PPO.
- **$H_{\text{frontier-2}}$ (Optimization Dominance in Needle-in-a-Haystack Regimes)**: When search depth increases ($d \ge 6$) and valid reasoning paths become exponentially sparse ($\rho \le 0.01$), Best-of-$N$ search collapses due to combinatorial explosion, while online PPO achieves statistically superior accuracy under matched FLOPs.
- **$H_{\text{frontier-3}}$ (Sample Diversity vs Parameter Specialization)**: Offline Rejection Sampling suffers from coverage plateauing under deep compositional graphs ($d \ge 8$), whereas online PPO shifts policy mass into previously unreached state regions.

### 4.3 Compute Accounting Invariant
To ensure strict fairness, all methods must adhere to a standardized FLOP accounting metric:
$$\text{Total FLOPs} = 2 \times |\theta| \times \left( N_{\text{fwd}} + 2 \cdot N_{\text{bwd}} \right)$$
- SFT: $N_{\text{fwd}} = B \cdot T$, $N_{\text{bwd}} = B \cdot T$.
- PPO: Rollouts ($N_{\text{fwd}}$) + Critic updates ($N_{\text{fwd}} + N_{\text{bwd}}$) + Actor updates ($N_{\text{fwd}} + N_{\text{bwd}}$).
- Best-of-$N$: Inference forward passes only ($N \times T$).
- Rejection Sampling: Initial sampling forward passes ($K \times T$) + Supervised fine-tuning passes on passing traces.

### 4.4 Experimental Execution Plan
1. **Module Construction**: Implement `src/rl_reasoning_lab/benchmarks/compute_frontier.py` with rigorous FLOP counters.
2. **Environment Parameterization**: Extend `GraphTraversalTask` to support configurable branching factor $b \in \{1, 2, 3, 4\}$ and distractors $k \in \{0, 3, 6\}$.
3. **Execution Script**: Author `experiments/follow_on_compute_frontier.py` executing across $n=15$ paired seeds.
4. **Verification & Claims**: Audit findings with bootstrap paired differences, generate Pareto frontier plots, and register `CLAIM-17` through `CLAIM-19` in `docs/claims.md`.

---

## 5. Summary

This proposed follow-on investigation leverages 100% of the research engineering, statistical audit, and verification infrastructure established in Phases 1–10. It converts the observational insights of the Phase 10 capstone into a decisive, publication-grade contribution on the foundational economics of reasoning systems.
