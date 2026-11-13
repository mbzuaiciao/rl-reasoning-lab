# `rl-reasoning-lab`

> **An inspectable, first-principles research laboratory investigating when reinforcement learning produces genuine multi-step reasoning improvement versus reward exploitation.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-128%20passing-brightgreen.svg)]()
[![Reproducibility](https://img.shields.io/badge/reproducibility-audited-success.svg)](docs/claims.md)

---

## 1. Central Research Question

Modern foundation models show dramatic accuracy gains on benchmark suites when fine-tuned with reinforcement learning against automated verifiers. However, benchmark accuracy is an aggregate terminal metric that conflates two fundamentally different phenomena:

> **When reinforcement learning improves benchmark accuracy, under what empirical conditions has the policy learned a better multi-step reasoning procedure, rather than merely a better way to produce rewarded outputs?**

`rl-reasoning-lab` investigates this question within a controlled, fully inspectable laboratory. By eliminating the confounding factors of web-scale pre-training, opaque RLHF libraries, and black-box LLM APIs, every component—token probabilities, attention matrices, intermediate deduction steps, environment rewards, verifier blind spots, and non-parametric bootstrap confidence intervals—is inspectable from first principles.

---

## 2. Core Research Navigation

| Document | Description | Reading Time |
| :--- | :--- | :---: |
| [`frontend/`](frontend/README.md) | **Interactive Research Frontend**: Apple-inspired reader for all 9 tutorials, results explorer, and interactive teaching demos. | Live Web |
| [`docs/research_summary.md`](docs/research_summary.md) | **Executive Research Brief**: Non-obvious dynamics, core findings, null results, and boundaries. | ~5 min |
| [`docs/phase10_capstone_report.md`](docs/phase10_capstone_report.md) | **Full Research Paper**: Complete experimental setup, treatment matrices, distributions, and answers to 19 empirical questions. | ~20 min |
| [`docs/claims.md`](docs/claims.md) | **Canonical Claim Registry**: 16 preregistered claims (CLAIM-01 to CLAIM-16) with bootstrap CIs, effect sizes ($d$), evidence strength, and status. | ~10 min |
| [`docs/phase10_research_plan.md`](docs/phase10_research_plan.md) | **Research Design & Preregistration**: Operational definitions of reasoning improvement, 7 criteria, and metric specifications. | ~10 min |
| [`docs/phase10_hypotheses.md`](docs/phase10_hypotheses.md) | **Preregistered Hypotheses (H1–H7)**: Directional predictions, explicit nulls ($H_0$), and falsification thresholds. | ~8 min |
| [`docs/phase10_failure_taxonomy.md`](docs/phase10_failure_taxonomy.md) | **Empirical Failure Taxonomy**: 10 distinct error modes with token-level examples and diagnostic probes. | ~10 min |
| [`docs/phase10_case_studies.md`](docs/phase10_case_studies.md) | **Qualitative Case Studies**: 10 representative trajectory inspections across training regimes. | ~10 min |
| [`docs/follow_on_research.md`](docs/follow_on_research.md) | **Follow-On Research Agenda**: Evaluation of candidate directions and concrete protocol for the Compute Frontier of reasoning. | ~8 min |
| [`docs/exp13_code_repair_report.md`](docs/exp13_code_repair_report.md) | **Experiment 13**: Training-free feasibility pilot for public test feedback in Python code repair. | ~3 min |
| [`docs/exp14_model_selection_report.md`](docs/exp14_model_selection_report.md) | **Experiment 14**: Cached-model inventory and bounded local Python repair comparison. | ~4 min |
| [`docs/exp15_repair_report.md`](docs/exp15_repair_report.md) | **Experiment 15**: Frozen local pilot for code-repair difficulty and public-test feedback. | ~3 min |
| [`docs/exp16_token_cap_report.md`](docs/exp16_token_cap_report.md) | **Experiment 16**: Paired diagnostic of the output-token cap for Qwen3.5-4B code repair. | ~3 min |
| [`docs/exp17_public_feedback_report.md`](docs/exp17_public_feedback_report.md) | **Experiment 17**: Paired code-repair revisions with and without public-test feedback. | ~3 min |

---

## 3. Headline Findings

All findings are evaluated across $n=10$ paired training seeds and audited using non-parametric bootstrap confidence intervals and Cohen's $d$ effect sizes traceable to [`experiments/results/phase10_capstone.json`](experiments/results/phase10_capstone.json):

1. **Outcome RL Induces a Severe Process Validity Gap ($45.8\%$)**: While Outcome-only PPO achieves $87.8\%$ in-distribution answer accuracy, complete-path validity plummets to $42.0\%$ ($d = 10.92$, 95% CI: $[+42.7\%, +49.5\%]$). Under out-of-distribution (OOD) depth shifts ($d=4..6$), accuracy collapses to $16.7\%$, and drops to $19.9\%$ when dataset shortcuts are counterbalanced ([CLAIM-10](docs/claims.md#claim-10-outcome-only-rl-induces-a-severe-process-gap)).
2. **Process Supervision Enables Robust OOD Generalization**: Rewarding valid intermediate transitions aligns answer accuracy ($84.7\%$) with complete-path validity ($86.5\%$), producing a $+44.8\%$ complete-path advantage over outcome RL ($d = 10.92$). This procedural grounding transfers to unseen depths ($66.5\%$) and single-edge counterfactual mutations ($74.2\%$; [CLAIM-11](docs/claims.md#claim-11-process-supervision-enables-ood-depth-and-counterfactual-generalization)).
3. **Flawed Proxy Verifiers Trigger Goodhart Divergence ($+68.8\%$) Eliminated by Invariant Gating**: Optimizing against an un-gated proxy verifier drives proxy reward to $94.5\%$ while true oracle utility collapses to $25.7\%$ ($64.9\%$ exploit rate; [CLAIM-12](docs/claims.md#claim-12-un-gated-proxy-verifiers-trigger-catastrophic-goodhart-divergence)). Enforcing lightweight topological invariant gates suppresses learned exploits to $1.3\%$ ($d = +18.20$; [CLAIM-13](docs/claims.md#claim-13-invariant-gating-suppresses-proxy-exploitation)).
4. **Offline Rejection Sampling Matches PPO at Zero Gradient Instability**: Generating multi-sample rollouts and supervised fine-tuning on oracle-passing traces achieves $83.3\%$ IID and $64.6\%$ OOD accuracy—matching Process PPO within $1.5\%$ without policy gradient variance, value function baselines, or clipping heuristics ([CLAIM-16](docs/claims.md#claim-16-test-time-search-matches-policy-rl-under-8x-token-cost)).
5. **Verifier Disagreement Dominates Policy Entropy for Selective Auditing**: Under a fixed $20\%$ oracle query budget, routing ambiguous rollouts via inter-verifier disagreement catches $69.0\%$ of latent exploits ($3.4\times$ the efficiency of uniform auditing, $d = +7.80$), whereas policy entropy catches only $48.0\%$ due to post-RL confidence collapse ([CLAIM-15](docs/claims.md#claim-15-verifier-disagreement-escalation-improves-audit-efficiency), [CLAIM-06](docs/claims.md#claim-06-rl-policy-optimization-degrades-confidence-calibration)).

---

## 4. Threats to Validity and Study Limitations

1. **Model Architecture Scale**: Experiments utilize a 2-layer autoregressive causal transformer (`TinyTransformer`, 64-dim hidden state, 2 attention heads). While causal attention, softmax parameterizations, and policy gradient dynamics match frontier foundation models, this architecture does not capture massive pre-trained world knowledge.
2. **Synthetic Graph Traversal Domain**: Reasoning tasks are defined over directed acyclic graphs ($|V|=8$). This provides exact, mathematically un-corrupted ground truth for every step, but lacks the semantic nuance and polysemy of natural language text.
3. **Chain-of-Thought Observability**: We assume the emitted discrete token trace represents the computational deduction. In heavily overparameterized models, internal latent reasoning representations may decouple from generated tokens.
4. **Verifier Availability**: Step-level process supervision requires verifiable domain transition rules. In open-ended domains lacking formal semantics, step verifiers must themselves be learned, introducing second-order Goodhart vulnerabilities.

---

## 5. Curriculum and Repository Lineage

The research capstone in Phase 10 rests on a cumulative 10-phase experimental curriculum built from first principles:

| Phase | Scientific & Algorithmic Domain | Key Theoretical Insights | Tutorials & Code |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **RL Foundations** | Bellman equations, dynamic programming, TD learning, Q-learning on discrete MDPs. | [`tutorials/01_rl_first_principles.md`](tutorials/01_rl_first_principles.md) |
| **Phase 2** | **Policy Gradients** | Likelihood ratio derivation, score function, baseline variance reduction ($d=-10.55$), GAE. | [`tutorials/02_policy_gradients.md`](tutorials/02_policy_gradients.md) |
| **Phase 3** | **PPO Foundations** | Importance sampling, clipped surrogate objective, empirical KL bounds, trust-region drift. | [`tutorials/03_ppo.md`](tutorials/03_ppo.md) |
| **Phase 4** | **LM Policy Optimization** | Autoregressive token MDP, causal transformer, SFT warm-start, reference KL regularization. | [`tutorials/04_language_models_as_policies.md`](tutorials/04_language_models_as_policies.md) |
| **Phase 5** | **RL for Reasoning** | Symbolic graph reasoning, step vs outcome rewards, credit assignment, Rejection Sampling. | [`tutorials/05_rl_for_reasoning.md`](tutorials/05_rl_for_reasoning.md) |
| **Phase 6** | **Rewards & Verifiers** | Goodhart divergence, heuristic proxy failure, verifier ensembles, topological invariant gates. | [`tutorials/06_rewards_and_verifiers.md`](tutorials/06_rewards_and_verifiers.md) |
| **Phase 7** | **Uncertainty & Calibration** | Optimization-induced overconfidence, ECE, selective prediction risk, epistemic ensembles. | [`tutorials/07_uncertainty_and_calibration.md`](tutorials/07_uncertainty_and_calibration.md) |
| **Phase 8** | **Experimental Science** | Seed sensitivity, best-seed bias, bootstrap CIs, paired tests, multiple testing correction. | [`tutorials/08_experimental_science.md`](tutorials/08_experimental_science.md) |
| **Phase 9** | **Research Engineering** | Atomic checkpoints, PRNG state isolation, fail-fast boundary validation, gradient forensics. | [`tutorials/09_research_engineering.md`](tutorials/09_research_engineering.md) |
| **Phase 10** | **Integrated Research Capstone** | Preregistered benchmark on genuine reasoning improvement vs reward exploitation. | [`docs/phase10_capstone_report.md`](docs/phase10_capstone_report.md) |

---

## 6. Reproducibility & Quickstart

All benchmarks and unit tests run deterministically and execute locally in seconds.

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/mbzuaiciao/rl-reasoning-lab.git
cd rl-reasoning-lab

# Set up virtual environment using uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Reproduce the Phase 10 Capstone Benchmark
```bash
# Execute the preregistered research benchmark (generates experiments/results/phase10_capstone.json)
python experiments/phase10_capstone.py
```

### Run Full Test Suite (128 Passing Tests)
```bash
pytest tests/ -v
```

### Run Linter and Formatting Checks
```bash
ruff check .
ruff format --check .
```

### Launch Interactive Research Frontend
```bash
cd frontend
npm install
npm run dev     # Starts local server at http://localhost:4321
npm test        # Runs frontend integrity & contract tests
npm run build   # Builds standalone static production site (dist/)
```
