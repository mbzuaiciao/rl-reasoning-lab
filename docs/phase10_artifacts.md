# Phase 10 Artifact Index & Navigation Guide

> *"A reproducible research system maps every hypothesis to its configuration, raw output, statistical audit, case study, and paper report."*

This index maps all components of the Phase 10 Capstone investigation so that independent researchers can navigate and verify the entire research lineage.

---

## Complete Provenance & Lineage Map

| Component | Repository Path | Description | Key Research Entity |
| :--- | :--- | :--- | :--- |
| **Research Plan** | [`docs/phase10_research_plan.md`](phase10_research_plan.md) | Operational definition of genuine reasoning, metric taxonomy, split specifications, and evidence discipline. | Operational Criteria 1–7 |
| **Preregistered Hypotheses** | [`docs/phase10_hypotheses.md`](phase10_hypotheses.md) | Formal predictions $H_1$ through $H_7$ with explicit null hypotheses ($H_0$) and bootstrap test plans. | Hypotheses H1–H7 |
| **Failure Taxonomy** | [`docs/phase10_failure_taxonomy.md`](phase10_failure_taxonomy.md) | 10 formal failure categories with mathematical definitions, detection metrics, and RL effects. | Failure Modes F1–F10 |
| **Qualitative Case Studies** | [`docs/phase10_case_studies.md`](phase10_case_studies.md) | 10 representative generated trajectories across success, shortcut, exploit, and OOD archetypes. | Cases 1–10 |
| **Capstone Paper Report** | [`docs/phase10_capstone_report.md`](phase10_capstone_report.md) | Publication-style research report with 10 sections answering all 19 final empirical questions. | Core Scientific Report |
| **Research Claim Registry** | [`docs/claims.md`](claims.md) | Canonical registry of audited empirical claims with effect sizes, 95% CIs, and replication status. | CLAIMS 01–16 |
| **Orchestrator Benchmark** | [`experiments/phase10_capstone.py`](../experiments/phase10_capstone.py) | Self-contained, deterministic runner executing Core Experiments A–F, probes, mechanisms, and replication. | Benchmark Orchestrator |
| **Machine-Readable Results**| [`experiments/results/phase10_capstone.json`](../experiments/results/phase10_capstone.json) | Complete JSON serialization of manifest, metrics, bootstrap CIs, OOD matrices, and frontiers. | Raw Empirical Data |
| **Unit & Invariant Tests** | [`tests/test_capstone.py`](../tests/test_capstone.py) | Automated tests verifying operational criteria, oracle insulation, budget accounting, and controls. | Invariant Test Suite |

---

## Hypothesis to Artifact Mapping

| Hypothesis | Investigated In | Primary Metric | Raw JSON Key | Validating Test |
| :--- | :--- | :--- | :--- | :--- |
| **H1: Outcome Process Gap** | Core Exp A | Answer Acc minus Path Validity | `h1_outcome_process_gap` | `test_outcome_process_validity_gap` |
| **H2: Process Generalization**| Core Exp A & B | Complete Path Validity Gain | `h2_process_path_gain` | `test_process_reward_path_validity_advantage` |
| **H3: Goodhart Divergence** | Core Exp C | Proxy Reward vs Oracle Utility | `h3_weak_proxy_divergence` | `test_goodhart_divergence_gap` |
| **H4: Reference KL Penalty** | Core Exp E | ECE & Entropy retention | `h4_kl_ece_reduction` | `test_kl_regularization_calibration` |
| **H5: Uncertainty Routing** | Core Exp F | Exploit Catch Rate / Budget | `h5_routing_catch_ratio` | `test_uncertainty_routing_efficiency` |
| **H6: Rejection Sampling** | Core Exp D | Accuracy vs Token Cost | `h6_rs_ppo_diff` | `test_rejection_sampling_parity` |
| **H7: Test-Time Search** | Core Exp D | SFT BoN-8 vs Greedy PPO | `h7_bon8_ppo_diff` | `test_bon_compute_frontier` |

---

## How to Reproduce All Results

To regenerate the entire Phase 10 experimental results from scratch:

```bash
# 1. Run the orchestrated benchmark
uv run python experiments/phase10_capstone.py

# 2. Run all invariant and unit tests
uv run pytest tests/test_capstone.py -v

# 3. Verify complete repository test suite (122+ tests)
uv run pytest tests/ -v
```
