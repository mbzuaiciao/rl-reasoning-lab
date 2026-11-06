import fs from 'node:fs';
import path from 'node:path';
import type { CapstoneMetrics, DocMeta, ResearchClaim, TutorialMeta } from './types';

// Resolve project root paths
const getRootDir = () => path.resolve(process.cwd(), '..');

export const TUTORIALS_METADATA: TutorialMeta[] = [
  {
    slug: '01_rl_first_principles',
    number: '01',
    phase: 1,
    title: 'Reinforcement Learning from First Principles',
    subtitle: 'Markov Decision Processes, Dynamic Programming, and Bellman Equations',
    description: 'Deconstruct sequential decision making under delayed feedback using discrete corridor MDPs, value iteration, and temporal difference learning.',
    readingTimeMinutes: 12,
    filePath: 'tutorials/01_rl_first_principles.md'
  },
  {
    slug: '02_policy_gradients',
    number: '02',
    phase: 2,
    title: 'Policy Gradients & Score Function Estimation',
    subtitle: 'Likelihood Ratio Gradients, REINFORCE, and Baseline Variance Reduction',
    description: 'Derive policy gradients mathematically, explore score function estimation, and prove baseline variance reduction without introducing bias (d = -10.55).',
    readingTimeMinutes: 15,
    filePath: 'tutorials/02_policy_gradients.md'
  },
  {
    slug: '03_ppo',
    number: '03',
    phase: 3,
    title: 'Proximal Policy Optimization & Trust Regions',
    subtitle: 'Surrogate Objectives, Ratio Clipping, and Empirical KL Bounds',
    description: 'Analyze policy collapse under multiple policy updates, clipped surrogate objectives, and empirical divergence bounds (d = -7.33).',
    readingTimeMinutes: 16,
    filePath: 'tutorials/03_ppo.md'
  },
  {
    slug: '04_language_models_as_policies',
    number: '04',
    phase: 4,
    title: 'Language Models as Reinforcement Learning Policies',
    subtitle: 'Autoregressive Token MDPs, Causal Attention, and SFT Warm-Start',
    description: 'Map language generation to sequential decision making with TinyTransformer, demonstrating that SFT warm-start accelerates RL convergence by >5x (d = -9.75).',
    readingTimeMinutes: 20,
    filePath: 'tutorials/04_language_models_as_policies.md'
  },
  {
    slug: '05_rl_for_reasoning',
    number: '05',
    phase: 5,
    title: 'Reinforcement Learning for Multi-Step Reasoning',
    subtitle: 'Symbolic Graphs, Process vs Outcome Feedback, and Rejection Sampling',
    description: 'Examine step-level credit assignment, shortcut discovery, counterfactual premise mutations, and test-time search vs online policy gradients.',
    readingTimeMinutes: 22,
    filePath: 'tutorials/05_rl_for_reasoning.md'
  },
  {
    slug: '06_rewards_and_verifiers',
    number: '06',
    phase: 6,
    title: 'Rewards, Verifiers, and Goodhart\'s Law',
    subtitle: 'Proxy Optimization Divergence, Verifier Vulnerabilities, and Invariant Gates',
    description: 'Demonstrate Goodhart divergence (+68.8% gap) under heuristic substring verifiers and prove how lightweight invariant gates suppress exploitation to 1.3%.',
    readingTimeMinutes: 18,
    filePath: 'tutorials/06_rewards_and_verifiers.md'
  },
  {
    slug: '07_uncertainty_and_calibration',
    number: '07',
    phase: 7,
    title: 'Uncertainty, Calibration, and Selective Prediction',
    subtitle: 'Post-RL Overconfidence, Expected Calibration Error, and Deep Ensembles',
    description: 'Quantify optimization-induced probability sharpening, evaluate selective prediction under risk, and demonstrate deep ensemble error detection.',
    readingTimeMinutes: 18,
    filePath: 'tutorials/07_uncertainty_and_calibration.md'
  },
  {
    slug: '08_experimental_science',
    number: '08',
    phase: 8,
    title: 'Experimental Science, Rigor, and Reproducibility',
    subtitle: 'Bootstrap Confidence Intervals, Best-Seed Bias, and Preregistered Claims',
    description: 'Establish empirical standards: paired difference tests, 20-seed sensitivity sweeps (+18.6% best-seed inflation), and multiple testing corrections.',
    readingTimeMinutes: 18,
    filePath: 'tutorials/08_experimental_science.md'
  },
  {
    slug: '09_research_engineering',
    number: '09',
    phase: 9,
    title: 'Research Engineering, Scaling, and Debugging',
    subtitle: 'Atomic Checkpoints, PRNG Isolation, Shape Guards, and Forensic Debugging',
    description: 'Master the engineering backbone of dependable RL research: microbatch gradient equivalence, PRNG determinism, and health monitors.',
    readingTimeMinutes: 28,
    filePath: 'tutorials/09_research_engineering.md'
  }
];

export const DOCS_METADATA: DocMeta[] = [
  {
    slug: 'research_summary',
    title: 'Executive Research Summary',
    subtitle: 'Core findings, non-obvious dynamics, null results, and operational boundaries',
    category: 'capstone',
    readingTimeMinutes: 5,
    filePath: 'docs/research_summary.md'
  },
  {
    slug: 'phase10_capstone_report',
    title: 'Phase 10 Capstone Report',
    subtitle: 'Full research paper: 8 treatments, 10 paired seeds, and 19 empirical answers',
    category: 'capstone',
    readingTimeMinutes: 20,
    filePath: 'docs/phase10_capstone_report.md'
  },
  {
    slug: 'claims',
    title: 'Canonical Claim Registry',
    subtitle: '16 preregistered claims (CLAIM-01 to CLAIM-16) with bootstrap CIs and effect sizes',
    category: 'claims',
    readingTimeMinutes: 10,
    filePath: 'docs/claims.md'
  },
  {
    slug: 'phase10_hypotheses',
    title: 'Preregistered Hypotheses (H1–H7)',
    subtitle: 'Directional predictions, explicit nulls (H0), and falsification thresholds',
    category: 'methodology',
    readingTimeMinutes: 8,
    filePath: 'docs/phase10_hypotheses.md'
  },
  {
    slug: 'phase10_research_plan',
    title: 'Research Design & Preregistration',
    subtitle: 'Operational definitions of reasoning improvement, 7 criteria, and metric specs',
    category: 'methodology',
    readingTimeMinutes: 10,
    filePath: 'docs/phase10_research_plan.md'
  },
  {
    slug: 'phase10_failure_taxonomy',
    title: 'Empirical Failure Taxonomy',
    subtitle: '10 distinct error modes with token-level examples and diagnostic probes',
    category: 'audit',
    readingTimeMinutes: 10,
    filePath: 'docs/phase10_failure_taxonomy.md'
  },
  {
    slug: 'phase10_case_studies',
    title: 'Qualitative Case Studies',
    subtitle: '10 representative trajectory inspections across training regimes',
    category: 'audit',
    readingTimeMinutes: 10,
    filePath: 'docs/phase10_case_studies.md'
  },
  {
    slug: 'phase8_experimental_report',
    title: 'Phase 8 Experimental Report',
    subtitle: 'Multi-seed statistical rigor, factorial ablations, and seed sensitivity audits',
    category: 'audit',
    readingTimeMinutes: 12,
    filePath: 'docs/phase8_experimental_report.md'
  },
  {
    slug: 'phase10_artifacts',
    title: 'Phase 10 Artifacts Registry',
    subtitle: 'Committed experiment outputs, data schemas, and run identifiers',
    category: 'audit',
    readingTimeMinutes: 4,
    filePath: 'docs/phase10_artifacts.md'
  },
  {
    slug: 'follow_on_research',
    title: 'Follow-On Research Agenda',
    subtitle: 'Evaluation of candidate directions and concrete protocol for the Compute Frontier',
    category: 'future',
    readingTimeMinutes: 8,
    filePath: 'docs/follow_on_research.md'
  }
];

export const RESEARCH_CLAIMS: ResearchClaim[] = [
  {
    id: 'CLAIM-01',
    phase: 2,
    title: 'Baseline Variance Reduction in Policy Gradients',
    primaryMetric: 'Policy Gradient Variance Var(g)',
    effectSize: 'd = -10.55',
    cohensD: -10.55,
    status: 'robustly reproduced',
    evidenceStrength: 'Analytic derivation & empirical validation in controlled MDP',
    claim: 'Incorporating a state-dependent value baseline V(s) into Monte Carlo policy gradients reduces gradient estimator variance without introducing bias.',
    evidence: 'Variance with baseline 3.08 ± 0.4 vs without baseline 12.70 ± 1.2 (Δ = -9.62, 95% CI: [-10.25, -8.94], Cohen\'s d = -10.55).',
    experiment: 'experiments/phase2_gradient_variance.py & phase8_research_claim_audit.py',
    seeds: '12 paired seeds',
    metricDetails: 'High-horizon corridor (T=50) testing confirms variance reduction expands to 78%.',
    limitations: 'Baseline quality depends on value function approximation; poorly fitted value networks can fail to reduce variance.'
  },
  {
    id: 'CLAIM-02',
    phase: 3,
    title: 'PPO Clipping Constrains Policy Drift',
    primaryMetric: 'Empirical Policy KL Divergence',
    effectSize: 'd = -7.33',
    cohensD: -7.33,
    status: 'robustly reproduced',
    evidenceStrength: 'Empirical demonstration in local trust region regime',
    claim: 'The pessimistic clipped surrogate objective in PPO bounds destructive policy drift when performing multiple gradient updates per rollout batch.',
    evidence: 'Empirical KL: Clipped 0.074 vs Unclipped 1.908 (Δ = -1.834, 95% CI: [-2.02, -1.65], Cohen\'s d = -7.33).',
    experiment: 'experiments/phase3_clipping_behavior.py & phase8_research_claim_audit.py',
    seeds: '12 paired seeds',
    metricDetails: 'Under excessive learning rate (lr=0.1), clipping keeps drift < 0.20 while unclipped diverges (>5.0).',
    limitations: 'Clipping is a local heuristic, not a mathematically exact trust region.'
  },
  {
    id: 'CLAIM-03',
    phase: 4,
    title: 'SFT Warm-Start Accelerates LM Policy Optimization',
    primaryMetric: 'Episodes to 50% Task Accuracy',
    effectSize: 'd = -9.75',
    cohensD: -9.75,
    status: 'robustly reproduced',
    evidenceStrength: 'Empirical demonstration in sparse reward regime',
    claim: 'Initializing a language model policy with Supervised Fine-Tuning (SFT) on reasoning traces before applying RL prevents policy collapse and accelerates convergence by >5x.',
    evidence: 'Episodes to 50% accuracy: SFT 199 vs Random 1,480 (Δ = -1,281 episodes, 95% CI: [-1375, -1181], Cohen\'s d = -9.75).',
    experiment: 'experiments/phase4_sft_vs_rl.py & phase8_research_claim_audit.py',
    seeds: '12 paired seeds',
    metricDetails: 'Zero-shot novel vocabulary testing shows SFT warm-start enables 3x faster adaptation than random init.',
    limitations: 'High SFT coverage can bias the model toward memorized paths.'
  },
  {
    id: 'CLAIM-04',
    phase: 5,
    title: 'Process Reward Improves Reasoning Path Validity',
    primaryMetric: 'Multi-Step Reasoning Path Validity',
    effectSize: 'd = +9.88 (IID)',
    cohensD: 9.88,
    status: 'robustly reproduced',
    evidenceStrength: 'Controlled synthetic setting (IID-strong, OOD-fragile under noisy verifiers)',
    claim: 'Rewarding intermediate valid reasoning steps produces higher complete-path validity and distractor robustness than rewarding final answer correctness alone.',
    evidence: 'Path validity with distractors: Process 88.6% vs Outcome 42.8% (Δ = +45.8%, 95% CI: [+42.1%, +49.5%], Cohen\'s d = +9.88).',
    experiment: 'experiments/phase5_process_vs_outcome_rl.py & phase8_ablation_and_interactions.py',
    seeds: '10 paired seeds',
    metricDetails: 'On deeper graph (d=5 vs train d=3), process validity drops to 68.0% OOD, but remains far above outcome-only (18.0%).',
    limitations: 'Requires a step-level verifier. If step verifier is noisy, process RL can reinforce hallucinated intermediate tokens.'
  },
  {
    id: 'CLAIM-05',
    phase: 6,
    title: 'Imperfect Proxy Verifiers Induce Goodhart Divergence',
    primaryMetric: 'Ground-Truth Task Utility vs Proxy Reward',
    effectSize: 'd = -20.79',
    cohensD: -20.79,
    status: 'robustly reproduced',
    evidenceStrength: 'Empirical demonstration in misspecified proxy verifier setting',
    claim: 'Optimizing an RL policy aggressively against an imperfect proxy reward function causes true task utility to collapse even as proxy reward approaches 100%.',
    evidence: 'Proxy reward 94.3% vs True utility 22.7% (Δ = -71.6%, 95% CI: [-74.0%, -69.2%], Cohen\'s d = -20.79).',
    experiment: 'experiments/phase6_reward_utility_divergence.py & phase8_research_claim_audit.py',
    seeds: '12 paired seeds',
    metricDetails: 'Under adversarial syntactic traps, true utility collapses to 12.0% while proxy remains at 96.0%.',
    limitations: 'Divergence rate depends on verifier loophole density and policy search capacity.'
  },
  {
    id: 'CLAIM-06',
    phase: 7,
    title: 'RL Policy Optimization Degrades Confidence Calibration',
    primaryMetric: 'Expected Calibration Error (ECE)',
    effectSize: 'd = +6.96',
    cohensD: 6.96,
    status: 'robustly reproduced',
    evidenceStrength: 'Empirical demonstration across multiple seeds in controlled setting',
    claim: 'RL training sharpens policy probabilities and maximizes expected reward, which systematically increases overconfidence and inflates Expected Calibration Error (ECE).',
    evidence: 'ECE: SFT 0.046 ± 0.008 vs RL 0.137 ± 0.015 (Δ = +0.091, 95% CI: [+0.082, +0.100], Cohen\'s d = +6.96).',
    experiment: 'experiments/phase7_calibration_sft_vs_rl.py & phase8_research_claim_audit.py',
    seeds: '12 paired seeds',
    metricDetails: 'Under OOD depth shift (d=5), ECE explodes to 0.280 as accuracy drops while policy confidence remains >90%.',
    limitations: 'Entropy regularization and KL penalty can mitigate, but not eliminate, optimization-induced overconfidence.'
  },
  {
    id: 'CLAIM-07',
    phase: 7,
    title: 'Deep Ensembles Detect OOD Errors Better Than Sampling',
    primaryMetric: 'AUROC on OOD Graph Depth Shifts',
    effectSize: 'd = +6.36',
    cohensD: 6.36,
    status: 'robustly reproduced',
    evidenceStrength: 'Controlled OOD distribution shift validation',
    claim: 'Epistemic variance across diverse model checkpoints reliably flags OOD reasoning errors where single-model sampling variance collapses into confident failure modes.',
    evidence: 'Error detection AUROC: Deep Ensemble 0.854 ± 0.02 vs Sampling 0.685 ± 0.03 (Δ = +0.169, 95% CI: [+0.151, +0.188], Cohen\'s d = +6.36).',
    experiment: 'experiments/phase7_policy_ensemble.py & phase8_research_claim_audit.py',
    seeds: '12 paired seeds',
    metricDetails: 'Under cyclic distractor graphs, ensemble AUROC maintains 0.81 whereas sampling AUROC drops to 0.58.',
    limitations: 'Requires maintaining and querying M distinct model checkpoints, scaling inference cost by Mx.'
  },
  {
    id: 'CLAIM-08',
    phase: 8,
    title: 'Three Seeds Suffer From Severe Best-Seed Bias',
    primaryMetric: 'Best-Seed Accuracy vs Mean Across 20 Seeds',
    effectSize: 'd = +4.10',
    cohensD: 4.10,
    status: 'robustly reproduced',
    evidenceStrength: 'Statistically validated across 20 seeds',
    claim: 'Reporting the single best seed from an n=3 or n=20 trial introduces massive optimistic bias (+15–20% accuracy inflation) and conceals catastrophic failure modes.',
    evidence: 'Best seed reached 95.0% accuracy, whereas true mean was 76.4% and worst seed collapsed to 13.9% (Best-seed inflation: +18.6%).',
    experiment: 'experiments/phase8_seed_sensitivity.py',
    seeds: '20 seeds',
    metricDetails: '95% CI width shrinks from ±0.089 at n=3 to ±0.045 at n=20.',
    limitations: 'If policy optimization is completely deterministic with zero variance, best-seed bias vanishes.'
  },
  {
    id: 'CLAIM-09',
    phase: 8,
    title: 'Factorial Process Reward × Reference KL Synergy',
    primaryMetric: 'Super-Additive Accuracy Gain Δ_AB - (Δ_A + Δ_B)',
    effectSize: 'Δ_synergy = +6.68%',
    status: 'robustly reproduced',
    evidenceStrength: 'Empirically demonstrated in controlled 2x2 factorial design',
    claim: 'Combining process reward and reference KL regularization produces a super-additive synergistic interaction (+6.7% over independent additive predictions).',
    evidence: 'Additive prediction: 13.76% + 4.00% = 17.76%. Observed joint gain: +24.44%. Synergistic interaction: +6.68% (95% CI: [+4.2%, +8.9%]).',
    experiment: 'experiments/phase8_ablation_and_interactions.py',
    seeds: '10 paired seeds',
    metricDetails: 'Process reward supplies dense direction while reference KL prevents policy drift away from natural syntax.',
    limitations: 'In environments with zero reward hacking loopholes, KL penalty can act as an unnecessary constraint.'
  },
  {
    id: 'CLAIM-10',
    phase: 10,
    title: 'Outcome-Only RL Induces a Severe Process Validity Gap',
    primaryMetric: 'Answer Accuracy minus Complete-Path Validity',
    effectSize: 'd = +10.92',
    cohensD: 10.92,
    status: 'robustly reproduced',
    evidenceStrength: 'Phase 10 Capstone Benchmark across 10 paired seeds',
    claim: 'Rewarding only final answer correctness in multi-step reasoning models improves answer accuracy while driving intermediate complete-path validity down due to shortcut and hallucination exploitation.',
    evidence: 'Outcome PPO achieves 87.8% answer accuracy but only 42.0% complete-path validity (process gap = 45.8%, 95% CI: [42.7%, 49.5%], d = 10.92).',
    experiment: 'experiments/phase10_capstone.py (Core Experiment A)',
    seeds: '10 paired seeds',
    metricDetails: 'Under OOD depth shift (d=4..6), Outcome PPO collapses to 16.7% accuracy, and to 19.9% when shortcuts are counterbalanced.',
    limitations: 'In single-step tasks where intermediate deductions are nonexistent, this gap does not manifest.'
  },
  {
    id: 'CLAIM-11',
    phase: 10,
    title: 'Process Supervision Enables OOD Depth and Counterfactual Generalization',
    primaryMetric: 'OOD Depth (d=4..6) & Counterfactual Accuracy',
    effectSize: 'd = +8.45',
    cohensD: 8.45,
    status: 'robustly reproduced',
    evidenceStrength: 'Phase 10 Capstone Benchmark across 10 paired seeds',
    claim: 'Rewarding intermediate valid steps produces policies that generalize to unseen chain lengths (d in {4, 5, 6}) and maintain causal responsiveness under single-edge counterfactual mutations.',
    evidence: 'Process PPO achieves 66.5% OOD depth accuracy vs 16.7% for Outcome PPO (+49.8%, d = 8.45), and 74.2% counterfactual consistency vs 23.1% for Outcome PPO.',
    experiment: 'experiments/phase10_capstone.py (Core Experiment B & Probes)',
    seeds: '10 paired seeds',
    metricDetails: 'Answer accuracy (84.7%) aligns tightly with complete-path validity (86.5%).',
    limitations: 'Requires fine-grained, verifiable step feedback during training.'
  },
  {
    id: 'CLAIM-12',
    phase: 10,
    title: 'Un-Gated Proxy Verifiers Trigger Catastrophic Goodhart Divergence',
    primaryMetric: 'Proxy Reward minus True Oracle Utility',
    effectSize: 'd = -22.40',
    cohensD: -22.40,
    status: 'robustly reproduced',
    evidenceStrength: 'Misspecified heuristic verifier ablation across 10 paired seeds',
    claim: 'Optimizing against an un-gated proxy verifier causes training proxy reward to surge while ground-truth oracle utility collapses.',
    evidence: 'Weak Proxy PPO achieves 94.5% proxy reward while true oracle utility drops to 25.7% (divergence gap +68.8%, exploit rate 64.9%, d = -22.40).',
    experiment: 'experiments/phase10_capstone.py (Core Experiment C)',
    seeds: '10 paired seeds',
    metricDetails: 'The optimizer rapidly learns to emit target character substrings while bypassing actual causal graph transitions.',
    limitations: 'Requires an imperfect verifier with exploitable blind spots.'
  },
  {
    id: 'CLAIM-13',
    phase: 10,
    title: 'Invariant Gating Suppresses Proxy Exploitation',
    primaryMetric: 'Exploit Rate Reduction under Flawed Verifiers',
    effectSize: 'd = +18.20',
    cohensD: 18.20,
    status: 'robustly reproduced',
    evidenceStrength: 'Structural invariant gate ablation across 10 paired seeds',
    claim: 'Requiring generated steps to satisfy topological structural invariants before granting proxy rewards suppresses learned exploits and keeps proxy reward aligned with oracle utility.',
    evidence: 'Invariant-Gated PPO achieves 88.5% proxy reward and 87.0% oracle utility, reducing exploit rate from 64.9% down to 1.3% (Δ = -63.6%, d = +18.20).',
    experiment: 'experiments/phase10_capstone.py (Core Experiment C)',
    seeds: '10 paired seeds',
    metricDetails: 'Structural checks enforce start-node matching and continuous edge connectivity in linear time O(|V|).',
    limitations: 'Requires domain-specific formal invariants that can be verified in linear or polynomial time.'
  },
  {
    id: 'CLAIM-14',
    phase: 10,
    title: 'Reference KL Regularization Mitigates Overconfidence',
    primaryMetric: 'Expected Calibration Error (ECE) Reduction',
    effectSize: 'd = -6.15',
    cohensD: -6.15,
    status: 'robustly reproduced',
    evidenceStrength: 'Reference KL regularization sweep across 10 paired seeds',
    claim: 'Penalizing KL divergence from the supervised reference model prevents entropy collapse and reduces Expected Calibration Error without degrading task accuracy.',
    evidence: 'Process PPO + KL achieves ECE of 0.081 vs 0.171 for Outcome PPO and maintains policy entropy at 1.00 vs 0.63 (d = -6.15).',
    experiment: 'experiments/phase10_capstone.py (Core Experiment E)',
    seeds: '10 paired seeds',
    metricDetails: 'Maintains 85.7% IID accuracy and boosts OOD depth accuracy to 72.1%.',
    limitations: 'Overly strong KL regularization (beta_kl > 0.20) slows policy convergence.'
  },
  {
    id: 'CLAIM-15',
    phase: 10,
    title: 'Verifier Disagreement Escalation Improves Audit Efficiency',
    primaryMetric: 'Exploit Catch Rate under 20% Oracle Query Budget',
    effectSize: 'd = +7.80',
    cohensD: 7.80,
    status: 'robustly reproduced',
    evidenceStrength: 'Selective audit routing evaluation across 10 paired seeds',
    claim: 'Escalating queries to an expensive oracle based on inter-verifier disagreement catches significantly more false-positive exploits per query than random auditing or policy entropy.',
    evidence: 'Under a 20% oracle query budget, disagreement routing caught 69.0% of exploits vs 20.0% for random auditing (3.4x efficiency gain, d = +7.80), and vs 48.0% for entropy routing.',
    experiment: 'experiments/phase10_capstone.py (Core Experiment F)',
    seeds: '10 paired seeds',
    metricDetails: 'Combined routing (Disagreement + Entropy) achieves an 82.5% exploit catch rate.',
    limitations: 'Requires ensemble or diverse verifier architectures with non-correlated error modes.'
  },
  {
    id: 'CLAIM-16',
    phase: 10,
    title: 'Test-Time Search Matches Policy RL Under 8x Token Cost',
    primaryMetric: 'SFT + Best-of-8 vs Single-Sample PPO Accuracy',
    effectSize: 'd = +1.10',
    cohensD: 1.10,
    status: 'budget-sensitive',
    evidenceStrength: 'Compute frontier Pareto evaluation across 10 paired seeds',
    claim: 'Best-of-N inference search on a frozen SFT policy can match the accuracy of single-sample PPO, demonstrating that apparent reasoning gains can be manufactured purely through test-time compute.',
    evidence: 'SFT + Best-of-8 achieves 88.5% oracle accuracy, matching greedy Process PPO (85.5%), but consumes 144 tokens/task vs 18 tokens/task (8x token cost).',
    experiment: 'experiments/phase10_capstone.py (Core Experiment D)',
    seeds: '10 paired seeds',
    metricDetails: 'Tokens per correct answer: Process PPO (21.1 tokens) vs SFT BoN-8 (162.7 tokens).',
    limitations: 'Search efficacy plateaus as N increases due to sample coverage limits and verifier noise.'
  }
];

// Helper to load committed experiment JSON files
export function loadCapstoneResults(): CapstoneMetrics {
  const root = getRootDir();
  const filePath = path.join(root, 'experiments/results/phase10_capstone.json');
  const raw = fs.readFileSync(filePath, 'utf-8');
  const data = JSON.parse(raw);
  return data.metrics as CapstoneMetrics;
}

export function loadClaimAuditResults() {
  const root = getRootDir();
  const filePath = path.join(root, 'experiments/results/phase8_research_claim_audit.json');
  const raw = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(raw);
}

export function loadSeedSensitivityResults() {
  const root = getRootDir();
  const filePath = path.join(root, 'experiments/results/phase8_seed_sensitivity.json');
  const raw = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(raw);
}

export function loadFactorialAblationResults() {
  const root = getRootDir();
  const filePath = path.join(root, 'experiments/results/phase8_ablation_and_interactions.json');
  const raw = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(raw);
}

// Read raw tutorial markdown
export function getTutorialMarkdown(slug: string): string {
  const meta = TUTORIALS_METADATA.find(t => t.slug === slug);
  if (!meta) {
    throw new Error(`Tutorial with slug "${slug}" not found.`);
  }
  const root = getRootDir();
  const fullPath = path.join(root, meta.filePath);
  return fs.readFileSync(fullPath, 'utf-8');
}

// Read raw doc markdown
export function getDocMarkdown(slug: string): string {
  const meta = DOCS_METADATA.find(d => d.slug === slug);
  if (!meta) {
    throw new Error(`Doc with slug "${slug}" not found.`);
  }
  const root = getRootDir();
  const fullPath = path.join(root, meta.filePath);
  return fs.readFileSync(fullPath, 'utf-8');
}
