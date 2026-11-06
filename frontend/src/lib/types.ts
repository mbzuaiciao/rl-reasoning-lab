export interface TutorialMeta {
  slug: string;
  phase: number;
  number: string;
  title: string;
  subtitle: string;
  description: string;
  readingTimeMinutes: number;
  filePath: string;
}

export interface DocMeta {
  slug: string;
  title: string;
  subtitle: string;
  category: 'capstone' | 'claims' | 'methodology' | 'audit' | 'future';
  readingTimeMinutes: number;
  filePath: string;
}

export interface TableOfContentsItem {
  id: string;
  text: string;
  depth: number;
}

export interface ProcessedMarkdown {
  html: string;
  toc: TableOfContentsItem[];
  title: string;
  raw: string;
}

export interface ResearchClaim {
  id: string;
  phase: number;
  title: string;
  primaryMetric: string;
  effectSize: string;
  cohensD?: number;
  status: 'robustly reproduced' | 'reproduced with smaller effect' | 'seed-sensitive' | 'budget-sensitive' | 'OOD-fragile' | 'inconclusive' | 'failed replication';
  evidenceStrength: string;
  claim: string;
  evidence: string;
  experiment: string;
  seeds: string;
  metricDetails: string;
  limitations: string;
}

export interface CapstoneMetrics {
  h1_outcome_process_gap: number;
  h1_gap_ci_lower: number;
  h1_gap_ci_upper: number;
  h2_process_path_gain: number;
  h2_path_ci_lower: number;
  h2_path_ci_upper: number;
  h2_cohens_d: number;
  h3_weak_proxy_divergence: number;
  h3_inv_proxy_divergence: number;
  h4_kl_ece_reduction: number;
  h5_routing_catch_ratio: number;
  h6_rs_ppo_diff: number;
  h7_bon8_ppo_diff: number;
  distribution_matrix: Record<string, {
    iid: number;
    ood_depth: number;
    distractors: number;
    counterfactual: number;
    new_symbols: number;
    shortcut_cb: number;
  }>;
  compute_frontier: Array<{
    method: string;
    tokens_per_task: number;
    oracle_accuracy: number;
    tokens_per_correct: number;
  }>;
  calibration_results: Record<string, {
    accuracy: number;
    ece: number;
    brier: number;
    entropy: number;
    selective_risk_80: number;
  }>;
  routing_results: Array<{
    strategy: string;
    oracle_budget: number;
    exploit_catch_rate: number;
    final_oracle_utility: number;
  }>;
  depth_extrapolation: Record<string, number[]>;
  replication: {
    rep_process_gain_mean: number;
    rep_process_gain_ci: [number, number];
    rep_goodhart_gap_mean: number;
    rep_goodhart_gap_ci: [number, number];
  };
}
