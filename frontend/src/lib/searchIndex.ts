import { TUTORIALS_METADATA, DOCS_METADATA, RESEARCH_CLAIMS } from './data';

export interface SearchEntry {
  id: string;
  type: 'tutorial' | 'doc' | 'claim' | 'finding';
  title: string;
  subtitle: string;
  url: string;
  badge?: string;
  keywords: string[];
}

export function buildSearchIndex(): SearchEntry[] {
  const entries: SearchEntry[] = [];

  // Index tutorials
  for (const t of TUTORIALS_METADATA) {
    entries.push({
      id: `tut-${t.slug}`,
      type: 'tutorial',
      title: `${t.number}: ${t.title}`,
      subtitle: t.subtitle,
      url: `/tutorials/${t.slug}`,
      badge: `Phase ${t.phase}`,
      keywords: [
        'tutorial',
        `phase ${t.phase}`,
        t.title.toLowerCase(),
        t.subtitle.toLowerCase(),
        ...t.description.toLowerCase().split(/\s+/)
      ]
    });
  }

  // Index docs
  for (const d of DOCS_METADATA) {
    entries.push({
      id: `doc-${d.slug}`,
      type: 'doc',
      title: d.title,
      subtitle: d.subtitle,
      url: `/docs/${d.slug}`,
      badge: d.category.toUpperCase(),
      keywords: [
        'doc',
        'report',
        'paper',
        d.title.toLowerCase(),
        d.subtitle.toLowerCase(),
        d.category.toLowerCase()
      ]
    });
  }

  // Index claims
  for (const c of RESEARCH_CLAIMS) {
    entries.push({
      id: `claim-${c.id}`,
      type: 'claim',
      title: `${c.id}: ${c.title}`,
      subtitle: `${c.primaryMetric} (${c.effectSize}) • Status: ${c.status}`,
      url: `/docs/claims#${c.id.toLowerCase()}`,
      badge: c.status,
      keywords: [
        'claim',
        c.id.toLowerCase(),
        c.title.toLowerCase(),
        c.primaryMetric.toLowerCase(),
        c.status.toLowerCase(),
        c.claim.toLowerCase()
      ]
    });
  }

  // Key findings shortcuts
  entries.push(
    {
      id: 'finding-process-gap',
      type: 'finding',
      title: 'Process Validity Gap (45.8%)',
      subtitle: 'Outcome PPO achieves 87.8% accuracy but only 42.0% complete-path validity',
      url: '/results#process-gap',
      badge: 'Finding',
      keywords: ['process gap', 'outcome ppo', 'validity', 'shortcut', 'hallucination']
    },
    {
      id: 'finding-goodhart',
      type: 'finding',
      title: 'Goodhart Divergence (+68.8%)',
      subtitle: 'Weak proxy verifier yields 94.5% proxy reward vs 25.7% true oracle utility',
      url: '/results#goodhart',
      badge: 'Finding',
      keywords: ['goodhart', 'proxy reward', 'oracle utility', 'exploit', 'invariant gate']
    },
    {
      id: 'finding-search',
      type: 'finding',
      title: 'Test-Time Search vs Policy RL (BoN-8)',
      subtitle: 'SFT + Best-of-8 matches Process PPO at 8x token expenditure',
      url: '/results#compute-frontier',
      badge: 'Finding',
      keywords: ['best of n', 'bon', 'search', 'compute frontier', 'token cost']
    },
    {
      id: 'demo-graph-reasoning',
      type: 'finding',
      title: 'Interactive Demo: Valid Graph Reasoning vs Shortcuts',
      subtitle: 'Step-by-step DAG traversal and counterfactual premise mutation simulator',
      url: '/demos#graph-reasoning',
      badge: 'Demo',
      keywords: ['interactive', 'demo', 'graph', 'dag', 'simulation', 'shortcut']
    },
    {
      id: 'demo-goodhart',
      type: 'finding',
      title: 'Interactive Demo: Goodhart Divergence & Invariant Gating',
      subtitle: 'Visual simulation of proxy optimization and topological gate protection',
      url: '/demos#goodhart-divergence',
      badge: 'Demo',
      keywords: ['interactive', 'demo', 'goodhart', 'proxy', 'invariant gate']
    }
  );

  return entries;
}
