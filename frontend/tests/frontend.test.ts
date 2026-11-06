import { describe, it, expect } from 'vitest';
import {
  TUTORIALS_METADATA,
  DOCS_METADATA,
  RESEARCH_CLAIMS,
  getTutorialMarkdown,
  getDocMarkdown,
  loadCapstoneResults,
  loadClaimAuditResults
} from '../src/lib/data';
import { processMarkdown, slugify } from '../src/lib/markdown';
import { buildSearchIndex } from '../src/lib/searchIndex';

describe('Tutorials Curriculum Verification', () => {
  it('should define all 9 tutorials with sequential phases', () => {
    expect(TUTORIALS_METADATA).toHaveLength(9);
    for (let i = 0; i < 9; i++) {
      expect(TUTORIALS_METADATA[i].phase).toBe(i + 1);
      expect(TUTORIALS_METADATA[i].slug).toBeTruthy();
      expect(TUTORIALS_METADATA[i].title).toBeTruthy();
    }
  });

  it('should successfully load raw markdown for all 9 tutorials from disk without duplication', () => {
    for (const tut of TUTORIALS_METADATA) {
      const raw = getTutorialMarkdown(tut.slug);
      expect(raw).toBeTruthy();
      expect(raw.length).toBeGreaterThan(500);
      expect(raw).toMatch(/^#\s+/m);
    }
  });

  it('should process tutorial markdown into HTML with KaTeX and headings', async () => {
    const raw = getTutorialMarkdown('01_rl_first_principles');
    const result = await processMarkdown(raw);

    expect(result.html).toBeTruthy();
    expect(result.html).toContain('katex');
    expect(result.toc.length).toBeGreaterThan(3);
    expect(result.title).toContain('Tutorial 01: Reinforcement Learning from First Principles');
  });
});

describe('Research Documentation Verification', () => {
  it('should define all 10 core research documents', () => {
    expect(DOCS_METADATA).toHaveLength(10);
    const slugs = DOCS_METADATA.map(d => d.slug);
    expect(slugs).toContain('phase10_capstone_report');
    expect(slugs).toContain('claims');
    expect(slugs).toContain('research_summary');
    expect(slugs).toContain('phase10_hypotheses');
  });

  it('should successfully load raw markdown for all 10 research docs from disk', () => {
    for (const doc of DOCS_METADATA) {
      const raw = getDocMarkdown(doc.slug);
      expect(raw).toBeTruthy();
      expect(raw.length).toBeGreaterThan(200);
    }
  });
});

describe('Committed Research Benchmark Artifacts Verification', () => {
  it('should read phase10_capstone.json and accurately match audited metrics', () => {
    const capstone = loadCapstoneResults();
    expect(capstone).toBeDefined();

    // Verify key hypothesis values
    expect(capstone.h1_outcome_process_gap).toBeCloseTo(0.4678, 3);
    expect(capstone.h2_process_path_gain).toBeCloseTo(0.4504, 3);
    expect(capstone.h3_weak_proxy_divergence).toBeCloseTo(0.6650, 3);

    // Verify distribution matrix has 8 treatments with 6 splits
    const models = Object.keys(capstone.distribution_matrix);
    expect(models).toHaveLength(8);
    expect(models).toContain('Outcome_PPO');
    expect(models).toContain('Process_PPO');
    expect(models).toContain('SFT_Baseline');

    for (const m of models) {
      const splits = capstone.distribution_matrix[m];
      expect(splits.iid).toBeGreaterThan(0);
      expect(splits.ood_depth).toBeGreaterThan(0);
      expect(splits.counterfactual).toBeGreaterThan(0);
    }

    // Verify depth extrapolation has 4 models with 6 depths
    const depthModels = Object.keys(capstone.depth_extrapolation);
    expect(depthModels).toHaveLength(4);
    for (const dm of depthModels) {
      expect(capstone.depth_extrapolation[dm]).toHaveLength(6);
    }

    // Verify calibration results
    expect(Object.keys(capstone.calibration_results)).toHaveLength(4);

    // Verify routing strategies
    expect(capstone.routing_results).toHaveLength(5);
  });

  it('should load phase8_research_claim_audit.json without error', () => {
    const audit = loadClaimAuditResults();
    expect(audit).toBeDefined();
    expect(audit.metrics.total_claims_audited).toBe(7);
  });
});

describe('Canonical Claim Registry Verification', () => {
  it('should register all 16 claims (CLAIM-01 through CLAIM-16)', () => {
    expect(RESEARCH_CLAIMS).toHaveLength(16);

    const ids = RESEARCH_CLAIMS.map(c => c.id);
    for (let i = 1; i <= 16; i++) {
      const expectedId = `CLAIM-${String(i).padStart(2, '0')}`;
      expect(ids).toContain(expectedId);
    }
  });

  it('should have complete metadata for every claim', () => {
    for (const claim of RESEARCH_CLAIMS) {
      expect(claim.id).toBeTruthy();
      expect(claim.title).toBeTruthy();
      expect(claim.primaryMetric).toBeTruthy();
      expect(claim.effectSize).toBeTruthy();
      expect(claim.status).toBeTruthy();
      expect(claim.claim).toBeTruthy();
      expect(claim.evidence).toBeTruthy();
      expect(claim.limitations).toBeTruthy();
    }
  });
});

describe('Search Index Verification', () => {
  it('should index tutorials, docs, claims, and interactive demos', () => {
    const index = buildSearchIndex();
    expect(index.length).toBeGreaterThan(35);

    const types = new Set(index.map(i => i.type));
    expect(types.has('tutorial')).toBe(true);
    expect(types.has('doc')).toBe(true);
    expect(types.has('claim')).toBe(true);
    expect(types.has('finding')).toBe(true);
  });
});

describe('Slugification Utility', () => {
  it('should generate clean URL slugs from heading strings', () => {
    expect(slugify('Section 1: The Intuition')).toBe('section-1-the-intuition');
    expect(slugify('H1 (Outcome-Only Process Gap)')).toBe('h1-outcome-only-process-gap');
  });
});
