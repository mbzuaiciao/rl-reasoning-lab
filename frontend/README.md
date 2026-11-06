# RL Reasoning Lab — Research Frontend

An Apple-inspired, responsive educational and research-reading frontend for `rl-reasoning-lab`.

The frontend presents the repository's first-principles reinforcement learning curriculum and audited Phase 10 empirical findings through an interactive static interface with zero backend services, model APIs, or databases.

---

## Architecture & Technology Stack

- **Framework**: [Astro 5](https://astro.build/) (Static Site Generation / SSG)
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom Apple-inspired design tokens (restrained typography, delicate glassmorphic blurs, accessible light/dark themes)
- **Math Rendering**: KaTeX via `remark-math` and `rehype-katex`
- **Syntax Highlighting**: Shiki dual-theme (`github-light` / `github-dark`)
- **Testing**: Vitest for build-time contract and content integrity verification

---

## Experiences

1. **Research Home (`/`)**:
   - Central scientific question: genuine multi-step reasoning improvement versus reward exploitation.
   - Core distinction: terminal answer accuracy ($87.8\%$) versus complete-path validity ($42.0\%$).
   - Headline findings with confidence intervals ($n=10$) and Cohen's $d$.
   - Methodological boundaries prominently displayed (2-layer transformer, $|V|=8$ DAGs, observable trace assumption, step verifier availability).

2. **Interactive Curriculum (`/tutorials` and `/tutorials/[slug]`)**:
   - Complete build-time reading interface for all 9 tutorials.
   - Table of contents with scroll-spy, KaTeX mathematical notation, and Shiki code highlighting.
   - Renders markdown directly from root repository files without duplication.

3. **Research Documentation (`/docs` and `/docs/[slug]`)**:
   - Access to all Phase 10 research papers, executive summary, hypotheses (H1–H7), failure taxonomy, and case studies.

4. **Results Explorer (`/results`)**:
   - Visual inspection grounded strictly in committed artifacts (`experiments/results/phase10_capstone.json` and `phase8_research_claim_audit.json`).
   - 7 interactive explorer views:
     - Process Gap ($45.8\%$, $d = 10.92$) & Core RL
     - Distribution Shifts Matrix ($8 \times 6$ treatments across shifts)
     - Depth Extrapolation Curves ($d=1$ to $d=6$)
     - Goodhart Divergence ($+68.8\%$ gap) & Invariant Gating ($1.3\%$ exploits)
     - Uncertainty, Calibration (ECE), and Selective Auditing
     - Compute Frontier (Test-Time Search BoN-8 vs Policy RL)
     - Canonical Claim Registry (16 audited claims)

5. **Interactive Teaching Demonstrations (`/demos`)**:
   - **Demo 1**: Multi-Step Causal Path vs Shortcut Exploitation with interactive premise mutations.
   - **Demo 2**: Goodhart Divergence & Invariant Gating Visualizer with live optimization slider.
   - Clearly marked as pedagogical simulations.

---

## Development & Build Commands

All commands are executed from the `frontend/` directory:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Run unit and contract tests (11 passing tests)
npm test

# Run Astro TypeScript typechecker
npm run check

# Build production static site (dist/)
npm run build

# Preview production build locally
npm run preview
```

---

## Directory Structure

```text
frontend/
├── astro.config.mjs         # Astro SSG configuration
├── tsconfig.json            # TypeScript configuration
├── tailwind.config.mjs      # Apple-inspired typography & palette tokens
├── package.json             # Scripts & dependencies
├── README.md                # Frontend documentation
├── screenshots/             # Captured desktop & mobile validation screenshots
├── src/
│   ├── components/
│   │   ├── Header.astro     # Sticky glassmorphic navigation
│   │   ├── Footer.astro     # Minimalist footer with scientific scope
│   │   ├── ThemeToggle.astro# Light/dark toggle with localStorage persistence
│   │   ├── SearchModal.astro# Client search dialog (⌘K)
│   │   ├── TableOfContents.astro # Sticky TOC with scroll-spy
│   │   ├── MetricCard.astro # Apple statistic card with Cohen's d & CIs
│   │   ├── LimitationBanner.astro # 4 core methodological boundaries
│   │   └── demos/           # Interactive pedagogical visualizers
│   ├── layouts/
│   │   └── BaseLayout.astro # Base HTML shell with FOUC prevention
│   ├── lib/
│   │   ├── data.ts          # Reads committed JSON & markdown at build time
│   │   ├── markdown.ts      # Unified / Remark / Rehype pipeline
│   │   ├── searchIndex.ts   # Pre-indexes tutorials, docs, and claims
│   │   └── types.ts         # TypeScript interfaces
│   ├── pages/
│   │   ├── index.astro      # Research home
│   │   ├── tutorials/       # All 9 tutorials
│   │   ├── docs/            # All 10 research docs
│   │   ├── results.astro    # Results explorer
│   │   ├── demos.astro      # Interactive teaching demos
│   │   └── api/search.json.ts # Statically generated search index
│   └── styles/
│       └── global.css       # KaTeX CSS, typography, prose styles
└── tests/
    └── frontend.test.ts     # Vitest tests verifying 9 tutorials, 10 docs, 16 claims, and capstone metrics
```
