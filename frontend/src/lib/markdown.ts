import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeShiki from '@shikijs/rehype';
import rehypeStringify from 'rehype-stringify';
import { visit } from 'unist-util-visit';
import type { ProcessedMarkdown, TableOfContentsItem } from './types';

// Slug generator matching standard GitHub/rehype-slug convention
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-');
}

// Rewrite relative repo markdown links to web routes
function rewriteLinkHref(href: string): string {
  if (!href) return href;
  if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('#') || href.startsWith('mailto:')) {
    return href;
  }
  const [base, anchor] = href.split('#');
  const hash = anchor ? `#${anchor}` : '';

  // Match tutorial files
  const tutMatch = base.match(/(?:tutorials\/)?(0[1-9]_[a-zA-Z0-9_]+)\.md$/);
  if (tutMatch) {
    return `/tutorials/${tutMatch[1]}${hash}`;
  }

  // Match docs files
  const docMatch = base.match(/(?:docs\/)?([a-zA-Z0-9_]+)\.md$/);
  if (docMatch) {
    return `/docs/${docMatch[1]}${hash}`;
  }

  // Match experiments / results json files
  if (base.includes('.json')) {
    return `/results${hash}`;
  }

  return href;
}

// Remark plugin to rewrite links and extract document title & TOC
function remarkStructurePlugin(meta: { toc: TableOfContentsItem[]; title: string }) {
  return (tree: any) => {
    visit(tree, 'heading', (node: any) => {
      // Extract text content from heading
      const textParts: string[] = [];
      visit(node, 'text', (t: any) => {
        textParts.push(t.value);
      });
      visit(node, 'inlineCode', (c: any) => {
        textParts.push(c.value);
      });
      const headingText = textParts.join('').trim();

      if (node.depth === 1 && !meta.title) {
        meta.title = headingText;
      }

      if (node.depth >= 2 && node.depth <= 3) {
        meta.toc.push({
          id: slugify(headingText),
          text: headingText,
          depth: node.depth
        });
      }
    });

    visit(tree, 'link', (node: any) => {
      if (node.url) {
        node.url = rewriteLinkHref(node.url);
      }
    });
  };
}

// Custom rehype plugin to add external link security attributes
function rehypeExternalLinksPlugin() {
  return (tree: any) => {
    visit(tree, 'element', (node: any) => {
      if (node.tagName === 'a' && node.properties?.href) {
        const href = String(node.properties.href);
        if (href.startsWith('http://') || href.startsWith('https://')) {
          node.properties.target = '_blank';
          node.properties.rel = 'noopener noreferrer';
        }
      }
    });
  };
}

export async function processMarkdown(raw: string): Promise<ProcessedMarkdown> {
  const meta = { toc: [] as TableOfContentsItem[], title: '' };

  const processor = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkStructurePlugin, meta)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeKatex)
    .use(rehypeSlug)
    .use(rehypeExternalLinksPlugin)
    .use(rehypeShiki, {
      themes: {
        light: 'github-light',
        dark: 'github-dark'
      }
    })
    .use(rehypeStringify, { allowDangerousHtml: true });

  const result = await processor.process(raw);
  const html = String(result);

  return {
    html,
    toc: meta.toc,
    title: meta.title || 'RL Reasoning Lab',
    raw
  };
}
