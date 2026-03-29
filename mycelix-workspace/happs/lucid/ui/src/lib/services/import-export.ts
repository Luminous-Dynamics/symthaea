// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Import/Export Module
 *
 * Supports:
 * - Export: JSON, Markdown, JSON-LD
 * - Import: Obsidian, Roam Research, Plain Markdown, JSON
 */

import type { Thought, CreateThoughtInput } from '@mycelix/lucid-client';
import { ThoughtType, EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel } from '@mycelix/lucid-client';

// ============================================================================
// EXPORT FUNCTIONS
// ============================================================================

/**
 * Export thoughts as JSON
 */
export function exportAsJSON(thoughts: Thought[]): string {
  return JSON.stringify(thoughts, null, 2);
}

/**
 * Export thoughts as Markdown
 */
export function exportAsMarkdown(thoughts: Thought[]): string {
  const lines: string[] = [
    '# LUCID Knowledge Export',
    '',
    `*Exported: ${new Date().toISOString()}*`,
    `*Total thoughts: ${thoughts.length}*`,
    '',
    '---',
    '',
  ];

  // Group by domain
  const byDomain = new Map<string, Thought[]>();
  thoughts.forEach((t) => {
    const domain = t.domain || 'Uncategorized';
    if (!byDomain.has(domain)) byDomain.set(domain, []);
    byDomain.get(domain)!.push(t);
  });

  for (const [domain, domainThoughts] of byDomain) {
    lines.push(`## ${domain}`);
    lines.push('');

    for (const thought of domainThoughts) {
      const epistemic = `E${thought.epistemic.empirical.slice(1)}N${thought.epistemic.normative.slice(1)}M${thought.epistemic.materiality.slice(1)}H${thought.epistemic.harmonic.slice(1)}`;

      lines.push(`### ${thought.thought_type}`);
      lines.push('');
      lines.push(thought.content);
      lines.push('');
      lines.push(`- **Confidence:** ${Math.round(thought.confidence * 100)}%`);
      lines.push(`- **Epistemic:** \`${epistemic}\``);
      if (thought.tags?.length) {
        lines.push(`- **Tags:** ${thought.tags.map((t) => `#${t}`).join(' ')}`);
      }
      lines.push(`- **Created:** ${new Date(thought.created_at / 1000).toISOString()}`);
      lines.push('');
    }
  }

  return lines.join('\n');
}

/**
 * Export as JSON-LD (Linked Data format)
 */
export function exportAsJSONLD(thoughts: Thought[]): string {
  const jsonld = {
    '@context': {
      '@vocab': 'https://schema.org/',
      lucid: 'https://mycelix.org/lucid/',
      epistemic: 'lucid:epistemic',
      confidence: 'lucid:confidence',
      thoughtType: 'lucid:thoughtType',
    },
    '@graph': thoughts.map((t) => ({
      '@id': `lucid:thought/${t.id}`,
      '@type': 'CreativeWork',
      name: t.thought_type,
      text: t.content,
      'lucid:epistemic': {
        empirical: t.epistemic.empirical,
        normative: t.epistemic.normative,
        materiality: t.epistemic.materiality,
        harmonic: t.epistemic.harmonic,
      },
      'lucid:confidence': t.confidence,
      'lucid:thoughtType': t.thought_type,
      keywords: t.tags,
      about: t.domain,
      dateCreated: new Date(t.created_at / 1000).toISOString(),
      dateModified: new Date(t.updated_at / 1000).toISOString(),
    })),
  };

  return JSON.stringify(jsonld, null, 2);
}

/**
 * Trigger download of export file
 */
export function downloadExport(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================================
// IMPORT FUNCTIONS
// ============================================================================

export interface ImportResult {
  thoughts: CreateThoughtInput[];
  errors: string[];
  warnings: string[];
}

/**
 * Import from JSON (LUCID format)
 */
export function importFromJSON(content: string): ImportResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const thoughts: CreateThoughtInput[] = [];

  try {
    const data = JSON.parse(content);
    const items = Array.isArray(data) ? data : [data];

    for (const item of items) {
      if (!item.content) {
        errors.push(`Skipped item: missing content`);
        continue;
      }

      thoughts.push({
        content: item.content,
        thought_type: item.thought_type || item.thoughtType || 'Note',
        confidence: item.confidence ?? 0.7,
        tags: item.tags || [],
        domain: item.domain,
        epistemic: item.epistemic || {
          empirical: EmpiricalLevel.E2,
          normative: NormativeLevel.N0,
          materiality: MaterialityLevel.M2,
          harmonic: HarmonicLevel.H2,
        },
      });
    }
  } catch (e) {
    errors.push(`JSON parse error: ${e}`);
  }

  return { thoughts, errors, warnings };
}

/**
 * Import from Obsidian Markdown
 * Parses frontmatter YAML and wikilinks
 */
export function importFromObsidian(content: string, filename?: string): ImportResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const thoughts: CreateThoughtInput[] = [];

  // Parse frontmatter
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  let frontmatter: Record<string, any> = {};
  let body = content;

  if (frontmatterMatch) {
    try {
      // Simple YAML parsing (key: value format)
      frontmatterMatch[1].split('\n').forEach((line) => {
        const match = line.match(/^(\w+):\s*(.+)$/);
        if (match) {
          let value: any = match[2].trim();
          // Parse arrays
          if (value.startsWith('[') && value.endsWith(']')) {
            value = value.slice(1, -1).split(',').map((s: string) => s.trim().replace(/["']/g, ''));
          }
          frontmatter[match[1]] = value;
        }
      });
      body = frontmatterMatch[2];
    } catch (e) {
      warnings.push('Failed to parse frontmatter');
    }
  }

  // Extract tags from content (both #tag and frontmatter)
  const hashTags = body.match(/#(\w+)/g)?.map((t) => t.slice(1)) || [];
  const fmTags = Array.isArray(frontmatter.tags) ? frontmatter.tags : [];
  const tags = [...new Set([...hashTags, ...fmTags])];

  // Convert wikilinks to plain text for now
  body = body.replace(/\[\[([^\]|]+)(\|[^\]]+)?\]\]/g, (_, link, alias) => alias ? alias.slice(1) : link);

  // Split by headings or paragraphs
  const sections = body.split(/\n#{1,6}\s+/).filter((s) => s.trim());

  if (sections.length === 0) {
    sections.push(body);
  }

  for (const section of sections) {
    const trimmed = section.trim();
    if (!trimmed || trimmed.length < 10) continue;

    // Detect thought type from content
    let thoughtType: string = 'Note';
    if (trimmed.includes('?')) thoughtType = 'Question';
    else if (frontmatter.type) thoughtType = frontmatter.type;

    thoughts.push({
      content: trimmed,
      thought_type: thoughtType as any,
      confidence: frontmatter.confidence ?? 0.7,
      tags,
      domain: frontmatter.domain || filename?.replace('.md', ''),
      epistemic: {
        empirical: frontmatter.empirical || 'E2',
        normative: frontmatter.normative || 'N0',
        materiality: frontmatter.materiality || 'M2',
        harmonic: frontmatter.harmonic || 'H2',
      },
    });
  }

  if (thoughts.length === 0) {
    warnings.push('No thoughts extracted from file');
  }

  return { thoughts, errors, warnings };
}

/**
 * Import from Roam Research JSON export
 */
export function importFromRoam(content: string): ImportResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const thoughts: CreateThoughtInput[] = [];

  try {
    const data = JSON.parse(content);

    function processBlock(block: any, parentTags: string[] = []): void {
      if (!block.string || block.string.trim().length < 5) return;

      // Extract tags from [[links]] and #tags
      const wikiLinks = block.string.match(/\[\[([^\]]+)\]\]/g)?.map((l: string) => l.slice(2, -2)) || [];
      const hashTags = block.string.match(/#(\w+)/g)?.map((t: string) => t.slice(1)) || [];
      const tags = [...new Set([...parentTags, ...wikiLinks.slice(0, 3), ...hashTags])];

      // Clean content
      let content = block.string
        .replace(/\[\[([^\]]+)\]\]/g, '$1')
        .replace(/#(\w+)/g, '')
        .trim();

      if (content.length > 10) {
        let thoughtType: string = 'Note';
        if (content.includes('?')) thoughtType = 'Question';
        else if (content.startsWith('TODO')) thoughtType = 'Goal';
        else if (content.startsWith('-')) thoughtType = 'Observation';

        thoughts.push({
          content,
          thought_type: thoughtType as any,
          confidence: 0.7,
          tags: tags.slice(0, 5),
          epistemic: {
            empirical: EmpiricalLevel.E2,
            normative: NormativeLevel.N0,
            materiality: MaterialityLevel.M2,
            harmonic: HarmonicLevel.H2,
          },
        });
      }

      // Process children recursively
      if (block.children) {
        block.children.forEach((child: any) => processBlock(child, tags));
      }
    }

    // Process all pages
    (Array.isArray(data) ? data : [data]).forEach((page: any) => {
      const pageTags = page.title ? [page.title] : [];
      if (page.children) {
        page.children.forEach((block: any) => processBlock(block, pageTags));
      }
    });
  } catch (e) {
    errors.push(`Roam JSON parse error: ${e}`);
  }

  return { thoughts, errors, warnings };
}

/**
 * Import from plain Markdown (one thought per paragraph or heading section)
 */
export function importFromMarkdown(content: string): ImportResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const thoughts: CreateThoughtInput[] = [];

  // Extract all tags
  const allTags = content.match(/#(\w+)/g)?.map((t) => t.slice(1)) || [];

  // Split by double newlines or headings
  const sections = content.split(/\n\n+|\n(?=#+\s)/).filter((s) => s.trim());

  for (const section of sections) {
    let trimmed = section.trim();

    // Skip if too short
    if (trimmed.length < 15) continue;

    // Extract heading if present
    const headingMatch = trimmed.match(/^#+\s+(.+)\n([\s\S]*)$/);
    if (headingMatch) {
      trimmed = headingMatch[2].trim() || headingMatch[1];
    }

    // Extract section-specific tags
    const sectionTags = trimmed.match(/#(\w+)/g)?.map((t) => t.slice(1)) || [];

    // Clean content
    const cleanContent = trimmed.replace(/#\w+/g, '').trim();

    if (cleanContent.length < 10) continue;

    let thoughtType: string = 'Note';
    if (cleanContent.includes('?')) thoughtType = 'Question';
    else if (cleanContent.startsWith('>')) thoughtType = 'Quote';
    else if (cleanContent.match(/^\d+\./)) thoughtType = 'Plan';

    thoughts.push({
      content: cleanContent,
      thought_type: thoughtType as any,
      confidence: 0.7,
      tags: [...new Set(sectionTags)].slice(0, 5),
      epistemic: {
        empirical: EmpiricalLevel.E2,
        normative: NormativeLevel.N0,
        materiality: MaterialityLevel.M2,
        harmonic: HarmonicLevel.H2,
      },
    });
  }

  return { thoughts, errors, warnings };
}

/**
 * Auto-detect format and import
 */
export function autoImport(content: string, filename?: string): ImportResult {
  // Try to detect format
  const trimmed = content.trim();

  // JSON (Roam or LUCID)
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    try {
      const data = JSON.parse(trimmed);
      // Check if it's Roam format (has 'children' arrays)
      const isRoam = Array.isArray(data) && data[0]?.children;
      if (isRoam) {
        return importFromRoam(content);
      }
      return importFromJSON(content);
    } catch {
      // Not valid JSON, treat as markdown
    }
  }

  // Obsidian (has frontmatter)
  if (trimmed.startsWith('---')) {
    return importFromObsidian(content, filename);
  }

  // Plain markdown
  return importFromMarkdown(content);
}

// ============================================================================
// WEB CLIPPER
// ============================================================================

/**
 * Generate bookmarklet code for web clipping
 */
export function generateBookmarklet(apiEndpoint: string): string {
  const code = `
    (function() {
      var selection = window.getSelection().toString().trim();
      var content = selection || document.title;
      var url = window.location.href;
      var thought = {
        content: content + '\\n\\nSource: ' + url,
        thought_type: 'Quote',
        confidence: 0.7,
        tags: ['web-clip'],
        source_url: url,
        source_title: document.title
      };
      fetch('${apiEndpoint}/clip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(thought)
      }).then(function() {
        alert('Clipped to LUCID!');
      }).catch(function(e) {
        prompt('Copy this to import manually:', JSON.stringify(thought));
      });
    })();
  `.replace(/\s+/g, ' ').trim();

  return `javascript:${encodeURIComponent(code)}`;
}

/**
 * Parse web clip from clipboard or dropped content
 *
 * SECURITY: Uses DOMParser instead of innerHTML to prevent XSS attacks.
 * DOMParser does not execute scripts during parsing.
 */
export function parseWebClip(html: string): CreateThoughtInput | null {
  // Use DOMParser for safe HTML parsing (does not execute scripts)
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  const div = doc.body;

  // Remove scripts and styles as extra safety measure
  div.querySelectorAll('script, style, iframe, object, embed').forEach((el) => el.remove());

  const text = div.textContent?.trim();
  if (!text || text.length < 10) return null;

  // Try to extract URL
  const link = div.querySelector('a[href]');
  const url = link?.getAttribute('href');

  return {
    content: text.slice(0, 2000) + (url ? `\n\nSource: ${url}` : ''),
    thought_type: ThoughtType.Quote,
    confidence: 0.5,
    tags: ['web-clip'],
    epistemic: {
      empirical: EmpiricalLevel.E1,
      normative: NormativeLevel.N0,
      materiality: MaterialityLevel.M2,
      harmonic: HarmonicLevel.H2,
    },
  };
}
