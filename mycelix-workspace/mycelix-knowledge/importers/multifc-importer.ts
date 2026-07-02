#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * MultiFC Dataset → Mycelix Claims Importer
 *
 * Parses the MultiFC (Multi-domain Fact Checking) dataset and converts
 * fact-checked claims to Mycelix Claims with verdict-based E/N/M classification.
 *
 * MultiFC paper: Augenstein et al. (2019) "MultiFC: A Real-World Multi-Domain
 *   Dataset for Evidence-Based Fact Checking of Claims"
 * Dataset: https://copenlu.github.io/publication/2019_emnlp_augenstein/
 * License: CC-BY 4.0
 *
 * Verdicts mapped to E levels:
 *   true / mostly-true / correct       → E3 (replicated/verified)
 *   half-true / mixture / unproven      → E1 (preliminary)
 *   mostly-false / false / pants-fire   → E0 (unverified/refuted)
 *   other / undetermined                → E1 (preliminary)
 *
 * Usage:
 *   npx ts-node importers/multifc-importer.ts <path-to-multifc.tsv> [--limit N]
 *
 * The MultiFC dataset is a TSV file with columns:
 *   claim_id, claim, label, claim_url, reason, categories, speaker,
 *   checker, tags, article_title, publish_date, claim_date
 *
 * Output:
 *   seed-data/claims/multifc-checked.json
 */

import * as fs from 'fs';
import * as readline from 'readline';
import {
  buildClaim,
  filterValid,
  writeSeedFile,
  enm,
  type Claim,
  type ClaimType,
  type EpistemicPosition,
} from './common';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_LIMIT = 5000;

const multifcPath = process.argv[2];
const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

if (!multifcPath || multifcPath.startsWith('--')) {
  console.error('Usage: npx ts-node importers/multifc-importer.ts <path-to-multifc.tsv> [--limit N]');
  console.error();
  console.error('Download MultiFC from: https://copenlu.github.io/publication/2019_emnlp_augenstein/');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Verdict → E/N/M mapping
// ---------------------------------------------------------------------------

/** Normalize raw verdict strings to canonical categories */
function normalizeVerdict(raw: string): 'true' | 'mixed' | 'false' | 'unknown' {
  const lower = raw.trim().toLowerCase().replace(/[^a-z-]/g, '');

  // True-ish
  if (['true', 'mostlytrue', 'correct', 'verified', 'accurate'].includes(lower)) {
    return 'true';
  }

  // Mixed
  if (['halftrue', 'mixture', 'halfflip', 'unproven', 'misleading', 'exaggerated', 'distorts'].includes(lower)) {
    return 'mixed';
  }

  // False-ish
  if (['false', 'mostlyfalse', 'pantsfire', 'pants-fire', 'incorrect', 'wrong', 'fabricated', 'fake'].includes(lower)) {
    return 'false';
  }

  return 'unknown';
}

function verdictClassification(verdict: 'true' | 'mixed' | 'false' | 'unknown'): EpistemicPosition {
  switch (verdict) {
    case 'true':
      return enm('E3', 'N0', 'M1');
    case 'mixed':
      return enm('E1', 'N0', 'M1');
    case 'false':
      return enm('E0', 'N0', 'M0');
    case 'unknown':
      return enm('E1', 'N0', 'M0');
  }
}

function verdictConfidence(verdict: 'true' | 'mixed' | 'false' | 'unknown'): number {
  switch (verdict) {
    case 'true': return 0.82;
    case 'mixed': return 0.50;
    case 'false': return 0.82;
    case 'unknown': return 0.30;
  }
}

function verdictTags(verdict: 'true' | 'mixed' | 'false' | 'unknown', categories?: string): string[] {
  const base = ['multifc', 'fact-check'];
  switch (verdict) {
    case 'true': base.push('verified-true'); break;
    case 'mixed': base.push('mixed-verdict'); break;
    case 'false': base.push('verified-false'); break;
    case 'unknown': base.push('unresolved'); break;
  }
  // Add category tags if available
  if (categories) {
    for (const cat of categories.split(',').slice(0, 3)) {
      const cleaned = cat.trim().toLowerCase().replace(/\s+/g, '-');
      if (cleaned && cleaned.length > 2) base.push(cleaned);
    }
  }
  return base;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

async function parseMultiFcTsv(filepath: string, maxClaims: number): Promise<Claim[]> {
  const claims: Claim[] = [];

  const fileStream = fs.createReadStream(filepath, { encoding: 'utf-8' });
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity,
  });

  let lineNum = 0;
  let parsed = 0;
  let skipped = 0;
  let isHeader = true;

  for await (const line of rl) {
    lineNum++;
    if (parsed >= maxClaims) break;

    // Skip header
    if (isHeader) {
      isHeader = false;
      continue;
    }

    const cols = line.split('\t');
    if (cols.length < 3) {
      skipped++;
      continue;
    }

    const claimText = cols[1]?.trim();
    const rawLabel = cols[2]?.trim();
    const claimUrl = cols[3]?.trim() || '';
    const categories = cols[5]?.trim() || '';

    if (!claimText || !rawLabel || claimText.length < 10) {
      skipped++;
      continue;
    }

    const verdict = normalizeVerdict(rawLabel);

    claims.push(
      buildClaim({
        source: 'multifc',
        content: claimText,
        classification: verdictClassification(verdict),
        claimType: 'Fact' as ClaimType,
        confidence: verdictConfidence(verdict),
        sources: claimUrl ? [claimUrl] : ['https://copenlu.github.io/publication/2019_emnlp_augenstein/'],
        tags: verdictTags(verdict, categories),
      }),
    );

    parsed++;

    if (parsed % 2000 === 0) {
      console.log(`  Parsed ${parsed} claims (line ${lineNum})...`);
    }
  }

  console.log(`  Total lines: ${lineNum}, parsed: ${parsed}, skipped: ${skipped}`);
  return claims;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== MultiFC Dataset → Mycelix Claims Importer ===');
  console.log(`Input:  ${multifcPath}`);
  console.log(`Limit:  ${limit}`);
  console.log();

  if (!fs.existsSync(multifcPath)) {
    console.error(`File not found: ${multifcPath}`);
    console.error('Download from: https://copenlu.github.io/publication/2019_emnlp_augenstein/');
    process.exit(1);
  }

  const stat = fs.statSync(multifcPath);
  console.log(`File size: ${(stat.size / 1024 / 1024).toFixed(1)} MB`);
  console.log();

  console.log('--- Parsing MultiFC TSV ---');
  const claims = await parseMultiFcTsv(multifcPath, limit);

  // Breakdown by verdict
  const trueCount = claims.filter((c) => c.tags.includes('verified-true')).length;
  const mixedCount = claims.filter((c) => c.tags.includes('mixed-verdict')).length;
  const falseCount = claims.filter((c) => c.tags.includes('verified-false')).length;
  const unknownCount = claims.filter((c) => c.tags.includes('unresolved')).length;
  console.log(`  TRUE: ${trueCount}, MIXED: ${mixedCount}, FALSE: ${falseCount}, UNKNOWN: ${unknownCount}`);
  console.log();

  const valid = filterValid(claims);
  const outFile = writeSeedFile('multifc-checked.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
