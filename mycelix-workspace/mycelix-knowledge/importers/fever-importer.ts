#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * FEVER Dataset → Mycelix Claims Importer
 *
 * Parses the FEVER (Fact Extraction and VERification) dataset JSONL files
 * and converts labeled claims to Mycelix Claims with E/N/M classification:
 *   SUPPORTS  → E3/N0/M1 (replicated, human-verified)
 *   REFUTES   → E0/N0/M0 (known false — valuable for factcheck training)
 *   NOT ENOUGH INFO → E1/N0/M0 (preliminary, insufficient evidence)
 *
 * FEVER dataset: https://fever.ai/dataset/fever.html
 * License: Creative Commons Attribution-ShareAlike 3.0
 *
 * Usage:
 *   npx ts-node importers/fever-importer.ts <path-to-fever-jsonl> [--limit N]
 *
 * Output:
 *   seed-data/claims/fever-train.json
 */

import * as fs from 'fs';
import * as readline from 'readline';
import {
  buildClaim,
  filterValid,
  writeSeedFile,
  SOURCE_ENM,
  type Claim,
  type ClaimType,
  type EpistemicPosition,
} from './common';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_LIMIT = 10000;

const feverPath = process.argv[2];
const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

if (!feverPath || feverPath.startsWith('--')) {
  console.error('Usage: npx ts-node importers/fever-importer.ts <path-to-fever.jsonl> [--limit N]');
  console.error();
  console.error('Download FEVER from: https://fever.ai/dataset/fever.html');
  console.error('  train.jsonl (~145MB, 145K claims)');
  console.error('  paper_dev.jsonl (~2MB, 19K claims)');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// FEVER JSONL format
// ---------------------------------------------------------------------------

interface FeverEntry {
  id: number;
  verifiable: string;       // "VERIFIABLE" | "NOT VERIFIABLE"
  label: string;            // "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"
  claim: string;            // The claim text
  evidence: any;            // Evidence annotations (not used for import)
}

// ---------------------------------------------------------------------------
// Mapping
// ---------------------------------------------------------------------------

function feverClassification(label: string): EpistemicPosition {
  switch (label) {
    case 'SUPPORTS':
      return SOURCE_ENM.fever_supported;
    case 'REFUTES':
      return SOURCE_ENM.fever_refuted;
    case 'NOT ENOUGH INFO':
      return SOURCE_ENM.fever_nei;
    default:
      return SOURCE_ENM.fever_nei;
  }
}

function feverConfidence(label: string): number {
  switch (label) {
    case 'SUPPORTS':
      return 0.85;
    case 'REFUTES':
      return 0.85; // high confidence it's false
    case 'NOT ENOUGH INFO':
      return 0.30;
    default:
      return 0.20;
  }
}

function feverClaimType(label: string): ClaimType {
  // REFUTED claims are still stored as "Fact" type — the E0 classification
  // signals they are known-false. This preserves the original claim structure.
  return 'Fact';
}

function feverTags(label: string): string[] {
  const base = ['fever-dataset', 'nlp', 'fact-verification'];
  switch (label) {
    case 'SUPPORTS':
      return [...base, 'verified-true'];
    case 'REFUTES':
      return [...base, 'verified-false'];
    case 'NOT ENOUGH INFO':
      return [...base, 'unresolved'];
    default:
      return base;
  }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

async function parseFeverJsonl(filepath: string, maxClaims: number): Promise<Claim[]> {
  const claims: Claim[] = [];

  const fileStream = fs.createReadStream(filepath, { encoding: 'utf-8' });
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity,
  });

  let lineNum = 0;
  let parsed = 0;
  let skipped = 0;

  for await (const line of rl) {
    lineNum++;
    if (parsed >= maxClaims) break;

    const trimmed = line.trim();
    if (!trimmed) continue;

    let entry: FeverEntry;
    try {
      entry = JSON.parse(trimmed);
    } catch {
      skipped++;
      continue;
    }

    // Skip non-verifiable entries (they have no label)
    if (!entry.label || !entry.claim) {
      skipped++;
      continue;
    }

    // Skip very short claims (noise)
    if (entry.claim.length < 10) {
      skipped++;
      continue;
    }

    claims.push(
      buildClaim({
        source: 'fever',
        content: entry.claim,
        classification: feverClassification(entry.label),
        claimType: feverClaimType(entry.label),
        confidence: feverConfidence(entry.label),
        sources: [`https://fever.ai/dataset/fever.html#${entry.id}`],
        tags: feverTags(entry.label),
      }),
    );

    parsed++;

    if (parsed % 5000 === 0) {
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
  console.log('=== FEVER Dataset → Mycelix Claims Importer ===');
  console.log(`Input:  ${feverPath}`);
  console.log(`Limit:  ${limit}`);
  console.log();

  if (!fs.existsSync(feverPath)) {
    console.error(`File not found: ${feverPath}`);
    console.error('Download from https://fever.ai/dataset/fever.html');
    process.exit(1);
  }

  const stat = fs.statSync(feverPath);
  console.log(`File size: ${(stat.size / 1024 / 1024).toFixed(1)} MB`);
  console.log();

  console.log('--- Parsing FEVER JSONL ---');
  const claims = await parseFeverJsonl(feverPath, limit);

  // Breakdown by label
  const supported = claims.filter((c) => c.tags.includes('verified-true')).length;
  const refuted = claims.filter((c) => c.tags.includes('verified-false')).length;
  const nei = claims.filter((c) => c.tags.includes('unresolved')).length;
  console.log(`  SUPPORTS: ${supported}, REFUTES: ${refuted}, NOT ENOUGH INFO: ${nei}`);
  console.log();

  const valid = filterValid(claims);
  const outFile = writeSeedFile('fever-train.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
