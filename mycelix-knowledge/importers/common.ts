// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Common utilities for Mycelix Knowledge Graph importers
 *
 * Shared types, E/N/M mapping helpers, content-hash deduplication,
 * and JSON output for all data source importers.
 */

import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

// ---------------------------------------------------------------------------
// Types matching the Holochain Claim struct (claims_integrity)
// ---------------------------------------------------------------------------

/** Epistemic position on the 3D E/N/M cube (continuous 0.0-1.0) */
export interface EpistemicPosition {
  empirical: number;
  normative: number;
  mythic: number; // maps to Materiality in GIS v4
}

/** Claim types matching the Rust ClaimType enum */
export type ClaimType =
  | 'Fact'
  | 'Opinion'
  | 'Prediction'
  | 'Hypothesis'
  | 'Definition'
  | 'Historical'
  | 'Normative'
  | 'Narrative';

/**
 * Claim struct matching the Holochain entry type.
 * Timestamps are microseconds (Holochain Timestamp convention).
 */
export interface Claim {
  id: string;
  content: string;
  classification: EpistemicPosition;
  author: string;
  sources: string[];
  tags: string[];
  claim_type: ClaimType;
  confidence: number;
  expires?: number;
  created: number;
  updated: number;
  version: number;
}

// ---------------------------------------------------------------------------
// E/N/M Discrete Level Helpers
// ---------------------------------------------------------------------------

/** Empirical levels E0-E4 mapped to continuous values */
export const EMPIRICAL_LEVELS: Record<string, number> = {
  E0: 0.0,   // Unverified
  E1: 0.25,  // Preliminary
  E2: 0.5,   // Tested
  E3: 0.75,  // Replicated
  E4: 1.0,   // Established
};

/** Normative levels N0-N3 mapped to continuous values */
export const NORMATIVE_LEVELS: Record<string, number> = {
  N0: 0.0,   // Raw
  N1: 0.33,  // Contested
  N2: 0.67,  // Emerging
  N3: 1.0,   // Endorsed
};

/** Materiality levels M0-M3 mapped to continuous values */
export const MATERIALITY_LEVELS: Record<string, number> = {
  M0: 0.0,   // Abstract
  M1: 0.33,  // Potential
  M2: 0.67,  // Applicable
  M3: 1.0,   // Transformative
};

/** Build an EpistemicPosition from discrete E/N/M level codes */
export function enm(e: string, n: string, m: string): EpistemicPosition {
  return {
    empirical: EMPIRICAL_LEVELS[e] ?? 0.0,
    normative: NORMATIVE_LEVELS[n] ?? 0.0,
    mythic: MATERIALITY_LEVELS[m] ?? 0.0,
  };
}

// ---------------------------------------------------------------------------
// Per-source E/N/M presets
// ---------------------------------------------------------------------------

export const SOURCE_ENM = {
  /** Wikidata verified facts: E4/N0/M3 */
  wikidata: enm('E4', 'N0', 'M3'),
  /** FEVER SUPPORTED: E3/N0/M1 */
  fever_supported: enm('E3', 'N0', 'M1'),
  /** FEVER REFUTED: E0/N0/M0 — known false, useful for training */
  fever_refuted: enm('E0', 'N0', 'M0'),
  /** FEVER NOT ENOUGH INFO: E1/N0/M0 */
  fever_nei: enm('E1', 'N0', 'M0'),
  /** GeoNames geographic facts: E4/N0/M3 */
  geonames: enm('E4', 'N0', 'M3'),
  /** OpenAlex highly-cited scientific: E3/N1/M2 */
  openal_high: enm('E3', 'N1', 'M2'),
  /** OpenAlex moderately-cited: E2/N1/M2 */
  openal_medium: enm('E2', 'N1', 'M2'),
  /** Government/institutional data (climate, health, energy): E4/N0/M3 */
  institutional: enm('E4', 'N0', 'M3'),
} as const;

// ---------------------------------------------------------------------------
// Content-hash deduplication
// ---------------------------------------------------------------------------

/**
 * Generate a deterministic claim ID from content.
 * Uses BLAKE2b-256 truncated to 16 hex chars with source prefix.
 */
export function claimId(source: string, content: string): string {
  const hash = crypto
    .createHash('sha256')
    .update(content.trim().toLowerCase())
    .digest('hex')
    .slice(0, 16);
  return `${source}:${hash}`;
}

/**
 * Deduplicate an array of claims by ID.
 * First occurrence wins.
 */
export function dedup(claims: Claim[]): Claim[] {
  const seen = new Set<string>();
  return claims.filter((c) => {
    if (seen.has(c.id)) return false;
    seen.add(c.id);
    return true;
  });
}

// ---------------------------------------------------------------------------
// Claim builder
// ---------------------------------------------------------------------------

const IMPORTER_DID = 'did:mycelix:knowledge-importer';

/** Current timestamp in Holochain microseconds */
export function nowMicros(): number {
  return Date.now() * 1000;
}

/** Build a Claim with sensible defaults */
export function buildClaim(opts: {
  source: string;
  content: string;
  classification: EpistemicPosition;
  claimType: ClaimType;
  confidence: number;
  sources: string[];
  tags: string[];
  domain?: string;
}): Claim {
  const now = nowMicros();
  return {
    id: claimId(opts.source, opts.content),
    content: opts.content,
    classification: opts.classification,
    author: IMPORTER_DID,
    sources: opts.sources,
    tags: opts.tags,
    claim_type: opts.claimType,
    confidence: opts.confidence,
    created: now,
    updated: now,
    version: 1,
  };
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

const SEED_DATA_DIR = path.resolve(__dirname, '..', 'seed-data', 'claims');

/** Write claims to a JSON file in seed-data/claims/ */
export function writeSeedFile(filename: string, claims: Claim[]): string {
  const deduplicated = dedup(claims);
  const filepath = path.join(SEED_DATA_DIR, filename);
  fs.mkdirSync(path.dirname(filepath), { recursive: true });
  fs.writeFileSync(filepath, JSON.stringify(deduplicated, null, 2));
  console.log(`[WRITE] ${filepath} — ${deduplicated.length} claims`);
  return filepath;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/** Validate a claim has required fields and sane values */
export function validateClaim(claim: Claim): string[] {
  const errors: string[] = [];
  if (!claim.id) errors.push('missing id');
  if (!claim.content || claim.content.length < 3) errors.push('content too short');
  if (claim.content.length > 10000) errors.push('content too long (>10K chars)');
  const c = claim.classification;
  if (c.empirical < 0 || c.empirical > 1) errors.push('empirical out of range');
  if (c.normative < 0 || c.normative > 1) errors.push('normative out of range');
  if (c.mythic < 0 || c.mythic > 1) errors.push('mythic out of range');
  if (claim.confidence < 0 || claim.confidence > 1) errors.push('confidence out of range');
  return errors;
}

/** Validate all claims, log errors, return only valid ones */
export function filterValid(claims: Claim[]): Claim[] {
  const valid: Claim[] = [];
  let invalid = 0;
  for (const c of claims) {
    const errs = validateClaim(c);
    if (errs.length > 0) {
      console.error(`  [INVALID] ${c.id}: ${errs.join(', ')}`);
      invalid++;
    } else {
      valid.push(c);
    }
  }
  if (invalid > 0) {
    console.warn(`  ${invalid} invalid claims filtered out`);
  }
  return valid;
}
