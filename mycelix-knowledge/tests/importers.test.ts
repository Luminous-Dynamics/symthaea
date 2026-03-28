// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for Mycelix Knowledge Graph importers — common utilities.
 *
 * Tests the shared types, E/N/M mapping, dedup, validation, and
 * claim builder from importers/common.ts.
 */

import { describe, it, expect } from 'vitest';
import {
  EMPIRICAL_LEVELS,
  NORMATIVE_LEVELS,
  MATERIALITY_LEVELS,
  enm,
  SOURCE_ENM,
  claimId,
  dedup,
  buildClaim,
  validateClaim,
  filterValid,
  nowMicros,
  type Claim,
  type EpistemicPosition,
} from '../importers/common';

// ---------------------------------------------------------------------------
// E/N/M Level Constants
// ---------------------------------------------------------------------------

describe('E/N/M level constants', () => {
  it('empirical levels span 0.0 to 1.0', () => {
    expect(EMPIRICAL_LEVELS.E0).toBe(0.0);
    expect(EMPIRICAL_LEVELS.E1).toBe(0.25);
    expect(EMPIRICAL_LEVELS.E2).toBe(0.5);
    expect(EMPIRICAL_LEVELS.E3).toBe(0.75);
    expect(EMPIRICAL_LEVELS.E4).toBe(1.0);
  });

  it('normative levels span 0.0 to 1.0', () => {
    expect(NORMATIVE_LEVELS.N0).toBe(0.0);
    expect(NORMATIVE_LEVELS.N3).toBe(1.0);
  });

  it('materiality levels span 0.0 to 1.0', () => {
    expect(MATERIALITY_LEVELS.M0).toBe(0.0);
    expect(MATERIALITY_LEVELS.M3).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// enm() helper
// ---------------------------------------------------------------------------

describe('enm()', () => {
  it('builds EpistemicPosition from discrete codes', () => {
    const pos = enm('E4', 'N0', 'M3');
    expect(pos.empirical).toBe(1.0);
    expect(pos.normative).toBe(0.0);
    expect(pos.mythic).toBe(1.0);
  });

  it('falls back to 0.0 for unknown codes', () => {
    const pos = enm('E99', 'N99', 'M99');
    expect(pos.empirical).toBe(0.0);
    expect(pos.normative).toBe(0.0);
    expect(pos.mythic).toBe(0.0);
  });

  it('handles mixed known and unknown codes', () => {
    const pos = enm('E3', 'N99', 'M2');
    expect(pos.empirical).toBe(0.75);
    expect(pos.normative).toBe(0.0);
    expect(pos.mythic).toBe(0.67);
  });
});

// ---------------------------------------------------------------------------
// SOURCE_ENM presets
// ---------------------------------------------------------------------------

describe('SOURCE_ENM presets', () => {
  it('wikidata is E4/N0/M3', () => {
    expect(SOURCE_ENM.wikidata.empirical).toBe(1.0);
    expect(SOURCE_ENM.wikidata.normative).toBe(0.0);
    expect(SOURCE_ENM.wikidata.mythic).toBe(1.0);
  });

  it('fever_supported is E3/N0/M1', () => {
    expect(SOURCE_ENM.fever_supported.empirical).toBe(0.75);
    expect(SOURCE_ENM.fever_supported.normative).toBe(0.0);
    expect(SOURCE_ENM.fever_supported.mythic).toBe(0.33);
  });

  it('fever_refuted is E0/N0/M0', () => {
    expect(SOURCE_ENM.fever_refuted.empirical).toBe(0.0);
    expect(SOURCE_ENM.fever_refuted.normative).toBe(0.0);
    expect(SOURCE_ENM.fever_refuted.mythic).toBe(0.0);
  });

  it('geonames is E4/N0/M3', () => {
    expect(SOURCE_ENM.geonames.empirical).toBe(1.0);
    expect(SOURCE_ENM.geonames.mythic).toBe(1.0);
  });

  it('institutional is E4/N0/M3', () => {
    expect(SOURCE_ENM.institutional.empirical).toBe(1.0);
    expect(SOURCE_ENM.institutional.mythic).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// claimId()
// ---------------------------------------------------------------------------

describe('claimId()', () => {
  it('produces deterministic IDs', () => {
    const id1 = claimId('wiki', 'The Earth is round.');
    const id2 = claimId('wiki', 'The Earth is round.');
    expect(id1).toBe(id2);
  });

  it('prefixes with source name', () => {
    const id = claimId('wikidata', 'Some fact.');
    expect(id.startsWith('wikidata:')).toBe(true);
  });

  it('produces 16-char hex hash after prefix', () => {
    const id = claimId('test', 'Some content.');
    const hash = id.split(':')[1];
    expect(hash).toMatch(/^[0-9a-f]{16}$/);
  });

  it('normalizes whitespace and case', () => {
    const id1 = claimId('src', '  Hello World  ');
    const id2 = claimId('src', 'hello world');
    expect(id1).toBe(id2);
  });

  it('different content produces different IDs', () => {
    const id1 = claimId('src', 'Claim A');
    const id2 = claimId('src', 'Claim B');
    expect(id1).not.toBe(id2);
  });

  it('different sources produce different IDs even with same content', () => {
    const id1 = claimId('wiki', 'Same fact.');
    const id2 = claimId('fever', 'Same fact.');
    expect(id1).not.toBe(id2);
  });
});

// ---------------------------------------------------------------------------
// dedup()
// ---------------------------------------------------------------------------

describe('dedup()', () => {
  const makeClaim = (id: string, content: string): Claim => ({
    id,
    content,
    classification: { empirical: 0.5, normative: 0.0, mythic: 0.5 },
    author: 'did:test',
    sources: [],
    tags: [],
    claim_type: 'Fact',
    confidence: 0.9,
    created: 0,
    updated: 0,
    version: 1,
  });

  it('removes duplicates by ID', () => {
    const claims = [
      makeClaim('a', 'First'),
      makeClaim('b', 'Second'),
      makeClaim('a', 'First duplicate'),
    ];
    const result = dedup(claims);
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe('a');
    expect(result[1].id).toBe('b');
  });

  it('keeps first occurrence on duplicate', () => {
    const claims = [
      makeClaim('x', 'Original content'),
      makeClaim('x', 'Different content same ID'),
    ];
    const result = dedup(claims);
    expect(result).toHaveLength(1);
    expect(result[0].content).toBe('Original content');
  });

  it('returns empty array for empty input', () => {
    expect(dedup([])).toHaveLength(0);
  });

  it('returns same array when no duplicates', () => {
    const claims = [makeClaim('a', 'A'), makeClaim('b', 'B'), makeClaim('c', 'C')];
    expect(dedup(claims)).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// buildClaim()
// ---------------------------------------------------------------------------

describe('buildClaim()', () => {
  it('builds a valid claim with all fields', () => {
    const claim = buildClaim({
      source: 'test',
      content: 'The sky is blue.',
      classification: enm('E4', 'N0', 'M2'),
      claimType: 'Fact',
      confidence: 0.95,
      sources: ['https://example.com'],
      tags: ['test', 'sky'],
    });

    expect(claim.id).toMatch(/^test:/);
    expect(claim.content).toBe('The sky is blue.');
    expect(claim.classification.empirical).toBe(1.0);
    expect(claim.author).toBe('did:mycelix:knowledge-importer');
    expect(claim.sources).toEqual(['https://example.com']);
    expect(claim.tags).toEqual(['test', 'sky']);
    expect(claim.claim_type).toBe('Fact');
    expect(claim.confidence).toBe(0.95);
    expect(claim.version).toBe(1);
  });

  it('sets timestamps in microseconds', () => {
    const before = Date.now() * 1000;
    const claim = buildClaim({
      source: 'test',
      content: 'Timestamp test claim content here.',
      classification: enm('E0', 'N0', 'M0'),
      claimType: 'Fact',
      confidence: 0.5,
      sources: [],
      tags: [],
    });
    const after = Date.now() * 1000;

    expect(claim.created).toBeGreaterThanOrEqual(before);
    expect(claim.created).toBeLessThanOrEqual(after);
    expect(claim.updated).toBe(claim.created);
  });

  it('generates deterministic ID from content', () => {
    const c1 = buildClaim({
      source: 'src',
      content: 'Identical content for deterministic test.',
      classification: enm('E0', 'N0', 'M0'),
      claimType: 'Fact',
      confidence: 0.5,
      sources: [],
      tags: [],
    });
    const c2 = buildClaim({
      source: 'src',
      content: 'Identical content for deterministic test.',
      classification: enm('E4', 'N3', 'M3'),
      claimType: 'Opinion',
      confidence: 1.0,
      sources: ['different'],
      tags: ['different'],
    });
    // Same source + content = same ID
    expect(c1.id).toBe(c2.id);
  });
});

// ---------------------------------------------------------------------------
// validateClaim()
// ---------------------------------------------------------------------------

describe('validateClaim()', () => {
  const validClaim = (): Claim =>
    buildClaim({
      source: 'test',
      content: 'A valid test claim about something.',
      classification: enm('E3', 'N1', 'M2'),
      claimType: 'Fact',
      confidence: 0.8,
      sources: ['https://example.com'],
      tags: ['test'],
    });

  it('returns no errors for valid claim', () => {
    expect(validateClaim(validClaim())).toHaveLength(0);
  });

  it('detects missing id', () => {
    const claim = validClaim();
    claim.id = '';
    expect(validateClaim(claim)).toContain('missing id');
  });

  it('detects content too short', () => {
    const claim = validClaim();
    claim.content = 'ab';
    expect(validateClaim(claim)).toContain('content too short');
  });

  it('detects content too long', () => {
    const claim = validClaim();
    claim.content = 'x'.repeat(10001);
    expect(validateClaim(claim)).toContain('content too long (>10K chars)');
  });

  it('detects empirical out of range', () => {
    const claim = validClaim();
    claim.classification.empirical = 1.5;
    expect(validateClaim(claim)).toContain('empirical out of range');
  });

  it('detects normative out of range', () => {
    const claim = validClaim();
    claim.classification.normative = -0.1;
    expect(validateClaim(claim)).toContain('normative out of range');
  });

  it('detects mythic out of range', () => {
    const claim = validClaim();
    claim.classification.mythic = 2.0;
    expect(validateClaim(claim)).toContain('mythic out of range');
  });

  it('detects confidence out of range', () => {
    const claim = validClaim();
    claim.confidence = -0.5;
    expect(validateClaim(claim)).toContain('confidence out of range');
  });

  it('reports multiple errors at once', () => {
    const claim = validClaim();
    claim.id = '';
    claim.content = 'x';
    claim.classification.empirical = 5.0;
    const errors = validateClaim(claim);
    expect(errors.length).toBeGreaterThanOrEqual(3);
  });
});

// ---------------------------------------------------------------------------
// filterValid()
// ---------------------------------------------------------------------------

describe('filterValid()', () => {
  it('keeps valid claims and removes invalid ones', () => {
    const good = buildClaim({
      source: 'test',
      content: 'A perfectly valid claim here.',
      classification: enm('E3', 'N0', 'M2'),
      claimType: 'Fact',
      confidence: 0.9,
      sources: [],
      tags: [],
    });
    const bad: Claim = { ...good, id: '', content: 'x', confidence: -1 };

    const result = filterValid([good, bad]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe(good.id);
  });
});

// ---------------------------------------------------------------------------
// nowMicros()
// ---------------------------------------------------------------------------

describe('nowMicros()', () => {
  it('returns microsecond timestamp', () => {
    const ts = nowMicros();
    // Should be roughly Date.now() * 1000
    const expected = Date.now() * 1000;
    expect(Math.abs(ts - expected)).toBeLessThan(1_000_000); // within 1 second
  });
});
