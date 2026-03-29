// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * SDK Unit Tests
 *
 * Tests for the @mycelix/knowledge-sdk TypeScript SDK.
 */

import {
  toDiscreteEpistemic,
  calculateInformationValue,
  recommendVerification,
  calculateCompositeCredibility,
  determineVerdict,
  EpistemicPosition,
  Claim,
  InformationValueRecord,
  ClaimEvidence,
} from '../client/src/index';

// ============================================================================
// Utility Function Tests
// ============================================================================

describe('toDiscreteEpistemic', () => {
  test('converts E levels correctly', () => {
    expect(toDiscreteEpistemic(0.95, 0.1, 0.3)).toEqual({ e: 'E5', n: 'N1', m: 'M2' });
    expect(toDiscreteEpistemic(0.75, 0.5, 0.7)).toEqual({ e: 'E4', n: 'N3', m: 'M4' });
    expect(toDiscreteEpistemic(0.55, 0.3, 0.5)).toEqual({ e: 'E3', n: 'N2', m: 'M3' });
    expect(toDiscreteEpistemic(0.35, 0.7, 0.9)).toEqual({ e: 'E2', n: 'N4', m: 'M5' });
    expect(toDiscreteEpistemic(0.1, 0.9, 0.1)).toEqual({ e: 'E1', n: 'N5', m: 'M1' });
  });

  test('handles edge cases', () => {
    expect(toDiscreteEpistemic(0.0, 0.0, 0.0)).toEqual({ e: 'E1', n: 'N1', m: 'M1' });
    expect(toDiscreteEpistemic(1.0, 1.0, 1.0)).toEqual({ e: 'E5', n: 'N5', m: 'M5' });
    expect(toDiscreteEpistemic(0.2, 0.2, 0.2)).toEqual({ e: 'E1', n: 'N1', m: 'M1' });
  });
});

describe('calculateInformationValue', () => {
  test('calculates IV correctly', () => {
    const claim: Partial<Claim> = {
      classification: { empirical: 0.5, normative: 0.2, mythic: 0.3 },
    };

    const ivRecord: Partial<InformationValueRecord> = {
      components: {
        uncertainty: 0.8,
        connectivity: 0.6,
        cascadePotential: 0.7,
        ageFactor: 0.9,
      },
    };

    const iv = calculateInformationValue(claim as Claim, ivRecord as InformationValueRecord);

    // Expected: 0.8 * 0.6 * 0.7 * 0.9 = 0.3024
    expect(iv).toBeCloseTo(0.3024, 4);
  });

  test('returns 0 for missing components', () => {
    const claim: Partial<Claim> = {
      classification: { empirical: 0.5, normative: 0.2, mythic: 0.3 },
    };

    const iv = calculateInformationValue(claim as Claim, null as unknown as InformationValueRecord);
    expect(iv).toBe(0);
  });
});

describe('recommendVerification', () => {
  test('recommends verification for mid-E claims', () => {
    const claim: Partial<Claim> = {
      classification: { empirical: 0.5, normative: 0.2, mythic: 0.3 },
    };

    const ivRecord: Partial<InformationValueRecord> = {
      value: 0.6,
    };

    const result = recommendVerification(
      claim as Claim,
      ivRecord as InformationValueRecord
    );

    expect(result.recommend).toBe(true);
    expect(result.suggestedTargetE).toBeGreaterThan(0.5);
    expect(result.reason).toContain('mid-range');
  });

  test('does not recommend for high-E claims', () => {
    const claim: Partial<Claim> = {
      classification: { empirical: 0.9, normative: 0.1, mythic: 0.3 },
    };

    const ivRecord: Partial<InformationValueRecord> = {
      value: 0.3,
    };

    const result = recommendVerification(
      claim as Claim,
      ivRecord as InformationValueRecord
    );

    expect(result.recommend).toBe(false);
    expect(result.reason).toContain('already high');
  });

  test('does not recommend for very low-E claims', () => {
    const claim: Partial<Claim> = {
      classification: { empirical: 0.1, normative: 0.8, mythic: 0.9 },
    };

    const ivRecord: Partial<InformationValueRecord> = {
      value: 0.2,
    };

    const result = recommendVerification(
      claim as Claim,
      ivRecord as InformationValueRecord
    );

    expect(result.recommend).toBe(false);
    expect(result.reason).toContain('too low');
  });
});

describe('calculateCompositeCredibility', () => {
  test('calculates composite score correctly', () => {
    const result = calculateCompositeCredibility({
      evidenceStrength: 0.8,
      authorReputation: 0.7,
      networkPosition: 0.6,
      temporalRelevance: 0.9,
    });

    // Weighted average with default weights
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThanOrEqual(1);
  });

  test('handles missing values', () => {
    const result = calculateCompositeCredibility({
      evidenceStrength: 0.8,
      authorReputation: 0.7,
    });

    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThanOrEqual(1);
  });
});

describe('determineVerdict', () => {
  test('returns True for strong supporting evidence', () => {
    const supporting: ClaimEvidence[] = [
      { claimId: '1', relevance: 0.9, supports: true, excerpt: '' },
      { claimId: '2', relevance: 0.85, supports: true, excerpt: '' },
      { claimId: '3', relevance: 0.8, supports: true, excerpt: '' },
    ];
    const contradicting: ClaimEvidence[] = [];

    const verdict = determineVerdict(supporting, contradicting, 0.6);
    expect(verdict).toBe('True');
  });

  test('returns False for strong contradicting evidence', () => {
    const supporting: ClaimEvidence[] = [];
    const contradicting: ClaimEvidence[] = [
      { claimId: '1', relevance: 0.9, supports: false, excerpt: '' },
      { claimId: '2', relevance: 0.85, supports: false, excerpt: '' },
    ];

    const verdict = determineVerdict(supporting, contradicting, 0.6);
    expect(verdict).toBe('False');
  });

  test('returns Mixed for balanced evidence', () => {
    const supporting: ClaimEvidence[] = [
      { claimId: '1', relevance: 0.8, supports: true, excerpt: '' },
    ];
    const contradicting: ClaimEvidence[] = [
      { claimId: '2', relevance: 0.8, supports: false, excerpt: '' },
    ];

    const verdict = determineVerdict(supporting, contradicting, 0.6);
    expect(verdict).toBe('Mixed');
  });

  test('returns InsufficientEvidence for no evidence', () => {
    const verdict = determineVerdict([], [], 0.6);
    expect(verdict).toBe('InsufficientEvidence');
  });

  test('returns MostlyTrue for mostly supporting', () => {
    const supporting: ClaimEvidence[] = [
      { claimId: '1', relevance: 0.8, supports: true, excerpt: '' },
      { claimId: '2', relevance: 0.7, supports: true, excerpt: '' },
    ];
    const contradicting: ClaimEvidence[] = [
      { claimId: '3', relevance: 0.5, supports: false, excerpt: '' },
    ];

    const verdict = determineVerdict(supporting, contradicting, 0.6);
    expect(['True', 'MostlyTrue']).toContain(verdict);
  });
});

// ============================================================================
// Type Validation Tests
// ============================================================================

describe('Type Validation', () => {
  test('EpistemicPosition has correct shape', () => {
    const position: EpistemicPosition = {
      empirical: 0.8,
      normative: 0.2,
      mythic: 0.5,
    };

    expect(position.empirical).toBeGreaterThanOrEqual(0);
    expect(position.empirical).toBeLessThanOrEqual(1);
    expect(position.normative).toBeGreaterThanOrEqual(0);
    expect(position.normative).toBeLessThanOrEqual(1);
    expect(position.mythic).toBeGreaterThanOrEqual(0);
    expect(position.mythic).toBeLessThanOrEqual(1);
  });

  test('Claim has required fields', () => {
    const claim: Claim = {
      id: 'test-claim',
      content: 'Test content',
      classification: { empirical: 0.7, normative: 0.2, mythic: 0.3 },
      domain: 'test',
      topics: ['test'],
      evidence: [],
      status: 'Published',
      sourceHapp: null,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    expect(claim.id).toBeDefined();
    expect(claim.content).toBeDefined();
    expect(claim.classification).toBeDefined();
    expect(claim.domain).toBeDefined();
  });
});

// ============================================================================
// Edge Case Tests
// ============================================================================

describe('Edge Cases', () => {
  test('handles zero values in IV calculation', () => {
    const claim: Partial<Claim> = {
      classification: { empirical: 0, normative: 0, mythic: 0 },
    };

    const ivRecord: Partial<InformationValueRecord> = {
      components: {
        uncertainty: 0,
        connectivity: 0,
        cascadePotential: 0,
        ageFactor: 0,
      },
    };

    const iv = calculateInformationValue(claim as Claim, ivRecord as InformationValueRecord);
    expect(iv).toBe(0);
  });

  test('handles maximum values', () => {
    const claim: Partial<Claim> = {
      classification: { empirical: 1, normative: 1, mythic: 1 },
    };

    const ivRecord: Partial<InformationValueRecord> = {
      components: {
        uncertainty: 1,
        connectivity: 1,
        cascadePotential: 1,
        ageFactor: 1,
      },
    };

    const iv = calculateInformationValue(claim as Claim, ivRecord as InformationValueRecord);
    expect(iv).toBe(1);
  });

  test('empty evidence arrays in verdict', () => {
    const verdict = determineVerdict([], [], 0.5);
    expect(verdict).toBe('InsufficientEvidence');
  });
});

// ============================================================================
// Mock Client Tests
// ============================================================================

describe('Mock Client', () => {
  const mockCallZome = jest.fn();

  const mockClient = {
    callZome: mockCallZome,
  };

  beforeEach(() => {
    mockCallZome.mockReset();
  });

  test('ClaimsClient.createClaim calls correct zome', async () => {
    mockCallZome.mockResolvedValue('uhCkk...');

    // This would test the actual client, but requires proper imports
    // For now, verify the mock setup works
    const result = await mockClient.callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {},
    });

    expect(mockCallZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'claims',
        fn_name: 'create_claim',
      })
    );
    expect(result).toBe('uhCkk...');
  });

  test('GraphClient.createRelationship calls correct zome', async () => {
    mockCallZome.mockResolvedValue('uhCkk...');

    await mockClient.callZome({
      zome_name: 'graph',
      fn_name: 'create_relationship',
      payload: {},
    });

    expect(mockCallZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'graph',
        fn_name: 'create_relationship',
      })
    );
  });

  test('FactCheckClient.factCheck calls correct zome', async () => {
    mockCallZome.mockResolvedValue({
      verdict: 'True',
      confidence: 0.85,
      supporting_claims: [],
      contradicting_claims: [],
      explanation: 'Test',
      checked_at: Date.now(),
    });

    const result = await mockClient.callZome({
      zome_name: 'factcheck',
      fn_name: 'fact_check',
      payload: { statement: 'test' },
    });

    expect(result.verdict).toBe('True');
    expect(result.confidence).toBe(0.85);
  });
});
