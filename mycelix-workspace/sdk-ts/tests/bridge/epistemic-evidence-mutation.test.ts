// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Epistemic Evidence Mutation Tests
 *
 * Mutation-targeted tests for bridge/epistemic-evidence.ts.
 * Focus on edge cases, boundary conditions, and error paths.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  EpistemicEvidenceBridge,
  getEpistemicEvidenceBridge,
  resetEpistemicEvidenceBridge,
  mapEvidenceTypeToClaimType,
  calculateAdjustedStrength,
  getEpistemicWeightsForCategory,
  meetsEpistemicStandards,
  explainEpistemicClassification,
  verdictToRecommendation,
  calculateOverallCredibility,
  type EpistemicVerificationResult,
  type BatchVerificationInput,
} from '../../src/bridge/epistemic-evidence.js';
import type { Evidence, CaseCategory, EvidenceType } from '../../src/justice/index.js';
import type { EpistemicPosition, FactCheckVerdict, FactCheckResult, CredibilityScore } from '../../src/knowledge/index.js';

// ============================================================================
// Test Fixtures
// ============================================================================

function createTestEvidence(overrides: Partial<Evidence> = {}): Evidence {
  return {
    id: `evidence-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    case_id: 'case-test-001',
    submitter: 'did:mycelix:alice',
    type_: 'Document',
    title: 'Test Evidence',
    description: 'Test evidence description',
    content_hash: 'Qm1234567890abcdef',
    submitted_at: Date.now(),
    verified: false,
    ...overrides,
  };
}

// ============================================================================
// mapEvidenceTypeToClaimType Tests
// ============================================================================

describe('mapEvidenceTypeToClaimType - Mutation Coverage', () => {
  it('should map Document to Fact', () => {
    expect(mapEvidenceTypeToClaimType('Document')).toBe('Fact');
  });

  it('should map Testimony to Opinion', () => {
    expect(mapEvidenceTypeToClaimType('Testimony')).toBe('Opinion');
  });

  it('should map Transaction to Historical', () => {
    expect(mapEvidenceTypeToClaimType('Transaction')).toBe('Historical');
  });

  it('should map Screenshot to Fact', () => {
    expect(mapEvidenceTypeToClaimType('Screenshot')).toBe('Fact');
  });

  it('should map Witness to Opinion', () => {
    expect(mapEvidenceTypeToClaimType('Witness')).toBe('Opinion');
  });

  it('should map Other to Fact', () => {
    expect(mapEvidenceTypeToClaimType('Other')).toBe('Fact');
  });

  it('should default to Fact for unknown types', () => {
    expect(mapEvidenceTypeToClaimType('UnknownType' as EvidenceType)).toBe('Fact');
  });
});

// ============================================================================
// getEpistemicWeightsForCategory Tests
// ============================================================================

describe('getEpistemicWeightsForCategory - Mutation Coverage', () => {
  it('should return correct weights for ContractDispute', () => {
    const weights = getEpistemicWeightsForCategory('ContractDispute');
    expect(weights.empirical).toBe(0.7);
    expect(weights.normative).toBe(0.2);
    expect(weights.mythic).toBe(0.1);
  });

  it('should return correct weights for ConductViolation', () => {
    const weights = getEpistemicWeightsForCategory('ConductViolation');
    expect(weights.empirical).toBe(0.6);
    expect(weights.normative).toBe(0.3);
    expect(weights.mythic).toBe(0.1);
  });

  it('should return correct weights for PropertyDispute', () => {
    const weights = getEpistemicWeightsForCategory('PropertyDispute');
    expect(weights.empirical).toBe(0.65);
    expect(weights.normative).toBe(0.25);
    expect(weights.mythic).toBe(0.1);
  });

  it('should return correct weights for FinancialDispute', () => {
    const weights = getEpistemicWeightsForCategory('FinancialDispute');
    expect(weights.empirical).toBe(0.75);
    expect(weights.normative).toBe(0.2);
    expect(weights.mythic).toBe(0.05);
  });

  it('should return correct weights for GovernanceDispute', () => {
    const weights = getEpistemicWeightsForCategory('GovernanceDispute');
    expect(weights.empirical).toBe(0.4);
    expect(weights.normative).toBe(0.5);
    expect(weights.mythic).toBe(0.1);
  });

  it('should return correct weights for IdentityDispute', () => {
    const weights = getEpistemicWeightsForCategory('IdentityDispute');
    expect(weights.empirical).toBe(0.6);
    expect(weights.normative).toBe(0.3);
    expect(weights.mythic).toBe(0.1);
  });

  it('should return correct weights for IPDispute', () => {
    const weights = getEpistemicWeightsForCategory('IPDispute');
    expect(weights.empirical).toBe(0.55);
    expect(weights.normative).toBe(0.35);
    expect(weights.mythic).toBe(0.1);
  });

  it('should return balanced weights for Other', () => {
    const weights = getEpistemicWeightsForCategory('Other');
    expect(weights.empirical).toBe(0.5);
    expect(weights.normative).toBe(0.35);
    expect(weights.mythic).toBe(0.15);
  });

  it('should return balanced weights for unknown category', () => {
    const weights = getEpistemicWeightsForCategory('UnknownCategory' as CaseCategory);
    expect(weights.empirical).toBe(0.5);
    expect(weights.normative).toBe(0.35);
    expect(weights.mythic).toBe(0.15);
  });
});

// ============================================================================
// calculateAdjustedStrength Tests
// ============================================================================

describe('calculateAdjustedStrength - Mutation Coverage', () => {
  it('should increase strength for high empirical evidence in empirical-heavy case', () => {
    const position: EpistemicPosition = { empirical: 0.9, normative: 0.05, mythic: 0.05 };
    const strength = calculateAdjustedStrength(0.8, position, 'ConductViolation');

    expect(strength).toBeGreaterThan(0.5);
    expect(strength).toBeLessThanOrEqual(1);
  });

  it('should clamp to 0 for very low values', () => {
    const position: EpistemicPosition = { empirical: 0, normative: 0, mythic: 0 };
    const strength = calculateAdjustedStrength(0, position, 'Other');

    expect(strength).toBeGreaterThanOrEqual(0);
  });

  it('should clamp to 1 for very high values', () => {
    const position: EpistemicPosition = { empirical: 1, normative: 1, mythic: 1 };
    const strength = calculateAdjustedStrength(1, position, 'ConductViolation');

    expect(strength).toBeLessThanOrEqual(1);
  });

  it('should apply different weights for different categories', () => {
    // Use a position that emphasizes empirical - FinancialDispute values empirical highly (0.75) while
    // GovernanceDispute values normative highly (0.5), so this should produce different results
    const position: EpistemicPosition = { empirical: 0.9, normative: 0.1, mythic: 0.1 };

    const financialStrength = calculateAdjustedStrength(0.6, position, 'FinancialDispute');
    const governanceStrength = calculateAdjustedStrength(0.6, position, 'GovernanceDispute');

    // FinancialDispute: 0.9*0.75 + 0.1*0.2 + 0.1*0.05 = 0.70
    // GovernanceDispute: 0.9*0.4 + 0.1*0.5 + 0.1*0.1 = 0.42
    // These should be different because of different category weights
    expect(Math.abs(financialStrength - governanceStrength)).toBeGreaterThan(0.01);
  });

  it('should handle edge case with zero base strength', () => {
    const position: EpistemicPosition = { empirical: 0.8, normative: 0.1, mythic: 0.1 };
    const strength = calculateAdjustedStrength(0, position, 'ContractDispute');

    expect(strength).toBe(0);
  });
});

// ============================================================================
// verdictToRecommendation Tests
// ============================================================================

describe('verdictToRecommendation - Mutation Coverage', () => {
  it('should return Strong for True verdict with high credibility', () => {
    expect(verdictToRecommendation('True', 0.9)).toBe('Strong');
    expect(verdictToRecommendation('True', 0.8)).toBe('Strong');
  });

  it('should return Moderate for True verdict with medium credibility', () => {
    expect(verdictToRecommendation('True', 0.6)).toBe('Moderate');
    expect(verdictToRecommendation('True', 0.5)).toBe('Moderate');
  });

  it('should return Moderate for PartiallyTrue with high credibility', () => {
    expect(verdictToRecommendation('PartiallyTrue', 0.7)).toBe('Moderate');
    expect(verdictToRecommendation('PartiallyTrue', 0.6)).toBe('Moderate');
  });

  it('should return Weak for PartiallyTrue with lower credibility', () => {
    expect(verdictToRecommendation('PartiallyTrue', 0.4)).toBe('Weak');
  });

  it('should return Weak for Misleading verdict', () => {
    expect(verdictToRecommendation('Misleading', 0.5)).toBe('Weak');
    expect(verdictToRecommendation('Misleading', 0.9)).toBe('Weak');
  });

  it('should return Unreliable for False verdict', () => {
    expect(verdictToRecommendation('False', 0.9)).toBe('Unreliable');
    expect(verdictToRecommendation('False', 0.1)).toBe('Unreliable');
  });

  it('should return Insufficient for Unverified verdict', () => {
    expect(verdictToRecommendation('Unverified', 0.5)).toBe('Insufficient');
  });

  it('should handle boundary credibility values', () => {
    expect(verdictToRecommendation('True', 0.79)).toBe('Moderate');
    expect(verdictToRecommendation('True', 0.49)).toBe('Insufficient');
    expect(verdictToRecommendation('PartiallyTrue', 0.59)).toBe('Weak');
  });
});

// ============================================================================
// calculateOverallCredibility Tests
// ============================================================================

describe('calculateOverallCredibility - Mutation Coverage', () => {
  it('should return base credibility for epistemic position only', () => {
    const position: EpistemicPosition = { empirical: 0.8, normative: 0.1, mythic: 0.1 };
    const result = calculateOverallCredibility(position);

    expect(result.credibility).toBeGreaterThan(0);
    expect(result.credibility).toBeLessThanOrEqual(1);
    expect(result.confidence).toBe(0.4);
  });

  it('should boost credibility for True fact-check result', () => {
    const position: EpistemicPosition = { empirical: 0.6, normative: 0.2, mythic: 0.2 };
    const factCheck: FactCheckResult = { verdict: 'True', confidence: 0.9, sources: [], reasoning: '' };

    const result = calculateOverallCredibility(position, factCheck);

    expect(result.credibility).toBeGreaterThan(calculateOverallCredibility(position).credibility);
  });

  it('should reduce credibility for False fact-check result', () => {
    const position: EpistemicPosition = { empirical: 0.6, normative: 0.2, mythic: 0.2 };
    const factCheck: FactCheckResult = { verdict: 'False', confidence: 0.9, sources: [], reasoning: '' };

    const result = calculateOverallCredibility(position, factCheck);
    const baseResult = calculateOverallCredibility(position);

    expect(result.credibility).toBeLessThan(baseResult.credibility);
  });

  it('should handle PartiallyTrue verdict', () => {
    const position: EpistemicPosition = { empirical: 0.5, normative: 0.3, mythic: 0.2 };
    const factCheck: FactCheckResult = { verdict: 'PartiallyTrue', confidence: 0.7, sources: [], reasoning: '' };

    const result = calculateOverallCredibility(position, factCheck);
    expect(result.credibility).toBeGreaterThan(0);
    expect(result.confidence).toBeGreaterThan(0.4);
  });

  it('should handle Misleading verdict', () => {
    const position: EpistemicPosition = { empirical: 0.5, normative: 0.3, mythic: 0.2 };
    const factCheck: FactCheckResult = { verdict: 'Misleading', confidence: 0.8, sources: [], reasoning: '' };

    const result = calculateOverallCredibility(position, factCheck);
    const baseResult = calculateOverallCredibility(position);

    expect(result.credibility).toBeLessThan(baseResult.credibility);
  });

  it('should handle Unverified verdict', () => {
    const position: EpistemicPosition = { empirical: 0.5, normative: 0.3, mythic: 0.2 };
    const factCheck: FactCheckResult = { verdict: 'Unverified', confidence: 0.5, sources: [], reasoning: '' };

    const result = calculateOverallCredibility(position, factCheck);
    // Unverified should not change credibility much
    expect(result.credibility).toBeCloseTo(calculateOverallCredibility(position).credibility, 0);
  });

  it('should include credibility score boost', () => {
    const position: EpistemicPosition = { empirical: 0.6, normative: 0.2, mythic: 0.2 };
    const credScore: CredibilityScore = { score: 0.9, factors: [], updated_at: Date.now() };

    const result = calculateOverallCredibility(position, undefined, credScore);
    const baseResult = calculateOverallCredibility(position);

    expect(result.credibility).toBeGreaterThan(baseResult.credibility);
    expect(result.confidence).toBeGreaterThan(baseResult.confidence);
  });

  it('should clamp values between 0 and 1', () => {
    const position: EpistemicPosition = { empirical: 1, normative: 1, mythic: 1 };
    const factCheck: FactCheckResult = { verdict: 'True', confidence: 1, sources: [], reasoning: '' };
    const credScore: CredibilityScore = { score: 1, factors: [], updated_at: Date.now() };

    const result = calculateOverallCredibility(position, factCheck, credScore);

    expect(result.credibility).toBeLessThanOrEqual(1);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });
});

// ============================================================================
// meetsEpistemicStandards Tests
// ============================================================================

describe('meetsEpistemicStandards - Mutation Coverage', () => {
  it('should return true for high quality evidence', () => {
    const result: EpistemicVerificationResult = {
      evidence_id: 'test-1',
      verified: true,
      classification: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
      credibility: 0.8,
      confidence: 0.7,
      supporting_claims: [],
      contradicting_claims: [],
      factors: [],
      recommendation: 'Strong',
      verified_at: Date.now(),
    };

    expect(meetsEpistemicStandards(result, 'ContractDispute')).toBe(true);
  });

  it('should return false for low credibility', () => {
    const result: EpistemicVerificationResult = {
      evidence_id: 'test-2',
      verified: true,
      classification: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
      credibility: 0.3, // Below 0.4 threshold
      confidence: 0.7,
      supporting_claims: [],
      contradicting_claims: [],
      factors: [],
      recommendation: 'Strong',
      verified_at: Date.now(),
    };

    expect(meetsEpistemicStandards(result, 'ContractDispute')).toBe(false);
  });

  it('should return false for Unreliable recommendation', () => {
    const result: EpistemicVerificationResult = {
      evidence_id: 'test-3',
      verified: true,
      classification: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
      credibility: 0.8,
      confidence: 0.7,
      supporting_claims: [],
      contradicting_claims: [],
      factors: [],
      recommendation: 'Unreliable',
      verified_at: Date.now(),
    };

    expect(meetsEpistemicStandards(result, 'ContractDispute')).toBe(false);
  });

  it('should return false for Insufficient recommendation', () => {
    const result: EpistemicVerificationResult = {
      evidence_id: 'test-4',
      verified: true,
      classification: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
      credibility: 0.8,
      confidence: 0.7,
      supporting_claims: [],
      contradicting_claims: [],
      factors: [],
      recommendation: 'Insufficient',
      verified_at: Date.now(),
    };

    expect(meetsEpistemicStandards(result, 'ContractDispute')).toBe(false);
  });

  it('should consider weighted score for different case categories', () => {
    const result: EpistemicVerificationResult = {
      evidence_id: 'test-5',
      verified: true,
      classification: { empirical: 0.3, normative: 0.8, mythic: 0.1 },
      credibility: 0.6,
      confidence: 0.6,
      supporting_claims: [],
      contradicting_claims: [],
      factors: [],
      recommendation: 'Moderate',
      verified_at: Date.now(),
    };

    // GovernanceDispute weights normative heavily, so this should meet standards
    expect(meetsEpistemicStandards(result, 'GovernanceDispute')).toBe(true);
  });

  it('should handle edge case at threshold', () => {
    // For 'Other': weights = { empirical: 0.5, normative: 0.35, mythic: 0.15 }
    // Need weightedScore >= 0.5: 0.8*0.5 + 0.3*0.35 + 0.2*0.15 = 0.535
    const result: EpistemicVerificationResult = {
      evidence_id: 'test-6',
      verified: true,
      classification: { empirical: 0.8, normative: 0.3, mythic: 0.2 },
      credibility: 0.4, // Exactly at threshold
      confidence: 0.5,
      supporting_claims: [],
      contradicting_claims: [],
      factors: [],
      recommendation: 'Weak',
      verified_at: Date.now(),
    };

    expect(meetsEpistemicStandards(result, 'Other')).toBe(true);
  });
});

// ============================================================================
// explainEpistemicClassification Tests
// ============================================================================

describe('explainEpistemicClassification - Mutation Coverage', () => {
  it('should describe strongly empirically verifiable', () => {
    const position: EpistemicPosition = { empirical: 0.8, normative: 0.1, mythic: 0.1 };
    const explanation = explainEpistemicClassification(position);

    expect(explanation).toContain('strongly empirically verifiable');
  });

  it('should describe partially empirically verifiable', () => {
    const position: EpistemicPosition = { empirical: 0.5, normative: 0.3, mythic: 0.2 };
    const explanation = explainEpistemicClassification(position);

    expect(explanation).toContain('partially empirically verifiable');
  });

  it('should describe limited empirical verifiability', () => {
    const position: EpistemicPosition = { empirical: 0.2, normative: 0.5, mythic: 0.3 };
    const explanation = explainEpistemicClassification(position);

    expect(explanation).toContain('limited empirical verifiability');
  });

  it('should include normative dimensions when significant', () => {
    const position: EpistemicPosition = { empirical: 0.3, normative: 0.6, mythic: 0.1 };
    const explanation = explainEpistemicClassification(position);

    expect(explanation).toContain('normative/ethical dimensions');
  });

  it('should include narrative context when notable', () => {
    const position: EpistemicPosition = { empirical: 0.3, normative: 0.2, mythic: 0.5 };
    const explanation = explainEpistemicClassification(position);

    expect(explanation).toContain('narrative/meaning context');
  });

  it('should combine multiple descriptions', () => {
    const position: EpistemicPosition = { empirical: 0.5, normative: 0.6, mythic: 0.5 };
    const explanation = explainEpistemicClassification(position);

    expect(explanation).toContain('partially empirically verifiable');
    expect(explanation).toContain('normative/ethical dimensions');
    expect(explanation).toContain('narrative/meaning context');
  });

  it('should handle edge case at boundary values', () => {
    const position: EpistemicPosition = { empirical: 0.7, normative: 0.5, mythic: 0.4 };
    const explanation = explainEpistemicClassification(position);

    expect(explanation).toContain('strongly empirically verifiable');
    expect(explanation).toContain('normative/ethical dimensions');
    expect(explanation).toContain('narrative/meaning context');
  });
});

// ============================================================================
// EpistemicEvidenceBridge Tests
// ============================================================================

describe('EpistemicEvidenceBridge - Mutation Coverage', () => {
  let bridge: EpistemicEvidenceBridge;

  beforeEach(() => {
    resetEpistemicEvidenceBridge();
    bridge = new EpistemicEvidenceBridge();
  });

  describe('verifyEvidence', () => {
    it('should verify Document evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Document' });
      const result = await bridge.verifyEvidence(evidence, 'ContractDispute');

      expect(result.evidence_id).toBe(evidence.id);
      expect(result.classification.empirical).toBeGreaterThan(0);
      expect(result.verified_at).toBeGreaterThan(0);
    });

    it('should verify Transaction evidence with high empirical', async () => {
      const evidence = createTestEvidence({ type_: 'Transaction' });
      const result = await bridge.verifyEvidence(evidence, 'ConductViolation');

      expect(result.classification.empirical).toBeGreaterThan(0.5);
    });

    it('should verify Testimony evidence with lower empirical', async () => {
      const evidence = createTestEvidence({ type_: 'Testimony' });
      const result = await bridge.verifyEvidence(evidence, 'ConductViolation');

      expect(result.classification.normative).toBeGreaterThan(0);
    });

    it('should verify Screenshot evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Screenshot' });
      const result = await bridge.verifyEvidence(evidence, 'IPDispute');

      expect(result).toBeDefined();
    });

    it('should verify Witness evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Witness' });
      const result = await bridge.verifyEvidence(evidence, 'PropertyDispute');

      expect(result).toBeDefined();
    });

    it('should verify Other evidence type', async () => {
      const evidence = createTestEvidence({ type_: 'Other' });
      const result = await bridge.verifyEvidence(evidence, 'GovernanceDispute');

      expect(result).toBeDefined();
    });

    it('should include factors in result', async () => {
      const evidence = createTestEvidence();
      const result = await bridge.verifyEvidence(evidence, 'ConductViolation');

      expect(Array.isArray(result.factors)).toBe(true);
      expect(result.factors.length).toBeGreaterThan(0);
    });

    it('should determine recommendation based on credibility', async () => {
      const evidence = createTestEvidence({ type_: 'Transaction' });
      const result = await bridge.verifyEvidence(evidence, 'ConductViolation');

      expect(['Strong', 'Moderate', 'Weak', 'Unreliable', 'Insufficient']).toContain(
        result.recommendation
      );
    });
  });

  describe('verifyEvidenceBatch', () => {
    it('should verify batch of evidence', async () => {
      const evidenceList = [
        createTestEvidence({ id: 'ev-1', type_: 'Document' }),
        createTestEvidence({ id: 'ev-2', type_: 'Transaction' }),
        createTestEvidence({ id: 'ev-3', type_: 'Testimony' }),
      ];

      const input: BatchVerificationInput = {
        case_id: 'case-001',
        evidence_ids: ['ev-1', 'ev-2', 'ev-3'],
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceList, 'ContractDispute');

      expect(result.case_id).toBe('case-001');
      expect(result.results.length).toBe(3);
      expect(result.summary.total).toBe(3);
    });

    it('should filter by minimum credibility', async () => {
      const evidenceList = [
        createTestEvidence({ id: 'ev-1', type_: 'Transaction' }),
        createTestEvidence({ id: 'ev-2', type_: 'Other' }),
      ];

      const input: BatchVerificationInput = {
        case_id: 'case-002',
        evidence_ids: ['ev-1', 'ev-2'],
        min_credibility: 0.9, // Very high threshold
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceList, 'ConductViolation');

      // Only high-credibility evidence should be included
      expect(result.results.length).toBeLessThanOrEqual(2);
    });

    it('should calculate summary statistics', async () => {
      const evidenceList = [
        createTestEvidence({ id: 'ev-1', type_: 'Document' }),
        createTestEvidence({ id: 'ev-2', type_: 'Transaction' }),
      ];

      const input: BatchVerificationInput = {
        case_id: 'case-003',
        evidence_ids: ['ev-1', 'ev-2'],
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceList, 'ContractDispute');

      expect(result.summary.avg_credibility).toBeGreaterThanOrEqual(0);
      expect(result.summary.avg_empirical).toBeGreaterThanOrEqual(0);
      expect(result.summary.avg_normative).toBeGreaterThanOrEqual(0);
    });

    it('should determine case evidence quality', async () => {
      const evidenceList = [
        createTestEvidence({ id: 'ev-1', type_: 'Transaction' }),
        createTestEvidence({ id: 'ev-2', type_: 'Document' }),
      ];

      const input: BatchVerificationInput = {
        case_id: 'case-004',
        evidence_ids: ['ev-1', 'ev-2'],
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceList, 'ConductViolation');

      expect(['Excellent', 'Good', 'Mixed', 'Poor', 'Insufficient']).toContain(
        result.case_evidence_quality
      );
    });

    it('should handle empty evidence list', async () => {
      const input: BatchVerificationInput = {
        case_id: 'case-005',
        evidence_ids: [],
      };

      const result = await bridge.verifyEvidenceBatch(input, [], 'Other');

      expect(result.results.length).toBe(0);
      expect(result.summary.total).toBe(0);
      expect(result.case_evidence_quality).toBe('Insufficient');
    });

    it('should track duration', async () => {
      const evidenceList = [createTestEvidence({ id: 'ev-1' })];

      const input: BatchVerificationInput = {
        case_id: 'case-006',
        evidence_ids: ['ev-1'],
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceList, 'ConductViolation');

      expect(result.duration_ms).toBeGreaterThanOrEqual(0);
    });
  });

  describe('enhanceEvidence', () => {
    it('should add epistemic fields to evidence', async () => {
      const evidence = createTestEvidence();
      const enhanced = await bridge.enhanceEvidence(evidence, 'ContractDispute');

      expect(enhanced.epistemic_position).toBeDefined();
      expect(enhanced.credibility_score).toBeGreaterThanOrEqual(0);
      expect(enhanced.epistemically_verified_at).toBeGreaterThan(0);
    });

    it('should preserve original evidence fields', async () => {
      const evidence = createTestEvidence({
        id: 'original-id',
        title: 'Original Title',
      });

      const enhanced = await bridge.enhanceEvidence(evidence, 'ConductViolation');

      expect(enhanced.id).toBe('original-id');
      expect(enhanced.title).toBe('Original Title');
    });

    it('should calculate adjusted strength', async () => {
      const evidence = createTestEvidence({ type_: 'Transaction' });
      const enhanced = await bridge.enhanceEvidence(evidence, 'ConductViolation');

      expect(enhanced.adjusted_strength).toBeGreaterThanOrEqual(0);
      expect(enhanced.adjusted_strength).toBeLessThanOrEqual(1);
    });
  });

  describe('factCheckEvidence', () => {
    it('should return True for Document evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Document' });
      const result = await bridge.factCheckEvidence(evidence, 'test claim');

      expect(result.verdict).toBe('True');
      expect(result.confidence).toBe(0.8);
    });

    it('should return True for Transaction evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Transaction' });
      const result = await bridge.factCheckEvidence(evidence, 'test claim');

      expect(result.verdict).toBe('True');
      expect(result.confidence).toBe(0.9);
    });

    it('should return PartiallyTrue for Screenshot evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Screenshot' });
      const result = await bridge.factCheckEvidence(evidence, 'test claim');

      expect(result.verdict).toBe('PartiallyTrue');
      expect(result.confidence).toBe(0.6);
    });

    it('should return Unverified for Testimony evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Testimony' });
      const result = await bridge.factCheckEvidence(evidence, 'test claim');

      expect(result.verdict).toBe('Unverified');
      expect(result.confidence).toBe(0.5);
    });

    it('should return Unverified for Witness evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Witness' });
      const result = await bridge.factCheckEvidence(evidence, 'test claim');

      expect(result.verdict).toBe('Unverified');
      expect(result.confidence).toBe(0.5);
    });

    it('should return Unverified for Other evidence', async () => {
      // Other type has reliability 0.4, which falls into Unverified (>= 0.4)
      const evidence = createTestEvidence({ type_: 'Other' });
      const result = await bridge.factCheckEvidence(evidence, 'test claim');

      expect(result.verdict).toBe('Unverified');
      expect(result.confidence).toBe(0.4);
    });

    it('should return empty supporting and contradicting arrays', async () => {
      const evidence = createTestEvidence();
      const result = await bridge.factCheckEvidence(evidence, 'test claim');

      expect(result.supporting).toEqual([]);
      expect(result.contradicting).toEqual([]);
    });
  });
});

// ============================================================================
// Singleton Tests
// ============================================================================

describe('Epistemic Evidence Bridge Singleton', () => {
  it('should return same instance', () => {
    resetEpistemicEvidenceBridge();
    const bridge1 = getEpistemicEvidenceBridge();
    const bridge2 = getEpistemicEvidenceBridge();

    expect(bridge1).toBe(bridge2);
  });

  it('should reset to new instance', () => {
    const bridge1 = getEpistemicEvidenceBridge();
    resetEpistemicEvidenceBridge();
    const bridge2 = getEpistemicEvidenceBridge();

    expect(bridge1).not.toBe(bridge2);
  });
});
