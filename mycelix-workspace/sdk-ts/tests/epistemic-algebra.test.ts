// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Algebra Tests
 *
 * Tests for mathematical operations combining claim confidences
 * with intellectual honesty.
 */

import { describe, it, expect } from 'vitest';
import {
  conjunction,
  disjunction,
  applyELevelCeiling,
  chainDegradation,
  corroborate,
  resolveConflicts,
  combineClaims,
  combineChain,
  propagateUncertainty,
  isValidConfidence,
  suggestConfidence,
  formatConfidence,
  E_LEVEL_CONFIDENCE_CEILING,
  DEFAULT_CHAIN_DEGRADATION,
  MAX_CORROBORATION_BOOST,
} from '../src/epistemic/algebra.js';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  createClaim,
} from '../src/epistemic/index.js';

describe('Epistemic Algebra - Conjunction (Weakest Link)', () => {
  it('should return minimum confidence from components', () => {
    const result = conjunction([0.9, 0.6, 0.8]);

    expect(result.confidence).toBe(0.6);
    expect(result.method).toBe('conjunction');
    expect(result.components).toEqual([0.9, 0.6, 0.8]);
  });

  it('should identify limiting factor', () => {
    const result = conjunction([0.95, 0.42, 0.88]);

    expect(result.confidence).toBe(0.42);
    expect(result.limitingFactor).toContain('Component 2');
    expect(result.limitingFactor).toContain('42.0%');
  });

  it('should handle single component', () => {
    const result = conjunction([0.75]);

    expect(result.confidence).toBe(0.75);
  });

  it('should handle empty array', () => {
    const result = conjunction([]);

    expect(result.confidence).toBe(0);
    expect(result.explanation).toContain('No components');
  });

  it('should work with two identical confidences', () => {
    const result = conjunction([0.8, 0.8, 0.8]);

    expect(result.confidence).toBe(0.8);
  });
});

describe('Epistemic Algebra - Disjunction (Maximum)', () => {
  it('should return maximum confidence from components', () => {
    const result = disjunction([0.4, 0.8, 0.6]);

    expect(result.confidence).toBe(0.8);
  });

  it('should identify best source', () => {
    const result = disjunction([0.3, 0.9, 0.5]);

    expect(result.confidence).toBe(0.9);
    expect(result.limitingFactor).toContain('Component 2');
  });

  it('should handle empty array', () => {
    const result = disjunction([]);

    expect(result.confidence).toBe(0);
  });
});

describe('Epistemic Algebra - E-Level Ceiling', () => {
  it('should cap E0_Unverified at 40%', () => {
    const result = applyELevelCeiling(0.95, EmpiricalLevel.E0_Unverified);

    expect(result.confidence).toBe(0.4);
    expect(result.ceilingApplied).toBe(true);
    expect(result.ceiling).toBe(0.4);
  });

  it('should cap E1_Testimonial at 60%', () => {
    const result = applyELevelCeiling(0.85, EmpiricalLevel.E1_Testimonial);

    expect(result.confidence).toBe(0.6);
    expect(result.ceilingApplied).toBe(true);
  });

  it('should cap E2_PrivateVerify at 80%', () => {
    const result = applyELevelCeiling(0.99, EmpiricalLevel.E2_PrivateVerify);

    expect(result.confidence).toBe(0.8);
    expect(result.ceilingApplied).toBe(true);
  });

  it('should cap E3_Cryptographic at 95%', () => {
    const result = applyELevelCeiling(0.99, EmpiricalLevel.E3_Cryptographic);

    expect(result.confidence).toBe(0.95);
    expect(result.ceilingApplied).toBe(true);
  });

  it('should cap E4_Consensus at 99%', () => {
    const result = applyELevelCeiling(1.0, EmpiricalLevel.E4_Consensus);

    expect(result.confidence).toBe(0.99);
    expect(result.ceilingApplied).toBe(true);
  });

  it('should not modify confidence below ceiling', () => {
    const result = applyELevelCeiling(0.5, EmpiricalLevel.E3_Cryptographic);

    expect(result.confidence).toBe(0.5);
    expect(result.ceilingApplied).toBe(false);
  });

  it('should verify all ceiling values are correct', () => {
    expect(E_LEVEL_CONFIDENCE_CEILING[EmpiricalLevel.E0_Unverified]).toBe(0.4);
    expect(E_LEVEL_CONFIDENCE_CEILING[EmpiricalLevel.E1_Testimonial]).toBe(0.6);
    expect(E_LEVEL_CONFIDENCE_CEILING[EmpiricalLevel.E2_PrivateVerify]).toBe(0.8);
    expect(E_LEVEL_CONFIDENCE_CEILING[EmpiricalLevel.E3_Cryptographic]).toBe(0.95);
    expect(E_LEVEL_CONFIDENCE_CEILING[EmpiricalLevel.E4_Consensus]).toBe(0.99);
  });
});

describe('Epistemic Algebra - Chain Degradation', () => {
  it('should degrade confidence through inference chain', () => {
    const result = chainDegradation(0.9, 3);

    // 0.9 * 0.95^3 ≈ 0.77
    expect(result.finalConfidence).toBeCloseTo(0.9 * Math.pow(0.95, 3), 5);
    expect(result.startingConfidence).toBe(0.9);
    expect(result.chainLength).toBe(3);
  });

  it('should not degrade for chain length 0', () => {
    const result = chainDegradation(0.9, 0);

    expect(result.finalConfidence).toBe(0.9);
  });

  it('should use default degradation factor', () => {
    expect(DEFAULT_CHAIN_DEGRADATION).toBe(0.95);
  });

  it('should allow custom degradation factor', () => {
    const result = chainDegradation(1.0, 2, 0.9);

    expect(result.finalConfidence).toBeCloseTo(0.81, 5);
    expect(result.degradationFactor).toBe(0.9);
  });

  it('should warn about long chains', () => {
    const result = chainDegradation(0.9, 6);

    expect(result.warning).toBeDefined();
    expect(result.warning).toContain('Long inference chain');
  });

  it('should significantly reduce confidence over long chains', () => {
    const result = chainDegradation(0.9, 10);

    // 0.9 * 0.95^10 ≈ 0.54, which is a significant reduction
    expect(result.finalConfidence).toBeLessThan(0.6);
    expect(result.finalConfidence).toBeGreaterThan(0.4);
  });

  it('should throw on negative chain length', () => {
    expect(() => chainDegradation(0.9, -1)).toThrow('Chain length cannot be negative');
  });
});

describe('Epistemic Algebra - Corroboration', () => {
  it('should boost confidence with independent sources', () => {
    const result = corroborate({
      independent: true,
      sourceCount: 3,
      confidences: [0.7, 0.7, 0.7],
    });

    expect(result.confidence).toBeGreaterThan(0.7);
    expect(result.method).toBe('corroboration');
  });

  it('should have diminishing returns', () => {
    const twoSources = corroborate({
      independent: true,
      sourceCount: 2,
      confidences: [0.6, 0.6],
    });

    const threeSources = corroborate({
      independent: true,
      sourceCount: 3,
      confidences: [0.6, 0.6, 0.6],
    });

    const fiveSources = corroborate({
      independent: true,
      sourceCount: 5,
      confidences: [0.6, 0.6, 0.6, 0.6, 0.6],
    });

    // Each additional source adds less boost
    const boost2to3 = threeSources.confidence - twoSources.confidence;
    const boost3to5 = (fiveSources.confidence - threeSources.confidence) / 2;

    expect(boost3to5).toBeLessThan(boost2to3);
  });

  it('should have lower boost for non-independent sources', () => {
    const independent = corroborate({
      independent: true,
      sourceCount: 3,
      confidences: [0.7, 0.7, 0.7],
    });

    const nonIndependent = corroborate({
      independent: false,
      sourceCount: 3,
      confidences: [0.7, 0.7, 0.7],
    });

    expect(nonIndependent.confidence).toBeLessThan(independent.confidence);
    expect(nonIndependent.explanation).toContain('non-independent');
  });

  it('should apply contradiction penalty', () => {
    const noContradiction = corroborate({
      independent: true,
      sourceCount: 3,
      confidences: [0.7, 0.7, 0.7],
      contradictionCount: 0,
    });

    const withContradiction = corroborate({
      independent: true,
      sourceCount: 3,
      confidences: [0.7, 0.7, 0.7],
      contradictionCount: 1,
    });

    expect(withContradiction.confidence).toBeLessThan(noContradiction.confidence);
    expect(withContradiction.explanation).toContain('Penalty');
  });

  it('should not exceed maximum boost', () => {
    const result = corroborate({
      independent: true,
      sourceCount: 100, // Many sources
      confidences: Array(100).fill(0.8),
    });

    expect(result.confidence).toBeLessThanOrEqual(0.99);
  });

  it('should return single source confidence unchanged', () => {
    const result = corroborate({
      independent: true,
      sourceCount: 1,
      confidences: [0.75],
    });

    expect(result.confidence).toBe(0.75);
    expect(result.explanation).toContain('Single source');
  });

  it('should handle empty sources', () => {
    const result = corroborate({
      independent: true,
      sourceCount: 0,
      confidences: [],
    });

    expect(result.confidence).toBe(0);
  });
});

describe('Epistemic Algebra - Conflict Resolution', () => {
  it('should handle single claim without conflict', () => {
    const result = resolveConflicts([{ id: 'a', confidence: 0.8, content: 'test' }]);

    expect(result.confidence).toBe(0.8);
    expect(result.conflicts).toHaveLength(0);
    expect(result.totalPenalty).toBe(0);
  });

  it('should detect conflicts when confidences differ significantly', () => {
    const result = resolveConflicts([
      { id: 'a', confidence: 0.9, content: 'claim A' },
      { id: 'b', confidence: 0.4, content: 'claim B' },
    ]);

    expect(result.conflicts.length).toBeGreaterThan(0);
    expect(result.totalPenalty).toBeGreaterThan(0);
  });

  it('should not detect conflicts for similar confidences', () => {
    const result = resolveConflicts([
      { id: 'a', confidence: 0.7, content: 'claim A' },
      { id: 'b', confidence: 0.75, content: 'claim B' },
    ]);

    expect(result.conflicts).toHaveLength(0);
  });

  it('should recommend human adjudication for significant disagreements', () => {
    const result = resolveConflicts([
      { id: 'a', confidence: 0.95, content: 'claim A' },
      { id: 'b', confidence: 0.2, content: 'claim B' },
      { id: 'c', confidence: 0.1, content: 'claim C' },
    ]);

    expect(result.recommendation).toContain('human adjudication');
  });
});

describe('Epistemic Algebra - Combine Claims', () => {
  const createTestClaim = (eLevel: EmpiricalLevel) =>
    createClaim('test', eLevel, NormativeLevel.N1_Communal, MaterialityLevel.M1_Temporal, 'test');

  it('should combine claims using conjunction mode', () => {
    const claims = [
      createTestClaim(EmpiricalLevel.E3_Cryptographic),
      createTestClaim(EmpiricalLevel.E2_PrivateVerify),
    ];

    const result = combineClaims(claims, { mode: 'conjunction' });

    // Should take minimum ceiling (E2 = 0.8)
    expect(result.confidence).toBe(0.8);
    expect(result.method).toBe('conjunction');
  });

  it('should combine claims using disjunction mode', () => {
    const claims = [
      createTestClaim(EmpiricalLevel.E1_Testimonial),
      createTestClaim(EmpiricalLevel.E3_Cryptographic),
    ];

    const result = combineClaims(claims, { mode: 'disjunction', applyECeiling: false });

    // Should take maximum ceiling (E3 = 0.95)
    expect(result.confidence).toBe(0.95);
  });

  it('should apply E-level ceiling by default', () => {
    const claims = [
      createTestClaim(EmpiricalLevel.E1_Testimonial), // ceiling 0.6
      createTestClaim(EmpiricalLevel.E2_PrivateVerify), // ceiling 0.8
    ];

    const result = combineClaims(claims, { mode: 'corroboration', areIndependent: true });

    // Even with corroboration boost, ceiling is lowest E-level (E1 = 0.6)
    expect(result.confidence).toBeLessThanOrEqual(0.6);
    expect(result.ceilingApplied).toBe(true);
  });

  it('should handle empty claims array', () => {
    const result = combineClaims([], { mode: 'conjunction' });

    expect(result.confidence).toBe(0);
  });
});

describe('Epistemic Algebra - Combine Chain', () => {
  const createTestClaim = (eLevel: EmpiricalLevel) =>
    createClaim('test', eLevel, NormativeLevel.N1_Communal, MaterialityLevel.M1_Temporal, 'test');

  it('should apply chain degradation across inference steps', () => {
    const claims = [
      createTestClaim(EmpiricalLevel.E3_Cryptographic),
      createTestClaim(EmpiricalLevel.E3_Cryptographic),
      createTestClaim(EmpiricalLevel.E3_Cryptographic),
    ];

    const result = combineChain(claims);

    // 3 claims = 2 inference steps
    // 0.95 * 0.95^2 ≈ 0.857
    expect(result.confidence).toBeLessThan(0.95);
    expect(result.method).toBe('chain');
  });

  it('should include chain warning for long chains', () => {
    const claims = Array(8)
      .fill(null)
      .map(() => createTestClaim(EmpiricalLevel.E3_Cryptographic));

    const result = combineChain(claims);

    // 8 claims = 7 inference steps (>5 triggers warning)
    expect(result.chainWarning).toBeDefined();
  });

  it('should handle single claim (no inference)', () => {
    const claims = [createTestClaim(EmpiricalLevel.E3_Cryptographic)];

    const result = combineChain(claims);

    expect(result.confidence).toBe(0.95); // Just the E3 ceiling
  });
});

describe('Epistemic Algebra - Uncertainty Propagation', () => {
  it('should reduce confidence for required dependencies', () => {
    const result = propagateUncertainty([
      { claimId: 'a', confidence: 0.8, dependencyType: 'requires' },
      { claimId: 'b', confidence: 0.9, dependencyType: 'requires' },
    ]);

    // 0.8 * 0.9 = 0.72
    expect(result.effectiveConfidence).toBeCloseTo(0.72, 2);
  });

  it('should boost confidence for supporting dependencies', () => {
    const result = propagateUncertainty([
      { claimId: 'a', confidence: 0.7, dependencyType: 'supports' },
    ]);

    // Supporting dependencies provide a boost, but capped at reasonable levels
    expect(result.effectiveConfidence).toBeGreaterThanOrEqual(0.7);
    expect(result.effectiveConfidence).toBeLessThanOrEqual(1);
  });

  it('should reduce confidence for weakening dependencies', () => {
    const result = propagateUncertainty([
      { claimId: 'a', confidence: 0.5, dependencyType: 'weakens' },
    ]);

    expect(result.effectiveConfidence).toBeLessThan(1);
  });

  it('should track uncertainty contributions', () => {
    const result = propagateUncertainty([
      { claimId: 'a', confidence: 0.8, dependencyType: 'requires' },
    ]);

    expect(result.uncertaintyContributions.has('a')).toBe(true);
  });

  it('should handle empty dependencies', () => {
    const result = propagateUncertainty([]);

    expect(result.effectiveConfidence).toBe(1);
    expect(result.totalUncertainty).toBe(0);
  });
});

describe('Epistemic Algebra - Validation', () => {
  it('should validate confidence within E-level ceiling', () => {
    const result = isValidConfidence(0.5, EmpiricalLevel.E2_PrivateVerify);

    expect(result.valid).toBe(true);
    expect(result.maxAllowed).toBe(0.8);
  });

  it('should reject confidence exceeding E-level ceiling', () => {
    const result = isValidConfidence(0.9, EmpiricalLevel.E1_Testimonial);

    expect(result.valid).toBe(false);
    expect(result.reason).toContain('exceeds');
    expect(result.reason).toContain('E1');
  });

  it('should reject confidence outside 0-1 range', () => {
    expect(isValidConfidence(-0.1, EmpiricalLevel.E3_Cryptographic).valid).toBe(false);
    expect(isValidConfidence(1.5, EmpiricalLevel.E3_Cryptographic).valid).toBe(false);
  });
});

describe('Epistemic Algebra - Suggest Confidence', () => {
  it('should suggest weak confidence at 40% of ceiling', () => {
    const result = suggestConfidence(EmpiricalLevel.E3_Cryptographic, 'weak');

    expect(result).toBeCloseTo(0.95 * 0.4, 2);
  });

  it('should suggest moderate confidence at 70% of ceiling', () => {
    const result = suggestConfidence(EmpiricalLevel.E3_Cryptographic, 'moderate');

    expect(result).toBeCloseTo(0.95 * 0.7, 2);
  });

  it('should suggest strong confidence at 90% of ceiling', () => {
    const result = suggestConfidence(EmpiricalLevel.E3_Cryptographic, 'strong');

    expect(result).toBeCloseTo(0.95 * 0.9, 2);
  });

  it('should respect E-level ceiling', () => {
    const result = suggestConfidence(EmpiricalLevel.E0_Unverified, 'strong');

    expect(result).toBeLessThanOrEqual(0.4); // E0 ceiling
  });
});

describe('Epistemic Algebra - Format Confidence', () => {
  it('should format extremely high confidence', () => {
    expect(formatConfidence(0.97)).toContain('extremely high');
  });

  it('should format high confidence', () => {
    expect(formatConfidence(0.88)).toContain('high confidence');
  });

  it('should format moderate confidence', () => {
    expect(formatConfidence(0.75)).toContain('moderate');
  });

  it('should format low-moderate confidence', () => {
    expect(formatConfidence(0.55)).toContain('low-moderate');
  });

  it('should format low confidence', () => {
    expect(formatConfidence(0.35)).toContain('low confidence');
  });

  it('should format very low confidence', () => {
    expect(formatConfidence(0.2)).toContain('very low');
  });
});
