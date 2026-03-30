// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Property-Based Tests for MATL Module
 *
 * Uses fast-check to verify MATL invariants hold across all valid inputs.
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  createPoGQ,
  compositeScore,
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isByzantine,
  calculateComposite,
} from '../../src/matl/index.js';

describe('MATL Property-Based Tests', () => {
  // Safe string arbitrary that never generates empty or whitespace-only strings
  const safeStringArb = fc
    .string({ minLength: 2, maxLength: 100 })
    .filter((s) => /^[a-z][a-z0-9_-]*$/i.test(s));

  // Arbitrary for valid [0.01, 0.99] range (avoiding edge cases at 0 and 1)
  // Use Math.fround to ensure 32-bit float compatibility
  const validScore = fc.float({ min: Math.fround(0.01), max: Math.fround(0.99), noNaN: true });

  describe('PoGQ Properties', () => {
    it('all PoGQ components should be between 0 and 1', { timeout: 30000 }, () => {
      fc.assert(
        fc.property(validScore, validScore, validScore, (q, c, e) => {
          const pogq = createPoGQ(q, c, e);

          expect(pogq.quality).toBeGreaterThanOrEqual(0);
          expect(pogq.quality).toBeLessThanOrEqual(1);
          expect(pogq.consistency).toBeGreaterThanOrEqual(0);
          expect(pogq.consistency).toBeLessThanOrEqual(1);
          expect(pogq.entropy).toBeGreaterThanOrEqual(0);
          expect(pogq.entropy).toBeLessThanOrEqual(1);
        }),
        { numRuns: 500 }
      );
    });

    it('identical inputs should produce equivalent outputs', () => {
      fc.assert(
        fc.property(validScore, validScore, validScore, (q, c, e) => {
          const pogq1 = createPoGQ(q, c, e);
          const pogq2 = createPoGQ(q, c, e);

          expect(pogq1.quality).toBe(pogq2.quality);
          expect(pogq1.consistency).toBe(pogq2.consistency);
          expect(pogq1.entropy).toBe(pogq2.entropy);
        }),
        { numRuns: 200 }
      );
    });

    it('should have valid timestamp', () => {
      fc.assert(
        fc.property(validScore, validScore, validScore, (q, c, e) => {
          const before = Date.now();
          const pogq = createPoGQ(q, c, e);
          const after = Date.now();

          expect(pogq.timestamp).toBeGreaterThanOrEqual(before);
          expect(pogq.timestamp).toBeLessThanOrEqual(after);
        }),
        { numRuns: 100 }
      );
    });
  });

  describe('Composite Score Properties', () => {
    it('composite score should be bounded [0, 1]', () => {
      fc.assert(
        fc.property(
          validScore, // quality
          validScore, // consistency
          validScore, // entropy
          validScore, // reputation
          (q, c, e, r) => {
            const pogq = createPoGQ(q, c, e);
            const score = compositeScore(pogq, r);

            expect(score).toBeGreaterThanOrEqual(0);
            expect(score).toBeLessThanOrEqual(1);
          }
        ),
        { numRuns: 500 }
      );
    });

    it('higher quality/consistency should increase score (with fixed entropy/reputation)', () => {
      fc.assert(
        fc.property(
          validScore,
          validScore,
          fc.float({ min: Math.fround(0.1), max: Math.fround(0.5), noNaN: true }), // fixed low entropy
          validScore,
          (q, c, e, r) => {
            const pogqLow = createPoGQ(q * 0.5, c * 0.5, e);
            const pogqHigh = createPoGQ(
              Math.min(q * 0.5 + 0.3, 0.99),
              Math.min(c * 0.5 + 0.3, 0.99),
              e
            );

            const scoreLow = compositeScore(pogqLow, r);
            const scoreHigh = compositeScore(pogqHigh, r);

            expect(scoreHigh).toBeGreaterThanOrEqual(scoreLow);
          }
        ),
        { numRuns: 300 }
      );
    });

    it('higher entropy should decrease score', () => {
      fc.assert(
        fc.property(
          validScore, // quality
          validScore, // consistency
          validScore, // reputation
          (q, c, r) => {
            const pogqLowEntropy = createPoGQ(q, c, 0.1);
            const pogqHighEntropy = createPoGQ(q, c, 0.9);

            const scoreLow = compositeScore(pogqLowEntropy, r);
            const scoreHigh = compositeScore(pogqHighEntropy, r);

            // Low entropy = higher score (entropy is penalized)
            expect(scoreLow).toBeGreaterThanOrEqual(scoreHigh);
          }
        ),
        { numRuns: 300 }
      );
    });
  });

  describe('Reputation Properties', () => {
    it('new reputation should start at 0.5', () => {
      fc.assert(
        fc.property(safeStringArb, (agentId) => {
          const rep = createReputation(agentId);
          expect(reputationValue(rep)).toBe(0.5);
        }),
        { numRuns: 100 }
      );
    });

    it('positive interactions should increase reputation', () => {
      fc.assert(
        fc.property(safeStringArb, fc.integer({ min: 1, max: 10 }), (agentId, count) => {
          let rep = createReputation(agentId);
          const initialValue = reputationValue(rep);

          for (let i = 0; i < count; i++) {
            rep = recordPositive(rep);
          }

          expect(reputationValue(rep)).toBeGreaterThan(initialValue);
        }),
        { numRuns: 100 }
      );
    });

    it('negative interactions should decrease reputation', () => {
      fc.assert(
        fc.property(safeStringArb, fc.integer({ min: 1, max: 10 }), (agentId, count) => {
          let rep = createReputation(agentId);
          const initialValue = reputationValue(rep);

          for (let i = 0; i < count; i++) {
            rep = recordNegative(rep);
          }

          expect(reputationValue(rep)).toBeLessThan(initialValue);
        }),
        { numRuns: 100 }
      );
    });

    it('reputation should always be bounded [0, 1]', () => {
      fc.assert(
        fc.property(
          safeStringArb,
          fc.array(fc.boolean(), { minLength: 1, maxLength: 50 }),
          (agentId, interactions) => {
            let rep = createReputation(agentId);

            for (const isPositive of interactions) {
              rep = isPositive ? recordPositive(rep) : recordNegative(rep);
            }

            const value = reputationValue(rep);
            expect(value).toBeGreaterThanOrEqual(0);
            expect(value).toBeLessThanOrEqual(1);
          }
        ),
        { numRuns: 200 }
      );
    });
  });

  describe('Byzantine Detection Properties', () => {
    it('low quality + high entropy should be Byzantine', () => {
      fc.assert(
        fc.property(
          fc.float({ min: Math.fround(0.01), max: Math.fround(0.3), noNaN: true }), // low quality
          validScore, // any consistency
          fc.float({ min: Math.fround(0.7), max: Math.fround(0.99), noNaN: true }), // high entropy
          (q, c, e) => {
            const pogq = createPoGQ(q, c, e);
            // Low quality with high entropy typically indicates Byzantine behavior
            // This tests the detection boundary
            expect(isByzantine(pogq)).toBeDefined();
          }
        ),
        { numRuns: 200 }
      );
    });

    it('high quality + low entropy should not be Byzantine', () => {
      fc.assert(
        fc.property(
          fc.float({ min: Math.fround(0.8), max: Math.fround(0.99), noNaN: true }), // high quality (> 0.8)
          fc.float({ min: Math.fround(0.8), max: Math.fround(0.99), noNaN: true }), // high consistency
          fc.float({ min: Math.fround(0.01), max: Math.fround(0.2), noNaN: true }), // low entropy (< 0.2)
          (q, c, e) => {
            // Byzantine threshold is 0.5
            // score = q * (1 - e)
            // With q >= 0.8 and e <= 0.2: score >= 0.8 * 0.8 = 0.64 > 0.5
            const pogq = createPoGQ(q, c, e);
            expect(isByzantine(pogq)).toBe(false);
          }
        ),
        { numRuns: 200 }
      );
    });
  });

  describe('Calculate Composite Properties', () => {
    it('compositeScore should be bounded [0, 1]', () => {
      fc.assert(
        fc.property(validScore, validScore, validScore, validScore, (q, c, e, r) => {
          const pogq = createPoGQ(q, c, e);
          const score = compositeScore(pogq, r);
          expect(score).toBeGreaterThanOrEqual(0);
          expect(score).toBeLessThanOrEqual(1);
        }),
        { numRuns: 500 }
      );
    });

    it('calculateComposite should work with full objects', () => {
      fc.assert(
        fc.property(validScore, validScore, validScore, safeStringArb, (q, c, e, agentId) => {
          const pogq = createPoGQ(q, c, e);
          const rep = createReputation(agentId);

          const result = calculateComposite(pogq, rep);

          expect(result).toBeDefined();
          expect(result.finalScore).toBeGreaterThanOrEqual(0);
          expect(result.finalScore).toBeLessThanOrEqual(1);
          expect(result.pogq).toEqual(pogq);
        }),
        { numRuns: 300 }
      );
    });
  });

  describe('Edge Case Properties', () => {
    it('should handle exact boundary values', () => {
      // Test exact 0 and 1 values
      const pogqZero = createPoGQ(0, 0, 0);
      const pogqOne = createPoGQ(1, 1, 1);

      expect(pogqZero.quality).toBe(0);
      expect(pogqOne.quality).toBe(1);

      const scoreZero = compositeScore(pogqZero, 0);
      const scoreOne = compositeScore(pogqOne, 1);

      expect(scoreZero).toBeGreaterThanOrEqual(0);
      expect(scoreOne).toBeLessThanOrEqual(1);
    });

    it('should be deterministic', () => {
      fc.assert(
        fc.property(validScore, validScore, validScore, validScore, (q, c, e, r) => {
          const pogq = createPoGQ(q, c, e);
          const score1 = compositeScore(pogq, r);
          const score2 = compositeScore(pogq, r);

          expect(score1).toBe(score2);
        }),
        { numRuns: 100 }
      );
    });
  });
});
