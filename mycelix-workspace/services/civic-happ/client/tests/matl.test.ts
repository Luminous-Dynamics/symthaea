import { describe, it, expect } from 'vitest';
import { MATL } from '../src/reputation.js';
import type { TrustScore } from '../src/types.js';

function makeTrustScore(overrides: Partial<TrustScore> = {}): TrustScore {
  return {
    agent_pubkey: new Uint8Array(39) as any,
    quality: 0.8,
    consistency: 0.7,
    reputation: 0.6,
    composite: 0.71,
    positive_count: 10,
    negative_count: 2,
    computed_at: Date.now() * 1000,
    confidence: 0.5,
    ...overrides,
  };
}

describe('MATL', () => {
  describe('computeComposite', () => {
    it('applies correct weights: 0.4·Q + 0.3·C + 0.3·R', () => {
      const result = MATL.computeComposite(0.8, 0.7, 0.6);
      const expected = 0.4 * 0.8 + 0.3 * 0.7 + 0.3 * 0.6;
      expect(result).toBeCloseTo(expected, 10);
    });

    it('clamps to 1.0 maximum', () => {
      expect(MATL.computeComposite(1.0, 1.0, 1.0)).toBe(1.0);
      expect(MATL.computeComposite(1.5, 1.5, 1.5)).toBe(1.0);
    });

    it('clamps to 0.0 minimum', () => {
      expect(MATL.computeComposite(0, 0, 0)).toBe(0.0);
      expect(MATL.computeComposite(-1, -1, -1)).toBe(0.0);
    });

    it('produces expected values for known inputs', () => {
      // All equal at 0.5 → 0.4*0.5 + 0.3*0.5 + 0.3*0.5 = 0.5
      expect(MATL.computeComposite(0.5, 0.5, 0.5)).toBeCloseTo(0.5, 10);

      // Quality-dominant: 1.0, 0.0, 0.0 → 0.4
      expect(MATL.computeComposite(1.0, 0.0, 0.0)).toBeCloseTo(0.4, 10);

      // Consistency-dominant: 0.0, 1.0, 0.0 → 0.3
      expect(MATL.computeComposite(0.0, 1.0, 0.0)).toBeCloseTo(0.3, 10);

      // Reputation-dominant: 0.0, 0.0, 1.0 → 0.3
      expect(MATL.computeComposite(0.0, 0.0, 1.0)).toBeCloseTo(0.3, 10);
    });
  });

  describe('isTrustworthy', () => {
    it('returns true when composite >= 0.55 and confidence >= 0.3', () => {
      expect(MATL.isTrustworthy(makeTrustScore({ composite: 0.55, confidence: 0.3 }))).toBe(true);
      expect(MATL.isTrustworthy(makeTrustScore({ composite: 0.9, confidence: 0.8 }))).toBe(true);
    });

    it('returns false when composite < 0.55', () => {
      expect(MATL.isTrustworthy(makeTrustScore({ composite: 0.54, confidence: 0.5 }))).toBe(false);
    });

    it('returns false when confidence < 0.3', () => {
      expect(MATL.isTrustworthy(makeTrustScore({ composite: 0.9, confidence: 0.29 }))).toBe(false);
    });

    it('returns false when both below threshold', () => {
      expect(MATL.isTrustworthy(makeTrustScore({ composite: 0.3, confidence: 0.1 }))).toBe(false);
    });
  });

  describe('getTrustLevel', () => {
    it('returns untrusted when confidence < 0.3', () => {
      expect(MATL.getTrustLevel(makeTrustScore({ composite: 0.9, confidence: 0.2 }))).toBe('untrusted');
    });

    it('returns low when composite < 0.4', () => {
      expect(MATL.getTrustLevel(makeTrustScore({ composite: 0.3, confidence: 0.5 }))).toBe('low');
    });

    it('returns moderate when composite >= 0.4 and < 0.55', () => {
      expect(MATL.getTrustLevel(makeTrustScore({ composite: 0.5, confidence: 0.5 }))).toBe('moderate');
    });

    it('returns high when composite >= 0.55 and < 0.75', () => {
      expect(MATL.getTrustLevel(makeTrustScore({ composite: 0.6, confidence: 0.5 }))).toBe('high');
    });

    it('returns very_high when composite >= 0.75', () => {
      expect(MATL.getTrustLevel(makeTrustScore({ composite: 0.8, confidence: 0.5 }))).toBe('very_high');
    });

    it('checks boundary values exactly', () => {
      expect(MATL.getTrustLevel(makeTrustScore({ composite: 0.4, confidence: 0.3 }))).toBe('moderate');
      expect(MATL.getTrustLevel(makeTrustScore({ composite: 0.55, confidence: 0.3 }))).toBe('high');
      expect(MATL.getTrustLevel(makeTrustScore({ composite: 0.75, confidence: 0.3 }))).toBe('very_high');
    });
  });

  describe('constants', () => {
    it('has correct weight values', () => {
      expect(MATL.QUALITY_WEIGHT).toBe(0.4);
      expect(MATL.CONSISTENCY_WEIGHT).toBe(0.3);
      expect(MATL.REPUTATION_WEIGHT).toBe(0.3);
    });

    it('weights sum to 1.0', () => {
      expect(MATL.QUALITY_WEIGHT + MATL.CONSISTENCY_WEIGHT + MATL.REPUTATION_WEIGHT).toBeCloseTo(1.0, 10);
    });

    it('has correct thresholds', () => {
      expect(MATL.TRUST_THRESHOLD).toBe(0.55);
      expect(MATL.MIN_CONFIDENCE).toBe(0.3);
    });
  });
});
