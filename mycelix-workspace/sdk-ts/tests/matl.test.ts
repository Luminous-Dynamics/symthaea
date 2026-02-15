/**
 * MATL Module Tests
 */

import { describe, it, expect } from 'vitest';
import {
  createPoGQ,
  compositeScore,
  isByzantine,
  isByzantineScore,
  createReputation,
  reputationValue,
  recordPositive,
  recordNegative,
  calculateComposite,
  isTrustworthy,
  isTrustworthyReputation,
  createAdaptiveThreshold,
  observe,
  getThreshold,
  isAnomalous,
  getThresholdStats,
  DEFAULT_BYZANTINE_THRESHOLD,
  MAX_BYZANTINE_TOLERANCE,
} from '../src/matl/index.js';
import { ValidationError, MycelixError } from '../src/errors';

describe('MATL - Proof of Gradient Quality', () => {
  it('should create a PoGQ with valid values', () => {
    const pogq = createPoGQ(0.95, 0.88, 0.12);

    expect(pogq.quality).toBe(0.95);
    expect(pogq.consistency).toBe(0.88);
    expect(pogq.entropy).toBe(0.12);
    expect(pogq.timestamp).toBeLessThanOrEqual(Date.now());
  });

  it('should throw validation error for values outside [0, 1]', () => {
    expect(() => createPoGQ(1.5, 0.5, 0.5)).toThrow(MycelixError);
    expect(() => createPoGQ(0.5, -0.1, 0.5)).toThrow(MycelixError);
    expect(() => createPoGQ(0.5, 0.5, 2.0)).toThrow(MycelixError);
  });

  it('should calculate composite score correctly', () => {
    const pogq = createPoGQ(0.9, 0.8, 0.1);
    const score = compositeScore(pogq, 0.85);

    // 0.4*0.9 + 0.3*0.8 + 0.3*0.85 - 0.1*0.1 = 0.36 + 0.24 + 0.255 - 0.01 = 0.845
    expect(score).toBeCloseTo(0.845, 2);
  });

  it('should detect Byzantine behavior from PoGQ', () => {
    // Low quality + high entropy = Byzantine
    const byzantinePogq = createPoGQ(0.2, 0.3, 0.9);
    expect(isByzantine(byzantinePogq)).toBe(true);

    // High quality + low entropy = Honest
    const honestPogq = createPoGQ(0.9, 0.9, 0.1);
    expect(isByzantine(honestPogq)).toBe(false);
  });

  it('should support custom Byzantine threshold', () => {
    const pogq = createPoGQ(0.6, 0.6, 0.3);

    // Score = 0.6 * (1 - 0.3) = 0.42
    expect(isByzantine(pogq, 0.4)).toBe(false); // Above threshold
    expect(isByzantine(pogq, 0.5)).toBe(true);  // Below threshold
  });

  it('should detect Byzantine behavior from score value', () => {
    expect(isByzantineScore(0.3)).toBe(true);
    expect(isByzantineScore(0.7)).toBe(false);
    expect(isByzantineScore(0.5)).toBe(false);
    expect(isByzantineScore(0.49)).toBe(true);
  });
});

describe('MATL - Reputation Score', () => {
  it('should create reputation with Bayesian priors', () => {
    const rep = createReputation('agent1');

    expect(rep.agentId).toBe('agent1');
    expect(rep.positiveCount).toBe(1);
    expect(rep.negativeCount).toBe(1);
    expect(reputationValue(rep)).toBe(0.5);
  });

  it('should throw for empty agent ID', () => {
    expect(() => createReputation('')).toThrow(MycelixError);
  });

  it('should increase reputation with positive interactions', () => {
    let rep = createReputation('agent1');
    rep = recordPositive(rep);
    rep = recordPositive(rep);

    // (1 + 2) / (1 + 2 + 1) = 3/4 = 0.75
    expect(reputationValue(rep)).toBe(0.75);
  });

  it('should decrease reputation with negative interactions', () => {
    let rep = createReputation('agent1');
    rep = recordNegative(rep);
    rep = recordNegative(rep);

    // 1 / (1 + 1 + 2) = 1/4 = 0.25
    expect(reputationValue(rep)).toBe(0.25);
  });

  it('should check trustworthiness', () => {
    const rep = createReputation('agent1');
    expect(isTrustworthyReputation(rep, 0.5)).toBe(true);
    expect(isTrustworthyReputation(rep, 0.6)).toBe(false);
  });
});

describe('MATL - Composite Score', () => {
  it('should calculate composite with confidence', () => {
    const pogq = createPoGQ(0.9, 0.8, 0.1);
    let rep = createReputation('agent1');
    rep = recordPositive(rep);
    rep = recordPositive(rep);

    const composite = calculateComposite(pogq, rep);

    expect(composite.finalScore).toBeGreaterThan(0.5);
    expect(composite.confidence).toBeGreaterThan(0);
    expect(composite.confidence).toBeLessThanOrEqual(1);
    expect(composite.pogqScore).toBeGreaterThan(0);
    expect(composite.reputationScore).toBe(reputationValue(rep));
  });

  it('should determine trustworthiness', () => {
    const pogq = createPoGQ(0.9, 0.9, 0.1);
    let rep = createReputation('agent1');
    rep = recordPositive(rep);
    rep = recordPositive(rep);
    rep = recordPositive(rep);

    const composite = calculateComposite(pogq, rep);

    expect(isTrustworthy(composite)).toBe(true);
    expect(isTrustworthy(composite, 0.99)).toBe(false);
  });

  it('should include isTrustworthy flag in composite', () => {
    const pogq = createPoGQ(0.9, 0.9, 0.1);
    let rep = createReputation('agent1');
    rep = recordPositive(rep);

    const composite = calculateComposite(pogq, rep);
    expect(composite.isTrustworthy).toBeDefined();
  });

  it('should handle reputation as input to isTrustworthy', () => {
    let rep = createReputation('agent1');
    rep = recordPositive(rep);
    rep = recordPositive(rep);

    // isTrustworthy should work with ReputationScore directly
    expect(isTrustworthy(rep, 0.7)).toBe(true);
    expect(isTrustworthy(rep, 0.8)).toBe(false);
  });

  it('should support configurable compositeScore options', () => {
    const pogq = createPoGQ(0.8, 0.6, 0.2);

    // Default weights (0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.5 - 0.1 * 0.2)
    const defaultScore = compositeScore(pogq, 0.5);

    // Custom weights emphasizing quality
    const customScore = compositeScore(pogq, 0.5, {
      qualityWeight: 0.7,
      consistencyWeight: 0.15,
      reputationWeight: 0.15,
      entropyPenalty: 0.05,
    });

    // Quality-weighted score should be higher since quality is 0.8
    expect(customScore).toBeGreaterThan(defaultScore);
  });

  it('should use defaults for missing options', () => {
    const pogq = createPoGQ(0.8, 0.8, 0.1);

    // Partial options - only override entropy penalty
    const score = compositeScore(pogq, 0.5, { entropyPenalty: 0 });

    // Should be slightly higher than default (no entropy penalty)
    const defaultScore = compositeScore(pogq, 0.5);
    expect(score).toBeGreaterThan(defaultScore);
  });

  it('should reject non-finite reputation', () => {
    const pogq = createPoGQ(0.8, 0.8, 0.1);

    expect(() => compositeScore(pogq, NaN)).toThrow(ValidationError);
    expect(() => compositeScore(pogq, Infinity)).toThrow(ValidationError);
    expect(() => compositeScore(pogq, -Infinity)).toThrow(ValidationError);
  });
});

describe('MATL - Adaptive Threshold', () => {
  it('should create adaptive threshold', () => {
    const at = createAdaptiveThreshold('node1');

    expect(at.nodeId).toBe('node1');
    expect(at.observations).toHaveLength(0);
    expect(at.windowSize).toBe(100);
  });

  it('should throw for invalid parameters', () => {
    expect(() => createAdaptiveThreshold('')).toThrow(MycelixError);
    expect(() => createAdaptiveThreshold('node1', 0)).toThrow(MycelixError);
    expect(() => createAdaptiveThreshold('node1', 100, 1.5)).toThrow(MycelixError);
  });

  it('should return default threshold with no history', () => {
    const at = createAdaptiveThreshold('node1');
    expect(getThreshold(at)).toBe(DEFAULT_BYZANTINE_THRESHOLD);
  });

  it('should adapt threshold based on history', () => {
    let at = createAdaptiveThreshold('node1', 10);

    // Add some observations
    for (let i = 0; i < 10; i++) {
      at = observe(at, 0.8);
    }

    // Threshold should be based on mean - sigma
    const threshold = getThreshold(at);
    expect(threshold).toBeLessThan(0.8);
    expect(threshold).toBeGreaterThan(0);
  });

  it('should detect anomalies', () => {
    let at = createAdaptiveThreshold('node1', 10);

    // Add varied observations to create some variance
    for (let i = 0; i < 5; i++) {
      at = observe(at, 0.9);
    }
    for (let i = 0; i < 5; i++) {
      at = observe(at, 0.8);
    }

    // Mean ~0.85, with variance. Low score should be anomalous
    expect(isAnomalous(at, 0.3)).toBe(true);
    // Score near mean should not be anomalous
    expect(isAnomalous(at, 0.85)).toBe(false);
  });

  it('should respect window size', () => {
    let at = createAdaptiveThreshold('node1', 5);

    for (let i = 0; i < 10; i++) {
      at = observe(at, 0.5);
    }

    expect(at.observations).toHaveLength(5);
  });

  it('should throw for score outside [0, 1] in observe', () => {
    const at = createAdaptiveThreshold('node1');
    expect(() => observe(at, 1.5)).toThrow(MycelixError);
    expect(() => observe(at, -0.1)).toThrow(MycelixError);
  });

  it('should provide threshold statistics', () => {
    let at = createAdaptiveThreshold('node1', 10);

    for (let i = 0; i < 5; i++) {
      at = observe(at, 0.7 + i * 0.05);
    }

    const stats = getThresholdStats(at);
    expect(stats.observationCount).toBe(5);
    expect(stats.mean).toBeGreaterThan(0);
    expect(stats.currentThreshold).toBeGreaterThan(0);
  });
});

describe('MATL - Constants', () => {
  it('should have correct Byzantine tolerance', () => {
    expect(MAX_BYZANTINE_TOLERANCE).toBe(0.34);
  });

  it('should have correct default threshold', () => {
    expect(DEFAULT_BYZANTINE_THRESHOLD).toBe(0.5);
  });
});

describe('MATL - ReputationCache', () => {
  it('should cache reputation values', async () => {
    const { ReputationCache } = await import('../src/matl/index.js');
    const cache = new ReputationCache();
    const rep = createReputation('agent-1');

    // First call is a miss
    const value1 = cache.getReputationValue(rep);
    const stats1 = cache.getStats();
    expect(stats1.misses).toBe(1);
    expect(stats1.hits).toBe(0);

    // Second call is a hit
    const value2 = cache.getReputationValue(rep);
    const stats2 = cache.getStats();
    expect(stats2.hits).toBe(1);
    expect(stats2.misses).toBe(1);

    // Values should be identical
    expect(value1).toBe(value2);
  });

  it('should return correct hit rate', async () => {
    const { ReputationCache } = await import('../src/matl/index.js');
    const cache = new ReputationCache();
    const rep = createReputation('agent-1');

    cache.getReputationValue(rep); // miss
    cache.getReputationValue(rep); // hit
    cache.getReputationValue(rep); // hit
    cache.getReputationValue(rep); // hit

    const stats = cache.getStats();
    expect(stats.hitRate).toBe(0.75); // 3 hits / 4 total
  });

  it('should evict oldest entry when at capacity', async () => {
    const { ReputationCache } = await import('../src/matl/index.js');
    const cache = new ReputationCache({ maxSize: 3 });

    const rep1 = createReputation('agent-1');
    const rep2 = createReputation('agent-2');
    const rep3 = createReputation('agent-3');
    const rep4 = createReputation('agent-4');

    cache.getReputationValue(rep1);
    cache.getReputationValue(rep2);
    cache.getReputationValue(rep3);
    expect(cache.getStats().size).toBe(3);

    // Adding 4th should evict oldest (rep1)
    cache.getReputationValue(rep4);
    expect(cache.getStats().size).toBe(3);

    // rep1 should be a miss now (was evicted)
    const statsBefore = cache.getStats();
    cache.getReputationValue(rep1);
    const statsAfter = cache.getStats();
    expect(statsAfter.misses).toBe(statsBefore.misses + 1);
  });

  it('should invalidate entries by agent', async () => {
    const { ReputationCache } = await import('../src/matl/index.js');
    const cache = new ReputationCache();

    const rep1 = createReputation('agent-1');
    const rep2 = createReputation('agent-2');

    cache.getReputationValue(rep1);
    cache.getReputationValue(recordPositive(rep1));
    cache.getReputationValue(rep2);

    expect(cache.getStats().size).toBe(3);

    const invalidated = cache.invalidate('agent-1');
    expect(invalidated).toBe(2);
    expect(cache.getStats().size).toBe(1);
  });

  it('should clear all entries', async () => {
    const { ReputationCache } = await import('../src/matl/index.js');
    const cache = new ReputationCache();

    cache.getReputationValue(createReputation('agent-1'));
    cache.getReputationValue(createReputation('agent-2'));

    expect(cache.getStats().size).toBe(2);
    cache.clear();
    expect(cache.getStats().size).toBe(0);
    expect(cache.getStats().hits).toBe(0);
    expect(cache.getStats().misses).toBe(0);
  });

  it('should compute composite scores with cached reputation', async () => {
    const { ReputationCache } = await import('../src/matl/index.js');
    const cache = new ReputationCache();

    const pogq = createPoGQ(0.9, 0.8, 0.1);
    const rep = createReputation('agent-1');

    const score = cache.getCompositeScore(pogq, rep);
    expect(score.finalScore).toBeGreaterThan(0);
    expect(score.finalScore).toBeLessThanOrEqual(1);
    expect(score.isTrustworthy).toBeDefined();

    // Reputation should be cached
    expect(cache.getStats().size).toBe(1);
  });

  it('should respect TTL for cache entries', async () => {
    const { ReputationCache } = await import('../src/matl/index.js');
    const cache = new ReputationCache({ ttlMs: 50 }); // 50ms TTL

    const rep = createReputation('agent-1');
    cache.getReputationValue(rep); // miss

    // Immediate second call is a hit
    cache.getReputationValue(rep);
    expect(cache.getStats().hits).toBe(1);

    // Wait for TTL to expire
    await new Promise(resolve => setTimeout(resolve, 60));

    // After TTL, should be a miss again
    cache.getReputationValue(rep);
    expect(cache.getStats().misses).toBe(2);
  });

  it('should prune expired entries', async () => {
    const { ReputationCache } = await import('../src/matl/index.js');
    const cache = new ReputationCache({ ttlMs: 30 });

    cache.getReputationValue(createReputation('agent-1'));
    cache.getReputationValue(createReputation('agent-2'));
    expect(cache.getStats().size).toBe(2);

    await new Promise(resolve => setTimeout(resolve, 40));

    const pruned = cache.prune();
    expect(pruned).toBe(2);
    expect(cache.getStats().size).toBe(0);
  });
});
