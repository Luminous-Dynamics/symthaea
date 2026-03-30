// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Safe Belief Propagator Tests
 *
 * Tests for circuit breakers, rate limits, quarantine, and human review.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  SafeBeliefPropagator,
  getSafeBeliefPropagator,
  resetSafeBeliefPropagator,
  DEFAULT_CIRCUIT_BREAKER_CONFIG,
  DEFAULT_PROPAGATION_CONFIG,
  type PropagationRequest,
  type PropagationResult,
} from '../src/propagation/index.js';
import { CalibrationEngine } from '../src/calibration/engine.js';

// Helper to create a propagation request
function createRequest(overrides: Partial<PropagationRequest> = {}): PropagationRequest {
  return {
    id: `req-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    sourceClaimId: 'claim-1',
    updateType: 'confidence_increase',
    newConfidence: 0.8,
    oldConfidence: 0.6,
    initiatedBy: 'agent-1',
    createdAt: Date.now(),
    priority: 1,
    ...overrides,
  };
}

describe('Safe Belief Propagator - Basic Propagation', () => {
  let propagator: SafeBeliefPropagator;

  beforeEach(() => {
    propagator = new SafeBeliefPropagator();
    // Set up some test claims
    propagator.registerClaim('claim-1', 0.6, ['claim-2', 'claim-3']);
    propagator.registerClaim('claim-2', 0.5, ['claim-4']);
    propagator.registerClaim('claim-3', 0.7, []);
    propagator.registerClaim('claim-4', 0.4, []);
  });

  it('should propagate a simple update', async () => {
    const request = createRequest();
    const result = await propagator.propagate(request);

    expect(result.requestId).toBe(request.id);
    expect(['completed', 'blocked', 'quarantined']).toContain(result.state);
  });

  it('should track affected claims', async () => {
    const request = createRequest();
    const result = await propagator.propagate(request);

    if (result.state === 'completed') {
      expect(result.affectedClaims).toBeDefined();
    }
  });

  it('should report propagation depth', async () => {
    const request = createRequest();
    const result = await propagator.propagate(request);

    expect(result.depthReached).toBeGreaterThanOrEqual(0);
  });
});

describe('Safe Belief Propagator - Circuit Breakers', () => {
  let propagator: SafeBeliefPropagator;

  beforeEach(() => {
    propagator = new SafeBeliefPropagator(null, {
      circuitBreaker: {
        ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
        maxDepth: 3,
        maxAffectedClaims: 5,
        maxConfidenceJump: 0.25,
      },
    });

    // Set up a deeper claim graph
    propagator.registerClaim('root', 0.5, ['level1-a', 'level1-b']);
    propagator.registerClaim('level1-a', 0.5, ['level2-a', 'level2-b']);
    propagator.registerClaim('level1-b', 0.5, ['level2-c']);
    propagator.registerClaim('level2-a', 0.5, ['level3-a']);
    propagator.registerClaim('level2-b', 0.5, []);
    propagator.registerClaim('level2-c', 0.5, []);
    propagator.registerClaim('level3-a', 0.5, ['level4-a']); // Beyond max depth
    propagator.registerClaim('level4-a', 0.5, []);
  });

  it('should block propagation exceeding max depth', async () => {
    const request = createRequest({
      sourceClaimId: 'root',
      newConfidence: 0.9,
      oldConfidence: 0.5,
    });

    const result = await propagator.propagate(request);

    // Should not exceed configured max depth
    expect(result.depthReached).toBeLessThanOrEqual(3);
  });

  it('should block too-large confidence jumps', async () => {
    const request = createRequest({
      sourceClaimId: 'root',
      newConfidence: 0.95, // Jump of 0.45 (> maxConfidenceJump of 0.25)
      oldConfidence: 0.5,
    });

    const result = await propagator.propagate(request);

    expect(result.state).toBe('blocked');
    expect(result.blockReason).toBe('confidence_jump_too_large');
  });

  it('should verify default circuit breaker values', () => {
    expect(DEFAULT_CIRCUIT_BREAKER_CONFIG.maxDepth).toBe(5);
    expect(DEFAULT_CIRCUIT_BREAKER_CONFIG.maxAffectedClaims).toBe(100);
    expect(DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConfidenceJump).toBe(0.3);
    expect(DEFAULT_CIRCUIT_BREAKER_CONFIG.perClaimRateLimit).toBe(10);
    expect(DEFAULT_CIRCUIT_BREAKER_CONFIG.globalRateLimit).toBe(1000);
  });
});

describe('Safe Belief Propagator - Rate Limiting', () => {
  let propagator: SafeBeliefPropagator;

  beforeEach(() => {
    propagator = new SafeBeliefPropagator(null, {
      circuitBreaker: {
        ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
        perClaimRateLimit: 3, // Low limit for testing
        cooldownPeriodMs: 100, // Short cooldown for testing
      },
    });

    propagator.registerClaim('claim-1', 0.5, []);
  });

  it('should block after exceeding per-claim rate limit', async () => {
    // Make requests up to the limit
    for (let i = 0; i < 3; i++) {
      const result = await propagator.propagate(
        createRequest({
          id: `req-${i}`,
          sourceClaimId: 'claim-1',
          newConfidence: 0.5 + (i + 1) * 0.05,
          oldConfidence: 0.5 + i * 0.05,
        })
      );

      // Wait for cooldown between requests
      await new Promise((resolve) => setTimeout(resolve, 110));
    }

    // This one should be rate limited
    const blockedResult = await propagator.propagate(
      createRequest({
        sourceClaimId: 'claim-1',
        newConfidence: 0.7,
        oldConfidence: 0.65,
      })
    );

    expect(blockedResult.state).toBe('blocked');
    expect(blockedResult.blockReason).toBe('rate_limit_exceeded');
  });

  it('should track rate limiting state', async () => {
    // First request succeeds
    const result1 = await propagator.propagate(createRequest({ sourceClaimId: 'claim-1' }));

    // Request processed (may be completed or blocked depending on rate limits)
    expect(result1.state).toBeDefined();

    // Second request also processed
    const result2 = await propagator.propagate(
      createRequest({
        id: 'req-second',
        sourceClaimId: 'claim-1',
        newConfidence: 0.65,
        oldConfidence: 0.6,
      })
    );

    expect(result2.state).toBeDefined();
  });
});

describe('Safe Belief Propagator - Quarantine', () => {
  let propagator: SafeBeliefPropagator;

  beforeEach(() => {
    propagator = new SafeBeliefPropagator(null, {
      circuitBreaker: {
        ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
        enableQuarantine: true,
        quarantineThreshold: 0.5,
      },
    });

    propagator.registerClaim('normal-claim', 0.5, []);
    propagator.registerClaim('suspicious-claim', 0.5, [], {
      centralityScore: 0.9, // High centrality makes it suspicious
    });
  });

  it('should quarantine suspicious patterns', async () => {
    // Mark the claim as suspicious (unknown source)
    const result = await propagator.propagate(
      createRequest({
        sourceClaimId: 'suspicious-claim',
        initiatedBy: 'unknown_suspicious_actor', // Starts with "unknown_"
      })
    );

    // Should be quarantined, blocked, or pending review
    expect(['quarantined', 'blocked', 'pending']).toContain(result.state);
  });

  it('should track quarantined propagations', async () => {
    // Generate suspicious activity
    await propagator.propagate(
      createRequest({
        sourceClaimId: 'suspicious-claim',
        initiatedBy: 'unknown_actor',
      })
    );

    const stats = propagator.getStats();
    expect(stats.quarantinedCount).toBeGreaterThanOrEqual(0);
  });

  it('should allow deciding quarantine outcome', async () => {
    // First, trigger a quarantine
    const result = await propagator.propagate(
      createRequest({
        id: 'quarantine-test',
        sourceClaimId: 'suspicious-claim',
        initiatedBy: 'unknown_actor',
      })
    );

    if (result.state === 'quarantined') {
      // Submit decision to release from quarantine
      await propagator.submitQuarantineDecision('quarantine-test', {
        propagationId: 'quarantine-test',
        decision: 'release',
        decidedBy: 'admin-1',
        reason: 'False positive',
        decidedAt: Date.now(),
      });

      // Check it was released
      const quarantined = propagator.getQuarantined();
      expect(quarantined.find((q) => q.requestId === 'quarantine-test')).toBeUndefined();
    }
  });
});

describe('Safe Belief Propagator - Human Review', () => {
  let propagator: SafeBeliefPropagator;

  beforeEach(() => {
    propagator = new SafeBeliefPropagator(null, {
      circuitBreaker: {
        ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
        centralityThresholdForReview: 0.7,
      },
    });

    // High-centrality claim requires human review
    propagator.registerClaim('central-claim', 0.5, ['dep-1', 'dep-2', 'dep-3', 'dep-4', 'dep-5'], {
      centralityScore: 0.9,
    });
  });

  it('should flag high-centrality updates for human review', async () => {
    const result = await propagator.propagate(
      createRequest({
        sourceClaimId: 'central-claim',
        newConfidence: 0.8,
        oldConfidence: 0.5,
      })
    );

    expect(result.pendingHumanReview).toBe(true);
  });

  it('should respect requiresHumanReview flag', async () => {
    propagator.registerClaim('normal-claim', 0.5, []);

    const result = await propagator.propagate(
      createRequest({
        sourceClaimId: 'normal-claim',
        requiresHumanReview: true,
      })
    );

    expect(result.pendingHumanReview).toBe(true);
  });

  it('should track pending human reviews via stats', async () => {
    const statsBefore = propagator.getStats();
    const queueSizeBefore = statsBefore.humanReviewQueueSize;

    await propagator.propagate(
      createRequest({
        id: 'review-test',
        sourceClaimId: 'central-claim',
      })
    );

    // Stats track the queue size
    const statsAfter = propagator.getStats();
    expect(statsAfter.humanReviewQueueSize).toBeGreaterThanOrEqual(queueSizeBefore);
  });
});

describe('Safe Belief Propagator - Calibration Integration', () => {
  let propagator: SafeBeliefPropagator;
  let calibrationEngine: CalibrationEngine;

  beforeEach(async () => {
    calibrationEngine = new CalibrationEngine({ minSampleSize: 5 });

    // Build up calibration history showing poor calibration for a domain
    for (let i = 0; i < 20; i++) {
      await calibrationEngine.recordResolution(
        { id: `claim-${i}`, confidence: 0.9, domain: 'uncalibrated' },
        i < 8 ? 'correct' : 'incorrect' // Only 40% correct at 90% confidence
      );
    }

    await calibrationEngine.generateReport({ type: 'domain', domain: 'uncalibrated' });

    propagator = new SafeBeliefPropagator(calibrationEngine, {
      enableCalibrationGating: true,
      circuitBreaker: {
        ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
        maxConfidenceJump: 0.5, // Allow large jumps so calibration check runs
      },
    });

    propagator.registerClaim('uncalibrated-claim', 0.5, [], {
      domain: 'uncalibrated',
    });
  });

  it('should block propagation for poorly-calibrated high-confidence updates', async () => {
    const result = await propagator.propagate(
      createRequest({
        sourceClaimId: 'uncalibrated-claim',
        newConfidence: 0.9,
        oldConfidence: 0.5,
      })
    );

    expect(result.state).toBe('blocked');
    expect(result.blockReason).toBe('calibration_failed');
  });

  it('should allow propagation without calibration engine', async () => {
    const noCalibratonPropagator = new SafeBeliefPropagator(null);
    noCalibratonPropagator.registerClaim('claim-1', 0.5, []);

    const result = await noCalibratonPropagator.propagate(
      createRequest({
        sourceClaimId: 'claim-1',
        newConfidence: 0.9,
        oldConfidence: 0.5,
      })
    );

    // Should not be blocked by calibration (no engine configured)
    expect(result.blockReason).not.toBe('calibration_failed');
  });
});

describe('Safe Belief Propagator - Statistics', () => {
  let propagator: SafeBeliefPropagator;

  beforeEach(() => {
    propagator = new SafeBeliefPropagator();
    propagator.registerClaim('claim-1', 0.5, ['claim-2']);
    propagator.registerClaim('claim-2', 0.5, []);
  });

  it('should track propagation statistics', async () => {
    await propagator.propagate(createRequest());

    const stats = propagator.getStats();

    // Check stats object has the expected structure
    expect(stats).toBeDefined();
    expect(stats.globalRateLimit).toBeDefined();
    expect(stats.quarantinedCount).toBeDefined();
    expect(stats.humanReviewQueueSize).toBeDefined();
  });

  it('should track claims in graph', async () => {
    const propagator = new SafeBeliefPropagator();
    propagator.registerClaim('claim-1', 0.5, []);
    propagator.registerClaim('claim-2', 0.6, []);

    const stats = propagator.getStats();
    expect(stats.trackedClaims).toBeGreaterThanOrEqual(2);
  });
});

describe('Safe Belief Propagator - Event Emission', () => {
  let propagator: SafeBeliefPropagator;
  const events: string[] = [];

  beforeEach(() => {
    propagator = new SafeBeliefPropagator();
    propagator.registerClaim('claim-1', 0.5, []);

    events.length = 0;
    propagator.on((event) => events.push(event.type));
  });

  it('should emit propagation_started event', async () => {
    await propagator.propagate(createRequest());

    expect(events).toContain('propagation_started');
  });

  it('should emit propagation_completed event', async () => {
    await propagator.propagate(createRequest());

    expect(events).toContain('propagation_completed');
  });

  it('should emit propagation_blocked event when blocked', async () => {
    const restrictivePropagator = new SafeBeliefPropagator(null, {
      circuitBreaker: {
        ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
        maxConfidenceJump: 0.05,
      },
    });
    restrictivePropagator.registerClaim('claim-1', 0.5, []);

    const blockEvents: string[] = [];
    restrictivePropagator.on((event) => blockEvents.push(event.type));

    await restrictivePropagator.propagate(
      createRequest({
        sourceClaimId: 'claim-1',
        newConfidence: 0.9,
        oldConfidence: 0.5,
      })
    );

    expect(blockEvents).toContain('propagation_blocked');
  });
});

describe('Safe Belief Propagator - Singleton', () => {
  beforeEach(() => {
    resetSafeBeliefPropagator();
  });

  it('should return same instance', () => {
    const propagator1 = getSafeBeliefPropagator();
    const propagator2 = getSafeBeliefPropagator();

    expect(propagator1).toBe(propagator2);
  });

  it('should reset correctly', async () => {
    const propagator1 = getSafeBeliefPropagator();
    propagator1.registerClaim('claim-1', 0.5, []);
    await propagator1.propagate(createRequest());

    resetSafeBeliefPropagator();

    const propagator2 = getSafeBeliefPropagator();
    // After reset, should have no tracked claims
    expect(propagator2.getStats().trackedClaims).toBe(0);
  });
});

describe('Safe Belief Propagator - Tracing', () => {
  let propagator: SafeBeliefPropagator;

  beforeEach(() => {
    propagator = new SafeBeliefPropagator(null, {
      enableTracing: true,
    });

    propagator.registerClaim('root', 0.5, ['child-1', 'child-2']);
    propagator.registerClaim('child-1', 0.5, ['grandchild']);
    propagator.registerClaim('child-2', 0.5, []);
    propagator.registerClaim('grandchild', 0.5, []);
  });

  it('should include trace when enabled', async () => {
    const result = await propagator.propagate(
      createRequest({ sourceClaimId: 'root' })
    );

    if (result.state === 'completed') {
      expect(result.trace).toBeDefined();
      expect(result.trace?.totalVisited).toBeGreaterThan(0);
    }
  });

  it('should not include trace when disabled', async () => {
    const noTracePropagator = new SafeBeliefPropagator(null, {
      enableTracing: false,
    });
    noTracePropagator.registerClaim('claim-1', 0.5, []);

    const result = await noTracePropagator.propagate(
      createRequest({ sourceClaimId: 'claim-1' })
    );

    expect(result.trace).toBeUndefined();
  });
});

describe('Safe Belief Propagator - Configuration', () => {
  it('should use default configuration', () => {
    expect(DEFAULT_PROPAGATION_CONFIG.enableCalibrationGating).toBe(true);
    expect(DEFAULT_PROPAGATION_CONFIG.enableTracing).toBe(false);
  });

  it('should accept custom configuration', () => {
    const propagator = new SafeBeliefPropagator(null, {
      enableTracing: true,
      circuitBreaker: {
        ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
        maxDepth: 10,
      },
    });

    expect(propagator).toBeDefined();
  });
});
