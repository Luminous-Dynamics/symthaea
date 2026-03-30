// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Security Module Integration Tests
 *
 * Tests security module integration with other SDK components:
 * - FL (Federated Learning) - secure gradient handling
 * - MATL (Adaptive Trust Layer) - trust-weighted security
 * - Bridge - secure message passing
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as security from '../../src/security/index.js';
import * as matl from '../../src/matl/index.js';
import * as fl from '../../src/fl/index.js';
import * as bridge from '../../src/bridge/index.js';

describe('Security + MATL Integration', () => {
  describe('Trust-Weighted Rate Limiting', () => {
    it('should apply different rate limits based on reputation', () => {
      // Create participants with different reputations
      const highTrustAgent = matl.createReputation('high-trust-agent');
      const lowTrustAgent = matl.createReputation('low-trust-agent');

      // Build up high trust agent reputation
      let highRep = highTrustAgent;
      for (let i = 0; i < 20; i++) {
        highRep = matl.recordPositive(highRep);
      }

      // Low trust agent has some negative interactions
      let lowRep = lowTrustAgent;
      for (let i = 0; i < 5; i++) {
        lowRep = matl.recordNegative(lowRep);
      }

      const highRepValue = matl.reputationValue(highRep);
      const lowRepValue = matl.reputationValue(lowRep);

      // Higher reputation should get more requests
      const highTrustLimit = Math.floor(10 * highRepValue);
      const lowTrustLimit = Math.floor(10 * lowRepValue);

      expect(highTrustLimit).toBeGreaterThan(lowTrustLimit);

      // Create rate limiters with trust-adjusted limits
      const highTrustLimiter = security.createRateLimiter('high-trust', {
        maxRequests: Math.max(1, highTrustLimit),
        windowMs: 1000,
      });

      const lowTrustLimiter = security.createRateLimiter('low-trust', {
        maxRequests: Math.max(1, lowTrustLimit),
        windowMs: 1000,
      });

      // High trust agent can make more requests
      let highState = highTrustLimiter;
      let highAllowedCount = 0;
      for (let i = 0; i < 20; i++) {
        const result = security.checkRateLimit(highState);
        if (result.allowed) highAllowedCount++;
        highState = result.state;
      }

      let lowState = lowTrustLimiter;
      let lowAllowedCount = 0;
      for (let i = 0; i < 20; i++) {
        const result = security.checkRateLimit(lowState);
        if (result.allowed) lowAllowedCount++;
        lowState = result.state;
      }

      expect(highAllowedCount).toBeGreaterThan(lowAllowedCount);
    });

    it('should track Byzantine behavior via audit log', () => {
      const auditLog = new security.SecurityAuditLog();
      const threshold = matl.createAdaptiveThreshold('byzantine-tracker');

      // Simulate several scores
      let tracker = threshold;
      const scores = [0.8, 0.75, 0.82, 0.3, 0.79]; // One anomaly at 0.3

      for (const score of scores) {
        tracker = matl.observe(tracker, score);

        if (matl.isAnomalous(tracker, score)) {
          auditLog.log(
            security.SecurityEventType.SUSPICIOUS_ACTIVITY,
            { score, threshold: matl.getThreshold(tracker) },
            'high',
            'byzantine-tracker'
          );
        }
      }

      const suspiciousEvents = auditLog.getEvents({
        type: security.SecurityEventType.SUSPICIOUS_ACTIVITY,
      });

      expect(suspiciousEvents.length).toBeGreaterThan(0);
      expect((suspiciousEvents[0].details as any).score).toBeLessThan(0.5);
    });
  });

  describe('Secure PoGQ Computation', () => {
    it('should hash PoGQ for tamper detection', async () => {
      const pogq = matl.createPoGQ(0.85, 0.9, 0.1);

      // Create a hash of the PoGQ
      const pogqData = JSON.stringify({
        quality: pogq.quality,
        consistency: pogq.consistency,
        entropy: pogq.entropy,
        timestamp: pogq.timestamp,
      });

      const hash = await security.hashHex(pogqData);

      // Verify hash is consistent
      const hash2 = await security.hashHex(pogqData);
      expect(hash).toBe(hash2);

      // Tampering should produce different hash
      const tamperedData = JSON.stringify({
        ...pogq,
        quality: 0.99, // Tampered
      });
      const tamperedHash = await security.hashHex(tamperedData);
      expect(tamperedHash).not.toBe(hash);
    });

    it('should HMAC-sign composite scores', async () => {
      const pogq = matl.createPoGQ(0.85, 0.9, 0.1);
      const reputation = matl.createReputation('test-agent');
      const composite = matl.calculateComposite(pogq, reputation);

      // Sign the composite score
      const key = security.secureRandomBytes(32);
      const scoreData = JSON.stringify({
        pogqScore: composite.pogqScore,
        reputationScore: composite.reputationScore,
        finalScore: composite.finalScore,
        isTrustworthy: composite.isTrustworthy,
      });

      const mac = await security.hmac(key, scoreData);

      // Verify signature
      expect(await security.verifyHmac(key, scoreData, mac)).toBe(true);

      // Tampered data should fail verification
      const tamperedData = scoreData.replace(
        composite.finalScore.toString(),
        '0.99'
      );
      expect(await security.verifyHmac(key, tamperedData, mac)).toBe(false);
    });
  });

  describe('Byzantine Fault Tolerance Validation', () => {
    it('should validate MATL Byzantine tolerance against security BFT rules', () => {
      // MATL allows up to 34% validated Byzantine tolerance
      const matlTolerance = matl.MAX_BYZANTINE_TOLERANCE; // 0.34

      // For n=10 nodes, calculate max Byzantine nodes
      const n = 10;
      const maxByzantineMATL = Math.floor(n * matlTolerance);
      const maxByzantineBFT = security.maxByzantineFailures(n);

      // Security BFT is more conservative (33%)
      expect(maxByzantineBFT).toBeLessThanOrEqual(maxByzantineMATL);

      // Verify BFT params are valid for conservative estimate
      const bftValidation = security.validateBftParams(n, maxByzantineBFT);
      expect(bftValidation.valid).toBe(true);
    });

    it('should check quorum for trust-weighted consensus', () => {
      const totalNodes = 10;
      const byzantineCount = 3;

      // Need 2f+1 honest nodes for consensus
      const requiredHonest = 2 * byzantineCount + 1;

      expect(
        security.hasQuorum(totalNodes, byzantineCount, requiredHonest)
      ).toBe(true);

      // With too many Byzantine nodes, no quorum
      expect(security.hasQuorum(totalNodes, 5, requiredHonest)).toBe(false);
    });
  });
});

describe('Security + FL Integration', () => {
  let coordinator: fl.FLCoordinator;

  beforeEach(() => {
    coordinator = new fl.FLCoordinator({
      minParticipants: 3,
      maxParticipants: 10,
      byzantineTolerance: 0.33,
      trustThreshold: 0.5,
    });
  });

  describe('Secure Gradient Transmission', () => {
    it('should hash gradients for integrity verification', async () => {
      const gradients = new Float64Array([0.1, -0.2, 0.3, 0.15, -0.05]);
      const serialized = fl.serializeGradients(gradients);

      // Hash the serialized gradients
      const hash = await security.hash(serialized);

      // Deserialization and re-hashing should match
      const deserialized = fl.deserializeGradients(serialized);
      const reSerialized = fl.serializeGradients(deserialized);
      const reHash = await security.hash(reSerialized);

      expect(security.constantTimeEqual(hash, reHash)).toBe(true);
    });

    it('should detect tampered gradients via HMAC', async () => {
      const gradients = new Float64Array([0.1, -0.2, 0.3, 0.15, -0.05]);
      const serialized = fl.serializeGradients(gradients);

      // Sign gradients
      const key = security.secureRandomBytes(32);
      const mac = await security.hmac(key, serialized);

      // Tamper with gradients
      const tamperedGradients = new Float64Array([
        0.99, -0.2, 0.3, 0.15, -0.05,
      ]);
      const tamperedSerialized = fl.serializeGradients(tamperedGradients);

      // Verification should fail
      expect(
        await security.verifyHmac(key, tamperedSerialized, mac)
      ).toBe(false);

      // Original should pass
      expect(await security.verifyHmac(key, serialized, mac)).toBe(true);
    });
  });

  describe('Rate-Limited FL Updates', () => {
    it('should rate limit participant submissions', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 3,
        windowMs: 10000,
      });

      // Register participants
      const participant1 = coordinator.registerParticipant('p1');
      const participant2 = coordinator.registerParticipant('p2');

      coordinator.startRound();

      // Each participant can only submit 3 updates
      for (let i = 0; i < 5; i++) {
        const p1Result = registry.check('p1');
        const p2Result = registry.check('p2');

        if (i < 3) {
          expect(p1Result.allowed).toBe(true);
          expect(p2Result.allowed).toBe(true);
        } else {
          expect(p1Result.allowed).toBe(false);
          expect(p2Result.allowed).toBe(false);
        }
      }
    });

    it('should log excessive submission attempts', () => {
      const auditLog = new security.SecurityAuditLog();
      const registry = new security.RateLimiterRegistry({
        maxRequests: 2,
        windowMs: 10000,
      });

      coordinator.registerParticipant('malicious');
      coordinator.startRound();

      // Simulate rapid submission attempts
      for (let i = 0; i < 10; i++) {
        const result = registry.check('malicious');
        if (!result.allowed) {
          auditLog.log(
            security.SecurityEventType.RATE_LIMIT_EXCEEDED,
            {
              participantId: 'malicious',
              attemptNumber: i + 1,
              remaining: result.remaining,
            },
            'medium',
            'malicious'
          );
        }
      }

      const rateLimitEvents = auditLog.getEvents({
        type: security.SecurityEventType.RATE_LIMIT_EXCEEDED,
      });

      // Should have logged 8 rate limit violations (10 attempts - 2 allowed)
      expect(rateLimitEvents.length).toBe(8);
    });
  });

  describe('Secure Participant IDs', () => {
    it('should generate secure participant IDs', () => {
      const id1 = security.secureUUID();
      const id2 = security.secureUUID();

      expect(id1).not.toBe(id2);

      coordinator.registerParticipant(id1);
      coordinator.registerParticipant(id2);

      const stats = coordinator.getRoundStats();
      expect(stats.participantCount).toBe(2);
    });

    it('should sanitize participant IDs before registration', () => {
      const unsafeId = '<script>alert("xss")</script>user123';
      const safeId = security.sanitizeId(unsafeId);

      expect(safeId).toBe('scriptalertxssscriptuser123');

      const participant = coordinator.registerParticipant(safeId);
      expect(participant.id).toBe(safeId);
    });
  });

  describe('Byzantine Detection with Security Logging', () => {
    it('should log potential Byzantine behavior', async () => {
      const auditLog = new security.SecurityAuditLog();

      // Create several PoGQ measurements
      const normalPogq = matl.createPoGQ(0.85, 0.9, 0.1);
      const suspiciousPogq = matl.createPoGQ(0.2, 0.3, 0.9); // Low quality, high entropy

      // Check for Byzantine behavior
      if (matl.isByzantine(normalPogq)) {
        auditLog.log(
          security.SecurityEventType.SUSPICIOUS_ACTIVITY,
          { participantId: 'normal', pogq: normalPogq },
          'high'
        );
      }

      if (matl.isByzantine(suspiciousPogq)) {
        auditLog.log(
          security.SecurityEventType.SUSPICIOUS_ACTIVITY,
          { participantId: 'suspicious', pogq: suspiciousPogq },
          'high'
        );
      }

      const suspiciousEvents = auditLog.getEvents({
        type: security.SecurityEventType.SUSPICIOUS_ACTIVITY,
      });

      // Only the suspicious one should be logged
      expect(suspiciousEvents.length).toBe(1);
      expect(
        (suspiciousEvents[0].details as any).participantId
      ).toBe('suspicious');
    });
  });
});

describe('Security + Bridge Integration', () => {
  describe('Secure Message Passing', () => {
    it('should hash cross-hApp reputation data for verification', async () => {
      const reputationScores: bridge.HappReputationScore[] = [
        { happ: 'happ-1', score: 0.9, weight: 1.0, lastUpdate: Date.now() },
        { happ: 'happ-2', score: 0.85, weight: 0.8, lastUpdate: Date.now() },
      ];

      // Create snapshot for hashing
      const reputationData = JSON.stringify({
        scores: reputationScores,
        aggregate: bridge.calculateAggregateReputation(reputationScores),
      });

      const hash = await security.hashHex(reputationData);

      // Hash should be consistent
      expect(hash.length).toBe(64);
      expect(await security.hashHex(reputationData)).toBe(hash);
    });

    it('should sign bridge messages with HMAC', async () => {
      const message = bridge.createReputationQuery('source-happ', 'agent-123');

      const key = security.secureRandomBytes(32);
      const messageData = JSON.stringify(message);

      const signature = await security.hmac(key, messageData);

      // Verification
      expect(await security.verifyHmac(key, messageData, signature)).toBe(true);

      // Tampered message fails
      const tamperedMessage = { ...message, sourceHapp: 'attacker' };
      expect(
        await security.verifyHmac(
          key,
          JSON.stringify(tamperedMessage),
          signature
        )
      ).toBe(false);
    });
  });

  describe('Secure hApp Registration', () => {
    it('should rate limit hApp registrations', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 5,
        windowMs: 60000,
      });

      const localBridge = new bridge.LocalBridge();

      // Simulate hApp registration with rate limiting
      const registrationAttempts = 10;
      let successfulRegistrations = 0;

      for (let i = 0; i < registrationAttempts; i++) {
        const result = registry.check('registrant-1');
        if (result.allowed) {
          localBridge.registerHapp('happ-' + i);
          successfulRegistrations++;
        }
      }

      expect(successfulRegistrations).toBe(5);
    });

    it('should sanitize hApp IDs before registration', () => {
      const unsafeId = '<script>happ</script>';
      const safeId = security.sanitizeId(unsafeId);

      expect(safeId).toBe('scripthappscript');

      const localBridge = new bridge.LocalBridge();
      localBridge.registerHapp(safeId);

      // Should be able to set reputation for sanitized hApp
      const rep = matl.createReputation('agent-1');
      localBridge.setReputation(safeId, 'agent-1', rep);

      const scores = localBridge.getCrossHappReputation('agent-1');
      expect(scores.length).toBe(1);
      expect(scores[0].happ).toBe(safeId);
    });
  });

  describe('Consensus with Security Validation', () => {
    it('should validate BFT parameters for bridge consensus', () => {
      // Typical bridge network sizes
      const scenarios = [
        { nodes: 4, maxByz: 1 },
        { nodes: 7, maxByz: 2 },
        { nodes: 10, maxByz: 3 },
        { nodes: 13, maxByz: 4 },
      ];

      for (const { nodes, maxByz } of scenarios) {
        const validation = security.validateBftParams(nodes, maxByz);
        expect(validation.valid).toBe(true);

        // Verify we can reach consensus
        const honestNeeded = maxByz + 1;
        expect(security.hasQuorum(nodes, maxByz, honestNeeded)).toBe(true);
      }
    });

    it('should log consensus failures', () => {
      const auditLog = new security.SecurityAuditLog();

      // Simulate consensus attempt with too many Byzantine nodes
      const totalNodes = 10;
      const byzantineNodes = 5;
      const honestNeeded = 6;

      if (!security.hasQuorum(totalNodes, byzantineNodes, honestNeeded)) {
        auditLog.log(
          security.SecurityEventType.SUSPICIOUS_ACTIVITY,
          {
            totalNodes,
            byzantineNodes,
            honestNeeded,
            reason: 'Insufficient honest nodes for consensus',
          },
          'critical'
        );
      }

      const criticalEvents = auditLog.getEvents({ severity: 'critical' });
      expect(criticalEvents.length).toBe(1);
      expect(
        (criticalEvents[0].details as any).reason
      ).toContain('Insufficient');
    });
  });
});

describe('End-to-End Security Workflow', () => {
  it('should demonstrate full security pipeline', async () => {
    const auditLog = new security.SecurityAuditLog();
    const rateLimiter = new security.RateLimiterRegistry({
      maxRequests: 100,
      windowMs: 60000,
    });

    // 1. Generate secure participant ID
    const participantId = security.secureUUID();
    expect(participantId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
    );

    // 2. Create FL coordinator and register participant
    const coordinator = new fl.FLCoordinator({
      minParticipants: 2,
      byzantineTolerance: 0.33,
    });
    coordinator.registerParticipant(participantId);

    // 3. Check rate limit before submission
    const rateCheck = rateLimiter.check(participantId);
    expect(rateCheck.allowed).toBe(true);

    // 4. Create and sign gradient update
    const gradients = new Float64Array([0.1, -0.05, 0.2, -0.1, 0.15]);
    const serialized = fl.serializeGradients(gradients);
    const key = security.secureRandomBytes(32);
    const signature = await security.hmac(key, serialized);

    // 5. Verify signature before accepting
    const isValid = await security.verifyHmac(key, serialized, signature);
    expect(isValid).toBe(true);

    // 6. Create PoGQ and check Byzantine behavior
    const pogq = matl.createPoGQ(0.85, 0.9, 0.1);
    const isByzantine = matl.isByzantine(pogq);

    if (isByzantine) {
      auditLog.log(
        security.SecurityEventType.SUSPICIOUS_ACTIVITY,
        { participantId, pogq },
        'high',
        participantId
      );
    }

    expect(isByzantine).toBe(false);

    // 7. Calculate composite score
    const reputation = matl.createReputation(participantId);
    const composite = matl.calculateComposite(pogq, reputation);
    expect(composite.isTrustworthy).toBe(true);

    // 8. Log successful operation
    auditLog.log(
      security.SecurityEventType.CRYPTO_OPERATION,
      {
        operation: 'gradient_submission',
        participantId,
        composite: composite.finalScore,
      },
      'low',
      participantId
    );

    const events = auditLog.getEvents({ entityId: participantId });
    expect(events.length).toBe(1);
    expect(events[0].type).toBe(security.SecurityEventType.CRYPTO_OPERATION);
  });

  it('should handle malicious participant detection', async () => {
    const auditLog = new security.SecurityAuditLog();
    const rateLimiter = new security.RateLimiterRegistry({
      maxRequests: 5,
      windowMs: 10000,
    });

    const maliciousId = security.secureUUID();

    // 1. Rapid-fire requests (rate limit violation)
    for (let i = 0; i < 20; i++) {
      const check = rateLimiter.check(maliciousId);
      if (!check.allowed) {
        auditLog.log(
          security.SecurityEventType.RATE_LIMIT_EXCEEDED,
          { attempt: i + 1 },
          'medium',
          maliciousId
        );
      }
    }

    // 2. Low-quality PoGQ (Byzantine indicator)
    const badPogq = matl.createPoGQ(0.1, 0.2, 0.95);
    if (matl.isByzantine(badPogq)) {
      auditLog.log(
        security.SecurityEventType.SUSPICIOUS_ACTIVITY,
        { reason: 'Byzantine PoGQ detected', pogq: badPogq },
        'high',
        maliciousId
      );
    }

    // 3. Check audit log for violations
    const allEvents = auditLog.getEvents({ entityId: maliciousId });
    const rateLimitEvents = allEvents.filter(
      (e) => e.type === security.SecurityEventType.RATE_LIMIT_EXCEEDED
    );
    const suspiciousEvents = allEvents.filter(
      (e) => e.type === security.SecurityEventType.SUSPICIOUS_ACTIVITY
    );

    expect(rateLimitEvents.length).toBe(15); // 20 - 5 allowed
    expect(suspiciousEvents.length).toBe(1);

    // 4. Total events should indicate problematic participant
    expect(allEvents.length).toBeGreaterThan(10);
  });
});

describe('Security Constants Validation', () => {
  it('should have consistent Byzantine tolerance between modules', () => {
    // MATL max tolerance
    const matlMax = matl.MAX_BYZANTINE_TOLERANCE; // 0.34

    // BFT rule: n >= 3f + 1, so f <= (n-1)/3
    // For percentage: f/n <= 1/3 ≈ 0.33
    const bftMaxPercentage = 1 / 3;

    // MATL is more permissive (uses additional trust mechanisms)
    expect(matlMax).toBeGreaterThan(bftMaxPercentage);

    // But both should be less than 50%
    expect(matlMax).toBeLessThan(0.5);
    expect(bftMaxPercentage).toBeLessThan(0.5);
  });

  it('should use secure random for all ID generation', () => {
    // Generate multiple IDs
    const ids = new Set<string>();
    for (let i = 0; i < 1000; i++) {
      ids.add(security.secureUUID());
    }

    // All should be unique
    expect(ids.size).toBe(1000);
  });
});
