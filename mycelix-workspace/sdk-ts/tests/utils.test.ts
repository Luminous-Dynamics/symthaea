// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Utilities Module Tests
 *
 * Tests for convenience functions and helpers.
 */

import { describe, it, expect, vi } from 'vitest';
import * as utils from '../src/utils/index.js';
import * as security from '../src/security/index.js';
import * as fl from '../src/fl/index.js';

describe('Utilities Module', () => {
  // ============================================================================
  // Timestamp Utilities Tests
  // ============================================================================

  describe('Timestamp Utilities', () => {
    it('should get current timestamp', () => {
      const before = Date.now();
      const ts = utils.now();
      const after = Date.now();

      expect(ts).toBeGreaterThanOrEqual(before);
      expect(ts).toBeLessThanOrEqual(after);
    });

    it('should validate timestamps', () => {
      expect(utils.isValidTimestamp(Date.now())).toBe(true);
      expect(utils.isValidTimestamp(0)).toBe(true);
      expect(utils.isValidTimestamp(1000000000000)).toBe(true);

      expect(utils.isValidTimestamp(-1)).toBe(false);
      expect(utils.isValidTimestamp(NaN)).toBe(false);
      expect(utils.isValidTimestamp(Infinity)).toBe(false);
      expect(utils.isValidTimestamp('2024-01-01')).toBe(false);
      expect(utils.isValidTimestamp(null)).toBe(false);
    });

    it('should check if timestamp is expired', () => {
      const now = Date.now();
      const oldTs = now - 10000; // 10 seconds ago
      const recentTs = now - 1000; // 1 second ago

      expect(utils.isExpiredTimestamp(oldTs, 5000)).toBe(true);
      expect(utils.isExpiredTimestamp(recentTs, 5000)).toBe(false);
      expect(utils.isExpiredTimestamp(now, 5000)).toBe(false);
    });

    it('should check if timestamp is in future', () => {
      const now = Date.now();
      const futureTs = now + 10000;
      const pastTs = now - 10000;

      expect(utils.isFutureTimestamp(futureTs)).toBe(true);
      expect(utils.isFutureTimestamp(pastTs)).toBe(false);
      expect(utils.isFutureTimestamp(now)).toBe(false);
    });

    it('should handle clock skew tolerance', () => {
      const now = Date.now();
      const slightlyFuture = now + 100;

      expect(utils.isFutureTimestamp(slightlyFuture, 0)).toBe(true);
      expect(utils.isFutureTimestamp(slightlyFuture, 200)).toBe(false);
    });

    it('should normalize timestamps from Date', () => {
      const date = new Date('2024-01-15T12:00:00Z');
      const ts = utils.normalizeTimestamp(date);

      expect(ts).toBe(date.getTime());
    });

    it('should normalize timestamps from number', () => {
      const ts = 1705320000000;
      expect(utils.normalizeTimestamp(ts)).toBe(ts);
    });

    it('should calculate timestamp age', () => {
      const now = Date.now();
      const oldTs = now - 5000;

      const age = utils.timestampAge(oldTs, now);
      expect(age).toBe(5000);
    });

    it('should format timestamp as ISO string', () => {
      const ts = 1705320000000;
      const formatted = utils.formatTimestamp(ts);

      expect(formatted).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });

    it('should parse ISO string to timestamp', () => {
      const isoString = '2024-01-15T12:00:00.000Z';
      const ts = utils.parseTimestamp(isoString);

      expect(ts).toBe(new Date(isoString).getTime());
    });

    it('should return null for invalid ISO strings', () => {
      expect(utils.parseTimestamp('invalid')).toBe(null);
      expect(utils.parseTimestamp('')).toBe(null);
      expect(utils.parseTimestamp('not-a-date')).toBe(null);
    });
  });

  // ============================================================================
  // Trust Assessment Tests
  // ============================================================================

  describe('Trust Assessment', () => {
    it('should perform quick trust check', () => {
      const result = utils.checkTrust('agent-1', 0.85, 0.9, 0.1);

      expect(result.trustworthy).toBe(true);
      expect(result.score).toBeGreaterThan(0.5);
      expect(result.byzantine).toBe(false);
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.details.quality).toBe(0.85);
      expect(result.details.consistency).toBe(0.9);
      expect(result.details.entropy).toBe(0.1);
    });

    it('should detect Byzantine behavior', () => {
      const result = utils.checkTrust('agent-1', 0.1, 0.2, 0.9, {
        trustThreshold: 0.5,
      });

      expect(result.trustworthy).toBe(false);
      expect(result.byzantine).toBe(true);
    });

    it('should use existing reputation', () => {
      const reputation = utils.buildReputation('agent-1', 20, 2);
      const result = utils.checkTrust('agent-1', 0.85, 0.9, 0.1, {
        existingReputation: reputation,
      });

      expect(result.trustworthy).toBe(true);
      expect(result.details.reputationScore).toBeGreaterThan(0.8);
    });

    it('should build reputation from counts', () => {
      const reputation = utils.buildReputation('agent-1', 10, 5);

      expect(reputation.positiveCount).toBe(11); // 10 + 1 prior
      expect(reputation.negativeCount).toBe(6); // 5 + 1 prior
    });
  });

  // ============================================================================
  // FL Helper Tests
  // ============================================================================

  describe('FL Helpers', () => {
    it('should create simple FL coordinator', () => {
      const coordinator = utils.createSimpleFLCoordinator({
        minParticipants: 2,
        byzantineTolerance: 0.3,
      });

      expect(coordinator).toBeDefined();
      const stats = coordinator.getRoundStats();
      expect(stats.participantCount).toBe(0);
    });

    it('should create gradient update from array', () => {
      const update = utils.createGradientUpdate('p1', 1, [0.1, -0.2, 0.3], {
        batchSize: 32,
        loss: 0.5,
        accuracy: 0.85,
      });

      expect(update.participantId).toBe('p1');
      expect(update.modelVersion).toBe(1);
      expect(update.gradients.length).toBe(3);
      expect(update.gradients[0]).toBe(0.1);
      expect(update.metadata.batchSize).toBe(32);
      expect(update.metadata.loss).toBe(0.5);
      expect(update.metadata.accuracy).toBe(0.85);
    });

    it('should run complete FL round', () => {
      const coordinator = utils.createSimpleFLCoordinator({ minParticipants: 2 });

      coordinator.registerParticipant('p1');
      coordinator.registerParticipant('p2');
      coordinator.registerParticipant('p3');

      const updates = [
        utils.createGradientUpdate('p1', 1, [0.1, 0.2], { batchSize: 32, loss: 0.5 }),
        utils.createGradientUpdate('p2', 1, [0.15, 0.18], { batchSize: 32, loss: 0.45 }),
        utils.createGradientUpdate('p3', 1, [0.12, 0.22], { batchSize: 32, loss: 0.48 }),
      ];

      const summary = utils.runFLRound(coordinator, updates);

      expect(summary.status).toBe('completed');
      expect(summary.aggregated).toBe(true);
      expect(summary.updateCount).toBe(3);
      expect(summary.duration).toBeDefined();
    });
  });

  // ============================================================================
  // Security Helper Tests
  // ============================================================================

  describe('Security Helpers', () => {
    it('should sign and verify message', async () => {
      const key = security.secureRandomBytes(32);
      const payload = { action: 'test', value: 42 };

      const message = await utils.signMessage(payload, key);

      expect(message.payload).toEqual(payload);
      expect(message.signature).toBeDefined();
      expect(message.timestamp).toBeDefined();

      const verification = await utils.verifyMessage(message, key);

      expect(verification.valid).toBe(true);
      expect(verification.expired).toBe(false);
      expect(verification.tampered).toBe(false);
    });

    it('should detect tampered message', async () => {
      const key = security.secureRandomBytes(32);
      const payload = { action: 'test', value: 42 };

      const message = await utils.signMessage(payload, key);

      // Tamper with payload
      const tamperedMessage = { ...message, payload: { action: 'evil', value: 0 } };
      const verification = await utils.verifyMessage(tamperedMessage, key);

      expect(verification.valid).toBe(false);
      expect(verification.tampered).toBe(true);
    });

    it('should detect expired message', async () => {
      const key = security.secureRandomBytes(32);
      const message = await utils.signMessage({ test: true }, key);

      // Backdate timestamp
      message.timestamp = Date.now() - 10000;

      const verification = await utils.verifyMessage(message, key, 5000);

      expect(verification.valid).toBe(false);
      expect(verification.expired).toBe(true);
    });

    it('should create rate-limited operation', async () => {
      let callCount = 0;
      const operation = () => ++callCount;

      const rateLimited = utils.createRateLimitedOperation(operation, 3, 10000);

      // First 3 calls should succeed
      const r1 = await rateLimited.execute();
      const r2 = await rateLimited.execute();
      const r3 = await rateLimited.execute();

      expect(r1).toBe(1);
      expect(r2).toBe(2);
      expect(r3).toBe(3);

      // 4th call should be rate limited
      const r4 = await rateLimited.execute();
      expect(r4).toBe(null);

      // After reset, should work again
      rateLimited.reset();
      const r5 = await rateLimited.execute();
      expect(r5).toBe(4);
    });
  });

  // ============================================================================
  // Epistemic Helper Tests
  // ============================================================================

  describe('Epistemic Helpers', () => {
    it('should create high-trust claim', () => {
      const claim = utils.createHighTrustClaim('User verified identity', 'issuer-1');

      expect(claim.content).toBe('User verified identity');
      expect(claim.classification.empirical).toBe(3); // E3_Cryptographic
      expect(claim.classification.normative).toBe(2); // N2_Network
      expect(claim.classification.materiality).toBe(2); // M2_Persistent
    });

    it('should create medium-trust claim', () => {
      const claim = utils.createMediumTrustClaim('Group approved', 'issuer-1');

      expect(claim.classification.empirical).toBe(2); // E2_PrivateVerify
      expect(claim.classification.normative).toBe(1); // N1_Communal
      expect(claim.classification.materiality).toBe(1); // M1_Temporal
    });

    it('should create low-trust claim', () => {
      const claim = utils.createLowTrustClaim('Personal statement', 'issuer-1');

      expect(claim.classification.empirical).toBe(1); // E1_Testimonial
      expect(claim.classification.normative).toBe(0); // N0_Personal
      expect(claim.classification.materiality).toBe(0); // M0_Ephemeral
    });
  });

  // ============================================================================
  // Bridge Helper Tests
  // ============================================================================

  describe('Bridge Helpers', () => {
    it('should aggregate reputation across hApps', () => {
      const aggregator = new utils.ReputationAggregator();

      aggregator.registerHapp('happ-1');
      aggregator.registerHapp('happ-2');

      aggregator.setReputation('happ-1', 'agent-1', 20, 2);
      aggregator.setReputation('happ-2', 'agent-1', 15, 3);

      const aggregate = aggregator.getAggregateReputation('agent-1');
      expect(aggregate).toBeGreaterThan(0.5);

      const detailed = aggregator.getDetailedReputation('agent-1');
      expect(detailed.length).toBe(2);

      const happs = aggregator.getHapps();
      expect(happs).toContain('happ-1');
      expect(happs).toContain('happ-2');
    });
  });

  // ============================================================================
  // Health Check Tests
  // ============================================================================

  describe('Health Checks', () => {
    it('should run all health checks', () => {
      const results = utils.runHealthChecks();

      expect(results.length).toBeGreaterThanOrEqual(5);

      for (const result of results) {
        expect(result.component).toBeDefined();
        expect(typeof result.healthy).toBe('boolean');
      }

      // All components should be healthy
      const allHealthy = results.every((r) => r.healthy);
      expect(allHealthy).toBe(true);
    });

    it('should provide SDK info', () => {
      const info = utils.getSdkInfo();

      expect(info.version).toBe('0.5.0');
      expect(info.modules).toContain('matl');
      expect(info.modules).toContain('fl');
      expect(info.modules).toContain('security');
      expect(info.modules).toContain('utils');
    });
  });

  // ============================================================================
  // Async Helper Tests
  // ============================================================================

  describe('Async Helpers', () => {
    it('should retry failing operations', async () => {
      let attempts = 0;
      const operation = async () => {
        attempts++;
        if (attempts < 3) {
          throw new Error('Not yet');
        }
        return 'success';
      };

      const result = await utils.retry(operation, {
        maxAttempts: 5,
        initialDelay: 10,
      });

      expect(result).toBe('success');
      expect(attempts).toBe(3);
    });

    it('should fail after max attempts', async () => {
      const operation = async () => {
        throw new Error('Always fails');
      };

      await expect(
        utils.retry(operation, { maxAttempts: 3, initialDelay: 10 })
      ).rejects.toThrow('Always fails');
    });

    it('should run operations in parallel with concurrency limit', async () => {
      const items = [1, 2, 3, 4, 5];
      const results = await utils.parallel(
        items,
        async (item) => item * 2,
        2
      );

      expect(results).toEqual([2, 4, 6, 8, 10]);
    });

    it('should create deferred promise', async () => {
      const deferred = utils.deferred<number>();

      setTimeout(() => deferred.resolve(42), 10);

      const result = await deferred.promise;
      expect(result).toBe(42);
    });

    it('should reject deferred promise', async () => {
      const deferred = utils.deferred<number>();

      setTimeout(() => deferred.reject(new Error('Failed')), 10);

      await expect(deferred.promise).rejects.toThrow('Failed');
    });

    it('should timeout a slow operation', async () => {
      const slowOperation = new Promise<string>((resolve) => {
        setTimeout(() => resolve('done'), 1000);
      });

      await expect(utils.timeout(slowOperation, 50)).rejects.toThrow(utils.TimeoutError);
      await expect(utils.timeout(slowOperation, 50)).rejects.toThrow('timed out');
    });

    it('should not timeout a fast operation', async () => {
      const fastOperation = new Promise<string>((resolve) => {
        setTimeout(() => resolve('fast'), 10);
      });

      const result = await utils.timeout(fastOperation, 100);
      expect(result).toBe('fast');
    });

    it('should withTimeout wrap operations', async () => {
      const result = await utils.withTimeout(
        async () => {
          await utils.sleep(10);
          return 42;
        },
        100
      );
      expect(result).toBe(42);
    });

    it('should sleep for specified duration', async () => {
      const start = Date.now();
      await utils.sleep(50);
      const elapsed = Date.now() - start;
      expect(elapsed).toBeGreaterThanOrEqual(45);
      expect(elapsed).toBeLessThan(150);
    });

    it('should debounce function calls', async () => {
      let callCount = 0;
      const fn = () => { callCount++; };
      const debounced = utils.debounce(fn, 50);

      // Call multiple times rapidly
      debounced();
      debounced();
      debounced();

      // Should not have been called yet
      expect(callCount).toBe(0);

      // Wait for debounce
      await utils.sleep(100);
      expect(callCount).toBe(1);

      // Cancel should prevent further calls
      debounced();
      debounced.cancel();
      await utils.sleep(100);
      expect(callCount).toBe(1);
    });

    it('should debounce with leading edge', async () => {
      let callCount = 0;
      const fn = () => { callCount++; };
      const debounced = utils.debounce(fn, 50, { leading: true });

      // First call should execute immediately
      debounced();
      expect(callCount).toBe(1);

      // Subsequent rapid calls should not execute
      debounced();
      debounced();
      expect(callCount).toBe(1);

      // After debounce period, next call should execute
      await utils.sleep(100);
      debounced();
      expect(callCount).toBe(2);
    });

    it('should throttle function calls', async () => {
      let callCount = 0;
      const fn = () => { callCount++; };
      // Disable trailing to make test more predictable
      const throttled = utils.throttle(fn, 50, { leading: true, trailing: false });

      // First call should execute immediately (leading edge)
      throttled();
      expect(callCount).toBe(1);

      // Rapid calls should be throttled
      throttled();
      throttled();
      expect(callCount).toBe(1);

      // After throttle period, should allow another call
      await utils.sleep(100);
      throttled();
      expect(callCount).toBe(2);

      // Cleanup
      throttled.cancel();
    });

    it('should throttle with trailing edge', async () => {
      let callCount = 0;
      const fn = () => { callCount++; };
      const throttled = utils.throttle(fn, 50, { leading: true, trailing: true });

      // First call executes immediately
      throttled();
      expect(callCount).toBe(1);

      // Rapid calls within throttle period are queued
      throttled();
      throttled();
      expect(callCount).toBe(1);

      // Wait for trailing edge to execute
      await utils.sleep(100);
      expect(callCount).toBe(2); // trailing call executed

      throttled.cancel();
    });

    it('should flush debounced function', async () => {
      let callCount = 0;
      const fn = () => { callCount++; };
      const debounced = utils.debounce(fn, 1000);

      debounced();
      expect(callCount).toBe(0);

      // Flush should execute immediately
      debounced.flush();
      expect(callCount).toBe(1);
    });
  });

  describe('Type Guards', () => {
    it('should validate ProofOfGradientQuality', () => {
      const valid = { quality: 0.9, consistency: 0.8, entropy: 0.1 };
      const invalid = { quality: 1.5, consistency: 0.8, entropy: 0.1 };
      const notAnObject = 'not an object';

      expect(utils.isProofOfGradientQuality(valid)).toBe(true);
      expect(utils.isProofOfGradientQuality(invalid)).toBe(false);
      expect(utils.isProofOfGradientQuality(notAnObject)).toBe(false);
      expect(utils.isProofOfGradientQuality(null)).toBe(false);
    });

    it('should validate ReputationScore', () => {
      const valid = {
        agentId: 'agent-1',
        positiveCount: 10,
        negativeCount: 2,
        lastUpdate: Date.now(),
      };
      const invalid = { agentId: 123, positiveCount: 10 };

      expect(utils.isReputationScore(valid)).toBe(true);
      expect(utils.isReputationScore(invalid)).toBe(false);
    });

    it('should validate EpistemicClaim', () => {
      const valid = {
        id: 'claim-1',
        content: 'Test claim',
        classification: { empirical: 1, normative: 1, materiality: 1 },
        issuedAt: Date.now(),
        issuer: 'test-issuer',
        evidence: [],
      };
      const invalidMissingContent = { id: 'claim-1', issuedAt: Date.now() };
      const invalidWrongPropertyNames = {
        id: 'claim-1',
        statement: 'Test claim', // Wrong: should be 'content'
        classification: { empirical: 1, normative: 1, materiality: 1 },
        createdAt: Date.now(), // Wrong: should be 'issuedAt'
      };

      expect(utils.isEpistemicClaim(valid)).toBe(true);
      expect(utils.isEpistemicClaim(invalidMissingContent)).toBe(false);
      expect(utils.isEpistemicClaim(invalidWrongPropertyNames)).toBe(false);
    });

    it('should validate GradientUpdate', () => {
      const valid = {
        participantId: 'p1',
        modelVersion: 1,
        gradients: new Float64Array([0.1, 0.2]),
        metadata: { batchSize: 32, loss: 0.5, timestamp: Date.now() },
      };
      const invalid = {
        participantId: 'p1',
        modelVersion: 1,
        gradients: [0.1, 0.2], // Not Float64Array!
        metadata: {},
      };

      expect(utils.isGradientUpdate(valid)).toBe(true);
      expect(utils.isGradientUpdate(invalid)).toBe(false);
    });

    it('should validate BridgeMessage', () => {
      const valid = {
        type: 'reputation_query',
        sourceHapp: 'mail-happ',
        timestamp: Date.now(),
        agent: 'agent-123',
      };
      const validCrossHappRep = {
        type: 'cross_happ_reputation',
        sourceHapp: 'bridge',
        timestamp: Date.now(),
        agent: 'agent-123',
        scores: [],
        aggregate: 0.5,
      };
      const invalidType = {
        type: 'invalid_type',
        sourceHapp: 'mail-happ',
        timestamp: Date.now(),
      };
      const invalidMissingSourceHapp = {
        type: 'reputation_query',
        timestamp: Date.now(),
      };

      expect(utils.isBridgeMessage(valid)).toBe(true);
      expect(utils.isBridgeMessage(validCrossHappRep)).toBe(true);
      expect(utils.isBridgeMessage(invalidType)).toBe(false);
      expect(utils.isBridgeMessage(invalidMissingSourceHapp)).toBe(false);
      expect(utils.isBridgeMessage(null)).toBe(false);
    });

    it('should validate SecureMessage', () => {
      const valid = {
        payload: { data: 'test' },
        signature: 'abc123',
        timestamp: Date.now(),
      };
      const invalid = {
        payload: { data: 'test' },
        signature: 123, // Should be string!
        timestamp: Date.now(),
      };

      expect(utils.isSecureMessage(valid)).toBe(true);
      expect(utils.isSecureMessage(invalid)).toBe(false);
    });

    it('should validate TrustCheckResult', () => {
      const valid = {
        trustworthy: true,
        score: 0.9,
        byzantine: false,
        confidence: 0.85,
        details: { pogqScore: 0.9 },
      };
      const invalid = { trustworthy: 'yes' };

      expect(utils.isTrustCheckResult(valid)).toBe(true);
      expect(utils.isTrustCheckResult(invalid)).toBe(false);
    });

    it('should validate HealthCheckResult', () => {
      const valid = { component: 'matl', healthy: true };
      const invalid = { component: 'matl' };

      expect(utils.isHealthCheckResult(valid)).toBe(true);
      expect(utils.isHealthCheckResult(invalid)).toBe(false);
    });

    it('should validate ConnectionState', () => {
      expect(utils.isConnectionState('disconnected')).toBe(true);
      expect(utils.isConnectionState('connecting')).toBe(true);
      expect(utils.isConnectionState('connected')).toBe(true);
      expect(utils.isConnectionState('error')).toBe(true);
      expect(utils.isConnectionState('invalid')).toBe(false);
      expect(utils.isConnectionState(123)).toBe(false);
      expect(utils.isConnectionState(null)).toBe(false);
    });

    it('should validate FLRound', () => {
      const valid = {
        roundId: 1,
        modelVersion: 1,
        participants: new Map(),
        updates: [],
        status: 'collecting',
        startTime: Date.now(),
      };
      const validCompleted = {
        roundId: 2,
        modelVersion: 1,
        participants: new Map(),
        updates: [],
        status: 'completed',
        startTime: Date.now(),
        endTime: Date.now(),
      };
      const invalidStatus = {
        roundId: 1,
        modelVersion: 1,
        participants: new Map(),
        updates: [],
        status: 'invalid',
        startTime: Date.now(),
      };

      expect(utils.isFLRound(valid)).toBe(true);
      expect(utils.isFLRound(validCompleted)).toBe(true);
      expect(utils.isFLRound(invalidStatus)).toBe(false);
      expect(utils.isFLRound(null)).toBe(false);
    });

    it('should validate EpistemicClassification', () => {
      const valid = { empirical: 2, normative: 1, materiality: 3 };
      const outOfRange = { empirical: 5, normative: 1, materiality: 1 };
      const missingField = { empirical: 1, normative: 1 };

      expect(utils.isEpistemicClassification(valid)).toBe(true);
      expect(utils.isEpistemicClassification(outOfRange)).toBe(false);
      expect(utils.isEpistemicClassification(missingField)).toBe(false);
    });

    it('should validate CompositeScore', () => {
      const valid = {
        pogq: { quality: 0.9, consistency: 0.8, entropy: 0.1, timestamp: Date.now() },
        reputation: { agentId: 'a1', positiveCount: 5, negativeCount: 1, lastUpdate: Date.now() },
        pogqScore: 0.85,
        reputationScore: 0.8,
        finalScore: 0.82,
        confidence: 0.9,
        timestamp: Date.now(),
        isTrustworthy: true,
      };
      const missingPogq = {
        reputation: { agentId: 'a1', positiveCount: 5, negativeCount: 1, lastUpdate: Date.now() },
        pogqScore: 0.85,
        reputationScore: 0.8,
        finalScore: 0.82,
        confidence: 0.9,
        timestamp: Date.now(),
        isTrustworthy: true,
      };

      expect(utils.isCompositeScore(valid)).toBe(true);
      expect(utils.isCompositeScore(missingPogq)).toBe(false);
    });

    it('should validate FLConfig', () => {
      const valid = {
        minParticipants: 3,
        maxParticipants: 100,
        roundTimeout: 60000,
        byzantineTolerance: 0.33,
        aggregationMethod: 'trust_weighted',
        trustThreshold: 0.5,
      };
      const invalidTolerance = {
        minParticipants: 3,
        maxParticipants: 100,
        roundTimeout: 60000,
        byzantineTolerance: 0.9, // Too high!
        aggregationMethod: 'trust_weighted',
        trustThreshold: 0.5,
      };

      expect(utils.isFLConfig(valid)).toBe(true);
      expect(utils.isFLConfig(invalidTolerance)).toBe(false);
    });

    it('should validate MycelixConfig', () => {
      const valid = {
        matl: { trustThreshold: 0.5 },
        fl: { minParticipants: 3 },
        security: { enabled: true },
        bridge: { timeout: 5000 },
        client: { appUrl: 'ws://localhost:8888' },
        logging: { level: 'info' },
      };
      const missingModule = {
        matl: {},
        fl: {},
        security: {},
        bridge: {},
        client: {},
        // Missing logging!
      };

      expect(utils.isMycelixConfig(valid)).toBe(true);
      expect(utils.isMycelixConfig(missingModule)).toBe(false);
    });

    it('should validate AggregatedGradient', () => {
      const valid = {
        gradients: new Float64Array([0.1, 0.2, 0.3]),
        modelVersion: 1,
        participantCount: 5,
        method: 'trust_weighted',
        timestamp: Date.now(),
      };
      const invalidGradients = {
        gradients: [0.1, 0.2, 0.3], // Not Float64Array!
        modelVersion: 1,
        participantCount: 5,
        method: 'trust_weighted',
        timestamp: Date.now(),
      };

      expect(utils.isAggregatedGradient(valid)).toBe(true);
      expect(utils.isAggregatedGradient(invalidGradients)).toBe(false);
    });

    it('should validate SerializedGradientUpdate', () => {
      const valid: fl.SerializedGradientUpdate = {
        participantId: 'agent-1',
        modelVersion: 1,
        gradients: 'AAAAAAAA8D8=', // base64 encoded
        metadata: {
          batchSize: 32,
          loss: 0.5,
          timestamp: Date.now(),
        },
      };

      const withAccuracy: fl.SerializedGradientUpdate = {
        ...valid,
        metadata: { ...valid.metadata, accuracy: 0.95 },
      };

      expect(utils.isSerializedGradientUpdate(valid)).toBe(true);
      expect(utils.isSerializedGradientUpdate(withAccuracy)).toBe(true);

      // Invalid cases
      expect(utils.isSerializedGradientUpdate(null)).toBe(false);
      expect(utils.isSerializedGradientUpdate({})).toBe(false);
      expect(utils.isSerializedGradientUpdate({ ...valid, participantId: '' })).toBe(false);
      expect(utils.isSerializedGradientUpdate({ ...valid, gradients: '' })).toBe(false);
      expect(utils.isSerializedGradientUpdate({ ...valid, modelVersion: NaN })).toBe(false);
      expect(utils.isSerializedGradientUpdate({ ...valid, metadata: null })).toBe(false);
      expect(utils.isSerializedGradientUpdate({ ...valid, metadata: { ...valid.metadata, batchSize: 0 } })).toBe(false);
      expect(utils.isSerializedGradientUpdate({ ...valid, metadata: { ...valid.metadata, loss: NaN } })).toBe(false);
      expect(utils.isSerializedGradientUpdate({ ...valid, metadata: { ...valid.metadata, accuracy: 'high' } })).toBe(false);
    });

    it('should validate SerializedAggregatedGradient', () => {
      const valid: fl.SerializedAggregatedGradient = {
        gradients: 'AAAAAAAA8D8=', // base64 encoded
        modelVersion: 1,
        participantCount: 5,
        aggregationMethod: 'fedAvg',
        timestamp: Date.now(),
      };

      const withExcluded: fl.SerializedAggregatedGradient = {
        ...valid,
        excludedCount: 2,
      };

      expect(utils.isSerializedAggregatedGradient(valid)).toBe(true);
      expect(utils.isSerializedAggregatedGradient(withExcluded)).toBe(true);

      // All aggregation methods
      expect(utils.isSerializedAggregatedGradient({ ...valid, aggregationMethod: 'trimmedMean' })).toBe(true);
      expect(utils.isSerializedAggregatedGradient({ ...valid, aggregationMethod: 'coordinateMedian' })).toBe(true);
      expect(utils.isSerializedAggregatedGradient({ ...valid, aggregationMethod: 'krum' })).toBe(true);
      expect(utils.isSerializedAggregatedGradient({ ...valid, aggregationMethod: 'trustWeighted' })).toBe(true);

      // Invalid cases
      expect(utils.isSerializedAggregatedGradient(null)).toBe(false);
      expect(utils.isSerializedAggregatedGradient({})).toBe(false);
      expect(utils.isSerializedAggregatedGradient({ ...valid, gradients: '' })).toBe(false);
      expect(utils.isSerializedAggregatedGradient({ ...valid, participantCount: 0 })).toBe(false);
      expect(utils.isSerializedAggregatedGradient({ ...valid, aggregationMethod: 'invalid' })).toBe(false);
      expect(utils.isSerializedAggregatedGradient({ ...valid, timestamp: -1 })).toBe(false);
      expect(utils.isSerializedAggregatedGradient({ ...valid, excludedCount: -1 })).toBe(false);
    });
  });

  // ============================================================================
  // Observable Event System Tests
  // ============================================================================

  describe('Observable Event System', () => {
    describe('Observable', () => {
      it('should subscribe and receive events', () => {
        const observable = new utils.Observable<number>();
        const received: number[] = [];

        observable.subscribe((event) => received.push(event));

        observable.emit(1);
        observable.emit(2);
        observable.emit(3);

        expect(received).toEqual([1, 2, 3]);
      });

      it('should support multiple observers', () => {
        const observable = new utils.Observable<string>();
        const received1: string[] = [];
        const received2: string[] = [];

        observable.subscribe((event) => received1.push(event));
        observable.subscribe((event) => received2.push(event));

        observable.emit('hello');

        expect(received1).toEqual(['hello']);
        expect(received2).toEqual(['hello']);
      });

      it('should unsubscribe correctly', () => {
        const observable = new utils.Observable<number>();
        const received: number[] = [];

        const subscription = observable.subscribe((event) => received.push(event));

        observable.emit(1);
        subscription.unsubscribe();
        observable.emit(2);

        expect(received).toEqual([1]);
        expect(subscription.closed).toBe(true);
      });

      it('should track statistics', () => {
        const observable = new utils.Observable<number>();

        observable.subscribe(() => {});
        observable.subscribe(() => {});

        observable.emit(1);
        observable.emit(2);

        const stats = observable.getStats();
        expect(stats.eventCount).toBe(2);
        expect(stats.observerCount).toBe(2);
        expect(stats.lastEventTime).toBeDefined();
      });

      it('should get last event', () => {
        const observable = new utils.Observable<string>();

        expect(observable.getLastEvent()).toBeUndefined();

        observable.emit('first');
        observable.emit('second');

        expect(observable.getLastEvent()).toBe('second');
      });

      it('should clear observers', () => {
        const observable = new utils.Observable<number>();

        observable.subscribe(() => {});
        observable.subscribe(() => {});

        expect(observable.getObserverCount()).toBe(2);

        observable.clear();

        expect(observable.getObserverCount()).toBe(0);
      });

      it('should reset stats', () => {
        const observable = new utils.Observable<number>();

        observable.emit(1);
        observable.emit(2);

        expect(observable.getEventCount()).toBe(2);

        observable.resetStats();

        expect(observable.getEventCount()).toBe(0);
        expect(observable.getLastEvent()).toBeUndefined();
      });

      it('should not break on observer errors', () => {
        const observable = new utils.Observable<number>();
        const received: number[] = [];

        observable.subscribe(() => {
          throw new Error('Observer error');
        });
        observable.subscribe((event) => received.push(event));

        // Should not throw
        observable.emit(1);

        expect(received).toEqual([1]);
      });

      it('should report hasObservers correctly', () => {
        const observable = new utils.Observable<number>();

        expect(observable.hasObservers()).toBe(false);

        const sub = observable.subscribe(() => {});
        expect(observable.hasObservers()).toBe(true);

        sub.unsubscribe();
        expect(observable.hasObservers()).toBe(false);
      });
    });

    describe('SdkEventBus', () => {
      it('should emit and receive events via onAll', () => {
        const bus = new utils.SdkEventBus();
        const received: utils.AnySdkEvent[] = [];

        bus.onAll((event) => received.push(event));

        const event = utils.createReputationUpdatedEvent('agent1', 0.5, 0.6, 'positive');
        bus.emit(event);

        expect(received.length).toBe(1);
        expect(received[0].type).toBe(utils.MatlEventType.REPUTATION_UPDATED);
      });

      it('should emit to type-specific subscribers', () => {
        const bus = new utils.SdkEventBus();
        const matlEvents: utils.ReputationUpdatedEvent[] = [];
        const flEvents: utils.RoundStartedEvent[] = [];

        bus.on(utils.MatlEventType.REPUTATION_UPDATED, (event) => {
          matlEvents.push(event);
        });
        bus.on(utils.FlEventType.ROUND_STARTED, (event) => {
          flEvents.push(event);
        });

        bus.emit(utils.createReputationUpdatedEvent('agent1', 0.5, 0.6, 'positive'));
        bus.emit(utils.createRoundStartedEvent(1, 1, 5));

        expect(matlEvents.length).toBe(1);
        expect(flEvents.length).toBe(1);
      });

      it('should maintain event history', () => {
        const bus = new utils.SdkEventBus({ maxHistory: 10 });

        for (let i = 0; i < 15; i++) {
          bus.emit(utils.createReputationUpdatedEvent(`agent${i}`, 0.5, 0.6, 'positive'));
        }

        const history = bus.getHistory();
        expect(history.length).toBe(10);
      });

      it('should filter history by type', () => {
        const bus = new utils.SdkEventBus();

        bus.emit(utils.createReputationUpdatedEvent('agent1', 0.5, 0.6, 'positive'));
        bus.emit(utils.createRoundStartedEvent(1, 1, 5));
        bus.emit(utils.createReputationUpdatedEvent('agent2', 0.5, 0.6, 'positive'));

        const matlHistory = bus.getHistory({ type: utils.MatlEventType.REPUTATION_UPDATED });
        expect(matlHistory.length).toBe(2);
      });

      it('should limit history results', () => {
        const bus = new utils.SdkEventBus();

        for (let i = 0; i < 10; i++) {
          bus.emit(utils.createReputationUpdatedEvent(`agent${i}`, 0.5, 0.6, 'positive'));
        }

        const limited = bus.getHistory({ limit: 3 });
        expect(limited.length).toBe(3);
      });

      it('should provide statistics', () => {
        const bus = new utils.SdkEventBus();

        bus.onAll(() => {});
        bus.on(utils.MatlEventType.REPUTATION_UPDATED, () => {});

        bus.emit(utils.createReputationUpdatedEvent('agent1', 0.5, 0.6, 'positive'));

        const stats = bus.getStats();
        expect(stats.totalEvents).toBe(1);
        expect(stats.historySize).toBe(1);
        expect(stats.observerCount).toBe(1);
        expect(stats.typeObserverCounts[utils.MatlEventType.REPUTATION_UPDATED]).toBe(1);
      });

      it('should clear history', () => {
        const bus = new utils.SdkEventBus();

        bus.emit(utils.createReputationUpdatedEvent('agent1', 0.5, 0.6, 'positive'));
        expect(bus.getHistory().length).toBe(1);

        bus.clearHistory();
        expect(bus.getHistory().length).toBe(0);
      });

      it('should clear all observers and history', () => {
        const bus = new utils.SdkEventBus();

        bus.onAll(() => {});
        bus.on(utils.MatlEventType.REPUTATION_UPDATED, () => {});
        bus.emit(utils.createReputationUpdatedEvent('agent1', 0.5, 0.6, 'positive'));

        bus.clear();

        const stats = bus.getStats();
        expect(stats.observerCount).toBe(0);
        expect(stats.historySize).toBe(0);
      });
    });

    describe('Event Factory Functions', () => {
      it('should create reputation updated event', () => {
        const event = utils.createReputationUpdatedEvent('agent1', 0.5, 0.6, 'positive');

        expect(event.type).toBe(utils.MatlEventType.REPUTATION_UPDATED);
        expect(event.agentId).toBe('agent1');
        expect(event.previousValue).toBe(0.5);
        expect(event.newValue).toBe(0.6);
        expect(event.change).toBe('positive');
        expect(event.timestamp).toBeDefined();
        expect(event.source).toBe('matl');
      });

      it('should create trust evaluated event', () => {
        const event = utils.createTrustEvaluatedEvent('agent1', 0.85, true, 0.9);

        expect(event.type).toBe(utils.MatlEventType.TRUST_EVALUATED);
        expect(event.agentId).toBe('agent1');
        expect(event.score).toBe(0.85);
        expect(event.trustworthy).toBe(true);
        expect(event.confidence).toBe(0.9);
      });

      it('should create message routed event', () => {
        const event = utils.createMessageRoutedEvent('reputation_query', 'mail-happ', 2);

        expect(event.type).toBe(utils.BridgeEventType.MESSAGE_ROUTED);
        expect(event.messageType).toBe('reputation_query');
        expect(event.sourceHapp).toBe('mail-happ');
        expect(event.handlerCount).toBe(2);
      });

      it('should create round started event', () => {
        const event = utils.createRoundStartedEvent(1, 5, 10);

        expect(event.type).toBe(utils.FlEventType.ROUND_STARTED);
        expect(event.roundId).toBe(1);
        expect(event.modelVersion).toBe(5);
        expect(event.participantCount).toBe(10);
      });

      it('should create round aggregated event', () => {
        const event = utils.createRoundAggregatedEvent(1, 5, 10, 2, 'trust_weighted', 150);

        expect(event.type).toBe(utils.FlEventType.ROUND_AGGREGATED);
        expect(event.roundId).toBe(1);
        expect(event.modelVersion).toBe(5);
        expect(event.participantCount).toBe(10);
        expect(event.excludedCount).toBe(2);
        expect(event.aggregationMethod).toBe('trust_weighted');
        expect(event.durationMs).toBe(150);
      });

      it('should create claim created event', () => {
        const event = utils.createClaimCreatedEvent('claim-1', 'issuer-1', 'E3N2M2');

        expect(event.type).toBe(utils.EpistemicEventType.CLAIM_CREATED);
        expect(event.claimId).toBe('claim-1');
        expect(event.issuer).toBe('issuer-1');
        expect(event.classificationCode).toBe('E3N2M2');
      });

      it('should create pool cleanup event', () => {
        const event = utils.createPoolCleanupEvent(5, 10);

        expect(event.type).toBe(utils.EpistemicEventType.POOL_CLEANUP);
        expect(event.removedCount).toBe(5);
        expect(event.remainingCount).toBe(10);
      });
    });

    describe('Global SDK Event Bus', () => {
      it('should export global sdkEvents instance', () => {
        expect(utils.sdkEvents).toBeDefined();
        expect(utils.sdkEvents).toBeInstanceOf(utils.SdkEventBus);
      });
    });
  });

  // ============================================================================
  // Batch Request Optimization Tests
  // ============================================================================

  describe('Batch Request Optimization', () => {
    describe('RequestBatcher', () => {
      it('should batch multiple requests together', async () => {
        const processCalls: number[][] = [];
        const batcher = new utils.RequestBatcher<number, number>(
          async (batch) => {
            processCalls.push([...batch]);
            return batch.map((n) => n * 2);
          },
          { batchSizeLimit: 10, timeLimitMs: 50 }
        );

        const results = await Promise.all([
          batcher.add(1),
          batcher.add(2),
          batcher.add(3),
        ]);

        expect(results).toEqual([2, 4, 6]);
        expect(processCalls.length).toBe(1);
        expect(processCalls[0]).toEqual([1, 2, 3]);
      });

      it('should auto-flush when batch size limit reached', async () => {
        const processCalls: number = 0;
        const batcher = new utils.RequestBatcher<number, number>(
          async (batch) => batch.map((n) => n),
          { batchSizeLimit: 3, timeLimitMs: 10000 }
        );

        const promises = [
          batcher.add(1),
          batcher.add(2),
          batcher.add(3),
        ];

        // Should flush immediately when limit reached
        const results = await Promise.all(promises);
        expect(results).toEqual([1, 2, 3]);
      });

      it('should track statistics correctly', async () => {
        const batcher = new utils.RequestBatcher<number, number>(
          async (batch) => batch.map((n) => n * 2),
          { batchSizeLimit: 5, timeLimitMs: 50 }
        );

        await Promise.all([batcher.add(1), batcher.add(2), batcher.add(3)]);

        const stats = batcher.getStats();
        expect(stats.totalRequests).toBe(3);
        expect(stats.totalBatches).toBe(1);
        expect(stats.averageBatchSize).toBe(3);
        expect(stats.requestsSaved).toBe(2); // 3 requests in 1 batch = 2 saved
      });

      it('should cancel pending requests', async () => {
        const batcher = new utils.RequestBatcher<number, number>(
          async (batch) => batch.map((n) => n * 2),
          { batchSizeLimit: 100, timeLimitMs: 10000 }
        );

        const promise = batcher.add(1);
        expect(batcher.getPendingCount()).toBe(1);

        batcher.cancel();
        expect(batcher.getPendingCount()).toBe(0);

        await expect(promise).rejects.toThrow('Batch cancelled');
      });

      it('should reject all on processor error', async () => {
        const batcher = new utils.RequestBatcher<number, number>(
          async () => {
            throw new Error('Processor failed');
          },
          { batchSizeLimit: 3, timeLimitMs: 50 }
        );

        const promises = [batcher.add(1), batcher.add(2), batcher.add(3)];

        await expect(Promise.all(promises)).rejects.toThrow('Processor failed');
      });

      it('should reset statistics', async () => {
        const batcher = new utils.RequestBatcher<number, number>(
          async (batch) => batch.map((n) => n),
          { batchSizeLimit: 10, timeLimitMs: 50 }
        );

        await Promise.all([batcher.add(1), batcher.add(2)]);
        batcher.resetStats();

        const stats = batcher.getStats();
        expect(stats.totalRequests).toBe(0);
        expect(stats.totalBatches).toBe(0);
      });
    });
  });

  // ============================================================================
  // Event Pipeline Composition Tests
  // ============================================================================

  describe('Event Pipeline Composition', () => {
    describe('EventPipeline', () => {
      it('should filter events', async () => {
        const source = new utils.Observable<number>();
        const received: number[] = [];

        const pipeline = new utils.EventPipeline(source)
          .filter((n) => n > 2)
          .subscribe((n) => received.push(n));

        source.emit(1);
        source.emit(2);
        source.emit(3);
        source.emit(4);

        expect(received).toEqual([3, 4]);

        pipeline.unsubscribe();
      });

      it('should map events', async () => {
        const source = new utils.Observable<number>();
        const received: string[] = [];

        const pipeline = new utils.EventPipeline(source)
          .map((n) => `value: ${n}`)
          .subscribe((s) => received.push(s));

        source.emit(1);
        source.emit(2);

        expect(received).toEqual(['value: 1', 'value: 2']);

        pipeline.unsubscribe();
      });

      it('should chain filter and map', async () => {
        const source = new utils.Observable<number>();
        const received: string[] = [];

        const pipeline = new utils.EventPipeline(source)
          .filter((n) => n % 2 === 0)
          .map((n) => `even: ${n}`)
          .subscribe((s) => received.push(s));

        source.emit(1);
        source.emit(2);
        source.emit(3);
        source.emit(4);

        expect(received).toEqual(['even: 2', 'even: 4']);

        pipeline.unsubscribe();
      });

      it('should buffer events', async () => {
        const source = new utils.Observable<number>();
        const received: number[][] = [];

        const pipeline = new utils.EventPipeline(source)
          .buffer(3)
          .subscribe((batch) => received.push(batch));

        source.emit(1);
        source.emit(2);
        source.emit(3);
        source.emit(4);
        source.emit(5);

        expect(received).toEqual([[1, 2, 3]]);

        pipeline.unsubscribe();
      });

      it('should take first N events', async () => {
        const source = new utils.Observable<number>();
        const received: number[] = [];

        const pipeline = new utils.EventPipeline(source)
          .take(2)
          .subscribe((n) => received.push(n));

        source.emit(1);
        source.emit(2);
        source.emit(3);
        source.emit(4);

        expect(received).toEqual([1, 2]);

        pipeline.unsubscribe();
      });

      it('should skip first N events', async () => {
        const source = new utils.Observable<number>();
        const received: number[] = [];

        const pipeline = new utils.EventPipeline(source)
          .skip(2)
          .subscribe((n) => received.push(n));

        source.emit(1);
        source.emit(2);
        source.emit(3);
        source.emit(4);

        expect(received).toEqual([3, 4]);

        pipeline.unsubscribe();
      });

      it('should emit distinct events', async () => {
        const source = new utils.Observable<{ id: number; value: string }>();
        const received: Array<{ id: number; value: string }> = [];

        const pipeline = new utils.EventPipeline(source)
          .distinct((e) => e.id)
          .subscribe((e) => received.push(e));

        source.emit({ id: 1, value: 'a' });
        source.emit({ id: 2, value: 'b' });
        source.emit({ id: 1, value: 'c' }); // Duplicate id
        source.emit({ id: 3, value: 'd' });

        expect(received.length).toBe(3);
        expect(received.map((r) => r.id)).toEqual([1, 2, 3]);

        pipeline.unsubscribe();
      });

      it('should create pipeline from helper function', () => {
        const source = new utils.Observable<number>();
        const p = utils.pipeline(source);

        expect(p).toBeInstanceOf(utils.EventPipeline);
      });
    });
  });

  // ============================================================================
  // Distributed Tracing Tests
  // ============================================================================

  describe('Distributed Tracing', () => {
    describe('TracingManager', () => {
      it('should start and end traces', () => {
        const manager = new utils.TracingManager();
        const context = manager.startTrace('testOperation');

        expect(context.traceId).toBeDefined();
        expect(context.spanId).toBeDefined();
        expect(context.startTime).toBeDefined();

        const completed = manager.endTrace(context.traceId);

        expect(completed).not.toBeNull();
        expect(completed?.traceId).toBe(context.traceId);
        expect(completed?.rootOperation).toBe('testOperation');
        expect(completed?.status).toBe('success');
        expect(completed?.totalDuration).toBeGreaterThanOrEqual(0);
      });

      it('should create child spans', () => {
        const manager = new utils.TracingManager();
        const context = manager.startTrace('parent');

        const childSpan = manager.startSpan(context, 'child');

        expect(childSpan.parentSpanId).toBe(context.spanId);
        expect(childSpan.traceId).toBe(context.traceId);
        expect(childSpan.operationName).toBe('child');

        manager.endSpan(childSpan.spanId, 'success');
        const completed = manager.endTrace(context.traceId);

        expect(completed?.spans.length).toBe(2);
      });

      it('should add tags to spans', () => {
        const manager = new utils.TracingManager();
        const context = manager.startTrace('operation');
        const span = manager.startSpan(context, 'child');

        manager.tagSpan(span.spanId, 'method', 'trustWeighted');
        manager.tagSpan(span.spanId, 'participants', 10);
        manager.tagSpan(span.spanId, 'success', true);

        const retrieved = manager.getSpan(span.spanId);
        expect(retrieved?.tags['method']).toBe('trustWeighted');
        expect(retrieved?.tags['participants']).toBe(10);
        expect(retrieved?.tags['success']).toBe(true);

        manager.endTrace(context.traceId);
      });

      it('should add log entries to spans', () => {
        const manager = new utils.TracingManager();
        const context = manager.startTrace('operation');

        manager.logToSpan(context.spanId, 'Starting operation', 'info');
        manager.logToSpan(context.spanId, 'Processing...', 'debug');
        manager.logToSpan(context.spanId, 'Warning!', 'warn');

        const span = manager.getSpan(context.spanId);
        expect(span?.logs.length).toBe(3);
        expect(span?.logs[0].message).toBe('Starting operation');
        expect(span?.logs[0].level).toBe('info');

        manager.endTrace(context.traceId);
      });

      it('should track error status', () => {
        const manager = new utils.TracingManager();
        const context = manager.startTrace('failing');

        manager.endSpan(context.spanId, 'error', new Error('Test error'));
        const completed = manager.endTrace(context.traceId);

        expect(completed?.status).toBe('error');
        expect(completed?.spans[0].error?.message).toBe('Test error');
      });

      it('should serialize and deserialize context', () => {
        const manager = new utils.TracingManager();
        const context = manager.startTrace('test', { key: 'value' });

        const serialized = manager.serializeContext(context);
        const deserialized = manager.deserializeContext(serialized);

        expect(deserialized?.traceId).toBe(context.traceId);
        expect(deserialized?.spanId).toBe(context.spanId);
        expect(deserialized?.baggage['key']).toBe('value');

        manager.endTrace(context.traceId);
      });

      it('should maintain completed traces history', () => {
        const manager = new utils.TracingManager({ maxCompletedTraces: 3 });

        for (let i = 0; i < 5; i++) {
          const ctx = manager.startTrace(`op${i}`);
          manager.endTrace(ctx.traceId);
        }

        const history = manager.getCompletedTraces();
        expect(history.length).toBe(3);
      });

      it('should notify listeners on trace completion', () => {
        const manager = new utils.TracingManager();
        const completed: utils.CompletedTrace[] = [];

        manager.onTraceComplete((trace) => completed.push(trace));

        const ctx = manager.startTrace('traced');
        manager.endTrace(ctx.traceId);

        expect(completed.length).toBe(1);
        expect(completed[0].rootOperation).toBe('traced');
      });
    });

    describe('traced helper', () => {
      it('should trace async operations', async () => {
        const result = await utils.traced('asyncOp', async () => {
          await utils.sleep(10);
          return 42;
        });

        expect(result).toBe(42);
      });

      it('should capture errors in traced operations', async () => {
        await expect(
          utils.traced('failingOp', async () => {
            throw new Error('Test failure');
          })
        ).rejects.toThrow('Test failure');
      });
    });

    describe('Global tracer', () => {
      it('should export global tracer instance', () => {
        expect(utils.tracer).toBeDefined();
        expect(utils.tracer).toBeInstanceOf(utils.TracingManager);
      });
    });
  });

  // ============================================================================
  // Policy-Based Authorization Tests
  // ============================================================================

  describe('Policy-Based Authorization', () => {
    describe('PolicyEngine', () => {
      it('should add and check policies', () => {
        const engine = new utils.PolicyEngine();

        engine.addPolicy({
          id: 'read-reputation',
          name: 'Read Reputation',
          resource: 'reputation',
          actions: ['read'],
        });

        const result = engine.check({
          agentId: 'agent1',
          resource: 'reputation',
          action: 'read',
        });

        expect(result.allowed).toBe(true);
        expect(result.matchedPolicy).toBe('read-reputation');
      });

      it('should deny by default when no policy matches', () => {
        const engine = new utils.PolicyEngine();

        const result = engine.check({
          agentId: 'agent1',
          resource: 'reputation',
          action: 'delete',
        });

        expect(result.allowed).toBe(false);
        expect(result.reason).toContain('default deny');
      });

      it('should evaluate conditions', () => {
        const engine = new utils.PolicyEngine();

        engine.addPolicy({
          id: 'read-own',
          name: 'Read Own Reputation',
          resource: 'reputation',
          actions: ['read'],
          condition: (ctx) => ctx.resourceId === ctx.agentId,
        });

        const ownResult = engine.check({
          agentId: 'agent1',
          resource: 'reputation',
          action: 'read',
          resourceId: 'agent1',
        });

        const otherResult = engine.check({
          agentId: 'agent1',
          resource: 'reputation',
          action: 'read',
          resourceId: 'agent2',
        });

        expect(ownResult.allowed).toBe(true);
        expect(otherResult.allowed).toBe(false);
      });

      it('should respect policy priority', () => {
        const engine = new utils.PolicyEngine();

        engine.addPolicy({
          id: 'low-priority',
          name: 'Low Priority',
          resource: 'reputation',
          actions: ['read'],
          priority: 1,
        });

        engine.addPolicy({
          id: 'high-priority',
          name: 'High Priority',
          resource: 'reputation',
          actions: ['read'],
          priority: 100,
        });

        const result = engine.check({
          agentId: 'agent1',
          resource: 'reputation',
          action: 'read',
        });

        expect(result.matchedPolicy).toBe('high-priority');
      });

      it('should remove policies', () => {
        const engine = new utils.PolicyEngine();

        engine.addPolicy({
          id: 'test-policy',
          name: 'Test',
          resource: 'reputation',
          actions: ['read'],
        });

        expect(engine.getPolicies().length).toBe(1);
        expect(engine.removePolicy('test-policy')).toBe(true);
        expect(engine.getPolicies().length).toBe(0);
      });

      it('should check multiple actions', () => {
        const engine = new utils.PolicyEngine();

        engine.addPolicy({
          id: 'read-only',
          name: 'Read Only',
          resource: 'claim',
          actions: ['read'],
        });

        const results = engine.checkMany('agent1', 'claim', ['read', 'create', 'delete']);

        expect(results.get('read')?.allowed).toBe(true);
        expect(results.get('create')?.allowed).toBe(false);
        expect(results.get('delete')?.allowed).toBe(false);
      });

      it('should get policies for resource', () => {
        const engine = new utils.PolicyEngine();

        engine.addPolicy({
          id: 'rep-read',
          name: 'Reputation Read',
          resource: 'reputation',
          actions: ['read'],
        });

        engine.addPolicy({
          id: 'claim-read',
          name: 'Claim Read',
          resource: 'claim',
          actions: ['read'],
        });

        const repPolicies = engine.getPoliciesForResource('reputation');
        expect(repPolicies.length).toBe(1);
        expect(repPolicies[0].id).toBe('rep-read');
      });
    });

    describe('Global policyEngine', () => {
      it('should export global policyEngine instance', () => {
        expect(utils.policyEngine).toBeDefined();
        expect(utils.policyEngine).toBeInstanceOf(utils.PolicyEngine);
      });
    });
  });

  // ============================================================================
  // Time-Series Analytics Tests
  // ============================================================================

  describe('Time-Series Analytics', () => {
    describe('TimeSeriesAnalytics', () => {
      it('should record and retrieve data points', () => {
        const analytics = new utils.TimeSeriesAnalytics<'metric1'>();

        analytics.record('metric1', 10);
        analytics.record('metric1', 20);
        analytics.record('metric1', 30);

        const series = analytics.getSeries('metric1', '1m');
        expect(series.length).toBe(3);
        expect(series.map((p) => p.value)).toEqual([10, 20, 30]);
      });

      it('should calculate basic statistics', () => {
        const analytics = new utils.TimeSeriesAnalytics<'metric'>();

        analytics.record('metric', 10);
        analytics.record('metric', 20);
        analytics.record('metric', 30);
        analytics.record('metric', 40);
        analytics.record('metric', 50);

        const stats = analytics.getStats('metric', '1m');

        expect(stats.count).toBe(5);
        expect(stats.min).toBe(10);
        expect(stats.max).toBe(50);
        expect(stats.avg).toBe(30);
        expect(stats.sum).toBe(150);
      });

      it('should calculate standard deviation', () => {
        const analytics = new utils.TimeSeriesAnalytics<'metric'>();

        // Values with known std dev
        analytics.record('metric', 2);
        analytics.record('metric', 4);
        analytics.record('metric', 4);
        analytics.record('metric', 4);
        analytics.record('metric', 5);
        analytics.record('metric', 5);
        analytics.record('metric', 7);
        analytics.record('metric', 9);

        const stats = analytics.getStats('metric', '1m');

        expect(stats.stdDev).toBeCloseTo(2, 1);
      });

      it('should analyze trends', () => {
        const analytics = new utils.TimeSeriesAnalytics<'increasing'>();
        const now = Date.now();

        // Increasing trend
        for (let i = 0; i < 10; i++) {
          analytics.record('increasing', i * 10, now + i * 1000);
        }

        const stats = analytics.getStats('increasing', '1m');
        expect(stats.trend.direction).toBe('increasing');
        expect(stats.trend.slope).toBeGreaterThan(0);
      });

      it('should get latest value', () => {
        const analytics = new utils.TimeSeriesAnalytics<'metric'>();

        analytics.record('metric', 10);
        analytics.record('metric', 20);
        analytics.record('metric', 30);

        const latest = analytics.getLatest('metric');
        expect(latest?.value).toBe(30);
      });

      it('should export and import data', () => {
        const analytics1 = new utils.TimeSeriesAnalytics<'metric'>();
        const analytics2 = new utils.TimeSeriesAnalytics<'metric'>();

        analytics1.record('metric', 10);
        analytics1.record('metric', 20);

        const exported = analytics1.export('metric');
        analytics2.import('metric', exported);

        const series = analytics2.getSeries('metric', '1m');
        expect(series.length).toBe(2);
      });

      it('should clear data', () => {
        const analytics = new utils.TimeSeriesAnalytics<'metric1' | 'metric2'>();

        analytics.record('metric1', 10);
        analytics.record('metric2', 20);

        analytics.clear('metric1');
        expect(analytics.getSeries('metric1', '1m').length).toBe(0);
        expect(analytics.getSeries('metric2', '1m').length).toBe(1);

        analytics.clear();
        expect(analytics.getMetrics().length).toBe(0);
      });

      it('should return empty stats for unknown metric', () => {
        const analytics = new utils.TimeSeriesAnalytics<'unknown'>();
        const stats = analytics.getStats('unknown', '1m');

        expect(stats.count).toBe(0);
        expect(stats.avg).toBe(0);
        expect(stats.trend.direction).toBe('stable');
      });
    });

    describe('Global analytics instances', () => {
      it('should export reputationAnalytics', () => {
        expect(utils.reputationAnalytics).toBeDefined();
        expect(utils.reputationAnalytics).toBeInstanceOf(utils.TimeSeriesAnalytics);
      });

      it('should export flAnalytics', () => {
        expect(utils.flAnalytics).toBeDefined();
        expect(utils.flAnalytics).toBeInstanceOf(utils.TimeSeriesAnalytics);
      });
    });
  });

  // ============================================================================
  // Claim Verification Framework Tests
  // ============================================================================

  describe('Claim Verification Framework', () => {
    describe('ClaimVerifier', () => {
      it('should register and use verifiers', async () => {
        const verifier = new utils.ClaimVerifier();

        verifier.registerVerifier('witness', async (proof) => {
          return proof.data === 'valid';
        });

        const result = await verifier.verify('claim-1', [
          { type: 'witness', data: 'valid', createdAt: Date.now() },
        ]);

        expect(result.verified).toBe(true);
        expect(result.passedCount).toBe(1);
      });

      it('should handle verification failure', async () => {
        const verifier = new utils.ClaimVerifier();

        verifier.registerVerifier('witness', async () => false);

        const result = await verifier.verify('claim-1', [
          { type: 'witness', data: 'invalid', createdAt: Date.now() },
        ], { strategy: 'all' });

        expect(result.verified).toBe(false);
        expect(result.failedCount).toBe(1);
      });

      it('should use "all" strategy by default', async () => {
        const verifier = new utils.ClaimVerifier();

        verifier.registerVerifier('witness', async () => true);

        const result = await verifier.verify('claim-1', [
          { type: 'witness', data: 'a', createdAt: Date.now() },
          { type: 'witness', data: 'b', createdAt: Date.now() },
        ]);

        expect(result.verified).toBe(true);
      });

      it('should use "any" strategy', async () => {
        const verifier = new utils.ClaimVerifier();
        let callCount = 0;

        verifier.registerVerifier('witness', async () => {
          callCount++;
          return callCount === 1; // First passes, second fails
        });

        const result = await verifier.verify(
          'claim-1',
          [
            { type: 'witness', data: 'a', createdAt: Date.now() },
            { type: 'witness', data: 'b', createdAt: Date.now() },
          ],
          { strategy: 'any' }
        );

        expect(result.verified).toBe(true);
        expect(result.passedCount).toBe(1);
      });

      it('should use "majority" strategy', async () => {
        const verifier = new utils.ClaimVerifier();
        let callCount = 0;

        verifier.registerVerifier('witness', async () => {
          callCount++;
          return callCount <= 2; // First 2 pass, third fails
        });

        const result = await verifier.verify(
          'claim-1',
          [
            { type: 'witness', data: 'a', createdAt: Date.now() },
            { type: 'witness', data: 'b', createdAt: Date.now() },
            { type: 'witness', data: 'c', createdAt: Date.now() },
          ],
          { strategy: 'majority' }
        );

        expect(result.verified).toBe(true);
        expect(result.passedCount).toBe(2);
      });

      it('should use "threshold" strategy', async () => {
        const verifier = new utils.ClaimVerifier();

        verifier.registerVerifier('witness', async () => true);

        const result = await verifier.verify(
          'claim-1',
          [
            { type: 'witness', data: 'a', createdAt: Date.now() },
            { type: 'witness', data: 'b', createdAt: Date.now() },
          ],
          { strategy: 'threshold', threshold: 2 }
        );

        expect(result.verified).toBe(true);
      });

      it('should check for required proof types', async () => {
        const verifier = new utils.ClaimVerifier();

        verifier.registerVerifier('signature', async () => true);

        const result = await verifier.verify(
          'claim-1',
          [{ type: 'signature', data: 'sig', createdAt: Date.now() }],
          { strategy: 'all', requiredTypes: ['signature', 'hash'] }
        );

        expect(result.verified).toBe(false);
        expect(result.missingProofs).toContain('hash');
      });

      it('should fail on expired proofs when configured', async () => {
        const verifier = new utils.ClaimVerifier();

        verifier.registerVerifier('witness', async () => true);

        const result = await verifier.verify(
          'claim-1',
          [
            {
              type: 'witness',
              data: 'test',
              createdAt: Date.now() - 10000,
              expiresAt: Date.now() - 5000,
            },
          ],
          { strategy: 'all', failOnExpired: true }
        );

        expect(result.verified).toBe(false);
        expect(result.proofResults[0].failureReason).toBe('Proof expired');
      });

      it('should create composite proofs', () => {
        const verifier = new utils.ClaimVerifier();

        const composite = verifier.createCompositeProof([
          { type: 'signature', data: 'sig1', createdAt: Date.now() },
          { type: 'hash', data: 'hash1', createdAt: Date.now() },
        ]);

        expect(composite.type).toBe('composite');
        expect(composite.algorithm).toBe('composite');
      });

      it('should get registered verifier types', () => {
        const verifier = new utils.ClaimVerifier();

        verifier.registerVerifier('signature', async () => true);
        verifier.registerVerifier('hash', async () => true);

        const types = verifier.getVerifierTypes();
        expect(types).toContain('signature');
        expect(types).toContain('hash');
      });
    });

    describe('Global claimVerifier', () => {
      it('should export global claimVerifier with default verifiers', () => {
        expect(utils.claimVerifier).toBeDefined();
        expect(utils.claimVerifier).toBeInstanceOf(utils.ClaimVerifier);

        const types = utils.claimVerifier.getVerifierTypes();
        expect(types).toContain('timestamp');
        expect(types).toContain('hash');
      });
    });
  });

  // ============================================================================
  // Schema Versioning Tests
  // ============================================================================

  describe('Schema Versioning', () => {
    describe('SchemaRegistry', () => {
      // Define test types
      interface V1Data {
        id: string;
        name: string;
      }

      interface V2Data {
        id: string;
        name: string;
        version: 2;
      }

      const isV1 = (data: unknown): data is V1Data => {
        const d = data as Record<string, unknown>;
        return (
          typeof d === 'object' &&
          d !== null &&
          typeof d.id === 'string' &&
          typeof d.name === 'string' &&
          !('version' in d)
        );
      };

      const isV2 = (data: unknown): data is V2Data => {
        const d = data as Record<string, unknown>;
        return (
          typeof d === 'object' &&
          d !== null &&
          typeof d.id === 'string' &&
          typeof d.name === 'string' &&
          d.version === 2
        );
      };

      it('should register schemas', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 1, minor: 0, patch: 0 },
          validate: isV1,
        });

        const schema = registry.getSchema('test');
        expect(schema).toBeDefined();
        expect(schema?.version).toEqual({ major: 1, minor: 0, patch: 0 });
      });

      it('should get latest schema by default', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 1, minor: 0, patch: 0 },
          validate: isV1,
        });

        registry.registerSchema({
          id: 'test',
          version: { major: 2, minor: 0, patch: 0 },
          validate: isV2,
        });

        const schema = registry.getSchema('test');
        expect(schema?.version.major).toBe(2);
      });

      it('should get specific version', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 1, minor: 0, patch: 0 },
          validate: isV1,
        });

        registry.registerSchema({
          id: 'test',
          version: { major: 2, minor: 0, patch: 0 },
          validate: isV2,
        });

        const schema = registry.getSchema('test', { major: 1, minor: 0, patch: 0 });
        expect(schema?.version.major).toBe(1);
      });

      it('should get all versions', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 1, minor: 0, patch: 0 },
          validate: isV1,
        });

        registry.registerSchema({
          id: 'test',
          version: { major: 2, minor: 0, patch: 0 },
          validate: isV2,
        });

        const versions = registry.getVersions('test');
        expect(versions.length).toBe(2);
      });

      it('should detect data version', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 1, minor: 0, patch: 0 },
          validate: isV1,
        });

        registry.registerSchema({
          id: 'test',
          version: { major: 2, minor: 0, patch: 0 },
          validate: isV2,
        });

        const v1Version = registry.detectVersion('test', { id: '1', name: 'Test' });
        const v2Version = registry.detectVersion('test', { id: '1', name: 'Test', version: 2 });

        expect(v1Version?.major).toBe(1);
        expect(v2Version?.major).toBe(2);
      });

      it('should migrate data between versions', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 1, minor: 0, patch: 0 },
          validate: isV1,
        });

        registry.registerSchema({
          id: 'test',
          version: { major: 2, minor: 0, patch: 0 },
          validate: isV2,
        });

        registry.registerMigration<V1Data, V2Data>(
          'test',
          { major: 1, minor: 0, patch: 0 },
          { major: 2, minor: 0, patch: 0 },
          (v1) => ({ ...v1, version: 2 })
        );

        const result = registry.migrate<V2Data>(
          { id: '1', name: 'Test' },
          'test',
          { major: 2, minor: 0, patch: 0 }
        );

        expect(result.success).toBe(true);
        expect(result.data?.version).toBe(2);
        expect(result.migrationPath).toEqual(['1.0.0', '2.0.0']);
      });

      it('should return success when already at target version', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 2, minor: 0, patch: 0 },
          validate: isV2,
        });

        const result = registry.migrate(
          { id: '1', name: 'Test', version: 2 },
          'test',
          { major: 2, minor: 0, patch: 0 }
        );

        expect(result.success).toBe(true);
        expect(result.migrationPath).toEqual([]);
      });

      it('should fail when no migration path exists', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 1, minor: 0, patch: 0 },
          validate: isV1,
        });

        registry.registerSchema({
          id: 'test',
          version: { major: 3, minor: 0, patch: 0 },
          // V3 requires 'uuid' field that v1 doesn't have
          validate: (data): data is { uuid: string } =>
            typeof data === 'object' && data !== null && 'uuid' in data,
        });

        const result = registry.migrate(
          { id: '1', name: 'Test' },
          'test',
          { major: 3, minor: 0, patch: 0 }
        );

        expect(result.success).toBe(false);
        expect(result.error).toContain('No migration path');
      });

      it('should get all schema IDs', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'schema1',
          version: { major: 1, minor: 0, patch: 0 },
          validate: () => true,
        });

        registry.registerSchema({
          id: 'schema2',
          version: { major: 1, minor: 0, patch: 0 },
          validate: () => true,
        });

        const ids = registry.getSchemaIds();
        expect(ids).toContain('schema1');
        expect(ids).toContain('schema2');
      });

      it('should clear all registrations', () => {
        const registry = new utils.SchemaRegistry();

        registry.registerSchema({
          id: 'test',
          version: { major: 1, minor: 0, patch: 0 },
          validate: () => true,
        });

        registry.clear();

        expect(registry.getSchemaIds().length).toBe(0);
      });
    });

    describe('Global schemaRegistry', () => {
      it('should export global schemaRegistry instance', () => {
        expect(utils.schemaRegistry).toBeDefined();
        expect(utils.schemaRegistry).toBeInstanceOf(utils.SchemaRegistry);
      });
    });
  });

  // ==========================================================================
  // CATEGORY 1: Real-time & Connectivity Tests
  // ==========================================================================
  describe('WebSocketManager', () => {
    it('should initialize with disconnected state', () => {
      const ws = new utils.WebSocketManager({ url: 'ws://localhost:8888' });
      expect(ws.getState()).toBe('disconnected');
      expect(ws.isConnected()).toBe(false);
    });

    it('should connect successfully', async () => {
      const ws = new utils.WebSocketManager({ url: 'ws://localhost:8888' });
      await ws.connect();
      expect(ws.getState()).toBe('connected');
      expect(ws.isConnected()).toBe(true);
    });

    it('should track message statistics', async () => {
      const ws = new utils.WebSocketManager<{ data: string }>({ url: 'ws://localhost:8888' });
      await ws.connect();

      ws.send('test', { data: 'hello' });
      const stats = ws.getStats();

      expect(stats.messagesSent).toBe(1);
      expect(stats.bytesSent).toBeGreaterThan(0);
    });

    it('should handle subscriptions', async () => {
      const ws = new utils.WebSocketManager<string>({ url: 'ws://localhost:8888' });
      const messages: utils.WebSocketMessage<string>[] = [];

      const unsubscribe = ws.subscribe('event', (msg) => messages.push(msg));

      ws.handleMessage({
        type: 'event',
        payload: 'test',
        timestamp: Date.now(),
        id: 'msg_1',
      });

      expect(messages.length).toBe(1);
      expect(messages[0].payload).toBe('test');

      unsubscribe();
    });

    it('should queue messages when disconnected', () => {
      const ws = new utils.WebSocketManager<string>({ url: 'ws://localhost:8888' });

      const sent = ws.send('test', 'queued message');
      expect(sent).toBe(false);
    });

    it('should disconnect properly', async () => {
      const ws = new utils.WebSocketManager({ url: 'ws://localhost:8888' });
      await ws.connect();
      ws.disconnect();

      expect(ws.getState()).toBe('disconnected');
      expect(ws.isConnected()).toBe(false);
    });
  });

  describe('ConnectionPool', () => {
    const createTestConnection = async (): Promise<utils.PooledConnection> => ({
      id: `conn_${Date.now()}_${Math.random()}`,
      isHealthy: () => true,
      close: async () => {},
    });

    it('should initialize with minimum connections', async () => {
      const pool = new utils.ConnectionPool({
        minConnections: 2,
        maxConnections: 5,
        idleTimeoutMs: 60000,
        acquireTimeoutMs: 5000,
        createConnection: createTestConnection,
      });

      await pool.initialize();
      const stats = pool.getStats();

      expect(stats.idleConnections).toBe(2);
      expect(stats.totalConnections).toBe(2);
    });

    it('should acquire and release connections', async () => {
      const pool = new utils.ConnectionPool({
        minConnections: 1,
        maxConnections: 5,
        idleTimeoutMs: 60000,
        acquireTimeoutMs: 5000,
        createConnection: createTestConnection,
      });

      await pool.initialize();
      const conn = await pool.acquire();

      expect(conn).toBeDefined();
      expect(conn.id).toBeDefined();

      pool.release(conn);
      const stats = pool.getStats();
      expect(stats.totalReleased).toBe(1);
    });

    it('should track statistics', async () => {
      const pool = new utils.ConnectionPool({
        minConnections: 1,
        maxConnections: 5,
        idleTimeoutMs: 60000,
        acquireTimeoutMs: 5000,
        createConnection: createTestConnection,
      });

      await pool.initialize();
      await pool.acquire();

      const stats = pool.getStats();
      expect(stats.totalAcquired).toBe(1);
      expect(stats.totalCreated).toBeGreaterThanOrEqual(1);
    });

    it('should close all connections', async () => {
      const pool = new utils.ConnectionPool({
        minConnections: 2,
        maxConnections: 5,
        idleTimeoutMs: 60000,
        acquireTimeoutMs: 5000,
        createConnection: createTestConnection,
      });

      await pool.initialize();
      await pool.close();

      const stats = pool.getStats();
      expect(stats.totalConnections).toBe(0);
    });
  });

  describe('OfflineQueue', () => {
    it('should enqueue operations', () => {
      const queue = new utils.OfflineQueue<{ action: string }>();

      const id = queue.enqueue('update', { action: 'test' });

      expect(id).toBeDefined();
      expect(queue.getQueue().length).toBe(1);
    });

    it('should process operations', async () => {
      const queue = new utils.OfflineQueue<string>();
      const processed: string[] = [];

      queue.enqueue('op1', 'data1');
      queue.enqueue('op2', 'data2');

      const result = await queue.process(async (op) => {
        processed.push(op.payload);
        return true;
      });

      expect(result.completed).toBe(2);
      expect(processed).toEqual(['data1', 'data2']);
    });

    it('should handle retries on failure', async () => {
      const queue = new utils.OfflineQueue<string>({ maxRetries: 2 });
      let attempts = 0;

      queue.enqueue('failing', 'data');

      await queue.process(async () => {
        attempts++;
        return false;
      });

      // Should be re-queued for retry
      expect(queue.getQueue().length).toBe(1);
    });

    it('should respect priority ordering', () => {
      const queue = new utils.OfflineQueue<string>();

      queue.enqueue('low', 'low-data', 1);
      queue.enqueue('high', 'high-data', 10);
      queue.enqueue('medium', 'medium-data', 5);

      const items = queue.getQueue();
      expect(items[0].priority).toBe(10);
      expect(items[1].priority).toBe(5);
      expect(items[2].priority).toBe(1);
    });

    it('should export and import queue', () => {
      const queue1 = new utils.OfflineQueue<string>();
      queue1.enqueue('test', 'data');

      const exported = queue1.export();

      const queue2 = new utils.OfflineQueue<string>();
      queue2.import(exported);

      expect(queue2.getQueue().length).toBe(1);
    });
  });

  describe('SignalHandler', () => {
    it('should register and handle signals', () => {
      const handler = new utils.SignalHandler<{ value: number }>();
      const received: utils.HolochainSignal<{ value: number }>[] = [];

      handler.on('test_signal', (signal) => received.push(signal));

      handler.handle({
        type: 'app',
        cellId: [new Uint8Array(), new Uint8Array()],
        zomeName: 'test',
        signalName: 'test_signal',
        payload: { value: 42 },
      });

      expect(received.length).toBe(1);
      expect(received[0].payload.value).toBe(42);
    });

    it('should handle global subscribers', () => {
      const handler = new utils.SignalHandler<string>();
      const received: string[] = [];

      handler.onAny((signal) => received.push(signal.signalName));

      handler.handle({
        type: 'app',
        cellId: [new Uint8Array(), new Uint8Array()],
        zomeName: 'test',
        signalName: 'signal1',
        payload: 'data',
      });

      handler.handle({
        type: 'app',
        cellId: [new Uint8Array(), new Uint8Array()],
        zomeName: 'test',
        signalName: 'signal2',
        payload: 'data',
      });

      expect(received).toEqual(['signal1', 'signal2']);
    });

    it('should buffer signals', () => {
      const handler = new utils.SignalHandler<string>({ bufferSize: 10 });

      for (let i = 0; i < 5; i++) {
        handler.handle({
          type: 'app',
          cellId: [new Uint8Array(), new Uint8Array()],
          zomeName: 'test',
          signalName: `signal_${i}`,
          payload: `data_${i}`,
        });
      }

      expect(handler.getBuffer().length).toBe(5);
    });
  });

  // ==========================================================================
  // CATEGORY 2: Developer Experience Tests
  // ==========================================================================
  describe('StructuredLogger', () => {
    it('should create logger with default config', () => {
      const logger = new utils.StructuredLogger();
      expect(logger).toBeDefined();
    });

    it('should respect log levels', () => {
      const logs: utils.LogEntry[] = [];
      const logger = new utils.StructuredLogger({
        level: 'warn',
        output: (entry) => logs.push(entry),
      });

      logger.debug('debug message');
      logger.info('info message');
      logger.warn('warn message');
      logger.error('error message');

      expect(logs.length).toBe(2);
      expect(logs[0].level).toBe('warn');
      expect(logs[1].level).toBe('error');
    });

    it('should create child loggers with context', () => {
      const logs: utils.LogEntry[] = [];
      const logger = new utils.StructuredLogger({
        level: 'debug',
        output: (entry) => logs.push(entry),
      });

      const childLogger = logger.child({ requestId: '123' });
      childLogger.info('test message');

      expect(logs[0].context?.requestId).toBe('123');
    });

    it('should support JSON format', () => {
      const originalLog = console.log;
      const logged: string[] = [];
      console.log = (msg: string) => logged.push(msg);

      const logger = new utils.StructuredLogger({
        level: 'info',
        format: 'json',
      });

      logger.info('test message', { key: 'value' });

      console.log = originalLog;

      expect(logged.length).toBe(1);
      const parsed = JSON.parse(logged[0]);
      expect(parsed.message).toBe('test message');
      expect(parsed.key).toBe('value');
    });
  });

  describe('DebugInspector', () => {
    it('should inspect objects', () => {
      const inspector = new utils.DebugInspector();
      const output = inspector.inspect({ a: 1, b: 'test' });

      expect(output).toContain('a');
      expect(output).toContain('1');
      expect(output).toContain('test');
    });

    it('should take and compare snapshots', () => {
      const inspector = new utils.DebugInspector();

      inspector.snapshot('test', { value: 1 });

      const { changed, diff } = inspector.compareToSnapshot('test', { value: 2 });

      expect(changed).toBe(true);
      expect(diff).toContain('1');
      expect(diff).toContain('2');
    });

    it('should diff two values', () => {
      const inspector = new utils.DebugInspector();

      const { changed, diff } = inspector.diff(
        { a: 1 },
        { a: 2 }
      );

      expect(changed).toBe(true);
      expect(diff).toBeDefined();
    });

    it('should trace function execution', () => {
      const inspector = new utils.DebugInspector();

      const result = inspector.trace(() => 42, 'test');

      expect(result).toBe(42);
    });
  });

  describe('Branded Types', () => {
    it('should create branded agent ID', () => {
      const agentId = utils.brandAgentId('agent123');
      expect(agentId).toBe('agent123');
    });

    it('should reject invalid agent ID', () => {
      expect(() => utils.brandAgentId('')).toThrow();
    });

    it('should create branded hApp ID', () => {
      const happId = utils.brandHappId('myapp');
      expect(happId).toBe('myapp');
    });

    it('should create branded trust score', () => {
      const score = utils.brandTrustScore(0.95);
      expect(score).toBe(0.95);
    });

    it('should reject out-of-range trust score', () => {
      expect(() => utils.brandTrustScore(1.5)).toThrow();
      expect(() => utils.brandTrustScore(-0.1)).toThrow();
    });

    it('should create branded timestamp', () => {
      const ts = utils.brandTimestamp(Date.now());
      expect(ts).toBeGreaterThan(0);
    });
  });

  describe('sdkHelpers', () => {
    it('should generate unique IDs', () => {
      const id1 = utils.sdkHelpers.generateId('test');
      const id2 = utils.sdkHelpers.generateId('test');

      expect(id1).not.toBe(id2);
      expect(id1.startsWith('test_')).toBe(true);
    });

    it('should deep clone objects', () => {
      const original = { a: { b: { c: 1 } } };
      const cloned = utils.sdkHelpers.deepClone(original);

      cloned.a.b.c = 2;

      expect(original.a.b.c).toBe(1);
    });

    it('should deep merge objects', () => {
      const target = { a: 1, b: { c: 2 } };
      const source = { b: { d: 3 }, e: 4 };

      const result = utils.sdkHelpers.deepMerge(target, source);

      expect(result.a).toBe(1);
      expect(result.b.c).toBe(2);
      expect(result.b.d).toBe(3);
      expect(result.e).toBe(4);
    });

    it('should safely parse JSON', () => {
      const valid = utils.sdkHelpers.safeJsonParse('{"a":1}', { a: 0 });
      expect(valid.a).toBe(1);

      const invalid = utils.sdkHelpers.safeJsonParse('invalid', { a: 0 });
      expect(invalid.a).toBe(0);
    });

    it('should format bytes', () => {
      expect(utils.sdkHelpers.formatBytes(500)).toBe('500.00 B');
      expect(utils.sdkHelpers.formatBytes(1500)).toBe('1.46 KB');
      expect(utils.sdkHelpers.formatBytes(1500000)).toBe('1.43 MB');
    });

    it('should format duration', () => {
      expect(utils.sdkHelpers.formatDuration(500)).toBe('500ms');
      expect(utils.sdkHelpers.formatDuration(5000)).toBe('5.00s');
      expect(utils.sdkHelpers.formatDuration(120000)).toBe('2.00m');
    });

    it('should chunk arrays', () => {
      const chunks = utils.sdkHelpers.chunk([1, 2, 3, 4, 5], 2);
      expect(chunks).toEqual([[1, 2], [3, 4], [5]]);
    });

    it('should memoize functions', () => {
      let callCount = 0;
      const expensive = utils.sdkHelpers.memoize((n: number) => {
        callCount++;
        return n * 2;
      });

      expect(expensive(5)).toBe(10);
      expect(expensive(5)).toBe(10);
      expect(callCount).toBe(1);
    });
  });

  // ==========================================================================
  // CATEGORY 3: Production Readiness Tests
  // ==========================================================================
  describe('MetricsCollector', () => {
    it('should define and increment counters', () => {
      const metrics = new utils.MetricsCollector();

      metrics.defineCounter('test_counter', 'Test counter');
      metrics.incrementCounter('test_counter');
      metrics.incrementCounter('test_counter', {}, 2);

      expect(metrics.getCounter('test_counter')).toBe(3);
    });

    it('should define and set gauges', () => {
      const metrics = new utils.MetricsCollector();

      metrics.defineGauge('test_gauge', 'Test gauge');
      metrics.setGauge('test_gauge', 42);

      expect(metrics.getGauge('test_gauge')).toBe(42);
    });

    it('should define and observe histograms', () => {
      const metrics = new utils.MetricsCollector();

      metrics.defineHistogram('test_histogram', 'Test histogram', [10, 50, 100]);
      metrics.observeHistogram('test_histogram', 25);
      metrics.observeHistogram('test_histogram', 75);

      const exported = metrics.exportJson();
      expect(exported['test_histogram']).toBeDefined();
    });

    it('should export Prometheus format', () => {
      const metrics = new utils.MetricsCollector();

      metrics.defineCounter('ops_total', 'Total operations');
      metrics.incrementCounter('ops_total', { type: 'read' });

      const prometheus = metrics.exportPrometheus();

      expect(prometheus).toContain('# HELP ops_total Total operations');
      expect(prometheus).toContain('# TYPE ops_total counter');
    });

    it('should support labels', () => {
      const metrics = new utils.MetricsCollector();

      metrics.defineCounter('labeled_counter', 'Labeled counter');
      metrics.incrementCounter('labeled_counter', { method: 'GET' });
      metrics.incrementCounter('labeled_counter', { method: 'POST' });

      expect(metrics.getCounter('labeled_counter', { method: 'GET' })).toBe(1);
      expect(metrics.getCounter('labeled_counter', { method: 'POST' })).toBe(1);
    });
  });

  describe('HealthChecker', () => {
    it('should register and run health checks', async () => {
      const health = new utils.HealthChecker('1.0.0');

      health.register('test', async () => ({ status: 'healthy' }));

      const report = await health.check();

      expect(report.status).toBe('healthy');
      expect(report.checks.length).toBe(1);
      expect(report.version).toBe('1.0.0');
    });

    it('should report degraded status', async () => {
      const health = new utils.HealthChecker();

      health.register('healthy', async () => ({ status: 'healthy' }));
      health.register('degraded', async () => ({ status: 'degraded' }));

      const report = await health.check();

      expect(report.status).toBe('degraded');
    });

    it('should report unhealthy status', async () => {
      const health = new utils.HealthChecker();

      health.register('healthy', async () => ({ status: 'healthy' }));
      health.register('unhealthy', async () => ({ status: 'unhealthy', message: 'Failed' }));

      const report = await health.check();

      expect(report.status).toBe('unhealthy');
    });

    it('should handle check failures', async () => {
      const health = new utils.HealthChecker();

      health.register('failing', async () => {
        throw new Error('Check failed');
      });

      const report = await health.check();

      expect(report.status).toBe('unhealthy');
      expect(report.checks[0].message).toContain('Check failed');
    });

    it('should track uptime', async () => {
      const health = new utils.HealthChecker();

      await new Promise((r) => setTimeout(r, 10));

      const report = await health.check();

      expect(report.uptime).toBeGreaterThan(0);
    });
  });

  describe('CircuitBreaker', () => {
    it('should start in closed state', () => {
      const breaker = new utils.CircuitBreaker();
      expect(breaker.getState()).toBe('closed');
    });

    it('should execute successful calls', async () => {
      const breaker = new utils.CircuitBreaker();

      const result = await breaker.call(async () => 42);

      expect(result).toBe(42);
      expect(breaker.getStats().successes).toBe(1);
    });

    it('should track failures', async () => {
      const breaker = new utils.CircuitBreaker({ failureThreshold: 3 });

      for (let i = 0; i < 2; i++) {
        try {
          await breaker.call(async () => {
            throw new Error('fail');
          });
        } catch {
          // expected
        }
      }

      expect(breaker.getStats().failures).toBe(2);
      expect(breaker.getState()).toBe('closed');
    });

    it('should open after threshold failures', async () => {
      const breaker = new utils.CircuitBreaker({ failureThreshold: 2 });

      for (let i = 0; i < 2; i++) {
        try {
          await breaker.call(async () => {
            throw new Error('fail');
          });
        } catch {
          // expected
        }
      }

      expect(breaker.getState()).toBe('open');
    });

    it('should reject calls when open', async () => {
      const breaker = new utils.CircuitBreaker({ failureThreshold: 1, resetTimeoutMs: 10000 });

      try {
        await breaker.call(async () => {
          throw new Error('fail');
        });
      } catch {
        // expected
      }

      await expect(breaker.call(async () => 42)).rejects.toThrow('Circuit is open');
    });

    it('should reset to closed state', async () => {
      const breaker = new utils.CircuitBreaker({ failureThreshold: 1 });

      try {
        await breaker.call(async () => {
          throw new Error('fail');
        });
      } catch {
        // expected
      }

      breaker.reset();

      expect(breaker.getState()).toBe('closed');
      expect(breaker.getStats().consecutiveFailures).toBe(0);
    });
  });

  // ==========================================================================
  // CATEGORY 4: Testing & Quality Tests
  // ==========================================================================
  describe('TestFixtureFactory', () => {
    it('should create MATL fixtures', () => {
      const factory = new utils.TestFixtureFactory();

      const fixture = factory.createMatlFixture();

      expect(fixture.agentId).toBeDefined();
      expect(fixture.pogq.quality).toBeGreaterThanOrEqual(0);
      expect(fixture.pogq.quality).toBeLessThanOrEqual(1);
    });

    it('should generate multiple fixtures', () => {
      const factory = new utils.TestFixtureFactory();

      const fixtures = factory.generateMatlFixtures(5);

      expect(fixtures.length).toBe(5);
    });

    it('should create Epistemic fixtures', () => {
      const factory = new utils.TestFixtureFactory();

      const fixture = factory.createEpistemicFixture();

      expect(fixture.claimText).toBeDefined();
      expect(fixture.expectedCode).toMatch(/E[0-3]-N[0-2]-M[0-2]/);
    });

    it('should create FL fixtures', () => {
      const factory = new utils.TestFixtureFactory();

      const fixture = factory.createFlFixture(3, 5);

      expect(fixture.participants.length).toBe(3);
      expect(fixture.participants[0].gradients.length).toBe(5);
    });

    it('should create Byzantine attack fixtures', () => {
      const factory = new utils.TestFixtureFactory();

      const fixture = factory.createByzantineFixture(5, 2);

      expect(fixture.participants.length).toBe(7);
    });

    it('should produce reproducible results with same seed', () => {
      const factory1 = new utils.TestFixtureFactory(12345);
      const factory2 = new utils.TestFixtureFactory(12345);

      const fixture1 = factory1.createMatlFixture();
      const fixture2 = factory2.createMatlFixture();

      expect(fixture1.pogq.quality).toBe(fixture2.pogq.quality);
    });
  });

  describe('mockFactory', () => {
    it('should create mock Holochain client', async () => {
      const client = utils.mockFactory.createMockClient();

      client.setResponse('test_zome', 'test_fn', { result: 'success' });

      const result = await client.callZome({
        zome_name: 'test_zome',
        fn_name: 'test_fn',
      });

      expect(result).toEqual({ result: 'success' });
    });

    it('should create mock WebSocket', () => {
      const ws = utils.mockFactory.createMockWebSocket();

      ws.connect();
      expect(ws.isConnected()).toBe(true);

      ws.disconnect();
      expect(ws.isConnected()).toBe(false);
    });

    it('should create mock metrics', () => {
      const metrics = utils.mockFactory.createMockMetrics();

      metrics.increment('counter');
      metrics.set('gauge', 42);

      expect(metrics.get('counter')).toBe(1);
      expect(metrics.get('gauge')).toBe(42);
    });

    it('should create delayed promises', async () => {
      const start = Date.now();
      const result = await utils.mockFactory.createDelayedPromise('value', 50);
      const elapsed = Date.now() - start;

      expect(result).toBe('value');
      expect(elapsed).toBeGreaterThanOrEqual(45);
    });
  });

  describe('propertyGenerators', () => {
    it('should generate trust scores in valid range', () => {
      for (let i = 0; i < 100; i++) {
        const score = utils.propertyGenerators.trustScore();
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(1);
      }
    });

    it('should generate arrays of trust scores', () => {
      const scores = utils.propertyGenerators.trustScores(5);
      expect(scores.length).toBe(5);
    });

    it('should generate gradients in range', () => {
      const gradients = utils.propertyGenerators.gradients(10, 2);
      expect(gradients.length).toBe(10);
      gradients.forEach((g) => {
        expect(g).toBeGreaterThanOrEqual(-2);
        expect(g).toBeLessThanOrEqual(2);
      });
    });

    it('should generate agent IDs', () => {
      const id = utils.propertyGenerators.agentId();
      expect(id.startsWith('agent_')).toBe(true);
    });

    it('should generate past timestamps', () => {
      const ts = utils.propertyGenerators.pastTimestamp();
      expect(ts).toBeLessThan(Date.now());
    });

    it('should generate future timestamps', () => {
      const ts = utils.propertyGenerators.futureTimestamp();
      expect(ts).toBeGreaterThan(Date.now());
    });
  });

  // ==========================================================================
  // CATEGORY 5: Documentation Tests
  // ==========================================================================
  describe('examples', () => {
    it('should have basic trust evaluation example', () => {
      expect(utils.examples.basicTrustEvaluation).toBeDefined();
      expect(utils.examples.basicTrustEvaluation).toContain('createPoGQ');
    });

    it('should have epistemic claim creation example', () => {
      expect(utils.examples.epistemicClaimCreation).toBeDefined();
      expect(utils.examples.epistemicClaimCreation).toContain('claim');
    });

    it('should have federated learning example', () => {
      expect(utils.examples.federatedLearningRound).toBeDefined();
      expect(utils.examples.federatedLearningRound).toContain('FLCoordinator');
    });

    it('should have cross-hApp bridge example', () => {
      expect(utils.examples.crossHappBridge).toBeDefined();
      expect(utils.examples.crossHappBridge).toContain('LocalBridge');
    });

    it('should have event pipeline example', () => {
      expect(utils.examples.eventPipeline).toBeDefined();
      expect(utils.examples.eventPipeline).toContain('pipeline');
    });

    it('should have distributed tracing example', () => {
      expect(utils.examples.distributedTracing).toBeDefined();
      expect(utils.examples.distributedTracing).toContain('traced');
    });
  });

  // ==========================================================================
  // CATEGORY 6: Advanced Features Tests
  // ==========================================================================
  describe('PluginManager', () => {
    it('should register plugins', async () => {
      const manager = new utils.PluginManager();

      await manager.register({
        id: 'test-plugin',
        name: 'Test Plugin',
        version: '1.0.0',
      });

      expect(manager.hasPlugin('test-plugin')).toBe(true);
    });

    it('should call plugin hooks', async () => {
      const manager = new utils.PluginManager();
      const calls: string[] = [];

      await manager.register({
        id: 'hook-plugin',
        name: 'Hook Plugin',
        version: '1.0.0',
        beforeOperation: async (op) => {
          calls.push(`before:${op}`);
          return {};
        },
        afterOperation: async (op, result) => {
          calls.push(`after:${op}`);
          return result;
        },
      });

      await manager.executeWithHooks('test', {}, async () => 'result');

      expect(calls).toEqual(['before:test', 'after:test']);
    });

    it('should check dependencies', async () => {
      const manager = new utils.PluginManager();

      await expect(manager.register({
        id: 'dependent',
        name: 'Dependent',
        version: '1.0.0',
        dependencies: ['missing'],
      })).rejects.toThrow('Missing dependency');
    });

    it('should unregister plugins', async () => {
      const manager = new utils.PluginManager();

      await manager.register({
        id: 'removable',
        name: 'Removable',
        version: '1.0.0',
      });

      await manager.unregister('removable');

      expect(manager.hasPlugin('removable')).toBe(false);
    });
  });

  describe('StateSync', () => {
    it('should set and get values', () => {
      const sync = new utils.StateSync<{ name: string }>();

      sync.set('user', { name: 'Alice' });

      expect(sync.get('user')).toEqual({ name: 'Alice' });
    });

    it('should track dirty entries', () => {
      const sync = new utils.StateSync<string>();

      sync.set('key1', 'value1');
      sync.set('key2', 'value2');

      const dirty = sync.getDirtyEntries();

      expect(dirty.length).toBe(2);
      expect(dirty[0].dirty).toBe(true);
    });

    it('should sync with remote', async () => {
      const sync = new utils.StateSync<string>();

      sync.set('local', 'value');

      const result = await sync.sync(async (changes) => {
        return changes.map((c) => ({ ...c, dirty: false }));
      });

      expect(result.synced).toBe(1);
    });

    it('should export and import state', () => {
      const sync1 = new utils.StateSync<number>();
      sync1.set('count', 42);

      const exported = sync1.export();

      const sync2 = new utils.StateSync<number>();
      sync2.import(exported);

      expect(sync2.get('count')).toBe(42);
    });
  });

  describe('EncryptedStorage', () => {
    it('should initialize with password', async () => {
      const storage = new utils.EncryptedStorage();

      await storage.init('password');

      expect(storage.isInitialized()).toBe(true);
    });

    it('should store and retrieve encrypted values', async () => {
      const storage = new utils.EncryptedStorage();
      await storage.init('password');

      await storage.set('secret', 'my-secret-value');
      const value = await storage.get('secret');

      expect(value).toBe('my-secret-value');
    });

    it('should throw when not initialized', async () => {
      const storage = new utils.EncryptedStorage();

      await expect(storage.set('key', 'value')).rejects.toThrow('Storage not initialized');
    });

    it('should lock storage', async () => {
      const storage = new utils.EncryptedStorage();
      await storage.init('password');

      storage.lock();

      expect(storage.isInitialized()).toBe(false);
    });
  });

  describe('MultiConductor', () => {
    it('should add and remove conductors', () => {
      const multi = new utils.MultiConductor();

      multi.addConductor('primary', 'ws://localhost:8888', true);
      multi.addConductor('secondary', 'ws://localhost:8889');

      expect(multi.getHealthyConductors().length).toBe(2);

      multi.removeConductor('secondary');

      expect(multi.getHealthyConductors().length).toBe(1);
    });

    it('should execute with failover', async () => {
      const multi = new utils.MultiConductor();

      multi.addConductor('test', 'ws://localhost:8888');

      const result = await multi.execute(async () => 'success');

      expect(result).toBe('success');
    });

    it('should throw when no healthy conductors', async () => {
      const multi = new utils.MultiConductor();

      await expect(multi.execute(async () => 'test')).rejects.toThrow('No healthy conductors');
    });

    it('should track conductor status', async () => {
      const multi = new utils.MultiConductor();

      multi.addConductor('test', 'ws://localhost:8888');

      const status = multi.getStatus();

      expect(status.length).toBe(1);
      expect(status[0].healthy).toBe(true);
    });
  });

  describe('Global instances', () => {
    it('should export global metrics instance', () => {
      expect(utils.metrics).toBeDefined();
      expect(utils.metrics).toBeInstanceOf(utils.MetricsCollector);
    });
  });
});
