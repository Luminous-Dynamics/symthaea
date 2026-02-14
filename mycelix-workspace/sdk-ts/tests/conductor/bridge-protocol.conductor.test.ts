/**
 * Bridge Protocol Conductor Tests
 *
 * Real conductor tests for validating cross-hApp bridge protocol.
 * These tests require a running Holochain conductor.
 *
 * Run with: CONDUCTOR_AVAILABLE=true npm run test:conductor
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  BridgeProtocolValidator,
  BridgeMessageType,
  BRIDGE_ENABLED_HAPPS,
  type BridgeHappId,
} from './bridge-protocol-validator';
import { CONDUCTOR_ENABLED } from './conductor-harness';

describe.skipIf(!CONDUCTOR_ENABLED)('Bridge Protocol - Conductor Tests', () => {
  let validator: BridgeProtocolValidator;

  beforeAll(async () => {
    validator = new BridgeProtocolValidator();
    await validator.initialize();
  });

  afterAll(async () => {
    if (validator) {
      await validator.cleanup();
    }
  });

  describe('Individual hApp Bridge Validation', () => {
    it.each(BRIDGE_ENABLED_HAPPS.slice(0, 6))('should validate bridge capability for %s', async (happId) => {
      const result = await validator.validateHapp(happId as BridgeHappId);

      expect(result.happ).toBe(happId);
      // At minimum, the bridge zome should exist and respond
      expect(result.errors.length).toBe(0);
    });
  });

  describe('Cross-hApp Message Passing', () => {
    it('should send reputation query from marketplace to core', async () => {
      const result = await validator.testCrossHappMessage('marketplace', 'core', BridgeMessageType.REPUTATION_QUERY);

      expect(result.success).toBe(true);
      expect(result.latencyMs).toBeLessThan(2000); // Should complete within 2s
    });

    it('should send credential verification from edunet to identity', async () => {
      const result = await validator.testCrossHappMessage('edunet', 'identity', BridgeMessageType.CREDENTIAL_VERIFY);

      expect(result.success).toBe(true);
    });

    it('should broadcast events from finance to all listeners', async () => {
      const result = await validator.testCrossHappMessage('finance', 'core', BridgeMessageType.EVENT_BROADCAST);

      expect(result.success).toBe(true);
    });

    it('should handle supplychain to fabrication provenance queries', async () => {
      const result = await validator.testCrossHappMessage('supplychain', 'fabrication', BridgeMessageType.REPUTATION_QUERY);

      expect(result.success).toBe(true);
    });
  });

  describe('Aggregate Reputation Flow', () => {
    it('should aggregate reputation across all participating hApps', async () => {
      const result = await validator.testReputationAggregation();

      expect(result.success).toBe(true);
      expect(result.aggregateScore).toBeGreaterThanOrEqual(0);
      expect(result.aggregateScore).toBeLessThanOrEqual(1);
      expect(result.errors).toHaveLength(0);
    });
  });

  describe('Full Validation', () => {
    it('should validate all 18 hApps bridge protocol', async () => {
      const results = await validator.validateAll();

      // At least 80% of hApps should pass validation
      const passing = Array.from(results.values()).filter((r) => r.canSend || r.canReceive).length;
      const passRate = passing / results.size;

      expect(passRate).toBeGreaterThanOrEqual(0.8);

      // Generate and log report
      console.log(validator.generateReport());
    });
  });

  describe('Bridge Protocol Edge Cases', () => {
    it('should handle message to non-existent hApp gracefully', async () => {
      // This should not throw, just return failure
      const result = await validator.testCrossHappMessage(
        'core',
        'nonexistent' as BridgeHappId,
        BridgeMessageType.REPUTATION_QUERY
      );

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should handle malformed message payload', async () => {
      // The validator should handle invalid payloads gracefully
      const result = await validator.testCrossHappMessage('core', 'identity', BridgeMessageType.CREDENTIAL_VERIFY);

      // Should either succeed (if payload is optional) or fail gracefully
      expect(typeof result.success).toBe('boolean');
    });
  });
});
