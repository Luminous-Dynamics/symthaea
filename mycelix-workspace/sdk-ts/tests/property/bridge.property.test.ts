/**
 * Property-Based Tests for Bridge Module
 *
 * Uses fast-check to verify bridge protocol properties:
 * - Message creation correctness
 * - Aggregate reputation bounds
 * - Message type consistency
 */

import { describe, it, expect, beforeEach } from 'vitest';
import fc from 'fast-check';
import {
  BridgeMessageType,
  calculateAggregateReputation,
  createReputationQuery,
  createCrossHappReputation,
  LocalBridge,
  BridgeRouter,
  type HappReputationScore,
} from '../../src/bridge/index.js';

describe('Bridge Property-Based Tests', () => {
  // Generate valid happ ID (non-empty)
  const happIdArb = fc
    .array(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789'.split('')), {
      minLength: 4,
      maxLength: 32,
    })
    .map((chars) => `happ-${chars.join('')}`);

  // Generate valid agent ID (non-empty hex string)
  const agentIdArb = fc
    .array(fc.constantFrom(...'0123456789abcdef'.split('')), {
      minLength: 32,
      maxLength: 64,
    })
    .map((chars) => chars.join(''));

  // Generate score [0, 1] - use Math.fround for 32-bit float compatibility
  const scoreArb = fc.float({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true });

  // Generate weight (non-negative) - must use 32-bit float values
  const weightArb = fc.float({ min: Math.fround(0.001), max: Math.fround(1000), noNaN: true, noDefaultInfinity: true });

  // Generate HappReputationScore with correct structure
  const happReputationScoreArb = fc.record({
    happ: happIdArb,
    score: scoreArb,
    weight: weightArb,
    lastUpdate: fc.nat({ max: Date.now() }),
  }) as fc.Arbitrary<HappReputationScore>;

  describe('Aggregate Reputation Properties', () => {
    it('aggregate reputation should be bounded [0, 1]', () => {
      fc.assert(
        fc.property(
          fc.array(happReputationScoreArb, { minLength: 1, maxLength: 20 }),
          (scores) => {
            const aggregate = calculateAggregateReputation(scores);
            expect(aggregate).toBeGreaterThanOrEqual(0);
            expect(aggregate).toBeLessThanOrEqual(1);
          }
        ),
        { numRuns: 100 }
      );
    });

    it('aggregate of single score should return that score', () => {
      fc.assert(
        fc.property(happReputationScoreArb, (score) => {
          const aggregate = calculateAggregateReputation([score]);
          expect(aggregate).toBeCloseTo(score.score, 10);
        }),
        { numRuns: 100 }
      );
    });

    it('aggregate of identical scores should return that score', () => {
      fc.assert(
        fc.property(
          scoreArb,
          fc.integer({ min: 2, max: 10 }),
          (scoreValue, count) => {
            const scores: HappReputationScore[] = [];
            for (let i = 0; i < count; i++) {
              scores.push({
                happ: `happ-${i}`,
                score: scoreValue,
                weight: 1,
                lastUpdate: Date.now(),
              });
            }
            const aggregate = calculateAggregateReputation(scores);
            expect(aggregate).toBeCloseTo(scoreValue, 10);
          }
        ),
        { numRuns: 100 }
      );
    });

    it('empty scores should return 0.5 (neutral)', () => {
      const aggregate = calculateAggregateReputation([]);
      expect(aggregate).toBe(0.5);
    });

    it('aggregate should be influenced by weight', () => {
      // High weight score with value 1 vs low weight score with value 0
      const highWeight: HappReputationScore = {
        happ: 'happ-high',
        score: 1,
        weight: 1000,
        lastUpdate: Date.now(),
      };
      const lowWeight: HappReputationScore = {
        happ: 'happ-low',
        score: 0,
        weight: 1,
        lastUpdate: Date.now(),
      };

      const aggregate = calculateAggregateReputation([highWeight, lowWeight]);
      // Should be closer to 1 (high weight)
      expect(aggregate).toBeGreaterThan(0.5);
    });

    it('weighted average should be correct', () => {
      const scores: HappReputationScore[] = [
        { happ: 'happ1', score: 1, weight: 2, lastUpdate: Date.now() },
        { happ: 'happ2', score: 0, weight: 2, lastUpdate: Date.now() },
      ];
      // Expected: (1*2 + 0*2) / (2+2) = 0.5
      const aggregate = calculateAggregateReputation(scores);
      expect(aggregate).toBeCloseTo(0.5, 10);
    });
  });

  describe('Message Creation Properties', () => {
    it('createReputationQuery should have correct type and fields', () => {
      fc.assert(
        fc.property(happIdArb, agentIdArb, (sourceHapp, agent) => {
          const msg = createReputationQuery(sourceHapp, agent);

          expect(msg.type).toBe(BridgeMessageType.ReputationQuery);
          expect(msg.sourceHapp).toBe(sourceHapp);
          expect(msg.agent).toBe(agent);
          expect(msg.timestamp).toBeLessThanOrEqual(Date.now());
        }),
        { numRuns: 100 }
      );
    });

    it('createCrossHappReputation should have correct structure', () => {
      fc.assert(
        fc.property(
          happIdArb,
          agentIdArb,
          fc.array(happReputationScoreArb, { minLength: 1, maxLength: 10 }),
          (sourceHapp, agent, scores) => {
            const msg = createCrossHappReputation(sourceHapp, agent, scores);

            expect(msg.type).toBe(BridgeMessageType.CrossHappReputation);
            expect(msg.sourceHapp).toBe(sourceHapp);
            expect(msg.agent).toBe(agent);
            expect(msg.scores).toEqual(scores);
            expect(msg.aggregate).toBeGreaterThanOrEqual(0);
            expect(msg.aggregate).toBeLessThanOrEqual(1);
          }
        ),
        { numRuns: 50 }
      );
    });

    it('createReputationQuery should throw on empty sourceHapp', () => {
      expect(() => createReputationQuery('', 'agent123')).toThrow();
    });

    it('createReputationQuery should throw on empty agent', () => {
      expect(() => createReputationQuery('happ-test', '')).toThrow();
    });
  });

  describe('Message Timestamp Properties', () => {
    it('timestamps should be valid', () => {
      fc.assert(
        fc.property(happIdArb, agentIdArb, (happ, agent) => {
          const before = Date.now();
          const msg = createReputationQuery(happ, agent);
          const after = Date.now();

          expect(msg.timestamp).toBeGreaterThanOrEqual(before);
          expect(msg.timestamp).toBeLessThanOrEqual(after);
        }),
        { numRuns: 100 }
      );
    });
  });

  describe('LocalBridge Properties', () => {
    let bridge: LocalBridge;

    beforeEach(() => {
      bridge = new LocalBridge();
    });

    it('registered happs should be discoverable', () => {
      fc.assert(
        fc.property(happIdArb, (happId) => {
          bridge = new LocalBridge();
          bridge.registerHapp(happId);

          const registered = bridge.getRegisteredHapps();
          expect(registered).toContain(happId);
        }),
        { numRuns: 50 }
      );
    });

    it('registered happs should be queryable via isRegistered', () => {
      fc.assert(
        fc.property(happIdArb, (happId) => {
          bridge = new LocalBridge();
          // Not registered yet
          expect(bridge.isRegistered(happId)).toBe(false);

          bridge.registerHapp(happId);

          // Now registered
          expect(bridge.isRegistered(happId)).toBe(true);
        }),
        { numRuns: 50 }
      );
    });

    it('unregistered happs should not appear in isRegistered', () => {
      fc.assert(
        fc.property(happIdArb, (happId) => {
          bridge = new LocalBridge();
          expect(bridge.isRegistered(happId)).toBe(false);
        }),
        { numRuns: 30 }
      );
    });

    it('multiple happs can be registered', () => {
      fc.assert(
        fc.property(
          fc.array(happIdArb, { minLength: 1, maxLength: 10 }),
          (happIds) => {
            bridge = new LocalBridge();
            const uniqueIds = [...new Set(happIds)];

            for (const happId of uniqueIds) {
              bridge.registerHapp(happId, []);
            }

            const registered = bridge.getRegisteredHapps();
            for (const happId of uniqueIds) {
              expect(registered).toContain(happId);
            }
          }
        ),
        { numRuns: 30 }
      );
    });
  });

  describe('BridgeRouter Properties', () => {
    let router: BridgeRouter;

    beforeEach(() => {
      router = new BridgeRouter({});
    });

    it('should route messages to correct handlers', async () => {
      const receivedMessages: any[] = [];

      router = new BridgeRouter({
        onReputationQuery: (msg) => {
          receivedMessages.push(msg);
        },
      });

      const msg = createReputationQuery('test-happ', 'agent123');
      await router.route(msg);

      expect(receivedMessages.length).toBe(1);
      expect(receivedMessages[0].type).toBe(BridgeMessageType.ReputationQuery);
    });

    it('middleware should be applied in order', async () => {
      const order: number[] = [];

      router = new BridgeRouter({});
      router.use(async (msg, next) => {
        order.push(1);
        await next();
        order.push(4);
      });
      router.use(async (msg, next) => {
        order.push(2);
        await next();
        order.push(3);
      });

      const msg = createReputationQuery('test-happ', 'agent123');
      await router.route(msg);

      expect(order).toEqual([1, 2, 3, 4]);
    });
  });

  describe('Cross-Happ Reputation Consistency', () => {
    it('aggregate in message should match calculated aggregate', () => {
      fc.assert(
        fc.property(
          happIdArb,
          agentIdArb,
          fc.array(happReputationScoreArb, { minLength: 1, maxLength: 10 }),
          (happ, agent, scores) => {
            const msg = createCrossHappReputation(happ, agent, scores);
            const calculated = calculateAggregateReputation(scores);

            expect(msg.aggregate).toBeCloseTo(calculated, 10);
          }
        ),
        { numRuns: 100 }
      );
    });
  });

  describe('Edge Cases', () => {
    it('should handle registering same happ twice', () => {
      const bridge = new LocalBridge();
      bridge.registerHapp('test-happ');
      bridge.registerHapp('test-happ'); // Should not throw
      expect(bridge.isRegistered('test-happ')).toBe(true);
    });

    it('should handle very long agent IDs', () => {
      const longAgent = 'a'.repeat(1000);
      const msg = createReputationQuery('test-happ', longAgent);
      expect(msg.agent).toBe(longAgent);
    });

    it('should handle special characters in happ ID', () => {
      const specialHapp = 'happ-test_123-abc';
      const msg = createReputationQuery(specialHapp, 'agent123');
      expect(msg.sourceHapp).toBe(specialHapp);
    });

    it('should handle zero weight in reputation (returns default 0.5)', () => {
      const scores: HappReputationScore[] = [
        {
          happ: 'happ-1',
          score: 0.9,
          weight: 0,
          lastUpdate: Date.now(),
        },
      ];
      // Zero weight should return default 0.5
      const aggregate = calculateAggregateReputation(scores);
      expect(aggregate).toBe(0.5);
    });
  });
});
