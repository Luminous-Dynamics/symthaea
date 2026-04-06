// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Bridge Module Benchmarks
 *
 * Performance tests for the cross-hApp bridge protocol:
 * - Message creation and routing
 * - LocalBridge operations
 * - BridgeRouter with middleware
 * - Reputation aggregation
 */

import { bench, describe } from 'vitest';
import {
  LocalBridge,
  BridgeRouter,
  BridgeMessageType,
  createReputationQuery,
  createCrossHappReputation,
  createHappRegistration,
  createCredentialVerification,
  createVerificationResult,
  createBroadcastEvent,
  calculateAggregateReputation,
  type HappReputationScore,
  type AnyBridgeMessage,
} from '../src/bridge/index.js';

// ============================================================================
// Message Creation Benchmarks
// ============================================================================

describe('Bridge: Message Creation', () => {
  bench('createReputationQuery()', () => {
    createReputationQuery('test-happ', 'agent-pubkey-12345');
  });

  bench('createCrossHappReputation() - 1 score', () => {
    const scores: HappReputationScore[] = [
      { happ: 'happ-1', score: 0.85, weight: 1, lastUpdate: Date.now() },
    ];
    createCrossHappReputation('test-happ', 'agent-1', scores);
  });

  bench('createCrossHappReputation() - 5 scores', () => {
    const scores: HappReputationScore[] = Array.from({ length: 5 }, (_, i) => ({
      happ: `happ-${i}`,
      score: 0.7 + Math.random() * 0.3,
      weight: 1 + Math.random() * 10,
      lastUpdate: Date.now() - i * 86400000,
    }));
    createCrossHappReputation('test-happ', 'agent-1', scores);
  });

  bench('createCrossHappReputation() - 10 scores', () => {
    const scores: HappReputationScore[] = Array.from({ length: 10 }, (_, i) => ({
      happ: `happ-${i}`,
      score: 0.5 + Math.random() * 0.5,
      weight: 1 + Math.random() * 5,
      lastUpdate: Date.now() - i * 3600000,
    }));
    createCrossHappReputation('test-happ', 'agent-1', scores);
  });

  bench('createHappRegistration()', () => {
    createHappRegistration('new-happ', {
      name: 'New Application',
      version: '1.0.0',
      capabilities: ['reputation', 'credentials', 'messaging'],
    });
  });

  bench('createCredentialVerification()', () => {
    createCredentialVerification('requester-happ', 'holder-agent', 'credential-id-12345', {
      purpose: 'identity-verification',
      requiredAttributes: ['name', 'email', 'verified_at'],
    });
  });

  bench('createVerificationResult()', () => {
    createVerificationResult(
      'verifier-happ',
      'agent-1',
      'credential-id-12345',
      true,
      {
        verifiedAt: Date.now(),
        expiresAt: Date.now() + 86400000,
        attributes: { name: 'Test User', verified: true },
      }
    );
  });

  bench('createBroadcastEvent()', () => {
    createBroadcastEvent('source-happ', 'reputation_updated', {
      agentId: 'agent-1',
      oldScore: 0.75,
      newScore: 0.82,
      reason: 'positive_interaction',
    });
  });
});

// ============================================================================
// LocalBridge Benchmarks
// ============================================================================

describe('Bridge: LocalBridge', () => {
  bench('LocalBridge creation', () => {
    new LocalBridge();
  });

  bench('registerHapp() - single', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('test-happ');
  });

  bench('registerHapp() - 10 happs', () => {
    const bridge = new LocalBridge();
    for (let i = 0; i < 10; i++) {
      bridge.registerHapp(`happ-${i}`);
    }
  });

  bench('registerHapp() - 50 happs', () => {
    const bridge = new LocalBridge();
    for (let i = 0; i < 50; i++) {
      bridge.registerHapp(`happ-${i}`);
    }
  });

  bench('isRegistered() - existing', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('test-happ');
    bridge.isRegistered('test-happ');
  });

  bench('isRegistered() - non-existing', () => {
    const bridge = new LocalBridge();
    bridge.isRegistered('unknown-happ');
  });

  bench('getRegisteredHapps() - 10 happs', () => {
    const bridge = new LocalBridge();
    for (let i = 0; i < 10; i++) {
      bridge.registerHapp(`happ-${i}`);
    }
    bridge.getRegisteredHapps();
  });

  bench('on() handler registration', () => {
    const bridge = new LocalBridge();
    bridge.on('test-happ', BridgeMessageType.ReputationQuery, () => {});
  });

  bench('send() - single message', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('target-happ');
    bridge.on('target-happ', BridgeMessageType.ReputationQuery, () => {});
    bridge.send('target-happ', createReputationQuery('source', 'agent-1'));
  });

  bench('send() - 100 messages', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('target-happ');
    bridge.on('target-happ', BridgeMessageType.ReputationQuery, () => {});
    for (let i = 0; i < 100; i++) {
      bridge.send('target-happ', createReputationQuery('source', `agent-${i}`));
    }
  });

  bench('broadcast() - 10 happs', () => {
    const bridge = new LocalBridge();
    for (let i = 0; i < 10; i++) {
      bridge.registerHapp(`happ-${i}`);
      bridge.on(`happ-${i}`, BridgeMessageType.BroadcastEvent, () => {});
    }
    bridge.broadcast(createBroadcastEvent('source', 'test_event', { data: 'test' }));
  });
});

// ============================================================================
// BridgeRouter Benchmarks
// ============================================================================

describe('Bridge: BridgeRouter', () => {
  bench('BridgeRouter creation', () => {
    new BridgeRouter({});
  });

  bench('BridgeRouter with handlers', () => {
    new BridgeRouter({
      onReputationQuery: () => {},
      onCrossHappReputation: () => {},
      onHappRegistration: () => {},
    });
  });

  bench('route() - single message', async () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
    });
    await router.route(createReputationQuery('test', 'agent-1'));
  });

  bench('route() - 50 messages', async () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
    });
    const promises = [];
    for (let i = 0; i < 50; i++) {
      promises.push(router.route(createReputationQuery('test', `agent-${i}`)));
    }
    await Promise.all(promises);
  });

  bench('use() middleware chain (3 middlewares)', async () => {
    const router = new BridgeRouter({});
    router.use(async (msg, next) => {
      await next();
    });
    router.use(async (msg, next) => {
      await next();
    });
    router.use(async (msg, next) => {
      await next();
    });
    await router.route(createReputationQuery('test', 'agent-1'));
  });

  bench('route() with 5 middlewares', async () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
    });
    for (let i = 0; i < 5; i++) {
      router.use(async (msg, next) => {
        await next();
      });
    }
    await router.route(createReputationQuery('test', 'agent-1'));
  });

  bench('route() different message types', async () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
      onCrossHappReputation: () => {},
      onBroadcastEvent: () => {},
    });

    await router.route(createReputationQuery('test', 'agent-1'));
    await router.route(
      createCrossHappReputation('test', 'agent-1', [
        { happ: 'h1', score: 0.8, weight: 1, lastUpdate: Date.now() },
      ])
    );
    await router.route(createBroadcastEvent('test', 'event', {}));
  });
});

// ============================================================================
// Reputation Aggregation Benchmarks
// ============================================================================

describe('Bridge: Reputation Aggregation', () => {
  bench('calculateAggregateReputation() - 1 score', () => {
    calculateAggregateReputation([
      { happ: 'happ-1', score: 0.85, weight: 1, lastUpdate: Date.now() },
    ]);
  });

  bench('calculateAggregateReputation() - 5 scores', () => {
    const scores: HappReputationScore[] = Array.from({ length: 5 }, (_, i) => ({
      happ: `happ-${i}`,
      score: 0.6 + Math.random() * 0.4,
      weight: 1 + i,
      lastUpdate: Date.now(),
    }));
    calculateAggregateReputation(scores);
  });

  bench('calculateAggregateReputation() - 10 scores', () => {
    const scores: HappReputationScore[] = Array.from({ length: 10 }, (_, i) => ({
      happ: `happ-${i}`,
      score: 0.5 + Math.random() * 0.5,
      weight: Math.random() * 10,
      lastUpdate: Date.now() - i * 3600000,
    }));
    calculateAggregateReputation(scores);
  });

  bench('calculateAggregateReputation() - 50 scores', () => {
    const scores: HappReputationScore[] = Array.from({ length: 50 }, (_, i) => ({
      happ: `happ-${i}`,
      score: Math.random(),
      weight: Math.random() * 100,
      lastUpdate: Date.now() - i * 60000,
    }));
    calculateAggregateReputation(scores);
  });

  bench('calculateAggregateReputation() - empty', () => {
    calculateAggregateReputation([]);
  });

  bench('calculateAggregateReputation() - uniform weights', () => {
    const scores: HappReputationScore[] = Array.from({ length: 20 }, (_, i) => ({
      happ: `happ-${i}`,
      score: 0.7 + (i % 3) * 0.1,
      weight: 1,
      lastUpdate: Date.now(),
    }));
    calculateAggregateReputation(scores);
  });

  bench('calculateAggregateReputation() - varying weights', () => {
    const scores: HappReputationScore[] = Array.from({ length: 20 }, (_, i) => ({
      happ: `happ-${i}`,
      score: 0.7 + (i % 3) * 0.1,
      weight: Math.pow(2, i % 5),
      lastUpdate: Date.now(),
    }));
    calculateAggregateReputation(scores);
  });
});

// ============================================================================
// Realistic Scenarios
// ============================================================================

describe('Bridge: Realistic Scenarios', () => {
  bench('cross-hApp reputation lookup workflow', async () => {
    const bridge = new LocalBridge();

    // Register happs
    const happs = ['marketplace', 'praxis', 'mail', 'supplychain'];
    happs.forEach((h) => bridge.registerHapp(h));

    // Set up handlers
    happs.forEach((h) => {
      bridge.on(h, BridgeMessageType.ReputationQuery, () => {});
    });

    // Query reputation across all happs
    const agentId = 'agent-12345';
    const query = createReputationQuery('marketplace', agentId);

    // Simulate sending to all happs
    happs.forEach((h) => bridge.send(h, query));

    // Aggregate results
    const scores: HappReputationScore[] = happs.map((h, i) => ({
      happ: h,
      score: 0.7 + Math.random() * 0.3,
      weight: i + 1,
      lastUpdate: Date.now(),
    }));

    const aggregate = calculateAggregateReputation(scores);
  });

  bench('hApp registration and capability exchange', async () => {
    const bridge = new LocalBridge();

    // Register multiple happs with capabilities
    for (let i = 0; i < 5; i++) {
      bridge.registerHapp(`happ-${i}`);
      bridge.on(`happ-${i}`, BridgeMessageType.HappRegistration, () => {});
    }

    // Broadcast registration
    const registration = createHappRegistration('new-happ', {
      name: 'New Application',
      version: '2.0.0',
      capabilities: ['reputation', 'credentials'],
    });

    bridge.broadcast(registration);
  });

  bench('credential verification flow', async () => {
    const router = new BridgeRouter({
      onCredentialVerification: async () => {},
      onVerificationResult: async () => {},
    });

    router.use(async (msg, next) => {
      // Audit logging middleware
      await next();
    });

    router.use(async (msg, next) => {
      // Rate limiting middleware
      await next();
    });

    // Request verification
    const request = createCredentialVerification(
      'requester-happ',
      'holder-agent',
      'credential-id',
      { purpose: 'authentication' }
    );
    await router.route(request);

    // Return result
    const result = createVerificationResult(
      'verifier-happ',
      'holder-agent',
      'credential-id',
      true,
      { verifiedAt: Date.now() }
    );
    await router.route(result);
  });

  bench('event broadcast to 20 subscribers', () => {
    const bridge = new LocalBridge();

    // Set up 20 subscriber happs
    for (let i = 0; i < 20; i++) {
      bridge.registerHapp(`subscriber-${i}`);
      bridge.on(`subscriber-${i}`, BridgeMessageType.BroadcastEvent, () => {});
    }

    // Broadcast event
    bridge.broadcast(
      createBroadcastEvent('publisher', 'user_action', {
        userId: 'user-1',
        action: 'purchase',
        timestamp: Date.now(),
      })
    );
  });

  bench('full ecosystem message flow (10 happs, 100 messages)', async () => {
    const bridge = new LocalBridge();
    const router = new BridgeRouter({
      onReputationQuery: () => {},
      onCrossHappReputation: () => {},
    });

    // Register happs
    for (let i = 0; i < 10; i++) {
      bridge.registerHapp(`happ-${i}`);
      bridge.on(`happ-${i}`, BridgeMessageType.ReputationQuery, () => {});
    }

    // Process messages
    const promises = [];
    for (let i = 0; i < 100; i++) {
      const sourceHapp = `happ-${i % 10}`;
      const targetHapp = `happ-${(i + 1) % 10}`;

      if (i % 2 === 0) {
        bridge.send(targetHapp, createReputationQuery(sourceHapp, `agent-${i}`));
      } else {
        promises.push(
          router.route(createReputationQuery(sourceHapp, `agent-${i}`))
        );
      }
    }

    await Promise.all(promises);
  });
});
