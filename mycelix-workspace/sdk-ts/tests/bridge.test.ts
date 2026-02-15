/**
 * Bridge Module Tests
 */

import { describe, it, expect } from 'vitest';
import {
  BridgeMessageType,
  calculateAggregateReputation,
  createReputationQuery,
  createCrossHappReputation,
  createCredentialVerification,
  createVerificationResult,
  createBroadcastEvent,
  createHappRegistration,
  createMessageHandler,
  LocalBridge,
  BridgeRouter,
  type AnyBridgeMessage,
} from '../src/bridge/index.js';
import { createReputation, recordPositive } from '../src/matl/index.js';

describe('Bridge - Aggregate Reputation', () => {
  it('should return 0.5 for empty scores', () => {
    expect(calculateAggregateReputation([])).toBe(0.5);
  });

  it('should calculate weighted average', () => {
    const scores = [
      { happ: 'mail', score: 0.9, weight: 100, lastUpdate: Date.now() },
      { happ: 'marketplace', score: 0.8, weight: 50, lastUpdate: Date.now() },
    ];

    // (0.9 * 100 + 0.8 * 50) / 150 = (90 + 40) / 150 = 130 / 150 = 0.867
    const aggregate = calculateAggregateReputation(scores);
    expect(aggregate).toBeCloseTo(0.867, 2);
  });

  it('should handle single score', () => {
    const scores = [
      { happ: 'single', score: 0.75, weight: 10, lastUpdate: Date.now() },
    ];

    expect(calculateAggregateReputation(scores)).toBe(0.75);
  });

  it('should handle zero total weight', () => {
    const scores = [
      { happ: 'zero', score: 0.9, weight: 0, lastUpdate: Date.now() },
    ];

    expect(calculateAggregateReputation(scores)).toBe(0.5);
  });
});

describe('Bridge - Message Creation', () => {
  it('should create reputation query', () => {
    const msg = createReputationQuery('mail-happ', 'agent123');

    expect(msg.type).toBe(BridgeMessageType.ReputationQuery);
    expect(msg.sourceHapp).toBe('mail-happ');
    expect(msg.agent).toBe('agent123');
    expect(msg.timestamp).toBeLessThanOrEqual(Date.now());
  });

  it('should create cross-hApp reputation', () => {
    const scores = [
      { happ: 'mail', score: 0.9, weight: 10, lastUpdate: Date.now() },
    ];
    const msg = createCrossHappReputation('bridge', 'agent123', scores);

    expect(msg.type).toBe(BridgeMessageType.CrossHappReputation);
    expect(msg.agent).toBe('agent123');
    expect(msg.scores).toHaveLength(1);
    expect(msg.aggregate).toBe(0.9);
  });

  it('should create credential verification', () => {
    const msg = createCredentialVerification('edunet', 'hash123', 'cert-issuer');

    expect(msg.type).toBe(BridgeMessageType.CredentialVerification);
    expect(msg.credentialHash).toBe('hash123');
    expect(msg.issuerHapp).toBe('cert-issuer');
  });

  it('should create verification result', () => {
    const msg = createVerificationResult(
      'cert-issuer',
      'hash123',
      true,
      'university',
      ['degree:cs', 'year:2024']
    );

    expect(msg.type).toBe(BridgeMessageType.VerificationResult);
    expect(msg.valid).toBe(true);
    expect(msg.claims).toContain('degree:cs');
  });

  it('should create broadcast event', () => {
    const payload = new TextEncoder().encode('test payload');
    const msg = createBroadcastEvent('mail', 'new_message', payload);

    expect(msg.type).toBe(BridgeMessageType.BroadcastEvent);
    expect(msg.eventType).toBe('new_message');
    expect(msg.payload).toEqual(payload);
  });

  it('should create hApp registration', () => {
    const msg = createHappRegistration(
      'new-happ',
      'marketplace',
      'Mycelix Marketplace',
      ['listings', 'transactions', 'reviews']
    );

    expect(msg.type).toBe(BridgeMessageType.HappRegistration);
    expect(msg.happId).toBe('marketplace');
    expect(msg.happName).toBe('Mycelix Marketplace');
    expect(msg.capabilities).toContain('listings');
  });
});

describe('Bridge - LocalBridge', () => {
  it('should register hApps', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');
    bridge.registerHapp('marketplace');

    // No error means success
    expect(true).toBe(true);
  });

  it('should handle message routing', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');
    bridge.registerHapp('marketplace');

    let received: any = null;
    bridge.on('marketplace', BridgeMessageType.ReputationQuery, (msg) => {
      received = msg;
    });

    const query = createReputationQuery('mail', 'agent123');
    bridge.send('marketplace', query);

    expect(received).not.toBeNull();
    expect(received.agent).toBe('agent123');
  });

  it('should broadcast messages', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');
    bridge.registerHapp('marketplace');
    bridge.registerHapp('edunet');

    const received: string[] = [];

    bridge.on('marketplace', BridgeMessageType.BroadcastEvent, () => {
      received.push('marketplace');
    });
    bridge.on('edunet', BridgeMessageType.BroadcastEvent, () => {
      received.push('edunet');
    });

    const event = createBroadcastEvent('mail', 'test', new Uint8Array());
    bridge.broadcast(event);

    expect(received).toContain('marketplace');
    expect(received).toContain('edunet');
    expect(received).not.toContain('mail'); // Source should not receive
  });

  it('should track cross-hApp reputation', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');
    bridge.registerHapp('marketplace');

    let rep = createReputation('agent1');
    rep = recordPositive(rep);
    rep = recordPositive(rep);
    bridge.setReputation('mail', 'agent1', rep);

    let rep2 = createReputation('agent1');
    rep2 = recordPositive(rep2);
    bridge.setReputation('marketplace', 'agent1', rep2);

    const scores = bridge.getCrossHappReputation('agent1');

    expect(scores).toHaveLength(2);
    expect(scores.find(s => s.happ === 'mail')?.score).toBe(0.75);
    expect(scores.find(s => s.happ === 'marketplace')?.score).toBeCloseTo(0.667, 2);
  });

  it('should calculate aggregate reputation', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');

    let rep = createReputation('agent1');
    rep = recordPositive(rep);
    rep = recordPositive(rep);
    bridge.setReputation('mail', 'agent1', rep);

    const aggregate = bridge.getAggregateReputation('agent1');
    expect(aggregate).toBe(0.75);
  });

  it('should return 0.5 for unknown agent', () => {
    const bridge = new LocalBridge();
    const aggregate = bridge.getAggregateReputation('unknown');
    expect(aggregate).toBe(0.5);
  });

  it('should check if hApp is registered', () => {
    const bridge = new LocalBridge();
    expect(bridge.isRegistered('mail')).toBe(false);
    bridge.registerHapp('mail');
    expect(bridge.isRegistered('mail')).toBe(true);
  });

  it('should get list of registered hApps', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');
    bridge.registerHapp('marketplace');
    const happs = bridge.getRegisteredHapps();
    expect(happs).toContain('mail');
    expect(happs).toContain('marketplace');
    expect(happs).toHaveLength(2);
  });
});

describe('Bridge - Input Validation', () => {
  it('should throw on empty sourceHapp in createReputationQuery', () => {
    expect(() => createReputationQuery('', 'agent')).toThrow();
  });

  it('should throw on empty agent in createReputationQuery', () => {
    expect(() => createReputationQuery('source', '')).toThrow();
  });

  it('should throw on empty credentialHash in createCredentialVerification', () => {
    expect(() => createCredentialVerification('source', '', 'issuer')).toThrow();
  });

  it('should throw on empty eventType in createBroadcastEvent', () => {
    expect(() => createBroadcastEvent('source', '', new Uint8Array())).toThrow();
  });

  it('should throw on empty happId in createHappRegistration', () => {
    expect(() => createHappRegistration('source', '', 'name', [])).toThrow();
  });

  it('should throw on invalid weight in calculateAggregateReputation', () => {
    expect(() => calculateAggregateReputation([
      { happ: 'mail', score: 0.9, weight: -1, lastUpdate: Date.now() },
    ])).toThrow(/Invalid weight/);
  });

  it('should throw on invalid score in calculateAggregateReputation', () => {
    expect(() => calculateAggregateReputation([
      { happ: 'mail', score: 1.5, weight: 1, lastUpdate: Date.now() },
    ])).toThrow(/Invalid score/);
  });

  it('should throw on empty happId in LocalBridge.registerHapp', () => {
    const bridge = new LocalBridge();
    expect(() => bridge.registerHapp('')).toThrow();
  });

  it('should throw on empty agentId in LocalBridge.getCrossHappReputation', () => {
    const bridge = new LocalBridge();
    expect(() => bridge.getCrossHappReputation('')).toThrow();
  });
});

describe('Bridge - LocalBridge Cleanup Methods', () => {
  it('should unregister a hApp', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');
    expect(bridge.isRegistered('mail')).toBe(true);
    bridge.unregisterHapp('mail');
    expect(bridge.isRegistered('mail')).toBe(false);
  });

  it('should clear all registrations', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');
    bridge.registerHapp('marketplace');
    expect(bridge.getRegisteredHapps()).toHaveLength(2);
    bridge.clear();
    expect(bridge.getRegisteredHapps()).toHaveLength(0);
  });

  it('should remove reputation for an agent', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');

    let rep = createReputation('agent1');
    rep = recordPositive(rep);
    bridge.setReputation('mail', 'agent1', rep);

    expect(bridge.getCrossHappReputation('agent1')).toHaveLength(1);
    bridge.removeReputation('mail', 'agent1');
    expect(bridge.getCrossHappReputation('agent1')).toHaveLength(0);
  });

  it('should remove a message handler', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('mail');

    let received = false;
    bridge.on('mail', BridgeMessageType.ReputationQuery, () => {
      received = true;
    });

    bridge.off('mail', BridgeMessageType.ReputationQuery);

    const query = createReputationQuery('other', 'agent1');
    bridge.send('mail', query);
    expect(received).toBe(false);
  });

  it('should throw on empty happId in unregisterHapp', () => {
    const bridge = new LocalBridge();
    expect(() => bridge.unregisterHapp('')).toThrow();
  });

  it('should throw on empty happId in removeReputation', () => {
    const bridge = new LocalBridge();
    expect(() => bridge.removeReputation('', 'agent1')).toThrow();
  });

  it('should throw on empty agentId in removeReputation', () => {
    const bridge = new LocalBridge();
    expect(() => bridge.removeReputation('mail', '')).toThrow();
  });
});

describe('BridgeRouter', () => {
  it('should route messages to correct handlers', async () => {
    const received: string[] = [];

    const router = new BridgeRouter({
      onReputationQuery: (msg) => {
        received.push(`query:${msg.agent}`);
      },
      onCredentialVerification: (msg) => {
        received.push(`verify:${msg.credentialHash}`);
      },
    });

    await router.route(createReputationQuery('source', 'agent1'));
    await router.route(createCredentialVerification('source', 'hash123', 'issuer'));

    expect(received).toContain('query:agent1');
    expect(received).toContain('verify:hash123');
  });

  it('should support fluent handler registration', async () => {
    const received: string[] = [];

    const router = new BridgeRouter()
      .on(BridgeMessageType.ReputationQuery, (msg) => {
        received.push(`query:${msg.agent}`);
      })
      .on(BridgeMessageType.BroadcastEvent, (msg) => {
        received.push(`event:${msg.eventType}`);
      });

    await router.route(createReputationQuery('source', 'agent1'));
    await router.route(createBroadcastEvent('source', 'test_event', new Uint8Array([1, 2, 3])));

    expect(received).toHaveLength(2);
  });

  it('should execute middleware in order', async () => {
    const order: number[] = [];

    const router = new BridgeRouter({
      onReputationQuery: () => {
        order.push(3);
      },
    });

    router.use(async (msg, next) => {
      order.push(1);
      await next();
      order.push(4);
    });

    router.use(async (msg, next) => {
      order.push(2);
      await next();
    });

    await router.route(createReputationQuery('source', 'agent1'));

    expect(order).toEqual([1, 2, 3, 4]);
  });

  it('should call onUnhandled for messages without handler', async () => {
    let unhandledMsg: AnyBridgeMessage | null = null;

    const router = new BridgeRouter({
      onUnhandled: (msg) => {
        unhandledMsg = msg;
      },
    });

    await router.route(createReputationQuery('source', 'agent1'));

    expect(unhandledMsg).not.toBeNull();
    expect(unhandledMsg!.type).toBe(BridgeMessageType.ReputationQuery);
  });

  it('should throw on unhandled when configured', async () => {
    const router = new BridgeRouter({}, { throwOnUnhandled: true });

    await expect(
      router.route(createReputationQuery('source', 'agent1'))
    ).rejects.toThrow();
  });

  it('should track routing statistics', async () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
    });

    await router.route(createReputationQuery('source', 'agent1'));
    await router.route(createReputationQuery('source', 'agent2'));
    await router.route(createBroadcastEvent('source', 'event', new Uint8Array()));

    const stats = router.getStats();

    expect(stats.messagesRouted).toBe(3);
    expect(stats.messagesUnhandled).toBe(1);
    expect(stats.byType[BridgeMessageType.ReputationQuery]).toBe(2);
    expect(stats.byType[BridgeMessageType.BroadcastEvent]).toBe(1);
  });

  it('should check hasHandler correctly', () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
    });

    expect(router.hasHandler(BridgeMessageType.ReputationQuery)).toBe(true);
    expect(router.hasHandler(BridgeMessageType.BroadcastEvent)).toBe(false);
  });

  it('should return handled types', () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
      onCredentialVerification: () => {},
    });

    const types = router.getHandledTypes();

    expect(types).toContain(BridgeMessageType.ReputationQuery);
    expect(types).toContain(BridgeMessageType.CredentialVerification);
    expect(types).not.toContain(BridgeMessageType.BroadcastEvent);
  });

  it('should route many messages in sequence', async () => {
    const received: string[] = [];

    const router = new BridgeRouter({
      onReputationQuery: (msg) => {
        received.push(msg.agent);
      },
    });

    await router.routeMany([
      createReputationQuery('source', 'agent1'),
      createReputationQuery('source', 'agent2'),
      createReputationQuery('source', 'agent3'),
    ]);

    expect(received).toEqual(['agent1', 'agent2', 'agent3']);
  });

  it('should route messages in parallel', async () => {
    const received: string[] = [];

    const router = new BridgeRouter({
      onReputationQuery: async (msg) => {
        await new Promise(resolve => setTimeout(resolve, 10));
        received.push(msg.agent);
      },
    });

    await router.routeParallel([
      createReputationQuery('source', 'agent1'),
      createReputationQuery('source', 'agent2'),
    ]);

    expect(received).toHaveLength(2);
  });

  it('should reset stats', async () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
    });

    await router.route(createReputationQuery('source', 'agent1'));
    expect(router.getStats().messagesRouted).toBe(1);

    router.resetStats();
    expect(router.getStats().messagesRouted).toBe(0);
  });

  it('should clear handlers and middleware', () => {
    const router = new BridgeRouter({
      onReputationQuery: () => {},
    });

    router.use(async (_, next) => next());

    expect(router.hasHandler(BridgeMessageType.ReputationQuery)).toBe(true);

    router.clear();

    expect(router.hasHandler(BridgeMessageType.ReputationQuery)).toBe(false);
  });
});

describe('createMessageHandler', () => {
  it('should create type-safe handler', () => {
    const handler = createMessageHandler(
      BridgeMessageType.ReputationQuery,
      (msg) => {
        expect(msg.agent).toBeDefined();
      }
    );

    expect(handler.type).toBe(BridgeMessageType.ReputationQuery);

    const query = createReputationQuery('source', 'agent1');
    expect(handler.matches(query)).toBe(true);

    const broadcast = createBroadcastEvent('source', 'event', new Uint8Array());
    expect(handler.matches(broadcast)).toBe(false);
  });
});
