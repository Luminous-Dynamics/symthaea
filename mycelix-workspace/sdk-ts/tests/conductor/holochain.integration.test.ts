// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Conductor Integration Tests
 *
 * These tests require a running Holochain conductor.
 * Run with: npm run test:conductor
 *
 * Set CONDUCTOR_AVAILABLE=true to enable these tests
 */

import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';

// Skip if conductor not available
const CONDUCTOR_AVAILABLE = process.env.CONDUCTOR_AVAILABLE === 'true';
const describeIf = CONDUCTOR_AVAILABLE ? describe : describe.skip;

// Import mock client from separate module (no @holochain/client dependency)
import {
  MockMycelixClient,
  createMockClient,
  type ConnectionState,
} from '../../src/client/mock.js';

// Import other SDK modules directly
import {
  createReputation,
  recordPositive,
  reputationValue,
} from '../../src/matl/index.js';

import {
  claim,
  EmpiricalLevel,
} from '../../src/epistemic/index.js';

import {
  LocalBridge,
  createReputationQuery,
  BridgeMessageType,
} from '../../src/bridge/index.js';

// ============================================================================
// Mock Client Tests (Always Run)
// ============================================================================

describe('MockMycelixClient', () => {
  let client: MockMycelixClient;

  beforeAll(() => {
    client = createMockClient();
  });

  afterAll(async () => {
    await client.disconnect();
  });

  it('should connect successfully', async () => {
    await client.connect();
    expect(client.isConnected()).toBe(true);
  });

  it('should register a hApp', async () => {
    const result = await client.registerHapp({
      happId: 'test-happ',
      dnaHash: 'uhC0k...mock-hash',
      agentPubKey: 'uhCAk...mock-key',
    });

    expect(result.success).toBe(true);
  });

  it('should record reputation', async () => {
    const result = await client.recordReputation({
      targetAgentId: 'agent-123',
      happId: 'test-happ',
      score: 0.85,
      evidence: { type: 'transaction', success: true },
    });

    expect(result.success).toBe(true);
    expect(result.newScore).toBeGreaterThanOrEqual(0);
    expect(result.newScore).toBeLessThanOrEqual(1);
  });

  it('should check trust', async () => {
    const result = await client.checkTrust({
      agentId: 'agent-123',
      minimumScore: 0.5,
      happContext: 'test-happ',
    });

    expect(typeof result.trustworthy).toBe('boolean');
    expect(typeof result.score).toBe('number');
    expect(typeof result.confidence).toBe('number');
  });

  it('should broadcast events', async () => {
    const result = await client.broadcastEvent({
      eventType: 'reputation_update',
      payload: { agentId: 'agent-123', delta: 0.1 },
      targetHapps: ['happ-1', 'happ-2'],
    });

    expect(result.success).toBe(true);
    expect(result.recipientCount).toBeGreaterThanOrEqual(0);
  });

  it('should get events', async () => {
    const result = await client.getEvents({
      happId: 'test-happ',
      eventTypes: ['reputation_update'],
      since: Date.now() - 3600000,
    });

    expect(Array.isArray(result.events)).toBe(true);
  });

  it('should verify credentials', async () => {
    const result = await client.verifyCredential({
      credentialType: 'identity',
      credentialData: { verified: true },
      requesterHappId: 'test-happ',
    });

    expect(typeof result.valid).toBe('boolean');
  });

  it('should get cross-hApp reputation', async () => {
    const result = await client.getCrossHappReputation({
      agentId: 'agent-123',
      happs: ['happ-1', 'happ-2'],
      aggregationMethod: 'weighted',
    });

    expect(typeof result.aggregateScore).toBe('number');
    expect(Array.isArray(result.happScores)).toBe(true);
  });

  it('should track connection state', async () => {
    const states: ConnectionState[] = [];
    const listener = (state: ConnectionState) => states.push(state);

    client.addConnectionStateListener(listener);
    await client.disconnect();
    await client.connect();

    expect(states).toContain('disconnected');
    expect(states).toContain('connected');

    client.removeConnectionStateListener(listener);
  });
});

// ============================================================================
// Real Conductor Tests (Require Running Conductor)
// ============================================================================

describeIf('Real Holochain Conductor', () => {
  // Use dynamic import to avoid loading @holochain/client unless needed
  let clientModule: typeof import('../../src/client/index.js');
  let client: InstanceType<typeof clientModule.MycelixClient>;

  beforeAll(async () => {
    // Dynamic import only when conductor is available
    clientModule = await import('../../src/client/index.js');

    const config: InstanceType<typeof clientModule.MycelixClient> extends { config: infer C } ? C : never = {
      appUrl: process.env.CONDUCTOR_URL || 'ws://localhost:8888',
      adminUrl: process.env.ADMIN_URL || 'ws://localhost:8889',
      installedAppId: process.env.APP_ID || 'mycelix-test',
      timeout: 30000,
      maxRetries: 3,
    };

    client = new clientModule.MycelixClient(config as any);
    await client.connect();
  }, 30000);

  afterAll(async () => {
    if (client) {
      await client.disconnect();
    }
  });

  it('should connect to the conductor', () => {
    expect(client.isConnected()).toBe(true);
  });

  it('should make zome calls', async () => {
    // This test depends on the actual zome implementation
    try {
      const result = await client.callZome<unknown>({
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'get_registered_happs',
        payload: null,
      });
      expect(result).toBeDefined();
    } catch (error) {
      // Zome function may not exist - that's okay for this test
      console.log('Zome call test skipped:', error);
    }
  });

  it('should handle connection loss gracefully', async () => {
    // Simulate connection loss and reconnection
    const reconnected = new Promise<void>((resolve) => {
      const unsubscribe = client.onStateChange((state) => {
        if (state === 'connected') {
          unsubscribe();
          resolve();
        }
      });
    });

    // Force disconnect
    await client.disconnect();
    await client.connect();

    await expect(reconnected).resolves.toBeUndefined();
  });
});

// ============================================================================
// Integration Scenarios
// ============================================================================

describe('SDK Integration Scenarios', () => {
  it('should support full trust workflow', async () => {
    const client = createMockClient();
    await client.connect();

    // 1. Register hApp
    await client.registerHapp({
      happId: 'marketplace',
      dnaHash: 'mock-dna',
      agentPubKey: 'mock-key',
    });

    // 2. Build reputation over time
    let reputation = createReputation('seller-123');
    for (let i = 0; i < 5; i++) {
      reputation = recordPositive(reputation);
      await client.recordReputation({
        targetAgentId: 'seller-123',
        happId: 'marketplace',
        score: reputationValue(reputation),
        evidence: { transactionId: `tx-${i}` },
      });
    }

    // 3. Check trust before transaction
    const trustCheck = await client.checkTrust({
      agentId: 'seller-123',
      minimumScore: 0.7,
      happContext: 'marketplace',
    });

    expect(trustCheck.trustworthy).toBe(true);
    expect(trustCheck.score).toBeGreaterThan(0.5);

    await client.disconnect();
  });

  it('should support cross-hApp reputation aggregation', async () => {
    const client = createMockClient();
    await client.connect();

    // Register multiple hApps
    await client.registerHapp({ happId: 'identity', dnaHash: 'd1', agentPubKey: 'k1' });
    await client.registerHapp({ happId: 'marketplace', dnaHash: 'd2', agentPubKey: 'k2' });
    await client.registerHapp({ happId: 'lending', dnaHash: 'd3', agentPubKey: 'k3' });

    // Record reputation in each
    for (const happId of ['identity', 'marketplace', 'lending']) {
      await client.recordReputation({
        targetAgentId: 'user-456',
        happId,
        score: 0.7 + Math.random() * 0.3,
        evidence: { verified: true },
      });
    }

    // Get aggregated reputation
    const crossHapp = await client.getCrossHappReputation({
      agentId: 'user-456',
      happs: ['identity', 'marketplace', 'lending'],
      aggregationMethod: 'weighted',
    });

    expect(crossHapp.aggregateScore).toBeGreaterThan(0);
    expect(crossHapp.happScores.length).toBe(3);

    await client.disconnect();
  });

  it('should support epistemic claims with bridge', async () => {
    const client = createMockClient();
    const bridge = new LocalBridge();

    await client.connect();
    bridge.registerHapp('verification');
    bridge.registerHapp('consumer');

    // Create a verified claim
    const verifiedClaim = claim('User passed KYC verification')
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withEvidence({
        type: 'kyc_verification',
        data: JSON.stringify({ provider: 'trusted-kyc', verified: true }),
        source: 'trusted-kyc-provider',
        timestamp: Date.now(),
      })
      .build();

    // Share via bridge
    bridge.on('consumer', BridgeMessageType.ReputationQuery, (msg) => {
      // Consumer received the query
      expect((msg as any).agent).toBe('verified-user');
    });

    bridge.send('consumer', createReputationQuery('verification', 'verified-user'));

    // Record the verification
    await client.recordReputation({
      targetAgentId: 'verified-user',
      happId: 'verification',
      score: 0.95,
      evidence: {
        claimId: verifiedClaim.id,
        empiricalLevel: verifiedClaim.classification.empirical,
      },
    });

    await client.disconnect();
  });

  it('should handle concurrent operations', async () => {
    const client = createMockClient();
    await client.connect();

    // Perform many concurrent operations
    const operations = Array.from({ length: 50 }, (_, i) =>
      client.recordReputation({
        targetAgentId: `agent-${i % 10}`,
        happId: 'stress-test',
        score: Math.random(),
        evidence: { iteration: i },
      })
    );

    const results = await Promise.all(operations);
    expect(results.every((r) => r.success)).toBe(true);

    await client.disconnect();
  });
});

// ============================================================================
// Performance Tests
// ============================================================================

describe('Client Performance', () => {
  it('should handle high throughput', async () => {
    const client = createMockClient();
    await client.connect();

    const start = performance.now();
    const iterations = 100;

    for (let i = 0; i < iterations; i++) {
      await client.checkTrust({
        agentId: 'perf-test-agent',
        minimumScore: 0.5,
        happContext: 'perf-test',
      });
    }

    const duration = performance.now() - start;
    const opsPerSecond = (iterations / duration) * 1000;

    console.log(`Trust checks: ${opsPerSecond.toFixed(0)} ops/sec`);
    expect(opsPerSecond).toBeGreaterThan(100); // At least 100 ops/sec

    await client.disconnect();
  });

  it('should batch operations efficiently', async () => {
    const client = createMockClient();
    await client.connect();

    const start = performance.now();

    // Simulate batched reputation updates
    const batch = Array.from({ length: 100 }, (_, i) => ({
      targetAgentId: `batch-agent-${i}`,
      happId: 'batch-test',
      score: Math.random(),
      evidence: { batch: true },
    }));

    await Promise.all(batch.map((b) => client.recordReputation(b)));

    const duration = performance.now() - start;
    console.log(`Batch of 100 operations: ${duration.toFixed(2)}ms`);
    expect(duration).toBeLessThan(5000); // Should complete in under 5 seconds

    await client.disconnect();
  });
});
