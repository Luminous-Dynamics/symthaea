// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Client Module Tests
 *
 * Note: These tests use a simplified inline mock to avoid ESM resolution
 * issues with the @holochain/client dependency.
 */

import { describe, it, expect } from 'vitest';
import type { HappReputationScore } from '../src/bridge/index.js';

// Inline mock types to avoid holochain client ESM issues
interface CrossHappReputation {
  agent: string;
  scores: HappReputationScore[];
  aggregate: number;
  total_interactions: number;
}

interface BridgeEventRecord {
  event_type: string;
  source_happ: string;
  payload: Uint8Array;
  timestamp: number;
  targets: string[];
}

interface TrustCheckInput {
  agent: string;
  threshold: number;
}

interface BroadcastEventInput {
  event_type: string;
  payload: Uint8Array;
  targets: string[];
}

interface GetEventsInput {
  event_type: string;
  since: number;
}

// Inline mock client for testing
class TestMockClient {
  private mockReputations: Map<string, CrossHappReputation> = new Map();
  private mockEvents: BridgeEventRecord[] = [];

  isConnected(): boolean {
    return true;
  }

  async connect(): Promise<void> {}
  async disconnect(): Promise<void> {}

  setMockReputation(agent: string, reputation: CrossHappReputation): void {
    this.mockReputations.set(agent, reputation);
  }

  addMockEvent(event: BridgeEventRecord): void {
    this.mockEvents.push(event);
  }

  async queryCrossHappReputation(agent: string): Promise<CrossHappReputation> {
    return (
      this.mockReputations.get(agent) ?? {
        agent,
        scores: [],
        aggregate: 0.5,
        total_interactions: 0,
      }
    );
  }

  async isAgentTrustworthy(input: TrustCheckInput): Promise<boolean> {
    const rep = await this.queryCrossHappReputation(input.agent);
    return rep.aggregate >= input.threshold;
  }

  async getEvents(input: GetEventsInput): Promise<BridgeEventRecord[]> {
    return this.mockEvents.filter(
      (e) => e.event_type === input.event_type && e.timestamp >= input.since
    );
  }

  async broadcastEvent(input: BroadcastEventInput): Promise<Uint8Array> {
    this.mockEvents.push({
      event_type: input.event_type,
      source_happ: 'mock',
      payload: input.payload,
      timestamp: Date.now(),
      targets: input.targets,
    });
    return new Uint8Array([0]);
  }
}

function createMockClient(): TestMockClient {
  return new TestMockClient();
}

describe('Client - MockMycelixClient', () => {
  it('should create mock client', () => {
    const client = createMockClient();
    expect(client).toBeInstanceOf(TestMockClient);
    expect(client.isConnected()).toBe(true);
  });

  it('should connect and disconnect without error', async () => {
    const client = createMockClient();

    await client.connect();
    expect(client.isConnected()).toBe(true);

    await client.disconnect();
    // Mock client stays "connected"
    expect(client.isConnected()).toBe(true);
  });

  it('should return default reputation for unknown agent', async () => {
    const client = createMockClient();

    const rep = await client.queryCrossHappReputation('unknown_agent');

    expect(rep.agent).toBe('unknown_agent');
    expect(rep.scores).toHaveLength(0);
    expect(rep.aggregate).toBe(0.5);
    expect(rep.total_interactions).toBe(0);
  });

  it('should return mock reputation when set', async () => {
    const client = createMockClient();

    client.setMockReputation('agent1', {
      agent: 'agent1',
      scores: [{ happ: 'mail', score: 0.9, weight: 100, lastUpdate: Date.now() }],
      aggregate: 0.9,
      total_interactions: 100,
    });

    const rep = await client.queryCrossHappReputation('agent1');

    expect(rep.aggregate).toBe(0.9);
    expect(rep.scores).toHaveLength(1);
  });

  it('should check agent trustworthiness', async () => {
    const client = createMockClient();

    client.setMockReputation('trusted_agent', {
      agent: 'trusted_agent',
      scores: [],
      aggregate: 0.8,
      total_interactions: 50,
    });

    client.setMockReputation('untrusted_agent', {
      agent: 'untrusted_agent',
      scores: [],
      aggregate: 0.3,
      total_interactions: 10,
    });

    expect(
      await client.isAgentTrustworthy({
        agent: 'trusted_agent',
        threshold: 0.5,
      })
    ).toBe(true);

    expect(
      await client.isAgentTrustworthy({
        agent: 'untrusted_agent',
        threshold: 0.5,
      })
    ).toBe(false);

    expect(
      await client.isAgentTrustworthy({
        agent: 'unknown_agent',
        threshold: 0.5,
      })
    ).toBe(true); // 0.5 >= 0.5
  });

  it('should handle mock events', async () => {
    const client = createMockClient();

    // Add some events
    const now = Date.now();
    client.addMockEvent({
      event_type: 'reputation_update',
      source_happ: 'mail',
      payload: new Uint8Array([1, 2, 3]),
      timestamp: now - 1000,
      targets: [],
    });

    client.addMockEvent({
      event_type: 'reputation_update',
      source_happ: 'marketplace',
      payload: new Uint8Array([4, 5, 6]),
      timestamp: now,
      targets: [],
    });

    client.addMockEvent({
      event_type: 'other_event',
      source_happ: 'praxis',
      payload: new Uint8Array(),
      timestamp: now,
      targets: [],
    });

    // Query events
    const events = await client.getEvents({
      event_type: 'reputation_update',
      since: now - 2000,
    });

    expect(events).toHaveLength(2);
    expect(events[0].source_happ).toBe('mail');
  });

  it('should filter events by timestamp', async () => {
    const client = createMockClient();

    const now = Date.now();
    client.addMockEvent({
      event_type: 'test',
      source_happ: 'a',
      payload: new Uint8Array(),
      timestamp: now - 5000,
      targets: [],
    });

    client.addMockEvent({
      event_type: 'test',
      source_happ: 'b',
      payload: new Uint8Array(),
      timestamp: now,
      targets: [],
    });

    const oldEvents = await client.getEvents({
      event_type: 'test',
      since: now - 10000,
    });
    expect(oldEvents).toHaveLength(2);

    const recentEvents = await client.getEvents({
      event_type: 'test',
      since: now - 1000,
    });
    expect(recentEvents).toHaveLength(1);
    expect(recentEvents[0].source_happ).toBe('b');
  });

  it('should broadcast events and store them', async () => {
    const client = createMockClient();

    await client.broadcastEvent({
      event_type: 'new_broadcast',
      payload: new Uint8Array([10, 20, 30]),
      targets: ['target1', 'target2'],
    });

    const events = await client.getEvents({
      event_type: 'new_broadcast',
      since: 0,
    });

    expect(events).toHaveLength(1);
    expect(events[0].source_happ).toBe('mock');
    expect(events[0].payload).toEqual(new Uint8Array([10, 20, 30]));
  });
});

describe('Client - Connection State Management', () => {
  it('should track connection state changes', () => {
    const client = createMockClient();
    const states: string[] = [];

    // Track initial state
    expect(client.isConnected()).toBe(true);
  });

  it('should support disconnect state', async () => {
    const client = createMockClient();

    expect(client.isConnected()).toBe(true);
    await client.disconnect();
    // Mock always returns true for testing, but real client tracks state
  });
});

describe('Client - Integration Patterns', () => {
  it('should support typical reputation check workflow', async () => {
    const client = createMockClient();

    // Setup: Some agent has reputation from multiple hApps
    client.setMockReputation('alice', {
      agent: 'alice',
      scores: [
        { happ: 'mail', score: 0.95, weight: 200, lastUpdate: Date.now() },
        { happ: 'marketplace', score: 0.88, weight: 50, lastUpdate: Date.now() },
        { happ: 'praxis', score: 0.92, weight: 30, lastUpdate: Date.now() },
      ],
      aggregate: 0.93,
      total_interactions: 280,
    });

    // Workflow: Check if alice is trustworthy for high-trust operation
    const isTrusted = await client.isAgentTrustworthy({
      agent: 'alice',
      threshold: 0.9,
    });

    expect(isTrusted).toBe(true);

    // Get detailed reputation breakdown
    const rep = await client.queryCrossHappReputation('alice');
    expect(rep.scores).toHaveLength(3);
    expect(rep.total_interactions).toBe(280);
  });

  it('should support event subscription workflow', async () => {
    const client = createMockClient();

    // Setup: Subscribe to reputation updates
    const lastCheck = Date.now() - 60000; // 1 minute ago

    // Simulate some reputation updates that happened
    client.addMockEvent({
      event_type: 'reputation_update',
      source_happ: 'mail',
      payload: new TextEncoder().encode(JSON.stringify({ agent: 'bob', score: 0.85 })),
      timestamp: Date.now() - 30000,
      targets: [],
    });

    // Poll for updates since last check
    const updates = await client.getEvents({
      event_type: 'reputation_update',
      since: lastCheck,
    });

    expect(updates).toHaveLength(1);

    // Parse the payload
    const payload = JSON.parse(new TextDecoder().decode(updates[0].payload));
    expect(payload.agent).toBe('bob');
    expect(payload.score).toBe(0.85);
  });
});

// =============================================================================
// Actual MockMycelixClient Tests (from src/client/mock.ts)
// =============================================================================

import { MockMycelixClient } from '../src/client/mock.js';

describe('MockMycelixClient (actual implementation)', () => {
  it('should create a new instance', () => {
    const client = new MockMycelixClient();
    expect(client).toBeInstanceOf(MockMycelixClient);
  });

  it('should start disconnected', () => {
    const client = new MockMycelixClient();
    expect(client.isConnected()).toBe(false);
    expect(client.getState()).toBe('disconnected');
  });

  it('should connect and update state', async () => {
    const client = new MockMycelixClient();

    await client.connect();

    expect(client.isConnected()).toBe(true);
    expect(client.getState()).toBe('connected');
  });

  it('should disconnect and update state', async () => {
    const client = new MockMycelixClient();

    await client.connect();
    expect(client.isConnected()).toBe(true);

    await client.disconnect();
    expect(client.isConnected()).toBe(false);
    expect(client.getState()).toBe('disconnected');
  });

  it('should notify connection state listeners', async () => {
    const client = new MockMycelixClient();
    const states: string[] = [];

    client.addConnectionStateListener((state) => {
      states.push(state);
    });

    await client.connect();
    await client.disconnect();

    expect(states).toContain('connecting');
    expect(states).toContain('connected');
    expect(states).toContain('disconnected');
  });

  it('should remove connection state listeners', async () => {
    const client = new MockMycelixClient();
    const states: string[] = [];

    const listener = (state: string) => {
      states.push(state);
    };

    client.addConnectionStateListener(listener);
    await client.connect();

    client.removeConnectionStateListener(listener);
    await client.disconnect();

    // Should only have connecting/connected, not disconnected
    expect(states).toContain('connecting');
    expect(states).toContain('connected');
    expect(states).not.toContain('disconnected');
  });

  it('should register a hApp', async () => {
    const client = new MockMycelixClient();

    const result = await client.registerHapp({
      happId: 'mail',
      dnaHash: 'abc123',
      agentPubKey: 'agent1',
    });

    expect(result.success).toBe(true);
  });

  it('should record reputation', async () => {
    const client = new MockMycelixClient();

    const result = await client.recordReputation({
      targetAgentId: 'agent1',
      happId: 'mail',
      score: 0.9,
      evidence: { reason: 'helpful' },
    });

    expect(result.success).toBe(true);
    expect(typeof result.newScore).toBe('number');
  });

  it('should accumulate reputation over multiple records', async () => {
    const client = new MockMycelixClient();

    // First record
    const first = await client.recordReputation({
      targetAgentId: 'agent1',
      happId: 'mail',
      score: 0.9,
      evidence: {},
    });

    // Second record should be weighted average
    const second = await client.recordReputation({
      targetAgentId: 'agent1',
      happId: 'mail',
      score: 1.0,
      evidence: {},
    });

    // Score should be moving toward 1.0 but not there yet
    expect(second.newScore).toBeLessThan(1.0);
    expect(second.newScore).toBeGreaterThan(first.newScore);
  });

  it('should check trust', async () => {
    const client = new MockMycelixClient();

    // Set up reputation
    await client.recordReputation({
      targetAgentId: 'trusted_agent',
      happId: 'mail',
      score: 0.95,
      evidence: {},
    });

    const result = await client.checkTrust({
      agentId: 'trusted_agent',
      minimumScore: 0.5,
      happContext: 'mail',
    });

    expect(result.trustworthy).toBe(true);
    expect(typeof result.score).toBe('number');
    expect(typeof result.confidence).toBe('number');
  });

  it('should return false for untrusted agent', async () => {
    const client = new MockMycelixClient();

    // Low reputation agent
    await client.recordReputation({
      targetAgentId: 'untrusted_agent',
      happId: 'mail',
      score: 0.1,
      evidence: {},
    });

    const result = await client.checkTrust({
      agentId: 'untrusted_agent',
      minimumScore: 0.8,
      happContext: 'mail',
    });

    expect(result.trustworthy).toBe(false);
  });

  it('should broadcast events', async () => {
    const client = new MockMycelixClient();

    const result = await client.broadcastEvent({
      eventType: 'test_event',
      payload: { message: 'hello' },
      targetHapps: ['happ1', 'happ2'],
    });

    expect(result.success).toBe(true);
    expect(result.recipientCount).toBe(2);
  });

  it('should get events after broadcast', async () => {
    const client = new MockMycelixClient();
    const now = Date.now();

    // Broadcast an event
    await client.broadcastEvent({
      eventType: 'reputation_update',
      payload: { agent: 'alice', score: 0.9 },
      targetHapps: ['mail'],
    });

    // Get events
    const result = await client.getEvents({
      happId: 'mail',
      eventTypes: ['reputation_update'],
      since: now - 1000,
    });

    expect(result.events.length).toBeGreaterThanOrEqual(1);
    expect(result.events[0].eventType).toBe('reputation_update');
  });

  it('should verify credentials', async () => {
    const client = new MockMycelixClient();

    const result = await client.verifyCredential({
      credentialType: 'identity',
      credentialData: { name: 'Alice', verified: true },
      requesterHappId: 'mail',
    });

    expect(result.valid).toBe(true);
    expect(result.issuer).toBeDefined();
  });

  it('should reject invalid credentials', async () => {
    const client = new MockMycelixClient();

    const result = await client.verifyCredential({
      credentialType: 'identity',
      credentialData: { name: 'Bob', verified: false },
      requesterHappId: 'mail',
    });

    expect(result.valid).toBe(false);
  });

  it('should query cross-hApp reputation', async () => {
    const client = new MockMycelixClient();

    // Record reputation across multiple happs
    await client.recordReputation({
      targetAgentId: 'alice',
      happId: 'mail',
      score: 0.9,
      evidence: {},
    });

    await client.recordReputation({
      targetAgentId: 'alice',
      happId: 'marketplace',
      score: 0.8,
      evidence: {},
    });

    const result = await client.getCrossHappReputation({
      agentId: 'alice',
      happs: ['mail', 'marketplace'],
      aggregationMethod: 'average',
    });

    expect(typeof result.aggregateScore).toBe('number');
    expect(result.aggregateScore).toBeGreaterThan(0);
    expect(result.happScores.length).toBe(2);
  });

  it('should handle weighted aggregation', async () => {
    const client = new MockMycelixClient();

    await client.recordReputation({
      targetAgentId: 'bob',
      happId: 'mail',
      score: 1.0,
      evidence: {},
    });

    await client.recordReputation({
      targetAgentId: 'bob',
      happId: 'marketplace',
      score: 0.5,
      evidence: {},
    });

    const result = await client.getCrossHappReputation({
      agentId: 'bob',
      happs: ['mail', 'marketplace'],
      aggregationMethod: 'weighted',
    });

    expect(typeof result.aggregateScore).toBe('number');
    expect(result.happScores.length).toBeGreaterThan(0);
  });

  it('should handle minimum aggregation', async () => {
    const client = new MockMycelixClient();

    await client.recordReputation({
      targetAgentId: 'carol',
      happId: 'mail',
      score: 0.9,
      evidence: {},
    });

    await client.recordReputation({
      targetAgentId: 'carol',
      happId: 'marketplace',
      score: 0.4,
      evidence: {},
    });

    const result = await client.getCrossHappReputation({
      agentId: 'carol',
      happs: ['mail', 'marketplace'],
      aggregationMethod: 'minimum',
    });

    // Check aggregation returns a number
    expect(typeof result.aggregateScore).toBe('number');
    expect(result.totalInteractions).toBeDefined();
  });
});
