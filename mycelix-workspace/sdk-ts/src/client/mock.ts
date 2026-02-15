/**
 * Mock Mycelix Client
 *
 * Standalone mock client for testing without @holochain/client dependency.
 * This module can be imported without triggering libsodium ESM issues.
 */

import type { HappReputationScore } from '../bridge/index.js';

/** Connection state for event handling */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

/** Connection state change listener */
export type ConnectionStateListener = (state: ConnectionState, error?: Error) => void;

/**
 * Cross-hApp reputation from bridge zome
 */
export interface CrossHappReputation {
  agent: string;
  scores: HappReputationScore[];
  aggregate: number;
  total_interactions: number;
}

/**
 * Bridge event record
 */
export interface BridgeEventRecord {
  event_type: string;
  source_happ: string;
  payload: Uint8Array;
  timestamp: number;
  targets: string[];
}

/**
 * Register hApp input
 */
export interface RegisterHappInput {
  happId: string;
  dnaHash: string;
  agentPubKey: string;
}

/**
 * Record reputation input
 */
export interface RecordReputationInput {
  targetAgentId: string;
  happId: string;
  score: number;
  evidence: Record<string, unknown>;
}

/**
 * Trust check input
 */
export interface TrustCheckInput {
  agentId: string;
  minimumScore: number;
  happContext: string;
}

/**
 * Broadcast event input
 */
export interface BroadcastEventInput {
  eventType: string;
  payload: Record<string, unknown>;
  targetHapps: string[];
}

/**
 * Get events input
 */
export interface GetEventsInput {
  happId: string;
  eventTypes: string[];
  since: number;
}

/**
 * Verify credential input
 */
export interface VerifyCredentialInput {
  credentialType: string;
  credentialData: Record<string, unknown>;
  requesterHappId: string;
}

/**
 * Cross-hApp reputation query input
 */
export interface CrossHappReputationInput {
  agentId: string;
  happs: string[];
  aggregationMethod: 'weighted' | 'average' | 'minimum';
}

/**
 * Mock Mycelix Client for testing
 *
 * Provides a complete mock implementation that simulates Holochain
 * conductor behavior without requiring actual network connectivity.
 */
export class MockMycelixClient {
  private connectionState: ConnectionState = 'disconnected';
  private stateListeners: Set<ConnectionStateListener> = new Set();
  private registeredHapps: Map<string, { happId: string; dnaHash: string; agentPubKey: string }> = new Map();
  private reputations: Map<string, Map<string, number>> = new Map(); // agent -> happ -> score
  private events: Array<{
    eventType: string;
    payload: Record<string, unknown>;
    timestamp: number;
    sourceHapp: string;
    targetHapps: string[];
  }> = [];

  constructor() {
    // Initialize with some default test data
  }

  /**
   * Connect to mock conductor
   */
  async connect(): Promise<void> {
    this.setState('connecting');
    // Simulate async connection
    await new Promise((resolve) => setTimeout(resolve, 10));
    this.setState('connected');
  }

  /**
   * Disconnect from mock conductor
   */
  async disconnect(): Promise<void> {
    this.setState('disconnected');
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connectionState === 'connected';
  }

  /**
   * Get current connection state
   */
  getState(): ConnectionState {
    return this.connectionState;
  }

  private setState(state: ConnectionState, error?: Error): void {
    this.connectionState = state;
    for (const listener of this.stateListeners) {
      try {
        listener(state, error);
      } catch {
        // Ignore listener errors
      }
    }
  }

  /**
   * Add connection state listener
   */
  addConnectionStateListener(listener: ConnectionStateListener): void {
    this.stateListeners.add(listener);
  }

  /**
   * Remove connection state listener
   */
  removeConnectionStateListener(listener: ConnectionStateListener): void {
    this.stateListeners.delete(listener);
  }

  /**
   * Register a hApp
   */
  async registerHapp(input: RegisterHappInput): Promise<{ success: boolean }> {
    this.registeredHapps.set(input.happId, input);
    return { success: true };
  }

  /**
   * Record reputation for an agent
   */
  async recordReputation(input: RecordReputationInput): Promise<{ success: boolean; newScore: number }> {
    if (!this.reputations.has(input.targetAgentId)) {
      this.reputations.set(input.targetAgentId, new Map());
    }

    const agentReps = this.reputations.get(input.targetAgentId)!;
    const currentScore = agentReps.get(input.happId) ?? 0.5;

    // Simple weighted average
    const newScore = currentScore * 0.7 + input.score * 0.3;
    agentReps.set(input.happId, newScore);

    return { success: true, newScore };
  }

  /**
   * Check if an agent is trustworthy
   */
  async checkTrust(input: TrustCheckInput): Promise<{
    trustworthy: boolean;
    score: number;
    confidence: number;
  }> {
    const agentReps = this.reputations.get(input.agentId);
    const score = agentReps?.get(input.happContext) ?? 0.5;
    const interactions = agentReps?.size ?? 0;

    // Confidence based on number of interactions
    const confidence = Math.min(1, interactions * 0.2);

    return {
      trustworthy: score >= input.minimumScore,
      score,
      confidence,
    };
  }

  /**
   * Broadcast event to other hApps
   */
  async broadcastEvent(input: BroadcastEventInput): Promise<{
    success: boolean;
    recipientCount: number;
  }> {
    this.events.push({
      eventType: input.eventType,
      payload: input.payload,
      timestamp: Date.now(),
      sourceHapp: 'mock-source',
      targetHapps: input.targetHapps,
    });

    return {
      success: true,
      recipientCount: input.targetHapps.length,
    };
  }

  /**
   * Get events
   */
  async getEvents(input: GetEventsInput): Promise<{
    events: Array<{
      eventType: string;
      payload: Record<string, unknown>;
      timestamp: number;
      sourceHapp: string;
    }>;
  }> {
    const filtered = this.events.filter(
      (e) =>
        input.eventTypes.includes(e.eventType) &&
        e.timestamp >= input.since &&
        (e.targetHapps.includes(input.happId) || e.targetHapps.length === 0)
    );

    return { events: filtered };
  }

  /**
   * Verify credential
   */
  async verifyCredential(input: VerifyCredentialInput): Promise<{
    valid: boolean;
    issuer?: string;
    expiresAt?: number;
  }> {
    // Mock verification - return valid for any "verified" data
    const valid = input.credentialData?.verified === true;

    return {
      valid,
      issuer: valid ? 'mock-issuer' : undefined,
      expiresAt: valid ? Date.now() + 86400000 : undefined,
    };
  }

  /**
   * Get cross-hApp reputation
   */
  async getCrossHappReputation(input: CrossHappReputationInput): Promise<{
    aggregateScore: number;
    happScores: Array<{ happId: string; score: number; weight: number }>;
    totalInteractions: number;
  }> {
    const agentReps = this.reputations.get(input.agentId);
    const happScores: Array<{ happId: string; score: number; weight: number }> = [];

    let totalScore = 0;
    let totalWeight = 0;

    for (const happId of input.happs) {
      const score = agentReps?.get(happId) ?? 0.5;
      const weight = 1 / input.happs.length;

      happScores.push({ happId, score, weight });
      totalScore += score * weight;
      totalWeight += weight;
    }

    const aggregateScore = totalWeight > 0 ? totalScore / totalWeight : 0.5;

    return {
      aggregateScore,
      happScores,
      totalInteractions: agentReps?.size ?? 0,
    };
  }

  /**
   * Make a generic zome call (mock implementation)
   */
  async zomeCall<T>(params: {
    zomeName: string;
    fnName: string;
    payload: unknown;
  }): Promise<T> {
    // Return mock response based on function name
    switch (params.fnName) {
      case 'get_agent_reputation':
        return { score: 0.75, interactions: 10 } as T;
      default:
        return {} as T;
    }
  }
}

/**
 * Create a mock client for testing
 */
export function createMockClient(): MockMycelixClient {
  return new MockMycelixClient();
}
