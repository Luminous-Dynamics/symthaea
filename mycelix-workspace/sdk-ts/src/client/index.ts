// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Client Integration
 *
 * WebSocket-based connection to Holochain conductor for real hApp communication.
 */

import { type AppClient, AppWebsocket, AdminWebsocket } from '@holochain/client';

import type { HappReputationScore } from '../bridge/index.js';

import { ConnectionError, ErrorCode, validate } from '../errors.js';

/** Connection state for event handling */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

/** Connection state change listener */
export type ConnectionStateListener = (state: ConnectionState, error?: Error) => void;

/**
 * Connection configuration
 */
export interface MycelixClientConfig {
  /** Conductor app WebSocket URL (default: ws://localhost:8888) */
  appUrl?: string;
  /** Conductor admin WebSocket URL (default: ws://localhost:8889) */
  adminUrl?: string;
  /** Timeout for WebSocket operations in ms (default: 30000) */
  timeout?: number;
  /** Installed app ID */
  installedAppId: string;
  /** Maximum retry attempts for connection (default: 3) */
  maxRetries?: number;
  /** Initial delay between retries in ms (default: 1000) */
  retryDelay?: number;
  /** Backoff multiplier for retry delay (default: 2) */
  retryBackoff?: number;
}

/**
 * Zome call parameters
 */
export interface ZomeCallParams {
  role_name: string;
  zome_name: string;
  fn_name: string;
  payload: unknown;
}

/**
 * Bridge zome call input types
 */
export interface RegisterHappInput {
  happ_id: string;
  happ_name: string;
  capabilities: string[];
}

export interface RecordReputationInput {
  agent: string;
  happ_id: string;
  happ_name: string;
  score: number;
  interactions: number;
}

export interface TrustCheckInput {
  agent: string;
  threshold: number;
}

export interface BroadcastEventInput {
  event_type: string;
  payload: Uint8Array;
  targets: string[];
}

export interface GetEventsInput {
  event_type: string;
  since: number;
}

export interface CredentialVerifyInput {
  credential_hash: string;
  issuer_happ: string;
}

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
 * Mycelix Client - Connects to Holochain conductor
 */
/** Signal handler callback type */
export type SignalCallback = (signal: unknown) => void;

export class MycelixClient {
  private config: Required<Omit<MycelixClientConfig, 'installedAppId'>> & { installedAppId: string };
  private appClient: AppClient | null = null;
  private adminClient: AdminWebsocket | null = null;
  private connectionState: ConnectionState = 'disconnected';
  private stateListeners: Set<ConnectionStateListener> = new Set();
  private signalListeners: Set<SignalCallback> = new Set();

  constructor(config: MycelixClientConfig) {
    validate().notEmpty('installedAppId', config.installedAppId).throwIfInvalid();

    this.config = {
      appUrl: config.appUrl ?? 'ws://localhost:8888',
      adminUrl: config.adminUrl ?? 'ws://localhost:8889',
      timeout: config.timeout ?? 30000,
      installedAppId: config.installedAppId,
      maxRetries: config.maxRetries ?? 3,
      retryDelay: config.retryDelay ?? 1000,
      retryBackoff: config.retryBackoff ?? 2,
    };
  }

  /**
   * Add a connection state listener
   */
  onStateChange(listener: ConnectionStateListener): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  /**
   * Subscribe to Holochain signals from the conductor.
   * Returns an unsubscribe function.
   *
   * @example
   * ```typescript
   * const unsubscribe = client.onSignal((signal) => {
   *   console.log('Received signal:', signal);
   * });
   *
   * // Later, to stop listening:
   * unsubscribe();
   * ```
   */
  onSignal(callback: SignalCallback): () => void {
    this.signalListeners.add(callback);
    return () => this.signalListeners.delete(callback);
  }

  /**
   * Dispatch a signal to all listeners
   */
  private dispatchSignal(signal: unknown): void {
    for (const listener of this.signalListeners) {
      try {
        listener(signal);
      } catch {
        // Don't let listener errors break signal handling
      }
    }
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
        // Don't let listener errors break state management
      }
    }
  }

  /**
   * Connect to the Holochain conductor with retry logic
   */
  async connect(): Promise<void> {
    if (this.connectionState === 'connected') return;

    this.setState('connecting');
    let lastError: Error | undefined;
    let delay = this.config.retryDelay;

    for (let attempt = 1; attempt <= this.config.maxRetries; attempt++) {
      try {
        this.appClient = await AppWebsocket.connect({
          url: new URL(this.config.appUrl),
        });

        // Set up signal handler
        const wsClient = this.appClient as AppWebsocket;
        wsClient.on('signal', (signal) => {
          this.dispatchSignal(signal);
        });

        this.setState('connected');
        return;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (attempt < this.config.maxRetries) {
          await new Promise((resolve) => setTimeout(resolve, delay));
          delay *= this.config.retryBackoff;
        }
      }
    }

    this.setState('error', lastError);
    throw new ConnectionError(
      `Failed to connect to Holochain conductor after ${this.config.maxRetries} attempts`,
      ErrorCode.TIMEOUT,
      this.config.appUrl,
      true,
      lastError
    );
  }

  /**
   * Connect to admin interface (for conductor management)
   */
  async connectAdmin(): Promise<void> {
    try {
      this.adminClient = await AdminWebsocket.connect({
        url: new URL(this.config.adminUrl),
      });
    } catch (error) {
      throw new ConnectionError(
        'Failed to connect to admin interface',
        ErrorCode.TIMEOUT,
        this.config.adminUrl,
        true,
        error instanceof Error ? error : undefined
      );
    }
  }

  /**
   * Disconnect from the conductor
   */
  async disconnect(): Promise<void> {
    if (this.appClient) {
      await (this.appClient as AppWebsocket).client.close();
      this.appClient = null;
    }
    if (this.adminClient) {
      await this.adminClient.client.close();
      this.adminClient = null;
    }
    this.setState('disconnected');
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connectionState === 'connected';
  }

  /**
   * Make a zome call with timeout
   */
  async callZome<T>(params: ZomeCallParams): Promise<T> {
    if (!this.appClient) {
      throw new ConnectionError(
        'Not connected to Holochain conductor',
        ErrorCode.TIMEOUT,
        this.config.appUrl,
        true
      );
    }

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(new ConnectionError(
          `Zome call timed out after ${this.config.timeout}ms`,
          ErrorCode.TIMEOUT,
          this.config.appUrl,
          true
        ));
      }, this.config.timeout);
    });

    const callPromise = this.appClient.callZome({
      cap_secret: undefined,
      role_name: params.role_name,
      zome_name: params.zome_name,
      fn_name: params.fn_name,
      payload: params.payload,
    });

    const result = await Promise.race([callPromise, timeoutPromise]);
    return result as T;
  }

  // =========================================================================
  // Bridge Zome Methods
  // =========================================================================

  /**
   * Register a hApp with the bridge
   */
  async registerHapp(input: RegisterHappInput): Promise<Uint8Array> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'register_happ',
      payload: input,
    });
  }

  /**
   * Get all registered hApps
   */
  async getRegisteredHapps(): Promise<Array<{ happ_id: string; happ_name: string; capabilities: string[] }>> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'get_registered_happs',
      payload: null,
    });
  }

  /**
   * Record reputation for an agent
   */
  async recordReputation(input: RecordReputationInput): Promise<Uint8Array> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'record_reputation',
      payload: input,
    });
  }

  /**
   * Query cross-hApp reputation for an agent
   */
  async queryCrossHappReputation(agent: string): Promise<CrossHappReputation> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'query_cross_happ_reputation',
      payload: agent,
    });
  }

  /**
   * Check if an agent is trustworthy
   */
  async isAgentTrustworthy(input: TrustCheckInput): Promise<boolean> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'is_agent_trustworthy',
      payload: input,
    });
  }

  /**
   * Broadcast an event to other hApps
   */
  async broadcastEvent(input: BroadcastEventInput): Promise<Uint8Array> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'broadcast_event',
      payload: input,
    });
  }

  /**
   * Get events of a specific type
   */
  async getEvents(input: GetEventsInput): Promise<BridgeEventRecord[]> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'get_events',
      payload: input,
    });
  }

  /**
   * Request credential verification
   */
  async requestCredentialVerification(input: CredentialVerifyInput): Promise<Uint8Array> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'request_credential_verification',
      payload: input,
    });
  }

  /**
   * Get pending credential verification requests
   */
  async getCredentialVerificationRequests(happId: string): Promise<Array<{
    credential_hash: string;
    issuer_happ: string;
    requester: string;
    requested_at: number;
    status: string;
  }>> {
    return this.callZome({
      role_name: 'bridge',
      zome_name: 'bridge_coordinator',
      fn_name: 'get_credential_verification_requests',
      payload: happId,
    });
  }
}

/**
 * Create a Mycelix client instance
 */
export function createClient(config: MycelixClientConfig): MycelixClient {
  return new MycelixClient(config);
}

/**
 * Mock client for testing without a real conductor
 */
export class MockMycelixClient extends MycelixClient {
  private mockReputations: Map<string, CrossHappReputation> = new Map();
  private mockEvents: BridgeEventRecord[] = [];
  private mockState: ConnectionState = 'connected';

  constructor(installedAppId = 'mock-app') {
    super({ installedAppId });
  }

  override async connect(): Promise<void> {
    this.mockState = 'connected';
  }

  override async disconnect(): Promise<void> {
    this.mockState = 'disconnected';
  }

  override isConnected(): boolean {
    return this.mockState === 'connected';
  }

  override getState(): ConnectionState {
    return this.mockState;
  }

  /**
   * Set mock reputation for testing
   */
  setMockReputation(agent: string, reputation: CrossHappReputation): void {
    this.mockReputations.set(agent, reputation);
  }

  /**
   * Add mock event for testing
   */
  addMockEvent(event: BridgeEventRecord): void {
    this.mockEvents.push(event);
  }

  override async queryCrossHappReputation(agent: string): Promise<CrossHappReputation> {
    return this.mockReputations.get(agent) ?? {
      agent,
      scores: [],
      aggregate: 0.5,
      total_interactions: 0,
    };
  }

  override async isAgentTrustworthy(input: TrustCheckInput): Promise<boolean> {
    const rep = await this.queryCrossHappReputation(input.agent);
    return rep.aggregate >= input.threshold;
  }

  override async getEvents(input: GetEventsInput): Promise<BridgeEventRecord[]> {
    return this.mockEvents.filter(
      (e) => e.event_type === input.event_type && e.timestamp >= input.since
    );
  }

  override async broadcastEvent(input: BroadcastEventInput): Promise<Uint8Array> {
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

/**
 * Create a mock client for testing
 */
export function createMockClient(): MockMycelixClient {
  return new MockMycelixClient();
}

// ============================================================================
// Unified Ecosystem Client
// ============================================================================

import {
  MycelixEcosystemClient as _MycelixEcosystemClient,
  MycelixEcosystemClientError as _MycelixClientError,
} from './unified';

import type {
  MycelixClientConfig as _EcosystemClientConfig,
  RetryConfig as _RetryConfig,
  DomainStatus as _DomainStatus,
} from './unified';

/**
 * Unified Ecosystem Client - Access all 9 Mycelix domain SDKs
 *
 * This is the recommended entry point for applications that need to interact
 * with multiple Mycelix hApps (Identity, Governance, Finance, Property,
 * Energy, Knowledge, Health, Media, Justice).
 *
 * @see MycelixEcosystemClient for comprehensive documentation
 */
export const MycelixEcosystemClient = _MycelixEcosystemClient;
export const EcosystemClientError = _MycelixClientError;
export type EcosystemClientConfig = _EcosystemClientConfig;
export type RetryConfig = _RetryConfig;
export type DomainStatus = _DomainStatus;

/**
 * Create a unified ecosystem client
 *
 * @example
 * ```typescript
 * const mycelix = await createEcosystemClient({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Access all domains
 * const myDid = await mycelix.identity.did.createDid();
 * const proposals = await mycelix.governance.proposals.getActiveProposals('dao-1');
 * const patient = await mycelix.health.patient.getPatient(hash);
 * ```
 */
export async function createEcosystemClient(
  config: EcosystemClientConfig
): Promise<_MycelixEcosystemClient> {
  return _MycelixEcosystemClient.connect(config);
}
