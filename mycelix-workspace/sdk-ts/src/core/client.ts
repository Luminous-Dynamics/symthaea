// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Unified Client
 *
 * The main entry point for the Mycelix SDK, providing a unified interface
 * to all Civilizational OS hApps through a single connection.
 *
 * @module @mycelix/sdk/core/client
 *
 * @example
 * ```typescript
 * import { Mycelix } from '@mycelix/sdk/core';
 *
 * // Connect to Holochain conductor
 * const mycelix = await Mycelix.connect({
 *   conductorUrl: 'ws://localhost:8888',
 *   appId: 'mycelix-civilizational-os',
 * });
 *
 * // Access domain-specific clients (lazy-loaded)
 * const myDid = await mycelix.identity.did.createDid();
 * const dao = await mycelix.governance.dao.createDao({ ... });
 * const loan = await mycelix.finance.lending.requestLoan({ ... });
 *
 * // Subscribe to signals across all hApps
 * mycelix.onSignal((signal) => {
 *   console.log('Received signal:', signal);
 * });
 *
 * // Clean up
 * await mycelix.disconnect();
 * ```
 */

import { type AppClient, AppWebsocket, encodeHashToBase64 } from '@holochain/client';

import {
  type SdkError,
  NetworkError,
} from './errors.js';
import { withRetry, type RetryConfig } from './retry.js';

// Import hApp clients (these will be lazily instantiated)
// Note: These imports will be updated as the individual clients are migrated
// to use the core patterns
import type { MycelixEnergyClient } from '../integrations/energy/client.js';
import type { MycelixFinanceClient } from '../integrations/finance/client.js';
import type { MycelixGovernanceClient } from '../integrations/governance/client.js';
import type { MycelixHealthClient } from '../integrations/health/client.js';
import type { MycelixIdentityClient } from '../integrations/identity/client.js';
import type { MycelixKnowledgeClient } from '../integrations/knowledge/client.js';
import type { MycelixPropertyClient } from '../integrations/property/client.js';
import type { MycelixJusticeClient } from '../justice/client.js';
import type { MycelixMediaClient } from '../media/client.js';

/**
 * Configuration for the Mycelix client
 */
export interface MycelixConfig {
  /** WebSocket URL for the Holochain conductor */
  conductorUrl: string;

  /** Installed app ID */
  appId: string;

  /** Timeout for operations in milliseconds (default: 30000) */
  timeout?: number;

  /** Retry configuration for connection and zome calls */
  retryConfig?: Partial<RetryConfig>;

  /** Enable debug logging (default: false) */
  debug?: boolean;
}

/**
 * Default configuration values
 */
const DEFAULT_CONFIG: Required<Omit<MycelixConfig, 'conductorUrl' | 'appId'>> = {
  timeout: 30000,
  retryConfig: {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000,
  },
  debug: false,
};

/**
 * Connection state
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

/**
 * Signal handler callback
 */
export type SignalHandler = (signal: unknown) => void;

/**
 * Connection state change listener
 */
export type ConnectionStateListener = (
  state: ConnectionState,
  error?: SdkError
) => void;

/**
 * Mycelix Client Options when already connected
 */
export interface MycelixClientOptions {
  /** App ID for reference */
  appId: string;

  /** Timeout for operations in milliseconds */
  timeout?: number;

  /** Retry configuration */
  retryConfig?: Partial<RetryConfig>;

  /** Enable debug logging */
  debug?: boolean;
}

/**
 * Mycelix - Unified Client for Civilizational OS
 *
 * Provides a single entry point for all Mycelix hApp interactions,
 * with lazy-loaded domain clients and unified signal handling.
 */
export class Mycelix {
  /** Underlying Holochain client */
  private readonly appClient: AppClient;

  /** Configuration */
  private readonly config: Required<MycelixClientOptions>;

  /** Connection state */
  private connectionState: ConnectionState = 'connected';

  /** State change listeners */
  private readonly stateListeners = new Set<ConnectionStateListener>();

  /** Signal handlers */
  private readonly signalHandlers = new Set<SignalHandler>();

  // Lazy-loaded hApp clients
  private _identity?: MycelixIdentityClient;
  private _governance?: MycelixGovernanceClient;
  private _finance?: MycelixFinanceClient;
  private _property?: MycelixPropertyClient;
  private _knowledge?: MycelixKnowledgeClient;
  private _energy?: MycelixEnergyClient;
  private _media?: MycelixMediaClient;
  private _justice?: MycelixJusticeClient;
  private _health?: MycelixHealthClient;

  /**
   * Private constructor - use static factory methods
   */
  private constructor(client: AppClient, config: Required<MycelixClientOptions>) {
    this.appClient = client;
    this.config = config;

    // Set up signal forwarding if client supports it
    if ('on' in client && typeof (client as any).on === 'function') {
      (client as AppWebsocket).on('signal', (signal: unknown) => {
        this.dispatchSignal(signal);
      });
    }
  }

  // ==========================================================================
  // Static Factory Methods
  // ==========================================================================

  /**
   * Connect to Holochain and create a Mycelix client
   *
   * @param config - Connection configuration
   * @returns Connected Mycelix client
   * @throws SdkError if connection fails
   *
   * @example
   * ```typescript
   * const mycelix = await Mycelix.connect({
   *   conductorUrl: 'ws://localhost:8888',
   *   appId: 'mycelix-civilizational-os',
   *   timeout: 30000,
   * });
   * ```
   */
  static async connect(config: MycelixConfig): Promise<Mycelix> {
    const fullConfig: Required<MycelixConfig> = {
      ...DEFAULT_CONFIG,
      ...config,
    };

    const log = (message: string) => {
      if (fullConfig.debug) {
        console.log(`[mycelix] ${message}`);
      }
    };

    log(`Connecting to ${fullConfig.conductorUrl}...`);

    try {
      const client = await withRetry(
        async () => {
          const ws = await AppWebsocket.connect({
            url: new URL(fullConfig.conductorUrl),
          });

          // Verify connection
          const appInfo = await ws.appInfo();
          if (!appInfo) {
            throw new Error('Failed to get app info');
          }

          log(`Connected to app: ${appInfo.installed_app_id}`);
          return ws as AppClient;
        },
        {
          ...fullConfig.retryConfig,
          onRetry: (attempt, _error, delay) => {
            log(`Connection attempt ${attempt} failed, retrying in ${delay}ms...`);
          },
        }
      );

      return new Mycelix(client, {
        appId: fullConfig.appId,
        timeout: fullConfig.timeout,
        retryConfig: fullConfig.retryConfig,
        debug: fullConfig.debug,
      });
    } catch (error) {
      throw new NetworkError(
        `Failed to connect to Holochain conductor at ${fullConfig.conductorUrl}`,
        { cause: error, url: fullConfig.conductorUrl }
      );
    }
  }

  /**
   * Create a Mycelix client from an existing AppClient
   *
   * Use this when you already have a Holochain connection.
   *
   * @param client - Existing AppClient
   * @param options - Client options
   * @returns Mycelix client
   *
   * @example
   * ```typescript
   * const existingClient = await AppWebsocket.connect({ url: '...' });
   * const mycelix = Mycelix.fromClient(existingClient, {
   *   appId: 'my-app',
   * });
   * ```
   */
  static fromClient(
    client: AppClient,
    options: MycelixClientOptions
  ): Mycelix {
    const fullConfig: Required<MycelixClientOptions> = {
      timeout: options.timeout ?? DEFAULT_CONFIG.timeout,
      retryConfig: options.retryConfig ?? DEFAULT_CONFIG.retryConfig,
      debug: options.debug ?? DEFAULT_CONFIG.debug,
      appId: options.appId,
    };

    return new Mycelix(client, fullConfig);
  }

  // ==========================================================================
  // Domain Clients (Lazy-Loaded)
  // ==========================================================================

  /**
   * Identity hApp client
   *
   * Provides access to DID management, verifiable credentials,
   * and identity federation.
   */
  get identity(): MycelixIdentityClient {
    if (!this._identity) {
      // Dynamically import and create client
      const { MycelixIdentityClient } = require('../integrations/identity/client.js');
      this._identity = MycelixIdentityClient.fromClient(this.appClient, {
        debug: this.config.debug,
        retry: {
          maxAttempts: (this.config.retryConfig.maxRetries ?? 3) + 1,
          delayMs: this.config.retryConfig.baseDelay ?? 1000,
          backoffMultiplier: 2,
        },
      });
    }
    return this._identity!;
  }

  /**
   * Governance hApp client
   *
   * Provides access to DAO management, proposals, voting,
   * and delegation.
   */
  get governance(): MycelixGovernanceClient {
    if (!this._governance) {
      const { MycelixGovernanceClient } = require('../integrations/governance/client.js');
      this._governance = MycelixGovernanceClient.fromClient(this.appClient);
    }
    return this._governance!;
  }

  /**
   * Finance hApp client
   *
   * Provides access to mutual credit, lending, escrow,
   * and treasury management.
   */
  get finance(): MycelixFinanceClient {
    if (!this._finance) {
      const { MycelixFinanceClient } = require('../integrations/finance/client.js');
      this._finance = MycelixFinanceClient.fromClient(this.appClient);
    }
    return this._finance!;
  }

  /**
   * Property hApp client
   *
   * Provides access to property registration, transfers,
   * and title verification.
   */
  get property(): MycelixPropertyClient {
    if (!this._property) {
      const { MycelixPropertyClient } = require('../integrations/property/client.js');
      this._property = MycelixPropertyClient.fromClient(this.appClient);
    }
    return this._property!;
  }

  /**
   * Knowledge hApp client
   *
   * Provides access to knowledge graphs, claims,
   * and fact-checking.
   */
  get knowledge(): MycelixKnowledgeClient {
    if (!this._knowledge) {
      const { MycelixKnowledgeClient } = require('../integrations/knowledge/client.js');
      this._knowledge = MycelixKnowledgeClient.fromClient(this.appClient);
    }
    return this._knowledge!;
  }

  /**
   * Energy hApp client
   *
   * Provides access to energy trading, production tracking,
   * and grid management.
   */
  get energy(): MycelixEnergyClient {
    if (!this._energy) {
      const { MycelixEnergyClient } = require('../integrations/energy/client.js');
      this._energy = MycelixEnergyClient.fromClient(this.appClient);
    }
    return this._energy!;
  }

  /**
   * Media hApp client
   *
   * Provides access to content publishing, licensing,
   * and royalty distribution.
   */
  get media(): MycelixMediaClient {
    if (!this._media) {
      const { MycelixMediaClient } = require('../media/client.js');
      this._media = MycelixMediaClient.fromClient(this.appClient);
    }
    return this._media!;
  }

  /**
   * Justice hApp client
   *
   * Provides access to dispute resolution, arbitration,
   * and enforcement.
   */
  get justice(): MycelixJusticeClient {
    if (!this._justice) {
      const { MycelixJusticeClient } = require('../justice/client.js');
      this._justice = MycelixJusticeClient.fromClient(this.appClient);
    }
    return this._justice!;
  }

  /**
   * Health hApp client
   *
   * Provides access to health records, consent management,
   * and provider verification.
   */
  get health(): MycelixHealthClient {
    if (!this._health) {
      const { MycelixHealthClient } = require('../integrations/health/client.js');
      this._health = MycelixHealthClient.fromClient(this.appClient);
    }
    return this._health!;
  }

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  /**
   * Get current connection state
   */
  getState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connectionState === 'connected';
  }

  /**
   * Get the current agent's public key
   */
  getAgentPubKey(): Uint8Array {
    return this.appClient.myPubKey;
  }

  /**
   * Get the current agent's public key as base64 string
   */
  getAgentPubKeyB64(): string {
    return encodeHashToBase64(this.appClient.myPubKey);
  }

  /**
   * Get the underlying AppClient
   *
   * Use this for advanced operations not covered by domain clients.
   */
  getAppClient(): AppClient {
    return this.appClient;
  }

  /**
   * Subscribe to connection state changes
   *
   * @param listener - Callback for state changes
   * @returns Unsubscribe function
   */
  onStateChange(listener: ConnectionStateListener): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  /**
   * Disconnect from Holochain
   */
  async disconnect(): Promise<void> {
    this.setConnectionState('disconnected');

    if ('client' in this.appClient && (this.appClient as AppWebsocket).client) {
      await (this.appClient as AppWebsocket).client.close();
    }

    // Clear cached clients
    this._identity = undefined;
    this._governance = undefined;
    this._finance = undefined;
    this._property = undefined;
    this._knowledge = undefined;
    this._energy = undefined;
    this._media = undefined;
    this._justice = undefined;
    this._health = undefined;
  }

  // ==========================================================================
  // Signal Handling
  // ==========================================================================

  /**
   * Subscribe to signals from all hApps
   *
   * @param handler - Signal handler callback
   * @returns Unsubscribe function
   *
   * @example
   * ```typescript
   * const unsubscribe = mycelix.onSignal((signal) => {
   *   console.log('Received signal:', signal);
   * });
   *
   * // Later: stop listening
   * unsubscribe();
   * ```
   */
  onSignal(handler: SignalHandler): () => void {
    this.signalHandlers.add(handler);
    return () => this.signalHandlers.delete(handler);
  }

  // ==========================================================================
  // Cross-hApp Operations
  // ==========================================================================

  /**
   * Get the current agent's DID (if one exists)
   *
   * Convenience method that resolves the DID for the connected agent.
   *
   * @returns DID string or null if none exists
   */
  async getMyDid(): Promise<string | null> {
    try {
      return await this.identity.getMyDid();
    } catch {
      return null;
    }
  }

  /**
   * Check health of all connected hApps
   *
   * @returns Object with health status for each hApp
   */
  async checkHealth(): Promise<Record<string, { ok: boolean; latencyMs?: number; error?: string }>> {
    const results: Record<string, { ok: boolean; latencyMs?: number; error?: string }> = {};

    // Check identity hApp
    try {
      const start = Date.now();
      await this.appClient.appInfo();
      results.conductor = { ok: true, latencyMs: Date.now() - start };
    } catch (error) {
      results.conductor = {
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }

    return results;
  }

  // ==========================================================================
  // Private Helpers
  // ==========================================================================

  private setConnectionState(state: ConnectionState, error?: SdkError): void {
    this.connectionState = state;
    Array.from(this.stateListeners).forEach((listener) => {
      try {
        listener(state, error);
      } catch {
        // Don't let listener errors break state management
      }
    });
  }

  private dispatchSignal(signal: unknown): void {
    Array.from(this.signalHandlers).forEach((handler) => {
      try {
        handler(signal);
      } catch {
        // Don't let handler errors break signal dispatch
      }
    });
  }
}

// Re-export for convenience
export type { MycelixIdentityClient } from '../integrations/identity/client.js';
export type { MycelixGovernanceClient } from '../integrations/governance/client.js';
