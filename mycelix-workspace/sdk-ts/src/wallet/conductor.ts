// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Conductor Connection Manager - The Spine
 *
 * Unified connection management for Holochain conductor with:
 * - Auto-reconnection with exponential backoff
 * - Connection state observability
 * - Agent context (signing)
 * - Signal handling
 * - Graceful degradation
 *
 * @example
 * ```typescript
 * const conductor = new ConductorManager({
 *   appUrl: 'ws://localhost:8888',
 *   installedAppId: 'mycelix-wallet',
 * });
 *
 * await conductor.connect();
 *
 * // Use the client for zome calls
 * const profile = await conductor.callZome({
 *   zome_name: 'profiles',
 *   fn_name: 'get_profile',
 *   payload: agentId,
 * });
 * ```
 */

import {
  type MycelixClient,
  createClient,
  type MycelixClientConfig,
  type ConnectionState,
  type ZomeCallParams,
} from '../client/index.js';
import { BehaviorSubject } from '../reactive/index.js';

// =============================================================================
// Types
// =============================================================================

/** Extended connection state with more detail */
export type ConductorState = {
  connection: ConnectionState;
  isOnline: boolean;
  lastConnectedAt: number | null;
  lastErrorAt: number | null;
  lastError: string | null;
  reconnectAttempts: number;
  agentPubKey: Uint8Array | null;
  installedAppId: string;
};

/** Conductor manager configuration */
export interface ConductorConfig extends MycelixClientConfig {
  /** Enable auto-reconnection (default: true) */
  autoReconnect?: boolean;
  /** Maximum reconnection attempts before giving up (default: 10, 0 = unlimited) */
  maxReconnectAttempts?: number;
  /** Initial reconnection delay in ms (default: 1000) */
  reconnectDelay?: number;
  /** Maximum reconnection delay in ms (default: 30000) */
  maxReconnectDelay?: number;
  /** Reconnect delay multiplier (default: 2) */
  reconnectBackoff?: number;
  /** Ping interval to detect disconnection in ms (default: 30000) */
  pingInterval?: number;
}

/** Signal from the conductor */
export interface ConductorSignal {
  type: string;
  zome_name: string;
  payload: unknown;
}

/** Signal listener callback */
export type SignalListener = (signal: ConductorSignal) => void;

// =============================================================================
// Conductor Manager
// =============================================================================

/**
 * Manages the WebSocket connection to the Holochain conductor.
 * Provides auto-reconnection, state observability, and a clean API for zome calls.
 */
export class ConductorManager {
  private client: MycelixClient;
  private config: Required<ConductorConfig>;
  private _state$: BehaviorSubject<ConductorState>;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private signalListeners: Set<SignalListener> = new Set();
  private isDestroyed = false;

  constructor(config: ConductorConfig) {
    this.config = {
      appUrl: config.appUrl ?? 'ws://localhost:8888',
      adminUrl: config.adminUrl ?? 'ws://localhost:8889',
      timeout: config.timeout ?? 30000,
      installedAppId: config.installedAppId,
      maxRetries: config.maxRetries ?? 3,
      retryDelay: config.retryDelay ?? 1000,
      retryBackoff: config.retryBackoff ?? 2,
      autoReconnect: config.autoReconnect ?? true,
      maxReconnectAttempts: config.maxReconnectAttempts ?? 10,
      reconnectDelay: config.reconnectDelay ?? 1000,
      maxReconnectDelay: config.maxReconnectDelay ?? 30000,
      reconnectBackoff: config.reconnectBackoff ?? 2,
      pingInterval: config.pingInterval ?? 30000,
    };

    this.client = createClient(this.config);

    this._state$ = new BehaviorSubject<ConductorState>({
      connection: 'disconnected',
      isOnline: typeof navigator !== 'undefined' ? navigator.onLine : true,
      lastConnectedAt: null,
      lastErrorAt: null,
      lastError: null,
      reconnectAttempts: 0,
      agentPubKey: null,
      installedAppId: this.config.installedAppId,
    });

    // Listen for connection state changes from the client
    this.client.onStateChange((state, error) => {
      this.handleConnectionStateChange(state, error);
    });

    // Listen for signals from the conductor
    this.client.onSignal((signal) => {
      this.dispatchSignal(signal as ConductorSignal);
    });

    // Monitor browser online/offline status
    if (typeof window !== 'undefined') {
      window.addEventListener('online', () => this.handleOnlineStatusChange(true));
      window.addEventListener('offline', () => this.handleOnlineStatusChange(false));
    }
  }

  // ===========================================================================
  // Reactive State
  // ===========================================================================

  /** Observable connection state */
  get state$(): BehaviorSubject<ConductorState> {
    return this._state$;
  }

  /** Current state snapshot */
  get state(): ConductorState {
    return this._state$.value;
  }

  /** Is currently connected */
  get isConnected(): boolean {
    return this._state$.value.connection === 'connected';
  }

  /** Is browser online */
  get isOnline(): boolean {
    return this._state$.value.isOnline;
  }

  /** Agent public key (after connection) */
  get agentPubKey(): Uint8Array | null {
    return this._state$.value.agentPubKey;
  }

  // ===========================================================================
  // Connection Management
  // ===========================================================================

  /**
   * Connect to the conductor.
   * Will automatically retry according to configuration.
   */
  async connect(): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('ConductorManager has been destroyed');
    }

    if (this.isConnected) {
      return;
    }

    // Reset reconnect attempts on explicit connect
    this.updateState({ reconnectAttempts: 0 });

    await this.attemptConnection();
  }

  /**
   * Disconnect from the conductor.
   * Stops auto-reconnection attempts.
   */
  async disconnect(): Promise<void> {
    this.clearReconnectTimer();
    this.clearPingTimer();

    await this.client.disconnect();

    this.updateState({
      connection: 'disconnected',
      agentPubKey: null,
    });
  }

  /**
   * Destroy the conductor manager.
   * Disconnects and cleans up all resources.
   */
  async destroy(): Promise<void> {
    this.isDestroyed = true;
    await this.disconnect();

    // Remove browser event listeners
    if (typeof window !== 'undefined') {
      window.removeEventListener('online', () => this.handleOnlineStatusChange(true));
      window.removeEventListener('offline', () => this.handleOnlineStatusChange(false));
    }

    this.signalListeners.clear();
  }

  // ===========================================================================
  // Zome Calls
  // ===========================================================================

  /**
   * Make a zome call to the conductor.
   *
   * @throws ConnectionError if not connected
   * @throws Error if the zome call fails
   */
  async callZome<T>(params: Omit<ZomeCallParams, 'role_name'> & { role_name?: string }): Promise<T> {
    if (!this.isConnected) {
      throw new Error('Not connected to conductor. Call connect() first.');
    }

    return this.client.callZome<T>({
      role_name: params.role_name ?? this.config.installedAppId,
      zome_name: params.zome_name,
      fn_name: params.fn_name,
      payload: params.payload,
    });
  }

  /**
   * Make a zome call with automatic retry on transient failures.
   */
  async callZomeWithRetry<T>(
    params: Omit<ZomeCallParams, 'role_name'> & { role_name?: string },
    maxRetries = 3
  ): Promise<T> {
    let lastError: Error | undefined;
    let delay = 500;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await this.callZome<T>(params);
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Don't retry on certain errors
        if (lastError.message.includes('not found') || lastError.message.includes('unauthorized')) {
          throw lastError;
        }

        if (attempt < maxRetries) {
          await new Promise((resolve) => setTimeout(resolve, delay));
          delay *= 2;
        }
      }
    }

    throw lastError;
  }

  // ===========================================================================
  // Signal Handling
  // ===========================================================================

  /**
   * Subscribe to conductor signals.
   * Returns unsubscribe function.
   */
  onSignal(listener: SignalListener): () => void {
    this.signalListeners.add(listener);
    return () => this.signalListeners.delete(listener);
  }

  private dispatchSignal(signal: ConductorSignal): void {
    for (const listener of this.signalListeners) {
      try {
        listener(signal);
      } catch {
        // Don't let listener errors break signal handling
      }
    }
  }

  // ===========================================================================
  // Internal Connection Logic
  // ===========================================================================

  private async attemptConnection(): Promise<void> {
    this.updateState({ connection: 'connecting' });

    try {
      await this.client.connect();

      // Connection successful
      this.updateState({
        connection: 'connected',
        lastConnectedAt: Date.now(),
        lastError: null,
        reconnectAttempts: 0,
      });

      // Start ping timer to detect disconnection
      this.startPingTimer();
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Connection failed';

      this.updateState({
        connection: 'error',
        lastErrorAt: Date.now(),
        lastError: errorMsg,
      });

      // Schedule reconnection if enabled
      if (this.config.autoReconnect && !this.isDestroyed) {
        this.scheduleReconnect();
      }

      throw error;
    }
  }

  private handleConnectionStateChange(state: ConnectionState, error?: Error): void {
    if (state === 'error' || state === 'disconnected') {
      const currentState = this._state$.value;

      if (currentState.connection === 'connected') {
        // We were connected and lost connection
        this.updateState({
          connection: state,
          lastErrorAt: Date.now(),
          lastError: error?.message ?? 'Connection lost',
          agentPubKey: null,
        });

        this.clearPingTimer();

        // Attempt to reconnect if enabled
        if (this.config.autoReconnect && !this.isDestroyed) {
          this.scheduleReconnect();
        }
      }
    }
  }

  private handleOnlineStatusChange(isOnline: boolean): void {
    this.updateState({ isOnline });

    if (isOnline && !this.isConnected && this.config.autoReconnect) {
      // Back online, try to reconnect
      this.updateState({ reconnectAttempts: 0 });
      this.scheduleReconnect(0); // Immediate reconnect
    }
  }

  private scheduleReconnect(delayOverride?: number): void {
    this.clearReconnectTimer();

    const currentAttempts = this._state$.value.reconnectAttempts;

    // Check if we've exceeded max attempts (0 = unlimited)
    if (this.config.maxReconnectAttempts > 0 && currentAttempts >= this.config.maxReconnectAttempts) {
      console.warn(`ConductorManager: Max reconnect attempts (${this.config.maxReconnectAttempts}) reached`);
      return;
    }

    // Calculate delay with exponential backoff
    const delay =
      delayOverride ??
      Math.min(
        this.config.reconnectDelay * Math.pow(this.config.reconnectBackoff, currentAttempts),
        this.config.maxReconnectDelay
      );

    this.reconnectTimer = setTimeout(() => {
      void (async () => {
        if (this.isDestroyed) return;

        this.updateState({ reconnectAttempts: currentAttempts + 1 });

        try {
          await this.attemptConnection();
        } catch {
          // attemptConnection will schedule next reconnect if needed
        }
      })();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private startPingTimer(): void {
    this.clearPingTimer();

    this.pingTimer = setInterval(() => {
      void (async () => {
        if (!this.isConnected || this.isDestroyed) {
          this.clearPingTimer();
          return;
        }

        try {
          // Simple ping to check connection is alive
          // This is a lightweight call that most conductors support
          await this.client.callZome({
            role_name: this.config.installedAppId,
            zome_name: 'ping',
            fn_name: 'ping',
            payload: null,
          });
        } catch {
          // Ping failed - connection might be dead
          // The next zome call will trigger reconnection
        }
      })();
    }, this.config.pingInterval);
  }

  private clearPingTimer(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private updateState(partial: Partial<ConductorState>): void {
    this._state$.next({ ...this._state$.value, ...partial });
  }

  // ===========================================================================
  // ZomeCallable Interface (for use with IdentityResolver, etc.)
  // ===========================================================================

  /**
   * Get the underlying client for direct usage.
   * Note: Prefer using callZome() for most use cases.
   */
  getClient(): MycelixClient {
    return this.client;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a conductor manager with default configuration.
 */
export function createConductorManager(config: ConductorConfig): ConductorManager {
  return new ConductorManager(config);
}

/**
 * Create a conductor manager configured for development.
 */
export function createDevConductorManager(installedAppId: string): ConductorManager {
  return new ConductorManager({
    installedAppId,
    appUrl: 'ws://localhost:8888',
    adminUrl: 'ws://localhost:8889',
    autoReconnect: true,
    maxReconnectAttempts: 0, // Unlimited for dev
    timeout: 60000, // Longer timeout for dev
  });
}

/**
 * Create a conductor manager configured for production.
 */
export function createProdConductorManager(
  installedAppId: string,
  appUrl: string
): ConductorManager {
  return new ConductorManager({
    installedAppId,
    appUrl,
    autoReconnect: true,
    maxReconnectAttempts: 10,
    timeout: 30000,
  });
}
