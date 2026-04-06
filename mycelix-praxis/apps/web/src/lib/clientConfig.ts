// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Client configuration for switching between mock and real Holochain clients
 *
 * This module provides a unified interface for accessing either the mock client
 * (for development/testing) or the real Holochain client (for production).
 */

import { HolochainClient, createHolochainClient, type HolochainClientConfig } from './holochainClient';
import { MockHolochainClient, getMockClient, type MockHolochainClientConfig } from '../services/mockHolochainClient';

// Type alias for backwards compatibility
export type MockClientOptions = MockHolochainClientConfig;

/**
 * Client mode: mock or real
 */
export type ClientMode = 'mock' | 'real';

/**
 * Combined client configuration
 */
export interface ClientConfig {
  /** Client mode (default: determined by environment) */
  mode?: ClientMode;

  /** Configuration for real Holochain client */
  holochain?: HolochainClientConfig;

  /** Configuration for mock client */
  mock?: MockClientOptions;
}

/**
 * Unified client interface
 *
 * This wraps either a HolochainClient or MockHolochainClient and provides
 * a unified interface for both.
 */
export class UnifiedClient {
  private client: HolochainClient | MockHolochainClient;
  private mode: ClientMode;

  constructor(config: ClientConfig = {}) {
    // Determine mode from config or environment
    this.mode = this.determineMode(config);

    // Create appropriate client
    if (this.mode === 'real') {
      this.client = createHolochainClient(config.holochain);
      console.log('[UnifiedClient] Using REAL Holochain client');
    } else {
      this.client = getMockClient(config.mock);
      console.log('[UnifiedClient] Using MOCK Holochain client');
    }
  }

  /**
   * Determine client mode from config and environment
   */
  private determineMode(config: ClientConfig): ClientMode {
    // Explicit mode in config takes priority
    if (config.mode) {
      return config.mode;
    }

    // Check environment variables
    if (import.meta.env.VITE_USE_REAL_CLIENT === 'true') {
      return 'real';
    }

    if (import.meta.env.VITE_USE_MOCK_CLIENT === 'true') {
      return 'mock';
    }

    // Default based on NODE_ENV
    if (import.meta.env.PROD) {
      return 'real';
    }

    return 'mock'; // Development default
  }

  /**
   * Get current client mode
   */
  getMode(): ClientMode {
    return this.mode;
  }

  /**
   * Check if using real client
   */
  isReal(): boolean {
    return this.mode === 'real';
  }

  /**
   * Check if using mock client
   */
  isMock(): boolean {
    return this.mode === 'mock';
  }

  /**
   * Get the underlying client
   */
  getClient(): HolochainClient | MockHolochainClient {
    return this.client;
  }

  // =============================================================================
  // Connection Management (unified interface)
  // =============================================================================

  async connect(): Promise<void> {
    return this.client.connect();
  }

  async disconnect(): Promise<void> {
    return this.client.disconnect();
  }

  isConnected(): boolean {
    if (this.client instanceof HolochainClient) {
      return this.client.isConnected();
    }
    // MockHolochainClient is always "connected"
    return true;
  }

  // =============================================================================
  // Zome Functions (proxied from underlying client)
  // =============================================================================

  get learning() {
    return (this.client as HolochainClient).learning;
  }

  get fl() {
    return (this.client as HolochainClient).fl;
  }

  get credential() {
    return (this.client as HolochainClient).credential;
  }

  get dao() {
    return (this.client as HolochainClient).dao;
  }

  get srs() {
    return (this.client as HolochainClient).srs;
  }

  get gamification() {
    return (this.client as HolochainClient).gamification;
  }

  get adaptive() {
    return (this.client as HolochainClient).adaptive;
  }

  get integration() {
    return (this.client as HolochainClient).integration;
  }
}

// =============================================================================
// Client Factory Functions
// =============================================================================

/**
 * Create a unified client (mock or real based on config/environment)
 */
export function createClient(config?: ClientConfig): UnifiedClient {
  return new UnifiedClient(config);
}

/**
 * Create a mock client (force mock mode)
 */
export function createMockClient(options?: MockClientOptions): UnifiedClient {
  return new UnifiedClient({
    mode: 'mock',
    mock: options,
  });
}

/**
 * Create a real Holochain client (force real mode)
 */
export function createRealClient(config?: HolochainClientConfig): UnifiedClient {
  return new UnifiedClient({
    mode: 'real',
    holochain: config,
  });
}

// =============================================================================
// Global Client Instance (Singleton)
// =============================================================================

let globalClient: UnifiedClient | null = null;

/**
 * Get the global client instance (creates if doesn't exist)
 */
export function getGlobalClient(config?: ClientConfig): UnifiedClient {
  if (!globalClient) {
    globalClient = createClient(config);
  }
  return globalClient;
}

/**
 * Set the global client instance
 */
export function setGlobalClient(client: UnifiedClient): void {
  globalClient = client;
}

/**
 * Reset the global client (disconnect and clear)
 */
export async function resetGlobalClient(): Promise<void> {
  if (globalClient) {
    await globalClient.disconnect();
    globalClient = null;
  }
}

// =============================================================================
// Environment Configuration Helper
// =============================================================================

/**
 * Get client configuration from environment variables
 */
export function getEnvConfig(): ClientConfig {
  const mode = import.meta.env.VITE_USE_REAL_CLIENT === 'true' ? 'real' : 'mock';

  return {
    mode,
    holochain: {
      appWsUrl: import.meta.env.VITE_APP_WS_URL || 'ws://localhost:8888',
      appId: import.meta.env.VITE_APP_ID || 'mycelix-praxis',
      roleName: import.meta.env.VITE_ROLE_NAME || 'praxis',
      verbose: import.meta.env.VITE_VERBOSE_LOGS === 'true',
      autoReconnect: import.meta.env.VITE_AUTO_RECONNECT !== 'false',
    },
    mock: {
      verbose: import.meta.env.VITE_VERBOSE_LOGS === 'true',
    },
  };
}

// =============================================================================
// React Hook (for convenience)
// =============================================================================

import { useState, useEffect } from 'react';

/**
 * React hook for using the Holochain client
 *
 * @example
 * function MyComponent() {
 *   const { client, connected, mode } = useHolochainClient();
 *
 *   useEffect(() => {
 *     if (connected) {
 *       client.learning.get_all_courses().then(courses => {
 *         console.log('Courses:', courses);
 *       });
 *     }
 *   }, [connected]);
 *
 *   return <div>Mode: {mode}, Connected: {connected}</div>;
 * }
 */
export function useHolochainClient(config?: ClientConfig) {
  const [client] = useState(() => getGlobalClient(config));
  const [connected, setConnected] = useState(false);
  const [mode] = useState(() => client.getMode());

  useEffect(() => {
    // Connect on mount
    client.connect().then(() => {
      setConnected(true);
    }).catch((error) => {
      console.error('[useHolochainClient] Connection failed:', error);
      setConnected(false);
    });

    // Set up status change listener (only for real client)
    let unsubscribe: (() => void) | undefined;
    const underlyingClient = client.getClient();
    if (underlyingClient instanceof HolochainClient) {
      unsubscribe = underlyingClient.onStatusChange((status) => {
        setConnected(status === 'connected');
      });
    }

    // Cleanup
    return () => {
      if (unsubscribe) {
        unsubscribe();
      }
    };
  }, [client]);

  return {
    client,
    connected,
    mode,
    isReal: mode === 'real',
    isMock: mode === 'mock',
  };
}

/**
 * Default export
 */
export default UnifiedClient;
