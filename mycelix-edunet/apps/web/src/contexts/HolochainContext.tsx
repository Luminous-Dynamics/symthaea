// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Context Provider
 *
 * Provides a unified Holochain client to all components in the app.
 * Handles connection management, reconnection, and status updates.
 */

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import {
  UnifiedClient,
  createClient,
  ClientMode,
  ClientConfig,
  getEnvConfig,
} from '../lib/clientConfig';

// =============================================================================
// Types
// =============================================================================

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface HolochainContextValue {
  /** The unified Holochain client */
  client: UnifiedClient | null;

  /** Current connection status */
  status: ConnectionStatus;

  /** Client mode (mock or real) */
  mode: ClientMode;

  /** Error message if connection failed */
  error: string | null;

  /** Whether using mock data */
  isMock: boolean;

  /** Whether using real Holochain */
  isReal: boolean;

  /** Manually trigger reconnection */
  reconnect: () => Promise<void>;

  /** Disconnect from conductor */
  disconnect: () => Promise<void>;
}

// =============================================================================
// Context
// =============================================================================

const HolochainContext = createContext<HolochainContextValue | null>(null);

// =============================================================================
// Provider Props
// =============================================================================

export interface HolochainProviderProps {
  children: ReactNode;
  /** Optional custom configuration */
  config?: ClientConfig;
  /** Whether to auto-connect on mount (default: true) */
  autoConnect?: boolean;
}

// =============================================================================
// Provider Component
// =============================================================================

export function HolochainProvider({
  children,
  config,
  autoConnect = true,
}: HolochainProviderProps) {
  const [client, setClient] = useState<UnifiedClient | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ClientMode>('mock');

  // Initialize client
  useEffect(() => {
    const initClient = async () => {
      try {
        // Use provided config or get from environment
        const clientConfig = config || getEnvConfig();
        const newClient = createClient(clientConfig);

        setClient(newClient);
        setMode(newClient.getMode());

        if (autoConnect) {
          setStatus('connecting');
          await newClient.connect();
          setStatus('connected');
          setError(null);
          console.log(`[HolochainProvider] Connected in ${newClient.getMode()} mode`);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to connect';
        console.error('[HolochainProvider] Connection error:', errorMessage);
        setError(errorMessage);
        setStatus('error');
      }
    };

    initClient();

    // Cleanup on unmount
    return () => {
      if (client) {
        client.disconnect().catch(console.error);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reconnect function
  const reconnect = useCallback(async () => {
    if (!client) {
      setError('No client initialized');
      return;
    }

    try {
      setStatus('connecting');
      setError(null);
      await client.connect();
      setStatus('connected');
      console.log('[HolochainProvider] Reconnected successfully');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to reconnect';
      console.error('[HolochainProvider] Reconnection error:', errorMessage);
      setError(errorMessage);
      setStatus('error');
    }
  }, [client]);

  // Disconnect function
  const disconnect = useCallback(async () => {
    if (!client) return;

    try {
      await client.disconnect();
      setStatus('disconnected');
      console.log('[HolochainProvider] Disconnected');
    } catch (err) {
      console.error('[HolochainProvider] Disconnect error:', err);
    }
  }, [client]);

  // Context value
  const value: HolochainContextValue = {
    client,
    status,
    mode,
    error,
    isMock: mode === 'mock',
    isReal: mode === 'real',
    reconnect,
    disconnect,
  };

  return (
    <HolochainContext.Provider value={value}>
      {children}
    </HolochainContext.Provider>
  );
}

// =============================================================================
// Hook
// =============================================================================

/**
 * Hook to access the Holochain context
 *
 * @example
 * function MyComponent() {
 *   const { client, status, isReal } = useHolochain();
 *
 *   if (status === 'connecting') return <Loading />;
 *   if (status === 'error') return <Error />;
 *
 *   return (
 *     <div>
 *       Mode: {isReal ? 'Real Holochain' : 'Mock'}
 *       <button onClick={() => client?.learning.list_courses()}>
 *         Load Courses
 *       </button>
 *     </div>
 *   );
 * }
 */
export function useHolochain(): HolochainContextValue {
  const context = useContext(HolochainContext);

  if (!context) {
    throw new Error('useHolochain must be used within a HolochainProvider');
  }

  return context;
}

// =============================================================================
// Convenience Hooks
// =============================================================================

/**
 * Hook to get just the client (throws if not connected)
 */
export function useHolochainClient(): UnifiedClient {
  const { client, status } = useHolochain();

  if (!client) {
    throw new Error('Holochain client not initialized');
  }

  if (status !== 'connected') {
    throw new Error(`Holochain client not connected (status: ${status})`);
  }

  return client;
}

/**
 * Hook to get connection status
 */
export function useHolochainStatus(): {
  status: ConnectionStatus;
  isConnected: boolean;
  isConnecting: boolean;
  isDisconnected: boolean;
  isError: boolean;
  error: string | null;
} {
  const { status, error } = useHolochain();

  return {
    status,
    isConnected: status === 'connected',
    isConnecting: status === 'connecting',
    isDisconnected: status === 'disconnected',
    isError: status === 'error',
    error,
  };
}

// =============================================================================
// Export
// =============================================================================

export default HolochainProvider;
