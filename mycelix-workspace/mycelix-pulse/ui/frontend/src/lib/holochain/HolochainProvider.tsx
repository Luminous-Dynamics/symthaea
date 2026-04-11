// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Provider
 *
 * React context provider for Holochain connection management
 */

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  ReactNode,
} from 'react';
import {
  connectToHolochain,
  disconnectFromHolochain,
  getHolochainClient,
  onConnectionStateChange,
  ConnectionState,
  HolochainConfig,
} from './index';
import type { MycelixMailClient } from '@mycelix/holochain-client';

// ==========================================
// Context Types
// ==========================================

export interface HolochainContextValue {
  client: MycelixMailClient | null;
  connectionState: ConnectionState;
  error: Error | null;
  connect: (config?: HolochainConfig) => Promise<void>;
  disconnect: () => Promise<void>;
  isConnected: boolean;
  isConnecting: boolean;
}

const HolochainContext = createContext<HolochainContextValue | null>(null);

// ==========================================
// Hook
// ==========================================

export function useHolochain(): HolochainContextValue {
  const context = useContext(HolochainContext);
  if (!context) {
    throw new Error('useHolochain must be used within a HolochainProvider');
  }
  return context;
}

// ==========================================
// Provider Props
// ==========================================

export interface HolochainProviderProps {
  children: ReactNode;
  config?: HolochainConfig;
  autoConnect?: boolean;
  onConnected?: () => void;
  onDisconnected?: () => void;
  onError?: (error: Error) => void;
  loadingComponent?: ReactNode;
  errorComponent?: (error: Error, retry: () => void) => ReactNode;
}

// ==========================================
// Provider Component
// ==========================================

export function HolochainProvider({
  children,
  config,
  autoConnect = true,
  onConnected,
  onDisconnected,
  onError,
  loadingComponent,
  errorComponent,
}: HolochainProviderProps) {
  const [client, setClient] = useState<MycelixMailClient | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [error, setError] = useState<Error | null>(null);

  // Subscribe to connection state changes
  useEffect(() => {
    return onConnectionStateChange((state, err) => {
      setConnectionState(state);
      setError(err || null);

      if (state === 'connected') {
        try {
          setClient(getHolochainClient());
          onConnected?.();
        } catch (e) {
          setClient(null);
        }
      } else if (state === 'disconnected') {
        setClient(null);
        onDisconnected?.();
      } else if (state === 'error' && err) {
        onError?.(err);
      }
    });
  }, [onConnected, onDisconnected, onError]);

  // Connect function
  const connect = useCallback(
    async (overrideConfig?: HolochainConfig) => {
      try {
        await connectToHolochain({ ...config, ...overrideConfig });
      } catch (e) {
        console.error('[HolochainProvider] Connection failed:', e);
        throw e;
      }
    },
    [config]
  );

  // Disconnect function
  const disconnect = useCallback(async () => {
    await disconnectFromHolochain();
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect().catch((e) => {
        console.error('[HolochainProvider] Auto-connect failed:', e);
      });
    }

    // Cleanup on unmount
    return () => {
      disconnectFromHolochain().catch(console.error);
    };
  }, [autoConnect, connect]);

  // Context value
  const value: HolochainContextValue = {
    client,
    connectionState,
    error,
    connect,
    disconnect,
    isConnected: connectionState === 'connected',
    isConnecting: connectionState === 'connecting',
  };

  // Render loading state
  if (connectionState === 'connecting' && loadingComponent) {
    return <>{loadingComponent}</>;
  }

  // Render error state
  if (connectionState === 'error' && error && errorComponent) {
    return <>{errorComponent(error, () => connect())}</>;
  }

  return (
    <HolochainContext.Provider value={value}>
      {children}
    </HolochainContext.Provider>
  );
}

// ==========================================
// Default Loading Component
// ==========================================

export function DefaultLoadingComponent() {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        backgroundColor: '#0f172a',
        color: '#f8fafc',
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          border: '3px solid #334155',
          borderTopColor: '#3b82f6',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }}
      />
      <h2 style={{ marginTop: 24, fontSize: 20, fontWeight: 500 }}>
        Connecting to Holochain...
      </h2>
      <p style={{ marginTop: 8, color: '#94a3b8', fontSize: 14 }}>
        Establishing secure P2P connection
      </p>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

// ==========================================
// Default Error Component
// ==========================================

export function DefaultErrorComponent({
  error,
  retry,
}: {
  error: Error;
  retry: () => void;
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        backgroundColor: '#0f172a',
        color: '#f8fafc',
        fontFamily: 'system-ui, sans-serif',
        padding: 24,
        textAlign: 'center',
      }}
    >
      <div
        style={{
          width: 64,
          height: 64,
          backgroundColor: '#7f1d1d',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 32,
        }}
      >
        ⚠️
      </div>
      <h2 style={{ marginTop: 24, fontSize: 20, fontWeight: 500 }}>
        Connection Failed
      </h2>
      <p style={{ marginTop: 8, color: '#94a3b8', fontSize: 14, maxWidth: 400 }}>
        {error.message}
      </p>
      <button
        onClick={retry}
        style={{
          marginTop: 24,
          padding: '12px 24px',
          backgroundColor: '#3b82f6',
          color: '#fff',
          border: 'none',
          borderRadius: 8,
          fontSize: 14,
          fontWeight: 500,
          cursor: 'pointer',
        }}
      >
        Retry Connection
      </button>
      <p style={{ marginTop: 16, color: '#64748b', fontSize: 12 }}>
        Make sure Holochain conductor is running
      </p>
    </div>
  );
}

// ==========================================
// Exports
// ==========================================

export default HolochainProvider;
