/**
 * React Provider for Mycelix SDK
 *
 * Provides React Context and Provider components for accessing the Mycelix client
 * throughout your application.
 *
 * @module frameworks/react/provider
 *
 * @example Basic Setup
 * ```tsx
 * import { MycelixProvider, useMycelix } from '@mycelix/sdk/react';
 *
 * function App() {
 *   return (
 *     <MycelixProvider config={{ conductorUrl: 'ws://localhost:8888', appId: 'my-app' }}>
 *       <YourApp />
 *     </MycelixProvider>
 *   );
 * }
 *
 * function YourApp() {
 *   const mycelix = useMycelix();
 *   // Access mycelix.identity, mycelix.governance, etc.
 * }
 * ```
 */

import * as React from 'react';

import type { Mycelix, MycelixConfig } from '../../core/index.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Connection state for the Mycelix client
 */
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

/**
 * Context value provided by MycelixProvider
 */
export interface MycelixContextValue {
  /** The connected Mycelix client instance (null if not connected) */
  mycelix: Mycelix | null;

  /** Current connection status */
  status: ConnectionStatus;

  /** Connection error (if status is 'error') */
  error: Error | null;

  /** Manually trigger connection */
  connect: () => Promise<void>;

  /** Disconnect from Holochain */
  disconnect: () => Promise<void>;

  /** Reconnect (disconnect then connect) */
  reconnect: () => Promise<void>;
}

/**
 * Props for MycelixProvider component
 */
export interface MycelixProviderProps {
  /** Mycelix configuration */
  config: MycelixConfig;

  /** Child components */
  children: React.ReactNode;

  /** Auto-connect on mount (default: true) */
  autoConnect?: boolean;

  /** Callback when connection succeeds */
  onConnect?: (mycelix: Mycelix) => void;

  /** Callback when connection fails */
  onError?: (error: Error) => void;

  /** Callback when disconnected */
  onDisconnect?: () => void;

  /** Custom loading component */
  loadingFallback?: React.ReactNode;

  /** Custom error component */
  errorFallback?: React.ReactNode | ((error: Error, retry: () => void) => React.ReactNode);
}

// =============================================================================
// Context
// =============================================================================

const MycelixContext = React.createContext<MycelixContextValue | null>(null);

// =============================================================================
// Provider Component
// =============================================================================

/**
 * MycelixProvider - Provides Mycelix context to your React application
 *
 * @example With auto-connect (default)
 * ```tsx
 * <MycelixProvider config={{ conductorUrl: 'ws://localhost:8888', appId: 'my-app' }}>
 *   <App />
 * </MycelixProvider>
 * ```
 *
 * @example With manual connect
 * ```tsx
 * <MycelixProvider
 *   config={config}
 *   autoConnect={false}
 *   onConnect={(m) => console.log('Connected!', m)}
 *   onError={(e) => console.error('Failed:', e)}
 * >
 *   <App />
 * </MycelixProvider>
 * ```
 *
 * @example With custom loading/error states
 * ```tsx
 * <MycelixProvider
 *   config={config}
 *   loadingFallback={<MySpinner />}
 *   errorFallback={(error, retry) => (
 *     <div>
 *       <p>Error: {error.message}</p>
 *       <button onClick={retry}>Retry</button>
 *     </div>
 *   )}
 * >
 *   <App />
 * </MycelixProvider>
 * ```
 */
export function MycelixProvider({
  config,
  children,
  autoConnect = true,
  onConnect,
  onError,
  onDisconnect,
  loadingFallback,
  errorFallback,
}: MycelixProviderProps): React.ReactElement {
  const [mycelix, setMycelix] = React.useState<Mycelix | null>(null);
  const [status, setStatus] = React.useState<ConnectionStatus>('disconnected');
  const [error, setError] = React.useState<Error | null>(null);

  // Track if component is mounted to prevent state updates after unmount
  const isMounted = React.useRef(true);
  React.useEffect(() => {
    return () => {
      isMounted.current = false;
    };
  }, []);

  const connect = React.useCallback(async () => {
    if (!isMounted.current) return;

    setStatus('connecting');
    setError(null);

    try {
      // Dynamically import to avoid bundling Holochain client for non-browser environments
      const { Mycelix } = await import('../../core/index.js');
      const client = await Mycelix.connect(config);

      if (!isMounted.current) {
        await client.disconnect();
        return;
      }

      setMycelix(client);
      setStatus('connected');
      onConnect?.(client);
    } catch (err) {
      if (!isMounted.current) return;

      const connectError = err instanceof Error ? err : new Error(String(err));
      setError(connectError);
      setStatus('error');
      onError?.(connectError);
    }
  }, [config, onConnect, onError]);

  const disconnect = React.useCallback(async () => {
    if (mycelix) {
      await mycelix.disconnect();
      if (isMounted.current) {
        setMycelix(null);
        setStatus('disconnected');
        onDisconnect?.();
      }
    }
  }, [mycelix, onDisconnect]);

  const reconnect = React.useCallback(async () => {
    await disconnect();
    await connect();
  }, [disconnect, connect]);

  // Auto-connect on mount
  React.useEffect(() => {
    if (autoConnect && status === 'disconnected') {
      connect();
    }

    // Cleanup on unmount
    return () => {
      if (mycelix) {
        mycelix.disconnect().catch(() => {
          // Ignore disconnect errors on unmount
        });
      }
    };
  }, []);

  const contextValue = React.useMemo<MycelixContextValue>(
    () => ({
      mycelix,
      status,
      error,
      connect,
      disconnect,
      reconnect,
    }),
    [mycelix, status, error, connect, disconnect, reconnect]
  );

  // Render loading state
  if (status === 'connecting' && loadingFallback) {
    return (
      <MycelixContext.Provider value={contextValue}>
        {loadingFallback}
      </MycelixContext.Provider>
    );
  }

  // Render error state
  if (status === 'error' && error && errorFallback) {
    const errorElement =
      typeof errorFallback === 'function' ? errorFallback(error, reconnect) : errorFallback;
    return (
      <MycelixContext.Provider value={contextValue}>
        {errorElement}
      </MycelixContext.Provider>
    );
  }

  return <MycelixContext.Provider value={contextValue}>{children}</MycelixContext.Provider>;
}

// =============================================================================
// Hooks
// =============================================================================

/**
 * useMycelixContext - Access the full Mycelix context
 *
 * @returns The Mycelix context value
 * @throws Error if used outside of MycelixProvider
 *
 * @example
 * ```tsx
 * function ConnectionStatus() {
 *   const { status, error, reconnect } = useMycelixContext();
 *
 *   if (status === 'error') {
 *     return <button onClick={reconnect}>Retry Connection</button>;
 *   }
 *
 *   return <span>Status: {status}</span>;
 * }
 * ```
 */
export function useMycelixContext(): MycelixContextValue {
  const context = React.useContext(MycelixContext);

  if (!context) {
    throw new Error(
      'useMycelixContext must be used within a MycelixProvider. ' +
        'Wrap your app with <MycelixProvider config={...}>.'
    );
  }

  return context;
}

/**
 * useMycelix - Access the Mycelix client instance
 *
 * @returns The connected Mycelix client
 * @throws Error if used outside of MycelixProvider or not connected
 *
 * @example
 * ```tsx
 * function CreateIdentity() {
 *   const mycelix = useMycelix();
 *
 *   const handleCreate = async () => {
 *     const did = await mycelix.identity.did.createDid();
 *     console.log('Created DID:', did);
 *   };
 *
 *   return <button onClick={handleCreate}>Create Identity</button>;
 * }
 * ```
 */
export function useMycelix(): Mycelix {
  const { mycelix, status } = useMycelixContext();

  if (!mycelix) {
    if (status === 'connecting') {
      throw new Error(
        'Mycelix is still connecting. Use useMycelixContext() to check status ' +
          'or wait for connection before rendering this component.'
      );
    }
    throw new Error(
      'Mycelix is not connected. Check that autoConnect is true or call connect() manually.'
    );
  }

  return mycelix;
}

/**
 * useConnectionStatus - Access connection status without requiring connection
 *
 * @returns Connection status and error
 *
 * @example
 * ```tsx
 * function ConnectButton() {
 *   const { status, error, connect } = useConnectionStatus();
 *
 *   if (status === 'connected') return null;
 *
 *   return (
 *     <button onClick={connect} disabled={status === 'connecting'}>
 *       {status === 'connecting' ? 'Connecting...' : 'Connect'}
 *     </button>
 *   );
 * }
 * ```
 */
export function useConnectionStatus(): {
  status: ConnectionStatus;
  error: Error | null;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  reconnect: () => Promise<void>;
} {
  const context = useMycelixContext();
  return {
    status: context.status,
    error: context.error,
    connect: context.connect,
    disconnect: context.disconnect,
    reconnect: context.reconnect,
  };
}

// =============================================================================
// HOC for Class Components
// =============================================================================

/**
 * withMycelix - Higher-order component for class components
 *
 * @example
 * ```tsx
 * interface Props {
 *   mycelix: Mycelix;
 * }
 *
 * class MyComponent extends React.Component<Props> {
 *   render() {
 *     return <div>Connected: {this.props.mycelix.isConnected()}</div>;
 *   }
 * }
 *
 * export default withMycelix(MyComponent);
 * ```
 */
export function withMycelix<P extends { mycelix: Mycelix }>(
  Component: React.ComponentType<P>
): React.ComponentType<Omit<P, 'mycelix'>> {
  const WrappedComponent = (props: Omit<P, 'mycelix'>) => {
    const mycelix = useMycelix();
    return <Component {...(props as P)} mycelix={mycelix} />;
  };

  WrappedComponent.displayName = `withMycelix(${Component.displayName || Component.name || 'Component'})`;

  return WrappedComponent;
}

export default MycelixProvider;
