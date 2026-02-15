/**
 * React Hooks for Mycelix SDK
 *
 * Provides React hooks for easy integration with all 8 Civilizational OS hApps.
 * Works with React 18+ and supports Suspense, concurrent features, and SSR.
 */

// Note: This module is designed to work without React as a hard dependency.
// It uses dynamic imports and type declarations to remain framework-agnostic
// while providing excellent DX when React is available.

// Re-export specialized hooks for all 8 Civilizational OS hApp modules
export * from './identity.js';
export * from './knowledge.js';
export * from './governance.js';
export * from './justice.js';
export * from './finance.js';
export * from './property.js';
export * from './energy.js';
export * from './media.js';

// ============================================================================
// Types
// ============================================================================

export interface QueryState<T> {
  data: T | undefined;
  loading: boolean;
  error: Error | undefined;
  refetch: () => Promise<void>;
}

export interface MutationState<T, V> {
  data: T | undefined;
  loading: boolean;
  error: Error | undefined;
  mutate: (variables: V) => Promise<T>;
  reset: () => void;
}

export interface MycelixContextValue {
  // Services
  identityService: any;
  financeService: any;
  propertyService: any;
  energyService: any;
  mediaService: any;
  governanceService: any;
  justiceService: any;
  knowledgeService: any;

  // Connection state
  connected: boolean;
  connecting: boolean;
  error: Error | undefined;

  // Methods
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
}

// ============================================================================
// Hook Factories (Framework-agnostic)
// ============================================================================

/**
 * Create a query hook factory
 */
export function createQueryHook<T, V = void>(
  _fetcher: (variables: V, ctx: MycelixContextValue) => Promise<T>
) {
  return function useQuery(_variables: V): QueryState<T> {
    // This is a placeholder implementation
    // The actual implementation requires React hooks
    throw new Error(
      'React hooks require the @mycelix/react package. ' +
        'Install it with: npm install @mycelix/react'
    );
  };
}

/**
 * Create a mutation hook factory
 */
export function createMutationHook<T, V>(
  _mutator: (variables: V, ctx: MycelixContextValue) => Promise<T>
) {
  return function useMutation(): MutationState<T, V> {
    throw new Error(
      'React hooks require the @mycelix/react package. ' +
        'Install it with: npm install @mycelix/react'
    );
  };
}

// ============================================================================
// NOTE: Domain-specific hooks are exported from submodules above.
// This file only contains cross-domain hooks and shared utilities.
// ============================================================================

// ============================================================================
// Cross-Domain Hooks
// ============================================================================

/**
 * Hook to fetch all entities related to a DID
 */
export function useRelatedEntities(_did: string): QueryState<{
  did: string;
  assets: any[];
  wallets: any[];
  content: any[];
  proposals: any[];
  cases: any[];
  claims: any[];
}> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch entity reputation
 */
export function useReputation(_did: string): QueryState<any> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Connection Hooks
// ============================================================================

/**
 * Hook to access Mycelix context
 */
export function useMycelix(): MycelixContextValue {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to check connection status
 */
export function useConnectionStatus(): { connected: boolean; connecting: boolean; error?: Error } {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// React Implementation Guide
// ============================================================================

/**
 * Example React implementation (requires React as a dependency):
 *
 * ```tsx
 * import { createContext, useContext, useState, useEffect, useCallback } from 'react';
 * import { getIdentityService, getFinanceService, ... } from '@mycelix/sdk';
 *
 * const MycelixContext = createContext<MycelixContextValue | null>(null);
 *
 * export function MycelixProvider({ children }: { children: React.ReactNode }) {
 *   const [connected, setConnected] = useState(false);
 *   const [services, setServices] = useState({});
 *
 *   const connect = useCallback(async () => {
 *     // Initialize services
 *     setServices({
 *       identityService: getIdentityService(),
 *       financeService: getFinanceService(),
 *       // ...
 *     });
 *     setConnected(true);
 *   }, []);
 *
 *   return (
 *     <MycelixContext.Provider value={{ ...services, connected, connect }}>
 *       {children}
 *     </MycelixContext.Provider>
 *   );
 * }
 *
 * export function useIdentityProfile(did: string) {
 *   const ctx = useContext(MycelixContext);
 *   const [state, setState] = useState({ loading: true, data: undefined, error: undefined });
 *
 *   useEffect(() => {
 *     ctx.identityService?.getProfile(did)
 *       .then(data => setState({ loading: false, data, error: undefined }))
 *       .catch(error => setState({ loading: false, data: undefined, error }));
 *   }, [did, ctx.identityService]);
 *
 *   return state;
 * }
 * ```
 */
