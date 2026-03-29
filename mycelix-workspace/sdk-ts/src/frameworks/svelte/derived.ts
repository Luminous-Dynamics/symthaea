// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Derived Store Utilities for Svelte
 *
 * Provides utilities for creating derived stores from Mycelix data.
 *
 * @module frameworks/svelte/derived
 *
 * @example Creating a derived store
 * ```svelte
 * <script>
 *   import { createDerivedStore, mycelix } from '@mycelix/sdk/svelte';
 *
 *   // Create a store that fetches DAO details
 *   const daoDetails = createDerivedStore(
 *     async (m) => m.governance.dao.getDao('my-dao-id')
 *   );
 * </script>
 *
 * {#if $daoDetails.loading}
 *   <p>Loading...</p>
 * {:else if $daoDetails.error}
 *   <p>Error: {$daoDetails.error.message}</p>
 * {:else if $daoDetails.data}
 *   <p>DAO: {$daoDetails.data.name}</p>
 * {/if}
 * ```
 */

import { mycelix } from './stores.js';

import type { Readable, AsyncState } from './stores.js';
import type { Mycelix } from '../../core/index.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Options for derived stores
 */
export interface DerivedStoreOptions {
  /** Refetch interval in milliseconds (0 = no auto-refetch) */
  refetchInterval?: number;
  /** Whether to fetch immediately on subscribe (default: true) */
  immediate?: boolean;
  /** Callback when fetch succeeds */
  onSuccess?: (data: unknown) => void;
  /** Callback when fetch fails */
  onError?: (error: Error) => void;
}

/**
 * Extended async store with additional methods
 */
export interface DerivedAsyncStore<T> extends Readable<AsyncState<T | null>> {
  /** Manually trigger a refetch */
  refetch: () => Promise<void>;
  /** Reset the store to initial state */
  reset: () => void;
}

// =============================================================================
// Internal Types
// =============================================================================

type Subscriber<T> = (value: T) => void;
type Unsubscriber = () => void;

// =============================================================================
// Derived Store Factory
// =============================================================================

/**
 * Create a derived store that fetches data from Mycelix
 *
 * The store automatically fetches data when Mycelix connects and
 * can be configured to refetch at intervals.
 *
 * @param selector - Async function that fetches data from Mycelix
 * @param options - Store options
 * @returns A readable store with async state
 *
 * @example Basic Usage
 * ```svelte
 * <script>
 *   import { createDerivedStore } from '@mycelix/sdk/svelte';
 *
 *   const userProfile = createDerivedStore(
 *     async (mycelix) => {
 *       const did = await mycelix.getMyDid();
 *       return mycelix.identity.profiles.getProfile(did);
 *     }
 *   );
 * </script>
 *
 * {#if $userProfile.data}
 *   <p>Welcome, {$userProfile.data.displayName}</p>
 * {/if}
 * ```
 *
 * @example With Auto-Refetch
 * ```svelte
 * <script>
 *   const liveBalance = createDerivedStore(
 *     async (m) => m.finance.balances.getBalances(did),
 *     { refetchInterval: 30000 } // Refresh every 30 seconds
 *   );
 * </script>
 * ```
 *
 * @example With Callbacks
 * ```svelte
 * <script>
 *   const proposals = createDerivedStore(
 *     async (m) => m.governance.proposals.listProposals({ status: 'active' }),
 *     {
 *       onSuccess: (data) => console.log('Loaded', data.length, 'proposals'),
 *       onError: (error) => toast.error(error.message),
 *     }
 *   );
 * </script>
 * ```
 */
export function createDerivedStore<T>(
  selector: (m: Mycelix) => Promise<T>,
  options: DerivedStoreOptions = {}
): DerivedAsyncStore<T> {
  const { refetchInterval = 0, immediate = true, onSuccess, onError } = options;

  // Internal state
  let state: AsyncState<T | null> = {
    data: null,
    loading: immediate,
    error: null,
  };

  const subscribers = new Set<Subscriber<AsyncState<T | null>>>();
  let currentClient: Mycelix | null = null;
  let intervalId: ReturnType<typeof setInterval> | null = null;
  let unsubscribeMycelix: Unsubscriber | null = null;

  // Notify all subscribers
  function notify(): void {
    subscribers.forEach((sub) => sub(state));
  }

  // Set state and notify
  function setState(partial: Partial<AsyncState<T | null>>): void {
    state = { ...state, ...partial };
    notify();
  }

  // Fetch data
  async function fetch(): Promise<void> {
    if (!currentClient) return;

    setState({ loading: true, error: null });

    try {
      const data = await selector(currentClient);
      setState({ data, loading: false });
      onSuccess?.(data);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState({ error, loading: false });
      onError?.(error);
    }
  }

  // Refetch method
  async function refetch(): Promise<void> {
    await fetch();
  }

  // Reset method
  function reset(): void {
    setState({ data: null, loading: false, error: null });
  }

  // Start subscription to mycelix store
  function start(): void {
    unsubscribeMycelix = mycelix.subscribe((client) => {
      currentClient = client;

      if (client && immediate) {
        fetch();
      }

      // Set up interval if configured
      if (client && refetchInterval > 0) {
        if (intervalId) clearInterval(intervalId);
        intervalId = setInterval(fetch, refetchInterval);
      }
    });
  }

  // Stop subscription
  function stop(): void {
    if (unsubscribeMycelix) {
      unsubscribeMycelix();
      unsubscribeMycelix = null;
    }
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  // Subscribe function
  function subscribe(run: Subscriber<AsyncState<T | null>>): Unsubscriber {
    // Start watching mycelix if this is the first subscriber
    if (subscribers.size === 0) {
      start();
    }

    subscribers.add(run);
    run(state);

    return () => {
      subscribers.delete(run);
      // Stop watching if no more subscribers
      if (subscribers.size === 0) {
        stop();
      }
    };
  }

  return {
    subscribe,
    refetch,
    reset,
  };
}

// =============================================================================
// Specialized Derived Stores
// =============================================================================

/**
 * Create a derived store for a specific DAO
 *
 * @param daoId - ID of the DAO
 * @returns Store with DAO details
 *
 * @example
 * ```svelte
 * <script>
 *   import { createDaoStore } from '@mycelix/sdk/svelte';
 *
 *   const dao = createDaoStore('my-dao-id');
 * </script>
 *
 * {#if $dao.data}
 *   <h1>{$dao.data.name}</h1>
 *   <p>{$dao.data.description}</p>
 * {/if}
 * ```
 */
export function createDaoStore(daoId: string, options?: DerivedStoreOptions) {
  return createDerivedStore(
    async (m) => m.governance.dao.getDao(daoId),
    options
  );
}

/**
 * Create a derived store for proposals in a DAO
 *
 * @param daoId - ID of the DAO
 * @param status - Optional status filter
 * @returns Store with proposals list
 *
 * @example
 * ```svelte
 * <script>
 *   import { createProposalsStore } from '@mycelix/sdk/svelte';
 *
 *   const activeProposals = createProposalsStore('my-dao', 'active');
 * </script>
 *
 * <ul>
 *   {#each $activeProposals.data ?? [] as proposal}
 *     <li>{proposal.title}</li>
 *   {/each}
 * </ul>
 * ```
 */
export function createProposalsStore(
  daoId: string,
  status?: 'draft' | 'active' | 'passed' | 'rejected' | 'executed' | 'expired',
  options?: DerivedStoreOptions
) {
  // Map lowercase status to governance zome format
  const statusMap: Record<string, string> = {
    draft: 'Draft',
    active: 'Active',
    passed: 'Passed',
    rejected: 'Failed',
    executed: 'Executed',
    expired: 'Cancelled',
  };
  return createDerivedStore(
    async (m) => m.governance.proposals.getProposals({
      dao_id: daoId,
      status_filter: status ? statusMap[status] as 'Draft' | 'Active' | 'Passed' | 'Failed' | 'Executed' | 'Vetoed' | 'Cancelled' : undefined,
    }),
    options
  );
}

/**
 * Create a derived store for user balances
 *
 * @param did - User's DID (or null for current user)
 * @returns Store with balance map
 *
 * @example
 * ```svelte
 * <script>
 *   import { createBalanceStore } from '@mycelix/sdk/svelte';
 *
 *   const myBalance = createBalanceStore(null); // Current user
 *   const aliceBalance = createBalanceStore('did:mycelix:alice');
 * </script>
 *
 * <p>My MYC: {$myBalance.data?.MYC ?? 0}</p>
 * <p>Alice's MYC: {$aliceBalance.data?.MYC ?? 0}</p>
 * ```
 */
export function createBalanceStore(
  did: string | null = null,
  options?: DerivedStoreOptions
) {
  return createDerivedStore(
    async (m) => {
      const didRecord = did || await m.getMyDid();
      const targetDid = didRecord ? (typeof didRecord === 'string' ? didRecord : (didRecord as { did: string }).did) : null;
      if (!targetDid) return {};
      return m.finance.wallets.getAllBalances(targetDid);
    },
    options
  );
}

/**
 * Create a derived store for user profile
 *
 * @param did - User's DID (or null for current user)
 * @returns Store with profile data
 *
 * @example
 * ```svelte
 * <script>
 *   import { createProfileStore } from '@mycelix/sdk/svelte';
 *
 *   export let did;
 *   $: profile = createProfileStore(did);
 * </script>
 *
 * {#if $profile.data}
 *   <img src={$profile.data.avatarUrl} alt={$profile.data.displayName} />
 *   <h2>{$profile.data.displayName}</h2>
 * {/if}
 * ```
 */
export function createProfileStore(
  did: string | null = null,
  options?: DerivedStoreOptions
) {
  return createDerivedStore(
    async (m) => {
      const didRecord = did || await m.getMyDid();
      const targetDid = didRecord ? (typeof didRecord === 'string' ? didRecord : (didRecord as { did: string }).did) : null;
      if (!targetDid) return null;
      // Profile operations may be available through credentials or a profile zome
      return (m.identity as { credentials?: { getProfile?: (did: string) => Promise<unknown> } }).credentials?.getProfile?.(targetDid) || null;
    },
    options
  );
}

// =============================================================================
// Computed Stores
// =============================================================================

/**
 * Create a computed store that transforms another store's data
 *
 * @param store - Source store
 * @param transform - Transform function
 * @returns Computed readable store
 *
 * @example
 * ```svelte
 * <script>
 *   import { proposals, computedStore } from '@mycelix/sdk/svelte';
 *
 *   // Count active proposals
 *   const activeCount = computedStore(proposals, (p) => p.filter(x => x.status === 'active').length);
 * </script>
 *
 * <p>Active proposals: {$activeCount}</p>
 * ```
 */
export function computedStore<S, T>(
  store: Readable<S>,
  transform: (value: S) => T
): Readable<T> {
  return {
    subscribe(run: Subscriber<T>): Unsubscriber {
      return store.subscribe((value) => {
        run(transform(value));
      });
    },
  };
}

/**
 * Combine multiple stores into one
 *
 * @param stores - Object of stores
 * @returns Combined store
 *
 * @example
 * ```svelte
 * <script>
 *   import { identity, walletBalance, proposals, combineStores } from '@mycelix/sdk/svelte';
 *
 *   const dashboard = combineStores({
 *     did: identity,
 *     balance: walletBalance,
 *     proposals,
 *   });
 * </script>
 *
 * <p>DID: {$dashboard.did}</p>
 * <p>Balance: {$dashboard.balance}</p>
 * <p>Proposals: {$dashboard.proposals.length}</p>
 * ```
 */
export function combineStores<T extends Record<string, Readable<unknown>>>(
  stores: T
): Readable<{ [K in keyof T]: T[K] extends Readable<infer V> ? V : never }> {
  type Combined = { [K in keyof T]: T[K] extends Readable<infer V> ? V : never };

  return {
    subscribe(run: Subscriber<Combined>): Unsubscriber {
      const values: Partial<Combined> = {};
      const unsubscribers: Unsubscriber[] = [];

      const keys = Object.keys(stores) as (keyof T)[];

      keys.forEach((key) => {
        const unsubscribe = stores[key].subscribe((value) => {
          values[key] = value as Combined[typeof key];
          run({ ...values } as Combined);
        });
        unsubscribers.push(unsubscribe);
      });

      return () => {
        unsubscribers.forEach((unsub) => unsub());
      };
    },
  };
}

// =============================================================================
// Type Re-exports
// =============================================================================

export type { Readable, AsyncState, DID } from './stores.js';
