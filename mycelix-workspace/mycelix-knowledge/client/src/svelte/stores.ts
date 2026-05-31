// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Svelte Stores for Mycelix Knowledge SDK
 *
 * @example
 * ```svelte
 * <script>
 *   import { createClaimStore, createFactCheckStore } from '@mycelix/knowledge-sdk/svelte';
 *   import { knowledgeClient } from './client';
 *
 *   const claimStore = createClaimStore(knowledgeClient);
 *   $: claimStore.load(claimId);
 *
 *   const factCheck = createFactCheckStore(knowledgeClient);
 * </script>
 *
 * {#if $claimStore.loading}
 *   <Spinner />
 * {:else if $claimStore.error}
 *   <Error message={$claimStore.error.message} />
 * {:else if $claimStore.data}
 *   <ClaimCard claim={$claimStore.data} />
 * {/if}
 * ```
 */

import { writable, derived, readable, get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';
import type {
  KnowledgeClient,
  Claim,
  ClaimSummary,
  FactCheckResult,
  EnhancedCredibilityScore,
  AuthorReputation,
  Relationship,
  InformationValueRecord,
  DependencyTree,
  PropagationResult,
  MarketSummary,
  CreateClaimInput,
  FactCheckInput,
  QueryOptions,
} from '../index';

// ============================================================================
// Types
// ============================================================================

interface AsyncStore<T> {
  subscribe: Readable<AsyncState<T>>['subscribe'];
  load: (...args: unknown[]) => Promise<void>;
  reset: () => void;
  refetch: () => Promise<void>;
}

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

interface MutationStore<TInput, TOutput> {
  subscribe: Readable<MutationState<TOutput>>['subscribe'];
  mutate: (input: TInput) => Promise<TOutput>;
  reset: () => void;
}

interface MutationState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

// ============================================================================
// Cache
// ============================================================================

const cache = new Map<string, { data: unknown; timestamp: number; ttl: number }>();
const DEFAULT_TTL = 60000;

function getCached<T>(key: string): T | null {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > entry.ttl) {
    cache.delete(key);
    return null;
  }
  return entry.data as T;
}

function setCache<T>(key: string, data: T, ttl: number = DEFAULT_TTL): void {
  cache.set(key, { data, timestamp: Date.now(), ttl });
}

export function invalidateCache(pattern?: string): void {
  if (!pattern) {
    cache.clear();
    return;
  }
  for (const key of cache.keys()) {
    if (key.includes(pattern)) {
      cache.delete(key);
    }
  }
}

// ============================================================================
// Store Factory
// ============================================================================

function createAsyncStore<T, TArgs extends unknown[]>(
  fetcher: (...args: TArgs) => Promise<T>,
  cacheKey?: (...args: TArgs) => string,
  ttl: number = DEFAULT_TTL
): AsyncStore<T> & { set: Writable<AsyncState<T>>['set'] } {
  const store = writable<AsyncState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  let lastArgs: TArgs | null = null;

  async function load(...args: TArgs): Promise<void> {
    lastArgs = args;

    // Check cache
    if (cacheKey) {
      const key = cacheKey(...args);
      const cached = getCached<T>(key);
      if (cached) {
        store.set({ data: cached, loading: false, error: null });
        return;
      }
    }

    store.update((s) => ({ ...s, loading: true, error: null }));

    try {
      const data = await fetcher(...args);

      // Update cache
      if (cacheKey) {
        setCache(cacheKey(...args), data, ttl);
      }

      store.set({ data, loading: false, error: null });
    } catch (err) {
      store.set({
        data: null,
        loading: false,
        error: err instanceof Error ? err : new Error(String(err)),
      });
    }
  }

  function reset(): void {
    store.set({ data: null, loading: false, error: null });
    lastArgs = null;
  }

  async function refetch(): Promise<void> {
    if (lastArgs) {
      // Invalidate cache before refetch
      if (cacheKey) {
        cache.delete(cacheKey(...lastArgs));
      }
      await load(...lastArgs);
    }
  }

  return {
    subscribe: store.subscribe,
    set: store.set,
    load: load as (...args: unknown[]) => Promise<void>,
    reset,
    refetch,
  };
}

function createMutationStore<TInput, TOutput>(
  mutator: (input: TInput) => Promise<TOutput>,
  onSuccess?: (data: TOutput) => void
): MutationStore<TInput, TOutput> {
  const store = writable<MutationState<TOutput>>({
    data: null,
    loading: false,
    error: null,
  });

  async function mutate(input: TInput): Promise<TOutput> {
    store.update((s) => ({ ...s, loading: true, error: null }));

    try {
      const data = await mutator(input);
      store.set({ data, loading: false, error: null });
      onSuccess?.(data);
      return data;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      store.set({ data: null, loading: false, error });
      throw error;
    }
  }

  function reset(): void {
    store.set({ data: null, loading: false, error: null });
  }

  return {
    subscribe: store.subscribe,
    mutate,
    reset,
  };
}

// ============================================================================
// Query Stores
// ============================================================================

/**
 * Create a store for fetching a single claim
 */
export function createClaimStore(client: KnowledgeClient) {
  return createAsyncStore<Claim, [string]>(
    (claimId) => client.claims.getClaim(claimId),
    (claimId) => `claim:${claimId}`
  );
}

/**
 * Create a store for searching claims
 */
export function createSearchStore(client: KnowledgeClient) {
  return createAsyncStore<ClaimSummary[], [string, QueryOptions?]>(
    (query, options) => client.query.search(query, options),
    (query, options) => `search:${query}:${JSON.stringify(options)}`,
    30000
  );
}

/**
 * Create a store for fact-checking
 */
export function createFactCheckStore(client: KnowledgeClient) {
  return createAsyncStore<FactCheckResult, [FactCheckInput]>(
    (input) => client.factcheck.factCheck(input),
    (input) => `factcheck:${input.statement}`,
    300000
  );
}

/**
 * Create a store for credibility scores
 */
export function createCredibilityStore(client: KnowledgeClient) {
  return createAsyncStore<EnhancedCredibilityScore, [string, 'Claim' | 'Author' | 'Source']>(
    (entityId, entityType) => client.inference.calculateEnhancedCredibility(entityId, entityType),
    (entityId, entityType) => `credibility:${entityType}:${entityId}`,
    120000
  );
}

/**
 * Create a store for author reputation
 */
export function createAuthorReputationStore(client: KnowledgeClient) {
  return createAsyncStore<AuthorReputation, [string]>(
    (authorDid) => client.inference.getAuthorReputation(authorDid),
    (authorDid) => `author:${authorDid}`,
    300000
  );
}

/**
 * Create a store for relationships
 */
export function createRelationshipsStore(client: KnowledgeClient) {
  return createAsyncStore<Relationship[], [string]>(
    (claimId) => client.graph.getRelationships(claimId),
    (claimId) => `relationships:${claimId}`
  );
}

/**
 * Create a store for dependency tree
 */
export function createDependencyTreeStore(client: KnowledgeClient) {
  return createAsyncStore<DependencyTree, [string, number?]>(
    (claimId, maxDepth = 3) => client.graph.getDependencyTree(claimId, maxDepth),
    (claimId, maxDepth = 3) => `tree:${claimId}:${maxDepth}`
  );
}

/**
 * Create a store for claim markets
 */
export function createClaimMarketsStore(client: KnowledgeClient) {
  return createAsyncStore<MarketSummary[], [string]>(
    (claimId) => client.marketsIntegration.getClaimMarkets(claimId),
    (claimId) => `markets:${claimId}`
  );
}

/**
 * Create a store for information value rankings
 */
export function createInformationValueStore(client: KnowledgeClient) {
  return createAsyncStore<InformationValueRecord[], [number?]>(
    (limit = 20) => client.graph.rankByInformationValue(limit),
    (limit = 20) => `iv:${limit}`
  );
}

// ============================================================================
// Mutation Stores
// ============================================================================

/**
 * Create a store for creating claims
 */
export function createCreateClaimStore(client: KnowledgeClient) {
  return createMutationStore<CreateClaimInput, string>(
    (input) => client.claims.createClaim(input),
    () => invalidateCache('search:')
  );
}

/**
 * Create a store for updating claims
 */
export function createUpdateClaimStore(client: KnowledgeClient) {
  return createMutationStore<
    { originalHash: string; content?: string; classification?: { empirical: number; normative: number; mythic: number } },
    string
  >(
    (input) => client.claims.updateClaim(input),
    (_, input) => invalidateCache(`claim:${input}`)
  );
}

/**
 * Create a store for creating relationships
 */
export function createCreateRelationshipStore(client: KnowledgeClient) {
  return createMutationStore<
    { sourceId: string; targetId: string; relationshipType: string; weight: number },
    string
  >((input) =>
    client.graph.createRelationship(input.sourceId, input.targetId, input.relationshipType as any, input.weight)
  );
}

/**
 * Create a store for spawning verification markets
 */
export function createSpawnMarketStore(client: KnowledgeClient) {
  return createMutationStore<
    { claimId: string; targetE: number; minConfidence?: number; closesAt?: number; initialSubsidy?: number },
    string
  >((input) =>
    client.claims.spawnVerificationMarket({
      claimId: input.claimId,
      targetE: input.targetE,
      minConfidence: input.minConfidence ?? 0.7,
      closesAt: input.closesAt ?? Date.now() + 14 * 24 * 60 * 60 * 1000,
      initialSubsidy: input.initialSubsidy ?? 100,
      tags: [],
    })
  );
}

/**
 * Create a store for belief propagation
 */
export function createPropagateBeliefStore(client: KnowledgeClient) {
  return createMutationStore<string, PropagationResult>(
    (claimId) => client.graph.propagateBelief(claimId),
    () => invalidateCache()
  );
}

// ============================================================================
// Derived Stores
// ============================================================================

/**
 * Create a combined store for enriched claim data
 */
export function createEnrichedClaimStore(client: KnowledgeClient) {
  const claimStore = createClaimStore(client);
  const credibilityStore = createCredibilityStore(client);
  const relationshipsStore = createRelationshipsStore(client);
  const marketsStore = createClaimMarketsStore(client);

  const combined = derived(
    [claimStore, credibilityStore, relationshipsStore, marketsStore] as [
      typeof claimStore,
      typeof credibilityStore,
      typeof relationshipsStore,
      typeof marketsStore
    ],
    ([$claim, $credibility, $relationships, $markets]) => ({
      claim: $claim.data,
      credibility: $credibility.data,
      relationships: $relationships.data,
      markets: $markets.data,
      loading: $claim.loading || $credibility.loading || $relationships.loading || $markets.loading,
      error: $claim.error || $credibility.error || $relationships.error || $markets.error,
    })
  );

  async function load(claimId: string): Promise<void> {
    await Promise.all([
      claimStore.load(claimId),
      credibilityStore.load(claimId, 'Claim'),
      relationshipsStore.load(claimId),
      marketsStore.load(claimId),
    ]);
  }

  return {
    subscribe: combined.subscribe,
    load,
    reset: () => {
      claimStore.reset();
      credibilityStore.reset();
      relationshipsStore.reset();
      marketsStore.reset();
    },
  };
}

// ============================================================================
// Reactive Stores (Auto-refresh)
// ============================================================================

/**
 * Create a store that auto-refreshes at an interval
 */
export function createAutoRefreshStore<T>(
  baseStore: AsyncStore<T>,
  intervalMs: number = 30000
): AsyncStore<T> & { stop: () => void } {
  let intervalId: ReturnType<typeof setInterval> | null = null;
  let lastArgs: unknown[] = [];

  const originalLoad = baseStore.load;

  const load = async (...args: unknown[]): Promise<void> => {
    lastArgs = args;
    await originalLoad(...args);

    // Start auto-refresh
    if (intervalId) clearInterval(intervalId);
    intervalId = setInterval(() => {
      originalLoad(...lastArgs);
    }, intervalMs);
  };

  const stop = (): void => {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  };

  const reset = (): void => {
    stop();
    baseStore.reset();
  };

  return {
    subscribe: baseStore.subscribe,
    load,
    reset,
    refetch: baseStore.refetch,
    stop,
  };
}
