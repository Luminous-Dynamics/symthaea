// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Mycelix Knowledge SDK
 *
 * @example
 * ```tsx
 * import { useClaim, useFactCheck, useCredibility } from '@mycelix/knowledge-sdk/react';
 *
 * function ClaimDisplay({ claimId }) {
 *   const { claim, loading, error, refetch } = useClaim(claimId);
 *   if (loading) return <Spinner />;
 *   if (error) return <Error message={error.message} />;
 *   return <ClaimCard claim={claim} />;
 * }
 * ```
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useKnowledgeClient } from './context';
import type {
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

interface UseQueryResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

interface UseMutationResult<TInput, TOutput> {
  mutate: (input: TInput) => Promise<TOutput>;
  loading: boolean;
  error: Error | null;
  data: TOutput | null;
  reset: () => void;
}

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

// ============================================================================
// Cache
// ============================================================================

const cache = new Map<string, CacheEntry<unknown>>();
const DEFAULT_TTL = 60000; // 1 minute

function getCached<T>(key: string): T | null {
  const entry = cache.get(key) as CacheEntry<T> | undefined;
  if (!entry) return null;
  if (Date.now() - entry.timestamp > entry.ttl) {
    cache.delete(key);
    return null;
  }
  return entry.data;
}

function setCache<T>(key: string, data: T, ttl: number = DEFAULT_TTL): void {
  cache.set(key, { data, timestamp: Date.now(), ttl });
}

function invalidateCache(pattern?: string): void {
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
// Core Hooks
// ============================================================================

/**
 * Hook to fetch and manage a single claim
 */
export function useClaim(claimId: string | null): UseQueryResult<Claim> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<Claim | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetch = useCallback(async () => {
    if (!claimId || !client) {
      setLoading(false);
      return;
    }

    const cacheKey = `claim:${claimId}`;
    const cached = getCached<Claim>(cacheKey);
    if (cached) {
      setData(cached);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const claim = await client.claims.getClaim(claimId);
      if (claim) {
        setCache(cacheKey, claim);
        setData(claim);
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [claimId, client]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Hook to search for claims
 */
export function useClaimSearch(
  query: string,
  options?: QueryOptions
): UseQueryResult<ClaimSummary[]> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<ClaimSummary[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const fetch = useCallback(async () => {
    if (!query || !client) {
      setData([]);
      setLoading(false);
      return;
    }

    const cacheKey = `search:${query}:${JSON.stringify(optionsRef.current)}`;
    const cached = getCached<ClaimSummary[]>(cacheKey);
    if (cached) {
      setData(cached);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const results = await client.query.search(query, optionsRef.current);
      setCache(cacheKey, results, 30000); // 30s cache for searches
      setData(results);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [query, client]);

  useEffect(() => {
    const debounce = setTimeout(fetch, 300); // Debounce search
    return () => clearTimeout(debounce);
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Hook to perform fact-checking
 */
export function useFactCheck(
  statement: string | null,
  options?: Partial<FactCheckInput>
): UseQueryResult<FactCheckResult> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<FactCheckResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const fetch = useCallback(async () => {
    if (!statement || !client) {
      setLoading(false);
      return;
    }

    const cacheKey = `factcheck:${statement}:${JSON.stringify(options)}`;
    const cached = getCached<FactCheckResult>(cacheKey);
    if (cached) {
      setData(cached);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await client.factcheck.factCheck({
        statement,
        ...options,
      });
      setCache(cacheKey, result, 300000); // 5 min cache for fact-checks
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [statement, options, client]);

  useEffect(() => {
    if (statement) {
      fetch();
    }
  }, [fetch, statement]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Hook to get credibility score for an entity
 */
export function useCredibility(
  entityId: string | null,
  entityType: 'Claim' | 'Author' | 'Source' = 'Claim'
): UseQueryResult<EnhancedCredibilityScore> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<EnhancedCredibilityScore | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetch = useCallback(async () => {
    if (!entityId || !client) {
      setLoading(false);
      return;
    }

    const cacheKey = `credibility:${entityType}:${entityId}`;
    const cached = getCached<EnhancedCredibilityScore>(cacheKey);
    if (cached) {
      setData(cached);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const score = await client.inference.calculateEnhancedCredibility(entityId, entityType);
      setCache(cacheKey, score, 120000); // 2 min cache
      setData(score);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [entityId, entityType, client]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Hook to get author reputation
 */
export function useAuthorReputation(authorDid: string | null): UseQueryResult<AuthorReputation> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<AuthorReputation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetch = useCallback(async () => {
    if (!authorDid || !client) {
      setLoading(false);
      return;
    }

    const cacheKey = `author:${authorDid}`;
    const cached = getCached<AuthorReputation>(cacheKey);
    if (cached) {
      setData(cached);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const reputation = await client.inference.getAuthorReputation(authorDid);
      setCache(cacheKey, reputation, 300000); // 5 min cache
      setData(reputation);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [authorDid, client]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Hook to get relationships for a claim
 */
export function useRelationships(claimId: string | null): UseQueryResult<Relationship[]> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<Relationship[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetch = useCallback(async () => {
    if (!claimId || !client) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const relationships = await client.graph.getRelationships(claimId);
      setData(relationships);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [claimId, client]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Hook to get dependency tree for belief graph visualization
 */
export function useDependencyTree(
  claimId: string | null,
  maxDepth: number = 3
): UseQueryResult<DependencyTree> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<DependencyTree | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetch = useCallback(async () => {
    if (!claimId || !client) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const tree = await client.graph.getDependencyTree(claimId, maxDepth);
      setData(tree);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [claimId, maxDepth, client]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Hook to get verification markets for a claim
 */
export function useClaimMarkets(claimId: string | null): UseQueryResult<MarketSummary[]> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<MarketSummary[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetch = useCallback(async () => {
    if (!claimId || !client) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const markets = await client.marketsIntegration.getClaimMarkets(claimId);
      setData(markets);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [claimId, client]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Hook to get information value rankings
 */
export function useInformationValueRanking(limit: number = 20): UseQueryResult<InformationValueRecord[]> {
  const client = useKnowledgeClient();
  const [data, setData] = useState<InformationValueRecord[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetch = useCallback(async () => {
    if (!client) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const rankings = await client.graph.rankByInformationValue(limit);
      setData(rankings);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [limit, client]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

// ============================================================================
// Mutation Hooks
// ============================================================================

/**
 * Hook to create a new claim
 */
export function useCreateClaim(): UseMutationResult<CreateClaimInput, string> {
  const client = useKnowledgeClient();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<string | null>(null);

  const mutate = useCallback(
    async (input: CreateClaimInput): Promise<string> => {
      if (!client) throw new Error('Knowledge client not initialized');

      setLoading(true);
      setError(null);

      try {
        const hash = await client.claims.createClaim(input);
        setData(hash);
        invalidateCache('search:');
        invalidateCache('claims:');
        return hash;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [client]
  );

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return { mutate, loading, error, data, reset };
}

/**
 * Hook to update a claim
 */
export function useUpdateClaim(): UseMutationResult<
  { originalHash: string; content?: string; classification?: { empirical: number; normative: number; mythic: number } },
  string
> {
  const client = useKnowledgeClient();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<string | null>(null);

  const mutate = useCallback(
    async (input: {
      originalHash: string;
      content?: string;
      classification?: { empirical: number; normative: number; mythic: number };
    }): Promise<string> => {
      if (!client) throw new Error('Knowledge client not initialized');

      setLoading(true);
      setError(null);

      try {
        const hash = await client.claims.updateClaim(input);
        setData(hash);
        invalidateCache(`claim:${input.originalHash}`);
        return hash;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [client]
  );

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return { mutate, loading, error, data, reset };
}

/**
 * Hook to create a relationship between claims
 */
export function useCreateRelationship(): UseMutationResult<
  {
    sourceId: string;
    targetId: string;
    relationshipType: string;
    weight: number;
  },
  string
> {
  const client = useKnowledgeClient();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<string | null>(null);

  const mutate = useCallback(
    async (input: {
      sourceId: string;
      targetId: string;
      relationshipType: string;
      weight: number;
    }): Promise<string> => {
      if (!client) throw new Error('Knowledge client not initialized');

      setLoading(true);
      setError(null);

      try {
        const hash = await client.graph.createRelationship(
          input.sourceId,
          input.targetId,
          input.relationshipType as any,
          input.weight
        );
        setData(hash);
        return hash;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [client]
  );

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return { mutate, loading, error, data, reset };
}

/**
 * Hook to spawn a verification market
 */
export function useSpawnVerificationMarket(): UseMutationResult<
  {
    claimId: string;
    targetE: number;
    minConfidence?: number;
    closesAt?: number;
    initialSubsidy?: number;
  },
  string
> {
  const client = useKnowledgeClient();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<string | null>(null);

  const mutate = useCallback(
    async (input: {
      claimId: string;
      targetE: number;
      minConfidence?: number;
      closesAt?: number;
      initialSubsidy?: number;
    }): Promise<string> => {
      if (!client) throw new Error('Knowledge client not initialized');

      setLoading(true);
      setError(null);

      try {
        const hash = await client.claims.spawnVerificationMarket({
          claimId: input.claimId,
          targetE: input.targetE,
          minConfidence: input.minConfidence ?? 0.7,
          closesAt: input.closesAt ?? Date.now() + 14 * 24 * 60 * 60 * 1000,
          initialSubsidy: input.initialSubsidy ?? 100,
          tags: [],
        });
        setData(hash);
        return hash;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [client]
  );

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return { mutate, loading, error, data, reset };
}

/**
 * Hook to trigger belief propagation
 */
export function usePropagateBelief(): UseMutationResult<string, PropagationResult> {
  const client = useKnowledgeClient();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<PropagationResult | null>(null);

  const mutate = useCallback(
    async (claimId: string): Promise<PropagationResult> => {
      if (!client) throw new Error('Knowledge client not initialized');

      setLoading(true);
      setError(null);

      try {
        const result = await client.graph.propagateBelief(claimId);
        setData(result);
        invalidateCache(); // Invalidate all caches after propagation
        return result;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [client]
  );

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return { mutate, loading, error, data, reset };
}

// ============================================================================
// Utility Hooks
// ============================================================================

/**
 * Hook to invalidate cache
 */
export function useInvalidateCache(): (pattern?: string) => void {
  return useCallback((pattern?: string) => {
    invalidateCache(pattern);
  }, []);
}

/**
 * Hook to get enriched claim with all related data
 */
export function useEnrichedClaim(claimId: string | null) {
  const claim = useClaim(claimId);
  const credibility = useCredibility(claimId);
  const relationships = useRelationships(claimId);
  const markets = useClaimMarkets(claimId);

  const loading = claim.loading || credibility.loading || relationships.loading || markets.loading;
  const error = claim.error || credibility.error || relationships.error || markets.error;

  const data = useMemo(() => {
    if (!claim.data) return null;
    return {
      claim: claim.data,
      credibility: credibility.data,
      relationships: relationships.data,
      markets: markets.data,
    };
  }, [claim.data, credibility.data, relationships.data, markets.data]);

  const refetch = useCallback(async () => {
    await Promise.all([
      claim.refetch(),
      credibility.refetch(),
      relationships.refetch(),
      markets.refetch(),
    ]);
  }, [claim, credibility, relationships, markets]);

  return { data, loading, error, refetch };
}
