import { initHolochainClient } from '$lib/holochain';
import { getProofTrail, getTrustGraph, requestProof } from '$lib/holochain/reputation';
import { getMockProofTrail, getMockTrustGraph, requestMockProof } from '$lib/mock/reputation';
import { markProofDenied, markProofFulfilled, markProofRequest } from '$lib/stores';
import type { ProofTrailItem, TrustGraphSnapshot } from '$types';

interface ReputationBundle {
  graph: TrustGraphSnapshot | null;
  proofTrail: ProofTrailItem[];
  usingMock: boolean;
  error?: string;
}

type LoadOptions = {
  /** Force bypassing cache */
  forceRefresh?: boolean;
  /** Custom cache TTL in ms (default: 2 minutes) */
  ttlMs?: number;
};

const cache = new Map<
  string,
  {
    expiresAt: number;
    bundle: ReputationBundle;
  }
>();

const DEFAULT_TTL_MS = 2 * 60 * 1000;

/**
 * Load a subject's trust graph and proof trail with graceful mock fallback.
 */
export async function loadReputationBundle(
  agentId: string,
  options: LoadOptions = {}
): Promise<ReputationBundle> {
  if (!agentId) {
    return { graph: null, proofTrail: [], usingMock: true, error: 'Missing agent id' };
  }

  const ttl = options.ttlMs ?? DEFAULT_TTL_MS;

  // Serve from cache when fresh
  if (!options.forceRefresh) {
    const cached = cache.get(agentId);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.bundle;
    }
  }

  try {
    const client = await initHolochainClient();
    const [graph, proofTrail] = await Promise.all([
      getTrustGraph(client, agentId),
      getProofTrail(client, agentId),
    ]);

    const bundle = { graph, proofTrail, usingMock: false };
    cache.set(agentId, { bundle, expiresAt: Date.now() + ttl });
    return bundle;
  } catch (error: any) {
    console.warn('Falling back to mock reputation data', error);
    const bundle = {
      graph: getMockTrustGraph(agentId),
      proofTrail: getMockProofTrail(agentId),
      usingMock: true,
      error: error?.message,
    };
    cache.set(agentId, { bundle, expiresAt: Date.now() + ttl });
    return bundle;
  }
}

/**
 * Clear in-memory cache (useful for tests)
 */
export function clearReputationCache() {
  cache.clear();
}

/**
 * Request a proof/attestation with live zome call, fallback to mock.
 */
export async function requestReputationProof(agentId: string, listingHash?: string) {
  if (!agentId) {
    return {
      usingMock: true,
      message: 'Missing agent id',
      status: 'error' as const,
    };
  }

  try {
    const client = await initHolochainClient();
    const result = await requestProof(client, agentId, listingHash);
    markProofRequest(agentId, listingHash, 'requested', result.message, false);
    return { ...result, usingMock: false };
  } catch (error: any) {
    console.warn('Falling back to mock proof request', error);
    const result = await requestMockProof(agentId, listingHash);
    markProofRequest(agentId, listingHash, 'requested', result.message, true);
    return { ...result, usingMock: true, error: error?.message };
  }
}

/**
 * Accept fulfillment signal and mark proof as fulfilled.
 */
export function fulfillProof(agentId: string, listingHash?: string, message?: string) {
  markProofFulfilled(agentId, listingHash);
  return {
    status: 'fulfilled' as const,
    message: message || 'Proof published',
  };
}

/**
 * Mark proof denied.
 */
export function denyProof(agentId: string, listingHash?: string, message?: string) {
  markProofDenied(agentId, listingHash, message);
  return {
    status: 'denied' as const,
    message: message || 'Proof denied',
  };
}
