import { beforeEach, describe, expect, it, vi } from 'vitest';
import { clearReputationCache, loadReputationBundle, requestReputationProof, fulfillProof, denyProof } from '$lib/reputation';
import { getMockProofTrail, getMockTrustGraph } from '$lib/mock/reputation';
import { getProofTrail, getTrustGraph, requestProof } from '$lib/holochain/reputation';
import { proofRequests } from '$lib/stores';
import type { ProofTrailItem, TrustGraphSnapshot } from '$types';

// Mock Holochain client init
vi.mock('$lib/holochain', () => ({
  initHolochainClient: vi.fn(async () => ({})),
}));

const proofTrailMock: ProofTrailItem[] = [
  {
    id: 'trail-1',
    label: 'Signed invoice',
    cid: 'bafy-mock',
    verified: true,
    issued_at: Date.now() - 1_000,
    issuer: 'peer_x',
  },
];

const graphMock: TrustGraphSnapshot = {
  subject: 'agent_alpha',
  nodes: [
    { id: 'agent_alpha', label: 'Seller', score: 95, role: 'seller', proof_count: 2 },
    { id: 'peer_x', label: 'Peer X', score: 90, role: 'buyer' },
  ],
  edges: [{ from: 'peer_x', to: 'agent_alpha', weight: 0.8, kind: 'transaction' }],
  summary: {
    score: 95,
    confidence: 0.88,
    attestations: 12,
    zk_capable: true,
    last_update: Date.now() - 500,
  },
  claims: [
    {
      id: 'claim-1',
      claim_type: 'transaction_settlement',
      issuer: 'peer_x',
      subject: 'agent_alpha',
      score: 10,
      description: 'On-time settlement',
      issued_at: Date.now() - 10_000,
    },
  ],
};

// Mock reputation zome wrappers
vi.mock('$lib/holochain/reputation', () => ({
  getTrustGraph: vi.fn(async () => graphMock),
  getProofTrail: vi.fn(async () => proofTrailMock),
  requestProof: vi.fn(async () => ({ status: 'requested' as const, message: 'live' })),
}));

describe('loadReputationBundle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    clearReputationCache();
  });

  it('returns live reputation data when zome calls succeed', async () => {
    const bundle = await loadReputationBundle('agent_alpha');
    expect(bundle.usingMock).toBe(false);
    expect(bundle.graph?.summary.score).toBe(95);
    expect(bundle.proofTrail).toHaveLength(1);
  });

  it('falls back to mock data when zome calls fail', async () => {
    vi.mocked(getTrustGraph).mockRejectedValueOnce(new Error('offline'));
    const bundle = await loadReputationBundle('agent_alpha');
    expect(bundle.usingMock).toBe(true);
    expect(bundle.graph).toEqual(getMockTrustGraph('agent_alpha'));
    expect(bundle.proofTrail).toEqual(getMockProofTrail('agent_alpha'));
  });

  it('handles missing agent id', async () => {
    const bundle = await loadReputationBundle('');
    expect(bundle.usingMock).toBe(true);
    expect(bundle.graph).toBeNull();
    expect(bundle.proofTrail).toHaveLength(0);
    expect(bundle.error).toBeDefined();
  });

  it('caches results and avoids duplicate zome calls', async () => {
    await loadReputationBundle('agent_alpha');
    await loadReputationBundle('agent_alpha');
    expect(getTrustGraph).toHaveBeenCalledTimes(1);
    expect(getProofTrail).toHaveBeenCalledTimes(1);
  });

  it('forces refresh when requested', async () => {
    await loadReputationBundle('agent_alpha');
    await loadReputationBundle('agent_alpha', { forceRefresh: true });
    expect(getTrustGraph).toHaveBeenCalledTimes(2);
  });

  it('requests proof via live zome call', async () => {
    const res = await requestReputationProof('agent_alpha', 'listing123');
    expect(res.usingMock).toBe(false);
    expect(requestProof).toHaveBeenCalledWith(expect.anything(), 'agent_alpha', 'listing123');
    const requests = await new Promise<any[]>((resolve) => {
      let unsubscribe: () => void = () => {};
      unsubscribe = proofRequests.subscribe((value) => {
        unsubscribe();
        resolve(value as any[]);
      });
    });
    expect(requests[0]).toMatchObject({ agent_id: 'agent_alpha', status: 'requested' });
  });

  it('marks proof fulfilled and denied', () => {
    const fulfilled = fulfillProof('agent_alpha', 'listing123');
    expect(fulfilled.status).toBe('fulfilled');
    const denied = denyProof('agent_beta', undefined, 'rejected');
    expect(denied.status).toBe('denied');
  });
});
