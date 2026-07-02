import type { ProofTrailItem, ReputationClaim, TrustGraphSnapshot } from '$types';

const now = Date.now();

const mockClaims: ReputationClaim[] = [
  {
    id: 'claim_txn_alpha',
    claim_type: 'transaction_settlement',
    issuer: 'peer_henry',
    subject: 'agent_alpha',
    score: 12,
    description: 'Settled escrow on-time with milestone proofs',
    issued_at: now - 1000 * 60 * 60 * 24 * 6,
    evidence_cid: 'bafy-alpha-proof',
    signature: 'sig-alpha',
    zk_range_proof: {
      lower_bound: 90,
      upper_bound: 100,
      curve: 'BLS12-381',
      statement: 'score >= 0.9',
    },
  },
  {
    id: 'claim_arbitration_beta',
    claim_type: 'arbitration_award',
    issuer: 'mrc_council',
    subject: 'agent_alpha',
    score: 8,
    description: 'Arbitrated dispute with 3/3 vote consensus',
    issued_at: now - 1000 * 60 * 60 * 24 * 14,
    evidence_cid: 'bafy-arb-proof',
    signature: 'sig-beta',
  },
  {
    id: 'claim_rating_gamma',
    claim_type: 'rating',
    issuer: 'peer_isabel',
    subject: 'agent_alpha',
    score: 6,
    description: '5-star post-purchase review (verified delivery)',
    issued_at: now - 1000 * 60 * 60 * 24 * 3,
    signature: 'sig-gamma',
  },
];

const mockProofTrail: ProofTrailItem[] = [
  {
    id: 'trail_delivery_receipt',
    label: 'Delivery receipt hash',
    hash: '0xabc123delivery',
    verified: true,
    issued_at: now - 1000 * 60 * 60 * 24 * 6,
    issuer: 'logistics_oracle',
  },
  {
    id: 'trail_ipfs_invoice',
    label: 'Signed invoice (IPFS)',
    cid: 'bafy-invoice',
    verified: true,
    issued_at: now - 1000 * 60 * 60 * 24 * 2,
    issuer: 'peer_isabel',
  },
  {
    id: 'trail_kyc',
    label: 'Lightweight self-attested ID',
    hash: '0xkyc123',
    verified: false,
    issued_at: now - 1000 * 60 * 60 * 24 * 30,
    issuer: 'agent_alpha',
  },
];

const mockGraph: TrustGraphSnapshot = {
  subject: 'agent_alpha',
  nodes: [
    { id: 'agent_alpha', label: 'Seller', score: 94, role: 'seller', proof_count: 3, recent_attestations: 4 },
    { id: 'peer_henry', label: 'Peer Henry', score: 91, role: 'buyer', proof_count: 1, recent_attestations: 1 },
    { id: 'peer_isabel', label: 'Peer Isabel', score: 88, role: 'buyer', proof_count: 1, recent_attestations: 1 },
    { id: 'mrc_council', label: 'MRC Council', score: 96, role: 'arbitrator', proof_count: 1, recent_attestations: 1 },
  ],
  edges: [
    { from: 'peer_henry', to: 'agent_alpha', weight: 0.82, kind: 'transaction', evidence_cid: 'bafy-alpha-proof' },
    { from: 'peer_isabel', to: 'agent_alpha', weight: 0.74, kind: 'review' },
    { from: 'mrc_council', to: 'agent_alpha', weight: 0.91, kind: 'arbitration', evidence_cid: 'bafy-arb-proof' },
  ],
  summary: {
    score: 94,
    confidence: 0.86,
    attestations: 24,
    zk_capable: true,
    last_update: now - 1000 * 60 * 5,
  },
  claims: mockClaims,
};

export function getMockTrustGraph(agentId: string): TrustGraphSnapshot {
  if (agentId === 'agent_alpha') return mockGraph;
  return {
    ...mockGraph,
    subject: agentId,
    nodes: mockGraph.nodes.map((node) =>
      node.id === 'agent_alpha'
        ? { ...node, id: agentId, label: 'Seller', score: 90 }
        : node
    ),
    edges: mockGraph.edges.map((edge) =>
      edge.to === 'agent_alpha' ? { ...edge, to: agentId } : edge
    ),
  };
}

export function getMockProofTrail(agentId: string): ProofTrailItem[] {
  return mockProofTrail.map((entry) => ({
    ...entry,
    id: `${entry.id}_${agentId}`,
  }));
}

export function getMockClaims(agentId: string): ReputationClaim[] {
  return mockClaims.map((claim) => ({
    ...claim,
    subject: agentId,
    id: `${claim.id}_${agentId}`,
  }));
}

export async function requestMockProof(agentId: string, listingHash?: string) {
  await new Promise((resolve) => setTimeout(resolve, 400));
  return {
    status: 'requested' as const,
    message: listingHash
      ? `Proof request recorded for listing ${listingHash}.`
      : `Proof request recorded for ${agentId}.`,
  };
}
