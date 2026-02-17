/**
 * Reputation, proof, and trust graph types
 *
 * These power the verifiable reputation surface (claims, proofs, graph edges).
 */

/** Claim/credential describing a trust event */
export interface ReputationClaim {
  /** Unique claim id (hash or action hash) */
  id: string;
  /** Claim category */
  claim_type:
    | 'transaction_settlement'
    | 'arbitration_award'
    | 'rating'
    | 'identity_verification'
    | 'milestone_delivery';
  /** Agent issuing the claim */
  issuer: string;
  /** Agent the claim is about */
  subject: string;
  /** Weighted score contribution (0-100) */
  score: number;
  /** Human-friendly context */
  description: string;
  /** When it was issued (ms) */
  issued_at: number;
  /** When it expires (ms) */
  expires_at?: number;
  /** Optional IPFS CID for supporting evidence */
  evidence_cid?: string;
  /** Signature for verification */
  signature?: string;
  /** Optional zero-knowledge range proof metadata */
  zk_range_proof?: {
    lower_bound: number;
    upper_bound: number;
    curve: string;
    statement: string;
  };
}

/** A single hop in the trust graph */
export interface TrustGraphEdge {
  from: string;
  to: string;
  /** Strength 0-1 */
  weight: number;
  kind: 'review' | 'arbitration' | 'transaction';
  evidence_cid?: string;
}

/** Agent node in the trust graph */
export interface TrustGraphNode {
  id: string;
  label: string;
  score: number;
  role?: 'buyer' | 'seller' | 'arbitrator' | 'admin';
  proof_count?: number;
  recent_attestations?: number;
}

/** Snapshot for a subject agent */
export interface TrustGraphSnapshot {
  subject: string;
  nodes: TrustGraphNode[];
  edges: TrustGraphEdge[];
  summary: {
    score: number;
    confidence: number;
    attestations: number;
    zk_capable: boolean;
    last_update: number;
  };
  claims: ReputationClaim[];
}

/** Proof trail item (hashed docs or IPFS CIDs) */
export interface ProofTrailItem {
  id: string;
  label: string;
  cid?: string;
  hash?: string;
  verified: boolean;
  issued_at: number;
  issuer: string;
}
