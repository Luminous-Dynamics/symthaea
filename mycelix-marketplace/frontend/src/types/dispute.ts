/**
 * Dispute Type Definitions
 *
 * Core data structures for MRC (Multi-Resonance Council) dispute resolution
 */

/**
 * Dispute status in resolution lifecycle
 */
export type DisputeStatus = 'pending' | 'active' | 'resolved' | 'rejected';

/**
 * Dispute reason categories
 */
export type DisputeReason =
  | 'defective_product'
  | 'wrong_item'
  | 'non_delivery'
  | 'not_as_described'
  | 'damaged_in_shipping'
  | 'seller_unresponsive'
  | 'other';

/**
 * Arbitrator vote
 */
export type ArbitratorVote = 'Approve' | 'Reject' | null;

/**
 * Arbitrator information
 */
export interface Arbitrator {
  /** Arbitrator agent ID */
  agent_id: string;

  /** Arbitrator username */
  username: string;

  /** PoGQ trust score (0-1 or 0-100) */
  pogq_score: number;

  /** Total cases arbitrated */
  cases_arbitrated: number;

  /** Arbitrator since timestamp */
  member_since: number;

  /** Whether currently active */
  is_active: boolean;
}

/**
 * Arbitrator vote record
 */
export interface ArbitratorVoteRecord {
  /** Arbitrator agent ID */
  arbitrator_id: string;

  /** Arbitrator username */
  arbitrator_name: string;

  /** Arbitrator PoGQ score */
  arbitrator_pogq: number;

  /** Vote cast */
  vote: ArbitratorVote;

  /** Vote reasoning */
  reasoning: string;

  /** Vote timestamp */
  voted_at: number;

  /** Vote weight (based on PoGQ) */
  vote_weight: number;
}

/**
 * Complete dispute data structure
 */
export interface Dispute {
  /** Dispute claim ID */
  claim_id: string;

  /** Associated transaction hash */
  transaction_hash: string;

  /** Transaction/listing title */
  title: string;

  /** Transaction/listing photo CID */
  photo_cid?: string;

  /** Buyer (claimant) agent ID */
  buyer_agent_id: string;

  /** Buyer username */
  buyer_name: string;

  /** Seller (respondent) agent ID */
  seller_agent_id: string;

  /** Seller username */
  seller_name: string;

  /** Seller trust score */
  seller_trust_score: number;

  /** Dispute reason */
  reason: DisputeReason;

  /** Detailed description */
  description: string;

  /** Evidence IPFS CIDs (photos, documents) */
  evidence_ipfs_cids: string[];

  /** Current status */
  status: DisputeStatus;

  /** Arbitrator panel assigned */
  arbitrators: Arbitrator[];

  /** Votes cast so far */
  votes: ArbitratorVoteRecord[];

  /** Number of votes cast */
  votes_cast: number;

  /** Number of votes required for consensus (66% of voting power) */
  votes_required: number;

  /** Total voting power of panel */
  total_voting_power: number;

  /** Current approval voting power */
  approval_voting_power: number;

  /** Current rejection voting power */
  rejection_voting_power: number;

  /** Whether consensus has been reached */
  consensus_reached: boolean;

  /** Final decision (if resolved) */
  final_decision?: 'approved' | 'rejected';

  /** Dispute filed timestamp */
  created_at: number;

  /** Dispute resolution timestamp */
  resolved_at?: number;

  /** Refund amount (if approved) */
  refund_amount?: number;

  /** My vote (if I'm an arbitrator) */
  my_vote?: ArbitratorVote;
}

/**
 * Dispute creation input
 */
export interface CreateDisputeInput {
  transaction_hash: string;
  reason: DisputeReason;
  description: string;
  evidence_ipfs_cids: string[];
}

/**
 * Arbitrator vote input
 */
export interface CastVoteInput {
  claim_id: string;
  vote: 'Approve' | 'Reject';
  reasoning: string;
}

/**
 * Dispute filter criteria
 */
export interface DisputeFilters {
  /** Filter by status */
  status?: DisputeStatus | 'all';

  /** Filter by assigned arbitrator */
  arbitrator_id?: string;
}

/**
 * Arbitrator profile for MRC interface
 */
export interface ArbitratorProfile extends Arbitrator {
  /** Number of pending cases */
  pending_cases: number;

  /** Number of active cases */
  active_cases: number;

  /** Number of resolved cases */
  resolved_cases: number;

  /** Approval rate (percentage) */
  approval_rate: number;

  /** Average resolution time (hours) */
  average_resolution_time: number;
}

/**
 * MRC statistics
 */
export interface MRCStats {
  /** Total disputes filed */
  total_disputes: number;

  /** Disputes resolved */
  disputes_resolved: number;

  /** Average resolution time (hours) */
  average_resolution_time: number;

  /** Consensus rate (percentage) */
  consensus_rate: number;

  /** Total active arbitrators */
  active_arbitrators: number;
}
