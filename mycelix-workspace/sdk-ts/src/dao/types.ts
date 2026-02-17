/**
 * Type Definitions for Mycelix-DAO
 *
 * All TypeScript types corresponding to Rust entry types.
 */

import { type ActionHash, type AgentPubKey, type Timestamp, type Record as HolochainRecord } from '@holochain/client';

// Re-export for convenience
export type { ActionHash, AgentPubKey, Timestamp, HolochainRecord };

// ============================================================================
// GOVERNANCE TYPES
// ============================================================================

/** Types of proposals */
export type ProposalType =
  | 'TechnicalMIP'
  | 'EconomicMIP'
  | 'GovernanceMIP'
  | 'SocialMIP'
  | 'CulturalMIP'
  | 'ConstitutionalAmendment'
  | 'TreasuryAllocation'
  | 'StreamingGrant'
  | 'MRCAppeal'
  | 'Impeachment'
  | 'EmergencyAction'
  | 'LocalDAODecision';

/** Voting mechanisms */
export type VotingMechanism =
  | { ReputationWeighted: { quorum: number; approval_threshold: number } }
  | { EqualWeight: { quorum: number; approval_threshold: number } }
  | { Quadratic: { budget_per_voter: number; approval_threshold: number } }
  | { Conviction: { threshold: number; decay_rate: number; min_conviction_time_hours: number } };

/** Proposal lifecycle status */
export type ProposalStatus =
  | 'Draft'
  | 'Review'
  | 'Voting'
  | 'Passed'
  | 'Failed'
  | 'Executed'
  | 'Cancelled'
  | 'Disputed'
  | 'Vetoed';

/** Vote choices */
export type VoteChoice = 'Yes' | 'No' | 'Abstain';

/** Delegation domains */
export type DelegationDomain =
  | 'All'
  | 'Governance'
  | 'Technical'
  | 'Economic'
  | 'Social'
  | 'Cultural';

// ============================================================================
// EPISTEMIC TYPES
// ============================================================================

/** Empirical verifiability levels (E-axis) */
export type EmpiricalLevel = 'E0' | 'E1' | 'E2' | 'E3' | 'E4';

/** Normative authority levels (N-axis) */
export type NormativeLevel = 'N0' | 'N1' | 'N2' | 'N3';

/** Materiality levels (M-axis) */
export type MaterialityLevel = 'M0' | 'M1' | 'M2' | 'M3';

/** 3D Epistemic Classification */
export interface EpistemicClassification {
  e: EmpiricalLevel;
  n: NormativeLevel;
  m: MaterialityLevel;
  justification?: string;
  override_from_default: boolean;
}

/** Verification methods */
export type VerificationMethod =
  | 'None'
  | 'Attestation'
  | 'AuditReview'
  | 'Signature'
  | 'ZKProof'
  | 'PublicReproduction';

/** Verification states */
export type VerificationState =
  | 'Pending'
  | 'Verified'
  | 'Disputed'
  | 'Superceded'
  | 'Invalid';

/** Dispute targets */
export type DisputeTarget =
  | { Proposal: ActionHash }
  | { Claim: ActionHash }
  | { Vote: ActionHash }
  | { MemberAction: ActionHash };

/** Disputed axes */
export type DisputedAxis = 'Empirical' | 'Normative' | 'Materiality' | 'Procedural';

/** Resolution bodies */
export type ResolutionBody =
  | 'MemberRedressCouncil'
  | 'KnowledgeCouncil'
  | 'AuditGuild'
  | 'ConstitutionalProcess'
  | { SectorDAO: string };

/** Claim relationship types */
export type ClaimRelationType =
  | 'Supports'
  | 'Refutes'
  | 'Clarifies'
  | 'Supercedes'
  | 'Implements'
  | 'Disputes';

// ============================================================================
// IDENTITY TYPES
// ============================================================================

/** Member status */
export type MemberStatus = 'Active' | 'Inactive' | 'Suspended' | 'Banned';

// ============================================================================
// OVERSIGHT TYPES
// ============================================================================

/** Oversight bodies */
export type OversightBody =
  | 'KnowledgeCouncil'
  | 'AuditGuild'
  | 'MemberRedressCouncil'
  | 'GuardianNetwork';

/** Case types */
export type CaseType =
  | 'MemberGrievance'
  | 'AuditFinding'
  | 'EpistemicDispute'
  | 'ConstitutionalReview'
  | 'EmergencyIntervention'
  | 'CartelInvestigation';

/** Case status */
export type CaseStatus =
  | 'Filed'
  | 'Assigned'
  | 'UnderReview'
  | 'Deliberating'
  | 'Decided'
  | 'Appealed'
  | 'Closed';

/** Guardian action types */
export type GuardianActionType =
  | 'PauseProposal'
  | 'UnpauseProposal'
  | 'EmergencyVeto'
  | 'Escalate'
  | 'FreezeFunds';

// ============================================================================
// TREASURY TYPES
// ============================================================================

/** Allocation recipients */
export type AllocationRecipient =
  | { Member: AgentPubKey }
  | { OversightBody: OversightBody }
  | { External: string }
  | { MultiSig: { signers: AgentPubKey[]; threshold: number } };

/** Vesting frequency */
export type VestingFrequency = 'Monthly' | 'Quarterly' | 'Yearly' | 'AtEnd';

/** Allocation status */
export type AllocationStatus =
  | 'Approved'
  | 'Vesting'
  | 'FullyDistributed'
  | 'Cancelled'
  | 'Clawedback';

// ============================================================================
// FULL ENTRY TYPES
// ============================================================================

/** A governance proposal */
export interface Proposal {
  id: string;
  proposal_type: ProposalType;
  title: string;
  description: string;
  classification: EpistemicClassification;
  voting_mechanism: VotingMechanism;
  status: ProposalStatus;
  voting_starts: Timestamp;
  voting_ends: Timestamp;
  quorum: number;
  approval_threshold: number;
  conviction_threshold?: number;
  funding_request?: number;
  author: AgentPubKey;
  created_at: Timestamp;
  implementation_url?: string;
  tags: string[];
}

/** A vote on a proposal */
export interface Vote {
  proposal: ActionHash;
  voter: AgentPubKey;
  choice: VoteChoice;
  weight: number;
  justification?: string;
  timestamp: Timestamp;
  delegated_power: number;
}

/** Conviction vote (time-weighted) */
export interface ConvictionVote {
  proposal: ActionHash;
  voter: AgentPubKey;
  weight: number;
  staked_at: Timestamp;
  withdrawn_at?: Timestamp;
  active: boolean;
}

/** Quadratic vote allocation */
export interface QuadraticVote {
  proposal: ActionHash;
  voter: AgentPubKey;
  credits_spent: number;
  effective_votes: number;
  round_id: string;
  timestamp: Timestamp;
}

/** Delegation of voting power */
export interface Delegation {
  delegator: AgentPubKey;
  delegate: AgentPubKey;
  domain: DelegationDomain;
  created_at: Timestamp;
  revoked_at?: Timestamp;
  active: boolean;
}

/** DKG Claim */
export interface DKGClaim {
  id: string;
  content: string;
  classification: EpistemicClassification;
  author: AgentPubKey;
  timestamp: Timestamp;
  verification: {
    method: VerificationMethod;
    status: VerificationState;
    proof_cid?: string;
    verified_at?: Timestamp;
    verifier?: AgentPubKey;
  };
  related_claims: Array<{ relation_type: ClaimRelationType; claim_id: string }>;
  proof_cid?: string;
}

/** Member profile */
export interface MemberProfile {
  agent: AgentPubKey;
  display_name: string;
  bio?: string;
  verification_level: EmpiricalLevel;
  sybil_score: number;
  reputation_aggregate: number;
  joined_at: Timestamp;
  status: MemberStatus;
  consistency_score: number;
  proposals_authored: number;
  votes_cast: number;
  delegations_received: number;
}

// ============================================================================
// RESULT TYPES
// ============================================================================

/** Vote tally result */
export interface VoteTally {
  yes_votes: number;
  no_votes: number;
  abstain_votes: number;
  total_weight: number;
  unique_voters: number;
  quorum_reached: boolean;
  approval_reached: boolean;
  result: 'Pending' | 'Passed' | 'Failed' | 'NoQuorum';
}

/** Conviction summary */
export interface ConvictionSummary {
  active_conviction: number;
  decaying_conviction: number;
  total_conviction: number;
  threshold: number;
  active_stakers: number;
  total_stakers: number;
  passes: boolean;
  progress_percentage: number;
}

/** Delegated power result */
export interface DelegatedPowerResult {
  direct_power: number;
  transitive_power: number;
  total_power: number;
  delegator_count: number;
}

/** Cross-hApp reputation */
export interface CrossHappReputation {
  agent: AgentPubKey;
  mail_reputation?: number;
  marketplace_reputation?: number;
  edunet_reputation?: number;
  supplychain_reputation?: number;
  aggregate: number;
}

/** Treasury summary */
export interface TreasurySummary {
  total_allocated: number;
  total_distributed: number;
  remaining: number;
  active_allocations: number;
}

// ============================================================================
// INPUT TYPES
// ============================================================================

/** Input for creating a proposal */
export interface CreateProposalInput {
  proposal_type: ProposalType;
  title: string;
  description: string;
  classification: EpistemicClassification;
  voting_mechanism?: VotingMechanism;
  voting_period_hours?: number;
  funding_request?: number;
  implementation_url?: string;
  tags: string[];
}

/** Input for casting a vote */
export interface CastVoteInput {
  proposal: ActionHash;
  choice: VoteChoice;
  justification?: string;
}

/** Input for quadratic voting */
export interface CastQuadraticVoteInput {
  proposal: ActionHash;
  credits: number;
  round_id: string;
}

/** Input for creating a delegation */
export interface CreateDelegationInput {
  delegate: AgentPubKey;
  domain: DelegationDomain;
}

/** Input for filing a dispute */
export interface FileDisputeInput {
  target: DisputeTarget;
  disputed_axis: DisputedAxis;
  justification: string;
}

/** Input for storing a DKG claim */
export interface StoreDKGClaimInput {
  content: string;
  classification: EpistemicClassification;
  verification_method?: VerificationMethod;
  proof_cid?: string;
  related_claims?: Array<{ relation_type: ClaimRelationType; claim_id: string }>;
}
