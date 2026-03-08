/**
 * Mycelix Governance SDK Types
 *
 * Type definitions for the Governance hApp SDK, providing DAOs, proposals,
 * MATL-weighted voting, and delegation capabilities.
 *
 * @module @mycelix/sdk/integrations/governance
 */

// Holochain types are referenced structurally via string identifiers in this module

// ============================================================================
// DAO Types
// ============================================================================

/**
 * Decentralized Autonomous Organization
 */
export interface Dao {
  /** Unique DAO identifier */
  id: string;
  /** Human-readable name */
  name: string;
  /** DAO description and purpose */
  description: string;
  /** Constitutional rules and governance policies */
  constitution: string;
  /** DID of the DAO founder */
  founder_did: string;
  /** Optional treasury wallet identifier */
  treasury_wallet_id?: string;
  /** Current member count */
  member_count: number;
  /** Creation timestamp (microseconds) */
  created: number;
  /** Last update timestamp (microseconds) */
  updated: number;
}

/**
 * Input for creating a new DAO
 */
export interface CreateDaoInput {
  /** Unique DAO identifier */
  id: string;
  /** Human-readable name */
  name: string;
  /** DAO description and purpose */
  description: string;
  /** Constitutional rules and governance policies */
  constitution: string;
  /** DID of the DAO founder (must match caller) */
  founder_did: string;
  /** Optional treasury wallet identifier */
  treasury_wallet_id?: string;
}

/**
 * Input for updating DAO settings
 */
export interface UpdateDaoInput {
  /** DAO identifier to update */
  dao_id: string;
  /** New name (optional) */
  name?: string;
  /** New description (optional) */
  description?: string;
  /** New treasury wallet ID (optional) */
  treasury_wallet_id?: string;
}

// ============================================================================
// Membership Types
// ============================================================================

/**
 * DAO membership record
 */
export interface Membership {
  /** DAO identifier */
  dao_id: string;
  /** Member's DID */
  member_did: string;
  /** Member's role in the DAO */
  role: MemberRole;
  /** Base voting power */
  voting_power: number;
  /** Timestamp when member joined (microseconds) */
  joined_at: number;
}

/**
 * Member roles with different privileges
 */
export type MemberRole = 'Member' | 'Moderator' | 'Admin' | 'Founder';

// ============================================================================
// Proposal Types
// ============================================================================

/**
 * Governance proposal
 */
export interface Proposal {
  /** Unique proposal identifier */
  id: string;
  /** DAO this proposal belongs to */
  dao_id: string;
  /** Proposal title */
  title: string;
  /** Detailed description */
  description: string;
  /** DID of the proposer */
  proposer_did: string;
  /** Type of proposal (affects voting period) */
  proposal_type: ProposalType;
  /** Current status */
  status: ProposalStatus;
  /** Voting period start (microseconds) */
  voting_starts: number;
  /** Voting period end (microseconds) */
  voting_ends: number;
  /** Required quorum percentage */
  quorum_percentage: number;
  /** Required approval threshold percentage */
  approval_threshold: number;
  /** Total yes votes (weighted) */
  yes_votes: number;
  /** Total no votes (weighted) */
  no_votes: number;
  /** Total abstain votes (weighted) */
  abstain_votes: number;
  /** Optional execution payload (JSON) */
  execution_payload?: string;
  /** Creation timestamp (microseconds) */
  created: number;
}

/**
 * Proposal types with different voting periods
 * - Standard: 7 days
 * - Emergency: 24 hours
 * - Constitutional: 30 days
 */
export type ProposalType = 'Standard' | 'Emergency' | 'Constitutional';

/**
 * Proposal lifecycle status
 */
export type ProposalStatus =
  | 'Draft'
  | 'Active'
  | 'Passed'
  | 'Failed'
  | 'Executed'
  | 'Vetoed'
  | 'Cancelled';

/**
 * Input for creating a proposal
 */
export interface CreateProposalInput {
  /** Unique proposal identifier */
  id: string;
  /** DAO this proposal belongs to */
  dao_id: string;
  /** Proposal title */
  title: string;
  /** Detailed description */
  description: string;
  /** DID of the proposer (must match caller) */
  proposer_did: string;
  /** Type of proposal */
  proposal_type: ProposalType;
  /** Required quorum percentage (0-100) */
  quorum_percentage: number;
  /** Required approval threshold percentage (0-100) */
  approval_threshold: number;
  /** Optional execution payload (JSON) */
  execution_payload?: string;
}

/**
 * Query parameters for listing proposals
 */
export interface DaoProposalsQuery {
  /** DAO identifier */
  dao_id: string;
  /** Filter by status (optional) */
  status_filter?: ProposalStatus;
  /** Maximum number of results (optional) */
  limit?: number;
}

/**
 * Proposal finalization result
 */
export interface ProposalResult {
  /** Proposal identifier */
  proposal_id: string;
  /** Final status */
  status: ProposalStatus;
  /** Total yes votes */
  yes_votes: number;
  /** Total no votes */
  no_votes: number;
  /** Total abstain votes */
  abstain_votes: number;
  /** Whether quorum was met */
  quorum_met: boolean;
  /** Approval percentage */
  approval_percentage: number;
  /** Whether proposal passed */
  passed: boolean;
}

// ============================================================================
// Voting Types
// ============================================================================

/**
 * Vote on a proposal
 */
export interface Vote {
  /** Unique vote identifier */
  id: string;
  /** Proposal being voted on */
  proposal_id: string;
  /** DID of the voter */
  voter_did: string;
  /** Vote choice */
  choice: VoteChoice;
  /** Voting weight (MATL-adjusted) */
  weight: number;
  /** Optional reasoning for the vote */
  reason?: string;
  /** If voting on behalf of a delegator */
  delegate_from?: string;
  /** Vote timestamp (microseconds) */
  created: number;
}

/**
 * Vote choice options
 */
export type VoteChoice = 'Yes' | 'No' | 'Abstain';

/**
 * Input for casting a vote
 */
export interface CastVoteInput {
  /** Unique vote identifier */
  id: string;
  /** Proposal being voted on */
  proposal_id: string;
  /** DID of the voter (must match caller) */
  voter_did: string;
  /** Vote choice */
  choice: VoteChoice;
  /** Claimed voting weight */
  weight: number;
  /** Optional reasoning */
  reason?: string;
  /** If voting on behalf of a delegator */
  delegate_from?: string;
}

/**
 * Query for voting power calculation
 */
export interface VotingPowerQuery {
  /** DAO identifier */
  dao_id: string;
  /** Voter's DID */
  voter_did: string;
}

/**
 * Input for MATL-weighted voting power calculation
 */
export interface MatlVotingPowerInput {
  /** DAO identifier */
  dao_id: string;
  /** Voter's DID */
  voter_did: string;
  /** MATL trust score (0-1) */
  matl_score: number;
  /** Participation rate (0-1) */
  participation_rate: number;
  /** Stake amount (if staking enabled) */
  stake_amount: number;
}

/**
 * Detailed voting power breakdown
 */
export interface VotingPowerResult {
  /** Final calculated voting power */
  voting_power: number;
  /** Base power from membership */
  base_power: number;
  /** MATL reputation multiplier */
  matl_multiplier: number;
  /** Bonus for participation */
  participation_bonus: number;
  /** Multiplier from staking */
  stake_multiplier: number;
}

/**
 * Quorum status for a proposal
 */
export interface QuorumStatus {
  /** Proposal identifier */
  proposal_id: string;
  /** Total votes received */
  total_votes: number;
  /** Required votes for quorum */
  required_votes: number;
  /** Whether quorum is met */
  quorum_met: boolean;
  /** Percentage of quorum achieved */
  percentage_of_quorum: number;
}

/**
 * Voting statistics for a proposal
 */
export interface VotingStats {
  /** Proposal identifier */
  proposal_id: string;
  /** Yes vote count */
  yes_votes: number;
  /** No vote count */
  no_votes: number;
  /** Abstain vote count */
  abstain_votes: number;
  /** Total vote count */
  total_votes: number;
  /** Yes percentage */
  yes_percentage: number;
  /** No percentage */
  no_percentage: number;
  /** Abstain percentage */
  abstain_percentage: number;
  /** Voting end timestamp */
  voting_ends: number;
  /** Whether proposal is in active voting */
  is_active: boolean;
}

// ============================================================================
// Delegation Types
// ============================================================================

/**
 * Vote delegation record
 */
export interface Delegation {
  /** Unique delegation identifier */
  id: string;
  /** DID of the person delegating */
  delegator_did: string;
  /** DID of the delegate */
  delegate_did: string;
  /** DAO this delegation applies to */
  dao_id: string;
  /** Scope of delegation */
  scope: DelegationScope;
  /** Category filter (if scope is 'Category') */
  category?: string;
  /** Expiration timestamp (optional) */
  expires_at?: number;
  /** Whether delegation is currently active */
  active: boolean;
  /** Creation timestamp (microseconds) */
  created: number;
}

/**
 * Delegation scope
 * - All: Delegate all voting power
 * - Category: Only delegate for specific proposal categories
 */
export type DelegationScope = 'All' | 'Category';

/**
 * Query for delegated power
 */
export interface DelegatedPowerQuery {
  /** DAO identifier */
  dao_id: string;
  /** Delegate's DID */
  delegate_did: string;
}

/**
 * Query to check if someone has delegated
 */
export interface HasDelegatedQuery {
  /** DAO identifier */
  dao_id: string;
  /** Delegator's DID */
  delegator_did: string;
}

// ============================================================================
// Cross-hApp Integration Types
// ============================================================================

/**
 * Participation score for cross-hApp reputation
 */
export interface ParticipationScore {
  /** DID of the participant */
  did: string;
  /** Number of proposals created */
  proposals_created: number;
  /** Number of votes cast */
  votes_cast: number;
  /** Participation rate (0-1) */
  participation_rate: number;
  /** Alignment with winning outcomes (0-1) */
  average_alignment: number;
  /** Trust score from delegators (0-1) */
  delegation_trust: number;
}

/**
 * Governance event for cross-hApp notification
 */
export interface GovernanceEvent {
  /** Type of event */
  event_type: GovernanceEventType;
  /** DAO identifier */
  dao_id: string;
  /** Subject DID (if applicable) */
  subject_did?: string;
  /** Proposal ID (if applicable) */
  proposal_id?: string;
  /** Additional event data */
  payload: Record<string, unknown>;
}

/**
 * Types of governance events
 */
export type GovernanceEventType =
  | 'DaoCreated'
  | 'ProposalCreated'
  | 'ProposalActivated'
  | 'ProposalPassed'
  | 'ProposalFailed'
  | 'VoteCast'
  | 'DelegationCreated'
  | 'MemberJoined';

// ============================================================================
// Error Types
// ============================================================================

/**
 * Governance SDK error codes
 */
export type GovernanceSdkErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_ERROR'
  | 'INVALID_INPUT'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'NOT_MEMBER'
  | 'ALREADY_VOTED'
  | 'VOTING_CLOSED'
  | 'QUORUM_NOT_MET'
  | 'PROPOSAL_NOT_ACTIVE'
  | 'INVALID_DELEGATION';

/**
 * Governance SDK error
 */
export class GovernanceSdkError extends Error {
  constructor(
    public readonly code: GovernanceSdkErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'GovernanceSdkError';
  }
}

// ============================================================================
// Holochain Zome Types — mirror Rust entry types from governance coordinator
// ============================================================================

// --- Proposal Tier & Phi-Weighted Voting ---

/** Proposal tier determines thresholds and timelock duration */
export type ProposalTier = 'Basic' | 'Major' | 'Constitutional';

/** Input for Φ-weighted vote tally */
export interface TallyPhiVotesInput {
  proposal_id: string;
  tier: ProposalTier;
  eligible_voters?: number;
  generate_reflection?: boolean;
}

/** Φ-weighted tally result */
export interface PhiWeightedTally {
  proposal_id: string;
  tier: ProposalTier;
  phi_votes_for: number;
  phi_votes_against: number;
  phi_abstentions: number;
  raw_votes_for: number;
  raw_votes_against: number;
  raw_abstentions: number;
  average_phi: number;
  total_phi_weight: number;
  eligible_voters: number;
  quorum_requirement: number;
  quorum_reached: boolean;
  approval_threshold: number;
  approved: boolean;
  /** Timestamp (microseconds) */
  tallied_at: number;
  final_tally: boolean;
  phi_tier_breakdown: PhiTierBreakdown;
  phi_enhanced_count: number;
  reputation_only_count: number;
  phi_coverage: number;
}

export interface PhiTierBreakdown {
  high_phi_votes: TallySegment;
  medium_phi_votes: TallySegment;
  low_phi_votes: TallySegment;
}

export interface TallySegment {
  votes_for: number;
  votes_against: number;
  abstentions: number;
  voter_count: number;
}

// --- Quadratic Voting ---

/** Input for allocating voice credits */
export interface AllocateCreditsInput {
  owner_did: string;
  amount: number;
  /** Period end timestamp (microseconds) */
  period_end: number;
}

/** Voice credit balance */
export interface VoiceCredits {
  owner: string;
  allocated: number;
  spent: number;
  remaining: number;
  period_start: number;
  period_end: number;
}

/** Input for casting a quadratic vote */
export interface CastQuadraticVoteInput {
  proposal_id: string;
  voter_did: string;
  choice: ZomeVoteChoice;
  credits_to_spend: number;
  reason?: string;
}

/** Quadratic vote record */
export interface QuadraticVote {
  id: string;
  proposal_id: string;
  voter: string;
  choice: ZomeVoteChoice;
  credits_spent: number;
  /** Effective weight = √credits_spent */
  effective_weight: number;
  reason?: string;
  voted_at: number;
}

/** Input for tallying quadratic votes */
export interface TallyQuadraticVotesInput {
  proposal_id: string;
  min_voters?: number;
}

/** Quadratic vote tally result */
export interface QuadraticTally {
  proposal_id: string;
  qv_for: number;
  qv_against: number;
  total_credits_spent: number;
  avg_credits_per_voter: number;
  voter_count: number;
  quorum_reached: boolean;
  approved: boolean;
  tallied_at: number;
  final_tally: boolean;
}

/** Zome-level vote choices (Rust enum: For/Against/Abstain) */
export type ZomeVoteChoice = 'For' | 'Against' | 'Abstain';

// --- Consciousness Metrics ---

/** Input for recording a consciousness snapshot */
export interface RecordSnapshotInput {
  phi: number;
  meta_awareness: number;
  self_model_accuracy: number;
  coherence: number;
  affective_valence: number;
  care_activation: number;
  source?: string;
}

/** Input for verifying consciousness gate */
export interface VerifyGateInput {
  action_type: GovernanceActionType;
  action_id?: string;
}

/** Result of consciousness gate verification */
export interface GateVerificationResult {
  passed: boolean;
  phi: number;
  required_phi: number;
  action_type: GovernanceActionType;
  failure_reason?: string;
  gate_id: string;
}

/** Enhanced gate verification with provenance tracking */
export interface GateVerificationResultV2 {
  passed: boolean;
  consciousness_level: number | null;
  required_consciousness: number;
  provenance: ConsciousnessProvenance;
  action_type: GovernanceActionType;
  failure_reason?: string;
}

/** Source of consciousness measurement */
export type ConsciousnessProvenance = 'Attested' | 'Snapshot' | 'Unavailable';

/** Governance action types that require Φ thresholds */
export type GovernanceActionType =
  | 'Basic'
  | 'ProposalSubmission'
  | 'Voting'
  | 'Constitutional';

/** Current Φ threshold requirements */
export interface PhiThresholds {
  basic: number;
  proposal_submission: number;
  voting: number;
  constitutional: number;
}

/** Input for querying agent consciousness snapshots */
export interface GetAgentSnapshotsInput {
  agent_did: string;
  limit?: number;
}

/** Input for assessing value alignment with Eight Harmonies */
export interface AssessValueAlignmentInput {
  proposal_id: string;
  proposal_content: string;
}

// --- Phi Attestation ---

/** Multi-dimensional consciousness vector entry */
export interface ConsciousnessVectorEntry {
  spectral_connectivity?: number;
  true_phi?: number;
  phi_fast?: number;
  entropy?: number;
  coherence?: number;
  epistemic_confidence?: number;
}

/** Input for recording a signed Φ attestation from Symthaea */
export interface RecordPhiAttestationInput {
  consciousness_level: number;
  cycle_id: number;
  captured_at_us: number;
  signature: number[];
  consciousness_vector?: ConsciousnessVectorEntry;
}

/** Consciousness attestation history entry */
export interface AttestationHistoryEntry {
  agent_did: string;
  consciousness_level: number;
  cycle_id: number;
  captured_at: number;
  signature: number[];
  source: string;
  consciousness_vector?: ConsciousnessVectorEntry;
}

/** Input for querying attestation history */
export interface GetAttestationHistoryInput {
  limit?: number;
}

/** Input for pruning stale attestations */
export interface PruneStaleAttestationsInput {
  keep_count: number;
}

// --- Consciousness-Aware Consensus ---

/** Input for registering as a consensus participant */
export interface RegisterConsensusParticipantInput {
  agent_did: string;
  k_vector_id: string;
  federated_rep_id?: string;
  reputation: number;
  matl_score: number;
  phi: number;
  federated_score: number;
}

/** Vote decision in consensus rounds */
export type VoteDecision = 'Approve' | 'Reject' | 'Abstain';

/** Input for casting a consciousness-weighted vote */
export interface CastWeightedVoteInput {
  proposal_id: string;
  proposal_type: ProposalType;
  round: number;
  decision: VoteDecision;
  harmonic_alignment?: number;
  reason?: string;
}

/** Input for calculating holistic vote weight */
export interface CalculateHolisticWeightInput {
  harmonic_alignment?: number;
}

/** Detailed holistic voting weight breakdown */
export interface HolisticVotingWeight {
  reputation: number;
  reputation_squared: number;
  consciousness_level: number;
  consciousness_multiplier: number;
  harmonic_alignment: number;
  harmonic_bonus: number;
  final_weight: number;
  was_capped: boolean;
  uncapped_weight: number;
  calculation_breakdown: string;
}

/** Query for votes in a consensus round */
export interface RoundVotesQuery {
  proposal_id: string;
  round: number;
}

/** Result of tallying a consensus round */
export interface RoundResult {
  proposal_id: string;
  round: number;
  proposal_type: ProposalType;
  total_weight: number;
  weighted_approvals: number;
  weighted_rejections: number;
  vote_count: number;
  required_threshold: number;
  approval_percentage: number;
  quorum_met: boolean;
  consensus_reached: boolean;
  rejected: boolean;
  result: string;
}

/** Consensus participant status with reputation and activity */
export interface ConsensusParticipantStatus {
  agent_did: string;
  is_active: boolean;
  base_reputation: number;
  effective_reputation: number;
  streak_count: number;
  streak_bonus: number;
  in_cooldown: boolean;
  current_phi: number;
  federated_score: number;
  rounds_participated: number;
  successful_votes: number;
  success_rate: number;
  slashing_events: number;
  can_vote_standard: boolean;
  can_vote_emergency: boolean;
  can_vote_constitutional: boolean;
}

// --- Reputation & Trust ---

/** 8-dimensional K-Vector trust model */
export interface KVector {
  agent_did: string;
  /** Reliability weight (0.25) */
  k_r: number;
  /** Accuracy weight (0.15) */
  k_a: number;
  /** Integrity weight (0.20) */
  k_i: number;
  /** Participation weight (0.15) */
  k_p: number;
  /** Mentorship weight (0.05) */
  k_m: number;
  /** Stewardship weight (0.10) */
  k_s: number;
  /** Harmony weight (0.05) */
  k_h: number;
  /** Topology weight (0.05) */
  k_topo: number;
  updated_at: number;
}

/** MATL trust score with component breakdown */
export interface MatlTrustScore {
  agent_did: string;
  pogq_score: number;
  tcdm_score: number;
  entropy_score: number;
  matl_score: number;
  k_vector_score: number;
  phi: number;
  calculated_at: number;
}

/** Input for updating multi-domain federated reputation */
export interface UpdateFederatedReputationInput {
  identity_verification?: number;
  credential_count?: number;
  credential_quality?: number;
  epistemic_contributions?: number;
  factcheck_accuracy?: number;
  dark_spots_resolved?: number;
  stake_weight?: number;
  payment_reliability?: number;
  escrow_completion_rate?: number;
  pogq_score?: number;
  fl_contributions?: number;
  byzantine_clean_rate?: number;
  voting_participation?: number;
  proposal_success_rate?: number;
  consensus_alignment?: number;
}

// --- Cross-hApp Integration ---

/** Governance query types for cross-hApp integration */
export type GovernanceQueryType =
  | 'ActiveProposals'
  | 'ProposalById'
  | 'VotingStatus'
  | 'VotingEligibility'
  | 'ConstitutionalRules';

/** Input for querying governance from another hApp */
export interface QueryGovernanceInput {
  source_happ: string;
  query_type: GovernanceQueryType;
  parameters: unknown;
}

/** Input for requesting cross-hApp execution */
export interface RequestExecutionInput {
  proposal_id: string;
  target_happ: string;
  action: string;
  parameters: unknown;
}

/** Execution status for cross-hApp actions */
export type ExecutionStatus =
  | 'Pending'
  | 'InProgress'
  | 'Completed'
  | 'Failed'
  | 'Reverted';

/** Input for acknowledging execution completion or failure */
export interface AcknowledgeExecutionInput {
  execution_id: string;
  status: ExecutionStatus;
  result?: string;
}

/** Bridge-level governance event types (distinct from SDK GovernanceEventType) */
export type BridgeGovernanceEventType =
  | 'ProposalCreated'
  | 'VotingStarted'
  | 'VoteReceived'
  | 'VotingEnded'
  | 'ProposalPassed'
  | 'ProposalFailed'
  | 'ProposalExecuted'
  | 'ConstitutionAmended';

/** Input for broadcasting a governance event */
export interface BroadcastGovernanceEventInput {
  event_type: BridgeGovernanceEventType;
  proposal_id?: string;
  subject: string;
  payload: string;
}

// --- Cross-Cluster Dispatch ---

/** Input for dispatching a call to the personal cluster */
export interface DispatchPersonalCallInput {
  zome_name: string;
  fn_name: string;
  payload: Uint8Array;
}

/** Input for dispatching a call to the identity cluster */
export interface DispatchIdentityCallInput {
  zome_name: string;
  fn_name: string;
  payload: Uint8Array;
}

/** Input for dispatching a call to the commons cluster */
export interface DispatchCommonsCallInput {
  zome_name: string;
  fn_name: string;
  payload: Uint8Array;
}

/** Input for dispatching a call to the civic cluster */
export interface DispatchCivicCallInput {
  zome_name: string;
  fn_name: string;
  payload: Uint8Array;
}

/** Input for enhanced voter trust check */
export interface CheckVoterTrustInput {
  did: string;
  min_reputation: number;
  require_mfa: boolean;
  min_assurance_level?: number;
}

// --- Phi Configuration ---

/** Dynamic Φ threshold configuration */
export interface GovernancePhiConfig {
  consciousness_gate_basic: number;
  consciousness_gate_proposal: number;
  consciousness_gate_voting: number;
  consciousness_gate_constitutional: number;
  min_voter_consciousness_standard: number;
  min_voter_consciousness_emergency: number;
  min_voter_consciousness_constitutional: number;
  max_voting_weight: number;
  min_true_phi_constitutional?: number;
  min_coherence_voting?: number;
  updated_at: number;
  changed_by_proposal?: string;
}

/** Input for updating Φ configuration */
export interface UpdatePhiConfigInput {
  proposal_id: string;
  consciousness_gate_basic?: number;
  consciousness_gate_proposal?: number;
  consciousness_gate_voting?: number;
  consciousness_gate_constitutional?: number;
  min_voter_consciousness_standard?: number;
  min_voter_consciousness_emergency?: number;
  min_voter_consciousness_constitutional?: number;
  max_voting_weight?: number;
}

/** Adaptive Φ threshold for a proposal type */
export interface AdaptiveThreshold {
  base_threshold: number;
  min_voter_consciousness: number;
  min_participation: number;
  quorum: number;
  max_extension_secs: number;
}

// --- Execution & Timelock ---

export type TimelockStatus = 'Pending' | 'Ready' | 'Executed' | 'Cancelled' | 'Failed';

export interface Timelock {
  id: string;
  proposal_id: string;
  actions: string;
  /** Timestamp (microseconds) */
  started: number;
  /** Timestamp (microseconds) */
  expires: number;
  status: TimelockStatus;
  cancellation_reason?: string;
}

export interface CreateTimelockInput {
  proposal_id: string;
  actions: string;
  duration_hours: number;
}

export interface ExecuteTimelockInput {
  timelock_id: string;
  executor_did: string;
}

export interface MarkTimelockReadyInput {
  timelock_id: string;
}

/** Input for guardian veto of a timelock */
export interface VetoTimelockInput {
  timelock_id: string;
  guardian_did: string;
  reason: string;
}

export interface GuardianVeto {
  id: string;
  timelock_id: string;
  guardian: string;
  reason: string;
  vetoed_at: number;
}

// --- Fund Management ---

export type AllocationStatus = 'Locked' | 'Released' | 'Refunded';

export interface FundAllocation {
  id: string;
  proposal_id: string;
  timelock_id: string;
  source_account: string;
  amount: number;
  currency: string;
  locked_at: number;
  status: AllocationStatus;
  status_reason?: string;
}

export interface LockFundsInput {
  proposal_id: string;
  timelock_id?: string;
  source_account: string;
  amount: number;
  currency?: string;
}

export interface ReleaseFundsInput {
  proposal_id: string;
  reason?: string;
}

export interface RefundFundsInput {
  proposal_id: string;
  reason: string;
}

// --- Council Decisions ---

export type DecisionType =
  | 'Operational'
  | 'Policy'
  | 'Resource'
  | 'Membership'
  | 'SubCouncil'
  | 'Constitutional';

export type DecisionStatus = 'Pending' | 'Approved' | 'Rejected' | 'Executed' | 'Vetoed';

export interface CouncilDecision {
  id: string;
  council_id: string;
  proposal_id?: string;
  title: string;
  content: string;
  decision_type: DecisionType;
  votes_for: number;
  votes_against: number;
  abstentions: number;
  phi_weighted_result: number;
  passed: boolean;
  status: DecisionStatus;
  created_at: number;
  executed_at?: number;
}

export interface RecordDecisionInput {
  council_id: string;
  proposal_id?: string;
  title: string;
  content: string;
  decision_type: DecisionType;
  votes_for: number;
  votes_against: number;
  abstentions: number;
  phi_weighted_result: number;
}

// --- Threshold Signing ---

export type DkgPhase = 'Registration' | 'Dealing' | 'Verification' | 'Complete' | 'Disbanded';

export type CommitteeScope =
  | 'All'
  | 'Constitutional'
  | 'Treasury'
  | 'Protocol'
  | { Custom: string[] };

export interface SigningCommittee {
  id: string;
  name: string;
  threshold: number;
  member_count: number;
  phase: DkgPhase;
  public_key?: number[];
  commitments: number[][];
  scope: CommitteeScope;
  created_at: number;
  active: boolean;
  epoch: number;
  min_phi?: number;
}

export interface CommitteeMember {
  committee_id: string;
  participant_id: number;
  agent: string;
  member_did: string;
  trust_score: number;
  public_share?: number[];
  vss_commitment?: number[];
  deal_submitted: boolean;
  qualified: boolean;
  registered_at: number;
}

export interface ThresholdSignature {
  id: string;
  committee_id: string;
  signed_content_hash: number[];
  signed_content_description: string;
  signature: number[];
  signer_count: number;
  signers: number[];
  verified: boolean;
  signed_at: number;
}

export interface CreateCommitteeInput {
  name: string;
  threshold: number;
  member_count: number;
  scope: CommitteeScope;
  min_phi?: number;
}

export interface RegisterMemberInput {
  committee_id: string;
  participant_id: number;
  member_did: string;
  trust_score: number;
}

export interface CombineSignaturesInput {
  committee_id: string;
  content_hash: number[];
  content_description: string;
  combined_signature: number[];
  signers: number[];
  verified: boolean;
}

// --- Real-Time Signals ---

/** Signals emitted by proposals coordinator */
export type ProposalSignal =
  | { type: 'ProposalCreated'; payload: { proposal_id: string; title: string; author: string } }
  | { type: 'ProposalStatusChanged'; payload: { proposal_id: string; old_status: string; new_status: string } }
  | { type: 'ContributionAdded'; payload: { proposal_id: string; contributor: string } }
  | { type: 'DiscussionReflectionGenerated'; payload: { proposal_id: string } };

/** Signals emitted by threshold-signing coordinator */
export type ThresholdSigningSignal =
  | { type: 'CommitteeCreated'; payload: { committee_id: string; threshold: number; member_count: number } }
  | { type: 'MemberRegistered'; payload: { committee_id: string; member_did: string } }
  | { type: 'DKGDealSubmitted'; payload: { committee_id: string; participant_id: number } }
  | { type: 'DKGFinalized'; payload: { committee_id: string; qualified_count: number } }
  | { type: 'SignatureShareSubmitted'; payload: { signature_id: string; participant_id: number } }
  | { type: 'ThresholdSignatureCreated'; payload: { signature_id: string; committee_id: string; verified: boolean } }
  | { type: 'CommitteeKeyRotated'; payload: { committee_id: string; new_epoch: number } };

/** Signals emitted by bridge coordinator */
export type BridgeSignal =
  | { type: 'ConsciousnessSnapshotRecorded'; payload: { agent_did: string; phi: number } }
  | { type: 'ConsciousnessGateVerified'; payload: { agent_did: string; passed: boolean; action_type: string } }
  | { type: 'ValueAlignmentAssessed'; payload: { proposal_id: string; agent_did: string; recommendation: string } };

/** Union of all governance signals */
export type GovernanceSignal = ProposalSignal | ThresholdSigningSignal | BridgeSignal;

/** Type guard for proposal signals */
export function isProposalSignal(signal: unknown): signal is ProposalSignal {
  return (
    typeof signal === 'object' &&
    signal !== null &&
    'type' in signal &&
    typeof (signal as { type: string }).type === 'string' &&
    ['ProposalCreated', 'ProposalStatusChanged', 'ContributionAdded', 'DiscussionReflectionGenerated']
      .includes((signal as { type: string }).type)
  );
}

/** Type guard for threshold-signing signals */
export function isThresholdSigningSignal(signal: unknown): signal is ThresholdSigningSignal {
  return (
    typeof signal === 'object' &&
    signal !== null &&
    'type' in signal &&
    typeof (signal as { type: string }).type === 'string' &&
    ['CommitteeCreated', 'MemberRegistered', 'DKGDealSubmitted', 'DKGFinalized',
     'SignatureShareSubmitted', 'ThresholdSignatureCreated', 'CommitteeKeyRotated']
      .includes((signal as { type: string }).type)
  );
}

/** Type guard for bridge signals */
export function isBridgeSignal(signal: unknown): signal is BridgeSignal {
  return (
    typeof signal === 'object' &&
    signal !== null &&
    'type' in signal &&
    typeof (signal as { type: string }).type === 'string' &&
    ['ConsciousnessSnapshotRecorded', 'ConsciousnessGateVerified', 'ValueAlignmentAssessed']
      .includes((signal as { type: string }).type)
  );
}

/** Type guard for any governance signal */
export function isGovernanceSignal(signal: unknown): signal is GovernanceSignal {
  return isProposalSignal(signal) || isThresholdSigningSignal(signal) || isBridgeSignal(signal);
}
