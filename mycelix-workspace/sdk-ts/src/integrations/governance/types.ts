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
