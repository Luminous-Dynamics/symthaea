/**
 * Governance hApp Client Types
 *
 * Type definitions for the Governance hApp Master SDK client,
 * providing complete coverage of all 7 governance zomes.
 *
 * @module @mycelix/sdk/clients/governance/types
 */

import type { ActionHash, AgentPubKey } from '../../generated/common';
// Note: EntryHash removed as unused

// ============================================================================
// Common Types
// ============================================================================

/**
 * Timestamp in microseconds (Holochain format)
 */
export type Timestamp = number;

/**
 * Base filter for list queries
 */
export interface PaginationFilter {
  /** Maximum results to return */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

// ============================================================================
// DAO Types
// ============================================================================

/**
 * Decentralized Autonomous Organization
 */
export interface DAO {
  /** DAO action hash (primary key) */
  id: ActionHash;
  /** Human-readable name */
  name: string;
  /** DAO description and purpose */
  description: string;
  /** Constitutional rules and governance policies */
  charter: string;
  /** Charter document hash (IPFS/content-addressed) */
  charterHash?: string;
  /** DID of the DAO founder */
  founderDid: string;
  /** Treasury wallet identifier */
  treasuryId?: string;
  /** Default voting period in hours */
  defaultVotingPeriodHours: number;
  /** Default quorum threshold (0.0-1.0) */
  defaultQuorum: number;
  /** Default approval threshold (0.0-1.0) */
  defaultThreshold: number;
  /** Current member count */
  memberCount: number;
  /** Total voting power in DAO */
  totalVotingPower: number;
  /** Whether DAO is active */
  active: boolean;
  /** Creation timestamp */
  createdAt: Timestamp;
  /** Last update timestamp */
  updatedAt: Timestamp;
}

/**
 * Input for creating a new DAO
 */
export interface CreateDAOInput {
  /** Human-readable name */
  name: string;
  /** DAO description and purpose */
  description: string;
  /** Constitutional rules and governance policies */
  charter: string;
  /** Charter document hash (optional) */
  charterHash?: string;
  /** Default voting period in hours */
  defaultVotingPeriodHours?: number;
  /** Default quorum threshold (0.0-1.0) */
  defaultQuorum?: number;
  /** Default approval threshold (0.0-1.0) */
  defaultThreshold?: number;
}

/**
 * Input for updating a DAO
 */
export interface UpdateDAOInput {
  /** DAO identifier */
  daoId: ActionHash;
  /** New name (optional) */
  name?: string;
  /** New description (optional) */
  description?: string;
  /** New charter (optional - requires governance approval) */
  charter?: string;
  /** New treasury ID (optional) */
  treasuryId?: string;
  /** New default voting period */
  defaultVotingPeriodHours?: number;
  /** New default quorum */
  defaultQuorum?: number;
  /** New default threshold */
  defaultThreshold?: number;
}

/**
 * Member roles with escalating privileges
 */
export type MemberRole = 'Member' | 'Moderator' | 'Admin' | 'Founder';

/**
 * DAO membership record
 */
export interface DAOMembership {
  /** Membership action hash */
  id: ActionHash;
  /** DAO identifier */
  daoId: ActionHash;
  /** Member's DID */
  memberDid: string;
  /** Member's agent public key */
  memberPubKey: AgentPubKey;
  /** Member's role */
  role: MemberRole;
  /** Base voting power */
  votingPower: number;
  /** Reputation score (0.0-1.0) */
  reputationScore: number;
  /** Whether membership is active */
  active: boolean;
  /** Timestamp when member joined */
  joinedAt: Timestamp;
  /** Last activity timestamp */
  lastActiveAt: Timestamp;
}

/**
 * Input for joining a DAO
 */
export interface JoinDAOInput {
  /** DAO to join */
  daoId: ActionHash;
  /** Initial voting power (may be 0 for new members) */
  initialVotingPower?: number;
}

/**
 * Filter for DAO queries
 */
export interface DAOFilter extends PaginationFilter {
  /** Filter by founder DID */
  founderDid?: string;
  /** Filter by active status */
  active?: boolean;
  /** Search by name */
  nameContains?: string;
}

// ============================================================================
// Proposal Types
// ============================================================================

/**
 * Proposal types with different voting periods
 */
export type ProposalType = 'Standard' | 'Emergency' | 'Constitutional';

/**
 * Proposal lifecycle status
 */
export type ProposalStatus =
  | 'Draft'
  | 'Active'
  | 'Passed'
  | 'Rejected'
  | 'Executed'
  | 'Expired'
  | 'Vetoed'
  | 'Cancelled';

/**
 * Governance proposal
 */
export interface Proposal {
  /** Proposal action hash (primary key) */
  id: ActionHash;
  /** DAO this proposal belongs to */
  daoId: ActionHash;
  /** Proposal title */
  title: string;
  /** Detailed description (markdown supported) */
  description: string;
  /** DID of the proposer */
  proposerDid: string;
  /** Type of proposal */
  proposalType: ProposalType;
  /** Current status */
  status: ProposalStatus;
  /** Category tag for filtering */
  category?: string;
  /** Voting period start (microseconds) */
  votingStartsAt: Timestamp;
  /** Voting period end (microseconds) */
  votingEndsAt: Timestamp;
  /** Required quorum (0.0-1.0) */
  quorum: number;
  /** Required approval threshold (0.0-1.0) */
  threshold: number;
  /** Total approve vote weight */
  approveWeight: number;
  /** Total reject vote weight */
  rejectWeight: number;
  /** Total abstain vote weight */
  abstainWeight: number;
  /** Unique voter count */
  voterCount: number;
  /** Optional execution payload (JSON) */
  executionPayload?: string;
  /** Target hApp for cross-hApp execution */
  executionTargetHapp?: string;
  /** Discussion link (forum, etc.) */
  discussionUrl?: string;
  /** Creation timestamp */
  createdAt: Timestamp;
  /** Execution timestamp (if executed) */
  executedAt?: Timestamp;
}

/**
 * Input for creating a proposal
 */
export interface CreateProposalInput {
  /** DAO this proposal belongs to */
  daoId: ActionHash;
  /** Proposal title */
  title: string;
  /** Detailed description */
  description: string;
  /** Type of proposal */
  proposalType: ProposalType;
  /** Category tag (optional) */
  category?: string;
  /** Custom quorum (optional, defaults to DAO setting) */
  quorum?: number;
  /** Custom threshold (optional, defaults to DAO setting) */
  threshold?: number;
  /** Custom voting period in hours (optional) */
  votingPeriodHours?: number;
  /** Optional execution payload */
  executionPayload?: string;
  /** Target hApp for execution */
  executionTargetHapp?: string;
  /** Discussion URL */
  discussionUrl?: string;
}

/**
 * Input for updating a proposal (draft status only)
 */
export interface UpdateProposalInput {
  /** Proposal ID */
  proposalId: ActionHash;
  /** New title */
  title?: string;
  /** New description */
  description?: string;
  /** New category */
  category?: string;
  /** New execution payload */
  executionPayload?: string;
  /** New discussion URL */
  discussionUrl?: string;
}

/**
 * Filter for proposal queries
 */
export interface ProposalFilter extends PaginationFilter {
  /** Filter by DAO */
  daoId?: ActionHash;
  /** Filter by status */
  status?: ProposalStatus;
  /** Filter by proposer */
  proposerDid?: string;
  /** Filter by type */
  proposalType?: ProposalType;
  /** Filter by category */
  category?: string;
  /** Include only proposals where voting is currently open */
  votingOpen?: boolean;
}

/**
 * Proposal finalization result
 */
export interface ProposalResult {
  /** Proposal identifier */
  proposalId: ActionHash;
  /** Final status */
  finalStatus: ProposalStatus;
  /** Total votes cast */
  totalVotes: number;
  /** Approve vote weight */
  approveWeight: number;
  /** Reject vote weight */
  rejectWeight: number;
  /** Abstain vote weight */
  abstainWeight: number;
  /** Whether quorum was met */
  quorumMet: boolean;
  /** Approval percentage (0-100) */
  approvalPercentage: number;
  /** Participation rate (0-100) */
  participationRate: number;
  /** Whether proposal passed */
  passed: boolean;
}

// ============================================================================
// Voting Types
// ============================================================================

/**
 * Vote choice options
 */
export type VoteChoice = 'Approve' | 'Reject' | 'Abstain';

/**
 * Vote record
 */
export interface Vote {
  /** Vote action hash */
  id: ActionHash;
  /** Proposal being voted on */
  proposalId: ActionHash;
  /** Voter's DID */
  voterDid: string;
  /** Vote choice */
  choice: VoteChoice;
  /** Vote weight (MATL-adjusted) */
  weight: number;
  /** Reasoning for vote (optional) */
  reason?: string;
  /** If voting on behalf of a delegator */
  delegatedFrom?: string;
  /** Vote timestamp */
  votedAt: Timestamp;
}

/**
 * Input for casting a vote
 */
export interface CastVoteInput {
  /** Proposal to vote on */
  proposalId: ActionHash;
  /** Vote choice */
  choice: VoteChoice;
  /** Vote weight (will be validated against voter's power) */
  weight: number;
  /** Reasoning (optional) */
  reason?: string;
  /** If voting on behalf of a delegator */
  delegatedFrom?: string;
}

/**
 * Voting power calculation query
 */
export interface VotingPowerQuery {
  /** DAO identifier */
  daoId: ActionHash;
  /** Voter's DID */
  voterDid: string;
  /** Include delegated power */
  includeDelegated?: boolean;
}

/**
 * Detailed voting power breakdown
 */
export interface VotingPowerBreakdown {
  /** Base power from membership */
  basePower: number;
  /** MATL reputation multiplier (0.5-2.0) */
  matlMultiplier: number;
  /** Bonus from participation history */
  participationBonus: number;
  /** Power received from delegations */
  delegatedPower: number;
  /** Stake multiplier (if staking enabled) */
  stakeMultiplier: number;
  /** Final calculated voting power */
  totalPower: number;
}

/**
 * Quorum status for a proposal
 */
export interface QuorumStatus {
  /** Proposal identifier */
  proposalId: ActionHash;
  /** Current vote weight */
  currentVotes: number;
  /** Required vote weight for quorum */
  requiredVotes: number;
  /** Whether quorum is met */
  quorumMet: boolean;
  /** Percentage of quorum achieved (0-100) */
  quorumPercentage: number;
}

/**
 * Voting statistics for a proposal
 */
export interface VotingStats {
  /** Proposal identifier */
  proposalId: ActionHash;
  /** Approve vote weight */
  approveWeight: number;
  /** Reject vote weight */
  rejectWeight: number;
  /** Abstain vote weight */
  abstainWeight: number;
  /** Total vote weight */
  totalWeight: number;
  /** Unique voter count */
  voterCount: number;
  /** Approve percentage (0-100) */
  approvePercentage: number;
  /** Reject percentage (0-100) */
  rejectPercentage: number;
  /** Abstain percentage (0-100) */
  abstainPercentage: number;
  /** Whether voting is active */
  isActive: boolean;
  /** Time remaining in seconds (0 if ended) */
  timeRemaining: number;
}

// ============================================================================
// Delegation Types
// ============================================================================

/**
 * Delegation scope
 */
export type DelegationScope = 'All' | 'Category';

/**
 * Vote delegation record
 */
export interface Delegation {
  /** Delegation action hash */
  id: ActionHash;
  /** DID of the person delegating */
  delegatorDid: string;
  /** DID of the delegate */
  delegateDid: string;
  /** DAO this delegation applies to */
  daoId: ActionHash;
  /** Scope of delegation */
  scope: DelegationScope;
  /** Category filter (if scope is 'Category') */
  category?: string;
  /** Percentage of power to delegate (0.0-1.0) */
  powerPercentage: number;
  /** Expiration timestamp (optional) */
  expiresAt?: Timestamp;
  /** Whether delegation is currently active */
  active: boolean;
  /** Creation timestamp */
  createdAt: Timestamp;
}

/**
 * Input for creating a delegation
 */
export interface CreateDelegationInput {
  /** DID of the delegate to receive power */
  delegateDid: string;
  /** DAO for this delegation */
  daoId: ActionHash;
  /** Scope of delegation */
  scope: DelegationScope;
  /** Category (required if scope is 'Category') */
  category?: string;
  /** Percentage of power to delegate (0.0-1.0, default 1.0) */
  powerPercentage?: number;
  /** Expiration (optional) */
  expiresAt?: Timestamp;
}

/**
 * Filter for delegation queries
 */
export interface DelegationFilter extends PaginationFilter {
  /** Filter by DAO */
  daoId?: ActionHash;
  /** Filter by delegator */
  delegatorDid?: string;
  /** Filter by delegate */
  delegateDid?: string;
  /** Include only active delegations */
  activeOnly?: boolean;
}

/**
 * Delegation chain (for detecting circular delegations)
 */
export interface DelegationChain {
  /** Starting delegator */
  startDid: string;
  /** Chain of delegates */
  chain: string[];
  /** Whether chain forms a cycle */
  hasCycle: boolean;
}

// ============================================================================
// Treasury Types
// ============================================================================

/**
 * Treasury configuration
 */
export interface Treasury {
  /** Treasury action hash */
  id: ActionHash;
  /** Associated DAO */
  daoId: ActionHash;
  /** Treasury name */
  name: string;
  /** Description */
  description: string;
  /** Governance threshold for allocations (0.0-1.0) */
  governanceThreshold: number;
  /** Spending limit without governance (per period) */
  discretionaryLimit: number;
  /** Discretionary period in hours */
  discretionaryPeriodHours: number;
  /** Multi-sig threshold (number of signers required) */
  multiSigThreshold: number;
  /** Multi-sig signers (DIDs) */
  multiSigSigners: string[];
  /** Whether treasury is active */
  active: boolean;
  /** Creation timestamp */
  createdAt: Timestamp;
}

/**
 * Treasury balance
 */
export interface TreasuryBalance {
  /** Treasury identifier */
  treasuryId: ActionHash;
  /** Currency code */
  currency: string;
  /** Available balance */
  available: number;
  /** Locked in pending allocations */
  locked: number;
  /** Total balance */
  total: number;
  /** Last updated */
  updatedAt: Timestamp;
}

/**
 * Treasury allocation status
 */
export type AllocationStatus =
  | 'Proposed'
  | 'Voting'
  | 'Approved'
  | 'Executed'
  | 'Rejected'
  | 'Cancelled';

/**
 * Treasury allocation record
 */
export interface TreasuryAllocation {
  /** Allocation action hash */
  id: ActionHash;
  /** Treasury identifier */
  treasuryId: ActionHash;
  /** Associated proposal (if governance-approved) */
  proposalId?: ActionHash;
  /** Purpose of allocation */
  purpose: string;
  /** Amount to allocate */
  amount: number;
  /** Currency code */
  currency: string;
  /** Recipient DID */
  recipientDid: string;
  /** Recipient wallet address (external) */
  recipientAddress?: string;
  /** Current status */
  status: AllocationStatus;
  /** Approvers (DIDs) */
  approvedBy: string[];
  /** Execution transaction hash */
  executionTxHash?: string;
  /** Request timestamp */
  requestedAt: Timestamp;
  /** Execution timestamp */
  executedAt?: Timestamp;
}

/**
 * Input for creating a treasury
 */
export interface CreateTreasuryInput {
  /** Associated DAO */
  daoId: ActionHash;
  /** Treasury name */
  name: string;
  /** Description */
  description: string;
  /** Governance threshold (0.0-1.0) */
  governanceThreshold?: number;
  /** Discretionary spending limit */
  discretionaryLimit?: number;
  /** Discretionary period in hours */
  discretionaryPeriodHours?: number;
  /** Multi-sig threshold */
  multiSigThreshold?: number;
  /** Multi-sig signers */
  multiSigSigners?: string[];
}

/**
 * Input for proposing an allocation
 */
export interface ProposeAllocationInput {
  /** Treasury identifier */
  treasuryId: ActionHash;
  /** Purpose of allocation */
  purpose: string;
  /** Amount to allocate */
  amount: number;
  /** Currency code */
  currency: string;
  /** Recipient DID */
  recipientDid: string;
  /** Recipient wallet address (optional) */
  recipientAddress?: string;
  /** Associated proposal ID (for governance-approved) */
  proposalId?: ActionHash;
}

// ============================================================================
// Execution Types
// ============================================================================

/**
 * Execution status
 */
export type ExecutionStatus =
  | 'Pending'
  | 'InProgress'
  | 'Completed'
  | 'Failed'
  | 'Reverted'
  | 'Cancelled';

/**
 * Proposal execution record
 */
export interface ProposalExecution {
  /** Execution action hash */
  id: ActionHash;
  /** Proposal being executed */
  proposalId: ActionHash;
  /** Target hApp for execution */
  targetHapp: string;
  /** Action to execute */
  action: string;
  /** Payload (JSON) */
  payload: string;
  /** Current status */
  status: ExecutionStatus;
  /** Executor DID */
  executorDid: string;
  /** Result data (JSON) */
  resultData?: string;
  /** Error message (if failed) */
  errorMessage?: string;
  /** Number of retry attempts */
  retryCount: number;
  /** Maximum retries allowed */
  maxRetries: number;
  /** Request timestamp */
  requestedAt: Timestamp;
  /** Start timestamp */
  startedAt?: Timestamp;
  /** Completion timestamp */
  completedAt?: Timestamp;
}

/**
 * Input for requesting execution
 */
export interface RequestExecutionInput {
  /** Proposal to execute */
  proposalId: ActionHash;
  /** Target hApp */
  targetHapp?: string;
  /** Action (defaults to proposal's execution action) */
  action?: string;
  /** Payload (defaults to proposal's execution payload) */
  payload?: string;
}

/**
 * Input for acknowledging execution completion
 */
export interface AcknowledgeExecutionInput {
  /** Execution identifier */
  executionId: ActionHash;
  /** Whether execution succeeded */
  success: boolean;
  /** Result data (JSON) */
  resultData?: string;
  /** Error message (if failed) */
  errorMessage?: string;
}

/**
 * Execution schedule for delayed execution
 */
export interface ExecutionSchedule {
  /** Schedule action hash */
  id: ActionHash;
  /** Proposal to execute */
  proposalId: ActionHash;
  /** Scheduled execution time */
  scheduledAt: Timestamp;
  /** Whether schedule is active */
  active: boolean;
  /** Creation timestamp */
  createdAt: Timestamp;
}

// ============================================================================
// Bridge Types (Cross-hApp Governance)
// ============================================================================

/**
 * Governance bridge event types
 */
export type GovernanceBridgeEventType =
  | 'DAOCreated'
  | 'ProposalCreated'
  | 'ProposalActivated'
  | 'ProposalPassed'
  | 'ProposalRejected'
  | 'ProposalExecuted'
  | 'VoteCast'
  | 'DelegationCreated'
  | 'DelegationRevoked'
  | 'MemberJoined'
  | 'TreasuryAllocated';

/**
 * Governance bridge event
 */
export interface GovernanceBridgeEvent {
  /** Event action hash */
  id: ActionHash;
  /** Event type */
  eventType: GovernanceBridgeEventType;
  /** Source hApp */
  sourceHapp: string;
  /** Related DAO */
  daoId?: ActionHash;
  /** Related proposal */
  proposalId?: ActionHash;
  /** Subject DID */
  subjectDid?: string;
  /** Event payload (JSON) */
  payload: string;
  /** Event timestamp */
  timestamp: Timestamp;
}

/**
 * Cross-hApp governance query
 */
export interface GovernanceQuery {
  /** Query type */
  queryType: 'ProposalStatus' | 'VotingPower' | 'DelegationChain' | 'ExecutionHistory' | 'TreasuryBalance';
  /** Query parameters (JSON) */
  queryParams: string;
  /** Source hApp making the query */
  sourceHapp: string;
}

/**
 * Cross-hApp proposal reference
 */
export interface CrossHappProposal {
  /** Local reference ID */
  id: ActionHash;
  /** Original proposal hash */
  originalProposalHash: string;
  /** Source hApp */
  sourceHapp: string;
  /** Proposal title */
  title: string;
  /** Proposal type */
  proposalType: ProposalType;
  /** Current status */
  status: ProposalStatus;
  /** Vote weight for */
  voteWeightFor: number;
  /** Vote weight against */
  voteWeightAgainst: number;
  /** Vote weight abstain */
  voteWeightAbstain: number;
  /** Voting end time */
  votingEndsAt: Timestamp;
  /** Creation timestamp */
  createdAt: Timestamp;
}

/**
 * Input for broadcasting governance event
 */
export interface BroadcastEventInput {
  /** Event type */
  eventType: GovernanceBridgeEventType;
  /** Related DAO */
  daoId?: ActionHash;
  /** Related proposal */
  proposalId?: ActionHash;
  /** Subject DID */
  subjectDid?: string;
  /** Event payload (JSON) */
  payload: string;
}

/**
 * Participation score for cross-hApp reputation
 */
export interface ParticipationScore {
  /** DID of the participant */
  did: string;
  /** Number of DAOs joined */
  daoMemberships: number;
  /** Number of proposals created */
  proposalsCreated: number;
  /** Number of votes cast */
  votesCast: number;
  /** Participation rate (0.0-1.0) */
  participationRate: number;
  /** Alignment with winning outcomes (0.0-1.0) */
  alignmentScore: number;
  /** Trust from delegators (0.0-1.0) */
  delegationTrust: number;
  /** Overall governance reputation (0.0-1.0) */
  overallScore: number;
}

// ============================================================================
// Error Types
// ============================================================================

/**
 * Governance SDK error codes
 */
export type GovernanceErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_CALL_ERROR'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'INVALID_INPUT'
  | 'NOT_MEMBER'
  | 'ALREADY_MEMBER'
  | 'ALREADY_VOTED'
  | 'VOTING_NOT_OPEN'
  | 'VOTING_CLOSED'
  | 'QUORUM_NOT_MET'
  | 'INSUFFICIENT_POWER'
  | 'DELEGATION_ERROR'
  | 'CIRCULAR_DELEGATION'
  | 'TREASURY_ERROR'
  | 'INSUFFICIENT_FUNDS'
  | 'EXECUTION_ERROR'
  | 'BRIDGE_ERROR';

/**
 * Governance SDK Error
 */
export class GovernanceError extends Error {
  constructor(
    public readonly code: GovernanceErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'GovernanceError';
  }
}
