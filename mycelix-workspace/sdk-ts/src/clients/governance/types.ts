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
export type ProposalType = 'Standard' | 'Emergency' | 'Constitutional' | 'Parameter' | 'Funding';

/**
 * Proposal lifecycle status
 */
export type ProposalStatus =
  | 'Draft'
  | 'Active'
  | 'Ended'
  | 'Approved'
  | 'Signed'
  | 'Rejected'
  | 'Executed'
  | 'Cancelled'
  | 'Failed';

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
 * Treasury allocation status (SDK-layer abstraction)
 */
export type AllocationStatus =
  | 'Proposed'
  | 'Voting'
  | 'Approved'
  | 'Executed'
  | 'Rejected'
  | 'Cancelled';

/**
 * Fund allocation status (matches Rust execution zome AllocationStatus)
 */
export type FundAllocationStatus =
  | 'Locked'
  | 'Released'
  | 'Refunded';

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
// Execution Types (matches Rust execution integrity zome)
// ============================================================================

/**
 * Timelock status
 */
export type TimelockStatus =
  | 'Pending'
  | 'Ready'
  | 'Executed'
  | 'Cancelled'
  | 'Failed';

/**
 * Execution result status
 */
export type ExecutionStatus =
  | 'Success'
  | 'PartialSuccess'
  | 'Failed';

/**
 * Timelock record — proposal actions locked until delay expires
 */
export interface Timelock {
  /** Timelock identifier */
  id: string;
  /** Associated proposal */
  proposalId: string;
  /** Actions to execute (JSON) */
  actions: string;
  /** Timelock start timestamp */
  started: Timestamp;
  /** Timelock expiry timestamp */
  expires: Timestamp;
  /** Current status */
  status: TimelockStatus;
  /** Reason for cancellation (if cancelled) */
  cancellationReason?: string;
}

/**
 * Execution record — result of executing a timelocked proposal
 */
export interface Execution {
  /** Execution identifier */
  id: string;
  /** Timelock this execution belongs to */
  timelockId: string;
  /** Original proposal */
  proposalId: string;
  /** Executor DID */
  executor: string;
  /** Execution result status */
  status: ExecutionStatus;
  /** Result data (JSON) */
  result?: string;
  /** Error message (if failed) */
  error?: string;
  /** Execution timestamp */
  executedAt: Timestamp;
}

/**
 * Guardian veto record
 */
export interface GuardianVeto {
  /** Veto identifier */
  id: string;
  /** Timelock being vetoed */
  timelockId: string;
  /** Guardian DID */
  guardian: string;
  /** Veto reason */
  reason: string;
  /** Veto timestamp */
  vetoedAt: Timestamp;
}

/**
 * Fund allocation record
 */
export interface FundAllocation {
  /** Allocation identifier */
  id: string;
  /** Associated proposal */
  proposalId: string;
  /** Associated timelock */
  timelockId: string;
  /** Source account */
  sourceAccount: string;
  /** Amount to allocate */
  amount: number;
  /** Currency code */
  currency: string;
  /** When funds were locked */
  lockedAt: Timestamp;
  /** Current status */
  status: FundAllocationStatus;
  /** Reason for status change */
  statusReason?: string;
}

/**
 * Input for creating a timelock
 */
export interface CreateTimelockInput {
  /** Associated proposal */
  proposalId: string;
  /** Actions to execute (JSON) */
  actions: string;
  /** Delay duration in hours */
  delayHours: number;
}

/**
 * Input for executing a timelock
 */
export interface ExecuteTimelockInput {
  /** Timelock identifier */
  timelockId: string;
}

/**
 * Input for vetoing a timelock
 */
export interface VetoTimelockInput {
  /** Timelock identifier */
  timelockId: string;
  /** Veto reason */
  reason: string;
}

// ============================================================================
// Bridge Types (Cross-hApp Governance)
// ============================================================================

/**
 * Governance bridge event types
 */
export type GovernanceBridgeEventType =
  | 'ProposalCreated'
  | 'VotingStarted'
  | 'VoteReceived'
  | 'VotingEnded'
  | 'ProposalPassed'
  | 'ProposalFailed'
  | 'ProposalExecuted'
  | 'ConstitutionAmended';

/**
 * Governance bridge event
 */
export interface GovernanceBridgeEvent {
  /** Event identifier */
  id: string;
  /** Event type */
  eventType: GovernanceBridgeEventType;
  /** Related proposal (optional) */
  proposalId?: string;
  /** Subject description */
  subject: string;
  /** Event payload (JSON) */
  payload: string;
  /** Source hApp */
  sourceHapp: string;
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
  /** Related proposal (optional) */
  proposalId?: string;
  /** Subject description */
  subject: string;
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

// ============================================================================
// Φ-Weighted Voting Types
// ============================================================================

/**
 * Proposal tier based on governance impact
 * Different tiers require different Φ thresholds
 */
export type ProposalTier = 'Basic' | 'Major' | 'Constitutional';

/**
 * Vote choice matching Rust VoteChoice enum (For/Against/Abstain)
 *
 * Note: The Rust integrity zome uses For/Against/Abstain.
 * The legacy VoteChoice type above uses Approve/Reject/Abstain.
 * New consciousness-weighted, quadratic, and verified voting methods use this type.
 */
export type ConsciousnessVoteChoice = 'For' | 'Against' | 'Abstain';

/**
 * How the consciousness value in a governance decision was obtained.
 * Matches Rust `ConsciousnessProvenance` enum.
 */
export type ConsciousnessProvenance = 'Attested' | 'Snapshot' | 'Unavailable';

/**
 * Consciousness weight components for consciousness-integrated governance
 */
export interface ConsciousnessWeight {
  /** Base consciousness score from consciousness metrics (0.0-1.0) */
  phiScore: number;
  /** How the consciousness score was obtained */
  phiProvenance: ConsciousnessProvenance;
  /** K-vector derived trust score (0.0-1.0) */
  kTrust: number;
  /** Stake-based weight */
  stakeWeight: number;
  /** Historical participation score (0.0-1.0) */
  participationScore: number;
  /** Reputation in relevant domain (0.0-1.0) */
  domainReputation: number;
}

/**
 * Consciousness-weighted vote with full consciousness integration
 */
export interface ConsciousnessWeightedVote {
  id: string;
  proposalId: string;
  proposalTier: ProposalTier;
  voter: string;
  choice: ConsciousnessVoteChoice;
  consciousnessWeight: ConsciousnessWeight;
  effectiveWeight: number;
  reason?: string;
  delegated: boolean;
  delegator?: string;
  delegationChain: string[];
  votedAt: Timestamp;
}

/**
 * Input for casting a consciousness-weighted vote
 */
export interface CastConsciousnessVoteInput {
  proposalId: string;
  voterDid: string;
  tier: ProposalTier;
  choice: ConsciousnessVoteChoice;
  reason?: string;
}

/**
 * Input for casting a delegated consciousness-weighted vote
 */
export interface CastDelegatedConsciousnessVoteInput {
  proposalId: string;
  delegateDid: string;
  tier: ProposalTier;
  choice: ConsciousnessVoteChoice;
  reason?: string;
}

// ============================================================================
// Quadratic Voting Types
// ============================================================================

/**
 * Quadratic vote record - weight = √(credits spent)
 */
export interface QuadraticVoteRecord {
  id: string;
  proposalId: string;
  voter: string;
  choice: ConsciousnessVoteChoice;
  creditsSpent: number;
  effectiveWeight: number;
  reason?: string;
  votedAt: Timestamp;
}

/**
 * Input for casting a quadratic vote
 */
export interface CastQuadraticVoteInput {
  proposalId: string;
  voterDid: string;
  choice: ConsciousnessVoteChoice;
  creditsToSpend: number;
  reason?: string;
}

/**
 * Voice credit balance for quadratic voting
 */
export interface VoiceCredits {
  owner: string;
  allocated: number;
  spent: number;
  remaining: number;
  periodStart: Timestamp;
  periodEnd: Timestamp;
}

/**
 * Input for allocating voice credits
 */
export interface AllocateCreditsInput {
  ownerDid: string;
  amount: number;
  periodEnd: Timestamp;
}

// ============================================================================
// Delegation with Decay Types
// ============================================================================

/**
 * Delegation decay model
 *
 * Rust serde: externally tagged enum
 * - "None" → no decay
 * - {"Linear":{"decay_days":30}} → linear
 * - {"Exponential":{"half_life_days":14}} → exponential
 * - {"Step":{"step_interval_days":7,"drop_per_step":0.1}} → step
 */
export type DelegationDecay =
  | 'None'
  | { Linear: { decay_days: number } }
  | { Exponential: { half_life_days: number } }
  | { Step: { step_interval_days: number; drop_per_step: number } };

/**
 * Delegation with decay support
 */
export interface DelegationWithDecay {
  id: string;
  delegator: string;
  delegate: string;
  percentage: number;
  topics?: string[];
  tierFilter?: ProposalTier[];
  active: boolean;
  decay: DelegationDecay;
  transitive: boolean;
  maxChainDepth: number;
  created: Timestamp;
  renewed: Timestamp;
  expires?: Timestamp;
}

/**
 * Input for creating a delegation with decay
 */
export interface CreateDelegationWithDecayInput {
  delegatorDid: string;
  delegateDid: string;
  percentage: number;
  topics?: string[];
  tierFilter?: ProposalTier[];
  decay?: DelegationDecay;
  transitive?: boolean;
  maxChainDepth?: number;
  expires?: Timestamp;
}

/**
 * Input for renewing a delegation (resets decay timer)
 */
export interface RenewDelegationInput {
  originalActionHash: ActionHash;
  newPercentage?: number;
}

/**
 * Input for revoking a delegation
 */
export interface RevokeDelegationInput {
  originalActionHash: ActionHash;
}

/**
 * Delegation with current effective percentage
 */
export interface EffectiveDelegation {
  delegation: DelegationWithDecay;
  effectivePercentage: number;
  isEffectivelyExpired: boolean;
}

// ============================================================================
// Vote Tally Types
// ============================================================================

/**
 * Input for tallying legacy votes
 */
export interface TallyVotesInput {
  proposalId: string;
  tier?: ProposalTier;
  quorumOverride?: number;
  approvalOverride?: number;
}

/**
 * Legacy vote tally
 */
export interface VoteTally {
  proposalId: string;
  votesFor: number;
  votesAgainst: number;
  abstentions: number;
  totalWeight: number;
  quorumReached: boolean;
  approved: boolean;
  talliedAt: Timestamp;
  finalTally: boolean;
}

/**
 * Input for tallying consciousness-weighted votes
 */
export interface TallyConsciousnessVotesInput {
  proposalId: string;
  tier: ProposalTier;
  eligibleVoters?: number;
  generateReflection?: boolean;
}

/**
 * Segment of tally for analysis
 */
export interface TallySegment {
  votesFor: number;
  votesAgainst: number;
  abstentions: number;
  voterCount: number;
}

/**
 * Breakdown by voter consciousness tier
 */
export interface ConsciousnessTierBreakdown {
  highConsciousnessVotes: TallySegment;
  mediumConsciousnessVotes: TallySegment;
  lowConsciousnessVotes: TallySegment;
}

/**
 * Consciousness-weighted vote tally with full breakdown
 */
export interface ConsciousnessWeightedTally {
  proposalId: string;
  tier: ProposalTier;
  consciousnessVotesFor: number;
  consciousnessVotesAgainst: number;
  consciousnessAbstentions: number;
  rawVotesFor: number;
  rawVotesAgainst: number;
  rawAbstentions: number;
  averageConsciousness: number;
  totalConsciousnessWeight: number;
  eligibleVoters: number;
  quorumRequirement: number;
  quorumReached: boolean;
  approvalThreshold: number;
  approved: boolean;
  talliedAt: Timestamp;
  finalTally: boolean;
  consciousnessTierBreakdown: ConsciousnessTierBreakdown;
  /** Votes with real consciousness data (Attested or Snapshot) */
  consciousnessEnhancedCount: number;
  /** Votes without consciousness data (reputation-only) */
  reputationOnlyCount: number;
  /** Fraction of votes with verified consciousness data (0.0-1.0) */
  consciousnessCoverage: number;
}

/**
 * Input for tallying quadratic votes
 */
export interface TallyQuadraticVotesInput {
  proposalId: string;
  minVoters?: number;
}

/**
 * Quadratic vote tally
 */
export interface QuadraticTally {
  proposalId: string;
  qvFor: number;
  qvAgainst: number;
  totalCreditsSpent: number;
  avgCreditsPerVoter: number;
  voterCount: number;
  quorumReached: boolean;
  approved: boolean;
  talliedAt: Timestamp;
  finalTally: boolean;
}

// ============================================================================
// ZK-STARK Verified Voting Types
// ============================================================================

/**
 * Proposal type for ZK eligibility proofs
 */
export type ZkProposalType =
  | 'Standard'
  | 'Constitutional'
  | 'ModelGovernance'
  | 'Emergency'
  | 'Treasury'
  | 'Membership';

/**
 * Input for storing an eligibility proof
 */
export interface StoreEligibilityProofInput {
  voterDid: string;
  voterCommitment: number[];
  proposalType: ZkProposalType;
  eligible: boolean;
  requirementsMet: number;
  activeRequirements: number;
  proofBytes: number[];
  validityHours?: number;
}

/**
 * ZK-STARK eligibility proof stored on-chain
 */
export interface EligibilityProof {
  id: string;
  voterDid: string;
  voterCommitment: number[];
  proposalType: ZkProposalType;
  eligible: boolean;
  requirementsMet: number;
  activeRequirements: number;
  proofBytes: number[];
  generatedAt: Timestamp;
  expiresAt?: Timestamp;
}

/**
 * Input for casting a verified vote
 */
export interface CastVerifiedVoteInput {
  proposalId: string;
  voterDid: string;
  tier: ProposalTier;
  choice: ConsciousnessVoteChoice;
  eligibilityProofHash: ActionHash;
  voterCommitment: number[];
  reason?: string;
}

/**
 * Verified vote record
 */
export interface VerifiedVote {
  id: string;
  proposalId: string;
  proposalTier: ProposalTier;
  voter: string;
  choice: ConsciousnessVoteChoice;
  eligibilityProofHash: ActionHash;
  voterCommitment: number[];
  effectiveWeight: number;
  reason?: string;
  votedAt: Timestamp;
}

/**
 * Input for tallying verified votes
 */
export interface TallyVerifiedVotesInput {
  proposalId: string;
  tier: ProposalTier;
  eligibleVoters?: number;
}

/**
 * Verified vote tally result
 */
export interface VerifiedVoteTallyResult {
  proposalId: string;
  tier: ProposalTier;
  votesFor: number;
  votesAgainst: number;
  abstentions: number;
  totalWeight: number;
  voterCount: number;
  quorumRequirement: number;
  quorumReached: boolean;
  approvalThreshold: number;
  approvalRate: number;
  approved: boolean;
  talliedAt: Timestamp;
}

// ============================================================================
// Proof Attestation Types
// ============================================================================

/**
 * Input for storing a proof attestation from an external verifier
 */
export interface StoreAttestationInput {
  proofActionHash: ActionHash;
  proofHash: number[];
  voterCommitment: number[];
  proposalType: ZkProposalType;
  verified: boolean;
  verifierPubkey: number[];
  signature: number[];
  securityLevel: string;
  verificationTimeMs: number;
  validityHours?: number;
}

/**
 * Proof attestation from an external verifier
 */
export interface ProofAttestation {
  id: string;
  proofHash: number[];
  proofActionHash: ActionHash;
  voterCommitment: number[];
  proposalType: ZkProposalType;
  verified: boolean;
  verifiedAt: Timestamp;
  expiresAt: Timestamp;
  verifierPubkey: number[];
  signature: number[];
  securityLevel: string;
  verificationTimeMs: number;
}

/**
 * Input for casting a vote with attestation check
 */
export interface CastAttestedVoteInput {
  proposalId: string;
  voterDid: string;
  tier: ProposalTier;
  choice: ConsciousnessVoteChoice;
  eligibilityProofHash: ActionHash;
  voterCommitment: number[];
  reason?: string;
}

// ============================================================================
// Collective Mirror / Reflection Types
// ============================================================================

/** Agreement structure pattern */
export type TopologyPattern = 'Mesh' | 'HubAndSpoke' | 'Polarized' | 'Monopole' | 'Unknown';

/** Echo chamber risk level */
export type EchoChamberRiskLevel = 'Low' | 'Moderate' | 'High' | 'Critical';

/** Trend direction for temporal analysis */
export type TrendDirection = 'Rising' | 'Stable' | 'Falling' | 'Unknown';

/**
 * Collective mirror reflection for a proposal's voting phase
 */
export interface ProposalReflection {
  id: string;
  proposalId: string;
  timestamp: Timestamp;
  voterCount: number;
  topologyPattern: TopologyPattern;
  centralization: number;
  clusterCount: number;
  absentHarmonies: string[];
  harmonyCoverage: number;
  averageEpistemicLevel: number;
  echoChamberRisk: EchoChamberRiskLevel;
  agreementVerified: boolean;
  agreementTrend: TrendDirection;
  centralizationTrend: TrendDirection;
  rapidConvergenceWarning: boolean;
  fragmentationWarning: boolean;
  votesFor: number;
  votesAgainst: number;
  abstentions: number;
  approvalRatio: number;
  polarization: number;
  suggestedInterventions: string[];
  reflectionPrompts: string[];
  needsReview: boolean;
  summary: string;
}

// ============================================================================
// Discussion Types (Proposals Zome)
// ============================================================================

/** Discussion contribution stance */
export type Stance = 'Support' | 'Oppose' | 'Neutral' | 'Question';

/**
 * Input for adding a discussion contribution
 */
export interface AddContributionInput {
  proposalId: string;
  contributorDid: string;
  content: string;
  harmonyTags?: string[];
  stance?: Stance;
  parentId?: string;
}

/**
 * Discussion contribution record
 */
export interface DiscussionContribution {
  id: string;
  proposalId: string;
  contributor: string;
  content: string;
  harmonyTags: string[];
  stance?: Stance;
  parentId?: string;
  createdAt: Timestamp;
  edited: boolean;
}

/**
 * Discussion readiness assessment
 */
export interface DiscussionReadiness {
  ready: boolean;
  reasoning: string;
  contributorCount: number;
  contributionCount: number;
  harmonyDiversity: number;
  unaddressedConcerns: string[];
}

// ============================================================================
// Bridge Consciousness Types
// ============================================================================

/**
 * Input for recording a consciousness snapshot.
 * Matches Rust `RecordSnapshotInput` struct.
 */
export interface RecordSnapshotInput {
  /** Consciousness level measurement (0.0-1.0) */
  consciousnessLevel?: number;
  /** @deprecated Use consciousnessLevel */
  phi?: number;
  /** Meta-awareness score (0.0-1.0) */
  metaAwareness: number;
  /** Self-model accuracy (0.0-1.0) */
  selfModelAccuracy: number;
  /** Coherence score (0.0-1.0) */
  coherence: number;
  /** Affective valence (-1.0 to 1.0) */
  affectiveValence: number;
  /** CARE activation (0.0-1.0) */
  careActivation: number;
  /** Source system (default: "symthaea") */
  source?: string;
}

/**
 * Type of governance action with associated consciousness thresholds.
 * Matches Rust `GovernanceActionType` enum.
 */
export type GovernanceActionType = 'Basic' | 'ProposalSubmission' | 'Voting' | 'Constitutional';

/**
 * Vote decision in consensus rounds.
 * Matches Rust `VoteDecision` enum.
 */
export type VoteDecision = 'Approve' | 'Reject' | 'Abstain';

/**
 * Bridge ProposalType (3-variant, distinct from the 5-variant types.ts ProposalType)
 * Matches Rust bridge integrity `ProposalType` enum.
 */
export type BridgeProposalType = 'Standard' | 'Emergency' | 'Constitutional';

/**
 * Input for verifying consciousness gate.
 * Matches Rust `VerifyGateInput` struct.
 */
export interface VerifyGateInput {
  actionType: GovernanceActionType;
  actionId?: string;
}

/**
 * Gate verification result (legacy v1).
 * Matches Rust `GateVerificationResult` struct.
 */
export interface GateVerificationResult {
  passed: boolean;
  consciousnessLevel: number;
  requiredConsciousness: number;
  /** @deprecated Use consciousnessLevel */
  phi: number;
  /** @deprecated Use requiredConsciousness */
  requiredPhi: number;
  actionType: GovernanceActionType;
  failureReason?: string;
  gateId: string;
}

/**
 * Gate verification result with provenance tracking (v2).
 * Matches Rust `GateVerificationResultV2` struct.
 */
export interface GateVerificationResultV2 {
  passed: boolean;
  /** Consciousness level, or null if unavailable */
  consciousnessLevel: number | null;
  requiredConsciousness: number;
  /** How the consciousness value was obtained */
  provenance: ConsciousnessProvenance;
  actionType: GovernanceActionType;
  failureReason?: string;
}

/**
 * Input for recording an authenticated consciousness attestation.
 * Matches Rust `RecordConsciousnessAttestationInput` struct.
 */
export interface RecordConsciousnessAttestationInput {
  /** Consciousness level (0.0-1.0) */
  consciousnessLevel: number;
  /** Symthaea cognitive cycle number */
  cycleId: number;
  /** Microseconds since Unix epoch when the consciousness level was captured */
  capturedAtUs: number;
  /** Ed25519 signature (64 bytes) over the attestation message */
  signature: number[];
}

/**
 * Input for assessing value alignment.
 * Matches Rust `AssessAlignmentInput` struct.
 */
export interface AssessAlignmentInput {
  /** Proposal identifier */
  proposalId: string;
  /** Proposal content to assess (1-4096 chars) */
  proposalContent: string;
}

/**
 * Input for getting agent snapshots
 */
export interface GetAgentSnapshotsInput {
  agentDid: string;
  limit?: number;
}

/**
 * Consciousness threshold requirements.
 * Matches Rust `ConsciousnessThresholdSummary` struct.
 */
export interface ConsciousnessThresholds {
  basic: number;
  proposalSubmission: number;
  voting: number;
  constitutional: number;
}

/**
 * Input for calculating holistic vote weight.
 * Matches Rust `CalculateWeightInput` struct.
 */
export interface CalculateWeightInput {
  harmonicAlignment?: number;
}

/**
 * Holistic voting weight result.
 * Matches Rust `HolisticVotingWeight` struct.
 */
export interface HolisticVotingWeight {
  reputation: number;
  reputationSquared: number;
  consciousnessLevel: number;
  /** @deprecated Use consciousnessLevel */
  phi: number;
  consciousnessMultiplier: number;
  harmonicAlignment: number;
  harmonicBonus: number;
  finalWeight: number;
  wasCapped: boolean;
  uncappedWeight: number;
  calculationBreakdown: string;
}

/**
 * Input for casting a weighted consensus vote.
 * Matches Rust `CastWeightedVoteInput` struct.
 */
export interface CastWeightedVoteInput {
  proposalId: string;
  proposalType: BridgeProposalType;
  round: number;
  decision: VoteDecision;
  harmonicAlignment?: number;
  reason?: string;
}

/**
 * Weighted vote result.
 * Matches Rust `WeightedVoteResult` struct.
 */
export interface WeightedVoteResult {
  voteId: string;
  weight: number;
  weightBreakdown: string;
  decision: VoteDecision;
  consciousnessAtVote: number;
  /** @deprecated Use consciousnessAtVote */
  phiAtVote: number;
  proposalType: BridgeProposalType;
  thresholdRequired: number;
}

/**
 * Adaptive threshold for a proposal type.
 * Matches Rust `AdaptiveThreshold` struct.
 */
export interface AdaptiveThreshold {
  /** Base threshold percentage (0.0-1.0) */
  baseThreshold: number;
  /** Minimum consciousness level required for voters */
  minVoterConsciousness: number;
  /** @deprecated Use minVoterConsciousness */
  minVoterPhi: number;
  /** Minimum participation (number of voters) */
  minParticipation: number;
  /** Quorum percentage (minimum % of eligible voters) */
  quorum: number;
  /** Time extension allowed (in seconds) */
  maxExtensionSecs: number;
}

/**
 * Participant status including streak and cooldown.
 * Matches Rust `ParticipantStatus` struct.
 */
export interface ParticipantStatus {
  agentDid: string;
  isActive: boolean;
  baseReputation: number;
  effectiveReputation: number;
  streakCount: number;
  streakBonus: number;
  inCooldown: boolean;
  currentConsciousness: number;
  federatedScore: number;
  roundsParticipated: number;
  successfulVotes: number;
  successRate: number;
  slashingEvents: number;
  canVoteStandard: boolean;
  canVoteEmergency: boolean;
  canVoteConstitutional: boolean;
}

/**
 * Input for updating federated reputation signals.
 * All fields are optional — only provided fields are updated.
 * Matches Rust `UpdateFederatedReputationInput` struct.
 */
export interface UpdateFederatedReputationInput {
  // Identity domain signals
  identityVerification?: number;
  credentialCount?: number;
  credentialQuality?: number;
  // Knowledge domain signals
  epistemicContributions?: number;
  factcheckAccuracy?: number;
  darkSpotsResolved?: number;
  // Finance domain signals (max 5% of final score)
  stakeWeight?: number;
  paymentReliability?: number;
  escrowCompletionRate?: number;
  // FL domain signals
  pogqScore?: number;
  flContributions?: number;
  byzantineCleanRate?: number;
  // Governance domain signals
  votingParticipation?: number;
  proposalSuccessRate?: number;
  consensusAlignment?: number;
}

/**
 * Input for getting round votes.
 * Matches Rust `GetRoundVotesInput` struct.
 */
export interface GetRoundVotesInput {
  proposalId: string;
  round: number;
}

/**
 * Input for calculating round result.
 * Matches Rust `CalculateRoundResultInput` struct.
 */
export interface CalculateRoundResultInput {
  proposalId: string;
  round: number;
  proposalType: BridgeProposalType;
  eligibleVoters: number;
}

/**
 * Consensus round result.
 * Matches Rust `RoundResult` struct.
 */
export interface RoundResult {
  proposalId: string;
  round: number;
  proposalType: BridgeProposalType;
  totalWeight: number;
  weightedApprovals: number;
  weightedRejections: number;
  voteCount: number;
  requiredThreshold: number;
  approvalPercentage: number;
  quorumMet: boolean;
  consensusReached: boolean;
  rejected: boolean;
  result: string;
}

/**
 * Input for dispatching personal cluster call
 */
export interface DispatchPersonalCallInput {
  zomeName: string;
  fnName: string;
  payload?: string;
}
