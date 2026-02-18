/**
 * @mycelix/sdk Governance Integration
 *
 * hApp-specific adapter for Mycelix-DAO governance providing:
 * - Proposal lifecycle management with reputation-weighted voting
 * - Delegate tracking and reputation cascading
 * - DAO membership and role management
 * - Cross-hApp governance coordination via Bridge
 * - Quadratic voting with MATL trust verification
 *
 * @packageDocumentation
 * @module integrations/governance
 * @see {@link GovernanceService} - Main service class
 * @see {@link getGovernanceService} - Singleton accessor
 *
 * @example Creating and voting on a proposal
 * ```typescript
 * import { getGovernanceService } from '@mycelix/sdk/integrations/governance';
 *
 * const gov = getGovernanceService();
 *
 * // Create a proposal
 * const proposal = gov.createProposal({
 *   title: 'Increase treasury allocation',
 *   description: 'Allocate 10% more to development fund',
 *   proposerId: 'did:mycelix:alice',
 *   daoId: 'luminous-core',
 *   votingPeriodHours: 72,
 *   quorumPercentage: 0.6,
 * });
 *
 * // Cast a vote
 * gov.castVote({
 *   proposalId: proposal.id,
 *   voterId: 'did:mycelix:bob',
 *   choice: 'approve',
 *   weight: 100,
 * });
 * ```
 */


import { LocalBridge } from '../../bridge/index.js';
import { type MycelixClient } from '../../client/index.js';
import {
  createReputation,
  recordPositive,
  type ReputationScore,
} from '../../matl/index.js';
import { GovernanceValidators } from '../../utils/validation.js';

// ============================================================================
// Bridge Zome Types (matching Rust governance_bridge zome)
// ============================================================================

/** Proposal type for cross-hApp queries */
export type BridgeProposalType = 'Standard' | 'Emergency' | 'Constitutional';

/** Execution status */
export type ExecutionStatus = 'Pending' | 'InProgress' | 'Completed' | 'Failed' | 'Reverted';

/** Governance query */
export interface GovernanceQuery {
  id: string;
  query_type: 'ProposalStatus' | 'VotingPower' | 'DelegationChain' | 'ExecutionHistory';
  query_params: string;
  source_happ: string;
  queried_at: number;
}

/** Proposal reference from bridge */
export interface ProposalReference {
  id: string;
  proposal_hash: string;
  source_happ: string;
  title: string;
  proposal_type: BridgeProposalType;
  status: ProposalStatus;
  vote_weight_for: number;
  vote_weight_against: number;
  vote_weight_abstain: number;
  ends_at: number;
  created_at: number;
}

/** Execution request for cross-hApp */
export interface ExecutionRequest {
  id: string;
  proposal_hash: string;
  source_happ: string;
  target_happ: string;
  action: string;
  payload: string;
  status: ExecutionStatus;
  requested_at: number;
  executed_at?: number;
  error_message?: string;
}

/** Governance bridge event types */
export type GovernanceBridgeEventType =
  | 'ProposalCreated'
  | 'ProposalPassed'
  | 'ProposalRejected'
  | 'VoteCast'
  | 'ExecutionRequested'
  | 'ExecutionCompleted'
  | 'DelegationChanged';

/** Governance bridge event */
export interface GovernanceBridgeEvent {
  id: string;
  event_type: GovernanceBridgeEventType;
  proposal_hash?: string;
  voter_did?: string;
  payload: string;
  source_happ: string;
  timestamp: number;
}

/** Query governance input */
export interface QueryGovernanceInput {
  query_type: GovernanceQuery['query_type'];
  query_params: string;
  source_happ: string;
}

/** Request execution input */
export interface RequestExecutionInput {
  proposal_hash: string;
  target_happ: string;
  action: string;
  payload: string;
}

/** Broadcast governance event input */
export interface BroadcastGovernanceEventInput {
  event_type: GovernanceBridgeEventType;
  proposal_hash?: string;
  voter_did?: string;
  payload: string;
}

// ============================================================================
// Governance-Specific Types
// ============================================================================

/** Proposal status lifecycle */
export type ProposalStatus =
  | 'draft'
  | 'active'
  | 'passed'
  | 'rejected'
  | 'executed'
  | 'expired';

/** Vote choice options */
export type VoteChoice = 'approve' | 'reject' | 'abstain';

/** DAO member role */
export type MemberRole = 'member' | 'delegate' | 'steward' | 'admin';

/** Proposal creation input */
export interface ProposalInput {
  title: string;
  description: string;
  proposerId: string;
  daoId: string;
  votingPeriodHours: number;
  quorumPercentage: number;
  executionPayload?: string;
  category?: string;
}

/** Full proposal record */
export interface Proposal {
  id: string;
  title: string;
  description: string;
  proposerId: string;
  daoId: string;
  status: ProposalStatus;
  votingEnds: number;
  quorumPercentage: number;
  approvesWeight: number;
  rejectsWeight: number;
  abstainWeight: number;
  executionPayload?: string;
  category?: string;
  createdAt: number;
  executedAt?: number;
}

/** Vote cast on a proposal */
export interface Vote {
  proposalId: string;
  voterId: string;
  choice: VoteChoice;
  weight: number;
  reason?: string;
  delegatedFrom?: string;
  timestamp: number;
}

/** DAO member profile */
export interface DAOMember {
  did: string;
  daoId: string;
  role: MemberRole;
  reputation: ReputationScore;
  votingPower: number;
  delegatedPower: number;
  delegateTo?: string;
  joinedAt: number;
  lastActive: number;
}

/** Delegation record */
export interface Delegation {
  delegatorId: string;
  delegateId: string;
  daoId: string;
  power: number;
  categories?: string[];
  expiresAt?: number;
  createdAt: number;
}

/** Proposal result summary */
export interface ProposalResult {
  proposalId: string;
  passed: boolean;
  quorumMet: boolean;
  totalVotes: number;
  approvalPercentage: number;
  participationRate: number;
}

// ============================================================================
// Governance Service
// ============================================================================

/**
 * Governance service for DAO proposal and voting management
 *
 * @remarks
 * Uses MATL reputation for vote weighting and trust verification.
 * Supports delegation with reputation cascading.
 */
export class GovernanceService {
  private proposals = new Map<string, Proposal>();
  private votes = new Map<string, Vote[]>();
  private members = new Map<string, DAOMember>();
  private delegations = new Map<string, Delegation[]>();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('governance');
  }

  /**
   * Create a new proposal
   * @throws {MycelixError} If proposal parameters are invalid
   */
  createProposal(input: ProposalInput): Proposal {
    // Validate proposal parameters
    GovernanceValidators.proposal(input.votingPeriodHours, input.quorumPercentage);

    // Verify proposer is a registered member
    const proposerKey = `${input.proposerId}:${input.daoId}`;
    if (!this.members.has(proposerKey)) {
      throw new Error('Proposer must be a registered member of the DAO');
    }

    const id = `proposal-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const proposal: Proposal = {
      id,
      title: input.title,
      description: input.description,
      proposerId: input.proposerId,
      daoId: input.daoId,
      status: 'active',
      votingEnds: Date.now() + input.votingPeriodHours * 60 * 60 * 1000,
      quorumPercentage: input.quorumPercentage,
      approvesWeight: 0,
      rejectsWeight: 0,
      abstainWeight: 0,
      executionPayload: input.executionPayload,
      category: input.category,
      createdAt: Date.now(),
    };

    this.proposals.set(id, proposal);
    this.votes.set(id, []);
    return proposal;
  }

  /**
   * Cast a vote on a proposal
   * @throws {Error} If voter is not a registered member
   */
  castVote(input: Omit<Vote, 'timestamp'>): Vote {
    const proposal = this.proposals.get(input.proposalId);
    if (!proposal) {
      throw new Error(`Proposal ${input.proposalId} not found`);
    }

    // Verify voter is a registered member
    const voterKey = `${input.voterId}:${proposal.daoId}`;
    if (!this.members.has(voterKey)) {
      throw new Error('Voter must be a registered member of the DAO');
    }

    if (proposal.status !== 'active') {
      throw new Error('Proposal is not active');
    }

    if (Date.now() > proposal.votingEnds) {
      throw new Error('Voting period has ended');
    }

    // Check for existing vote
    const existingVotes = this.votes.get(input.proposalId) || [];
    if (existingVotes.some((v) => v.voterId === input.voterId)) {
      throw new Error('Already voted on this proposal');
    }

    const vote: Vote = { ...input, timestamp: Date.now() };
    existingVotes.push(vote);
    this.votes.set(input.proposalId, existingVotes);

    // Update proposal tallies
    switch (input.choice) {
      case 'approve':
        proposal.approvesWeight += input.weight;
        break;
      case 'reject':
        proposal.rejectsWeight += input.weight;
        break;
      case 'abstain':
        proposal.abstainWeight += input.weight;
        break;
    }

    // Record participation reputation
    const member = this.members.get(`${input.voterId}:${proposal.daoId}`);
    if (member) {
      member.reputation = recordPositive(member.reputation);
      member.lastActive = Date.now();
    }

    return vote;
  }

  /**
   * Finalize a proposal after voting period
   */
  finalizeProposal(proposalId: string): ProposalResult {
    const proposal = this.proposals.get(proposalId);
    if (!proposal) {
      throw new Error(`Proposal ${proposalId} not found`);
    }

    const totalWeight =
      proposal.approvesWeight + proposal.rejectsWeight + proposal.abstainWeight;
    const totalPossibleWeight = this.getTotalVotingPower(proposal.daoId);
    const participationRate = totalPossibleWeight > 0 ? totalWeight / totalPossibleWeight : 0;
    const quorumMet = participationRate >= proposal.quorumPercentage;

    const approvalPercentage =
      totalWeight > 0 ? proposal.approvesWeight / totalWeight : 0;
    const passed = quorumMet && approvalPercentage > 0.5;

    proposal.status = passed ? 'passed' : 'rejected';

    return {
      proposalId,
      passed,
      quorumMet,
      totalVotes: this.votes.get(proposalId)?.length || 0,
      approvalPercentage,
      participationRate,
    };
  }

  /**
   * Register a DAO member
   */
  registerMember(member: Omit<DAOMember, 'reputation' | 'lastActive'>): DAOMember {
    const fullMember: DAOMember = {
      ...member,
      reputation: createReputation(member.did),
      lastActive: Date.now(),
    };
    this.members.set(`${member.did}:${member.daoId}`, fullMember);
    return fullMember;
  }

  /**
   * Delegate voting power to another member
   */
  delegate(input: Omit<Delegation, 'createdAt'>): Delegation {
    const delegation: Delegation = { ...input, createdAt: Date.now() };
    const key = `${input.delegatorId}:${input.daoId}`;
    const existing = this.delegations.get(key) || [];
    existing.push(delegation);
    this.delegations.set(key, existing);

    // Update delegate's delegated power
    const delegateMember = this.members.get(`${input.delegateId}:${input.daoId}`);
    if (delegateMember) {
      delegateMember.delegatedPower += input.power;
    }

    return delegation;
  }

  /**
   * Get total voting power in a DAO
   */
  getTotalVotingPower(daoId: string): number {
    let total = 0;
    for (const [key, member] of this.members) {
      if (key.endsWith(`:${daoId}`)) {
        total += member.votingPower + member.delegatedPower;
      }
    }
    return total;
  }

  /**
   * Get member's effective voting power (including delegations)
   */
  getEffectiveVotingPower(did: string, daoId: string): number {
    const member = this.members.get(`${did}:${daoId}`);
    if (!member) return 0;

    // If delegating, power goes to delegate
    if (member.delegateTo) return 0;

    return member.votingPower + member.delegatedPower;
  }

  /**
   * Get proposal by ID
   */
  getProposal(proposalId: string): Proposal | undefined {
    return this.proposals.get(proposalId);
  }

  /**
   * Get votes for a proposal
   */
  getVotes(proposalId: string): Vote[] {
    return this.votes.get(proposalId) || [];
  }

  /**
   * Get active proposals for a DAO
   */
  getActiveProposals(daoId: string): Proposal[] {
    return Array.from(this.proposals.values()).filter(
      (p) => p.daoId === daoId && p.status === 'active'
    );
  }

  /**
   * Cross-hApp reputation query
   */
  async getReputationFromOtherHapps(did: string): Promise<number> {
    // In production, would use bridge to query other hApps
    const scores = this.bridge.getCrossHappReputation(did);
    const total = scores.reduce((sum, s) => sum + s.score, 0);
    return scores.length > 0 ? total / scores.length : 0.5;
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

let instance: GovernanceService | null = null;

/**
 * Get the singleton GovernanceService instance
 */
export function getGovernanceService(): GovernanceService {
  if (!instance) {
    instance = new GovernanceService();
  }
  return instance;
}

/**
 * Reset service (for testing)
 */
export function resetGovernanceService(): void {
  instance = null;
}

// ============================================================================
// Governance Bridge Client (Holochain Zome Calls)
// ============================================================================

const GOVERNANCE_ROLE = 'governance';
const BRIDGE_ZOME = 'governance_bridge';

/**
 * Governance Bridge Client - Direct Holochain zome calls for cross-hApp governance
 */
export class GovernanceBridgeClient {
  constructor(private client: MycelixClient) {}

  /**
   * Query governance information
   */
  async queryGovernance(input: QueryGovernanceInput): Promise<ProposalReference[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_governance',
      payload: input,
    });
  }

  /**
   * Request execution of a passed proposal on another hApp
   */
  async requestExecution(input: RequestExecutionInput): Promise<ExecutionRequest> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'request_execution',
      payload: input,
    });
  }

  /**
   * Get pending executions for a hApp
   */
  async getPendingExecutions(targetHapp: string): Promise<ExecutionRequest[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_pending_executions',
      payload: targetHapp,
    });
  }

  /**
   * Acknowledge execution completion
   */
  async acknowledgeExecution(executionId: string, success: boolean, errorMessage?: string): Promise<ExecutionRequest> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'acknowledge_execution',
      payload: { execution_id: executionId, success, error_message: errorMessage },
    });
  }

  /**
   * Broadcast a governance event
   */
  async broadcastGovernanceEvent(input: BroadcastGovernanceEventInput): Promise<GovernanceBridgeEvent> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_governance_event',
      payload: input,
    });
  }

  /**
   * Get recent governance events
   */
  async getRecentEvents(limit?: number): Promise<GovernanceBridgeEvent[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit ?? 50,
    });
  }

  /**
   * Get events by proposal
   */
  async getEventsByProposal(proposalHash: string): Promise<GovernanceBridgeEvent[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_proposal',
      payload: proposalHash,
    });
  }
}

// Bridge client singleton
let bridgeInstance: GovernanceBridgeClient | null = null;

export function getGovernanceBridgeClient(client: MycelixClient): GovernanceBridgeClient {
  if (!bridgeInstance) bridgeInstance = new GovernanceBridgeClient(client);
  return bridgeInstance;
}

export function resetGovernanceBridgeClient(): void {
  bridgeInstance = null;
}

// ============================================================================
// Holochain Zome Clients (New SDK)
// ============================================================================

/**
 * New Holochain-based Governance SDK
 *
 * These clients provide direct zome calls to the governance Holochain hApp.
 * Use MycelixGovernanceClient for the unified interface.
 *
 * @example
 * ```typescript
 * import { MycelixGovernanceClient } from '@mycelix/sdk/integrations/governance';
 *
 * const governance = await MycelixGovernanceClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Use the unified client
 * const dao = await governance.dao.createDao({...});
 * const proposal = await governance.proposals.createProposal({...});
 * await governance.voting.castVote({...});
 * ```
 */
export { MycelixGovernanceClient } from './client';
export type {
  GovernanceClientConfig,
  GovernanceConnectionOptions,
} from './client';

// Zome clients for advanced usage
export {
  DaoClient,
  ProposalClient,
  VotingClient,
  DelegationClient,
} from './zomes';
export type {
  DaoClientConfig,
  ProposalClientConfig,
  VotingClientConfig,
  DelegationClientConfig,
} from './zomes';

// Types from the new SDK
export type {
  // DAO types
  Dao,
  CreateDaoInput,
  UpdateDaoInput,
  Membership as DaoMembership,
  MemberRole as DaoMemberRole,
  // Proposal types
  Proposal as ZomeProposal,
  CreateProposalInput,
  ProposalType,
  ProposalStatus as ZomeProposalStatus,
  ProposalResult as ZomeProposalResult,
  DaoProposalsQuery,
  // Voting types
  Vote as ZomeVote,
  VoteChoice as ZomeVoteChoice,
  CastVoteInput,
  VotingPowerQuery,
  VotingPowerResult,
  MatlVotingPowerInput,
  QuorumStatus,
  VotingStats,
  // Delegation types
  Delegation as ZomeDelegation,
  DelegationScope,
  DelegatedPowerQuery,
  HasDelegatedQuery,
  // Cross-hApp types
  ParticipationScore,
  GovernanceEvent as ZomeGovernanceEvent,
  GovernanceEventType as ZomeGovernanceEventType,
  // Error types
  GovernanceSdkErrorCode,
  // Phi-weighted voting
  ProposalTier,
  TallyPhiVotesInput,
  PhiWeightedTally,
  PhiTierBreakdown,
  TallySegment,
  // Quadratic voting
  AllocateCreditsInput,
  VoiceCredits,
  CastQuadraticVoteInput,
  QuadraticVote,
  TallyQuadraticVotesInput,
  QuadraticTally,
  // Consciousness metrics
  RecordSnapshotInput,
  VerifyGateInput,
  GateVerificationResult,
  GovernanceActionType,
  PhiThresholds,
  // Execution & timelock
  TimelockStatus,
  Timelock,
  CreateTimelockInput,
  ExecuteTimelockInput,
  MarkTimelockReadyInput,
  VetoTimelockInput,
  GuardianVeto,
  // Fund management
  AllocationStatus,
  FundAllocation,
  LockFundsInput,
  ReleaseFundsInput,
  RefundFundsInput,
  // Council decisions
  DecisionType,
  DecisionStatus,
  CouncilDecision,
  RecordDecisionInput,
  // Threshold signing
  DkgPhase,
  CommitteeScope,
  SigningCommittee,
  CommitteeMember,
  ThresholdSignature,
  CreateCommitteeInput,
  RegisterMemberInput,
  CombineSignaturesInput,
  // Signals
  ProposalSignal,
  ThresholdSigningSignal,
  BridgeSignal,
  GovernanceSignal,
} from './types';

export {
  GovernanceSdkError,
  isProposalSignal,
  isThresholdSigningSignal,
  isBridgeSignal,
  isGovernanceSignal,
} from './types';
