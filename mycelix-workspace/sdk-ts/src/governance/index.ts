/**
 * Mycelix Governance Module
 *
 * TypeScript client for the mycelix-governance hApp.
 * Provides DAO management, proposals, voting, and delegation.
 *
 * @module @mycelix/sdk/governance
 */

// ============================================================================
// Types
// ============================================================================

/** Holochain record wrapper */
export interface HolochainRecord<T = unknown> {
  signed_action: {
    hashed: { hash: string; content: unknown };
    signature: string;
  };
  entry: { Present: T };
}

/** Generic zome call interface */
export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Proposal Types
// ============================================================================

/** Proposal type */
export type ProposalType = 'Standard' | 'Emergency' | 'Constitutional';

/** Proposal status */
export type ProposalStatus =
  | 'Draft'
  | 'Active'
  | 'Passed'
  | 'Rejected'
  | 'Executed'
  | 'Expired'
  | 'Cancelled';

/** Vote choice */
export type VoteChoice = 'Approve' | 'Reject' | 'Abstain';

/** A DAO proposal */
export interface Proposal {
  /** Proposal ID */
  id: string;
  /** DAO this proposal belongs to */
  dao_id: string;
  /** Title */
  title: string;
  /** Detailed description */
  description: string;
  /** Proposer's DID */
  proposer: string;
  /** Proposal type */
  proposal_type: ProposalType;
  /** Current status */
  status: ProposalStatus;
  /** Voting period in hours */
  voting_period_hours: number;
  /** Required quorum (0.0-1.0) */
  quorum: number;
  /** Approval threshold (0.0-1.0) */
  threshold: number;
  /** Total approve weight */
  approve_weight: number;
  /** Total reject weight */
  reject_weight: number;
  /** Total abstain weight */
  abstain_weight: number;
  /** Execution payload (JSON) */
  execution_payload?: string;
  /** Category tag */
  category?: string;
  /** When voting ends (timestamp) */
  voting_ends: number;
  /** Creation timestamp */
  created: number;
  /** Execution timestamp */
  executed_at?: number;
}

/** Input for creating a proposal */
export interface CreateProposalInput {
  dao_id: string;
  title: string;
  description: string;
  proposal_type: ProposalType;
  voting_period_hours: number;
  quorum: number;
  threshold?: number;
  execution_payload?: string;
  category?: string;
}

/** A vote on a proposal */
export interface Vote {
  /** Vote ID */
  id: string;
  /** Proposal being voted on */
  proposal_id: string;
  /** Voter's DID */
  voter: string;
  /** Vote choice */
  choice: VoteChoice;
  /** Vote weight */
  weight: number;
  /** Reason for vote */
  reason?: string;
  /** If vote was delegated */
  delegated_from?: string;
  /** Vote timestamp */
  voted_at: number;
}

/** Input for casting a vote */
export interface CastVoteInput {
  proposal_id: string;
  choice: VoteChoice;
  weight: number;
  reason?: string;
}

// ============================================================================
// DAO Types
// ============================================================================

/** Member role in a DAO */
export type MemberRole = 'Member' | 'Delegate' | 'Steward' | 'Admin';

/** DAO configuration */
export interface DAO {
  /** DAO ID */
  id: string;
  /** DAO name */
  name: string;
  /** Description */
  description: string;
  /** Creator's DID */
  creator: string;
  /** Charter/constitution hash */
  charter_hash?: string;
  /** Default voting period in hours */
  default_voting_period: number;
  /** Default quorum */
  default_quorum: number;
  /** Default threshold */
  default_threshold: number;
  /** Member count */
  member_count: number;
  /** Total voting power */
  total_voting_power: number;
  /** Creation timestamp */
  created: number;
  /** Last update timestamp */
  updated: number;
}

/** Input for creating a DAO */
export interface CreateDAOInput {
  name: string;
  description: string;
  charter_hash?: string;
  default_voting_period: number;
  default_quorum: number;
  default_threshold: number;
}

/** DAO member */
export interface DAOMember {
  /** Member's DID */
  did: string;
  /** DAO ID */
  dao_id: string;
  /** Member role */
  role: MemberRole;
  /** Base voting power */
  voting_power: number;
  /** Delegated power received */
  delegated_power: number;
  /** DID of delegate (if delegating) */
  delegate_to?: string;
  /** When member joined */
  joined_at: number;
  /** Last activity timestamp */
  last_active: number;
}

/** Delegation record */
export interface Delegation {
  /** Delegation ID */
  id: string;
  /** Delegator's DID */
  delegator: string;
  /** Delegate's DID */
  delegate: string;
  /** DAO ID */
  dao_id: string;
  /** Delegated power */
  power: number;
  /** Categories to delegate (optional) */
  categories?: string[];
  /** Expiration timestamp */
  expires_at?: number;
  /** Creation timestamp */
  created: number;
}

/** Input for creating a delegation */
export interface CreateDelegationInput {
  delegate: string;
  dao_id: string;
  power: number;
  categories?: string[];
  expires_at?: number;
}

// ============================================================================
// Execution Types
// ============================================================================

/** Execution status */
export type ExecutionStatus = 'Pending' | 'InProgress' | 'Completed' | 'Failed' | 'Reverted';

/** Execution record */
export interface Execution {
  /** Execution ID */
  id: string;
  /** Proposal ID */
  proposal_id: string;
  /** Target hApp */
  target_happ: string;
  /** Action to execute */
  action: string;
  /** Payload */
  payload: string;
  /** Current status */
  status: ExecutionStatus;
  /** Requested timestamp */
  requested_at: number;
  /** Executed timestamp */
  executed_at?: number;
  /** Error message if failed */
  error?: string;
}

// ============================================================================
// Proposal Client
// ============================================================================

const GOVERNANCE_ROLE = 'governance';
const PROPOSALS_ZOME = 'proposals';

/**
 * Proposals Client - Manage DAO proposals
 *
 * @example
 * ```typescript
 * const proposals = new ProposalsClient(conductor);
 *
 * const proposal = await proposals.createProposal({
 *   dao_id: 'dao-1',
 *   title: 'Increase Treasury Allocation',
 *   description: 'Allocate 10% more to development fund',
 *   proposal_type: 'Standard',
 *   voting_period_hours: 72,
 *   quorum: 0.6,
 * });
 * ```
 */
export class ProposalsClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Create a new proposal */
  async createProposal(input: CreateProposalInput): Promise<HolochainRecord<Proposal>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: PROPOSALS_ZOME,
      fn_name: 'create_proposal',
      payload: input,
    }) as Promise<HolochainRecord<Proposal>>;
  }

  /** Get proposal by ID */
  async getProposal(proposalId: string): Promise<HolochainRecord<Proposal> | null> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: PROPOSALS_ZOME,
      fn_name: 'get_proposal',
      payload: proposalId,
    }) as Promise<HolochainRecord<Proposal> | null>;
  }

  /** Get proposals by DAO */
  async getProposalsByDAO(daoId: string): Promise<HolochainRecord<Proposal>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: PROPOSALS_ZOME,
      fn_name: 'get_proposals_by_dao',
      payload: daoId,
    }) as Promise<HolochainRecord<Proposal>[]>;
  }

  /** Get active proposals for a DAO */
  async getActiveProposals(daoId: string): Promise<HolochainRecord<Proposal>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: PROPOSALS_ZOME,
      fn_name: 'get_active_proposals',
      payload: daoId,
    }) as Promise<HolochainRecord<Proposal>[]>;
  }

  /** Get proposals by proposer */
  async getProposalsByProposer(proposerDid: string): Promise<HolochainRecord<Proposal>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: PROPOSALS_ZOME,
      fn_name: 'get_proposals_by_proposer',
      payload: proposerDid,
    }) as Promise<HolochainRecord<Proposal>[]>;
  }

  /** Cancel a proposal (proposer only) */
  async cancelProposal(proposalId: string): Promise<HolochainRecord<Proposal>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: PROPOSALS_ZOME,
      fn_name: 'cancel_proposal',
      payload: proposalId,
    }) as Promise<HolochainRecord<Proposal>>;
  }

  /** Finalize a proposal after voting ends */
  async finalizeProposal(proposalId: string): Promise<HolochainRecord<Proposal>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: PROPOSALS_ZOME,
      fn_name: 'finalize_proposal',
      payload: proposalId,
    }) as Promise<HolochainRecord<Proposal>>;
  }
}

// ============================================================================
// Voting Client
// ============================================================================

const VOTING_ZOME = 'voting';

/**
 * Voting Client - Cast and manage votes
 *
 * @example
 * ```typescript
 * const voting = new VotingClient(conductor);
 *
 * await voting.castVote({
 *   proposal_id: 'proposal-1',
 *   choice: 'Approve',
 *   weight: 100,
 *   reason: 'Great initiative!',
 * });
 * ```
 */
export class VotingClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Cast a vote on a proposal */
  async castVote(input: CastVoteInput): Promise<HolochainRecord<Vote>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: VOTING_ZOME,
      fn_name: 'cast_vote',
      payload: input,
    }) as Promise<HolochainRecord<Vote>>;
  }

  /** Get votes for a proposal */
  async getVotesForProposal(proposalId: string): Promise<HolochainRecord<Vote>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: VOTING_ZOME,
      fn_name: 'get_votes_for_proposal',
      payload: proposalId,
    }) as Promise<HolochainRecord<Vote>[]>;
  }

  /** Get votes by voter */
  async getVotesByVoter(voterDid: string): Promise<HolochainRecord<Vote>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: VOTING_ZOME,
      fn_name: 'get_votes_by_voter',
      payload: voterDid,
    }) as Promise<HolochainRecord<Vote>[]>;
  }

  /** Check if user has voted on a proposal */
  async hasVoted(proposalId: string, voterDid: string): Promise<boolean> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: VOTING_ZOME,
      fn_name: 'has_voted',
      payload: { proposal_id: proposalId, voter: voterDid },
    }) as Promise<boolean>;
  }

  /** Get vote tally for a proposal */
  async getVoteTally(proposalId: string): Promise<{
    approve: number;
    reject: number;
    abstain: number;
    total: number;
  }> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: VOTING_ZOME,
      fn_name: 'get_vote_tally',
      payload: proposalId,
    }) as Promise<{ approve: number; reject: number; abstain: number; total: number }>;
  }
}

// ============================================================================
// DAO Client
// ============================================================================

const DAO_ZOME = 'dao';

/**
 * DAO Client - Manage DAOs and membership
 *
 * @example
 * ```typescript
 * const dao = new DAOClient(conductor);
 *
 * const myDAO = await dao.createDAO({
 *   name: 'Luminous DAO',
 *   description: 'Decentralized governance for Luminous',
 *   default_voting_period: 72,
 *   default_quorum: 0.5,
 *   default_threshold: 0.5,
 * });
 * ```
 */
export class DAOClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Create a new DAO */
  async createDAO(input: CreateDAOInput): Promise<HolochainRecord<DAO>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DAO_ZOME,
      fn_name: 'create_dao',
      payload: input,
    }) as Promise<HolochainRecord<DAO>>;
  }

  /** Get DAO by ID */
  async getDAO(daoId: string): Promise<HolochainRecord<DAO> | null> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DAO_ZOME,
      fn_name: 'get_dao',
      payload: daoId,
    }) as Promise<HolochainRecord<DAO> | null>;
  }

  /** List all DAOs */
  async listDAOs(): Promise<HolochainRecord<DAO>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DAO_ZOME,
      fn_name: 'list_daos',
      payload: null,
    }) as Promise<HolochainRecord<DAO>[]>;
  }

  /** Join a DAO */
  async joinDAO(daoId: string, votingPower: number = 1): Promise<HolochainRecord<DAOMember>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DAO_ZOME,
      fn_name: 'join_dao',
      payload: { dao_id: daoId, voting_power: votingPower },
    }) as Promise<HolochainRecord<DAOMember>>;
  }

  /** Get DAO members */
  async getMembers(daoId: string): Promise<HolochainRecord<DAOMember>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DAO_ZOME,
      fn_name: 'get_members',
      payload: daoId,
    }) as Promise<HolochainRecord<DAOMember>[]>;
  }

  /** Get member by DID */
  async getMember(daoId: string, did: string): Promise<HolochainRecord<DAOMember> | null> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DAO_ZOME,
      fn_name: 'get_member',
      payload: { dao_id: daoId, did },
    }) as Promise<HolochainRecord<DAOMember> | null>;
  }

  /** Update member role */
  async updateMemberRole(
    daoId: string,
    did: string,
    role: MemberRole
  ): Promise<HolochainRecord<DAOMember>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DAO_ZOME,
      fn_name: 'update_member_role',
      payload: { dao_id: daoId, did, role },
    }) as Promise<HolochainRecord<DAOMember>>;
  }
}

// ============================================================================
// Delegation Client
// ============================================================================

const DELEGATION_ZOME = 'delegation';

/**
 * Delegation Client - Manage voting power delegation
 *
 * @example
 * ```typescript
 * const delegation = new DelegationClient(conductor);
 *
 * await delegation.delegate({
 *   delegate: 'did:mycelix:bob',
 *   dao_id: 'dao-1',
 *   power: 50,
 * });
 * ```
 */
export class DelegationClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Create a delegation */
  async delegate(input: CreateDelegationInput): Promise<HolochainRecord<Delegation>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DELEGATION_ZOME,
      fn_name: 'delegate',
      payload: input,
    }) as Promise<HolochainRecord<Delegation>>;
  }

  /** Revoke a delegation */
  async revokeDelegation(delegationId: string): Promise<void> {
    await this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DELEGATION_ZOME,
      fn_name: 'revoke_delegation',
      payload: delegationId,
    });
  }

  /** Get delegations for a delegator */
  async getDelegationsFrom(delegatorDid: string): Promise<HolochainRecord<Delegation>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DELEGATION_ZOME,
      fn_name: 'get_delegations_from',
      payload: delegatorDid,
    }) as Promise<HolochainRecord<Delegation>[]>;
  }

  /** Get delegations to a delegate */
  async getDelegationsTo(delegateDid: string): Promise<HolochainRecord<Delegation>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DELEGATION_ZOME,
      fn_name: 'get_delegations_to',
      payload: delegateDid,
    }) as Promise<HolochainRecord<Delegation>[]>;
  }

  /** Get effective voting power for a member */
  async getEffectiveVotingPower(did: string, daoId: string): Promise<number> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: DELEGATION_ZOME,
      fn_name: 'get_effective_voting_power',
      payload: { did, dao_id: daoId },
    }) as Promise<number>;
  }
}

// ============================================================================
// Execution Client
// ============================================================================

const EXECUTION_ZOME = 'execution';

/**
 * Execution Client - Execute passed proposals
 */
export class ExecutionClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Request execution of a passed proposal */
  async requestExecution(
    proposalId: string,
    targetHapp: string
  ): Promise<HolochainRecord<Execution>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: EXECUTION_ZOME,
      fn_name: 'request_execution',
      payload: { proposal_id: proposalId, target_happ: targetHapp },
    }) as Promise<HolochainRecord<Execution>>;
  }

  /** Get pending executions */
  async getPendingExecutions(targetHapp?: string): Promise<HolochainRecord<Execution>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: EXECUTION_ZOME,
      fn_name: 'get_pending_executions',
      payload: targetHapp ?? null,
    }) as Promise<HolochainRecord<Execution>[]>;
  }

  /** Acknowledge execution completion */
  async acknowledgeExecution(
    executionId: string,
    success: boolean,
    error?: string
  ): Promise<HolochainRecord<Execution>> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: EXECUTION_ZOME,
      fn_name: 'acknowledge_execution',
      payload: { execution_id: executionId, success, error },
    }) as Promise<HolochainRecord<Execution>>;
  }

  /** Get executions for a proposal */
  async getExecutionsForProposal(proposalId: string): Promise<HolochainRecord<Execution>[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: EXECUTION_ZOME,
      fn_name: 'get_executions_for_proposal',
      payload: proposalId,
    }) as Promise<HolochainRecord<Execution>[]>;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all Governance hApp clients (ZomeCallable pattern)
 *
 * For the full 9-client factory (including Constitution, Councils,
 * ThresholdSigning, Bridge), use the clients/governance/ module
 * exported from the main SDK barrel.
 *
 * @example
 * ```typescript
 * const { proposals, voting, dao, delegation, execution } = createGovernanceClients(conductor);
 * ```
 */
export function createGovernanceClients(client: ZomeCallable) {
  return {
    proposals: new ProposalsClient(client),
    voting: new VotingClient(client),
    dao: new DAOClient(client),
    delegation: new DelegationClient(client),
    execution: new ExecutionClient(client),
  };
}

export default {
  ProposalsClient,
  VotingClient,
  DAOClient,
  DelegationClient,
  ExecutionClient,
  createGovernanceClients,
};

// ============================================================================
// Unified Cross-Domain Voting
// ============================================================================

export * from './unified-voting.js';
