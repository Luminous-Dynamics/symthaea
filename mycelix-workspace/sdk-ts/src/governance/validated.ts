/**
 * @mycelix/sdk Governance Validated Clients
 *
 * Provides input-validated versions of all Governance clients.
 * All inputs are validated using Zod schemas before being sent to the conductor.
 *
 * @packageDocumentation
 * @module governance/validated
 */

import { z } from 'zod';

import { MycelixError, ErrorCode } from '../errors.js';

import {
  ProposalsClient,
  VotingClient,
  DAOClient,
  DelegationClient,
  ExecutionClient,
  type ZomeCallable,
  type HolochainRecord,
  type Proposal,
  type CreateProposalInput,
  type Vote,
  type CastVoteInput,
  type DAO,
  type CreateDAOInput,
  type DAOMember,
  type MemberRole,
  type Delegation,
  type CreateDelegationInput,
  type Execution,
} from './index.js';

// ============================================================================
// Validation Schemas
// ============================================================================

const didSchema = z
  .string()
  .refine((val) => val.startsWith('did:'), { message: 'Must be a valid DID (start with "did:")' });

const proposalTypeSchema = z.enum(['Standard', 'Emergency', 'Constitutional']);

const voteChoiceSchema = z.enum(['Approve', 'Reject', 'Abstain']);

const memberRoleSchema = z.enum(['Member', 'Delegate', 'Steward', 'Admin']);

const createProposalInputSchema = z.object({
  dao_id: z.string().min(1, 'DAO ID is required'),
  title: z.string().min(1, 'Title is required').max(200, 'Title too long'),
  description: z.string().min(1, 'Description is required'),
  proposal_type: proposalTypeSchema,
  voting_period_hours: z
    .number()
    .min(1, 'Voting period must be at least 1 hour')
    .max(720, 'Voting period cannot exceed 30 days'),
  quorum: z.number().min(0, 'Quorum must be positive').max(1, 'Quorum cannot exceed 1'),
  threshold: z.number().min(0).max(1).optional(),
  execution_payload: z.string().optional(),
  category: z.string().optional(),
});

const castVoteInputSchema = z.object({
  proposal_id: z.string().min(1, 'Proposal ID is required'),
  choice: voteChoiceSchema,
  weight: z.number().min(0, 'Weight must be positive'),
  reason: z.string().optional(),
});

const createDAOInputSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100, 'Name too long'),
  description: z.string().min(1, 'Description is required'),
  charter_hash: z.string().optional(),
  default_voting_period: z.number().min(1).max(720),
  default_quorum: z.number().min(0).max(1),
  default_threshold: z.number().min(0).max(1),
});

const createDelegationInputSchema = z.object({
  delegate: didSchema,
  dao_id: z.string().min(1),
  power: z.number().min(0, 'Power must be positive'),
  categories: z.array(z.string()).optional(),
  expires_at: z.number().positive().optional(),
});

// ============================================================================
// Validation Utility
// ============================================================================

function validateOrThrow<T>(schema: z.ZodSchema<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errors = result.error.issues.map((e) => `${e.path.join('.')}: ${e.message}`).join('; ');
    throw new MycelixError(
      `Validation failed for ${context}: ${errors}`,
      ErrorCode.INVALID_ARGUMENT
    );
  }
  return result.data;
}

// ============================================================================
// Validated Proposals Client
// ============================================================================

/**
 * Validated Proposals Client - All inputs are validated before zome calls
 */
export class ValidatedProposalsClient {
  private client: ProposalsClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new ProposalsClient(zomeClient);
  }

  async createProposal(input: CreateProposalInput): Promise<HolochainRecord<Proposal>> {
    validateOrThrow(createProposalInputSchema, input, 'createProposal input');
    return this.client.createProposal(input);
  }

  async getProposal(proposalId: string): Promise<HolochainRecord<Proposal> | null> {
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    return this.client.getProposal(proposalId);
  }

  async getProposalsByDAO(daoId: string): Promise<HolochainRecord<Proposal>[]> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    return this.client.getProposalsByDAO(daoId);
  }

  async getActiveProposals(daoId: string): Promise<HolochainRecord<Proposal>[]> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    return this.client.getActiveProposals(daoId);
  }

  async getProposalsByProposer(proposerDid: string): Promise<HolochainRecord<Proposal>[]> {
    validateOrThrow(didSchema, proposerDid, 'proposerDid');
    return this.client.getProposalsByProposer(proposerDid);
  }

  async cancelProposal(proposalId: string): Promise<HolochainRecord<Proposal>> {
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    return this.client.cancelProposal(proposalId);
  }

  async finalizeProposal(proposalId: string): Promise<HolochainRecord<Proposal>> {
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    return this.client.finalizeProposal(proposalId);
  }
}

// ============================================================================
// Validated Voting Client
// ============================================================================

/**
 * Validated Voting Client
 */
export class ValidatedVotingClient {
  private client: VotingClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new VotingClient(zomeClient);
  }

  async castVote(input: CastVoteInput): Promise<HolochainRecord<Vote>> {
    validateOrThrow(castVoteInputSchema, input, 'castVote input');
    return this.client.castVote(input);
  }

  async getVotesForProposal(proposalId: string): Promise<HolochainRecord<Vote>[]> {
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    return this.client.getVotesForProposal(proposalId);
  }

  async getVotesByVoter(voterDid: string): Promise<HolochainRecord<Vote>[]> {
    validateOrThrow(didSchema, voterDid, 'voterDid');
    return this.client.getVotesByVoter(voterDid);
  }

  async hasVoted(proposalId: string, voterDid: string): Promise<boolean> {
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    validateOrThrow(didSchema, voterDid, 'voterDid');
    return this.client.hasVoted(proposalId, voterDid);
  }

  async getVoteTally(proposalId: string): Promise<{
    approve: number;
    reject: number;
    abstain: number;
    total: number;
  }> {
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    return this.client.getVoteTally(proposalId);
  }
}

// ============================================================================
// Validated DAO Client
// ============================================================================

/**
 * Validated DAO Client
 */
export class ValidatedDAOClient {
  private client: DAOClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new DAOClient(zomeClient);
  }

  async createDAO(input: CreateDAOInput): Promise<HolochainRecord<DAO>> {
    validateOrThrow(createDAOInputSchema, input, 'createDAO input');
    return this.client.createDAO(input);
  }

  async getDAO(daoId: string): Promise<HolochainRecord<DAO> | null> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    return this.client.getDAO(daoId);
  }

  async listDAOs(): Promise<HolochainRecord<DAO>[]> {
    return this.client.listDAOs();
  }

  async joinDAO(daoId: string, votingPower: number = 1): Promise<HolochainRecord<DAOMember>> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    validateOrThrow(z.number().min(0), votingPower, 'votingPower');
    return this.client.joinDAO(daoId, votingPower);
  }

  async getMembers(daoId: string): Promise<HolochainRecord<DAOMember>[]> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    return this.client.getMembers(daoId);
  }

  async getMember(daoId: string, did: string): Promise<HolochainRecord<DAOMember> | null> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    validateOrThrow(didSchema, did, 'did');
    return this.client.getMember(daoId, did);
  }

  async updateMemberRole(
    daoId: string,
    did: string,
    role: MemberRole
  ): Promise<HolochainRecord<DAOMember>> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    validateOrThrow(didSchema, did, 'did');
    validateOrThrow(memberRoleSchema, role, 'role');
    return this.client.updateMemberRole(daoId, did, role);
  }
}

// ============================================================================
// Validated Delegation Client
// ============================================================================

/**
 * Validated Delegation Client
 */
export class ValidatedDelegationClient {
  private client: DelegationClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new DelegationClient(zomeClient);
  }

  async delegate(input: CreateDelegationInput): Promise<HolochainRecord<Delegation>> {
    validateOrThrow(createDelegationInputSchema, input, 'delegate input');
    return this.client.delegate(input);
  }

  async revokeDelegation(delegationId: string): Promise<void> {
    validateOrThrow(z.string().min(1), delegationId, 'delegationId');
    return this.client.revokeDelegation(delegationId);
  }

  async getDelegationsFrom(delegatorDid: string): Promise<HolochainRecord<Delegation>[]> {
    validateOrThrow(didSchema, delegatorDid, 'delegatorDid');
    return this.client.getDelegationsFrom(delegatorDid);
  }

  async getDelegationsTo(delegateDid: string): Promise<HolochainRecord<Delegation>[]> {
    validateOrThrow(didSchema, delegateDid, 'delegateDid');
    return this.client.getDelegationsTo(delegateDid);
  }

  async getEffectiveVotingPower(did: string, daoId: string): Promise<number> {
    validateOrThrow(didSchema, did, 'did');
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    return this.client.getEffectiveVotingPower(did, daoId);
  }
}

// ============================================================================
// Validated Execution Client
// ============================================================================

/**
 * Validated Execution Client
 */
export class ValidatedExecutionClient {
  private client: ExecutionClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new ExecutionClient(zomeClient);
  }

  async requestExecution(
    proposalId: string,
    targetHapp: string
  ): Promise<HolochainRecord<Execution>> {
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    validateOrThrow(z.string().min(1), targetHapp, 'targetHapp');
    return this.client.requestExecution(proposalId, targetHapp);
  }

  async getPendingExecutions(targetHapp?: string): Promise<HolochainRecord<Execution>[]> {
    if (targetHapp !== undefined) {
      validateOrThrow(z.string().min(1), targetHapp, 'targetHapp');
    }
    return this.client.getPendingExecutions(targetHapp);
  }

  async acknowledgeExecution(
    executionId: string,
    success: boolean,
    error?: string
  ): Promise<HolochainRecord<Execution>> {
    validateOrThrow(z.string().min(1), executionId, 'executionId');
    validateOrThrow(z.boolean(), success, 'success');
    if (error !== undefined) {
      validateOrThrow(z.string(), error, 'error');
    }
    return this.client.acknowledgeExecution(executionId, success, error);
  }

  async getExecutionsForProposal(proposalId: string): Promise<HolochainRecord<Execution>[]> {
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    return this.client.getExecutionsForProposal(proposalId);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all validated Governance hApp clients (ZomeCallable pattern)
 *
 * For the full 9-client factory (including Constitution, Councils,
 * ThresholdSigning, Bridge), use the clients/governance/ module
 * exported from the main SDK barrel.
 */
export function createValidatedGovernanceClients(client: ZomeCallable) {
  return {
    proposals: new ValidatedProposalsClient(client),
    voting: new ValidatedVotingClient(client),
    dao: new ValidatedDAOClient(client),
    delegation: new ValidatedDelegationClient(client),
    execution: new ValidatedExecutionClient(client),
  };
}
