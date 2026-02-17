/**
 * Proposal Zome Client
 *
 * Handles proposal creation, lifecycle management, and queries.
 *
 * @module @mycelix/sdk/integrations/governance/zomes/proposal
 */

import { GovernanceSdkError } from '../types';

import type {
  Proposal,
  CreateProposalInput,
  ProposalStatus,
  ProposalResult,
  DaoProposalsQuery,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Proposal client
 */
export interface ProposalClientConfig {
  /** Role ID for the governance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: ProposalClientConfig = {
  roleId: 'governance',
  zomeName: 'governance',
};

/**
 * Client for Proposal operations
 *
 * @example
 * ```typescript
 * const proposals = new ProposalClient(holochainClient);
 *
 * // Create a new proposal
 * const proposal = await proposals.createProposal({
 *   id: 'prop-001',
 *   dao_id: 'mycelix-core',
 *   title: 'Increase validator rewards',
 *   description: 'Proposal to increase rewards by 10%...',
 *   proposer_did: 'did:mycelix:abc123',
 *   proposal_type: 'Standard',
 *   quorum_percentage: 33,
 *   approval_threshold: 51,
 * });
 *
 * // Activate the proposal to start voting
 * await proposals.activateProposal('prop-001');
 *
 * // Later, finalize the proposal
 * const result = await proposals.finalizeProposal('prop-001');
 * console.log(`Proposal ${result.passed ? 'passed' : 'failed'}`);
 * ```
 */
export class ProposalClient {
  private readonly config: ProposalClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<ProposalClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      throw new GovernanceSdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Extract entry from a Holochain record
   */
  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new GovernanceSdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Proposal CRUD
  // ============================================================================

  /**
   * Create a new proposal
   *
   * Proposals are created in 'Draft' status and must be activated
   * to start the voting period.
   *
   * @param input - Proposal creation parameters
   * @returns The created proposal
   * @throws {GovernanceSdkError} If proposer is not a DAO member
   */
  async createProposal(input: CreateProposalInput): Promise<Proposal> {
    const now = Date.now() * 1000; // Convert to microseconds
    const proposalInput: Proposal = {
      ...input,
      status: 'Draft' as ProposalStatus,
      voting_starts: 0,
      voting_ends: 0,
      yes_votes: 0,
      no_votes: 0,
      abstain_votes: 0,
      created: now,
    };

    const record = await this.call<HolochainRecord>('create_proposal', proposalInput);
    return this.extractEntry<Proposal>(record);
  }

  /**
   * Get a proposal by ID
   *
   * @param proposalId - Proposal identifier (base64-encoded ActionHash)
   * @returns The proposal or null if not found
   */
  async getProposal(proposalId: string): Promise<Proposal | null> {
    const record = await this.call<HolochainRecord | null>('get_proposal', proposalId);
    if (!record) return null;
    return this.extractEntry<Proposal>(record);
  }

  /**
   * Get all proposals for a DAO
   *
   * @param query - Query parameters
   * @returns Array of proposals
   */
  async getProposals(query: DaoProposalsQuery): Promise<Proposal[]> {
    return this.call<Proposal[]>('get_dao_proposals', query);
  }

  /**
   * Get active proposals for a DAO
   *
   * @param daoId - DAO identifier
   * @param limit - Maximum number of results
   * @returns Array of active proposals
   */
  async getActiveProposals(daoId: string, limit?: number): Promise<Proposal[]> {
    return this.getProposals({
      dao_id: daoId,
      status_filter: 'Active',
      limit,
    });
  }

  // ============================================================================
  // Proposal Lifecycle
  // ============================================================================

  /**
   * Activate a draft proposal to start voting
   *
   * Sets voting period based on proposal type:
   * - Standard: 7 days
   * - Emergency: 24 hours
   * - Constitutional: 30 days
   *
   * @param proposalId - Proposal identifier
   * @returns The activated proposal
   * @throws {GovernanceSdkError} If proposal is not in Draft status
   */
  async activateProposal(proposalId: string): Promise<Proposal> {
    const record = await this.call<HolochainRecord>('activate_proposal', proposalId);
    return this.extractEntry<Proposal>(record);
  }

  /**
   * Cancel a proposal
   *
   * Can only be called by the proposer on Draft or Active proposals.
   *
   * @param proposalId - Proposal identifier
   * @returns The cancelled proposal
   * @throws {GovernanceSdkError} If caller is not the proposer
   */
  async cancelProposal(proposalId: string): Promise<Proposal> {
    const record = await this.call<HolochainRecord>('cancel_proposal', proposalId);
    return this.extractEntry<Proposal>(record);
  }

  /**
   * Finalize a proposal after voting period ends
   *
   * Calculates final results and sets status to Passed or Failed.
   *
   * @param proposalId - Proposal identifier
   * @returns Finalization result with vote counts
   * @throws {GovernanceSdkError} If voting period has not ended
   */
  async finalizeProposal(proposalId: string): Promise<ProposalResult> {
    return this.call<ProposalResult>('finalize_proposal', proposalId);
  }

  /**
   * Execute a passed proposal
   *
   * Triggers the execution_payload if present.
   *
   * @param proposalId - Proposal identifier
   * @returns The executed proposal
   * @throws {GovernanceSdkError} If proposal has not passed
   */
  async executeProposal(proposalId: string): Promise<Proposal> {
    const record = await this.call<HolochainRecord>('execute_proposal', proposalId);
    return this.extractEntry<Proposal>(record);
  }

  // ============================================================================
  // Proposal Queries
  // ============================================================================

  /**
   * Check if a proposal is currently in voting period
   *
   * @param proposal - Proposal to check
   * @returns True if voting is open
   */
  isVotingOpen(proposal: Proposal): boolean {
    const now = Date.now() * 1000;
    return (
      proposal.status === 'Active' &&
      now >= proposal.voting_starts &&
      now <= proposal.voting_ends
    );
  }

  /**
   * Get time remaining in voting period
   *
   * @param proposal - Proposal to check
   * @returns Milliseconds remaining, or 0 if voting has ended
   */
  getTimeRemaining(proposal: Proposal): number {
    const now = Date.now() * 1000;
    if (now >= proposal.voting_ends) return 0;
    return Math.floor((proposal.voting_ends - now) / 1000);
  }

  /**
   * Calculate current approval percentage
   *
   * @param proposal - Proposal to check
   * @returns Approval percentage (0-100)
   */
  getApprovalPercentage(proposal: Proposal): number {
    const totalVotes = proposal.yes_votes + proposal.no_votes;
    if (totalVotes === 0) return 0;
    return (proposal.yes_votes / totalVotes) * 100;
  }

  /**
   * Check if proposal has reached quorum
   *
   * @param proposal - Proposal to check
   * @param totalMemberVotingPower - Total voting power in DAO
   * @returns True if quorum is met
   */
  hasQuorum(proposal: Proposal, totalMemberVotingPower: number): boolean {
    const totalVotes = proposal.yes_votes + proposal.no_votes + proposal.abstain_votes;
    const requiredVotes = (proposal.quorum_percentage / 100) * totalMemberVotingPower;
    return totalVotes >= requiredVotes;
  }
}
