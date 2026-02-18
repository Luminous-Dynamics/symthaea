/**
 * Proposals Zome Client
 *
 * Handles proposal creation, lifecycle management, and queries
 * for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/proposals
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Proposal,
  CreateProposalInput,
  UpdateProposalInput,
  ProposalFilter,
  ProposalResult,
  ProposalStatus,
  ProposalType,
  AddContributionInput,
  DiscussionReadiness,
} from './types';
// GovernanceError removed (unused)
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Proposals client
 */
export interface ProposalsClientConfig extends Partial<ZomeClientConfig> {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
}

const DEFAULT_CONFIG: ProposalsClientConfig = {
  roleName: 'governance',
};

/**
 * Voting periods by proposal type (in hours)
 */
export const VOTING_PERIODS: Record<ProposalType, number> = {
  Standard: 168,       // 7 days
  Emergency: 24,       // 24 hours
  Constitutional: 720, // 30 days
  Parameter: 168,      // 7 days (same as standard)
  Funding: 168,        // 7 days (same as standard)
};

/**
 * Client for Proposal operations
 *
 * Manages the complete proposal lifecycle from creation through execution.
 *
 * @example
 * ```typescript
 * import { ProposalsClient } from '@mycelix/sdk/clients/governance';
 *
 * const proposals = new ProposalsClient(appClient);
 *
 * // Create a new proposal
 * const proposal = await proposals.createProposal({
 *   daoId: 'uhCkkp...',
 *   title: 'Fund Community Garden Project',
 *   description: 'Allocate 5000 MYC tokens to establish...',
 *   proposalType: 'Standard',
 *   executionPayload: JSON.stringify({ action: 'transfer', amount: 5000 }),
 * });
 *
 * // Activate the proposal (start voting)
 * await proposals.activateProposal(proposal.id);
 *
 * // Later, finalize the proposal
 * const result = await proposals.finalizeProposal(proposal.id);
 * if (result.passed) {
 *   console.log('Proposal passed!');
 * }
 * ```
 */
export class ProposalsClient extends ZomeClient {
  protected readonly zomeName = 'proposals';
  

  constructor(client: AppClient, config: ProposalsClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
    
  }

  // ============================================================================
  // Proposal CRUD Operations
  // ============================================================================

  /**
   * Create a new proposal
   *
   * Proposals start in 'Draft' status and must be activated to begin voting.
   * Only DAO members can create proposals.
   *
   * @param input - Proposal creation parameters
   * @returns The created proposal
   */
  async createProposal(input: CreateProposalInput): Promise<Proposal> {
    const record = await this.callZomeOnce<HolochainRecord>('create_proposal', {
      dao_id: input.daoId,
      title: input.title,
      description: input.description,
      proposal_type: input.proposalType,
      category: input.category,
      quorum: input.quorum,
      threshold: input.threshold,
      voting_period_hours: input.votingPeriodHours,
      execution_payload: input.executionPayload,
      execution_target_happ: input.executionTargetHapp,
      discussion_url: input.discussionUrl,
    });
    return this.mapProposal(record);
  }

  /**
   * Get a proposal by ID
   *
   * @param proposalId - Proposal action hash
   * @returns The proposal or null if not found
   */
  async getProposal(proposalId: ActionHash): Promise<Proposal | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_proposal', proposalId);
    if (!record) return null;
    return this.mapProposal(record);
  }

  /**
   * Update a proposal (draft only)
   *
   * Only the proposer can update, and only while in Draft status.
   *
   * @param input - Update parameters
   * @returns The updated proposal
   */
  async updateProposal(input: UpdateProposalInput): Promise<Proposal> {
    const record = await this.callZomeOnce<HolochainRecord>('update_proposal', {
      proposal_id: input.proposalId,
      title: input.title,
      description: input.description,
      category: input.category,
      execution_payload: input.executionPayload,
      discussion_url: input.discussionUrl,
    });
    return this.mapProposal(record);
  }

  /**
   * List proposals with optional filters
   *
   * @param filter - Filter parameters
   * @returns Array of proposals
   */
  async listProposals(filter?: ProposalFilter): Promise<Proposal[]> {
    const records = await this.callZome<HolochainRecord[]>('list_proposals', {
      dao_id: filter?.daoId,
      status: filter?.status,
      proposer_did: filter?.proposerDid,
      proposal_type: filter?.proposalType,
      category: filter?.category,
      voting_open: filter?.votingOpen,
      limit: filter?.limit,
      offset: filter?.offset,
    });
    return records.map(r => this.mapProposal(r));
  }

  /**
   * Get active proposals for a DAO
   *
   * @param daoId - DAO identifier
   * @param limit - Maximum results
   * @returns Array of active proposals
   */
  async getActiveProposals(daoId: ActionHash, limit?: number): Promise<Proposal[]> {
    return this.listProposals({
      daoId,
      status: 'Active',
      limit,
    });
  }

  /**
   * Get proposals by proposer
   *
   * @param proposerDid - Proposer's DID
   * @returns Array of proposals
   */
  async getProposalsByProposer(proposerDid: string): Promise<Proposal[]> {
    return this.listProposals({ proposerDid });
  }

  /**
   * Get proposals by category
   *
   * @param daoId - DAO identifier
   * @param category - Category tag
   * @returns Array of proposals
   */
  async getProposalsByCategory(daoId: ActionHash, category: string): Promise<Proposal[]> {
    return this.listProposals({ daoId, category });
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
   */
  async activateProposal(proposalId: ActionHash): Promise<Proposal> {
    const record = await this.callZomeOnce<HolochainRecord>('activate_proposal', proposalId);
    return this.mapProposal(record);
  }

  /**
   * Cancel a proposal
   *
   * Only the proposer can cancel, and only for Draft or Active proposals.
   *
   * @param proposalId - Proposal identifier
   * @returns The cancelled proposal
   */
  async cancelProposal(proposalId: ActionHash): Promise<Proposal> {
    const record = await this.callZomeOnce<HolochainRecord>('cancel_proposal', proposalId);
    return this.mapProposal(record);
  }

  /**
   * Veto a proposal (admin/founder only)
   *
   * @param proposalId - Proposal identifier
   * @param reason - Reason for veto
   * @returns The vetoed proposal
   */
  async vetoProposal(proposalId: ActionHash, reason: string): Promise<Proposal> {
    const record = await this.callZomeOnce<HolochainRecord>('veto_proposal', {
      proposal_id: proposalId,
      reason,
    });
    return this.mapProposal(record);
  }

  /**
   * Finalize a proposal after voting period ends
   *
   * Calculates final results and sets status to Passed or Rejected.
   * Can be called by anyone after voting ends.
   *
   * @param proposalId - Proposal identifier
   * @returns Finalization result with vote tallies
   */
  async finalizeProposal(proposalId: ActionHash): Promise<ProposalResult> {
    const result = await this.callZomeOnce<any>('finalize_proposal', proposalId);
    return {
      proposalId: result.proposal_id,
      finalStatus: result.final_status,
      totalVotes: result.total_votes,
      approveWeight: result.approve_weight,
      rejectWeight: result.reject_weight,
      abstainWeight: result.abstain_weight,
      quorumMet: result.quorum_met,
      approvalPercentage: result.approval_percentage,
      participationRate: result.participation_rate,
      passed: result.passed,
    };
  }

  /**
   * Mark a proposal as executed
   *
   * Called after successful execution to update status.
   *
   * @param proposalId - Proposal identifier
   * @returns The executed proposal
   */
  async markExecuted(proposalId: ActionHash): Promise<Proposal> {
    const record = await this.callZomeOnce<HolochainRecord>('mark_executed', proposalId);
    return this.mapProposal(record);
  }

  // ============================================================================
  // Proposal Queries & Helpers
  // ============================================================================

  /**
   * Check if a proposal is currently in voting period
   *
   * @param proposal - Proposal to check
   * @returns True if voting is open
   */
  isVotingOpen(proposal: Proposal): boolean {
    const now = Date.now() * 1000; // microseconds
    return (
      proposal.status === 'Active' &&
      now >= proposal.votingStartsAt &&
      now <= proposal.votingEndsAt
    );
  }

  /**
   * Get time remaining in voting period
   *
   * @param proposal - Proposal to check
   * @returns Seconds remaining, or 0 if voting has ended
   */
  getTimeRemaining(proposal: Proposal): number {
    const now = Date.now() * 1000;
    if (now >= proposal.votingEndsAt) return 0;
    return Math.floor((proposal.votingEndsAt - now) / 1000000);
  }

  /**
   * Calculate current approval percentage
   *
   * @param proposal - Proposal to check
   * @returns Approval percentage (0-100)
   */
  getApprovalPercentage(proposal: Proposal): number {
    const totalVotes = proposal.approveWeight + proposal.rejectWeight;
    if (totalVotes === 0) return 0;
    return (proposal.approveWeight / totalVotes) * 100;
  }

  /**
   * Check if proposal has reached quorum
   *
   * @param proposal - Proposal to check
   * @param totalVotingPower - Total voting power in DAO
   * @returns True if quorum is met
   */
  hasQuorum(proposal: Proposal, totalVotingPower: number): boolean {
    const totalVotes = proposal.approveWeight + proposal.rejectWeight + proposal.abstainWeight;
    const requiredVotes = proposal.quorum * totalVotingPower;
    return totalVotes >= requiredVotes;
  }

  /**
   * Check if proposal would pass with current votes
   *
   * @param proposal - Proposal to check
   * @param totalVotingPower - Total voting power in DAO
   * @returns True if proposal would pass
   */
  wouldPass(proposal: Proposal, totalVotingPower: number): boolean {
    if (!this.hasQuorum(proposal, totalVotingPower)) return false;
    return this.getApprovalPercentage(proposal) >= proposal.threshold * 100;
  }

  /**
   * Get voting period for proposal type
   *
   * @param proposalType - Type of proposal
   * @returns Voting period in hours
   */
  getVotingPeriod(proposalType: ProposalType): number {
    return VOTING_PERIODS[proposalType];
  }

  /**
   * Get proposal status description
   *
   * @param status - Proposal status
   * @returns Human-readable description
   */
  getStatusDescription(status: ProposalStatus): string {
    const descriptions: Record<ProposalStatus, string> = {
      Draft: 'Proposal is being prepared and can be edited',
      Active: 'Proposal is currently open for voting',
      Ended: 'Voting period has ended, awaiting tally',
      Approved: 'Proposal has been approved by voters',
      Signed: 'Proposal has been signed by threshold committee',
      Rejected: 'Proposal was rejected by voters',
      Executed: 'Proposal has been executed',
      Cancelled: 'Proposal was cancelled by proposer',
      Failed: 'Proposal execution failed',
    };
    return descriptions[status];
  }

  /**
   * Get recent proposals across all DAOs
   *
   * @param limit - Maximum results (default: 10)
   * @returns Array of recent proposals
   */
  async getRecentProposals(limit: number = 10): Promise<Proposal[]> {
    const records = await this.callZome<HolochainRecord[]>('get_recent_proposals', limit);
    return records.map(r => this.mapProposal(r));
  }

  /**
   * Get proposal history (all status changes)
   *
   * @param proposalId - Proposal identifier
   * @returns Array of status changes with timestamps
   */
  async getProposalHistory(
    proposalId: ActionHash
  ): Promise<Array<{ status: ProposalStatus; timestamp: number; actor?: string }>> {
    return this.callZome('get_proposal_history', proposalId);
  }

  // ============================================================================
  // Discussion System
  // ============================================================================

  /**
   * Generate next proposal ID (e.g. MIP-0001)
   *
   * @returns The next sequential proposal ID
   */
  async generateProposalId(): Promise<string> {
    return this.callZome<string>('generate_proposal_id', null);
  }

  /**
   * Add a contribution to a proposal's discussion
   *
   * Supports threaded replies, harmony tagging, and stance signaling.
   *
   * @param input - Contribution parameters
   * @returns The created contribution record
   */
  async addContribution(input: AddContributionInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('add_contribution', {
      proposal_id: input.proposalId,
      contributor_did: input.contributorDid,
      content: input.content,
      harmony_tags: input.harmonyTags,
      stance: input.stance,
      parent_id: input.parentId,
    });
  }

  /**
   * Get all contributions for a proposal's discussion
   *
   * @param proposalId - Proposal identifier
   * @returns Array of contribution records (sorted by timestamp)
   */
  async getDiscussion(proposalId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_discussion', proposalId);
  }

  /**
   * Get replies to a specific contribution
   *
   * @param contributionId - Contribution identifier
   * @returns Array of reply records (sorted by timestamp)
   */
  async getReplies(contributionId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_replies', contributionId);
  }

  /**
   * Generate a reflection on the discussion phase
   *
   * Analyzes participation, harmony coverage, stance distribution,
   * and thread depth to mirror the discussion quality.
   *
   * @param proposalId - Proposal identifier
   * @returns The discussion reflection record
   */
  async reflectOnDiscussion(proposalId: string): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('reflect_on_discussion', proposalId);
  }

  /**
   * Get all discussion reflections for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of reflection records
   */
  async getDiscussionReflections(proposalId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_discussion_reflections', proposalId);
  }

  /**
   * Check if a proposal's discussion is ready for voting
   *
   * Evaluates contributor count, harmony diversity, and
   * whether key concerns have been addressed.
   *
   * @param proposalId - Proposal identifier
   * @returns Discussion readiness assessment
   */
  async isDiscussionReady(proposalId: string): Promise<DiscussionReadiness> {
    const result = await this.callZome<any>('is_discussion_ready', proposalId);
    return {
      ready: result.ready,
      reasoning: result.reasoning,
      contributorCount: result.contributor_count,
      contributionCount: result.contribution_count,
      harmonyDiversity: result.harmony_diversity,
      unaddressedConcerns: result.unaddressed_concerns,
    };
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Map Holochain record to Proposal type
   */
  private mapProposal(record: HolochainRecord): Proposal {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      daoId: entry.dao_id,
      title: entry.title,
      description: entry.description,
      proposerDid: entry.proposer_did,
      proposalType: entry.proposal_type,
      status: entry.status,
      category: entry.category,
      votingStartsAt: entry.voting_starts_at,
      votingEndsAt: entry.voting_ends_at,
      quorum: entry.quorum,
      threshold: entry.threshold,
      approveWeight: entry.approve_weight,
      rejectWeight: entry.reject_weight,
      abstainWeight: entry.abstain_weight,
      voterCount: entry.voter_count,
      executionPayload: entry.execution_payload,
      executionTargetHapp: entry.execution_target_happ,
      discussionUrl: entry.discussion_url,
      createdAt: entry.created_at,
      executedAt: entry.executed_at,
    };
  }
}
