/**
 * Voting Zome Client
 *
 * Handles vote casting, MATL-weighted power calculation, and vote queries
 * for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/voting
 */

import type { AppClient, Record as HolochainRecord } from '@holochain/client';
import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';
import type {
  Vote,
  CastVoteInput,
  VotingPowerQuery,
  VotingPowerBreakdown,
  QuorumStatus,
  VotingStats,
  VoteChoice,
} from './types';
import type { ActionHash } from '../../generated/common';
// Note: GovernanceError removed as unused

/**
 * Configuration for the Voting client
 */
export interface VotingClientConfig extends Partial<ZomeClientConfig> {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
}

const DEFAULT_CONFIG: VotingClientConfig = {
  roleName: 'governance',
};

/**
 * Client for Voting operations
 *
 * Provides comprehensive voting functionality including MATL-weighted voting,
 * delegated voting, and detailed statistics.
 *
 * @example
 * ```typescript
 * import { VotingClient } from '@mycelix/sdk/clients/governance';
 *
 * const voting = new VotingClient(appClient);
 *
 * // Get your voting power
 * const power = await voting.getVotingPower({
 *   daoId: 'uhCkkp...',
 *   voterDid: 'did:mycelix:alice',
 *   includeDelegated: true,
 * });
 *
 * // Cast a vote
 * const vote = await voting.castVote({
 *   proposalId: 'uhCkkq...',
 *   choice: 'Approve',
 *   weight: power.totalPower,
 *   reason: 'Strong alignment with community values',
 * });
 *
 * // Check voting stats
 * const stats = await voting.getVotingStats('uhCkkq...');
 * console.log(`Approval: ${stats.approvePercentage.toFixed(1)}%`);
 * ```
 */
export class VotingClient extends ZomeClient {
  protected readonly zomeName = 'voting';

  constructor(client: AppClient, config: VotingClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Vote Casting
  // ============================================================================

  /**
   * Cast a vote on a proposal
   *
   * Vote weight is validated against the voter's available voting power.
   * Rejects if:
   * - Voter is not a DAO member
   * - Proposal is not in Active status
   * - Voting period has not started or has ended
   * - Voter has already voted
   * - Claimed weight exceeds available power
   *
   * @param input - Vote parameters
   * @returns The cast vote
   */
  async castVote(input: CastVoteInput): Promise<Vote> {
    const record = await this.callZomeOnce<HolochainRecord>('cast_vote', {
      proposal_id: input.proposalId,
      choice: input.choice,
      weight: input.weight,
      reason: input.reason,
      delegated_from: input.delegatedFrom,
    });
    return this.mapVote(record);
  }

  /**
   * Cast vote with automatic weight calculation
   *
   * Automatically uses maximum available voting power.
   *
   * @param proposalId - Proposal to vote on
   * @param choice - Vote choice
   * @param reason - Optional reason
   * @returns The cast vote
   */
  async castVoteAuto(
    proposalId: ActionHash,
    choice: VoteChoice,
    reason?: string
  ): Promise<Vote> {
    const record = await this.callZomeOnce<HolochainRecord>('cast_vote_auto', {
      proposal_id: proposalId,
      choice,
      reason,
    });
    return this.mapVote(record);
  }

  /**
   * Cast delegated vote on behalf of a delegator
   *
   * @param proposalId - Proposal to vote on
   * @param delegatorDid - DID of the person who delegated
   * @param choice - Vote choice
   * @param reason - Optional reason
   * @returns The cast vote
   */
  async castDelegatedVote(
    proposalId: ActionHash,
    delegatorDid: string,
    choice: VoteChoice,
    reason?: string
  ): Promise<Vote> {
    const record = await this.callZomeOnce<HolochainRecord>('cast_delegated_vote', {
      proposal_id: proposalId,
      delegator_did: delegatorDid,
      choice,
      reason,
    });
    return this.mapVote(record);
  }

  /**
   * Change an existing vote (if allowed by DAO rules)
   *
   * @param proposalId - Proposal identifier
   * @param newChoice - New vote choice
   * @param reason - Updated reason
   * @returns The updated vote
   */
  async changeVote(
    proposalId: ActionHash,
    newChoice: VoteChoice,
    reason?: string
  ): Promise<Vote> {
    const record = await this.callZomeOnce<HolochainRecord>('change_vote', {
      proposal_id: proposalId,
      new_choice: newChoice,
      reason,
    });
    return this.mapVote(record);
  }

  // ============================================================================
  // Vote Queries
  // ============================================================================

  /**
   * Get all votes for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of votes
   */
  async getVotesForProposal(proposalId: ActionHash): Promise<Vote[]> {
    const records = await this.callZome<HolochainRecord[]>('get_votes_for_proposal', proposalId);
    return records.map(r => this.mapVote(r));
  }

  /**
   * Get a voter's vote on a proposal
   *
   * @param proposalId - Proposal identifier
   * @param voterDid - Voter's DID
   * @returns The vote or null if not voted
   */
  async getVote(proposalId: ActionHash, voterDid: string): Promise<Vote | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_vote', {
      proposal_id: proposalId,
      voter_did: voterDid,
    });
    if (!record) return null;
    return this.mapVote(record);
  }

  /**
   * Get my vote on a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns My vote or null
   */
  async getMyVote(proposalId: ActionHash): Promise<Vote | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_my_vote', proposalId);
    if (!record) return null;
    return this.mapVote(record);
  }

  /**
   * Check if a voter has already voted on a proposal
   *
   * @param proposalId - Proposal identifier
   * @param voterDid - Voter's DID
   * @returns True if already voted
   */
  async hasVoted(proposalId: ActionHash, voterDid: string): Promise<boolean> {
    return this.callZome<boolean>('has_voted', {
      proposal_id: proposalId,
      voter_did: voterDid,
    });
  }

  /**
   * Get votes by a specific voter across all proposals
   *
   * @param voterDid - Voter's DID
   * @param limit - Maximum results
   * @returns Array of votes
   */
  async getVotesByVoter(voterDid: string, limit?: number): Promise<Vote[]> {
    const records = await this.callZome<HolochainRecord[]>('get_votes_by_voter', {
      voter_did: voterDid,
      limit,
    });
    return records.map(r => this.mapVote(r));
  }

  // ============================================================================
  // Voting Power
  // ============================================================================

  /**
   * Get total voting power for a voter
   *
   * @param query - Voting power query parameters
   * @returns Total voting power
   */
  async getVotingPower(query: VotingPowerQuery): Promise<number> {
    return this.callZome<number>('get_voting_power', {
      dao_id: query.daoId,
      voter_did: query.voterDid,
      include_delegated: query.includeDelegated ?? true,
    });
  }

  /**
   * Get detailed voting power breakdown
   *
   * Returns all components of voting power calculation including
   * MATL reputation multiplier and delegated power.
   *
   * @param query - Voting power query
   * @returns Detailed breakdown
   */
  async getVotingPowerBreakdown(query: VotingPowerQuery): Promise<VotingPowerBreakdown> {
    const result = await this.callZome<any>('get_voting_power_breakdown', {
      dao_id: query.daoId,
      voter_did: query.voterDid,
      include_delegated: query.includeDelegated ?? true,
    });
    return {
      basePower: result.base_power,
      matlMultiplier: result.matl_multiplier,
      participationBonus: result.participation_bonus,
      delegatedPower: result.delegated_power,
      stakeMultiplier: result.stake_multiplier,
      totalPower: result.total_power,
    };
  }

  /**
   * Calculate MATL-weighted voting power
   *
   * Applies reputation-based adjustments to voting power.
   *
   * @param daoId - DAO identifier
   * @param voterDid - Voter's DID
   * @param matlScore - MATL trust score (0-1)
   * @param participationRate - Participation rate (0-1)
   * @returns Adjusted voting power
   */
  async calculateMATLVotingPower(
    daoId: ActionHash,
    voterDid: string,
    matlScore: number,
    participationRate: number
  ): Promise<VotingPowerBreakdown> {
    const result = await this.callZome<any>('calculate_matl_voting_power', {
      dao_id: daoId,
      voter_did: voterDid,
      matl_score: matlScore,
      participation_rate: participationRate,
    });
    return {
      basePower: result.base_power,
      matlMultiplier: result.matl_multiplier,
      participationBonus: result.participation_bonus,
      delegatedPower: result.delegated_power,
      stakeMultiplier: result.stake_multiplier,
      totalPower: result.total_power,
    };
  }

  // ============================================================================
  // Quorum & Statistics
  // ============================================================================

  /**
   * Check quorum status for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Quorum status with progress
   */
  async checkQuorum(proposalId: ActionHash): Promise<QuorumStatus> {
    const result = await this.callZome<any>('check_quorum', proposalId);
    return {
      proposalId: result.proposal_id,
      currentVotes: result.current_votes,
      requiredVotes: result.required_votes,
      quorumMet: result.quorum_met,
      quorumPercentage: result.quorum_percentage,
    };
  }

  /**
   * Get detailed voting statistics for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Voting statistics
   */
  async getVotingStats(proposalId: ActionHash): Promise<VotingStats> {
    const result = await this.callZome<any>('get_voting_stats', proposalId);
    return {
      proposalId: result.proposal_id,
      approveWeight: result.approve_weight,
      rejectWeight: result.reject_weight,
      abstainWeight: result.abstain_weight,
      totalWeight: result.total_weight,
      voterCount: result.voter_count,
      approvePercentage: result.approve_percentage,
      rejectPercentage: result.reject_percentage,
      abstainPercentage: result.abstain_percentage,
      isActive: result.is_active,
      timeRemaining: result.time_remaining,
    };
  }

  // ============================================================================
  // Vote Analysis Helpers
  // ============================================================================

  /**
   * Aggregate votes by choice (client-side calculation)
   *
   * @param votes - Array of votes
   * @returns Aggregated totals
   */
  aggregateVotes(votes: Vote[]): {
    approve: number;
    reject: number;
    abstain: number;
    total: number;
    voterCount: number;
  } {
    const result = { approve: 0, reject: 0, abstain: 0, total: 0, voterCount: votes.length };

    for (const vote of votes) {
      result.total += vote.weight;
      switch (vote.choice) {
        case 'Approve':
          result.approve += vote.weight;
          break;
        case 'Reject':
          result.reject += vote.weight;
          break;
        case 'Abstain':
          result.abstain += vote.weight;
          break;
      }
    }

    return result;
  }

  /**
   * Get vote distribution percentages (client-side calculation)
   *
   * @param votes - Array of votes
   * @returns Percentages by choice
   */
  getVoteDistribution(votes: Vote[]): {
    approve: number;
    reject: number;
    abstain: number;
  } {
    const agg = this.aggregateVotes(votes);
    if (agg.total === 0) {
      return { approve: 0, reject: 0, abstain: 0 };
    }

    return {
      approve: (agg.approve / agg.total) * 100,
      reject: (agg.reject / agg.total) * 100,
      abstain: (agg.abstain / agg.total) * 100,
    };
  }

  /**
   * Get top voters by weight
   *
   * @param votes - Array of votes
   * @param limit - Maximum results
   * @returns Top voters sorted by weight
   */
  getTopVoters(votes: Vote[], limit: number = 10): Vote[] {
    return [...votes].sort((a, b) => b.weight - a.weight).slice(0, limit);
  }

  /**
   * Group votes by choice
   *
   * @param votes - Array of votes
   * @returns Votes grouped by choice
   */
  groupVotesByChoice(votes: Vote[]): Record<VoteChoice, Vote[]> {
    return {
      Approve: votes.filter(v => v.choice === 'Approve'),
      Reject: votes.filter(v => v.choice === 'Reject'),
      Abstain: votes.filter(v => v.choice === 'Abstain'),
    };
  }

  /**
   * Calculate participation rate
   *
   * @param totalVoted - Total voting power that participated
   * @param totalEligible - Total eligible voting power
   * @returns Participation rate (0-1)
   */
  calculateParticipationRate(totalVoted: number, totalEligible: number): number {
    if (totalEligible === 0) return 0;
    return totalVoted / totalEligible;
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Map Holochain record to Vote type
   */
  private mapVote(record: HolochainRecord): Vote {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      proposalId: entry.proposal_id,
      voterDid: entry.voter_did,
      choice: entry.choice,
      weight: entry.weight,
      reason: entry.reason,
      delegatedFrom: entry.delegated_from,
      votedAt: entry.voted_at,
    };
  }
}
