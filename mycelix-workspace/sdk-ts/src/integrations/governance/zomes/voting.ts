// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Voting Zome Client
 *
 * Handles vote casting, MATL-weighted power calculation, and vote queries.
 *
 * @module @mycelix/sdk/integrations/governance/zomes/voting
 */

import { GovernanceSdkError } from '../types';

import type {
  Vote,
  CastVoteInput,
  VotingPowerQuery,
  VotingPowerResult,
  MatlVotingPowerInput,
  QuorumStatus,
  VotingStats,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Voting client
 */
export interface VotingClientConfig {
  /** Role ID for the governance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: VotingClientConfig = {
  roleId: 'governance',
  zomeName: 'governance',
};

/**
 * Client for Voting operations
 *
 * @example
 * ```typescript
 * const voting = new VotingClient(holochainClient);
 *
 * // Check voting power before voting
 * const power = await voting.getVotingPower({
 *   dao_id: 'mycelix-core',
 *   voter_did: 'did:mycelix:abc123',
 * });
 *
 * // Cast a vote
 * await voting.castVote({
 *   id: 'vote-001',
 *   proposal_id: 'prop-001',
 *   voter_did: 'did:mycelix:abc123',
 *   choice: 'Yes',
 *   weight: power,
 *   reason: 'This aligns with our mission',
 * });
 *
 * // Check voting stats
 * const stats = await voting.getVotingStats('prop-001');
 * console.log(`Yes: ${stats.yes_percentage}%`);
 * ```
 */
export class VotingClient {
  private readonly config: VotingClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<VotingClientConfig> = {}
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
      const errorMessage = error instanceof Error ? error.message : String(error);

      // Map common errors to specific codes
      if (errorMessage.includes('already voted')) {
        throw new GovernanceSdkError('ALREADY_VOTED', 'You have already voted on this proposal', error);
      }
      if (errorMessage.includes('not active')) {
        throw new GovernanceSdkError('PROPOSAL_NOT_ACTIVE', 'Proposal is not in active voting status', error);
      }
      if (errorMessage.includes('not a member')) {
        throw new GovernanceSdkError('NOT_MEMBER', 'Voter is not a member of this DAO', error);
      }
      if (errorMessage.includes('Voting period has ended')) {
        throw new GovernanceSdkError('VOTING_CLOSED', 'Voting period has ended', error);
      }

      throw new GovernanceSdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${errorMessage}`,
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
  // Vote Casting
  // ============================================================================

  /**
   * Cast a vote on a proposal
   *
   * Vote weight is validated against the voter's membership voting power.
   * The vote is rejected if:
   * - Voter is not a DAO member
   * - Proposal is not in Active status
   * - Voting period has not started or has ended
   * - Voter has already voted
   *
   * @param input - Vote parameters
   * @returns The cast vote
   * @throws {GovernanceSdkError} See conditions above
   */
  async castVote(input: CastVoteInput): Promise<Vote> {
    const now = Date.now() * 1000;
    const voteInput: Vote = {
      ...input,
      created: now,
    };

    const record = await this.call<HolochainRecord>('cast_vote', voteInput);
    return this.extractEntry<Vote>(record);
  }

  /**
   * Get all votes for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of votes
   */
  async getVotesForProposal(proposalId: string): Promise<Vote[]> {
    const records = await this.call<HolochainRecord[]>('get_votes_for_proposal', proposalId);
    return records.map(r => this.extractEntry<Vote>(r));
  }

  // ============================================================================
  // Voting Power
  // ============================================================================

  /**
   * Get voting power for a voter in a DAO
   *
   * Returns the base membership voting power. For MATL-weighted power,
   * use calculateMatlVotingPower().
   *
   * @param query - Voting power query
   * @returns Voting power value
   */
  async getVotingPower(query: VotingPowerQuery): Promise<number> {
    return this.call<number>('get_voting_power', query);
  }

  /**
   * Calculate MATL-weighted voting power
   *
   * Voting power formula:
   * - Base power from membership
   * - MATL multiplier (0.5 to 2.0 based on trust score)
   * - Participation bonus (up to 20%)
   * - Stake multiplier (if staking enabled)
   *
   * @param input - MATL voting power parameters
   * @returns Detailed voting power breakdown
   */
  async calculateMatlVotingPower(input: MatlVotingPowerInput): Promise<VotingPowerResult> {
    return this.call<VotingPowerResult>('calculate_matl_voting_power', input);
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
  async checkQuorum(proposalId: string): Promise<QuorumStatus> {
    return this.call<QuorumStatus>('check_quorum', proposalId);
  }

  /**
   * Get detailed voting statistics for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Voting statistics
   */
  async getVotingStats(proposalId: string): Promise<VotingStats> {
    return this.call<VotingStats>('get_voting_stats', proposalId);
  }

  // ============================================================================
  // Vote Analysis
  // ============================================================================

  /**
   * Check if a voter has already voted on a proposal
   *
   * @param proposalId - Proposal identifier
   * @param voterDid - Voter's DID
   * @returns True if already voted
   */
  async hasVoted(proposalId: string, voterDid: string): Promise<boolean> {
    const votes = await this.getVotesForProposal(proposalId);
    return votes.some(v => v.voter_did === voterDid);
  }

  /**
   * Get a voter's vote on a proposal
   *
   * @param proposalId - Proposal identifier
   * @param voterDid - Voter's DID
   * @returns The vote or null if not voted
   */
  async getVote(proposalId: string, voterDid: string): Promise<Vote | null> {
    const votes = await this.getVotesForProposal(proposalId);
    return votes.find(v => v.voter_did === voterDid) || null;
  }

  /**
   * Aggregate votes by choice
   *
   * @param votes - Array of votes
   * @returns Vote counts by choice
   */
  aggregateVotes(votes: Vote[]): { yes: number; no: number; abstain: number; total: number } {
    const result = { yes: 0, no: 0, abstain: 0, total: 0 };

    for (const vote of votes) {
      result.total += vote.weight;
      switch (vote.choice) {
        case 'Yes':
          result.yes += vote.weight;
          break;
        case 'No':
          result.no += vote.weight;
          break;
        case 'Abstain':
          result.abstain += vote.weight;
          break;
      }
    }

    return result;
  }

  /**
   * Get vote distribution percentages
   *
   * @param votes - Array of votes
   * @returns Percentages by choice
   */
  getVoteDistribution(votes: Vote[]): { yes: number; no: number; abstain: number } {
    const agg = this.aggregateVotes(votes);
    if (agg.total === 0) {
      return { yes: 0, no: 0, abstain: 0 };
    }

    return {
      yes: (agg.yes / agg.total) * 100,
      no: (agg.no / agg.total) * 100,
      abstain: (agg.abstain / agg.total) * 100,
    };
  }
}
