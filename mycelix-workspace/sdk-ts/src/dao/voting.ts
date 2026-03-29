// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Voting Client
 *
 * Cast votes and tally results across all voting mechanisms.
 */

import { type ActionHash, type Record as HolochainRecord } from '@holochain/client';

import type { DAOClient } from './client';
import type {
  Vote,
  QuadraticVote,
  VoteTally,
  CastVoteInput,
  CastQuadraticVoteInput,
} from './types';

/**
 * Client for voting operations
 */
export class VotingClient {
  constructor(private client: DAOClient) {}

  private get zome() {
    return this.client.getZomeName('governance');
  }

  /**
   * Cast a vote on a proposal (Reputation-Weighted or Equal-Weight)
   *
   * @example
   * ```typescript
   * await voting.castVote({
   *   proposal: proposalHash,
   *   choice: 'Yes',
   *   justification: 'I support this because...',
   * });
   * ```
   */
  async castVote(input: CastVoteInput): Promise<Vote> {
    const record = await this.client.callZome<HolochainRecord>(
      this.zome,
      'cast_vote',
      input
    );

    return this.decodeVote(record);
  }

  /**
   * Cast a quadratic vote on a treasury proposal
   *
   * @example
   * ```typescript
   * // Spend 25 credits for 5 effective votes
   * await voting.castQuadraticVote({
   *   proposal: proposalHash,
   *   credits: 25,
   *   round_id: 'round-001',
   * });
   * ```
   */
  async castQuadraticVote(input: CastQuadraticVoteInput): Promise<QuadraticVote> {
    const record = await this.client.callZome<HolochainRecord>(
      this.zome,
      'cast_quadratic_vote',
      input
    );

    return this.decodeQuadraticVote(record);
  }

  /**
   * Get all votes for a proposal
   */
  async getVotesForProposal(proposal: ActionHash): Promise<Vote[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_votes_for_proposal',
      proposal
    );

    return records.map(r => this.decodeVote(r));
  }

  /**
   * Tally votes for a proposal
   *
   * @example
   * ```typescript
   * const tally = await voting.tallyVotes(proposalHash);
   * console.log(`Result: ${tally.result}`);
   * console.log(`Yes: ${tally.yes_votes}, No: ${tally.no_votes}`);
   * console.log(`Quorum reached: ${tally.quorum_reached}`);
   * ```
   */
  async tallyVotes(proposal: ActionHash): Promise<VoteTally> {
    return this.client.callZome<VoteTally>(
      this.zome,
      'tally_votes',
      proposal
    );
  }

  /**
   * Check if the current user has voted on a proposal
   */
  async hasVoted(proposal: ActionHash): Promise<boolean> {
    const votes = await this.getVotesForProposal(proposal);
    const myPubKey = await this.client.getMyPubKey();

    return votes.some(v => v.voter.toString() === myPubKey.toString());
  }

  /**
   * Get the user's vote on a proposal (if any)
   */
  async getMyVote(proposal: ActionHash): Promise<Vote | null> {
    const votes = await this.getVotesForProposal(proposal);
    const myPubKey = await this.client.getMyPubKey();

    return votes.find(v => v.voter.toString() === myPubKey.toString()) || null;
  }

  /**
   * Calculate effective votes for a quadratic vote
   */
  calculateEffectiveVotes(credits: number): number {
    return Math.sqrt(credits);
  }

  /**
   * Calculate credits needed for desired effective votes
   */
  calculateCreditsNeeded(effectiveVotes: number): number {
    return Math.pow(effectiveVotes, 2);
  }

  private decodeVote(record: HolochainRecord): Vote {
    return (record as any).entry.Present.entry as Vote;
  }

  private decodeQuadraticVote(record: HolochainRecord): QuadraticVote {
    return (record as any).entry.Present.entry as QuadraticVote;
  }
}
