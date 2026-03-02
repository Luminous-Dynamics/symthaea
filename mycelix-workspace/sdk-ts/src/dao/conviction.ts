/**
 * Conviction Voting Client
 *
 * Time-weighted voting for continuous funding proposals.
 */

import { type ActionHash, type AgentPubKey, type Record } from '@holochain/client';

import type { DAOClient } from './client';
import type { ConvictionVote, ConvictionSummary } from './types';

/**
 * Half-life in hours (default: 7 days)
 */
const DEFAULT_HALF_LIFE_HOURS = 168;

/**
 * Client for conviction voting operations
 */
export class ConvictionClient {
  constructor(private client: DAOClient) {}

  private get zome() {
    return this.client.getZomeName('governance');
  }

  /**
   * Stake conviction on a proposal
   *
   * @example
   * ```typescript
   * // Stake your voting power on a streaming grant
   * await conviction.stake(proposalHash);
   *
   * // Check progress
   * const summary = await conviction.getSummary(proposalHash);
   * console.log(`Progress: ${summary.progress_percentage.toFixed(1)}%`);
   * ```
   */
  async stake(proposal: ActionHash): Promise<ConvictionVote> {
    const record = await this.client.callZome<Record>(
      this.zome,
      'stake_conviction',
      proposal
    );

    return this.decodeConviction(record);
  }

  /**
   * Withdraw conviction from a proposal
   */
  async withdraw(proposal: ActionHash): Promise<ConvictionVote> {
    const record = await this.client.callZome<Record>(
      this.zome,
      'withdraw_conviction',
      proposal
    );

    return this.decodeConviction(record);
  }

  /**
   * Move conviction from one proposal to another
   *
   * Note: Conviction resets when moved!
   */
  async move(fromProposal: ActionHash, toProposal: ActionHash): Promise<ConvictionVote> {
    const record = await this.client.callZome<Record>(
      this.zome,
      'move_conviction',
      { from_proposal: fromProposal, to_proposal: toProposal }
    );

    return this.decodeConviction(record);
  }

  /**
   * Get conviction summary for a proposal
   */
  async getSummary(proposal: ActionHash): Promise<ConvictionSummary> {
    return this.client.callZome<ConvictionSummary>(
      this.zome,
      'get_total_conviction',
      proposal
    );
  }

  /**
   * Check if a proposal passes based on current conviction
   */
  async checkPasses(proposal: ActionHash): Promise<boolean> {
    return this.client.callZome<boolean>(
      this.zome,
      'check_conviction_passes',
      proposal
    );
  }

  /**
   * Calculate conviction threshold for a funding request
   */
  async calculateThreshold(
    fundingRequest: number,
    availableTreasury: number,
    totalActiveStake: number
  ): Promise<number> {
    return this.client.callZome<number>(
      this.zome,
      'calculate_conviction_threshold',
      {
        funding_request: fundingRequest,
        available_treasury: availableTreasury,
        total_active_stake: totalActiveStake,
      }
    );
  }

  /**
   * Calculate conviction at a given time (client-side)
   *
   * conviction(t) = weight × (1 - e^(-t/τ))
   */
  calculateConviction(
    weight: number,
    hoursStaked: number,
    halfLifeHours: number = DEFAULT_HALF_LIFE_HOURS
  ): number {
    return weight * (1 - Math.exp(-hoursStaked / halfLifeHours));
  }

  /**
   * Calculate time to reach target conviction
   *
   * Solves: target = weight × (1 - e^(-t/τ))
   * t = -τ × ln(1 - target/weight)
   */
  calculateTimeToTarget(
    weight: number,
    targetConviction: number,
    halfLifeHours: number = DEFAULT_HALF_LIFE_HOURS
  ): number {
    if (targetConviction >= weight) {
      return Infinity; // Can never reach
    }

    return -halfLifeHours * Math.log(1 - targetConviction / weight);
  }

  /**
   * Estimate time for proposal to pass
   */
  async estimateTimeToPass(proposal: ActionHash): Promise<{
    hours: number;
    days: number;
    willPass: boolean;
  }> {
    const summary = await this.getSummary(proposal);

    if (summary.passes) {
      return { hours: 0, days: 0, willPass: true };
    }

    if (summary.active_stakers === 0) {
      return { hours: Infinity, days: Infinity, willPass: false };
    }

    // Estimate based on current staking rate
    // Note: remainingNeeded could be used for more precise estimation
    // const remainingNeeded = summary.threshold - summary.total_conviction;

    // Rough estimate: assume current active stake continues growing
    // This is a simplification; real calculation would need per-staker data
    const hours = this.calculateTimeToTarget(
      summary.active_conviction * 1.5, // Assume some growth
      summary.threshold,
      DEFAULT_HALF_LIFE_HOURS
    );

    return {
      hours,
      days: hours / 24,
      willPass: hours < Infinity,
    };
  }

  /**
   * Get all active convictions for the current agent
   */
  async getMyConvictions(): Promise<ConvictionVote[]> {
    const myPubKey = await this.client.getMyPubKey();
    return this.getConvictionsFrom(myPubKey);
  }

  /**
   * Get all active convictions for a given agent
   */
  async getConvictionsFrom(agent: AgentPubKey): Promise<ConvictionVote[]> {
    const records = await this.client.callZome<Record[]>(
      this.zome,
      'get_voter_convictions',
      agent
    );
    return records.map((r) => this.decodeConviction(r));
  }

  private decodeConviction(record: Record): ConvictionVote {
    return (record as any).entry.Present.entry as ConvictionVote;
  }
}
