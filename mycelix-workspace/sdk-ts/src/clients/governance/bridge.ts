/**
 * Bridge Zome Client
 *
 * Handles cross-hApp governance coordination, event broadcasting,
 * and inter-hApp queries for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/bridge
 */

import type { AppClient, Record as HolochainRecord } from '@holochain/client';
import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';
import type {
  GovernanceBridgeEvent,
  GovernanceBridgeEventType,
  GovernanceQuery,
  CrossHappProposal,
  BroadcastEventInput,
  ParticipationScore,
} from './types';
// GovernanceError removed (unused)
import type { ActionHash } from '../../generated/common';

/**
 * Configuration for the Bridge client
 */
export interface BridgeClientConfig extends Partial<ZomeClientConfig> {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
  /** Source hApp identifier for bridge events */
  sourceHapp?: string;
}

const DEFAULT_CONFIG: BridgeClientConfig = {
  roleName: 'governance',
  sourceHapp: 'governance',
};

/**
 * Client for Cross-hApp Bridge operations
 *
 * Enables governance coordination across multiple hApps in the Mycelix ecosystem:
 * - Broadcasting governance events to other hApps
 * - Querying governance state from other hApps
 * - Coordinating cross-hApp proposal execution
 * - Sharing participation/reputation scores
 *
 * @example
 * ```typescript
 * import { BridgeClient } from '@mycelix/sdk/clients/governance';
 *
 * const bridge = new BridgeClient(appClient, { sourceHapp: 'my-happ' });
 *
 * // Broadcast a governance event
 * await bridge.broadcastEvent({
 *   eventType: 'ProposalPassed',
 *   daoId: 'uhCkkp...',
 *   proposalId: 'uhCkkq...',
 *   payload: JSON.stringify({ action: 'treasury_allocation', amount: 5000 }),
 * });
 *
 * // Query governance from another hApp
 * const proposals = await bridge.queryGovernance({
 *   queryType: 'ProposalStatus',
 *   queryParams: JSON.stringify({ daoId: 'uhCkkp...' }),
 *   sourceHapp: 'finance',
 * });
 *
 * // Get participation score for reputation
 * const score = await bridge.getParticipationScore('did:mycelix:alice');
 * ```
 */
export class BridgeClient extends ZomeClient {
  protected readonly zomeName = 'bridge';

  constructor(client: AppClient, config: BridgeClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Event Broadcasting
  // ============================================================================

  /**
   * Broadcast a governance event to other hApps
   *
   * Events are stored on-chain and can be queried by other hApps.
   *
   * @param input - Event parameters
   * @returns The broadcast event
   */
  async broadcastEvent(input: BroadcastEventInput): Promise<GovernanceBridgeEvent> {
    const record = await this.callZomeOnce<HolochainRecord>('broadcast_governance_event', {
      event_type: input.eventType,
      proposal_id: input.proposalId,
      subject: input.subject,
      payload: input.payload,
    });
    return this.mapEvent(record);
  }

  /**
   * Broadcast proposal created event
   *
   * @param daoId - DAO identifier
   * @param proposalId - Proposal identifier
   * @param proposerDid - Proposer's DID
   * @param title - Proposal title
   * @returns The broadcast event
   */
  async broadcastProposalCreated(
    proposalId: string,
    proposerDid: string,
    title: string
  ): Promise<GovernanceBridgeEvent> {
    return this.broadcastEvent({
      eventType: 'ProposalCreated',
      proposalId,
      subject: `Proposal created by ${proposerDid}`,
      payload: JSON.stringify({ title, proposer: proposerDid }),
    });
  }

  /**
   * Broadcast proposal passed event
   *
   * @param daoId - DAO identifier
   * @param proposalId - Proposal identifier
   * @param approvalPercentage - Final approval percentage
   * @returns The broadcast event
   */
  async broadcastProposalPassed(
    proposalId: string,
    approvalPercentage: number
  ): Promise<GovernanceBridgeEvent> {
    return this.broadcastEvent({
      eventType: 'ProposalPassed',
      proposalId,
      subject: `Proposal passed with ${approvalPercentage}% approval`,
      payload: JSON.stringify({ approval_percentage: approvalPercentage }),
    });
  }

  /**
   * Broadcast proposal rejected event
   *
   * @param daoId - DAO identifier
   * @param proposalId - Proposal identifier
   * @param reason - Rejection reason (quorum, threshold, etc.)
   * @returns The broadcast event
   */
  async broadcastProposalFailed(
    proposalId: string,
    reason: string
  ): Promise<GovernanceBridgeEvent> {
    return this.broadcastEvent({
      eventType: 'ProposalFailed',
      proposalId,
      subject: `Proposal failed: ${reason}`,
      payload: JSON.stringify({ reason }),
    });
  }

  /**
   * Broadcast vote cast event
   *
   * @param daoId - DAO identifier
   * @param proposalId - Proposal identifier
   * @param voterDid - Voter's DID
   * @param choice - Vote choice
   * @param weight - Vote weight
   * @returns The broadcast event
   */
  async broadcastVoteReceived(
    proposalId: string,
    voterDid: string,
    choice: string,
    weight: number
  ): Promise<GovernanceBridgeEvent> {
    return this.broadcastEvent({
      eventType: 'VoteReceived',
      proposalId,
      subject: `Vote by ${voterDid}: ${choice}`,
      payload: JSON.stringify({ voter: voterDid, choice, weight }),
    });
  }

  /**
   * Broadcast member joined event
   *
   * @param daoId - DAO identifier
   * @param memberDid - New member's DID
   * @returns The broadcast event
   */
  async broadcastConstitutionAmended(
    proposalId: string,
    amendmentDescription: string
  ): Promise<GovernanceBridgeEvent> {
    return this.broadcastEvent({
      eventType: 'ConstitutionAmended',
      proposalId,
      subject: amendmentDescription,
      payload: JSON.stringify({ description: amendmentDescription }),
    });
  }

  // ============================================================================
  // Event Queries
  // ============================================================================

  /**
   * Get recent governance events
   *
   * @param limit - Maximum results (default: 50)
   * @returns Array of events
   */
  async getRecentEvents(limit: number = 50): Promise<GovernanceBridgeEvent[]> {
    const records = await this.callZome<HolochainRecord[]>('get_recent_events', limit);
    return records.map(r => this.mapEvent(r));
  }

  /**
   * Get events by type
   *
   * @param eventType - Event type to filter
   * @param limit - Maximum results
   * @returns Array of events
   */
  async getEventsByType(
    eventType: GovernanceBridgeEventType,
    limit?: number
  ): Promise<GovernanceBridgeEvent[]> {
    const records = await this.callZome<HolochainRecord[]>('get_events_by_type', {
      event_type: eventType,
      limit,
    });
    return records.map(r => this.mapEvent(r));
  }

  /**
   * Get events for a DAO
   *
   * @param daoId - DAO identifier
   * @param limit - Maximum results
   * @returns Array of events
   */
  async getEventsForDAO(daoId: ActionHash, limit?: number): Promise<GovernanceBridgeEvent[]> {
    const records = await this.callZome<HolochainRecord[]>('get_events_for_dao', {
      dao_id: daoId,
      limit,
    });
    return records.map(r => this.mapEvent(r));
  }

  /**
   * Get events for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of events
   */
  async getEventsForProposal(proposalId: ActionHash): Promise<GovernanceBridgeEvent[]> {
    const records = await this.callZome<HolochainRecord[]>(
      'get_events_for_proposal',
      proposalId
    );
    return records.map(r => this.mapEvent(r));
  }

  /**
   * Get events by subject DID
   *
   * @param subjectDid - Subject's DID
   * @param limit - Maximum results
   * @returns Array of events
   */
  async getEventsBySubject(subjectDid: string, limit?: number): Promise<GovernanceBridgeEvent[]> {
    const records = await this.callZome<HolochainRecord[]>('get_events_by_subject', {
      subject_did: subjectDid,
      limit,
    });
    return records.map(r => this.mapEvent(r));
  }

  /**
   * Get events from a source hApp
   *
   * @param sourceHapp - Source hApp identifier
   * @param limit - Maximum results
   * @returns Array of events
   */
  async getEventsFromHapp(sourceHapp: string, limit?: number): Promise<GovernanceBridgeEvent[]> {
    const records = await this.callZome<HolochainRecord[]>('get_events_from_happ', {
      source_happ: sourceHapp,
      limit,
    });
    return records.map(r => this.mapEvent(r));
  }

  // ============================================================================
  // Cross-hApp Queries
  // ============================================================================

  /**
   * Query governance information
   *
   * @param query - Query parameters
   * @returns Query results (varies by query type)
   */
  async queryGovernance(query: GovernanceQuery): Promise<unknown> {
    return this.callZome('query_governance', {
      query_type: query.queryType,
      query_params: query.queryParams,
      source_happ: query.sourceHapp,
    });
  }

  /**
   * Query proposal status from another hApp
   *
   * @param proposalHash - Proposal hash
   * @param sourceHapp - Source hApp
   * @returns Proposal status
   */
  async queryProposalStatus(
    proposalHash: string,
    sourceHapp: string
  ): Promise<CrossHappProposal | null> {
    const result = await this.queryGovernance({
      queryType: 'ProposalStatus',
      queryParams: JSON.stringify({ proposal_hash: proposalHash }),
      sourceHapp,
    });
    if (!result) return null;
    return this.mapCrossHappProposal(result);
  }

  /**
   * Query voting power for a DID
   *
   * @param daoId - DAO identifier
   * @param voterDid - Voter's DID
   * @param sourceHapp - Source hApp
   * @returns Voting power
   */
  async queryVotingPower(
    daoId: string,
    voterDid: string,
    sourceHapp: string
  ): Promise<number> {
    const result = await this.queryGovernance({
      queryType: 'VotingPower',
      queryParams: JSON.stringify({ dao_id: daoId, voter_did: voterDid }),
      sourceHapp,
    });
    return result as number;
  }

  /**
   * Query delegation chain
   *
   * @param daoId - DAO identifier
   * @param startDid - Starting DID
   * @param sourceHapp - Source hApp
   * @returns Delegation chain
   */
  async queryDelegationChain(
    daoId: string,
    startDid: string,
    sourceHapp: string
  ): Promise<string[]> {
    const result = await this.queryGovernance({
      queryType: 'DelegationChain',
      queryParams: JSON.stringify({ dao_id: daoId, start_did: startDid }),
      sourceHapp,
    });
    return result as string[];
  }

  // ============================================================================
  // Cross-hApp Proposals
  // ============================================================================

  /**
   * Register a proposal reference from another hApp
   *
   * @param proposal - Cross-hApp proposal data
   * @returns The registered reference
   */
  async registerCrossHappProposal(
    proposal: Omit<CrossHappProposal, 'id'>
  ): Promise<CrossHappProposal> {
    const record = await this.callZomeOnce<HolochainRecord>('register_cross_happ_proposal', {
      original_proposal_hash: proposal.originalProposalHash,
      source_happ: proposal.sourceHapp,
      title: proposal.title,
      proposal_type: proposal.proposalType,
      status: proposal.status,
      vote_weight_for: proposal.voteWeightFor,
      vote_weight_against: proposal.voteWeightAgainst,
      vote_weight_abstain: proposal.voteWeightAbstain,
      voting_ends_at: proposal.votingEndsAt,
      created_at: proposal.createdAt,
    });
    return this.mapCrossHappProposal(this.extractEntry(record));
  }

  /**
   * Get cross-hApp proposals
   *
   * @param sourceHapp - Optional filter by source hApp
   * @returns Array of cross-hApp proposals
   */
  async getCrossHappProposals(sourceHapp?: string): Promise<CrossHappProposal[]> {
    const records = await this.callZome<HolochainRecord[]>('get_cross_happ_proposals', {
      source_happ: sourceHapp,
    });
    return records.map(r => this.mapCrossHappProposal(this.extractEntry(r)));
  }

  /**
   * Update cross-hApp proposal status
   *
   * @param referenceId - Local reference ID
   * @param status - New status
   * @param voteWeights - Updated vote weights
   * @returns Updated proposal
   */
  async updateCrossHappProposal(
    referenceId: ActionHash,
    status: string,
    voteWeights?: { for: number; against: number; abstain: number }
  ): Promise<CrossHappProposal> {
    const record = await this.callZomeOnce<HolochainRecord>('update_cross_happ_proposal', {
      reference_id: referenceId,
      status,
      vote_weight_for: voteWeights?.for,
      vote_weight_against: voteWeights?.against,
      vote_weight_abstain: voteWeights?.abstain,
    });
    return this.mapCrossHappProposal(this.extractEntry(record));
  }

  // ============================================================================
  // Participation & Reputation
  // ============================================================================

  /**
   * Get participation score for a DID
   *
   * Returns governance participation metrics for cross-hApp reputation.
   *
   * @param did - DID to query
   * @returns Participation score
   */
  async getParticipationScore(did: string): Promise<ParticipationScore> {
    const result = await this.callZome<any>('get_participation_score', did);
    return {
      did: result.did,
      daoMemberships: result.dao_memberships,
      proposalsCreated: result.proposals_created,
      votesCast: result.votes_cast,
      participationRate: result.participation_rate,
      alignmentScore: result.alignment_score,
      delegationTrust: result.delegation_trust,
      overallScore: result.overall_score,
    };
  }

  /**
   * Report participation score to other hApps
   *
   * @param score - Participation score to report
   */
  async reportParticipationScore(score: ParticipationScore): Promise<void> {
    await this.callZomeOnce('report_participation_score', {
      did: score.did,
      dao_memberships: score.daoMemberships,
      proposals_created: score.proposalsCreated,
      votes_cast: score.votesCast,
      participation_rate: score.participationRate,
      alignment_score: score.alignmentScore,
      delegation_trust: score.delegationTrust,
      overall_score: score.overallScore,
    });
  }

  /**
   * Get aggregated reputation from multiple hApps
   *
   * @param did - DID to query
   * @returns Aggregated reputation scores by hApp
   */
  async getAggregatedReputation(
    did: string
  ): Promise<Array<{ happ: string; score: number }>> {
    return this.callZome('get_aggregated_reputation', did);
  }

  // ============================================================================
  // hApp Registration
  // ============================================================================

  /**
   * Register a hApp for cross-hApp governance
   *
   * @param happId - hApp identifier
   * @param happName - Human-readable name
   * @param capabilities - List of governance capabilities
   * @returns Registration status
   */
  async registerHapp(
    happId: string,
    happName: string,
    capabilities: string[]
  ): Promise<{ registered: boolean; happId: string }> {
    return this.callZomeOnce('register_happ', {
      happ_id: happId,
      happ_name: happName,
      capabilities,
    });
  }

  /**
   * Get registered hApps
   *
   * @returns Array of registered hApps
   */
  async getRegisteredHapps(): Promise<
    Array<{
      happId: string;
      happName: string;
      capabilities: string[];
      registeredAt: number;
    }>
  > {
    const results = await this.callZome<any[]>('get_registered_happs', null);
    return results.map(r => ({
      happId: r.happ_id,
      happName: r.happ_name,
      capabilities: r.capabilities,
      registeredAt: r.registered_at,
    }));
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Map Holochain record to GovernanceBridgeEvent type
   */
  private mapEvent(record: HolochainRecord): GovernanceBridgeEvent {
    const entry = this.extractEntry<any>(record);
    return {
      id: entry.id,
      eventType: entry.event_type,
      proposalId: entry.proposal_id,
      subject: entry.subject,
      payload: entry.payload,
      sourceHapp: entry.source_happ,
      timestamp: entry.timestamp,
    };
  }

  /**
   * Map to CrossHappProposal type
   */
  private mapCrossHappProposal(entry: any): CrossHappProposal {
    return {
      id: entry.id,
      originalProposalHash: entry.original_proposal_hash,
      sourceHapp: entry.source_happ,
      title: entry.title,
      proposalType: entry.proposal_type,
      status: entry.status,
      voteWeightFor: entry.vote_weight_for,
      voteWeightAgainst: entry.vote_weight_against,
      voteWeightAbstain: entry.vote_weight_abstain,
      votingEndsAt: entry.voting_ends_at,
      createdAt: entry.created_at,
    };
  }
}
