// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Bridge Zome Client
 *
 * Handles cross-hApp governance coordination, event broadcasting,
 * and inter-hApp queries for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/bridge
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  GovernanceBridgeEvent,
  GovernanceBridgeEventType,
  GovernanceQuery,
  CrossHappProposal,
  BroadcastEventInput,
  BridgeProposalType,
  ParticipationScore,
  RecordSnapshotInput,
  RecordConsciousnessAttestationInput,
  VerifyGateInput,
  GateVerificationResult,
  GateVerificationResultV2,
  AssessAlignmentInput,
  GetAgentSnapshotsInput,
  ConsciousnessThresholds,
  ConsciousnessProvenance,
  CalculateWeightInput,
  HolisticVotingWeight,
  CastWeightedVoteInput,
  WeightedVoteResult,
  AdaptiveThreshold,
  ParticipantStatus,
  UpdateFederatedReputationInput,
  GetRoundVotesInput,
  CalculateRoundResultInput,
  RoundResult,
  DispatchPersonalCallInput,
} from './types';
// GovernanceError removed (unused)
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

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
  // Consciousness Integration
  // ============================================================================

  /**
   * Record a consciousness snapshot for an agent
   *
   * Establishes consciousness state before governance actions.
   *
   * @param input - Snapshot parameters
   * @returns The snapshot record
   */
  async recordConsciousnessSnapshot(input: RecordSnapshotInput): Promise<HolochainRecord> {
    const level = input.phi ?? input.consciousnessLevel;
    return this.callZomeOnce<HolochainRecord>('record_consciousness_snapshot', {
      phi: level,
      meta_awareness: input.metaAwareness,
      self_model_accuracy: input.selfModelAccuracy,
      coherence: input.coherence,
      affective_valence: input.affectiveValence,
      care_activation: input.careActivation,
      source: input.source ?? null,
    });
  }

  /**
   * Verify consciousness gate for a governance action (legacy v1)
   *
   * @param input - Gate verification parameters
   * @returns Gate verification result
   */
  async verifyConsciousnessGate(input: VerifyGateInput): Promise<GateVerificationResult> {
    const result = await this.callZome<any>('verify_consciousness_gate', {
      action_type: input.actionType,
      action_id: input.actionId ?? null,
    });
    const level = result.consciousness_level ?? result.phi;
    const required = result.required_consciousness ?? result.required_phi;
    return {
      passed: result.passed,
      consciousnessLevel: level,
      requiredConsciousness: required,
      phi: level,
      requiredPhi: required,
      actionType: result.action_type,
      failureReason: result.failure_reason ?? undefined,
      gateId: result.gate_id,
    };
  }

  /**
   * Record an authenticated consciousness attestation (preferred over consciousness snapshot)
   *
   * Creates a signed attestation linking an agent's Symthaea consciousness level
   * to their governance identity. Preferred over `recordConsciousnessSnapshot`
   * because attestations include cryptographic signatures.
   *
   * @param input - Attestation parameters (consciousnessLevel, cycleId, signature)
   * @returns The attestation record
   */
  async recordConsciousnessAttestation(input: RecordConsciousnessAttestationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_consciousness_attestation', {
      consciousness_level: input.consciousnessLevel,
      cycle_id: input.cycleId,
      captured_at_us: input.capturedAtUs,
      signature: input.signature,
    });
  }

  /** @deprecated Use recordConsciousnessAttestation */
  async recordPhiAttestation(input: { phi: number; cycleId: number; capturedAtUs: number; signature: Uint8Array | number[] }): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_phi_attestation', {
      phi: input.phi,
      cycle_id: input.cycleId,
      captured_at_us: input.capturedAtUs,
      signature: input.signature,
    });
  }

  /**
   * Verify consciousness gate with provenance tracking (v2)
   *
   * Returns the consciousness level along with how it was obtained (Attested, Snapshot,
   * or Unavailable). Preferred over `verifyConsciousnessGate` because it
   * honestly reports when no real consciousness data is available.
   *
   * @param input - Gate verification parameters
   * @returns Gate verification result with provenance
   */
  async verifyConsciousnessGateV2(input: VerifyGateInput): Promise<GateVerificationResultV2> {
    const result = await this.callZome<any>('verify_consciousness_gate_v2', {
      action_type: input.actionType,
      action_id: input.actionId ?? null,
    });
    return {
      passed: result.passed,
      consciousnessLevel: result.consciousness_level ?? null,
      requiredConsciousness: result.required_consciousness,
      provenance: result.provenance as ConsciousnessProvenance,
      actionType: result.action_type,
      failureReason: result.failure_reason ?? undefined,
    };
  }

  /**
   * Assess value alignment between an agent and a proposal
   *
   * @param input - Alignment assessment parameters
   * @returns The alignment record
   */
  async assessValueAlignment(input: AssessAlignmentInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('assess_value_alignment', {
      proposal_id: input.proposalId,
      proposal_content: input.proposalContent,
    });
  }

  /**
   * Get agent's consciousness history
   *
   * @param agentDid - Agent's DID
   * @returns The consciousness history record or null
   */
  async getAgentConsciousnessHistory(agentDid: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_agent_consciousness_history', agentDid);
  }

  /**
   * Get recent consciousness snapshots for an agent
   *
   * @param input - Query parameters
   * @returns Array of snapshot records
   */
  async getAgentSnapshots(input: GetAgentSnapshotsInput): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_agent_snapshots', {
      agent_did: input.agentDid,
      limit: input.limit,
    });
  }

  /**
   * Get value alignments for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of alignment records
   */
  async getProposalAlignments(proposalId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_proposal_alignments', proposalId);
  }

  /**
   * Get current consciousness threshold requirements
   *
   * @returns Consciousness thresholds for each proposal tier
   */
  async getConsciousnessThresholds(): Promise<ConsciousnessThresholds> {
    const result = await this.callZome<any>('get_consciousness_thresholds', null);
    return {
      basic: result.basic,
      proposalSubmission: result.proposal_submission,
      voting: result.voting,
      constitutional: result.constitutional,
    };
  }

  /** @deprecated Use getConsciousnessThresholds */
  getPhiThresholds = this.getConsciousnessThresholds.bind(this);

  // ============================================================================
  // Weighted Consensus Voting
  // ============================================================================

  /**
   * Register as a consensus participant
   *
   * @returns The participant registration record
   */
  async registerConsensusParticipant(): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('register_consensus_participant', null);
  }

  /**
   * Calculate holistic vote weight for an agent
   *
   * Uses: Reputation² x (0.7 + 0.3 x Φ) x (1 + 0.2 x HarmonicAlignment)
   *
   * @param input - Weight calculation parameters
   * @returns Holistic voting weight breakdown
   */
  async calculateHolisticVoteWeight(input: CalculateWeightInput): Promise<HolisticVotingWeight> {
    const result = await this.callZome<any>('calculate_holistic_vote_weight', {
      harmonic_alignment: input.harmonicAlignment ?? null,
    });
    const level = result.consciousness_level ?? result.phi;
    return {
      reputation: result.reputation,
      reputationSquared: result.reputation_squared,
      consciousnessLevel: level,
      phi: level,
      consciousnessMultiplier: result.consciousness_multiplier,
      harmonicAlignment: result.harmonic_alignment,
      harmonicBonus: result.harmonic_bonus,
      finalWeight: result.final_weight,
      wasCapped: result.was_capped,
      uncappedWeight: result.uncapped_weight,
      calculationBreakdown: result.calculation_breakdown,
    };
  }

  /**
   * Cast a weighted consensus vote with consciousness gate verification
   *
   * @param input - Weighted vote parameters
   * @returns Weighted vote result
   */
  async castWeightedVote(input: CastWeightedVoteInput): Promise<WeightedVoteResult> {
    const result = await this.callZome<any>('cast_weighted_vote', {
      proposal_id: input.proposalId,
      proposal_type: input.proposalType,
      round: input.round,
      decision: input.decision,
      harmonic_alignment: input.harmonicAlignment ?? null,
      reason: input.reason ?? null,
    });
    return {
      voteId: result.vote_id,
      weight: result.weight,
      weightBreakdown: result.weight_breakdown,
      decision: result.decision,
      consciousnessAtVote: result.phi_at_vote,
      phiAtVote: result.phi_at_vote,
      proposalType: result.proposal_type,
      thresholdRequired: result.threshold_required,
    };
  }

  /**
   * Get adaptive threshold for a proposal type
   *
   * @param proposalType - Proposal type
   * @returns Adaptive threshold
   */
  async getAdaptiveThreshold(proposalType: BridgeProposalType): Promise<AdaptiveThreshold> {
    const result = await this.callZome<any>('get_adaptive_threshold', proposalType);
    const minPhi = result.min_voter_consciousness ?? result.min_voter_phi;
    return {
      baseThreshold: result.base_threshold,
      minVoterConsciousness: minPhi,
      minVoterPhi: minPhi,
      minParticipation: result.min_participation,
      quorum: result.quorum,
      maxExtensionSecs: result.max_extension_secs,
    };
  }

  /**
   * Get participant's current status including streak and cooldown
   *
   * @returns Participant status
   */
  async getParticipantStatus(): Promise<ParticipantStatus> {
    const result = await this.callZome<any>('get_participant_status', null);
    return {
      agentDid: result.agent_did,
      isActive: result.is_active,
      baseReputation: result.base_reputation,
      effectiveReputation: result.effective_reputation,
      streakCount: result.streak_count,
      streakBonus: result.streak_bonus,
      inCooldown: result.in_cooldown,
      currentConsciousness: result.current_phi,
      federatedScore: result.federated_score,
      roundsParticipated: result.rounds_participated,
      successfulVotes: result.successful_votes,
      successRate: result.success_rate,
      slashingEvents: result.slashing_events,
      canVoteStandard: result.can_vote_standard,
      canVoteEmergency: result.can_vote_emergency,
      canVoteConstitutional: result.can_vote_constitutional,
    };
  }

  /**
   * Update federated reputation from another hApp
   *
   * @param input - Reputation update parameters
   * @returns The updated reputation record
   */
  async updateFederatedReputation(input: UpdateFederatedReputationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('update_federated_reputation', {
      identity_verification: input.identityVerification ?? null,
      credential_count: input.credentialCount ?? null,
      credential_quality: input.credentialQuality ?? null,
      epistemic_contributions: input.epistemicContributions ?? null,
      factcheck_accuracy: input.factcheckAccuracy ?? null,
      dark_spots_resolved: input.darkSpotsResolved ?? null,
      stake_weight: input.stakeWeight ?? null,
      payment_reliability: input.paymentReliability ?? null,
      escrow_completion_rate: input.escrowCompletionRate ?? null,
      pogq_score: input.pogqScore ?? null,
      fl_contributions: input.flContributions ?? null,
      byzantine_clean_rate: input.byzantineCleanRate ?? null,
      voting_participation: input.votingParticipation ?? null,
      proposal_success_rate: input.proposalSuccessRate ?? null,
      consensus_alignment: input.consensusAlignment ?? null,
    });
  }

  /**
   * Get votes for a consensus round
   *
   * @param input - Round query parameters
   * @returns Array of vote records
   */
  async getRoundVotes(input: GetRoundVotesInput): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_round_votes', {
      proposal_id: input.proposalId,
      round: input.round,
    });
  }

  /**
   * Calculate consensus round result
   *
   * @param input - Round calculation parameters
   * @returns Round result
   */
  async calculateRoundResult(input: CalculateRoundResultInput): Promise<RoundResult> {
    const result = await this.callZome<any>('calculate_round_result', {
      proposal_id: input.proposalId,
      round: input.round,
      proposal_type: input.proposalType,
      eligible_voters: input.eligibleVoters,
    });
    return {
      proposalId: result.proposal_id,
      round: result.round,
      proposalType: result.proposal_type,
      totalWeight: result.total_weight,
      weightedApprovals: result.weighted_approvals,
      weightedRejections: result.weighted_rejections,
      voteCount: result.vote_count,
      requiredThreshold: result.required_threshold,
      approvalPercentage: result.approval_percentage,
      quorumMet: result.quorum_met,
      consensusReached: result.consensus_reached,
      rejected: result.rejected,
      result: result.result,
    };
  }

  // ============================================================================
  // Personal Cluster Bridge
  // ============================================================================

  /**
   * Dispatch a call to the personal cluster via cross-cluster bridge
   *
   * @param input - Personal call parameters
   * @returns Raw response from personal cluster
   */
  async dispatchPersonalCall(input: DispatchPersonalCallInput): Promise<unknown> {
    return this.callZome('dispatch_personal_call', {
      zome_name: input.zomeName,
      fn_name: input.fnName,
      payload: input.payload,
    });
  }

  /**
   * Request a consciousness credential from the personal cluster
   *
   * @returns Consciousness credential presentation
   */
  async requestConsciousnessCredential(): Promise<unknown> {
    return this.callZome('request_phi_credential', null);
  }

  /** @deprecated Use requestConsciousnessCredential */
  requestPhiCredential = this.requestConsciousnessCredential.bind(this);

  /**
   * Request K-vector trust data from the personal cluster
   *
   * @returns K-vector trust presentation
   */
  async requestKVector(): Promise<unknown> {
    return this.callZome('request_k_vector', null);
  }

  /**
   * Request ZK identity proof from the personal cluster
   *
   * @returns Identity proof without full profile disclosure
   */
  async requestIdentityProof(): Promise<unknown> {
    return this.callZome('request_identity_proof', null);
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
