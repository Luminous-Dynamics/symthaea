// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Voting Zome Client
 *
 * Handles vote casting, MATL-weighted power calculation, and vote queries
 * for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/voting
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Vote,
  CastVoteInput,
  VotingPowerQuery,
  VotingPowerBreakdown,
  QuorumStatus,
  VotingStats,
  VoteChoice,
  CastConsciousnessVoteInput,
  CastDelegatedConsciousnessVoteInput,
  CastQuadraticVoteInput,
  AllocateCreditsInput,
  VoiceCredits,
  CreateDelegationWithDecayInput,
  RenewDelegationInput,
  RevokeDelegationInput,
  EffectiveDelegation,
  TallyVotesInput,
  TallyConsciousnessVotesInput,
  TallyQuadraticVotesInput,
  StoreEligibilityProofInput,
  CastVerifiedVoteInput,
  TallyVerifiedVotesInput,
  VerifiedVoteTallyResult,
  StoreAttestationInput,
  CastAttestedVoteInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';
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
  // Φ-Weighted Voting
  // ============================================================================

  /**
   * Cast a Φ-weighted vote on a proposal
   *
   * Voter must meet the Φ threshold for the proposal's tier:
   * - Basic: Φ ≥ 0.3
   * - Major: Φ ≥ 0.4
   * - Constitutional: Φ ≥ 0.6
   *
   * @param input - Consciousness vote parameters
   * @returns The cast consciousness-weighted vote record
   */
  async castConsciousnessWeightedVote(input: CastConsciousnessVoteInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('cast_phi_weighted_vote', {
      proposal_id: input.proposalId,
      voter_did: input.voterDid,
      tier: input.tier,
      choice: input.choice,
      reason: input.reason,
    });
  }

  /**
   * Cast a delegated consciousness-weighted vote
   *
   * Resolves the delegation chain and applies transitive weight with decay.
   *
   * @param input - Delegated consciousness vote parameters
   * @returns The cast delegated vote record
   */
  async castDelegatedConsciousnessVote(input: CastDelegatedConsciousnessVoteInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('cast_delegated_phi_vote', {
      proposal_id: input.proposalId,
      delegate_did: input.delegateDid,
      tier: input.tier,
      choice: input.choice,
      reason: input.reason,
    });
  }

  // --- Phi backward-compat aliases ---

  /** @deprecated Use castConsciousnessWeightedVote */
  castPhiWeightedVote = this.castConsciousnessWeightedVote.bind(this);

  /** @deprecated Use castDelegatedConsciousnessVote */
  castDelegatedPhiVote = this.castDelegatedConsciousnessVote.bind(this);

  // ============================================================================
  // Quadratic Voting
  // ============================================================================

  /**
   * Cast a quadratic vote (weight = √credits)
   *
   * @param input - Quadratic vote parameters
   * @returns The cast quadratic vote record
   */
  async castQuadraticVote(input: CastQuadraticVoteInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('cast_quadratic_vote', {
      proposal_id: input.proposalId,
      voter_did: input.voterDid,
      choice: input.choice,
      credits_to_spend: input.creditsToSpend,
      reason: input.reason,
    });
  }

  /**
   * Allocate voice credits to a voter for quadratic voting
   *
   * @param input - Credit allocation parameters
   * @returns The allocated credits record
   */
  async allocateVoiceCredits(input: AllocateCreditsInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('allocate_voice_credits', {
      owner_did: input.ownerDid,
      amount: input.amount,
      period_end: input.periodEnd,
    });
  }

  /**
   * Query a voter's current voice credit balance
   *
   * @param voterDid - Voter's DID
   * @returns Voice credit balance
   */
  async queryVoiceCredits(voterDid: string): Promise<VoiceCredits> {
    const result = await this.callZome<any>('query_voice_credits', voterDid);
    return {
      owner: result.owner,
      allocated: result.allocated,
      spent: result.spent,
      remaining: result.remaining,
      periodStart: result.period_start,
      periodEnd: result.period_end,
    };
  }

  // ============================================================================
  // Delegation with Decay
  // ============================================================================

  /**
   * Create a delegation with configurable decay model
   *
   * @param input - Delegation parameters
   * @returns The created delegation record
   */
  async createDelegation(input: CreateDelegationWithDecayInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_delegation', {
      delegator_did: input.delegatorDid,
      delegate_did: input.delegateDid,
      percentage: input.percentage,
      topics: input.topics,
      tier_filter: input.tierFilter,
      decay: input.decay,
      transitive: input.transitive,
      max_chain_depth: input.maxChainDepth,
      expires: input.expires,
    });
  }

  /**
   * Renew a delegation (resets decay timer)
   *
   * @param input - Renewal parameters
   * @returns The renewed delegation record
   */
  async renewDelegation(input: RenewDelegationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('renew_delegation', {
      original_action_hash: input.originalActionHash,
      new_percentage: input.newPercentage,
    });
  }

  /**
   * Revoke a delegation
   *
   * @param input - Revocation parameters
   * @returns The revoked delegation record
   */
  async revokeDelegation(input: RevokeDelegationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('revoke_delegation', {
      original_action_hash: input.originalActionHash,
    });
  }

  /**
   * Get delegations with current effective percentages (accounting for decay)
   *
   * @param voterDid - Voter's DID
   * @returns Array of effective delegations
   */
  async getEffectiveDelegations(voterDid: string): Promise<EffectiveDelegation[]> {
    const results = await this.callZome<any[]>('get_effective_delegations', voterDid);
    return results.map(r => ({
      delegation: {
        id: r.delegation.id,
        delegator: r.delegation.delegator,
        delegate: r.delegation.delegate,
        percentage: r.delegation.percentage,
        topics: r.delegation.topics,
        tierFilter: r.delegation.tier_filter,
        active: r.delegation.active,
        decay: r.delegation.decay,
        transitive: r.delegation.transitive,
        maxChainDepth: r.delegation.max_chain_depth,
        created: r.delegation.created,
        renewed: r.delegation.renewed,
        expires: r.delegation.expires,
      },
      effectivePercentage: r.effective_percentage,
      isEffectivelyExpired: r.is_effectively_expired,
    }));
  }

  // ============================================================================
  // Tallying
  // ============================================================================

  /**
   * Get raw vote records for a proposal (legacy)
   *
   * @param proposalId - Proposal identifier
   * @returns Array of vote records
   */
  async getProposalVotes(proposalId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_proposal_votes', proposalId);
  }

  /**
   * Tally votes for a proposal with configurable thresholds
   *
   * @param input - Tally parameters
   * @returns The tally record
   */
  async tallyVotes(input: TallyVotesInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('tally_votes', {
      proposal_id: input.proposalId,
      tier: input.tier,
      quorum_override: input.quorumOverride,
      approval_override: input.approvalOverride,
    });
  }

  /**
   * Tally consciousness-weighted votes for a proposal
   *
   * Automatically advances proposal to Approved if threshold met.
   * Optionally generates a collective mirror reflection.
   *
   * @param input - Consciousness tally parameters
   * @returns The consciousness-weighted tally record
   */
  async tallyConsciousnessVotes(input: TallyConsciousnessVotesInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('tally_phi_votes', {
      proposal_id: input.proposalId,
      tier: input.tier,
      eligible_voters: input.eligibleVoters,
      generate_reflection: input.generateReflection,
    });
  }

  /**
   * Tally quadratic votes for a proposal
   *
   * @param input - Quadratic tally parameters
   * @returns The quadratic tally record
   */
  async tallyQuadraticVotes(input: TallyQuadraticVotesInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('tally_quadratic_votes', {
      proposal_id: input.proposalId,
      min_voters: input.minVoters,
    });
  }

  /**
   * Get the legacy tally for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns The tally record or null
   */
  async getProposalTally(proposalId: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_proposal_tally', proposalId);
  }

  /**
   * Get the consciousness-weighted tally for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns The consciousness-weighted tally record or null
   */
  async getConsciousnessTally(proposalId: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_phi_tally', proposalId);
  }

  /**
   * Get the quadratic tally for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns The quadratic tally record or null
   */
  async getQuadraticTally(proposalId: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_quadratic_tally', proposalId);
  }

  // ============================================================================
  // ZK-STARK Verified Voting
  // ============================================================================

  /**
   * Store an eligibility proof on-chain
   *
   * @param input - Proof storage parameters
   * @returns The stored proof record
   */
  async storeEligibilityProof(input: StoreEligibilityProofInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('store_eligibility_proof', {
      voter_did: input.voterDid,
      voter_commitment: input.voterCommitment,
      proposal_type: input.proposalType,
      eligible: input.eligible,
      requirements_met: input.requirementsMet,
      active_requirements: input.activeRequirements,
      proof_bytes: input.proofBytes,
      validity_hours: input.validityHours,
    });
  }

  /**
   * Cast a vote with ZK eligibility verification
   *
   * Voter must have a valid, unexpired eligibility proof.
   *
   * @param input - Verified vote parameters
   * @returns The cast verified vote record
   */
  async castVerifiedVote(input: CastVerifiedVoteInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('cast_verified_vote', {
      proposal_id: input.proposalId,
      voter_did: input.voterDid,
      tier: input.tier,
      choice: input.choice,
      eligibility_proof_hash: input.eligibilityProofHash,
      voter_commitment: input.voterCommitment,
      reason: input.reason,
    });
  }

  /**
   * Get all verified votes for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of verified vote records
   */
  async getVerifiedVotes(proposalId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_verified_votes', proposalId);
  }

  /**
   * Get voter's eligibility proofs
   *
   * @param voterDid - Voter's DID
   * @returns Array of proof records
   */
  async getVoterProofs(voterDid: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_voter_proofs', voterDid);
  }

  /**
   * Tally verified votes for a proposal
   *
   * @param input - Tally parameters
   * @returns The verified vote tally
   */
  async tallyVerifiedVotes(input: TallyVerifiedVotesInput): Promise<VerifiedVoteTallyResult> {
    const result = await this.callZome<any>('tally_verified_votes', {
      proposal_id: input.proposalId,
      tier: input.tier,
      eligible_voters: input.eligibleVoters,
    });
    return {
      proposalId: result.proposal_id,
      tier: result.tier,
      votesFor: result.votes_for,
      votesAgainst: result.votes_against,
      abstentions: result.abstentions,
      totalWeight: result.total_weight,
      voterCount: result.voter_count,
      quorumRequirement: result.quorum_requirement,
      quorumReached: result.quorum_reached,
      approvalThreshold: result.approval_threshold,
      approvalRate: result.approval_rate,
      approved: result.approved,
      talliedAt: result.tallied_at,
    };
  }

  // ============================================================================
  // Proof Attestation
  // ============================================================================

  /**
   * Store a proof attestation from an external verifier
   *
   * @param input - Attestation parameters
   * @returns The stored attestation record
   */
  async storeProofAttestation(input: StoreAttestationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('store_proof_attestation', {
      proof_action_hash: input.proofActionHash,
      proof_hash: input.proofHash,
      voter_commitment: input.voterCommitment,
      proposal_type: input.proposalType,
      verified: input.verified,
      verifier_pubkey: input.verifierPubkey,
      signature: input.signature,
      security_level: input.securityLevel,
      verification_time_ms: input.verificationTimeMs,
      validity_hours: input.validityHours,
    });
  }

  /**
   * Get attestations for a proof
   *
   * @param proofActionHash - Action hash of the proof
   * @returns Array of attestation records
   */
  async getProofAttestations(proofActionHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_proof_attestations', proofActionHash);
  }

  /**
   * Check if a proof has a valid (verified and not expired) attestation
   *
   * @param proofActionHash - Action hash of the proof
   * @returns True if valid attestation exists
   */
  async hasValidAttestation(proofActionHash: ActionHash): Promise<boolean> {
    return this.callZome<boolean>('has_valid_attestation', proofActionHash);
  }

  /**
   * Get attestations by verifier
   *
   * @param verifierPubkey - Verifier's public key bytes
   * @returns Array of attestation records
   */
  async getVerifierAttestations(verifierPubkey: number[]): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_verifier_attestations', verifierPubkey);
  }

  /**
   * Cast a verified vote with attestation check
   *
   * Enhanced version that also requires a valid attestation from an external verifier.
   *
   * @param input - Attested vote parameters
   * @returns The cast attested vote record
   */
  async castAttestedVote(input: CastAttestedVoteInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('cast_attested_vote', {
      proposal_id: input.proposalId,
      voter_did: input.voterDid,
      tier: input.tier,
      choice: input.choice,
      eligibility_proof_hash: input.eligibilityProofHash,
      voter_commitment: input.voterCommitment,
      reason: input.reason,
    });
  }

  // ============================================================================
  // Collective Mirror Reflections
  // ============================================================================

  /**
   * Generate a collective mirror reflection for a proposal's voting phase
   *
   * Analyzes topology, shadow (absent harmonies), signal integrity,
   * and trajectory of current voting patterns.
   *
   * @param proposalId - Proposal identifier
   * @returns The reflection record
   */
  async reflectOnProposal(proposalId: string): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('reflect_on_proposal', {
      proposal_id: proposalId,
    });
  }

  /**
   * Get all collective mirror reflections for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of reflection records (sorted by timestamp)
   */
  async getProposalReflections(proposalId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_proposal_reflections', proposalId);
  }

  /**
   * Get the latest reflection for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns The latest reflection record or null
   */
  async getLatestReflection(proposalId: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_latest_reflection', proposalId);
  }

  /**
   * Check if a proposal needs human review based on collective sensing
   *
   * Returns true if echo chamber risk is high, rapid convergence detected,
   * fragmentation warning, low harmony coverage, or high centralization.
   *
   * @param proposalId - Proposal identifier
   * @returns True if review is recommended
   */
  async proposalNeedsReview(proposalId: string): Promise<boolean> {
    return this.callZome<boolean>('proposal_needs_review', proposalId);
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
