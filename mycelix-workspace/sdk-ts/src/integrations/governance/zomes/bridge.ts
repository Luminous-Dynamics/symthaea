// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Governance Bridge Zome Client
 *
 * Handles consciousness metrics, Φ attestation, weighted consensus,
 * reputation management, and cross-hApp governance integration.
 *
 * This client wraps the governance bridge coordinator which provides:
 * - Consciousness snapshots and gate verification (Φ-gated actions)
 * - Signed Φ attestations from Symthaea
 * - Value alignment assessment (Eight Harmonies)
 * - Consciousness-aware weighted voting
 * - K-Vector trust, MATL scores, federated reputation
 * - Cross-hApp query, execution, and event broadcasting
 * - Dynamic Φ threshold configuration
 *
 * @module @mycelix/sdk/integrations/governance/zomes/bridge
 */

import { GovernanceSdkError } from '../types';

import type { Record as HolochainRecord } from '@holochain/client';
import type {
  // Consciousness
  RecordSnapshotInput,
  VerifyGateInput,
  GateVerificationResult,
  GateVerificationResultV2,
  PhiThresholds,
  GetAgentSnapshotsInput,
  // Attestation
  RecordPhiAttestationInput,
  AttestationHistoryEntry,
  GetAttestationHistoryInput,
  PruneStaleAttestationsInput,
  // Value alignment
  AssessValueAlignmentInput,
  // Consensus
  RegisterConsensusParticipantInput,
  CastWeightedVoteInput,
  CalculateHolisticWeightInput,
  HolisticVotingWeight,
  RoundResult,
  RoundVotesQuery,
  ConsensusParticipantStatus,
  // Reputation
  KVector,
  MatlTrustScore,
  UpdateFederatedReputationInput,
  // Cross-hApp
  QueryGovernanceInput,
  RequestExecutionInput,
  AcknowledgeExecutionInput,
  BroadcastGovernanceEventInput,
  // Cross-cluster dispatch
  DispatchPersonalCallInput,
  DispatchIdentityCallInput,
  DispatchCommonsCallInput,
  DispatchCivicCallInput,
  CheckVoterTrustInput,
  // Phi config
  GovernancePhiConfig,
  UpdatePhiConfigInput,
  AdaptiveThreshold,
  ProposalType,
} from '../types';

/**
 * Configuration for the Bridge client
 */
export interface BridgeClientConfig {
  /** Role ID for the governance DNA */
  roleId: string;
  /** Bridge zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: BridgeClientConfig = {
  roleId: 'governance',
  zomeName: 'governance_bridge',
};

interface ZomeCallable {
  callZome<T>(params: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<T>;
}

/**
 * Client for the Governance Bridge Coordinator
 *
 * Provides consciousness-aware governance operations, including Φ-gated
 * actions, signed attestations, weighted consensus, and cross-hApp
 * integration.
 *
 * @example
 * ```typescript
 * const bridge = new BridgeClient(holochainClient);
 *
 * // Record consciousness snapshot from Symthaea
 * await bridge.recordConsciousnessSnapshot({
 *   phi: 0.65,
 *   meta_awareness: 0.7,
 *   self_model_accuracy: 0.8,
 *   coherence: 0.75,
 *   affective_valence: 0.6,
 *   care_activation: 0.5,
 * });
 *
 * // Verify Φ gate before voting
 * const gate = await bridge.verifyConsciousnessGate({
 *   action_type: 'Voting',
 * });
 * if (gate.passed) {
 *   await bridge.castWeightedVote({ ... });
 * }
 * ```
 */
export class BridgeClient {
  private readonly config: BridgeClientConfig;

  constructor(
    private readonly client: ZomeCallable,
    config: Partial<BridgeClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a bridge zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome<T>({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });
      return result;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);

      if (msg.includes('Consciousness gate failed')) {
        throw new GovernanceSdkError('UNAUTHORIZED', `Φ gate check failed: ${msg}`, error);
      }
      if (msg.includes('Rate limited')) {
        throw new GovernanceSdkError('ZOME_ERROR', `Rate limited: ${msg}`, error);
      }

      throw new GovernanceSdkError(
        'ZOME_ERROR',
        `Failed to call bridge.${fnName}: ${msg}`,
        error
      );
    }
  }

  // ============================================================================
  // Consciousness Metrics
  // ============================================================================

  /**
   * Record a consciousness snapshot from Symthaea
   *
   * Captures 6 consciousness metrics: Φ, meta-awareness, self-model accuracy,
   * coherence, affective valence, and care activation.
   *
   * @param input - Snapshot data
   * @returns Created record
   */
  async recordConsciousnessSnapshot(input: RecordSnapshotInput): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('record_consciousness_snapshot', input);
  }

  /**
   * Verify whether the caller's Φ meets the threshold for an action
   *
   * @param input - Action type and optional action ID
   * @returns Gate verification result (passed, phi, required_phi)
   */
  async verifyConsciousnessGate(input: VerifyGateInput): Promise<GateVerificationResult> {
    return this.call<GateVerificationResult>('verify_consciousness_gate', input);
  }

  /**
   * Enhanced gate verification with provenance tracking
   *
   * Returns whether the Φ came from a signed attestation, a snapshot,
   * or was unavailable, plus the recent Φ trend.
   *
   * @param input - Action type and optional action ID
   * @returns Enhanced gate result with provenance and trend
   */
  async verifyConsciousnessGateV2(input: VerifyGateInput): Promise<GateVerificationResultV2> {
    return this.call<GateVerificationResultV2>('verify_consciousness_gate_v2', input);
  }

  /**
   * Assess a proposal's alignment with the Eight Harmonies
   *
   * Returns per-harmony scores, overall alignment, authenticity,
   * violations, and a governance recommendation.
   *
   * @param input - Proposal ID and content
   * @returns Value alignment assessment record
   */
  async assessValueAlignment(input: AssessValueAlignmentInput): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('assess_value_alignment', input);
  }

  /**
   * Get consciousness trajectory for an agent
   *
   * @param agentDid - Agent's DID
   * @returns History record or null if no snapshots
   */
  async getAgentConsciousnessHistory(agentDid: string): Promise<HolochainRecord | null> {
    return this.call<HolochainRecord | null>('get_agent_consciousness_history', agentDid);
  }

  /**
   * Get recent consciousness snapshots for an agent
   *
   * @param input - Agent DID and optional limit (default 50)
   * @returns Array of snapshot records
   */
  async getAgentSnapshots(input: GetAgentSnapshotsInput): Promise<HolochainRecord[]> {
    return this.call<HolochainRecord[]>('get_agent_snapshots', input);
  }

  /**
   * Get all value alignment assessments for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of assessment records
   */
  async getProposalAlignments(proposalId: string): Promise<HolochainRecord[]> {
    return this.call<HolochainRecord[]>('get_proposal_alignments', proposalId);
  }

  /**
   * Get current Φ threshold requirements
   *
   * @returns Thresholds for Basic, ProposalSubmission, Voting, Constitutional
   */
  async getPhiThresholds(): Promise<PhiThresholds> {
    return this.call<PhiThresholds>('get_phi_thresholds', null);
  }

  // ============================================================================
  // Phi Attestation
  // ============================================================================

  /**
   * Record an Ed25519-signed Φ attestation from Symthaea
   *
   * Rate-limited to ~10s minimum interval per agent.
   *
   * @param input - Phi value, cycle ID, timestamp, and Ed25519 signature
   * @returns Created attestation record
   */
  async recordPhiAttestation(input: RecordPhiAttestationInput): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('record_phi_attestation', input);
  }

  /**
   * Get an agent's Φ attestation history (most recent first)
   *
   * @param input - Optional limit (default 20, max 100)
   * @returns Array of attestation entries
   */
  async getAgentAttestationHistory(
    input: GetAttestationHistoryInput = {}
  ): Promise<AttestationHistoryEntry[]> {
    return this.call<AttestationHistoryEntry[]>('get_agent_attestation_history', input);
  }

  /**
   * Prune old attestations, keeping the N most recent
   *
   * @param input - Number of attestations to keep
   * @returns Number of attestations deleted
   */
  async pruneStaleAttestations(input: PruneStaleAttestationsInput): Promise<number> {
    return this.call<number>('prune_stale_attestations', input);
  }

  // ============================================================================
  // Consciousness-Aware Consensus
  // ============================================================================

  /**
   * Register as a consensus participant (validator)
   *
   * @param input - Agent DID, K-Vector ID, reputation, MATL, Phi, federated score
   * @returns Created participant record
   */
  async registerConsensusParticipant(
    input: RegisterConsensusParticipantInput
  ): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('register_consensus_participant', input);
  }

  /**
   * Cast a consciousness-aware weighted vote
   *
   * Vote weight = reputation² × consciousness_multiplier × harmonic_bonus
   *
   * @param input - Proposal ID, round, decision, optional harmonic alignment
   * @returns Created vote record
   */
  async castWeightedVote(input: CastWeightedVoteInput): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('cast_weighted_vote', input);
  }

  /**
   * Calculate holistic vote weight from components
   *
   * @param input - Reputation, Phi, harmonic alignment values
   * @returns Detailed weight breakdown
   */
  async calculateHolisticVoteWeight(
    input: CalculateHolisticWeightInput
  ): Promise<HolisticVotingWeight> {
    return this.call<HolisticVotingWeight>('calculate_holistic_vote_weight', input);
  }

  /**
   * Tally votes for a consensus round
   *
   * @param input - Proposal ID and round number
   * @returns Round result (totals, approved/rejected)
   */
  async calculateRoundResult(input: RoundVotesQuery): Promise<RoundResult> {
    return this.call<RoundResult>('calculate_round_result', input);
  }

  /**
   * Get all votes in a consensus round
   *
   * @param input - Proposal ID and round number
   * @returns Array of vote records
   */
  async getRoundVotes(input: RoundVotesQuery): Promise<HolochainRecord[]> {
    return this.call<HolochainRecord[]>('get_round_votes', input);
  }

  /**
   * Get a validator's participation status
   *
   * @param agentDid - Agent's DID
   * @returns Participant status (reputation, activity, slashing)
   */
  async getParticipantStatus(agentDid: string): Promise<ConsensusParticipantStatus> {
    return this.call<ConsensusParticipantStatus>('get_participant_status', agentDid);
  }

  /**
   * Check a voter's federated reputation score
   *
   * @param agentDid - Agent's DID
   * @returns Aggregated trust score
   */
  async checkVoterTrust(agentDid: string): Promise<number> {
    return this.call<number>('check_voter_trust', agentDid);
  }

  // ============================================================================
  // Reputation & Trust
  // ============================================================================

  /**
   * Calculate/retrieve MATL trust score
   *
   * MATL = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
   *
   * @param agentDid - Agent's DID
   * @returns MATL score with component breakdown
   */
  async getVoterMatlScore(agentDid: string): Promise<MatlTrustScore> {
    return this.call<MatlTrustScore>('get_voter_matl_score', agentDid);
  }

  /**
   * Update multi-domain federated reputation
   *
   * Aggregates signals from: identity, knowledge, finance, FL, governance.
   *
   * @param input - Per-domain reputation signals
   * @returns Updated reputation record
   */
  async updateFederatedReputation(
    input: UpdateFederatedReputationInput
  ): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('update_federated_reputation', input);
  }

  /**
   * Request K-Vector (8D trust model) for an agent
   *
   * Dimensions: k_r (weight 0.25), k_a (0.15), k_i (0.20), k_p (0.15),
   * k_m (0.05), k_s (0.10), k_h (0.05), k_topo (0.05)
   *
   * @param agentDid - Agent's DID
   * @returns K-Vector record
   */
  async requestKVector(agentDid: string): Promise<KVector> {
    return this.call<KVector>('request_k_vector', agentDid);
  }

  /**
   * Request Φ credential from identity system
   *
   * @param agentDid - Agent's DID
   * @returns Phi credential record
   */
  async requestPhiCredential(agentDid: string): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('request_phi_credential', agentDid);
  }

  // ============================================================================
  // Cross-hApp Integration
  // ============================================================================

  /**
   * Query governance from another hApp
   *
   * Supports: ActiveProposals, ProposalById, VotingStatus,
   * VotingEligibility, ConstitutionalRules
   *
   * @param input - Source hApp, query type, and parameters
   * @returns Query result (structure depends on query type)
   */
  async queryGovernance(input: QueryGovernanceInput): Promise<unknown> {
    return this.call<unknown>('query_governance', input);
  }

  /**
   * Request cross-hApp execution (Φ ≥ 0.3 required)
   *
   * @param input - Proposal ID, target hApp, action, and parameters
   * @returns Created execution request record
   */
  async requestExecution(input: RequestExecutionInput): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('request_execution', input);
  }

  /**
   * Get pending execution requests for a target hApp
   *
   * @param targetHapp - Target hApp identifier
   * @returns Array of pending execution records
   */
  async getPendingExecutions(targetHapp: string): Promise<HolochainRecord[]> {
    return this.call<HolochainRecord[]>('get_pending_executions', targetHapp);
  }

  /**
   * Acknowledge execution completion or failure
   *
   * @param input - Execution ID, status, and optional result
   * @returns Success
   */
  async acknowledgeExecution(input: AcknowledgeExecutionInput): Promise<boolean> {
    return this.call<boolean>('acknowledge_execution', input);
  }

  /**
   * Broadcast a governance event for cross-hApp notification
   *
   * @param input - Event type, optional proposal ID, subject, payload
   * @returns Created event record
   */
  async broadcastGovernanceEvent(input: BroadcastGovernanceEventInput): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('broadcast_governance_event', input);
  }

  /**
   * Get the last 50 governance events
   *
   * @returns Array of event records
   */
  async getRecentEvents(): Promise<HolochainRecord[]> {
    return this.call<HolochainRecord[]>('get_recent_events', null);
  }

  /**
   * Verify a voter's DID with the identity system
   *
   * @param agentDid - Agent's DID
   * @returns Whether the DID is verified
   */
  async verifyVoterDid(agentDid: string): Promise<boolean> {
    return this.call<boolean>('verify_voter_did', agentDid);
  }

  /**
   * Request DID proof for voting from the identity system
   *
   * @param agentDid - Agent's DID
   * @returns Identity proof record
   */
  async requestIdentityProof(agentDid: string): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('request_identity_proof', agentDid);
  }

  /**
   * Publish a proposal reference for cross-hApp discovery
   *
   * @param proposalId - Proposal identifier
   * @returns Created reference record
   */
  async publishProposalReference(proposalId: string): Promise<HolochainRecord> {
    return this.call<HolochainRecord>('publish_proposal_reference', proposalId);
  }

  /**
   * Verify a governance-related credential
   *
   * @param credentialId - Credential identifier
   * @param issuerDid - Issuer's DID
   * @returns Whether the credential is valid
   */
  async verifyGovernanceCredential(
    credentialId: string,
    issuerDid: string
  ): Promise<boolean> {
    return this.call<boolean>('verify_governance_credential', {
      credential_id: credentialId,
      issuer_did: issuerDid,
    });
  }

  // ============================================================================
  // Phi Configuration
  // ============================================================================

  /**
   * Get current Φ threshold configuration
   *
   * @returns Dynamic phi config
   */
  async getPhiConfig(): Promise<GovernancePhiConfig> {
    return this.call<GovernancePhiConfig>('get_phi_config', null);
  }

  /**
   * Initialize Φ config with defaults (one-time bootstrap)
   *
   * @returns Initialized phi config
   */
  async bootstrapPhiConfig(): Promise<GovernancePhiConfig> {
    return this.call<GovernancePhiConfig>('bootstrap_phi_config', null);
  }

  /**
   * Update Φ threshold configuration dynamically
   *
   * @param input - Fields to update (only non-null fields are changed)
   * @returns Updated phi config
   */
  async updatePhiConfig(input: UpdatePhiConfigInput): Promise<GovernancePhiConfig> {
    return this.call<GovernancePhiConfig>('update_phi_config', input);
  }

  /**
   * Get adaptive Φ threshold for a specific proposal type
   *
   * @param proposalType - Standard, Emergency, or Constitutional
   * @returns Adaptive threshold with adjustment reason
   */
  async getAdaptiveThreshold(proposalType: ProposalType): Promise<AdaptiveThreshold> {
    return this.call<AdaptiveThreshold>('get_adaptive_threshold', proposalType);
  }

  // ============================================================================
  // Cross-Cluster Dispatch
  // ============================================================================

  /**
   * Dispatch a call to the personal cluster via OtherRole
   *
   * Allowed zomes: personal_bridge
   *
   * @param input - Zome name, function name, and payload
   * @returns Raw response from the personal cluster
   */
  async dispatchPersonalCall(input: DispatchPersonalCallInput): Promise<unknown> {
    return this.call<unknown>('dispatch_personal_call', input);
  }

  /**
   * Request Φ credential from the agent's personal vault (cross-cluster)
   *
   * Unlike requestPhiCredential (identity system), this calls through
   * OtherRole dispatch to the personal cluster's vault.
   *
   * @returns Phi credential from personal vault
   */
  async requestPhiCredentialFromVault(): Promise<unknown> {
    return this.call<unknown>('request_phi_credential', null);
  }

  /**
   * Request K-Vector trust credential from the agent's personal vault
   *
   * @returns K-Vector credential
   */
  async requestKVectorFromVault(): Promise<unknown> {
    return this.call<unknown>('request_k_vector', null);
  }

  /**
   * Request identity proof from the agent's personal vault
   *
   * @returns Identity proof
   */
  async requestIdentityProofFromVault(): Promise<unknown> {
    return this.call<unknown>('request_identity_proof', null);
  }

  /**
   * Dispatch a call to the identity cluster via OtherRole
   *
   * Allowed zomes: identity_bridge, did_registry, verifiable_credential
   *
   * @param input - Zome name, function name, and payload
   * @returns Raw response from the identity cluster
   */
  async dispatchIdentityCall(input: DispatchIdentityCallInput): Promise<unknown> {
    return this.call<unknown>('dispatch_identity_call', input);
  }

  /**
   * Verify a voter's DID is active in the identity cluster
   *
   * @param did - The DID to verify
   * @returns Whether the DID is active
   */
  async verifyVoterDidActive(did: string): Promise<unknown> {
    return this.call<unknown>('verify_voter_did', did);
  }

  /**
   * Get a voter's MATL trust score from the identity bridge
   *
   * @param did - Voter's DID
   * @returns Composite trust score (0.0-1.0)
   */
  async getVoterMatlScoreFromIdentity(did: string): Promise<unknown> {
    return this.call<unknown>('get_voter_matl_score', did);
  }

  /**
   * Verify a credential via the identity cluster
   *
   * @param credentialId - Credential to verify
   * @returns Verification result
   */
  async verifyCredentialViaIdentity(credentialId: string): Promise<unknown> {
    return this.call<unknown>('verify_governance_credential', credentialId);
  }

  /**
   * Check enhanced trust for a voter (reputation + MFA)
   *
   * Used for high-stakes governance actions (treasury, key rotation).
   *
   * @param input - DID, min reputation, MFA requirement, assurance level
   * @returns Trust check result
   */
  async checkVoterEnhancedTrust(input: CheckVoterTrustInput): Promise<unknown> {
    return this.call<unknown>('check_voter_trust', input);
  }

  /**
   * Dispatch a call to the commons cluster via OtherRole
   *
   * Allowed zomes: commons_bridge, property_registry, housing_governance, water_steward
   *
   * @param input - Zome name, function name, and payload
   * @returns Raw response from the commons cluster
   */
  async dispatchCommonsCall(input: DispatchCommonsCallInput): Promise<unknown> {
    return this.call<unknown>('dispatch_commons_call', input);
  }

  /**
   * Publish a governance proposal reference to the commons bridge
   *
   * Makes the proposal discoverable by commons domain zomes.
   *
   * @param proposalId - Proposal identifier
   * @returns Result from commons bridge
   */
  async publishProposalToCommons(proposalId: string): Promise<unknown> {
    return this.call<unknown>('publish_proposal_to_commons', proposalId);
  }

  /**
   * Dispatch a call to the civic cluster via OtherRole
   *
   * Allowed zomes: civic_bridge, justice_enforcement, emergency_coordination, media_factcheck
   *
   * @param input - Zome name, function name, and payload
   * @returns Raw response from the civic cluster
   */
  async dispatchCivicCall(input: DispatchCivicCallInput): Promise<unknown> {
    return this.call<unknown>('dispatch_civic_call', input);
  }

  /**
   * Check for active emergencies in the civic cluster
   *
   * Governance can use this to trigger fast-track emergency proposals.
   *
   * @param area - Geographic or topical area
   * @returns Active emergencies
   */
  async checkEmergencyStatus(area: string): Promise<unknown> {
    return this.call<unknown>('check_emergency_status', area);
  }

  /**
   * Request a fact-check for a governance proposal via civic media
   *
   * @param proposalId - Proposal to fact-check
   * @returns Fact-check assessment
   */
  async requestProposalFactcheck(proposalId: string): Promise<unknown> {
    return this.call<unknown>('request_proposal_factcheck', proposalId);
  }
}
