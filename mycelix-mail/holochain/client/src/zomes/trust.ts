// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Zome Client (MATL Algorithm)
 * Mycelix Advanced Trust Logic with transitive trust and Byzantine detection
 */

import type { AppClient, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  TrustAttestation,
  TrustScore,
  TrustCategory,
  TrustLevel,
  TrustEvidence,
  CreateAttestationInput,
  GetTrustScoreInput,
  ByzantineFlag,
} from '../types';

export class TrustZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'mycelix_mail',
    private zomeName: string = 'mail_trust'
  ) {}

  private async callZome<T>(fnName: string, payload?: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null,
    });
    return result as T;
  }

  // ==================== ATTESTATIONS ====================

  /**
   * Create a trust attestation for another agent
   */
  async createAttestation(input: CreateAttestationInput): Promise<ActionHash> {
    return this.callZome('create_attestation', {
      subject: input.subject,
      category: input.category,
      level: input.level,
      confidence: input.confidence,
      evidence: input.evidence ?? [],
      context: input.context ?? null,
      stake_amount: input.stake_amount ?? 0,
      expires_in_days: input.expires_in_days ?? null,
    });
  }

  /**
   * Revoke an attestation
   */
  async revokeAttestation(attestationHash: ActionHash, reason: string): Promise<void> {
    await this.callZome('revoke_attestation', {
      attestation_hash: attestationHash,
      reason,
    });
  }

  /**
   * Update attestation level
   */
  async updateAttestation(
    attestationHash: ActionHash,
    newLevel: TrustLevel,
    newConfidence?: number
  ): Promise<ActionHash> {
    return this.callZome('update_attestation', {
      attestation_hash: attestationHash,
      new_level: newLevel,
      new_confidence: newConfidence ?? null,
    });
  }

  /**
   * Add evidence to existing attestation
   */
  async addEvidence(attestationHash: ActionHash, evidence: TrustEvidence): Promise<void> {
    await this.callZome('add_evidence', {
      attestation_hash: attestationHash,
      evidence,
    });
  }

  /**
   * Get attestations I've made
   */
  async getMyAttestations(): Promise<Array<{ hash: ActionHash; attestation: TrustAttestation }>> {
    return this.callZome('get_my_attestations', null);
  }

  /**
   * Get attestations about a subject
   */
  async getAttestationsAbout(
    subject: AgentPubKey,
    category?: TrustCategory
  ): Promise<Array<{ hash: ActionHash; attestation: TrustAttestation }>> {
    return this.callZome('get_attestations_about', {
      subject,
      category: category ?? null,
    });
  }

  /**
   * Get attestations by a specific attestor
   */
  async getAttestationsBy(
    attestor: AgentPubKey
  ): Promise<Array<{ hash: ActionHash; attestation: TrustAttestation }>> {
    return this.callZome('get_attestations_by', attestor);
  }

  // ==================== TRUST SCORING (MATL) ====================

  /**
   * Get trust score for an agent
   * Uses MATL algorithm with transitive trust calculation
   */
  async getTrustScore(input: GetTrustScoreInput): Promise<TrustScore> {
    return this.callZome('get_trust_score', {
      subject: input.subject,
      category: input.category ?? null,
      include_transitive: input.include_transitive ?? true,
      max_depth: input.max_depth ?? 5,
    });
  }

  /**
   * Get trust scores for multiple agents
   */
  async getBatchTrustScores(
    subjects: AgentPubKey[],
    category?: TrustCategory
  ): Promise<Map<string, TrustScore>> {
    const result = await this.callZome<Array<[string, TrustScore]>>('get_batch_trust_scores', {
      subjects,
      category: category ?? null,
    });
    return new Map(result);
  }

  /**
   * Check if agent meets trust threshold
   */
  async meetsTrustThreshold(
    subject: AgentPubKey,
    threshold: number,
    category?: TrustCategory
  ): Promise<boolean> {
    return this.callZome('meets_trust_threshold', {
      subject,
      threshold,
      category: category ?? null,
    });
  }

  /**
   * Get trust path between two agents
   */
  async getTrustPath(
    from: AgentPubKey,
    to: AgentPubKey,
    category?: TrustCategory
  ): Promise<{
    path: AgentPubKey[];
    total_trust: number;
    hops: number;
  } | null> {
    return this.callZome('get_trust_path', {
      from,
      to,
      category: category ?? null,
    });
  }

  /**
   * Get web of trust visualization data
   */
  async getWebOfTrust(
    center: AgentPubKey,
    depth?: number
  ): Promise<{
    nodes: Array<{ agent: AgentPubKey; score: number }>;
    edges: Array<{ from: AgentPubKey; to: AgentPubKey; weight: number }>;
  }> {
    return this.callZome('get_web_of_trust', {
      center,
      depth: depth ?? 3,
    });
  }

  // ==================== BYZANTINE DETECTION ====================

  /**
   * Get Byzantine flags for an agent
   */
  async getByzantineFlags(subject: AgentPubKey): Promise<ByzantineFlag[]> {
    return this.callZome('get_byzantine_flags', subject);
  }

  /**
   * Report suspicious behavior
   */
  async reportSuspiciousBehavior(
    subject: AgentPubKey,
    description: string,
    evidence?: string
  ): Promise<ActionHash> {
    return this.callZome('report_suspicious_behavior', {
      subject,
      description,
      evidence: evidence ?? null,
    });
  }

  /**
   * Check for Sybil patterns
   */
  async checkSybilPatterns(subject: AgentPubKey): Promise<{
    is_suspicious: boolean;
    indicators: string[];
    confidence: number;
  }> {
    return this.callZome('check_sybil_patterns', subject);
  }

  // ==================== INTRODUCTIONS ====================

  /**
   * Create trust introduction (vouch for connection)
   */
  async createIntroduction(
    introducer: AgentPubKey,
    introducee: AgentPubKey,
    message?: string
  ): Promise<ActionHash> {
    return this.callZome('create_introduction', {
      introducer,
      introducee,
      message: message ?? null,
    });
  }

  /**
   * Accept introduction
   */
  async acceptIntroduction(introductionHash: ActionHash): Promise<void> {
    await this.callZome('accept_introduction', introductionHash);
  }

  /**
   * Get pending introductions
   */
  async getPendingIntroductions(): Promise<Array<{
    hash: ActionHash;
    introducer: AgentPubKey;
    introducee: AgentPubKey;
    message: string | null;
    created_at: number;
  }>> {
    return this.callZome('get_pending_introductions', null);
  }

  // ==================== DISPUTES ====================

  /**
   * File a trust dispute
   */
  async fileDispute(
    attestationHash: ActionHash,
    reason: string,
    evidence?: string
  ): Promise<ActionHash> {
    return this.callZome('file_dispute', {
      attestation_hash: attestationHash,
      reason,
      evidence: evidence ?? null,
    });
  }

  /**
   * Get disputes for an attestation
   */
  async getDisputes(attestationHash: ActionHash): Promise<Array<{
    hash: ActionHash;
    disputant: AgentPubKey;
    reason: string;
    status: 'Open' | 'Resolved' | 'Dismissed';
  }>> {
    return this.callZome('get_disputes', attestationHash);
  }

  // ==================== STAKING ====================

  /**
   * Stake on attestation (put reputation on the line)
   */
  async stakeOnAttestation(attestationHash: ActionHash, amount: number): Promise<void> {
    await this.callZome('stake_on_attestation', {
      attestation_hash: attestationHash,
      amount,
    });
  }

  /**
   * Get my total stake
   */
  async getMyStake(): Promise<{ total: number; at_risk: number; available: number }> {
    return this.callZome('get_my_stake', null);
  }
}
