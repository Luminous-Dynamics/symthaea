/**
 * Identity Client
 *
 * Member registration, verification, and profile management.
 */

import { type ActionHash, type AgentPubKey, type Record } from '@holochain/client';

import type { DAOClient } from './client';
import type {
  MemberProfile,
  MemberStatus,
  EmpiricalLevel,
} from './types';

/**
 * Verification request type
 */
export interface VerificationRequest {
  id: string;
  agent: AgentPubKey;
  requested_level: EmpiricalLevel;
  evidence_cid?: string;
  status: 'Pending' | 'Approved' | 'Rejected';
  reviewer?: AgentPubKey;
  created_at: number;
  reviewed_at?: number;
}

/**
 * Identity attestation from another member
 */
export interface Attestation {
  subject: AgentPubKey;
  attester: AgentPubKey;
  claim: string;
  confidence: number; // 0.0-1.0
  created_at: number;
  revoked_at?: number;
}

/**
 * Client for identity operations
 */
export class IdentityClient {
  constructor(private client: DAOClient) {}

  private get zome() {
    return this.client.getZomeName('identity');
  }

  // =========================================================================
  // REGISTRATION
  // =========================================================================

  /**
   * Register as a new member
   *
   * @example
   * ```typescript
   * const profile = await identity.register({
   *   display_name: 'alice_dao',
   *   bio: 'Passionate about decentralized governance',
   * });
   * ```
   */
  async register(input: {
    display_name: string;
    bio?: string;
  }): Promise<MemberProfile> {
    const record = await this.client.callZome<Record>(
      this.zome,
      'register_member',
      input
    );

    return this.decodeProfile(record);
  }

  /**
   * Check if the current user is registered
   */
  async isRegistered(): Promise<boolean> {
    const profile = await this.getMyProfile();
    return profile !== null;
  }

  // =========================================================================
  // PROFILES
  // =========================================================================

  /**
   * Get the current user's profile
   */
  async getMyProfile(): Promise<MemberProfile | null> {
    const myPubKey = await this.client.getMyPubKey();
    return this.getProfile(myPubKey);
  }

  /**
   * Get a member's profile by public key
   */
  async getProfile(agent: AgentPubKey): Promise<MemberProfile | null> {
    const record = await this.client.callZome<Record | null>(
      this.zome,
      'get_member_profile',
      agent
    );

    if (!record) return null;
    return this.decodeProfile(record);
  }

  /**
   * Update the current user's profile
   */
  async updateProfile(updates: {
    display_name?: string;
    bio?: string;
  }): Promise<MemberProfile> {
    const record = await this.client.callZome<Record>(
      this.zome,
      'update_member_profile',
      updates
    );

    return this.decodeProfile(record);
  }

  /**
   * Get all registered members
   */
  async getAllMembers(): Promise<MemberProfile[]> {
    const records = await this.client.callZome<Record[]>(
      this.zome,
      'get_all_members',
      null
    );

    return records.map(r => this.decodeProfile(r));
  }

  /**
   * Get members by status
   */
  async getMembersByStatus(status: MemberStatus): Promise<MemberProfile[]> {
    const records = await this.client.callZome<Record[]>(
      this.zome,
      'get_members_by_status',
      status
    );

    return records.map(r => this.decodeProfile(r));
  }

  /**
   * Search members by display name
   */
  async searchMembers(query: string): Promise<MemberProfile[]> {
    const records = await this.client.callZome<Record[]>(
      this.zome,
      'search_members',
      query
    );

    return records.map(r => this.decodeProfile(r));
  }

  // =========================================================================
  // VERIFICATION
  // =========================================================================

  /**
   * Request verification level upgrade
   *
   * @example
   * ```typescript
   * // Request E2 verification with evidence
   * await identity.requestVerification('E2', 'bafybeig...');
   * ```
   */
  async requestVerification(
    level: EmpiricalLevel,
    evidenceCid?: string
  ): Promise<VerificationRequest> {
    const record = await this.client.callZome<Record>(
      this.zome,
      'request_verification',
      { requested_level: level, evidence_cid: evidenceCid }
    );

    return (record as any).entry.Present.entry as VerificationRequest;
  }

  /**
   * Get pending verification requests (for reviewers)
   */
  async getPendingVerifications(): Promise<VerificationRequest[]> {
    const records = await this.client.callZome<Record[]>(
      this.zome,
      'get_pending_verifications',
      null
    );

    return records.map(r => (r as any).entry.Present.entry as VerificationRequest);
  }

  /**
   * Review a verification request (for authorized reviewers)
   */
  async reviewVerification(
    requestId: string,
    approved: boolean,
    reason?: string
  ): Promise<VerificationRequest> {
    const record = await this.client.callZome<Record>(
      this.zome,
      'review_verification',
      { request_id: requestId, approved, reason }
    );

    return (record as any).entry.Present.entry as VerificationRequest;
  }

  /**
   * Get current user's verification level
   */
  async getMyVerificationLevel(): Promise<EmpiricalLevel> {
    const profile = await this.getMyProfile();
    return profile?.verification_level || 'E0';
  }

  // =========================================================================
  // SYBIL RESISTANCE
  // =========================================================================

  /**
   * Get sybil resistance score for an agent
   *
   * Higher score = more trusted (0.0-1.0)
   */
  async getSybilScore(agent?: AgentPubKey): Promise<number> {
    const pubKey = agent || await this.client.getMyPubKey();
    return this.client.callZome<number>(
      this.zome,
      'get_sybil_score',
      pubKey
    );
  }

  /**
   * Report suspicious activity for sybil analysis
   */
  async reportSuspiciousActivity(
    target: AgentPubKey,
    reason: string,
    evidence?: string
  ): Promise<void> {
    await this.client.callZome(
      this.zome,
      'report_suspicious_activity',
      { target, reason, evidence }
    );
  }

  // =========================================================================
  // ATTESTATIONS
  // =========================================================================

  /**
   * Create an attestation for another member
   *
   * @example
   * ```typescript
   * // Attest that alice is a domain expert
   * await identity.attest(alicePubKey, 'Domain expert in renewable energy', 0.9);
   * ```
   */
  async attest(
    subject: AgentPubKey,
    claim: string,
    confidence: number
  ): Promise<Attestation> {
    if (confidence < 0 || confidence > 1) {
      throw new Error('Confidence must be between 0 and 1');
    }

    const record = await this.client.callZome<Record>(
      this.zome,
      'create_attestation',
      { subject, claim, confidence }
    );

    return (record as any).entry.Present.entry as Attestation;
  }

  /**
   * Get attestations about a member
   */
  async getAttestationsFor(agent: AgentPubKey): Promise<Attestation[]> {
    const records = await this.client.callZome<Record[]>(
      this.zome,
      'get_attestations_for',
      agent
    );

    return records.map(r => (r as any).entry.Present.entry as Attestation);
  }

  /**
   * Get attestations made by a member
   */
  async getAttestationsBy(agent: AgentPubKey): Promise<Attestation[]> {
    const records = await this.client.callZome<Record[]>(
      this.zome,
      'get_attestations_by',
      agent
    );

    return records.map(r => (r as any).entry.Present.entry as Attestation);
  }

  /**
   * Revoke an attestation
   */
  async revokeAttestation(attestationHash: ActionHash): Promise<void> {
    await this.client.callZome(
      this.zome,
      'revoke_attestation',
      attestationHash
    );
  }

  // =========================================================================
  // REPUTATION
  // =========================================================================

  /**
   * Get aggregate reputation for a member
   */
  async getReputation(agent?: AgentPubKey): Promise<{
    aggregate: number;
    proposals_authored: number;
    votes_cast: number;
    delegations_received: number;
    consistency_score: number;
  }> {
    const pubKey = agent || await this.client.getMyPubKey();
    return this.client.callZome(
      this.zome,
      'get_member_reputation',
      pubKey
    );
  }

  /**
   * Get reputation leaderboard
   */
  async getReputationLeaderboard(limit: number = 10): Promise<{
    agent: AgentPubKey;
    display_name: string;
    reputation: number;
  }[]> {
    return this.client.callZome(
      this.zome,
      'get_reputation_leaderboard',
      limit
    );
  }

  // =========================================================================
  // ADMIN (for oversight bodies)
  // =========================================================================

  /**
   * Suspend a member (requires oversight authorization)
   */
  async suspendMember(
    agent: AgentPubKey,
    reason: string,
    duration_hours?: number
  ): Promise<void> {
    await this.client.callZome(
      this.zome,
      'suspend_member',
      { agent, reason, duration_hours }
    );
  }

  /**
   * Reinstate a suspended member
   */
  async reinstateMember(agent: AgentPubKey): Promise<void> {
    await this.client.callZome(
      this.zome,
      'reinstate_member',
      agent
    );
  }

  private decodeProfile(record: Record): MemberProfile {
    return (record as any).entry.Present.entry as MemberProfile;
  }
}
