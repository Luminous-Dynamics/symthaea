/**
 * Threshold-Signing Zome Client
 *
 * Handles DKG ceremony management, signing committees, and threshold
 * signature operations for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/threshold-signing
 */

import type { AppClient, Record as HolochainRecord } from '@holochain/client';
import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

// ============================================================================
// Types
// ============================================================================

export type DkgPhase =
  | 'Registration'
  | 'Dealing'
  | 'Verification'
  | 'Complete';

export interface SigningCommittee {
  id: string;
  name: string;
  threshold: u32;
  minMembers: u32;
  maxMembers: u32;
  phase: DkgPhase;
  epoch: u32;
  publicKey?: Uint8Array;
  publicCommitments: Uint8Array[];
  qualifiedMembers: string[];
  createdAt: number;
  createdBy: string;
}

export interface CommitteeMember {
  committeeId: string;
  participantId: u32;
  agent: string;
  memberDid: string;
  trustScore: number;
  publicShare?: Uint8Array;
  vssCommitment?: Uint8Array;
  dealSubmitted: boolean;
  qualified: boolean;
  registeredAt: number;
}

export interface ThresholdSignature {
  committeeId: string;
  contentHash: Uint8Array;
  combinedSignature: Uint8Array;
  signerCount: u32;
  signerIds: u32[];
  verified: boolean;
  createdAt: number;
}

export interface SignatureShare {
  committeeId: string;
  contentHash: Uint8Array;
  participantId: u32;
  share: Uint8Array;
  createdAt: number;
}

type u32 = number;

export interface CreateCommitteeInput {
  name: string;
  threshold: number;
  minMembers: number;
  maxMembers: number;
}

export interface RegisterMemberInput {
  committeeId: string;
  memberDid: string;
  trustScore: number;
}

export interface SubmitDkgDealInput {
  committeeId: string;
  participantId: number;
  vssCommitment: Uint8Array;
  publicShare: Uint8Array;
}

export interface FinalizeDkgInput {
  committeeId: string;
  combinedPublicKey: Uint8Array;
  publicCommitments: Uint8Array[];
  qualifiedMembers: string[];
}

export interface SubmitSignatureShareInput {
  committeeId: string;
  contentHash: Uint8Array;
  participantId: number;
  share: Uint8Array;
}

export interface CombineSignaturesInput {
  committeeId: string;
  contentHash: Uint8Array;
  combinedSignature: Uint8Array;
  signerCount: number;
  signerIds: number[];
  verified: boolean;
}

export interface RotateKeysInput {
  committeeId: string;
  reason: string;
}

// ============================================================================
// Configuration
// ============================================================================

export interface ThresholdSigningClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: ThresholdSigningClientConfig = {
  roleName: 'governance',
};

// ============================================================================
// Client
// ============================================================================

/**
 * Client for Threshold-Signing operations
 *
 * Manages DKG ceremonies, signing committees, and threshold signature
 * creation/verification using Feldman DKG protocol.
 *
 * @example
 * ```typescript
 * const signing = new ThresholdSigningClient(appClient);
 *
 * // Create a signing committee (2-of-3)
 * const committee = await signing.createCommittee({
 *   name: 'Treasury Signers',
 *   threshold: 2,
 *   minMembers: 3,
 *   maxMembers: 5,
 * });
 *
 * // Register members
 * await signing.registerMember({
 *   committeeId: committee.id,
 *   memberDid: 'did:mycelix:uhCAk...',
 *   trustScore: 0.9,
 * });
 *
 * // After DKG ceremony completes, create threshold signatures
 * await signing.submitSignatureShare({
 *   committeeId: committee.id,
 *   contentHash: new Uint8Array(32),
 *   participantId: 1,
 *   share: signatureShareBytes,
 * });
 * ```
 */
export class ThresholdSigningClient extends ZomeClient {
  protected readonly zomeName = 'threshold_signing';

  constructor(client: AppClient, config: ThresholdSigningClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Committee Management
  // ============================================================================

  async createCommittee(input: CreateCommitteeInput): Promise<SigningCommittee> {
    const record = await this.callZomeOnce<HolochainRecord>('create_committee', {
      name: input.name,
      threshold: input.threshold,
      min_members: input.minMembers,
      max_members: input.maxMembers,
    });
    return this.mapCommittee(record);
  }

  async getCommittee(committeeId: string): Promise<SigningCommittee | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_committee', committeeId);
    if (!record) return null;
    return this.mapCommittee(record);
  }

  async getAllCommittees(): Promise<SigningCommittee[]> {
    const records = await this.callZome<HolochainRecord[]>('get_all_committees', null);
    return records.map((r) => this.mapCommittee(r));
  }

  async getCommitteeHistory(committeeId: string): Promise<SigningCommittee[]> {
    const records = await this.callZome<HolochainRecord[]>('get_committee_history', committeeId);
    return records.map((r) => this.mapCommittee(r));
  }

  // ============================================================================
  // DKG Ceremony
  // ============================================================================

  async registerMember(input: RegisterMemberInput): Promise<CommitteeMember> {
    const record = await this.callZomeOnce<HolochainRecord>('register_member', {
      committee_id: input.committeeId,
      member_did: input.memberDid,
      trust_score: input.trustScore,
    });
    return this.mapMember(record);
  }

  async submitDkgDeal(input: SubmitDkgDealInput): Promise<CommitteeMember> {
    const record = await this.callZomeOnce<HolochainRecord>('submit_dkg_deal', {
      committee_id: input.committeeId,
      participant_id: input.participantId,
      vss_commitment: Array.from(input.vssCommitment),
      public_share: Array.from(input.publicShare),
    });
    return this.mapMember(record);
  }

  async finalizeDkg(input: FinalizeDkgInput): Promise<SigningCommittee> {
    const record = await this.callZomeOnce<HolochainRecord>('finalize_dkg', {
      committee_id: input.committeeId,
      combined_public_key: Array.from(input.combinedPublicKey),
      public_commitments: input.publicCommitments.map((c) => Array.from(c)),
      qualified_members: input.qualifiedMembers,
    });
    return this.mapCommittee(record);
  }

  // ============================================================================
  // Signature Operations
  // ============================================================================

  async submitSignatureShare(input: SubmitSignatureShareInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('submit_signature_share', {
      committee_id: input.committeeId,
      content_hash: Array.from(input.contentHash),
      participant_id: input.participantId,
      share: Array.from(input.share),
    });
  }

  async combineSignatures(input: CombineSignaturesInput): Promise<ThresholdSignature> {
    const record = await this.callZomeOnce<HolochainRecord>('combine_signatures', {
      committee_id: input.committeeId,
      content_hash: Array.from(input.contentHash),
      combined_signature: Array.from(input.combinedSignature),
      signer_count: input.signerCount,
      signer_ids: input.signerIds,
      verified: input.verified,
    });
    return this.mapSignature(record);
  }

  async getSignatureShares(committeeId: string, contentHash: Uint8Array): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_signature_shares', {
      committee_id: committeeId,
      content_hash: Array.from(contentHash),
    });
  }

  async getProposalSignature(proposalId: string): Promise<ThresholdSignature | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_proposal_signature', proposalId);
    if (!record) return null;
    return this.mapSignature(record);
  }

  async getCommitteeMembers(committeeId: string): Promise<CommitteeMember[]> {
    const records = await this.callZome<HolochainRecord[]>('get_committee_members', committeeId);
    return records.map((r) => this.mapMember(r));
  }

  // ============================================================================
  // Key Rotation
  // ============================================================================

  async rotateCommitteeKeys(input: RotateKeysInput): Promise<SigningCommittee> {
    const record = await this.callZomeOnce<HolochainRecord>('rotate_committee_keys', {
      committee_id: input.committeeId,
      reason: input.reason,
    });
    return this.mapCommittee(record);
  }

  // ============================================================================
  // Mappers
  // ============================================================================

  private mapCommittee(record: HolochainRecord): SigningCommittee {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      name: entry.name,
      threshold: entry.threshold,
      minMembers: entry.min_members,
      maxMembers: entry.max_members,
      phase: entry.phase,
      epoch: entry.epoch,
      publicKey: entry.public_key ? new Uint8Array(entry.public_key) : undefined,
      publicCommitments: (entry.public_commitments ?? []).map((c: number[]) => new Uint8Array(c)),
      qualifiedMembers: entry.qualified_members ?? [],
      createdAt: entry.created_at,
      createdBy: entry.created_by,
    };
  }

  private mapMember(record: HolochainRecord): CommitteeMember {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      committeeId: entry.committee_id,
      participantId: entry.participant_id,
      agent: entry.agent,
      memberDid: entry.member_did,
      trustScore: entry.trust_score,
      publicShare: entry.public_share ? new Uint8Array(entry.public_share) : undefined,
      vssCommitment: entry.vss_commitment ? new Uint8Array(entry.vss_commitment) : undefined,
      dealSubmitted: entry.deal_submitted,
      qualified: entry.qualified,
      registeredAt: entry.registered_at,
    };
  }

  private mapSignature(record: HolochainRecord): ThresholdSignature {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      committeeId: entry.committee_id,
      contentHash: new Uint8Array(entry.content_hash ?? []),
      combinedSignature: new Uint8Array(entry.combined_signature ?? []),
      signerCount: entry.signer_count,
      signerIds: entry.signer_ids ?? [],
      verified: entry.verified,
      createdAt: entry.created_at,
    };
  }
}
