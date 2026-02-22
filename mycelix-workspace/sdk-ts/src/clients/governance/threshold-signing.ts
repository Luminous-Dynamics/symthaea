/**
 * Threshold-Signing Zome Client
 *
 * Handles DKG ceremony management, signing committees, and threshold
 * signature operations for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/threshold-signing
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type { AppClient, Record as HolochainRecord } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

export type DkgPhase =
  | 'Registration'
  | 'Dealing'
  | 'Verification'
  | 'Complete'
  | 'Disbanded';

export type CommitteeScope =
  | 'All'
  | 'Constitutional'
  | 'Treasury'
  | 'Protocol'
  | { Custom: string[] };

export interface SigningCommittee {
  id: string;
  name: string;
  threshold: u32;
  memberCount: u32;
  phase: DkgPhase;
  publicKey?: Uint8Array;
  commitments: Uint8Array[];
  scope: CommitteeScope;
  createdAt: number;
  active: boolean;
  epoch: u32;
  minPhi?: number;
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
  id: string;
  committeeId: string;
  signedContentHash: Uint8Array;
  signedContentDescription: string;
  signature: Uint8Array;
  signerCount: u32;
  signers: u32[];
  verified: boolean;
  signedAt: number;
}

export interface SignatureShare {
  signatureId: string;
  participantId: u32;
  signer: string;
  share: Uint8Array;
  contentHash: Uint8Array;
  submittedAt: number;
}

type u32 = number;

export interface CreateCommitteeInput {
  name: string;
  threshold: number;
  memberCount: number;
  scope: CommitteeScope;
  minPhi?: number;
}

export interface RegisterMemberInput {
  committeeId: string;
  participantId: number;
  memberDid: string;
  trustScore: number;
}

export interface SubmitDkgDealInput {
  committeeId: string;
  vssCommitment: Uint8Array;
}

export interface FinalizeDkgInput {
  committeeId: string;
  combinedPublicKey: Uint8Array;
  publicCommitments: Uint8Array[];
  qualifiedMembers: number[];
}

export interface SubmitSignatureShareInput {
  committeeId: string;
  signatureId: string;
  participantId: number;
  share: Uint8Array;
  contentHash: Uint8Array;
}

export interface CombineSignaturesInput {
  committeeId: string;
  contentHash: Uint8Array;
  contentDescription: string;
  combinedSignature: Uint8Array;
  signers: number[];
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
      member_count: input.memberCount,
      scope: input.scope,
      min_phi: input.minPhi ?? null,
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
      participant_id: input.participantId,
      member_did: input.memberDid,
      trust_score: input.trustScore,
    });
    return this.mapMember(record);
  }

  async submitDkgDeal(input: SubmitDkgDealInput): Promise<CommitteeMember> {
    const record = await this.callZomeOnce<HolochainRecord>('submit_dkg_deal', {
      committee_id: input.committeeId,
      vss_commitment: Array.from(input.vssCommitment),
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
      signature_id: input.signatureId,
      participant_id: input.participantId,
      share: Array.from(input.share),
      content_hash: Array.from(input.contentHash),
    });
  }

  async combineSignatures(input: CombineSignaturesInput): Promise<ThresholdSignature> {
    const record = await this.callZomeOnce<HolochainRecord>('combine_signatures', {
      committee_id: input.committeeId,
      content_hash: Array.from(input.contentHash),
      content_description: input.contentDescription,
      combined_signature: Array.from(input.combinedSignature),
      signers: input.signers,
    });
    return this.mapSignature(record);
  }

  async getSignatureShares(signatureId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_signature_shares', signatureId);
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
    const record = await this.callZomeOnce<HolochainRecord>('rotate_committee_keys', input.committeeId);
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
      memberCount: entry.member_count,
      phase: entry.phase,
      publicKey: entry.public_key ? new Uint8Array(entry.public_key) : undefined,
      commitments: (entry.commitments ?? []).map((c: number[]) => new Uint8Array(c)),
      scope: entry.scope,
      createdAt: entry.created_at,
      active: entry.active,
      epoch: entry.epoch,
      minPhi: entry.min_phi ?? undefined,
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
      id: entry.id,
      committeeId: entry.committee_id,
      signedContentHash: new Uint8Array(entry.signed_content_hash ?? []),
      signedContentDescription: entry.signed_content_description,
      signature: new Uint8Array(entry.signature ?? []),
      signerCount: entry.signer_count,
      signers: entry.signers ?? [],
      verified: entry.verified,
      signedAt: entry.signed_at,
    };
  }
}
