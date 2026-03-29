// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Credential Client
 *
 * Client for K-Vector trust credentials with ZKP proofs in Mycelix-Identity.
 * Enables trust credential issuance, verification, and selective disclosure.
 *
 * @module @mycelix/sdk/clients/identity/trust
 */

import { DEFAULT_IDENTITY_CLIENT_CONFIG } from './types.js';
import { ZomeClient } from '../../core/zome-client.js';

import type {
  TrustCredential,
  TrustPresentation,
  AttestationRequest,
  TrustTier,
  TrustVerificationResult,
  IssueTrustCredentialInput,
  SelfAttestTrustInput,
  CreateTrustPresentationInput,
  RequestAttestationInput,
  RevokeTrustCredentialInput,
  IdentityClientConfig,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';


/**
 * Trust credential record wrapper
 */
export interface TrustCredentialRecord {
  hash: Uint8Array;
  credential: TrustCredential;
}

/**
 * Trust presentation record wrapper
 */
export interface TrustPresentationRecord {
  hash: Uint8Array;
  presentation: TrustPresentation;
}

/**
 * Attestation request record wrapper
 */
export interface AttestationRequestRecord {
  hash: Uint8Array;
  request: AttestationRequest;
}

/**
 * Trust Credential Client
 *
 * Provides K-Vector trust credential management with zero-knowledge proofs.
 * Supports trust credential issuance, self-attestation, selective disclosure,
 * and verification.
 *
 * @example
 * ```typescript
 * const trustClient = new TrustClient(appClient);
 *
 * // Issue a trust credential (with ZK proof from off-chain computation)
 * const credential = await trustClient.issueTrustCredential({
 *   subject_did: 'did:mycelix:alice123',
 *   issuer_did: 'did:mycelix:validator456',
 *   kvector_commitment: commitmentBytes,
 *   range_proof: proofBytes,
 *   trust_score_lower: 0.7,
 *   trust_score_upper: 0.85,
 *   expires_at: Date.now() + (365 * 24 * 60 * 60 * 1000), // 1 year
 * });
 *
 * // Self-attest trust (lower weight than third-party attestation)
 * const selfAttestation = await trustClient.selfAttestTrust({
 *   self_did: 'did:mycelix:bob789',
 *   kvector_commitment: commitmentBytes,
 *   range_proof: proofBytes,
 *   trust_score_lower: 0.5,
 *   trust_score_upper: 0.6,
 * });
 *
 * // Create selective disclosure presentation
 * const presentation = await trustClient.createPresentation({
 *   credential_id: credential.credential.id,
 *   subject_did: 'did:mycelix:alice123',
 *   disclosed_tier: 'High',
 *   disclose_range: false, // Only reveal tier, not exact score
 *   trust_range: { lower: 0.7, upper: 0.85 },
 *   presentation_proof: presentationProofBytes,
 *   verifier_did: 'did:mycelix:marketplace001',
 *   purpose: 'Marketplace access verification',
 * });
 * ```
 */
export class TrustClient extends ZomeClient {
  protected readonly zomeName = 'trust_credential';

  constructor(client: AppClient, config: Partial<IdentityClientConfig> = {}) {
    const mergedConfig = { ...DEFAULT_IDENTITY_CLIENT_CONFIG, ...config };
    super(client, {
      roleName: mergedConfig.roleName,
      timeout: mergedConfig.timeout,
    });
  }

  // ==========================================================================
  // CREDENTIAL ISSUANCE
  // ==========================================================================

  /**
   * Issue a new trust credential
   *
   * Creates a trust credential with K-Vector commitment and ZKP proof.
   * The issuer vouches for the subject's trust properties.
   *
   * @param input - Trust credential parameters
   * @returns Created trust credential record
   */
  async issueTrustCredential(input: IssueTrustCredentialInput): Promise<TrustCredentialRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('issue_trust_credential', input);
    return this.recordToCredentialRecord(record);
  }

  /**
   * Self-attest trust
   *
   * An agent can self-attest their K-Vector with proof.
   * Self-attested credentials have lower trust weight than third-party attestations.
   *
   * @param input - Self-attestation parameters
   * @returns Created trust credential record
   */
  async selfAttestTrust(input: SelfAttestTrustInput): Promise<TrustCredentialRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('self_attest_trust', input);
    return this.recordToCredentialRecord(record);
  }

  /**
   * Revoke a trust credential
   *
   * @param input - Revocation parameters
   * @returns Updated trust credential record
   */
  async revokeCredential(input: RevokeTrustCredentialInput): Promise<TrustCredentialRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('revoke_credential', input);
    return this.recordToCredentialRecord(record);
  }

  // ==========================================================================
  // PRESENTATIONS
  // ==========================================================================

  /**
   * Create a trust presentation with selective disclosure
   *
   * Creates a presentation that reveals only the trust tier (or optionally
   * the score range) without revealing the exact score.
   *
   * @param input - Presentation parameters
   * @returns Created presentation record
   */
  async createPresentation(input: CreateTrustPresentationInput): Promise<TrustPresentationRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('create_presentation', input);
    return this.recordToPresentationRecord(record);
  }

  // ==========================================================================
  // ATTESTATION REQUESTS
  // ==========================================================================

  /**
   * Request attestation from another agent
   *
   * Creates a request for trust attestation with specified requirements.
   *
   * @param input - Attestation request parameters
   * @returns Created request record
   */
  async requestAttestation(input: RequestAttestationInput): Promise<AttestationRequestRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('request_attestation', input);
    return this.recordToRequestRecord(record);
  }

  // ==========================================================================
  // QUERIES
  // ==========================================================================

  /**
   * Get trust credentials for a subject
   *
   * Returns all non-revoked trust credentials for a given DID.
   *
   * @param subjectDid - DID of the subject
   * @returns Array of trust credential records
   */
  async getSubjectCredentials(subjectDid: string): Promise<TrustCredentialRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_subject_credentials', subjectDid);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  /**
   * Get credentials by trust tier
   *
   * Returns all non-revoked credentials at a specific trust tier.
   *
   * @param tier - Trust tier to query
   * @returns Array of trust credential records
   */
  async getCredentialsByTier(tier: TrustTier): Promise<TrustCredentialRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_credentials_by_tier', tier);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  // ==========================================================================
  // VERIFICATION
  // ==========================================================================

  /**
   * Verify a trust credential
   *
   * Performs on-chain verification of commitment format and tier consistency.
   * Full STARK proof verification should be done off-chain.
   *
   * @param credentialId - ID of the credential to verify
   * @returns Verification result
   */
  async verifyCredential(credentialId: string): Promise<TrustVerificationResult> {
    return this.callZome<TrustVerificationResult>('verify_credential', credentialId);
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private recordToCredentialRecord(record: HolochainRecord): TrustCredentialRecord {
    const credential = this.extractEntry<TrustCredential>(record);
    return {
      hash: this.getActionHash(record),
      credential,
    };
  }

  private recordToPresentationRecord(record: HolochainRecord): TrustPresentationRecord {
    const presentation = this.extractEntry<TrustPresentation>(record);
    return {
      hash: this.getActionHash(record),
      presentation,
    };
  }

  private recordToRequestRecord(record: HolochainRecord): AttestationRequestRecord {
    const request = this.extractEntry<AttestationRequest>(record);
    return {
      hash: this.getActionHash(record),
      request,
    };
  }
}
