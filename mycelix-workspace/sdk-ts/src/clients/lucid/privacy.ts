// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Privacy Zome Client
 *
 * Sharing policies, access grants, access logging,
 * and zero-knowledge proof attestations.
 *
 * @module @mycelix/sdk/clients/lucid/privacy
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  SetPolicyInput,
  GrantAccessInput,
  CheckAccessInput,
  LogAccessInput,
  SubmitAttestationInput,
  ValidAttestationInput,
} from './types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface PrivacyClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: PrivacyClientConfig = {
  roleName: 'lucid',
};

/**
 * Client for the LUCID Privacy zome
 *
 * Manages sharing policies, access control, audit logging,
 * and ZK proof attestations.
 */
export class PrivacyClient extends ZomeClient {
  protected readonly zomeName = 'lucid_privacy';

  constructor(client: AppClient, config: PrivacyClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Sharing Policies
  // ============================================================================

  /** Set sharing policy for a thought */
  async setSharingPolicy(input: SetPolicyInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('set_sharing_policy', input);
  }

  /** Get sharing policy for a thought */
  async getSharingPolicy(thoughtId: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_sharing_policy', thoughtId);
  }

  // ============================================================================
  // Access Control
  // ============================================================================

  /** Grant access to an agent for a specific thought */
  async grantAccess(input: GrantAccessInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('grant_access', input);
  }

  /** Get all grants for a thought */
  async getThoughtGrants(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thought_grants', thoughtId);
  }

  /** Check if an agent has access to a thought */
  async checkAccess(input: CheckAccessInput): Promise<boolean> {
    return this.callZome<boolean>('check_access', input);
  }

  // ============================================================================
  // Access Logging
  // ============================================================================

  /** Log an access event for audit purposes */
  async logAccess(input: LogAccessInput): Promise<Uint8Array> {
    return this.callZomeOnce<Uint8Array>('log_access', input);
  }

  // ============================================================================
  // ZK Proof Attestations
  // ============================================================================

  /** Submit a ZK proof attestation after verification */
  async submitProofAttestation(input: SubmitAttestationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('submit_proof_attestation', input);
  }

  /** Get attestation by proof CID */
  async getProofAttestation(proofCid: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_proof_attestation', proofCid);
  }

  /** Get all attestations for a subject */
  async getSubjectAttestations(subjectHash: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_subject_attestations', subjectHash);
  }

  /** Get attestations by proof type */
  async getAttestationsByType(proofType: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_attestations_by_type', proofType);
  }

  /** Get attestations created by a specific verifier */
  async getVerifierAttestations(verifier: Uint8Array): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_verifier_attestations', verifier);
  }

  /** Check if a subject has a valid (non-expired, verified) attestation */
  async hasValidAttestation(input: ValidAttestationInput): Promise<boolean> {
    return this.callZome<boolean>('has_valid_attestation', input);
  }
}
