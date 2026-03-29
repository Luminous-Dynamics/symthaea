// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * MFA (Multi-Factor Authentication) Zome Client
 *
 * Client for Multi-Factor Decentralized Identity (MFDI) in Mycelix-Identity.
 * Implements the MFDI specification v1.0 with 9 factor types and 5 assurance levels.
 */

import { IdentitySdkError, IdentitySdkErrorCode } from '../types.js';

import type { AppClient } from '@holochain/client';

// =============================================================================
// Types
// =============================================================================

/**
 * Authentication factor types as defined in MFDI specification
 */
export type FactorType =
  | 'PrimaryKeyPair'
  | 'HardwareKey'
  | 'Biometric'
  | 'SocialRecovery'
  | 'ReputationAttestation'
  | 'GitcoinPassport'
  | 'VerifiableCredential'
  | 'RecoveryPhrase'
  | 'SecurityQuestions';

/**
 * Factor categories for grouping
 */
export type FactorCategory =
  | 'knowledge'
  | 'possession'
  | 'inherence'
  | 'social'
  | 'reputation';

/**
 * Assurance levels from MFDI specification
 */
export type AssuranceLevel =
  | 'Anonymous'
  | 'Basic'
  | 'Verified'
  | 'HighlyAssured'
  | 'ConstitutionallyCritical';

/**
 * Enrolled authentication factor
 */
export interface EnrolledFactor {
  id: string;
  factorType: FactorType;
  category: FactorCategory;
  createdAt: number;
  lastUsedAt: number | null;
  metadata: Record<string, unknown>;
  verified: boolean;
}

/**
 * Complete MFA state for a DID
 */
export interface MfaState {
  did: string;
  factors: EnrolledFactor[];
  assuranceLevel: AssuranceLevel;
  lastVerifiedAt: number | null;
  flEligible: boolean;
}

/**
 * Factor enrollment input
 */
export interface EnrollFactorInput {
  factorType: FactorType;
  metadata?: Record<string, unknown>;
  proof?: FactorProof;
}

/**
 * Proof for factor verification
 */
export type FactorProof =
  | { type: 'signature'; signature: string; message: string }
  | { type: 'webauthn'; authenticatorData: string; clientDataHash: string; signature: string }
  | { type: 'biometric'; templateHash: string; response: string }
  | { type: 'gitcoinPassport'; score: number; checkedAt: number; stamps: string[] }
  | { type: 'verifiableCredential'; credential: string; issuer: string; credentialType: string }
  | { type: 'socialRecovery'; guardianSignatures: GuardianAttestation[]; threshold: number }
  | { type: 'knowledge'; answerHash: string };

/**
 * Guardian attestation for social recovery
 */
export interface GuardianAttestation {
  guardianDid: string;
  signature: string;
  timestamp: number;
}

/**
 * Verification challenge
 */
export interface VerificationChallenge {
  challengeId: string;
  factorId: string;
  factorType: FactorType;
  challenge: string;
  expiresAt: number;
}

/**
 * Verification result
 */
export interface MfaVerificationResult {
  success: boolean;
  factorId: string;
  verifiedAt: number;
  newAssuranceLevel?: AssuranceLevel;
}

/**
 * FL (Federated Learning) eligibility status
 */
export interface FlEligibilityResult {
  eligible: boolean;
  requirements: FlRequirement[];
  currentScore: number;
  requiredScore: number;
}

/**
 * FL requirement
 */
export interface FlRequirement {
  name: string;
  met: boolean;
  description: string;
}

/**
 * Factor enrollment history entry
 */
export interface FactorEnrollment {
  factorId: string;
  factorType: FactorType;
  action: 'enrolled' | 'removed' | 'verified' | 'failed';
  timestamp: number;
  metadata?: Record<string, unknown>;
}

// =============================================================================
// Client
// =============================================================================

/**
 * MFA Zome Client
 *
 * @example
 * ```typescript
 * const mfaClient = new MfaClient(appClient, 'mycelix_identity');
 *
 * // Get MFA state for a DID
 * const state = await mfaClient.getMfaState('did:mycelix:abc123...');
 * console.log(`Assurance Level: ${state.assuranceLevel}`);
 *
 * // Enroll a new factor
 * const factor = await mfaClient.enrollFactor('did:mycelix:abc123...', {
 *   factorType: 'HardwareKey',
 *   metadata: { keyName: 'YubiKey 5' }
 * });
 * ```
 */
export class MfaClient {
  private readonly roleName: string;
  private readonly zomeName = 'mfa';

  constructor(
    private readonly client: AppClient,
    roleName: string
  ) {
    this.roleName = roleName;
  }

  /**
   * Get MFA state for a DID
   *
   * @param did - DID string
   * @returns MFA state or null if not found
   */
  async getMfaState(did: string): Promise<MfaState | null> {
    this.validateDid(did);
    const result = await this.call<MfaState | null>('get_mfa_state', did);
    return result;
  }

  /**
   * Create initial MFA state for a DID
   *
   * @param did - DID string
   * @param primaryKeyHash - Hash of the primary key
   * @returns Created MFA state
   */
  async createMfaState(did: string, primaryKeyHash: string): Promise<MfaState> {
    this.validateDid(did);
    const result = await this.call<MfaState>('create_mfa_state', {
      did,
      primary_key_hash: primaryKeyHash,
    });
    return result;
  }

  /**
   * Enroll a new authentication factor
   *
   * @param did - DID string
   * @param input - Factor enrollment input
   * @returns Enrolled factor details
   */
  async enrollFactor(did: string, input: EnrollFactorInput): Promise<EnrolledFactor> {
    this.validateDid(did);
    const result = await this.call<EnrolledFactor>('enroll_factor', {
      did,
      factor_type: input.factorType,
      metadata: input.metadata ?? {},
      proof: input.proof,
    });
    return result;
  }

  /**
   * Remove an enrolled factor
   *
   * @param did - DID string
   * @param factorId - ID of the factor to remove
   * @returns Updated MFA state
   */
  async removeFactor(did: string, factorId: string): Promise<MfaState> {
    this.validateDid(did);
    const result = await this.call<MfaState>('remove_factor', {
      did,
      factor_id: factorId,
    });
    return result;
  }

  /**
   * Generate a verification challenge for a factor
   *
   * @param did - DID string
   * @param factorId - ID of the factor to verify
   * @returns Verification challenge
   */
  async generateChallenge(did: string, factorId: string): Promise<VerificationChallenge> {
    this.validateDid(did);
    const result = await this.call<VerificationChallenge>('generate_verification_challenge', {
      did,
      factor_id: factorId,
    });
    return result;
  }

  /**
   * Verify a factor with proof
   *
   * @param did - DID string
   * @param factorId - ID of the factor to verify
   * @param proof - Verification proof
   * @returns Verification result
   */
  async verifyFactor(did: string, factorId: string, proof: FactorProof): Promise<MfaVerificationResult> {
    this.validateDid(did);
    const result = await this.call<MfaVerificationResult>('verify_factor_proof', {
      did,
      factor_id: factorId,
      proof,
    });
    return result;
  }

  /**
   * Get current assurance level for a DID
   *
   * @param did - DID string
   * @returns Assurance level and score
   */
  async getAssuranceLevel(did: string): Promise<{ level: AssuranceLevel; score: number }> {
    this.validateDid(did);
    const result = await this.call<{ level: AssuranceLevel; score: number }>(
      'calculate_assurance_level',
      did
    );
    return result;
  }

  /**
   * Check FL (Federated Learning) eligibility
   *
   * @param did - DID string
   * @returns FL eligibility status with requirements
   */
  async checkFlEligibility(did: string): Promise<FlEligibilityResult> {
    this.validateDid(did);
    const result = await this.call<FlEligibilityResult>('check_fl_eligibility', did);
    return result;
  }

  /**
   * Get factor enrollment history
   *
   * @param did - DID string
   * @param limit - Maximum number of entries (default 50)
   * @returns List of enrollment history entries
   */
  async getEnrollmentHistory(did: string, limit = 50): Promise<FactorEnrollment[]> {
    this.validateDid(did);
    const result = await this.call<FactorEnrollment[]>('get_enrollment_history', {
      did,
      limit,
    });
    return result;
  }

  /**
   * Get all factors by type
   *
   * @param did - DID string
   * @param factorType - Type of factors to retrieve
   * @returns List of enrolled factors of the specified type
   */
  async getFactorsByType(did: string, factorType: FactorType): Promise<EnrolledFactor[]> {
    this.validateDid(did);
    const state = await this.getMfaState(did);
    if (!state) return [];
    return state.factors.filter((f) => f.factorType === factorType);
  }

  /**
   * Get MFA summary (lightweight version of state)
   *
   * @param did - DID string
   * @returns Summary with factor count, assurance level, etc.
   */
  async getMfaSummary(did: string): Promise<{
    factorCount: number;
    assuranceLevel: AssuranceLevel;
    flEligible: boolean;
    categories: FactorCategory[];
  } | null> {
    const state = await this.getMfaState(did);
    if (!state) return null;

    const categories = [...new Set(state.factors.map((f) => f.category))];

    return {
      factorCount: state.factors.length,
      assuranceLevel: state.assuranceLevel,
      flEligible: state.flEligible,
      categories,
    };
  }

  // =============================================================================
  // Private Helpers
  // =============================================================================

  private validateDid(did: string): void {
    if (!did.startsWith('did:mycelix:')) {
      throw new IdentitySdkError(
        IdentitySdkErrorCode.INVALID_DID_FORMAT,
        'DID must start with "did:mycelix:"',
        { did }
      );
    }
  }

  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.roleName,
        zome_name: this.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new IdentitySdkError(
        IdentitySdkErrorCode.ZOME_CALL_FAILED,
        `MFA zome call failed: ${fnName}: ${message}`,
        { fnName, payload }
      );
    }
  }
}

export default MfaClient;
