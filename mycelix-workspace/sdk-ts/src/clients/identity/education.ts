// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Education Credential Client
 *
 * Client for academic credential management in Mycelix-Identity.
 * Supports W3C VC 2.0 compliant academic credentials, legacy imports,
 * DNS-DID verification, and epistemic claim publication.
 *
 * @module @mycelix/sdk/clients/identity/education
 */

import { DEFAULT_IDENTITY_CLIENT_CONFIG } from './types.js';
import { ZomeClient } from '../../core/zome-client.js';

import type {
  AcademicCredential,
  EpistemicClaimReference,
  CreateAcademicCredentialInput,
  CreateAcademicCredentialOutput,
  StartLegacyImportInput,
  RequestAcademicRevocationInput,
  DnssecStatus,
  IdentityClientConfig,
} from './types.js';
import type { AppClient, ActionHash } from '@holochain/client';


/**
 * Academic credential record wrapper
 */
export interface AcademicCredentialRecord {
  hash: ActionHash;
  credential: AcademicCredential;
}

/**
 * DNS-DID verification result
 */
export interface VerifyDnsDidResult {
  verified: boolean;
  dnssec_status: DnssecStatus;
  resolved_did: string | null;
  verification_record: DnsVerificationRecord;
  error: string | null;
}

/**
 * DNS verification record
 */
export interface DnsVerificationRecord {
  timestamp: number;
  resolver: string;
  dnssec_status: DnssecStatus;
  ttl_seconds: number;
}

/**
 * Legacy import result
 */
export interface ImportCredentialResult {
  row_number: number;
  success: boolean;
  credential_id: string | null;
  action_hash: ActionHash | null;
  error: ImportError | null;
}

/**
 * Import error
 */
export interface ImportError {
  row: number;
  field: string;
  message: string;
  code: string;
}

/**
 * Epistemic position summary
 */
export interface EpistemicPosition {
  empirical: number;
  normative: number;
  materiality: number;
  label: string;
}

/**
 * Epistemic claim publication output
 */
export interface PublishCredentialAsClaimOutput {
  claim_ref_hash: ActionHash;
  claim_id: string;
  epistemic_position: EpistemicPosition;
}

/**
 * Education Credential Client
 *
 * Provides academic credential management with institutional verification,
 * DNS-DID binding, legacy system imports, and epistemic claim publication.
 *
 * @example
 * ```typescript
 * const educationClient = new EducationClient(appClient);
 *
 * // Create an academic credential
 * const result = await educationClient.createAcademicCredential({
 *   issuer: {
 *     id: 'did:web:university.edu',
 *     name: 'Example University',
 *     url: 'https://university.edu',
 *   },
 *   subject: {
 *     id: 'did:mycelix:student123',
 *     name: 'Alice Johnson',
 *     student_id: 'STU-2024-001',
 *   },
 *   achievement: {
 *     degree_type: 'Bachelor',
 *     degree_name: 'Bachelor of Science',
 *     field_of_study: 'Computer Science',
 *     conferral_date: '2024-05-15',
 *     gpa: 3.85,
 *     honors: ['Cum Laude'],
 *   },
 *   dns_did: {
 *     domain: 'university.edu',
 *     did: 'did:web:university.edu',
 *     txt_record: '_did.university.edu',
 *   },
 *   revocation_registry_id: 'university:revocation:2024',
 *   valid_from: '2024-05-15',
 *   proof: {
 *     type: 'DataIntegrityProof',
 *     created: new Date().toISOString(),
 *     verificationMethod: 'did:web:university.edu#key-1',
 *     proofPurpose: 'assertionMethod',
 *     proofValue: signatureValue,
 *   },
 * });
 *
 * // Publish as epistemic claim
 * const claim = await educationClient.publishCredentialAsEpistemicClaim({
 *   credential_action_hash: result.action_hash,
 * });
 * console.log(`Published claim ${claim.claim_id} at E${claim.epistemic_position.empirical}`);
 * ```
 */
export class EducationClient extends ZomeClient {
  protected readonly zomeName = 'education';

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
   * Create a new academic credential
   *
   * Creates a W3C VC 2.0 compliant academic credential with ZK commitment
   * and DNS-DID verification support.
   *
   * @param input - Academic credential parameters
   * @returns Created credential output with action hash and ID
   */
  async createAcademicCredential(
    input: CreateAcademicCredentialInput
  ): Promise<CreateAcademicCredentialOutput> {
    return this.callZomeOnce<CreateAcademicCredentialOutput>('create_academic_credential', input);
  }

  /**
   * Get an academic credential by action hash
   *
   * @param actionHash - Action hash of the credential
   * @returns Academic credential or null if not found
   */
  async getAcademicCredential(actionHash: ActionHash): Promise<AcademicCredential | null> {
    return this.callZome<AcademicCredential | null>('get_academic_credential', actionHash);
  }

  /**
   * Get credentials by institution DID
   *
   * @param institutionDid - DID of the issuing institution
   * @returns Array of credential action hashes
   */
  async getCredentialsByInstitution(institutionDid: string): Promise<ActionHash[]> {
    return this.callZome<ActionHash[]>('get_credentials_by_institution', institutionDid);
  }

  /**
   * Get credentials by subject DID
   *
   * @param subjectDid - DID of the credential subject (student)
   * @returns Array of credential action hashes
   */
  async getCredentialsBySubject(subjectDid: string): Promise<ActionHash[]> {
    return this.callZome<ActionHash[]>('get_credentials_by_subject', subjectDid);
  }

  // ==========================================================================
  // LEGACY IMPORT
  // ==========================================================================

  /**
   * Start a legacy import batch
   *
   * Initializes a batch import session for credentials from legacy systems.
   *
   * @param input - Import batch parameters
   * @returns Batch ID for tracking
   */
  async startLegacyImport(input: StartLegacyImportInput): Promise<string> {
    return this.callZomeOnce<string>('start_legacy_import', input);
  }

  /**
   * Import a credential from CSV data
   *
   * Processes a single row from a legacy CSV import.
   *
   * @param input - CSV row data
   * @returns Import result with success/failure status
   */
  async importCredentialFromCsv(input: {
    batch_id: string;
    row_number: number;
    student_id: string;
    first_name: string;
    last_name: string;
    degree_name: string;
    major: string;
    conferral_date: string;
    gpa?: number;
    honors?: string[];
  }): Promise<ImportCredentialResult> {
    return this.callZomeOnce<ImportCredentialResult>('import_credential_from_csv', input);
  }

  // ==========================================================================
  // DNS-DID VERIFICATION
  // ==========================================================================

  /**
   * Record DNS-DID verification result
   *
   * Records the result of DNS-DID verification performed off-chain.
   *
   * @param input - Verification data
   * @returns Verification result
   */
  async recordDnsDidVerification(input: {
    domain: string;
    expected_did: string;
    dnssec_status: DnssecStatus;
    resolved_did: string | null;
  }): Promise<VerifyDnsDidResult> {
    return this.callZomeOnce<VerifyDnsDidResult>('record_dns_did_verification', input);
  }

  /**
   * Verify ZK commitment
   *
   * Verifies that a ZK commitment matches the expected values.
   *
   * @param input - Commitment verification parameters
   * @returns true if commitment is valid
   */
  async verifyZkCommitment(input: {
    credential_id: string;
    subject_id: string;
    nonce: number[];
    expected_commitment: number[];
  }): Promise<boolean> {
    return this.callZome<boolean>('verify_zk_commitment', input);
  }

  // ==========================================================================
  // REVOCATION
  // ==========================================================================

  /**
   * Request academic credential revocation
   *
   * Creates a revocation request that must be approved by the institution.
   *
   * @param input - Revocation request parameters
   * @returns Action hash of the revocation request
   */
  async requestAcademicRevocation(input: RequestAcademicRevocationInput): Promise<ActionHash> {
    return this.callZomeOnce<ActionHash>('request_academic_revocation', input);
  }

  // ==========================================================================
  // EPISTEMIC CLAIMS (DKG INTEGRATION)
  // ==========================================================================

  /**
   * Publish a credential as an epistemic claim
   *
   * Transforms the credential's achievement metadata into a subject-predicate-object
   * triple and stores an EpistemicClaimReference linking to the knowledge graph.
   *
   * Academic credentials are classified at E3/N2/M3:
   * - E3 (Cryptographic): Signed by institution with verifiable proof
   * - N2 (Network): Accredited institution + registrar consensus
   * - M3 (Immutable): Permanent on-chain record
   *
   * @param input - Credential to publish
   * @returns Published claim output with ID and epistemic position
   */
  async publishCredentialAsEpistemicClaim(input: {
    credential_action_hash: ActionHash;
  }): Promise<PublishCredentialAsClaimOutput> {
    return this.callZomeOnce<PublishCredentialAsClaimOutput>(
      'publish_credential_as_epistemic_claim',
      input
    );
  }

  /**
   * Get epistemic claims for a credential
   *
   * @param credentialId - ID of the credential
   * @returns Array of epistemic claim references
   */
  async getEpistemicClaimsForCredential(credentialId: string): Promise<EpistemicClaimReference[]> {
    return this.callZome<EpistemicClaimReference[]>(
      'get_epistemic_claims_for_credential',
      credentialId
    );
  }

  /**
   * Get epistemic claims for a subject
   *
   * @param subjectDid - DID of the subject (student)
   * @returns Array of epistemic claim references
   */
  async getEpistemicClaimsForSubject(subjectDid: string): Promise<EpistemicClaimReference[]> {
    return this.callZome<EpistemicClaimReference[]>('get_epistemic_claims_for_subject', subjectDid);
  }
}
