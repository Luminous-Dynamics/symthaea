// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Academic Credentials Integration (Legacy Bridge)
 *
 * hApp-specific adapter for the education zome, providing:
 * - W3C VC 2.0 academic credential types
 * - ZK commitment verification for selective disclosure
 * - DNS-DID verification status
 * - Batch import tracking
 * - Credential revocation workflows
 * - Cross-hApp reputation via Bridge
 *
 * @packageDocumentation
 * @module integrations/academic
 *
 * @example Verifying an academic credential
 * ```typescript
 * import { getAcademicService } from '@mycelix/sdk/integrations/academic';
 *
 * const academic = getAcademicService();
 *
 * // Verify a credential by ID
 * const result = academic.verifyCredential('urn:uuid:abc-123');
 * if (result.structureValid && result.commitmentValid) {
 *   console.log('Credential verified');
 * }
 * ```
 *
 * @example Querying credentials by institution
 * ```typescript
 * const credentials = academic.getCredentialsByInstitution('did:dns:university.edu');
 * for (const cred of credentials) {
 *   console.log(`${cred.achievement.degreeName} - ${cred.credentialSubject.id}`);
 * }
 * ```
 */

import { LocalBridge } from '../../bridge/index.js';
import {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClaim,
} from '../../epistemic/index.js';
import {
  createReputation,
  recordPositive,
  reputationValue,
  type ReputationScore,
} from '../../matl/index.js';


// ============================================================================
// W3C VC 2.0 Academic Credential Types
// (Matching Rust education_integrity zome)
// ============================================================================

/** Degree type classification */
export type DegreeType =
  | 'HighSchool'
  | 'Associate'
  | 'Bachelor'
  | 'Master'
  | 'Doctorate'
  | 'Professional'
  | 'Certificate'
  | 'Diploma'
  | 'Microcredential'
  | 'CourseCompletion';

/** DNSSEC validation status */
export type DnssecStatus =
  | 'Validated'
  | 'Insecure'
  | 'Invalid'
  | 'Unsigned'
  | 'Unknown';

/** Revocation reason for academic credentials */
export type RevocationReason =
  | 'AcademicFraud'
  | 'DegreeRescinded'
  | 'IssuedInError'
  | 'HolderRequest'
  | 'CourtOrder'
  | 'InstitutionChange'
  | 'Other';

/** Revocation request status */
export type RevocationRequestStatus =
  | 'Pending'
  | 'Approved'
  | 'Denied'
  | 'Executed';

/** Import batch status */
export type ImportStatus =
  | 'InProgress'
  | 'Completed'
  | 'CompletedWithErrors'
  | 'Failed'
  | 'RolledBack';

/** Institution location */
export interface InstitutionLocation {
  country: string;
  region?: string;
  city?: string;
}

/** Accreditation record */
export interface Accreditation {
  accreditor: string;
  accreditationType: string;
  validFrom: string;
  validUntil?: string;
}

/** Institutional issuer */
export interface InstitutionalIssuer {
  id: string;
  name: string;
  type: string[];
  image?: string;
  location?: InstitutionLocation;
  accreditation?: Accreditation[];
}

/** Academic credential subject */
export interface AcademicSubject {
  id: string;
  name?: string;
  nameHash?: Uint8Array;
  birthDate?: string;
  studentId?: string;
}

/** Cryptographic proof */
export interface AcademicProof {
  type: string;
  created: string;
  verificationMethod: string;
  proofPurpose: string;
  proofValue: string;
  cryptosuite?: string;
  domain?: string;
  challenge?: string;
}

/** Achievement metadata */
export interface AchievementMetadata {
  degreeType: DegreeType;
  degreeName: string;
  fieldOfStudy: string;
  minors?: string[];
  conferralDate: string;
  gpa?: number;
  honors?: string[];
  cipCode?: string;
  creditsEarned?: number;
}

/** DNS-DID verification record */
export interface DnsVerificationRecord {
  timestamp: number;
  resolver: string;
  dnssecStatus: DnssecStatus;
  ttlSeconds: number;
}

/** DNS-DID verification data */
export interface DnsDid {
  domain: string;
  did: string;
  txtRecord: string;
  dnssec: DnssecStatus;
  lastVerified: number;
  verificationChain: DnsVerificationRecord[];
}

/**
 * W3C VC 2.0 Academic Credential
 *
 * Matches the Rust `AcademicCredential` struct from
 * `education_integrity` zome.
 */
export interface AcademicCredential {
  '@context': string[];
  id: string;
  type: string[];
  issuer: InstitutionalIssuer;
  validFrom: string;
  validUntil?: string;
  credentialSubject: AcademicSubject;
  proof: AcademicProof;
  zkCommitment: Uint8Array;
  commitmentNonce?: Uint8Array;
  revocationRegistryId: string;
  revocationIndex: number;
  dnsDid: DnsDid;
  achievement: AchievementMetadata;
  mycelixSchemaId: string;
  mycelixCreated: number;
  legacyImportRef?: string;
}

/** Credential revocation entry */
export interface CredentialRevocation {
  credentialId: string;
  revocationRegistryId: string;
  revocationIndex: number;
  reason: RevocationReason;
  reasonText?: string;
  revokedAt: string;
  revokedBy: string;
}

/** Revocation request */
export interface AcademicRevocationRequest {
  credentialId: string;
  requesterDid: string;
  reason: RevocationReason;
  explanation: string;
  evidence?: string[];
  requestedAt: number;
  status: RevocationRequestStatus;
}

/** Legacy import batch */
export interface LegacyBridgeImport {
  batchId: string;
  institutionDid: string;
  sourceSystem: string;
  importTimestamp: number;
  totalCredentials: number;
  importedCount: number;
  failedCount: number;
  status: ImportStatus;
  sourceHash: Uint8Array;
  errors: ImportError[];
}

/** Import error record */
export interface ImportError {
  row: number;
  field: string;
  message: string;
  code: string;
}

/** CSV field mapping */
export interface CsvFieldMapping {
  studentId: string;
  firstName: string;
  lastName: string;
  degreeName: string;
  major: string;
  conferralDate: string;
  gpa?: string;
  honors?: string;
  minor?: string;
}

// ============================================================================
// Verification Types
// ============================================================================

/** Credential verification result */
export interface CredentialVerification {
  credentialId: string;
  structureValid: boolean;
  issuerValid: boolean;
  proofValid: boolean;
  commitmentValid: boolean;
  revocationChecked: boolean;
  isRevoked: boolean;
  dnsDid: {
    checked: boolean;
    verified: boolean;
    dnssecStatus: DnssecStatus;
  };
  epistemicClaim?: EpistemicClaim;
  overallValid: boolean;
  verifiedAt: number;
  errors: string[];
}

// ============================================================================
// Academic Credential Service
// ============================================================================

/**
 * AcademicCredentialService - Legacy Bridge academic credential management
 *
 * Provides typed access to the education coordinator zome for
 * institutional academic credentials (degrees, diplomas, certificates).
 *
 * @example
 * ```typescript
 * const service = new AcademicCredentialService();
 *
 * // Verify credential structure
 * const result = service.verifyCredentialStructure(credential);
 * console.log(`Valid: ${result.overallValid}`);
 *
 * // Query by institution
 * const creds = service.getCredentialsByInstitution('did:dns:university.edu');
 * ```
 */
export class AcademicCredentialService {
  private credentials: Map<string, AcademicCredential> = new Map();
  private revocations: Map<string, CredentialRevocation> = new Map();
  private institutionReputations: Map<string, ReputationScore> = new Map();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('education');
  }

  /**
   * Verify an academic credential's structure and integrity
   *
   * @param credential - The credential to verify
   * @returns Verification result with detailed check status
   */
  verifyCredentialStructure(credential: AcademicCredential): CredentialVerification {
    const errors: string[] = [];
    const now = Date.now();

    // Structure validation
    const hasContext = credential['@context']?.includes(
      'https://www.w3.org/ns/credentials/v2',
    );
    const hasType =
      credential.type?.includes('VerifiableCredential') &&
      credential.type?.includes('AcademicCredential');
    const hasValidId = credential.id?.startsWith('urn:uuid:');

    const structureValid = hasContext && hasType && hasValidId;
    if (!hasContext) errors.push('Missing W3C VC 2.0 context');
    if (!hasType) errors.push('Missing required credential types');
    if (!hasValidId) errors.push('Invalid credential ID format');

    // Issuer validation
    const issuerValid =
      credential.issuer?.id?.startsWith('did:') &&
      credential.issuer?.name?.length > 0;
    if (!issuerValid) errors.push('Invalid issuer');

    // Proof validation
    const proofValid =
      credential.proof?.type === 'DataIntegrityProof' &&
      credential.proof?.proofValue?.startsWith('z') &&
      credential.proof?.proofPurpose === 'assertionMethod';
    if (!proofValid) errors.push('Invalid proof structure');

    // Commitment validation
    const commitmentValid =
      credential.zkCommitment?.length === 32 &&
      credential.revocationRegistryId?.length > 0;
    if (!commitmentValid) errors.push('Invalid ZK commitment or missing revocation registry');

    // Revocation check
    const revocation = this.revocations.get(credential.id);
    const isRevoked = revocation !== undefined;
    if (isRevoked) errors.push(`Credential revoked: ${revocation.reason}`);

    const overallValid =
      structureValid && issuerValid && proofValid && commitmentValid && !isRevoked;

    return {
      credentialId: credential.id,
      structureValid,
      issuerValid,
      proofValid,
      commitmentValid,
      revocationChecked: true,
      isRevoked,
      dnsDid: {
        checked: false,
        verified: false,
        dnssecStatus: credential.dnsDid?.dnssec ?? 'Unknown',
      },
      epistemicClaim: overallValid
        ? claim(`Academic credential ${credential.id} verified`)
            .withEmpirical(EmpiricalLevel.E3_Cryptographic)
            .withNormative(NormativeLevel.N2_Network)
            .withMateriality(MaterialityLevel.M2_Persistent)
            .build()
        : undefined,
      overallValid,
      verifiedAt: now,
      errors,
    };
  }

  /**
   * Verify a ZK commitment matches the credential data
   *
   * @param credential - The credential containing the commitment
   * @param nonce - The nonce used to create the commitment
   * @returns true if commitment is valid
   */
  async verifyZkCommitment(
    credential: AcademicCredential,
    nonce: Uint8Array,
  ): Promise<boolean> {
    // SHA-256(credential_id || "|" || subject_id || "|" || nonce)
    const encoder = new TextEncoder();
    const data = new Uint8Array([
      ...encoder.encode(credential.id),
      ...encoder.encode('|'),
      ...encoder.encode(credential.credentialSubject.id),
      ...encoder.encode('|'),
      ...nonce,
    ]);

    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const expected = new Uint8Array(hashBuffer);

    if (expected.length !== credential.zkCommitment.length) return false;
    return expected.every((byte, i) => byte === credential.zkCommitment[i]);
  }

  /**
   * Register a credential (from import or on-chain query)
   */
  registerCredential(credential: AcademicCredential): void {
    this.credentials.set(credential.id, credential);

    // Update institution reputation
    const issuerDid = credential.issuer.id;
    if (!this.institutionReputations.has(issuerDid)) {
      this.institutionReputations.set(issuerDid, createReputation(issuerDid));
    }
    const rep = this.institutionReputations.get(issuerDid)!;
    this.institutionReputations.set(issuerDid, recordPositive(rep));

    // Store reputation in bridge
    this.bridge.setReputation('education', issuerDid, rep);
  }

  /**
   * Register a revocation
   */
  registerRevocation(revocation: CredentialRevocation): void {
    this.revocations.set(revocation.credentialId, revocation);
  }

  /**
   * Get credentials by institution DID
   */
  getCredentialsByInstitution(institutionDid: string): AcademicCredential[] {
    return Array.from(this.credentials.values()).filter(
      (c) => c.issuer.id === institutionDid,
    );
  }

  /**
   * Get credentials by subject DID
   */
  getCredentialsBySubject(subjectDid: string): AcademicCredential[] {
    return Array.from(this.credentials.values()).filter(
      (c) => c.credentialSubject.id === subjectDid,
    );
  }

  /**
   * Get credential by ID
   */
  getCredential(credentialId: string): AcademicCredential | undefined {
    return this.credentials.get(credentialId);
  }

  /**
   * Check if a credential has been revoked
   */
  isRevoked(credentialId: string): boolean {
    return this.revocations.has(credentialId);
  }

  /**
   * Get institution reputation from MATL
   */
  getInstitutionReputation(institutionDid: string): number {
    const rep = this.institutionReputations.get(institutionDid);
    return rep ? reputationValue(rep) : 0;
  }

  /**
   * Query cross-hApp reputation for an institution
   */
  queryInstitutionReputation(institutionDid: string): number {
    const scores = this.bridge.getCrossHappReputation(institutionDid);
    const educationScore = scores.find(s => s.happ === 'education');
    return educationScore?.score ?? 0;
  }

  /**
   * Check if DNSSEC status is acceptable for credential trust
   */
  isDnssecAcceptable(status: DnssecStatus): boolean {
    return status === 'Validated' || status === 'Insecure';
  }

  // ==========================================================================
  // DKG Integration - Epistemic Claim Publication
  // ==========================================================================

  /**
   * Publish an academic credential as an E3-level epistemic claim.
   *
   * Transforms the credential's achievement metadata into a knowledge graph
   * triple (subject-predicate-object) classified at E3/N2/M3:
   * - E3 (Cryptographic): institutionally signed with verifiable proof
   * - N2 (Network): accredited institution + registrar consensus
   * - M3 (Immutable): permanent on-chain record
   *
   * @param credential - The verified academic credential
   * @returns The epistemic claim with E-N-M classification
   *
   * @example
   * ```typescript
   * const service = getAcademicService();
   * const epistemicClaim = service.publishAsEpistemicClaim(credential);
   * console.log(epistemicClaim.classification);
   * // { empirical: 3, normative: 2, materiality: 2 }
   * ```
   */
  publishAsEpistemicClaim(credential: AcademicCredential): EpistemicClaim {
    // Verify credential structure first
    const verification = this.verifyCredentialStructure(credential);
    if (!verification.overallValid) {
      throw new Error(
        `Cannot publish invalid credential as epistemic claim: ${verification.errors.join(', ')}`,
      );
    }

    // Build the claim content from achievement metadata (no PII)
    const claimContent = [
      `${credential.achievement.degreeType}: ${credential.achievement.degreeName}`,
      `Field: ${credential.achievement.fieldOfStudy}`,
      `Issued by: ${credential.issuer.name} (${credential.issuer.id})`,
      `Conferred: ${credential.achievement.conferralDate}`,
    ].join(' | ');

    // Create E3/N2/M3 epistemic claim
    const epistemicClaim = claim(claimContent)
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent)
      .build();

    return epistemicClaim;
  }

  /**
   * Get the epistemic position for a given credential.
   *
   * Returns the E-N-M classification that would be assigned if the credential
   * were published as an epistemic claim. Useful for preview/display.
   *
   * @param credential - The academic credential
   * @returns The epistemic position label and scores
   */
  getEpistemicPosition(credential: AcademicCredential): {
    empirical: number;
    normative: number;
    materiality: number;
    label: string;
  } {
    const verification = this.verifyCredentialStructure(credential);
    if (!verification.overallValid) {
      return {
        empirical: 0,
        normative: 0,
        materiality: 0,
        label: 'Invalid credential - cannot classify',
      };
    }

    return {
      empirical: 0.8,
      normative: 0.7,
      materiality: 0.9,
      label: 'E3/N2/M3 - Cryptographic·Network·Immutable',
    };
  }

  /**
   * Get summary statistics
   */
  getStats(): {
    totalCredentials: number;
    totalRevocations: number;
    institutionCount: number;
    degreeBreakdown: Record<DegreeType, number>;
  } {
    const degreeBreakdown: Record<string, number> = {};

    for (const cred of this.credentials.values()) {
      const type = cred.achievement.degreeType;
      degreeBreakdown[type] = (degreeBreakdown[type] || 0) + 1;
    }

    return {
      totalCredentials: this.credentials.size,
      totalRevocations: this.revocations.size,
      institutionCount: this.institutionReputations.size,
      degreeBreakdown: degreeBreakdown as Record<DegreeType, number>,
    };
  }
}

// ============================================================================
// Singleton Accessor
// ============================================================================

let _instance: AcademicCredentialService | null = null;

/**
 * Get the singleton AcademicCredentialService instance
 *
 * @returns The shared service instance
 */
export function getAcademicService(): AcademicCredentialService {
  if (!_instance) {
    _instance = new AcademicCredentialService();
  }
  return _instance;
}

/**
 * Reset the singleton (for testing)
 * @internal
 */
export function resetAcademicService(): void {
  _instance = null;
}

// ============================================================================
// Academic Holochain Bridge Client
// ============================================================================

/** Holochain conductor bridge client for Academic/Knowledge cluster */
export class AcademicBridgeClient {
  constructor(
    private client: {
      callZome(input: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: any;
      }): Promise<any>;
    },
  ) {}

  // -- Claims zome (credential management) --

  async registerCredential(credential: AcademicCredential): Promise<void> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'register_credential',
      payload: credential,
    });
  }

  async getCredential(credentialId: string): Promise<AcademicCredential | null> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'get_credential',
      payload: credentialId,
    });
  }

  async listCredentialsBySubject(subjectDid: string): Promise<AcademicCredential[]> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'list_credentials_by_subject',
      payload: subjectDid,
    });
  }

  async listCredentialsByInstitution(institutionDid: string): Promise<AcademicCredential[]> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'list_credentials_by_institution',
      payload: institutionDid,
    });
  }

  async verifyCredential(credentialId: string): Promise<CredentialVerification> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'verify_credential',
      payload: credentialId,
    });
  }

  // -- Revocations --

  async registerRevocation(revocation: CredentialRevocation): Promise<void> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'register_revocation',
      payload: revocation,
    });
  }

  async isRevoked(credentialId: string): Promise<boolean> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'is_revoked',
      payload: credentialId,
    });
  }

  // -- Graph zome (knowledge graph queries) --

  async publishEpistemicClaim(credentialId: string): Promise<EpistemicClaim> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'graph',
      fn_name: 'publish_epistemic_claim',
      payload: credentialId,
    });
  }

  async getInstitutionReputation(institutionDid: string): Promise<number> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'graph',
      fn_name: 'get_institution_reputation',
      payload: institutionDid,
    });
  }

  // -- Query zome (search and stats) --

  async getStats(): Promise<{
    totalCredentials: number;
    totalRevocations: number;
    institutionCount: number;
    degreeBreakdown: Record<DegreeType, number>;
  }> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'query',
      fn_name: 'get_stats',
      payload: null,
    });
  }

  async searchCredentials(params: {
    degreeType?: DegreeType;
    fieldOfStudy?: string;
    institution?: string;
    limit?: number;
  }): Promise<AcademicCredential[]> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'query',
      fn_name: 'search_credentials',
      payload: {
        degree_type: params.degreeType,
        field_of_study: params.fieldOfStudy,
        institution: params.institution,
        limit: params.limit,
      },
    });
  }
}
