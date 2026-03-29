// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Identity hApp Client Types
 *
 * Comprehensive type definitions for all Identity hApp zomes.
 * These types mirror the Rust structs from mycelix-identity zomes.
 *
 * @module @mycelix/sdk/clients/identity/types
 */

import type { ActionHash, AgentPubKey, Timestamp, Record as HolochainRecord } from '@holochain/client';

// Re-export Holochain types
export type { ActionHash, AgentPubKey, Timestamp, HolochainRecord };

// =============================================================================
// DID REGISTRY TYPES
// =============================================================================

/**
 * Verification method for cryptographic operations
 *
 * Wire format uses W3C DID Core camelCase field names.
 */
export interface VerificationMethod {
  id: string;
  /** W3C: "type" (was type_ in Rust) */
  type: string;
  controller: string;
  /** W3C: "publicKeyMultibase" */
  publicKeyMultibase: string;
  /** Algorithm identifier (multicodec u16). Absent defaults to Ed25519. */
  algorithm?: number;
}

/**
 * Service endpoint for discovery
 *
 * Wire format uses W3C DID Core camelCase field names.
 */
export interface ServiceEndpoint {
  id: string;
  /** W3C: "type" (was type_ in Rust) */
  type: string;
  /** W3C: "serviceEndpoint" */
  serviceEndpoint: string;
}

/**
 * DID Document entry type
 *
 * Wire format uses W3C DID Core camelCase field names.
 */
export interface DidDocument {
  /** The DID identifier (did:mycelix:<agent_pub_key>) */
  id: string;
  /** Controller of this DID (usually self) */
  controller: AgentPubKey;
  /** Verification methods (public keys) */
  verificationMethod: VerificationMethod[];
  /** Authentication methods */
  authentication: string[];
  /** Key agreement methods for encryption (W3C DID Core §5.3.3) */
  keyAgreement?: string[];
  /** Service endpoints */
  service: ServiceEndpoint[];
  /** Creation timestamp */
  created: Timestamp;
  /** Last update timestamp */
  updated: Timestamp;
  /** Version number for updates */
  version: number;
}

/**
 * DID Deactivation record
 */
export interface DidDeactivation {
  did: string;
  reason: string;
  deactivatedAt: Timestamp;
}

/**
 * Input for updating a DID document
 */
export interface UpdateDidInput {
  verificationMethod?: VerificationMethod[];
  authentication?: string[];
  service?: ServiceEndpoint[];
}

/**
 * Record wrapper for DID documents
 */
export interface DidRecord {
  hash: ActionHash;
  document: DidDocument;
}

// =============================================================================
// VERIFIABLE CREDENTIAL TYPES
// =============================================================================

/**
 * Credential issuer - can be DID string or object with id
 */
export type CredentialIssuer =
  | string
  | {
      id: string;
      name?: string;
      issuer_type?: string[];
    };

/**
 * Credential subject containing the claims
 */
export interface CredentialSubject {
  id: string;
  claims: Record<string, unknown>;
}

/**
 * Reference to credential schema
 */
export interface CredentialSchemaRef {
  id: string;
  schema_type: string;
}

/**
 * Credential status for revocation
 *
 * Wire format uses W3C VC Data Model camelCase field names.
 */
export interface CredentialStatus {
  id: string;
  /** W3C: "type" */
  type: string;
  /** W3C: "statusPurpose" */
  statusPurpose?: string;
  /** W3C: "statusListIndex" */
  statusListIndex?: string;
  /** W3C: "statusListCredential" */
  statusListCredential?: string;
}

/**
 * Cryptographic proof
 *
 * Wire format uses W3C VC Data Model camelCase field names.
 */
export interface CredentialProof {
  /** W3C: "type" (was proof_type in Rust) */
  type: string;
  created: string;
  /** W3C: "verificationMethod" */
  verificationMethod: string;
  /** W3C: "proofPurpose" */
  proofPurpose: string;
  /** W3C: "proofValue" (multibase encoded) */
  proofValue: string;
  cryptosuite?: string;
}

/**
 * W3C Verifiable Credential
 *
 * Wire format uses W3C VC Data Model camelCase field names.
 */
export interface VerifiableCredential {
  /** JSON-LD context */
  '@context': string[];
  id: string;
  /** W3C: "type" (was credential_type in Rust) */
  type: string[];
  issuer: CredentialIssuer;
  /** W3C: "validFrom" */
  validFrom: string;
  /** W3C: "validUntil" */
  validUntil?: string;
  /** W3C: "credentialSubject" */
  credentialSubject: CredentialSubject;
  /** W3C: "credentialSchema" */
  credentialSchema?: CredentialSchemaRef;
  /** W3C: "credentialStatus" */
  credentialStatus?: CredentialStatus;
  proof: CredentialProof;
  mycelix_schema_id: string;
  mycelix_created: Timestamp;
}

/**
 * Verifiable Presentation
 *
 * Wire format uses W3C VC Data Model camelCase field names.
 */
export interface VerifiablePresentation {
  /** JSON-LD context */
  '@context': string[];
  id: string;
  /** W3C: "type" (was presentation_type in Rust) */
  type: string[];
  holder: string;
  /** W3C: "verifiableCredential" */
  verifiableCredential: VerifiableCredential[];
  proof: CredentialProof;
  mycelix_created: Timestamp;
}

/**
 * Derived credential for selective disclosure
 */
export interface DerivedCredential {
  original_credential_id: string;
  original_issuer: string;
  holder: string;
  selected_claims: string[];
  derived_content: CredentialSubject;
  derivation_proof: DerivationProof;
  created: Timestamp;
  expires?: Timestamp;
}

/**
 * Derivation proof for selective disclosure
 */
export interface DerivationProof {
  /** W3C: "type" (was proof_type in Rust) */
  type: string;
  original_credential_hash: number[];
  claim_proofs: ClaimProof[];
  holder_signature: number[];
}

/**
 * Individual claim proof
 */
export interface ClaimProof {
  claim_key: string;
  merkle_path?: string[];
  commitment?: string;
}

/**
 * Credential evidence
 */
export interface CredentialEvidence {
  evidence_type: string;
  id: string;
  description?: string;
}

/**
 * Status of credential request
 */
export type RequestStatus = 'Pending' | 'UnderReview' | 'Approved' | 'Rejected' | 'Issued';

/**
 * Credential issuance request
 */
export interface CredentialRequest {
  id: string;
  requester_did: string;
  issuer_did: string;
  schema_id: string;
  provided_claims: Record<string, unknown>;
  evidence: CredentialEvidence[];
  status: RequestStatus;
  created: Timestamp;
  updated: Timestamp;
}

/**
 * Verification result
 */
export interface VerificationResult {
  credential_id: string;
  valid: boolean;
  checks_passed: string[];
  errors: string[];
  verified_at: Timestamp;
}

/**
 * Input for issuing a credential
 */
export interface IssueCredentialInput {
  subject_did: string;
  schema_id: string;
  claims: Record<string, unknown>;
  credential_types?: string[];
  issuer_name?: string;
  expiration_days?: number;
  enable_revocation?: boolean;
}

/**
 * Input for creating a presentation
 */
export interface CreatePresentationInput {
  credential_ids: string[];
  challenge?: string;
  domain?: string;
}

/**
 * Input for creating a derived credential
 */
export interface CreateDerivedInput {
  credential_id: string;
  selected_claims: string[];
  expires_hours?: number;
}

/**
 * Input for requesting a credential
 */
export interface RequestCredentialInput {
  issuer_did: string;
  schema_id: string;
  claims: Record<string, unknown>;
  evidence?: CredentialEvidence[];
}

/**
 * Input for updating request status
 */
export interface UpdateRequestStatusInput {
  request_id: string;
  new_status: RequestStatus;
}

/**
 * Record wrapper for credentials
 */
export interface CredentialRecord {
  hash: ActionHash;
  credential: VerifiableCredential;
}

/**
 * Record wrapper for presentations
 */
export interface PresentationRecord {
  hash: ActionHash;
  presentation: VerifiablePresentation;
}

/**
 * Record wrapper for credential requests
 */
export interface RequestRecord {
  hash: ActionHash;
  request: CredentialRequest;
}

/**
 * Credential status response
 */
export interface CredentialStatusResponse {
  credential_id: string;
  is_valid: boolean;
  status_type: string;
  reason?: string;
  checked_at: Timestamp;
}

// =============================================================================
// RECOVERY TYPES
// =============================================================================

/**
 * Recovery configuration
 */
export interface RecoveryConfig {
  did: string;
  owner: AgentPubKey;
  trustees: string[];
  threshold: number;
  time_lock: number;
  active: boolean;
  created: Timestamp;
  updated: Timestamp;
}

/**
 * Recovery status
 */
export type RecoveryStatus =
  | 'Pending'
  | 'Approved'
  | 'Rejected'
  | 'Cancelled'
  | 'ReadyToExecute'
  | 'Completed';

/**
 * Recovery request
 */
export interface RecoveryRequest {
  id: string;
  did: string;
  new_agent: AgentPubKey;
  initiated_by: string;
  reason: string;
  status: RecoveryStatus;
  created: Timestamp;
  time_lock_expires?: Timestamp;
}

/**
 * Vote decision
 */
export type VoteDecision = 'Approve' | 'Reject' | 'Abstain';

/**
 * Recovery vote
 */
export interface RecoveryVote {
  request_id: string;
  trustee: string;
  vote: VoteDecision;
  comment?: string;
  voted_at: Timestamp;
}

/**
 * Input for setting up recovery
 */
export interface SetupRecoveryInput {
  did: string;
  trustees: string[];
  threshold: number;
  time_lock?: number;
}

/**
 * Input for initiating recovery
 */
export interface InitiateRecoveryInput {
  did: string;
  initiator_did: string;
  new_agent: AgentPubKey;
  reason: string;
}

/**
 * Input for voting on recovery
 */
export interface VoteOnRecoveryInput {
  request_id: string;
  trustee_did: string;
  vote: VoteDecision;
  comment?: string;
}

// =============================================================================
// CREDENTIAL SCHEMA TYPES
// =============================================================================

/**
 * Credential schema
 */
export interface CredentialSchema {
  id: string;
  name: string;
  description: string;
  version: string;
  author: string;
  schema: string;
  required_fields: string[];
  optional_fields: string[];
  credential_type: string;
  default_expiration?: number;
  revocable: boolean;
  active: boolean;
  created: Timestamp;
  updated: Timestamp;
}

/**
 * Schema endorsement
 */
export interface SchemaEndorsement {
  schema_id: string;
  endorser: string;
  trust_level: number;
  comment?: string;
  endorsed_at: Timestamp;
}

/**
 * Input for updating a schema
 */
export interface UpdateSchemaInput {
  schema_id: string;
  name?: string;
  description?: string;
  version?: string;
  schema?: string;
  required_fields?: string[];
  optional_fields?: string[];
  default_expiration?: number;
  active?: boolean;
}

/**
 * Input for endorsing a schema
 */
export interface EndorseSchemaInput {
  schema_id: string;
  endorser_did: string;
  trust_level: number;
  comment?: string;
}

// =============================================================================
// REVOCATION TYPES
// =============================================================================

/**
 * Revocation status
 */
export type RevocationStatus = 'Active' | 'Revoked' | 'Suspended';

/**
 * Revocation entry
 */
export interface RevocationEntry {
  credential_id: string;
  issuer: string;
  status: RevocationStatus;
  reason: string;
  effective_from: Timestamp;
  recorded_at: Timestamp;
  suspension_end?: Timestamp;
}

/**
 * Revocation check result
 */
export interface RevocationCheckResult {
  credential_id: string;
  status: RevocationStatus;
  reason?: string;
  checked_at: Timestamp;
}

/**
 * Input for revoking a credential
 */
export interface RevokeCredentialInput {
  credential_id: string;
  issuer_did: string;
  reason: string;
  effective_from?: Timestamp;
}

/**
 * Input for suspending a credential
 */
export interface SuspendCredentialInput {
  credential_id: string;
  issuer_did: string;
  reason: string;
  suspension_end: Timestamp;
}

/**
 * Input for reinstating a credential
 */
export interface ReinstateCredentialInput {
  credential_id: string;
  issuer_did: string;
  reason: string;
}

// =============================================================================
// TRUST CREDENTIAL TYPES
// =============================================================================

/**
 * Trust tier levels
 */
export type TrustTier =
  | 'Untrusted'
  | 'Minimal'
  | 'Low'
  | 'Moderate'
  | 'Standard'
  | 'High'
  | 'VeryHigh'
  | 'Trusted'
  | 'HighlyTrusted'
  | 'Exemplary';

/**
 * Trust score range
 */
export interface TrustScoreRange {
  lower: number;
  upper: number;
}

/**
 * Trust credential
 */
export interface TrustCredential {
  id: string;
  subject_did: string;
  issuer_did: string;
  kvector_commitment: number[];
  range_proof: number[];
  trust_score_range: TrustScoreRange;
  trust_tier: TrustTier;
  issued_at: Timestamp;
  expires_at?: Timestamp;
  revoked: boolean;
  revocation_reason?: string;
  supersedes?: string;
}

/**
 * Trust presentation for selective disclosure
 */
export interface TrustPresentation {
  id: string;
  credential_id: string;
  subject_did: string;
  disclosed_tier: TrustTier;
  disclosed_range?: TrustScoreRange;
  presentation_proof: number[];
  verifier_did?: string;
  purpose: string;
  presented_at: Timestamp;
  nonce: number[];
}

/**
 * K-Vector component
 */
export type KVectorComponent =
  | 'Reliability'
  | 'Competence'
  | 'Integrity'
  | 'Benevolence'
  | 'Openness';

/**
 * Attestation status
 */
export type AttestationStatus = 'Pending' | 'Fulfilled' | 'Rejected' | 'Expired';

/**
 * Attestation request
 */
export interface AttestationRequest {
  id: string;
  requester_did: string;
  subject_did: string;
  components: KVectorComponent[];
  min_trust_score?: number;
  min_tier?: TrustTier;
  purpose: string;
  expires_at: Timestamp;
  status: AttestationStatus;
  created_at: Timestamp;
}

/**
 * Input for issuing a trust credential
 */
export interface IssueTrustCredentialInput {
  subject_did: string;
  issuer_did: string;
  kvector_commitment: number[];
  range_proof: number[];
  trust_score_lower: number;
  trust_score_upper: number;
  expires_at?: Timestamp;
  supersedes?: string;
}

/**
 * Input for self-attesting trust
 */
export interface SelfAttestTrustInput {
  self_did: string;
  kvector_commitment: number[];
  range_proof: number[];
  trust_score_lower: number;
  trust_score_upper: number;
  expires_at?: Timestamp;
  supersedes?: string;
}

/**
 * Input for creating a trust presentation
 */
export interface CreateTrustPresentationInput {
  credential_id: string;
  subject_did: string;
  disclosed_tier: TrustTier;
  disclose_range: boolean;
  trust_range: TrustScoreRange;
  presentation_proof: number[];
  verifier_did?: string;
  purpose: string;
}

/**
 * Input for requesting attestation
 */
export interface RequestAttestationInput {
  requester_did: string;
  subject_did: string;
  components: KVectorComponent[];
  min_trust_score?: number;
  min_tier?: TrustTier;
  purpose: string;
  expires_at: Timestamp;
}

/**
 * Input for revoking a trust credential
 */
export interface RevokeTrustCredentialInput {
  credential_id: string;
  subject_did: string;
  reason: string;
}

/**
 * Trust credential verification result
 */
export interface TrustVerificationResult {
  credential_id: string;
  commitment_valid: boolean;
  tier_consistent: boolean;
  not_revoked: boolean;
  not_expired: boolean;
  proof_format_valid: boolean;
  message: string;
}

// =============================================================================
// BRIDGE TYPES
// =============================================================================

/**
 * hApp registration
 */
export interface HappRegistration {
  happ_id: string;
  happ_name: string;
  capabilities: string[];
  matl_score: number;
  registered_at: Timestamp;
}

/**
 * Identity query
 */
export interface IdentityQuery {
  id: string;
  did: string;
  source_happ: string;
  requested_fields: string[];
  queried_at: Timestamp;
}

/**
 * Identity verification
 */
export interface IdentityVerification {
  id: string;
  did: string;
  is_valid: boolean;
  is_deactivated: boolean;
  matl_score: number;
  credential_count: number;
  did_created?: Timestamp;
  verified_at: Timestamp;
}

/**
 * Identity verification result
 */
export interface IdentityVerificationResult {
  verification_hash: ActionHash;
  did: string;
  is_valid: boolean;
  matl_score: number;
  credential_count: number;
}

/**
 * Identity reputation
 */
export interface IdentityReputation {
  did: string;
  source_happ: string;
  score: number;
  interactions: number;
  last_updated: Timestamp;
}

/**
 * Aggregated reputation
 */
export interface AggregatedReputation {
  did: string;
  aggregate_score: number;
  sources: ReputationSource[];
  total_interactions: number;
}

/**
 * Reputation source
 */
export interface ReputationSource {
  source_happ: string;
  score: number;
  interactions: number;
}

/**
 * Bridge event types
 */
export type BridgeEventType =
  | 'IdentityCreated'
  | 'IdentityUpdated'
  | 'CredentialIssued'
  | 'CredentialRevoked'
  | 'TrustAttested'
  | 'RecoveryInitiated'
  | 'HappRegistered';

/**
 * Bridge event
 */
export interface BridgeEvent {
  id: string;
  event_type: BridgeEventType;
  subject: string;
  payload: string;
  source_happ: string;
  timestamp: Timestamp;
}

/**
 * Input for registering a hApp
 */
export interface RegisterHappInput {
  happ_id: string;
  happ_name: string;
  capabilities: string[];
}

/**
 * Input for querying identity
 */
export interface QueryIdentityInput {
  did: string;
  source_happ: string;
  requested_fields: string[];
}

/**
 * Input for reporting reputation
 */
export interface ReportReputationInput {
  did: string;
  source_happ: string;
  score: number;
  interactions: number;
}

/**
 * Input for broadcasting an event
 */
export interface BroadcastEventInput {
  event_type: BridgeEventType;
  subject: string;
  payload: string;
}

/**
 * Input for getting recent events
 */
export interface GetEventsInput {
  event_type?: BridgeEventType;
  since?: number;
  limit?: number;
}

/**
 * Input for trust check
 */
export interface TrustCheckInput {
  did: string;
  threshold: number;
}

// =============================================================================
// EDUCATION TYPES
// =============================================================================

/**
 * Degree type
 */
export type DegreeType =
  | 'HighSchool'
  | 'Associate'
  | 'Bachelor'
  | 'Master'
  | 'Doctorate'
  | 'Professional'
  | 'Certificate'
  | 'Diploma';

/**
 * DNSSEC status
 */
export type DnssecStatus = 'Secure' | 'Insecure' | 'Bogus' | 'Indeterminate';

/**
 * DNS DID record
 */
export interface DnsDid {
  domain: string;
  did: string;
  txt_record: string;
}

/**
 * DNS verification record
 */
export interface DnsVerificationRecord {
  timestamp: Timestamp;
  resolver: string;
  dnssec_status: DnssecStatus;
  ttl_seconds: number;
}

/**
 * Institutional issuer
 */
export interface InstitutionalIssuer {
  id: string;
  name: string;
  url?: string;
  image?: string;
  email?: string;
}

/**
 * Academic subject
 */
export interface AcademicSubject {
  id: string;
  name?: string;
  student_id?: string;
}

/**
 * Achievement metadata
 */
export interface AchievementMetadata {
  degree_type: DegreeType;
  degree_name: string;
  field_of_study: string;
  conferral_date: string;
  gpa?: number;
  honors?: string[];
  thesis_title?: string;
}

/**
 * Academic proof
 *
 * Wire format uses W3C VC Data Model camelCase field names.
 */
export interface AcademicProof {
  /** W3C: "type" */
  type: string;
  created: string;
  /** W3C: "verificationMethod" */
  verificationMethod: string;
  /** W3C: "proofPurpose" */
  proofPurpose: string;
  /** W3C: "proofValue" */
  proofValue: string;
  /** Multicodec algorithm ID (e.g. 0xED=Ed25519, 0x1206=ML-DSA-65, 0xF101=Hybrid) */
  algorithm?: number;
}

/**
 * Academic credential
 *
 * Wire format uses W3C VC Data Model camelCase field names.
 */
export interface AcademicCredential {
  /** JSON-LD context */
  '@context': string[];
  id: string;
  /** W3C: "type" */
  type: string[];
  issuer: InstitutionalIssuer;
  /** W3C: "validFrom" */
  validFrom: string;
  /** W3C: "validUntil" */
  validUntil?: string;
  /** W3C: "credentialSubject" */
  credentialSubject: AcademicSubject;
  proof: AcademicProof;
  zk_commitment: number[];
  commitment_nonce?: number[];
  revocation_registry_id: string;
  revocation_index: number;
  dns_did: DnsDid;
  achievement: AchievementMetadata;
  mycelix_schema_id: string;
  mycelix_created: Timestamp;
  legacy_import_ref?: string;
}

/**
 * Import status
 */
export type ImportStatus = 'InProgress' | 'Completed' | 'Failed' | 'PartiallyCompleted';

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
 * Legacy bridge import
 */
export interface LegacyBridgeImport {
  batch_id: string;
  institution_did: string;
  source_system: string;
  import_timestamp: Timestamp;
  total_credentials: number;
  imported_count: number;
  failed_count: number;
  status: ImportStatus;
  source_hash: number[];
  errors: ImportError[];
}

/**
 * Revocation reason
 */
export type RevocationReason =
  | 'FraudulentClaim'
  | 'DataError'
  | 'InstitutionalPolicy'
  | 'StudentRequest'
  | 'Other';

/**
 * Revocation request status
 */
export type RevocationRequestStatus = 'Pending' | 'Approved' | 'Rejected';

/**
 * Academic revocation request
 */
export interface AcademicRevocationRequest {
  credential_id: string;
  requester_did: string;
  reason: RevocationReason;
  explanation: string;
  evidence?: string[];
  requested_at: Timestamp;
  status: RevocationRequestStatus;
}

/**
 * Epistemic claim reference
 */
export interface EpistemicClaimReference {
  credential_id: string;
  credential_hash: ActionHash;
  claim_id: string;
  source_happ: string;
  subject: string;
  predicate: string;
  object: string;
  epistemic_e: number;
  epistemic_n: number;
  epistemic_m: number;
  published_at: Timestamp;
}

/**
 * Input for creating academic credential
 */
export interface CreateAcademicCredentialInput {
  issuer: InstitutionalIssuer;
  subject: AcademicSubject;
  achievement: AchievementMetadata;
  dns_did: DnsDid;
  revocation_registry_id: string;
  valid_from: string;
  valid_until?: string;
  proof: AcademicProof;
}

/**
 * Output from creating academic credential
 */
export interface CreateAcademicCredentialOutput {
  action_hash: ActionHash;
  credential_id: string;
  revocation_index: number;
}

/**
 * Input for starting legacy import
 */
export interface StartLegacyImportInput {
  institution_did: string;
  source_system: string;
  source_hash: number[];
  total_credentials: number;
}

/**
 * Input for requesting academic revocation
 */
export interface RequestAcademicRevocationInput {
  credential_id: string;
  reason: RevocationReason;
  explanation: string;
  evidence?: string[];
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/**
 * Identity SDK error codes
 */
export enum IdentitySdkErrorCode {
  // DID errors
  DID_NOT_FOUND = 'DID_NOT_FOUND',
  DID_ALREADY_EXISTS = 'DID_ALREADY_EXISTS',
  DID_DEACTIVATED = 'DID_DEACTIVATED',
  INVALID_DID_FORMAT = 'INVALID_DID_FORMAT',

  // Credential errors
  CREDENTIAL_NOT_FOUND = 'CREDENTIAL_NOT_FOUND',
  CREDENTIAL_EXPIRED = 'CREDENTIAL_EXPIRED',
  CREDENTIAL_REVOKED = 'CREDENTIAL_REVOKED',
  INVALID_CREDENTIAL = 'INVALID_CREDENTIAL',
  SCHEMA_NOT_FOUND = 'SCHEMA_NOT_FOUND',

  // Authorization errors
  UNAUTHORIZED = 'UNAUTHORIZED',
  NOT_ISSUER = 'NOT_ISSUER',
  NOT_HOLDER = 'NOT_HOLDER',
  NOT_TRUSTEE = 'NOT_TRUSTEE',

  // Recovery errors
  RECOVERY_NOT_FOUND = 'RECOVERY_NOT_FOUND',
  THRESHOLD_NOT_MET = 'THRESHOLD_NOT_MET',
  TIME_LOCK_ACTIVE = 'TIME_LOCK_ACTIVE',

  // Network errors
  CONNECTION_FAILED = 'CONNECTION_FAILED',
  ZOME_CALL_FAILED = 'ZOME_CALL_FAILED',
  TIMEOUT = 'TIMEOUT',

  // Validation errors
  INVALID_INPUT = 'INVALID_INPUT',
  VALIDATION_FAILED = 'VALIDATION_FAILED',

  // General errors
  UNKNOWN = 'UNKNOWN',
}

/**
 * Identity SDK error with structured information
 */
export class IdentitySdkError extends Error {
  constructor(
    public readonly code: IdentitySdkErrorCode,
    message: string,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'IdentitySdkError';
  }

  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      details: this.details,
    };
  }
}

// =============================================================================
// CLIENT CONFIGURATION
// =============================================================================

/**
 * Identity client configuration
 */
export interface IdentityClientConfig {
  /** Role name in the hApp */
  roleName?: string;
  /** Timeout for zome calls in milliseconds */
  timeout?: number;
  /** Enable debug logging */
  debug?: boolean;
}

/**
 * Default identity client configuration
 */
export const DEFAULT_IDENTITY_CLIENT_CONFIG: Required<IdentityClientConfig> = {
  roleName: 'identity',
  timeout: 30000,
  debug: false,
};
