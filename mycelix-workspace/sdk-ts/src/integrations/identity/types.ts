/**
 * @mycelix/sdk Identity Types
 *
 * TypeScript mirrors of Rust structs from mycelix-identity zomes.
 * These types ensure client-side type safety before data hits the network.
 */

import type { ActionHash, AgentPubKey, Timestamp } from '@holochain/client';

// Re-export for convenience
export type { ActionHash, AgentPubKey, Timestamp };

// ============================================================================
// DID REGISTRY TYPES
// ============================================================================

/**
 * Verification method for cryptographic operations
 * Mirrors: VerificationMethod in did_registry/integrity/src/lib.rs
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
 * Mirrors: ServiceEndpoint in did_registry/integrity/src/lib.rs
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
 * Mirrors: DidDocument in did_registry/integrity/src/lib.rs
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
 * Mirrors: DidDeactivation in did_registry/integrity/src/lib.rs
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
  original_action_hash: ActionHash;
  updates: {
    verificationMethod?: VerificationMethod[];
    authentication?: string[];
    service?: ServiceEndpoint[];
  };
}

// ============================================================================
// VERIFIABLE CREDENTIAL TYPES
// ============================================================================

/**
 * Credential issuer - can be DID string or object with id
 * Mirrors: CredentialIssuer in verifiable_credential/integrity/src/lib.rs
 */
export type CredentialIssuer =
  | string
  | {
      id: string;
      name?: string;
      type?: string[];
    };

/**
 * Credential subject containing the claims
 * Mirrors: CredentialSubject in verifiable_credential/integrity/src/lib.rs
 */
export interface CredentialSubject {
  id: string;
  [key: string]: unknown;
}

/**
 * Reference to credential schema
 */
export interface CredentialSchemaRef {
  id: string;
  type: string;
}

/**
 * Credential status for revocation
 */
export interface CredentialStatus {
  id: string;
  type: string;
  statusPurpose?: string;
  statusListIndex?: string;
  statusListCredential?: string;
}

/**
 * Cryptographic proof
 * Mirrors: CredentialProof in verifiable_credential/integrity/src/lib.rs
 */
export interface CredentialProof {
  /** Proof type (Ed25519Signature2020, DataIntegrityProof, etc.) */
  type: string;
  /** When the proof was created (ISO 8601) */
  created: string;
  /** Verification method used (DID URL) */
  verificationMethod: string;
  /** Purpose of the proof */
  proofPurpose: string;
  /** The actual signature/proof value (multibase encoded) */
  proofValue: string;
  /** For DataIntegrityProof: cryptosuite used */
  cryptosuite?: string;
}

/**
 * W3C Verifiable Credential
 * Mirrors: VerifiableCredential in verifiable_credential/integrity/src/lib.rs
 */
export interface VerifiableCredential {
  /** JSON-LD context */
  '@context': string[];
  /** Unique credential identifier */
  id: string;
  /** Credential types (must include "VerifiableCredential") */
  type: string[];
  /** DID of the issuer */
  issuer: CredentialIssuer;
  /** Issuance date (ISO 8601) */
  validFrom: string;
  /** Expiration date (optional, ISO 8601) */
  validUntil?: string;
  /** The claims being made */
  credentialSubject: CredentialSubject;
  /** Schema reference */
  credentialSchema?: CredentialSchemaRef;
  /** Credential status (for revocation checking) */
  credentialStatus?: CredentialStatus;
  /** Cryptographic proof */
  proof: CredentialProof;
  /** Mycelix-specific: schema ID used */
  mycelix_schema_id: string;
  /** Mycelix-specific: creation timestamp */
  mycelix_created: Timestamp;
}

/**
 * Verifiable Presentation
 * Mirrors: VerifiablePresentation in verifiable_credential/integrity/src/lib.rs
 */
export interface VerifiablePresentation {
  '@context': string[];
  id: string;
  type: string[];
  holder: string;
  verifiableCredential: VerifiableCredential[];
  proof: CredentialProof;
  mycelix_created: Timestamp;
}

/**
 * Evidence supporting a credential request
 */
export interface CredentialEvidence {
  type: string;
  id: string;
  description?: string;
}

/**
 * Status of credential request
 */
export type RequestStatus = 'Pending' | 'UnderReview' | 'Approved' | 'Rejected' | 'Issued';

/**
 * Credential issuance request
 * Mirrors: CredentialRequest in verifiable_credential/integrity/src/lib.rs
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
  is_valid: boolean;
  errors: string[];
  issuer_verified: boolean;
  signature_valid: boolean;
  not_expired: boolean;
  not_revoked: boolean;
  schema_valid: boolean;
}

/**
 * Credential status response
 */
export interface CredentialStatusResponse {
  credential_id: string;
  is_revoked: boolean;
  is_expired: boolean;
  revocation_reason?: string;
  revoked_at?: Timestamp;
}

// ============================================================================
// INPUT TYPES FOR ZOME CALLS
// ============================================================================

/**
 * Input for issuing a credential
 */
export interface IssueCredentialInput {
  subject_did: string;
  schema_id: string;
  claims: Record<string, unknown>;
  expiration_days?: number;
}

/**
 * Input for creating a presentation
 */
export interface CreatePresentationInput {
  credential_ids: string[];
  verifier_did?: string;
  challenge?: string;
}

/**
 * Input for creating a derived credential (selective disclosure)
 */
export interface CreateDerivedInput {
  original_credential_id: string;
  selected_claims: string[];
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
  reason?: string;
}

// ============================================================================
// RECORD WRAPPERS
// ============================================================================

/**
 * Record wrapper for DID documents
 */
export interface DidRecord {
  hash: ActionHash;
  document: DidDocument;
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

// ============================================================================
// ERROR TYPES
// ============================================================================

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

// ============================================================================
// SDK CONFIGURATION
// ============================================================================

/**
 * Identity SDK configuration options
 */
export interface IdentityConfig {
  /** Role name in the hApp */
  roleName?: string;
  /** Enable debug logging */
  debug?: boolean;
  /** Retry configuration */
  retry?: {
    maxAttempts: number;
    delayMs: number;
    backoffMultiplier: number;
  };
}

/**
 * Default configuration values
 */
export const DEFAULT_IDENTITY_CONFIG: Required<IdentityConfig> = {
  roleName: 'identity',
  debug: false,
  retry: {
    maxAttempts: 3,
    delayMs: 1000,
    backoffMultiplier: 2,
  },
};
