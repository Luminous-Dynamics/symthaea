// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Identity Client - Type Definitions
 *
 * Shared types for identity operations across all Mycelix applications.
 */

/**
 * Assurance levels following eIDAS-inspired tiers:
 * - E0: Unverified (self-attested only)
 * - E1: Email verified
 * - E2: Phone verified + social recovery configured
 * - E3: Government ID verified (KYC)
 * - E4: Biometric + multi-factor (high-value transactions)
 */
export type AssuranceLevel = 'E0' | 'E1' | 'E2' | 'E3' | 'E4';

/**
 * Numeric mapping for assurance levels (for comparisons)
 */
export const ASSURANCE_LEVEL_VALUE: Record<AssuranceLevel, number> = {
  E0: 0,
  E1: 1,
  E2: 2,
  E3: 3,
  E4: 4,
};

/**
 * DID Document structure (W3C DID Core compatible)
 */
export interface DidDocument {
  /** DID identifier (did:mycelix:...) */
  id: string;
  /** Controller agent pub key */
  controller: string;
  /** Verification methods (keys) */
  verificationMethod: VerificationMethod[];
  /** Authentication methods */
  authentication: string[];
  /** Service endpoints */
  service: ServiceEndpoint[];
  /** Creation timestamp */
  created: number;
  /** Last update timestamp */
  updated: number;
  /** Document version */
  version: number;
}

/**
 * Verification method (key) in a DID document
 */
export interface VerificationMethod {
  /** Method ID (e.g., did:mycelix:xyz#keys-1) */
  id: string;
  /** Key type */
  type: string;
  /** Controller DID */
  controller: string;
  /** Public key in multibase format */
  publicKeyMultibase: string;
}

/**
 * Service endpoint in a DID document
 */
export interface ServiceEndpoint {
  /** Service ID */
  id: string;
  /** Service type (e.g., "LinkedDomains", "CredentialStore") */
  type: string;
  /** Service endpoint URL or other identifier */
  serviceEndpoint: string;
}

/**
 * Result of DID resolution
 */
export interface DidResolutionResult {
  /** Whether resolution was successful */
  success: boolean;
  /** The resolved DID document (if found) */
  didDocument: DidDocument | null;
  /** Error message if resolution failed */
  error?: string;
  /** Whether this is from cache */
  cached?: boolean;
  /** Resolution metadata */
  metadata?: {
    /** Time taken to resolve (ms) */
    resolveTime: number;
    /** Source of resolution (local, dht, fallback) */
    source: 'local' | 'dht' | 'fallback';
  };
}

/**
 * Verifiable Credential structure (W3C VC compatible)
 */
export interface VerifiableCredential {
  /** Credential context */
  '@context': string[];
  /** Credential ID */
  id: string;
  /** Credential types */
  type: string[];
  /** Issuer DID */
  issuer: string;
  /** Issuance date */
  issuanceDate: string;
  /** Expiration date (optional) */
  expirationDate?: string;
  /** Credential subject (the claims) */
  credentialSubject: Record<string, any>;
  /** Credential status (for revocation checking) */
  credentialStatus?: CredentialStatus;
  /** Proof/signature */
  proof?: CredentialProof;
}

/**
 * Credential status for revocation checking
 */
export interface CredentialStatus {
  /** Status ID */
  id: string;
  /** Status type (e.g., "RevocationList2020Status") */
  type: string;
  /** Revocation list index */
  revocationListIndex?: string;
  /** Revocation list credential */
  revocationListCredential?: string;
}

/**
 * Credential proof/signature
 */
export interface CredentialProof {
  /** Proof type */
  type: string;
  /** Creation timestamp */
  created: string;
  /** Verification method used */
  verificationMethod: string;
  /** Proof purpose */
  proofPurpose: string;
  /** Proof value (signature) */
  proofValue: string;
}

/**
 * Result of credential verification
 */
export interface CredentialVerificationResult {
  /** Whether verification was successful */
  valid: boolean;
  /** Verification checks performed */
  checks: {
    /** Signature is valid */
    signatureValid: boolean;
    /** Credential has not expired */
    notExpired: boolean;
    /** Credential is not revoked */
    notRevoked: boolean;
    /** Issuer is trusted */
    issuerTrusted: boolean;
    /** Schema is valid */
    schemaValid: boolean;
  };
  /** Error message if verification failed */
  error?: string;
  /** The verified credential (if valid) */
  credential?: VerifiableCredential;
  /** Assurance level derived from the credential */
  assuranceLevel?: AssuranceLevel;
}

/**
 * Revocation status check result
 */
export interface RevocationStatus {
  /** Credential ID that was checked */
  credentialId: string;
  /** Current status */
  status: 'active' | 'revoked' | 'suspended';
  /** Reason for revocation/suspension (if applicable) */
  reason?: string;
  /** When the status was checked */
  checkedAt: number;
}

/**
 * Identity verification request (for cross-hApp queries)
 */
export interface IdentityVerificationRequest {
  /** DID to verify */
  did: string;
  /** Minimum assurance level required */
  minAssuranceLevel?: AssuranceLevel;
  /** Specific credentials to check */
  requiredCredentials?: string[];
  /** Requesting hApp identifier */
  sourceHapp: string;
}

/**
 * Identity verification response
 */
export interface IdentityVerificationResponse {
  /** Verification ID */
  id: string;
  /** DID that was verified */
  did: string;
  /** Whether the DID exists and is active */
  isValid: boolean;
  /** Whether the DID is deactivated */
  isDeactivated: boolean;
  /** Current assurance level */
  assuranceLevel: AssuranceLevel;
  /** MATL reputation score (0.0 - 1.0) */
  matlScore: number;
  /** Total credentials held by this DID */
  credentialCount: number;
  /** When the DID was created */
  didCreated?: number;
  /** Verification timestamp */
  verifiedAt: number;
}

/**
 * Cross-hApp reputation data
 */
export interface CrossHappReputation {
  /** DID this reputation is for */
  did: string;
  /** Reputation scores from various hApps */
  scores: HappReputationScore[];
  /** Aggregated overall score */
  aggregateScore: number;
  /** Last update timestamp */
  lastUpdated: number;
}

/**
 * Reputation score from a single hApp
 */
export interface HappReputationScore {
  /** hApp identifier */
  happId: string;
  /** hApp name */
  happName: string;
  /** Reputation score (0.0 - 1.0) */
  score: number;
  /** Number of interactions in this hApp */
  interactions: number;
  /** When this score was last updated */
  updatedAt: number;
}

/**
 * Options for identity client initialization
 */
export interface IdentityClientOptions {
  /** Holochain app client */
  appClient: any;
  /** Role name for identity DNA (default: 'identity') */
  roleName?: string;
  /** Enable caching (default: true) */
  enableCache?: boolean;
  /** Cache TTL in milliseconds (default: 5 minutes) */
  cacheTtl?: number;
  /** Fallback mode if identity hApp is unavailable */
  fallbackMode?: 'warn' | 'error' | 'silent';
}

/**
 * Configuration for high-value transaction verification
 */
export interface HighValueTransactionConfig {
  /** Minimum value threshold for enhanced verification */
  threshold: number;
  /** Currency for threshold */
  currency: string;
  /** Required assurance level */
  requiredAssuranceLevel: AssuranceLevel;
  /** Whether to require specific credentials */
  requiredCredentials?: string[];
}
