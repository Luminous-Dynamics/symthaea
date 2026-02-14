/**
 * Identity types (DID, VerifiableCredential, etc.)
 *
 * Auto-generated TypeScript types from Rust SDK.
 * DO NOT EDIT MANUALLY - regenerate with: pnpm generate:types
 *
 * @module @mycelix/sdk/generated/identity
 * @generated
 */

/**
 * Decentralized Identifier
 * @generated
 */
export interface DID {
  /** The DID string (did:mycelix:...) */
  id: string;
  /** Controller agent key */
  controller: string;
  /** Creation timestamp */
  created: number;
  /** Last update timestamp */
  updated: number;
}

/**
 * W3C Verifiable Credential
 * @generated
 */
export interface VerifiableCredential {
  /** JSON-LD context */
  context: string[];
  /** Credential ID */
  id: string;
  /** Credential types */
  type: string[];
  /** Issuer DID */
  issuer: string;
  /** ISO 8601 issuance date */
  issuanceDate: string;
  /** ISO 8601 expiration date */
  expirationDate?: string;
  /** Credential claims */
  credentialSubject: CredentialSubject;
  /** Cryptographic proof */
  proof?: CredentialProof;
}

/**
 * Subject of a verifiable credential
 * @generated
 */
export interface CredentialSubject {
  /** Subject DID */
  id: string;
}

/**
 * Cryptographic proof for a credential
 * @generated
 */
export interface CredentialProof {
  /** Proof type */
  type: string;
  /** Creation timestamp */
  created: string;
  /** Verification method ID */
  verificationMethod: string;
  /** Purpose of proof */
  proofPurpose: string;
  /** Proof signature value */
  proofValue: string;
}

/**
 * Public key verification method
 * @generated
 */
export interface VerificationMethod {
  /** Method ID */
  id: string;
  /** Key type */
  type: string;
  /** Controller DID */
  controller: string;
  /** Multibase-encoded public key */
  publicKeyMultibase: string;
}

/**
 * DID service endpoint
 * @generated
 */
export interface ServiceEndpoint {
  /** Service ID */
  id: string;
  /** Service type */
  type: string;
  /** Endpoint URL */
  serviceEndpoint: string;
}
