/**
 * Mycelix Identity Client
 *
 * Shared identity library for all Mycelix applications.
 * Provides DID resolution, credential verification, and assurance level checks.
 *
 * @packageDocumentation
 */

// Main client
export { IdentityClient } from './client';

// Types
export type {
  // Core types
  AssuranceLevel,
  DidDocument,
  DidResolutionResult,
  VerificationMethod,
  ServiceEndpoint,

  // Credentials
  VerifiableCredential,
  CredentialVerificationResult,
  CredentialStatus,
  CredentialProof,
  RevocationStatus,

  // Identity verification
  IdentityVerificationRequest,
  IdentityVerificationResponse,

  // Reputation
  CrossHappReputation,
  HappReputationScore,

  // Configuration
  IdentityClientOptions,
  HighValueTransactionConfig,
} from './types';

// Constants
export { ASSURANCE_LEVEL_VALUE } from './types';

// Utility functions
export {
  compareAssuranceLevels,
  isValidDID,
  parseDID,
  formatAssuranceLevel,
} from './utils';
