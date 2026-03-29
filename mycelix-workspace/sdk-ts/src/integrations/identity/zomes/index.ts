// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Identity Zome Clients
 *
 * Exports all zome clients for the mycelix-identity hApp.
 */

export { DidRegistryClient } from './did-registry.js';
export { VerifiableCredentialClient } from './verifiable-credential.js';
export { MfaClient } from './mfa.js';
export type {
  FactorType,
  FactorCategory,
  AssuranceLevel,
  EnrolledFactor,
  MfaState,
  EnrollFactorInput,
  FactorProof,
  GuardianAttestation,
  VerificationChallenge,
  MfaVerificationResult,
  FlEligibilityResult,
  FlRequirement,
  FactorEnrollment,
} from './mfa.js';

// Future zome clients (to be implemented):
// export { CredentialSchemaClient } from './credential-schema.js';
// export { RevocationClient } from './revocation.js';
// export { RecoveryClient } from './recovery.js';
// export { TrustCredentialClient } from './trust-credential.js';
// export { IdentityBridgeClient } from './bridge.js';
