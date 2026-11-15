#![cfg_attr(not(feature = "std"), no_std)]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request-bound balance proof types (tag = 2)
// ---------------------------------------------------------------------------

/// Versioned public request context for a request-bound balance proof.
///
/// All digests are references supplied by the verifier/request owner. This
/// type does not authenticate their issuers, verify a source signature, or
/// prove that the private balance opens the referenced source statement.
/// Those are separate qualification layers.
///
/// The guest commits to a domain-separated digest of every field so a receipt
/// cannot be substituted across request, audience, verifier, purpose, policy,
/// source-reference, challenge, or threshold contexts.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BalanceProofRequestV1 {
    /// Unique request identity within the verifier's request namespace.
    pub request_id: [u8; 16],
    /// Digest of the source statement to which the caller says this request
    /// relates. It is a reference only; this guest does not authenticate it.
    pub source_statement_reference_digest: [u8; 32],
    /// Digest of the claim schema, including field and unit semantics.
    pub claim_schema_digest: [u8; 32],
    /// Digest of the exact predicate semantics (operator, bounds, units,
    /// rounding/fixed-point rules and comparison boundary).
    pub predicate_digest: [u8; 32],
    /// Digest identifying the issuer/key reference selected by the caller.
    /// It is not proof that the issuer or key is authorized.
    pub issuer_key_reference_digest: [u8; 32],
    /// Pairwise or purpose-scoped audience identifier digest.
    pub audience_digest: [u8; 32],
    /// Digest identifying the verifier/request authority context.
    pub verifier_digest: [u8; 32],
    /// Digest of the declared purpose.
    pub purpose_digest: [u8; 32],
    /// Digest of the applicable policy/profile.
    pub policy_digest: [u8; 32],
    /// Fresh challenge supplied by the verifier-side request owner.
    /// This guest only binds it; it does not issue or consume challenges.
    pub verifier_challenge: [u8; 32],
    /// Public threshold tested against the private balance.
    pub required_minimum: u64,
}

/// Private witness plus the public request context for tag 2.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BalanceProofRequestBoundInputV1 {
    /// Private balance witness. It is not included in the committed output.
    pub balance: u64,
    /// Public request context whose exact digest is committed by the guest.
    pub request: BalanceProofRequestV1,
}

/// Public journal committed by a tag-2 guest execution.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BalanceProofRequestBoundOutputV1 {
    /// Whether the private balance meets the request threshold.
    pub sufficient: bool,
    /// Threshold applied by the guest.
    pub required_minimum: u64,
    /// Domain-separated digest of the exact request fields.
    pub request_digest: [u8; 32],
}

/// Typed errors for malformed request context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BalanceProofRequestValidationError {
    EmptyRequestId,
    EmptySourceStatementReferenceDigest,
    EmptyClaimSchemaDigest,
    EmptyPredicateDigest,
    EmptyIssuerKeyReferenceDigest,
    EmptyAudienceDigest,
    EmptyVerifierDigest,
    EmptyPurposeDigest,
    EmptyPolicyDigest,
    EmptyVerifierChallenge,
}

impl core::fmt::Display for BalanceProofRequestValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::EmptyRequestId => "request_id must not be all zero",
            Self::EmptySourceStatementReferenceDigest => {
                "source_statement_reference_digest must not be all zero"
            }
            Self::EmptyClaimSchemaDigest => "claim_schema_digest must not be all zero",
            Self::EmptyPredicateDigest => "predicate_digest must not be all zero",
            Self::EmptyIssuerKeyReferenceDigest => {
                "issuer_key_reference_digest must not be all zero"
            }
            Self::EmptyAudienceDigest => "audience_digest must not be all zero",
            Self::EmptyVerifierDigest => "verifier_digest must not be all zero",
            Self::EmptyPurposeDigest => "purpose_digest must not be all zero",
            Self::EmptyPolicyDigest => "policy_digest must not be all zero",
            Self::EmptyVerifierChallenge => "verifier_challenge must not be all zero",
        };
        f.write_str(message)
    }
}

impl BalanceProofRequestV1 {
    /// Validate required request-context fields.
    ///
    /// This is structural validation only. In particular, it does not prove
    /// that the challenge was issued by a verifier or that any referenced
    /// issuer/source/policy is trusted.
    pub fn validate(&self) -> Result<(), BalanceProofRequestValidationError> {
        if !contains_nonzero_byte(&self.request_id) {
            return Err(BalanceProofRequestValidationError::EmptyRequestId);
        }
        if !contains_nonzero_byte(&self.source_statement_reference_digest) {
            return Err(
                BalanceProofRequestValidationError::EmptySourceStatementReferenceDigest,
            );
        }
        if !contains_nonzero_byte(&self.claim_schema_digest) {
            return Err(BalanceProofRequestValidationError::EmptyClaimSchemaDigest);
        }
        if !contains_nonzero_byte(&self.predicate_digest) {
            return Err(BalanceProofRequestValidationError::EmptyPredicateDigest);
        }
        if !contains_nonzero_byte(&self.issuer_key_reference_digest) {
            return Err(BalanceProofRequestValidationError::EmptyIssuerKeyReferenceDigest);
        }
        if !contains_nonzero_byte(&self.audience_digest) {
            return Err(BalanceProofRequestValidationError::EmptyAudienceDigest);
        }
        if !contains_nonzero_byte(&self.verifier_digest) {
            return Err(BalanceProofRequestValidationError::EmptyVerifierDigest);
        }
        if !contains_nonzero_byte(&self.purpose_digest) {
            return Err(BalanceProofRequestValidationError::EmptyPurposeDigest);
        }
        if !contains_nonzero_byte(&self.policy_digest) {
            return Err(BalanceProofRequestValidationError::EmptyPolicyDigest);
        }
        if !contains_nonzero_byte(&self.verifier_challenge) {
            return Err(BalanceProofRequestValidationError::EmptyVerifierChallenge);
        }
        Ok(())
    }

    /// Compute the canonical, domain-separated request digest.
    ///
    /// Encoding is fixed-width and explicit: byte arrays are appended in the
    /// field order below and the threshold is big-endian. No Serde or JSON
    /// serialization bytes participate in this digest.
    pub fn canonical_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};

        const DOMAIN: &[u8] = b"symthaea-balance-proof-request-v1\0";

        let mut hasher = Sha256::new();
        hasher.update(DOMAIN);
        hasher.update(self.request_id);
        hasher.update(self.source_statement_reference_digest);
        hasher.update(self.claim_schema_digest);
        hasher.update(self.predicate_digest);
        hasher.update(self.issuer_key_reference_digest);
        hasher.update(self.audience_digest);
        hasher.update(self.verifier_digest);
        hasher.update(self.purpose_digest);
        hasher.update(self.policy_digest);
        hasher.update(self.verifier_challenge);
        hasher.update(self.required_minimum.to_be_bytes());
        hasher.finalize().into()
    }
}

impl BalanceProofRequestBoundOutputV1 {
    /// Check whether this public journal is bound to the exact expected request.
    pub fn matches_request(&self, expected: &BalanceProofRequestV1) -> bool {
        self.required_minimum == expected.required_minimum
            && self.request_digest == expected.canonical_digest()
    }
}

fn contains_nonzero_byte<const N: usize>(bytes: &[u8; N]) -> bool {
    bytes.iter().any(|byte| *byte != 0)
}

#[cfg(test)]
mod request_bound_tests {
    use super::*;

    fn sample_request() -> BalanceProofRequestV1 {
        BalanceProofRequestV1 {
            request_id: [1; 16],
            source_statement_reference_digest: [2; 32],
            claim_schema_digest: [4; 32],
            predicate_digest: [5; 32],
            issuer_key_reference_digest: [3; 32],
            audience_digest: [6; 32],
            verifier_digest: [7; 32],
            purpose_digest: [8; 32],
            policy_digest: [9; 32],
            verifier_challenge: [10; 32],
            required_minimum: 500,
        }
    }

    #[test]
    fn request_digest_is_deterministic() {
        let request = sample_request();
        assert_eq!(request.canonical_digest(), request.canonical_digest());
        assert_eq!(request.validate(), Ok(()));
    }

    #[test]
    fn request_digest_matches_fixed_known_answer() {
        let request = sample_request();
        assert_eq!(
            request.canonical_digest(),
            [
                0xf1, 0x73, 0x03, 0x7a, 0x11, 0x26, 0xd4, 0x95,
                0xe2, 0xb2, 0x28, 0xfa, 0x79, 0xd6, 0x13, 0x2c,
                0xee, 0x08, 0x3d, 0x97, 0x61, 0x34, 0x98, 0x32,
                0x2b, 0x5a, 0x7d, 0x2f, 0x3f, 0xda, 0x53, 0x8d,
            ]
        );
    }

    #[test]
    fn request_validation_rejects_each_missing_context_reference() {
        let original = sample_request();

        let mut changed = original.clone();
        changed.source_statement_reference_digest = [0; 32];
        assert_eq!(changed.validate(), Err(BalanceProofRequestValidationError::EmptySourceStatementReferenceDigest));

        let mut changed = original.clone();
        changed.claim_schema_digest = [0; 32];
        assert_eq!(changed.validate(), Err(BalanceProofRequestValidationError::EmptyClaimSchemaDigest));

        let mut changed = original.clone();
        changed.predicate_digest = [0; 32];
        assert_eq!(changed.validate(), Err(BalanceProofRequestValidationError::EmptyPredicateDigest));

        let mut changed = original.clone();
        changed.issuer_key_reference_digest = [0; 32];
        assert_eq!(changed.validate(), Err(BalanceProofRequestValidationError::EmptyIssuerKeyReferenceDigest));

        let mut changed = original.clone();
        changed.audience_digest = [0; 32];
        assert_eq!(changed.validate(), Err(BalanceProofRequestValidationError::EmptyAudienceDigest));

        let mut changed = original.clone();
        changed.verifier_digest = [0; 32];
        assert_eq!(changed.validate(), Err(BalanceProofRequestValidationError::EmptyVerifierDigest));

        let mut changed = original.clone();
        changed.purpose_digest = [0; 32];
        assert_eq!(changed.validate(), Err(BalanceProofRequestValidationError::EmptyPurposeDigest));

        let mut changed = original;
        changed.policy_digest = [0; 32];
        assert_eq!(changed.validate(), Err(BalanceProofRequestValidationError::EmptyPolicyDigest));
    }

    #[test]
    fn request_digest_binds_every_context_field() {
        let original = sample_request();
        let original_digest = original.canonical_digest();

        let mut changed = original.clone();
        changed.request_id[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.source_statement_reference_digest[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.claim_schema_digest[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.predicate_digest[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.issuer_key_reference_digest[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.audience_digest[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.verifier_digest[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.purpose_digest[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.policy_digest[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original.clone();
        changed.verifier_challenge[0] ^= 1;
        assert_ne!(changed.canonical_digest(), original_digest);

        let mut changed = original;
        changed.required_minimum += 1;
        assert_ne!(changed.canonical_digest(), original_digest);
    }

    #[test]
    fn request_validation_rejects_zero_request_id() {
        let mut request = sample_request();
        request.request_id = [0; 16];
        assert_eq!(
            request.validate(),
            Err(BalanceProofRequestValidationError::EmptyRequestId)
        );
    }

    #[test]
    fn request_validation_rejects_zero_challenge() {
        let mut request = sample_request();
        request.verifier_challenge = [0; 32];
        assert_eq!(
            request.validate(),
            Err(BalanceProofRequestValidationError::EmptyVerifierChallenge)
        );
    }

    #[test]
    fn output_matches_only_exact_request_digest_and_threshold() {
        let request = sample_request();
        let output = BalanceProofRequestBoundOutputV1 {
            sufficient: true,
            required_minimum: request.required_minimum,
            request_digest: request.canonical_digest(),
        };
        assert!(output.matches_request(&request));

        let mut other = request.clone();
        other.audience_digest[0] ^= 1;
        assert!(!output.matches_request(&other));
    }
}
