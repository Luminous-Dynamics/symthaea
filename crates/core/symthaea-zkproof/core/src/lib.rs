#![cfg_attr(not(feature = "std"), no_std)]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

extern crate alloc;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// Data passed from host to guest (zkVM)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvolutionInput {
    pub episodes: Vec<Vec<f32>>, // HDC vectors (e.g., 1024D)
    pub tau_scale: f32,
    pub threshold: f32,
}

/// Data committed by the guest as public output
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvolutionOutput {
    pub average_phi: f32,
    pub tau_scale: f32,
    pub episode_count: u32,
}

// ---------------------------------------------------------------------------
// Balance proof types (tag = 1)
// ---------------------------------------------------------------------------

/// Private input for a ZK balance sufficiency proof.
///
/// The `balance` field is PRIVATE — it is consumed inside the zkVM guest
/// but never appears in the committed journal output.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BalanceProofInput {
    /// Actual balance (PRIVATE — never revealed in the proof output).
    pub balance: u64,
    /// Minimum balance required (will be committed as public output).
    pub required_minimum: u64,
    /// Caller-chosen nonce to prevent proof replay (public output).
    pub nonce: u64,
}

/// Public output committed by the guest for a balance proof.
///
/// The actual balance is NOT included — only the boolean result,
/// the threshold, and the nonce are revealed.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BalanceProofOutput {
    /// Whether `balance >= required_minimum`.
    pub sufficient: bool,
    /// The minimum that was checked against (public).
    pub required_minimum: u64,
    /// Replay-prevention nonce (public).
    pub nonce: u64,
}


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

        const DOMAIN: &[u8] = b"symthaea-balance-proof-request-v1\\0";

        let mut hasher = Sha256::new();
        hasher.update(DOMAIN);
        hasher.update(self.request_id);
        hasher.update(self.source_statement_reference_digest);
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
            issuer_key_reference_digest: [3; 32],
            audience_digest: [4; 32],
            verifier_digest: [5; 32],
            purpose_digest: [6; 32],
            policy_digest: [7; 32],
            verifier_challenge: [8; 32],
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
