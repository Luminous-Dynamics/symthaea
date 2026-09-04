// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Affine locally-generated challenges for Xenia witness-currentness checks.
//!
//! The Xenia wire protocol continues to carry a raw 32-byte challenge. This
//! module hardens the Symthaea integration boundary by generating that value
//! from the operating system CSPRNG, retaining the source/witness scope beside
//! it, and consuming the pending challenge when currentness verification runs.
//!
//! Loss of a pending challenge across process restart is intentionally
//! fail-closed: the old response is abandoned and a new challenge must be
//! issued. V0.1 does not persist outstanding challenges and does not claim
//! cross-process replay prevention for callers that bypass this typed API.

use core::fmt;

use getrandom::getrandom;
use symthaea_authority_time::VerifiedAuthorityTime;
use thiserror::Error;

use crate::{
    VerifiedXeniaWitnessFrontierV1, XeniaSignedWitnessFrontierAnchorV1,
    XeniaSignedWitnessFrontierObservationV1, XeniaWitnessFrontierExpectationV1,
    XeniaWitnessFrontierFreshnessPolicyV1, XeniaWitnessFrontierVerificationError,
    derive_xenia_witness_frontier_source_id, verify_xenia_witness_frontier_v1,
    xenia_witness_frontier_time_subject_digest_v1,
};

const ZERO16: [u8; 16] = [0; 16];
const ZERO32: [u8; 32] = [0; 32];

/// Stable non-challenge bindings for one Xenia witness-currentness request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XeniaWitnessCurrentnessScopeV1 {
    /// Exact trusted Xenia ledger public key.
    pub trusted_ledger_public_key: [u8; 32],
    /// Reviewed Xenia witness-source epoch.
    pub source_epoch: u64,
    /// Exact reviewed Xenia witness-anchor policy commitment.
    pub anchor_policy_digest: [u8; 32],
    /// Exact local qualification-witness identity being queried.
    pub witness_id: [u8; 16],
}

impl XeniaWitnessCurrentnessScopeV1 {
    fn validate(self) -> Result<(), XeniaWitnessCurrentnessChallengeError> {
        if self.trusted_ledger_public_key == ZERO32
            || self.source_epoch == 0
            || self.anchor_policy_digest == ZERO32
            || self.witness_id == ZERO16
        {
            return Err(XeniaWitnessCurrentnessChallengeError::InvalidScope);
        }
        derive_xenia_witness_frontier_source_id(
            self.trusted_ledger_public_key,
            self.anchor_policy_digest,
        )
        .map_err(|_| XeniaWitnessCurrentnessChallengeError::InvalidScope)?;
        Ok(())
    }
}

/// One locally generated, process-local Xenia currentness challenge.
///
/// This type intentionally does not implement `Clone` or `Copy`. The normal
/// production flow exposes the raw challenge bytes only so they can be sent to
/// Xenia, derives the authority-time subject from this exact object, and then
/// consumes the object during verification.
pub struct PendingXeniaWitnessCurrentnessChallengeV1 {
    scope: XeniaWitnessCurrentnessScopeV1,
    challenge: [u8; 32],
}

impl fmt::Debug for PendingXeniaWitnessCurrentnessChallengeV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingXeniaWitnessCurrentnessChallengeV1")
            .field("scope", &self.scope)
            .field("challenge_present", &true)
            .finish()
    }
}

impl PendingXeniaWitnessCurrentnessChallengeV1 {
    /// Generate a fresh challenge from the operating-system CSPRNG.
    pub fn generate(
        scope: XeniaWitnessCurrentnessScopeV1,
    ) -> Result<Self, XeniaWitnessCurrentnessChallengeError> {
        scope.validate()?;
        let mut challenge = [0u8; 32];
        getrandom(&mut challenge)
            .map_err(|_| XeniaWitnessCurrentnessChallengeError::EntropyUnavailable)?;
        Self::from_challenge(scope, challenge)
    }

    /// Raw challenge bytes that must be echoed by the signed Xenia observation.
    ///
    /// The bytes are public protocol material, not a secret. Returning them does
    /// not make this pending object cloneable and does not create a second typed
    /// verification permit.
    pub fn challenge(&self) -> [u8; 32] {
        self.challenge
    }

    /// Stable source/witness bindings retained beside this pending challenge.
    pub fn scope(&self) -> XeniaWitnessCurrentnessScopeV1 {
        self.scope
    }

    /// Build the exact subject a `VerifiedAuthorityTime` fact must bind for this
    /// pending challenge and signed durable anchor.
    pub fn authority_time_subject_digest(
        &self,
        anchor: &XeniaSignedWitnessFrontierAnchorV1,
        freshness: XeniaWitnessFrontierFreshnessPolicyV1,
    ) -> Result<[u8; 32], XeniaWitnessCurrentnessChallengeError> {
        xenia_witness_frontier_time_subject_digest_v1(anchor, self.expectation(), freshness)
            .map_err(XeniaWitnessCurrentnessChallengeError::Verification)
    }

    /// Consume this exact pending challenge while independently verifying the
    /// durable Xenia anchor and fresh signed observation.
    ///
    /// A returned `VerifiedXeniaWitnessFrontierV1` is chronology evidence only;
    /// consuming this challenge does not mint capability or retry authority.
    pub fn verify(
        self,
        anchor: &XeniaSignedWitnessFrontierAnchorV1,
        observation: &XeniaSignedWitnessFrontierObservationV1,
        authority_time: &VerifiedAuthorityTime,
        freshness: XeniaWitnessFrontierFreshnessPolicyV1,
    ) -> Result<VerifiedXeniaWitnessFrontierV1, XeniaWitnessCurrentnessChallengeError> {
        verify_xenia_witness_frontier_v1(
            anchor,
            observation,
            self.expectation(),
            authority_time,
            freshness,
        )
        .map_err(XeniaWitnessCurrentnessChallengeError::Verification)
    }

    fn from_challenge(
        scope: XeniaWitnessCurrentnessScopeV1,
        challenge: [u8; 32],
    ) -> Result<Self, XeniaWitnessCurrentnessChallengeError> {
        scope.validate()?;
        if challenge == ZERO32 {
            return Err(XeniaWitnessCurrentnessChallengeError::EntropyReturnedZero);
        }
        Ok(Self { scope, challenge })
    }

    fn expectation(&self) -> XeniaWitnessFrontierExpectationV1 {
        XeniaWitnessFrontierExpectationV1 {
            trusted_ledger_public_key: self.scope.trusted_ledger_public_key,
            source_epoch: self.scope.source_epoch,
            anchor_policy_digest: self.scope.anchor_policy_digest,
            witness_id: self.scope.witness_id,
            challenge: self.challenge,
        }
    }
}

/// Fail-closed challenge generation/verification errors.
#[derive(Debug, Error)]
pub enum XeniaWitnessCurrentnessChallengeError {
    /// Source/key/policy/witness scope contains zero or unsupported structure.
    #[error("Xenia witness-currentness scope is invalid")]
    InvalidScope,
    /// The operating-system CSPRNG could not provide challenge bytes.
    #[error("operating-system entropy unavailable for Xenia witness-currentness challenge")]
    EntropyUnavailable,
    /// The CSPRNG returned the all-zero value, which V1 reserves as malformed.
    #[error("operating-system entropy returned the reserved all-zero challenge")]
    EntropyReturnedZero,
    /// The underlying independent Xenia evidence/currentness verification failed.
    #[error("Xenia witness-currentness verification failed: {0}")]
    Verification(#[from] XeniaWitnessFrontierVerificationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> XeniaWitnessCurrentnessScopeV1 {
        XeniaWitnessCurrentnessScopeV1 {
            trusted_ledger_public_key: [0x11; 32],
            source_epoch: 3,
            anchor_policy_digest: [0x22; 32],
            witness_id: [0x33; 16],
        }
    }

    #[test]
    fn deterministic_fixture_retains_exact_scope_and_challenge() {
        let pending = PendingXeniaWitnessCurrentnessChallengeV1::from_challenge(
            scope(),
            [0x44; 32],
        )
        .unwrap();
        assert_eq!(pending.scope(), scope());
        assert_eq!(pending.challenge(), [0x44; 32]);
        let expected = pending.expectation();
        assert_eq!(expected.trusted_ledger_public_key, [0x11; 32]);
        assert_eq!(expected.source_epoch, 3);
        assert_eq!(expected.anchor_policy_digest, [0x22; 32]);
        assert_eq!(expected.witness_id, [0x33; 16]);
        assert_eq!(expected.challenge, [0x44; 32]);
    }

    #[test]
    fn all_zero_challenge_is_never_constructible() {
        assert!(matches!(
            PendingXeniaWitnessCurrentnessChallengeV1::from_challenge(scope(), ZERO32),
            Err(XeniaWitnessCurrentnessChallengeError::EntropyReturnedZero)
        ));
    }

    #[test]
    fn malformed_scope_is_rejected_before_entropy_use() {
        let mut invalid = scope();
        invalid.witness_id = ZERO16;
        assert!(matches!(
            PendingXeniaWitnessCurrentnessChallengeV1::generate(invalid),
            Err(XeniaWitnessCurrentnessChallengeError::InvalidScope)
        ));
    }

    #[test]
    fn production_generator_returns_nonzero_process_local_challenges() {
        let first = PendingXeniaWitnessCurrentnessChallengeV1::generate(scope()).unwrap();
        let second = PendingXeniaWitnessCurrentnessChallengeV1::generate(scope()).unwrap();
        assert_ne!(first.challenge(), ZERO32);
        assert_ne!(second.challenge(), ZERO32);
        assert_ne!(first.challenge(), second.challenge());
    }
}
