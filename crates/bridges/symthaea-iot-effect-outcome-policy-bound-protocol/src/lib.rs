// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical policy-bound reconciliation challenge protocol.
//!
//! This crate deliberately adds **context**, not authority. It wraps one already-issued V1
//! reconciliation challenge with the exact outcome-policy generation/digest that a remote or
//! firmware outcome verifier is expected to answer under. The wrapper has a distinct V2 wire
//! representation and digest domain; V1 bytes and semantics remain unchanged.
//!
//! The expected policy identity is caller-supplied because challenge issuance is non-authorizing.
//! A malicious or stale caller can therefore only cause later verification to fail: the owner-local
//! outcome guard must require exact equality among this advertised identity, signed V2 evidence,
//! current guard policy, and the held reconciliation-trust publication before journal closure.

#![deny(unsafe_code)]

use symthaea_authority::Digest32;
use symthaea_iot_actuation_effect_reconciliation_challenge::{
    EffectReconciliationChallengeV1, EffectReconciliationChallengeValidationError,
    ReconciliationSourceStateV1,
};
use thiserror::Error;

pub const POLICY_BOUND_RECONCILIATION_CHALLENGE_SCHEMA_VERSION: u16 = 2;
pub const POLICY_BOUND_RECONCILIATION_CHALLENGE_WIRE_MAGIC: &[u8] =
    b"SYMTHAEA-IOT-RECON-CHALLENGE-V2-POLICY-BOUND\0";

const POLICY_BOUND_CHALLENGE_DOMAIN: &[u8] =
    b"symthaea-iot-effect-reconciliation-challenge-policy-bound-v2\0";

/// Non-authorizing policy identity advertised to a remote/device-class outcome verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedOutcomePolicyIdentityV2 {
    generation: u64,
    digest: Digest32,
}

impl ExpectedOutcomePolicyIdentityV2 {
    pub fn new(generation: u64, digest: Digest32) -> Result<Self, PolicyBoundChallengeV2Error> {
        let identity = Self { generation, digest };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(self) -> Result<(), PolicyBoundChallengeV2Error> {
        if self.generation == 0 {
            return Err(PolicyBoundChallengeV2Error::ExpectedPolicyGenerationZero);
        }
        if self.digest == Digest32([0; 32]) {
            return Err(PolicyBoundChallengeV2Error::ExpectedPolicyDigestZero);
        }
        Ok(())
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn digest(self) -> Digest32 {
        self.digest
    }
}

/// V2 challenge that binds one exact V1 unresolved-attempt challenge to one expected outcome policy.
///
/// The embedded V1 challenge is intentionally preserved byte-for-byte at its own wire layer. V2
/// canonical bytes length-prefix those exact V1 bytes and append the expected policy identity under
/// a distinct V2 magic/domain. This avoids maintaining a second copy of the causal field encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyBoundEffectReconciliationChallengeV2 {
    schema_version: u16,
    base: EffectReconciliationChallengeV1,
    expected_policy: ExpectedOutcomePolicyIdentityV2,
}

impl PolicyBoundEffectReconciliationChallengeV2 {
    pub fn new(
        base: EffectReconciliationChallengeV1,
        expected_policy: ExpectedOutcomePolicyIdentityV2,
    ) -> Result<Self, PolicyBoundChallengeV2Error> {
        let challenge = Self {
            schema_version: POLICY_BOUND_RECONCILIATION_CHALLENGE_SCHEMA_VERSION,
            base,
            expected_policy,
        };
        challenge.validate()?;
        Ok(challenge)
    }

    pub fn validate(&self) -> Result<(), PolicyBoundChallengeV2Error> {
        if self.schema_version != POLICY_BOUND_RECONCILIATION_CHALLENGE_SCHEMA_VERSION {
            return Err(PolicyBoundChallengeV2Error::UnsupportedSchema);
        }
        self.base.validate()?;
        self.expected_policy.validate()?;
        Ok(())
    }

    /// Exact language-neutral V2 bytes.
    ///
    /// Layout:
    ///
    /// ```text
    /// magic
    /// schema_version:u16 BE
    /// base_v1_len:u64 BE
    /// base_v1_canonical_bytes:[u8;base_v1_len]
    /// expected_policy_generation:u64 BE
    /// expected_policy_digest:[u8;32]
    /// ```
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PolicyBoundChallengeV2Error> {
        self.validate()?;
        let base = self.base.canonical_bytes()?;
        let mut out = Vec::with_capacity(
            POLICY_BOUND_RECONCILIATION_CHALLENGE_WIRE_MAGIC.len()
                + 2
                + 8
                + base.len()
                + 8
                + 32,
        );
        out.extend_from_slice(POLICY_BOUND_RECONCILIATION_CHALLENGE_WIRE_MAGIC);
        out.extend_from_slice(&self.schema_version.to_be_bytes());
        out.extend_from_slice(&(base.len() as u64).to_be_bytes());
        out.extend_from_slice(&base);
        out.extend_from_slice(&self.expected_policy.generation.to_be_bytes());
        out.extend_from_slice(&self.expected_policy.digest.0);
        Ok(out)
    }

    pub fn digest(&self) -> Result<Digest32, PolicyBoundChallengeV2Error> {
        let bytes = self.canonical_bytes()?;
        let mut h = blake3::Hasher::new();
        h.update(POLICY_BOUND_CHALLENGE_DOMAIN);
        h.update(&(bytes.len() as u64).to_be_bytes());
        h.update(&bytes);
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub const fn base(&self) -> &EffectReconciliationChallengeV1 {
        &self.base
    }

    pub const fn expected_policy(&self) -> ExpectedOutcomePolicyIdentityV2 {
        self.expected_policy
    }

    pub fn is_fresh_at(&self, now_unix_ms: u64) -> bool {
        self.base.is_fresh_at(now_unix_ms)
    }

    pub const fn journal_generation(&self) -> u64 {
        self.base.journal_generation()
    }

    pub const fn journal_digest(&self) -> Digest32 {
        self.base.journal_digest()
    }

    pub const fn correlation_digest(&self) -> Digest32 {
        self.base.correlation_digest()
    }

    pub const fn command_digest(&self) -> Digest32 {
        self.base.command_digest()
    }

    pub fn device(&self) -> &symthaea_authority::ResourceRef {
        self.base.device()
    }

    pub fn operation(&self) -> &symthaea_authority::Operation {
        self.base.operation()
    }

    pub fn executor(&self) -> &symthaea_authority::PrincipalId {
        self.base.executor()
    }

    pub const fn sequence(&self) -> u64 {
        self.base.sequence()
    }

    pub const fn source_state(&self) -> ReconciliationSourceStateV1 {
        self.base.source_state()
    }

    pub const fn attempt_common_fenced_at_unix_ms(&self) -> u64 {
        self.base.attempt_common_fenced_at_unix_ms()
    }

    pub const fn attempt_wall_valid_until_unix_ms(&self) -> u64 {
        self.base.attempt_wall_valid_until_unix_ms()
    }

    pub const fn issued_at_unix_ms(&self) -> u64 {
        self.base.issued_at_unix_ms()
    }

    pub const fn expires_at_unix_ms(&self) -> u64 {
        self.base.expires_at_unix_ms()
    }
}

#[derive(Debug, Error)]
pub enum PolicyBoundChallengeV2Error {
    #[error("unsupported policy-bound reconciliation challenge schema")]
    UnsupportedSchema,
    #[error("expected outcome-policy generation is zero")]
    ExpectedPolicyGenerationZero,
    #[error("expected outcome-policy digest is zero")]
    ExpectedPolicyDigestZero,
    #[error("embedded V1 reconciliation challenge is invalid: {0}")]
    BaseChallenge(#[from] EffectReconciliationChallengeValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_policy_identity_rejects_zero_components() {
        assert!(matches!(
            ExpectedOutcomePolicyIdentityV2::new(0, Digest32([1; 32])),
            Err(PolicyBoundChallengeV2Error::ExpectedPolicyGenerationZero)
        ));
        assert!(matches!(
            ExpectedOutcomePolicyIdentityV2::new(1, Digest32([0; 32])),
            Err(PolicyBoundChallengeV2Error::ExpectedPolicyDigestZero)
        ));
    }
}
