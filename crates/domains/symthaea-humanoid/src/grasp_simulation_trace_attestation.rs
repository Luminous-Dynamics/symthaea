// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authentication boundary for sealed Grasp simulation causal traces.
//!
//! This module proves one deliberately narrow proposition: an identified producer
//! authenticated one exact, structurally valid sealed causal-trace digest under an
//! exact key/scheme/revocation policy. It does **not** grant motor authority, assert
//! that the trace belongs to a particular external qualification subject beyond
//! what the sealed trace itself commits internally, or prove that an external
//! simulator process physically executed the recorded transition. Those stronger
//! correspondences remain separate structural/promotion evidence.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::grasp_simulation_controller::causal_trace::HumanoidGraspSimulationCausalTrace;

pub const HUMANOID_GRASP_SIMULATION_TRACE_CLAIM_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_SIMULATION_TRACE_VERIFICATION_SCHEMA_VERSION: u32 = 1;
const MAX_AUTHENTICATION_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidGraspSimulationTraceAuthentication {
    scheme_id: String,
    signature: Vec<u8>,
}

impl HumanoidGraspSimulationTraceAuthentication {
    pub fn new(scheme_id: impl Into<String>, signature: Vec<u8>) -> Option<Self> {
        let value = Self {
            scheme_id: scheme_id.into(),
            signature,
        };
        value.validate().then_some(value)
    }

    pub fn scheme_id(&self) -> &str {
        &self.scheme_id
    }

    pub fn signature(&self) -> &[u8] {
        &self.signature
    }

    fn validate(&self) -> bool {
        valid_id(&self.scheme_id)
            && !self.signature.is_empty()
            && self.signature.len() <= MAX_AUTHENTICATION_BYTES
    }
}

/// Canonical producer claim over one exact sealed simulation trace.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspSimulationTraceClaim {
    schema_version: u32,
    trace_digest: HumanoidEvidenceDigest,
    producer_id: String,
    producer_artifact_digest: HumanoidEvidenceDigest,
    key_id: String,
    revocation_epoch: u64,
    attested_at_unix_millis: u64,
    valid_until_unix_millis: u64,
    authentication: HumanoidGraspSimulationTraceAuthentication,
    statement_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspSimulationTraceClaim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trace_digest: HumanoidEvidenceDigest,
        producer_id: impl Into<String>,
        producer_artifact_digest: HumanoidEvidenceDigest,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        attested_at_unix_millis: u64,
        valid_until_unix_millis: u64,
        authentication: HumanoidGraspSimulationTraceAuthentication,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_GRASP_SIMULATION_TRACE_CLAIM_SCHEMA_VERSION,
            trace_digest,
            producer_id: producer_id.into(),
            producer_artifact_digest,
            key_id: key_id.into(),
            revocation_epoch,
            attested_at_unix_millis,
            valid_until_unix_millis,
            authentication,
            statement_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.base_valid() {
            return None;
        }
        value.statement_digest = digest_claim_statement(&value);
        value.validate().then_some(value)
    }

    fn base_valid(&self) -> bool {
        self.schema_version == HUMANOID_GRASP_SIMULATION_TRACE_CLAIM_SCHEMA_VERSION
            && !self.trace_digest.is_zero()
            && valid_id(&self.producer_id)
            && !self.producer_artifact_digest.is_zero()
            && valid_id(&self.key_id)
            && self.attested_at_unix_millis > 0
            && self.valid_until_unix_millis >= self.attested_at_unix_millis
            && self.authentication.validate()
    }

    pub fn validate(&self) -> bool {
        self.base_valid()
            && !self.statement_digest.is_zero()
            && self.statement_digest == digest_claim_statement(self)
    }

    pub const fn trace_digest(&self) -> HumanoidEvidenceDigest {
        self.trace_digest
    }

    pub fn producer_id(&self) -> &str {
        &self.producer_id
    }

    pub const fn producer_artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.producer_artifact_digest
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub const fn revocation_epoch(&self) -> u64 {
        self.revocation_epoch
    }

    pub const fn attested_at_unix_millis(&self) -> u64 {
        self.attested_at_unix_millis
    }

    pub const fn valid_until_unix_millis(&self) -> u64 {
        self.valid_until_unix_millis
    }

    pub fn authentication(&self) -> &HumanoidGraspSimulationTraceAuthentication {
        &self.authentication
    }

    pub const fn statement_digest(&self) -> HumanoidEvidenceDigest {
        self.statement_digest
    }
}

/// Two-phase signing helper. The exact statement digest is fixed before any
/// external signature bytes are attached.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidUnsignedGraspSimulationTraceClaim {
    trace_digest: HumanoidEvidenceDigest,
    producer_id: String,
    producer_artifact_digest: HumanoidEvidenceDigest,
    key_id: String,
    revocation_epoch: u64,
    attested_at_unix_millis: u64,
    valid_until_unix_millis: u64,
    scheme_id: String,
    statement_digest: HumanoidEvidenceDigest,
}

impl HumanoidUnsignedGraspSimulationTraceClaim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trace_digest: HumanoidEvidenceDigest,
        producer_id: impl Into<String>,
        producer_artifact_digest: HumanoidEvidenceDigest,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        attested_at_unix_millis: u64,
        valid_until_unix_millis: u64,
        scheme_id: impl Into<String>,
    ) -> Option<Self> {
        let producer_id = producer_id.into();
        let key_id = key_id.into();
        let scheme_id = scheme_id.into();
        let placeholder = HumanoidGraspSimulationTraceAuthentication::new(
            scheme_id.clone(),
            vec![0x00],
        )?;
        let preview = HumanoidGraspSimulationTraceClaim::new(
            trace_digest,
            producer_id.clone(),
            producer_artifact_digest,
            key_id.clone(),
            revocation_epoch,
            attested_at_unix_millis,
            valid_until_unix_millis,
            placeholder,
        )?;
        let statement_digest = preview.statement_digest();
        (!statement_digest.is_zero()).then_some(Self {
            trace_digest,
            producer_id,
            producer_artifact_digest,
            key_id,
            revocation_epoch,
            attested_at_unix_millis,
            valid_until_unix_millis,
            scheme_id,
            statement_digest,
        })
    }

    pub const fn trace_digest(&self) -> HumanoidEvidenceDigest {
        self.trace_digest
    }

    pub const fn statement_digest(&self) -> HumanoidEvidenceDigest {
        self.statement_digest
    }

    pub fn attach_signature(self, signature: Vec<u8>) -> Option<HumanoidGraspSimulationTraceClaim> {
        let authentication = HumanoidGraspSimulationTraceAuthentication::new(
            self.scheme_id.clone(),
            signature,
        )?;
        let claim = HumanoidGraspSimulationTraceClaim::new(
            self.trace_digest,
            self.producer_id,
            self.producer_artifact_digest,
            self.key_id,
            self.revocation_epoch,
            self.attested_at_unix_millis,
            self.valid_until_unix_millis,
            authentication,
        )?;
        (claim.statement_digest() == self.statement_digest).then_some(claim)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanoidGraspSimulationTraceVerificationWindow {
    pub valid_until_unix_millis: u64,
    pub revocation_epoch: u64,
}

impl HumanoidGraspSimulationTraceVerificationWindow {
    fn validate_for(
        &self,
        claim: &HumanoidGraspSimulationTraceClaim,
        now_unix_millis: u64,
    ) -> bool {
        now_unix_millis > 0
            && self.valid_until_unix_millis >= now_unix_millis
            && self.valid_until_unix_millis <= claim.valid_until_unix_millis
            && self.revocation_epoch >= claim.revocation_epoch
    }
}

/// External trust root for simulation-trace producers/executors.
pub trait HumanoidGraspSimulationTraceVerifier {
    /// Exact accepted producer/key/scheme/trust-root/revocation policy identity.
    fn verifier_digest(&self) -> HumanoidEvidenceDigest;

    fn verify(
        &self,
        claim: &HumanoidGraspSimulationTraceClaim,
        now_unix_millis: u64,
    ) -> Option<HumanoidGraspSimulationTraceVerificationWindow>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspSimulationTraceVerificationFailure {
    InvalidTime,
    InvalidTrace,
    InvalidClaim,
    TraceDigestMismatch,
    InvalidVerifierIdentity,
    VerificationRejected,
    InvalidVerificationWindow,
    InvalidVerifiedEvidence,
}

/// Opaque record of one accepted trace-producer verification decision.
pub struct HumanoidVerifiedGraspSimulationTrace {
    schema_version: u32,
    trace_digest: HumanoidEvidenceDigest,
    statement_digest: HumanoidEvidenceDigest,
    producer_id: String,
    producer_artifact_digest: HumanoidEvidenceDigest,
    key_id: String,
    scheme_id: String,
    verifier_digest: HumanoidEvidenceDigest,
    revocation_epoch: u64,
    verification_valid_until_unix_millis: u64,
    verification_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidVerifiedGraspSimulationTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidVerifiedGraspSimulationTrace")
            .field("trace_digest", &self.trace_digest)
            .field("producer_id", &self.producer_id)
            .field("producer_artifact_digest", &self.producer_artifact_digest)
            .field("key_id", &self.key_id)
            .field("scheme_id", &self.scheme_id)
            .field("verifier_digest", &self.verifier_digest)
            .field("revocation_epoch", &self.revocation_epoch)
            .field("verification_digest", &self.verification_digest)
            .finish()
    }
}

impl HumanoidVerifiedGraspSimulationTrace {
    pub const fn trace_digest(&self) -> HumanoidEvidenceDigest {
        self.trace_digest
    }

    pub fn producer_id(&self) -> &str {
        &self.producer_id
    }

    pub const fn producer_artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.producer_artifact_digest
    }

    pub const fn verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.verifier_digest
    }

    pub const fn verification_digest(&self) -> HumanoidEvidenceDigest {
        self.verification_digest
    }

    pub const fn valid_until_unix_millis(&self) -> u64 {
        self.verification_valid_until_unix_millis
    }

    pub fn validate_for(
        &self,
        trace: &HumanoidGraspSimulationCausalTrace,
        now_unix_millis: u64,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_SIMULATION_TRACE_VERIFICATION_SCHEMA_VERSION
            && trace.validate()
            && self.trace_digest == trace.trace_digest()
            && !self.trace_digest.is_zero()
            && !self.statement_digest.is_zero()
            && valid_id(&self.producer_id)
            && !self.producer_artifact_digest.is_zero()
            && valid_id(&self.key_id)
            && valid_id(&self.scheme_id)
            && !self.verifier_digest.is_zero()
            && now_unix_millis > 0
            && self.verification_valid_until_unix_millis >= now_unix_millis
            && !self.verification_digest.is_zero()
            && self.verification_digest == digest_verification(self)
    }
}

pub fn verify_humanoid_grasp_simulation_trace_producer(
    trace: &HumanoidGraspSimulationCausalTrace,
    claim: &HumanoidGraspSimulationTraceClaim,
    verifier: &dyn HumanoidGraspSimulationTraceVerifier,
    now_unix_millis: u64,
) -> Result<HumanoidVerifiedGraspSimulationTrace, HumanoidGraspSimulationTraceVerificationFailure> {
    if now_unix_millis == 0 {
        return Err(HumanoidGraspSimulationTraceVerificationFailure::InvalidTime);
    }
    if !trace.validate() {
        return Err(HumanoidGraspSimulationTraceVerificationFailure::InvalidTrace);
    }
    if !claim.validate()
        || now_unix_millis < claim.attested_at_unix_millis
        || now_unix_millis > claim.valid_until_unix_millis
    {
        return Err(HumanoidGraspSimulationTraceVerificationFailure::InvalidClaim);
    }
    if claim.trace_digest != trace.trace_digest() {
        return Err(HumanoidGraspSimulationTraceVerificationFailure::TraceDigestMismatch);
    }

    // Capture verifier identity once. Never re-read it after the external verify
    // call; a stateful verifier must not be able to change trust-root identity
    // between decision and evidence binding.
    let verifier_digest = verifier.verifier_digest();
    if verifier_digest.is_zero() {
        return Err(HumanoidGraspSimulationTraceVerificationFailure::InvalidVerifierIdentity);
    }
    let window = verifier
        .verify(claim, now_unix_millis)
        .ok_or(HumanoidGraspSimulationTraceVerificationFailure::VerificationRejected)?;
    if !window.validate_for(claim, now_unix_millis) {
        return Err(HumanoidGraspSimulationTraceVerificationFailure::InvalidVerificationWindow);
    }

    let mut value = HumanoidVerifiedGraspSimulationTrace {
        schema_version: HUMANOID_GRASP_SIMULATION_TRACE_VERIFICATION_SCHEMA_VERSION,
        trace_digest: trace.trace_digest(),
        statement_digest: claim.statement_digest(),
        producer_id: claim.producer_id.clone(),
        producer_artifact_digest: claim.producer_artifact_digest,
        key_id: claim.key_id.clone(),
        scheme_id: claim.authentication.scheme_id.clone(),
        verifier_digest,
        revocation_epoch: window.revocation_epoch,
        verification_valid_until_unix_millis: window.valid_until_unix_millis,
        verification_digest: HumanoidEvidenceDigest::ZERO,
    };
    value.verification_digest = digest_verification(&value);
    value
        .validate_for(trace, now_unix_millis)
        .then_some(value)
        .ok_or(HumanoidGraspSimulationTraceVerificationFailure::InvalidVerifiedEvidence)
}

fn digest_claim_statement(value: &HumanoidGraspSimulationTraceClaim) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-trace-claim.v1");
    h.u32(value.schema_version)
        .digest(value.trace_digest)
        .string(&value.producer_id)
        .digest(value.producer_artifact_digest)
        .string(&value.key_id)
        .u64(value.revocation_epoch)
        .u64(value.attested_at_unix_millis)
        .u64(value.valid_until_unix_millis)
        .string(&value.authentication.scheme_id);
    h.finish()
}

fn digest_verification(value: &HumanoidVerifiedGraspSimulationTrace) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-trace-verification.v1");
    h.u32(value.schema_version)
        .digest(value.trace_digest)
        .digest(value.statement_digest)
        .string(&value.producer_id)
        .digest(value.producer_artifact_digest)
        .string(&value.key_id)
        .string(&value.scheme_id)
        .digest(value.verifier_digest)
        .u64(value.revocation_epoch)
        .u64(value.verification_valid_until_unix_millis);
    h.finish()
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> HumanoidEvidenceDigest {
        HumanoidEvidenceDigest::from_bytes([byte; 32])
    }

    #[test]
    fn two_phase_signing_preserves_exact_statement_digest() {
        let unsigned = HumanoidUnsignedGraspSimulationTraceClaim::new(
            digest(1),
            "sim-executor-a",
            digest(2),
            "key-a",
            7,
            1_000,
            2_000,
            "ml-dsa-87",
        )
        .unwrap();
        let statement = unsigned.statement_digest();
        let claim = unsigned.attach_signature(vec![0xA5; 64]).unwrap();
        assert_eq!(claim.statement_digest(), statement);
        assert_eq!(claim.trace_digest(), digest(1));
        assert!(claim.validate());
    }

    #[test]
    fn authentication_scheme_is_part_of_signed_statement() {
        let a = HumanoidUnsignedGraspSimulationTraceClaim::new(
            digest(1),
            "sim-executor-a",
            digest(2),
            "key-a",
            7,
            1_000,
            2_000,
            "ml-dsa-87",
        )
        .unwrap();
        let b = HumanoidUnsignedGraspSimulationTraceClaim::new(
            digest(1),
            "sim-executor-a",
            digest(2),
            "key-a",
            7,
            1_000,
            2_000,
            "ed25519",
        )
        .unwrap();
        assert_ne!(a.statement_digest(), b.statement_digest());
    }

    #[test]
    fn trace_and_producer_identity_change_the_statement() {
        let a = HumanoidUnsignedGraspSimulationTraceClaim::new(
            digest(1),
            "sim-executor-a",
            digest(2),
            "key-a",
            7,
            1_000,
            2_000,
            "scheme-a",
        )
        .unwrap();
        let different_trace = HumanoidUnsignedGraspSimulationTraceClaim::new(
            digest(3),
            "sim-executor-a",
            digest(2),
            "key-a",
            7,
            1_000,
            2_000,
            "scheme-a",
        )
        .unwrap();
        let different_producer = HumanoidUnsignedGraspSimulationTraceClaim::new(
            digest(1),
            "sim-executor-b",
            digest(2),
            "key-a",
            7,
            1_000,
            2_000,
            "scheme-a",
        )
        .unwrap();
        assert_ne!(a.statement_digest(), different_trace.statement_digest());
        assert_ne!(a.statement_digest(), different_producer.statement_digest());
    }

    #[test]
    fn malformed_authentication_and_time_windows_fail_closed() {
        assert!(HumanoidGraspSimulationTraceAuthentication::new("", vec![1]).is_none());
        assert!(HumanoidGraspSimulationTraceAuthentication::new("scheme", Vec::new()).is_none());
        assert!(HumanoidUnsignedGraspSimulationTraceClaim::new(
            digest(1),
            "producer",
            digest(2),
            "key",
            0,
            2_000,
            1_000,
            "scheme",
        )
        .is_none());
    }
}
