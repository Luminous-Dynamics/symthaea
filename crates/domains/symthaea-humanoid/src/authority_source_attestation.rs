// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport-neutral authenticated authority-source attestations.
//!
//! The humanoid domain should not implement Xenia/Mycelix key management, signature
//! policy, or revocation itself. It does, however, need a non-bypassable boundary
//! between arbitrary source strings/scales and authority evidence accepted by the
//! final motor path. This module defines that boundary.
//!
//! An attestation contains a canonical SHA-256 statement plus opaque authentication
//! bytes. A caller-selected verifier is the explicit trust root. Successful
//! verification returns a non-constructible `HumanoidVerifiedAuthoritySource` whose
//! lifetime is bounded by both the source claim and the verifier's current
//! revocation/freshness window.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::qualification::HumanoidQualificationSubject;
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_AUTHORITY_SOURCE_ATTESTATION_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_AUTHORITY_VERIFICATION_SCHEMA_VERSION: u32 = 1;
const MAX_AUTHENTICATION_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HumanoidAuthoritySourceKind {
    Operator,
    Qualification,
    Physical,
    Epistemic,
    Cognitive,
}

/// Authentication material is intentionally scheme-neutral. For example, Xenia
/// may choose an ML-DSA/hybrid scheme while a HIL lab may use a hardware-backed
/// institutional signer. The verifier interprets these bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidAuthorityAuthentication {
    scheme_id: String,
    signature: Vec<u8>,
}

impl HumanoidAuthorityAuthentication {
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

/// Canonical source statement to be authenticated by an upstream authority
/// service. `evidence_digest` must identify the actual source-specific evidence
/// artifact, not merely a display label.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidAuthoritySourceAttestation {
    schema_version: u32,
    source_kind: HumanoidAuthoritySourceKind,
    subject_digest: HumanoidEvidenceDigest,
    evidence_digest: HumanoidEvidenceDigest,
    scale: f32,
    evaluated_at_s: f64,
    valid_until_s: f64,
    issuer_id: String,
    key_id: String,
    revocation_epoch: u64,
    statement_digest: HumanoidEvidenceDigest,
    authentication: HumanoidAuthorityAuthentication,
}

impl HumanoidAuthoritySourceAttestation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        source_kind: HumanoidAuthoritySourceKind,
        evidence_digest: HumanoidEvidenceDigest,
        scale: f32,
        evaluated_at_s: f64,
        valid_until_s: f64,
        issuer_id: impl Into<String>,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        authentication: HumanoidAuthorityAuthentication,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_AUTHORITY_SOURCE_ATTESTATION_SCHEMA_VERSION,
            source_kind,
            subject_digest: digest_subject(subject)?,
            evidence_digest,
            scale,
            evaluated_at_s,
            valid_until_s,
            issuer_id: issuer_id.into(),
            key_id: key_id.into(),
            revocation_epoch,
            statement_digest: HumanoidEvidenceDigest::ZERO,
            authentication,
        };
        if !value.validate_shape(subject) {
            return None;
        }
        value.statement_digest = digest_statement(&value);
        value.validate_for(subject).then_some(value)
    }

    pub const fn source_kind(&self) -> HumanoidAuthoritySourceKind {
        self.source_kind
    }

    pub const fn evidence_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_digest
    }

    pub const fn statement_digest(&self) -> HumanoidEvidenceDigest {
        self.statement_digest
    }

    pub const fn scale(&self) -> f32 {
        self.scale
    }

    pub const fn evaluated_at_s(&self) -> f64 {
        self.evaluated_at_s
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.valid_until_s
    }

    pub fn issuer_id(&self) -> &str {
        &self.issuer_id
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub const fn revocation_epoch(&self) -> u64 {
        self.revocation_epoch
    }

    pub fn authentication(&self) -> &HumanoidAuthorityAuthentication {
        &self.authentication
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.validate_shape(subject)
            && !self.statement_digest.is_zero()
            && self.statement_digest == digest_statement(self)
    }

    fn validate_shape(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_AUTHORITY_SOURCE_ATTESTATION_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && !self.evidence_digest.is_zero()
            && self.scale.is_finite()
            && (0.0..=1.0).contains(&self.scale)
            && self.evaluated_at_s.is_finite()
            && self.evaluated_at_s >= 0.0
            && self.valid_until_s.is_finite()
            && self.valid_until_s >= self.evaluated_at_s
            && valid_id(&self.issuer_id)
            && valid_id(&self.key_id)
            && self.authentication.validate()
    }
}

/// Freshness/revocation window returned by the trusted verifier after it has
/// authenticated the statement. This value cannot extend the source claim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumanoidAuthorityVerificationWindow {
    pub valid_until_s: f64,
    pub revocation_epoch: u64,
}

impl HumanoidAuthorityVerificationWindow {
    fn validate_for(&self, attestation: &HumanoidAuthoritySourceAttestation, now_s: f64) -> bool {
        self.valid_until_s.is_finite()
            && self.valid_until_s >= now_s
            && self.valid_until_s <= attestation.valid_until_s
            && self.revocation_epoch >= attestation.revocation_epoch
    }
}

/// Trust-root interface implemented by Xenia/Mycelix, a hardware trust service,
/// or another application-selected verifier.
///
/// The verifier must authenticate `attestation.statement_digest()` against the
/// opaque authentication material, enforce issuer/key policy and revocation, and
/// return a bounded validity window for its current verification decision.
pub trait HumanoidAuthorityAttestationVerifier {
    /// Cryptographic identity of the exact verifier policy/keyset/configuration.
    fn verifier_digest(&self) -> HumanoidEvidenceDigest;

    fn verify(
        &self,
        attestation: &HumanoidAuthoritySourceAttestation,
        now_s: f64,
    ) -> Option<HumanoidAuthorityVerificationWindow>;
}

/// Opaque proof that one exact source attestation was accepted by one explicit
/// verifier trust root. External callers cannot construct this type directly.
pub struct HumanoidVerifiedAuthoritySource {
    source_kind: HumanoidAuthoritySourceKind,
    subject_digest: HumanoidEvidenceDigest,
    statement_digest: HumanoidEvidenceDigest,
    evidence_digest: HumanoidEvidenceDigest,
    verifier_digest: HumanoidEvidenceDigest,
    verification_digest: HumanoidEvidenceDigest,
    revocation_epoch: u64,
    snapshot: HumanoidAuthoritySourceSnapshot,
}

impl std::fmt::Debug for HumanoidVerifiedAuthoritySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidVerifiedAuthoritySource")
            .field("source_kind", &self.source_kind)
            .field("statement_digest", &self.statement_digest)
            .field("verifier_digest", &self.verifier_digest)
            .field("verification_digest", &self.verification_digest)
            .field("revocation_epoch", &self.revocation_epoch)
            .field("valid_until_s", &self.snapshot.valid_until_s)
            .finish()
    }
}

impl HumanoidVerifiedAuthoritySource {
    pub const fn source_kind(&self) -> HumanoidAuthoritySourceKind {
        self.source_kind
    }

    pub const fn statement_digest(&self) -> HumanoidEvidenceDigest {
        self.statement_digest
    }

    pub const fn evidence_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_digest
    }

    pub const fn verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.verifier_digest
    }

    pub const fn verification_digest(&self) -> HumanoidEvidenceDigest {
        self.verification_digest
    }

    pub const fn scale(&self) -> f32 {
        self.snapshot.scale
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.snapshot.valid_until_s
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        expected_kind: HumanoidAuthoritySourceKind,
        now_s: f64,
    ) -> bool {
        self.source_kind == expected_kind
            && digest_subject(subject) == Some(self.subject_digest)
            && !self.statement_digest.is_zero()
            && !self.evidence_digest.is_zero()
            && !self.verifier_digest.is_zero()
            && !self.verification_digest.is_zero()
            && self.snapshot.validate_at(now_s)
            && self.verification_digest == digest_verification(self)
    }

    pub(crate) fn source_snapshot(&self) -> HumanoidAuthoritySourceSnapshot {
        self.snapshot.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidAuthorityAttestationVerificationFailure {
    InvalidTime,
    InvalidAttestation,
    WrongSourceKind,
    InvalidVerifierIdentity,
    VerificationRejected,
    InvalidVerificationWindow,
    InvalidVerifiedSource,
}

/// Authenticate one source claim and return the opaque evidence object accepted by
/// the final embodied-authority path.
pub fn verify_humanoid_authority_source(
    subject: &HumanoidQualificationSubject,
    expected_kind: HumanoidAuthoritySourceKind,
    attestation: &HumanoidAuthoritySourceAttestation,
    verifier: &dyn HumanoidAuthorityAttestationVerifier,
    now_s: f64,
) -> Result<HumanoidVerifiedAuthoritySource, HumanoidAuthorityAttestationVerificationFailure> {
    if !now_s.is_finite() || now_s < 0.0 {
        return Err(HumanoidAuthorityAttestationVerificationFailure::InvalidTime);
    }
    if !attestation.validate_for(subject) {
        return Err(HumanoidAuthorityAttestationVerificationFailure::InvalidAttestation);
    }
    if attestation.source_kind != expected_kind {
        return Err(HumanoidAuthorityAttestationVerificationFailure::WrongSourceKind);
    }
    if now_s < attestation.evaluated_at_s || now_s > attestation.valid_until_s {
        return Err(HumanoidAuthorityAttestationVerificationFailure::InvalidAttestation);
    }
    let verifier_digest = verifier.verifier_digest();
    if verifier_digest.is_zero() {
        return Err(HumanoidAuthorityAttestationVerificationFailure::InvalidVerifierIdentity);
    }
    let window = verifier
        .verify(attestation, now_s)
        .ok_or(HumanoidAuthorityAttestationVerificationFailure::VerificationRejected)?;
    if !window.validate_for(attestation, now_s) {
        return Err(HumanoidAuthorityAttestationVerificationFailure::InvalidVerificationWindow);
    }

    let evidence_id = format!(
        "attested-{}-sha256:{}-via:{}",
        source_kind_id(expected_kind),
        attestation.statement_digest.to_hex(),
        verifier_digest.to_hex(),
    );
    let snapshot = HumanoidAuthoritySourceSnapshot {
        evidence_id,
        scale: attestation.scale,
        evaluated_at_s: attestation.evaluated_at_s,
        valid_until_s: attestation.valid_until_s.min(window.valid_until_s),
    };
    if !snapshot.validate_at(now_s) {
        return Err(HumanoidAuthorityAttestationVerificationFailure::InvalidVerificationWindow);
    }

    let mut verified = HumanoidVerifiedAuthoritySource {
        source_kind: expected_kind,
        subject_digest: attestation.subject_digest,
        statement_digest: attestation.statement_digest,
        evidence_digest: attestation.evidence_digest,
        verifier_digest,
        verification_digest: HumanoidEvidenceDigest::ZERO,
        revocation_epoch: window.revocation_epoch,
        snapshot,
    };
    verified.verification_digest = digest_verification(&verified);
    if !verified.validate_for(subject, expected_kind, now_s) {
        return Err(HumanoidAuthorityAttestationVerificationFailure::InvalidVerifiedSource);
    }
    Ok(verified)
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.authority-attestation-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_statement(attestation: &HumanoidAuthoritySourceAttestation) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.authority-source-statement.v1");
    h.u32(attestation.schema_version)
        .u64(source_kind_id(attestation.source_kind))
        .digest(attestation.subject_digest)
        .digest(attestation.evidence_digest)
        .f32(attestation.scale)
        .f64(attestation.evaluated_at_s)
        .f64(attestation.valid_until_s)
        .string(&attestation.issuer_id)
        .string(&attestation.key_id)
        .u64(attestation.revocation_epoch);
    h.finish()
}

fn digest_verification(source: &HumanoidVerifiedAuthoritySource) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.authority-source-verification.v1");
    h.u32(HUMANOID_AUTHORITY_VERIFICATION_SCHEMA_VERSION)
        .u64(source_kind_id(source.source_kind))
        .digest(source.subject_digest)
        .digest(source.statement_digest)
        .digest(source.evidence_digest)
        .digest(source.verifier_digest)
        .u64(source.revocation_epoch)
        .f32(source.snapshot.scale)
        .f64(source.snapshot.evaluated_at_s)
        .f64(source.snapshot.valid_until_s);
    h.finish()
}

fn source_kind_id(kind: HumanoidAuthoritySourceKind) -> u64 {
    match kind {
        HumanoidAuthoritySourceKind::Operator => 1,
        HumanoidAuthoritySourceKind::Qualification => 2,
        HumanoidAuthoritySourceKind::Physical => 3,
        HumanoidAuthoritySourceKind::Epistemic => 4,
        HumanoidAuthoritySourceKind::Cognitive => 5,
    }
}

fn task_id(task: HumanoidTask) -> u64 {
    match task {
        HumanoidTask::Stand => 1,
        HumanoidTask::Walk => 2,
        HumanoidTask::Run => 3,
        HumanoidTask::Reach => 4,
        HumanoidTask::Grasp => 5,
    }
}

fn actuation_mode_id(mode: ActuationMode) -> u64 {
    match mode {
        ActuationMode::NormalizedTorque => 1,
        ActuationMode::TorqueNewtonMetres => 2,
        ActuationMode::NormalizedPosition => 3,
        ActuationMode::PositionTargetRadians => 4,
    }
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
    use crate::morphology::HumanoidMorphology;

    struct TestVerifier;

    impl HumanoidAuthorityAttestationVerifier for TestVerifier {
        fn verifier_digest(&self) -> HumanoidEvidenceDigest {
            HumanoidEvidenceDigest::from_bytes([7; 32])
        }

        fn verify(
            &self,
            attestation: &HumanoidAuthoritySourceAttestation,
            now_s: f64,
        ) -> Option<HumanoidAuthorityVerificationWindow> {
            (attestation.authentication().scheme_id() == "test-signature-v1").then_some(
                HumanoidAuthorityVerificationWindow {
                    valid_until_s: (now_s + 0.25).min(attestation.valid_until_s()),
                    revocation_epoch: attestation.revocation_epoch(),
                },
            )
        }
    }

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "attestation-test-backend-v1",
        )
    }

    fn attestation(kind: HumanoidAuthoritySourceKind) -> HumanoidAuthoritySourceAttestation {
        HumanoidAuthoritySourceAttestation::new(
            &subject(),
            kind,
            HumanoidEvidenceDigest::from_bytes([3; 32]),
            0.8,
            1.0,
            2.0,
            "test-issuer",
            "test-key-1",
            4,
            HumanoidAuthorityAuthentication::new("test-signature-v1", vec![1, 2, 3]).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn verifier_window_shortens_source_validity() {
        let verified = verify_humanoid_authority_source(
            &subject(),
            HumanoidAuthoritySourceKind::Physical,
            &attestation(HumanoidAuthoritySourceKind::Physical),
            &TestVerifier,
            1.25,
        )
        .unwrap();
        assert_eq!(verified.valid_until_s(), 1.5);
        assert!(verified.validate_for(
            &subject(),
            HumanoidAuthoritySourceKind::Physical,
            1.49
        ));
        assert!(!verified.validate_for(
            &subject(),
            HumanoidAuthoritySourceKind::Physical,
            1.51
        ));
    }

    #[test]
    fn wrong_source_kind_is_rejected() {
        let error = verify_humanoid_authority_source(
            &subject(),
            HumanoidAuthoritySourceKind::Epistemic,
            &attestation(HumanoidAuthoritySourceKind::Physical),
            &TestVerifier,
            1.25,
        )
        .unwrap_err();
        assert_eq!(error, HumanoidAuthorityAttestationVerificationFailure::WrongSourceKind);
    }
}
