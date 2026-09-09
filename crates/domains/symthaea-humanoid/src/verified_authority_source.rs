// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strong transport-neutral authority-source verification boundary.
//!
//! This module supersedes the original Reach-specific attestation handoff for
//! public motor authority. The authenticated statement binds its authentication
//! scheme ID, while the runtime source identity is the digest of the complete
//! verification decision (including verifier identity, revocation epoch and the
//! bounded verification window).
//!
//! Authentication, key management and revocation remain external responsibilities.
//! Xenia/Mycelix, a hardware trust service or another application verifier can
//! implement `HumanoidAuthoritySourceVerifier` without coupling this crate to one
//! cryptographic stack.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::qualification::HumanoidQualificationSubject;
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_VERIFIED_AUTHORITY_ATTESTATION_SCHEMA_VERSION: u32 = 2;
pub const HUMANOID_VERIFIED_AUTHORITY_DECISION_SCHEMA_VERSION: u32 = 2;
const MAX_AUTHENTICATION_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HumanoidVerifiedAuthorityKind {
    Operator,
    Qualification,
    Physical,
    Epistemic,
    Cognitive,
}

/// Scheme-neutral authentication material.
///
/// The `scheme_id` is part of the authenticated statement. Signature bytes are
/// intentionally not part of that statement: they authenticate it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidAuthorityAuthenticationEvidence {
    scheme_id: String,
    signature: Vec<u8>,
}

impl HumanoidAuthorityAuthenticationEvidence {
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

/// Canonical source claim supplied to an external verifier.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidAuthoritySourceClaim {
    schema_version: u32,
    source_kind: HumanoidVerifiedAuthorityKind,
    subject_digest: HumanoidEvidenceDigest,
    evidence_digest: HumanoidEvidenceDigest,
    scale: f32,
    evaluated_at_s: f64,
    valid_until_s: f64,
    issuer_id: String,
    key_id: String,
    revocation_epoch: u64,
    authentication: HumanoidAuthorityAuthenticationEvidence,
    statement_digest: HumanoidEvidenceDigest,
}

impl HumanoidAuthoritySourceClaim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        source_kind: HumanoidVerifiedAuthorityKind,
        evidence_digest: HumanoidEvidenceDigest,
        scale: f32,
        evaluated_at_s: f64,
        valid_until_s: f64,
        issuer_id: impl Into<String>,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        authentication: HumanoidAuthorityAuthenticationEvidence,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_VERIFIED_AUTHORITY_ATTESTATION_SCHEMA_VERSION,
            source_kind,
            subject_digest: digest_subject(subject)?,
            evidence_digest,
            scale,
            evaluated_at_s,
            valid_until_s,
            issuer_id: issuer_id.into(),
            key_id: key_id.into(),
            revocation_epoch,
            authentication,
            statement_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.validate_shape(subject) {
            return None;
        }
        value.statement_digest = digest_statement(&value);
        value.validate_for(subject).then_some(value)
    }

    pub const fn source_kind(&self) -> HumanoidVerifiedAuthorityKind {
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

    pub fn authentication(&self) -> &HumanoidAuthorityAuthenticationEvidence {
        &self.authentication
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.validate_shape(subject)
            && !self.statement_digest.is_zero()
            && self.statement_digest == digest_statement(self)
    }

    fn validate_shape(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_VERIFIED_AUTHORITY_ATTESTATION_SCHEMA_VERSION
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

/// Revocation/freshness horizon returned by the external trust root.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumanoidAuthorityVerificationDecisionWindow {
    pub valid_until_s: f64,
    pub revocation_epoch: u64,
}

impl HumanoidAuthorityVerificationDecisionWindow {
    fn validate_for(&self, claim: &HumanoidAuthoritySourceClaim, now_s: f64) -> bool {
        self.valid_until_s.is_finite()
            && self.valid_until_s >= now_s
            && self.valid_until_s <= claim.valid_until_s
            && self.revocation_epoch >= claim.revocation_epoch
    }
}

/// Application-selected authentication/revocation trust root.
pub trait HumanoidAuthoritySourceVerifier {
    /// Identity of the verifier policy, accepted issuer/key set, trust roots and
    /// revocation configuration used for this decision.
    fn verifier_digest(&self) -> HumanoidEvidenceDigest;

    /// Authenticate `claim.statement_digest()` using `claim.authentication()` and
    /// enforce scheme, issuer, key and revocation policy.
    fn verify(
        &self,
        claim: &HumanoidAuthoritySourceClaim,
        now_s: f64,
    ) -> Option<HumanoidAuthorityVerificationDecisionWindow>;
}

/// Opaque result of one exact verification decision.
///
/// Its runtime evidence ID is derived from `verification_digest`, not merely the
/// signed statement. Therefore changes to verifier policy, revocation epoch or
/// verification validity window remain visible in the final motor-authority audit.
pub struct HumanoidVerifiedAuthoritySource {
    source_kind: HumanoidVerifiedAuthorityKind,
    subject_digest: HumanoidEvidenceDigest,
    statement_digest: HumanoidEvidenceDigest,
    evidence_digest: HumanoidEvidenceDigest,
    verifier_digest: HumanoidEvidenceDigest,
    revocation_epoch: u64,
    verification_digest: HumanoidEvidenceDigest,
    snapshot: HumanoidAuthoritySourceSnapshot,
}

impl std::fmt::Debug for HumanoidVerifiedAuthoritySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidVerifiedAuthoritySource")
            .field("source_kind", &self.source_kind)
            .field("statement_digest", &self.statement_digest)
            .field("evidence_digest", &self.evidence_digest)
            .field("verifier_digest", &self.verifier_digest)
            .field("revocation_epoch", &self.revocation_epoch)
            .field("verification_digest", &self.verification_digest)
            .field("valid_until_s", &self.snapshot.valid_until_s)
            .finish()
    }
}

impl HumanoidVerifiedAuthoritySource {
    pub const fn source_kind(&self) -> HumanoidVerifiedAuthorityKind {
        self.source_kind
    }

    pub const fn evidence_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_digest
    }

    pub const fn statement_digest(&self) -> HumanoidEvidenceDigest {
        self.statement_digest
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
        expected_kind: HumanoidVerifiedAuthorityKind,
        now_s: f64,
    ) -> bool {
        self.source_kind == expected_kind
            && digest_subject(subject) == Some(self.subject_digest)
            && !self.statement_digest.is_zero()
            && !self.evidence_digest.is_zero()
            && !self.verifier_digest.is_zero()
            && !self.verification_digest.is_zero()
            && self.snapshot.evidence_id
                == format!(
                    "verified-authority-{}-sha256:{}",
                    source_kind_id(self.source_kind),
                    self.verification_digest.to_hex(),
                )
            && self.snapshot.validate_at(now_s)
            && self.verification_digest == digest_verification(self)
    }

    pub(crate) fn source_snapshot(&self) -> HumanoidAuthoritySourceSnapshot {
        self.snapshot.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidAuthoritySourceVerificationFailure {
    InvalidTime,
    InvalidClaim,
    WrongSourceKind,
    InvalidVerifierIdentity,
    VerificationRejected,
    InvalidVerificationWindow,
    InvalidVerifiedSource,
}

pub fn verify_humanoid_authority_source(
    subject: &HumanoidQualificationSubject,
    expected_kind: HumanoidVerifiedAuthorityKind,
    claim: &HumanoidAuthoritySourceClaim,
    verifier: &dyn HumanoidAuthoritySourceVerifier,
    now_s: f64,
) -> Result<HumanoidVerifiedAuthoritySource, HumanoidAuthoritySourceVerificationFailure> {
    if !now_s.is_finite() || now_s < 0.0 {
        return Err(HumanoidAuthoritySourceVerificationFailure::InvalidTime);
    }
    if !claim.validate_for(subject) {
        return Err(HumanoidAuthoritySourceVerificationFailure::InvalidClaim);
    }
    if claim.source_kind != expected_kind {
        return Err(HumanoidAuthoritySourceVerificationFailure::WrongSourceKind);
    }
    if now_s < claim.evaluated_at_s || now_s > claim.valid_until_s {
        return Err(HumanoidAuthoritySourceVerificationFailure::InvalidClaim);
    }
    let verifier_digest = verifier.verifier_digest();
    if verifier_digest.is_zero() {
        return Err(HumanoidAuthoritySourceVerificationFailure::InvalidVerifierIdentity);
    }
    let window = verifier
        .verify(claim, now_s)
        .ok_or(HumanoidAuthoritySourceVerificationFailure::VerificationRejected)?;
    if !window.validate_for(claim, now_s) {
        return Err(HumanoidAuthoritySourceVerificationFailure::InvalidVerificationWindow);
    }

    let bounded_until_s = claim.valid_until_s.min(window.valid_until_s);
    let provisional_snapshot = HumanoidAuthoritySourceSnapshot {
        evidence_id: "pending-verification-digest".to_string(),
        scale: claim.scale,
        evaluated_at_s: claim.evaluated_at_s,
        valid_until_s: bounded_until_s,
    };
    if !provisional_snapshot.validate_at(now_s) {
        return Err(HumanoidAuthoritySourceVerificationFailure::InvalidVerificationWindow);
    }

    let mut verified = HumanoidVerifiedAuthoritySource {
        source_kind: expected_kind,
        subject_digest: claim.subject_digest,
        statement_digest: claim.statement_digest,
        evidence_digest: claim.evidence_digest,
        verifier_digest,
        revocation_epoch: window.revocation_epoch,
        verification_digest: HumanoidEvidenceDigest::ZERO,
        snapshot: provisional_snapshot,
    };
    verified.verification_digest = digest_verification(&verified);
    if verified.verification_digest.is_zero() {
        return Err(HumanoidAuthoritySourceVerificationFailure::InvalidVerifiedSource);
    }
    verified.snapshot.evidence_id = format!(
        "verified-authority-{}-sha256:{}",
        source_kind_id(expected_kind),
        verified.verification_digest.to_hex(),
    );
    if !verified.validate_for(subject, expected_kind, now_s) {
        return Err(HumanoidAuthoritySourceVerificationFailure::InvalidVerifiedSource);
    }
    Ok(verified)
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.verified-authority-subject.v2");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_statement(claim: &HumanoidAuthoritySourceClaim) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.authority-source-statement.v2");
    h.u32(claim.schema_version)
        .u64(source_kind_id(claim.source_kind))
        .digest(claim.subject_digest)
        .digest(claim.evidence_digest)
        .f32(claim.scale)
        .f64(claim.evaluated_at_s)
        .f64(claim.valid_until_s)
        .string(&claim.issuer_id)
        .string(&claim.key_id)
        .u64(claim.revocation_epoch)
        .string(claim.authentication.scheme_id());
    h.finish()
}

/// Deliberately excludes `snapshot.evidence_id`: that field is derived from this
/// digest, so including it would create a circular identity definition.
fn digest_verification(source: &HumanoidVerifiedAuthoritySource) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.authority-source-verification.v2");
    h.u32(HUMANOID_VERIFIED_AUTHORITY_DECISION_SCHEMA_VERSION)
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

fn source_kind_id(kind: HumanoidVerifiedAuthorityKind) -> u64 {
    match kind {
        HumanoidVerifiedAuthorityKind::Operator => 1,
        HumanoidVerifiedAuthorityKind::Qualification => 2,
        HumanoidVerifiedAuthorityKind::Physical => 3,
        HumanoidVerifiedAuthorityKind::Epistemic => 4,
        HumanoidVerifiedAuthorityKind::Cognitive => 5,
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

    struct AcceptingVerifier {
        digest: HumanoidEvidenceDigest,
        valid_until_s: f64,
        revocation_epoch: u64,
    }

    impl HumanoidAuthoritySourceVerifier for AcceptingVerifier {
        fn verifier_digest(&self) -> HumanoidEvidenceDigest {
            self.digest
        }

        fn verify(
            &self,
            _claim: &HumanoidAuthoritySourceClaim,
            _now_s: f64,
        ) -> Option<HumanoidAuthorityVerificationDecisionWindow> {
            Some(HumanoidAuthorityVerificationDecisionWindow {
                valid_until_s: self.valid_until_s,
                revocation_epoch: self.revocation_epoch,
            })
        }
    }

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "verified-authority-test-backend",
        )
    }

    fn claim(scheme_id: &str) -> HumanoidAuthoritySourceClaim {
        HumanoidAuthoritySourceClaim::new(
            &subject(),
            HumanoidVerifiedAuthorityKind::Physical,
            HumanoidEvidenceDigest::from_bytes([3; 32]),
            0.8,
            10.0,
            20.0,
            "physical-monitor",
            "key-7",
            4,
            HumanoidAuthorityAuthenticationEvidence::new(scheme_id, vec![7; 64]).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn authentication_scheme_is_part_of_signed_statement() {
        assert_ne!(claim("ml-dsa-87").statement_digest(), claim("hybrid-v1").statement_digest());
    }

    #[test]
    fn verification_window_and_revocation_epoch_change_runtime_identity() {
        let a = verify_humanoid_authority_source(
            &subject(),
            HumanoidVerifiedAuthorityKind::Physical,
            &claim("ml-dsa-87"),
            &AcceptingVerifier {
                digest: HumanoidEvidenceDigest::from_bytes([9; 32]),
                valid_until_s: 15.0,
                revocation_epoch: 4,
            },
            11.0,
        )
        .unwrap();
        let b = verify_humanoid_authority_source(
            &subject(),
            HumanoidVerifiedAuthorityKind::Physical,
            &claim("ml-dsa-87"),
            &AcceptingVerifier {
                digest: HumanoidEvidenceDigest::from_bytes([9; 32]),
                valid_until_s: 14.0,
                revocation_epoch: 5,
            },
            11.0,
        )
        .unwrap();
        assert_ne!(a.verification_digest(), b.verification_digest());
    }
}
