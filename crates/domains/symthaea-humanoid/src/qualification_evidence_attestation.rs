// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authentication and revocation boundary for qualification-evidence producers.
//!
//! This domain is deliberately separate from motor authority. A verified recorder
//! proves who authenticated one exact immutable qualification artifact under one
//! exact verifier policy; it does not grant permission to actuate, increase an
//! authority scale, or promote a capability by itself.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::grasp_contact_evidence::HumanoidGraspContactPolicy;
use crate::grasp_controller_qualification::{
    HumanoidGraspControllerCandidate, HumanoidGraspControllerQualificationPolicy,
};
use crate::grasp_measurement_coverage::{
    HumanoidGraspMeasurementCoveragePolicy, HumanoidGraspQualificationMeasurementChain,
    HumanoidMeasuredGraspControllerQualificationTrial,
};
use crate::grasp_qualification_validation::validate_complete_measured_grasp_qualification;
use crate::grasp_retention_evidence::HumanoidGraspRetentionPolicy;
use crate::qualification::HumanoidQualificationSubject;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_QUALIFICATION_EVIDENCE_CLAIM_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_QUALIFICATION_EVIDENCE_DECISION_SCHEMA_VERSION: u32 = 1;
const MAX_AUTHENTICATION_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HumanoidQualificationEvidenceKind {
    GraspMeasuredControllerTrial,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidQualificationEvidenceAuthentication {
    scheme_id: String,
    signature: Vec<u8>,
}

impl HumanoidQualificationEvidenceAuthentication {
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

/// Canonical producer claim for one immutable qualification artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidQualificationEvidenceClaim {
    schema_version: u32,
    evidence_kind: HumanoidQualificationEvidenceKind,
    subject_digest: HumanoidEvidenceDigest,
    evidence_digest: HumanoidEvidenceDigest,
    provenance_context_digest: HumanoidEvidenceDigest,
    producer_id: String,
    producer_artifact_digest: HumanoidEvidenceDigest,
    key_id: String,
    revocation_epoch: u64,
    attested_at_unix_millis: u64,
    valid_until_unix_millis: u64,
    authentication: HumanoidQualificationEvidenceAuthentication,
    statement_digest: HumanoidEvidenceDigest,
}

impl HumanoidQualificationEvidenceClaim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        evidence_kind: HumanoidQualificationEvidenceKind,
        evidence_digest: HumanoidEvidenceDigest,
        provenance_context_digest: HumanoidEvidenceDigest,
        producer_id: impl Into<String>,
        producer_artifact_digest: HumanoidEvidenceDigest,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        attested_at_unix_millis: u64,
        valid_until_unix_millis: u64,
        authentication: HumanoidQualificationEvidenceAuthentication,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_QUALIFICATION_EVIDENCE_CLAIM_SCHEMA_VERSION,
            evidence_kind,
            subject_digest: digest_subject(subject)?,
            evidence_digest,
            provenance_context_digest,
            producer_id: producer_id.into(),
            producer_artifact_digest,
            key_id: key_id.into(),
            revocation_epoch,
            attested_at_unix_millis,
            valid_until_unix_millis,
            authentication,
            statement_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.validate_shape(subject) {
            return None;
        }
        value.statement_digest = digest_claim_statement(&value);
        value.validate_for(subject).then_some(value)
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.validate_shape(subject)
            && !self.statement_digest.is_zero()
            && self.statement_digest == digest_claim_statement(self)
    }

    fn validate_shape(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_QUALIFICATION_EVIDENCE_CLAIM_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && !self.evidence_digest.is_zero()
            && !self.provenance_context_digest.is_zero()
            && valid_id(&self.producer_id)
            && !self.producer_artifact_digest.is_zero()
            && valid_id(&self.key_id)
            && self.attested_at_unix_millis > 0
            && self.valid_until_unix_millis >= self.attested_at_unix_millis
            && self.authentication.validate()
    }

    pub const fn evidence_kind(&self) -> HumanoidQualificationEvidenceKind {
        self.evidence_kind
    }

    pub const fn evidence_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_digest
    }

    pub const fn provenance_context_digest(&self) -> HumanoidEvidenceDigest {
        self.provenance_context_digest
    }

    pub const fn producer_artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.producer_artifact_digest
    }

    pub fn producer_id(&self) -> &str {
        &self.producer_id
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

    pub fn authentication(&self) -> &HumanoidQualificationEvidenceAuthentication {
        &self.authentication
    }

    pub const fn statement_digest(&self) -> HumanoidEvidenceDigest {
        self.statement_digest
    }
}

/// Two-phase helper: obtain the canonical statement digest before signing.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidUnsignedQualificationEvidenceClaim {
    subject: HumanoidQualificationSubject,
    evidence_kind: HumanoidQualificationEvidenceKind,
    evidence_digest: HumanoidEvidenceDigest,
    provenance_context_digest: HumanoidEvidenceDigest,
    producer_id: String,
    producer_artifact_digest: HumanoidEvidenceDigest,
    key_id: String,
    revocation_epoch: u64,
    attested_at_unix_millis: u64,
    valid_until_unix_millis: u64,
    scheme_id: String,
    statement_digest: HumanoidEvidenceDigest,
}

impl HumanoidUnsignedQualificationEvidenceClaim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: HumanoidQualificationSubject,
        evidence_kind: HumanoidQualificationEvidenceKind,
        evidence_digest: HumanoidEvidenceDigest,
        provenance_context_digest: HumanoidEvidenceDigest,
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
        let placeholder = HumanoidQualificationEvidenceAuthentication::new(
            scheme_id.clone(),
            vec![0x00],
        )?;
        let preview = HumanoidQualificationEvidenceClaim::new(
            &subject,
            evidence_kind,
            evidence_digest,
            provenance_context_digest,
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
            subject,
            evidence_kind,
            evidence_digest,
            provenance_context_digest,
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

    pub const fn statement_digest(&self) -> HumanoidEvidenceDigest {
        self.statement_digest
    }

    pub const fn evidence_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_digest
    }

    pub fn attach_signature(self, signature: Vec<u8>) -> Option<HumanoidQualificationEvidenceClaim> {
        let authentication = HumanoidQualificationEvidenceAuthentication::new(
            self.scheme_id.clone(),
            signature,
        )?;
        let claim = HumanoidQualificationEvidenceClaim::new(
            &self.subject,
            self.evidence_kind,
            self.evidence_digest,
            self.provenance_context_digest,
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
pub struct HumanoidQualificationEvidenceVerificationWindow {
    pub valid_until_unix_millis: u64,
    pub revocation_epoch: u64,
}

impl HumanoidQualificationEvidenceVerificationWindow {
    fn validate_for(&self, claim: &HumanoidQualificationEvidenceClaim, now_unix_millis: u64) -> bool {
        now_unix_millis > 0
            && self.valid_until_unix_millis >= now_unix_millis
            && self.valid_until_unix_millis <= claim.valid_until_unix_millis
            && self.revocation_epoch >= claim.revocation_epoch
    }
}

/// External trust root for qualification-evidence producers.
pub trait HumanoidQualificationEvidenceVerifier {
    /// Exact accepted producer/key/scheme/trust-root/revocation policy identity.
    fn verifier_digest(&self) -> HumanoidEvidenceDigest;

    fn verify(
        &self,
        claim: &HumanoidQualificationEvidenceClaim,
        now_unix_millis: u64,
    ) -> Option<HumanoidQualificationEvidenceVerificationWindow>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidQualificationEvidenceVerificationFailure {
    InvalidTime,
    InvalidEvidenceLineage,
    InvalidClaim,
    WrongEvidenceKind,
    EvidenceDigestMismatch,
    ProvenanceContextMismatch,
    InvalidVerifierIdentity,
    VerificationRejected,
    InvalidVerificationWindow,
    InvalidVerifiedEvidence,
}

/// Opaque result of one exact qualification-evidence verification decision.
pub struct HumanoidVerifiedQualificationEvidence {
    schema_version: u32,
    evidence_kind: HumanoidQualificationEvidenceKind,
    subject_digest: HumanoidEvidenceDigest,
    evidence_digest: HumanoidEvidenceDigest,
    provenance_context_digest: HumanoidEvidenceDigest,
    statement_digest: HumanoidEvidenceDigest,
    producer_artifact_digest: HumanoidEvidenceDigest,
    verifier_digest: HumanoidEvidenceDigest,
    revocation_epoch: u64,
    verification_valid_until_unix_millis: u64,
    verification_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidVerifiedQualificationEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidVerifiedQualificationEvidence")
            .field("evidence_kind", &self.evidence_kind)
            .field("evidence_digest", &self.evidence_digest)
            .field("provenance_context_digest", &self.provenance_context_digest)
            .field("producer_artifact_digest", &self.producer_artifact_digest)
            .field("verifier_digest", &self.verifier_digest)
            .field("revocation_epoch", &self.revocation_epoch)
            .field("verification_digest", &self.verification_digest)
            .finish()
    }
}

impl HumanoidVerifiedQualificationEvidence {
    pub const fn evidence_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_digest
    }

    pub const fn provenance_context_digest(&self) -> HumanoidEvidenceDigest {
        self.provenance_context_digest
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
        subject: &HumanoidQualificationSubject,
        expected_kind: HumanoidQualificationEvidenceKind,
        now_unix_millis: u64,
    ) -> bool {
        self.schema_version == HUMANOID_QUALIFICATION_EVIDENCE_DECISION_SCHEMA_VERSION
            && self.evidence_kind == expected_kind
            && digest_subject(subject) == Some(self.subject_digest)
            && !self.evidence_digest.is_zero()
            && !self.provenance_context_digest.is_zero()
            && !self.statement_digest.is_zero()
            && !self.producer_artifact_digest.is_zero()
            && !self.verifier_digest.is_zero()
            && self.verification_valid_until_unix_millis >= now_unix_millis
            && !self.verification_digest.is_zero()
            && self.verification_digest == digest_verification(self)
    }
}

/// Verify authenticated recorder provenance for one complete measured Grasp trial.
///
/// The underlying measured trial must recursively validate through every semantic
/// policy first. Only then can producer authentication be accepted for the bundle.
#[allow(clippy::too_many_arguments)]
pub fn verify_measured_grasp_qualification_recorder(
    subject: &HumanoidQualificationSubject,
    candidate: &HumanoidGraspControllerCandidate,
    controller_policy: &HumanoidGraspControllerQualificationPolicy,
    contact_policy: &HumanoidGraspContactPolicy,
    retention_policy: &HumanoidGraspRetentionPolicy,
    measurement_chain: &HumanoidGraspQualificationMeasurementChain,
    measurement_policy: &HumanoidGraspMeasurementCoveragePolicy,
    measured_trial: &HumanoidMeasuredGraspControllerQualificationTrial,
    claim: &HumanoidQualificationEvidenceClaim,
    verifier: &dyn HumanoidQualificationEvidenceVerifier,
    now_unix_millis: u64,
) -> Result<HumanoidVerifiedQualificationEvidence, HumanoidQualificationEvidenceVerificationFailure> {
    if now_unix_millis == 0 {
        return Err(HumanoidQualificationEvidenceVerificationFailure::InvalidTime);
    }
    if !validate_complete_measured_grasp_qualification(
        subject,
        candidate,
        controller_policy,
        contact_policy,
        retention_policy,
        measurement_chain,
        measurement_policy,
        measured_trial,
    ) {
        return Err(HumanoidQualificationEvidenceVerificationFailure::InvalidEvidenceLineage);
    }
    if !claim.validate_for(subject)
        || now_unix_millis < claim.attested_at_unix_millis
        || now_unix_millis > claim.valid_until_unix_millis
    {
        return Err(HumanoidQualificationEvidenceVerificationFailure::InvalidClaim);
    }
    if claim.evidence_kind != HumanoidQualificationEvidenceKind::GraspMeasuredControllerTrial {
        return Err(HumanoidQualificationEvidenceVerificationFailure::WrongEvidenceKind);
    }
    if claim.evidence_digest != measured_trial.bundle_digest() {
        return Err(HumanoidQualificationEvidenceVerificationFailure::EvidenceDigestMismatch);
    }
    if claim.provenance_context_digest != measurement_chain.chain_digest() {
        return Err(HumanoidQualificationEvidenceVerificationFailure::ProvenanceContextMismatch);
    }

    let verifier_digest = verifier.verifier_digest();
    if verifier_digest.is_zero() {
        return Err(HumanoidQualificationEvidenceVerificationFailure::InvalidVerifierIdentity);
    }
    let window = verifier
        .verify(claim, now_unix_millis)
        .ok_or(HumanoidQualificationEvidenceVerificationFailure::VerificationRejected)?;
    if !window.validate_for(claim, now_unix_millis) {
        return Err(HumanoidQualificationEvidenceVerificationFailure::InvalidVerificationWindow);
    }

    let mut verified = HumanoidVerifiedQualificationEvidence {
        schema_version: HUMANOID_QUALIFICATION_EVIDENCE_DECISION_SCHEMA_VERSION,
        evidence_kind: claim.evidence_kind,
        subject_digest: digest_subject(subject)
            .ok_or(HumanoidQualificationEvidenceVerificationFailure::InvalidClaim)?,
        evidence_digest: claim.evidence_digest,
        provenance_context_digest: claim.provenance_context_digest,
        statement_digest: claim.statement_digest,
        producer_artifact_digest: claim.producer_artifact_digest,
        verifier_digest,
        revocation_epoch: window.revocation_epoch,
        verification_valid_until_unix_millis: window.valid_until_unix_millis,
        verification_digest: HumanoidEvidenceDigest::ZERO,
    };
    verified.verification_digest = digest_verification(&verified);
    if !verified.validate_for(
        subject,
        HumanoidQualificationEvidenceKind::GraspMeasuredControllerTrial,
        now_unix_millis,
    ) {
        return Err(HumanoidQualificationEvidenceVerificationFailure::InvalidVerifiedEvidence);
    }
    Ok(verified)
}

fn digest_claim_statement(value: &HumanoidQualificationEvidenceClaim) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.qualification-evidence-claim.v1");
    h.u32(value.schema_version)
        .u64(evidence_kind_id(value.evidence_kind))
        .digest(value.subject_digest)
        .digest(value.evidence_digest)
        .digest(value.provenance_context_digest)
        .string(&value.producer_id)
        .digest(value.producer_artifact_digest)
        .string(&value.key_id)
        .u64(value.revocation_epoch)
        .u64(value.attested_at_unix_millis)
        .u64(value.valid_until_unix_millis)
        .string(value.authentication.scheme_id());
    h.finish()
}

fn digest_verification(value: &HumanoidVerifiedQualificationEvidence) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.qualification-evidence-verification.v1");
    h.u32(value.schema_version)
        .u64(evidence_kind_id(value.evidence_kind))
        .digest(value.subject_digest)
        .digest(value.evidence_digest)
        .digest(value.provenance_context_digest)
        .digest(value.statement_digest)
        .digest(value.producer_artifact_digest)
        .digest(value.verifier_digest)
        .u64(value.revocation_epoch)
        .u64(value.verification_valid_until_unix_millis);
    h.finish()
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.qualification-evidence-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn evidence_kind_id(kind: HumanoidQualificationEvidenceKind) -> u64 {
    match kind {
        HumanoidQualificationEvidenceKind::GraspMeasuredControllerTrial => 1,
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
