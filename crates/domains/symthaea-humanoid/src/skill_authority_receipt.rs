// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh, skill-atomic binding for the five restrictive authority sources.
//!
//! `HumanoidAuthorityEnvelope` intentionally contains only five scales. That is
//! useful inside the deterministic execution boundary, but insufficient as proof
//! that those values were evaluated for the same skill, validation epoch, body,
//! backend, and time window as a live execution permit.
//!
//! This module adds that missing local binding. It does **not** authenticate an
//! operator, create qualification evidence, or verify hardware by itself. Each
//! upstream subsystem supplies an explicit source snapshot; issuance only binds
//! those already-established restrictive values to one atomic skill permit.

use crate::execution::HumanoidAuthorityEnvelope;
use crate::morphology::HumanoidMorphology;
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::ActuationMode;

pub const HUMANOID_SKILL_AUTHORITY_RECEIPT_SCHEMA_VERSION: u32 = 1;

/// One already-evaluated restrictive authority source.
///
/// There is deliberately no Default. `evidence_id` is a local provenance label
/// or upstream evidence identity, not a signature. `valid_until_s` makes stale
/// authority fail closed even if a caller retains the value in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidAuthoritySourceSnapshot {
    pub evidence_id: String,
    pub scale: f32,
    pub evaluated_at_s: f64,
    pub valid_until_s: f64,
}

impl HumanoidAuthoritySourceSnapshot {
    pub fn validate_at(&self, now_s: f64) -> bool {
        valid_evidence_id(&self.evidence_id)
            && self.scale.is_finite()
            && (0.0..=1.0).contains(&self.scale)
            && self.evaluated_at_s.is_finite()
            && self.evaluated_at_s >= 0.0
            && self.valid_until_s.is_finite()
            && self.valid_until_s >= self.evaluated_at_s
            && now_s.is_finite()
            && now_s >= self.evaluated_at_s
            && now_s <= self.valid_until_s
    }
}

/// All five independent restriction sources for one issuance instant.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidSkillAuthorityEvidence {
    pub operator: HumanoidAuthoritySourceSnapshot,
    pub qualification: HumanoidAuthoritySourceSnapshot,
    pub physical: HumanoidAuthoritySourceSnapshot,
    pub epistemic: HumanoidAuthoritySourceSnapshot,
    pub cognitive: HumanoidAuthoritySourceSnapshot,
}

impl HumanoidSkillAuthorityEvidence {
    pub fn validate_at(&self, now_s: f64) -> bool {
        self.operator.validate_at(now_s)
            && self.qualification.validate_at(now_s)
            && self.physical.validate_at(now_s)
            && self.epistemic.validate_at(now_s)
            && self.cognitive.validate_at(now_s)
    }

    fn envelope(&self) -> HumanoidAuthorityEnvelope {
        HumanoidAuthorityEnvelope {
            operator: self.operator.scale,
            qualification: self.qualification.scale,
            physical: self.physical.scale,
            epistemic: self.epistemic.scale,
            cognitive: self.cognitive.scale,
        }
    }

    fn earliest_expiry_s(&self) -> f64 {
        [
            self.operator.valid_until_s,
            self.qualification.valid_until_s,
            self.physical.valid_until_s,
            self.epistemic.valid_until_s,
            self.cognitive.valid_until_s,
        ]
        .into_iter()
        .fold(f64::INFINITY, f64::min)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidSkillAuthorityReceiptIssueFailure {
    InvalidIssuanceTime,
    InvalidPermitLineage,
    InvalidSourceEvidence,
    InvalidExpiry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidSkillAuthorityReceiptValidationFailure {
    InvalidValidationTime,
    ReceiptExpired,
    ValidationEpochMismatch,
    MorphologyMismatch,
    ActuationModeMismatch,
    BackendProfileMismatch,
    RequirementSubjectsMismatch,
    ReceiptFingerprintMismatch,
}

/// Opaque local proof that five authority sources were fresh and bound to one
/// exact atomic skill permit.
///
/// This type is intentionally non-Clone and non-Serialize. It is move-only in
/// ordinary safe Rust and is meant to be consumed by the motor finalization path.
pub struct HumanoidSkillAuthorityReceipt {
    schema_version: u32,
    validation_epoch: u64,
    morphology: HumanoidMorphology,
    actuation_mode: ActuationMode,
    backend_profile_id: String,
    requirement_subject_fingerprints: Vec<u64>,
    issued_at_s: f64,
    valid_until_s: f64,
    receipt_fingerprint: u64,
    evidence: HumanoidSkillAuthorityEvidence,
    envelope: HumanoidAuthorityEnvelope,
}

impl std::fmt::Debug for HumanoidSkillAuthorityReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidSkillAuthorityReceipt")
            .field("schema_version", &self.schema_version)
            .field("validation_epoch", &self.validation_epoch)
            .field("morphology", &self.morphology)
            .field("actuation_mode", &self.actuation_mode)
            .field("backend_profile_id", &self.backend_profile_id)
            .field(
                "requirement_subject_fingerprints",
                &self.requirement_subject_fingerprints,
            )
            .field("issued_at_s", &self.issued_at_s)
            .field("valid_until_s", &self.valid_until_s)
            .field("receipt_fingerprint", &self.receipt_fingerprint)
            .field("effective_scale", &self.envelope.effective_scale())
            .finish()
    }
}

impl HumanoidSkillAuthorityReceipt {
    pub const fn validation_epoch(&self) -> u64 {
        self.validation_epoch
    }

    pub const fn morphology(&self) -> HumanoidMorphology {
        self.morphology
    }

    pub const fn actuation_mode(&self) -> ActuationMode {
        self.actuation_mode
    }

    pub fn backend_profile_id(&self) -> &str {
        &self.backend_profile_id
    }

    pub fn requirement_subject_fingerprints(&self) -> &[u64] {
        &self.requirement_subject_fingerprints
    }

    pub const fn issued_at_s(&self) -> f64 {
        self.issued_at_s
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.valid_until_s
    }

    pub const fn receipt_fingerprint(&self) -> u64 {
        self.receipt_fingerprint
    }

    pub const fn authority_envelope(&self) -> HumanoidAuthorityEnvelope {
        self.envelope
    }

    pub fn source_evidence(&self) -> &HumanoidSkillAuthorityEvidence {
        &self.evidence
    }

    /// Revalidate identity, expiry, and the receipt's internal checksum against
    /// the exact live permit immediately before motor finalization.
    pub fn validate_for_permit(
        &self,
        permit: &HumanoidSkillExecutionPermit<'_>,
        now_s: f64,
    ) -> Result<(), HumanoidSkillAuthorityReceiptValidationFailure> {
        if !now_s.is_finite() || now_s < 0.0 {
            return Err(HumanoidSkillAuthorityReceiptValidationFailure::InvalidValidationTime);
        }
        if now_s < self.issued_at_s || now_s > self.valid_until_s {
            return Err(HumanoidSkillAuthorityReceiptValidationFailure::ReceiptExpired);
        }
        if self.validation_epoch != permit.epoch() {
            return Err(HumanoidSkillAuthorityReceiptValidationFailure::ValidationEpochMismatch);
        }
        if self.morphology != permit.morphology() {
            return Err(HumanoidSkillAuthorityReceiptValidationFailure::MorphologyMismatch);
        }
        if self.actuation_mode != permit.actuation_mode() {
            return Err(HumanoidSkillAuthorityReceiptValidationFailure::ActuationModeMismatch);
        }
        if self.backend_profile_id != permit.backend_profile_id() {
            return Err(HumanoidSkillAuthorityReceiptValidationFailure::BackendProfileMismatch);
        }
        let expected_subjects = permit
            .requirements()
            .iter()
            .map(|requirement| requirement.request.subject_fingerprint)
            .collect::<Vec<_>>();
        if self.requirement_subject_fingerprints != expected_subjects {
            return Err(
                HumanoidSkillAuthorityReceiptValidationFailure::RequirementSubjectsMismatch,
            );
        }
        let expected_fingerprint = authority_receipt_fingerprint(
            self.schema_version,
            self.validation_epoch,
            self.morphology,
            self.actuation_mode,
            &self.backend_profile_id,
            &self.requirement_subject_fingerprints,
            self.issued_at_s,
            self.valid_until_s,
            &self.evidence,
        );
        if expected_fingerprint == 0 || expected_fingerprint != self.receipt_fingerprint {
            return Err(
                HumanoidSkillAuthorityReceiptValidationFailure::ReceiptFingerprintMismatch,
            );
        }
        Ok(())
    }
}

/// Bind five already-established authority source snapshots to the exact live
/// atomic skill permit.
pub fn issue_humanoid_skill_authority_receipt(
    permit: &HumanoidSkillExecutionPermit<'_>,
    evidence: HumanoidSkillAuthorityEvidence,
    now_s: f64,
) -> Result<HumanoidSkillAuthorityReceipt, HumanoidSkillAuthorityReceiptIssueFailure> {
    if !now_s.is_finite() || now_s < 0.0 {
        return Err(HumanoidSkillAuthorityReceiptIssueFailure::InvalidIssuanceTime);
    }
    let requirement_subject_fingerprints = permit
        .requirements()
        .iter()
        .map(|requirement| requirement.request.subject_fingerprint)
        .collect::<Vec<_>>();
    if permit.epoch() == 0
        || permit.backend_profile_id().trim().is_empty()
        || requirement_subject_fingerprints.is_empty()
        || requirement_subject_fingerprints
            .iter()
            .any(|fingerprint| *fingerprint == 0)
    {
        return Err(HumanoidSkillAuthorityReceiptIssueFailure::InvalidPermitLineage);
    }
    if !evidence.validate_at(now_s) {
        return Err(HumanoidSkillAuthorityReceiptIssueFailure::InvalidSourceEvidence);
    }
    let valid_until_s = evidence.earliest_expiry_s();
    if !valid_until_s.is_finite() || valid_until_s < now_s {
        return Err(HumanoidSkillAuthorityReceiptIssueFailure::InvalidExpiry);
    }
    let envelope = evidence.envelope();
    let receipt_fingerprint = authority_receipt_fingerprint(
        HUMANOID_SKILL_AUTHORITY_RECEIPT_SCHEMA_VERSION,
        permit.epoch(),
        permit.morphology(),
        permit.actuation_mode(),
        permit.backend_profile_id(),
        &requirement_subject_fingerprints,
        now_s,
        valid_until_s,
        &evidence,
    );
    if receipt_fingerprint == 0 {
        return Err(HumanoidSkillAuthorityReceiptIssueFailure::InvalidPermitLineage);
    }
    Ok(HumanoidSkillAuthorityReceipt {
        schema_version: HUMANOID_SKILL_AUTHORITY_RECEIPT_SCHEMA_VERSION,
        validation_epoch: permit.epoch(),
        morphology: permit.morphology(),
        actuation_mode: permit.actuation_mode(),
        backend_profile_id: permit.backend_profile_id().to_string(),
        requirement_subject_fingerprints,
        issued_at_s: now_s,
        valid_until_s,
        receipt_fingerprint,
        evidence,
        envelope,
    })
}

fn valid_evidence_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

#[allow(clippy::too_many_arguments)]
fn authority_receipt_fingerprint(
    schema_version: u32,
    epoch: u64,
    morphology: HumanoidMorphology,
    actuation_mode: ActuationMode,
    backend_profile_id: &str,
    requirement_subject_fingerprints: &[u64],
    issued_at_s: f64,
    valid_until_s: f64,
    evidence: &HumanoidSkillAuthorityEvidence,
) -> u64 {
    if schema_version != HUMANOID_SKILL_AUTHORITY_RECEIPT_SCHEMA_VERSION
        || epoch == 0
        || backend_profile_id.trim().is_empty()
        || requirement_subject_fingerprints.is_empty()
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, schema_version as u64);
    feed_u64(&mut hash, epoch);
    feed_u64(&mut hash, morphology_id(morphology));
    feed_u64(&mut hash, actuation_mode_id(actuation_mode));
    feed_bytes(&mut hash, backend_profile_id.as_bytes());
    feed_u64(&mut hash, requirement_subject_fingerprints.len() as u64);
    for fingerprint in requirement_subject_fingerprints {
        feed_u64(&mut hash, *fingerprint);
    }
    feed_u64(&mut hash, issued_at_s.to_bits());
    feed_u64(&mut hash, valid_until_s.to_bits());
    for source in [
        &evidence.operator,
        &evidence.qualification,
        &evidence.physical,
        &evidence.epistemic,
        &evidence.cognitive,
    ] {
        feed_bytes(&mut hash, source.evidence_id.as_bytes());
        feed_u64(&mut hash, source.scale.to_bits() as u64);
        feed_u64(&mut hash, source.evaluated_at_s.to_bits());
        feed_u64(&mut hash, source.valid_until_s.to_bits());
    }
    if hash == 0 { 1 } else { hash }
}

fn morphology_id(morphology: HumanoidMorphology) -> u64 {
    match morphology {
        HumanoidMorphology::Dmc21 => 1,
        HumanoidMorphology::Dexterous53 => 2,
        HumanoidMorphology::FullSpine => 3,
        HumanoidMorphology::WithNeckWrist => 4,
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

fn feed_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

fn feed_bytes(hash: &mut u64, bytes: &[u8]) {
    feed_u64(hash, bytes.len() as u64);
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: &str, scale: f32, evaluated_at_s: f64, valid_until_s: f64) -> HumanoidAuthoritySourceSnapshot {
        HumanoidAuthoritySourceSnapshot {
            evidence_id: id.into(),
            scale,
            evaluated_at_s,
            valid_until_s,
        }
    }

    fn evidence() -> HumanoidSkillAuthorityEvidence {
        HumanoidSkillAuthorityEvidence {
            operator: source("operator-test-v1", 1.0, 1.0, 2.0),
            qualification: source("qualification-test-v1", 0.9, 0.8, 3.0),
            physical: source("physical-test-v1", 0.8, 1.1, 1.8),
            epistemic: source("epistemic-test-v1", 0.7, 1.2, 1.6),
            cognitive: source("cognitive-test-v1", 0.6, 1.0, 2.5),
        }
    }

    #[test]
    fn source_evidence_rejects_stale_or_future_evaluation() {
        assert!(source("good", 1.0, 1.0, 2.0).validate_at(1.5));
        assert!(!source("stale", 1.0, 1.0, 1.4).validate_at(1.5));
        assert!(!source("future", 1.0, 1.6, 2.0).validate_at(1.5));
    }

    #[test]
    fn evidence_envelope_uses_exact_source_scales_and_earliest_expiry() {
        let evidence = evidence();
        let envelope = evidence.envelope();
        assert_eq!(envelope.operator, 1.0);
        assert_eq!(envelope.qualification, 0.9);
        assert_eq!(envelope.physical, 0.8);
        assert_eq!(envelope.epistemic, 0.7);
        assert_eq!(envelope.cognitive, 0.6);
        assert_eq!(envelope.effective_scale(), 0.6);
        assert_eq!(evidence.earliest_expiry_s(), 1.6);
    }
}
