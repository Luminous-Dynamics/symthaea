// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cryptographic commitments for finalized Reach authority audits.
//!
//! The runtime authority objects are intentionally move-only and disappear when a
//! prepared command is finalized. `HumanoidReachAuthorityReceiptAudit` retains the
//! provenance needed by later qualification evidence, but its historical
//! `receipt_fingerprint` and `scope_fingerprint` are only 64-bit local checksums.
//!
//! This module upgrades that handoff without changing the mature execution path:
//! it commits the exact Reach subject, authority-source identities/scales, permit
//! epoch/time window, execution purpose/basis, deployment scope and finalization
//! instant with domain-separated SHA-256. The old fingerprints remain secondary
//! corruption checks; promotion-grade code should bind these digests instead.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{
    HumanoidExecutionPurpose, HumanoidQualificationAuthorityBasis,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_execution::{
    HumanoidPermittedReachExecutionResult, HumanoidReachAuthorityReceiptAudit,
};
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_REACH_AUTHORITY_COMMITMENT_SCHEMA_VERSION: u32 = 1;

/// Collision-resistant identity for one finalized, purpose-scoped Reach authority
/// decision. The three digests deliberately separate the pre-scope receipt,
/// execution scope, and finalization event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanoidReachAuthorityCommitment {
    subject_digest: HumanoidEvidenceDigest,
    receipt_digest: HumanoidEvidenceDigest,
    scope_digest: HumanoidEvidenceDigest,
    finalization_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachAuthorityCommitment {
    /// Revalidate the finalized audit against the actual Reach execution result and
    /// commit it cryptographically. This is an evidence identity, not a signature.
    pub fn from_finalized_execution(
        subject: &HumanoidQualificationSubject,
        result: &HumanoidPermittedReachExecutionResult,
    ) -> Option<Self> {
        if !validate_finalized_authority(subject, result) {
            return None;
        }
        let subject_digest = digest_subject(subject)?;
        let receipt_digest = digest_receipt_audit(subject_digest, &result.authority_receipt)?;
        let scope_digest = digest_scope_audit(receipt_digest, &result.authority_receipt)?;
        let finalization_digest = digest_finalization(
            scope_digest,
            result.authority_receipt.finalized_at_s,
            result.execution.report.authority_scale,
        )?;
        Some(Self {
            subject_digest,
            receipt_digest,
            scope_digest,
            finalization_digest,
        })
    }

    pub const fn subject_digest(&self) -> HumanoidEvidenceDigest {
        self.subject_digest
    }

    pub const fn receipt_digest(&self) -> HumanoidEvidenceDigest {
        self.receipt_digest
    }

    pub const fn scope_digest(&self) -> HumanoidEvidenceDigest {
        self.scope_digest
    }

    pub const fn finalization_digest(&self) -> HumanoidEvidenceDigest {
        self.finalization_digest
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        result: &HumanoidPermittedReachExecutionResult,
    ) -> bool {
        if !validate_finalized_authority(subject, result) {
            return false;
        }
        let Some(subject_digest) = digest_subject(subject) else {
            return false;
        };
        let Some(receipt_digest) = digest_receipt_audit(subject_digest, &result.authority_receipt)
        else {
            return false;
        };
        let Some(scope_digest) = digest_scope_audit(receipt_digest, &result.authority_receipt) else {
            return false;
        };
        let Some(finalization_digest) = digest_finalization(
            scope_digest,
            result.authority_receipt.finalized_at_s,
            result.execution.report.authority_scale,
        ) else {
            return false;
        };
        self.subject_digest == subject_digest
            && self.receipt_digest == receipt_digest
            && self.scope_digest == scope_digest
            && self.finalization_digest == finalization_digest
            && !self.finalization_digest.is_zero()
    }
}

fn validate_finalized_authority(
    subject: &HumanoidQualificationSubject,
    result: &HumanoidPermittedReachExecutionResult,
) -> bool {
    if !subject.validate()
        || subject.task != HumanoidTask::Reach
        || result.preparation.validation_epoch == 0
        || result.authority_receipt.validation_epoch != result.preparation.validation_epoch
        || result.authority_receipt.requirement_subject_fingerprints.as_slice()
            != [subject.fingerprint()]
    {
        return false;
    }

    let authority = &result.authority_receipt;
    if authority.receipt_fingerprint == 0
        || authority.scope_fingerprint == 0
        || !valid_id(&authority.scope_id)
        || authority.qualification_basis != authority.execution_purpose.required_qualification_basis()
        || !authority.issued_at_s.is_finite()
        || !authority.valid_until_s.is_finite()
        || !authority.finalized_at_s.is_finite()
        || authority.issued_at_s < 0.0
        || authority.valid_until_s < authority.issued_at_s
        || authority.finalized_at_s < authority.issued_at_s
        || authority.finalized_at_s > authority.valid_until_s
        || authority.finalized_at_s < result.preparation.prepared_at_s
    {
        return false;
    }

    let ids = [
        authority.operator_evidence_id.as_str(),
        authority.qualification_evidence_id.as_str(),
        authority.physical_evidence_id.as_str(),
        authority.epistemic_evidence_id.as_str(),
        authority.cognitive_evidence_id.as_str(),
    ];
    if !ids.into_iter().all(valid_id) {
        return false;
    }

    let expected_scales = [
        authority.operator_scale,
        authority.qualification_scale,
        authority.physical_scale,
        authority.epistemic_scale,
        authority.cognitive_scale,
    ];
    if !expected_scales
        .into_iter()
        .all(|scale| scale.is_finite() && (0.0..=1.0).contains(&scale))
    {
        return false;
    }

    let reported = result.execution.report.authority;
    let reported_scales = [
        reported.operator,
        reported.qualification,
        reported.physical,
        reported.epistemic,
        reported.cognitive,
    ];
    if !expected_scales
        .into_iter()
        .zip(reported_scales)
        .all(|(expected, actual)| expected.to_bits() == actual.to_bits())
    {
        return false;
    }
    reported.effective_scale().to_bits() == result.execution.report.authority_scale.to_bits()
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.authority-subject.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITMENT_SCHEMA_VERSION)
        .u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_receipt_audit(
    subject_digest: HumanoidEvidenceDigest,
    audit: &HumanoidReachAuthorityReceiptAudit,
) -> Option<HumanoidEvidenceDigest> {
    if subject_digest.is_zero()
        || audit.receipt_fingerprint == 0
        || audit.validation_epoch == 0
        || audit.requirement_subject_fingerprints.is_empty()
        || !audit.issued_at_s.is_finite()
        || !audit.valid_until_s.is_finite()
        || audit.issued_at_s < 0.0
        || audit.valid_until_s < audit.issued_at_s
    {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.authority-receipt-audit.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITMENT_SCHEMA_VERSION)
        .digest(subject_digest)
        .u64(audit.validation_epoch)
        .f64(audit.issued_at_s)
        .f64(audit.valid_until_s)
        .usize(audit.requirement_subject_fingerprints.len());
    for fingerprint in &audit.requirement_subject_fingerprints {
        h.u64(*fingerprint);
    }
    for (id, scale) in [
        (&audit.operator_evidence_id, audit.operator_scale),
        (&audit.qualification_evidence_id, audit.qualification_scale),
        (&audit.physical_evidence_id, audit.physical_scale),
        (&audit.epistemic_evidence_id, audit.epistemic_scale),
        (&audit.cognitive_evidence_id, audit.cognitive_scale),
    ] {
        if !valid_id(id) || !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
            return None;
        }
        h.string(id).f32(scale);
    }
    // Secondary compatibility checksum. Collision resistance comes from the
    // complete canonical contents above, not from this legacy value.
    h.u64(audit.receipt_fingerprint);
    Some(h.finish())
}

fn digest_scope_audit(
    receipt_digest: HumanoidEvidenceDigest,
    audit: &HumanoidReachAuthorityReceiptAudit,
) -> Option<HumanoidEvidenceDigest> {
    if receipt_digest.is_zero()
        || audit.scope_fingerprint == 0
        || !valid_id(&audit.scope_id)
        || audit.qualification_basis != audit.execution_purpose.required_qualification_basis()
    {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.authority-scope-audit.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITMENT_SCHEMA_VERSION)
        .digest(receipt_digest)
        .u64(purpose_id(audit.execution_purpose))
        .u64(basis_id(audit.qualification_basis))
        .string(&audit.scope_id)
        .u64(audit.scope_fingerprint);
    Some(h.finish())
}

fn digest_finalization(
    scope_digest: HumanoidEvidenceDigest,
    finalized_at_s: f64,
    effective_authority_scale: f32,
) -> Option<HumanoidEvidenceDigest> {
    if scope_digest.is_zero()
        || !finalized_at_s.is_finite()
        || finalized_at_s < 0.0
        || !effective_authority_scale.is_finite()
        || !(0.0..=1.0).contains(&effective_authority_scale)
    {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.authority-finalization.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITMENT_SCHEMA_VERSION)
        .digest(scope_digest)
        .f64(finalized_at_s)
        .f32(effective_authority_scale);
    Some(h.finish())
}

fn purpose_id(purpose: HumanoidExecutionPurpose) -> u64 {
    match purpose {
        HumanoidExecutionPurpose::SimulationQualification => 1,
        HumanoidExecutionPurpose::HilQualification => 2,
        HumanoidExecutionPurpose::PhysicalQualification => 3,
        HumanoidExecutionPurpose::Operational => 4,
    }
}

fn basis_id(basis: HumanoidQualificationAuthorityBasis) -> u64 {
    match basis {
        HumanoidQualificationAuthorityBasis::TrialProtocol => 1,
        HumanoidQualificationAuthorityBasis::QualifiedCapability => 2,
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

    #[test]
    fn subject_digest_changes_with_backend_identity() {
        let a = HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "backend-a",
        );
        let b = HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "backend-b",
        );
        assert_ne!(digest_subject(&a), digest_subject(&b));
    }

    #[test]
    fn scope_digest_is_domain_separated_from_receipt_digest() {
        let subject = HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "backend-a",
        );
        let subject_digest = digest_subject(&subject).unwrap();
        let audit = HumanoidReachAuthorityReceiptAudit {
            receipt_fingerprint: 11,
            scope_fingerprint: 12,
            scope_id: "sim-reach-v1".into(),
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            qualification_basis: HumanoidQualificationAuthorityBasis::TrialProtocol,
            validation_epoch: 7,
            issued_at_s: 1.0,
            valid_until_s: 3.0,
            finalized_at_s: 2.0,
            requirement_subject_fingerprints: vec![subject.fingerprint()],
            operator_evidence_id: "operator:v1".into(),
            qualification_evidence_id: "qualification:v1".into(),
            physical_evidence_id: "physical:v1".into(),
            epistemic_evidence_id: "epistemic:v1".into(),
            cognitive_evidence_id: "cognitive:v1".into(),
            operator_scale: 1.0,
            qualification_scale: 1.0,
            physical_scale: 0.9,
            epistemic_scale: 0.8,
            cognitive_scale: 0.7,
        };
        let receipt = digest_receipt_audit(subject_digest, &audit).unwrap();
        let scope = digest_scope_audit(receipt, &audit).unwrap();
        assert_ne!(receipt, scope);
    }
}
