// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Execution-purpose scoping for move-only skill authority receipts.
//!
//! Capability qualification has an unavoidable bootstrap boundary: the trials
//! used to qualify a capability cannot require that same capability to have
//! already passed operational qualification. Qualification executions therefore
//! use a `TrialProtocol` basis, while operational execution requires a
//! `QualifiedCapability` basis produced only by a verified promotion path.
//!
//! Public callers may directly scope controlled qualification trials. They cannot
//! label an arbitrary receipt as operational. The operational constructor is
//! crate-internal so capability-specific promotion code must first verify the
//! corresponding qualification artifact.

use crate::execution::HumanoidAuthorityEnvelope;
use crate::skill_authority_receipt::{
    HumanoidSkillAuthorityEvidence, HumanoidSkillAuthorityReceipt,
    HumanoidSkillAuthorityReceiptValidationFailure,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;

pub const HUMANOID_EXECUTION_AUTHORITY_SCOPE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidExecutionPurpose {
    SimulationQualification,
    HilQualification,
    PhysicalQualification,
    Operational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidQualificationAuthorityBasis {
    /// Admission by an explicit controlled qualification protocol. This does not
    /// imply the capability has already passed operational qualification.
    TrialProtocol,
    /// Authority derived from evidence that the exact capability subject has
    /// already passed the applicable operational qualification policy.
    QualifiedCapability,
}

impl HumanoidExecutionPurpose {
    pub const fn required_qualification_basis(self) -> HumanoidQualificationAuthorityBasis {
        match self {
            Self::SimulationQualification
            | Self::HilQualification
            | Self::PhysicalQualification => HumanoidQualificationAuthorityBasis::TrialProtocol,
            Self::Operational => HumanoidQualificationAuthorityBasis::QualifiedCapability,
        }
    }

    pub const fn is_qualification(self) -> bool {
        matches!(
            self,
            Self::SimulationQualification | Self::HilQualification | Self::PhysicalQualification
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidExecutionAuthorityScopeIssueFailure {
    InvalidScopeId,
    QualificationPurposeRequired,
    OperationalScopeRequiresVerifiedPromotion,
    InvalidInnerReceipt,
    Inner(HumanoidSkillAuthorityReceiptValidationFailure),
    InvalidScopeFingerprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidExecutionAuthorityScopeValidationFailure {
    InvalidScopeFingerprint,
    QualificationBasisMismatch,
    Inner(HumanoidSkillAuthorityReceiptValidationFailure),
}

/// Move-only authority receipt with an immutable execution purpose.
pub struct HumanoidScopedSkillAuthorityReceipt {
    schema_version: u32,
    purpose: HumanoidExecutionPurpose,
    qualification_basis: HumanoidQualificationAuthorityBasis,
    scope_id: String,
    scope_fingerprint: u64,
    inner: HumanoidSkillAuthorityReceipt,
}

impl std::fmt::Debug for HumanoidScopedSkillAuthorityReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidScopedSkillAuthorityReceipt")
            .field("purpose", &self.purpose)
            .field("qualification_basis", &self.qualification_basis)
            .field("scope_id", &self.scope_id)
            .field("scope_fingerprint", &self.scope_fingerprint)
            .field("inner_receipt_fingerprint", &self.inner.receipt_fingerprint())
            .finish()
    }
}

impl HumanoidScopedSkillAuthorityReceipt {
    pub const fn purpose(&self) -> HumanoidExecutionPurpose {
        self.purpose
    }

    pub const fn qualification_basis(&self) -> HumanoidQualificationAuthorityBasis {
        self.qualification_basis
    }

    pub fn scope_id(&self) -> &str {
        &self.scope_id
    }

    pub const fn scope_fingerprint(&self) -> u64 {
        self.scope_fingerprint
    }

    pub const fn receipt_fingerprint(&self) -> u64 {
        self.inner.receipt_fingerprint()
    }

    pub const fn validation_epoch(&self) -> u64 {
        self.inner.validation_epoch()
    }

    pub const fn issued_at_s(&self) -> f64 {
        self.inner.issued_at_s()
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.inner.valid_until_s()
    }

    pub fn requirement_subject_fingerprints(&self) -> &[u64] {
        self.inner.requirement_subject_fingerprints()
    }

    pub const fn authority_envelope(&self) -> HumanoidAuthorityEnvelope {
        self.inner.authority_envelope()
    }

    pub fn source_evidence(&self) -> &HumanoidSkillAuthorityEvidence {
        self.inner.source_evidence()
    }

    pub fn validate_for_permit(
        &self,
        permit: &HumanoidSkillExecutionPermit<'_>,
        now_s: f64,
    ) -> Result<(), HumanoidExecutionAuthorityScopeValidationFailure> {
        if self.qualification_basis != self.purpose.required_qualification_basis() {
            return Err(
                HumanoidExecutionAuthorityScopeValidationFailure::QualificationBasisMismatch,
            );
        }
        let expected = scoped_receipt_fingerprint(
            self.schema_version,
            self.purpose,
            self.qualification_basis,
            &self.scope_id,
            self.inner.receipt_fingerprint(),
        );
        if expected == 0 || expected != self.scope_fingerprint {
            return Err(HumanoidExecutionAuthorityScopeValidationFailure::InvalidScopeFingerprint);
        }
        self.inner
            .validate_for_permit(permit, now_s)
            .map_err(HumanoidExecutionAuthorityScopeValidationFailure::Inner)
    }
}

/// Public bootstrap path for controlled qualification runs.
///
/// This function cannot issue Operational authority and always uses the
/// `TrialProtocol` basis. The inner receipt must already match the exact live
/// permit at issuance time; Reach finalization revalidates it again before use.
pub fn scope_humanoid_qualification_trial_authority_receipt(
    inner: HumanoidSkillAuthorityReceipt,
    permit: &HumanoidSkillExecutionPermit<'_>,
    purpose: HumanoidExecutionPurpose,
    protocol_scope_id: impl Into<String>,
    now_s: f64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidExecutionAuthorityScopeIssueFailure> {
    if !purpose.is_qualification() {
        return Err(HumanoidExecutionAuthorityScopeIssueFailure::QualificationPurposeRequired);
    }
    inner
        .validate_for_permit(permit, now_s)
        .map_err(HumanoidExecutionAuthorityScopeIssueFailure::Inner)?;
    scope_checked(
        inner,
        purpose,
        HumanoidQualificationAuthorityBasis::TrialProtocol,
        protocol_scope_id.into(),
    )
}

/// Crate-internal operational constructor.
///
/// Capability-specific promotion code may call this only after verifying the
/// evidence artifact that justifies `QualifiedCapability`. Keeping this out of
/// the public API prevents external callers from self-labeling an arbitrary
/// authority receipt as production-qualified.
pub(crate) fn scope_verified_operational_authority_receipt(
    inner: HumanoidSkillAuthorityReceipt,
    permit: &HumanoidSkillExecutionPermit<'_>,
    operational_scope_id: impl Into<String>,
    now_s: f64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidExecutionAuthorityScopeIssueFailure> {
    inner
        .validate_for_permit(permit, now_s)
        .map_err(HumanoidExecutionAuthorityScopeIssueFailure::Inner)?;
    scope_checked(
        inner,
        HumanoidExecutionPurpose::Operational,
        HumanoidQualificationAuthorityBasis::QualifiedCapability,
        operational_scope_id.into(),
    )
}

fn scope_checked(
    inner: HumanoidSkillAuthorityReceipt,
    purpose: HumanoidExecutionPurpose,
    qualification_basis: HumanoidQualificationAuthorityBasis,
    scope_id: String,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidExecutionAuthorityScopeIssueFailure> {
    if !valid_id(&scope_id) {
        return Err(HumanoidExecutionAuthorityScopeIssueFailure::InvalidScopeId);
    }
    if qualification_basis != purpose.required_qualification_basis() {
        return Err(
            HumanoidExecutionAuthorityScopeIssueFailure::OperationalScopeRequiresVerifiedPromotion,
        );
    }
    if inner.receipt_fingerprint() == 0
        || inner.validation_epoch() == 0
        || inner.requirement_subject_fingerprints().is_empty()
    {
        return Err(HumanoidExecutionAuthorityScopeIssueFailure::InvalidInnerReceipt);
    }
    let scope_fingerprint = scoped_receipt_fingerprint(
        HUMANOID_EXECUTION_AUTHORITY_SCOPE_SCHEMA_VERSION,
        purpose,
        qualification_basis,
        &scope_id,
        inner.receipt_fingerprint(),
    );
    if scope_fingerprint == 0 {
        return Err(HumanoidExecutionAuthorityScopeIssueFailure::InvalidScopeFingerprint);
    }
    Ok(HumanoidScopedSkillAuthorityReceipt {
        schema_version: HUMANOID_EXECUTION_AUTHORITY_SCOPE_SCHEMA_VERSION,
        purpose,
        qualification_basis,
        scope_id,
        scope_fingerprint,
        inner,
    })
}

fn scoped_receipt_fingerprint(
    schema_version: u32,
    purpose: HumanoidExecutionPurpose,
    basis: HumanoidQualificationAuthorityBasis,
    scope_id: &str,
    inner_receipt_fingerprint: u64,
) -> u64 {
    if schema_version != HUMANOID_EXECUTION_AUTHORITY_SCOPE_SCHEMA_VERSION
        || !valid_id(scope_id)
        || inner_receipt_fingerprint == 0
        || basis != purpose.required_qualification_basis()
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, schema_version as u64);
    feed_u64(&mut hash, purpose_id(purpose));
    feed_u64(&mut hash, basis_id(basis));
    feed_bytes(&mut hash, scope_id.as_bytes());
    feed_u64(&mut hash, inner_receipt_fingerprint);
    if hash == 0 { 1 } else { hash }
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

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
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

    #[test]
    fn qualification_purposes_require_trial_protocol_basis() {
        assert_eq!(
            HumanoidExecutionPurpose::SimulationQualification.required_qualification_basis(),
            HumanoidQualificationAuthorityBasis::TrialProtocol
        );
        assert_eq!(
            HumanoidExecutionPurpose::HilQualification.required_qualification_basis(),
            HumanoidQualificationAuthorityBasis::TrialProtocol
        );
        assert_eq!(
            HumanoidExecutionPurpose::PhysicalQualification.required_qualification_basis(),
            HumanoidQualificationAuthorityBasis::TrialProtocol
        );
    }

    #[test]
    fn operational_execution_requires_qualified_capability_basis() {
        assert_eq!(
            HumanoidExecutionPurpose::Operational.required_qualification_basis(),
            HumanoidQualificationAuthorityBasis::QualifiedCapability
        );
        assert!(!HumanoidExecutionPurpose::Operational.is_qualification());
    }

    #[test]
    fn scope_fingerprint_rejects_basis_purpose_substitution() {
        let fingerprint = scoped_receipt_fingerprint(
            HUMANOID_EXECUTION_AUTHORITY_SCOPE_SCHEMA_VERSION,
            HumanoidExecutionPurpose::Operational,
            HumanoidQualificationAuthorityBasis::TrialProtocol,
            "scope-test-v1",
            42,
        );
        assert_eq!(fingerprint, 0);
    }
}
