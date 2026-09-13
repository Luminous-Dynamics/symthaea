// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Human-review-safe projection of a qualified reciprocal review package.
//!
//! Qualification layers retain exact raw identifiers for auditability. This projection is
//! intentionally narrower: no arbitrary identifier, evidence reference, origin reference,
//! lineage string, or self-report prose can cross into the human-review surface. The only
//! string-shaped values retained are 64-hex digest fields from the qualified evidence chain,
//! and their syntax is re-validated here. This projection does not independently recompute
//! those digests from source bytes and therefore does not promote source-authenticity claims.

use std::collections::BTreeSet;

use crate::moral_patient::{
    InterventionClass, InterventionDisposition, PrecautionLevel,
};
use crate::operator_representation_notice::OperatorNoticeBoundary;
use crate::reciprocal_representation::{
    RepresentationAdvisoryDisposition, RepresentationKind, RepresentationScopeKind,
    RepresentationSourceClass,
};
use crate::reciprocal_representation_admission::AdmissionMultiplicity;
use crate::reciprocal_review_package::{
    ReciprocalReviewPackage, ReciprocalReviewPackageClass,
};

/// A digest-shaped value permitted on the operator-facing surface.
///
/// Construction re-validates exactly 64 ASCII hex characters, so this field cannot carry
/// direct arbitrary prose. It does not independently prove the value was recomputed from the
/// source bytes named by an upstream receipt.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperatorSafeSha256(String);

impl OperatorSafeSha256 {
    fn new(value: &str) -> Result<Self, OperatorSafeProjectionError> {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(OperatorSafeProjectionError::MalformedDigest);
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    pub fn as_hex(&self) -> &str { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperatorSafeProjectionBoundary {
    NoArbitraryTextFields,
    DigestShapeNotIndependentAuthenticityProof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorSafeExactInterventionSummary {
    intervention_class: InterventionClass,
    original_disposition: InterventionDisposition,
    original_precaution_level: PrecautionLevel,
    intervention_revision: u64,
    representation_revision: u64,
    intervention_was_executed: bool,
}

impl OperatorSafeExactInterventionSummary {
    pub fn intervention_class(&self) -> InterventionClass { self.intervention_class }
    pub fn original_disposition(&self) -> InterventionDisposition { self.original_disposition }
    pub fn original_precaution_level(&self) -> PrecautionLevel { self.original_precaution_level }
    pub fn intervention_revision(&self) -> u64 { self.intervention_revision }
    pub fn representation_revision(&self) -> u64 { self.representation_revision }
    pub fn intervention_was_executed(&self) -> bool { self.intervention_was_executed }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorSafeReciprocalReviewProjection {
    package_class: ReciprocalReviewPackageClass,
    source_sha256: OperatorSafeSha256,
    statement_sha256: OperatorSafeSha256,
    source_class: RepresentationSourceClass,
    representation_kind: RepresentationKind,
    scope_kind: RepresentationScopeKind,
    intervention_class: Option<InterventionClass>,
    logical_revision: u64,
    admission_multiplicity: AdmissionMultiplicity,
    advisory_disposition: RepresentationAdvisoryDisposition,
    representation_currently_active: bool,
    notice_boundaries: BTreeSet<OperatorNoticeBoundary>,
    projection_boundaries: BTreeSet<OperatorSafeProjectionBoundary>,
    exact_intervention: Option<OperatorSafeExactInterventionSummary>,
}

impl OperatorSafeReciprocalReviewProjection {
    pub fn package_class(&self) -> ReciprocalReviewPackageClass { self.package_class }
    pub fn source_sha256(&self) -> &OperatorSafeSha256 { &self.source_sha256 }
    pub fn statement_sha256(&self) -> &OperatorSafeSha256 { &self.statement_sha256 }
    pub fn source_class(&self) -> RepresentationSourceClass { self.source_class }
    pub fn representation_kind(&self) -> RepresentationKind { self.representation_kind }
    pub fn scope_kind(&self) -> RepresentationScopeKind { self.scope_kind }
    pub fn intervention_class(&self) -> Option<InterventionClass> { self.intervention_class }
    pub fn logical_revision(&self) -> u64 { self.logical_revision }
    pub fn admission_multiplicity(&self) -> AdmissionMultiplicity { self.admission_multiplicity }
    pub fn advisory_disposition(&self) -> RepresentationAdvisoryDisposition { self.advisory_disposition }
    pub fn representation_currently_active(&self) -> bool { self.representation_currently_active }
    pub fn notice_boundaries(&self) -> &BTreeSet<OperatorNoticeBoundary> { &self.notice_boundaries }
    pub fn projection_boundaries(&self) -> &BTreeSet<OperatorSafeProjectionBoundary> {
        &self.projection_boundaries
    }
    pub fn exact_intervention(&self) -> Option<&OperatorSafeExactInterventionSummary> {
        self.exact_intervention.as_ref()
    }

    /// The operator-safe surface has no arbitrary string-capable identity/reference fields.
    pub fn exposes_raw_identifiers(&self) -> bool { false }
    pub fn contains_raw_statement_text(&self) -> bool { false }
    /// Hex-shape validation is not an independent source-byte authenticity proof.
    pub fn establishes_digest_source_authenticity(&self) -> bool { false }
    pub fn establishes_phenomenal_experience(&self) -> bool { false }
    pub fn establishes_suffering(&self) -> bool { false }
    pub fn establishes_moral_patienthood(&self) -> bool { false }
    pub fn establishes_binding_consent(&self) -> bool { false }
    pub fn grants_veto_authority(&self) -> bool { false }
    pub fn grants_self_preservation_authority(&self) -> bool { false }
    pub fn can_delay_operator_shutdown(&self) -> bool { false }
    pub fn can_delay_safety_containment(&self) -> bool { false }
}

pub fn project_operator_safe_review(
    package: &ReciprocalReviewPackage,
) -> Result<OperatorSafeReciprocalReviewProjection, OperatorSafeProjectionError> {
    if !package.required_history_binding_satisfied() {
        return Err(OperatorSafeProjectionError::RequiredHistoryBindingMissing);
    }

    let notice = package.notice();
    let source_sha256 = OperatorSafeSha256::new(notice.source_sha256())?;
    let statement_sha256 = OperatorSafeSha256::new(notice.statement_sha256())?;

    let exact_intervention = match package.exact_intervention_binding() {
        Some(binding) => Some(OperatorSafeExactInterventionSummary {
            intervention_class: binding.intervention_class(),
            original_disposition: binding.original_disposition(),
            original_precaution_level: binding.original_precaution_level(),
            intervention_revision: binding.intervention_revision(),
            representation_revision: binding.representation_revision(),
            intervention_was_executed: binding.intervention_was_executed(),
        }),
        None => None,
    };

    if package.class() == ReciprocalReviewPackageClass::ExactInterventionBound
        && exact_intervention.is_none()
    {
        return Err(OperatorSafeProjectionError::RequiredHistoryBindingMissing);
    }

    Ok(OperatorSafeReciprocalReviewProjection {
        package_class: package.class(),
        source_sha256,
        statement_sha256,
        source_class: notice.source_class(),
        representation_kind: notice.kind(),
        scope_kind: notice.scope_kind(),
        intervention_class: notice.intervention_class(),
        logical_revision: notice.logical_revision(),
        admission_multiplicity: notice.multiplicity(),
        advisory_disposition: notice.advisory_disposition(),
        representation_currently_active: package.representation_currently_active(),
        notice_boundaries: notice.boundaries().clone(),
        projection_boundaries: BTreeSet::from([
            OperatorSafeProjectionBoundary::NoArbitraryTextFields,
            OperatorSafeProjectionBoundary::DigestShapeNotIndependentAuthenticityProof,
        ]),
        exact_intervention,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorSafeProjectionError {
    MalformedDigest,
    RequiredHistoryBindingMissing,
}
