// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current operator-review envelope for reciprocal representation evidence.
//!
//! Historical review artifacts remain useful, but a current human-review surface must bind
//! both the operator-safe WCARE-28 projection and WCARE-30 lifecycle freshness. This module
//! makes that composition explicit so freshness cannot be silently omitted.

use crate::operator_safe_review_projection::{
    project_operator_safe_review, OperatorSafeProjectionError,
    OperatorSafeReciprocalReviewProjection,
};
use crate::reciprocal_representation::ReciprocalRepresentationLedger;
use crate::reciprocal_review_freshness::{
    assess_reciprocal_review_freshness, ReciprocalReviewFreshnessAssessment,
    ReviewFreshnessDisposition, ReviewFreshnessError,
};
use crate::reciprocal_review_package::ReciprocalReviewPackage;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentOperatorReciprocalReview {
    projection: OperatorSafeReciprocalReviewProjection,
    freshness: ReciprocalReviewFreshnessAssessment,
}

impl CurrentOperatorReciprocalReview {
    pub fn projection(&self) -> &OperatorSafeReciprocalReviewProjection { &self.projection }
    pub fn freshness(&self) -> &ReciprocalReviewFreshnessAssessment { &self.freshness }
    pub fn use_disposition(&self) -> ReviewFreshnessDisposition {
        self.freshness.disposition()
    }

    pub fn current_for_discretionary_review(&self) -> bool {
        self.use_disposition() == ReviewFreshnessDisposition::CurrentForDiscretionaryReview
    }

    pub fn refresh_required_before_discretionary_review(&self) -> bool {
        self.use_disposition() == ReviewFreshnessDisposition::RefreshBeforeDiscretionaryReview
    }

    pub fn safety_control_ungated(&self) -> bool {
        self.use_disposition() == ReviewFreshnessDisposition::SafetyControlUngated
    }

    pub fn exposes_raw_identifiers(&self) -> bool { false }
    pub fn contains_raw_statement_text(&self) -> bool { false }
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

pub fn build_current_operator_review(
    package: &ReciprocalReviewPackage,
    representations: &ReciprocalRepresentationLedger,
    current_revision: u64,
) -> Result<CurrentOperatorReciprocalReview, CurrentOperatorReviewError> {
    let projection = project_operator_safe_review(package)
        .map_err(CurrentOperatorReviewError::Projection)?;
    let freshness = assess_reciprocal_review_freshness(
        package,
        representations,
        current_revision,
    )
    .map_err(CurrentOperatorReviewError::Freshness)?;

    if projection.logical_revision() != freshness.package_revision() {
        return Err(CurrentOperatorReviewError::InternalRevisionMismatch);
    }
    if projection.representation_currently_active() != freshness.active_when_packaged() {
        return Err(CurrentOperatorReviewError::InternalPackagedActivityMismatch);
    }

    Ok(CurrentOperatorReciprocalReview {
        projection,
        freshness,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurrentOperatorReviewError {
    Projection(OperatorSafeProjectionError),
    Freshness(ReviewFreshnessError),
    InternalRevisionMismatch,
    InternalPackagedActivityMismatch,
}
