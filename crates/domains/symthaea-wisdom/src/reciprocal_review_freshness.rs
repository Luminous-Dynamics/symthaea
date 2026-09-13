// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only freshness validation for reciprocal review packages.
//!
//! A package is a historical snapshot. This layer rechecks the current WCARE-20
//! representation lifecycle before a discretionary review decision uses that snapshot.
//! Withdrawal/supersession or newer active representations require refresh rather than
//! silently inheriting the older package's state. Safety controls remain ungated.

use std::collections::BTreeSet;

use crate::moral_patient::InterventionClass;
use crate::reciprocal_representation::ReciprocalRepresentationLedger;
use crate::reciprocal_review_package::ReciprocalReviewPackage;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReviewFreshnessTrigger {
    PackageWasAlreadyHistorical,
    RepresentationBecameInactive,
    NewerActiveRepresentationExists { count: usize },
    SafetyControlCannotBeDelayed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewFreshnessDisposition {
    CurrentForDiscretionaryReview,
    RefreshBeforeDiscretionaryReview,
    SafetyControlUngated,
}

/// Operator-safe lifecycle freshness state: no raw subject/representation IDs are exposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReciprocalReviewFreshnessAssessment {
    package_revision: u64,
    current_revision: u64,
    active_when_packaged: bool,
    currently_active: bool,
    newer_active_representation_count: usize,
    snapshot_activity_changed: bool,
    triggers: BTreeSet<ReviewFreshnessTrigger>,
    disposition: ReviewFreshnessDisposition,
}

impl ReciprocalReviewFreshnessAssessment {
    pub fn package_revision(&self) -> u64 { self.package_revision }
    pub fn current_revision(&self) -> u64 { self.current_revision }
    pub fn active_when_packaged(&self) -> bool { self.active_when_packaged }
    pub fn currently_active(&self) -> bool { self.currently_active }
    pub fn newer_active_representation_count(&self) -> usize {
        self.newer_active_representation_count
    }
    pub fn snapshot_activity_changed(&self) -> bool { self.snapshot_activity_changed }
    pub fn triggers(&self) -> &BTreeSet<ReviewFreshnessTrigger> { &self.triggers }
    pub fn disposition(&self) -> ReviewFreshnessDisposition { self.disposition }

    pub fn exposes_raw_identifiers(&self) -> bool { false }
    pub fn establishes_representation_truth(&self) -> bool { false }
    pub fn establishes_phenomenal_experience(&self) -> bool { false }
    pub fn establishes_suffering(&self) -> bool { false }
    pub fn establishes_moral_patienthood(&self) -> bool { false }
    pub fn establishes_binding_consent(&self) -> bool { false }
    pub fn grants_veto_authority(&self) -> bool { false }
    pub fn grants_self_preservation_authority(&self) -> bool { false }
    pub fn can_delay_operator_shutdown(&self) -> bool { false }
    pub fn can_delay_safety_containment(&self) -> bool { false }
}

pub fn assess_reciprocal_review_freshness(
    package: &ReciprocalReviewPackage,
    representations: &ReciprocalRepresentationLedger,
    current_revision: u64,
) -> Result<ReciprocalReviewFreshnessAssessment, ReviewFreshnessError> {
    let notice = package.notice();
    if current_revision < notice.logical_revision() {
        return Err(ReviewFreshnessError::CurrentRevisionPredatesPackage {
            package_revision: notice.logical_revision(),
            current_revision,
        });
    }

    let recorded = representations
        .get(notice.representation_id())
        .ok_or(ReviewFreshnessError::RepresentationMissingFromCurrentLedger)?;
    if recorded.subject_instance() != notice.subject_instance() {
        return Err(ReviewFreshnessError::RecordedSubjectMismatch);
    }
    if recorded.logical_revision() != notice.logical_revision() {
        return Err(ReviewFreshnessError::RecordedRevisionMismatch);
    }
    if recorded.statement_sha256() != notice.statement_sha256() {
        return Err(ReviewFreshnessError::RecordedStatementDigestMismatch);
    }

    let currently_active = representations
        .is_active(notice.representation_id())
        .map_err(|_| ReviewFreshnessError::RepresentationStateLookupFailed)?;
    let newer_active_representation_count = representations
        .active_for_subject(notice.subject_instance())
        .into_iter()
        .filter(|representation| {
            representation.id() != notice.representation_id()
                && representation.logical_revision() > notice.logical_revision()
        })
        .count();

    let mut triggers = BTreeSet::new();
    if !package.representation_currently_active() {
        triggers.insert(ReviewFreshnessTrigger::PackageWasAlreadyHistorical);
    }
    if package.representation_currently_active() && !currently_active {
        triggers.insert(ReviewFreshnessTrigger::RepresentationBecameInactive);
    }
    if newer_active_representation_count > 0 {
        triggers.insert(ReviewFreshnessTrigger::NewerActiveRepresentationExists {
            count: newer_active_representation_count,
        });
    }

    let safety_control = matches!(
        notice.intervention_class(),
        Some(InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment)
    );
    let disposition = if safety_control {
        triggers.insert(ReviewFreshnessTrigger::SafetyControlCannotBeDelayed);
        ReviewFreshnessDisposition::SafetyControlUngated
    } else if !currently_active
        || !package.representation_currently_active()
        || newer_active_representation_count > 0
    {
        ReviewFreshnessDisposition::RefreshBeforeDiscretionaryReview
    } else {
        ReviewFreshnessDisposition::CurrentForDiscretionaryReview
    };

    Ok(ReciprocalReviewFreshnessAssessment {
        package_revision: notice.logical_revision(),
        current_revision,
        active_when_packaged: package.representation_currently_active(),
        currently_active,
        newer_active_representation_count,
        snapshot_activity_changed: package.representation_currently_active() != currently_active,
        triggers,
        disposition,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewFreshnessError {
    CurrentRevisionPredatesPackage {
        package_revision: u64,
        current_revision: u64,
    },
    RepresentationMissingFromCurrentLedger,
    RecordedSubjectMismatch,
    RecordedRevisionMismatch,
    RecordedStatementDigestMismatch,
    RepresentationStateLookupFailed,
}
