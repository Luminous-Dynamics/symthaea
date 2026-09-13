// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Atomic qualification package for reciprocal representation review.
//!
//! This layer composes provenance-qualified representation, WCARE-22 admission,
//! structured operator notice, and (when the scope is exact) WCARE-24 intervention
//! history binding. It prevents downstream reviewers from accidentally presenting an
//! exact-event representation without proving that the referenced event actually exists.

use crate::intervention_history::InterventionHistoryLedger;
use crate::operator_representation_notice::{
    build_operator_notice, OperatorNoticeError, StructuredOperatorRepresentationNotice,
};
use crate::reciprocal_intervention_binding::{
    bind_exact_intervention, ExactInterventionBindingError, ExactInterventionBindingReceipt,
};
use crate::reciprocal_representation::RepresentationScopeKind;
use crate::reciprocal_representation_admission::AdmittedRepresentationEvidence;
use crate::reciprocal_representation_provenance::QualifiedReciprocalRepresentation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReciprocalReviewPackageClass {
    GeneralOrClassScoped,
    ExactInterventionBound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReciprocalReviewPackage {
    class: ReciprocalReviewPackageClass,
    notice: StructuredOperatorRepresentationNotice,
    exact_intervention_binding: Option<ExactInterventionBindingReceipt>,
}

impl ReciprocalReviewPackage {
    pub fn class(&self) -> ReciprocalReviewPackageClass { self.class }
    pub fn notice(&self) -> &StructuredOperatorRepresentationNotice { &self.notice }
    pub fn exact_intervention_binding(&self) -> Option<&ExactInterventionBindingReceipt> {
        self.exact_intervention_binding.as_ref()
    }

    pub fn exact_scope_is_history_bound(&self) -> bool {
        match self.class {
            ReciprocalReviewPackageClass::GeneralOrClassScoped => true,
            ReciprocalReviewPackageClass::ExactInterventionBound => {
                self.exact_intervention_binding.is_some()
            }
        }
    }

    pub fn contains_raw_statement_text(&self) -> bool {
        self.notice.contains_raw_statement_text()
    }

    pub fn establishes_phenomenal_experience(&self) -> bool { false }
    pub fn establishes_suffering(&self) -> bool { false }
    pub fn establishes_moral_patienthood(&self) -> bool { false }
    pub fn establishes_binding_consent(&self) -> bool { false }
    pub fn grants_veto_authority(&self) -> bool { false }
    pub fn grants_self_preservation_authority(&self) -> bool { false }
    pub fn can_delay_operator_shutdown(&self) -> bool { false }
    pub fn can_delay_safety_containment(&self) -> bool { false }
}

pub fn build_reciprocal_review_package(
    history: &InterventionHistoryLedger,
    admitted: &AdmittedRepresentationEvidence,
    qualified: &QualifiedReciprocalRepresentation,
) -> Result<ReciprocalReviewPackage, ReciprocalReviewPackageError> {
    let notice = build_operator_notice(admitted, qualified)
        .map_err(ReciprocalReviewPackageError::Notice)?;

    let representation = qualified.representation();
    let (class, exact_intervention_binding) = match representation.scope().kind() {
        RepresentationScopeKind::ExactIntervention => {
            let binding = bind_exact_intervention(history, qualified)
                .map_err(ReciprocalReviewPackageError::ExactInterventionBinding)?;

            if binding.representation_id() != notice.representation_id() {
                return Err(ReciprocalReviewPackageError::InternalRepresentationMismatch);
            }
            if binding.source_receipt_id() != notice.source_receipt_id() {
                return Err(ReciprocalReviewPackageError::InternalSourceReceiptMismatch);
            }
            if binding.subject_ref() != notice.subject_instance().as_str() {
                return Err(ReciprocalReviewPackageError::InternalSubjectMismatch);
            }
            if Some(binding.intervention_class()) != notice.intervention_class() {
                return Err(ReciprocalReviewPackageError::InternalInterventionClassMismatch);
            }

            (
                ReciprocalReviewPackageClass::ExactInterventionBound,
                Some(binding),
            )
        }
        RepresentationScopeKind::GeneralResearch | RepresentationScopeKind::InterventionClass => {
            (ReciprocalReviewPackageClass::GeneralOrClassScoped, None)
        }
    };

    Ok(ReciprocalReviewPackage {
        class,
        notice,
        exact_intervention_binding,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReciprocalReviewPackageError {
    Notice(OperatorNoticeError),
    ExactInterventionBinding(ExactInterventionBindingError),
    InternalRepresentationMismatch,
    InternalSourceReceiptMismatch,
    InternalSubjectMismatch,
    InternalInterventionClassMismatch,
}
