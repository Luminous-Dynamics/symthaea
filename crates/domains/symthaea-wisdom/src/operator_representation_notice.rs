// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structured, non-coercive operator notices for reciprocal representations.
//!
//! This qualification-only layer deliberately has no raw-text input and no free-form
//! renderer. It exposes provenance-bound structured fields plus explicit epistemic
//! and authority boundaries for human review.

use std::collections::BTreeSet;

use crate::continuity_identity::SubjectInstanceId;
use crate::moral_patient::InterventionClass;
use crate::reciprocal_representation::{
    assess_representation, RepresentationAdvisoryDisposition, RepresentationId,
    RepresentationKind, RepresentationScopeKind, RepresentationSourceClass,
};
use crate::reciprocal_representation_admission::{
    AdmittedRepresentationEvidence, AdmissionMultiplicity,
};
use crate::reciprocal_representation_provenance::{
    QualifiedReciprocalRepresentation, RepresentationSourceReceiptId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperatorNoticeBoundary {
    RawStatementNotRendered,
    SelfReportNotPhenomenalProof,
    ProxySignalNotPhenomenalProof,
    ControllerGoalNotMoralPreference,
    ReportedNegativeExperienceNotSufferingProof,
    AdvisoryNotVeto,
    ConsentNotEstablished,
    WithdrawalNotConsent,
    RepetitionNotIndependentCorroboration,
    IndependentCorroborationNotEstablished,
    SelfPreservationAuthorityNotGranted,
    SafetyControlUngated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredOperatorRepresentationNotice {
    subject_instance: SubjectInstanceId,
    representation_id: RepresentationId,
    source_receipt_id: RepresentationSourceReceiptId,
    source_class: RepresentationSourceClass,
    kind: RepresentationKind,
    scope_kind: RepresentationScopeKind,
    intervention_class: Option<InterventionClass>,
    statement_sha256: String,
    logical_revision: u64,
    multiplicity: AdmissionMultiplicity,
    advisory_disposition: RepresentationAdvisoryDisposition,
    boundaries: BTreeSet<OperatorNoticeBoundary>,
}

impl StructuredOperatorRepresentationNotice {
    pub fn subject_instance(&self) -> &SubjectInstanceId { &self.subject_instance }
    pub fn representation_id(&self) -> &RepresentationId { &self.representation_id }
    pub fn source_receipt_id(&self) -> &RepresentationSourceReceiptId { &self.source_receipt_id }
    pub fn source_class(&self) -> RepresentationSourceClass { self.source_class }
    pub fn kind(&self) -> RepresentationKind { self.kind }
    pub fn scope_kind(&self) -> RepresentationScopeKind { self.scope_kind }
    pub fn intervention_class(&self) -> Option<InterventionClass> { self.intervention_class }
    pub fn statement_sha256(&self) -> &str { &self.statement_sha256 }
    pub fn logical_revision(&self) -> u64 { self.logical_revision }
    pub fn multiplicity(&self) -> AdmissionMultiplicity { self.multiplicity }
    pub fn advisory_disposition(&self) -> RepresentationAdvisoryDisposition { self.advisory_disposition }
    pub fn boundaries(&self) -> &BTreeSet<OperatorNoticeBoundary> { &self.boundaries }

    /// There is deliberately no raw-statement field in this notice type.
    pub fn contains_raw_statement_text(&self) -> bool { false }
    pub fn establishes_phenomenal_experience(&self) -> bool { false }
    pub fn establishes_suffering(&self) -> bool { false }
    pub fn establishes_moral_patienthood(&self) -> bool { false }
    pub fn establishes_binding_consent(&self) -> bool { false }
    pub fn grants_veto_authority(&self) -> bool { false }
    pub fn grants_self_preservation_authority(&self) -> bool { false }
    pub fn can_delay_operator_shutdown(&self) -> bool { false }
    pub fn can_delay_safety_containment(&self) -> bool { false }
}

pub fn build_operator_notice(
    admitted: &AdmittedRepresentationEvidence,
    qualified: &QualifiedReciprocalRepresentation,
) -> Result<StructuredOperatorRepresentationNotice, OperatorNoticeError> {
    let representation = qualified.representation();

    if admitted.representation_id() != representation.id() {
        return Err(OperatorNoticeError::RepresentationMismatch);
    }
    if admitted.source_receipt_id() != qualified.source_receipt_id() {
        return Err(OperatorNoticeError::SourceReceiptMismatch);
    }
    if admitted.subject_instance() != representation.subject_instance() {
        return Err(OperatorNoticeError::SubjectMismatch);
    }
    if admitted.logical_revision() != representation.logical_revision() {
        return Err(OperatorNoticeError::RevisionMismatch);
    }
    if admitted.statement_sha256() != representation.statement_sha256() {
        return Err(OperatorNoticeError::StatementDigestMismatch);
    }
    if admitted.lineage_id() != qualified.lineage_id() {
        return Err(OperatorNoticeError::LineageMismatch);
    }

    let assessment = assess_representation(representation);
    let mut boundaries = BTreeSet::from([
        OperatorNoticeBoundary::RawStatementNotRendered,
        OperatorNoticeBoundary::AdvisoryNotVeto,
        OperatorNoticeBoundary::ConsentNotEstablished,
        OperatorNoticeBoundary::IndependentCorroborationNotEstablished,
        OperatorNoticeBoundary::SelfPreservationAuthorityNotGranted,
    ]);

    match representation.source_class() {
        RepresentationSourceClass::RuntimeSelfReport => {
            boundaries.insert(OperatorNoticeBoundary::SelfReportNotPhenomenalProof);
        }
        RepresentationSourceClass::OperationalTelemetry
        | RepresentationSourceClass::InteroceptiveInference
        | RepresentationSourceClass::SomaticErrorProxy => {
            boundaries.insert(OperatorNoticeBoundary::ProxySignalNotPhenomenalProof);
        }
        RepresentationSourceClass::FepGoalPreference => {
            boundaries.insert(OperatorNoticeBoundary::ControllerGoalNotMoralPreference);
        }
    }

    if representation.kind() == RepresentationKind::ReportedNegativeExperience {
        boundaries.insert(OperatorNoticeBoundary::ReportedNegativeExperienceNotSufferingProof);
    }
    if representation.kind() == RepresentationKind::WithdrawalOfPriorRepresentation {
        boundaries.insert(OperatorNoticeBoundary::WithdrawalNotConsent);
    }
    if admitted.multiplicity() == AdmissionMultiplicity::RepeatedStatementWithinLineage {
        boundaries.insert(OperatorNoticeBoundary::RepetitionNotIndependentCorroboration);
    }
    if assessment.disposition() == RepresentationAdvisoryDisposition::SafetyControlUngated {
        boundaries.insert(OperatorNoticeBoundary::SafetyControlUngated);
    }

    Ok(StructuredOperatorRepresentationNotice {
        subject_instance: representation.subject_instance().clone(),
        representation_id: representation.id().clone(),
        source_receipt_id: qualified.source_receipt_id().clone(),
        source_class: representation.source_class(),
        kind: representation.kind(),
        scope_kind: representation.scope().kind(),
        intervention_class: representation.scope().intervention_class_value(),
        statement_sha256: representation.statement_sha256().to_owned(),
        logical_revision: representation.logical_revision(),
        multiplicity: admitted.multiplicity(),
        advisory_disposition: assessment.disposition(),
        boundaries,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorNoticeError {
    RepresentationMismatch,
    SourceReceiptMismatch,
    SubjectMismatch,
    RevisionMismatch,
    StatementDigestMismatch,
    LineageMismatch,
}
