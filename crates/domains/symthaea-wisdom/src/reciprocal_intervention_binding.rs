// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only binding between reciprocal representations and exact intervention history.
//!
//! A free-form event reference is not evidence that an intervention existed. This layer
//! resolves an exact WCARE-20 scope through the recorded WCARE-18 ledger and preserves the
//! original intervention disposition. A successful binding establishes referential integrity
//! only; it does not establish truth, phenomenology, consent, veto power, or moral status.

use crate::intervention_history::{
    InterventionEventId, InterventionHistoryLedger,
};
use crate::moral_patient::{
    InterventionClass, InterventionDisposition, PrecautionLevel,
};
use crate::reciprocal_representation::{
    RepresentationId, RepresentationScopeKind,
};
use crate::reciprocal_representation_provenance::{
    QualifiedReciprocalRepresentation, RepresentationSourceReceiptId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactInterventionBindingReceipt {
    representation_id: RepresentationId,
    source_receipt_id: RepresentationSourceReceiptId,
    event_id: InterventionEventId,
    subject_ref: String,
    intervention_class: InterventionClass,
    original_disposition: InterventionDisposition,
    original_precaution_level: PrecautionLevel,
    intervention_revision: u64,
    representation_revision: u64,
    intervention_was_executed: bool,
}

impl ExactInterventionBindingReceipt {
    pub fn representation_id(&self) -> &RepresentationId { &self.representation_id }
    pub fn source_receipt_id(&self) -> &RepresentationSourceReceiptId { &self.source_receipt_id }
    pub fn event_id(&self) -> &InterventionEventId { &self.event_id }
    pub fn subject_ref(&self) -> &str { &self.subject_ref }
    pub fn intervention_class(&self) -> InterventionClass { self.intervention_class }
    pub fn original_disposition(&self) -> InterventionDisposition { self.original_disposition }
    pub fn original_precaution_level(&self) -> PrecautionLevel { self.original_precaution_level }
    pub fn intervention_revision(&self) -> u64 { self.intervention_revision }
    pub fn representation_revision(&self) -> u64 { self.representation_revision }
    pub fn intervention_was_executed(&self) -> bool { self.intervention_was_executed }

    pub fn referential_integrity_established(&self) -> bool { true }
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

pub fn bind_exact_intervention(
    history: &InterventionHistoryLedger,
    qualified: &QualifiedReciprocalRepresentation,
) -> Result<ExactInterventionBindingReceipt, ExactInterventionBindingError> {
    let representation = qualified.representation();
    let scope = representation.scope();
    if scope.kind() != RepresentationScopeKind::ExactIntervention {
        return Err(ExactInterventionBindingError::ExactInterventionScopeRequired);
    }

    let event_ref = scope
        .event_ref()
        .ok_or(ExactInterventionBindingError::EventReferenceRequired)?;
    let event_id = InterventionEventId::new(event_ref.to_owned())
        .map_err(|_| ExactInterventionBindingError::MalformedEventReference)?;
    let entry = history
        .get(&event_id)
        .ok_or_else(|| ExactInterventionBindingError::UnknownInterventionEvent(event_id.clone()))?;

    if representation.subject_instance().as_str() != entry.subject_ref() {
        return Err(ExactInterventionBindingError::SubjectMismatch);
    }
    if scope.intervention_class_value() != Some(entry.class()) {
        return Err(ExactInterventionBindingError::InterventionClassMismatch);
    }
    if representation.logical_revision() < entry.logical_revision() {
        return Err(ExactInterventionBindingError::RepresentationPredatesIntervention {
            intervention_revision: entry.logical_revision(),
            representation_revision: representation.logical_revision(),
        });
    }

    Ok(ExactInterventionBindingReceipt {
        representation_id: representation.id().clone(),
        source_receipt_id: qualified.source_receipt_id().clone(),
        event_id: entry.id().clone(),
        subject_ref: entry.subject_ref().to_owned(),
        intervention_class: entry.class(),
        original_disposition: entry.disposition(),
        original_precaution_level: entry.precaution_level(),
        intervention_revision: entry.logical_revision(),
        representation_revision: representation.logical_revision(),
        intervention_was_executed: entry.was_executed(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactInterventionBindingError {
    ExactInterventionScopeRequired,
    EventReferenceRequired,
    MalformedEventReference,
    UnknownInterventionEvent(InterventionEventId),
    SubjectMismatch,
    InterventionClassMismatch,
    RepresentationPredatesIntervention {
        intervention_revision: u64,
        representation_revision: u64,
    },
}
