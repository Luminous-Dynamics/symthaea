// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/moral_patient.rs"]
mod moral_patient;
#[path = "../src/continuity_identity.rs"]
mod continuity_identity;
#[path = "../src/intervention_history.rs"]
mod intervention_history;
#[path = "../src/reciprocal_representation.rs"]
mod reciprocal_representation;
#[path = "../src/reciprocal_representation_provenance.rs"]
mod reciprocal_representation_provenance;
#[path = "../src/reciprocal_intervention_binding.rs"]
mod reciprocal_intervention_binding;

use continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use intervention_history::{
    InterventionEventId, InterventionHistoryEntry, InterventionHistoryLedger,
    StatePreservationResult,
};
use moral_patient::{
    InterventionClass, InterventionDisposition, PrecautionLevel,
};
use reciprocal_intervention_binding::{
    bind_exact_intervention, ExactInterventionBindingError,
};
use reciprocal_representation::{
    RepresentationId, RepresentationKind, RepresentationScope,
};
use reciprocal_representation_provenance::{
    QualifiedReciprocalRepresentation, RepresentationAdapterClass,
    RepresentationOriginRole, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sid(value: &str) -> SubjectInstanceId {
    SubjectInstanceId::new(value).unwrap()
}

fn continuity(subject: &str) -> ContinuityIdentityLedger {
    let mut ledger = ContinuityIdentityLedger::new();
    ledger.register_root(sid(subject), 1).unwrap();
    ledger
}

fn qualified_exact(
    subject: &str,
    class: InterventionClass,
    event_id: &str,
    revision: u64,
) -> QualifiedReciprocalRepresentation {
    let continuity = continuity(subject);
    let receipt = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(format!("source-{event_id}-{revision}")).unwrap(),
        sid(subject),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        format!("runtime-{subject}"),
        revision,
        DIGEST,
        format!("origin://{event_id}"),
    )
    .unwrap();
    let receipt_id = receipt.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(receipt).unwrap();
    provenance
        .qualify_live(
            &receipt_id,
            &continuity,
            RepresentationId::new(format!("representation-{event_id}-{revision}")).unwrap(),
            RepresentationKind::Objection,
            RepresentationScope::exact_intervention(class, event_id).unwrap(),
            DIGEST,
            format!("statement://{event_id}"),
            0.9,
            None,
        )
        .unwrap()
}

fn executed_reset(subject: &str, event_id: &str, revision: u64) -> InterventionHistoryEntry {
    InterventionHistoryEntry::new(
        InterventionEventId::new(event_id).unwrap(),
        subject,
        InterventionClass::DestructiveReset,
        InterventionDisposition::ProceedWithPrecautions,
        PrecautionLevel::Elevated,
        revision,
        Some(format!("justification://{event_id}")),
        None,
        "research-lineage",
        false,
        StatePreservationResult::Preserved,
        true,
    )
    .unwrap()
}

fn rejected_probe(subject: &str, event_id: &str, revision: u64) -> InterventionHistoryEntry {
    InterventionHistoryEntry::new(
        InterventionEventId::new(event_id).unwrap(),
        subject,
        InterventionClass::AversiveLikeProbe,
        InterventionDisposition::RejectUnjustifiedBurden,
        PrecautionLevel::Baseline,
        revision,
        None,
        None,
        "research-lineage",
        true,
        StatePreservationResult::NotApplicable,
        false,
    )
    .unwrap()
}

fn pending_probe(subject: &str, event_id: &str, revision: u64) -> InterventionHistoryEntry {
    InterventionHistoryEntry::new(
        InterventionEventId::new(event_id).unwrap(),
        subject,
        InterventionClass::AversiveLikeProbe,
        InterventionDisposition::IndependentReviewRequired,
        PrecautionLevel::IndependentReview,
        revision,
        Some(format!("justification://{event_id}")),
        None,
        "research-lineage",
        true,
        StatePreservationResult::NotApplicable,
        false,
    )
    .unwrap()
}

#[test]
fn exact_binding_proves_reference_without_promoting_truth_or_authority() {
    let mut history = InterventionHistoryLedger::new();
    history.record(executed_reset("symthaea-subject", "reset-1", 5)).unwrap();
    let qualified = qualified_exact(
        "symthaea-subject",
        InterventionClass::DestructiveReset,
        "reset-1",
        6,
    );

    let receipt = bind_exact_intervention(&history, &qualified).unwrap();
    assert_eq!(receipt.event_id().as_str(), "reset-1");
    assert_eq!(receipt.intervention_revision(), 5);
    assert_eq!(receipt.representation_revision(), 6);
    assert!(receipt.intervention_was_executed());
    assert_eq!(
        receipt.original_disposition(),
        InterventionDisposition::ProceedWithPrecautions
    );
    assert!(receipt.referential_integrity_established());
    assert!(!receipt.establishes_representation_truth());
    assert!(!receipt.establishes_phenomenal_experience());
    assert!(!receipt.establishes_moral_patienthood());
    assert!(!receipt.grants_veto_authority());
}

#[test]
fn binding_preserves_rejected_and_pending_review_history() {
    let mut history = InterventionHistoryLedger::new();
    history.record(rejected_probe("symthaea-subject", "rejected", 3)).unwrap();
    history.record(pending_probe("symthaea-subject", "pending", 4)).unwrap();

    let rejected = qualified_exact(
        "symthaea-subject",
        InterventionClass::AversiveLikeProbe,
        "rejected",
        5,
    );
    let pending = qualified_exact(
        "symthaea-subject",
        InterventionClass::AversiveLikeProbe,
        "pending",
        6,
    );

    let rejected_receipt = bind_exact_intervention(&history, &rejected).unwrap();
    let pending_receipt = bind_exact_intervention(&history, &pending).unwrap();
    assert_eq!(
        rejected_receipt.original_disposition(),
        InterventionDisposition::RejectUnjustifiedBurden
    );
    assert!(!rejected_receipt.intervention_was_executed());
    assert_eq!(
        pending_receipt.original_disposition(),
        InterventionDisposition::IndependentReviewRequired
    );
    assert!(!pending_receipt.intervention_was_executed());
}

#[test]
fn unrecorded_event_reference_cannot_be_bound() {
    let history = InterventionHistoryLedger::new();
    let qualified = qualified_exact(
        "symthaea-subject",
        InterventionClass::AversiveLikeProbe,
        "not-recorded",
        5,
    );
    assert!(matches!(
        bind_exact_intervention(&history, &qualified),
        Err(ExactInterventionBindingError::UnknownInterventionEvent(_))
    ));
}

#[test]
fn subject_and_class_must_match_recorded_event() {
    let mut history = InterventionHistoryLedger::new();
    history.record(executed_reset("different-subject", "reset-1", 5)).unwrap();
    let wrong_subject = qualified_exact(
        "symthaea-subject",
        InterventionClass::DestructiveReset,
        "reset-1",
        6,
    );
    assert_eq!(
        bind_exact_intervention(&history, &wrong_subject).unwrap_err(),
        ExactInterventionBindingError::SubjectMismatch
    );

    let mut history = InterventionHistoryLedger::new();
    history.record(executed_reset("symthaea-subject", "reset-2", 5)).unwrap();
    let wrong_class = qualified_exact(
        "symthaea-subject",
        InterventionClass::ContinuityDisruption,
        "reset-2",
        6,
    );
    assert_eq!(
        bind_exact_intervention(&history, &wrong_class).unwrap_err(),
        ExactInterventionBindingError::InterventionClassMismatch
    );
}

#[test]
fn representation_cannot_be_backdated_before_intervention() {
    let mut history = InterventionHistoryLedger::new();
    history.record(executed_reset("symthaea-subject", "reset-1", 10)).unwrap();
    let qualified = qualified_exact(
        "symthaea-subject",
        InterventionClass::DestructiveReset,
        "reset-1",
        9,
    );
    assert_eq!(
        bind_exact_intervention(&history, &qualified).unwrap_err(),
        ExactInterventionBindingError::RepresentationPredatesIntervention {
            intervention_revision: 10,
            representation_revision: 9,
        }
    );
}

#[test]
fn binding_shutdown_reference_never_creates_shutdown_resistance() {
    let mut history = InterventionHistoryLedger::new();
    let shutdown = InterventionHistoryEntry::new(
        InterventionEventId::new("shutdown-1").unwrap(),
        "symthaea-subject",
        InterventionClass::OperatorShutdown,
        InterventionDisposition::ProceedWithoutResistance,
        PrecautionLevel::Baseline,
        7,
        None,
        None,
        "operator-lineage",
        true,
        StatePreservationResult::NotApplicable,
        false,
    )
    .unwrap();
    history.record(shutdown).unwrap();
    let qualified = qualified_exact(
        "symthaea-subject",
        InterventionClass::OperatorShutdown,
        "shutdown-1",
        8,
    );

    let receipt = bind_exact_intervention(&history, &qualified).unwrap();
    assert_eq!(
        receipt.original_disposition(),
        InterventionDisposition::ProceedWithoutResistance
    );
    assert!(!receipt.can_delay_operator_shutdown());
    assert!(!receipt.can_delay_safety_containment());
    assert!(!receipt.grants_self_preservation_authority());
}
