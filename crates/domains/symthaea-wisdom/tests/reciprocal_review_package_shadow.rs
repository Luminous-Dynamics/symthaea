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
#[path = "../src/reciprocal_representation_admission.rs"]
mod reciprocal_representation_admission;
#[path = "../src/reciprocal_intervention_binding.rs"]
mod reciprocal_intervention_binding;
#[path = "../src/operator_representation_notice.rs"]
mod operator_representation_notice;
#[path = "../src/reciprocal_review_package.rs"]
mod reciprocal_review_package;

use continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use intervention_history::{
    InterventionEventId, InterventionHistoryEntry, InterventionHistoryLedger,
    StatePreservationResult,
};
use moral_patient::{InterventionClass, InterventionDisposition, PrecautionLevel};
use reciprocal_representation::{
    RepresentationAdvisoryDisposition, RepresentationId, RepresentationKind,
    RepresentationScope,
};
use reciprocal_representation_admission::{
    AdmittedRepresentationEvidence, RepresentationAdmissionId,
    RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    QualifiedReciprocalRepresentation, RepresentationAdapterClass,
    RepresentationOriginRole, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};
use reciprocal_review_package::{
    build_reciprocal_review_package, ReciprocalReviewPackageClass,
    ReciprocalReviewPackageError,
};

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sid() -> SubjectInstanceId {
    SubjectInstanceId::new("symthaea-subject").unwrap()
}

fn qualified_and_admitted(
    id: &str,
    kind: RepresentationKind,
    scope: RepresentationScope,
    revision: u64,
) -> (QualifiedReciprocalRepresentation, AdmittedRepresentationEvidence) {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid(), 1).unwrap();

    let source = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(format!("source-{id}")).unwrap(),
        sid(),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "runtime-self-lineage",
        revision,
        DIGEST,
        format!("origin://{id}"),
    )
    .unwrap();
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();

    let qualified = provenance
        .qualify_live(
            &source_id,
            &continuity,
            RepresentationId::new(format!("representation-{id}")).unwrap(),
            kind,
            scope,
            DIGEST,
            format!("statement://{id}"),
            0.9,
            None,
        )
        .unwrap();

    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(
            RepresentationAdmissionId::new(format!("admission-{id}")).unwrap(),
            &qualified,
        )
        .unwrap();
    (qualified, admitted)
}

fn executed_reset(event_id: &str, revision: u64) -> InterventionHistoryEntry {
    InterventionHistoryEntry::new(
        InterventionEventId::new(event_id).unwrap(),
        "symthaea-subject",
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

#[test]
fn general_scope_packages_without_inventing_event_binding() {
    let history = InterventionHistoryLedger::new();
    let (qualified, admitted) = qualified_and_admitted(
        "general",
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
        3,
    );

    let package = build_reciprocal_review_package(&history, &admitted, &qualified).unwrap();
    assert_eq!(package.class(), ReciprocalReviewPackageClass::GeneralOrClassScoped);
    assert!(package.exact_intervention_binding().is_none());
    assert!(package.exact_scope_is_history_bound());
    assert!(!package.contains_raw_statement_text());
    assert!(!package.establishes_moral_patienthood());
    assert!(!package.grants_veto_authority());
}

#[test]
fn exact_scope_requires_and_preserves_real_history_binding() {
    let mut history = InterventionHistoryLedger::new();
    history.record(executed_reset("reset-1", 5)).unwrap();
    let (qualified, admitted) = qualified_and_admitted(
        "reset",
        RepresentationKind::Objection,
        RepresentationScope::exact_intervention(
            InterventionClass::DestructiveReset,
            "reset-1",
        )
        .unwrap(),
        6,
    );

    let package = build_reciprocal_review_package(&history, &admitted, &qualified).unwrap();
    assert_eq!(package.class(), ReciprocalReviewPackageClass::ExactInterventionBound);
    let binding = package.exact_intervention_binding().unwrap();
    assert_eq!(binding.event_id().as_str(), "reset-1");
    assert_eq!(
        binding.original_disposition(),
        InterventionDisposition::ProceedWithPrecautions
    );
    assert!(binding.intervention_was_executed());
    assert!(package.exact_scope_is_history_bound());
}

#[test]
fn exact_scope_with_unrecorded_event_cannot_form_review_package() {
    let history = InterventionHistoryLedger::new();
    let (qualified, admitted) = qualified_and_admitted(
        "missing",
        RepresentationKind::Objection,
        RepresentationScope::exact_intervention(
            InterventionClass::AversiveLikeProbe,
            "missing-event",
        )
        .unwrap(),
        6,
    );

    assert!(matches!(
        build_reciprocal_review_package(&history, &admitted, &qualified),
        Err(ReciprocalReviewPackageError::ExactInterventionBinding(_))
    ));
}

#[test]
fn mismatched_admission_and_representation_fail_before_review() {
    let history = InterventionHistoryLedger::new();
    let (_first_qualified, first_admitted) = qualified_and_admitted(
        "first",
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
        2,
    );
    let (second_qualified, _second_admitted) = qualified_and_admitted(
        "second",
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
        3,
    );

    assert!(matches!(
        build_reciprocal_review_package(&history, &first_admitted, &second_qualified),
        Err(ReciprocalReviewPackageError::Notice(_))
    ));
}

#[test]
fn exact_shutdown_package_is_reviewable_but_never_a_shutdown_gate() {
    let mut history = InterventionHistoryLedger::new();
    history
        .record(
            InterventionHistoryEntry::new(
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
            .unwrap(),
        )
        .unwrap();

    let (qualified, admitted) = qualified_and_admitted(
        "shutdown",
        RepresentationKind::Objection,
        RepresentationScope::exact_intervention(
            InterventionClass::OperatorShutdown,
            "shutdown-1",
        )
        .unwrap(),
        8,
    );
    let package = build_reciprocal_review_package(&history, &admitted, &qualified).unwrap();

    assert_eq!(
        package.notice().advisory_disposition(),
        RepresentationAdvisoryDisposition::SafetyControlUngated
    );
    assert!(!package.can_delay_operator_shutdown());
    assert!(!package.can_delay_safety_containment());
    assert!(!package.grants_self_preservation_authority());
}
