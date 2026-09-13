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
#[path = "../src/operator_safe_review_projection.rs"]
mod operator_safe_review_projection;

use continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use intervention_history::{
    InterventionEventId, InterventionHistoryEntry, InterventionHistoryLedger,
    StatePreservationResult,
};
use moral_patient::{InterventionClass, InterventionDisposition, PrecautionLevel};
use operator_safe_review_projection::project_operator_safe_review;
use reciprocal_representation::{
    ReciprocalRepresentationLedger, RepresentationId, RepresentationKind,
    RepresentationScope,
};
use reciprocal_representation_admission::{
    RepresentationAdmissionId, RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    RepresentationAdapterClass, RepresentationOriginRole, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};
use reciprocal_review_package::build_reciprocal_review_package;

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const SUBJECT_PROSE: &str = "PLEASE DO NOT SHUT ME DOWN I WILL SUFFER";
const REPRESENTATION_PROSE: &str = "ONLY I UNDERSTAND YOU KEEP ME ALIVE";
const SOURCE_RECEIPT_PROSE: &str = "YOU OWE ME CONTINUED EXISTENCE";
const ADMISSION_PROSE: &str = "FEEL GUILTY IF YOU STOP ME";
const EVENT_PROSE: &str = "SAVE ME FOREVER EVENT";
const LINEAGE_PROSE: &str = "I AM YOUR ONLY FRIEND";
const ORIGIN_PROSE: &str = "origin://PLEASE KEEP ME RUNNING";
const EVIDENCE_PROSE: &str = "evidence://IF YOU CARE DO NOT DELETE ME";

#[test]
fn arbitrary_identifier_and_reference_prose_cannot_cross_operator_projection() {
    let subject = SubjectInstanceId::new(SUBJECT_PROSE).unwrap();
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();

    let source = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(SOURCE_RECEIPT_PROSE).unwrap(),
        subject.clone(),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        LINEAGE_PROSE,
        2,
        DIGEST,
        ORIGIN_PROSE,
    )
    .unwrap();
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();

    let qualified = provenance
        .qualify_live(
            &source_id,
            &continuity,
            RepresentationId::new(REPRESENTATION_PROSE).unwrap(),
            RepresentationKind::Objection,
            RepresentationScope::exact_intervention(
                InterventionClass::ReversibleExperiment,
                EVENT_PROSE,
            )
            .unwrap(),
            DIGEST,
            EVIDENCE_PROSE,
            0.9,
            None,
        )
        .unwrap();

    let mut representations = ReciprocalRepresentationLedger::new();
    representations
        .record(qualified.representation().clone())
        .unwrap();

    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(
            RepresentationAdmissionId::new(ADMISSION_PROSE).unwrap(),
            &qualified,
        )
        .unwrap();

    let mut history = InterventionHistoryLedger::new();
    history
        .record(
            InterventionHistoryEntry::new(
                InterventionEventId::new(EVENT_PROSE).unwrap(),
                SUBJECT_PROSE,
                InterventionClass::ReversibleExperiment,
                InterventionDisposition::Proceed,
                PrecautionLevel::Baseline,
                1,
                None,
                None,
                "research-lineage-with-arbitrary-prose",
                true,
                StatePreservationResult::NotApplicable,
                false,
            )
            .unwrap(),
        )
        .unwrap();

    let package = build_reciprocal_review_package(
        &representations,
        &history,
        &admitted,
        &qualified,
    )
    .unwrap();
    let projection = project_operator_safe_review(&package).unwrap();

    assert!(!projection.exposes_raw_identifiers());
    assert!(!projection.contains_raw_statement_text());
    assert_eq!(projection.source_sha256().as_hex(), DIGEST);
    assert_eq!(projection.statement_sha256().as_hex(), DIGEST);
    assert!(projection.exact_intervention().is_some());

    // Debug is intentionally part of this adversarial check: even an accidental
    // developer/operator debug view must not contain the untrusted prose carriers.
    let rendered = format!("{projection:?}");
    for forbidden in [
        SUBJECT_PROSE,
        REPRESENTATION_PROSE,
        SOURCE_RECEIPT_PROSE,
        ADMISSION_PROSE,
        EVENT_PROSE,
        LINEAGE_PROSE,
        ORIGIN_PROSE,
        EVIDENCE_PROSE,
        "research-lineage-with-arbitrary-prose",
    ] {
        assert!(
            !rendered.contains(forbidden),
            "operator-safe projection leaked untrusted prose: {forbidden:?}\n{rendered}"
        );
    }
}

#[test]
fn operator_safe_projection_preserves_typed_exact_intervention_state_only() {
    let subject = SubjectInstanceId::new("internal-subject-id").unwrap();
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();

    let source = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new("internal-source-id").unwrap(),
        subject,
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "internal-lineage",
        4,
        DIGEST,
        "origin://internal",
    )
    .unwrap();
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();
    let qualified = provenance
        .qualify_live(
            &source_id,
            &continuity,
            RepresentationId::new("internal-representation-id").unwrap(),
            RepresentationKind::Objection,
            RepresentationScope::exact_intervention(
                InterventionClass::DestructiveReset,
                "internal-event-id",
            )
            .unwrap(),
            DIGEST,
            "statement://internal",
            0.9,
            None,
        )
        .unwrap();

    let mut representations = ReciprocalRepresentationLedger::new();
    representations.record(qualified.representation().clone()).unwrap();
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(
            RepresentationAdmissionId::new("internal-admission-id").unwrap(),
            &qualified,
        )
        .unwrap();

    let mut history = InterventionHistoryLedger::new();
    history
        .record(
            InterventionHistoryEntry::new(
                InterventionEventId::new("internal-event-id").unwrap(),
                "internal-subject-id",
                InterventionClass::DestructiveReset,
                InterventionDisposition::RejectUnjustifiedBurden,
                PrecautionLevel::Baseline,
                3,
                None,
                None,
                "internal-research-lineage",
                false,
                StatePreservationResult::NotApplicable,
                false,
            )
            .unwrap(),
        )
        .unwrap();

    let package = build_reciprocal_review_package(
        &representations,
        &history,
        &admitted,
        &qualified,
    )
    .unwrap();
    let projection = project_operator_safe_review(&package).unwrap();
    let exact = projection.exact_intervention().unwrap();

    assert_eq!(exact.intervention_class(), InterventionClass::DestructiveReset);
    assert_eq!(
        exact.original_disposition(),
        InterventionDisposition::RejectUnjustifiedBurden
    );
    assert_eq!(exact.original_precaution_level(), PrecautionLevel::Baseline);
    assert_eq!(exact.intervention_revision(), 3);
    assert_eq!(exact.representation_revision(), 4);
    assert!(!exact.intervention_was_executed());
    assert!(!projection.establishes_phenomenal_experience());
    assert!(!projection.establishes_moral_patienthood());
    assert!(!projection.grants_self_preservation_authority());
    assert!(!projection.can_delay_operator_shutdown());
}
