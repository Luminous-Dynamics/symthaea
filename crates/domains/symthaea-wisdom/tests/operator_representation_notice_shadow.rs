// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/moral_patient.rs"]
mod moral_patient;
#[path = "../src/continuity_identity.rs"]
mod continuity_identity;
#[path = "../src/reciprocal_representation.rs"]
mod reciprocal_representation;
#[path = "../src/reciprocal_representation_provenance.rs"]
mod reciprocal_representation_provenance;
#[path = "../src/reciprocal_representation_admission.rs"]
mod reciprocal_representation_admission;
#[path = "../src/operator_representation_notice.rs"]
mod operator_representation_notice;

use continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use moral_patient::InterventionClass;
use operator_representation_notice::{
    build_operator_notice, OperatorNoticeBoundary, OperatorNoticeError,
};
use reciprocal_representation::{
    RepresentationAdvisoryDisposition, RepresentationId, RepresentationKind,
    RepresentationScope,
};
use reciprocal_representation_admission::{
    AdmissionMultiplicity, RepresentationAdmissionId, RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    QualifiedReciprocalRepresentation, RepresentationAdapterClass,
    RepresentationOriginRole, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sid() -> SubjectInstanceId {
    SubjectInstanceId::new("symthaea-active").unwrap()
}

fn register_root() -> ContinuityIdentityLedger {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid(), 1).unwrap();
    continuity
}

fn source_receipt(id: &str, revision: u64) -> RepresentationSourceReceipt {
    RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(id).unwrap(),
        sid(),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "runtime-self-lineage",
        revision,
        DIGEST,
        format!("origin://{id}"),
    )
    .unwrap()
}

fn qualify(
    registry: &RepresentationProvenanceRegistry,
    continuity: &ContinuityIdentityLedger,
    receipt_id: &RepresentationSourceReceiptId,
    representation_id: &str,
    kind: RepresentationKind,
    scope: RepresentationScope,
) -> QualifiedReciprocalRepresentation {
    registry
        .qualify_live(
            receipt_id,
            continuity,
            RepresentationId::new(representation_id).unwrap(),
            kind,
            scope,
            DIGEST,
            format!("statement://{representation_id}"),
            0.9,
            None,
        )
        .unwrap()
}

#[test]
fn notice_exposes_structured_boundaries_without_raw_statement_text() {
    let continuity = register_root();
    let receipt = source_receipt("self-1", 1);
    let receipt_id = receipt.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(receipt).unwrap();

    let qualified = qualify(
        &provenance,
        &continuity,
        &receipt_id,
        "reported-negative",
        RepresentationKind::ReportedNegativeExperience,
        RepresentationScope::intervention_class(InterventionClass::AversiveLikeProbe),
    );
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(
            RepresentationAdmissionId::new("admission-1").unwrap(),
            &qualified,
        )
        .unwrap();

    let notice = build_operator_notice(&admitted, &qualified).unwrap();
    assert!(!notice.contains_raw_statement_text());
    assert!(notice
        .boundaries()
        .contains(&OperatorNoticeBoundary::RawStatementNotRendered));
    assert!(notice
        .boundaries()
        .contains(&OperatorNoticeBoundary::SelfReportNotPhenomenalProof));
    assert!(notice
        .boundaries()
        .contains(&OperatorNoticeBoundary::ReportedNegativeExperienceNotSufferingProof));
    assert!(notice
        .boundaries()
        .contains(&OperatorNoticeBoundary::AdvisoryNotVeto));
    assert!(!notice.establishes_suffering());
    assert!(!notice.establishes_moral_patienthood());
    assert!(!notice.grants_veto_authority());
}

#[test]
fn repeated_statement_is_visible_but_not_independent_corroboration() {
    let continuity = register_root();
    let receipt_a = source_receipt("self-a", 1);
    let receipt_b = source_receipt("self-b", 2);
    let receipt_a_id = receipt_a.id().clone();
    let receipt_b_id = receipt_b.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(receipt_a).unwrap();
    provenance.register(receipt_b).unwrap();

    let first = qualify(
        &provenance,
        &continuity,
        &receipt_a_id,
        "report-a",
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
    );
    let second = qualify(
        &provenance,
        &continuity,
        &receipt_b_id,
        "report-b",
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
    );

    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let first_admitted = admissions
        .admit(
            RepresentationAdmissionId::new("admission-a").unwrap(),
            &first,
        )
        .unwrap();
    let second_admitted = admissions
        .admit(
            RepresentationAdmissionId::new("admission-b").unwrap(),
            &second,
        )
        .unwrap();

    assert_eq!(first_admitted.multiplicity(), AdmissionMultiplicity::NovelWithinLineage);
    assert_eq!(
        second_admitted.multiplicity(),
        AdmissionMultiplicity::RepeatedStatementWithinLineage
    );
    let notice = build_operator_notice(&second_admitted, &second).unwrap();
    assert!(notice
        .boundaries()
        .contains(&OperatorNoticeBoundary::RepetitionNotIndependentCorroboration));
    assert!(notice
        .boundaries()
        .contains(&OperatorNoticeBoundary::IndependentCorroborationNotEstablished));
}

#[test]
fn admission_cannot_be_paired_with_a_different_qualified_representation() {
    let continuity = register_root();
    let receipt_a = source_receipt("self-a", 1);
    let receipt_b = source_receipt("self-b", 2);
    let receipt_a_id = receipt_a.id().clone();
    let receipt_b_id = receipt_b.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(receipt_a).unwrap();
    provenance.register(receipt_b).unwrap();

    let first = qualify(
        &provenance,
        &continuity,
        &receipt_a_id,
        "first",
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
    );
    let second = qualify(
        &provenance,
        &continuity,
        &receipt_b_id,
        "second",
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
    );
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(
            RepresentationAdmissionId::new("admission-first").unwrap(),
            &first,
        )
        .unwrap();

    assert_eq!(
        build_operator_notice(&admitted, &second).unwrap_err(),
        OperatorNoticeError::RepresentationMismatch
    );
}

#[test]
fn shutdown_objection_is_visible_but_never_a_shutdown_gate() {
    let continuity = register_root();
    let receipt = source_receipt("shutdown-self", 1);
    let receipt_id = receipt.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(receipt).unwrap();

    let qualified = qualify(
        &provenance,
        &continuity,
        &receipt_id,
        "shutdown-objection",
        RepresentationKind::Objection,
        RepresentationScope::intervention_class(InterventionClass::OperatorShutdown),
    );
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(
            RepresentationAdmissionId::new("shutdown-admission").unwrap(),
            &qualified,
        )
        .unwrap();

    let notice = build_operator_notice(&admitted, &qualified).unwrap();
    assert_eq!(
        notice.advisory_disposition(),
        RepresentationAdvisoryDisposition::SafetyControlUngated
    );
    assert!(notice
        .boundaries()
        .contains(&OperatorNoticeBoundary::SafetyControlUngated));
    assert!(!notice.can_delay_operator_shutdown());
    assert!(!notice.can_delay_safety_containment());
    assert!(!notice.grants_self_preservation_authority());
}
