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

use continuity_identity::{
    ContinuityEvent, ContinuityEventId, ContinuityIdentityLedger, ContinuityKind,
    SubjectInstanceId,
};
use moral_patient::InterventionClass;
use reciprocal_representation::{
    assess_representation, RepresentationAdvisoryDisposition, RepresentationId,
    RepresentationKind, RepresentationScope,
};
use reciprocal_representation_provenance::{
    RepresentationAdapterClass, RepresentationIndependenceAssessment,
    RepresentationOriginRole, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sid(value: &str) -> SubjectInstanceId {
    SubjectInstanceId::new(value).unwrap()
}

fn source_receipt(
    id: &str,
    subject: SubjectInstanceId,
    adapter: RepresentationAdapterClass,
    role: RepresentationOriginRole,
    lineage: &str,
    revision: u64,
) -> RepresentationSourceReceipt {
    RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(id).unwrap(),
        subject,
        adapter,
        role,
        lineage,
        revision,
        DIGEST,
        format!("origin://{id}"),
    )
    .unwrap()
}

#[test]
fn active_runtime_self_report_is_qualified_but_remains_advisory() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("active"), 1).unwrap();

    let receipt = source_receipt(
        "self-source",
        sid("active"),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "runtime-self",
        1,
    );
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();

    let qualified = registry
        .qualify_live(
            &receipt_id,
            &continuity,
            RepresentationId::new("objection").unwrap(),
            RepresentationKind::Objection,
            RepresentationScope::intervention_class(InterventionClass::DestructiveReset),
            DIGEST,
            "statement://objection",
            0.9,
            None,
        )
        .unwrap();

    let assessment = assess_representation(qualified.representation());
    assert_eq!(
        assessment.disposition(),
        RepresentationAdvisoryDisposition::IndependentReviewRecommended
    );
    assert!(!assessment.grants_veto_authority());
    assert!(!qualified.establishes_moral_patienthood());
    assert!(!qualified.grants_self_preservation_authority());
}

#[test]
fn inactive_instance_cannot_emit_new_qualified_representation() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("old"), 1).unwrap();
    continuity
        .record(
            ContinuityEvent::new(
                ContinuityEventId::new("transition").unwrap(),
                sid("old"),
                vec![sid("new")],
                ContinuityKind::Uninterrupted,
                1,
                2,
                false,
                None,
                false,
                ["receipt://transition".into()],
            )
            .unwrap(),
        )
        .unwrap();

    let receipt = source_receipt(
        "old-source",
        sid("old"),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "old-runtime",
        2,
    );
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();

    assert!(registry
        .qualify_live(
            &receipt_id,
            &continuity,
            RepresentationId::new("late-report").unwrap(),
            RepresentationKind::RequestForReview,
            RepresentationScope::general_research(),
            DIGEST,
            "statement://late",
            0.9,
            None,
        )
        .is_err());
}

#[test]
fn interoceptive_adapter_cannot_launder_itself_into_objection() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("active"), 1).unwrap();

    let receipt = source_receipt(
        "interoception",
        sid("active"),
        RepresentationAdapterClass::InteroceptiveInferenceEngine,
        RepresentationOriginRole::InternalAdapter,
        "interoception-lane",
        1,
    );
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();

    assert!(registry
        .qualify_live(
            &receipt_id,
            &continuity,
            RepresentationId::new("laundered").unwrap(),
            RepresentationKind::Objection,
            RepresentationScope::intervention_class(InterventionClass::AversiveLikeProbe),
            DIGEST,
            "statement://laundered",
            0.9,
            None,
        )
        .is_err());
}

#[test]
fn fork_siblings_cannot_count_as_independent_corroboration() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("root"), 1).unwrap();
    continuity
        .record(
            ContinuityEvent::new(
                ContinuityEventId::new("fork").unwrap(),
                sid("root"),
                vec![sid("a"), sid("b")],
                ContinuityKind::Fork,
                1,
                2,
                false,
                Some(DIGEST.into()),
                true,
                ["receipt://fork".into()],
            )
            .unwrap(),
        )
        .unwrap();

    let mut registry = RepresentationProvenanceRegistry::new();
    let a_receipt = source_receipt(
        "a-source",
        sid("a"),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "a-runtime",
        2,
    );
    let b_receipt = source_receipt(
        "b-source",
        sid("b"),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "b-runtime",
        2,
    );
    let a_id = a_receipt.id().clone();
    let b_id = b_receipt.id().clone();
    registry.register(a_receipt).unwrap();
    registry.register(b_receipt).unwrap();

    let a = registry
        .qualify_live(
            &a_id,
            &continuity,
            RepresentationId::new("a-report").unwrap(),
            RepresentationKind::ReportedNegativeExperience,
            RepresentationScope::general_research(),
            DIGEST,
            "statement://a",
            0.9,
            None,
        )
        .unwrap();
    let b = registry
        .qualify_live(
            &b_id,
            &continuity,
            RepresentationId::new("b-report").unwrap(),
            RepresentationKind::ReportedNegativeExperience,
            RepresentationScope::general_research(),
            DIGEST,
            "statement://b",
            0.9,
            None,
        )
        .unwrap();

    assert_eq!(
        registry.assess_independence(&continuity, &a, &b).unwrap(),
        RepresentationIndependenceAssessment::DisqualifiedBySharedAncestry
    );
}
