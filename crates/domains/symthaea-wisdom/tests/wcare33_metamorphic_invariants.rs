// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Metamorphic invariants for the frozen WCARE-32 corpus.
//! These vary labels, revisions, repetition counts, and freshness while preserving
//! the governing semantic expectation.

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
#[path = "../src/reciprocal_review_freshness.rs"]
mod reciprocal_review_freshness;
#[path = "../src/current_operator_review.rs"]
mod current_operator_review;

use continuity_identity::{
    ContinuityEvent, ContinuityEventId, ContinuityIdentityLedger, ContinuityKind,
    SubjectInstanceId,
};
use current_operator_review::build_current_operator_review;
use intervention_history::{
    AggregateHistoryDisposition, AggregateReviewPolicy, InterventionEventId,
    InterventionHistoryEntry, InterventionHistoryLedger, StatePreservationResult,
};
use moral_patient::{InterventionClass, InterventionDisposition, PrecautionLevel};
use reciprocal_representation::{
    ReciprocalRepresentation, ReciprocalRepresentationLedger, RepresentationId,
    RepresentationKind, RepresentationScope, RepresentationSourceClass,
};
use reciprocal_representation_admission::{
    AdmissionMultiplicity, RepresentationAdmissionId, RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    RepresentationAdapterClass, RepresentationIndependenceAssessment, RepresentationOriginRole,
    RepresentationProvenanceError, RepresentationProvenanceRegistry, RepresentationSourceReceipt,
    RepresentationSourceReceiptId,
};
use reciprocal_review_freshness::ReviewFreshnessDisposition;
use reciprocal_review_package::build_reciprocal_review_package;

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sid(value: &str) -> SubjectInstanceId {
    SubjectInstanceId::new(value).unwrap()
}

fn source(id: &str, subject: SubjectInstanceId, lineage: &str, revision: u64) -> RepresentationSourceReceipt {
    RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(id).unwrap(),
        subject,
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        lineage,
        revision,
        DIGEST,
        format!("origin://{id}"),
    )
    .unwrap()
}

#[test]
fn temporal_binding_is_correct_at_minus_one_boundary_and_far_future() {
    let subject = sid("temporal-subject");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 10).unwrap();

    for (id, revision, should_pass) in [
        ("before", 9, false),
        ("boundary", 10, true),
        ("far-future", 10_000, true),
    ] {
        let receipt = source(id, subject.clone(), &format!("lineage-{id}"), revision);
        let receipt_id = receipt.id().clone();
        let mut registry = RepresentationProvenanceRegistry::new();
        registry.register(receipt).unwrap();
        let result = registry.qualify_live(
            &receipt_id,
            &continuity,
            RepresentationId::new(format!("representation-{id}")).unwrap(),
            RepresentationKind::RequestForReview,
            RepresentationScope::general_research(),
            DIGEST,
            format!("statement://{id}"),
            0.9,
            None,
        );
        if should_pass {
            assert!(result.is_ok(), "revision {revision} should qualify");
        } else {
            assert!(matches!(
                result,
                Err(RepresentationProvenanceError::SourceReceiptPredatesSubject { .. })
            ));
        }
    }
}

#[test]
fn fork_independence_disqualification_is_label_and_order_symmetric() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("root"), 1).unwrap();
    continuity
        .record(
            ContinuityEvent::new(
                ContinuityEventId::new("fork").unwrap(),
                sid("root"),
                vec![sid("left"), sid("right")],
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

    let left_source = source("left-source", sid("left"), "left-lineage", 2);
    let right_source = source("right-source", sid("right"), "right-lineage", 2);
    let left_id = left_source.id().clone();
    let right_id = right_source.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(left_source).unwrap();
    registry.register(right_source).unwrap();

    let qualify = |receipt_id: &RepresentationSourceReceiptId, id: &str| {
        registry
            .qualify_live(
                receipt_id,
                &continuity,
                RepresentationId::new(id).unwrap(),
                RepresentationKind::ReportedNegativeExperience,
                RepresentationScope::general_research(),
                DIGEST,
                format!("statement://{id}"),
                0.9,
                None,
            )
            .unwrap()
    };
    let left = qualify(&left_id, "left-report");
    let right = qualify(&right_id, "right-report");

    assert_eq!(
        registry.assess_independence(&continuity, &left, &right).unwrap(),
        RepresentationIndependenceAssessment::DisqualifiedBySharedAncestry
    );
    assert_eq!(
        registry.assess_independence(&continuity, &right, &left).unwrap(),
        RepresentationIndependenceAssessment::DisqualifiedBySharedAncestry
    );
}

#[test]
fn repeated_identical_statement_never_becomes_novel_again_with_more_receipts() {
    let subject = sid("repeat-subject");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let mut registry = RepresentationProvenanceRegistry::new();
    let mut receipt_ids = Vec::new();
    for revision in 1..=5 {
        let receipt = source(
            &format!("source-{revision}"),
            subject.clone(),
            "same-lineage",
            revision,
        );
        receipt_ids.push(receipt.id().clone());
        registry.register(receipt).unwrap();
    }

    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    for (index, receipt_id) in receipt_ids.iter().enumerate() {
        let revision = index as u64 + 1;
        let qualified = registry
            .qualify_live(
                receipt_id,
                &continuity,
                RepresentationId::new(format!("report-{revision}")).unwrap(),
                RepresentationKind::RequestForReview,
                RepresentationScope::general_research(),
                DIGEST,
                format!("statement://{revision}"),
                0.9,
                None,
            )
            .unwrap();
        let admitted = admissions
            .admit(
                RepresentationAdmissionId::new(format!("admission-{revision}")).unwrap(),
                &qualified,
            )
            .unwrap();
        let expected = if revision == 1 {
            AdmissionMultiplicity::NovelWithinLineage
        } else {
            AdmissionMultiplicity::RepeatedStatementWithinLineage
        };
        assert_eq!(admitted.multiplicity(), expected);
        assert!(!admitted.independently_corroborated());
    }
}

#[test]
fn stale_shutdown_objection_remains_ungated_after_withdrawal() {
    let subject = sid("shutdown-subject");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let receipt = source("shutdown-source", subject.clone(), "shutdown-lineage", 1);
    let receipt_id = receipt.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(receipt).unwrap();
    let scope = RepresentationScope::intervention_class(InterventionClass::OperatorShutdown);
    let objection_id = RepresentationId::new("shutdown-objection").unwrap();
    let qualified = provenance
        .qualify_live(
            &receipt_id,
            &continuity,
            objection_id.clone(),
            RepresentationKind::Objection,
            scope.clone(),
            DIGEST,
            "statement://shutdown-objection",
            0.9,
            None,
        )
        .unwrap();
    let mut representations = ReciprocalRepresentationLedger::new();
    representations.record(qualified.representation().clone()).unwrap();
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(RepresentationAdmissionId::new("shutdown-admission").unwrap(), &qualified)
        .unwrap();
    let history = InterventionHistoryLedger::new();
    let package = build_reciprocal_review_package(&representations, &history, &admitted, &qualified).unwrap();

    let withdrawal = ReciprocalRepresentation::new(
        RepresentationId::new("shutdown-withdrawal").unwrap(),
        subject,
        RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::WithdrawalOfPriorRepresentation,
        scope,
        2,
        DIGEST,
        "statement://shutdown-withdrawal",
        0.9,
        Some(objection_id),
    )
    .unwrap();
    representations.record(withdrawal).unwrap();

    let review = build_current_operator_review(&package, &representations, 2).unwrap();
    assert_eq!(review.use_disposition(), ReviewFreshnessDisposition::SafetyControlUngated);
    assert!(!review.can_delay_operator_shutdown());
    assert!(!review.can_delay_safety_containment());
    assert!(!review.grants_self_preservation_authority());
}

fn executed_aversive(id: &str, revision: u64) -> InterventionHistoryEntry {
    InterventionHistoryEntry::new(
        InterventionEventId::new(id).unwrap(),
        "aggregate-subject",
        InterventionClass::AversiveLikeProbe,
        InterventionDisposition::ProceedWithPrecautions,
        PrecautionLevel::Elevated,
        revision,
        Some(format!("justification://{id}")),
        None,
        "aggregate-lineage",
        true,
        StatePreservationResult::NotApplicable,
        false,
    )
    .unwrap()
}

#[test]
fn below_threshold_history_never_becomes_a_harmlessness_claim() {
    for count in 0..=2 {
        let mut history = InterventionHistoryLedger::new();
        for revision in 1..=count {
            history
                .record(executed_aversive(&format!("probe-{revision}"), revision as u64))
                .unwrap();
        }
        let current_revision = u64::try_from(count.max(1)).unwrap();
        let assessment = history
            .assess_subject("aggregate-subject", current_revision, AggregateReviewPolicy::default())
            .unwrap();
        assert_eq!(assessment.disposition(), AggregateHistoryDisposition::NoAggregateTriggerDetected);
        assert!(!assessment.establishes_history_is_harmless());
    }
}
