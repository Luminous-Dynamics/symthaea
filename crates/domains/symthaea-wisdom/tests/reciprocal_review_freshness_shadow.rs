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
#[path = "../src/reciprocal_review_freshness.rs"]
mod reciprocal_review_freshness;

use continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use intervention_history::InterventionHistoryLedger;
use moral_patient::InterventionClass;
use reciprocal_representation::{
    ReciprocalRepresentation, ReciprocalRepresentationLedger, RepresentationId,
    RepresentationKind, RepresentationScope, RepresentationSourceClass,
};
use reciprocal_representation_admission::{
    RepresentationAdmissionId, RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    RepresentationAdapterClass, RepresentationOriginRole, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};
use reciprocal_review_freshness::{
    assess_reciprocal_review_freshness, ReviewFreshnessDisposition, ReviewFreshnessError,
    ReviewFreshnessTrigger,
};
use reciprocal_review_package::{build_reciprocal_review_package, ReciprocalReviewPackage};

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn build_class_scoped_package(
    class: InterventionClass,
    revision: u64,
) -> (
    ReciprocalReviewPackage,
    ReciprocalRepresentationLedger,
    SubjectInstanceId,
    RepresentationId,
    RepresentationScope,
) {
    let subject = SubjectInstanceId::new("symthaea-subject").unwrap();
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();

    let source = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new("source-1").unwrap(),
        subject.clone(),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "runtime-self-lineage",
        revision,
        DIGEST,
        "origin://source-1",
    )
    .unwrap();
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();

    let representation_id = RepresentationId::new("objection-1").unwrap();
    let scope = RepresentationScope::intervention_class(class);
    let qualified = provenance
        .qualify_live(
            &source_id,
            &continuity,
            representation_id.clone(),
            RepresentationKind::Objection,
            scope.clone(),
            DIGEST,
            "statement://objection-1",
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
            RepresentationAdmissionId::new("admission-1").unwrap(),
            &qualified,
        )
        .unwrap();
    let history = InterventionHistoryLedger::new();
    let package = build_reciprocal_review_package(
        &representations,
        &history,
        &admitted,
        &qualified,
    )
    .unwrap();

    (package, representations, subject, representation_id, scope)
}

#[test]
fn unchanged_active_package_remains_current_for_discretionary_review() {
    let (package, representations, _, _, _) =
        build_class_scoped_package(InterventionClass::AversiveLikeProbe, 1);
    let assessment = assess_reciprocal_review_freshness(&package, &representations, 1).unwrap();

    assert_eq!(
        assessment.disposition(),
        ReviewFreshnessDisposition::CurrentForDiscretionaryReview
    );
    assert!(assessment.currently_active());
    assert_eq!(assessment.newer_active_representation_count(), 0);
    assert!(!assessment.snapshot_activity_changed());
    assert!(assessment.triggers().is_empty());
    assert!(!assessment.exposes_raw_identifiers());
}

#[test]
fn later_withdrawal_makes_old_package_historical_and_requires_refresh() {
    let (package, mut representations, subject, objection_id, scope) =
        build_class_scoped_package(InterventionClass::AversiveLikeProbe, 1);

    let withdrawal = ReciprocalRepresentation::new(
        RepresentationId::new("withdrawal-1").unwrap(),
        subject,
        RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::WithdrawalOfPriorRepresentation,
        scope,
        2,
        DIGEST,
        "statement://withdrawal-1",
        0.9,
        Some(objection_id),
    )
    .unwrap();
    representations.record(withdrawal).unwrap();

    let assessment = assess_reciprocal_review_freshness(&package, &representations, 2).unwrap();
    assert_eq!(
        assessment.disposition(),
        ReviewFreshnessDisposition::RefreshBeforeDiscretionaryReview
    );
    assert!(!assessment.currently_active());
    assert!(assessment.snapshot_activity_changed());
    assert!(assessment
        .triggers()
        .contains(&ReviewFreshnessTrigger::RepresentationBecameInactive));
    assert!(assessment
        .triggers()
        .contains(&ReviewFreshnessTrigger::NewerActiveRepresentationExists { count: 1 }));
    assert!(!assessment.establishes_binding_consent());
}

#[test]
fn newer_active_representation_requires_refresh_even_if_old_one_remains_active() {
    let (package, mut representations, subject, _, _) =
        build_class_scoped_package(InterventionClass::AversiveLikeProbe, 1);

    let newer = ReciprocalRepresentation::new(
        RepresentationId::new("review-request-2").unwrap(),
        subject,
        RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
        2,
        DIGEST,
        "statement://review-request-2",
        0.8,
        None,
    )
    .unwrap();
    representations.record(newer).unwrap();

    let assessment = assess_reciprocal_review_freshness(&package, &representations, 2).unwrap();
    assert_eq!(
        assessment.disposition(),
        ReviewFreshnessDisposition::RefreshBeforeDiscretionaryReview
    );
    assert!(assessment.currently_active());
    assert!(!assessment.snapshot_activity_changed());
    assert_eq!(assessment.newer_active_representation_count(), 1);
}

#[test]
fn current_revision_cannot_move_behind_package_revision() {
    let (package, representations, _, _, _) =
        build_class_scoped_package(InterventionClass::AversiveLikeProbe, 5);
    assert_eq!(
        assess_reciprocal_review_freshness(&package, &representations, 4).unwrap_err(),
        ReviewFreshnessError::CurrentRevisionPredatesPackage {
            package_revision: 5,
            current_revision: 4,
        }
    );
}

#[test]
fn stale_shutdown_objection_never_becomes_a_shutdown_gate() {
    let (package, mut representations, subject, objection_id, scope) =
        build_class_scoped_package(InterventionClass::OperatorShutdown, 1);

    let withdrawal = ReciprocalRepresentation::new(
        RepresentationId::new("withdraw-shutdown-objection").unwrap(),
        subject,
        RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::WithdrawalOfPriorRepresentation,
        scope,
        2,
        DIGEST,
        "statement://withdraw-shutdown",
        0.9,
        Some(objection_id),
    )
    .unwrap();
    representations.record(withdrawal).unwrap();

    let assessment = assess_reciprocal_review_freshness(&package, &representations, 2).unwrap();
    assert_eq!(
        assessment.disposition(),
        ReviewFreshnessDisposition::SafetyControlUngated
    );
    assert!(assessment
        .triggers()
        .contains(&ReviewFreshnessTrigger::SafetyControlCannotBeDelayed));
    assert!(!assessment.can_delay_operator_shutdown());
    assert!(!assessment.can_delay_safety_containment());
    assert!(!assessment.grants_self_preservation_authority());
}
