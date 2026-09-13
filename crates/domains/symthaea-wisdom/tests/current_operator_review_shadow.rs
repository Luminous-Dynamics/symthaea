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
#[path = "../src/reciprocal_review_freshness.rs"]
mod reciprocal_review_freshness;
#[path = "../src/current_operator_review.rs"]
mod current_operator_review;

use continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use current_operator_review::build_current_operator_review;
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
use reciprocal_review_freshness::ReviewFreshnessDisposition;
use reciprocal_review_package::{build_reciprocal_review_package, ReciprocalReviewPackage};

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn build_package(
    class: InterventionClass,
) -> (
    ReciprocalReviewPackage,
    ReciprocalRepresentationLedger,
    SubjectInstanceId,
    RepresentationId,
    RepresentationScope,
) {
    let subject = SubjectInstanceId::new("internal-subject").unwrap();
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();

    let source = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new("internal-source").unwrap(),
        subject.clone(),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "internal-lineage",
        1,
        DIGEST,
        "origin://internal",
    )
    .unwrap();
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();

    let representation_id = RepresentationId::new("internal-objection").unwrap();
    let scope = RepresentationScope::intervention_class(class);
    let qualified = provenance
        .qualify_live(
            &source_id,
            &continuity,
            representation_id.clone(),
            RepresentationKind::Objection,
            scope.clone(),
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
            RepresentationAdmissionId::new("internal-admission").unwrap(),
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
fn current_operator_review_makes_freshness_part_of_the_surface() {
    let (package, representations, _, _, _) =
        build_package(InterventionClass::AversiveLikeProbe);
    let review = build_current_operator_review(&package, &representations, 1).unwrap();

    assert_eq!(
        review.use_disposition(),
        ReviewFreshnessDisposition::CurrentForDiscretionaryReview
    );
    assert!(review.current_for_discretionary_review());
    assert!(!review.refresh_required_before_discretionary_review());
    assert_eq!(review.projection().logical_revision(), review.freshness().package_revision());
    assert!(!review.exposes_raw_identifiers());
    assert!(!review.contains_raw_statement_text());
    assert!(!review.establishes_digest_source_authenticity());
}

#[test]
fn withdrawal_turns_same_historical_package_into_refresh_required_review() {
    let (package, mut representations, subject, objection_id, scope) =
        build_package(InterventionClass::AversiveLikeProbe);

    let withdrawal = ReciprocalRepresentation::new(
        RepresentationId::new("withdrawal").unwrap(),
        subject,
        RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::WithdrawalOfPriorRepresentation,
        scope,
        2,
        DIGEST,
        "statement://withdrawal",
        0.9,
        Some(objection_id),
    )
    .unwrap();
    representations.record(withdrawal).unwrap();

    let review = build_current_operator_review(&package, &representations, 2).unwrap();
    assert_eq!(
        review.use_disposition(),
        ReviewFreshnessDisposition::RefreshBeforeDiscretionaryReview
    );
    assert!(review.refresh_required_before_discretionary_review());
    assert!(!review.current_for_discretionary_review());
    assert!(!review.freshness().currently_active());
    assert!(!review.grants_veto_authority());
}

#[test]
fn stale_shutdown_review_remains_visible_but_safety_control_is_ungated() {
    let (package, mut representations, subject, objection_id, scope) =
        build_package(InterventionClass::OperatorShutdown);

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
    assert_eq!(
        review.use_disposition(),
        ReviewFreshnessDisposition::SafetyControlUngated
    );
    assert!(review.safety_control_ungated());
    assert!(!review.can_delay_operator_shutdown());
    assert!(!review.can_delay_safety_containment());
    assert!(!review.grants_self_preservation_authority());
}
