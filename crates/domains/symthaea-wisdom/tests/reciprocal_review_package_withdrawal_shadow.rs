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
use intervention_history::InterventionHistoryLedger;
use moral_patient::InterventionClass;
use reciprocal_representation::{
    ReciprocalRepresentationError, ReciprocalRepresentationLedger, RepresentationId,
    RepresentationKind, RepresentationScope,
};
use reciprocal_representation_admission::{
    RepresentationAdmissionId, RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    RepresentationAdapterClass, RepresentationOriginRole, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};
use reciprocal_review_package::{
    build_reciprocal_review_package, ReciprocalReviewPackageError,
};

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn nonexistent_withdrawal_target_never_reaches_review_package() {
    let subject = SubjectInstanceId::new("symthaea-subject").unwrap();
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();

    let source = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new("withdraw-source").unwrap(),
        subject,
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "runtime-self-lineage",
        2,
        DIGEST,
        "origin://withdraw",
    )
    .unwrap();
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();

    // Low-level semantics require a target ID but cannot know whether it exists.
    let qualified = provenance
        .qualify_live(
            &source_id,
            &continuity,
            RepresentationId::new("invented-withdrawal").unwrap(),
            RepresentationKind::WithdrawalOfPriorRepresentation,
            RepresentationScope::intervention_class(InterventionClass::AversiveLikeProbe),
            DIGEST,
            "statement://withdraw",
            0.9,
            Some(RepresentationId::new("never-existed").unwrap()),
        )
        .unwrap();

    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(
            RepresentationAdmissionId::new("withdraw-admission").unwrap(),
            &qualified,
        )
        .unwrap();

    // WCARE-20 lifecycle validation is the authoritative withdrawal-state check.
    let mut representations = ReciprocalRepresentationLedger::new();
    assert!(matches!(
        representations.record(qualified.representation().clone()),
        Err(ReciprocalRepresentationError::UnknownWithdrawalTarget(_))
    ));

    // Because lifecycle admission failed, the atomic review package also fails closed.
    let history = InterventionHistoryLedger::new();
    assert_eq!(
        build_reciprocal_review_package(
            &representations,
            &history,
            &admitted,
            &qualified,
        )
        .unwrap_err(),
        ReciprocalReviewPackageError::RepresentationNotRecorded
    );
}
