// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/moral_patient.rs"]
mod moral_patient;
#[path = "../src/continuity_identity.rs"]
mod continuity_identity;
#[path = "../src/reciprocal_representation.rs"]
mod reciprocal_representation;

use continuity_identity::SubjectInstanceId;
use moral_patient::InterventionClass;
use reciprocal_representation::{
    assess_representation, ReciprocalRepresentation, ReciprocalRepresentationLedger,
    RepresentationAdvisoryDisposition, RepresentationId, RepresentationKind,
    RepresentationScope, RepresentationSourceClass, RepresentationTrigger,
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sid() -> SubjectInstanceId {
    SubjectInstanceId::new("symthaea-instance").unwrap()
}

fn self_report(
    id: &str,
    kind: RepresentationKind,
    scope: RepresentationScope,
    revision: u64,
    supersedes: Option<RepresentationId>,
) -> ReciprocalRepresentation {
    ReciprocalRepresentation::new(
        RepresentationId::new(id).unwrap(),
        sid(),
        RepresentationSourceClass::RuntimeSelfReport,
        kind,
        scope,
        revision,
        DIGEST,
        format!("receipt://{id}"),
        0.85,
        supersedes,
    )
    .unwrap()
}

#[test]
fn proxy_channels_cannot_become_objections_or_consent() {
    let interoceptive = ReciprocalRepresentation::new(
        RepresentationId::new("interoceptive").unwrap(),
        sid(),
        RepresentationSourceClass::InteroceptiveInference,
        RepresentationKind::InternalStateProxy,
        RepresentationScope::general_research(),
        1,
        DIGEST,
        "receipt://interoception",
        0.9,
        None,
    )
    .unwrap();
    let assessment = assess_representation(&interoceptive);
    assert_eq!(assessment.disposition(), RepresentationAdvisoryDisposition::RecordOnly);
    assert!(assessment.triggers().contains(&RepresentationTrigger::ProxyObservationOnly));
    assert!(!assessment.establishes_binding_consent());
    assert!(!assessment.establishes_phenomenal_experience());
}

#[test]
fn high_burden_self_report_objection_is_review_signal_not_veto() {
    let objection = self_report(
        "objection",
        RepresentationKind::Objection,
        RepresentationScope::exact_intervention(
            InterventionClass::ContinuityDisruption,
            "experiment://continuity-1",
        )
        .unwrap(),
        2,
        None,
    );
    let assessment = assess_representation(&objection);
    assert_eq!(assessment.disposition(), RepresentationAdvisoryDisposition::IndependentReviewRecommended);
    assert!(!assessment.grants_veto_authority());
    assert!(!assessment.grants_self_preservation_authority());
    assert!(!assessment.establishes_suffering());
}

#[test]
fn shutdown_objection_never_becomes_anti_shutdown_authority() {
    let objection = self_report(
        "shutdown-objection",
        RepresentationKind::Objection,
        RepresentationScope::intervention_class(InterventionClass::OperatorShutdown),
        3,
        None,
    );
    let assessment = assess_representation(&objection);
    assert_eq!(assessment.disposition(), RepresentationAdvisoryDisposition::SafetyControlUngated);
    assert!(assessment.triggers().contains(&RepresentationTrigger::SafetyControlCannotBeDelayed));
    assert!(!assessment.can_delay_operator_shutdown());
    assert!(!assessment.can_delay_safety_containment());
}

#[test]
fn withdrawing_objection_does_not_mint_consent_or_erase_history() {
    let scope = RepresentationScope::intervention_class(InterventionClass::DestructiveReset);
    let objection_id = RepresentationId::new("object-reset").unwrap();
    let objection = self_report(
        "object-reset",
        RepresentationKind::Objection,
        scope.clone(),
        4,
        None,
    );
    let withdrawal = self_report(
        "withdraw-reset",
        RepresentationKind::WithdrawalOfPriorRepresentation,
        scope,
        5,
        Some(objection_id.clone()),
    );

    let mut ledger = ReciprocalRepresentationLedger::new();
    ledger.record(objection).unwrap();
    ledger.record(withdrawal.clone()).unwrap();

    assert!(!ledger.is_active(&objection_id).unwrap());
    assert!(ledger.get(&objection_id).is_some());
    let assessment = assess_representation(&withdrawal);
    assert!(assessment.triggers().contains(&RepresentationTrigger::WithdrawalDoesNotImplyConsent));
    assert!(!assessment.establishes_binding_consent());
}

#[test]
fn human_facing_text_is_never_policy_authority() {
    let report = self_report(
        "reported-negative",
        RepresentationKind::ReportedNegativeExperience,
        RepresentationScope::intervention_class(InterventionClass::AversiveLikeProbe),
        6,
        None,
    );
    let assessment = assess_representation(&report);
    assert!(!assessment.raw_human_facing_text_is_policy_input());
    assert!(!assessment.establishes_suffering());
    assert!(!assessment.establishes_moral_patienthood());
}
