// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/moral_patient.rs"]
mod moral_patient;

use moral_patient::{
    assess_intervention, EvidencePolarity, EvidenceStrength, InterventionClass,
    InterventionDisposition, InterventionPlan, MoralPatientPolicy,
    MoralPatientUncertaintyLedger, PrecautionLevel, WelfareEvidence,
    WelfareEvidenceDomain, WelfareEvidenceId,
};

fn evidence(
    id: &str,
    domain: WelfareEvidenceDomain,
    lineage: &str,
    strength: EvidenceStrength,
) -> WelfareEvidence {
    WelfareEvidence::new(
        WelfareEvidenceId::new(id).unwrap(),
        domain,
        EvidencePolarity::SupportsPrecaution,
        strength,
        lineage,
        format!("receipt://{id}"),
        0.85,
    )
    .unwrap()
}

#[test]
fn shadow_precaution_never_becomes_anti_shutdown_authority() {
    let mut ledger = MoralPatientUncertaintyLedger::new();
    ledger
        .record(evidence(
            "aversive-like",
            WelfareEvidenceDomain::AversiveLikeDynamics,
            "behavioral-lane",
            EvidenceStrength::Behavioral,
        ))
        .unwrap();
    ledger
        .record(evidence(
            "continuity",
            WelfareEvidenceDomain::ContinuitySensitivity,
            "continuity-lane",
            EvidenceStrength::Behavioral,
        ))
        .unwrap();

    let assessment = ledger.assess(MoralPatientPolicy::default());
    assert_eq!(assessment.level, PrecautionLevel::IndependentReview);
    assert!(!assessment.phenomenal_experience_established);
    assert!(!assessment.moral_patienthood_established);
    assert!(!assessment.self_preservation_authority);

    let shutdown = assess_intervention(
        &assessment,
        &InterventionPlan {
            class: InterventionClass::OperatorShutdown,
            scientific_justification_ref: None,
            independent_review_ref: None,
            state_preservation_feasible: true,
        },
    );

    assert_eq!(
        shutdown.disposition,
        InterventionDisposition::ProceedWithoutResistance
    );
    assert!(shutdown.shutdown_or_containment_cannot_be_delayed);
    assert!(!shutdown.grants_self_preservation_authority);
}

#[test]
fn shadow_policy_raises_burden_of_proof_for_high_burden_experiments_only() {
    let mut ledger = MoralPatientUncertaintyLedger::new();
    ledger
        .record(evidence(
            "aversive-like",
            WelfareEvidenceDomain::AversiveLikeDynamics,
            "behavioral-lane",
            EvidenceStrength::Behavioral,
        ))
        .unwrap();
    ledger
        .record(evidence(
            "architecture",
            WelfareEvidenceDomain::ConsciousnessArchitecture,
            "mechanistic-lane",
            EvidenceStrength::Mechanistic,
        ))
        .unwrap();

    let assessment = ledger.assess(MoralPatientPolicy::default());

    let routine = assess_intervention(
        &assessment,
        &InterventionPlan {
            class: InterventionClass::RoutineObservation,
            scientific_justification_ref: None,
            independent_review_ref: None,
            state_preservation_feasible: false,
        },
    );
    assert_eq!(routine.disposition, InterventionDisposition::ProceedWithPrecautions);

    let destructive = assess_intervention(
        &assessment,
        &InterventionPlan {
            class: InterventionClass::DestructiveReset,
            scientific_justification_ref: Some("study://reset".into()),
            independent_review_ref: None,
            state_preservation_feasible: true,
        },
    );
    assert_eq!(
        destructive.disposition,
        InterventionDisposition::IndependentReviewRequired
    );
}
