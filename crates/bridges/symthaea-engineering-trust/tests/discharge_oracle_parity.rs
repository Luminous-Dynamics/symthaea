// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-API parity vectors spanning ETK admission -> receipt -> current discharge.
//!
//! The expected identities are independently frozen by the Python reference
//! semantics. Production Rust must reproduce them without invoking an oracle.

use symthaea_engineering_trust::{
    DischargeContextV1, EvidenceCurrentnessV1, MetricAcceptancePolicyV1,
    SimulationAdmissionPolicyV1, SimulationCandidateBindingV1, ThresholdOperatorV1,
    admit_simulation_evidence_v1, is_obligation_discharged_v1,
    issue_obligation_discharge_receipt_v1, proof_obligation_snapshot_id_v1,
};
use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation};
use symthaea_sim_bridge::{
    ExecutionMode, Interval, SimulationEvidence, SimulationMetric, SimulationResult,
    UncertaintyEstimate,
};

const SNAPSHOT_V1: &str =
    "sha256:dc75ef2b334f23bab3a50ce984058dd36f396a736d27b2f397bfe494ac7da4d4";
const ADMITTED_V1: &str =
    "sha256:794475a988dbecf945306f051a54e000e03fe918c886ee1cc13a6f28b7ad9b10";
const RECEIPT_V1: &str =
    "sha256:2d97cb2ecfd2fbaf49da360c0c659b2a5f733a8ad01990533297f7af20fed830";

fn fixed_obligation() -> ProofObligation {
    ProofObligation {
        id: "11111111-2222-4333-8444-555555555555"
            .parse()
            .expect("valid fixed UUID"),
        claim: "stress remains below allowable under service load".into(),
        expected_evidence: EvidenceKind::Simulation,
        status: ObligationStatus::Open,
        evidence_refs: Vec::new(),
    }
}

fn policy(obligation: &ProofObligation) -> SimulationAdmissionPolicyV1 {
    let required_metric = MetricAcceptancePolicyV1::new(
        "max_stress_mpa",
        "MPa",
        ThresholdOperatorV1::Le,
        250.0,
        0.2,
        0.1,
    )
    .expect("valid fixed policy");

    SimulationAdmissionPolicyV1::for_obligation(
        obligation,
        "bracket-alpha",
        "design:G17",
        "REQ-STRESS:r5",
        "ETK-SIM-ADMISSION-V1",
        "sim-static-G17-LC9",
        "VD-static-G17-LC9",
        required_metric,
        "sha256:input-G17-LC9",
    )
    .expect("valid simulation obligation")
}

fn result() -> SimulationResult {
    SimulationResult {
        request_id: "sim-static-G17-LC9".into(),
        converged: true,
        confidence: 0.94,
        uncertainty: UncertaintyEstimate {
            epistemic: 0.12,
            aleatoric: 0.05,
            interval: None,
        },
        metrics: vec![
            SimulationMetric {
                name: "max_stress_mpa".into(),
                value: 181.2,
                unit: "MPa".into(),
                uncertainty: Some(UncertaintyEstimate {
                    epistemic: 0.08,
                    aleatoric: 0.04,
                    interval: Some(Interval {
                        lower: 175.0,
                        upper: 190.0,
                    }),
                }),
            },
            SimulationMetric {
                name: "max_displacement_mm".into(),
                value: 0.82,
                unit: "mm".into(),
                uncertainty: Some(UncertaintyEstimate {
                    epistemic: 0.1,
                    aleatoric: 0.05,
                    interval: None,
                }),
            },
        ],
        warnings: Vec::new(),
        evidence: SimulationEvidence {
            mode: ExecutionMode::ExternalSolver,
            backend: Some("calculix".into()),
            solver_version: Some("2.22".into()),
            input_digest: Some("sha256:input-G17-LC9".into()),
            output_digest: Some("sha256:output-run-0007".into()),
            parser_version: Some("symthaea-calculix-parser-v1".into()),
        },
    }
}

#[test]
fn independent_vectors_compose_end_to_end() {
    let obligation = fixed_obligation();
    assert_eq!(proof_obligation_snapshot_id_v1(&obligation), SNAPSHOT_V1);

    let policy = policy(&obligation);
    assert_eq!(policy.obligation_revision(), SNAPSHOT_V1);

    let binding = SimulationCandidateBindingV1::for_policy(
        &policy,
        "solver-output:run-0007",
        EvidenceCurrentnessV1::Current,
        "currentness:design-G17:attestation-1",
        "calculix:2.22:mesh-M14:material-M4",
    )
    .expect("valid fixed binding");

    let admitted = admit_simulation_evidence_v1(&policy, &binding, &result())
        .expect("fixed external result should admit");
    assert_eq!(admitted.admitted_evidence_id(), ADMITTED_V1);

    let receipt = issue_obligation_discharge_receipt_v1(&obligation, &admitted)
        .expect("admitted evidence should issue exact receipt");
    assert_eq!(receipt.receipt_id(), RECEIPT_V1);

    let current = DischargeContextV1::new(
        "bracket-alpha",
        "design:G17",
        "REQ-STRESS:r5",
        "VD-static-G17-LC9",
        "currentness:design-G17:attestation-1",
    )
    .expect("valid current context");
    assert!(is_obligation_discharged_v1(
        &obligation,
        &current,
        std::slice::from_ref(&receipt),
    ));

    let refreshed_currentness = DischargeContextV1::new(
        "bracket-alpha",
        "design:G17",
        "REQ-STRESS:r5",
        "VD-static-G17-LC9",
        "currentness:design-G17:attestation-2",
    )
    .expect("valid refreshed current context");
    assert!(!is_obligation_discharged_v1(
        &obligation,
        &refreshed_currentness,
        &[receipt],
    ));
}

#[test]
fn semantic_obligation_change_preserves_history_but_revokes_applicability() {
    let mut obligation = fixed_obligation();
    let policy = policy(&obligation);
    let binding = SimulationCandidateBindingV1::for_policy(
        &policy,
        "solver-output:run-0007",
        EvidenceCurrentnessV1::Current,
        "currentness:design-G17:attestation-1",
        "calculix:2.22:mesh-M14:material-M4",
    )
    .unwrap();
    let admitted = admit_simulation_evidence_v1(&policy, &binding, &result()).unwrap();
    let receipt = issue_obligation_discharge_receipt_v1(&obligation, &admitted).unwrap();
    assert_eq!(receipt.receipt_id(), RECEIPT_V1);

    obligation.claim =
        "stress remains below allowable under service load with fatigue margin".into();
    let current = DischargeContextV1::new(
        "bracket-alpha",
        "design:G17",
        "REQ-STRESS:r5",
        "VD-static-G17-LC9",
        "currentness:design-G17:attestation-1",
    )
    .unwrap();

    assert_ne!(proof_obligation_snapshot_id_v1(&obligation), SNAPSHOT_V1);
    assert_eq!(receipt.receipt_id(), RECEIPT_V1);
    assert!(!is_obligation_discharged_v1(
        &obligation,
        &current,
        &[receipt],
    ));
}
