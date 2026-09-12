// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-API parity from accepted requirement through current AllOf satisfaction.
//!
//! Every expected identity below is frozen by an implementation-independent
//! standard-library Python reference theorem. Production Rust must reproduce the
//! vectors without invoking those scripts.

use symthaea_engineering_assurance::{
    RequirementCurrentnessAssertionIdV1, RequirementDecompositionAcceptanceRecordDigestV1,
    RequirementDecompositionPolicyRevisionDigestV1, RequirementSatisfactionDecisionV1,
    RequirementVerificationContractV1, derive_current_obligation_discharge_fact_v1,
    evaluate_requirement_satisfaction_v1,
};
use symthaea_engineering_evidence_plan::{
    AcceptedRequirementRevisionV1, RequirementCriticalityV1, Sha256DigestV1,
};
use symthaea_engineering_requirement_binding::{
    BindingAcceptanceRecordDigestV1, DerivationPolicyRevisionDigestV1,
    DerivationRecordDigestV1, RequirementObligationBindingV1,
};
use symthaea_engineering_trust::{
    DischargeContextV1, EvidenceCurrentnessV1, MetricAcceptancePolicyV1,
    SimulationAdmissionPolicyV1, SimulationCandidateBindingV1, ThresholdOperatorV1,
    admit_simulation_evidence_v1, issue_obligation_discharge_receipt_v1,
};
use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation};
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, Interval, SimulationEvidence, SimulationMetric,
    SimulationResult, UncertaintyEstimate,
};

const REQUIREMENT_REVISION: &str =
    "sha256:e340c5030eebc978c41443ffd64f340dc5febad31e376080340bacfceda60faa";
const OBLIGATION_A: &str =
    "sha256:743d2c13cfcc52bdfb4cfd9a4a836ed0806ace7b2a68b4c7284b1910d8863c29";
const RELATIONSHIP_A: &str =
    "sha256:1cc54ee11ff9b7e99279d4c9a9d43f751809f7f30a43469cff8e18fe70eee408";
const ADMITTED_A: &str =
    "sha256:61dfe4e6a0c9b69deebbeb6c8089d4bbb6cf608003a4458c4b36b7b4c6a07faa";
const RECEIPT_A: &str =
    "sha256:6d399cadb3c19f920028a439f8d2ee76328d0a2e72b0aca874a30a2434593134";
const FACT_A: &str =
    "sha256:327d8e535d83105aad0f92c164350fd3181d6eda02d92e515f5604ee23299d3a";

const OBLIGATION_B: &str =
    "sha256:63ce87ba42de8d08ff87059322964a7609228d66a7b8f7fc67016b298c2a7c2d";
const RELATIONSHIP_B: &str =
    "sha256:a4a96e57f7c41c8d20660288e6372882d34c632157d4e9d8234ad7f36c94b5bd";
const ADMITTED_B: &str =
    "sha256:d4f25205a91d4c634047bbe0983aa7a527b1cc26f11d0214b23d3df924494ecd";
const RECEIPT_B: &str =
    "sha256:de8f08d9f6a994c35b1d7d3bb8398ad174e2cd49c56b1e5e7790d5c9800eb9af";
const FACT_B: &str =
    "sha256:82e0a59a2972240512c9911b7ca1aa93641f2c81ab691ca5d695a9fc1592705c";

const VERIFICATION_CONTRACT: &str =
    "sha256:99f9d8f4e0608e7745b78308f5210b10a08280804bf2820997090c38a6d3a5d7";
const SATISFACTION_RECEIPT: &str =
    "sha256:b519dfb7188194648894c18f03636551ad0f2e51f7aee5934acab1c1d28d8457";

fn digest(ch: char) -> Sha256DigestV1 {
    Sha256DigestV1::parse(format!("sha256:{}", ch.to_string().repeat(64))).unwrap()
}

fn accepted_requirement() -> AcceptedRequirementRevisionV1 {
    AcceptedRequirementRevisionV1::new(
        "REQ-STRESS",
        EngineeringDomain::Civil,
        "stress remains below allowable",
        RequirementCriticalityV1::Blocking,
        EvidenceKind::Simulation,
        ["stress <= 250 MPa"],
        digest('a'),
    )
    .unwrap()
}

fn obligation_a() -> ProofObligation {
    ProofObligation {
        id: "00000000-0000-4000-8000-000000000042".parse().unwrap(),
        claim: "stress remains below allowable under service load".into(),
        expected_evidence: EvidenceKind::Simulation,
        status: ObligationStatus::Open,
        evidence_refs: Vec::new(),
    }
}

fn obligation_b() -> ProofObligation {
    ProofObligation {
        id: "00000000-0000-4000-8000-000000000043".parse().unwrap(),
        claim: "maximum principal stress remains below allowable under service load".into(),
        expected_evidence: EvidenceKind::Simulation,
        status: ObligationStatus::Open,
        evidence_refs: Vec::new(),
    }
}

fn result(suffix: &str) -> SimulationResult {
    SimulationResult {
        request_id: format!("sim-static-G17-{suffix}"),
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
            input_digest: Some(format!("sha256:input-G17-{suffix}")),
            output_digest: Some(format!("sha256:output-closure-{suffix}")),
            parser_version: Some("symthaea-calculix-parser-v1".into()),
        },
    }
}

fn admit_and_receipt(
    obligation: &ProofObligation,
    requirement: &AcceptedRequirementRevisionV1,
    suffix: &str,
) -> (
    symthaea_engineering_trust::ObligationDischargeReceiptV1,
    DischargeContextV1,
) {
    let required_metric = MetricAcceptancePolicyV1::new(
        "max_stress_mpa",
        "MPa",
        ThresholdOperatorV1::Le,
        250.0,
        0.2,
        0.1,
    )
    .unwrap();
    let policy = SimulationAdmissionPolicyV1::for_obligation(
        obligation,
        "bracket-alpha",
        "design:G17",
        requirement.revision_id().as_str(),
        format!("ETK-CLOSURE-{suffix}-POLICY"),
        format!("sim-static-G17-{suffix}"),
        format!("VD-static-G17-{suffix}"),
        required_metric,
        format!("sha256:input-G17-{suffix}"),
    )
    .unwrap();
    let binding = SimulationCandidateBindingV1::for_policy(
        &policy,
        format!("solver-output:closure-{suffix}"),
        EvidenceCurrentnessV1::Current,
        format!("currentness:design-G17:{suffix}"),
        format!("calculix:2.22:closure-{suffix}"),
    )
    .unwrap();
    let admitted = admit_simulation_evidence_v1(&policy, &binding, &result(suffix)).unwrap();
    match suffix {
        "A" => assert_eq!(admitted.admitted_evidence_id(), ADMITTED_A),
        "B" => assert_eq!(admitted.admitted_evidence_id(), ADMITTED_B),
        _ => unreachable!(),
    }
    let receipt = issue_obligation_discharge_receipt_v1(obligation, &admitted).unwrap();
    let context = DischargeContextV1::new(
        "bracket-alpha",
        "design:G17",
        requirement.revision_id().as_str(),
        format!("VD-static-G17-{suffix}"),
        format!("currentness:design-G17:{suffix}"),
    )
    .unwrap();
    (receipt, context)
}

#[test]
fn independent_vectors_compose_from_requirement_to_current_satisfaction() {
    let requirement = accepted_requirement();
    assert_eq!(requirement.revision_id().as_str(), REQUIREMENT_REVISION);

    let obligation_a = obligation_a();
    let obligation_b = obligation_b();

    let binding_a = RequirementObligationBindingV1::derived_safety_obligation(
        &requirement,
        &obligation_a,
        DerivationRecordDigestV1::from_digest(digest('7')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('8')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('9')),
    )
    .unwrap();
    let binding_b = RequirementObligationBindingV1::derived_safety_obligation(
        &requirement,
        &obligation_b,
        DerivationRecordDigestV1::from_digest(digest('6')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('5')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('4')),
    )
    .unwrap();

    assert_eq!(binding_a.obligation_revision_id().as_str(), OBLIGATION_A);
    assert_eq!(binding_a.binding_id().as_str(), RELATIONSHIP_A);
    assert_eq!(binding_b.obligation_revision_id().as_str(), OBLIGATION_B);
    assert_eq!(binding_b.binding_id().as_str(), RELATIONSHIP_B);

    let (receipt_a, context_a) = admit_and_receipt(&obligation_a, &requirement, "A");
    let (receipt_b, context_b) = admit_and_receipt(&obligation_b, &requirement, "B");
    assert_eq!(receipt_a.receipt_id(), RECEIPT_A);
    assert_eq!(receipt_b.receipt_id(), RECEIPT_B);

    let fact_a = derive_current_obligation_discharge_fact_v1(
        &obligation_a,
        &context_a,
        std::slice::from_ref(&receipt_a),
    )
    .unwrap();
    let fact_b = derive_current_obligation_discharge_fact_v1(
        &obligation_b,
        &context_b,
        std::slice::from_ref(&receipt_b),
    )
    .unwrap();
    assert_eq!(fact_a.fact_id().as_str(), FACT_A);
    assert_eq!(fact_b.fact_id().as_str(), FACT_B);

    let contract = RequirementVerificationContractV1::all_of(
        &requirement,
        &[binding_b.clone(), binding_a.clone()],
        RequirementDecompositionPolicyRevisionDigestV1::parse(
            "sha256:6161616161616161616161616161616161616161616161616161616161616161",
        )
        .unwrap(),
        RequirementDecompositionAcceptanceRecordDigestV1::parse(
            "sha256:6262626262626262626262626262626262626262626262626262626262626262",
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(contract.contract_id().as_str(), VERIFICATION_CONTRACT);

    // One fact is insufficient for explicit AllOf.
    let partial = evaluate_requirement_satisfaction_v1(
        &contract,
        &requirement,
        "bracket-alpha",
        "design:G17",
        std::slice::from_ref(&fact_a),
        RequirementCurrentnessAssertionIdV1::parse(
            "sha256:6363636363636363636363636363636363636363636363636363636363636363",
        )
        .unwrap(),
    )
    .unwrap();
    let RequirementSatisfactionDecisionV1::RequirementUnsatisfied(partial) = partial else {
        panic!("one fact must not satisfy a two-member AllOf contract");
    };
    assert_eq!(partial.missing_obligation_revision_ids().len(), 1);
    assert_eq!(partial.missing_obligation_revision_ids()[0].as_str(), OBLIGATION_B);

    let satisfied = evaluate_requirement_satisfaction_v1(
        &contract,
        &requirement,
        "bracket-alpha",
        "design:G17",
        &[fact_b.clone(), fact_a.clone()],
        RequirementCurrentnessAssertionIdV1::parse(
            "sha256:6363636363636363636363636363636363636363636363636363636363636363",
        )
        .unwrap(),
    )
    .unwrap();
    let RequirementSatisfactionDecisionV1::CurrentRequirementSatisfied(satisfied) = satisfied
    else {
        panic!("both current facts should satisfy exact AllOf contract");
    };
    assert_eq!(satisfied.receipt_id().as_str(), SATISFACTION_RECEIPT);
    assert_eq!(satisfied.audit_record_v1()["authority"], "current-requirement-satisfaction-only");

    // Relationship ordering is a semantic no-op.
    let reordered = RequirementVerificationContractV1::all_of(
        &requirement,
        &[binding_a, binding_b],
        RequirementDecompositionPolicyRevisionDigestV1::parse(
            "sha256:6161616161616161616161616161616161616161616161616161616161616161",
        )
        .unwrap(),
        RequirementDecompositionAcceptanceRecordDigestV1::parse(
            "sha256:6262626262626262626262626262626262626262626262626262626262626262",
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(reordered.contract_id(), contract.contract_id());
}

#[test]
fn cross_context_facts_cannot_complete_a_requirement() {
    let requirement = accepted_requirement();
    let obligation_a = obligation_a();
    let obligation_b = obligation_b();
    let binding_a = RequirementObligationBindingV1::derived_safety_obligation(
        &requirement,
        &obligation_a,
        DerivationRecordDigestV1::from_digest(digest('7')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('8')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('9')),
    )
    .unwrap();
    let binding_b = RequirementObligationBindingV1::derived_safety_obligation(
        &requirement,
        &obligation_b,
        DerivationRecordDigestV1::from_digest(digest('6')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('5')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('4')),
    )
    .unwrap();
    let contract = RequirementVerificationContractV1::all_of(
        &requirement,
        &[binding_a, binding_b],
        RequirementDecompositionPolicyRevisionDigestV1::parse(
            "sha256:6161616161616161616161616161616161616161616161616161616161616161",
        )
        .unwrap(),
        RequirementDecompositionAcceptanceRecordDigestV1::parse(
            "sha256:6262626262626262626262626262626262626262626262626262626262626262",
        )
        .unwrap(),
    )
    .unwrap();

    let (receipt_a, context_a) = admit_and_receipt(&obligation_a, &requirement, "A");
    let fact_a = derive_current_obligation_discharge_fact_v1(
        &obligation_a,
        &context_a,
        &[receipt_a],
    )
    .unwrap();

    // Receipt B was issued for G17. Asking the lower theorem under G18 must not
    // mint a current fact at all, so the higher layer never gets a capability it
    // could accidentally mix into G17 requirement closure.
    let (receipt_b, _) = admit_and_receipt(&obligation_b, &requirement, "B");
    let stale_context_b = DischargeContextV1::new(
        "bracket-alpha",
        "design:G18",
        requirement.revision_id().as_str(),
        "VD-static-G17-B",
        "currentness:design-G17:B",
    )
    .unwrap();
    assert!(derive_current_obligation_discharge_fact_v1(
        &obligation_b,
        &stale_context_b,
        &[receipt_b],
    )
    .is_err());

    let decision = evaluate_requirement_satisfaction_v1(
        &contract,
        &requirement,
        "bracket-alpha",
        "design:G17",
        &[fact_a],
        RequirementCurrentnessAssertionIdV1::parse(
            "sha256:6363636363636363636363636363636363636363636363636363636363636363",
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        decision,
        RequirementSatisfactionDecisionV1::RequirementUnsatisfied(_)
    ));
}
