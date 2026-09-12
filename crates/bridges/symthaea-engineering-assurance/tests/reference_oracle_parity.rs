// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-API parity for the hardened plan-bound ETK V2 assurance chain.
//!
//! Every expected identity below is frozen by an implementation-independent
//! standard-library Python reference theorem. Production Rust must reproduce the
//! vectors without invoking that oracle as an authority source.

use symthaea_engineering_assurance::{
    AssuranceErrorV2, PlanBoundSimulationEvidencePlanV2,
    RequirementCurrentnessAssertionIdV2, RequirementDecompositionAcceptanceRecordDigestV2,
    RequirementDecompositionPolicyRevisionDigestV2, RequirementSatisfactionDecisionV2,
    RequirementVerificationContractV2, RequirementVerificationMemberV2,
    admit_plan_bound_simulation_v2, canonical_binary64_v2,
    derive_current_plan_bound_discharge_fact_v2, evaluate_requirement_satisfaction_v2,
    issue_plan_bound_discharge_receipt_v2,
};
use symthaea_engineering_evidence_plan::{
    AcceptedRequirementRevisionV1, CurrentnessAssertionV1, MetricOperatorV1,
    MetricPredicateV1, RequirementCriticalityV1, Sha256DigestV1,
    SimulationEvidencePolicyRevisionV1, SubjectRevisionV1, TwinKindV1, TwinRevisionV1,
    ValidityDomainRevisionV1, WarningPolicyV1,
};
use symthaea_engineering_requirement_binding::{
    BindingAcceptanceRecordDigestV1, DerivationPolicyRevisionDigestV1,
    DerivationRecordDigestV1, RequirementObligationBindingV1,
};
use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation};
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, Interval, SimulationEvidence, SimulationMetric,
    SimulationRequest, SimulationResult, SolverKind, UncertaintyEstimate,
};

const REQUIREMENT_REVISION: &str =
    "sha256:e340c5030eebc978c41443ffd64f340dc5febad31e376080340bacfceda60faa";
const SUBJECT_REVISION: &str =
    "sha256:38a9505d423fa3464020107b4e8abc8acc6ac6af33bcc98a2418480e1d33390e";
const TWIN_REVISION: &str =
    "sha256:19d558d6e7579f0f44c71398c7ddac659677aca0fa0e65ed46227670e4876cd9";
const VALIDITY_REVISION: &str =
    "sha256:90bff4adf8917f2ae071f405d5c46afb310b12a98dd2e4eec845de1261a24890";
const CURRENTNESS_ASSERTION: &str =
    "sha256:f85bd7d5129b08a3256cfdd5b3506f3cae1dbc625b097a8bf95bf1f6fe709dd9";
const REQUEST_V2: &str =
    "sha256:695db7bfd3570d020ecef240303d4ba7cc8fc8ef461f4a7acf1c08491c75f165";
const POLICY_V2: &str =
    "sha256:7f58b186d470cd62256df2aa14f6be85e35a52ba7de0b91755e4fed3cebe1a09";

const OBLIGATION_A: &str =
    "sha256:743d2c13cfcc52bdfb4cfd9a4a836ed0806ace7b2a68b4c7284b1910d8863c29";
const RELATIONSHIP_A: &str =
    "sha256:1cc54ee11ff9b7e99279d4c9a9d43f751809f7f30a43469cff8e18fe70eee408";
const PLAN_A: &str =
    "sha256:3b55051507d38ebde23b9f5b5e6ad03c1cadae81ba56abde25ff3b7213ce030b";
const ADMITTED_A: &str =
    "sha256:daca55e4fdd2c173606e035b63c9721d6ed86c121d28543e3f326366881ce6ed";
const RECEIPT_A: &str =
    "sha256:391e20b8697ad72aa29de0b99b6d317d2dc704542ea88a0e081578ccf720561d";
const FACT_A: &str =
    "sha256:a258c2988a48f27de28d9f0f712f01a3235702ae473537bbe8487f38e2f96b6e";

const OBLIGATION_B: &str =
    "sha256:63ce87ba42de8d08ff87059322964a7609228d66a7b8f7fc67016b298c2a7c2d";
const RELATIONSHIP_B: &str =
    "sha256:a4a96e57f7c41c8d20660288e6372882d34c632157d4e9d8234ad7f36c94b5bd";
const PLAN_B: &str =
    "sha256:926d82f36922fa3e38ac369c3e841462ea9024460ceb8f83c7007908c2e010e7";
const ADMITTED_B: &str =
    "sha256:7f9a4bc46a34f3d97f1fcfbbd4b3479a61de0d601545677dbbf765321d4ddc3b";
const RECEIPT_B: &str =
    "sha256:b0a2d81d49dd5bba72a6d1b885d75953768fa0e31ab97d66042a98ee4190ed18";
const FACT_B: &str =
    "sha256:36050614ef2fbe51c52c229e77d5a0616b29a4e38f24e52cfc2511a92df864d3";

const VERIFICATION_CONTRACT: &str =
    "sha256:79a8a3e5cda3f89ff66c4f5954f52fb92e3b3e33418d50230b6fd7311be1d800";
const SATISFACTION_RECEIPT: &str =
    "sha256:883d74833c424d3a5dfa0a29519fb69a5d2a2e53ecfd0c85629e399ed1b5dce8";

fn digest(ch: char) -> Sha256DigestV1 {
    Sha256DigestV1::parse(format!("sha256:{}", ch.to_string().repeat(64))).unwrap()
}

fn repeated_pair(pair: &str) -> String {
    format!("sha256:{}", pair.repeat(32))
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

fn raw_request() -> SimulationRequest {
    let mut request = SimulationRequest::new(
        "sim-static-G17-LC9",
        EngineeringDomain::Civil,
        SolverKind::FiniteElement,
        "check bracket service stress",
    )
    .with_parameter("load_n", 10_000.0, "N", "load-case:LC9")
    .with_parameter("thickness_mm", 8.0, "mm", "design:G17");
    request.requested_metrics = vec![
        "max_stress_mpa".into(),
        "max_displacement_mm".into(),
    ];
    request
}

fn evidence_policy() -> SimulationEvidencePolicyRevisionV1 {
    SimulationEvidencePolicyRevisionV1::new(
        "service-stress-policy",
        MetricPredicateV1::new(
            "max_stress_mpa",
            "MPa",
            MetricOperatorV1::Le,
            250.0,
            0.2,
            0.1,
        )
        .unwrap(),
        WarningPolicyV1::DenyAny,
    )
    .unwrap()
}

fn solver_result(output_digest: &str) -> SimulationResult {
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
            input_digest: Some(digest('6').as_str().to_string()),
            output_digest: Some(output_digest.into()),
            parser_version: Some("symthaea-calculix-parser-v1".into()),
        },
    }
}

struct SemanticContext {
    requirement: AcceptedRequirementRevisionV1,
    subject: SubjectRevisionV1,
    twin: TwinRevisionV1,
    validity: ValidityDomainRevisionV1,
    currentness: CurrentnessAssertionV1,
    policy: SimulationEvidencePolicyRevisionV1,
    request: SimulationRequest,
}

fn semantic_context() -> SemanticContext {
    let requirement = accepted_requirement();
    let subject = SubjectRevisionV1::new("design", "bracket-alpha", digest('b')).unwrap();
    let twin = TwinRevisionV1::new(
        &subject,
        TwinKindV1::Design,
        digest('c'),
        digest('d'),
        None,
    )
    .unwrap();
    let validity = ValidityDomainRevisionV1::new(
        &subject,
        &twin,
        digest('e'),
        digest('f'),
        vec![
            ("load_case".into(), digest('1')),
            ("material_state".into(), digest('2')),
            ("boundary_conditions".into(), digest('3')),
        ],
    )
    .unwrap();
    let currentness =
        CurrentnessAssertionV1::new(&twin, &validity, digest('4'), 1_789_123_456_000).unwrap();
    SemanticContext {
        requirement,
        subject,
        twin,
        validity,
        currentness,
        policy: evidence_policy(),
        request: raw_request(),
    }
}

fn plan_for(
    context: &SemanticContext,
    obligation: &ProofObligation,
    rendered_input: Sha256DigestV1,
) -> PlanBoundSimulationEvidencePlanV2 {
    PlanBoundSimulationEvidencePlanV2::new(
        &context.subject,
        &context.twin,
        &context.requirement,
        obligation,
        &context.request,
        &context.policy,
        &context.validity,
        &context.currentness,
        rendered_input,
    )
    .unwrap()
}

#[test]
fn independent_v2_vectors_compose_end_to_end() {
    let context = semantic_context();
    assert_eq!(context.requirement.revision_id().as_str(), REQUIREMENT_REVISION);
    assert_eq!(context.subject.revision_id().as_str(), SUBJECT_REVISION);
    assert_eq!(context.twin.revision_id().as_str(), TWIN_REVISION);
    assert_eq!(context.validity.revision_id().as_str(), VALIDITY_REVISION);
    assert_eq!(context.currentness.assertion_id().as_str(), CURRENTNESS_ASSERTION);

    let obligation_a = obligation_a();
    let obligation_b = obligation_b();
    let binding_a = RequirementObligationBindingV1::derived_safety_obligation(
        &context.requirement,
        &obligation_a,
        DerivationRecordDigestV1::from_digest(digest('7')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('8')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('9')),
    )
    .unwrap();
    let binding_b = RequirementObligationBindingV1::derived_safety_obligation(
        &context.requirement,
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

    let plan_a = plan_for(&context, &obligation_a, digest('6'));
    let plan_b = plan_for(&context, &obligation_b, digest('6'));
    assert_eq!(plan_a.request_revision_id().as_str(), REQUEST_V2);
    assert_eq!(plan_a.evidence_policy_revision_id().as_str(), POLICY_V2);
    assert_eq!(plan_a.plan_id().as_str(), PLAN_A);
    assert_eq!(plan_b.plan_id().as_str(), PLAN_B);

    let admitted_a = admit_plan_bound_simulation_v2(
        &plan_a,
        &obligation_a,
        &solver_result(digest('7').as_str()),
        "solver-output:plan-v2-A",
        "calculix:2.22:plan-v2-A",
    )
    .unwrap();
    let admitted_b = admit_plan_bound_simulation_v2(
        &plan_b,
        &obligation_b,
        &solver_result(digest('8').as_str()),
        "solver-output:plan-v2-B",
        "calculix:2.22:plan-v2-B",
    )
    .unwrap();
    assert_eq!(admitted_a.admitted_evidence_id().as_str(), ADMITTED_A);
    assert_eq!(admitted_b.admitted_evidence_id().as_str(), ADMITTED_B);
    assert_eq!(admitted_a.output_digest().as_str(), digest('7').as_str());
    assert_eq!(
        admitted_a.audit_record_v2()["authority"].as_str(),
        Some("plan-bound-admitted-evidence-only")
    );
    assert_eq!(
        admitted_a.audit_record_v2()["solver_output_digest"].as_str(),
        Some(digest('7').as_str())
    );

    let receipt_a =
        issue_plan_bound_discharge_receipt_v2(&plan_a, &obligation_a, &admitted_a).unwrap();
    let receipt_b =
        issue_plan_bound_discharge_receipt_v2(&plan_b, &obligation_b, &admitted_b).unwrap();
    assert_eq!(receipt_a.receipt_id().as_str(), RECEIPT_A);
    assert_eq!(receipt_b.receipt_id().as_str(), RECEIPT_B);
    assert_eq!(receipt_a.admitted_evidence_id().as_str(), ADMITTED_A);
    assert_eq!(
        receipt_a.audit_record_v2()["plan_bound_admitted_evidence_id"].as_str(),
        Some(ADMITTED_A)
    );

    let fact_a =
        derive_current_plan_bound_discharge_fact_v2(&plan_a, &obligation_a, &receipt_a).unwrap();
    let fact_b =
        derive_current_plan_bound_discharge_fact_v2(&plan_b, &obligation_b, &receipt_b).unwrap();
    assert_eq!(fact_a.fact_id().as_str(), FACT_A);
    assert_eq!(fact_b.fact_id().as_str(), FACT_B);
    assert_eq!(fact_a.witness_receipt_id().as_str(), RECEIPT_A);
    assert_eq!(
        fact_a.audit_record_v2()["plan_bound_discharge_receipt_id"].as_str(),
        Some(RECEIPT_A)
    );

    let member_a = RequirementVerificationMemberV2::new(&binding_a, &plan_a).unwrap();
    let member_b = RequirementVerificationMemberV2::new(&binding_b, &plan_b).unwrap();
    let contract = RequirementVerificationContractV2::all_of(
        &context.requirement,
        vec![member_b.clone(), member_a.clone()],
        RequirementDecompositionPolicyRevisionDigestV2::parse(repeated_pair("61")).unwrap(),
        RequirementDecompositionAcceptanceRecordDigestV2::parse(repeated_pair("62")).unwrap(),
    )
    .unwrap();
    assert_eq!(contract.contract_id().as_str(), VERIFICATION_CONTRACT);
    assert_eq!(
        contract.audit_record_v2()["decomposition_policy_revision_id"].as_str(),
        Some(repeated_pair("61").as_str())
    );
    assert_eq!(
        contract.audit_record_v2()["decomposition_acceptance_record_digest"].as_str(),
        Some(repeated_pair("62").as_str())
    );

    let partial = evaluate_requirement_satisfaction_v2(
        &contract,
        &context.requirement,
        &context.subject,
        &context.twin,
        std::slice::from_ref(&fact_a),
        RequirementCurrentnessAssertionIdV2::parse(repeated_pair("63")).unwrap(),
    )
    .unwrap();
    let RequirementSatisfactionDecisionV2::RequirementUnsatisfied(partial) = partial else {
        panic!("one plan-bound fact must not satisfy two-member AllOf")
    };
    assert_eq!(partial.missing_evidence_plan_ids().len(), 1);
    assert_eq!(partial.missing_evidence_plan_ids()[0].as_str(), PLAN_B);

    let satisfied = evaluate_requirement_satisfaction_v2(
        &contract,
        &context.requirement,
        &context.subject,
        &context.twin,
        &[fact_b.clone(), fact_a.clone()],
        RequirementCurrentnessAssertionIdV2::parse(repeated_pair("63")).unwrap(),
    )
    .unwrap();
    let RequirementSatisfactionDecisionV2::CurrentRequirementSatisfied(satisfied) = satisfied else {
        panic!("both exact plan-bound facts should satisfy AllOf")
    };
    assert_eq!(satisfied.receipt_id().as_str(), SATISFACTION_RECEIPT);
    let satisfaction_audit = satisfied.audit_record_v2();
    assert_eq!(
        satisfaction_audit["authority"].as_str(),
        Some("current-requirement-satisfaction-only")
    );
    assert_eq!(
        satisfaction_audit["currentness_assertion_id"].as_str(),
        Some(repeated_pair("63").as_str())
    );
    assert_eq!(
        satisfaction_audit["current_discharge_facts"]
            .as_array()
            .expect("used fact audit array")
            .len(),
        2
    );

    let reordered = RequirementVerificationContractV2::all_of(
        &context.requirement,
        vec![member_a, member_b],
        RequirementDecompositionPolicyRevisionDigestV2::parse(repeated_pair("61")).unwrap(),
        RequirementDecompositionAcceptanceRecordDigestV2::parse(repeated_pair("62")).unwrap(),
    )
    .unwrap();
    assert_eq!(reordered.contract_id(), contract.contract_id());
}

#[test]
fn v2_requires_exact_plan_context_not_merely_a_discharged_obligation() {
    let context = semantic_context();
    let obligation = obligation_a();
    let binding = RequirementObligationBindingV1::derived_safety_obligation(
        &context.requirement,
        &obligation,
        DerivationRecordDigestV1::from_digest(digest('7')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('8')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('9')),
    )
    .unwrap();
    let plan = plan_for(&context, &obligation, digest('6'));
    let alternate_plan = plan_for(&context, &obligation, digest('7'));

    let member = RequirementVerificationMemberV2::new(&binding, &plan).unwrap();
    let alternate_member = RequirementVerificationMemberV2::new(&binding, &alternate_plan).unwrap();
    let multi_plan_contract = RequirementVerificationContractV2::all_of(
        &context.requirement,
        vec![member.clone(), alternate_member],
        RequirementDecompositionPolicyRevisionDigestV2::parse(repeated_pair("61")).unwrap(),
        RequirementDecompositionAcceptanceRecordDigestV2::parse(repeated_pair("62")).unwrap(),
    )
    .unwrap();
    assert_ne!(multi_plan_contract.contract_id().as_str(), VERIFICATION_CONTRACT);

    let duplicate = RequirementVerificationContractV2::all_of(
        &context.requirement,
        vec![member.clone(), member],
        RequirementDecompositionPolicyRevisionDigestV2::parse(repeated_pair("61")).unwrap(),
        RequirementDecompositionAcceptanceRecordDigestV2::parse(repeated_pair("62")).unwrap(),
    );
    assert_eq!(duplicate.unwrap_err(), AssuranceErrorV2::DuplicateEvidencePlan);

    let admitted = admit_plan_bound_simulation_v2(
        &plan,
        &obligation,
        &solver_result(digest('7').as_str()),
        "solver-output:plan-v2-A",
        "calculix:2.22:plan-v2-A",
    )
    .unwrap();
    let receipt = issue_plan_bound_discharge_receipt_v2(&plan, &obligation, &admitted).unwrap();
    assert_eq!(
        derive_current_plan_bound_discharge_fact_v2(&alternate_plan, &obligation, &receipt)
            .unwrap_err(),
        AssuranceErrorV2::HistoricalPlan
    );
}

#[test]
fn v2_canonicalization_provenance_and_warning_gates_fail_closed() {
    assert_eq!(canonical_binary64_v2(-0.0).unwrap(), "f64:0000000000000000");
    assert_eq!(canonical_binary64_v2(0.1).unwrap(), "f64:3fb999999999999a");
    assert!(canonical_binary64_v2(f64::NAN).is_err());

    let context = semantic_context();
    let obligation = obligation_a();
    let plan = plan_for(&context, &obligation, digest('6'));

    let mut reordered_request = raw_request();
    reordered_request.parameters.reverse();
    reordered_request.requested_metrics.reverse();
    let reordered_plan = PlanBoundSimulationEvidencePlanV2::new(
        &context.subject,
        &context.twin,
        &context.requirement,
        &obligation,
        &reordered_request,
        &context.policy,
        &context.validity,
        &context.currentness,
        digest('6'),
    )
    .unwrap();
    assert_eq!(reordered_plan.plan_id(), plan.plan_id());

    let mut reordered_result = solver_result(digest('7').as_str());
    reordered_result.metrics.reverse();
    let reordered_admitted = admit_plan_bound_simulation_v2(
        &plan,
        &obligation,
        &reordered_result,
        "solver-output:plan-v2-A",
        "calculix:2.22:plan-v2-A",
    )
    .unwrap();
    assert_eq!(reordered_admitted.admitted_evidence_id().as_str(), ADMITTED_A);

    let mut wrong_input = solver_result(digest('9').as_str());
    wrong_input.evidence.input_digest = Some(digest('7').as_str().to_string());
    assert_eq!(
        admit_plan_bound_simulation_v2(
            &plan,
            &obligation,
            &wrong_input,
            "solver-output:bad-input",
            "calculix:2.22:bad-input",
        )
        .unwrap_err(),
        AssuranceErrorV2::PlanResultBindingMismatch
    );

    let mut malformed_output = solver_result(digest('9').as_str());
    malformed_output.evidence.output_digest = Some("sha256:not-a-digest".into());
    assert_eq!(
        admit_plan_bound_simulation_v2(
            &plan,
            &obligation,
            &malformed_output,
            "solver-output:malformed-output",
            "calculix:2.22:malformed-output",
        )
        .unwrap_err(),
        AssuranceErrorV2::InvalidDigest("solver output digest")
    );

    let mut warning = solver_result(digest('9').as_str());
    warning.warnings.push("mesh warning".into());
    assert_eq!(
        admit_plan_bound_simulation_v2(
            &plan,
            &obligation,
            &warning,
            "solver-output:bad-warning",
            "calculix:2.22:bad-warning",
        )
        .unwrap_err(),
        AssuranceErrorV2::WarningDenied
    );

    let mut duplicate_warning = solver_result(digest('9').as_str());
    duplicate_warning.warnings = vec!["mesh warning".into(), "mesh warning".into()];
    assert_eq!(
        admit_plan_bound_simulation_v2(
            &plan,
            &obligation,
            &duplicate_warning,
            "solver-output:duplicate-warning",
            "calculix:2.22:duplicate-warning",
        )
        .unwrap_err(),
        AssuranceErrorV2::DuplicateWarning("mesh warning".into())
    );
}
