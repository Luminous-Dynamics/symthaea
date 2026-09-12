// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-API parity with the independent ETK-3C native analytical oracle.

use symthaea_engineering_analysis_trust::{
    AcceptanceRecordDigestV1, AlgorithmRevisionDigestV1, AnalysisConfigurationDigestV1,
    AcceptedAnalysisRequirementV1, AnalysisTrustErrorV1, AnalyticalAcceptancePolicyV1,
    AnalyticalMethodV1, CurrentnessAssertionV1, CurrentnessAttestationDigestV1,
    ExecutionArtifactDigestV1, ImplementationArtifactDigestV1, ModelQualificationRecordDigestV1,
    ModelRevisionDigestV1, NativeAnalyticalPlanV1, NativeAnalyticalResultV1,
    RectangularCantileverInputV1, Sha256DigestV1, SubjectRevisionV1, SubjectStateDigestV1,
    TwinKindV1, TwinRevisionV1, TwinSchemaDigestV1, TwinStateDigestV1,
    ValidityDimensionDigestV1, ValidityDomainRevisionV1, admit_native_analytical_evidence_v1,
    analytical_obligation_revision_v1, canonical_binary64_v1,
    derive_current_native_analytical_discharge_fact_v1,
    issue_native_analytical_discharge_receipt_v1,
};
use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation};

const REQUIREMENT: &str =
    "sha256:10891514f6551c85b10674a3df04ab2c6d19f74f014adeacc671fa31deecd419";
const OBLIGATION: &str =
    "sha256:ea6e2f0f5e3ec37524325eda24b95ead75fa502f5d2a0b9410a407aed3a12a8c";
const SUBJECT: &str =
    "sha256:38a9505d423fa3464020107b4e8abc8acc6ac6af33bcc98a2418480e1d33390e";
const TWIN: &str =
    "sha256:19d558d6e7579f0f44c71398c7ddac659677aca0fa0e65ed46227670e4876cd9";
const VALIDITY: &str =
    "sha256:90bff4adf8917f2ae071f405d5c46afb310b12a98dd2e4eec845de1261a24890";
const CURRENTNESS: &str =
    "sha256:f85bd7d5129b08a3256cfdd5b3506f3cae1dbc625b097a8bf95bf1f6fe709dd9";
const REFRESHED_CURRENTNESS: &str =
    "sha256:e4008d7a50289e654a63b2db6d964a5550030bc6ab670757cd78c0c58b529a1b";
const METHOD: &str =
    "sha256:73bb060e12b166cecfea9a73c1f274f4443f1a540a777e7ac8260c65808bedc3";
const INPUT: &str =
    "sha256:3b9a62ea2ce031b73c188e8c0138fb1ebcda20e77d7f2401502ec493323d3c86";
const POLICY: &str =
    "sha256:7edc8fcbc03184413cc9c275537bdefb2d1c18f281406dc1adacd9a53a0a3bb5";
const PLAN: &str =
    "sha256:b86b76c504bd7b7601981d40988ceafd6c0a6580c941e4c91a65054c57749432";
const ADMITTED: &str =
    "sha256:7ea504bb399216df14d5792b44f574e18c62163049c4406e23e4dc3783e86bde";
const RECEIPT: &str =
    "sha256:5d740e2cfe67fa0bc0145cc7d79a9410e9c36b276ac78f12243ad5e013f7a1ec";
const CURRENT_FACT: &str =
    "sha256:47e1d641da233d0d70e6084a97bbe55fbf9d0487a3b6f7978ca03790fbeb494c";

fn digest(ch: char) -> String {
    format!("sha256:{}", ch.to_string().repeat(64))
}

fn raw(ch: char) -> Sha256DigestV1 {
    Sha256DigestV1::parse(digest(ch)).unwrap()
}

macro_rules! premise {
    ($ty:ty, $ch:expr) => {
        <$ty>::from_digest(raw($ch))
    };
}

fn requirement() -> AcceptedAnalysisRequirementV1 {
    AcceptedAnalysisRequirementV1::civil_service_stress_250_mpa(premise!(
        AcceptanceRecordDigestV1,
        'a'
    ))
}

fn obligation() -> ProofObligation {
    ProofObligation {
        id: "00000000-0000-4000-8000-000000000042"
            .parse()
            .unwrap(),
        claim: "stress remains below allowable under service load".into(),
        expected_evidence: EvidenceKind::Analysis,
        status: ObligationStatus::Open,
        evidence_refs: Vec::new(),
    }
}

struct ContextFixture {
    subject: SubjectRevisionV1,
    twin: TwinRevisionV1,
    validity: ValidityDomainRevisionV1,
    currentness: CurrentnessAssertionV1,
}

fn context(attestation: char) -> ContextFixture {
    let subject = SubjectRevisionV1::new(
        "design",
        "bracket-alpha",
        premise!(SubjectStateDigestV1, 'b'),
    )
    .unwrap();
    let twin = TwinRevisionV1::new(
        &subject,
        TwinKindV1::Design,
        premise!(TwinStateDigestV1, 'c'),
        premise!(TwinSchemaDigestV1, 'd'),
        None,
    )
    .unwrap();
    let validity = ValidityDomainRevisionV1::new(
        &subject,
        &twin,
        premise!(ModelRevisionDigestV1, 'e'),
        premise!(AnalysisConfigurationDigestV1, 'f'),
        vec![
            ("load_case".into(), premise!(ValidityDimensionDigestV1, '1')),
            (
                "material_state".into(),
                premise!(ValidityDimensionDigestV1, '2'),
            ),
            (
                "boundary_conditions".into(),
                premise!(ValidityDimensionDigestV1, '3'),
            ),
        ],
    )
    .unwrap();
    let currentness = CurrentnessAssertionV1::new(
        &twin,
        &validity,
        premise!(CurrentnessAttestationDigestV1, attestation),
        1_789_123_456_000,
    )
    .unwrap();
    ContextFixture {
        subject,
        twin,
        validity,
        currentness,
    }
}

fn method() -> AnalyticalMethodV1 {
    AnalyticalMethodV1::euler_bernoulli_beam(
        premise!(ImplementationArtifactDigestV1, 'a'),
        premise!(AlgorithmRevisionDigestV1, 'b'),
    )
}

fn input(method: &AnalyticalMethodV1, load_n: f64) -> RectangularCantileverInputV1 {
    RectangularCantileverInputV1::new(
        method,
        2.0,
        0.05,
        0.1,
        200.0e9,
        250.0e6,
        load_n,
    )
    .unwrap()
}

fn policy(threshold: f64, max_error: f64) -> AnalyticalAcceptancePolicyV1 {
    AnalyticalAcceptancePolicyV1::factor_of_safety_ge(
        threshold,
        max_error,
        premise!(ModelQualificationRecordDigestV1, '9'),
    )
    .unwrap()
}

fn plan(
    method: &AnalyticalMethodV1,
    input: &RectangularCantileverInputV1,
    policy: &AnalyticalAcceptancePolicyV1,
    context: &ContextFixture,
) -> NativeAnalyticalPlanV1 {
    NativeAnalyticalPlanV1::new(
        &requirement(),
        &context.subject,
        &context.twin,
        &context.validity,
        &context.currentness,
        &obligation(),
        method,
        input,
        policy,
    )
    .unwrap()
}

fn result(
    method: &AnalyticalMethodV1,
    input: &RectangularCantileverInputV1,
    fos: f64,
    stress_pa: f64,
    deflection_m: f64,
    moment_nm: f64,
    error: f64,
) -> NativeAnalyticalResultV1 {
    NativeAnalyticalResultV1::for_input(
        method,
        input,
        premise!(ExecutionArtifactDigestV1, '8'),
        fos,
        stress_pa,
        deflection_m,
        moment_nm,
        error,
    )
    .unwrap()
}

#[test]
fn independent_vectors_compose_end_to_end() {
    assert_eq!(requirement().revision_id().as_str(), REQUIREMENT);
    assert_eq!(
        analytical_obligation_revision_v1(&obligation())
            .unwrap()
            .as_str(),
        OBLIGATION
    );
    assert_eq!(canonical_binary64_v1(-0.0).unwrap(), "f64:0000000000000000");
    assert_eq!(canonical_binary64_v1(0.1).unwrap(), "f64:3fb999999999999a");

    let context = context('4');
    assert_eq!(context.subject.revision_id().as_str(), SUBJECT);
    assert_eq!(context.twin.revision_id().as_str(), TWIN);
    assert_eq!(context.validity.revision_id().as_str(), VALIDITY);
    assert_eq!(context.currentness.assertion_id().as_str(), CURRENTNESS);

    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy(2.0, 0.05);
    assert_eq!(method.revision_id().as_str(), METHOD);
    assert_eq!(input.revision_id().as_str(), INPUT);
    assert_eq!(policy.revision_id().as_str(), POLICY);

    let plan = plan(&method, &input, &policy, &context);
    assert_eq!(plan.plan_id().as_str(), PLAN);
    assert_eq!(plan.requirement_revision_id().as_str(), REQUIREMENT);
    assert_eq!(plan.obligation_revision_id().as_str(), OBLIGATION);
    assert_eq!(plan.subject_revision_id().as_str(), SUBJECT);
    assert_eq!(plan.twin_revision_id().as_str(), TWIN);
    assert_eq!(plan.validity_domain_revision_id().as_str(), VALIDITY);
    assert_eq!(plan.currentness_assertion_id().as_str(), CURRENTNESS);

    let candidate = result(
        &method,
        &input,
        10.416666666666666,
        24.0e6,
        0.0032,
        2000.0,
        0.02,
    );
    let admitted =
        admit_native_analytical_evidence_v1(&plan, &method, &input, &policy, &candidate).unwrap();
    assert_eq!(admitted.admitted_evidence_id().as_str(), ADMITTED);

    let receipt = issue_native_analytical_discharge_receipt_v1(&plan, &admitted).unwrap();
    assert_eq!(receipt.receipt_id().as_str(), RECEIPT);

    let current = derive_current_native_analytical_discharge_fact_v1(&plan, &receipt).unwrap();
    assert_eq!(current.fact_id().as_str(), CURRENT_FACT);
    assert_eq!(current.witness_receipt_id().as_str(), RECEIPT);
    assert_eq!(current.requirement_revision_id().as_str(), REQUIREMENT);
    assert_eq!(current.obligation_revision_id().as_str(), OBLIGATION);
}

#[test]
fn role_safe_premises_preserve_vector_bytes_but_not_type_interchangeability() {
    let raw_a = raw('a');
    assert_eq!(
        AcceptanceRecordDigestV1::from_digest(raw_a.clone()).as_str(),
        ImplementationArtifactDigestV1::from_digest(raw_a).as_str()
    );
    // Equal bytes are intentionally distinct Rust types; call sites must name
    // which authority premise role they are supplying.
}

#[test]
fn analysis_is_not_external_simulation_semantics() {
    let mut wrong = obligation();
    wrong.expected_evidence = EvidenceKind::Simulation;
    assert_eq!(
        analytical_obligation_revision_v1(&wrong).unwrap_err(),
        AnalysisTrustErrorV1::NotAnalysisObligation
    );
}

#[test]
fn semantic_context_rejects_cross_subject_and_cross_twin_composition() {
    let subject_a = SubjectRevisionV1::new(
        "design",
        "bracket-alpha",
        premise!(SubjectStateDigestV1, 'b'),
    )
    .unwrap();
    let subject_b = SubjectRevisionV1::new(
        "design",
        "bracket-beta",
        premise!(SubjectStateDigestV1, 'b'),
    )
    .unwrap();
    let twin_a = TwinRevisionV1::new(
        &subject_a,
        TwinKindV1::Design,
        premise!(TwinStateDigestV1, 'c'),
        premise!(TwinSchemaDigestV1, 'd'),
        None,
    )
    .unwrap();

    assert_eq!(
        ValidityDomainRevisionV1::new(
            &subject_b,
            &twin_a,
            premise!(ModelRevisionDigestV1, 'e'),
            premise!(AnalysisConfigurationDigestV1, 'f'),
            Vec::<(String, ValidityDimensionDigestV1)>::new(),
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::TwinSubjectMismatch
    );

    let context_a = context('4');
    let twin_b = TwinRevisionV1::new(
        &subject_b,
        TwinKindV1::Design,
        premise!(TwinStateDigestV1, 'c'),
        premise!(TwinSchemaDigestV1, 'd'),
        None,
    )
    .unwrap();
    assert_eq!(
        CurrentnessAssertionV1::new(
            &twin_b,
            &context_a.validity,
            premise!(CurrentnessAttestationDigestV1, '4'),
            1_789_123_456_000,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::ValidityContextMismatch
    );
}

#[test]
fn input_drift_and_weak_policy_fail_before_authority() {
    let method = method();
    let original = input(&method, 1000.0);
    let changed = input(&method, 1000.0000000000001);
    let context = context('4');
    let policy = policy(2.0, 0.05);
    let plan = plan(&method, &original, &policy, &context);
    assert_ne!(original.revision_id(), changed.revision_id());

    let changed_result = result(
        &method,
        &changed,
        10.416666666666666,
        24.0e6,
        0.0032,
        2000.0,
        0.02,
    );
    assert_eq!(
        admit_native_analytical_evidence_v1(
            &plan,
            &method,
            &changed,
            &policy,
            &changed_result,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::PlanBindingMismatch
    );

    let weak = policy(0.5, 0.05);
    assert_eq!(
        NativeAnalyticalPlanV1::new(
            &requirement(),
            &context.subject,
            &context.twin,
            &context.validity,
            &context.currentness,
            &obligation(),
            &method,
            &original,
            &weak,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::PolicyDoesNotDischargeRequirement
    );
}

#[test]
fn analytical_equations_and_conservative_acceptance_fail_closed() {
    let method = method();
    let input = input(&method, 1000.0);
    let context = context('4');
    let policy = policy(2.0, 0.05);
    let plan = plan(&method, &input, &policy, &context);

    for (candidate, expected) in [
        (
            result(
                &method,
                &input,
                10.416666666666666,
                24.0e6,
                0.004,
                2000.0,
                0.02,
            ),
            AnalysisTrustErrorV1::AnalyticalEquationMismatch("maximum deflection"),
        ),
        (
            result(
                &method,
                &input,
                10.416666666666666,
                24.0e6,
                0.0032,
                1999.0,
                0.02,
            ),
            AnalysisTrustErrorV1::AnalyticalEquationMismatch("maximum moment"),
        ),
        (
            result(&method, &input, 10.0, 24.0e6, 0.0032, 2000.0, 0.02),
            AnalysisTrustErrorV1::InconsistentFactorOfSafety,
        ),
    ] {
        assert_eq!(
            admit_native_analytical_evidence_v1(
                &plan,
                &method,
                &input,
                &policy,
                &candidate,
            )
            .unwrap_err(),
            expected
        );
    }

    let excessive_error = result(
        &method,
        &input,
        10.416666666666666,
        24.0e6,
        0.0032,
        2000.0,
        0.06,
    );
    assert_eq!(
        admit_native_analytical_evidence_v1(
            &plan,
            &method,
            &input,
            &policy,
            &excessive_error,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::ModelErrorBudgetExceeded
    );

    let strict = policy(10.3, 0.05);
    let strict_plan = plan(&method, &input, &strict, &context);
    let nominal = result(
        &method,
        &input,
        10.416666666666666,
        24.0e6,
        0.0032,
        2000.0,
        0.02,
    );
    assert_eq!(
        admit_native_analytical_evidence_v1(
            &strict_plan,
            &method,
            &input,
            &strict,
            &nominal,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::AcceptancePredicateFailed
    );
}

#[test]
fn currentness_refresh_makes_old_receipt_historical() {
    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy(2.0, 0.05);
    let old_context = context('4');
    let new_context = context('5');
    assert_eq!(old_context.currentness.assertion_id().as_str(), CURRENTNESS);
    assert_eq!(new_context.currentness.assertion_id().as_str(), REFRESHED_CURRENTNESS);

    let old_plan = plan(&method, &input, &policy, &old_context);
    let new_plan = plan(&method, &input, &policy, &new_context);
    let candidate = result(
        &method,
        &input,
        10.416666666666666,
        24.0e6,
        0.0032,
        2000.0,
        0.02,
    );
    let admitted = admit_native_analytical_evidence_v1(
        &old_plan,
        &method,
        &input,
        &policy,
        &candidate,
    )
    .unwrap();
    let receipt = issue_native_analytical_discharge_receipt_v1(&old_plan, &admitted).unwrap();

    assert_eq!(
        derive_current_native_analytical_discharge_fact_v1(&new_plan, &receipt).unwrap_err(),
        AnalysisTrustErrorV1::HistoricalPlan
    );
}

#[test]
fn malformed_content_identity_and_non_finite_numbers_are_rejected() {
    assert!(ExecutionArtifactDigestV1::parse("sha256:not-a-digest").is_err());
    assert!(AcceptanceRecordDigestV1::parse(format!("sha256:{}", "A".repeat(64))).is_err());
    assert!(canonical_binary64_v1(f64::NAN).is_err());
}
