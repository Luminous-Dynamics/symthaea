// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-API parity with the independent ETK-3C bounded-currentness V2 oracle.

use symthaea_engineering_analysis_trust::{
    AcceptanceRecordDigestV1, AlgorithmRevisionDigestV1, AnalysisConfigurationDigestV1,
    AcceptedAnalysisRequirementV1, AnalysisTrustErrorV1, AnalyticalAcceptancePolicyV1,
    AnalyticalMethodV1, CurrentnessAssertionV2, CurrentnessAttestationDigestV1,
    ExecutionArtifactDigestV1, ImplementationArtifactDigestV1, ModelQualificationRecordDigestV1,
    ModelRevisionDigestV1, NativeAnalyticalPlanV1, NativeAnalyticalResultV1,
    RectangularCantileverInputV1, Sha256DigestV1, SubjectRevisionV1, SubjectStateDigestV1,
    TwinKindV1, TwinRevisionV1, TwinSchemaDigestV1, TwinStateDigestV1,
    ValidityDimensionDigestV1, ValidityDomainRevisionV1, admit_native_analytical_evidence_v1,
    analytical_obligation_revision_v1, canonical_binary64_v1,
    derive_current_native_analytical_discharge_fact_v2,
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
const CURRENTNESS_V2: &str =
    "sha256:deab1cf24d0cb23c7e72697f9bc41277b82997696a13ea1c25fb3c7a7d0994c6";
const METHOD: &str =
    "sha256:73bb060e12b166cecfea9a73c1f274f4443f1a540a777e7ac8260c65808bedc3";
const INPUT: &str =
    "sha256:3b9a62ea2ce031b73c188e8c0138fb1ebcda20e77d7f2401502ec493323d3c86";
const POLICY: &str =
    "sha256:7edc8fcbc03184413cc9c275537bdefb2d1c18f281406dc1adacd9a53a0a3bb5";
const PLAN_V2: &str =
    "sha256:bc308f5399ca9032131dcbed3f53991ab104465b15471521b0388be7f611b5e1";
const ADMITTED_V2: &str =
    "sha256:228f0ceb1566fa133d8647ccc5859ffd45957ab5138d67fa4b943a79fcdf0ec7";
const RECEIPT_V2: &str =
    "sha256:ce0b9d96d83e829d3878e7303e35ac607dbed8512a2202e7c224ca4c32ee0b81";
const CURRENT_FACT_V2: &str =
    "sha256:2d6e90e20460efc8f2f3d24c72e2c7a07be77e0143119d38f23385a868dad6eb";

const OBSERVED_AT_UNIX_MS: u64 = 1_789_123_456_000;
const VALID_UNTIL_UNIX_MS: u64 = 1_789_209_856_000;
const EVALUATED_AT_UNIX_MS: u64 = 1_789_123_457_000;

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
    currentness: CurrentnessAssertionV2,
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
    let currentness = CurrentnessAssertionV2::new(
        &twin,
        &validity,
        premise!(CurrentnessAttestationDigestV1, attestation),
        OBSERVED_AT_UNIX_MS,
        VALID_UNTIL_UNIX_MS,
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
fn v2_vectors_compose_end_to_end_and_preserve_upstream_semantics() {
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
    assert_eq!(context.currentness.assertion_id().as_str(), CURRENTNESS_V2);
    assert_eq!(context.currentness.observed_at_unix_ms(), OBSERVED_AT_UNIX_MS);
    assert_eq!(context.currentness.valid_until_unix_ms(), VALID_UNTIL_UNIX_MS);

    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy(2.0, 0.05);
    assert_eq!(method.revision_id().as_str(), METHOD);
    assert_eq!(input.revision_id().as_str(), INPUT);
    assert_eq!(policy.revision_id().as_str(), POLICY);

    let plan = plan(&method, &input, &policy, &context);
    assert_eq!(plan.plan_id().as_str(), PLAN_V2);
    assert_eq!(plan.requirement_revision_id().as_str(), REQUIREMENT);
    assert_eq!(plan.obligation_revision_id().as_str(), OBLIGATION);
    assert_eq!(plan.subject_revision_id().as_str(), SUBJECT);
    assert_eq!(plan.twin_revision_id().as_str(), TWIN);
    assert_eq!(plan.validity_domain_revision_id().as_str(), VALIDITY);
    assert_eq!(plan.currentness_assertion_id().as_str(), CURRENTNESS_V2);

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
    assert_eq!(admitted.admitted_evidence_id().as_str(), ADMITTED_V2);

    let receipt = issue_native_analytical_discharge_receipt_v1(&plan, &admitted).unwrap();
    assert_eq!(receipt.receipt_id().as_str(), RECEIPT_V2);

    let current = derive_current_native_analytical_discharge_fact_v2(
        &plan,
        &receipt,
        EVALUATED_AT_UNIX_MS,
    )
    .unwrap();
    assert_eq!(current.fact_id().as_str(), CURRENT_FACT_V2);
    assert_eq!(current.witness_receipt_id().as_str(), RECEIPT_V2);
    assert_eq!(current.requirement_revision_id().as_str(), REQUIREMENT);
    assert_eq!(current.obligation_revision_id().as_str(), OBLIGATION);
    assert_eq!(current.evaluated_at_unix_ms(), EVALUATED_AT_UNIX_MS);
}

#[test]
fn bounded_currentness_rejects_invalid_windows_and_out_of_window_evaluation() {
    let base = context('4');
    assert_eq!(
        CurrentnessAssertionV2::new(
            &base.twin,
            &base.validity,
            premise!(CurrentnessAttestationDigestV1, '4'),
            OBSERVED_AT_UNIX_MS,
            OBSERVED_AT_UNIX_MS,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::InvalidCurrentnessWindow
    );

    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy(2.0, 0.05);
    let plan = plan(&method, &input, &policy, &base);
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
    let receipt = issue_native_analytical_discharge_receipt_v1(&plan, &admitted).unwrap();

    assert_eq!(
        derive_current_native_analytical_discharge_fact_v2(
            &plan,
            &receipt,
            OBSERVED_AT_UNIX_MS - 1,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::FreshnessNotYetValid
    );
    assert_eq!(
        derive_current_native_analytical_discharge_fact_v2(
            &plan,
            &receipt,
            VALID_UNTIL_UNIX_MS + 1,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::FreshnessExpired
    );

    // Inclusive boundaries are intentionally valid.
    derive_current_native_analytical_discharge_fact_v2(&plan, &receipt, OBSERVED_AT_UNIX_MS)
        .unwrap();
    derive_current_native_analytical_discharge_fact_v2(&plan, &receipt, VALID_UNTIL_UNIX_MS)
        .unwrap();
}

#[test]
fn refreshed_currentness_makes_old_receipt_historical() {
    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy(2.0, 0.05);
    let old_context = context('4');
    let refreshed_context = context('5');
    let old_plan = plan(&method, &input, &policy, &old_context);
    let refreshed_plan = plan(&method, &input, &policy, &refreshed_context);
    assert_ne!(old_plan.plan_id(), refreshed_plan.plan_id());

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
        derive_current_native_analytical_discharge_fact_v2(
            &refreshed_plan,
            &receipt,
            EVALUATED_AT_UNIX_MS,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::HistoricalPlan
    );
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
        CurrentnessAssertionV2::new(
            &twin_b,
            &context_a.validity,
            premise!(CurrentnessAttestationDigestV1, '4'),
            OBSERVED_AT_UNIX_MS,
            VALID_UNTIL_UNIX_MS,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::ValidityContextMismatch
    );
}

#[test]
fn input_drift_weak_policy_and_equation_errors_fail_before_authority() {
    let method = method();
    let original = input(&method, 1000.0);
    let changed = input(&method, 1000.0000000000001);
    let context = context('4');
    let policy = policy(2.0, 0.05);
    let plan = plan(&method, &original, &policy, &context);

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

    let wrong_deflection = result(
        &method,
        &original,
        10.416666666666666,
        24.0e6,
        0.004,
        2000.0,
        0.02,
    );
    assert_eq!(
        admit_native_analytical_evidence_v1(
            &plan,
            &method,
            &original,
            &policy,
            &wrong_deflection,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::AnalyticalEquationMismatch("maximum deflection")
    );

    let wrong_moment = result(
        &method,
        &original,
        10.416666666666666,
        24.0e6,
        0.0032,
        1999.0,
        0.02,
    );
    assert_eq!(
        admit_native_analytical_evidence_v1(
            &plan,
            &method,
            &original,
            &policy,
            &wrong_moment,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::AnalyticalEquationMismatch("maximum moment")
    );
}

#[test]
fn conservative_margin_and_model_error_fail_closed() {
    let method = method();
    let input = input(&method, 1000.0);
    let context = context('4');

    let strict_policy = policy(10.3, 0.05);
    let strict_plan = plan(&method, &input, &strict_policy, &context);
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
            &strict_policy,
            &nominal,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::AcceptancePredicateFailed
    );

    let policy = policy(2.0, 0.05);
    let plan = plan(&method, &input, &policy, &context);
    let excessive = result(
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
            &excessive,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::ModelErrorBudgetExceeded
    );
}

#[test]
fn malformed_content_identity_and_nan_are_rejected() {
    assert!(ExecutionArtifactDigestV1::parse("sha256:not-a-digest").is_err());
    assert!(Sha256DigestV1::parse(format!("sha256:{}", "A".repeat(64))).is_err());
    assert!(canonical_binary64_v1(f64::NAN).is_err());
}
