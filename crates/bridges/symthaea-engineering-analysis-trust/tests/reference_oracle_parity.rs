// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-API parity with the independent ETK-3C native analytical oracle.

use symthaea_engineering_analysis_trust::{
    AnalysisTrustErrorV1, AnalyticalAcceptancePolicyV1, AnalyticalMethodV1,
    CurrentnessAssertionIdV1, ExecutionArtifactDigestV1, ModelQualificationRecordDigestV1,
    NativeAnalyticalPlanV1, NativeAnalyticalResultV1, RectangularCantileverInputV1,
    RequirementRevisionIdV1, Sha256DigestV1, SubjectRevisionIdV1, TwinRevisionIdV1,
    ValidityDomainRevisionIdV1, admit_native_analytical_evidence_v1,
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

fn method() -> AnalyticalMethodV1 {
    AnalyticalMethodV1::euler_bernoulli_beam(
        Sha256DigestV1::parse(digest('a')).unwrap(),
        Sha256DigestV1::parse(digest('b')).unwrap(),
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

fn policy() -> AnalyticalAcceptancePolicyV1 {
    AnalyticalAcceptancePolicyV1::factor_of_safety_ge(
        2.0,
        0.05,
        ModelQualificationRecordDigestV1::parse(digest('9')).unwrap(),
    )
    .unwrap()
}

fn plan_with_currentness(
    method: &AnalyticalMethodV1,
    input: &RectangularCantileverInputV1,
    policy: &AnalyticalAcceptancePolicyV1,
    currentness: &str,
) -> NativeAnalyticalPlanV1 {
    NativeAnalyticalPlanV1::new(
        RequirementRevisionIdV1::parse(REQUIREMENT).unwrap(),
        SubjectRevisionIdV1::parse(SUBJECT).unwrap(),
        TwinRevisionIdV1::parse(TWIN).unwrap(),
        ValidityDomainRevisionIdV1::parse(VALIDITY).unwrap(),
        CurrentnessAssertionIdV1::parse(currentness).unwrap(),
        &obligation(),
        method,
        input,
        policy,
    )
    .unwrap()
}

fn result(
    factor_of_safety: f64,
    max_bending_stress_pa: f64,
    model_relative_error_bound: f64,
) -> NativeAnalyticalResultV1 {
    NativeAnalyticalResultV1::new(
        ExecutionArtifactDigestV1::parse(digest('8')).unwrap(),
        factor_of_safety,
        max_bending_stress_pa,
        0.0032,
        2000.0,
        model_relative_error_bound,
    )
    .unwrap()
}

#[test]
fn independent_vectors_compose_end_to_end() {
    assert_eq!(
        analytical_obligation_revision_v1(&obligation())
            .unwrap()
            .as_str(),
        OBLIGATION
    );
    assert_eq!(canonical_binary64_v1(-0.0).unwrap(), "f64:0000000000000000");
    assert_eq!(canonical_binary64_v1(0.1).unwrap(), "f64:3fb999999999999a");

    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy();
    assert_eq!(method.revision_id().as_str(), METHOD);
    assert_eq!(input.revision_id().as_str(), INPUT);
    assert_eq!(policy.revision_id().as_str(), POLICY);

    let plan = plan_with_currentness(&method, &input, &policy, CURRENTNESS);
    assert_eq!(plan.plan_id().as_str(), PLAN);
    assert_eq!(plan.obligation_revision_id().as_str(), OBLIGATION);

    let result = result(10.416666666666666, 24.0e6, 0.02);
    let admitted =
        admit_native_analytical_evidence_v1(&plan, &method, &input, &policy, &result).unwrap();
    assert_eq!(admitted.admitted_evidence_id().as_str(), ADMITTED);

    let receipt = issue_native_analytical_discharge_receipt_v1(&plan, &admitted).unwrap();
    assert_eq!(receipt.receipt_id().as_str(), RECEIPT);

    let current = derive_current_native_analytical_discharge_fact_v1(&plan, &receipt).unwrap();
    assert_eq!(current.fact_id().as_str(), CURRENT_FACT);
    assert_eq!(current.witness_receipt_id().as_str(), RECEIPT);
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
fn input_drift_cannot_reuse_an_old_plan() {
    let method = method();
    let original = input(&method, 1000.0);
    let changed = input(&method, 1000.0000000000001);
    let policy = policy();
    let plan = plan_with_currentness(&method, &original, &policy, CURRENTNESS);
    assert_ne!(original.revision_id(), changed.revision_id());

    let result = result(10.416666666666666, 24.0e6, 0.02);
    assert_eq!(
        admit_native_analytical_evidence_v1(&plan, &method, &changed, &policy, &result)
            .unwrap_err(),
        AnalysisTrustErrorV1::PlanBindingMismatch
    );
}

#[test]
fn conservative_margin_and_model_error_fail_closed() {
    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy();
    let plan = plan_with_currentness(&method, &input, &policy, CURRENTNESS);

    let low_fos = 2.01;
    let low_stress = 250.0e6 / low_fos;
    let low = result(low_fos, low_stress, 0.02);
    assert_eq!(
        admit_native_analytical_evidence_v1(&plan, &method, &input, &policy, &low)
            .unwrap_err(),
        AnalysisTrustErrorV1::AcceptancePredicateFailed
    );

    let excessive_error = result(10.416666666666666, 24.0e6, 0.06);
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
}

#[test]
fn inconsistent_factor_of_safety_fails_closed() {
    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy();
    let plan = plan_with_currentness(&method, &input, &policy, CURRENTNESS);
    let inconsistent = result(10.416666666666666, 25.0e6, 0.02);
    assert_eq!(
        admit_native_analytical_evidence_v1(
            &plan,
            &method,
            &input,
            &policy,
            &inconsistent,
        )
        .unwrap_err(),
        AnalysisTrustErrorV1::InconsistentFactorOfSafety
    );
}

#[test]
fn currentness_refresh_makes_old_receipt_historical() {
    let method = method();
    let input = input(&method, 1000.0);
    let policy = policy();
    let historical = plan_with_currentness(&method, &input, &policy, CURRENTNESS);
    let refreshed = plan_with_currentness(&method, &input, &policy, REFRESHED_CURRENTNESS);
    assert_ne!(historical.plan_id(), refreshed.plan_id());

    let result = result(10.416666666666666, 24.0e6, 0.02);
    let admitted = admit_native_analytical_evidence_v1(
        &historical,
        &method,
        &input,
        &policy,
        &result,
    )
    .unwrap();
    let receipt =
        issue_native_analytical_discharge_receipt_v1(&historical, &admitted).unwrap();

    assert_eq!(
        derive_current_native_analytical_discharge_fact_v1(&refreshed, &receipt).unwrap_err(),
        AnalysisTrustErrorV1::HistoricalPlan
    );
}

#[test]
fn malformed_content_identity_is_rejected_before_authority() {
    assert!(ExecutionArtifactDigestV1::parse("sha256:not-a-digest").is_err());
    assert!(Sha256DigestV1::parse(format!("sha256:{}", "A".repeat(64))).is_err());
    assert!(canonical_binary64_v1(f64::NAN).is_err());
}
