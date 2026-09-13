// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent public-API regression for evaluation-time identity binding.

use symthaea_engineering_analysis_trust::{
    AcceptanceRecordDigestV1, AlgorithmRevisionDigestV1, AnalysisConfigurationDigestV1,
    AcceptedAnalysisRequirementV1, AnalyticalAcceptancePolicyV1, AnalyticalMethodV1,
    CurrentnessAssertionV2, CurrentnessAttestationDigestV1, ExecutionArtifactDigestV1,
    ImplementationArtifactDigestV1, ModelQualificationRecordDigestV1, ModelRevisionDigestV1,
    NativeAnalyticalPlanV1, NativeAnalyticalResultV1, RectangularCantileverInputV1,
    Sha256DigestV1, SubjectRevisionV1, SubjectStateDigestV1, TwinKindV1, TwinRevisionV1,
    TwinSchemaDigestV1, TwinStateDigestV1, ValidityDimensionDigestV1,
    ValidityDomainRevisionV1, admit_native_analytical_evidence_v1,
    derive_current_native_analytical_discharge_fact_v2,
    issue_native_analytical_discharge_receipt_v1,
};
use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation};

fn raw(ch: char) -> Sha256DigestV1 {
    Sha256DigestV1::parse(format!("sha256:{}", ch.to_string().repeat(64))).unwrap()
}

macro_rules! premise {
    ($ty:ty, $ch:expr) => {
        <$ty>::from_digest(raw($ch))
    };
}

#[test]
fn evaluation_time_changes_present_fact_identity() {
    let requirement = AcceptedAnalysisRequirementV1::civil_service_stress_250_mpa(premise!(
        AcceptanceRecordDigestV1,
        'a'
    ));
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
        vec![("load_case".into(), premise!(ValidityDimensionDigestV1, '1'))],
    )
    .unwrap();

    const OBSERVED: u64 = 1_789_123_456_000;
    const VALID_UNTIL: u64 = 1_789_209_856_000;
    let currentness = CurrentnessAssertionV2::new(
        &twin,
        &validity,
        premise!(CurrentnessAttestationDigestV1, '4'),
        OBSERVED,
        VALID_UNTIL,
    )
    .unwrap();

    let obligation = ProofObligation {
        id: "00000000-0000-4000-8000-000000000042".parse().unwrap(),
        claim: "stress remains below allowable under service load".into(),
        expected_evidence: EvidenceKind::Analysis,
        status: ObligationStatus::Open,
        evidence_refs: Vec::new(),
    };
    let method = AnalyticalMethodV1::euler_bernoulli_beam(
        premise!(ImplementationArtifactDigestV1, 'a'),
        premise!(AlgorithmRevisionDigestV1, 'b'),
    );
    let input = RectangularCantileverInputV1::new(
        &method, 2.0, 0.05, 0.1, 200.0e9, 250.0e6, 1000.0,
    )
    .unwrap();
    let policy = AnalyticalAcceptancePolicyV1::factor_of_safety_ge(
        2.0,
        0.05,
        premise!(ModelQualificationRecordDigestV1, '9'),
    )
    .unwrap();
    let plan = NativeAnalyticalPlanV1::new(
        &requirement,
        &subject,
        &twin,
        &validity,
        &currentness,
        &obligation,
        &method,
        &input,
        &policy,
    )
    .unwrap();
    let result = NativeAnalyticalResultV1::for_input(
        &method,
        &input,
        premise!(ExecutionArtifactDigestV1, '8'),
        10.416666666666666,
        24.0e6,
        0.0032,
        2000.0,
        0.02,
    )
    .unwrap();
    let admitted =
        admit_native_analytical_evidence_v1(&plan, &method, &input, &policy, &result).unwrap();
    let receipt = issue_native_analytical_discharge_receipt_v1(&plan, &admitted).unwrap();

    let at_observation =
        derive_current_native_analytical_discharge_fact_v2(&plan, &receipt, OBSERVED).unwrap();
    let one_millisecond_later =
        derive_current_native_analytical_discharge_fact_v2(&plan, &receipt, OBSERVED + 1).unwrap();

    assert_ne!(at_observation.fact_id(), one_millisecond_later.fact_id());
}
