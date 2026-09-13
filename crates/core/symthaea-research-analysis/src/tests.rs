use symthaea_research_protocol::{
    AnalysisPlanRef, BaselineSpec, CanonicalConfirmatoryProtocol, CanonicalConfirmatoryRunBinding,
    ConfirmatoryDecisionRule, ConfirmatoryEndpointSpec, HypothesisDirection, HypothesisRole,
    HypothesisSpec, MetricRole, MetricSchemaBinding, MetricSpec, MetricValueSchema,
    MultiplicityPolicy, ResearchProtocol, ResearchRunRegistration, StoppingRule,
};
use symthaea_research_result::{
    ClaimDisposition, ClaimInterpretation, CompleteEndpointResearchResultV2,
    ConfirmatoryClaimBinding, ConfirmatoryResearchResultV2, MetricOutcome, MetricResult,
    ResearchResultManifest, ResultArtifactKind, ResultArtifactRef, ResultClaim,
};

use crate::{
    AnalysisExecutionStatus, AnalysisInputBindingV1, AnalysisInputRole, AnalysisOutputBindingV1,
    AnalysisVerificationClass, AnalysisVerifierResult, AnalysisVerifierV1,
    ExternalAnalysisQualifiedResultV1, FrozenAnalysisExecutionReceiptV1,
    FrozenAnalysisInputSpecV1, FrozenExternalAnalysisPlanV1, ResearchAnalysisError,
};

fn protocol() -> CanonicalConfirmatoryProtocol {
    let frozen = ResearchProtocol::new(
        "external-analysis-v1",
        "1",
        "Does the frozen external comparison support the primary hypothesis?",
        vec![HypothesisSpec::new(
            "h-score",
            "score exceeds the frozen comparator under the declared analysis",
            HypothesisRole::Primary,
            HypothesisDirection::GreaterThan,
        )
        .unwrap()],
        vec![MetricSpec::new(
            "score",
            "primary score",
            "points",
            MetricRole::Primary,
            "held-out mean",
        )
        .unwrap()],
        vec![BaselineSpec::new(
            "baseline",
            "frozen baseline",
            "bench/baseline-v1",
        )
        .unwrap()],
        vec![],
        StoppingRule::FixedEpisodeCount(8),
        MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
        AnalysisPlanRef::new("analysis", "1", "sha256:analysis-plan").unwrap(),
        "frozen scenario set",
        "frozen seeds",
    )
    .unwrap()
    .freeze(1_000)
    .unwrap();

    CanonicalConfirmatoryProtocol::new(
        frozen,
        vec![MetricSchemaBinding::new(
            "score",
            MetricValueSchema::numeric("points").unwrap(),
        )
        .unwrap()],
        vec![ConfirmatoryEndpointSpec::new(
            "score-endpoint",
            "h-score",
            vec!["score".into()],
            vec!["baseline".into()],
            ConfirmatoryDecisionRule::external("analysis/score", "sha256:rule").unwrap(),
        )
        .unwrap()],
    )
    .unwrap()
}

fn run_binding(protocol: &CanonicalConfirmatoryProtocol) -> CanonicalConfirmatoryRunBinding {
    let run = ResearchRunRegistration::new(
        protocol.frozen_protocol(),
        "run-001",
        1_100,
        "deadbeef",
        "sha256:dataset",
        "sha256:repro",
        "sha256:seeds",
    )
    .unwrap();
    protocol.bind_run(run).unwrap()
}

fn plan(
    protocol: &CanonicalConfirmatoryProtocol,
    minimum: AnalysisVerificationClass,
) -> FrozenExternalAnalysisPlanV1 {
    FrozenExternalAnalysisPlanV1::new(
        protocol,
        "score-endpoint",
        "sha256:analysis-executable",
        "sha256:argv-config-entrypoint",
        "sha256:toolchain",
        "sha256:environment",
        minimum,
        vec![
            FrozenAnalysisInputSpecV1::new(
                AnalysisInputRole::endpoint_metric("score"),
                None,
                "sha256:score-schema",
            )
            .unwrap(),
            FrozenAnalysisInputSpecV1::new(
                AnalysisInputRole::baseline_observation("baseline"),
                Some("sha256:baseline-source-plan".into()),
                "sha256:baseline-schema",
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn complete_result(
    protocol: &CanonicalConfirmatoryProtocol,
    run_binding: &CanonicalConfirmatoryRunBinding,
    disposition: ClaimDisposition,
) -> CompleteEndpointResearchResultV2 {
    let metric_artifact = ResultArtifactRef::new(
        "score-input",
        ResultArtifactKind::Metrics,
        "sha256:score-input",
        "serialized endpoint metric input",
    )
    .unwrap();
    let baseline_artifact = ResultArtifactRef::new(
        "baseline-input",
        ResultArtifactKind::Metrics,
        "sha256:baseline-input",
        "serialized baseline observation",
    )
    .unwrap();
    let analysis_artifact = ResultArtifactRef::new(
        "analysis-output",
        ResultArtifactKind::Analysis,
        "sha256:analysis-output",
        "frozen external analysis output",
    )
    .unwrap();
    let metric = MetricResult::new(
        "score",
        MetricOutcome::Numeric {
            value: 2.0,
            unit: "points".into(),
        },
    )
    .unwrap()
    .with_artifact("score-input")
    .unwrap();
    let claim = ResultClaim::new(
        "claim-score",
        "frozen external analysis supports the primary hypothesis",
        disposition,
        ClaimInterpretation::Confirmatory,
    )
    .unwrap()
    .for_hypothesis("h-score")
    .unwrap()
    .with_metric("score")
    .unwrap()
    .with_artifact("analysis-output")
    .unwrap();
    let result = ResearchResultManifest::new(
        protocol.frozen_protocol(),
        run_binding.run().clone(),
        "result-001",
        2_000,
        vec![],
        vec![],
        false,
        vec![metric_artifact, baseline_artifact, analysis_artifact],
        vec![metric],
        vec![claim],
    )
    .unwrap();
    let v2 = ConfirmatoryResearchResultV2::new(
        protocol,
        run_binding,
        result,
        vec![ConfirmatoryClaimBinding::new("claim-score", "score-endpoint")
            .unwrap()
            .with_analysis_artifact("analysis-output")
            .unwrap()],
    )
    .unwrap();
    CompleteEndpointResearchResultV2::new(protocol, run_binding, v2).unwrap()
}

fn verifier(
    class: AnalysisVerificationClass,
    reexecuted_output_digest: Option<String>,
) -> AnalysisVerifierV1 {
    AnalysisVerifierV1::new(
        "verifier-v1",
        "sha256:verifier-identity",
        class == AnalysisVerificationClass::IndependentReexecutionExact,
        class,
        "sha256:verification-evidence",
        reexecuted_output_digest,
        AnalysisVerifierResult::Passed,
    )
    .unwrap()
}

fn receipt(
    protocol: &CanonicalConfirmatoryProtocol,
    run_binding: &CanonicalConfirmatoryRunBinding,
    plan: &FrozenExternalAnalysisPlanV1,
    verifier: AnalysisVerifierV1,
) -> crate::Result<FrozenAnalysisExecutionReceiptV1> {
    let inputs = vec![
        AnalysisInputBindingV1::new(
            &plan.ordered_inputs[0],
            run_binding,
            "score-input",
            "sha256:score-input",
        )?,
        AnalysisInputBindingV1::new(
            &plan.ordered_inputs[1],
            run_binding,
            "baseline-input",
            "sha256:baseline-input",
        )?,
    ];
    FrozenAnalysisExecutionReceiptV1::new(
        protocol,
        run_binding,
        plan,
        inputs,
        Some(AnalysisOutputBindingV1::new(
            "analysis-output",
            "sha256:analysis-output",
        )?),
        AnalysisExecutionStatus::Completed,
        Some("sha256:execution-witness".into()),
        verifier,
        vec!["sha256:custody-receipt-b".into(), "sha256:custody-receipt-a".into()],
    )
}

#[test]
fn baseline_observation_requires_precommitted_source_plan() {
    let error = FrozenAnalysisInputSpecV1::new(
        AnalysisInputRole::baseline_observation("baseline"),
        None,
        "sha256:baseline-schema",
    )
    .unwrap_err();
    assert!(matches!(error, ResearchAnalysisError::MissingSourcePlan(_)));
}

#[test]
fn invocation_digest_changes_plan_identity() {
    let protocol = protocol();
    let first = plan(
        &protocol,
        AnalysisVerificationClass::ExecutionWitnessBindingValidated,
    );
    let second = FrozenExternalAnalysisPlanV1::new(
        &protocol,
        "score-endpoint",
        "sha256:analysis-executable",
        "sha256:different-argv-config-entrypoint",
        "sha256:toolchain",
        "sha256:environment",
        AnalysisVerificationClass::ExecutionWitnessBindingValidated,
        first.ordered_inputs.clone(),
    )
    .unwrap();
    assert_ne!(first.digest, second.digest);
}

#[test]
fn binding_only_verification_cannot_support_evidentiary_conclusion() {
    let protocol = protocol();
    let run_binding = run_binding(&protocol);
    let plan = plan(&protocol, AnalysisVerificationClass::BindingOnly);
    let receipt = receipt(
        &protocol,
        &run_binding,
        &plan,
        verifier(AnalysisVerificationClass::BindingOnly, None),
    )
    .unwrap();
    assert!(!receipt.supports_evidentiary_conclusion(&plan));

    let result = complete_result(
        &protocol,
        &run_binding,
        ClaimDisposition::ConsistentWithHypothesis,
    );
    let error = ExternalAnalysisQualifiedResultV1::new(
        &protocol,
        &run_binding,
        result,
        vec![plan],
        vec![receipt],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ResearchAnalysisError::InsufficientExecutionVerification(_)
    ));
}

#[test]
fn exact_reexecution_output_mismatch_is_first_class_failure() {
    let protocol = protocol();
    let run_binding = run_binding(&protocol);
    let plan = plan(
        &protocol,
        AnalysisVerificationClass::DeterministicReexecutionExact,
    );
    let error = receipt(
        &protocol,
        &run_binding,
        &plan,
        verifier(
            AnalysisVerificationClass::DeterministicReexecutionExact,
            Some("sha256:different-output".into()),
        ),
    )
    .unwrap_err();
    assert_eq!(error, ResearchAnalysisError::ReexecutionOutputMismatch);
}

#[test]
fn weaker_verifier_than_frozen_minimum_is_rejected() {
    let protocol = protocol();
    let run_binding = run_binding(&protocol);
    let plan = plan(
        &protocol,
        AnalysisVerificationClass::DeterministicReexecutionExact,
    );
    let error = receipt(
        &protocol,
        &run_binding,
        &plan,
        verifier(AnalysisVerificationClass::ExecutionWitnessBindingValidated, None),
    )
    .unwrap_err();
    assert!(matches!(error, ResearchAnalysisError::InvalidVerifier(_)));
}

#[test]
fn custody_receipt_order_is_canonicalized_without_claiming_custody_validation() {
    let protocol = protocol();
    let run_binding = run_binding(&protocol);
    let plan = plan(
        &protocol,
        AnalysisVerificationClass::ExecutionWitnessBindingValidated,
    );
    let receipt = receipt(
        &protocol,
        &run_binding,
        &plan,
        verifier(AnalysisVerificationClass::ExecutionWitnessBindingValidated, None),
    )
    .unwrap();
    assert_eq!(
        receipt.custody_access_receipt_digests,
        vec![
            "sha256:custody-receipt-a".to_string(),
            "sha256:custody-receipt-b".to_string(),
        ]
    );
}

#[test]
fn exact_bound_execution_receipt_qualifies_external_result_relationship() {
    let protocol = protocol();
    let run_binding = run_binding(&protocol);
    let plan = plan(
        &protocol,
        AnalysisVerificationClass::ExecutionWitnessBindingValidated,
    );
    let receipt = receipt(
        &protocol,
        &run_binding,
        &plan,
        verifier(AnalysisVerificationClass::ExecutionWitnessBindingValidated, None),
    )
    .unwrap();
    let result = complete_result(
        &protocol,
        &run_binding,
        ClaimDisposition::ConsistentWithHypothesis,
    );
    let qualified = ExternalAnalysisQualifiedResultV1::new(
        &protocol,
        &run_binding,
        result,
        vec![plan],
        vec![receipt],
    )
    .unwrap();
    qualified.validate_against(&protocol, &run_binding).unwrap();
}

#[test]
fn missing_receipt_cannot_qualify_external_endpoint() {
    let protocol = protocol();
    let run_binding = run_binding(&protocol);
    let plan = plan(
        &protocol,
        AnalysisVerificationClass::ExecutionWitnessBindingValidated,
    );
    let result = complete_result(
        &protocol,
        &run_binding,
        ClaimDisposition::ConsistentWithHypothesis,
    );
    let error = ExternalAnalysisQualifiedResultV1::new(
        &protocol,
        &run_binding,
        result,
        vec![plan],
        vec![],
    )
    .unwrap_err();
    assert!(matches!(error, ResearchAnalysisError::MissingReceipt(_)));
}
