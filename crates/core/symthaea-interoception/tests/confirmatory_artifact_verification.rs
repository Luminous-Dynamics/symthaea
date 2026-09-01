use symthaea_interoception::{
    evaluate_confirmatory_study_bound, execute_study, extract_study_blinded_metrics,
    validate_confirmatory_evaluation_bound, DrivePhase, EvidenceRunClass, ExecutionLimits,
    ExclusionCriterion, ExclusionCriterionDecision, ExclusionDecisionReceipt,
    ExclusionDecisionStatus, ExpectedRelation, ExperimentArmSpec, ExperimentPreregistration,
    HypothesisSpec, InteroceptiveDrive, InteroceptiveDynamicsConfig, NativeInteroceptiveState,
    OutcomeRef, RegisteredMeasure, RegisteredMetricSpec, StudyPreregistration, ViabilityChannel,
    EXCLUSION_DECISION_RECEIPT_SCHEMA_VERSION, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
    INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION, PREREGISTRATION_SCHEMA_VERSION,
    STUDY_PREREGISTRATION_SCHEMA_VERSION,
};

fn limits() -> ExecutionLimits {
    ExecutionLimits {
        max_steps_per_arm: 32,
        max_total_steps: 64,
    }
}

fn study() -> StudyPreregistration {
    let control = ExperimentArmSpec {
        arm_id: "control".into(),
        blind_code: "blind-alpha".into(),
        initial_state: NativeInteroceptiveState::default(),
        dynamics_config: InteroceptiveDynamicsConfig::default(),
        phases: vec![DrivePhase {
            steps: 8,
            drive: InteroceptiveDrive::ZERO,
        }],
        interventions: vec![],
    };
    let load = ExperimentArmSpec {
        arm_id: "load".into(),
        blind_code: "blind-beta".into(),
        initial_state: NativeInteroceptiveState::default(),
        dynamics_config: InteroceptiveDynamicsConfig::default(),
        phases: vec![DrivePhase {
            steps: 8,
            drive: InteroceptiveDrive::ZERO
                .with_rate(ViabilityChannel::ComputeReserve, -0.02),
        }],
        interventions: vec![],
    };

    StudyPreregistration {
        schema_version: STUDY_PREREGISTRATION_SCHEMA_VERSION,
        run_class: EvidenceRunClass::Confirmatory,
        protocol: ExperimentPreregistration {
            schema_version: PREREGISTRATION_SCHEMA_VERSION,
            model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
            snapshot_schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
            protocol_id: "confirmatory-artifact-verification-v1".into(),
            analysis_version: "analysis-v1".into(),
            blind_arm_identity_during_primary_analysis: true,
            arms: vec![control, load],
            metrics: vec![RegisteredMetricSpec {
                metric_id: "terminal_weighted".into(),
                measure: RegisteredMeasure::TerminalHomeostaticWeightedDeviation,
            }],
            hypotheses: vec![HypothesisSpec {
                hypothesis_id: "h1".into(),
                primary: true,
                left: OutcomeRef {
                    arm_id: "load".into(),
                    metric_id: "terminal_weighted".into(),
                },
                relation: ExpectedRelation::GreaterThan,
                right: OutcomeRef {
                    arm_id: "control".into(),
                    metric_id: "terminal_weighted".into(),
                },
            }],
            exclusions: vec![ExclusionCriterion {
                criterion_id: "mechanical-integrity".into(),
                description: "Exclude only if preregistered mechanical evidence fails.".into(),
            }],
        },
    }
}

fn exclusions(
    study: &StudyPreregistration,
    execution: &symthaea_interoception::StudyExecutionTrace,
) -> ExclusionDecisionReceipt {
    ExclusionDecisionReceipt {
        schema_version: EXCLUSION_DECISION_RECEIPT_SCHEMA_VERSION,
        run_class: EvidenceRunClass::Confirmatory,
        study_preregistration_sha256: study.sha256().expect("study digest"),
        study_execution_sha256: execution.sha256().expect("execution digest"),
        decisions: vec![ExclusionCriterionDecision {
            criterion_id: "mechanical-integrity".into(),
            status: ExclusionDecisionStatus::NotTriggered,
            evidence_sha256: "a".repeat(64),
        }],
    }
}

#[test]
fn stored_confirmatory_evaluation_must_reproduce_from_complete_locked_evidence() {
    let study = study();
    let execution = execute_study(&study, limits()).expect("execute study");
    let exclusions = exclusions(&study, &execution);
    let blinded = extract_study_blinded_metrics(&study, &execution, &exclusions, limits())
        .expect("extract blinded metrics");
    let evaluation = evaluate_confirmatory_study_bound(
        &study,
        &execution,
        &exclusions,
        &blinded,
        limits(),
    )
    .expect("bound confirmatory evaluation");

    validate_confirmatory_evaluation_bound(
        &study,
        &execution,
        &exclusions,
        &blinded,
        &evaluation,
        limits(),
    )
    .expect("stored evaluation must validate");

    let mut tampered = evaluation.clone();
    tampered.evaluation.outcomes[0].satisfied = !tampered.evaluation.outcomes[0].satisfied;

    let errors = validate_confirmatory_evaluation_bound(
        &study,
        &execution,
        &exclusions,
        &blinded,
        &tampered,
        limits(),
    )
    .expect_err("tampered semantic evaluation must fail exact recomputation");
    assert!(errors
        .iter()
        .any(|error| error.contains("does not exactly reproduce")));
}
