use symthaea_interoception::{
    evaluate_confirmatory_study_bound, execute_study, extract_study_blinded_metrics, DrivePhase,
    EvidenceRunClass, ExecutionLimits, ExclusionCriterion, ExclusionCriterionDecision,
    ExclusionDecisionReceipt, ExclusionDecisionStatus, ExpectedRelation, ExperimentArmSpec,
    ExperimentPreregistration, HypothesisSpec, InteroceptiveDrive, InteroceptiveDynamicsConfig,
    NativeInteroceptiveState, OutcomeRef, RegisteredMeasure, RegisteredMetricSpec, RunDisposition,
    StudyPreregistration, ViabilityChannel, EXCLUSION_DECISION_RECEIPT_SCHEMA_VERSION,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION, INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
    PREREGISTRATION_SCHEMA_VERSION, STUDY_PREREGISTRATION_SCHEMA_VERSION,
};

fn protocol(blind: bool) -> ExperimentPreregistration {
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

    ExperimentPreregistration {
        schema_version: PREREGISTRATION_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        snapshot_schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
        protocol_id: "study-boundary-v1".into(),
        analysis_version: "analysis-v1".into(),
        blind_arm_identity_during_primary_analysis: blind,
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
            description: "Exclude only when preregistered mechanical evidence fails.".into(),
        }],
    }
}

fn study(run_class: EvidenceRunClass) -> StudyPreregistration {
    StudyPreregistration {
        schema_version: STUDY_PREREGISTRATION_SCHEMA_VERSION,
        run_class,
        protocol: protocol(true),
    }
}

fn limits() -> ExecutionLimits {
    ExecutionLimits {
        max_steps_per_arm: 64,
        max_total_steps: 128,
    }
}

fn receipt(
    study: &StudyPreregistration,
    execution: &symthaea_interoception::StudyExecutionTrace,
    status: ExclusionDecisionStatus,
) -> ExclusionDecisionReceipt {
    ExclusionDecisionReceipt {
        schema_version: EXCLUSION_DECISION_RECEIPT_SCHEMA_VERSION,
        run_class: study.run_class,
        study_preregistration_sha256: study.sha256().expect("study digest"),
        study_execution_sha256: execution.sha256().expect("execution digest"),
        decisions: vec![ExclusionCriterionDecision {
            criterion_id: "mechanical-integrity".into(),
            status,
            evidence_sha256: "a".repeat(64),
        }],
    }
}

#[test]
fn exploratory_and_confirmatory_studies_have_distinct_locked_identities() {
    let exploratory = study(EvidenceRunClass::Exploratory);
    let confirmatory = study(EvidenceRunClass::Confirmatory);

    assert_ne!(
        exploratory.sha256().expect("exploratory digest"),
        confirmatory.sha256().expect("confirmatory digest")
    );
}

#[test]
fn confirmatory_study_requires_blinded_primary_analysis() {
    let study = StudyPreregistration {
        schema_version: STUDY_PREREGISTRATION_SCHEMA_VERSION,
        run_class: EvidenceRunClass::Confirmatory,
        protocol: protocol(false),
    };

    let errors = study
        .validate()
        .expect_err("unblinded confirmatory study must fail");
    assert!(errors.iter().any(|error| error.contains("must blind")));
}

#[test]
fn exclusion_receipt_requires_every_registered_decision_and_evidence() {
    let study = study(EvidenceRunClass::Confirmatory);
    let execution = execute_study(&study, limits()).expect("execute study");

    let mut missing = receipt(&study, &execution, ExclusionDecisionStatus::NotTriggered);
    missing.decisions.clear();
    assert!(missing.validate_against(&study, &execution, limits()).is_err());

    let mut malformed = receipt(&study, &execution, ExclusionDecisionStatus::NotTriggered);
    malformed.decisions[0].evidence_sha256 = "not-a-digest".into();
    assert!(malformed
        .validate_against(&study, &execution, limits())
        .is_err());
}

#[test]
fn exclusion_disposition_fails_closed() {
    let study = study(EvidenceRunClass::Confirmatory);
    let execution = execute_study(&study, limits()).expect("execute study");

    let include = receipt(&study, &execution, ExclusionDecisionStatus::NotTriggered);
    assert_eq!(
        include
            .disposition_against(&study, &execution, limits())
            .expect("include disposition"),
        RunDisposition::Include
    );

    let exclude = receipt(&study, &execution, ExclusionDecisionStatus::Triggered);
    assert_eq!(
        exclude
            .disposition_against(&study, &execution, limits())
            .expect("exclude disposition"),
        RunDisposition::Exclude
    );

    let indeterminate = receipt(&study, &execution, ExclusionDecisionStatus::Indeterminate);
    assert_eq!(
        indeterminate
            .disposition_against(&study, &execution, limits())
            .expect("indeterminate disposition"),
        RunDisposition::Indeterminate
    );
}

#[test]
fn exploratory_run_cannot_be_promoted_to_confirmatory_evaluation() {
    let study = study(EvidenceRunClass::Exploratory);
    let execution = execute_study(&study, limits()).expect("execute exploratory study");
    let exclusions = receipt(&study, &execution, ExclusionDecisionStatus::NotTriggered);
    let blinded = extract_study_blinded_metrics(&study, &execution, &exclusions, limits())
        .expect("extract blinded metrics");

    assert!(!blinded.confirmatory_eligible());
    let errors = evaluate_confirmatory_study_bound(
        &study,
        &execution,
        &exclusions,
        &blinded,
        limits(),
    )
    .expect_err("exploratory study must not become confirmatory evidence");
    assert!(errors.iter().any(|error| error.contains("exploratory")));
}

#[test]
fn excluded_or_indeterminate_confirmatory_runs_cannot_be_confirmed() {
    let study = study(EvidenceRunClass::Confirmatory);
    let execution = execute_study(&study, limits()).expect("execute confirmatory study");

    for status in [
        ExclusionDecisionStatus::Triggered,
        ExclusionDecisionStatus::Indeterminate,
    ] {
        let exclusions = receipt(&study, &execution, status);
        let blinded = extract_study_blinded_metrics(&study, &execution, &exclusions, limits())
            .expect("extract blinded metrics");
        assert!(!blinded.confirmatory_eligible());
        assert!(evaluate_confirmatory_study_bound(
            &study,
            &execution,
            &exclusions,
            &blinded,
            limits(),
        )
        .is_err());
    }
}

#[test]
fn tampered_blinded_metrics_are_rejected_before_unblinding() {
    let study = study(EvidenceRunClass::Confirmatory);
    let execution = execute_study(&study, limits()).expect("execute confirmatory study");
    let exclusions = receipt(&study, &execution, ExclusionDecisionStatus::NotTriggered);
    let mut blinded = extract_study_blinded_metrics(&study, &execution, &exclusions, limits())
        .expect("extract blinded metrics");
    blinded.blinded.values[0].value += 0.25;

    let errors = evaluate_confirmatory_study_bound(
        &study,
        &execution,
        &exclusions,
        &blinded,
        limits(),
    )
    .expect_err("tampered blinded metrics must fail exact recomputation");
    assert!(errors
        .iter()
        .any(|error| error.contains("does not exactly reproduce")));
}

#[test]
fn included_confirmatory_run_can_produce_bound_hypothesis_evidence() {
    let study = study(EvidenceRunClass::Confirmatory);
    let execution = execute_study(&study, limits()).expect("execute confirmatory study");
    let exclusions = receipt(&study, &execution, ExclusionDecisionStatus::NotTriggered);
    let blinded = extract_study_blinded_metrics(&study, &execution, &exclusions, limits())
        .expect("extract blinded metrics");

    assert!(blinded.confirmatory_eligible());
    let evaluation = evaluate_confirmatory_study_bound(
        &study,
        &execution,
        &exclusions,
        &blinded,
        limits(),
    )
    .expect("included confirmatory study should evaluate");
    assert_eq!(evaluation.evaluation.outcomes.len(), 1);
    assert!(evaluation.evaluation.outcomes[0].satisfied);
    assert_eq!(evaluation.study_preregistration_sha256, study.sha256().unwrap());
    assert_eq!(evaluation.study_blinded_metric_sha256.len(), 64);
    assert_eq!(evaluation.exclusion_decision_sha256.len(), 64);
}
