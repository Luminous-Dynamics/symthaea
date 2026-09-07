use symthaea_interoception::{
    evaluate_hypotheses, execute_preregistration, extract_blinded_metrics, DrivePhase,
    ExecutionLimits, ExclusionCriterion, ExpectedRelation, ExperimentArmSpec,
    ExperimentPreregistration, HypothesisSpec, InteroceptiveDrive, InteroceptiveDynamicsConfig,
    NativeInteroceptiveState, OutcomeRef, RegisteredMeasure, RegisteredMetricSpec,
    ViabilityChannel, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
    INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION, PREREGISTRATION_SCHEMA_VERSION,
};

fn protocol() -> ExperimentPreregistration {
    let control = ExperimentArmSpec {
        arm_id: "semantic-control".into(),
        blind_code: "blind-x3".into(),
        initial_state: NativeInteroceptiveState::default(),
        dynamics_config: InteroceptiveDynamicsConfig::default(),
        phases: vec![DrivePhase {
            steps: 8,
            drive: InteroceptiveDrive::ZERO,
        }],
        interventions: vec![],
    };
    let load = ExperimentArmSpec {
        arm_id: "semantic-load".into(),
        blind_code: "blind-m8".into(),
        initial_state: NativeInteroceptiveState::default(),
        dynamics_config: InteroceptiveDynamicsConfig::default(),
        phases: vec![DrivePhase {
            steps: 8,
            drive: InteroceptiveDrive::ZERO
                .with_rate(ViabilityChannel::ComputeReserve, -0.10),
        }],
        interventions: vec![],
    };

    ExperimentPreregistration {
        schema_version: PREREGISTRATION_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        snapshot_schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
        protocol_id: "blinded-analysis-v1".into(),
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
                arm_id: "semantic-load".into(),
                metric_id: "terminal_weighted".into(),
            },
            relation: ExpectedRelation::GreaterByAtLeast {
                minimum_difference: 0.10,
            },
            right: OutcomeRef {
                arm_id: "semantic-control".into(),
                metric_id: "terminal_weighted".into(),
            },
        }],
        exclusions: vec![ExclusionCriterion {
            criterion_id: "mechanical-integrity".into(),
            description: "Exclude only on preregistered mechanical integrity failure.".into(),
        }],
    }
}

fn limits() -> ExecutionLimits {
    ExecutionLimits {
        max_steps_per_arm: 32,
        max_total_steps: 64,
    }
}

#[test]
fn blinded_metrics_can_be_locked_before_hypothesis_unblinding() {
    let protocol = protocol();
    let trace = execute_preregistration(&protocol, limits()).expect("execute protocol");
    let blinded =
        extract_blinded_metrics(&trace, &protocol, limits()).expect("extract blinded metrics");
    let locked_digest = blinded.sha256().expect("hash blinded metrics");

    let json = serde_json::to_string(&blinded).expect("serialize blinded metrics");
    assert!(!json.contains("semantic-control"));
    assert!(!json.contains("semantic-load"));
    assert!(json.contains("blind-x3"));
    assert!(json.contains("blind-m8"));

    let evaluation = evaluate_hypotheses(&protocol, &blinded).expect("evaluate hypotheses");
    assert_eq!(evaluation.blinded_metric_sha256, locked_digest);
    assert_eq!(evaluation.outcomes.len(), 1);
    assert!(evaluation.outcomes[0].satisfied);
}

#[test]
fn blinded_report_rejects_duplicate_metric_pairs() {
    let protocol = protocol();
    let trace = execute_preregistration(&protocol, limits()).expect("execute protocol");
    let mut blinded =
        extract_blinded_metrics(&trace, &protocol, limits()).expect("extract blinded metrics");
    blinded.values.push(blinded.values[0].clone());

    assert!(blinded.validate_against(&protocol).is_err());
}

#[test]
fn minimum_effect_relations_do_not_reduce_to_direction_only() {
    let relation = ExpectedRelation::GreaterByAtLeast {
        minimum_difference: 0.5,
    };
    assert!(relation.is_satisfied_by(1.0, 0.4));
    assert!(!relation.is_satisfied_by(1.0, 0.6));

    let inverse = ExpectedRelation::LessByAtLeast {
        minimum_difference: 0.5,
    };
    assert!(inverse.is_satisfied_by(0.4, 1.0));
    assert!(!inverse.is_satisfied_by(0.6, 1.0));
}

#[test]
fn tampering_with_trace_blocks_metric_extraction() {
    let protocol = protocol();
    let mut trace = execute_preregistration(&protocol, limits()).expect("execute protocol");
    trace.arms[0].steps[0].homeostasis.weighted_deviation = 0.25;

    assert!(extract_blinded_metrics(&trace, &protocol, limits()).is_err());
}
