use symthaea_interoception::{
    execute_preregistration, DrivePhase, ExecutionLimits, ExclusionCriterion, ExpectedRelation,
    ExperimentArmSpec, ExperimentPreregistration, HypothesisSpec, InteroceptiveDrive,
    InteroceptiveDynamicsConfig, InteroceptiveIntervention, NativeInteroceptiveState, OutcomeRef,
    RegisteredMeasure, RegisteredMetricSpec, ScheduledIntervention, ViabilityChannel,
    EXECUTION_TRACE_SCHEMA_VERSION, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
    INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION, PREREGISTRATION_SCHEMA_VERSION,
};

fn protocol() -> ExperimentPreregistration {
    let passive = ExperimentArmSpec {
        arm_id: "semantic-control".into(),
        blind_code: "blind-a17".into(),
        initial_state: NativeInteroceptiveState::default(),
        dynamics_config: InteroceptiveDynamicsConfig::default(),
        phases: vec![DrivePhase {
            steps: 8,
            drive: InteroceptiveDrive::ZERO,
        }],
        interventions: vec![ScheduledIntervention {
            before_step: 0,
            intervention: InteroceptiveIntervention::add(ViabilityChannel::Integrity, -0.05),
        }],
    };
    let driven = ExperimentArmSpec {
        arm_id: "semantic-load".into(),
        blind_code: "blind-q42".into(),
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
        protocol_id: "executor-replay-v1".into(),
        analysis_version: "analysis-v1".into(),
        blind_arm_identity_during_primary_analysis: true,
        arms: vec![passive, driven],
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
            relation: ExpectedRelation::GreaterThan,
            right: OutcomeRef {
                arm_id: "semantic-control".into(),
                metric_id: "terminal_weighted".into(),
            },
        }],
        exclusions: vec![ExclusionCriterion {
            criterion_id: "integrity".into(),
            description: "Exclude only on preregistered mechanical integrity failure.".into(),
        }],
    }
}

fn limits() -> ExecutionLimits {
    ExecutionLimits {
        max_steps_per_arm: 64,
        max_total_steps: 128,
    }
}

#[test]
fn execution_is_deterministic_and_exactly_replayable() {
    let protocol = protocol();
    let left = execute_preregistration(&protocol, limits()).expect("execute left");
    let right = execute_preregistration(&protocol, limits()).expect("execute right");

    assert_eq!(left, right);
    assert_eq!(left.schema_version, EXECUTION_TRACE_SCHEMA_VERSION);
    assert_eq!(left.protocol_sha256.len(), 64);
    assert_eq!(left.resolved_config_sha256.len(), 64);
    assert_eq!(left.input_sequence_sha256.len(), 64);
    left.validate_against(&protocol, limits())
        .expect("trace must replay exactly");
}

#[test]
fn execution_trace_keeps_semantic_arm_ids_out_of_blinded_export() {
    let protocol = protocol();
    let trace = execute_preregistration(&protocol, limits()).expect("execute protocol");
    let json = serde_json::to_string(&trace).expect("serialize trace");

    assert!(!json.contains("semantic-control"));
    assert!(!json.contains("semantic-load"));
    assert!(json.contains("blind-a17"));
    assert!(json.contains("blind-q42"));
}

#[test]
fn tampered_trace_fails_protocol_replay_validation() {
    let protocol = protocol();
    let mut trace = execute_preregistration(&protocol, limits()).expect("execute protocol");
    trace.arms[0].steps[0].homeostasis.weighted_deviation += 0.1;

    assert!(trace.validate_against(&protocol, limits()).is_err());
}

#[test]
fn scheduled_intervention_executes_before_the_declared_step() {
    let protocol = protocol();
    let trace = execute_preregistration(&protocol, limits()).expect("execute protocol");
    let first = &trace.arms[0].steps[0];

    assert_eq!(first.step_index, 0);
    assert_eq!(first.intervention_records.len(), 1);
    assert_eq!(first.intervention_records[0].cycle, 0);
    assert_eq!(first.transition.cycle_before, 0);
    assert_eq!(first.transition.cycle_after, 1);
}

#[test]
fn execution_limits_are_hard_failures_not_silent_truncation() {
    let protocol = protocol();
    let too_small = ExecutionLimits {
        max_steps_per_arm: 7,
        max_total_steps: 128,
    };

    let errors = execute_preregistration(&protocol, too_small)
        .expect_err("execution must reject rather than truncate");
    assert!(errors
        .iter()
        .any(|error| error.contains("exceeding max_steps_per_arm")));
}
