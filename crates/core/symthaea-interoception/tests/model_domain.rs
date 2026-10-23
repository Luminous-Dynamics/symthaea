use symthaea_interoception::{
    DrivePhase, ExclusionCriterion, ExpectedRelation, ExperimentArmSpec, ExperimentPreregistration,
    HypothesisSpec, InteroceptiveDrive, InteroceptiveDynamicsConfig, InteroceptiveSnapshot,
    NativeInteroceptiveModel, NativeInteroceptiveState, OutcomeRef, RegisteredMeasure,
    RegisteredMetricSpec, ViabilityChannel, AllostaticConfig,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION, INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
    PREREGISTRATION_SCHEMA_VERSION,
};

fn out_of_domain_state() -> NativeInteroceptiveState {
    NativeInteroceptiveState::default().with_value(ViabilityChannel::ComputeReserve, 1.25)
}

#[test]
fn model_try_new_rejects_state_outside_declared_numeric_domain() {
    let error = NativeInteroceptiveModel::try_new(
        out_of_domain_state(),
        InteroceptiveDynamicsConfig::default(),
    )
    .expect_err("out-of-domain initial state must fail");

    assert!(error.contains("compute_reserve"));
    assert!(error.contains("outside model domain"));
}

#[test]
fn preregistration_rejects_out_of_domain_initial_state_before_execution() {
    let arm = ExperimentArmSpec {
        arm_id: "invalid-domain".into(),
        blind_code: "blind-invalid".into(),
        initial_state: out_of_domain_state(),
        dynamics_config: InteroceptiveDynamicsConfig::default(),
        phases: vec![DrivePhase {
            steps: 4,
            drive: InteroceptiveDrive::ZERO,
        }],
        interventions: vec![],
    };

    let protocol = ExperimentPreregistration {
        schema_version: PREREGISTRATION_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        snapshot_schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
        protocol_id: "invalid-domain-v1".into(),
        analysis_version: "analysis-v1".into(),
        blind_arm_identity_during_primary_analysis: true,
        arms: vec![arm],
        metrics: vec![RegisteredMetricSpec {
            metric_id: "terminal".into(),
            measure: RegisteredMeasure::TerminalHomeostaticWeightedDeviation,
        }],
        hypotheses: vec![HypothesisSpec {
            hypothesis_id: "h1".into(),
            primary: true,
            left: OutcomeRef {
                arm_id: "invalid-domain".into(),
                metric_id: "terminal".into(),
            },
            relation: ExpectedRelation::EqualWithin {
                absolute_tolerance: 0.0,
            },
            right: OutcomeRef {
                arm_id: "invalid-domain".into(),
                metric_id: "terminal".into(),
            },
        }],
        exclusions: vec![ExclusionCriterion {
            criterion_id: "mechanical".into(),
            description: "mechanical integrity".into(),
        }],
    };

    let errors = protocol
        .validate()
        .expect_err("out-of-domain initial state must invalidate preregistration");
    assert!(errors
        .iter()
        .any(|error| error.contains("invalid dynamics/state contract")));
}

#[test]
fn snapshot_deserialization_rejects_domain_mismatch_without_constructing_invalid_model() {
    let model = NativeInteroceptiveModel::default();
    let snapshot = InteroceptiveSnapshot::capture_with_drive(
        &model,
        InteroceptiveDrive::ZERO,
        AllostaticConfig::default(),
    );
    let mut value = serde_json::to_value(snapshot).expect("serialize snapshot value");

    value["state"]["channels"][0]["value"] = serde_json::json!(1.25);

    let decoded = serde_json::from_value::<InteroceptiveSnapshot>(value);
    assert!(decoded.is_err());
    let message = decoded.expect_err("domain-mismatched snapshot must fail").to_string();
    assert!(message.contains("outside model domain"));
}
