use symthaea_interoception::{
    AllostaticConfig, DrivePhase, ExclusionCriterion, ExpectedRelation, ExperimentArmSpec,
    ExperimentPreregistration, HypothesisSpec, InteroceptiveDrive, InteroceptiveDynamicsConfig,
    InteroceptiveIntervention, NativeInteroceptiveState, OutcomeRef, ProtocolForecastSpec,
    RegisteredMeasure, RegisteredMetricSpec, ScheduledIntervention, ViabilityChannel,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION, INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
    PREREGISTRATION_SCHEMA_VERSION,
};

fn preregistration() -> ExperimentPreregistration {
    let control = ExperimentArmSpec {
        arm_id: "control".into(),
        blind_code: "arm-kappa".into(),
        initial_state: NativeInteroceptiveState::default(),
        dynamics_config: InteroceptiveDynamicsConfig::default(),
        phases: vec![DrivePhase {
            steps: 24,
            drive: InteroceptiveDrive::ZERO,
        }],
        interventions: vec![],
    };
    let load = ExperimentArmSpec {
        arm_id: "load".into(),
        blind_code: "arm-sigma".into(),
        initial_state: NativeInteroceptiveState::default(),
        dynamics_config: InteroceptiveDynamicsConfig::default(),
        phases: vec![DrivePhase {
            steps: 24,
            drive: InteroceptiveDrive::ZERO
                .with_rate(ViabilityChannel::ComputeReserve, -0.03),
        }],
        interventions: vec![],
    };

    ExperimentPreregistration {
        schema_version: PREREGISTRATION_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        snapshot_schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
        protocol_id: "native-regulation-ordering-v1".into(),
        analysis_version: "analysis-v1".into(),
        blind_arm_identity_during_primary_analysis: true,
        arms: vec![control, load],
        metrics: vec![RegisteredMetricSpec {
            metric_id: "terminal_weighted_deviation".into(),
            measure: RegisteredMeasure::TerminalHomeostaticWeightedDeviation,
        }],
        hypotheses: vec![HypothesisSpec {
            hypothesis_id: "h1".into(),
            primary: true,
            left: OutcomeRef {
                arm_id: "load".into(),
                metric_id: "terminal_weighted_deviation".into(),
            },
            relation: ExpectedRelation::GreaterThan,
            right: OutcomeRef {
                arm_id: "control".into(),
                metric_id: "terminal_weighted_deviation".into(),
            },
        }],
        exclusions: vec![ExclusionCriterion {
            criterion_id: "mechanical-integrity".into(),
            description: "Exclude a run only if a preregistered mechanical invariant fails.".into(),
        }],
    }
}

fn incompatible_future_debt_metric(metric_id: &str) -> RegisteredMetricSpec {
    RegisteredMetricSpec {
        metric_id: metric_id.into(),
        measure: RegisteredMeasure::TerminalForecastDiscountedDebt {
            forecast: ProtocolForecastSpec::DynamicsAwareConstantDrive {
                config: AllostaticConfig {
                    horizon_steps: 8,
                    dt: 0.5,
                    discount: 0.95,
                },
                drive: InteroceptiveDrive::ZERO,
            },
        },
    }
}

#[test]
fn valid_preregistration_round_trips_and_has_stable_digest() {
    let protocol = preregistration();
    protocol.validate().expect("valid preregistration");
    let digest = protocol.sha256().expect("protocol digest");
    assert_eq!(digest.len(), 64);

    let encoded = protocol.canonical_json().expect("canonical protocol json");
    let decoded: ExperimentPreregistration =
        serde_json::from_slice(&encoded).expect("deserialize protocol");
    assert_eq!(decoded, protocol);
    assert_eq!(decoded.sha256().expect("decoded digest"), digest);
}

#[test]
fn protocol_digest_changes_when_the_prospective_plan_changes() {
    let left = preregistration();
    let mut right = left.clone();
    right.arms[1].phases[0].drive =
        InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, -0.04);

    assert_ne!(
        left.sha256().expect("left digest"),
        right.sha256().expect("right digest")
    );
}

#[test]
fn preregistration_rejects_unknown_outcome_references() {
    let mut protocol = preregistration();
    protocol.hypotheses[0].left.arm_id = "missing-arm".into();
    protocol.hypotheses[0].right.metric_id = "missing-metric".into();

    let errors = protocol.validate().expect_err("unknown references must fail");
    assert!(errors.iter().any(|error| error.contains("unknown arm")));
    assert!(errors.iter().any(|error| error.contains("unknown metric")));
}

#[test]
fn preregistration_rejects_out_of_range_interventions() {
    let mut protocol = preregistration();
    protocol.arms[0].interventions.push(ScheduledIntervention {
        before_step: 24,
        intervention: InteroceptiveIntervention::set(ViabilityChannel::Integrity, 0.8),
    });

    assert!(protocol.validate().is_err());
}

#[test]
fn dynamics_aware_metric_must_match_every_arm_timestep() {
    let mut protocol = preregistration();
    protocol.metrics[0] = incompatible_future_debt_metric("future_debt");
    protocol.hypotheses[0].left.metric_id = "future_debt".into();
    protocol.hypotheses[0].right.metric_id = "future_debt".into();

    let errors = protocol
        .validate()
        .expect_err("incompatible forecast timestep must fail");
    assert!(errors.iter().any(|error| error.contains("dt incompatible")));
}

#[test]
fn unreferenced_dynamics_aware_metric_must_still_match_every_arm_timestep() {
    let mut protocol = preregistration();
    protocol
        .metrics
        .push(incompatible_future_debt_metric("unreferenced_future_debt"));

    let errors = protocol
        .validate()
        .expect_err("every exported metric must be executable for every arm");
    assert!(errors.iter().any(|error| {
        error.contains("unreferenced_future_debt")
            && error.contains("dt incompatible")
            && error.contains("control")
    }));
}
