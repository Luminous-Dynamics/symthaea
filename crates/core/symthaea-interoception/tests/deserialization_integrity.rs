use serde_json::json;
use symthaea_interoception::{
    AllostaticConfig, InteroceptiveDynamicsConfig, InteroceptiveSnapshot,
    NativeInteroceptiveModel, NativeInteroceptiveState, ViabilityVariable,
};

#[test]
fn viability_variable_try_new_rejects_invalid_geometry() {
    assert!(ViabilityVariable::try_new(0.5, 0.8, 0.7, 0.2, 1.0, 1.0, 1.0).is_err());
    assert!(ViabilityVariable::try_new(0.5, 0.4, 0.6, 0.2, 1.0, -1.0, 1.0).is_err());
    assert!(ViabilityVariable::try_new(0.5, 0.4, 0.6, 0.2, 1.0, 1.0, -1.0).is_err());
}

#[test]
fn deserialization_cannot_bypass_viability_invariants() {
    let state = NativeInteroceptiveState::default();
    let mut encoded = serde_json::to_value(&state).expect("serialize state");

    encoded["channels"][0]["preferred_low"] = json!(0.9);
    encoded["channels"][0]["preferred_high"] = json!(0.8);
    assert!(serde_json::from_value::<NativeInteroceptiveState>(encoded).is_err());

    let state = NativeInteroceptiveState::default();
    let mut encoded = serde_json::to_value(&state).expect("serialize state");
    encoded["channels"][0]["precision"] = json!(-0.5);
    assert!(serde_json::from_value::<NativeInteroceptiveState>(encoded).is_err());
}

#[test]
fn deserialization_rejects_invalid_dynamics_and_forecast_configs() {
    let invalid_dynamics = json!({
        "step_dt": 2.0,
        "recovery_rate": 0.6,
        "min_value": 0.0,
        "max_value": 1.0
    });
    assert!(serde_json::from_value::<InteroceptiveDynamicsConfig>(invalid_dynamics).is_err());

    let invalid_forecast = json!({
        "horizon_steps": 16,
        "dt": 1.0,
        "discount": 1.1
    });
    assert!(serde_json::from_value::<AllostaticConfig>(invalid_forecast).is_err());
}

#[test]
fn snapshot_deserialization_rejects_forged_homeostatic_report() {
    let model = NativeInteroceptiveModel::default();
    let snapshot = InteroceptiveSnapshot::capture_kinematic(&model, AllostaticConfig::default());
    let mut encoded = serde_json::to_value(&snapshot).expect("serialize snapshot");

    encoded["homeostasis"]["weighted_deviation"] = json!(0.25);
    assert!(serde_json::from_value::<InteroceptiveSnapshot>(encoded).is_err());
}

#[test]
fn snapshot_deserialization_rejects_forged_forecast_report() {
    let model = NativeInteroceptiveModel::default();
    let snapshot = InteroceptiveSnapshot::capture_kinematic(&model, AllostaticConfig::default());
    let mut encoded = serde_json::to_value(&snapshot).expect("serialize snapshot");

    encoded["forecast"]["Kinematic"]["report"]["discounted_debt"] = json!(0.25);
    assert!(serde_json::from_value::<InteroceptiveSnapshot>(encoded).is_err());
}

#[test]
fn valid_snapshot_still_round_trips_after_semantic_validation() {
    let model = NativeInteroceptiveModel::default();
    let snapshot = InteroceptiveSnapshot::capture_kinematic(&model, AllostaticConfig::default());
    let encoded = serde_json::to_vec(&snapshot).expect("serialize snapshot");
    let decoded: InteroceptiveSnapshot =
        serde_json::from_slice(&encoded).expect("deserialize validated snapshot");

    assert_eq!(decoded, snapshot);
    assert!(decoded.validate().is_ok());
}
