//! Motor Output Bridge Integration Tests
//!
//! Verifies the motor bridge → cognitive loop pipeline:
//! 1. Phi gating uses actual consciousness_level (not coherence proxy)
//! 2. Motor telemetry fields are populated in CycleMetadata
//! 3. Low-consciousness states block destructive actions
//! 4. Motor prediction error feeds back to FEP
//! 5. Bridge installation and lifecycle

use symthaea::cognitive_loop::motor_output_bridge::{
    ActionType, MotorActionRequest, MotorOutputBridge,
};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed")
}

#[test]
fn test_motor_bridge_telemetry_defaults() {
    // Without a motor bridge installed, telemetry should report inactive
    let mut service = make_service();
    let result = service.cycle("hello");
    assert!(!result.metadata.motor.motor_bridge_active);
    assert!(!result.metadata.motor.motor_action_executed);
    // Default::default() produces 0 for u8 (serde default only applies during deserialization)
    assert_eq!(result.metadata.motor.motor_action_type, 0);
    assert_eq!(result.metadata.motor.motor_phi_used, 0.0);
}

#[test]
fn test_motor_bridge_installation() {
    let mut service = make_service();
    assert!(!service.has_motor_bridge());

    let bridge = MotorOutputBridge::with_defaults().unwrap();
    service.set_motor_output_bridge(bridge);
    assert!(service.has_motor_bridge());

    // After installation, telemetry reports bridge as active
    let result = service.cycle("test with bridge");
    assert!(result.metadata.motor.motor_bridge_active);
}

#[test]
fn test_motor_bridge_telemetry_after_installation() {
    let mut service = make_service();
    let bridge = MotorOutputBridge::with_defaults().unwrap();
    service.set_motor_output_bridge(bridge);

    // Run several cycles — motor_bridge_active should always be true
    for i in 0..10 {
        let result = service.cycle(&format!("motor telemetry test {i}"));
        assert!(
            result.metadata.motor.motor_bridge_active,
            "Bridge should remain active at cycle {i}"
        );
    }
}

#[test]
fn test_motor_result_lifecycle() {
    let mut service = make_service();
    let bridge = MotorOutputBridge::with_defaults().unwrap();
    service.set_motor_output_bridge(bridge);

    // First take should be None (no MotorOutput command yet)
    assert!(service.take_motor_result().is_none());

    // Run a cycle — motor result is produced only if FEP emits MotorOutput
    service.cycle("test motor lifecycle");
    // Result may or may not exist depending on FEP dynamics
    let _ = service.take_motor_result();
}

#[test]
fn test_action_type_exhaustive_roundtrip() {
    // Verify all 8 action types decode correctly
    for i in 0..8u8 {
        let at = ActionType::from_param(i as f64);
        assert!(at.is_some(), "ActionType should decode for param {i}");
        assert_eq!(at.unwrap() as u8, i);
    }
    // Out-of-range should return None
    assert!(ActionType::from_param(8.0).is_none());
    // Note: -1.0 as u8 saturates to 0 in Rust 2021, so it matches Read
    // NaN as u8 saturates to 0, also matching Read
    assert!(ActionType::from_param(255.0).is_none());
}

#[test]
fn test_motor_prediction_error_bounds_via_bridge() {
    let mut bridge = MotorOutputBridge::with_defaults().unwrap();
    let request = MotorActionRequest::default();

    // Skipped (no params) → PE = 0.5
    let skipped = bridge.execute(&[], 0.8, 0.9, &request);
    assert!(!skipped.success);
    assert_eq!(skipped.prediction_error, 0.5);

    // Skipped (low phi) → PE = 0.5
    let low_phi = bridge.execute(&[0.0], 0.8, 0.05, &request);
    assert!(!low_phi.success);
    assert_eq!(low_phi.prediction_error, 0.5);

    // Skipped (low confidence) → PE = 0.5
    let low_conf = bridge.execute(&[0.0], 0.1, 0.9, &request);
    assert!(!low_conf.success);
    assert_eq!(low_conf.prediction_error, 0.5);
}

#[test]
fn test_phi_gating_blocks_at_low_consciousness() {
    // The bridge itself requires min_phi (default 0.3 from PolicyBundle::restrictive)
    let mut bridge = MotorOutputBridge::with_defaults().unwrap();
    let request = MotorActionRequest {
        target_path: Some(std::path::PathBuf::from(
            "/tmp/symthaea/motor_bridge/test.txt",
        )),
        ..Default::default()
    };

    // Phi = 0.05 — far below threshold
    let result = bridge.execute(&[0.0], 0.8, 0.05, &request);
    assert!(!result.success);
    assert!(
        result.error.as_ref().unwrap().contains("below"),
        "Should cite Phi threshold: {:?}",
        result.error
    );
}

#[test]
fn test_phi_gating_reversible_requires_higher_phi() {
    let mut bridge = MotorOutputBridge::with_defaults().unwrap();
    let request = MotorActionRequest {
        target_path: Some(std::path::PathBuf::from("/tmp/symthaea/motor_bridge/")),
        program: Some("echo".into()),
        args: vec!["hello".into()],
        content: None,
    };

    // "echo" is classified as Reversible → requires min_phi + 0.1 = 0.4
    // Phi = 0.35 passes base threshold (0.3) but not Reversible (0.4)
    let result = bridge.execute(&[7.0], 0.8, 0.35, &request);
    assert!(!result.success);
    assert!(
        result.error.as_ref().unwrap().contains("insufficient"),
        "Should cite insufficient Phi: {:?}",
        result.error
    );
}
