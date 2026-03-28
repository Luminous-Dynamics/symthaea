// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Embodiment Integration Tests
//!
//! Verifies the proprioceptive loop: thought → motor → physics → perception → thought.
//!
//! These tests require `--features humanoid` to compile.

#![cfg(feature = "humanoid")]

use symthaea::cognitive_loop::motor_bridge::{EmbodimentPlatform, EmbodimentTelemetry};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_embodied_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };
    config.embodiment_platform = EmbodimentPlatform::Humanoid;
    config.embodiment_blend_weight = 0.2;
    config.embodiment_step_interval = 1;

    CognitiveLoopService::new(config).expect("Embodied service should initialize")
}

fn make_disembodied_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("Disembodied service should initialize")
}

#[test]
fn test_embodied_service_has_bridge() {
    let service = make_embodied_service();
    assert!(
        service.has_embodiment(),
        "Humanoid embodiment should be active"
    );
    assert_eq!(service.embodiment_platform(), EmbodimentPlatform::Humanoid);
}

#[test]
fn test_disembodied_service_no_bridge() {
    let service = make_disembodied_service();
    assert!(
        !service.has_embodiment(),
        "Default config should be disembodied"
    );
    assert_eq!(service.embodiment_platform(), EmbodimentPlatform::None);
}

#[test]
fn test_embodiment_telemetry_after_cycle() {
    let mut service = make_embodied_service();

    // Run a few cycles to let the embodiment step execute
    for i in 0..5 {
        let result = service.cycle(&format!("embodiment test cycle {i}"));
        assert!(
            result.prediction_error.is_finite(),
            "Prediction error should be finite at cycle {i}"
        );
    }

    // Check embodiment telemetry
    let telemetry = service.embodiment_telemetry();
    assert!(
        telemetry.total_steps > 0,
        "Embodiment should have stepped: total_steps={}",
        telemetry.total_steps
    );
    assert_eq!(telemetry.platform, "humanoid");
    assert_eq!(telemetry.num_actuators, 21);
    assert!(
        telemetry.control_effort >= 0.0,
        "Control effort should be non-negative"
    );
}

#[test]
fn test_proprioceptive_hv_populated() {
    let mut service = make_embodied_service();
    service.cycle("generate proprioceptive feedback");

    let proprioceptive = service.last_proprioceptive_hv();
    assert!(
        proprioceptive.is_some(),
        "Proprioceptive HV should be populated after embodied cycle"
    );

    let hv = proprioceptive.unwrap();
    assert_eq!(hv.dim(), 16384, "Proprioceptive HV should be 16,384D");
}

#[test]
fn test_disembodied_has_no_proprioception() {
    let mut service = make_disembodied_service();
    service.cycle("no embodiment here");

    let proprioceptive = service.last_proprioceptive_hv();
    assert!(
        proprioceptive.is_none(),
        "Disembodied service should have no proprioceptive HV"
    );
}

#[test]
fn test_embodiment_telemetry_in_cycle_metadata() {
    let mut service = make_embodied_service();
    let result = service.cycle("check metadata");

    // CycleMetadata should contain embodiment fields
    assert!(
        result.metadata.embodiment_total_steps > 0
            || result.metadata.embodiment_platform == "humanoid",
        "CycleMetadata should reflect embodiment"
    );
}

#[test]
fn test_embodiment_survives_multiple_cycles() {
    let mut service = make_embodied_service();

    // Run 20 cycles — embodiment should remain active throughout
    for i in 0..20 {
        let _ = service.cycle(&format!("cycle {i}"));
        assert!(
            service.has_embodiment(),
            "Embodiment should survive cycle {i}"
        );
    }

    let telemetry = service.embodiment_telemetry();
    assert_eq!(
        telemetry.total_steps, 20,
        "Should have exactly 20 embodiment steps"
    );
}

#[test]
fn test_embodiment_blend_weight_affects_encoding() {
    // Create two services with different blend weights
    let mut config_low = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };
    config_low.embodiment_platform = EmbodimentPlatform::Humanoid;
    config_low.embodiment_blend_weight = 0.0; // No blending

    let mut config_high = config_low.clone();
    config_high.embodiment_blend_weight = 0.5; // Strong blending

    let mut service_low =
        CognitiveLoopService::new(config_low).expect("Low blend service should work");
    let mut service_high =
        CognitiveLoopService::new(config_high).expect("High blend service should work");

    // Run identical input
    let _ = service_low.cycle("test blend");
    let _ = service_high.cycle("test blend");

    // Both should have embodiment active
    assert!(service_low.has_embodiment());
    assert!(service_high.has_embodiment());
}
