// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: IMU fusion flows through the full cognitive loop.
//!
//! The `sensor_fusion::RoleFillerImuFusion` module is unit-tested in its
//! own file. This file exercises the end-to-end path: inject an IMU
//! reading, run a cycle, verify the thought_vector shifts vs. baseline.
//!
//! Pattern copied from `cognitive_loop::tests::sensor_blending` — same
//! approach as the radio/STT sensor-blend integration test.
//!
//! Run: cargo test --features sensor-imu --test sensor_fusion_imu

#![cfg(feature = "sensor-imu")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::perception::sensor_fusion::{ImuReading, RoleFillerImuFusion};

fn reading(accel: [f32; 3], gyro: [f32; 3]) -> ImuReading {
    ImuReading {
        accel,
        gyro,
        timestamp_us: 0,
    }
}

#[test]
fn imu_blend_does_not_panic() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.install_imu_fusion(Box::new(RoleFillerImuFusion::new()));
    service.inject_imu_reading(reading([1.2, 0.0, 9.8], [0.0, 0.1, 0.0]));
    let _ = service.cycle("walking forward");
}

#[test]
fn imu_blend_keeps_downstream_finite() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.install_imu_fusion(Box::new(RoleFillerImuFusion::new()));
    service.inject_imu_reading(reading([2.0, -1.0, 9.81], [0.1, 0.05, -0.02]));

    let result = service.cycle("walking forward");

    assert!(
        result.thought_vector.iter().all(|v| v.is_finite()),
        "IMU blend must not introduce NaN/Inf downstream"
    );
    assert!(
        result.thought_vector.iter().any(|v| *v != 0.0),
        "thought_vector must be non-trivial"
    );
}

#[test]
fn imu_blend_shifts_thought_vector_from_baseline() {
    // Baseline: no IMU fusion installed.
    let input = "sensing the world";
    let mut baseline = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let baseline_result = baseline.cycle(input);

    // Same config + input, but with IMU fusion and a non-zero reading.
    let mut with_imu = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    with_imu.install_imu_fusion(Box::new(RoleFillerImuFusion::new()));
    with_imu.inject_imu_reading(reading([3.5, 1.2, 9.8], [0.3, -0.2, 0.1]));
    let blended_result = with_imu.cycle(input);

    let delta: f32 = baseline_result
        .thought_vector
        .iter()
        .zip(blended_result.thought_vector.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        delta > 0.0,
        "IMU blend should shift thought_vector from baseline, got L1 delta {delta}"
    );
}

#[test]
fn no_fusion_installed_behaves_like_baseline() {
    // Injecting a reading without a fusion module should be a no-op.
    let input = "no fusion module";

    let mut a = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result_a = a.cycle(input);

    let mut b = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // fusion NOT installed — injection alone should not shift the output.
    b.inject_imu_reading(reading([1.0, 2.0, 9.8], [0.5, 0.5, 0.5]));
    let result_b = b.cycle(input);

    assert_eq!(
        result_a.thought_vector, result_b.thought_vector,
        "injected reading without a fusion module must be ignored"
    );
}

#[test]
fn clear_imu_fusion_restores_baseline() {
    // With fusion installed and a reading injected, output differs from
    // baseline. After clear_imu_fusion(), the cycle should match baseline.
    let input = "install-then-clear";
    let mut baseline = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let baseline_result = baseline.cycle(input);

    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    svc.install_imu_fusion(Box::new(RoleFillerImuFusion::new()));
    svc.inject_imu_reading(reading([5.0, 0.0, 9.8], [0.0, 0.5, 0.0]));
    svc.clear_imu_fusion();

    let cleared_result = svc.cycle(input);
    assert_eq!(
        baseline_result.thought_vector, cleared_result.thought_vector,
        "after clear_imu_fusion, cycle output must match baseline"
    );
}
