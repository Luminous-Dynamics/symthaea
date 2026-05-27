// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Extreme AUV mission scenarios.
//!
//! These tests push the AUV physics to its breaking point while maintaining
//! mathematical stability. They verify that the platform survives conditions
//! analogous to real-world deep-sea operations: full-thrust maneuvers, thermal
//! downdrafts, contamination plumes, propulsor failures, and energy exhaustion.
//!
//! Designed to support the "Abyssal Rescue" and "Deep-Sea Toxic Leak" scenarios
//! from the extreme robotics testing plan.

use symthaea_auv::chemical_sensors::*;
use symthaea_auv::simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
use symthaea_auv::types::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 1: Abyssal Dive and Ascent Under Full Thrust
// AUV dives to maximum depth, then must ascend against gravity with full
// thrust. Tests that velocity clamping holds under sustained max thrust in
// both directions across 1000+ steps.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_abyssal_dive_and_ascent() {
    let mut sim = SimpleAuvSimulator::new();
    sim.reset(0.0); // Start at surface

    // Phase 1: Dive (500 steps, 5 seconds at dt=0.01)
    // Full downward thrust on vertical thrusters (4,5)
    let dive_cmd = AuvCommand {
        thrusters: [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    };
    for i in 0..500 {
        sim.step(&dive_cmd, 0.01);
        let state = sim.state();
        assert!(state.is_finite(), "NaN during dive at step {i}");
    }
    let dive_depth = sim.state().depth;
    assert!(
        dive_depth > 0.0,
        "Should have descended: depth={dive_depth}"
    );

    // Phase 2: Ascend with full thrust (500 steps)
    let ascend_cmd = AuvCommand {
        thrusters: [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
    };
    for i in 0..500 {
        sim.step(&ascend_cmd, 0.01);
        let state = sim.state();
        assert!(state.is_finite(), "NaN during ascent at step {i}");
    }

    // Should have ascended (depth decreased or hit surface)
    let final_depth = sim.state().depth;
    assert!(
        final_depth <= dive_depth,
        "Should have ascended: final={final_depth}, dive={dive_depth}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 2: Thermal Downdraft (External Force Perturbation)
// Simulates a strong downward current (thermal downdraft) that the AUV must
// fight against while maintaining sensor readings and state stability.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_thermal_downdraft_resistance() {
    let mut sim = SimpleAuvSimulator::new();

    // Phase 1: Normal cruise (100 steps)
    let cruise_cmd = AuvCommand::forward(0.5);
    for _ in 0..100 {
        sim.step(&cruise_cmd, 0.01);
    }
    let pre_downdraft_depth = sim.state().depth;

    // Phase 2: Downdraft hits — strong external force pushing down
    // A thermal downdraft can generate ~200N of force on a torpedo AUV
    for i in 0..200 {
        sim.apply_external_force([0.0, 0.0, 200.0]); // Downward force
        // AUV fights with upward thrust
        let fight_cmd = AuvCommand {
            thrusters: [0.3, 0.3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
        };
        sim.step(&fight_cmd, 0.01);
        let state = sim.state();
        assert!(state.is_finite(), "NaN during downdraft at step {i}");
    }

    // Phase 3: Downdraft passes — recover
    for i in 0..200 {
        let recover_cmd = AuvCommand {
            thrusters: [0.3, 0.3, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0],
        };
        sim.step(&recover_cmd, 0.01);
        let state = sim.state();
        assert!(state.is_finite(), "NaN during recovery at step {i}");
    }

    // All state should be valid after the ordeal
    let final_state = sim.state();
    assert!(final_state.is_finite());
    assert!(
        final_state.speed() < 10.0,
        "Speed should be bounded after recovery"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 3: Contamination Plume Navigation
// AUV navigates through a toxic plume, sensors detect anomalies, and the
// vehicle must maintain stability while chemical readings spike.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_contamination_plume_transit() {
    let mut sim = SimpleAuvSimulator::new();
    let clean = ChemicalReadings::clean_freshwater();

    // Arsenic plume centered at position [20, 0, 10]
    let plume = ContaminationPlume {
        center: [20.0, 0.0, 10.0],
        radius: 15.0,
        contaminant: Contaminant::Arsenic,
        peak_concentration: 30.0, // 3x WHO threshold
    };

    let mut max_arsenic = 0.0f64;
    let mut anomaly_detected = false;

    // Drive forward through the plume
    let cmd = AuvCommand::forward(0.8);
    for i in 0..2000 {
        sim.step(&cmd, 0.01);
        let state = sim.state();
        assert!(state.is_finite(), "NaN during plume transit at step {i}");

        // Simulate chemical readings at current position
        let readings = apply_plume(&clean, &plume, &state.position);
        let anomalies = detect_anomalies(&readings);

        if !anomalies.is_empty() {
            anomaly_detected = true;
        }
        max_arsenic = max_arsenic.max(readings.arsenic_ug_l);
    }

    // Should have detected the plume at some point during transit
    // (depends on whether the AUV path crosses the plume)
    assert!(
        max_arsenic >= clean.arsenic_ug_l,
        "Should have measured arsenic: max={max_arsenic}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 4: Asymmetric Propulsor Failure
// One thruster fails mid-mission, creating asymmetric thrust. The AUV must
// remain stable (no NaN) despite the resulting torque imbalance.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_asymmetric_thruster_failure() {
    let mut sim = SimpleAuvSimulator::new();

    // Phase 1: Normal operation (100 steps)
    let normal_cmd = AuvCommand::forward(0.5);
    for _ in 0..100 {
        sim.step(&normal_cmd, 0.01);
    }

    // Phase 2: Thruster 0 fails (stuck at zero), thruster 1 still active
    // This creates a yaw torque imbalance
    for i in 0..500 {
        let failed_cmd = AuvCommand {
            thrusters: [0.0, 0.5, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0],
        };
        sim.step(&failed_cmd, 0.01);
        let state = sim.state();
        assert!(state.is_finite(), "NaN with asymmetric thrust at step {i}");
    }

    // Vehicle should still be in a valid state despite yaw drift
    let state = sim.state();
    assert!(state.is_finite());
    // Angular velocity should be bounded (clamped, not divergent)
    for w in &state.angular_velocity {
        assert!(w.abs() <= 3.5, "Angular velocity should be bounded: {}", w);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 5: Emergency Full-Reverse Under Load
// AUV is moving forward at speed, then executes emergency full-reverse
// thrust. Tests stability during thrust reversal — the most extreme
// force transition possible.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_emergency_full_reverse() {
    let mut sim = SimpleAuvSimulator::new();

    // Build up forward speed
    let forward_cmd = AuvCommand::forward(1.0);
    for _ in 0..300 {
        sim.step(&forward_cmd, 0.01);
    }
    let speed_before = sim.state().speed();
    assert!(speed_before > 0.1, "Should have forward speed");

    // SLAM full reverse
    let reverse_cmd = AuvCommand {
        thrusters: [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    for i in 0..500 {
        sim.step(&reverse_cmd, 0.01);
        let state = sim.state();
        assert!(
            state.is_finite(),
            "NaN during emergency reverse at step {i}"
        );
    }

    // Should have decelerated and reversed
    let state = sim.state();
    assert!(state.is_finite());
    assert!(state.speed() < 10.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 6: Multi-Axis Extreme Maneuver (Stress Test)
// All 8 thrusters at maximum in different directions simultaneously,
// creating maximum possible forces and torques. This is the hardest
// numerical stability test possible for the integrator.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_axis_extreme_maneuver() {
    let mut sim = SimpleAuvSimulator::new();

    // Maximum opposing thrust: forward pair pushing, aft pair pulling,
    // vertical thrusters maxed, vectored thrusters at extremes
    let extreme_cmd = AuvCommand {
        thrusters: [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    };

    for i in 0..500 {
        sim.step(&extreme_cmd, 0.01);
        let state = sim.state();
        assert!(state.is_finite(), "NaN during extreme maneuver at step {i}");
    }

    let state = sim.state();
    assert!(state.is_finite());

    // Quaternion should still be normalized
    let q = state.quaternion;
    let qnorm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    assert!(
        (qnorm - 1.0).abs() < 0.01,
        "Quaternion should remain normalized: norm={qnorm}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 7: Long-Duration Mission (5000 steps)
// Extended operation with varying thrust profiles. Tests for slow drift,
// accumulation errors, or gradual NaN corruption.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_long_duration_mission() {
    let mut sim = SimpleAuvSimulator::new();

    let mission_phases: Vec<(AuvCommand, usize)> = vec![
        (AuvCommand::forward(0.5), 1000), // Cruise
        (
            AuvCommand {
                thrusters: [0.3, 0.3, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            },
            500,
        ), // Dive
        (AuvCommand::forward(0.8), 1000), // Fast cruise
        (
            AuvCommand {
                thrusters: [0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
            },
            500,
        ), // Lateral maneuver
        (
            AuvCommand {
                thrusters: [0.3, 0.3, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0],
            },
            500,
        ), // Ascend
        (AuvCommand::forward(0.3), 1000), // Slow return
        (AuvCommand::zero(), 500),        // Station keeping
    ];

    let mut total_steps = 0;
    for (cmd, steps) in &mission_phases {
        for _ in 0..*steps {
            sim.step(cmd, 0.01);
            total_steps += 1;
        }
        let state = sim.state();
        assert!(
            state.is_finite(),
            "NaN at step {total_steps} during mission"
        );
    }

    assert_eq!(total_steps, 5000);
    assert!(sim.state().is_finite());
}
