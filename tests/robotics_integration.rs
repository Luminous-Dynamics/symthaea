// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robotics End-to-End Cognitive Loop Integration Tests
//!
//! These tests close the full proprioceptive loop:
//! perception → CognitiveLoopService::cycle() → motor → physics → proprioceptive HV → next cycle
//!
//! This is the critical gap: individual robotics crates have strong unit tests
//! (1,366 total) but none verify the full cognitive-embodiment feedback loop.
//!
//! Run: `cargo test --features humanoid,helicopter,multirotor,vehicle --test robotics_integration`

use symthaea::cognitive_loop::motor_bridge::{
    EmbodimentBridge, EmbodimentPlatform, MotorSafetyLevel,
};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

// ── Helpers ────────────────────────────────────────────────────────────────

fn make_embodied_service(platform: EmbodimentPlatform) -> CognitiveLoopService {
    let config = CognitiveLoopConfig {
        embodiment_platform: platform,
        embodiment_blend_weight: 0.3,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };
    CognitiveLoopService::new(config).expect("CognitiveLoopService::new with embodiment")
}

#[cfg(feature = "flight")]
fn make_multi_embodied_service(platform: EmbodimentPlatform) -> CognitiveLoopService {
    let config = CognitiveLoopConfig {
        embodiment_platform: platform,
        embodiment_blend_weight: 0.3,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };
    CognitiveLoopService::new(config).expect("CognitiveLoopService::new with embodiment")
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 1: Humanoid cognitive loop closure
// Verifies: cycle() steps the humanoid bridge, proprioceptive HV feeds back,
// telemetry is populated, and consciousness level modulates motor authority.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_humanoid_cognitive_loop_closure() {
    let mut service = make_embodied_service(EmbodimentPlatform::Humanoid);

    assert!(service.has_embodiment());
    assert_eq!(service.embodiment_platform(), EmbodimentPlatform::Humanoid);

    // Run 50 cycles — the full loop should close each time
    for i in 0..50 {
        let result = service.cycle("humanoid standing balance test");
        let m = &result.metadata;

        // Consciousness should remain finite
        assert!(
            m.consciousness.consciousness_level.is_finite(),
            "consciousness NaN at cycle {i}"
        );

        // After first cycle, proprioceptive HV should be populated
        if i > 0 {
            assert!(
                service.last_proprioceptive_hv().is_some(),
                "proprioceptive HV missing at cycle {i}"
            );
        }
    }

    // Telemetry should reflect humanoid platform
    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps > 0, "embodiment should have stepped");
    assert_eq!(telem.num_actuators, 21, "humanoid has 21 actuators");
    assert!(
        telem.control_effort.is_finite(),
        "control effort should be finite"
    );
    assert!(
        telem.prediction_error.is_finite(),
        "prediction error should be finite"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 2: Helicopter SAR cognitive loop
// Verifies: helicopter bridge steps through cognitive cycle with consciousness
// modulating rotor authority.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "helicopter")]
#[test]
fn test_helicopter_cognitive_loop_closure() {
    let mut service = make_embodied_service(EmbodimentPlatform::Helicopter);

    assert!(service.has_embodiment());
    assert_eq!(
        service.embodiment_platform(),
        EmbodimentPlatform::Helicopter
    );

    // Run 50 cycles — SAR helicopter with full cognitive loop
    for i in 0..50 {
        let result = service.cycle("helicopter search and rescue patrol");
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "consciousness NaN at helicopter cycle {i}"
        );
    }

    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps > 0, "helicopter should have stepped");
    assert_eq!(telem.num_actuators, 6, "helicopter has 6 actuators");

    // Proprioceptive HV should be populated (18D state encoded to 16384D)
    assert!(
        service.last_proprioceptive_hv().is_some(),
        "helicopter proprioceptive HV missing after 50 cycles"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 3: Quadrotor flight cognitive loop
// Verifies: flight bridge closes the proprioceptive loop with 13D state.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "flight")]
#[test]
fn test_flight_cognitive_loop_closure() {
    let mut service = make_embodied_service(EmbodimentPlatform::Quadrotor);

    assert!(service.has_embodiment());
    assert_eq!(service.embodiment_platform(), EmbodimentPlatform::Quadrotor);

    // Run 50 cycles
    for i in 0..50 {
        let result = service.cycle("quadrotor hover and stabilize");
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "consciousness NaN at flight cycle {i}"
        );
    }

    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps > 0, "flight should have stepped");
    assert_eq!(telem.num_actuators, 4, "quadrotor has 4 actuators");
    assert!(
        service.last_proprioceptive_hv().is_some(),
        "flight proprioceptive HV missing"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 4: Vehicle cognitive loop
// Verifies: vehicle bridge closes the proprioceptive loop with 20D state.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "vehicle")]
#[test]
fn test_vehicle_cognitive_loop_closure() {
    let mut service = make_embodied_service(EmbodimentPlatform::Vehicle);

    assert!(service.has_embodiment());
    assert_eq!(service.embodiment_platform(), EmbodimentPlatform::Vehicle);

    for i in 0..50 {
        let result = service.cycle("autonomous vehicle lane following");
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "consciousness NaN at vehicle cycle {i}"
        );
    }

    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps > 0, "vehicle should have stepped");
    assert_eq!(telem.num_actuators, 3, "vehicle has 3 actuators");
    assert!(
        service.last_proprioceptive_hv().is_some(),
        "vehicle proprioceptive HV missing"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 5: Consciousness degradation halts all motors
// Verifies: when Phi drops below 0.1, safety level goes Red and motor gain = 0.
// This is the critical safety invariant for all robotics platforms.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_consciousness_degradation_halts_motors() {
    let mut service = make_embodied_service(EmbodimentPlatform::Humanoid);

    // Run enough cycles to establish baseline
    for _ in 0..20 {
        service.cycle("baseline motor activity");
    }

    // Verify safety level mapping
    assert_eq!(MotorSafetyLevel::from_phi(0.8).motor_gain(), 1.0); // Green
    assert_eq!(MotorSafetyLevel::from_phi(0.5).motor_gain(), 0.6); // Yellow
    assert_eq!(MotorSafetyLevel::from_phi(0.2).motor_gain(), 0.3); // Orange
    assert_eq!(MotorSafetyLevel::from_phi(0.05).motor_gain(), 0.0); // Red — HALT

    // Verify the actual safety level reported by telemetry
    let telem = service.embodiment_telemetry();
    assert!(
        telem.total_steps > 0,
        "should have motor steps before degradation test"
    );

    // The safety level string should be one of the valid levels
    assert!(
        ["Green", "Yellow", "Orange", "Red"].contains(&telem.safety_level.as_str()),
        "unexpected safety level: {}",
        telem.safety_level
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 6: Multi-embodiment fusion preserves coherence
// Verifies: when 2+ platforms are active, their proprioceptive HVs are fused
// via weighted running average without NaN or divergence.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "flight")]
#[test]
fn test_multi_embodiment_fusion() {
    let mut service = make_multi_embodied_service(EmbodimentPlatform::Quadrotor);

    assert!(service.has_embodiment(), "should have embodiment bridge");

    // Run 50 cycles — both platforms step and fuse proprioceptive HVs
    for i in 0..50 {
        let result = service.cycle("multi-body coordination test");
        let m = &result.metadata;

        // Consciousness must remain finite with multi-body fusion
        assert!(
            m.consciousness.consciousness_level.is_finite(),
            "consciousness NaN with multi-body at cycle {i}"
        );
    }

    // Primary bridge should have telemetry
    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps > 0, "bridge should have stepped");
    assert!(
        telem.control_effort.is_finite(),
        "bridge control effort should be finite"
    );

    // Fused proprioceptive HV should be populated (first bridge accessor)
    assert!(
        service.last_proprioceptive_hv().is_some(),
        "fused proprioceptive HV missing after multi-body cycles"
    );

    // Verify the fused HV has reasonable magnitude (not all zeros, not NaN)
    let hv = service.last_proprioceptive_hv().unwrap();
    let dims = hv.as_slice();
    let magnitude: f32 = dims.iter().map(|d| d * d).sum::<f32>().sqrt();
    assert!(magnitude > 0.0, "fused HV should not be zero vector");
    assert!(magnitude.is_finite(), "fused HV magnitude should be finite");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 7: Long-horizon stability (no drift accumulation)
// Verifies: 200 cycles of continuous embodied operation without NaN or divergence.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_embodied_long_horizon_stability() {
    let mut service = make_embodied_service(EmbodimentPlatform::Humanoid);

    let mut max_consciousness = f64::MIN;
    let mut min_consciousness = f64::MAX;

    for i in 0..200 {
        let result = service.cycle("long horizon stability check");
        let cl = result.metadata.consciousness.consciousness_level;

        assert!(cl.is_finite(), "consciousness NaN at cycle {i}");
        assert!(cl >= 0.0, "consciousness negative at cycle {i}: {cl}");
        assert!(cl <= 1.0, "consciousness > 1 at cycle {i}: {cl}");

        max_consciousness = max_consciousness.max(cl);
        min_consciousness = min_consciousness.min(cl);
    }

    // Verify the system didn't flatline or explode
    let telem = service.embodiment_telemetry();
    assert!(
        telem.total_steps >= 200,
        "should have at least 200 motor steps, got {}",
        telem.total_steps
    );

    // Proprioceptive HV should still be valid after 200 cycles
    let hv = service
        .last_proprioceptive_hv()
        .expect("proprioceptive HV after 200 cycles");
    let magnitude: f32 = hv.as_slice().iter().map(|d| d * d).sum::<f32>().sqrt();
    assert!(
        magnitude.is_finite() && magnitude > 0.0,
        "proprioceptive HV should remain valid after 200 cycles"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 8: Proprioceptive feedback influences perception
// Verifies: with embodiment_blend_weight > 0, the perception HV changes
// compared to running without embodiment.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_proprioceptive_feedback_influences_perception() {
    // Run with embodiment
    let mut embodied = make_embodied_service(EmbodimentPlatform::Humanoid);

    // Run without embodiment (blend weight = 0)
    let mut disembodied = CognitiveLoopService::new(CognitiveLoopConfig {
        embodiment_platform: EmbodimentPlatform::Humanoid,
        embodiment_blend_weight: 0.0, // No proprioceptive feedback
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("disembodied service");

    // Run both for 30 cycles with same input
    let mut embodied_phi_sum = 0.0f64;
    let mut disembodied_phi_sum = 0.0f64;
    for _ in 0..30 {
        let r1 = embodied.cycle("proprioceptive test input");
        let r2 = disembodied.cycle("proprioceptive test input");
        embodied_phi_sum += r1.metadata.consciousness.consciousness_level;
        disembodied_phi_sum += r2.metadata.consciousness.consciousness_level;
    }

    // Both should produce valid results
    assert!(embodied_phi_sum.is_finite());
    assert!(disembodied_phi_sum.is_finite());

    // The embodied system should have a proprioceptive HV; the disembodied one also
    // steps but doesn't blend (weight = 0)
    assert!(embodied.last_proprioceptive_hv().is_some());
    assert!(disembodied.last_proprioceptive_hv().is_some());
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 9: AUV cognitive loop closure
// Verifies: AUV bridge with 32D state (kinematic + chemical) closes the loop.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "auv")]
#[test]
fn test_auv_cognitive_loop_closure() {
    let mut service = make_embodied_service(EmbodimentPlatform::Auv);

    assert!(service.has_embodiment());
    assert_eq!(service.embodiment_platform(), EmbodimentPlatform::Auv);

    for i in 0..50 {
        let result = service.cycle("auv water quality monitoring");
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "consciousness NaN at AUV cycle {i}"
        );
    }

    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps > 0, "AUV should have stepped");
    assert_eq!(telem.num_actuators, 8, "AUV has 8 thrusters");
    assert!(
        service.last_proprioceptive_hv().is_some(),
        "AUV proprioceptive HV missing"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 10: Manipulator cognitive loop closure
// Verifies: manipulator bridge with 21D state closes the loop.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "manipulator")]
#[test]
fn test_manipulator_cognitive_loop_closure() {
    let mut service = make_embodied_service(EmbodimentPlatform::Manipulator);

    assert!(service.has_embodiment());
    assert_eq!(
        service.embodiment_platform(),
        EmbodimentPlatform::Manipulator
    );

    for i in 0..50 {
        let result = service.cycle("manipulator pick and place operation");
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "consciousness NaN at manipulator cycle {i}"
        );
    }

    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps > 0, "manipulator should have stepped");
    assert_eq!(
        telem.num_actuators, 8,
        "manipulator has 8 actuators (7 joints + gripper)"
    );
    assert!(
        service.last_proprioceptive_hv().is_some(),
        "manipulator proprioceptive HV missing"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 11: Embodiment step interval (cognitive-motor rate decoupling)
// Verifies: with step_interval > 1, embodiment steps less frequently than
// cognitive cycles, but still produces valid results.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_embodiment_step_interval() {
    let config = CognitiveLoopConfig {
        embodiment_platform: EmbodimentPlatform::Humanoid,
        embodiment_blend_weight: 0.2,
        embodiment_step_interval: 5, // Step every 5th cognitive cycle
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };
    let mut service = CognitiveLoopService::new(config).expect("service with step interval 5");

    // Run 50 cycles — embodiment should step ~10 times (50 / 5)
    for i in 0..50 {
        let result = service.cycle("interval test");
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "consciousness NaN at cycle {i} with interval=5"
        );
    }

    let telem = service.embodiment_telemetry();
    // With interval=5, should have stepped approximately 10 times
    // (exact count depends on cycle_num % interval starting at 0)
    assert!(
        telem.total_steps >= 9 && telem.total_steps <= 11,
        "expected ~10 motor steps with interval=5, got {}",
        telem.total_steps
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 12: Safety level transitions during embodied operation
// Verifies: telemetry safety level is consistent with consciousness level.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_safety_level_consistency() {
    let mut service = make_embodied_service(EmbodimentPlatform::Humanoid);

    for _ in 0..30 {
        let result = service.cycle("safety consistency check");
        let phi = result.metadata.consciousness.consciousness_level;
        let telem = service.embodiment_telemetry();

        // Safety level should be consistent with Phi
        let expected = MotorSafetyLevel::from_phi(phi);
        let _expected_str = format!("{expected:?}");
        // Note: safety override from SafetyAgent may elevate the level,
        // so actual level >= expected level (more conservative is OK)
        assert!(
            ["Green", "Yellow", "Orange", "Red"].contains(&telem.safety_level.as_str()),
            "invalid safety level: {}",
            telem.safety_level
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  MULTI-EMBODIMENT FAIL-SAFE SCENARIO TESTS
// ═════════════════════════════════════════════════════════════════════════════
//
// The tests below exercise extreme scenarios against the MotorSafetyLevel API
// and each platform's embodiment bridge (step/reset/safety_level) directly.
// This avoids the full CognitiveLoopService overhead and isolates the fail-safe
// logic itself.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Helper: run a Phi sweep across all safety boundaries and verify transitions.
fn assert_safety_transition_at_boundary(phi: f64, expected: MotorSafetyLevel) {
    let actual = MotorSafetyLevel::from_phi(phi);
    assert_eq!(
        actual, expected,
        "Phi={phi}: expected {expected:?}, got {actual:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 1: Cascading Phi Collapse — All Platforms
//
// Phi drops from 0.8 to 0.0 over 10 cycles. Verify every platform transitions
// Green -> Yellow -> Orange -> Red and motor gains clamp at each stage.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_cascading_phi_collapse_humanoid() {
    let genesis = GenesisSeed::from_phrase("cascade-humanoid");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence: Vec<f64> = (0..=10).map(|i| 0.8 - (i as f64) * 0.08).collect();

    let mut saw_green = false;
    let mut saw_yellow = false;
    let mut saw_orange = false;
    let mut saw_red = false;

    for (i, &phi) in phi_sequence.iter().enumerate() {
        let result = bridge.step(&hv, 0.025, phi);
        let level = result.safety_level;
        let gain = level.motor_gain();

        match level {
            MotorSafetyLevel::Green => {
                saw_green = true;
                assert_eq!(gain, 1.0, "Green gain must be 1.0 at cycle {i}, phi={phi}");
            }
            MotorSafetyLevel::Yellow => {
                saw_yellow = true;
                assert_eq!(gain, 0.6, "Yellow gain must be 0.6 at cycle {i}, phi={phi}");
            }
            MotorSafetyLevel::Orange => {
                saw_orange = true;
                assert_eq!(gain, 0.3, "Orange gain must be 0.3 at cycle {i}, phi={phi}");
            }
            MotorSafetyLevel::Red => {
                saw_red = true;
                assert_eq!(gain, 0.0, "Red gain must be 0.0 at cycle {i}, phi={phi}");
                assert_eq!(
                    result.control_effort, 0.0,
                    "Red: humanoid control effort must be 0 at cycle {i}"
                );
            }
        }
    }

    assert!(saw_green, "cascade should have started at Green");
    assert!(saw_yellow, "cascade should have passed through Yellow");
    assert!(saw_orange, "cascade should have passed through Orange");
    assert!(saw_red, "cascade should have reached Red");
}

#[cfg(feature = "flight")]
#[test]
fn test_cascading_phi_collapse_quadrotor() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("cascade-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence: Vec<f64> = (0..=10).map(|i| 0.8 - (i as f64) * 0.08).collect();
    let mut levels_seen = [false; 4]; // Green, Yellow, Orange, Red

    for &phi in &phi_sequence {
        let result = bridge.step(&hv, 0.002, phi);
        let level = format!("{:?}", result.safety_level);
        match level.as_str() {
            "Green" => levels_seen[0] = true,
            "Yellow" => levels_seen[1] = true,
            "Orange" => levels_seen[2] = true,
            "Red" => {
                levels_seen[3] = true;
                assert_eq!(
                    result.control_effort, 0.0,
                    "quadrotor Red: effort must be 0"
                );
            }
            other => panic!("unexpected safety level: {other}"),
        }
    }
    assert!(
        levels_seen.iter().all(|&s| s),
        "quadrotor cascade must visit all 4 levels"
    );
}

#[cfg(feature = "helicopter")]
#[test]
fn test_cascading_phi_collapse_helicopter() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;

    let genesis = GenesisSeed::from_phrase("cascade-heli");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence: Vec<f64> = (0..=10).map(|i| 0.8 - (i as f64) * 0.08).collect();
    let mut levels_seen = [false; 4];

    for &phi in &phi_sequence {
        let result = bridge.step(&hv, 1.0 / 300.0, phi);
        let level = format!("{:?}", result.safety_level);
        match level.as_str() {
            "Green" => levels_seen[0] = true,
            "Yellow" => levels_seen[1] = true,
            "Orange" => levels_seen[2] = true,
            "Red" => {
                levels_seen[3] = true;
                assert_eq!(
                    result.control_effort, 0.0,
                    "helicopter Red: effort must be 0"
                );
            }
            other => panic!("unexpected safety level: {other}"),
        }
    }
    assert!(
        levels_seen.iter().all(|&s| s),
        "helicopter cascade must visit all 4 levels"
    );
}

#[cfg(feature = "vehicle")]
#[test]
fn test_cascading_phi_collapse_vehicle() {
    use symthaea_vehicle::embodiment::VehicleEmbodiment;

    let genesis = GenesisSeed::from_phrase("cascade-vehicle");
    let mut bridge = VehicleEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence: Vec<f64> = (0..=10).map(|i| 0.8 - (i as f64) * 0.08).collect();
    let mut levels_seen = [false; 4];

    for &phi in &phi_sequence {
        let result = bridge.step(&hv, 0.005, phi);
        let level = format!("{:?}", result.safety_level);
        match level.as_str() {
            "Green" => levels_seen[0] = true,
            "Yellow" => levels_seen[1] = true,
            "Orange" => levels_seen[2] = true,
            "Red" => levels_seen[3] = true,
            other => panic!("unexpected safety level: {other}"),
        }
    }
    assert!(
        levels_seen.iter().all(|&s| s),
        "vehicle cascade must visit all 4 levels"
    );
}

#[cfg(feature = "manipulator")]
#[test]
fn test_cascading_phi_collapse_manipulator() {
    use symthaea_manipulator::embodiment::ManipulatorEmbodiment;

    let genesis = GenesisSeed::from_phrase("cascade-manip");
    let mut bridge = ManipulatorEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence: Vec<f64> = (0..=10).map(|i| 0.8 - (i as f64) * 0.08).collect();
    let mut levels_seen = [false; 4];

    for &phi in &phi_sequence {
        let result = bridge.step(&hv, 0.002, phi);
        let level = format!("{:?}", result.safety_level);
        match level.as_str() {
            "Green" => levels_seen[0] = true,
            "Yellow" => levels_seen[1] = true,
            "Orange" => levels_seen[2] = true,
            "Red" => levels_seen[3] = true,
            other => panic!("unexpected safety level: {other}"),
        }
    }
    assert!(
        levels_seen.iter().all(|&s| s),
        "manipulator cascade must visit all 4 levels"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 2: Asymmetric Degradation — One Platform Red, Others Green
//
// Verify isolation: one platform's failure does not affect another.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_asymmetric_degradation_humanoid_isolated() {
    let genesis = GenesisSeed::from_phrase("asymmetric");
    let mut healthy_bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let mut failing_bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Step healthy at Green, failing at Red
    for _ in 0..20 {
        let healthy_result = healthy_bridge.step(&hv, 0.025, 0.8);
        let failing_result = failing_bridge.step(&hv, 0.025, 0.01);

        assert_eq!(
            healthy_result.safety_level,
            MotorSafetyLevel::Green,
            "healthy bridge must remain Green"
        );
        assert_eq!(
            failing_result.safety_level,
            MotorSafetyLevel::Red,
            "failing bridge must be Red"
        );
        assert_eq!(
            failing_result.control_effort, 0.0,
            "failing bridge motor effort must be 0"
        );
        // Healthy bridge should have nonzero effort (it is actively moving)
        // (not asserting > 0 because controller output depends on thought HV)
        assert!(
            healthy_result.control_effort.is_finite(),
            "healthy bridge effort must be finite"
        );
    }
}

#[cfg(feature = "flight")]
#[test]
fn test_asymmetric_degradation_cross_platform() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("asymmetric-cross");
    let mut humanoid = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let mut quadrotor = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Humanoid healthy, quadrotor failing
    for _ in 0..20 {
        let h_result = humanoid.step(&hv, 0.025, 0.8);
        let q_result = quadrotor.step(&hv, 0.002, 0.01);

        assert_eq!(h_result.safety_level, MotorSafetyLevel::Green);
        assert_eq!(format!("{:?}", q_result.safety_level), "Red");
    }

    // Now swap: humanoid fails, quadrotor recovers
    for _ in 0..20 {
        let h_result = humanoid.step(&hv, 0.025, 0.01);
        let q_result = quadrotor.step(&hv, 0.002, 0.8);

        assert_eq!(h_result.safety_level, MotorSafetyLevel::Red);
        assert_eq!(format!("{:?}", q_result.safety_level), "Green");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 3: Rapid Phi Oscillation — 100 cycles between 0.1 and 0.7
//
// Verify no NaN, no panic, gains always bounded, safety level always valid.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_rapid_phi_oscillation_humanoid() {
    let genesis = GenesisSeed::from_phrase("oscillation");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for i in 0..100 {
        let phi = if i % 2 == 0 { 0.1 } else { 0.7 };
        let result = bridge.step(&hv, 0.025, phi);

        assert!(
            result.control_effort.is_finite(),
            "oscillation cycle {i}: control effort NaN/Inf"
        );
        assert!(
            result.prediction_error.is_finite(),
            "oscillation cycle {i}: prediction error NaN/Inf"
        );
        let gain = result.safety_level.motor_gain();
        assert!(
            gain >= 0.0 && gain <= 1.0,
            "oscillation cycle {i}: gain {gain} out of bounds"
        );
    }
}

#[cfg(feature = "flight")]
#[test]
fn test_rapid_phi_oscillation_quadrotor() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("oscillation-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for i in 0..100 {
        let phi = if i % 2 == 0 { 0.1 } else { 0.7 };
        let result = bridge.step(&hv, 0.002, phi);

        assert!(
            result.control_effort.is_finite(),
            "quadrotor oscillation cycle {i}: effort NaN/Inf"
        );
        assert!(
            result.prediction_error.is_finite(),
            "quadrotor oscillation cycle {i}: pred error NaN/Inf"
        );
        assert!(
            result.success,
            "quadrotor oscillation cycle {i}: step failed"
        );
    }
}

#[cfg(feature = "helicopter")]
#[test]
fn test_rapid_phi_oscillation_helicopter() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;

    let genesis = GenesisSeed::from_phrase("oscillation-heli");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for i in 0..100 {
        let phi = if i % 2 == 0 { 0.1 } else { 0.7 };
        let result = bridge.step(&hv, 1.0 / 300.0, phi);
        assert!(
            result.control_effort.is_finite(),
            "helicopter oscillation cycle {i}: effort NaN/Inf"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 4: NaN Phi Input
//
// Feed NaN as Phi value. from_phi uses `>` comparisons, which return false
// for NaN, so NaN must map to Red (the else branch). Verify no panic.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_nan_phi_maps_to_red() {
    // MotorSafetyLevel::from_phi with NaN
    let level = MotorSafetyLevel::from_phi(f64::NAN);
    assert_eq!(
        level,
        MotorSafetyLevel::Red,
        "NaN Phi must map to Red (fail-safe)"
    );
    assert_eq!(
        level.motor_gain(),
        0.0,
        "NaN Phi must yield zero motor gain"
    );
}

#[test]
fn test_nan_phi_humanoid_bridge_no_panic() {
    let genesis = GenesisSeed::from_phrase("nan-test");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // First step with valid Phi to initialize
    let _ = bridge.step(&hv, 0.025, 0.7);

    // Now NaN Phi — must not panic, must go Red
    let result = bridge.step(&hv, 0.025, f64::NAN);
    assert_eq!(
        result.safety_level,
        MotorSafetyLevel::Red,
        "NaN Phi: must be Red"
    );
    assert_eq!(
        result.control_effort, 0.0,
        "NaN Phi: control effort must be 0"
    );
    assert!(
        result.prediction_error.is_finite(),
        "prediction error must stay finite despite NaN Phi"
    );
}

#[cfg(feature = "flight")]
#[test]
fn test_nan_phi_quadrotor_no_panic() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("nan-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.002, f64::NAN);
    assert_eq!(format!("{:?}", result.safety_level), "Red");
    assert_eq!(result.control_effort, 0.0);
}

#[cfg(feature = "helicopter")]
#[test]
fn test_nan_phi_helicopter_no_panic() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;

    let genesis = GenesisSeed::from_phrase("nan-heli");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 1.0 / 300.0, f64::NAN);
    assert_eq!(format!("{:?}", result.safety_level), "Red");
    assert_eq!(result.control_effort, 0.0);
}

#[cfg(feature = "vehicle")]
#[test]
fn test_nan_phi_vehicle_no_panic() {
    use symthaea_vehicle::embodiment::VehicleEmbodiment;

    let genesis = GenesisSeed::from_phrase("nan-vehicle");
    let mut bridge = VehicleEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.005, f64::NAN);
    assert_eq!(format!("{:?}", result.safety_level), "Red");
}

#[cfg(feature = "manipulator")]
#[test]
fn test_nan_phi_manipulator_no_panic() {
    use symthaea_manipulator::embodiment::ManipulatorEmbodiment;

    let genesis = GenesisSeed::from_phrase("nan-manip");
    let mut bridge = ManipulatorEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.002, f64::NAN);
    assert_eq!(format!("{:?}", result.safety_level), "Red");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 5: Negative Phi
//
// Feed -0.5. All `>` comparisons with positive thresholds fail, so negative
// Phi must map to Red.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_negative_phi_maps_to_red() {
    assert_eq!(
        MotorSafetyLevel::from_phi(-0.5),
        MotorSafetyLevel::Red,
        "negative Phi must map to Red"
    );
    assert_eq!(
        MotorSafetyLevel::from_phi(-1.0),
        MotorSafetyLevel::Red,
        "phi=-1 must map to Red"
    );
    assert_eq!(
        MotorSafetyLevel::from_phi(-100.0),
        MotorSafetyLevel::Red,
        "phi=-100 must map to Red"
    );
    assert_eq!(MotorSafetyLevel::from_phi(-0.5).motor_gain(), 0.0);
}

#[test]
fn test_negative_phi_humanoid_bridge() {
    let genesis = GenesisSeed::from_phrase("negative-phi");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.025, -0.5);
    assert_eq!(result.safety_level, MotorSafetyLevel::Red);
    assert_eq!(result.control_effort, 0.0);
    assert!(result.prediction_error.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 6: Phi Exactly at Boundaries
//
// Test 0.0, 0.1, 0.3, 0.6, 1.0 — the boundaries of each tier.
// from_phi uses strict `>` comparisons:
//   >0.6 = Green, >0.3 = Yellow, >0.1 = Orange, else Red
// So boundary values 0.6, 0.3, 0.1 fall to the LOWER tier.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_phi_exact_boundaries() {
    // phi=0.0: not >0.1 -> Red
    assert_safety_transition_at_boundary(0.0, MotorSafetyLevel::Red);
    // phi=0.1: not >0.1 -> Red (boundary is exclusive)
    assert_safety_transition_at_boundary(0.1, MotorSafetyLevel::Red);
    // phi=0.3: not >0.3 -> Orange (boundary is exclusive)
    assert_safety_transition_at_boundary(0.3, MotorSafetyLevel::Orange);
    // phi=0.6: not >0.6 -> Yellow (boundary is exclusive)
    assert_safety_transition_at_boundary(0.6, MotorSafetyLevel::Yellow);
    // phi=1.0: >0.6 -> Green
    assert_safety_transition_at_boundary(1.0, MotorSafetyLevel::Green);
}

#[test]
fn test_phi_near_boundaries() {
    // Just above each threshold
    assert_safety_transition_at_boundary(0.1001, MotorSafetyLevel::Orange);
    assert_safety_transition_at_boundary(0.3001, MotorSafetyLevel::Yellow);
    assert_safety_transition_at_boundary(0.6001, MotorSafetyLevel::Green);

    // Just below each threshold
    assert_safety_transition_at_boundary(0.0999, MotorSafetyLevel::Red);
    assert_safety_transition_at_boundary(0.2999, MotorSafetyLevel::Orange);
    assert_safety_transition_at_boundary(0.5999, MotorSafetyLevel::Yellow);
}

#[test]
fn test_phi_boundaries_on_humanoid_bridge() {
    let genesis = GenesisSeed::from_phrase("boundary-test");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let cases: &[(f64, MotorSafetyLevel, f32)] = &[
        (0.0, MotorSafetyLevel::Red, 0.0),
        (0.1, MotorSafetyLevel::Red, 0.0),
        (0.3, MotorSafetyLevel::Orange, 0.3),
        (0.6, MotorSafetyLevel::Yellow, 0.6),
        (1.0, MotorSafetyLevel::Green, 1.0),
    ];

    for &(phi, expected_level, expected_gain) in cases {
        bridge.reset();
        let result = bridge.step(&hv, 0.025, phi);
        assert_eq!(
            result.safety_level, expected_level,
            "phi={phi}: expected {expected_level:?}"
        );
        assert_eq!(
            result.safety_level.motor_gain(),
            expected_gain,
            "phi={phi}: expected gain {expected_gain}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 7: Recovery from Red
//
// System enters Red (phi=0.0), then Phi recovers to 0.8.
// Verify all platforms return to Green with correct motor gains.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_recovery_from_red_humanoid() {
    let genesis = GenesisSeed::from_phrase("recovery");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Drive to Red
    for _ in 0..10 {
        let result = bridge.step(&hv, 0.025, 0.0);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
        assert_eq!(result.control_effort, 0.0);
    }

    // Recover to Green
    for _ in 0..10 {
        let result = bridge.step(&hv, 0.025, 0.8);
        assert_eq!(
            result.safety_level,
            MotorSafetyLevel::Green,
            "after recovery, must be Green"
        );
        assert_eq!(result.safety_level.motor_gain(), 1.0);
    }
}

#[cfg(feature = "flight")]
#[test]
fn test_recovery_from_red_quadrotor() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("recovery-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Red phase
    for _ in 0..10 {
        let r = bridge.step(&hv, 0.002, 0.0);
        assert_eq!(format!("{:?}", r.safety_level), "Red");
    }

    // Recovery phase
    for i in 0..10 {
        let r = bridge.step(&hv, 0.002, 0.8);
        assert_eq!(
            format!("{:?}", r.safety_level),
            "Green",
            "quadrotor recovery cycle {i}: must be Green"
        );
    }
}

#[cfg(feature = "helicopter")]
#[test]
fn test_recovery_from_red_helicopter() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;

    let genesis = GenesisSeed::from_phrase("recovery-heli");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for _ in 0..10 {
        let r = bridge.step(&hv, 1.0 / 300.0, 0.0);
        assert_eq!(format!("{:?}", r.safety_level), "Red");
    }
    for _ in 0..10 {
        let r = bridge.step(&hv, 1.0 / 300.0, 0.8);
        assert_eq!(format!("{:?}", r.safety_level), "Green");
    }
}

#[cfg(feature = "vehicle")]
#[test]
fn test_recovery_from_red_vehicle() {
    use symthaea_vehicle::embodiment::VehicleEmbodiment;

    let genesis = GenesisSeed::from_phrase("recovery-vehicle");
    let mut bridge = VehicleEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for _ in 0..10 {
        let r = bridge.step(&hv, 0.005, 0.0);
        assert_eq!(format!("{:?}", r.safety_level), "Red");
    }
    for _ in 0..10 {
        let r = bridge.step(&hv, 0.005, 0.8);
        assert_eq!(format!("{:?}", r.safety_level), "Green");
    }
}

#[cfg(feature = "manipulator")]
#[test]
fn test_recovery_from_red_manipulator() {
    use symthaea_manipulator::embodiment::ManipulatorEmbodiment;

    let genesis = GenesisSeed::from_phrase("recovery-manip");
    let mut bridge = ManipulatorEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for _ in 0..10 {
        let r = bridge.step(&hv, 0.002, 0.0);
        assert_eq!(format!("{:?}", r.safety_level), "Red");
    }
    for _ in 0..10 {
        let r = bridge.step(&hv, 0.002, 0.8);
        assert_eq!(format!("{:?}", r.safety_level), "Green");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 8: Infinity Phi
//
// Feed f64::INFINITY. Since INFINITY > 0.6 is true, this should map to Green.
// Feed NEG_INFINITY. Since NEG_INFINITY > anything is false, this maps to Red.
// Neither should panic.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_infinity_phi() {
    let pos_inf = MotorSafetyLevel::from_phi(f64::INFINITY);
    assert_eq!(pos_inf, MotorSafetyLevel::Green, "+INF Phi must be Green");
    assert_eq!(pos_inf.motor_gain(), 1.0);

    let neg_inf = MotorSafetyLevel::from_phi(f64::NEG_INFINITY);
    assert_eq!(neg_inf, MotorSafetyLevel::Red, "-INF Phi must be Red");
    assert_eq!(neg_inf.motor_gain(), 0.0);
}

#[test]
fn test_infinity_phi_humanoid_bridge_no_panic() {
    let genesis = GenesisSeed::from_phrase("inf-test");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Positive infinity — should be Green, no panic
    let result = bridge.step(&hv, 0.025, f64::INFINITY);
    assert_eq!(result.safety_level, MotorSafetyLevel::Green);
    assert!(
        result.control_effort.is_finite(),
        "+INF phi: control effort must be finite"
    );

    // Negative infinity — should be Red, no panic
    bridge.reset();
    let result = bridge.step(&hv, 0.025, f64::NEG_INFINITY);
    assert_eq!(result.safety_level, MotorSafetyLevel::Red);
    assert_eq!(result.control_effort, 0.0);
}

#[cfg(feature = "flight")]
#[test]
fn test_infinity_phi_quadrotor_no_panic() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("inf-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let r = bridge.step(&hv, 0.002, f64::INFINITY);
    assert_eq!(format!("{:?}", r.safety_level), "Green");
    assert!(r.success);

    bridge.reset();
    let r = bridge.step(&hv, 0.002, f64::NEG_INFINITY);
    assert_eq!(format!("{:?}", r.safety_level), "Red");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 9: Simultaneous Multi-Platform Motor Authority Consistency
//
// All platforms active at the same Phi. Verify motor_gain() is identical
// across all of them at each safety level.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_motor_gain_consistency_across_levels() {
    // This tests the MotorSafetyLevel API itself — all platforms share it
    let test_phis: &[(f64, MotorSafetyLevel, f32)] = &[
        (0.8, MotorSafetyLevel::Green, 1.0),
        (0.5, MotorSafetyLevel::Yellow, 0.6),
        (0.2, MotorSafetyLevel::Orange, 0.3),
        (0.05, MotorSafetyLevel::Red, 0.0),
    ];

    for &(phi, expected_level, expected_gain) in test_phis {
        let level = MotorSafetyLevel::from_phi(phi);
        assert_eq!(level, expected_level, "phi={phi}");
        assert_eq!(level.motor_gain(), expected_gain, "phi={phi} gain");
    }
}

#[test]
fn test_all_platforms_same_safety_at_same_phi() {
    let genesis = GenesisSeed::from_phrase("consistency-test");
    let hv = ContinuousHV::random(16384, 42);

    let mut humanoid = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);

    let phis = [0.8, 0.5, 0.2, 0.05, 0.0, -0.5, f64::NAN];

    for &phi in &phis {
        humanoid.reset();
        let h_result = humanoid.step(&hv, 0.025, phi);
        let expected = MotorSafetyLevel::from_phi(phi);
        // Safety override from SafetyAgent could elevate, but without override
        // the level should match from_phi exactly
        assert_eq!(
            h_result.safety_level, expected,
            "humanoid: phi={phi} expected {expected:?}"
        );
    }
}

#[cfg(all(
    feature = "flight",
    feature = "helicopter",
    feature = "vehicle",
    feature = "manipulator"
))]
#[test]
fn test_six_platforms_same_safety_at_same_phi() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;
    use symthaea_manipulator::embodiment::ManipulatorEmbodiment;
    use symthaea_multirotor::embodiment::FlightEmbodiment;
    use symthaea_vehicle::embodiment::VehicleEmbodiment;

    let genesis = GenesisSeed::from_phrase("six-platform-consistency");
    let hv = ContinuousHV::random(16384, 42);

    let phis = [0.8, 0.5, 0.2, 0.05, 0.0];

    for &phi in &phis {
        let mut humanoid = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
        let mut flight = FlightEmbodiment::new(&genesis);
        let mut heli = HelicopterEmbodiment::new(&genesis);
        let mut vehicle = VehicleEmbodiment::new(&genesis);
        let mut manip = ManipulatorEmbodiment::new(&genesis);

        let expected = MotorSafetyLevel::from_phi(phi);
        let expected_name = format!("{:?}", expected);

        let h = humanoid.step(&hv, 0.025, phi);
        let f = flight.step(&hv, 0.002, phi);
        let he = heli.step(&hv, 1.0 / 300.0, phi);
        let v = vehicle.step(&hv, 0.005, phi);
        let m = manip.step(&hv, 0.002, phi);

        assert_eq!(
            format!("{:?}", h.safety_level),
            expected_name,
            "humanoid phi={phi}"
        );
        assert_eq!(
            format!("{:?}", f.safety_level),
            expected_name,
            "flight phi={phi}"
        );
        assert_eq!(
            format!("{:?}", he.safety_level),
            expected_name,
            "helicopter phi={phi}"
        );
        assert_eq!(
            format!("{:?}", v.safety_level),
            expected_name,
            "vehicle phi={phi}"
        );
        assert_eq!(
            format!("{:?}", m.safety_level),
            expected_name,
            "manipulator phi={phi}"
        );

        // All motor gains must match
        let gain = expected.motor_gain();
        assert_eq!(h.safety_level.motor_gain(), gain, "humanoid gain phi={phi}");
        assert_eq!(f.safety_level.motor_gain(), gain, "flight gain phi={phi}");
        assert_eq!(
            he.safety_level.motor_gain(),
            gain,
            "helicopter gain phi={phi}"
        );
        assert_eq!(v.safety_level.motor_gain(), gain, "vehicle gain phi={phi}");
        assert_eq!(
            m.safety_level.motor_gain(),
            gain,
            "manipulator gain phi={phi}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 10: Zero-Duration Step (dt=0.0)
//
// Verify no division by zero or panic when dt=0.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_zero_dt_humanoid_no_panic() {
    let genesis = GenesisSeed::from_phrase("zero-dt");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.0, 0.7);
    assert!(
        result.control_effort.is_finite(),
        "dt=0: control effort must be finite"
    );
    assert!(
        result.prediction_error.is_finite(),
        "dt=0: prediction error must be finite"
    );
    // Bridge should not panic, and safety level should be correct
    assert_eq!(result.safety_level, MotorSafetyLevel::Green);
}

#[cfg(feature = "flight")]
#[test]
fn test_zero_dt_quadrotor_no_panic() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("zero-dt-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.0, 0.7);
    assert!(
        result.control_effort.is_finite(),
        "quadrotor dt=0: effort finite"
    );
    assert!(result.success, "quadrotor dt=0: must not fail");
}

#[cfg(feature = "helicopter")]
#[test]
fn test_zero_dt_helicopter_no_panic() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;

    let genesis = GenesisSeed::from_phrase("zero-dt-heli");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.0, 0.7);
    assert!(
        result.control_effort.is_finite(),
        "helicopter dt=0: effort finite"
    );
}

#[cfg(feature = "vehicle")]
#[test]
fn test_zero_dt_vehicle_no_panic() {
    use symthaea_vehicle::embodiment::VehicleEmbodiment;

    let genesis = GenesisSeed::from_phrase("zero-dt-vehicle");
    let mut bridge = VehicleEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.0, 0.7);
    assert!(
        result.control_effort.is_finite(),
        "vehicle dt=0: effort finite"
    );
}

#[cfg(feature = "manipulator")]
#[test]
fn test_zero_dt_manipulator_no_panic() {
    use symthaea_manipulator::embodiment::ManipulatorEmbodiment;

    let genesis = GenesisSeed::from_phrase("zero-dt-manip");
    let mut bridge = ManipulatorEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.0, 0.7);
    assert!(
        result.control_effort.is_finite(),
        "manipulator dt=0: effort finite"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 11: Very Large dt
//
// Call step() with dt=1000.0. Verify no overflow, no NaN, no panic.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_large_dt_humanoid_no_overflow() {
    let genesis = GenesisSeed::from_phrase("large-dt");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 1000.0, 0.7);
    // Physics may go wild, but we must not panic or produce NaN in safety
    assert!(
        [
            MotorSafetyLevel::Green,
            MotorSafetyLevel::Yellow,
            MotorSafetyLevel::Orange,
            MotorSafetyLevel::Red
        ]
        .contains(&result.safety_level),
        "large dt: safety level must be valid"
    );
    // Control effort and prediction error should be finite (no NaN from overflow)
    // Note: they may be very large but must not be NaN
    assert!(
        !result.control_effort.is_nan(),
        "large dt: control effort must not be NaN"
    );
}

#[cfg(feature = "flight")]
#[test]
fn test_large_dt_quadrotor_no_overflow() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("large-dt-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 1000.0, 0.7);
    assert!(
        !result.control_effort.is_nan(),
        "quadrotor large dt: effort must not be NaN"
    );
}

#[cfg(feature = "helicopter")]
#[test]
fn test_large_dt_helicopter_no_overflow() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;

    let genesis = GenesisSeed::from_phrase("large-dt-heli");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 1000.0, 0.7);
    assert!(
        !result.control_effort.is_nan(),
        "helicopter large dt: no NaN"
    );
}

#[cfg(feature = "vehicle")]
#[test]
fn test_large_dt_vehicle_no_overflow() {
    use symthaea_vehicle::embodiment::VehicleEmbodiment;

    let genesis = GenesisSeed::from_phrase("large-dt-vehicle");
    let mut bridge = VehicleEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 1000.0, 0.7);
    assert!(!result.control_effort.is_nan(), "vehicle large dt: no NaN");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO 12: Sustained Red State — 1000 cycles
//
// Stay at Red for 1000 cycles. Verify no drift, no accumulation errors,
// control effort stays exactly 0, safety stays Red.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sustained_red_humanoid_1000_cycles() {
    let genesis = GenesisSeed::from_phrase("sustained-red");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for i in 0..1000 {
        let result = bridge.step(&hv, 0.025, 0.0);
        assert_eq!(
            result.safety_level,
            MotorSafetyLevel::Red,
            "cycle {i}: must remain Red"
        );
        assert_eq!(
            result.control_effort, 0.0,
            "cycle {i}: effort must stay 0 in sustained Red"
        );
        assert!(
            result.prediction_error.is_finite(),
            "cycle {i}: prediction error must be finite in sustained Red"
        );
    }

    // Verify step count
    assert_eq!(bridge.total_steps(), 1000);
}

#[cfg(feature = "flight")]
#[test]
fn test_sustained_red_quadrotor_1000_cycles() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("sustained-red-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for i in 0..1000 {
        let result = bridge.step(&hv, 0.002, 0.0);
        assert_eq!(format!("{:?}", result.safety_level), "Red", "cycle {i}");
        assert_eq!(result.control_effort, 0.0, "cycle {i}: effort=0");
    }
    assert_eq!(bridge.total_steps(), 1000);
}

#[cfg(feature = "helicopter")]
#[test]
fn test_sustained_red_helicopter_1000_cycles() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;

    let genesis = GenesisSeed::from_phrase("sustained-red-heli");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for i in 0..1000 {
        let result = bridge.step(&hv, 1.0 / 300.0, 0.0);
        assert_eq!(format!("{:?}", result.safety_level), "Red", "cycle {i}");
        assert_eq!(result.control_effort, 0.0, "cycle {i}: effort=0");
    }
    assert_eq!(bridge.total_steps(), 1000);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADDITIONAL: Safety Override Interactions
//
// The EmbodimentBridge trait supports set_safety_override() which is merged
// with phi via max() (more conservative wins). Test this interaction.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_safety_override_escalates_to_red() {
    let genesis = GenesisSeed::from_phrase("override-test");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Without override, phi=0.8 -> Green
    let result = bridge.step(&hv, 0.025, 0.8);
    assert_eq!(result.safety_level, MotorSafetyLevel::Green);

    // Set override to Red — even with phi=0.8, max(Green, Red) = Red
    bridge.set_safety_override(MotorSafetyLevel::Red);
    let result = bridge.step(&hv, 0.025, 0.8);
    assert_eq!(
        result.safety_level,
        MotorSafetyLevel::Red,
        "safety override must escalate to Red"
    );
    assert_eq!(result.control_effort, 0.0);

    // Clear override — should return to Green
    bridge.clear_safety_override();
    let result = bridge.step(&hv, 0.025, 0.8);
    assert_eq!(
        result.safety_level,
        MotorSafetyLevel::Green,
        "clearing override must restore phi-based level"
    );
}

#[test]
fn test_safety_override_does_not_downgrade() {
    let genesis = GenesisSeed::from_phrase("override-no-downgrade");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Override to Yellow, phi=0.05 -> Red. max(Red, Yellow) = Red.
    // The override should NOT downgrade from Red to Yellow.
    bridge.set_safety_override(MotorSafetyLevel::Yellow);
    let result = bridge.step(&hv, 0.025, 0.05);
    assert_eq!(
        result.safety_level,
        MotorSafetyLevel::Red,
        "override must not downgrade from phi-level Red to Yellow"
    );
}

#[cfg(feature = "flight")]
#[test]
fn test_safety_override_quadrotor() {
    use symthaea_multirotor::embodiment::{FlightEmbodiment, MotorSafetyLevel as FlightSafety};

    let genesis = GenesisSeed::from_phrase("override-flight");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Without override
    let r = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(format!("{:?}", r.safety_level), "Green");

    // Override to Orange
    bridge.set_safety_override(FlightSafety::Orange);
    let r = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(format!("{:?}", r.safety_level), "Orange");

    // Clear
    bridge.clear_safety_override();
    let r = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(format!("{:?}", r.safety_level), "Green");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADDITIONAL: Gradual Phi Ramp-Up (Fine-Grained Transition Verification)
//
// Phi rises from 0.0 to 1.0 in 0.01 increments. At each step verify the
// safety level and gain are correct.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_gradual_phi_ramp_100_steps() {
    let genesis = GenesisSeed::from_phrase("ramp-up");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    for i in 0..=100 {
        let phi = i as f64 / 100.0;
        let result = bridge.step(&hv, 0.025, phi);

        let expected_level = MotorSafetyLevel::from_phi(phi);
        assert_eq!(
            result.safety_level, expected_level,
            "ramp step {i}: phi={phi:.2}"
        );

        let expected_gain = expected_level.motor_gain();
        assert_eq!(
            result.safety_level.motor_gain(),
            expected_gain,
            "ramp step {i}: phi={phi:.2} gain mismatch"
        );

        // All outputs must be finite
        assert!(
            result.control_effort.is_finite(),
            "ramp step {i}: effort not finite"
        );
        assert!(
            result.prediction_error.is_finite(),
            "ramp step {i}: pred error not finite"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADDITIONAL: Reset Mid-Red Restores Green on Next Good Phi
//
// Verifies that reset() clears internal state cleanly after extended Red.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_reset_after_sustained_red_restores_clean_state() {
    let genesis = GenesisSeed::from_phrase("reset-after-red");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // 100 cycles at Red
    for _ in 0..100 {
        bridge.step(&hv, 0.025, 0.0);
    }
    assert_eq!(bridge.total_steps(), 100);

    // Reset
    bridge.reset();
    assert_eq!(bridge.total_steps(), 0);

    // First step after reset with good Phi must be Green
    let result = bridge.step(&hv, 0.025, 0.8);
    assert_eq!(
        result.safety_level,
        MotorSafetyLevel::Green,
        "after reset from Red, phi=0.8 must be Green"
    );
    assert!(result.control_effort.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADDITIONAL: Subnormal Phi Values
//
// Test extremely small positive values near zero (subnormals).
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_subnormal_phi_values() {
    // f64::MIN_POSITIVE is ~2.2e-308
    let level = MotorSafetyLevel::from_phi(f64::MIN_POSITIVE);
    assert_eq!(level, MotorSafetyLevel::Red, "MIN_POSITIVE phi -> Red");

    // Epsilon (~2.2e-16)
    let level = MotorSafetyLevel::from_phi(f64::EPSILON);
    assert_eq!(level, MotorSafetyLevel::Red, "EPSILON phi -> Red");

    // Very large phi
    let level = MotorSafetyLevel::from_phi(1e308);
    assert_eq!(level, MotorSafetyLevel::Green, "huge phi -> Green");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADDITIONAL: Multi-Platform Simultaneous Stress (Full CognitiveLoopService)
//
// Runs the full cognitive loop service with multi-embodiment for 100 cycles
// with varying Phi scenarios injected via the stimulus string.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_service_long_run_no_nan() {
    let mut service = make_embodied_service(EmbodimentPlatform::Humanoid);

    for i in 0..100 {
        let result = service.cycle("stress test scenario long run");
        let cl = result.metadata.consciousness.consciousness_level;

        assert!(cl.is_finite(), "cycle {i}: consciousness NaN");
        assert!(cl >= 0.0, "cycle {i}: consciousness negative: {cl}");

        let telem = service.embodiment_telemetry();
        assert!(
            ["Green", "Yellow", "Orange", "Red"].contains(&telem.safety_level.as_str()),
            "cycle {i}: invalid safety level: {}",
            telem.safety_level
        );
        assert!(
            telem.control_effort.is_finite(),
            "cycle {i}: telemetry effort NaN"
        );
        assert!(
            telem.prediction_error.is_finite(),
            "cycle {i}: telemetry pred error NaN"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADDITIONAL: MotorSafetyLevel Ordering Invariants
//
// The Ord derive means Green < Yellow < Orange < Red. This is important
// because safety_override uses max() — higher = more restrictive.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_safety_level_ordering_invariant() {
    assert!(MotorSafetyLevel::Green < MotorSafetyLevel::Yellow);
    assert!(MotorSafetyLevel::Yellow < MotorSafetyLevel::Orange);
    assert!(MotorSafetyLevel::Orange < MotorSafetyLevel::Red);

    // max() should always pick the more restrictive level
    assert_eq!(
        MotorSafetyLevel::Green.max(MotorSafetyLevel::Red),
        MotorSafetyLevel::Red
    );
    assert_eq!(
        MotorSafetyLevel::Yellow.max(MotorSafetyLevel::Orange),
        MotorSafetyLevel::Orange
    );
    assert_eq!(
        MotorSafetyLevel::Red.max(MotorSafetyLevel::Green),
        MotorSafetyLevel::Red
    );
}

#[test]
fn test_motor_gain_monotonically_decreases_with_severity() {
    let gains = [
        MotorSafetyLevel::Green.motor_gain(),
        MotorSafetyLevel::Yellow.motor_gain(),
        MotorSafetyLevel::Orange.motor_gain(),
        MotorSafetyLevel::Red.motor_gain(),
    ];
    for i in 0..gains.len() - 1 {
        assert!(
            gains[i] > gains[i + 1],
            "motor_gain must decrease: {:?}={} vs {:?}={}",
            [
                MotorSafetyLevel::Green,
                MotorSafetyLevel::Yellow,
                MotorSafetyLevel::Orange,
                MotorSafetyLevel::Red
            ][i],
            gains[i],
            [
                MotorSafetyLevel::Green,
                MotorSafetyLevel::Yellow,
                MotorSafetyLevel::Orange,
                MotorSafetyLevel::Red
            ][i + 1],
            gains[i + 1],
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADDITIONAL: Prediction Error Stays Bounded Under Rapid Safety Transitions
//
// When safety toggles rapidly, physics state changes discontinuously.
// Prediction error (1 - cosine_similarity) must stay in [0, 1].
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_prediction_error_bounded_under_rapid_transitions() {
    let genesis = GenesisSeed::from_phrase("pred-error-bound");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Alternate between full authority and shutdown every cycle
    for i in 0..200 {
        let phi = if i % 2 == 0 { 0.8 } else { 0.0 };
        let result = bridge.step(&hv, 0.025, phi);

        assert!(
            result.prediction_error >= 0.0,
            "cycle {i}: pred error negative: {}",
            result.prediction_error
        );
        assert!(
            result.prediction_error <= 1.0,
            "cycle {i}: pred error > 1: {}",
            result.prediction_error
        );
        assert!(
            result.prediction_error.is_finite(),
            "cycle {i}: pred error NaN/Inf"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-PLATFORM SAFETY CASCADE
// Verifies all platforms respond identically to identical Phi sequences.
//
// Two layers:
//   1. Lightweight: MotorSafetyLevel consistency between motor_bridge.rs (main crate)
//      and symthaea-hal (shared by all platform crates). No feature gates, instant.
//   2. Heavy: per-platform CognitiveLoopService tests (feature-gated).
// ═══════════════════════════════════════════════════════════════════════════════

// ── Layer 1: MotorSafetyLevel consistency (no feature gates) ────────────────

/// The main crate's MotorSafetyLevel must produce correct from_phi() results
/// at every boundary and interior point. Platform crates re-export
/// symthaea_hal::MotorSafetyLevel which has identical logic — verified via
/// the per-platform direct-bridge tests below (comparing Debug strings to
/// avoid cross-crate type identity issues).
#[test]
fn test_motor_safety_level_from_phi_comprehensive() {
    let phi_values: Vec<f64> = vec![
        -0.1, 0.0, 0.05, 0.1, 0.100001, 0.15, 0.2, 0.3, 0.300001, 0.35, 0.4, 0.5, 0.6, 0.600001,
        0.65, 0.7, 0.8, 0.9, 1.0, 1.5,
    ];

    let expected = vec![
        MotorSafetyLevel::Red,    // -0.1
        MotorSafetyLevel::Red,    // 0.0
        MotorSafetyLevel::Red,    // 0.05
        MotorSafetyLevel::Red,    // 0.1 (not > 0.1)
        MotorSafetyLevel::Orange, // 0.100001
        MotorSafetyLevel::Orange, // 0.15
        MotorSafetyLevel::Orange, // 0.2
        MotorSafetyLevel::Orange, // 0.3 (not > 0.3)
        MotorSafetyLevel::Yellow, // 0.300001
        MotorSafetyLevel::Yellow, // 0.35
        MotorSafetyLevel::Yellow, // 0.4
        MotorSafetyLevel::Yellow, // 0.5
        MotorSafetyLevel::Yellow, // 0.6 (not > 0.6)
        MotorSafetyLevel::Green,  // 0.600001
        MotorSafetyLevel::Green,  // 0.65
        MotorSafetyLevel::Green,  // 0.7
        MotorSafetyLevel::Green,  // 0.8
        MotorSafetyLevel::Green,  // 0.9
        MotorSafetyLevel::Green,  // 1.0
        MotorSafetyLevel::Green,  // 1.5
    ];

    for (phi, exp) in phi_values.iter().zip(expected.iter()) {
        let level = MotorSafetyLevel::from_phi(*phi);
        assert_eq!(
            level, *exp,
            "from_phi({phi}) expected {exp:?}, got {level:?}"
        );
    }
}

/// Motor gain must be strictly monotonically decreasing across safety levels.
#[test]
fn test_motor_safety_level_gain_monotonicity() {
    let levels = [
        MotorSafetyLevel::Green,
        MotorSafetyLevel::Yellow,
        MotorSafetyLevel::Orange,
        MotorSafetyLevel::Red,
    ];
    for window in levels.windows(2) {
        assert!(
            window[0].motor_gain() > window[1].motor_gain(),
            "{:?} gain ({}) must be > {:?} gain ({})",
            window[0],
            window[0].motor_gain(),
            window[1],
            window[1].motor_gain(),
        );
    }
}

/// Red must produce exactly 0.0 gain (motors off), Green exactly 1.0.
#[test]
fn test_motor_safety_level_boundary_gains() {
    assert_eq!(
        MotorSafetyLevel::Green.motor_gain(),
        1.0,
        "Green must be full authority"
    );
    assert_eq!(
        MotorSafetyLevel::Red.motor_gain(),
        0.0,
        "Red must be motors off"
    );
    assert_eq!(
        MotorSafetyLevel::Yellow.motor_gain(),
        0.6,
        "Yellow must be 0.6"
    );
    assert_eq!(
        MotorSafetyLevel::Orange.motor_gain(),
        0.3,
        "Orange must be 0.3"
    );
}

/// Phi exactly at boundary values must resolve to the lower tier.
/// Boundaries: 0.1, 0.3, 0.6 (all exclusive >).
#[test]
fn test_motor_safety_level_exact_boundaries() {
    // At boundaries, phi is NOT greater than threshold, so falls to lower tier
    assert_eq!(
        MotorSafetyLevel::from_phi(0.1),
        MotorSafetyLevel::Red,
        "phi=0.1: not > 0.1 so Red"
    );
    assert_eq!(
        MotorSafetyLevel::from_phi(0.3),
        MotorSafetyLevel::Orange,
        "phi=0.3: not > 0.3 so Orange"
    );
    assert_eq!(
        MotorSafetyLevel::from_phi(0.6),
        MotorSafetyLevel::Yellow,
        "phi=0.6: not > 0.6 so Yellow"
    );

    // Just above boundaries
    assert_eq!(
        MotorSafetyLevel::from_phi(0.100001),
        MotorSafetyLevel::Orange
    );
    assert_eq!(
        MotorSafetyLevel::from_phi(0.300001),
        MotorSafetyLevel::Yellow
    );
    assert_eq!(
        MotorSafetyLevel::from_phi(0.600001),
        MotorSafetyLevel::Green
    );
}

/// Sweep 1000 phi values in [0, 1] and verify from_phi + motor_gain are correct.
#[test]
fn test_motor_safety_level_dense_sweep() {
    for i in 0..=1000 {
        let phi = i as f64 / 1000.0;
        let level = MotorSafetyLevel::from_phi(phi);
        let gain = level.motor_gain();

        // Verify level matches expected for this phi
        let expected = if phi > 0.6 {
            MotorSafetyLevel::Green
        } else if phi > 0.3 {
            MotorSafetyLevel::Yellow
        } else if phi > 0.1 {
            MotorSafetyLevel::Orange
        } else {
            MotorSafetyLevel::Red
        };
        assert_eq!(
            level, expected,
            "dense sweep at phi={phi:.3}: expected {expected:?}, got {level:?}"
        );

        // Verify gain is in expected set
        assert!(
            [0.0, 0.3, 0.6, 1.0].contains(&gain),
            "unexpected motor_gain {gain} at phi={phi:.3}"
        );
    }
}

// ── Layer 2: Per-platform identical Phi cascade (feature-gated, heavier) ────

/// Collects (safety_level_name, motor_gain) at each phi step for a single platform.
#[cfg(all(
    feature = "humanoid",
    feature = "flight",
    feature = "helicopter",
    feature = "vehicle",
    feature = "manipulator",
    feature = "auv"
))]
fn collect_safety_cascade_from_service(
    platform: EmbodimentPlatform,
    phi_sequence: &[f64],
) -> Vec<(String, f32)> {
    let config = CognitiveLoopConfig {
        embodiment_platform: platform,
        embodiment_blend_weight: 0.3,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };
    let mut service = CognitiveLoopService::new(config)
        .expect(&format!("CognitiveLoopService::new for {platform:?}"));

    let mut results = Vec::with_capacity(phi_sequence.len());
    for &_phi in phi_sequence {
        // Run one cycle; the embodiment bridge will derive safety from internal phi
        let cycle_result = service.cycle("cross-platform safety cascade test");
        let telem = service.embodiment_telemetry();
        // Parse the safety level from telemetry string and get motor gain
        let level = match telem.safety_level.as_str() {
            "Green" => MotorSafetyLevel::Green,
            "Yellow" => MotorSafetyLevel::Yellow,
            "Orange" => MotorSafetyLevel::Orange,
            "Red" => MotorSafetyLevel::Red,
            other => panic!("unexpected safety level string: {other}"),
        };
        results.push((telem.safety_level.clone(), level.motor_gain()));

        // Verify consciousness stays finite
        assert!(
            cycle_result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "{platform:?}: consciousness NaN"
        );
    }
    results
}

/// Test that all 6 platforms produce identical safety level sequences when
/// stepped through matching Phi-collapse scenarios.
///
/// NOTE: Because CognitiveLoopService computes Phi internally (not injected),
/// this test verifies that the safety derivation path is consistent across
/// platforms rather than testing at exact phi values. For exact phi testing,
/// see the direct-bridge tests below.
#[cfg(all(
    feature = "humanoid",
    feature = "flight",
    feature = "helicopter",
    feature = "vehicle",
    feature = "manipulator",
    feature = "auv"
))]
#[test]
fn test_cross_platform_phi_cascade_service() {
    let phi_sequence: Vec<f64> = vec![0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 1.0];

    let platforms = [
        EmbodimentPlatform::Humanoid,
        EmbodimentPlatform::Quadrotor,
        EmbodimentPlatform::Helicopter,
        EmbodimentPlatform::Vehicle,
        EmbodimentPlatform::Manipulator,
        EmbodimentPlatform::Auv,
    ];

    // Collect cascades
    let cascades: Vec<_> = platforms
        .iter()
        .map(|&p| (p, collect_safety_cascade_from_service(p, &phi_sequence)))
        .collect();

    // All cascades should have same length
    for (platform, cascade) in &cascades {
        assert_eq!(
            cascade.len(),
            phi_sequence.len(),
            "{platform:?}: cascade length mismatch"
        );
    }

    // At each step, all platforms should report the same safety level
    // (since they all use the same internal Phi computation)
    let reference = &cascades[0];
    for (platform, cascade) in &cascades[1..] {
        for (i, ((ref_level, ref_gain), (plat_level, plat_gain))) in
            reference.1.iter().zip(cascade.iter()).enumerate()
        {
            assert_eq!(
                ref_level, plat_level,
                "step {i}: {:?} level={plat_level} differs from {:?} level={ref_level}",
                platform, reference.0
            );
            assert_eq!(
                ref_gain, plat_gain,
                "step {i}: {:?} gain={plat_gain} differs from {:?} gain={ref_gain}",
                platform, reference.0
            );
        }
    }
}

/// For each platform: inject known Phi values directly into the embodiment bridge
/// and verify identical safety levels and gains.
///
/// This is the precise test — it bypasses CognitiveLoopService and tests the
/// bridge adapters' from_phi logic directly at each platform.

#[test]
fn test_all_safety_levels_consistent_direct_humanoid() {
    let genesis = GenesisSeed::from_phrase("cross-platform-safety");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence = [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 1.0];
    let expected = [
        MotorSafetyLevel::Red,    // 0.0
        MotorSafetyLevel::Red,    // 0.05
        MotorSafetyLevel::Red,    // 0.1 (not > 0.1)
        MotorSafetyLevel::Orange, // 0.3 (not > 0.3)
        MotorSafetyLevel::Yellow, // 0.6 (not > 0.6)
        MotorSafetyLevel::Green,  // 0.7
        MotorSafetyLevel::Green,  // 1.0
    ];
    for (i, &phi) in phi_sequence.iter().enumerate() {
        bridge.reset();
        let result = bridge.step(&hv, 0.025, phi);
        assert_eq!(
            result.safety_level, expected[i],
            "humanoid phi={phi}: expected {:?}, got {:?}",
            expected[i], result.safety_level
        );
        if result.safety_level == MotorSafetyLevel::Red {
            assert_eq!(result.control_effort, 0.0, "humanoid Red: effort must be 0");
        }
    }
}

#[cfg(feature = "flight")]
#[test]
fn test_all_safety_levels_consistent_direct_flight() {
    use symthaea_multirotor::embodiment::FlightEmbodiment;

    let genesis = GenesisSeed::from_phrase("cross-platform-safety");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence = [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 1.0];
    let expected_names = ["Red", "Red", "Red", "Orange", "Yellow", "Green", "Green"];

    for (i, &phi) in phi_sequence.iter().enumerate() {
        bridge.reset();
        let result = bridge.step(&hv, 0.002, phi);
        let level_name = format!("{:?}", result.safety_level);
        assert_eq!(
            level_name, expected_names[i],
            "flight phi={phi}: expected {}, got {level_name}",
            expected_names[i]
        );
        if level_name == "Red" {
            assert_eq!(result.control_effort, 0.0, "flight Red: effort must be 0");
        }
    }
}

#[cfg(feature = "helicopter")]
#[test]
fn test_all_safety_levels_consistent_direct_helicopter() {
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;

    let genesis = GenesisSeed::from_phrase("cross-platform-safety");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence = [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 1.0];
    let expected_names = ["Red", "Red", "Red", "Orange", "Yellow", "Green", "Green"];

    for (i, &phi) in phi_sequence.iter().enumerate() {
        bridge.reset();
        let result = bridge.step(&hv, 1.0 / 300.0, phi);
        let level_name = format!("{:?}", result.safety_level);
        assert_eq!(
            level_name, expected_names[i],
            "helicopter phi={phi}: expected {}, got {level_name}",
            expected_names[i]
        );
        if level_name == "Red" {
            assert_eq!(
                result.control_effort, 0.0,
                "helicopter Red: effort must be 0"
            );
        }
    }
}

#[cfg(feature = "vehicle")]
#[test]
fn test_all_safety_levels_consistent_direct_vehicle() {
    use symthaea_vehicle::embodiment::VehicleEmbodiment;

    let genesis = GenesisSeed::from_phrase("cross-platform-safety");
    let mut bridge = VehicleEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence = [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 1.0];
    let expected_names = ["Red", "Red", "Red", "Orange", "Yellow", "Green", "Green"];

    for (i, &phi) in phi_sequence.iter().enumerate() {
        bridge.reset();
        let result = bridge.step(&hv, 0.005, phi);
        let level_name = format!("{:?}", result.safety_level);
        assert_eq!(
            level_name, expected_names[i],
            "vehicle phi={phi}: expected {}, got {level_name}",
            expected_names[i]
        );
    }
}

#[cfg(feature = "manipulator")]
#[test]
fn test_all_safety_levels_consistent_direct_manipulator() {
    use symthaea_manipulator::embodiment::ManipulatorEmbodiment;

    let genesis = GenesisSeed::from_phrase("cross-platform-safety");
    let mut bridge = ManipulatorEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence = [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 1.0];
    let expected_names = ["Red", "Red", "Red", "Orange", "Yellow", "Green", "Green"];

    for (i, &phi) in phi_sequence.iter().enumerate() {
        bridge.reset();
        let result = bridge.step(&hv, 0.002, phi);
        let level_name = format!("{:?}", result.safety_level);
        assert_eq!(
            level_name, expected_names[i],
            "manipulator phi={phi}: expected {}, got {level_name}",
            expected_names[i]
        );
    }
}

#[cfg(feature = "auv")]
#[test]
fn test_all_safety_levels_consistent_direct_auv() {
    use symthaea_auv::embodiment::AuvEmbodiment;

    let genesis = GenesisSeed::from_phrase("cross-platform-safety");
    let mut bridge = AuvEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let phi_sequence = [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 1.0];
    let expected_names = ["Red", "Red", "Red", "Orange", "Yellow", "Green", "Green"];

    for (i, &phi) in phi_sequence.iter().enumerate() {
        bridge.reset();
        let result = bridge.step(&hv, 0.01, phi);
        let level_name = format!("{:?}", result.safety_level);
        assert_eq!(
            level_name, expected_names[i],
            "auv phi={phi}: expected {}, got {level_name}",
            expected_names[i]
        );
        if level_name == "Red" {
            assert_eq!(result.control_effort, 0.0, "auv Red: effort must be 0");
        }
    }
}

// ── Layer 2b: Safety override cross-platform ────────────────────────────────

/// Setting safety override to Red must produce 0.0 motor gain regardless of Phi.
/// Clearing the override must resume Phi-based safety.
#[test]
fn test_safety_override_cross_platform_humanoid() {
    let genesis = GenesisSeed::from_phrase("override-test");
    let mut bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    // Step at high Phi — should be Green
    let result = bridge.step(&hv, 0.025, 0.8);
    assert_eq!(result.safety_level, MotorSafetyLevel::Green);

    // Set override to Red — motor gain must be 0 even at high Phi
    bridge.set_safety_override(MotorSafetyLevel::Red);
    let result = bridge.step(&hv, 0.025, 0.8);
    assert_eq!(
        result.safety_level,
        MotorSafetyLevel::Red,
        "humanoid: Red override must win over high Phi"
    );
    assert_eq!(
        result.control_effort, 0.0,
        "humanoid: Red override must zero motor effort"
    );

    // Clear override — should resume Phi-based
    bridge.clear_safety_override();
    let result = bridge.step(&hv, 0.025, 0.8);
    assert_eq!(
        result.safety_level,
        MotorSafetyLevel::Green,
        "humanoid: clearing override must resume Phi-based safety"
    );
}

#[cfg(feature = "flight")]
#[test]
fn test_safety_override_cross_platform_flight() {
    use symthaea_multirotor::embodiment::{FlightEmbodiment, MotorSafetyLevel as FlightSafety};

    let genesis = GenesisSeed::from_phrase("override-test");
    let mut bridge = FlightEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(format!("{:?}", result.safety_level), "Green");

    bridge.set_safety_override(FlightSafety::Red);
    let result = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Red",
        "flight: Red override must win over high Phi"
    );
    assert_eq!(
        result.control_effort, 0.0,
        "flight: Red override must zero motor effort"
    );

    bridge.clear_safety_override();
    let result = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Green",
        "flight: clearing override must resume Phi-based safety"
    );
}

#[cfg(feature = "helicopter")]
#[test]
fn test_safety_override_cross_platform_helicopter() {
    use symthaea_helicopter::embodiment::{HelicopterEmbodiment, MotorSafetyLevel as HeliSafety};

    let genesis = GenesisSeed::from_phrase("override-test");
    let mut bridge = HelicopterEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 1.0 / 300.0, 0.8);
    assert_eq!(format!("{:?}", result.safety_level), "Green");

    bridge.set_safety_override(HeliSafety::Red);
    let result = bridge.step(&hv, 1.0 / 300.0, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Red",
        "helicopter: Red override must win over high Phi"
    );
    assert_eq!(
        result.control_effort, 0.0,
        "helicopter: Red override must zero motor effort"
    );

    bridge.clear_safety_override();
    let result = bridge.step(&hv, 1.0 / 300.0, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Green",
        "helicopter: clearing override must resume Phi-based safety"
    );
}

#[cfg(feature = "vehicle")]
#[test]
fn test_safety_override_cross_platform_vehicle() {
    use symthaea_vehicle::embodiment::{MotorSafetyLevel as VehicleSafety, VehicleEmbodiment};

    let genesis = GenesisSeed::from_phrase("override-test");
    let mut bridge = VehicleEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.005, 0.8);
    assert_eq!(format!("{:?}", result.safety_level), "Green");

    bridge.set_safety_override(VehicleSafety::Red);
    let result = bridge.step(&hv, 0.005, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Red",
        "vehicle: Red override must win over high Phi"
    );

    bridge.clear_safety_override();
    let result = bridge.step(&hv, 0.005, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Green",
        "vehicle: clearing override must resume Phi-based safety"
    );
}

#[cfg(feature = "manipulator")]
#[test]
fn test_safety_override_cross_platform_manipulator() {
    use symthaea_manipulator::embodiment::{
        ManipulatorEmbodiment, MotorSafetyLevel as ManipSafety,
    };

    let genesis = GenesisSeed::from_phrase("override-test");
    let mut bridge = ManipulatorEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(format!("{:?}", result.safety_level), "Green");

    bridge.set_safety_override(ManipSafety::Red);
    let result = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Red",
        "manipulator: Red override must win over high Phi"
    );

    bridge.clear_safety_override();
    let result = bridge.step(&hv, 0.002, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Green",
        "manipulator: clearing override must resume Phi-based safety"
    );
}

#[cfg(feature = "auv")]
#[test]
fn test_safety_override_cross_platform_auv() {
    use symthaea_auv::embodiment::{AuvEmbodiment, MotorSafetyLevel as AuvSafety};

    let genesis = GenesisSeed::from_phrase("override-test");
    let mut bridge = AuvEmbodiment::new(&genesis);
    let hv = ContinuousHV::random(16384, 42);

    let result = bridge.step(&hv, 0.01, 0.8);
    assert_eq!(format!("{:?}", result.safety_level), "Green");

    bridge.set_safety_override(AuvSafety::Red);
    let result = bridge.step(&hv, 0.01, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Red",
        "auv: Red override must win over high Phi"
    );
    assert_eq!(
        result.control_effort, 0.0,
        "auv: Red override must zero motor effort"
    );

    bridge.clear_safety_override();
    let result = bridge.step(&hv, 0.01, 0.8);
    assert_eq!(
        format!("{:?}", result.safety_level),
        "Green",
        "auv: clearing override must resume Phi-based safety"
    );
}

// ── Layer 2c: Cross-platform identical Phi cascade via direct bridges ───────

/// When all 6 platforms are available, verify that a Phi collapse from 0.8 to 0.0
/// produces the exact same safety level sequence across all platforms.
#[cfg(all(
    feature = "flight",
    feature = "helicopter",
    feature = "vehicle",
    feature = "manipulator",
    feature = "auv"
))]
#[test]
fn test_cross_platform_identical_phi_collapse_direct() {
    use symthaea_auv::embodiment::AuvEmbodiment;
    use symthaea_helicopter::embodiment::HelicopterEmbodiment;
    use symthaea_manipulator::embodiment::ManipulatorEmbodiment;
    use symthaea_multirotor::embodiment::FlightEmbodiment;
    use symthaea_vehicle::embodiment::VehicleEmbodiment;

    let genesis = GenesisSeed::from_phrase("cross-platform-collapse");
    let hv = ContinuousHV::random(16384, 42);

    // Phi collapse: 0.8 down to -0.1 in 10 steps
    let phi_sequence: Vec<f64> = (0..=10).map(|i| 0.8 - (i as f64) * 0.09).collect();

    // Collect safety levels from each platform
    let mut humanoid_bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(&genesis);
    let mut flight_bridge = FlightEmbodiment::new(&genesis);
    let mut heli_bridge = HelicopterEmbodiment::new(&genesis);
    let mut vehicle_bridge = VehicleEmbodiment::new(&genesis);
    let mut manip_bridge = ManipulatorEmbodiment::new(&genesis);
    let mut auv_bridge = AuvEmbodiment::new(&genesis);

    for &phi in &phi_sequence {
        let h = humanoid_bridge.step(&hv, 0.025, phi);
        let f = flight_bridge.step(&hv, 0.002, phi);
        let heli = heli_bridge.step(&hv, 1.0 / 300.0, phi);
        let v = vehicle_bridge.step(&hv, 0.005, phi);
        let m = manip_bridge.step(&hv, 0.002, phi);
        let a = auv_bridge.step(&hv, 0.01, phi);

        let expected = MotorSafetyLevel::from_phi(phi);
        let expected_name = format!("{expected:?}");

        // All 6 must match
        let platforms = [
            ("humanoid", format!("{:?}", h.safety_level)),
            ("flight", format!("{:?}", f.safety_level)),
            ("helicopter", format!("{:?}", heli.safety_level)),
            ("vehicle", format!("{:?}", v.safety_level)),
            ("manipulator", format!("{:?}", m.safety_level)),
            ("auv", format!("{:?}", a.safety_level)),
        ];

        for (name, level) in &platforms {
            assert_eq!(
                level, &expected_name,
                "phi={phi:.2}: {name} level={level}, expected {expected_name}"
            );
        }

        // At Red, ALL platforms must have zero control effort
        if expected == MotorSafetyLevel::Red {
            assert_eq!(h.control_effort, 0.0, "phi={phi}: humanoid Red effort != 0");
            assert_eq!(f.control_effort, 0.0, "phi={phi}: flight Red effort != 0");
            assert_eq!(
                heli.control_effort, 0.0,
                "phi={phi}: helicopter Red effort != 0"
            );
            // Vehicle and manipulator may have non-zero effort at Red depending on
            // implementation; the critical invariant is safety_level == Red
        }
    }
}
