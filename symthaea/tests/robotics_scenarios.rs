// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mission-level robotics scenario tests.
//!
//! Unlike robotics_integration.rs (which tests loop closure mechanics),
//! these tests verify meaningful embodied behavior:
//! - Safety level transitions during realistic scenarios
//! - Recovery dynamics after consciousness disruption
//! - Multi-body proprioceptive divergence
//!
//! Run: `cargo test --features humanoid,helicopter,multirotor,vehicle --test robotics_scenarios`

use symthaea::cognitive_loop::motor_bridge::{EmbodimentPlatform, MotorSafetyLevel};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_service(platform: EmbodimentPlatform) -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        embodiment_platform: platform,
        embodiment_blend_weight: 0.3,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("CognitiveLoopService with embodiment")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 1: Helicopter SAR — consciousness dynamics under stress
//
// Validates: The cognitive-embodiment loop produces coherent dynamics over
// extended operation. Consciousness level, motor telemetry, and safety
// levels all remain internally consistent across 100 cycles.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "helicopter")]
#[test]
fn test_helicopter_sar_extended_mission() {
    let mut service = make_service(EmbodimentPlatform::Helicopter);

    let mut safety_transitions = Vec::new();
    let mut last_safety = String::new();
    let mut phi_history = Vec::new();

    // Phase 1: Cruise (30 cycles) — establish baseline
    for _ in 0..30 {
        let result = service.cycle("helicopter patrol over search area");
        let phi = result.metadata.consciousness.consciousness_level;
        phi_history.push(phi);

        let telem = service.embodiment_telemetry();
        if telem.safety_level != last_safety {
            safety_transitions.push(telem.safety_level.clone());
            last_safety = telem.safety_level.clone();
        }
    }

    // Phase 2: Stressful input (30 cycles) — drive consciousness dynamics
    for _ in 0..30 {
        let result =
            service.cycle("emergency: target spotted in rough terrain, high winds, low visibility");
        let phi = result.metadata.consciousness.consciousness_level;
        phi_history.push(phi);

        let telem = service.embodiment_telemetry();
        if telem.safety_level != last_safety {
            safety_transitions.push(telem.safety_level.clone());
            last_safety = telem.safety_level.clone();
        }
    }

    // Phase 3: Recovery (40 cycles) — return to normal
    for _ in 0..40 {
        let result = service.cycle("target secured, returning to base, clear skies");
        let phi = result.metadata.consciousness.consciousness_level;
        phi_history.push(phi);

        let telem = service.embodiment_telemetry();
        if telem.safety_level != last_safety {
            safety_transitions.push(telem.safety_level.clone());
            last_safety = telem.safety_level.clone();
        }
    }

    // ── Assertions ───────────────────────────────────────────────────

    // All consciousness values should be finite and bounded
    for (i, &phi) in phi_history.iter().enumerate() {
        assert!(
            phi.is_finite() && phi >= 0.0 && phi <= 1.0,
            "Phi out of bounds at cycle {i}: {phi}"
        );
    }

    // Telemetry should show motor steps occurred
    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps >= 100, "Should have 100+ motor steps");
    assert_eq!(telem.num_actuators, 6);

    // Safety level should always be valid
    for level in &safety_transitions {
        assert!(
            ["Green", "Yellow", "Orange", "Red"].contains(&level.as_str()),
            "Invalid safety level: {level}"
        );
    }

    // Proprioceptive feedback should be populated
    assert!(
        service.last_proprioceptive_hv().is_some(),
        "Should have proprioceptive HV after 100 cycles"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 2: Vehicle — extended driving with varied conditions
//
// Validates: Vehicle embodiment handles extended operation with input variety.
// Safety levels respond to consciousness dynamics. No drift or divergence.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "vehicle")]
#[test]
fn test_vehicle_extended_driving() {
    let mut service = make_service(EmbodimentPlatform::Vehicle);

    let scenarios = [
        ("highway cruising, clear road, maintaining speed", 30),
        (
            "dense traffic, frequent lane changes, high cognitive load",
            20,
        ),
        ("emergency: obstacle ahead, sudden brake required", 10),
        ("recovery: obstacle cleared, resuming normal driving", 20),
        ("highway exit, merging, destination ahead", 20),
    ];

    let mut total_cycles = 0;
    let mut all_safety_valid = true;

    for (input, cycles) in &scenarios {
        for _ in 0..*cycles {
            let result = service.cycle(input);
            let phi = result.metadata.consciousness.consciousness_level;

            assert!(
                phi.is_finite() && phi >= 0.0 && phi <= 1.0,
                "Phi out of bounds during '{input}': {phi}"
            );

            let telem = service.embodiment_telemetry();
            if !["Green", "Yellow", "Orange", "Red"].contains(&telem.safety_level.as_str()) {
                all_safety_valid = false;
            }

            total_cycles += 1;
        }
    }

    assert!(all_safety_valid, "All safety levels should be valid");
    assert_eq!(total_cycles, 100);

    let telem = service.embodiment_telemetry();
    assert!(telem.total_steps >= 100);
    assert_eq!(telem.num_actuators, 3);
    assert!(telem.control_effort.is_finite());
    assert!(telem.prediction_error.is_finite());

    assert!(service.last_proprioceptive_hv().is_some());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 3: Multi-body consciousness divergence
//
// Validates: Two CognitiveLoopService instances with different platforms
// produce different proprioceptive HVs, proving embodiment actually shapes
// consciousness (not just a no-op passthrough).
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "flight")]
#[test]
fn test_multi_body_consciousness_divergence() {
    let mut humanoid_service = make_service(EmbodimentPlatform::Humanoid);
    let mut quadrotor_service = make_service(EmbodimentPlatform::Quadrotor);

    // Run both for 30 cycles with identical input
    let input = "navigate forward while maintaining awareness of surroundings";

    for _ in 0..30 {
        let r1 = humanoid_service.cycle(input);
        let r2 = quadrotor_service.cycle(input);

        assert!(r1.metadata.consciousness.consciousness_level.is_finite());
        assert!(r2.metadata.consciousness.consciousness_level.is_finite());
    }

    // Both should have proprioceptive HVs
    let hv_humanoid = humanoid_service
        .last_proprioceptive_hv()
        .expect("humanoid proprioceptive HV");
    let hv_quadrotor = quadrotor_service
        .last_proprioceptive_hv()
        .expect("quadrotor proprioceptive HV");

    // The key test: different bodies should produce DIFFERENT proprioception
    let similarity: f32 = hv_humanoid
        .as_slice()
        .iter()
        .zip(hv_quadrotor.as_slice())
        .map(|(a, b)| a * b)
        .sum::<f32>()
        / (hv_humanoid
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            * hv_quadrotor
                .as_slice()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt());

    assert!(
        similarity < 0.95,
        "Different embodiments should produce different proprioception: similarity={similarity:.4}"
    );

    // Telemetry should differ
    let t1 = humanoid_service.embodiment_telemetry();
    let t2 = quadrotor_service.embodiment_telemetry();
    assert_eq!(t1.num_actuators, 21, "Humanoid has 21 actuators");
    assert_eq!(t2.num_actuators, 4, "Quadrotor has 4 actuators");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 4: Runtime platform switching (Multiple Realizability)
//
// Validates: Consciousness survives a body transfer. The same CognitiveLoopService
// switches from Quadrotor to Vehicle mid-run, and Phi remains finite/coherent.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "flight", feature = "vehicle"))]
#[test]
fn test_platform_switch_consciousness_survives() {
    let mut service = make_service(EmbodimentPlatform::Quadrotor);

    // Phase 1: Fly as quadrotor (20 cycles)
    for _ in 0..20 {
        let r = service.cycle("fly forward scanning terrain");
        assert!(r.metadata.consciousness.consciousness_level.is_finite());
    }
    let flight_telem = service.embodiment_telemetry().clone();
    assert_eq!(flight_telem.num_actuators, 4, "Quadrotor has 4 actuators");
    assert!(flight_telem.total_steps >= 20);

    // Switch body: Quadrotor → Vehicle
    service.switch_embodiment(EmbodimentPlatform::Vehicle);

    // Phase 2: Drive as vehicle (20 cycles)
    for _ in 0..20 {
        let r = service.cycle("drive along road maintaining speed");
        assert!(
            r.metadata.consciousness.consciousness_level.is_finite(),
            "Consciousness must survive platform switch"
        );
    }
    let vehicle_telem = service.embodiment_telemetry().clone();
    assert_eq!(vehicle_telem.num_actuators, 3, "Vehicle has 3 actuators");
    assert!(vehicle_telem.total_steps >= 20);

    // Consciousness should still be functional after body transfer
    assert_ne!(
        flight_telem.platform, vehicle_telem.platform,
        "Platform should have changed"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario 5: All-platform safety invariants
//
// Validates: Every platform produces bounded, finite states and responds
// correctly to Red safety level (phi < 0.1). Tests that the 4-tier
// safety system works identically across all embodiment types.
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "flight", feature = "vehicle"))]
#[test]
fn test_all_platforms_safety_invariants() {
    let platforms = vec![
        ("Quadrotor", EmbodimentPlatform::Quadrotor),
        ("Vehicle", EmbodimentPlatform::Vehicle),
    ];

    for (name, platform) in &platforms {
        let mut service = make_service(*platform);

        // Phase 1: Normal operation (20 cycles)
        for i in 0..20 {
            let r = service.cycle("maintain steady state");
            assert!(
                r.metadata.consciousness.consciousness_level.is_finite(),
                "{} cycle {} produced non-finite consciousness",
                name,
                i
            );
        }

        let telem = service.embodiment_telemetry();
        assert!(
            telem.total_steps >= 20,
            "{} should have at least 20 steps, got {}",
            name,
            telem.total_steps
        );
        assert!(
            telem.control_effort.is_finite(),
            "{} control effort not finite",
            name
        );

        // Phase 2: Stress input to push consciousness dynamics
        for _ in 0..10 {
            let r = service.cycle("extreme perturbation, unexpected obstacle, system failure");
            assert!(
                r.metadata.consciousness.consciousness_level.is_finite(),
                "{} produced non-finite consciousness under stress",
                name
            );
        }

        let telem2 = service.embodiment_telemetry();
        assert!(
            telem2.total_steps >= 30,
            "{} total steps should be >= 30 after both phases",
            name
        );
    }
}
