// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: drive a real Symthaea `OrbitalEmbodiment` through the
//! adapter. Orbital is the smallest platform that implements
//! [`EmbodimentBridge`] (4 unit tests, 208 LOC in symthaea-orbital), so it's
//! a cheap way to prove the adapter works end-to-end — not just against a
//! mock.

use symthaea_core::embodiment::MotorSafetyLevel;
use symthaea_core::genesis::GenesisSeed;
use symthaea_orbital::embodiment::OrbitalEmbodiment;
use symtropy_robotics_bridge_core::MotorPlanner;
use symtropy_symthaea_adapter::EmbodimentPlanner;

/// Orbital's `NUM_ACTUATORS` constant is crate-private; mirror it here so
/// the test fails loudly if the count ever changes.
const ORBITAL_ACTUATORS: usize = 7;

#[test]
fn adapter_drives_real_orbital_bridge() {
    let orbital = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("integration-test"));
    let mut planner = EmbodimentPlanner::new(Box::new(orbital));

    // Observation is a small state vector; the adapter tiles it into the
    // 16,384D thought HV. Green-tier phi gain.
    let observation = [0.1_f64, 0.2, 0.3, 0.4];
    let cmds = planner.plan(&observation, 1.0, ORBITAL_ACTUATORS);

    assert_eq!(cmds.len(), ORBITAL_ACTUATORS);
    assert_eq!(planner.bridge().total_steps(), 1);
    // Orbital starts in Green tier at phi=1.0.
    assert_eq!(planner.bridge().safety_level(), MotorSafetyLevel::Green);
    for c in &cmds {
        assert!(c.is_finite(), "orbital produced non-finite command: {c}");
        assert!(*c >= 0.0 && *c <= 1.0, "out of range: {c}");
    }
}

#[test]
fn adapter_accumulates_bridge_steps_across_ticks() {
    let orbital = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("integration-test"));
    let mut planner = EmbodimentPlanner::new(Box::new(orbital));

    for _ in 0..10 {
        let _ = planner.plan(&[0.1, 0.2, 0.3, 0.4], 1.0, ORBITAL_ACTUATORS);
    }
    assert_eq!(planner.bridge().total_steps(), 10);
}

#[test]
fn low_phi_gain_parks_real_bridge() {
    let orbital = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("integration-test"));
    let mut planner = EmbodimentPlanner::new(Box::new(orbital));

    // phi=0.15 → Orange tier. Orbital internally zeros out torques at
    // gain <= 0.3 and the adapter also caps output at the caller's gain.
    let cmds = planner.plan(&[0.1, 0.2, 0.3, 0.4], 0.15, ORBITAL_ACTUATORS);
    assert_eq!(planner.bridge().safety_level(), MotorSafetyLevel::Orange);
    for c in &cmds {
        assert!(*c <= 0.15 + 1e-9, "gain cap violated under Orange: {c}");
    }
}

#[test]
fn moral_gate_propagates_to_real_bridge() {
    use symthaea_core::embodiment::MoralGateInput;

    let orbital = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("integration-test"));
    let mut planner = EmbodimentPlanner::new(Box::new(orbital));

    planner.bridge_mut().apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_SAFE,
        consent_violation: false,
        ahimsa_violated: true,
    });

    // Ahimsa violation forces Red regardless of the incoming phi gain.
    // Red → motor_gain 0.0 → adapter emits all zeros.
    let cmds = planner.plan(&[0.1, 0.2, 0.3, 0.4], 1.0, ORBITAL_ACTUATORS);
    assert_eq!(planner.bridge().safety_level(), MotorSafetyLevel::Red);
    for c in &cmds {
        assert_eq!(*c, 0.0, "ahimsa violation should zero all commands");
    }
}

#[test]
fn reset_via_adapter_clears_bridge_state() {
    let orbital = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("integration-test"));
    let mut planner = EmbodimentPlanner::new(Box::new(orbital));

    for _ in 0..5 {
        let _ = planner.plan(&[0.1, 0.2, 0.3, 0.4], 1.0, ORBITAL_ACTUATORS);
    }
    assert_eq!(planner.bridge().total_steps(), 5);

    planner.bridge_mut().reset();
    assert_eq!(planner.bridge().total_steps(), 0);
}
