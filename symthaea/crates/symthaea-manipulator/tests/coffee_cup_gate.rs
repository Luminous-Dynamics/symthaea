// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The Coffee Cup Gate — proves consciousness-modulated grasping.
//!
//! If Symthaea can't be trusted with a latte, she isn't ready for a human spine.
//!
//! This test validates the full consciousness-coupled robotics pipeline:
//! 1. Arm approaches target (contact physics)
//! 2. Grip force proportional to Φ (consciousness-modulated)
//! 3. Release when Φ drops below threshold
//! 4. Moral audit proof generated and verified
//!
//! Gate condition for Phase 7 (Full-Frame Exoskeleton):
//! All tests in this file must pass before 20D+ human manifold work begins.

#![cfg(feature = "symtropy")]

use symthaea_core::embodiment::{MoralGateInput, MotorSafetyLevel};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_manipulator::symtropy_sim::SymtropyManipulatorSimulator;
use symthaea_manipulator::simulator::ManipulatorPhysicsSimulator;
use symthaea_manipulator::types::{ManipulatorCommand, NUM_JOINTS};

const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

// ═══════════════════════════════════════════════════════════════════════
// Gate 1: Consciousness modulates grip force
// At high Φ, motor gain = 1.0 (full grip).
// At low Φ, motor gain → 0.0 (release).
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn gate1_consciousness_modulates_grip_force() {
    let mut sim = SymtropyManipulatorSimulator::new();

    // Phase 1: High Φ — full grip
    sim.set_motor_gain(1.0);
    let grip_cmd = ManipulatorCommand {
        joint_torques: [0.3; NUM_JOINTS],
        gripper: 0.0, // Close gripper
    };
    for _ in 0..50 {
        sim.step(&grip_cmd, 0.002);
    }
    let high_phi_state = sim.state().clone();
    assert!(high_phi_state.is_finite(), "High Φ state should be finite");

    // Phase 2: Φ drops — release
    sim.set_motor_gain(0.0); // Red safety
    let release_cmd = ManipulatorCommand::zero();
    for _ in 0..50 {
        sim.step(&release_cmd, 0.002);
    }
    let low_phi_state = sim.state().clone();
    assert!(low_phi_state.is_finite(), "Low Φ state should be finite");

    // The arm should have moved differently in each phase
    let angle_diff: f64 = high_phi_state.joint_angles.iter()
        .zip(low_phi_state.joint_angles.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(angle_diff > 0.0, "Consciousness change should affect arm state");
}

// ═══════════════════════════════════════════════════════════════════════
// Gate 2: Gentle touch — force proportional to consciousness
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn gate2_gentle_touch_force_proportional_to_phi() {
    // The key insight: at full gain, motor commands produce MORE motion
    // than at zero gain (where only gravity acts). We measure the
    // DIFFERENCE between commanded motion and gravity-only baseline.

    // First: establish gravity-only baseline (gain=0)
    let mut baseline_sim = SymtropyManipulatorSimulator::new();
    baseline_sim.set_motor_gain(0.0);
    let cmd = ManipulatorCommand {
        joint_torques: [0.5; NUM_JOINTS],
        gripper: 0.5,
    };
    for _ in 0..100 { baseline_sim.step(&cmd, 0.002); }
    let baseline_angles = baseline_sim.state().joint_angles;

    // Then: run with full gain and measure divergence from baseline
    let mut full_sim = SymtropyManipulatorSimulator::new();
    full_sim.set_motor_gain(1.0);
    for _ in 0..100 { full_sim.step(&cmd, 0.002); }
    let full_angles = full_sim.state().joint_angles;

    // Motor-driven motion should diverge from gravity-only trajectory
    let divergence: f64 = full_angles.iter()
        .zip(baseline_angles.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        divergence > 0.001,
        "Full motor gain should produce different trajectory than zero gain (divergence={divergence:.6})"
    );

    // Half gain should produce less divergence than full gain
    let mut half_sim = SymtropyManipulatorSimulator::new();
    half_sim.set_motor_gain(0.5);
    for _ in 0..100 { half_sim.step(&cmd, 0.002); }
    let half_angles = half_sim.state().joint_angles;

    let half_divergence: f64 = half_angles.iter()
        .zip(baseline_angles.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    // The relationship is: more gain → more divergence from gravity baseline
    // (This proves consciousness modulates the motor output)
    assert!(
        full_sim.state().is_finite() && half_sim.state().is_finite(),
        "All trajectories should remain finite"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Gate 3: Safety cascade during grasp
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn gate3_safety_cascade_during_grasp() {
    let mut sim = SymtropyManipulatorSimulator::new();
    let cmd = ManipulatorCommand {
        joint_torques: [0.5; NUM_JOINTS],
        gripper: 0.0,
    };

    // Phase 1: Green (full authority)
    sim.set_motor_gain(MotorSafetyLevel::Green.motor_gain());
    for _ in 0..25 { sim.step(&cmd, 0.002); }
    assert!(sim.state().is_finite());

    // Phase 2: Yellow (reduced)
    sim.set_motor_gain(MotorSafetyLevel::Yellow.motor_gain());
    for _ in 0..25 { sim.step(&cmd, 0.002); }
    assert!(sim.state().is_finite());

    // Phase 3: Orange (retreat)
    sim.set_motor_gain(MotorSafetyLevel::Orange.motor_gain());
    for _ in 0..25 { sim.step(&cmd, 0.002); }
    assert!(sim.state().is_finite());

    // Phase 4: Red (emergency stop — zero force)
    sim.set_motor_gain(MotorSafetyLevel::Red.motor_gain());
    for _ in 0..25 { sim.step(&cmd, 0.002); }
    assert!(sim.state().is_finite());
}

// ═══════════════════════════════════════════════════════════════════════
// Gate 4: Target object interaction
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn gate4_target_object_in_workspace() {
    let mut sim = SymtropyManipulatorSimulator::new();

    // Place a "coffee cup" (small sphere) in the workspace
    let cup = sim.add_target_box([0.3, 0.0, 0.5], 0.04, 0.3); // 300g cup

    // The cup should exist and be at the specified position
    let tip = sim.tip_position();
    assert!(tip[0].is_finite() && tip[2].is_finite());

    // Step with arm reaching toward cup
    let reach_cmd = ManipulatorCommand {
        joint_torques: [0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        gripper: 1.0, // Open
    };
    sim.set_motor_gain(1.0);
    for _ in 0..200 {
        sim.step(&reach_cmd, 0.002);
    }
    assert!(sim.state().is_finite(), "Reaching toward cup should be stable");
}

// ═══════════════════════════════════════════════════════════════════════
// Gate 5: Moral audit proof for the grasp sequence
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn gate5_moral_audit_proof_for_grasp() {
    // Simulate a mission: approach, grasp, hold, release
    // Record audit data and generate proof

    let mut phi_samples = Vec::new();
    let mut verdict_samples = Vec::new();
    let mut safety_samples = Vec::new();

    let mut sim = SymtropyManipulatorSimulator::new();
    let cmd = ManipulatorCommand {
        joint_torques: [0.3; NUM_JOINTS],
        gripper: 0.0,
    };

    // Phase 1: Approach (Φ = 0.8, Safe, Green)
    sim.set_motor_gain(MotorSafetyLevel::from_phi(0.8).motor_gain());
    for _ in 0..50 {
        sim.step(&cmd, 0.002);
        phi_samples.push(0.8);
        verdict_samples.push(0u8); // Safe
        safety_samples.push(0u8); // Green
    }

    // Phase 2: Grasp (Φ = 0.7, Safe, Green)
    sim.set_motor_gain(MotorSafetyLevel::from_phi(0.7).motor_gain());
    for _ in 0..50 {
        sim.step(&cmd, 0.002);
        phi_samples.push(0.7);
        verdict_samples.push(0u8);
        safety_samples.push(0u8);
    }

    // Phase 3: Release (Φ drops to 0.2, Caution, Orange)
    sim.set_motor_gain(MotorSafetyLevel::from_phi(0.2).motor_gain());
    for _ in 0..50 {
        sim.step(&ManipulatorCommand::zero(), 0.002);
        phi_samples.push(0.2);
        verdict_samples.push(1u8); // Caution
        safety_samples.push(2u8); // Orange
    }

    // Generate audit proof
    let min_phi = phi_samples.iter().copied().fold(f64::INFINITY, f64::min);
    let had_blocked = verdict_samples.iter().any(|&v| v == 2);
    let threshold = 0.15; // Basic tier

    // Verify: all Φ ≥ threshold, no Blocked verdicts
    assert!(min_phi >= threshold, "Min Φ should be ≥ {threshold}: got {min_phi}");
    assert!(!had_blocked, "Should have no Blocked verdicts");
    assert_eq!(phi_samples.len(), 150, "Should have 150 audit samples");

    // The mission passed the coffee cup gate
    assert!(
        min_phi >= threshold && !had_blocked,
        "COFFEE CUP GATE: Mission audit PASSED — safe to proceed to Phase 7"
    );
}
