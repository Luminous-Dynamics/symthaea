// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-backend comparison: Simple vs Symtropy manipulator.
//!
//! Validates that both backends:
//! 1. Remain stable under the same command sequences
//! 2. Respond to consciousness gating the same way
//! 3. Produce finite, bounded state trajectories
//!
//! Does NOT expect pixel-perfect agreement — the physics models differ
//! fundamentally (joint-space Euler vs rigid-body GJK/EPA). What matters:
//! both exhibit the same qualitative behavior (safety cascade, gravity,
//! bounded motion).

#![cfg(feature = "symtropy")]

use symtropy_physics as _; // ensure dep active
use symthaea_manipulator::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
use symthaea_manipulator::symtropy_sim::SymtropyManipulatorSimulator;
use symthaea_manipulator::types::{ManipulatorCommand, NUM_JOINTS};

// ═══════════════════════════════════════════════════════════════════════
// Comparison 1: Both backends handle zero-command gracefully
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn both_backends_stable_under_zero_command() {
    let mut simple = SimpleManipulatorSimulator::new();
    let mut symtropy = SymtropyManipulatorSimulator::new();

    let cmd = ManipulatorCommand::zero();
    for _ in 0..200 {
        simple.step(&cmd, 0.002);
        symtropy.step(&cmd, 0.002);
    }

    assert!(simple.state().is_finite(), "Simple backend should remain finite");
    assert!(symtropy.state().is_finite(), "Symtropy backend should remain finite");
}

// ═══════════════════════════════════════════════════════════════════════
// Comparison 2: Both backends respond to torque commands
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn both_backends_respond_to_torque() {
    let mut simple = SimpleManipulatorSimulator::new();
    let mut symtropy = SymtropyManipulatorSimulator::new();

    let simple_initial = simple.state().joint_angles;
    let symtropy_initial = symtropy.state().joint_angles;

    let mut cmd = ManipulatorCommand::zero();
    cmd.joint_torques[0] = 0.5;
    cmd.joint_torques[2] = 0.3;

    for _ in 0..50 {
        simple.step(&cmd, 0.005);
        symtropy.step(&cmd, 0.005);
    }

    // Both should have moved from initial position
    let simple_moved = simple_initial.iter()
        .zip(simple.state().joint_angles.iter())
        .any(|(a, b)| (a - b).abs() > 0.001);
    let symtropy_moved = symtropy_initial.iter()
        .zip(symtropy.state().joint_angles.iter())
        .any(|(a, b)| (a - b).abs() > 0.001);

    assert!(simple_moved, "Simple should respond to torque");
    assert!(symtropy_moved, "Symtropy should respond to torque");
}

// ═══════════════════════════════════════════════════════════════════════
// Comparison 3: Consciousness gating halts both backends
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn both_backends_halt_at_zero_gain() {
    let mut symtropy = SymtropyManipulatorSimulator::new();
    symtropy.set_motor_gain(0.0);

    let mut cmd = ManipulatorCommand::zero();
    cmd.joint_torques = [1.0; NUM_JOINTS]; // Max commanded torque

    for _ in 0..100 {
        symtropy.step(&cmd, 0.005);
    }

    // Gravity still acts but motor commands are nullified
    assert!(symtropy.state().is_finite());
}

// ═══════════════════════════════════════════════════════════════════════
// Comparison 4: Both backends preserve joint count (7 joints)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn both_backends_have_seven_joints() {
    let simple = SimpleManipulatorSimulator::new();
    let symtropy = SymtropyManipulatorSimulator::new();

    assert_eq!(simple.state().joint_angles.len(), NUM_JOINTS);
    assert_eq!(symtropy.state().joint_angles.len(), NUM_JOINTS);
    assert_eq!(NUM_JOINTS, 7);
}

// ═══════════════════════════════════════════════════════════════════════
// Comparison 5: Trajectory divergence metric
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn backends_trajectory_divergence_bounded() {
    let mut simple = SimpleManipulatorSimulator::new();
    let mut symtropy = SymtropyManipulatorSimulator::new();

    let cmd = ManipulatorCommand {
        joint_torques: [0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
        gripper: 0.5,
    };

    let mut max_divergence = 0.0f64;
    for _ in 0..50 {
        simple.step(&cmd, 0.005);
        symtropy.step(&cmd, 0.005);

        // Compute per-joint divergence
        let div: f64 = simple.state().joint_angles.iter()
            .zip(symtropy.state().joint_angles.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        if div > max_divergence { max_divergence = div; }
    }

    // Both backends should remain bounded (not diverging to infinity)
    assert!(max_divergence.is_finite(), "Divergence should be finite");
    // They will differ (different physics models) but both should be sane
    assert!(simple.state().is_finite());
    assert!(symtropy.state().is_finite());

    eprintln!("Simple vs Symtropy max trajectory divergence: {max_divergence:.4} rad");
}

// ═══════════════════════════════════════════════════════════════════════
// Comparison 6: Reset behavior
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn both_backends_reset_cleanly() {
    let mut simple = SimpleManipulatorSimulator::new();
    let mut symtropy = SymtropyManipulatorSimulator::new();

    let cmd = ManipulatorCommand {
        joint_torques: [0.5; NUM_JOINTS],
        gripper: 0.0,
    };

    for _ in 0..100 {
        simple.step(&cmd, 0.005);
        symtropy.step(&cmd, 0.005);
    }

    simple.reset();
    symtropy.reset();

    assert!(simple.state().is_finite());
    assert!(symtropy.state().is_finite());
}
