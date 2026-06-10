// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Extended tests for manipulator trajectory planning, IK edge cases,
//! workspace safety interactions, and embodiment bridge stability.
//!
//! These complement the existing inline tests by covering:
//! - IK failure modes (unreachable targets)
//! - Multi-waypoint trajectory chaining
//! - Safety level transitions during operation
//! - Workspace clearance gradients
//! - Long-horizon stability
//! - Human proximity zone interactions with safety levels

use symthaea_manipulator::kinematics::ManipulatorKinematics;
use symthaea_manipulator::workspace_safety::{ManipulatorSafetyLevel, WorkspaceBoundary};
// ManipulatorState available if needed
use symthaea_manipulator::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};

// ═══════════════════════════════════════════════════════════════════════════════
// IK failure modes
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_ik_unreachable_target_returns_none() {
    let kin = ManipulatorKinematics::default_7dof();
    // Target well beyond max reach (0.855m)
    let target = [5.0, 5.0, 5.0];
    let q0 = vec![0.0; 7];
    let result = kin.ik_dls(&target, &q0, 0.1, 200, 0.01);
    assert!(
        result.is_none(),
        "IK should return None for unreachable target"
    );
}

#[test]
fn test_ik_target_below_floor() {
    let kin = ManipulatorKinematics::default_7dof();
    // Target at negative z (below floor)
    let target = [0.3, 0.0, -0.5];
    let q0 = vec![0.0; 7];
    let result = kin.ik_dls(&target, &q0, 0.1, 200, 0.01);
    // May or may not converge (joint limits might prevent it)
    // but if it does, the solution should still respect joint limits
    if let Some(q) = result {
        assert!(
            kin.within_limits(&q),
            "Solution must respect joint limits even for low target"
        );
    }
}

#[test]
fn test_ik_target_at_origin() {
    let kin = ManipulatorKinematics::default_7dof();
    // Target at the arm base — likely unreachable due to link offsets
    let target = [0.0, 0.0, 0.0];
    let q0 = vec![0.0; 7];
    let result = kin.ik_dls(&target, &q0, 0.1, 200, 0.01);
    // Either fails or produces a clamped solution
    if let Some(q) = result {
        assert!(kin.within_limits(&q));
    }
}

#[test]
fn test_ik_zero_tolerance_never_converges() {
    let kin = ManipulatorKinematics::default_7dof();
    let q_target = vec![0.1, 0.3, -0.2, -1.5, 0.1, 0.5, 0.0];
    let target_pos = kin.end_effector_position(&q_target);
    let q0 = vec![0.0; 7];
    // Zero tolerance — practically impossible to achieve
    let result = kin.ik_dls(&target_pos, &q0, 0.1, 10, 0.0);
    // Should fail with very few iterations and zero tolerance
    // (unless it gets astronomically lucky)
    // This tests that the algorithm doesn't panic on zero tolerance
    let _ = result; // Just check it doesn't crash
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-waypoint trajectory chaining
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_waypoint_trajectory() {
    let kin = ManipulatorKinematics::default_7dof();

    // Define waypoints as known reachable positions (from FK)
    let configs = [
        vec![0.0, 0.3, 0.0, -1.5, 0.0, 0.5, 0.0],
        vec![0.3, 0.5, -0.2, -1.0, 0.1, 0.8, 0.0],
        vec![-0.2, 0.4, 0.1, -1.2, -0.1, 0.6, 0.0],
        vec![0.0, 0.3, 0.0, -1.5, 0.0, 0.5, 0.0], // Return to start
    ];

    let waypoints: Vec<[f64; 3]> = configs
        .iter()
        .map(|q| kin.end_effector_position(q))
        .collect();

    // Chain IK solutions: each waypoint uses previous solution as starting config
    let mut current_q = vec![0.0; 7];
    let mut all_solutions = Vec::new();

    for (i, target) in waypoints.iter().enumerate() {
        let result = kin.ik_dls(target, &current_q, 0.1, 200, 0.01);
        assert!(
            result.is_some(),
            "IK should converge for waypoint {i}: target={target:?}"
        );
        let q = result.unwrap();

        // Verify solution accuracy
        let achieved = kin.end_effector_position(&q);
        let error = ((achieved[0] - target[0]).powi(2)
            + (achieved[1] - target[1]).powi(2)
            + (achieved[2] - target[2]).powi(2))
        .sqrt();
        assert!(error < 0.02, "Waypoint {i} error too large: {error:.4}");

        // Joint limits respected
        assert!(kin.within_limits(&q), "Waypoint {i} violates joint limits");

        all_solutions.push(q.clone());
        current_q = q;
    }

    assert_eq!(all_solutions.len(), 4);
}

#[test]
fn test_trajectory_continuity() {
    let kin = ManipulatorKinematics::default_7dof();

    // Two nearby targets
    let q1 = vec![0.1, 0.3, 0.0, -1.5, 0.0, 0.5, 0.0];
    let q2 = vec![0.15, 0.35, 0.0, -1.45, 0.0, 0.55, 0.0];
    let pos1 = kin.end_effector_position(&q1);
    let pos2 = kin.end_effector_position(&q2);

    // Solve IK for both, starting from same initial
    let q0 = vec![0.0; 7];
    let sol1 = kin.ik_dls(&pos1, &q0, 0.1, 200, 0.01).unwrap();
    let sol2 = kin.ik_dls(&pos2, &sol1, 0.1, 200, 0.01).unwrap();

    // Solutions for nearby targets (using warm-start) should be nearby in joint space
    let joint_dist: f64 = sol1
        .iter()
        .zip(&sol2)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        joint_dist < 1.0,
        "Nearby targets should produce nearby joint solutions: dist={joint_dist:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workspace clearance gradients
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_clearance_decreases_toward_boundary() {
    let boundary = WorkspaceBoundary::default();
    let safety = ManipulatorSafetyLevel::Green;

    // Points at increasing radius
    let clearances: Vec<f64> = (1..=8)
        .map(|i| {
            let r = i as f64 * 0.1;
            boundary.clearance(&[r, 0.0, 0.3], safety)
        })
        .collect();

    // Clearance should monotonically decrease as we approach the boundary
    for i in 1..clearances.len() {
        assert!(
            clearances[i] <= clearances[i - 1] + 1e-10,
            "Clearance should decrease toward boundary: c[{}]={:.4} > c[{}]={:.4}",
            i,
            clearances[i],
            i - 1,
            clearances[i - 1]
        );
    }
}

#[test]
fn test_clearance_negative_outside_boundary() {
    let boundary = WorkspaceBoundary::default();

    // Well outside workspace
    let c = boundary.clearance(&[2.0, 0.0, 0.3], ManipulatorSafetyLevel::Green);
    assert!(
        c < 0.0,
        "Clearance outside workspace should be negative: {c}"
    );
}

#[test]
fn test_clearance_with_human_zone() {
    let boundary = WorkspaceBoundary {
        human_zones: vec![[0.4, 0.0, 0.3, 0.15]], // Human at (0.4, 0, 0.3) radius 15cm
        ..Default::default()
    };

    // Near human
    let near = boundary.clearance(&[0.3, 0.0, 0.3], ManipulatorSafetyLevel::Green);
    // Far from human
    let far = boundary.clearance(&[0.0, 0.5, 0.3], ManipulatorSafetyLevel::Green);

    assert!(
        near < far,
        "Clearance near human should be less than far: near={near:.4}, far={far:.4}"
    );
}

#[test]
fn test_clearance_min_height_contribution() {
    let boundary = WorkspaceBoundary::default();
    let safety = ManipulatorSafetyLevel::Green;

    // Just above min height (0.05m)
    let low = boundary.clearance(&[0.3, 0.0, 0.06], safety);
    let high = boundary.clearance(&[0.3, 0.0, 0.3], safety);

    assert!(
        low < high,
        "Low position should have less clearance: low={low:.4}, high={high:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Safety level interactions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_safety_level_torque_gain_ordering() {
    let gains: Vec<f32> = [
        ManipulatorSafetyLevel::Green,
        ManipulatorSafetyLevel::Yellow,
        ManipulatorSafetyLevel::Orange,
        ManipulatorSafetyLevel::Red,
    ]
    .iter()
    .map(|l| l.torque_gain())
    .collect();

    // Gains should be monotonically decreasing
    for i in 1..gains.len() {
        assert!(
            gains[i] <= gains[i - 1],
            "Torque gain should decrease with safety level: {:?}={} > {:?}={}",
            i,
            gains[i],
            i - 1,
            gains[i - 1]
        );
    }
    assert_eq!(gains[3], 0.0, "Red should have zero gain");
}

#[test]
fn test_workspace_fraction_ordering() {
    let fractions: Vec<f64> = [
        ManipulatorSafetyLevel::Green,
        ManipulatorSafetyLevel::Yellow,
        ManipulatorSafetyLevel::Orange,
        ManipulatorSafetyLevel::Red,
    ]
    .iter()
    .map(|l| l.workspace_fraction())
    .collect();

    for i in 1..fractions.len() {
        assert!(
            fractions[i] <= fractions[i - 1],
            "Workspace fraction should decrease: {:?}",
            fractions
        );
    }
}

#[test]
fn test_position_inside_green_but_outside_yellow() {
    let boundary = WorkspaceBoundary::default();
    // Position near max reach (0.855m): inside Green (100%) but outside Yellow (80% = 0.684m)
    let pos: [f64; 3] = [0.7, 0.0, 0.3];
    let radius = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();

    assert!(
        boundary.is_within(&pos, ManipulatorSafetyLevel::Green),
        "Should be inside Green workspace"
    );
    assert!(
        !boundary.is_within(&pos, ManipulatorSafetyLevel::Yellow),
        "Should be outside Yellow workspace (80%)"
    );
}

#[test]
fn test_orange_red_workspace_zero() {
    let boundary = WorkspaceBoundary::default();
    // Any position should be outside Orange/Red workspace (fraction = 0)
    let center = [0.0, 0.0, 0.3];
    assert!(!boundary.is_within(&center, ManipulatorSafetyLevel::Orange));
    assert!(!boundary.is_within(&center, ManipulatorSafetyLevel::Red));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multiple human proximity zones
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_multiple_human_zones() {
    let boundary = WorkspaceBoundary {
        human_zones: vec![
            [0.3, 0.0, 0.3, 0.15],  // Human 1
            [-0.3, 0.0, 0.3, 0.15], // Human 2 (opposite side)
        ],
        ..Default::default()
    };

    // Near human 1
    assert!(!boundary.is_within(&[0.25, 0.0, 0.3], ManipulatorSafetyLevel::Green));
    // Near human 2
    assert!(!boundary.is_within(&[-0.25, 0.0, 0.3], ManipulatorSafetyLevel::Green));
    // Between both humans (should be safe)
    assert!(boundary.is_within(&[0.0, 0.0, 0.3], ManipulatorSafetyLevel::Green));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Simulator stability
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_simulator_long_horizon_stability() {
    let mut sim = SimpleManipulatorSimulator::new();
    let cmd = symthaea_manipulator::types::ManipulatorCommand::zero();

    // Run 1000 steps with zero command — should not diverge
    for i in 0..1000 {
        sim.step(&cmd, 0.001);
        let state = sim.state();
        assert!(state.is_finite(), "State should remain finite at step {i}");
    }
}

#[test]
fn test_simulator_max_torque_stability() {
    let mut sim = SimpleManipulatorSimulator::new();
    let cmd = symthaea_manipulator::types::ManipulatorCommand {
        joint_torques: [87.0; 7], // Max Panda torque
        gripper: 1.0,
    };

    // Max torque for 100 steps should not produce NaN
    for i in 0..100 {
        sim.step(&cmd, 0.001);
        let state = sim.state();
        assert!(
            state.is_finite(),
            "State should remain finite under max torque at step {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FK/IK consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fk_ik_roundtrip_multiple_configs() {
    let kin = ManipulatorKinematics::default_7dof();

    let configs = [
        vec![0.0, 0.3, 0.0, -1.5, 0.0, 0.5, 0.0],
        vec![0.5, 0.2, -0.3, -1.0, 0.2, 0.8, -0.1],
        vec![-0.3, 0.5, 0.1, -2.0, -0.2, 0.3, 0.2],
    ];

    for (idx, q_target) in configs.iter().enumerate() {
        let target_pos = kin.end_effector_position(q_target);

        // Different starting point to test convergence robustness
        let q0 = vec![0.0; 7];
        let result = kin.ik_dls(&target_pos, &q0, 0.1, 300, 0.005);

        assert!(result.is_some(), "FK-IK roundtrip failed for config {idx}");
        let q_solution = result.unwrap();
        let achieved = kin.end_effector_position(&q_solution);

        let error = ((achieved[0] - target_pos[0]).powi(2)
            + (achieved[1] - target_pos[1]).powi(2)
            + (achieved[2] - target_pos[2]).powi(2))
        .sqrt();

        assert!(
            error < 0.01,
            "Config {idx} FK-IK roundtrip error: {error:.4}"
        );
    }
}

#[test]
fn test_num_joints() {
    let kin = ManipulatorKinematics::default_7dof();
    assert_eq!(kin.num_joints(), 7);
}

#[test]
fn test_within_limits() {
    let kin = ManipulatorKinematics::default_7dof();
    let valid = vec![0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
    assert!(kin.within_limits(&valid));

    let invalid = vec![10.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]; // J1 out of range
    assert!(!kin.within_limits(&invalid));
}
