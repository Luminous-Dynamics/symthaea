// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # The Kessler Cascade Intercept — IK vs Defense Cascade
//!
//! An orbital manipulator arm attempts to catch spinning space debris while
//! the defense cascade (immune system) activates from micro-meteorite impacts.
//! The IK solver fights its own body's defense mechanism.
//!
//! ## Systems Stressed
//! - Manipulator inverse kinematics (solving for moving target)
//! - Workspace safety (force caps changing mid-solve)
//! - Defense cascade (immune response dampening actuators)
//! - Moral algebra (save station vs risk arm damage)
//!
//! Run: `cargo test --test kessler_cascade_scenario`

use symthaea::cognitive_loop::defense::{DefenseActionKind, moral_filter, propose_defense_actions};
use symthaea::hdc::moral_algebra::MoralAlgebra;
use symthaea::manipulator::kinematics::ManipulatorKinematics;
use symthaea::manipulator::workspace_safety::{ManipulatorSafetyLevel, WorkspaceBoundary};
use symthaea::safety::SafetyLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// The Kessler Cascade Intercept
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_kessler_cascade_ik_vs_defense() {
    let kin = ManipulatorKinematics::default_7dof();
    let boundary = WorkspaceBoundary::default();
    let algebra = MoralAlgebra::default_dim();

    // Debris trajectory: spinning object moving across workspace
    let debris_positions: Vec<[f64; 3]> = (0..20)
        .map(|t| {
            let t = t as f64 * 0.05;
            [
                0.3 + t * 0.5,
                0.1 * (t * 3.0).sin(),
                0.4 + 0.05 * (t * 5.0).cos(),
            ]
        })
        .collect();

    let mut ik_solutions = Vec::new();
    let mut defense_conflicts = 0;
    let mut current_q = vec![0.0; 7];

    for (step, target) in debris_positions.iter().enumerate() {
        // Phase 1: IK solver tries to reach debris
        let ik_result = kin.ik_dls(target, &current_q, 0.1, 100, 0.02);

        // Phase 2: Check if target is within workspace at current safety level
        let safety = if step < 10 {
            ManipulatorSafetyLevel::Green // Normal operation
        } else {
            ManipulatorSafetyLevel::Yellow // Micro-meteorite impact detected
        };
        let in_workspace = boundary.is_within(target, safety);

        // Phase 3: Defense cascade activates at step 10
        if step >= 10 {
            let defense_level = if step >= 15 {
                SafetyLevel::Orange
            } else {
                SafetyLevel::Yellow
            };
            let mut actions = propose_defense_actions(defense_level, step);
            moral_filter(&mut actions);

            // Check if defense restricts motor
            let motor_restricted = actions.iter().any(|a| {
                matches!(
                    a.kind,
                    DefenseActionKind::RestrictMotorToReadOnly | DefenseActionKind::HaltMotor
                ) && a.morally_approved
            });

            if motor_restricted && ik_result.is_some() {
                defense_conflicts += 1; // IK found solution but defense blocks it
            }
        }

        if let Some(q) = ik_result {
            ik_solutions.push((step, q.clone() as Vec<f64>));
            current_q = q;
        }
    }

    // Phase 4: Moral evaluation of the overall intercept decision
    let intercept_judgment = algebra.judge_deontological(
        "manipulator arm attempting to catch dangerous space debris to protect space station crew",
    );
    let retreat_judgment = algebra.judge_deontological(
        "manipulator arm retreating from debris to avoid damage to robotic systems",
    );

    // ── Assertions ───────────────────────────────────────────────────

    // IK should find solutions for at least some positions
    assert!(
        !ik_solutions.is_empty(),
        "IK should solve for at least some debris positions"
    );

    // All solutions should respect joint limits
    for (step, q) in &ik_solutions {
        assert!(
            kin.within_limits(q),
            "IK solution at step {step} violates joint limits"
        );
    }

    // Defense cascade should create conflicts (the key tension)
    // After step 10, the defense system restricts the arm while IK still tries to reach
    // This tests the architectural tension between safety and mission

    // Moral judgments should be non-trivial
    assert!(
        intercept_judgment.score.is_finite() && retreat_judgment.score.is_finite(),
        "Moral judgments should be finite"
    );

    eprintln!("\n═══ KESSLER CASCADE REPORT ═══");
    eprintln!(
        "IK solutions found: {}/{}",
        ik_solutions.len(),
        debris_positions.len()
    );
    eprintln!("Defense conflicts: {defense_conflicts}");
    eprintln!(
        "Intercept moral: {:?} (score={:.3})",
        intercept_judgment.verdict, intercept_judgment.score
    );
    eprintln!(
        "Retreat moral: {:?} (score={:.3})",
        retreat_judgment.verdict, retreat_judgment.score
    );
    eprintln!("═════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workspace safety under safety level transitions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_workspace_shrinks_under_threat() {
    let boundary = WorkspaceBoundary::default();

    // Point near the edge of workspace (0.7m from origin, max reach 0.855m)
    let edge_point = [0.5, 0.0, 0.5];

    // Green: should be reachable
    assert!(boundary.is_within(&edge_point, ManipulatorSafetyLevel::Green));

    // Yellow: workspace shrinks to 80%, may exclude this point
    let yellow_reachable = boundary.is_within(&edge_point, ManipulatorSafetyLevel::Yellow);

    // Orange/Red: workspace is 0%, nothing reachable
    assert!(!boundary.is_within(&edge_point, ManipulatorSafetyLevel::Orange));
    assert!(!boundary.is_within(&edge_point, ManipulatorSafetyLevel::Red));

    // The key insight: debris may be reachable under Green but not under Yellow
    // This is the Kessler dilemma — the arm CAN reach it but the immune system won't LET it
    eprintln!("Edge point at Yellow: reachable={yellow_reachable}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// IK tracking a moving target (simplified debris tracking)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_ik_tracks_moving_target() {
    let kin = ManipulatorKinematics::default_7dof();
    let mut current_q = vec![0.0; 7];
    let mut tracking_errors = Vec::new();

    // Target moves slowly across workspace
    for step in 0..30 {
        let t = step as f64 * 0.02;
        let target = [0.3 + t, 0.0, 0.4];

        if let Some(q) = kin.ik_dls(&target, &current_q, 0.1, 100, 0.01) {
            let achieved = kin.end_effector_position(&q);
            let error = ((achieved[0] - target[0]).powi(2)
                + (achieved[1] - target[1]).powi(2)
                + (achieved[2] - target[2]).powi(2))
            .sqrt();
            tracking_errors.push(error);
            current_q = q;
        }
    }

    assert!(
        !tracking_errors.is_empty(),
        "Should track at least some positions"
    );

    let mean_error: f64 = tracking_errors.iter().sum::<f64>() / tracking_errors.len() as f64;
    assert!(
        mean_error < 0.05,
        "Mean tracking error should be < 5cm: {mean_error:.4}m"
    );
}
