// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for manipulator kinematics.
//!
//! Tests invariants of the forward/inverse kinematics system that
//! should hold for ALL valid joint configurations, not just the
//! handful of hand-picked cases in the unit tests.

use proptest::prelude::*;
use symthaea_manipulator::kinematics::ManipulatorKinematics;

// ============================================================================
// Strategy: valid joint angle vector (within limits)
// ============================================================================

fn arb_joint_angles() -> impl Strategy<Value = Vec<f64>> {
    let kin = ManipulatorKinematics::default_7dof();
    let limits = kin.joint_limits.clone();

    limits
        .into_iter()
        .map(|[lo, hi]| lo..=hi)
        .collect::<Vec<_>>()
        .prop_map(|ranges| ranges)
}

// Helper: generate joint angles within limits
fn arb_q() -> impl Strategy<Value = Vec<f64>> {
    // Panda joint limits (hardcoded to avoid borrowing issues)
    (
        -2.89..=2.89f64,  // J1
        -1.76..=1.76f64,  // J2
        -2.89..=2.89f64,  // J3
        -3.07..=-0.07f64, // J4 (elbow, negative range)
        -2.89..=2.89f64,  // J5
        -0.02..=3.75f64,  // J6
        -2.89..=2.89f64,  // J7
    )
        .prop_map(|(j1, j2, j3, j4, j5, j6, j7)| vec![j1, j2, j3, j4, j5, j6, j7])
}

// ============================================================================
// Property 1: Forward kinematics always produces finite positions
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn fk_always_finite(q in arb_q()) {
        let kin = ManipulatorKinematics::default_7dof();
        let pos = kin.end_effector_position(&q);

        prop_assert!(pos[0].is_finite(), "FK x should be finite: {:?}", pos);
        prop_assert!(pos[1].is_finite(), "FK y should be finite: {:?}", pos);
        prop_assert!(pos[2].is_finite(), "FK z should be finite: {:?}", pos);
    }
}

// ============================================================================
// Property 2: FK end-effector is within workspace bounds
// The Panda arm has ~0.855m reach. FK should never exceed this.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn fk_within_workspace(q in arb_q()) {
        let kin = ManipulatorKinematics::default_7dof();
        let pos = kin.end_effector_position(&q);
        let radius = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();

        // Max reach is sum of all link lengths (~1.1m for Panda)
        // Use generous bound to account for all configurations
        prop_assert!(
            radius < 1.5,
            "FK radius {radius:.4} exceeds workspace at q={q:?}"
        );
    }
}

// ============================================================================
// Property 3: Jacobian has correct shape and finite entries
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn jacobian_always_valid(q in arb_q()) {
        let kin = ManipulatorKinematics::default_7dof();
        let jac = kin.jacobian(&q, 1e-6);

        prop_assert_eq!(jac.len(), 6, "Jacobian should have 6 rows");
        for (i, row) in jac.iter().enumerate() {
            prop_assert!(row.len() == 7, "Row {} should have 7 columns", i);
            for (j, &val) in row.iter().enumerate() {
                prop_assert!(
                    val.is_finite(),
                    "Jacobian[{i}][{j}] should be finite: {val}"
                );
            }
        }
    }
}

// ============================================================================
// Property 4: IK solution (when it converges) respects joint limits
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn ik_solution_within_limits(q_target in arb_q()) {
        let kin = ManipulatorKinematics::default_7dof();
        let target_pos = kin.end_effector_position(&q_target);

        let q0 = vec![0.0; 7];
        let result = kin.ik_dls(&target_pos, &q0, 0.1, 200, 0.01);

        if let Some(q_solution) = result {
            prop_assert!(
                kin.within_limits(&q_solution),
                "IK solution violates limits: {q_solution:?}"
            );
        }
        // It's OK for IK to not converge — some targets are hard to reach
        // from zero config. The property is about the solution when it exists.
    }
}

// ============================================================================
// Property 5: FK-IK roundtrip accuracy (when IK converges)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn fk_ik_roundtrip(q_target in arb_q()) {
        let kin = ManipulatorKinematics::default_7dof();
        let target_pos = kin.end_effector_position(&q_target);

        // Use the target config as initial guess (warm start) for better convergence
        let q0 = q_target.iter().map(|v| v * 0.9).collect::<Vec<_>>();
        let result = kin.ik_dls(&target_pos, &q0, 0.1, 300, 0.005);

        if let Some(q_solution) = result {
            let achieved = kin.end_effector_position(&q_solution);
            let error = ((achieved[0] - target_pos[0]).powi(2)
                + (achieved[1] - target_pos[1]).powi(2)
                + (achieved[2] - target_pos[2]).powi(2))
            .sqrt();

            prop_assert!(
                error < 0.02,
                "FK-IK roundtrip error {error:.6} at target {target_pos:?}"
            );
        }
    }
}

// ============================================================================
// Property 6: within_limits is consistent with joint limits
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn within_limits_correct(q in arb_q()) {
        let kin = ManipulatorKinematics::default_7dof();
        // Angles generated by arb_q() are within limits by construction
        prop_assert!(
            kin.within_limits(&q),
            "Angles within strategy range should pass within_limits: {q:?}"
        );
    }
}

// ============================================================================
// Property 7: FK determinism — same input always produces same output
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn fk_deterministic(q in arb_q()) {
        let kin = ManipulatorKinematics::default_7dof();
        let pos1 = kin.end_effector_position(&q);
        let pos2 = kin.end_effector_position(&q);

        prop_assert_eq!(pos1[0].to_bits(), pos2[0].to_bits(), "FK x not deterministic");
        prop_assert_eq!(pos1[1].to_bits(), pos2[1].to_bits(), "FK y not deterministic");
        prop_assert_eq!(pos1[2].to_bits(), pos2[2].to_bits(), "FK z not deterministic");
    }
}
