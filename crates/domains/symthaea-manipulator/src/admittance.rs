// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Admittance control: compliant response to sensed end-effector force.
//!
//! The cognitive controller (`controller.rs`) is a pure feedforward torque
//! generator — it has no notion of contact force at all. Without this
//! module, an external push is either invisible to control (ignored) or
//! fought at full cognitive authority, which the project's own design note
//! on admittance control warns causes chatter: "Phi-proportional stiffness;
//! never fight external force with full IK/torque authority."
//!
//! This closes that gap: given the previous step's measured end-effector
//! force and the current safety tier, compute a Jacobian-transpose torque
//! correction that partially cancels the cognitive command in proportion
//! to sensed force and to how uncertain/unsafe the current tier is — the
//! arm yields more under contact when confidence is low, stays stiffer
//! when confident and unopposed.
//!
//! Deliberately not exercised at [`MotorSafetyLevel::Red`]: `SafeFallback`'s
//! GravityHold already takes over there with its own full-authority
//! command (see `embodiment.rs`), bypassing this path entirely.

use crate::kinematics::ManipulatorKinematics;
use crate::types::{ManipulatorCommand, NUM_JOINTS};
use symthaea_core::embodiment::MotorSafetyLevel;

/// Compliance gain per safety tier: how much of the sensed force's
/// equivalent torque is subtracted from the cognitive command. `0.0` is
/// fully stiff (ignore force entirely); `1.0` is fully compliant (cancel
/// the force's torque contribution exactly, net response driven only by
/// physical dynamics).
///
/// Monotonically increasing with tier severity, matching `MotorSafetyLevel`'s
/// own Phi-derived escalation (`from_phi`): the same signal that says "less
/// certain" also says "yield more to whatever is pushing back."
pub fn compliance_gain(safety: MotorSafetyLevel) -> f64 {
    match safety {
        MotorSafetyLevel::Green => 0.2,
        MotorSafetyLevel::Yellow => 0.45,
        MotorSafetyLevel::Orange => 0.75,
        // Not reached via the normal embodiment path (GravityHold
        // pre-empts it), but defined for completeness/consistency rather
        // than leaving an implicit gap in the match.
        MotorSafetyLevel::Red => 1.0,
    }
}

/// Apply the admittance correction to `cmd` in place: subtract
/// `compliance_gain(safety) * (J^T · sensed_force)`, normalized into the
/// command's `[-1, 1]` torque range via `max_torques`, then clamp.
///
/// `sensed_force` should be the *previous* step's measured end-effector
/// force (`ManipulatorState::end_effector_force`) — this is a one-tick
/// feedback loop, same latency convention as any other sensor-driven
/// control correction here.
pub fn apply_admittance(
    cmd: &mut ManipulatorCommand,
    kinematics: &ManipulatorKinematics,
    q: &[f64],
    sensed_force: &[f64; 3],
    max_torques: &[f64; NUM_JOINTS],
    safety: MotorSafetyLevel,
) {
    let gain = compliance_gain(safety);
    if gain <= 0.0 || sensed_force.iter().all(|&f| f == 0.0) {
        return;
    }
    let tau_ext = kinematics.cartesian_force_to_joint_torque(q, sensed_force);
    for j in 0..NUM_JOINTS {
        let normalized = (tau_ext[j] / max_torques[j]) as f32;
        cmd.joint_torques[j] -= gain as f32 * normalized;
    }
    *cmd = cmd.clamped();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_authority_cmd() -> ManipulatorCommand {
        ManipulatorCommand {
            joint_torques: [1.0; NUM_JOINTS],
            gripper: 0.5,
        }
    }

    #[test]
    fn test_compliance_gain_monotonic_with_severity() {
        assert!(
            compliance_gain(MotorSafetyLevel::Green) < compliance_gain(MotorSafetyLevel::Yellow)
        );
        assert!(
            compliance_gain(MotorSafetyLevel::Yellow) < compliance_gain(MotorSafetyLevel::Orange)
        );
        assert!(compliance_gain(MotorSafetyLevel::Orange) < compliance_gain(MotorSafetyLevel::Red));
    }

    #[test]
    fn test_zero_force_leaves_command_unchanged() {
        let kin = ManipulatorKinematics::default_7dof();
        let q = vec![0.0; 7];
        let max_torques = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];
        let mut cmd = full_authority_cmd();
        let before = cmd;
        apply_admittance(
            &mut cmd,
            &kin,
            &q,
            &[0.0, 0.0, 0.0],
            &max_torques,
            MotorSafetyLevel::Orange,
        );
        assert_eq!(cmd.joint_torques, before.joint_torques);
    }

    #[test]
    fn test_nonzero_force_reduces_at_least_one_joint_torque() {
        let kin = ManipulatorKinematics::default_7dof();
        let q = vec![0.1, 0.2, 0.0, -1.0, 0.0, 1.0, 0.0];
        let max_torques = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];
        let mut cmd = full_authority_cmd();
        let before = cmd;
        apply_admittance(
            &mut cmd,
            &kin,
            &q,
            &[20.0, 0.0, 0.0],
            &max_torques,
            MotorSafetyLevel::Orange,
        );
        assert!(
            cmd.joint_torques
                .iter()
                .zip(before.joint_torques.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "admittance correction had no effect: {:?} vs {:?}",
            cmd.joint_torques,
            before.joint_torques
        );
    }

    #[test]
    fn test_higher_safety_tier_yields_more() {
        let kin = ManipulatorKinematics::default_7dof();
        let q = vec![0.1, 0.2, 0.0, -1.0, 0.0, 1.0, 0.0];
        let max_torques = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];
        let force = [20.0, 5.0, -3.0];

        let mut cmd_green = full_authority_cmd();
        apply_admittance(
            &mut cmd_green,
            &kin,
            &q,
            &force,
            &max_torques,
            MotorSafetyLevel::Green,
        );

        let mut cmd_orange = full_authority_cmd();
        apply_admittance(
            &mut cmd_orange,
            &kin,
            &q,
            &force,
            &max_torques,
            MotorSafetyLevel::Orange,
        );

        let green_deviation: f32 = cmd_green
            .joint_torques
            .iter()
            .map(|t| (1.0 - t).abs())
            .sum();
        let orange_deviation: f32 = cmd_orange
            .joint_torques
            .iter()
            .map(|t| (1.0 - t).abs())
            .sum();
        assert!(
            orange_deviation > green_deviation,
            "Orange (less confident) should yield more than Green: {orange_deviation} vs {green_deviation}"
        );
    }

    #[test]
    fn test_result_stays_clamped() {
        let kin = ManipulatorKinematics::default_7dof();
        let q = vec![0.1, 0.2, 0.0, -1.0, 0.0, 1.0, 0.0];
        let max_torques = [1.0; NUM_JOINTS]; // tiny max torque -> easy to overflow normalization
        let mut cmd = full_authority_cmd();
        apply_admittance(
            &mut cmd,
            &kin,
            &q,
            &[1000.0, 1000.0, 1000.0],
            &max_torques,
            MotorSafetyLevel::Red,
        );
        for t in cmd.joint_torques {
            assert!((-1.0..=1.0).contains(&t), "torque escaped clamp: {t}");
        }
    }
}
