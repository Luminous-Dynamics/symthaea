// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rigid-body arm dynamics simulator.

use crate::kinematics::ManipulatorKinematics;
use crate::types::{ManipulatorCommand, ManipulatorState, NUM_JOINTS};

pub trait ManipulatorPhysicsSimulator {
    fn step(&mut self, cmd: &ManipulatorCommand, dt: f64);
    fn state(&self) -> &ManipulatorState;
    fn reset(&mut self);
}

/// Simple joint-space dynamics simulator.
///
/// Models each joint as: I * ddq = torque - damping * dq - gravity_torque
///
/// Gravity model: simplified sum of distal link contributions.
/// For joint i, gravity torque = sum_j(m_j * g * d_j * cos(q_i)) for j >= i.
/// This is a first-order approximation valid for arms where all joints
/// rotate about roughly horizontal axes (e.g., Panda-class).
pub struct SimpleManipulatorSimulator {
    state: ManipulatorState,
    kinematics: ManipulatorKinematics,
    /// Effective inertia per joint (kg·m²).
    inertias: [f64; NUM_JOINTS],
    /// Damping per joint (Nm·s/rad).
    damping: [f64; NUM_JOINTS],
    /// Max torque per joint (Nm).
    max_torques: [f64; NUM_JOINTS],
    /// Link masses (kg) — used for gravity torque computation.
    link_masses: [f64; NUM_JOINTS],
    /// Center-of-mass distance from joint axis (m) per link.
    link_com_distances: [f64; NUM_JOINTS],
    /// Gravitational acceleration (m/s²).
    gravity: f64,
    /// External forces applied to the end-effector [Fx, Fy, Fz] in Newtons.
    /// Set by external systems (e.g., human proximity force field) and added
    /// to `end_effector_force` each step. Reset to zero after each step.
    pub external_forces: [f64; 3],
}

impl SimpleManipulatorSimulator {
    pub fn new() -> Self {
        Self {
            state: ManipulatorState::home(),
            kinematics: ManipulatorKinematics::default_7dof(),
            inertias: [2.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2],
            damping: [5.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0],
            max_torques: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], // Panda-like
            // Panda-class link masses (kg): heavy at shoulder, light at wrist
            link_masses: [4.0, 4.0, 3.0, 2.5, 1.5, 1.0, 0.5],
            // CoM distance from joint axis (m): roughly half the link length
            link_com_distances: [0.15, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
            gravity: 9.81,
            external_forces: [0.0; 3],
        }
    }

    /// Compute simplified gravity torque at joint `i`.
    ///
    /// Gravity torque = sum of (mass_j * g * com_dist_j * cos(angle_i))
    /// for all links j distal to joint i (j >= i).
    ///
    /// This is a first-order approximation: it treats each joint's angle
    /// as the dominant contributor to that link's gravitational moment arm.
    /// Accurate for configurations near home pose; degrades for highly
    /// coupled configurations but captures the dominant gravity effect.
    fn gravity_torque_at(&self, joint: usize) -> f64 {
        let mut tau = 0.0;
        for j in joint..NUM_JOINTS {
            tau += self.link_masses[j]
                * self.gravity
                * self.link_com_distances[j]
                * self.state.joint_angles[joint].cos();
        }
        tau
    }
}

impl Default for SimpleManipulatorSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl ManipulatorPhysicsSimulator for SimpleManipulatorSimulator {
    fn step(&mut self, cmd: &ManipulatorCommand, dt: f64) {
        for i in 0..NUM_JOINTS {
            // Scale normalized torque to actual Nm
            let torque = cmd.joint_torques[i] as f64 * self.max_torques[i];
            // Joint dynamics: I * ddq = torque - damping * dq - gravity
            let gravity = self.gravity_torque_at(i);
            let ddq = (torque - self.damping[i] * self.state.joint_velocities[i] - gravity)
                / self.inertias[i];
            // Semi-implicit Euler
            self.state.joint_velocities[i] += ddq * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;
            // Clamp to joint limits
            let limits = self.kinematics.joint_limits[i];
            if self.state.joint_angles[i] < limits[0] {
                self.state.joint_angles[i] = limits[0];
                self.state.joint_velocities[i] = self.state.joint_velocities[i].max(0.0);
            }
            if self.state.joint_angles[i] > limits[1] {
                self.state.joint_angles[i] = limits[1];
                self.state.joint_velocities[i] = self.state.joint_velocities[i].min(0.0);
            }
        }
        // Update end-effector position via FK
        self.state.end_effector_position = self
            .kinematics
            .end_effector_position(&self.state.joint_angles);
        // Apply external forces to end-effector force feedback
        for i in 0..3 {
            self.state.end_effector_force[i] = self.external_forces[i];
        }
        self.external_forces = [0.0; 3]; // Reset after application
                                         // Update gripper
        self.state.gripper_opening = cmd.gripper as f64;
    }

    fn state(&self) -> &ManipulatorState {
        &self.state
    }

    fn reset(&mut self) {
        self.state = ManipulatorState::home();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_torque_reaches_equilibrium() {
        let mut sim = SimpleManipulatorSimulator::new();
        // Give initial velocity
        sim.state.joint_velocities[0] = 1.0;
        let cmd = ManipulatorCommand::zero();
        // With gravity + damping, the joint should reach a gravity-determined
        // equilibrium (not necessarily zero velocity, but bounded).
        for _ in 0..5000 {
            sim.step(&cmd, 0.001);
        }
        // Velocity should be bounded (damping prevents runaway)
        assert!(
            sim.state().joint_velocities[0].abs() < 2.0,
            "Damping should bound velocity: got {}",
            sim.state().joint_velocities[0]
        );
        // State must remain finite
        assert!(sim.state().is_finite(), "State must remain finite");
    }

    #[test]
    fn test_torque_moves_joint() {
        let mut sim = SimpleManipulatorSimulator::new();
        let initial_angle = sim.state().joint_angles[0];
        let mut cmd = ManipulatorCommand::zero();
        cmd.joint_torques[0] = 0.5; // Positive torque on J1
        for _ in 0..100 {
            sim.step(&cmd, 0.001);
        }
        assert!(
            sim.state().joint_angles[0] > initial_angle,
            "Positive torque should increase angle"
        );
    }

    #[test]
    fn test_joint_limits_enforced() {
        let mut sim = SimpleManipulatorSimulator::new();
        let mut cmd = ManipulatorCommand::zero();
        cmd.joint_torques[0] = 1.0; // Max torque
        for _ in 0..100000 {
            sim.step(&cmd, 0.001);
        }
        let limits = sim.kinematics.joint_limits[0];
        assert!(
            sim.state().joint_angles[0] <= limits[1] + 0.01,
            "Should not exceed upper limit"
        );
    }

    #[test]
    fn test_end_effector_updates() {
        let mut sim = SimpleManipulatorSimulator::new();
        let initial_pos = sim.state().end_effector_position;
        let mut cmd = ManipulatorCommand::zero();
        cmd.joint_torques[1] = 0.3; // Move shoulder
        for _ in 0..500 {
            sim.step(&cmd, 0.001);
        }
        let new_pos = sim.state().end_effector_position;
        let dist = ((new_pos[0] - initial_pos[0]).powi(2)
            + (new_pos[1] - initial_pos[1]).powi(2)
            + (new_pos[2] - initial_pos[2]).powi(2))
        .sqrt();
        assert!(dist > 0.001, "End effector should move: dist={dist}");
    }

    #[test]
    fn test_reset() {
        let mut sim = SimpleManipulatorSimulator::new();
        let mut cmd = ManipulatorCommand::zero();
        cmd.joint_torques[0] = 0.5;
        sim.step(&cmd, 0.01);
        sim.reset();
        assert_eq!(sim.state().joint_velocities, [0.0; NUM_JOINTS]);
    }

    #[test]
    fn test_gripper_tracks_command() {
        let mut sim = SimpleManipulatorSimulator::new();
        let mut cmd = ManipulatorCommand::zero();
        cmd.gripper = 0.3;
        sim.step(&cmd, 0.01);
        assert!((sim.state().gripper_opening - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_gravity_causes_drift_with_zero_torque() {
        let mut sim = SimpleManipulatorSimulator::new();
        let initial_angles = sim.state().joint_angles;
        let cmd = ManipulatorCommand::zero();
        for _ in 0..500 {
            sim.step(&cmd, 0.001);
        }
        let max_drift = (0..NUM_JOINTS)
            .map(|i| (sim.state().joint_angles[i] - initial_angles[i]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_drift > 0.001,
            "Gravity should cause joints to drift: max_drift={}",
            max_drift
        );
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Arbitrary joint torques must never produce NaN/Inf.
            #[test]
            fn arbitrary_torques_stay_finite(
                t0 in -1.0f32..1.0, t1 in -1.0f32..1.0,
                t2 in -1.0f32..1.0, t3 in -1.0f32..1.0,
                t4 in -1.0f32..1.0, t5 in -1.0f32..1.0,
                t6 in -1.0f32..1.0, gripper in 0.0f32..1.0,
                dt in 0.0001f64..0.01,
                steps in 1usize..300,
            ) {
                let mut sim = SimpleManipulatorSimulator::new();
                let cmd = ManipulatorCommand {
                    joint_torques: [t0, t1, t2, t3, t4, t5, t6],
                    gripper,
                };
                for _ in 0..steps {
                    sim.step(&cmd, dt);
                }
                prop_assert!(sim.state().is_finite(), "Manipulator state diverged to NaN/Inf");
            }

            /// Joint angles must stay within limits regardless of input.
            #[test]
            fn joints_within_limits(
                t0 in -1.0f32..1.0, t1 in -1.0f32..1.0,
                t2 in -1.0f32..1.0, t3 in -1.0f32..1.0,
                t4 in -1.0f32..1.0, t5 in -1.0f32..1.0,
                t6 in -1.0f32..1.0,
            ) {
                let mut sim = SimpleManipulatorSimulator::new();
                let cmd = ManipulatorCommand {
                    joint_torques: [t0, t1, t2, t3, t4, t5, t6],
                    gripper: 0.5,
                };
                for _ in 0..1000 {
                    sim.step(&cmd, 0.001);
                }
                for i in 0..NUM_JOINTS {
                    let limits = sim.kinematics.joint_limits[i];
                    prop_assert!(
                        sim.state().joint_angles[i] >= limits[0] - 0.01
                            && sim.state().joint_angles[i] <= limits[1] + 0.01,
                        "Joint {} out of limits: {} not in [{}, {}]",
                        i, sim.state().joint_angles[i], limits[0], limits[1]
                    );
                }
            }
        }
    }
}
