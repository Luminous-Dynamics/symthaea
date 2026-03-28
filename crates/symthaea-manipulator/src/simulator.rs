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
/// Models each joint as: I * ddq = torque - damping * dq - gravity_compensation
pub struct SimpleManipulatorSimulator {
    state: ManipulatorState,
    kinematics: ManipulatorKinematics,
    /// Effective inertia per joint (kg·m²).
    inertias: [f64; NUM_JOINTS],
    /// Damping per joint (Nm·s/rad).
    damping: [f64; NUM_JOINTS],
    /// Max torque per joint (Nm).
    max_torques: [f64; NUM_JOINTS],
}

impl SimpleManipulatorSimulator {
    pub fn new() -> Self {
        Self {
            state: ManipulatorState::home(),
            kinematics: ManipulatorKinematics::default_7dof(),
            inertias: [2.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2],
            damping: [5.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0],
            max_torques: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], // Panda-like
        }
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
            // Joint dynamics: I * ddq = torque - damping * dq
            let ddq =
                (torque - self.damping[i] * self.state.joint_velocities[i]) / self.inertias[i];
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
    fn test_zero_torque_damps() {
        let mut sim = SimpleManipulatorSimulator::new();
        // Give initial velocity
        sim.state.joint_velocities[0] = 1.0;
        let cmd = ManipulatorCommand::zero();
        for _ in 0..1000 {
            sim.step(&cmd, 0.001);
        }
        // Velocity should decay toward zero via damping
        assert!(
            sim.state().joint_velocities[0].abs() < 0.1,
            "Should damp to near-zero"
        );
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
}
