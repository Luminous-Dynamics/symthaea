// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for 7-DOF industrial manipulator arm.

use serde::{Deserialize, Serialize};

pub const NUM_JOINTS: usize = 7;
pub const NUM_STATE_CHANNELS: usize = 21; // 7 angles + 7 velocities + 3 end-effector + 3 force/torque + 1 gripper

/// Manipulator state (21D).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ManipulatorState {
    pub joint_angles: [f64; NUM_JOINTS],
    pub joint_velocities: [f64; NUM_JOINTS],
    pub end_effector_position: [f64; 3],
    pub end_effector_force: [f64; 3],
    pub gripper_opening: f64, // 0.0 = closed, 1.0 = fully open
}

impl ManipulatorState {
    pub fn home() -> Self {
        Self {
            joint_angles: [
                0.0,
                0.0,
                0.0,
                -std::f64::consts::FRAC_PI_2,
                0.0,
                std::f64::consts::FRAC_PI_2,
                0.0,
            ],
            joint_velocities: [0.0; NUM_JOINTS],
            end_effector_position: [0.3, 0.0, 0.5],
            end_effector_force: [0.0; 3],
            gripper_opening: 1.0,
        }
    }

    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut c = [0.0f32; NUM_STATE_CHANNELS];
        for i in 0..NUM_JOINTS {
            c[i] = self.joint_angles[i] as f32;
        }
        for i in 0..NUM_JOINTS {
            c[NUM_JOINTS + i] = self.joint_velocities[i] as f32;
        }
        for i in 0..3 {
            c[14 + i] = self.end_effector_position[i] as f32;
        }
        for i in 0..3 {
            c[17 + i] = self.end_effector_force[i] as f32;
        }
        c[20] = self.gripper_opening as f32;
        c
    }

    pub fn is_finite(&self) -> bool {
        self.joint_angles.iter().all(|v| v.is_finite())
            && self.end_effector_position.iter().all(|v| v.is_finite())
    }

    /// Domain-specific bounds check for physically plausible arm state.
    ///
    /// Returns `true` if all joint angles are within their limits (±2.9 rad
    /// for Panda-class joints) and end-effector is within reach envelope.
    pub fn is_bounded(&self) -> bool {
        self.is_finite()
            && self.joint_angles.iter().all(|&a| a.abs() < 3.0)
            && self.joint_velocities.iter().all(|&v| v.abs() < 10.0)
            && self.end_effector_position.iter().all(|&p| p.abs() < 2.0)
    }
}

/// Motor command (7 joint torques + 1 gripper).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ManipulatorCommand {
    pub joint_torques: [f32; NUM_JOINTS],
    pub gripper: f32, // 0.0 = close, 1.0 = open
}

impl ManipulatorCommand {
    pub fn zero() -> Self {
        Self {
            joint_torques: [0.0; NUM_JOINTS],
            gripper: 0.5,
        }
    }

    pub fn clamped(mut self) -> Self {
        for t in &mut self.joint_torques {
            *t = t.clamp(-1.0, 1.0);
        }
        self.gripper = self.gripper.clamp(0.0, 1.0);
        self
    }

    pub fn control_effort(&self) -> f32 {
        self.joint_torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_JOINTS as f32
    }
}

/// Configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManipulatorConfig {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
}

impl ManipulatorConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

impl Default for ManipulatorConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-manipulator-industrial".to_string(),
            learning_rate: 0.0005,
            network_layers: 2,
            neurons_per_layer: 8,
            physics_hz: 500.0,
            cognitive_interval: 25, // 20Hz cognitive
            steps_per_episode: 2000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_home_state() {
        let state = ManipulatorState::home();
        assert!(state.is_finite());
        assert_eq!(state.gripper_opening, 1.0);
    }

    #[test]
    fn test_channels_count() {
        let channels = ManipulatorState::home().to_channels();
        assert_eq!(channels.len(), NUM_STATE_CHANNELS);
    }

    #[test]
    fn test_command_zero() {
        let cmd = ManipulatorCommand::zero();
        assert_eq!(cmd.control_effort(), 0.0);
    }

    #[test]
    fn test_command_clamped() {
        let cmd = ManipulatorCommand {
            joint_torques: [2.0; NUM_JOINTS],
            gripper: 1.5,
        };
        let clamped = cmd.clamped();
        assert!(clamped.joint_torques.iter().all(|&t| t <= 1.0));
        assert!(clamped.gripper <= 1.0);
    }
}
