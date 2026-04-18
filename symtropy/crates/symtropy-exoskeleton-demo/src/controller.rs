// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Assistive controller: impedance-style torques toward a walking-gait
//! reference, counteracting velocity deviations. The `AssistanceMode`
//! returned by the platform crate from Φ scales the output directly.

use symthaea_exoskeleton::types::{
    AssistanceMode, ExoskeletonCommand, ExoskeletonState, NUM_ACTUATORS, NUM_JOINTS,
};

pub struct AssistiveController {
    /// Target joint angles the exoskeleton "wants" to help reach.
    /// For a walking demo this is roughly the neutral standing pose.
    pub target_angles: [f64; NUM_JOINTS],
    /// Joint-space proportional gain (rad → N·m).
    pub kp: f64,
    /// Joint-space derivative gain (rad/s → N·m).
    pub kd: f64,
}

impl Default for AssistiveController {
    fn default() -> Self {
        Self {
            target_angles: [0.05, 0.1, 0.0, 0.05, 0.1, 0.0],
            kp: 12.0,
            kd: 1.2,
        }
    }
}

impl AssistiveController {
    /// Compute the raw (un-gated) assist command, then apply the mode's
    /// torque + stiffness factors from `AssistanceMode`.
    pub fn compute(&self, state: &ExoskeletonState, mode: AssistanceMode) -> ExoskeletonCommand {
        let torque_factor = mode.torque_factor();
        let stiffness_factor = mode.stiffness_factor() as f32;

        let mut torques = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_JOINTS {
            let err = self.target_angles[i] - state.joint_angles[i];
            let vel = state.joint_velocities[i];
            // Raw PD command, normalized to the [-1, 1] torque channel
            // (simulator multiplies by config.max_torques internally).
            let raw = (self.kp * err - self.kd * vel) / 60.0;
            torques[i] = (raw as f32).clamp(-1.0, 1.0) * torque_factor;
        }

        ExoskeletonCommand {
            joint_torques: torques,
            stiffness_gain: 0.5 * stiffness_factor,
            damping_gain: 0.3 * stiffness_factor,
        }
    }
}
