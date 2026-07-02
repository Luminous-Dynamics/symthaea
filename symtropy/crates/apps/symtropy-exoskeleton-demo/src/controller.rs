// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Assistive controller: impedance-style torques toward a walking-gait
//! reference, counteracting velocity deviations. The `AssistanceMode`
//! returned by the platform crate from Φ scales the output directly.

use symthaea_exoskeleton::types::{
    AssistanceMode, ExoskeletonCommand, ExoskeletonState, NUM_ACTUATORS, NUM_JOINTS,
};

/// Represents the persistent state of the entity's physical and mental condition.
/// These values are typically updated by the physics callback system.
#[derive(Debug, Clone, Copy)]
pub struct ConsciousnessState {
    /// 0.0 (perfect) to 1.0 (total exhaustion). Higher means worse.
    pub fatigue: f64,
    /// 0.0 (none) to 1.0 (severe injury). Higher means worse.
    pub trauma: f64,
    /// 0.0 (calm) to 1.0 (panic/stress). Higher means worse.
    pub stress: f64,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        ConsciousnessState {
            fatigue: 0.0,
            trauma: 0.0,
            stress: 0.0,
        }
    }
}

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
    ///
    /// The output is scaled by the current `ConsciousnessState` to simulate
    /// performance degradation due to fatigue or trauma.
    pub fn compute(
        &self,
        state: &ExoskeletonState,
        mode: AssistanceMode,
        state_metrics: ConsciousnessState,
    ) -> ExoskeletonCommand {
        let torque_factor = mode.torque_factor();
        let stiffness_factor = mode.stiffness_factor() as f32;

        // Calculate a combined performance degradation factor.
        // Fatigue and Trauma are the primary dampeners on motor output.
        // Stress might affect the stiffness/damping gains more.
        let performance_factor =
            1.0 - (state_metrics.fatigue * 0.5 + state_metrics.trauma * 0.3).clamp(0.0, 1.0);

        let mut torques = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_JOINTS {
            let err = self.target_angles[i] - state.joint_angles[i];
            let vel = state.joint_velocities[i];

            // Raw PD command, normalized to the [-1, 1] torque channel
            // (simulator multiplies by config.max_torques internally).
            let raw = (self.kp * err - self.kd * vel) / 60.0;

            // Apply performance degradation factor to the raw torque command
            let scaled_raw = raw * performance_factor;

            torques[i] = (scaled_raw as f32).clamp(-1.0, 1.0) * torque_factor;
        }

        ExoskeletonCommand {
            joint_torques: torques,
            // Stress can affect how much the system resists external forces (stiffness/damping)
            stiffness_gain: 0.5 * stiffness_factor * (1.0 - state_metrics.stress * 0.2).max(0.5),
            damping_gain: 0.3 * stiffness_factor * (1.0 - state_metrics.stress * 0.2).max(0.5),
        }
    }
}
