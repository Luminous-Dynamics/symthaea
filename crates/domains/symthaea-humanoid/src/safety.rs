// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic command projection for simulator and hardware parity.
//!
//! This layer is deliberately independent of learned confidence, Phi, and FEP.
//! Cognitive signals may request less authority, but they cannot bypass joint,
//! velocity, finite-value, morphology, or slew-rate constraints.

use serde::{Deserialize, Serialize};

use crate::morphology::HumanoidMorphology;
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    /// Maximum magnitude accepted for normalized command modes.
    pub max_normalized_command: f32,
    /// Maximum normalized command change per second.
    pub max_command_slew_per_second: f32,
    /// Distance from a joint limit where authority begins to taper.
    pub joint_limit_margin_rad: f64,
    /// Velocity at which commands accelerating in the same direction are removed.
    pub joint_velocity_soft_limit_rad_s: f64,
}

impl Default for SafetyEnvelope {
    fn default() -> Self {
        Self {
            max_normalized_command: 1.0,
            max_command_slew_per_second: 8.0,
            joint_limit_margin_rad: 0.12,
            joint_velocity_soft_limit_rad_s: 12.0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyProjectionReport {
    pub rejected: bool,
    pub non_finite_values: usize,
    pub morphology_mismatch: bool,
    pub magnitude_clips: usize,
    pub slew_clips: usize,
    pub joint_limit_interventions: usize,
    pub velocity_interventions: usize,
}

impl SafetyProjectionReport {
    pub fn intervened(&self) -> bool {
        self.rejected
            || self.magnitude_clips > 0
            || self.slew_clips > 0
            || self.joint_limit_interventions > 0
            || self.velocity_interventions > 0
    }
}

#[derive(Debug, Clone)]
pub struct ProjectedCommand {
    pub command: HumanoidCommand,
    pub report: SafetyProjectionReport,
}

/// Stateful projector because slew limits depend on the last applied command.
pub struct HumanoidSafetyProjector {
    morphology: HumanoidMorphology,
    envelope: SafetyEnvelope,
    previous: HumanoidCommand,
}

impl HumanoidSafetyProjector {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_envelope(morphology, SafetyEnvelope::default())
    }

    pub fn with_envelope(morphology: HumanoidMorphology, envelope: SafetyEnvelope) -> Self {
        Self {
            morphology,
            envelope,
            previous: HumanoidCommand::zero_for(morphology.num_actuators()),
        }
    }

    pub fn reset(&mut self) {
        self.previous = HumanoidCommand::zero_for(self.morphology.num_actuators());
    }

    pub fn project(
        &mut self,
        requested: &HumanoidCommand,
        state: &HumanoidState,
        actuation_mode: ActuationMode,
        dt: f64,
    ) -> ProjectedCommand {
        let n = self.morphology.num_actuators();
        let mut report = SafetyProjectionReport::default();

        if requested.num_actuators() != n
            || state.joint_angles.len() != n
            || state.joint_velocities.len() != n
        {
            report.rejected = true;
            report.morphology_mismatch = true;
            let command = HumanoidCommand::zero_for(n);
            self.previous = command.clone();
            return ProjectedCommand { command, report };
        }

        let mut values = requested.torques.clone();
        for value in &mut values {
            if !value.is_finite() {
                *value = 0.0;
                report.non_finite_values += 1;
                report.rejected = true;
            }
        }
        if report.rejected {
            let command = HumanoidCommand::zero_for(n);
            self.previous = command.clone();
            return ProjectedCommand { command, report };
        }

        let normalized = matches!(
            actuation_mode,
            ActuationMode::NormalizedTorque | ActuationMode::NormalizedPosition
        );
        if normalized {
            let limit = self.envelope.max_normalized_command.clamp(0.0, 1.0);
            for value in &mut values {
                let clipped = value.clamp(-limit, limit);
                if clipped != *value {
                    report.magnitude_clips += 1;
                    *value = clipped;
                }
            }
        }

        let limits = self.morphology.joint_limits();
        let margin = self.envelope.joint_limit_margin_rad.max(1.0e-6);
        let velocity_limit = self.envelope.joint_velocity_soft_limit_rad_s.max(1.0e-6);
        for i in 0..n {
            let [low, high] = limits[i];
            let angle = state.joint_angles[i];
            let velocity = state.joint_velocities[i];

            if values[i] > 0.0 {
                let authority = ((high - angle) / margin).clamp(0.0, 1.0) as f32;
                if authority < 1.0 {
                    values[i] *= authority;
                    report.joint_limit_interventions += 1;
                }
                if velocity >= velocity_limit {
                    values[i] = 0.0;
                    report.velocity_interventions += 1;
                }
            } else if values[i] < 0.0 {
                let authority = ((angle - low) / margin).clamp(0.0, 1.0) as f32;
                if authority < 1.0 {
                    values[i] *= authority;
                    report.joint_limit_interventions += 1;
                }
                if velocity <= -velocity_limit {
                    values[i] = 0.0;
                    report.velocity_interventions += 1;
                }
            }
        }

        let max_step = (self.envelope.max_command_slew_per_second * dt.max(0.0) as f32).max(0.0);
        for (value, previous) in values.iter_mut().zip(self.previous.torques.iter()) {
            let low = *previous - max_step;
            let high = *previous + max_step;
            let clipped = value.clamp(low, high);
            if clipped != *value {
                report.slew_clips += 1;
                *value = clipped;
            }
        }

        let command = HumanoidCommand { torques: values };
        self.previous = command.clone();
        ProjectedCommand { command, report }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_commands_fail_closed() {
        let mut projector = HumanoidSafetyProjector::new(HumanoidMorphology::Dmc21);
        let state = HumanoidState::standing();
        let result = projector.project(
            &HumanoidCommand::zero_for(20),
            &state,
            ActuationMode::NormalizedTorque,
            0.025,
        );
        assert!(result.report.rejected);
        assert!(result.command.torques.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn repeated_projection_obeys_slew_rate() {
        let envelope = SafetyEnvelope {
            max_command_slew_per_second: 2.0,
            ..SafetyEnvelope::default()
        };
        let mut projector =
            HumanoidSafetyProjector::with_envelope(HumanoidMorphology::Dmc21, envelope);
        let state = HumanoidState::standing();
        let requested = HumanoidCommand::from_raw(&vec![1.0; 21]);
        let result = projector.project(&requested, &state, ActuationMode::NormalizedTorque, 0.025);
        assert!(
            result
                .command
                .torques
                .iter()
                .all(|value| *value <= 0.050_001)
        );
        assert_eq!(result.report.slew_clips, 21);
    }

    #[test]
    fn removes_commands_that_push_through_joint_limits() {
        let mut projector = HumanoidSafetyProjector::new(HumanoidMorphology::Dmc21);
        let mut state = HumanoidState::standing();
        let limits = HumanoidMorphology::Dmc21.joint_limits();
        state.joint_angles[0] = limits[0][1];
        let mut requested = HumanoidCommand::zero();
        requested.torques[0] = 1.0;
        let result = projector.project(&requested, &state, ActuationMode::NormalizedTorque, 1.0);
        assert_eq!(result.command.torques[0], 0.0);
        assert!(result.report.joint_limit_interventions > 0);
    }
}
