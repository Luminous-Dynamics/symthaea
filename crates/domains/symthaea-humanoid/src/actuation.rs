// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit conversion from canonical policy intent to backend actuation.
//!
//! The humanoid policy always emits normalized torque intent. Backends advertise
//! what they physically accept; this adapter performs the conversion rather than
//! allowing the same vector to silently mean torque in one simulator and joint
//! position in another.

use crate::morphology::HumanoidMorphology;
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActuationAdaptationError {
    ActuatorCount { expected: usize, actual: usize },
    StateCount { expected: usize, actual: usize },
    NonFiniteValue { index: usize },
}

#[derive(Debug, Clone)]
pub struct ActuationAdaptation {
    pub command: HumanoidCommand,
    pub source_mode: ActuationMode,
    pub target_mode: ActuationMode,
    pub clipped_joints: usize,
}

#[derive(Debug, Clone)]
pub struct ActuationAdapter {
    /// Maximum target-position displacement produced by full normalized intent.
    pub max_position_step_rad: f64,
}

impl Default for ActuationAdapter {
    fn default() -> Self {
        Self {
            max_position_step_rad: 0.20,
        }
    }
}

impl ActuationAdapter {
    pub fn adapt_normalized_torque_intent(
        &self,
        intent: &HumanoidCommand,
        state: &HumanoidState,
        morphology: HumanoidMorphology,
        target_mode: ActuationMode,
    ) -> Result<ActuationAdaptation, ActuationAdaptationError> {
        let n = morphology.num_actuators();
        if intent.num_actuators() != n {
            return Err(ActuationAdaptationError::ActuatorCount {
                expected: n,
                actual: intent.num_actuators(),
            });
        }
        if state.joint_angles.len() != n {
            return Err(ActuationAdaptationError::StateCount {
                expected: n,
                actual: state.joint_angles.len(),
            });
        }
        for (index, value) in intent.torques.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ActuationAdaptationError::NonFiniteValue { index });
            }
        }

        let limits = morphology.joint_limits();
        let torque_scales = morphology.joint_torque_scales();
        let mut clipped_joints = 0usize;
        let mut output = Vec::with_capacity(n);

        match target_mode {
            ActuationMode::NormalizedTorque => {
                output.extend(intent.torques.iter().map(|value| value.clamp(-1.0, 1.0)));
            }
            ActuationMode::TorqueNewtonMetres => {
                output.extend(
                    intent
                        .torques
                        .iter()
                        .zip(torque_scales.iter())
                        .map(|(value, scale)| value.clamp(-1.0, 1.0) * *scale as f32),
                );
            }
            ActuationMode::PositionTargetRadians | ActuationMode::NormalizedPosition => {
                for i in 0..n {
                    let [low, high] = limits[i];
                    let requested = state.joint_angles[i]
                        + intent.torques[i].clamp(-1.0, 1.0) as f64 * self.max_position_step_rad;
                    let target = requested.clamp(low, high);
                    if target != requested {
                        clipped_joints += 1;
                    }

                    if target_mode == ActuationMode::PositionTargetRadians {
                        output.push(target as f32);
                    } else {
                        let range = high - low;
                        let normalized = if range <= f64::EPSILON {
                            0.0
                        } else {
                            (2.0 * (target - low) / range - 1.0).clamp(-1.0, 1.0)
                        };
                        output.push(normalized as f32);
                    }
                }
            }
        }

        Ok(ActuationAdaptation {
            command: HumanoidCommand { torques: output },
            source_mode: ActuationMode::NormalizedTorque,
            target_mode,
            clipped_joints,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torque_backend_preserves_normalized_intent() {
        let adapter = ActuationAdapter::default();
        let state = HumanoidState::standing();
        let intent = HumanoidCommand::from_raw(&vec![0.5; 21]);
        let adapted = adapter
            .adapt_normalized_torque_intent(
                &intent,
                &state,
                HumanoidMorphology::Dmc21,
                ActuationMode::NormalizedTorque,
            )
            .unwrap();
        assert_eq!(adapted.command.torques, intent.torques);
    }

    #[test]
    fn position_backend_receives_radian_targets() {
        let adapter = ActuationAdapter {
            max_position_step_rad: 0.1,
        };
        let state = HumanoidState::standing();
        let intent = HumanoidCommand::from_raw(&vec![0.5; 21]);
        let adapted = adapter
            .adapt_normalized_torque_intent(
                &intent,
                &state,
                HumanoidMorphology::Dmc21,
                ActuationMode::PositionTargetRadians,
            )
            .unwrap();
        assert!((adapted.command.torques[0] - 0.05).abs() < 1.0e-6);
    }

    #[test]
    fn torque_nm_backend_uses_morphology_scale() {
        let adapter = ActuationAdapter::default();
        let state = HumanoidState::standing();
        let intent = HumanoidCommand::from_raw(&vec![0.5; 21]);
        let adapted = adapter
            .adapt_normalized_torque_intent(
                &intent,
                &state,
                HumanoidMorphology::Dmc21,
                ActuationMode::TorqueNewtonMetres,
            )
            .unwrap();
        assert!((adapted.command.torques[0] - 50.0).abs() < 1.0e-6);
    }
}
