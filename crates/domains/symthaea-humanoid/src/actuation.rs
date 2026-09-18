// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit conversion from canonical policy intent to backend actuation.
//!
//! The humanoid policy always emits normalized torque intent. Backends advertise
//! what they physically accept; this adapter performs the conversion rather than
//! allowing the same vector to silently mean torque in one simulator and joint
//! position in another.
//!
//! [`PhysicalActuationCommand`] binds the numeric vector to its physical mode.
//! The legacy [`HumanoidCommand`] compatibility view remains available while
//! simulators/backends migrate, but claim-bearing boundaries should consume the
//! typed command so numeric equality cannot erase torque-vs-position semantics.

use crate::morphology::HumanoidMorphology;
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActuationAdaptationError {
    ActuatorCount { expected: usize, actual: usize },
    StateCount { expected: usize, actual: usize },
    NonFiniteValue { index: usize },
    NormalizedValueOutOfRange { index: usize },
}

/// Backend-facing actuation values with their physical interpretation bound.
///
/// `values` deliberately does not use a torque-specific field name: the same
/// carrier can represent normalized torque, physical torque, normalized joint
/// position, or absolute joint position, but the accompanying mode is part of
/// the command identity and validation contract.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalActuationCommand {
    pub mode: ActuationMode,
    pub values: Vec<f32>,
}

impl PhysicalActuationCommand {
    pub fn new(
        mode: ActuationMode,
        values: Vec<f32>,
        expected_actuators: usize,
    ) -> Result<Self, ActuationAdaptationError> {
        let command = Self { mode, values };
        command.validate_for(expected_actuators)?;
        Ok(command)
    }

    pub fn validate_for(&self, expected_actuators: usize) -> Result<(), ActuationAdaptationError> {
        if self.values.len() != expected_actuators {
            return Err(ActuationAdaptationError::ActuatorCount {
                expected: expected_actuators,
                actual: self.values.len(),
            });
        }
        let normalized = matches!(
            self.mode,
            ActuationMode::NormalizedTorque | ActuationMode::NormalizedPosition
        );
        for (index, value) in self.values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ActuationAdaptationError::NonFiniteValue { index });
            }
            if normalized && !(-1.0..=1.0).contains(&value) {
                return Err(ActuationAdaptationError::NormalizedValueOutOfRange { index });
            }
        }
        Ok(())
    }

    pub fn num_actuators(&self) -> usize {
        self.values.len()
    }

    /// Explicit compatibility projection for legacy simulator/backend APIs.
    ///
    /// The returned `HumanoidCommand` carries only numbers; callers must retain
    /// `self.mode` separately. New physical boundaries should prefer this typed
    /// command directly rather than treating the compatibility vector as a
    /// semantically complete actuation record.
    pub fn legacy_vector(&self) -> HumanoidCommand {
        HumanoidCommand {
            torques: self.values.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActuationAdaptation {
    /// Typed backend-facing command. This is the claim-bearing representation.
    pub physical_command: PhysicalActuationCommand,
    /// Compatibility vector for existing simulator/backend APIs. Its physical
    /// meaning is `physical_command.mode`; the field name does not redefine it.
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

        let physical_command = PhysicalActuationCommand::new(target_mode, output, n)?;
        let command = physical_command.legacy_vector();
        Ok(ActuationAdaptation {
            physical_command,
            command,
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
    fn physical_mode_is_part_of_command_identity() {
        let torque = PhysicalActuationCommand::new(
            ActuationMode::NormalizedTorque,
            vec![0.25; 21],
            21,
        )
        .unwrap();
        let position = PhysicalActuationCommand::new(
            ActuationMode::NormalizedPosition,
            vec![0.25; 21],
            21,
        )
        .unwrap();
        assert_ne!(torque, position);
    }

    #[test]
    fn normalized_physical_command_rejects_out_of_range_values() {
        let result = PhysicalActuationCommand::new(
            ActuationMode::NormalizedPosition,
            vec![1.5; 21],
            21,
        );
        assert!(matches!(
            result,
            Err(ActuationAdaptationError::NormalizedValueOutOfRange { .. })
        ));
    }

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
        assert_eq!(adapted.physical_command.mode, ActuationMode::NormalizedTorque);
        assert_eq!(adapted.physical_command.values, intent.torques);
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
        assert_eq!(
            adapted.physical_command.mode,
            ActuationMode::PositionTargetRadians
        );
        assert!((adapted.physical_command.values[0] - 0.05).abs() < 1.0e-6);
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
        assert_eq!(
            adapted.physical_command.mode,
            ActuationMode::TorqueNewtonMetres
        );
        assert!((adapted.physical_command.values[0] - 50.0).abs() < 1.0e-6);
        assert!((adapted.command.torques[0] - 50.0).abs() < 1.0e-6);
    }
}
