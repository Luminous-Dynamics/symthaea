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
//! Adapted payloads keep their morphology and target [`ActuationMode`] attached
//! to the values. They deliberately do not convert back into [`HumanoidCommand`],
//! because that type has canonical normalized-torque semantics.

use crate::morphology::HumanoidMorphology;
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActuationAdaptationError {
    ActuatorCount { expected: usize, actual: usize },
    StateCount { expected: usize, actual: usize },
    NonFiniteValue { index: usize },
    NonFiniteStateValue { index: usize },
}

/// Backend-ready actuator values whose physical interpretation cannot be
/// detached from the payload.
///
/// There is intentionally no public generic `(mode, values)` constructor and
/// no conversion into [`HumanoidCommand`]. Instances are created by
/// [`ActuationAdapter`], which keeps normalized-torque policy intent separate
/// from backend actuation semantics.
#[derive(Debug, Clone)]
pub struct AdaptedActuationCommand {
    morphology: HumanoidMorphology,
    mode: ActuationMode,
    values: Vec<f32>,
}

impl AdaptedActuationCommand {
    /// Exact morphology whose actuator ordering/count this payload uses.
    pub fn morphology(&self) -> HumanoidMorphology {
        self.morphology
    }

    /// Physical interpretation of every value in this payload.
    pub fn mode(&self) -> ActuationMode {
        self.mode
    }

    /// Borrow the actuator values without losing their attached mode.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of actuator values carried by this command.
    pub fn num_actuators(&self) -> usize {
        self.values.len()
    }
}

/// Result of explicitly adapting one canonical normalized-torque policy intent
/// to one backend actuation mode.
///
/// This is a semantic conversion result only. It does **not** prove that the
/// supplied [`HumanoidState`] came from physical sensors, is current, or is
/// otherwise authoritative for real hardware.
#[derive(Debug, Clone)]
pub struct ActuationAdaptation {
    command: AdaptedActuationCommand,
    source_mode: ActuationMode,
    clipped_joints: usize,
}

impl ActuationAdaptation {
    /// Backend-ready, mode-tagged payload.
    pub fn command(&self) -> &AdaptedActuationCommand {
        &self.command
    }

    /// Semantic mode of the policy input consumed by the adapter.
    pub fn source_mode(&self) -> ActuationMode {
        self.source_mode
    }

    /// Target mode is derived from the command itself; there is no detached
    /// sidecar field that can disagree with the payload's meaning.
    pub fn target_mode(&self) -> ActuationMode {
        self.command.mode()
    }

    /// Number of target joints clipped to morphology limits while adapting.
    pub fn clipped_joints(&self) -> usize {
        self.clipped_joints
    }
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
        for (index, value) in state.joint_angles.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ActuationAdaptationError::NonFiniteStateValue { index });
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

        for (index, value) in output.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ActuationAdaptationError::NonFiniteValue { index });
            }
        }

        Ok(ActuationAdaptation {
            command: AdaptedActuationCommand {
                morphology,
                mode: target_mode,
                values: output,
            },
            source_mode: ActuationMode::NormalizedTorque,
            clipped_joints,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torque_backend_preserves_normalized_intent_and_mode() {
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
        assert_eq!(adapted.command().values(), intent.torques.as_slice());
        assert_eq!(adapted.command().mode(), ActuationMode::NormalizedTorque);
        assert_eq!(adapted.target_mode(), adapted.command().mode());
        assert_eq!(adapted.source_mode(), ActuationMode::NormalizedTorque);
        assert_eq!(adapted.command().morphology(), HumanoidMorphology::Dmc21);
    }

    #[test]
    fn position_backend_receives_mode_tagged_radian_targets() {
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
        assert!((adapted.command().values()[0] - 0.05).abs() < 1.0e-6);
        assert_eq!(
            adapted.command().mode(),
            ActuationMode::PositionTargetRadians
        );
    }

    #[test]
    fn normalized_position_backend_is_bounded_and_mode_tagged() {
        let adapter = ActuationAdapter::default();
        let state = HumanoidState::standing();
        let intent = HumanoidCommand::from_raw(&vec![1.0; 21]);
        let adapted = adapter
            .adapt_normalized_torque_intent(
                &intent,
                &state,
                HumanoidMorphology::Dmc21,
                ActuationMode::NormalizedPosition,
            )
            .unwrap();

        assert_eq!(adapted.target_mode(), ActuationMode::NormalizedPosition);
        assert_eq!(adapted.command().num_actuators(), 21);
        assert!(
            adapted
                .command()
                .values()
                .iter()
                .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
        );
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
        assert!((adapted.command().values()[0] - 50.0).abs() < 1.0e-6);
        assert_eq!(adapted.target_mode(), ActuationMode::TorqueNewtonMetres);
    }

    #[test]
    fn rejects_non_finite_policy_intent() {
        let adapter = ActuationAdapter::default();
        let state = HumanoidState::standing();
        let mut intent = HumanoidCommand::from_raw(&vec![0.0; 21]);
        intent.torques[3] = f32::NAN;
        let result = adapter.adapt_normalized_torque_intent(
            &intent,
            &state,
            HumanoidMorphology::Dmc21,
            ActuationMode::NormalizedPosition,
        );
        assert!(matches!(
            result,
            Err(ActuationAdaptationError::NonFiniteValue { index: 3 })
        ));
    }

    #[test]
    fn rejects_non_finite_state_used_for_position_adaptation() {
        let adapter = ActuationAdapter::default();
        let mut state = HumanoidState::standing();
        state.joint_angles[4] = f64::NAN;
        let intent = HumanoidCommand::from_raw(&vec![0.0; 21]);
        let result = adapter.adapt_normalized_torque_intent(
            &intent,
            &state,
            HumanoidMorphology::Dmc21,
            ActuationMode::NormalizedPosition,
        );
        assert!(matches!(
            result,
            Err(ActuationAdaptationError::NonFiniteStateValue { index: 4 })
        ));
    }

    #[test]
    fn clipping_count_is_preserved_without_detaching_mode() {
        let adapter = ActuationAdapter {
            max_position_step_rad: 0.2,
        };
        let mut state = HumanoidState::standing();
        let limits = HumanoidMorphology::Dmc21.joint_limits();
        state.joint_angles[0] = limits[0][1];
        let mut values = vec![0.0; 21];
        values[0] = 1.0;
        let intent = HumanoidCommand::from_raw(&values);

        let adapted = adapter
            .adapt_normalized_torque_intent(
                &intent,
                &state,
                HumanoidMorphology::Dmc21,
                ActuationMode::PositionTargetRadians,
            )
            .unwrap();

        assert_eq!(adapted.clipped_joints(), 1);
        assert_eq!(adapted.target_mode(), ActuationMode::PositionTargetRadians);
        assert!((adapted.command().values()[0] as f64 - limits[0][1]).abs() < 1.0e-6);
    }
}
