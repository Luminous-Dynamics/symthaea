// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit physical semantics for the PCA9685 hobby-servo backend.
//!
//! The PCA9685 path drives PWM pulse targets for position-controlled hobby
//! servos. It is therefore a position-actuated backend, not a torque actuator.
//! This module provides a typed admission boundary for new callers while the
//! legacy `HumanoidCommand`-based runtime remains available for compatibility.

use symthaea_humanoid::{
    ActuationMode, HumanoidCommand, PhysicalActuationCommand, types::NUM_ACTUATORS,
};

use crate::error::{HalError, HalResult};

/// Stable semantic profile for the current 21-axis PCA9685 bench embodiment.
pub const PCA9685_POSITION_PROFILE_ID: &str = "symthaea.humanoid.pca9685-position-21.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pca9685PositionCapabilities {
    pub profile_id: &'static str,
    pub actuation_mode: ActuationMode,
    pub actuator_count: usize,
    /// PWM register readback does not prove physical joint position.
    pub physical_position_feedback_required_for_position_claim: bool,
    /// The current hobby-servo path does not expose qualified joint torque feedback.
    pub torque_feedback_available: bool,
}

impl Default for Pca9685PositionCapabilities {
    fn default() -> Self {
        Self {
            profile_id: PCA9685_POSITION_PROFILE_ID,
            actuation_mode: ActuationMode::NormalizedPosition,
            actuator_count: NUM_ACTUATORS,
            physical_position_feedback_required_for_position_claim: true,
            torque_feedback_available: false,
        }
    }
}

/// Validate a typed command for the PCA9685 position backend and lower it into
/// the legacy numeric carrier consumed by the existing servo implementation.
///
/// The lowering is deliberately explicit: a normalized-torque command with the
/// same numeric values is rejected rather than silently reinterpreted as joint
/// position.
pub fn lower_pca9685_position_command(
    command: &PhysicalActuationCommand,
) -> HalResult<HumanoidCommand> {
    if command.mode != ActuationMode::NormalizedPosition {
        return Err(HalError::Safety(format!(
            "PCA9685 profile {} accepts {:?}, received {:?}",
            PCA9685_POSITION_PROFILE_ID,
            ActuationMode::NormalizedPosition,
            command.mode
        )));
    }
    command
        .validate_for(NUM_ACTUATORS)
        .map_err(|error| HalError::Safety(format!("invalid PCA9685 position command: {error:?}")))?;
    Ok(command.legacy_vector())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_declares_position_not_torque_semantics() {
        let capabilities = Pca9685PositionCapabilities::default();
        assert_eq!(
            capabilities.actuation_mode,
            ActuationMode::NormalizedPosition
        );
        assert!(!capabilities.torque_feedback_available);
        assert!(capabilities.physical_position_feedback_required_for_position_claim);
    }

    #[test]
    fn typed_normalized_position_is_admitted() {
        let command = PhysicalActuationCommand::new(
            ActuationMode::NormalizedPosition,
            vec![0.25; NUM_ACTUATORS],
            NUM_ACTUATORS,
        )
        .unwrap();
        let lowered = lower_pca9685_position_command(&command).unwrap();
        assert_eq!(lowered.torques, command.values);
    }

    #[test]
    fn numerically_identical_torque_command_is_rejected() {
        let command = PhysicalActuationCommand::new(
            ActuationMode::NormalizedTorque,
            vec![0.25; NUM_ACTUATORS],
            NUM_ACTUATORS,
        )
        .unwrap();
        let result = lower_pca9685_position_command(&command);
        assert!(result.is_err());
    }

    #[test]
    fn physical_torque_command_is_rejected() {
        let command = PhysicalActuationCommand::new(
            ActuationMode::TorqueNewtonMetres,
            vec![5.0; NUM_ACTUATORS],
            NUM_ACTUATORS,
        )
        .unwrap();
        let result = lower_pca9685_position_command(&command);
        assert!(result.is_err());
    }
}
