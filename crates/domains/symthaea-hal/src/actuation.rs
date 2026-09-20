// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit command semantics for physical HAL actuation.
//!
//! The current PCA9685 hobby-servo backend ultimately drives position through
//! per-joint calibration. Two position representations are intentionally kept
//! distinct here:
//!
//! - [`NormalizedPositionCommand`] is **HAL-calibration-relative**. `-1` and
//!   `+1` refer to the exact calibrated endpoints of this backend/profile.
//! - [`PositionTargetRadiansCommand`] is an absolute semantic joint-position
//!   target in radians and is the preferred cross-layer representation.
//!
//! A morphology-normalized position value is not automatically equivalent to a
//! calibration-normalized position value. Cross-layer adapters therefore use
//! radians unless an exact normalization-profile equivalence is separately
//! proven.
//!
//! This module deliberately does not implement `From<HumanoidCommand>` or
//! `TryFrom<HumanoidCommand>`. `HumanoidCommand` has canonical normalized-torque
//! semantics, so an automatic conversion would recreate semantic laundering.

use symthaea_humanoid::actuation::AdaptedActuationCommand;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_humanoid::types::{ActuationMode, NUM_ACTUATORS};

use crate::error::{HalError, HalResult};

/// A complete calibration-relative normalized-position demand for the DMC21 HAL.
///
/// Each value is unitless and must be finite and within `[-1, +1]`. The
/// normalization basis is the exact HAL calibration profile: `-1` denotes that
/// joint's calibrated minimum, `0` the calibrated midpoint, and `+1` the
/// calibrated maximum.
///
/// This is **not** a universal normalized-position representation. In
/// particular, values normalized against humanoid morphology limits must not be
/// converted into this type merely because they also lie in `[-1, +1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedPositionCommand {
    values: [f32; NUM_ACTUATORS],
}

impl NormalizedPositionCommand {
    /// The semantic mode carried by this command type.
    pub const ACTUATION_MODE: ActuationMode = ActuationMode::NormalizedPosition;

    /// Validate and construct a complete calibration-relative command.
    pub fn try_from_array(values: [f32; NUM_ACTUATORS]) -> HalResult<Self> {
        validate_normalized_values(&values)?;
        Ok(Self { values })
    }

    /// Validate and construct from a dynamically sized slice.
    pub fn try_from_slice(values: &[f32]) -> HalResult<Self> {
        if values.len() != NUM_ACTUATORS {
            return Err(HalError::Safety(format!(
                "normalized-position command has {} actuators; expected exactly {}",
                values.len(),
                NUM_ACTUATORS
            )));
        }

        let mut array = [0.0_f32; NUM_ACTUATORS];
        array.copy_from_slice(values);
        Self::try_from_array(array)
    }

    /// A zero/neutral calibration-relative demand.
    pub fn zero() -> Self {
        Self {
            values: [0.0; NUM_ACTUATORS],
        }
    }

    /// Borrow the validated command values.
    pub fn values(&self) -> &[f32; NUM_ACTUATORS] {
        &self.values
    }

    /// Return the explicit actuation semantics for this command.
    pub const fn actuation_mode(&self) -> ActuationMode {
        Self::ACTUATION_MODE
    }
}

/// A complete absolute DMC21 joint-position target expressed in radians.
///
/// This type validates quantity/mode/shape only. It deliberately does not
/// clamp against morphology or calibration limits at construction: a caller
/// must receive an explicit admission/rejection disposition from the layer that
/// owns those exact limits rather than silently changing the requested target.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionTargetRadiansCommand {
    values: [f32; NUM_ACTUATORS],
}

impl PositionTargetRadiansCommand {
    /// The semantic mode carried by this command type.
    pub const ACTUATION_MODE: ActuationMode = ActuationMode::PositionTargetRadians;

    /// Validate and construct a complete absolute-radian command.
    pub fn try_from_array(values: [f32; NUM_ACTUATORS]) -> HalResult<Self> {
        validate_finite_values(&values, "position-target-radians")?;
        Ok(Self { values })
    }

    /// Validate and construct from a dynamically sized slice.
    pub fn try_from_slice(values: &[f32]) -> HalResult<Self> {
        if values.len() != NUM_ACTUATORS {
            return Err(HalError::Safety(format!(
                "position-target-radians command has {} actuators; expected exactly {}",
                values.len(),
                NUM_ACTUATORS
            )));
        }

        let mut array = [0.0_f32; NUM_ACTUATORS];
        array.copy_from_slice(values);
        Self::try_from_array(array)
    }

    /// Accept a humanoid adaptation only when its morphology and semantic mode
    /// exactly match this physical HAL boundary.
    ///
    /// This proves only semantic compatibility. It does not prove the
    /// `HumanoidState` used upstream was physical/current, nor that calibration,
    /// authority, or safety admission is current.
    pub fn try_from_adapted(command: &AdaptedActuationCommand) -> HalResult<Self> {
        if command.morphology() != HumanoidMorphology::Dmc21 {
            return Err(HalError::Safety(format!(
                "position-target-radians HAL requires Dmc21 morphology, got {:?}",
                command.morphology()
            )));
        }
        if command.mode() != Self::ACTUATION_MODE {
            return Err(HalError::Safety(format!(
                "position-target-radians HAL requires {:?}, got {:?}",
                Self::ACTUATION_MODE,
                command.mode()
            )));
        }
        Self::try_from_slice(command.values())
    }

    /// Borrow the validated absolute-radian targets.
    pub fn values(&self) -> &[f32; NUM_ACTUATORS] {
        &self.values
    }

    /// Return the explicit actuation semantics for this command.
    pub const fn actuation_mode(&self) -> ActuationMode {
        Self::ACTUATION_MODE
    }
}

impl TryFrom<&AdaptedActuationCommand> for PositionTargetRadiansCommand {
    type Error = HalError;

    fn try_from(command: &AdaptedActuationCommand) -> Result<Self, Self::Error> {
        Self::try_from_adapted(command)
    }
}

fn validate_normalized_values(values: &[f32; NUM_ACTUATORS]) -> HalResult<()> {
    validate_finite_values(values, "normalized-position")?;
    for (index, value) in values.iter().copied().enumerate() {
        if !(-1.0..=1.0).contains(&value) {
            return Err(HalError::Safety(format!(
                "normalized-position command actuator {index} is outside [-1, 1]: {value}"
            )));
        }
    }
    Ok(())
}

fn validate_finite_values(
    values: &[f32; NUM_ACTUATORS],
    command_name: &str,
) -> HalResult<()> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(HalError::Safety(format!(
                "{command_name} command contains non-finite value at actuator {index}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_humanoid::actuation::ActuationAdapter;
    use symthaea_humanoid::types::{HumanoidCommand, HumanoidState};

    #[test]
    fn advertises_calibration_relative_normalized_position_semantics() {
        let command = NormalizedPositionCommand::zero();
        assert_eq!(
            command.actuation_mode(),
            ActuationMode::NormalizedPosition
        );
    }

    #[test]
    fn normalized_position_accepts_closed_normalized_range() {
        let mut values = [0.0; NUM_ACTUATORS];
        values[0] = -1.0;
        values[1] = 1.0;
        let command = NormalizedPositionCommand::try_from_array(values).unwrap();
        assert_eq!(command.values()[0], -1.0);
        assert_eq!(command.values()[1], 1.0);
    }

    #[test]
    fn normalized_position_rejects_out_of_range_value_instead_of_clamping() {
        let mut values = [0.0; NUM_ACTUATORS];
        values[7] = 1.01;
        assert!(NormalizedPositionCommand::try_from_array(values).is_err());
    }

    #[test]
    fn normalized_position_rejects_non_finite_value() {
        let mut values = [0.0; NUM_ACTUATORS];
        values[3] = f32::NAN;
        assert!(NormalizedPositionCommand::try_from_array(values).is_err());
    }

    #[test]
    fn normalized_position_rejects_wrong_dynamic_actuator_count() {
        assert!(NormalizedPositionCommand::try_from_slice(&[0.0; NUM_ACTUATORS - 1]).is_err());
        assert!(NormalizedPositionCommand::try_from_slice(&[0.0; NUM_ACTUATORS + 1]).is_err());
    }

    #[test]
    fn radians_command_accepts_finite_absolute_targets() {
        let mut values = [0.0; NUM_ACTUATORS];
        values[0] = -0.75;
        values[1] = 1.25;
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();
        assert_eq!(command.actuation_mode(), ActuationMode::PositionTargetRadians);
        assert_eq!(command.values()[0], -0.75);
        assert_eq!(command.values()[1], 1.25);
    }

    #[test]
    fn radians_command_rejects_non_finite_targets() {
        let mut values = [0.0; NUM_ACTUATORS];
        values[8] = f32::INFINITY;
        assert!(PositionTargetRadiansCommand::try_from_array(values).is_err());
    }

    #[test]
    fn accepts_only_dmc21_radian_adaptation() {
        let adapter = ActuationAdapter::default();
        let state = HumanoidState::standing();
        let intent = HumanoidCommand::from_raw(&vec![0.25; NUM_ACTUATORS]);
        let adapted = adapter
            .adapt_normalized_torque_intent(
                &intent,
                &state,
                HumanoidMorphology::Dmc21,
                ActuationMode::PositionTargetRadians,
            )
            .unwrap();

        let command = PositionTargetRadiansCommand::try_from(adapted.command()).unwrap();
        assert_eq!(command.actuation_mode(), ActuationMode::PositionTargetRadians);
        assert_eq!(command.values(), adapted.command().values());
    }

    #[test]
    fn rejects_normalized_position_adaptation_even_though_values_are_bounded() {
        let adapter = ActuationAdapter::default();
        let state = HumanoidState::standing();
        let intent = HumanoidCommand::from_raw(&vec![0.25; NUM_ACTUATORS]);
        let adapted = adapter
            .adapt_normalized_torque_intent(
                &intent,
                &state,
                HumanoidMorphology::Dmc21,
                ActuationMode::NormalizedPosition,
            )
            .unwrap();

        assert!(PositionTargetRadiansCommand::try_from(adapted.command()).is_err());
    }

    #[test]
    fn rejects_torque_adaptation_at_position_boundary() {
        let adapter = ActuationAdapter::default();
        let state = HumanoidState::standing();
        let intent = HumanoidCommand::from_raw(&vec![0.25; NUM_ACTUATORS]);
        let adapted = adapter
            .adapt_normalized_torque_intent(
                &intent,
                &state,
                HumanoidMorphology::Dmc21,
                ActuationMode::NormalizedTorque,
            )
            .unwrap();

        assert!(PositionTargetRadiansCommand::try_from(adapted.command()).is_err());
    }
}
