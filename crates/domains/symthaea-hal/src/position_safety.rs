// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mode-specific safety admission for absolute humanoid position targets.
//!
//! This module intentionally contains **no second e-stop, watchdog, or sensor
//! trip state**. Those belong to the shared hardware-safety core. Its only
//! theorem is that an exact [`PositionTargetRadiansCommand`] lies inside an
//! explicitly supplied per-actuator absolute-radian command envelope.
//!
//! Admission never clamps or scalar-scales the target. In particular, degraded
//! cognition/prediction error must not turn an absolute target `q` into
//! `gain * q`: that changes the requested physical position rather than merely
//! reducing command authority.

use symthaea_humanoid::types::NUM_ACTUATORS;

use crate::actuation::PositionTargetRadiansCommand;
use crate::error::{HalError, HalResult};

/// Per-actuator absolute joint-position command limits in radians.
///
/// These are **safety operating limits**, not calibration/mechanical endpoints.
/// A safety envelope may be strictly narrower than the calibration range.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionTargetEnvelope {
    min_radians: [f32; NUM_ACTUATORS],
    max_radians: [f32; NUM_ACTUATORS],
}

impl PositionTargetEnvelope {
    /// Construct and validate explicit per-actuator safety limits.
    pub fn try_new(
        min_radians: [f32; NUM_ACTUATORS],
        max_radians: [f32; NUM_ACTUATORS],
    ) -> HalResult<Self> {
        for index in 0..NUM_ACTUATORS {
            let min = min_radians[index];
            let max = max_radians[index];
            if !min.is_finite() || !max.is_finite() {
                return Err(HalError::Safety(format!(
                    "position safety envelope joint {index} has non-finite bounds: min={min}, max={max}"
                )));
            }
            if min > max {
                return Err(HalError::Safety(format!(
                    "position safety envelope joint {index} has inverted bounds: min={min}, max={max}"
                )));
            }
        }
        Ok(Self {
            min_radians,
            max_radians,
        })
    }

    /// Borrow the lower absolute-radian limits.
    pub fn min_radians(&self) -> &[f32; NUM_ACTUATORS] {
        &self.min_radians
    }

    /// Borrow the upper absolute-radian limits.
    pub fn max_radians(&self) -> &[f32; NUM_ACTUATORS] {
        &self.max_radians
    }

    /// Admit a target only if every actuator is already inside the configured
    /// absolute-radian safety envelope.
    ///
    /// The target is never clamped or otherwise rewritten.
    pub fn admit(
        &self,
        command: &PositionTargetRadiansCommand,
    ) -> HalResult<EnvelopeAdmittedPositionCommand> {
        for (index, target) in command.values().iter().copied().enumerate() {
            let min = self.min_radians[index];
            let max = self.max_radians[index];
            if target < min || target > max {
                return Err(HalError::Safety(format!(
                    "position target joint {index} is outside safety envelope: target={target} rad, allowed=[{min}, {max}] rad"
                )));
            }
        }
        Ok(EnvelopeAdmittedPositionCommand {
            command: command.clone(),
        })
    }
}

/// A position target whose values were admitted unchanged by one supplied
/// [`PositionTargetEnvelope`].
///
/// Construction is private. This wrapper proves only value-in-envelope under
/// the envelope object used for admission. It does not prove that envelope's
/// identity/currentness/qualification, physical state, calibration, authority,
/// workspace safety, or execution success.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvelopeAdmittedPositionCommand {
    command: PositionTargetRadiansCommand,
}

impl EnvelopeAdmittedPositionCommand {
    /// Borrow the unchanged semantic command.
    pub fn command(&self) -> &PositionTargetRadiansCommand {
        &self.command
    }

    /// Consume the admission wrapper and recover the unchanged command.
    pub fn into_command(self) -> PositionTargetRadiansCommand {
        self.command
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::CalibrationProfile;

    fn envelope(min: f32, max: f32) -> PositionTargetEnvelope {
        PositionTargetEnvelope::try_new(
            [min; NUM_ACTUATORS],
            [max; NUM_ACTUATORS],
        )
        .unwrap()
    }

    #[test]
    fn admits_in_range_target_without_rewriting_values() {
        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[0] = -0.4;
        values[1] = 0.7;
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();
        let admitted = envelope(-1.0, 1.0).admit(&command).unwrap();

        assert_eq!(admitted.command().values(), command.values());
        assert_eq!(admitted.into_command(), command);
    }

    #[test]
    fn rejects_target_above_one_joint_limit_instead_of_clamping() {
        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[7] = 0.500_1;
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();
        assert!(envelope(-0.5, 0.5).admit(&command).is_err());
        assert_eq!(command.values()[7], 0.500_1);
    }

    #[test]
    fn rejects_target_below_one_joint_limit() {
        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[4] = -0.500_1;
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();
        assert!(envelope(-0.5, 0.5).admit(&command).is_err());
    }

    #[test]
    fn rejects_non_finite_and_inverted_envelopes() {
        let mut mins = [-1.0_f32; NUM_ACTUATORS];
        let maxs = [1.0_f32; NUM_ACTUATORS];
        mins[2] = f32::NAN;
        assert!(PositionTargetEnvelope::try_new(mins, maxs).is_err());

        let mut mins = [-1.0_f32; NUM_ACTUATORS];
        let mut maxs = [1.0_f32; NUM_ACTUATORS];
        mins[3] = 0.4;
        maxs[3] = 0.3;
        assert!(PositionTargetEnvelope::try_new(mins, maxs).is_err());
    }

    #[test]
    fn asymmetric_limits_are_supported() {
        let mins = [-0.25_f32; NUM_ACTUATORS];
        let maxs = [0.75_f32; NUM_ACTUATORS];
        let envelope = PositionTargetEnvelope::try_new(mins, maxs).unwrap();

        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[0] = -0.25;
        values[1] = 0.75;
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();
        assert!(envelope.admit(&command).is_ok());
    }

    #[test]
    fn safety_envelope_can_be_narrower_than_calibration_range() {
        let calibration = CalibrationProfile::default_21();
        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[0] = 1.0; // inside default +/-90 degree calibration range
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();

        assert!(
            calibration
                .position_targets_radians_to_pulses(&command)
                .is_ok()
        );
        assert!(envelope(-0.5, 0.5).admit(&command).is_err());
    }

    #[test]
    fn boundary_admission_is_inclusive() {
        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[0] = -0.5;
        values[1] = 0.5;
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();
        assert!(envelope(-0.5, 0.5).admit(&command).is_ok());
    }
}
