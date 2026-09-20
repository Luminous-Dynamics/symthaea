// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict absolute-position calibration for the DMC21 PWM backend.
//!
//! This module is intentionally separate from the legacy normalized
//! `torque_to_*` helpers in `calibration.rs`. It consumes only an explicitly
//! typed [`PositionTargetRadiansCommand`] and maps semantic physical joint
//! angles to electrical PWM pulse widths.
//!
//! Servo reversal belongs to the physical-angle → electrical-pulse mapping:
//! reversing a servo changes which pulse realizes a physical angle; it does not
//! change the semantic angle requested by the caller.

use symthaea_humanoid::types::NUM_ACTUATORS;

use crate::actuation::PositionTargetRadiansCommand;
use crate::calibration::{CalibrationProfile, JointCalibration};
use crate::error::{HalError, HalResult};

impl CalibrationProfile {
    /// Convert a complete absolute-radian joint-position command to PWM pulses.
    ///
    /// This path is intentionally strict:
    ///
    /// - the calibration profile is structurally validated first;
    /// - every target must be finite;
    /// - radians are converted explicitly to degrees because the legacy
    ///   calibration document currently stores degree-valued endpoints;
    /// - a target outside the calibrated physical range is rejected rather than
    ///   silently clamped;
    /// - `reversed` changes the electrical pulse direction while preserving the
    ///   semantic physical target angle.
    ///
    /// A successful conversion proves only deterministic conversion under these
    /// calibration bytes. It does not prove that the calibration is qualified,
    /// current, or bound to the connected device.
    pub fn position_targets_radians_to_pulses(
        &self,
        command: &PositionTargetRadiansCommand,
    ) -> HalResult<[u16; NUM_ACTUATORS]> {
        self.validate()?;

        let mut pulses = [0_u16; NUM_ACTUATORS];
        for (index, (&position_rad, joint)) in command
            .values()
            .iter()
            .zip(self.joints.iter())
            .enumerate()
        {
            pulses[index] = position_radians_to_pulse_us(joint, index, position_rad)?;
        }
        Ok(pulses)
    }
}

fn position_radians_to_pulse_us(
    calibration: &JointCalibration,
    index: usize,
    position_rad: f32,
) -> HalResult<u16> {
    if !position_rad.is_finite() {
        return Err(HalError::Safety(format!(
            "non-finite position target for joint {index} ({})",
            calibration.name
        )));
    }

    let angle_deg = position_rad.to_degrees();
    if angle_deg < calibration.angle_min_deg || angle_deg > calibration.angle_max_deg {
        return Err(HalError::Safety(format!(
            "position target for joint {index} ({}) is outside calibrated range: {angle_deg}deg not in [{}deg, {}deg]",
            calibration.name, calibration.angle_min_deg, calibration.angle_max_deg
        )));
    }

    let angle_range = calibration.angle_max_deg - calibration.angle_min_deg;
    debug_assert!(angle_range > 0.0);

    let physical_fraction = (angle_deg - calibration.angle_min_deg) / angle_range;
    let electrical_fraction = if calibration.reversed {
        1.0 - physical_fraction
    } else {
        physical_fraction
    };

    let pulse_span = calibration.pulse_max_us as f32 - calibration.pulse_min_us as f32;
    let pulse = calibration.pulse_min_us as f32 + electrical_fraction * pulse_span;

    // The clamp here protects only integer-rounding at an already-admitted
    // endpoint; semantic out-of-range targets were rejected above.
    Ok((pulse.round() as u16).clamp(calibration.pulse_min_us, calibration.pulse_max_us))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn default_profile_maps_absolute_radian_endpoints() {
        let profile = CalibrationProfile::default_21();
        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[0] = -FRAC_PI_2;
        values[1] = 0.0;
        values[2] = FRAC_PI_2;
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();

        let pulses = profile.position_targets_radians_to_pulses(&command).unwrap();
        assert_eq!(pulses[0], 500);
        assert_eq!(pulses[1], 1500);
        assert_eq!(pulses[2], 2500);
    }

    #[test]
    fn reversed_servo_changes_electrical_direction_not_semantic_angle() {
        let mut profile = CalibrationProfile::default_21();
        profile.joints[0].reversed = true;

        let mut positive = [0.0_f32; NUM_ACTUATORS];
        positive[0] = FRAC_PI_2;
        let positive = PositionTargetRadiansCommand::try_from_array(positive).unwrap();
        let positive_pulse = profile.position_targets_radians_to_pulses(&positive).unwrap()[0];

        let mut negative = [0.0_f32; NUM_ACTUATORS];
        negative[0] = -FRAC_PI_2;
        let negative = PositionTargetRadiansCommand::try_from_array(negative).unwrap();
        let negative_pulse = profile.position_targets_radians_to_pulses(&negative).unwrap()[0];

        assert_eq!(positive_pulse, 500);
        assert_eq!(negative_pulse, 2500);
    }

    #[test]
    fn rejects_target_outside_calibrated_range_instead_of_clamping() {
        let profile = CalibrationProfile::default_21();
        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[5] = FRAC_PI_2 + 0.01;
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();

        assert!(profile.position_targets_radians_to_pulses(&command).is_err());
    }

    #[test]
    fn custom_asymmetric_angle_range_uses_explicit_unit_conversion() {
        let mut profile = CalibrationProfile::default_21();
        profile.joints[0].angle_min_deg = -30.0;
        profile.joints[0].angle_max_deg = 60.0;

        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[0] = 15.0_f32.to_radians();
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();

        let pulse = profile.position_targets_radians_to_pulses(&command).unwrap()[0];
        assert_eq!(pulse, 1500);
    }

    #[test]
    fn calibration_range_is_independent_of_morphology_normalization() {
        let mut profile = CalibrationProfile::default_21();
        profile.joints[0].angle_min_deg = -10.0;
        profile.joints[0].angle_max_deg = 10.0;

        let mut values = [0.0_f32; NUM_ACTUATORS];
        values[0] = 0.5; // about 28.65 degrees: valid radians, outside this calibration.
        let command = PositionTargetRadiansCommand::try_from_array(values).unwrap();

        assert!(profile.position_targets_radians_to_pulses(&command).is_err());
    }
}
