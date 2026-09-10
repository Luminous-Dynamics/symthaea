// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic wearer/frame fit calibration primitives.
//!
//! These records make anthropometric assumptions explicit before a controller
//! treats a reference frame as fitted to a wearer. They do not diagnose health,
//! prescribe a device, or certify physical fit.

use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WearerProfile {
    pub mass_kg: f64,
    pub standing_height_m: f64,
    pub torso_length_m: f64,
    pub left_arm_length_m: f64,
    pub right_arm_length_m: f64,
    pub left_leg_length_m: f64,
    pub right_leg_length_m: f64,
}

impl WearerProfile {
    pub fn reference_adult() -> Self {
        Self {
            mass_kg: 75.0,
            standing_height_m: 1.75,
            torso_length_m: 0.55,
            left_arm_length_m: 0.62,
            right_arm_length_m: 0.62,
            left_leg_length_m: 0.90,
            right_leg_length_m: 0.90,
        }
    }

    pub fn is_finite_and_positive(&self) -> bool {
        [
            self.mass_kg,
            self.standing_height_m,
            self.torso_length_m,
            self.left_arm_length_m,
            self.right_arm_length_m,
            self.left_leg_length_m,
            self.right_leg_length_m,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameFitEnvelope {
    pub min_mass_kg: f64,
    pub max_mass_kg: f64,
    pub min_height_m: f64,
    pub max_height_m: f64,
    /// Maximum accepted relative left/right limb-length mismatch.
    pub max_limb_asymmetry_fraction: f64,
    /// Maximum scale away from the reference geometry for any major segment group.
    pub max_segment_scale_deviation: f64,
}

impl Default for FrameFitEnvelope {
    fn default() -> Self {
        Self {
            min_mass_kg: 45.0,
            max_mass_kg: 120.0,
            min_height_m: 1.45,
            max_height_m: 2.05,
            max_limb_asymmetry_fraction: 0.12,
            max_segment_scale_deviation: 0.30,
        }
    }
}

impl FrameFitEnvelope {
    pub fn is_valid(&self) -> bool {
        self.min_mass_kg.is_finite()
            && self.max_mass_kg.is_finite()
            && self.min_mass_kg > 0.0
            && self.max_mass_kg > self.min_mass_kg
            && self.min_height_m.is_finite()
            && self.max_height_m.is_finite()
            && self.min_height_m > 0.0
            && self.max_height_m > self.min_height_m
            && self.max_limb_asymmetry_fraction.is_finite()
            && (0.0..1.0).contains(&self.max_limb_asymmetry_fraction)
            && self.max_segment_scale_deviation.is_finite()
            && (0.0..1.0).contains(&self.max_segment_scale_deviation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameCalibration {
    pub mass_ratio: f64,
    pub height_scale: f64,
    pub torso_scale: f64,
    pub left_arm_scale: f64,
    pub right_arm_scale: f64,
    pub left_leg_scale: f64,
    pub right_leg_scale: f64,
    pub arm_asymmetry_fraction: f64,
    pub leg_asymmetry_fraction: f64,
}

impl FrameCalibration {
    pub fn max_segment_scale_deviation(&self) -> f64 {
        [
            self.torso_scale,
            self.left_arm_scale,
            self.right_arm_scale,
            self.left_leg_scale,
            self.right_leg_scale,
        ]
        .into_iter()
        .map(|scale| (scale - 1.0).abs())
        .fold(0.0, f64::max)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationError {
    InvalidWearerProfile,
    InvalidFitEnvelope,
    MassOutsideEnvelope { mass_kg: f64 },
    HeightOutsideEnvelope { height_m: f64 },
    LimbAsymmetryOutsideEnvelope { fraction: f64 },
    SegmentScaleOutsideEnvelope { deviation: f64 },
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWearerProfile => write!(f, "wearer profile contains invalid measurements"),
            Self::InvalidFitEnvelope => write!(f, "frame fit envelope is invalid"),
            Self::MassOutsideEnvelope { mass_kg } => {
                write!(f, "wearer mass {mass_kg} kg is outside frame envelope")
            }
            Self::HeightOutsideEnvelope { height_m } => {
                write!(f, "wearer height {height_m} m is outside frame envelope")
            }
            Self::LimbAsymmetryOutsideEnvelope { fraction } => write!(
                f,
                "wearer limb asymmetry {fraction} exceeds frame envelope"
            ),
            Self::SegmentScaleOutsideEnvelope { deviation } => write!(
                f,
                "wearer segment scale deviation {deviation} exceeds frame envelope"
            ),
        }
    }
}

impl Error for CalibrationError {}

fn relative_asymmetry(left: f64, right: f64) -> f64 {
    let mean = 0.5 * (left + right);
    (left - right).abs() / mean.max(f64::EPSILON)
}

/// Calibrate one wearer against an explicit reference geometry and frame envelope.
pub fn calibrate_wearer(
    wearer: WearerProfile,
    reference: WearerProfile,
    envelope: FrameFitEnvelope,
) -> Result<FrameCalibration, CalibrationError> {
    if !wearer.is_finite_and_positive() || !reference.is_finite_and_positive() {
        return Err(CalibrationError::InvalidWearerProfile);
    }
    if !envelope.is_valid() {
        return Err(CalibrationError::InvalidFitEnvelope);
    }
    if !(envelope.min_mass_kg..=envelope.max_mass_kg).contains(&wearer.mass_kg) {
        return Err(CalibrationError::MassOutsideEnvelope {
            mass_kg: wearer.mass_kg,
        });
    }
    if !(envelope.min_height_m..=envelope.max_height_m).contains(&wearer.standing_height_m) {
        return Err(CalibrationError::HeightOutsideEnvelope {
            height_m: wearer.standing_height_m,
        });
    }

    let arm_asymmetry_fraction =
        relative_asymmetry(wearer.left_arm_length_m, wearer.right_arm_length_m);
    let leg_asymmetry_fraction =
        relative_asymmetry(wearer.left_leg_length_m, wearer.right_leg_length_m);
    let max_asymmetry = arm_asymmetry_fraction.max(leg_asymmetry_fraction);
    if max_asymmetry > envelope.max_limb_asymmetry_fraction {
        return Err(CalibrationError::LimbAsymmetryOutsideEnvelope {
            fraction: max_asymmetry,
        });
    }

    let calibration = FrameCalibration {
        mass_ratio: wearer.mass_kg / reference.mass_kg,
        height_scale: wearer.standing_height_m / reference.standing_height_m,
        torso_scale: wearer.torso_length_m / reference.torso_length_m,
        left_arm_scale: wearer.left_arm_length_m / reference.left_arm_length_m,
        right_arm_scale: wearer.right_arm_length_m / reference.right_arm_length_m,
        left_leg_scale: wearer.left_leg_length_m / reference.left_leg_length_m,
        right_leg_scale: wearer.right_leg_length_m / reference.right_leg_length_m,
        arm_asymmetry_fraction,
        leg_asymmetry_fraction,
    };
    let deviation = calibration.max_segment_scale_deviation();
    if deviation > envelope.max_segment_scale_deviation {
        return Err(CalibrationError::SegmentScaleOutsideEnvelope { deviation });
    }
    Ok(calibration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_wearer_calibrates_to_unity() {
        let reference = WearerProfile::reference_adult();
        let calibration = calibrate_wearer(reference, reference, FrameFitEnvelope::default())
            .expect("reference wearer should fit reference frame");
        assert_eq!(calibration.mass_ratio, 1.0);
        assert_eq!(calibration.height_scale, 1.0);
        assert_eq!(calibration.max_segment_scale_deviation(), 0.0);
    }

    #[test]
    fn out_of_envelope_height_fails_closed() {
        let mut wearer = WearerProfile::reference_adult();
        wearer.standing_height_m = 2.5;
        assert!(matches!(
            calibrate_wearer(
                wearer,
                WearerProfile::reference_adult(),
                FrameFitEnvelope::default()
            ),
            Err(CalibrationError::HeightOutsideEnvelope { .. })
        ));
    }

    #[test]
    fn excessive_asymmetry_fails_closed() {
        let mut wearer = WearerProfile::reference_adult();
        wearer.left_leg_length_m = 0.65;
        wearer.right_leg_length_m = 0.95;
        assert!(matches!(
            calibrate_wearer(
                wearer,
                WearerProfile::reference_adult(),
                FrameFitEnvelope::default()
            ),
            Err(CalibrationError::LimbAsymmetryOutsideEnvelope { .. })
        ));
    }

    #[test]
    fn same_measurements_produce_same_calibration() {
        let wearer = WearerProfile {
            mass_kg: 82.0,
            standing_height_m: 1.82,
            torso_length_m: 0.58,
            left_arm_length_m: 0.65,
            right_arm_length_m: 0.65,
            left_leg_length_m: 0.94,
            right_leg_length_m: 0.94,
        };
        let a = calibrate_wearer(
            wearer,
            WearerProfile::reference_adult(),
            FrameFitEnvelope::default(),
        );
        let b = calibrate_wearer(
            wearer,
            WearerProfile::reference_adult(),
            FrameFitEnvelope::default(),
        );
        assert_eq!(a, b);
    }
}
