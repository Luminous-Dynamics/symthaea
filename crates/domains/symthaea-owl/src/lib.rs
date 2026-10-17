// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal physical-geometry primitives for an owl-inspired embodied robot.
//!
//! This crate currently provides one deliberately bounded primitive: a signed
//! two-microphone time-difference-of-arrival (TDOA) to azimuth conversion for a
//! far-field source in the microphones' forward hemisphere. It is not a signal
//! processor, a 3-D localizer, or a model of biological owl hearing.
//!
//! Convention: right_minus_left_delay_seconds = t_right - t_left. The
//! microphone baseline is horizontal and orthogonal to the array's forward
//! axis. Positive azimuth is to the right of that axis; a source to the right
//! reaches the right microphone first, so its TDOA is negative.
//!
//! For spacing d, sound speed c, and azimuth theta:
//! t_right - t_left = -(d / c) * sin(theta).
//! The inverse is only unique for the declared front hemisphere,
//! theta in [-pi/2, pi/2]. A single pair does not distinguish front/back or
//! provide elevation; reverberation, calibration error and finite range need
//! separate treatment.

pub mod spatial_likelihood;

use core::fmt;

/// Errors returned by the bounded far-field two-microphone geometry model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TdoaGeometryError {
    /// One or more inputs were NaN or infinite.
    NonFiniteInput,
    /// Microphone spacing must be strictly positive.
    NonPositiveSpacing,
    /// Speed of sound must be strictly positive.
    NonPositiveSoundSpeed,
    /// Finite inputs produced a non-finite or underflowed physical delay bound.
    InvalidDerivedGeometry,
    /// The measured lag exceeds the maximum path-delay allowed by spacing.
    PhysicallyImpossibleDelay {
        measured_delay_seconds: f64,
        maximum_absolute_delay_seconds: f64,
    },
}

impl fmt::Display for TdoaGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput => write!(f, "TDOA geometry inputs must be finite"),
            Self::NonPositiveSpacing => write!(f, "microphone spacing must be positive"),
            Self::NonPositiveSoundSpeed => write!(f, "speed of sound must be positive"),
            Self::InvalidDerivedGeometry => {
                write!(f, "inputs do not produce a finite positive physical delay bound")
            }
            Self::PhysicallyImpossibleDelay {
                measured_delay_seconds,
                maximum_absolute_delay_seconds,
            } => write!(
                f,
                "measured TDOA {measured_delay_seconds} s exceeds physical bound ±{maximum_absolute_delay_seconds} s"
            ),
        }
    }
}

impl std::error::Error for TdoaGeometryError {}

/// A bearing estimate under the limited geometry stated by bearing_from_tdoa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinauralBearing {
    /// Estimated azimuth from the forward axis, positive to the robot's right.
    pub azimuth_radians: f64,
    /// Estimated azimuth in degrees, positive to the robot's right.
    pub azimuth_degrees: f64,
    /// Input delay using the documented right-channel-minus-left-channel convention.
    pub measured_tdoa_seconds: f64,
    /// Maximum absolute far-field delay magnitude for this microphone spacing/sound speed.
    pub maximum_absolute_tdoa_seconds: f64,
}

/// Convert a signed microphone time difference into a far-field azimuth estimate.
///
/// The caller must supply a calibrated TDOA estimate, calibrated baseline
/// spacing (metres), and sound speed (metres/second). Positive TDOA means the
/// right channel arrives later; that corresponds to a source on the left under
/// this convention. The function intentionally errors on a lag outside the
/// physical bound instead of clamping it to an apparently confident direction.
///
/// # Limitations
///
/// This is geometry only. It assumes a far-field source, a known horizontal
/// baseline orthogonal to the forward axis, synchronized/calibrated channels,
/// and a source in the forward hemisphere. It does not handle TDOA extraction,
/// reverberation, source motion, microphone frequency response, front/back
/// ambiguity, elevation, or uncertainty calibration.
pub fn bearing_from_tdoa(
    right_minus_left_delay_seconds: f64,
    microphone_spacing_m: f64,
    speed_of_sound_m_s: f64,
) -> Result<BinauralBearing, TdoaGeometryError> {
    if !right_minus_left_delay_seconds.is_finite()
        || !microphone_spacing_m.is_finite()
        || !speed_of_sound_m_s.is_finite()
    {
        return Err(TdoaGeometryError::NonFiniteInput);
    }
    if microphone_spacing_m <= 0.0 {
        return Err(TdoaGeometryError::NonPositiveSpacing);
    }
    if speed_of_sound_m_s <= 0.0 {
        return Err(TdoaGeometryError::NonPositiveSoundSpeed);
    }

    let max_delay = microphone_spacing_m / speed_of_sound_m_s;
    if !max_delay.is_finite() || max_delay <= 0.0 {
        return Err(TdoaGeometryError::InvalidDerivedGeometry);
    }
    if right_minus_left_delay_seconds.abs() > max_delay {
        return Err(TdoaGeometryError::PhysicallyImpossibleDelay {
            measured_delay_seconds: right_minus_left_delay_seconds,
            maximum_absolute_delay_seconds: max_delay,
        });
    }

    // Dividing by the physical maximum delay keeps the inverse within [-1, 1]
    // after the strict physical-bound check.
    let sin_azimuth = -(right_minus_left_delay_seconds / max_delay);
    let azimuth_radians = sin_azimuth.asin();
    let azimuth_degrees = azimuth_radians.to_degrees();

    Ok(BinauralBearing {
        azimuth_radians,
        azimuth_degrees,
        measured_tdoa_seconds: right_minus_left_delay_seconds,
        maximum_absolute_tdoa_seconds: max_delay,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOUND_SPEED_M_S: f64 = 343.0;
    const SPACING_M: f64 = 0.10;

    fn delay_for_azimuth(azimuth_degrees: f64) -> f64 {
        -(SPACING_M / SOUND_SPEED_M_S) * azimuth_degrees.to_radians().sin()
    }

    #[test]
    fn zero_delay_is_broadside() {
        let bearing = bearing_from_tdoa(0.0, SPACING_M, SOUND_SPEED_M_S).unwrap();
        assert!(bearing.azimuth_radians.abs() < 1e-12);
        assert!(bearing.azimuth_degrees.abs() < 1e-10);
    }

    #[test]
    fn negative_tdoa_means_source_to_the_right() {
        let delay = delay_for_azimuth(30.0);
        let bearing = bearing_from_tdoa(delay, SPACING_M, SOUND_SPEED_M_S).unwrap();
        assert!((bearing.azimuth_degrees - 30.0).abs() < 1e-9);
    }

    #[test]
    fn positive_tdoa_means_source_to_the_left() {
        let delay = delay_for_azimuth(-20.0);
        let bearing = bearing_from_tdoa(delay, SPACING_M, SOUND_SPEED_M_S).unwrap();
        assert!((bearing.azimuth_degrees + 20.0).abs() < 1e-9);
    }

    #[test]
    fn physical_delay_boundary_maps_to_broadside_edges() {
        let max_delay = SPACING_M / SOUND_SPEED_M_S;
        let right = bearing_from_tdoa(-max_delay, SPACING_M, SOUND_SPEED_M_S).unwrap();
        let left = bearing_from_tdoa(max_delay, SPACING_M, SOUND_SPEED_M_S).unwrap();
        assert!((right.azimuth_degrees - 90.0).abs() < 1e-8);
        assert!((left.azimuth_degrees + 90.0).abs() < 1e-8);
    }

    #[test]
    fn impossible_delay_is_rejected_not_clamped() {
        let max_delay = SPACING_M / SOUND_SPEED_M_S;
        let result = bearing_from_tdoa(max_delay * 1.01, SPACING_M, SOUND_SPEED_M_S);
        assert!(matches!(
            result,
            Err(TdoaGeometryError::PhysicallyImpossibleDelay { .. })
        ));
    }

    #[test]
    fn zero_spacing_is_rejected() {
        assert_eq!(
            bearing_from_tdoa(0.0, 0.0, SOUND_SPEED_M_S),
            Err(TdoaGeometryError::NonPositiveSpacing)
        );
    }

    #[test]
    fn zero_sound_speed_is_rejected() {
        assert_eq!(
            bearing_from_tdoa(0.0, SPACING_M, 0.0),
            Err(TdoaGeometryError::NonPositiveSoundSpeed)
        );
    }

    #[test]
    fn non_finite_inputs_are_rejected() {
        assert_eq!(
            bearing_from_tdoa(f64::NAN, SPACING_M, SOUND_SPEED_M_S),
            Err(TdoaGeometryError::NonFiniteInput)
        );
        assert_eq!(
            bearing_from_tdoa(0.0, f64::INFINITY, SOUND_SPEED_M_S),
            Err(TdoaGeometryError::NonFiniteInput)
        );
    }

    #[test]
    fn invalid_derived_delay_bound_is_rejected() {
        let result = bearing_from_tdoa(0.0, f64::MAX, f64::MIN_POSITIVE);
        assert_eq!(result, Err(TdoaGeometryError::InvalidDerivedGeometry));
    }
}
