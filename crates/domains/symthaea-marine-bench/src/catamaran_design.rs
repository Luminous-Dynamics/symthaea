// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unit-checked, physics-transparent catamaran geometry utilities.
//!
//! This module computes dimensionless geometry descriptors and Froude number only.
//! It does not estimate resistance, seakeeping, slamming, stability, strength,
//! energy use, safety, or an operating envelope. Those require declared methods
//! validated against comparable experimental/reference data.

use serde::Serialize;
use std::fmt;

/// Standard gravitational acceleration in metres per second squared.
pub const STANDARD_GRAVITY_M_S2: f64 = 9.806_65;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryField {
    LengthWaterlineM,
    OverallBeamM,
    DemihullBeamM,
    HullCenterlineSpacingM,
    DesignDraftM,
    WetDeckClearanceM,
    SpeedMps,
    SpacingRatioOverLwl,
}

impl fmt::Display for GeometryField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::LengthWaterlineM => "length_waterline_m",
            Self::OverallBeamM => "overall_beam_m",
            Self::DemihullBeamM => "demihull_beam_m",
            Self::HullCenterlineSpacingM => "hull_centerline_spacing_m",
            Self::DesignDraftM => "design_draft_m",
            Self::WetDeckClearanceM => "wet_deck_clearance_m",
            Self::SpeedMps => "speed_m_s",
            Self::SpacingRatioOverLwl => "hull_spacing_over_lwl",
        };
        f.write_str(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GeometryError {
    NonFinite(GeometryField),
    NotPositive(GeometryField),
    Negative(GeometryField),
    /// Supplied overall beam is smaller than centreline spacing plus one full
    /// demihull beam, which is the minimum outer-to-outer width for symmetric hulls.
    BeamEnvelopeInconsistent {
        minimum_overall_beam_m: f64,
        supplied_overall_beam_m: f64,
    },
    EmptySpacingSweep,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::NotPositive(field) => write!(f, "{field} must be greater than zero"),
            Self::Negative(field) => write!(f, "{field} must be non-negative"),
            Self::BeamEnvelopeInconsistent {
                minimum_overall_beam_m,
                supplied_overall_beam_m,
            } => write!(
                f,
                "overall beam {supplied_overall_beam_m} m is smaller than the minimum                  {minimum_overall_beam_m} m implied by hull spacing and demihull beam"
            ),
            Self::EmptySpacingSweep => write!(f, "at least one hull-spacing ratio is required"),
        }
    }
}

impl std::error::Error for GeometryError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatamaranGeometry {
    /// Waterline length in metres for a declared loading/trim condition.
    pub length_waterline_m: f64,
    /// Overall outer-to-outer beam in metres.
    pub overall_beam_m: f64,
    /// Beam of one demihull in metres, measured consistently with the stated waterline.
    pub demihull_beam_m: f64,
    /// Centreline-to-centreline distance in metres.
    pub hull_centerline_spacing_m: f64,
    /// Design draft in metres for the same declared loading condition.
    pub design_draft_m: f64,
    /// Static wet-deck clearance in metres at the declared loading condition.
    /// Dynamic clearance and slamming require a separate wave-response model.
    pub wet_deck_clearance_m: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DimensionlessGeometry {
    pub overall_beam_over_lwl: f64,
    pub demihull_beam_over_lwl: f64,
    pub hull_spacing_over_lwl: f64,
    pub demihull_slenderness_lwl_over_beam: f64,
    pub draft_over_lwl: f64,
    pub wet_deck_clearance_over_draft: f64,
    pub wet_deck_clearance_over_lwl: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullSpacingCandidate {
    pub centerline_spacing_ratio_over_lwl: f64,
    pub centerline_spacing_m: f64,
}

fn positive(field: GeometryField, value: f64) -> Result<f64, GeometryError> {
    if !value.is_finite() {
        return Err(GeometryError::NonFinite(field));
    }
    if value <= 0.0 {
        return Err(GeometryError::NotPositive(field));
    }
    Ok(value)
}

/// Calculates dimensionless geometry descriptors for consistent comparison.
///
/// This validates the most basic symmetric plan-view beam envelope. It does not
/// assert that the hull is structurally feasible, hydrostatically stable, buildable,
/// efficient, safe in waves, or approved for operation.
impl CatamaranGeometry {
    pub fn dimensionless(self) -> Result<DimensionlessGeometry, GeometryError> {
        let lwl = positive(GeometryField::LengthWaterlineM, self.length_waterline_m)?;
        let overall_beam = positive(GeometryField::OverallBeamM, self.overall_beam_m)?;
        let demihull_beam = positive(GeometryField::DemihullBeamM, self.demihull_beam_m)?;
        let spacing = positive(
            GeometryField::HullCenterlineSpacingM,
            self.hull_centerline_spacing_m,
        )?;
        let draft = positive(GeometryField::DesignDraftM, self.design_draft_m)?;
        let wet_deck = positive(GeometryField::WetDeckClearanceM, self.wet_deck_clearance_m)?;

        let minimum_beam = spacing + demihull_beam;
        if !minimum_beam.is_finite() || overall_beam + f64::EPSILON < minimum_beam {
            return Err(GeometryError::BeamEnvelopeInconsistent {
                minimum_overall_beam_m: minimum_beam,
                supplied_overall_beam_m: overall_beam,
            });
        }

        Ok(DimensionlessGeometry {
            overall_beam_over_lwl: overall_beam / lwl,
            demihull_beam_over_lwl: demihull_beam / lwl,
            hull_spacing_over_lwl: spacing / lwl,
            demihull_slenderness_lwl_over_beam: lwl / demihull_beam,
            draft_over_lwl: draft / lwl,
            wet_deck_clearance_over_draft: wet_deck / draft,
            wet_deck_clearance_over_lwl: wet_deck / lwl,
        })
    }
}

/// Converts a list of dimensionless centreline-spacing ratios into physical spacing
/// candidates. Ratios are caller-specified design-space inputs, not recommended values.
/// Candidate order is preserved for deterministic experiment manifests.
pub fn hull_spacing_sweep(
    length_waterline_m: f64,
    spacing_ratios_over_lwl: &[f64],
) -> Result<Vec<HullSpacingCandidate>, GeometryError> {
    let lwl = positive(GeometryField::LengthWaterlineM, length_waterline_m)?;
    if spacing_ratios_over_lwl.is_empty() {
        return Err(GeometryError::EmptySpacingSweep);
    }

    spacing_ratios_over_lwl
        .iter()
        .copied()
        .map(|ratio| {
            let ratio = positive(GeometryField::SpacingRatioOverLwl, ratio)?;
            let spacing = ratio * lwl;
            if !spacing.is_finite() {
                return Err(GeometryError::NonFinite(
                    GeometryField::HullCenterlineSpacingM,
                ));
            }
            Ok(HullSpacingCandidate {
                centerline_spacing_ratio_over_lwl: ratio,
                centerline_spacing_m: spacing,
            })
        })
        .collect()
}

/// Returns the length-based Froude number, Fr = speed / sqrt(g * L).
///
/// Speed is in metres per second and waterline length is in metres. A zero speed
/// is valid and yields zero; negative/non-finite speeds and non-positive lengths
/// are rejected. Froude number is a similarity descriptor, not a resistance model.
pub fn froude_number(
    speed_m_s: f64,
    length_waterline_m: f64,
) -> Result<f64, GeometryError> {
    if !speed_m_s.is_finite() {
        return Err(GeometryError::NonFinite(GeometryField::SpeedMps));
    }
    if speed_m_s < 0.0 {
        return Err(GeometryError::Negative(GeometryField::SpeedMps));
    }
    let lwl = positive(GeometryField::LengthWaterlineM, length_waterline_m)?;
    Ok(speed_m_s / (STANDARD_GRAVITY_M_S2 * lwl).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_geometry() -> CatamaranGeometry {
        CatamaranGeometry {
            length_waterline_m: 20.0,
            overall_beam_m: 10.5,
            demihull_beam_m: 2.5,
            hull_centerline_spacing_m: 8.0,
            design_draft_m: 1.2,
            wet_deck_clearance_m: 0.9,
        }
    }

    #[test]
    fn dimensionless_geometry_matches_hand_calculations() {
        let ratios = example_geometry().dimensionless().expect("valid geometry");
        assert!((ratios.overall_beam_over_lwl - 0.525).abs() < 1e-12);
        assert!((ratios.demihull_beam_over_lwl - 0.125).abs() < 1e-12);
        assert!((ratios.hull_spacing_over_lwl - 0.4).abs() < 1e-12);
        assert!((ratios.demihull_slenderness_lwl_over_beam - 8.0).abs() < 1e-12);
        assert!((ratios.draft_over_lwl - 0.06).abs() < 1e-12);
        assert!((ratios.wet_deck_clearance_over_draft - 0.75).abs() < 1e-12);
        assert!((ratios.wet_deck_clearance_over_lwl - 0.045).abs() < 1e-12);
    }

    #[test]
    fn rejects_zero_or_negative_geometry() {
        let mut geometry = example_geometry();
        geometry.length_waterline_m = 0.0;
        assert_eq!(
            geometry.dimensionless().unwrap_err(),
            GeometryError::NotPositive(GeometryField::LengthWaterlineM)
        );

        geometry = example_geometry();
        geometry.wet_deck_clearance_m = -0.1;
        assert_eq!(
            geometry.dimensionless().unwrap_err(),
            GeometryError::NotPositive(GeometryField::WetDeckClearanceM)
        );
    }

    #[test]
    fn rejects_non_finite_geometry() {
        let mut geometry = example_geometry();
        geometry.hull_centerline_spacing_m = f64::NAN;
        assert_eq!(
            geometry.dimensionless().unwrap_err(),
            GeometryError::NonFinite(GeometryField::HullCenterlineSpacingM)
        );
    }

    #[test]
    fn rejects_inconsistent_overall_beam() {
        let mut geometry = example_geometry();
        geometry.overall_beam_m = 10.4;
        assert_eq!(
            geometry.dimensionless().unwrap_err(),
            GeometryError::BeamEnvelopeInconsistent {
                minimum_overall_beam_m: 10.5,
                supplied_overall_beam_m: 10.4,
            }
        );
    }

    #[test]
    fn allows_exact_minimum_outer_beam_envelope() {
        assert!(example_geometry().dimensionless().is_ok());
    }

    #[test]
    fn hull_spacing_sweep_is_dimensionless_and_deterministic() {
        let sweep = hull_spacing_sweep(20.0, &[0.35, 0.40, 0.45]).expect("valid sweep");
        assert_eq!(sweep.len(), 3);
        assert!((sweep[0].centerline_spacing_m - 7.0).abs() < 1e-12);
        assert!((sweep[1].centerline_spacing_m - 8.0).abs() < 1e-12);
        assert!((sweep[2].centerline_spacing_m - 9.0).abs() < 1e-12);
        assert_eq!(sweep[1].centerline_spacing_ratio_over_lwl, 0.40);
    }

    #[test]
    fn rejects_empty_or_invalid_spacing_sweeps() {
        assert_eq!(
            hull_spacing_sweep(20.0, &[]).unwrap_err(),
            GeometryError::EmptySpacingSweep
        );
        assert_eq!(
            hull_spacing_sweep(20.0, &[0.35, 0.0]).unwrap_err(),
            GeometryError::NotPositive(GeometryField::SpacingRatioOverLwl)
        );
        assert_eq!(
            hull_spacing_sweep(20.0, &[0.35, f64::INFINITY]).unwrap_err(),
            GeometryError::NonFinite(GeometryField::SpacingRatioOverLwl)
        );
    }

    #[test]
    fn froude_number_matches_definition_and_accepts_zero_speed() {
        let speed = 5.0_f64;
        let length = 25.0_f64;
        let expected = speed / (STANDARD_GRAVITY_M_S2 * length).sqrt();
        assert!((froude_number(speed, length).expect("valid point") - expected).abs() < 1e-12);
        assert_eq!(froude_number(0.0, length).expect("stopped vessel"), 0.0);
    }

    #[test]
    fn rejects_negative_or_non_finite_froude_inputs() {
        assert_eq!(
            froude_number(-1.0, 25.0).unwrap_err(),
            GeometryError::Negative(GeometryField::SpeedMps)
        );
        assert_eq!(
            froude_number(f64::NAN, 25.0).unwrap_err(),
            GeometryError::NonFinite(GeometryField::SpeedMps)
        );
        assert_eq!(
            froude_number(1.0, 0.0).unwrap_err(),
            GeometryError::NotPositive(GeometryField::LengthWaterlineM)
        );
    }
}
