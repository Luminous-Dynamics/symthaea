// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Geometry-only analytical applicability screening for C0.
//!
//! This module is an admission-policy screen. Passing it does not establish
//! physical beam validity, lateral stability, elastic response, fixture
//! ideality, safety, or physical prediction accuracy.

use crate::c0_normalized::{C0NormalizedError, PositiveRationalV1};
use crate::design_parameters::DesignLengthUm;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt;

pub const C0_APPLICABILITY_SCHEMA_ID: &str = "symthaea.robot-design.c0-applicability.v1";
pub const C0_APPLICABILITY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C0ApplicabilityError {
    NonPositiveDimension(&'static str),
    ArithmeticOverflow(&'static str),
    Rational(C0NormalizedError),
}

impl fmt::Display for C0ApplicabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveDimension(field) => {
                write!(formatter, "C0 applicability dimension must be positive: {field}")
            }
            Self::ArithmeticOverflow(context) => {
                write!(formatter, "C0 applicability arithmetic overflow: {context}")
            }
            Self::Rational(error) => write!(formatter, "C0 applicability rational error: {error}"),
        }
    }
}

impl std::error::Error for C0ApplicabilityError {}

impl From<C0NormalizedError> for C0ApplicabilityError {
    fn from(value: C0NormalizedError) -> Self {
        Self::Rational(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0ApplicabilityProfileId([u8; 32]);

impl C0ApplicabilityProfileId {
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }
}

/// Explicit C0 geometry-screen policy.
///
/// `max_height_to_width_ratio` is a campaign/profile admission rule only. It is
/// not a buckling or lateral-stability theorem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0ApplicabilityProfileV1 {
    pub max_conservative_shear_to_bending_ratio: PositiveRationalV1,
    pub max_height_to_width_ratio: PositiveRationalV1,
}

impl C0ApplicabilityProfileV1 {
    pub fn id(self) -> C0ApplicabilityProfileId {
        let mut out = Vec::new();
        put_str(&mut out, C0_APPLICABILITY_SCHEMA_ID);
        put_u32(&mut out, C0_APPLICABILITY_SCHEMA_VERSION);
        put_rational(
            &mut out,
            self.max_conservative_shear_to_bending_ratio,
        );
        put_rational(&mut out, self.max_height_to_width_ratio);
        C0ApplicabilityProfileId(Sha256::digest(out).into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C0ApplicabilityDispositionV1 {
    AdmittedGeometryScreen,
    RejectedConservativeShearBound,
    RejectedSectionAspectProfile,
    RejectedMultipleGeometryScreens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0ApplicabilityAssessmentV1 {
    pub profile_id: C0ApplicabilityProfileId,
    /// Conservative isotropic upper bound: `3.6 * (h/L)^2`, represented
    /// exactly as `18 h^2 / (5 L^2)`.
    pub conservative_shear_to_bending_upper_bound: PositiveRationalV1,
    pub height_to_width_ratio: PositiveRationalV1,
    pub disposition: C0ApplicabilityDispositionV1,
}

pub fn assess_c0_geometry_applicability(
    span: DesignLengthUm,
    width: DesignLengthUm,
    height: DesignLengthUm,
    profile: C0ApplicabilityProfileV1,
) -> Result<C0ApplicabilityAssessmentV1, C0ApplicabilityError> {
    let span = positive_u128(span, "span")?;
    let width = positive_u128(width, "width")?;
    let height = positive_u128(height, "height")?;

    let h_squared = height
        .checked_mul(height)
        .ok_or(C0ApplicabilityError::ArithmeticOverflow("height squared"))?;
    let l_squared = span
        .checked_mul(span)
        .ok_or(C0ApplicabilityError::ArithmeticOverflow("span squared"))?;
    let shear_numerator = 18_u128
        .checked_mul(h_squared)
        .ok_or(C0ApplicabilityError::ArithmeticOverflow(
            "conservative shear numerator",
        ))?;
    let shear_denominator = 5_u128
        .checked_mul(l_squared)
        .ok_or(C0ApplicabilityError::ArithmeticOverflow(
            "conservative shear denominator",
        ))?;

    let shear_upper = PositiveRationalV1::new(shear_numerator, shear_denominator)?;
    let aspect = PositiveRationalV1::new(height, width)?;

    let shear_ok = shear_upper.checked_cmp(
        profile.max_conservative_shear_to_bending_ratio,
    )? != Ordering::Greater;
    let aspect_ok = aspect.checked_cmp(profile.max_height_to_width_ratio)?
        != Ordering::Greater;

    let disposition = match (shear_ok, aspect_ok) {
        (true, true) => C0ApplicabilityDispositionV1::AdmittedGeometryScreen,
        (false, true) => C0ApplicabilityDispositionV1::RejectedConservativeShearBound,
        (true, false) => C0ApplicabilityDispositionV1::RejectedSectionAspectProfile,
        (false, false) => C0ApplicabilityDispositionV1::RejectedMultipleGeometryScreens,
    };

    Ok(C0ApplicabilityAssessmentV1 {
        profile_id: profile.id(),
        conservative_shear_to_bending_upper_bound: shear_upper,
        height_to_width_ratio: aspect,
        disposition,
    })
}

fn positive_u128(
    value: DesignLengthUm,
    field: &'static str,
) -> Result<u128, C0ApplicabilityError> {
    let value = value.micrometres();
    if value == 0 {
        Err(C0ApplicabilityError::NonPositiveDimension(field))
    } else {
        Ok(u128::from(value))
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u128(out: &mut Vec<u8>, value: u128) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_be_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

fn put_rational(out: &mut Vec<u8>, value: PositiveRationalV1) {
    put_u128(out, value.numerator());
    put_u128(out, value.denominator());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn um(value: u64) -> DesignLengthUm {
        DesignLengthUm::from_micrometres(value)
    }

    fn profile() -> C0ApplicabilityProfileV1 {
        C0ApplicabilityProfileV1 {
            max_conservative_shear_to_bending_ratio: PositiveRationalV1::new(1, 100).unwrap(),
            max_height_to_width_ratio: PositiveRationalV1::new(2, 1).unwrap(),
        }
    }

    #[test]
    fn baseline_300mm_span_6mm_height_has_exact_shear_upper_bound() {
        let assessment = assess_c0_geometry_applicability(
            um(300_000),
            um(20_000),
            um(6_000),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.conservative_shear_to_bending_upper_bound,
            PositiveRationalV1::new(18, 12_500).unwrap()
        );
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::AdmittedGeometryScreen
        );
    }

    #[test]
    fn selected_reference_candidate_remains_inside_geometry_screen() {
        let assessment = assess_c0_geometry_applicability(
            um(300_000),
            um(16_000),
            um(7_200),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::AdmittedGeometryScreen
        );
    }

    #[test]
    fn thick_section_can_fail_conservative_shear_budget() {
        let assessment = assess_c0_geometry_applicability(
            um(300_000),
            um(40_000),
            um(30_000),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::RejectedConservativeShearBound
        );
    }

    #[test]
    fn synthetic_tall_narrow_section_can_fail_aspect_profile() {
        let assessment = assess_c0_geometry_applicability(
            um(1_000_000),
            um(10_000),
            um(25_000),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::RejectedSectionAspectProfile
        );
    }

    #[test]
    fn zero_dimension_rejects_before_ratio_math() {
        assert!(assess_c0_geometry_applicability(
            um(300_000),
            um(0),
            um(6_000),
            profile()
        )
        .is_err());
    }

    #[test]
    fn changing_policy_changes_profile_identity() {
        let first = profile();
        let second = C0ApplicabilityProfileV1 {
            max_conservative_shear_to_bending_ratio: PositiveRationalV1::new(1, 200).unwrap(),
            ..first
        };
        assert_ne!(first.id(), second.id());
    }
}
