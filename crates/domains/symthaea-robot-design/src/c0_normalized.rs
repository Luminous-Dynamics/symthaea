// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact normalized mechanics for the bounded C0 JointLinkCoupon profile.
//!
//! This module proves only dimensionless analytical ratios under a declared
//! matched-invariant design profile. It does not establish absolute material
//! properties, physical fixture ideality, strength, safety, or R2.

use crate::design_parameters::{DesignLengthUm, DesignParameterSetId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt;

pub const C0_NORMALIZED_SCHEMA_ID: &str = "symthaea.robot-design.c0-normalized.v1";
pub const C0_NORMALIZED_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C0NormalizedError {
    NonPositiveDimension(&'static str),
    ZeroRationalPart(&'static str),
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for C0NormalizedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveDimension(field) => {
                write!(formatter, "C0 dimension must be positive: {field}")
            }
            Self::ZeroRationalPart(field) => {
                write!(formatter, "positive rational {field} must be non-zero")
            }
            Self::ArithmeticOverflow(context) => {
                write!(formatter, "C0 exact arithmetic overflow: {context}")
            }
        }
    }
}

impl std::error::Error for C0NormalizedError {}

/// Canonical positive rational used for C0 identity and exact comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PositiveRationalV1 {
    numerator: u128,
    denominator: u128,
}

impl PositiveRationalV1 {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, C0NormalizedError> {
        if numerator == 0 {
            return Err(C0NormalizedError::ZeroRationalPart("numerator"));
        }
        if denominator == 0 {
            return Err(C0NormalizedError::ZeroRationalPart("denominator"));
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    pub const fn denominator(self) -> u128 {
        self.denominator
    }

    /// Presentation only. Float conversion never participates in identity or
    /// exact feasibility decisions.
    pub fn to_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    pub fn checked_cmp(self, other: Self) -> Result<Ordering, C0NormalizedError> {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or(C0NormalizedError::ArithmeticOverflow("rational comparison left"))?;
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .ok_or(C0NormalizedError::ArithmeticOverflow("rational comparison right"))?;
        Ok(left.cmp(&right))
    }
}

impl fmt::Display for PositiveRationalV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.numerator, self.denominator)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0MatchedInvariantProfileId([u8; 32]);

impl C0MatchedInvariantProfileId {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0NormalizedEvaluationId([u8; 32]);

impl C0NormalizedEvaluationId {
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

/// Exact section projection consumed by the normalized C0 evaluator.
///
/// Length, load, material state and support semantics are deliberately absent:
/// they are frozen by the matched-invariant profile and therefore cancel from
/// the normalized formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0RectangularSectionV1 {
    pub parameter_set_id: DesignParameterSetId,
    pub width: DesignLengthUm,
    pub height: DesignLengthUm,
}

impl C0RectangularSectionV1 {
    pub fn new(
        parameter_set_id: DesignParameterSetId,
        width: DesignLengthUm,
        height: DesignLengthUm,
    ) -> Result<Self, C0NormalizedError> {
        if width.micrometres() == 0 {
            return Err(C0NormalizedError::NonPositiveDimension("width"));
        }
        if height.micrometres() == 0 {
            return Err(C0NormalizedError::NonPositiveDimension("height"));
        }
        Ok(Self {
            parameter_set_id,
            width,
            height,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0NormalizedEvaluationV1 {
    pub baseline_parameter_set_id: DesignParameterSetId,
    pub candidate_parameter_set_id: DesignParameterSetId,
    pub matched_invariant_profile_id: C0MatchedInvariantProfileId,
    pub mass_ratio: PositiveRationalV1,
    pub deflection_ratio: PositiveRationalV1,
    pub stress_ratio: PositiveRationalV1,
}

impl C0NormalizedEvaluationV1 {
    pub fn id(self) -> C0NormalizedEvaluationId {
        let mut out = Vec::new();
        put_str(&mut out, C0_NORMALIZED_SCHEMA_ID);
        put_u32(&mut out, C0_NORMALIZED_SCHEMA_VERSION);
        out.extend_from_slice(&self.baseline_parameter_set_id.into_bytes());
        out.extend_from_slice(&self.candidate_parameter_set_id.into_bytes());
        out.extend_from_slice(&self.matched_invariant_profile_id.into_bytes());
        put_rational(&mut out, self.mass_ratio);
        put_rational(&mut out, self.deflection_ratio);
        put_rational(&mut out, self.stress_ratio);
        C0NormalizedEvaluationId(Sha256::digest(out).into())
    }
}

pub fn evaluate_c0_normalized(
    baseline: C0RectangularSectionV1,
    candidate: C0RectangularSectionV1,
    matched_invariant_profile_id: C0MatchedInvariantProfileId,
) -> Result<C0NormalizedEvaluationV1, C0NormalizedError> {
    let b0 = u128::from(baseline.width.micrometres());
    let h0 = u128::from(baseline.height.micrometres());
    let b = u128::from(candidate.width.micrometres());
    let h = u128::from(candidate.height.micrometres());

    let mass_ratio = PositiveRationalV1::new(
        checked_mul(b, h, "candidate area")?,
        checked_mul(b0, h0, "baseline area")?,
    )?;

    let deflection_ratio = PositiveRationalV1::new(
        checked_mul(b0, checked_pow(h0, 3, "baseline h^3")?, "baseline b*h^3")?,
        checked_mul(b, checked_pow(h, 3, "candidate h^3")?, "candidate b*h^3")?,
    )?;

    let stress_ratio = PositiveRationalV1::new(
        checked_mul(b0, checked_pow(h0, 2, "baseline h^2")?, "baseline b*h^2")?,
        checked_mul(b, checked_pow(h, 2, "candidate h^2")?, "candidate b*h^2")?,
    )?;

    Ok(C0NormalizedEvaluationV1 {
        baseline_parameter_set_id: baseline.parameter_set_id,
        candidate_parameter_set_id: candidate.parameter_set_id,
        matched_invariant_profile_id,
        mass_ratio,
        deflection_ratio,
        stress_ratio,
    })
}

fn checked_mul(left: u128, right: u128, context: &'static str) -> Result<u128, C0NormalizedError> {
    left.checked_mul(right)
        .ok_or(C0NormalizedError::ArithmeticOverflow(context))
}

fn checked_pow(mut base: u128, mut exponent: u32, context: &'static str) -> Result<u128, C0NormalizedError> {
    let mut result = 1_u128;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result
                .checked_mul(base)
                .ok_or(C0NormalizedError::ArithmeticOverflow(context))?;
        }
        exponent >>= 1;
        if exponent > 0 {
            base = base
                .checked_mul(base)
                .ok_or(C0NormalizedError::ArithmeticOverflow(context))?;
        }
    }
    Ok(result)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
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

    fn set_id(byte: u8) -> DesignParameterSetId {
        DesignParameterSetId::from_bytes([byte; 32])
    }

    fn profile_id(byte: u8) -> C0MatchedInvariantProfileId {
        C0MatchedInvariantProfileId::from_bytes([byte; 32])
    }

    fn section(id_byte: u8, width_um: u64, height_um: u64) -> C0RectangularSectionV1 {
        C0RectangularSectionV1::new(
            set_id(id_byte),
            DesignLengthUm::from_micrometres(width_um),
            DesignLengthUm::from_micrometres(height_um),
        )
        .unwrap()
    }

    #[test]
    fn baseline_against_itself_is_exactly_one() {
        let baseline = section(1, 20_000, 6_000);
        let result = evaluate_c0_normalized(baseline, baseline, profile_id(9)).unwrap();
        let one = PositiveRationalV1::new(1, 1).unwrap();
        assert_eq!(result.mass_ratio, one);
        assert_eq!(result.deflection_ratio, one);
        assert_eq!(result.stress_ratio, one);
    }

    #[test]
    fn known_08_width_12_height_candidate_matches_oracle() {
        let result = evaluate_c0_normalized(
            section(1, 20_000, 6_000),
            section(2, 16_000, 7_200),
            profile_id(9),
        )
        .unwrap();
        assert_eq!(result.mass_ratio, PositiveRationalV1::new(24, 25).unwrap());
        assert_eq!(
            result.deflection_ratio,
            PositiveRationalV1::new(625, 864).unwrap()
        );
        assert_eq!(
            result.stress_ratio,
            PositiveRationalV1::new(125, 144).unwrap()
        );
    }

    #[test]
    fn rational_inputs_reduce_canonically() {
        assert_eq!(
            PositiveRationalV1::new(48, 50).unwrap(),
            PositiveRationalV1::new(24, 25).unwrap()
        );
    }

    #[test]
    fn width_height_orientation_is_semantic() {
        let baseline = section(1, 20_000, 6_000);
        let normal = evaluate_c0_normalized(baseline, section(2, 16_000, 7_200), profile_id(9))
            .unwrap();
        let swapped = evaluate_c0_normalized(baseline, section(3, 7_200, 16_000), profile_id(9))
            .unwrap();
        assert_ne!(normal.deflection_ratio, swapped.deflection_ratio);
        assert_ne!(normal.stress_ratio, swapped.stress_ratio);
    }

    #[test]
    fn zero_dimensions_reject() {
        assert!(C0RectangularSectionV1::new(
            set_id(1),
            DesignLengthUm::from_micrometres(0),
            DesignLengthUm::from_micrometres(6_000)
        )
        .is_err());
    }

    #[test]
    fn overflow_fails_closed() {
        let baseline = section(1, u64::MAX, u64::MAX);
        let candidate = section(2, u64::MAX - 1, u64::MAX - 1);
        assert!(matches!(
            evaluate_c0_normalized(baseline, candidate, profile_id(9)),
            Err(C0NormalizedError::ArithmeticOverflow(_))
        ));
    }

    #[test]
    fn exact_comparison_does_not_use_float_rounding() {
        let less = PositiveRationalV1::new(24, 25).unwrap();
        let one = PositiveRationalV1::new(1, 1).unwrap();
        assert_eq!(less.checked_cmp(one).unwrap(), Ordering::Less);
    }

    #[test]
    fn evaluation_identity_binds_subjects_and_profile() {
        let first = evaluate_c0_normalized(
            section(1, 20_000, 6_000),
            section(2, 16_000, 7_200),
            profile_id(9),
        )
        .unwrap();
        let changed_candidate_identity = evaluate_c0_normalized(
            section(1, 20_000, 6_000),
            section(3, 16_000, 7_200),
            profile_id(9),
        )
        .unwrap();
        let changed_profile = evaluate_c0_normalized(
            section(1, 20_000, 6_000),
            section(2, 16_000, 7_200),
            profile_id(10),
        )
        .unwrap();
        assert_ne!(first.id(), changed_candidate_identity.id());
        assert_ne!(first.id(), changed_profile.id());
    }
}
