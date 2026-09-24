// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact C0 dimensional-tolerance robustness evaluation.
//!
//! This module evaluates a frozen rectangular candidate over a declared
//! dimensional box using monotonic analytical worst corners. A passing result
//! means only `robust under this design-side tolerance assumption`; it does not
//! establish that any manufacturing process can hold the declared tolerance.

use crate::c0_normalized::{C0NormalizedError, PositiveRationalV1};
use crate::design_parameters::DesignLengthUm;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt;

pub const C0_ROBUSTNESS_SCHEMA_ID: &str = "symthaea.robot-design.c0-robustness.v1";
pub const C0_ROBUSTNESS_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C0RobustnessError {
    NonPositiveDimension(&'static str),
    InvalidToleranceBox(&'static str),
    ArithmeticOverflow(&'static str),
    Rational(C0NormalizedError),
}

impl fmt::Display for C0RobustnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveDimension(field) => {
                write!(formatter, "C0 robustness dimension must be positive: {field}")
            }
            Self::InvalidToleranceBox(field) => {
                write!(formatter, "C0 robustness lower bound exceeds upper bound: {field}")
            }
            Self::ArithmeticOverflow(context) => {
                write!(formatter, "C0 robustness arithmetic overflow: {context}")
            }
            Self::Rational(error) => write!(formatter, "C0 robustness rational error: {error}"),
        }
    }
}

impl std::error::Error for C0RobustnessError {}

impl From<C0NormalizedError> for C0RobustnessError {
    fn from(value: C0NormalizedError) -> Self {
        Self::Rational(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0DimensionBoxV1 {
    pub width_lower: DesignLengthUm,
    pub width_upper: DesignLengthUm,
    pub height_lower: DesignLengthUm,
    pub height_upper: DesignLengthUm,
}

impl C0DimensionBoxV1 {
    pub fn validate(self) -> Result<(), C0RobustnessError> {
        for (value, field) in [
            (self.width_lower, "width_lower"),
            (self.width_upper, "width_upper"),
            (self.height_lower, "height_lower"),
            (self.height_upper, "height_upper"),
        ] {
            if value.micrometres() == 0 {
                return Err(C0RobustnessError::NonPositiveDimension(field));
            }
        }
        if self.width_lower > self.width_upper {
            return Err(C0RobustnessError::InvalidToleranceBox("width"));
        }
        if self.height_lower > self.height_upper {
            return Err(C0RobustnessError::InvalidToleranceBox("height"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0RobustnessPolicyV1 {
    pub max_mass_ratio: PositiveRationalV1,
    pub max_deflection_ratio: PositiveRationalV1,
    pub max_stress_ratio: PositiveRationalV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0RobustnessProfileId([u8; 32]);

impl C0RobustnessProfileId {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C0RobustnessDispositionV1 {
    RobustUnderDeclaredBox,
    MassTargetNotRobust,
    DeflectionProtectionNotRobust,
    StressProtectionNotRobust,
    MultipleProtectedBoundsNotRobust,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0RobustnessAssessmentV1 {
    pub profile_id: C0RobustnessProfileId,
    pub worst_mass_ratio: PositiveRationalV1,
    pub worst_deflection_ratio: PositiveRationalV1,
    pub worst_stress_ratio: PositiveRationalV1,
    pub disposition: C0RobustnessDispositionV1,
}

pub fn assess_c0_tolerance_robustness(
    baseline_width: DesignLengthUm,
    baseline_height: DesignLengthUm,
    candidate_box: C0DimensionBoxV1,
    policy: C0RobustnessPolicyV1,
) -> Result<C0RobustnessAssessmentV1, C0RobustnessError> {
    candidate_box.validate()?;
    let b0 = positive(baseline_width, "baseline_width")?;
    let h0 = positive(baseline_height, "baseline_height")?;
    let blo = u128::from(candidate_box.width_lower.micrometres());
    let bhi = u128::from(candidate_box.width_upper.micrometres());
    let hlo = u128::from(candidate_box.height_lower.micrometres());
    let hhi = u128::from(candidate_box.height_upper.micrometres());

    // Monotonic worst corners for the frozen C0 rectangular formulas.
    let worst_mass_ratio = PositiveRationalV1::new(
        checked_mul(bhi, hhi, "candidate upper area")?,
        checked_mul(b0, h0, "baseline area")?,
    )?;
    let worst_deflection_ratio = PositiveRationalV1::new(
        checked_mul(
            b0,
            checked_pow(h0, 3, "baseline h^3")?,
            "baseline b*h^3",
        )?,
        checked_mul(
            blo,
            checked_pow(hlo, 3, "candidate lower h^3")?,
            "candidate lower b*h^3",
        )?,
    )?;
    let worst_stress_ratio = PositiveRationalV1::new(
        checked_mul(
            b0,
            checked_pow(h0, 2, "baseline h^2")?,
            "baseline b*h^2",
        )?,
        checked_mul(
            blo,
            checked_pow(hlo, 2, "candidate lower h^2")?,
            "candidate lower b*h^2",
        )?,
    )?;

    let mass_ok = worst_mass_ratio.checked_cmp(policy.max_mass_ratio)? != Ordering::Greater;
    let deflection_ok = worst_deflection_ratio.checked_cmp(policy.max_deflection_ratio)?
        != Ordering::Greater;
    let stress_ok = worst_stress_ratio.checked_cmp(policy.max_stress_ratio)? != Ordering::Greater;

    let failure_count = usize::from(!mass_ok) + usize::from(!deflection_ok) + usize::from(!stress_ok);
    let disposition = match (mass_ok, deflection_ok, stress_ok, failure_count) {
        (true, true, true, 0) => C0RobustnessDispositionV1::RobustUnderDeclaredBox,
        (false, true, true, 1) => C0RobustnessDispositionV1::MassTargetNotRobust,
        (true, false, true, 1) => C0RobustnessDispositionV1::DeflectionProtectionNotRobust,
        (true, true, false, 1) => C0RobustnessDispositionV1::StressProtectionNotRobust,
        _ => C0RobustnessDispositionV1::MultipleProtectedBoundsNotRobust,
    };

    Ok(C0RobustnessAssessmentV1 {
        profile_id: robustness_profile_id(baseline_width, baseline_height, candidate_box, policy),
        worst_mass_ratio,
        worst_deflection_ratio,
        worst_stress_ratio,
        disposition,
    })
}

fn robustness_profile_id(
    baseline_width: DesignLengthUm,
    baseline_height: DesignLengthUm,
    candidate_box: C0DimensionBoxV1,
    policy: C0RobustnessPolicyV1,
) -> C0RobustnessProfileId {
    let mut out = Vec::new();
    put_str(&mut out, C0_ROBUSTNESS_SCHEMA_ID);
    put_u32(&mut out, C0_ROBUSTNESS_SCHEMA_VERSION);
    for value in [
        baseline_width,
        baseline_height,
        candidate_box.width_lower,
        candidate_box.width_upper,
        candidate_box.height_lower,
        candidate_box.height_upper,
    ] {
        put_u64(&mut out, value.micrometres());
    }
    for value in [
        policy.max_mass_ratio,
        policy.max_deflection_ratio,
        policy.max_stress_ratio,
    ] {
        put_u128(&mut out, value.numerator());
        put_u128(&mut out, value.denominator());
    }
    C0RobustnessProfileId(Sha256::digest(out).into())
}

fn positive(value: DesignLengthUm, field: &'static str) -> Result<u128, C0RobustnessError> {
    if value.micrometres() == 0 {
        Err(C0RobustnessError::NonPositiveDimension(field))
    } else {
        Ok(u128::from(value.micrometres()))
    }
}

fn checked_mul(left: u128, right: u128, context: &'static str) -> Result<u128, C0RobustnessError> {
    left.checked_mul(right)
        .ok_or(C0RobustnessError::ArithmeticOverflow(context))
}

fn checked_pow(mut base: u128, mut exponent: u32, context: &'static str) -> Result<u128, C0RobustnessError> {
    let mut result = 1_u128;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result
                .checked_mul(base)
                .ok_or(C0RobustnessError::ArithmeticOverflow(context))?;
        }
        exponent >>= 1;
        if exponent > 0 {
            base = base
                .checked_mul(base)
                .ok_or(C0RobustnessError::ArithmeticOverflow(context))?;
        }
    }
    Ok(result)
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn um(value: u64) -> DesignLengthUm {
        DesignLengthUm::from_micrometres(value)
    }

    fn policy() -> C0RobustnessPolicyV1 {
        C0RobustnessPolicyV1 {
            max_mass_ratio: PositiveRationalV1::new(97, 100).unwrap(),
            max_deflection_ratio: PositiveRationalV1::new(19, 20).unwrap(),
            max_stress_ratio: PositiveRationalV1::new(1, 1).unwrap(),
        }
    }

    #[test]
    fn narrow_tolerance_box_preserves_all_reference_protections() {
        let assessment = assess_c0_tolerance_robustness(
            um(20_000),
            um(6_000),
            C0DimensionBoxV1 {
                width_lower: um(15_950),
                width_upper: um(16_050),
                height_lower: um(7_150),
                height_upper: um(7_250),
            },
            policy(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0RobustnessDispositionV1::RobustUnderDeclaredBox
        );
    }

    #[test]
    fn wide_box_can_break_mass_target_while_structural_guards_still_pass() {
        let assessment = assess_c0_tolerance_robustness(
            um(20_000),
            um(6_000),
            C0DimensionBoxV1 {
                width_lower: um(15_500),
                width_upper: um(16_750),
                height_lower: um(7_000),
                height_upper: um(7_500),
            },
            policy(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0RobustnessDispositionV1::MassTargetNotRobust
        );
        assert_eq!(
            assessment.worst_mass_ratio,
            PositiveRationalV1::new(67, 64).unwrap()
        );
        assert!(
            assessment
                .worst_deflection_ratio
                .checked_cmp(policy().max_deflection_ratio)
                .unwrap()
                != Ordering::Greater
        );
        assert!(
            assessment
                .worst_stress_ratio
                .checked_cmp(policy().max_stress_ratio)
                .unwrap()
                != Ordering::Greater
        );
    }

    #[test]
    fn invalid_box_rejects() {
        assert!(assess_c0_tolerance_robustness(
            um(20_000),
            um(6_000),
            C0DimensionBoxV1 {
                width_lower: um(17_000),
                width_upper: um(16_000),
                height_lower: um(7_000),
                height_upper: um(7_200),
            },
            policy(),
        )
        .is_err());
    }
}
