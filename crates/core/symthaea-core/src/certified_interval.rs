// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Crate-private outward-certified binary64 interval arithmetic.
//!
//! This module is the hardened shared kernel for MANIFOLD-006 certified numerics.
//! It deliberately exposes no public API. Exact fast paths are admitted only when
//! their real-valued exactness follows from the input representation, never from
//! the classification of an already-rounded result.

use core::fmt;

/// Fail-closed finite-binary64 arithmetic error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CertifiedIntervalError {
    reason: String,
}

impl CertifiedIntervalError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub(crate) fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for CertifiedIntervalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for CertifiedIntervalError {}

/// Conservative closed enclosure of an exact-real scalar value or interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OutwardInterval {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

impl OutwardInterval {
    /// Exact finite binary64 point interval.
    pub(crate) fn point(value: f64) -> Result<Self, CertifiedIntervalError> {
        let value = finite("outward point", value)?;
        Ok(Self {
            lower: value,
            upper: value,
        })
    }

    /// Construct an already-directed finite ordered interval.
    pub(crate) fn ordered(
        lower: f64,
        upper: f64,
        operation: &str,
    ) -> Result<Self, CertifiedIntervalError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(CertifiedIntervalError::new(format!(
                "{operation} produced invalid enclosure [{lower}, {upper}]"
            )));
        }
        // Preserve the qualified 006C distinction: unlike point(), ordered()
        // retains supplied signed-zero endpoint bits.
        Ok(Self { lower, upper })
    }

    /// Conservative interval addition.
    pub(crate) fn add(self, other: Self) -> Result<Self, CertifiedIntervalError> {
        let lower = outward_add(self.lower, other.lower)?.lower;
        let upper = outward_add(self.upper, other.upper)?.upper;
        Self::ordered(lower, upper, "outward interval addition")
    }

    /// Conservative interval subtraction.
    pub(crate) fn sub(self, other: Self) -> Result<Self, CertifiedIntervalError> {
        let lower = outward_sub(self.lower, other.upper)?.lower;
        let upper = outward_sub(self.upper, other.lower)?.upper;
        Self::ordered(lower, upper, "outward interval subtraction")
    }

    /// Conservative multiplication by one finite strictly-positive scalar.
    pub(crate) fn mul_positive(self, scalar: f64) -> Result<Self, CertifiedIntervalError> {
        if !scalar.is_finite() || scalar <= 0.0 {
            return Err(CertifiedIntervalError::new(format!(
                "outward interval multiplication requires finite positive scalar, got {scalar}"
            )));
        }
        let lower = outward_mul(self.lower, scalar)?.lower;
        let upper = outward_mul(self.upper, scalar)?.upper;
        Self::ordered(lower, upper, "outward interval multiplication")
    }

    /// Conservative division by one finite strictly-positive scalar.
    pub(crate) fn div_positive(self, scalar: f64) -> Result<Self, CertifiedIntervalError> {
        if !scalar.is_finite() || scalar <= 0.0 {
            return Err(CertifiedIntervalError::new(format!(
                "outward interval division requires finite positive scalar, got {scalar}"
            )));
        }
        let lower = outward_div(self.lower, scalar)?.lower;
        let upper = outward_div(self.upper, scalar)?.upper;
        Self::ordered(lower, upper, "outward interval division")
    }
}

/// Directed enclosure of one finite binary64 addition.
pub(crate) fn outward_add(
    left: f64,
    right: f64,
) -> Result<OutwardInterval, CertifiedIntervalError> {
    let left = finite("outward addition left operand", left)?;
    let right = finite("outward addition right operand", right)?;
    if right == 0.0 {
        return OutwardInterval::point(left);
    }
    if left == 0.0 {
        return OutwardInterval::point(right);
    }

    let sum = finite("outward addition nearest result", left + right)?;
    let virtual_right = sum - left;
    let virtual_left = sum - virtual_right;
    let right_roundoff = right - virtual_right;
    let left_roundoff = left - virtual_left;
    let residual = finite(
        "outward addition residual",
        left_roundoff + right_roundoff,
    )?;

    if residual == 0.0 {
        return OutwardInterval::point(sum);
    }
    if residual > 0.0 {
        OutwardInterval::ordered(sum, next_up(sum)?, "directed outward addition")
    } else {
        OutwardInterval::ordered(next_down(sum)?, sum, "directed outward addition")
    }
}

/// Directed enclosure of one finite binary64 subtraction.
pub(crate) fn outward_sub(
    left: f64,
    right: f64,
) -> Result<OutwardInterval, CertifiedIntervalError> {
    outward_add(left, finite("outward subtraction negation", -right)?)
}

/// Directed enclosure of one finite binary64 multiplication.
pub(crate) fn outward_mul(
    left: f64,
    right: f64,
) -> Result<OutwardInterval, CertifiedIntervalError> {
    let left = finite("outward multiplication left operand", left)?;
    let right = finite("outward multiplication right operand", right)?;
    if left == 0.0 || right == 0.0 {
        return OutwardInterval::point(0.0);
    }
    if left == 1.0 {
        return OutwardInterval::point(right);
    }
    if right == 1.0 {
        return OutwardInterval::point(left);
    }
    if left == -1.0 {
        return OutwardInterval::point(finite("exact multiplication by -1", -right)?);
    }
    if right == -1.0 {
        return OutwardInterval::point(finite("exact multiplication by -1", -left)?);
    }

    let product = finite("outward multiplication nearest result", left * right)?;
    if exact_normal_power_of_two_product(left, right) {
        return OutwardInterval::point(product);
    }

    OutwardInterval::ordered(
        next_down(product)?,
        next_up(product)?,
        "outward multiplication enclosure",
    )
}

/// Directed enclosure of one finite binary64 division.
pub(crate) fn outward_div(
    left: f64,
    right: f64,
) -> Result<OutwardInterval, CertifiedIntervalError> {
    let left = finite("outward division numerator", left)?;
    let right = finite("outward division denominator", right)?;
    if right == 0.0 {
        return Err(CertifiedIntervalError::new(
            "division by zero in outward robust arithmetic",
        ));
    }
    if left == 0.0 {
        return OutwardInterval::point(0.0);
    }
    if right == 1.0 {
        return OutwardInterval::point(left);
    }
    if right == -1.0 {
        return OutwardInterval::point(finite("exact division by -1", -left)?);
    }

    let quotient = finite("outward division nearest result", left / right)?;
    if exact_normal_power_of_two_quotient(left, right) {
        return OutwardInterval::point(quotient);
    }

    OutwardInterval::ordered(
        next_down(quotient)?,
        next_up(quotient)?,
        "outward division enclosure",
    )
}

/// Next finite representable binary64 value above `value`.
pub(crate) fn next_up(value: f64) -> Result<f64, CertifiedIntervalError> {
    let value = finite("next-up input", value)?;
    if value == f64::MAX {
        return Err(CertifiedIntervalError::new(
            "next-up would leave finite binary64 domain",
        ));
    }
    if value == 0.0 {
        return Ok(f64::from_bits(1));
    }
    let bits = value.to_bits();
    let stepped = if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    };
    finite("next-up result", stepped)
}

/// Next finite representable binary64 value below `value`.
pub(crate) fn next_down(value: f64) -> Result<f64, CertifiedIntervalError> {
    let value = finite("next-down input", value)?;
    if value == -f64::MAX {
        return Err(CertifiedIntervalError::new(
            "next-down would leave finite binary64 domain",
        ));
    }
    if value == 0.0 {
        return Ok(-f64::from_bits(1));
    }
    let bits = value.to_bits();
    let stepped = if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    };
    finite("next-down result", stepped)
}

/// Whether `value` is a finite normal exact power of two.
pub(crate) fn is_normal_power_of_two(value: f64) -> bool {
    if !value.is_normal() || value <= 0.0 {
        return false;
    }
    value.to_bits() & ((1_u64 << 52) - 1) == 0
}

/// Canonicalize either signed zero to positive zero.
pub(crate) fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// Return the unbiased exponent of a finite normal value.
fn normal_unbiased_exponent(value: f64) -> Option<i32> {
    let magnitude = value.abs();
    if !magnitude.is_normal() {
        return None;
    }
    let raw = ((magnitude.to_bits() >> 52) & 0x7ff) as i32;
    Some(raw - 1023)
}

/// Return the unbiased exponent when `value` is a finite normal exact power of two.
fn normal_power_of_two_exponent(value: f64) -> Option<i32> {
    let magnitude = value.abs();
    if !is_normal_power_of_two(magnitude) {
        return None;
    }
    normal_unbiased_exponent(magnitude)
}

/// Prove that multiplication by a power of two is exact and remains normal.
///
/// This intentionally reasons from the *input exponents*. A rounded product that
/// happens to be normal is not evidence that the exact product was normal: the
/// exact real value may lie just below `MIN_POSITIVE` and round upward to it.
fn exact_normal_power_of_two_product(left: f64, right: f64) -> bool {
    if let (Some(scale), Some(other)) = (
        normal_power_of_two_exponent(left),
        normal_unbiased_exponent(right),
    ) {
        let shifted = scale + other;
        if (-1022..=1023).contains(&shifted) {
            return true;
        }
    }
    if let (Some(other), Some(scale)) = (
        normal_unbiased_exponent(left),
        normal_power_of_two_exponent(right),
    ) {
        let shifted = other + scale;
        return (-1022..=1023).contains(&shifted);
    }
    false
}

/// Prove that division by a power of two is exact and remains normal.
fn exact_normal_power_of_two_quotient(numerator: f64, denominator: f64) -> bool {
    let Some(numerator_exponent) = normal_unbiased_exponent(numerator) else {
        return false;
    };
    let Some(scale_exponent) = normal_power_of_two_exponent(denominator) else {
        return false;
    };
    let shifted = numerator_exponent - scale_exponent;
    (-1022..=1023).contains(&shifted)
}

fn finite(operation: &str, value: f64) -> Result<f64, CertifiedIntervalError> {
    if !value.is_finite() {
        return Err(CertifiedIntervalError::new(format!(
            "robust reachability {operation} became non-finite"
        )));
    }
    Ok(canonical_zero(value))
}
