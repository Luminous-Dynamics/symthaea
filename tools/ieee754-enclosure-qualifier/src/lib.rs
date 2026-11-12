// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-only outward enclosures for scalar IEEE-754 binary64
//! arithmetic. This crate is not a production dependency of aircraft models.
//!
//! Each operation is evaluated once with the target's binary64 arithmetic,
//! then expanded to the immediately adjacent representable value on both
//! sides. For a correctly-rounded basic IEEE-754 operation this contains the
//! exact real result for finite binary64 operands. Overflow is represented by
//! a one-sided unbounded interval. Division by zero and non-finite inputs fail.

#![forbid(unsafe_code)]

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl BinaryOp {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "sub",
            Self::Multiply => "mul",
            Self::Divide => "div",
        }
    }

    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Subtract => a - b,
            Self::Multiply => a * b,
            Self::Divide => a / b,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Enclosure {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnclosureError {
    NonFiniteInput,
    DivisionByZero,
    InvalidResult,
}

impl fmt::Display for EnclosureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for EnclosureError {}

/// Smallest representable binary64 value strictly greater than x.
///
/// NaNs and positive infinity are returned unchanged. Both signed zeros map
/// to the smallest positive subnormal, matching the IEEE nextUp operation.
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return -f64::MAX;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Greatest representable binary64 value strictly less than x.
///
/// NaNs and negative infinity are returned unchanged. Both signed zeros map
/// to the smallest negative subnormal, matching the IEEE nextDown operation.
pub fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == f64::INFINITY {
        return f64::MAX;
    }
    if x == 0.0 {
        return f64::from_bits((1_u64 << 63) | 1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Enclose the exact real result of one basic arithmetic operation on two
/// finite binary64 operands. Each endpoint is expanded by one representable
/// value around the nearest-rounded hardware result. Multi-operation
/// expressions must compose these enclosures operation by operation; applying
/// this function only to the final nearest-rounded expression is not covered.
pub fn enclose_binary(
    op: BinaryOp,
    a: f64,
    b: f64,
) -> Result<Enclosure, EnclosureError> {
    if !a.is_finite() || !b.is_finite() {
        return Err(EnclosureError::NonFiniteInput);
    }
    if op == BinaryOp::Divide && b == 0.0 {
        return Err(EnclosureError::DivisionByZero);
    }

    let rounded = op.apply(a, b);
    if rounded.is_nan() {
        return Err(EnclosureError::InvalidResult);
    }

    let lower = next_down(rounded);
    let upper = next_up(rounded);
    if lower.is_nan() || upper.is_nan() || lower > upper {
        return Err(EnclosureError::InvalidResult);
    }
    Ok(Enclosure { lower, upper })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_up_and_down_cross_signed_zero_correctly() {
        let smallest = f64::from_bits(1);
        let negative_smallest = f64::from_bits((1_u64 << 63) | 1);
        assert_eq!(next_up(0.0).to_bits(), smallest.to_bits());
        assert_eq!(next_up(-0.0).to_bits(), smallest.to_bits());
        assert_eq!(next_down(0.0).to_bits(), negative_smallest.to_bits());
        assert_eq!(next_down(-0.0).to_bits(), negative_smallest.to_bits());
    }

    #[test]
    fn next_up_and_down_handle_finite_extremes_and_infinities() {
        assert_eq!(next_up(f64::MAX), f64::INFINITY);
        assert_eq!(next_down(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(next_up(f64::NEG_INFINITY), -f64::MAX);
        assert_eq!(next_down(f64::INFINITY), f64::MAX);
    }

    #[test]
    fn zero_result_gets_a_signed_enclosure() {
        let result = enclose_binary(BinaryOp::Subtract, 1.0, 1.0).unwrap();
        assert_eq!(result.lower.to_bits(), f64::from_bits((1_u64 << 63) | 1).to_bits());
        assert_eq!(result.upper.to_bits(), f64::from_bits(1).to_bits());
    }

    #[test]
    fn overflow_is_an_explicit_one_sided_unbounded_enclosure() {
        let result = enclose_binary(BinaryOp::Multiply, f64::MAX, 2.0).unwrap();
        assert_eq!(result.lower, f64::MAX);
        assert_eq!(result.upper, f64::INFINITY);

        let negative = enclose_binary(BinaryOp::Multiply, -f64::MAX, 2.0).unwrap();
        assert_eq!(negative.lower, f64::NEG_INFINITY);
        assert_eq!(negative.upper, -f64::MAX);
    }

    #[test]
    fn division_by_zero_and_non_finite_inputs_fail_closed() {
        assert_eq!(
            enclose_binary(BinaryOp::Divide, 1.0, -0.0),
            Err(EnclosureError::DivisionByZero)
        );
        assert_eq!(
            enclose_binary(BinaryOp::Add, f64::INFINITY, 1.0),
            Err(EnclosureError::NonFiniteInput)
        );
    }

    #[test]
    fn subnormal_underflow_is_enclosed() {
        let smallest = f64::from_bits(1);
        let result = enclose_binary(BinaryOp::Multiply, smallest, 0.5).unwrap();
        assert!(result.lower < 0.0);
        assert!(result.upper > 0.0);
    }
}
