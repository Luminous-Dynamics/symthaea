// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Small, fail-closed interval arithmetic for certified numerical work.
//!
//! Arithmetic outputs are widened one representable f64 step outward. This is
//! not a claim that arbitrary libm functions are correctly rounded; functions
//! such as sin, exp, and ln need separately reviewed bounds before being added.
//!
//! The enclosure guarantee assumes ordinary IEEE-754 binary64 round-to-nearest
//! operations and compilation without unsafe fast-math reassociation.

use std::fmt;

/// Errors while constructing or evaluating a real interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalError {
    /// NaN cannot define an ordered endpoint.
    NaNBound,
    /// The lower endpoint exceeds the upper endpoint or denotes an empty
    /// extended-real singleton at infinity.
    InvalidBounds,
    /// A point interval must identify a finite real number.
    NonFinitePoint,
    /// Division is undefined as a bounded interval when the divisor includes 0.
    DivisionByIntervalContainingZero,
    /// The interval contains zero and the exponent is zero.
    IndeterminatePower,
}

impl fmt::Display for IntervalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaNBound => write!(f, "interval endpoints must not be NaN"),
            Self::InvalidBounds => write!(f, "interval bounds are reversed or empty"),
            Self::NonFinitePoint => write!(f, "a point interval requires a finite value"),
            Self::DivisionByIntervalContainingZero => {
                write!(f, "cannot divide by an interval containing zero")
            }
            Self::IndeterminatePower => write!(f, "zero-to-zero is undefined over this interval"),
        }
    }
}

impl std::error::Error for IntervalError {}

/// Closed interval over the real numbers with outward-rounded binary64 arithmetic.
///
/// Infinite bounds can represent unbounded enclosures after overflow. NaN is
/// never stored. Fields are private so operations preserve endpoint ordering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    /// Construct an interval after validating endpoint ordering.
    ///
    /// A lower negative infinity and upper positive infinity are allowed.
    /// NaN, reversed bounds, [+infinity,+infinity], and [-infinity,-infinity]
    /// are rejected.
    pub fn new(lower: f64, upper: f64) -> Result<Self, IntervalError> {
        if lower.is_nan() || upper.is_nan() {
            return Err(IntervalError::NaNBound);
        }
        if lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
            return Err(IntervalError::InvalidBounds);
        }
        Ok(Self { lower, upper })
    }

    /// Construct the exact singleton interval containing a finite value.
    pub fn point(value: f64) -> Result<Self, IntervalError> {
        if !value.is_finite() {
            return Err(IntervalError::NonFinitePoint);
        }
        Ok(Self { lower: value, upper: value })
    }

    /// The whole real line, used as a conservative fallback for indeterminate
    /// endpoint calculations such as infinity divided by infinity.
    pub const fn entire() -> Self {
        Self { lower: f64::NEG_INFINITY, upper: f64::INFINITY }
    }

    /// Lowest value in the enclosure.
    pub const fn lower(self) -> f64 {
        self.lower
    }

    /// Highest value in the enclosure.
    pub const fn upper(self) -> f64 {
        self.upper
    }

    /// Whether a value belongs to this enclosure. NaN is never contained.
    pub fn contains(self, value: f64) -> bool {
        !value.is_nan() && self.lower <= value && value <= self.upper
    }

    /// Whether zero is included in this interval.
    pub fn contains_zero(self) -> bool {
        self.lower <= 0.0 && self.upper >= 0.0
    }

    /// Whether all represented real values are strictly positive.
    pub fn is_strictly_positive(self) -> bool {
        self.lower > 0.0
    }

    /// Whether all represented real values are strictly negative.
    pub fn is_strictly_negative(self) -> bool {
        self.upper < 0.0
    }

    /// Exact point zero, unlike an interval that merely contains zero.
    pub fn is_exact_zero(self) -> bool {
        self.lower == 0.0 && self.upper == 0.0
    }

    /// Add intervals, widening both endpoints to enclose rounding error.
    pub fn checked_add(self, rhs: Self) -> Self {
        if rhs.is_exact_zero() {
            return self;
        }
        if self.is_exact_zero() {
            return rhs;
        }
        Self::from_outward(self.lower + rhs.lower, self.upper + rhs.upper)
    }

    /// Subtract intervals, widening both endpoints to enclose rounding error.
    pub fn checked_subtract(self, rhs: Self) -> Self {
        if rhs.is_exact_zero() {
            return self;
        }
        Self::from_outward(self.lower - rhs.upper, self.upper - rhs.lower)
    }

    /// Multiply intervals using all four endpoint products.
    ///
    /// The endpoint helper treats zero times infinity as the limiting value
    /// zero for hull computation. Any other unexpected NaN widens to the whole
    /// line; it is never silently discarded by min/max.
    pub fn checked_multiply(self, rhs: Self) -> Self {
        if self.is_exact_zero() || rhs.is_exact_zero() {
            return Self { lower: 0.0, upper: 0.0 };
        }
        if self.is_exact_point(1.0) {
            return rhs;
        }
        if rhs.is_exact_point(1.0) {
            return self;
        }
        if self.is_exact_point(-1.0) {
            return rhs.negate();
        }
        if rhs.is_exact_point(-1.0) {
            return self.negate();
        }

        let products = [
            endpoint_product(self.lower, rhs.lower),
            endpoint_product(self.lower, rhs.upper),
            endpoint_product(self.upper, rhs.lower),
            endpoint_product(self.upper, rhs.upper),
        ];
        if products.iter().any(|value| value.is_nan()) {
            return Self::entire();
        }

        let lower = products.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::from_outward(lower, upper)
    }

    /// Divide intervals. The divisor must exclude zero.
    ///
    /// If an endpoint quotient is indeterminate, the entire line is returned:
    /// this loses precision but never creates a falsely narrow enclosure.
    pub fn checked_divide(self, rhs: Self) -> Result<Self, IntervalError> {
        if rhs.contains_zero() {
            return Err(IntervalError::DivisionByIntervalContainingZero);
        }

        let quotients = [
            self.lower / rhs.lower,
            self.lower / rhs.upper,
            self.upper / rhs.lower,
            self.upper / rhs.upper,
        ];
        if quotients.iter().any(|value| value.is_nan()) {
            return Ok(Self::entire());
        }

        let lower = quotients.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = quotients.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(Self::from_outward(lower, upper))
    }

    /// Negate an interval exactly by reflecting and swapping endpoints.
    pub fn negate(self) -> Self {
        Self { lower: -self.upper, upper: -self.lower }
    }

    /// Square an interval without the dependency blow-up of x times x.
    pub fn square(self) -> Self {
        if self.is_exact_zero() {
            return self;
        }
        let left = self.lower * self.lower;
        let right = self.upper * self.upper;
        let upper = left.max(right);
        if upper.is_nan() {
            return Self::entire();
        }

        if self.contains_zero() {
            // Zero is attained, so the exact lower bound is zero.
            Self { lower: 0.0, upper: upper.next_up() }
        } else {
            let lower = left.min(right);
            Self::from_outward(lower, upper)
        }
    }

    /// Integer power with a checked domain for zero and negative exponents.
    ///
    /// Exponentiation by squaring limits the number of interval multiplications
    /// to logarithmic time. Repeated use is an enclosure, not an exact symbolic
    /// simplification; dependency can widen the result.
    pub fn checked_powi(self, exponent: i32) -> Result<Self, IntervalError> {
        if exponent == 0 {
            if self.contains_zero() {
                return Err(IntervalError::IndeterminatePower);
            }
            return Ok(Self { lower: 1.0, upper: 1.0 });
        }
        if exponent < 0 {
            if self.contains_zero() {
                return Err(IntervalError::DivisionByIntervalContainingZero);
            }
            let positive = self.powi_nonnegative(exponent.unsigned_abs());
            return Self { lower: 1.0, upper: 1.0 }.checked_divide(positive);
        }
        Ok(self.powi_nonnegative(exponent as u32))
    }

    /// Absolute value is exact endpoint reflection and requires no rounding.
    pub fn absolute(self) -> Self {
        if self.lower >= 0.0 {
            self
        } else if self.upper <= 0.0 {
            self.negate()
        } else {
            Self { lower: 0.0, upper: (-self.lower).max(self.upper) }
        }
    }

    fn powi_nonnegative(self, mut exponent: u32) -> Self {
        if exponent == 0 {
            return Self { lower: 1.0, upper: 1.0 };
        }
        if exponent == 1 {
            return self;
        }
        if exponent == 2 {
            return self.square();
        }

        let mut result = Self { lower: 1.0, upper: 1.0 };
        let mut base = self;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.checked_multiply(base);
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.square();
            }
        }
        result
    }

    fn is_exact_point(self, value: f64) -> bool {
        self.lower == value && self.upper == value
    }

    /// Widen nearest-rounded endpoint results one representable value outward.
    /// NaN is never stored as a bound.
    fn from_outward(raw_lower: f64, raw_upper: f64) -> Self {
        if raw_lower.is_nan() || raw_upper.is_nan() || raw_lower > raw_upper {
            return Self::entire();
        }
        let lower = raw_lower.next_down();
        let upper = raw_upper.next_up();
        if lower.is_nan() || upper.is_nan() || lower > upper {
            Self::entire()
        } else {
            Self { lower, upper }
        }
    }
}

/// Product of interval endpoints. Zero times infinity contributes zero to the
/// hull; NaN at that corner would otherwise be silently mishandled by min/max.
fn endpoint_product(lhs: f64, rhs: f64) -> f64 {
    if (lhs == 0.0 && rhs.is_infinite()) || (rhs == 0.0 && lhs.is_infinite()) {
        0.0
    } else {
        lhs * rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval::new(lower, upper).expect("test interval should be valid")
    }

    fn encloses(value: Interval, lower: f64, upper: f64) {
        assert!(
            value.lower() <= lower && value.upper() >= upper,
            "{value:?} must enclose [{lower}, {upper}]"
        );
    }

    #[test]
    fn rejects_nan_reversed_and_empty_infinite_bounds() {
        assert_eq!(Interval::new(f64::NAN, 1.0), Err(IntervalError::NaNBound));
        assert_eq!(Interval::new(0.0, f64::NAN), Err(IntervalError::NaNBound));
        assert_eq!(Interval::new(2.0, 1.0), Err(IntervalError::InvalidBounds));
        assert_eq!(Interval::new(f64::INFINITY, f64::INFINITY), Err(IntervalError::InvalidBounds));
        assert_eq!(Interval::new(f64::NEG_INFINITY, f64::NEG_INFINITY), Err(IntervalError::InvalidBounds));
        assert_eq!(Interval::point(f64::INFINITY), Err(IntervalError::NonFinitePoint));
    }

    #[test]
    fn outward_add_subtract_multiply_and_divide_enclose_extrema() {
        let a = interval(1.0, 2.0);
        let b = interval(3.0, 4.0);
        encloses(a.checked_add(b), 4.0, 6.0);
        encloses(a.checked_subtract(b), -3.0, -1.0);
        encloses(a.checked_multiply(b), 3.0, 8.0);
        encloses(a.checked_divide(b).unwrap(), 0.25, 2.0 / 3.0);

        encloses(interval(-2.0, -1.0).checked_multiply(b), -8.0, -3.0);
        encloses(
            interval(-2.0, 3.0).checked_multiply(interval(-4.0, 5.0)),
            -12.0,
            15.0,
        );
    }

    #[test]
    fn square_tightens_sign_when_interval_crosses_zero() {
        let squared = interval(-2.0, 3.0).square();
        assert!(squared.lower() >= 0.0);
        encloses(squared, 0.0, 9.0);

        let negative = interval(-4.0, -2.0).square();
        encloses(negative, 4.0, 16.0);
        assert!(negative.is_strictly_positive());
    }

    #[test]
    fn integer_powers_check_domains_and_preserve_even_power_sign() {
        let crossing = interval(-2.0, 3.0);
        let squared = crossing.checked_powi(2).unwrap();
        assert!(squared.lower() >= 0.0);
        encloses(squared, 0.0, 9.0);

        let cube = crossing.checked_powi(3).unwrap();
        encloses(cube, -8.0, 27.0);
        let reciprocal_square = interval(2.0, 4.0).checked_powi(-2).unwrap();
        encloses(reciprocal_square, 1.0 / 16.0, 1.0 / 4.0);

        assert_eq!(
            interval(-1.0, 1.0).checked_powi(0),
            Err(IntervalError::IndeterminatePower)
        );
        assert_eq!(
            interval(-1.0, 1.0).checked_powi(-2),
            Err(IntervalError::DivisionByIntervalContainingZero)
        );
    }

    #[test]
    fn absolute_value_preserves_nonnegative_sign() {
        assert_eq!(interval(-4.0, -2.0).absolute(), interval(2.0, 4.0));
        assert_eq!(interval(-2.0, 3.0).absolute(), interval(0.0, 3.0));
        assert_eq!(interval(2.0, 4.0).absolute(), interval(2.0, 4.0));
    }

    #[test]
    fn division_rejects_any_denominator_that_contains_zero() {
        assert_eq!(
            interval(1.0, 2.0).checked_divide(interval(-1.0, 1.0)),
            Err(IntervalError::DivisionByIntervalContainingZero)
        );
        assert_eq!(
            interval(1.0, 2.0).checked_divide(Interval::entire()),
            Err(IntervalError::DivisionByIntervalContainingZero)
        );
    }

    #[test]
    fn overflow_and_underflow_widen_without_nan_bounds() {
        let max = Interval::point(f64::MAX).unwrap();
        let overflow = max.checked_multiply(Interval::point(2.0).unwrap());
        assert_eq!(overflow.lower(), f64::MAX);
        assert_eq!(overflow.upper(), f64::INFINITY);

        let tiny = Interval::point(f64::from_bits(1)).unwrap();
        let underflow = tiny.checked_multiply(Interval::point(0.5).unwrap());
        assert!(underflow.contains(0.0));
        assert!(underflow.contains(f64::from_bits(1)));
        assert!(!underflow.lower().is_nan() && !underflow.upper().is_nan());
    }

    #[test]
    fn indeterminate_infinite_quotient_fails_wide() {
        let unbounded_positive = interval(f64::MAX, f64::INFINITY);
        let result = unbounded_positive
            .checked_divide(interval(f64::MAX, f64::INFINITY))
            .expect("divisor excludes zero");
        assert_eq!(result, Interval::entire());
    }

    #[test]
    fn small_integer_grid_encloses_exact_arithmetic() {
        for a0 in -8i32..=8 {
            for a1 in a0..=8 {
                let a = interval(a0 as f64, a1 as f64);
                for b0 in -8i32..=8 {
                    for b1 in b0..=8 {
                        let b = interval(b0 as f64, b1 as f64);
                        encloses(a.checked_add(b), (a0 + b0) as f64, (a1 + b1) as f64);
                        encloses(
                            a.checked_subtract(b),
                            (a0 - b1) as f64,
                            (a1 - b0) as f64,
                        );

                        let products = [a0 * b0, a0 * b1, a1 * b0, a1 * b1];
                        let expected_lower = *products.iter().min().unwrap() as f64;
                        let expected_upper = *products.iter().max().unwrap() as f64;
                        encloses(a.checked_multiply(b), expected_lower, expected_upper);

                        if !b.contains_zero() {
                            let quotients = [
                                a0 as f64 / b0 as f64,
                                a0 as f64 / b1 as f64,
                                a1 as f64 / b0 as f64,
                                a1 as f64 / b1 as f64,
                            ];
                            let expected_lower =
                                quotients.iter().copied().fold(f64::INFINITY, f64::min);
                            let expected_upper =
                                quotients.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            encloses(a.checked_divide(b).unwrap(), expected_lower, expected_upper);
                        }
                    }
                }
            }
        }
    }
}
