// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Small, fail-closed interval arithmetic for conservative numerical enclosures.
//!
//! Basic arithmetic widens one representable f64 step outward. Square root uses
//! its documented IEEE-754 correctly rounded contract; natural logarithm and
//! exponential use range reduction, interval series arithmetic and explicit
//! remainder bounds. Sine/cosine use bounded small-argument series; outside that
//! region their valid but coarse enclosure is [-1, 1].
//!
//! The enclosure guarantee assumes ordinary IEEE-754 binary64 round-to-nearest
//! operations and compilation without unsafe fast-math reassociation.
//! See the repository-root INTERVAL_ARITHMETIC.md for the current scope and limits.

use std::fmt;
use std::sync::OnceLock;

// For z <= 1/3, the tail after the first 64 terms of
// ln(m) = 2 * sum(z^(2j+1)/(2j+1)), z=(m-1)/(m+1), is
// less than 9 / (516 * 3^129) < 2^-134. This is exactly 2^-134.
const LN_SERIES_TERMS: usize = 64;
const LN_SERIES_TAIL_BOUND: f64 = f64::from_bits(889_u64 << 52);

// For |r| <= 0.36, the exponential Taylor remainder after terms 0..39 is
// exp(xi)*r^40/40!, whose absolute value is < 2^-200.
const EXP_SERIES_TERMS: usize = 40;
const EXP_SERIES_TAIL_BOUND: f64 = f64::from_bits(823_u64 << 52);
const EXP_REDUCTION_LIMIT: f64 = 0.36;

// Trigonometric endpoint series are evaluated only for |x| <= 2. Taylor
// remainder bounds there are < 2^-119 for cosine and < 2^-123 for sine;
// the wider exactly represented allowance 2^-80 covers either remainder.
// Critical-point handling keeps interval ranges sound beyond sine's monotone
// region without requiring platform-dependent argument reduction.
const TRIG_SMALL_ARGUMENT_LIMIT: f64 = 2.0;
const TRIG_SERIES_TAIL_BOUND: f64 = f64::from_bits(943_u64 << 52);

// The atan(1/5) and atan(1/239) series below each use a symmetric 2^-120
// remainder allowance, comfortably wider than their first omitted terms.
const PI_ATAN_SERIES_TAIL_BOUND: f64 = f64::from_bits(903_u64 << 52);

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
    /// Square root is undefined on an interval containing negative real values.
    SquareRootOfNegativeInterval,
    /// Natural logarithm is undefined for intervals that include nonpositive reals.
    LogarithmOfNonPositiveInterval,
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
            Self::SquareRootOfNegativeInterval => {
                write!(f, "cannot take square root of an interval containing negative values")
            }
            Self::LogarithmOfNonPositiveInterval => {
                write!(f, "cannot take logarithm of an interval containing nonpositive values")
            }
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

    /// Whether a finite real value belongs to this enclosure.
    ///
    /// Infinite endpoints represent unbounded real ranges; infinity itself is
    /// not treated as a member of the represented real-number set.
    pub fn contains(self, value: f64) -> bool {
        value.is_finite() && self.lower <= value && value <= self.upper
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

    /// Square root with an outward enclosure of the correctly-rounded result.
    ///
    /// Rust documents `f64::sqrt` as returning the IEEE-754 correctly rounded
    /// square root. Taking the adjacent representable values outward therefore
    /// encloses the real result. Zero is kept exact to preserve the nonnegative
    /// invariant; positive infinity stays unbounded. Any interval containing a
    /// negative real input is rejected rather than silently clipping its domain.
    pub fn checked_sqrt(self) -> Result<Self, IntervalError> {
        if self.lower < 0.0 {
            return Err(IntervalError::SquareRootOfNegativeInterval);
        }

        let lower_root = self.lower.sqrt();
        let upper_root = self.upper.sqrt();
        let lower = if self.lower == 0.0 {
            0.0
        } else {
            lower_root.next_down()
        };
        let upper = if self.upper == 0.0 {
            0.0
        } else {
            upper_root.next_up()
        };

        // Monotonicity of sqrt preserves endpoint order, and the branches above
        // preserve exact zero and valid infinite upper bounds.
        Ok(Self { lower, upper })
    }

    /// Natural logarithm enclosed by range reduction and a bounded atanh series.
    ///
    /// Inputs must be strictly positive. The implementation decomposes each
    /// finite endpoint exactly as m*2^k with 1 <= m < 2, then encloses
    /// ln(m)=2*atanh((m-1)/(m+1)) using 64 positive series terms and an explicit
    /// remainder bound. ln(2) itself is enclosed by evaluating the same series
    /// at m=2. No platform `f64::ln` result is used as a bound.
    pub fn checked_ln(self) -> Result<Self, IntervalError> {
        if self.lower <= 0.0 {
            return Err(IntervalError::LogarithmOfNonPositiveInterval);
        }

        let ln_two = ln_mantissa_enclosure(2.0)?;
        let lower_log = ln_positive_point(self.lower, ln_two)?;
        let upper_bound = if self.upper == f64::INFINITY {
            f64::INFINITY
        } else {
            ln_positive_point(self.upper, ln_two)?.upper
        };

        // ln is monotone on (0,+infinity); use the lower endpoint's lower
        // enclosure and upper endpoint's upper enclosure.
        if lower_log.lower > upper_bound {
            return Ok(Self::entire());
        }
        Self::new(lower_log.lower, upper_bound)
    }

    /// Exponential enclosed by range reduction and a Taylor series with a
    /// bounded remainder.
    ///
    /// The finite-point path reduces x = r + k*ln(2), with |r| <= 0.36 checked
    /// against an interval enclosure of ln(2). The first 40 terms of exp(r) are
    /// evaluated using interval arithmetic; the absolute Taylor remainder is
    /// bounded by 2^-200. Scaling uses an exactly represented power of two.
    /// Extremely large or tiny inputs widen conservatively instead of relying
    /// on the platform's unspecified-precision `f64::exp`.
    pub fn checked_exp(self) -> Result<Self, IntervalError> {
        let ln_two = ln_mantissa_enclosure(2.0)?;
        let lower = if self.lower == f64::NEG_INFINITY {
            0.0
        } else {
            exp_positive_point(self.lower, ln_two)?.lower.max(0.0)
        };
        let upper = if self.upper == f64::INFINITY {
            f64::INFINITY
        } else {
            exp_positive_point(self.upper, ln_two)?.upper
        };

        // exp is strictly positive and monotone over the real line.
        Self::new(lower, upper)
    }

    /// Sine enclosure with a bounded local series and a conservative global fallback.
    ///
    /// For intervals wholly inside [-2, 2], endpoint Taylor polynomials through
    /// degree 39 are evaluated with interval arithmetic and widened by 2^-80.
    /// The endpoint hull is augmented by -1 or +1 when the interval enters the
    /// corresponding possible sine-extremum region. This may be wider than the
    /// true range, but avoids unqualified argument reduction. Larger/unbounded
    /// intervals return [-1, 1].
    pub fn checked_sin(self) -> Result<Self, IntervalError> {
        if self.lower < -TRIG_SMALL_ARGUMENT_LIMIT
            || self.upper > TRIG_SMALL_ARGUMENT_LIMIT
        {
            return Ok(Self { lower: -1.0, upper: 1.0 });
        }

        let lower_point = sin_point_enclosure(self.lower)?;
        let upper_point = sin_point_enclosure(self.upper)?;
        let mut lower = lower_point.lower.min(upper_point.lower).max(-1.0);
        let mut upper = lower_point.upper.max(upper_point.upper).min(1.0);
        // Include an extremum exactly when this interval overlaps our cached,
        // series-derived enclosure of +/-pi/2. If that enclosure cannot be
        // constructed, fail wide rather than make an ungrounded extrema claim.
        let Some(half_pi) = half_pi_enclosure() else {
            return Ok(Self { lower: -1.0, upper: 1.0 });
        };
        let negative_half_pi = half_pi.negate();
        if self.lower <= half_pi.upper && self.upper >= half_pi.lower {
            upper = 1.0;
        }
        if self.lower <= negative_half_pi.upper && self.upper >= negative_half_pi.lower {
            lower = -1.0;
        }
        if lower <= upper {
            Ok(Self { lower, upper })
        } else {
            Ok(Self { lower: -1.0, upper: 1.0 })
        }
    }

    /// Cosine enclosure with a bounded local series and a conservative global fallback.
    ///
    /// For intervals wholly inside [-2, 2], endpoint Taylor polynomials through
    /// degree 38 use interval arithmetic and a 2^-80 remainder allowance. Since
    /// 2 < pi, cosine has no critical point there except its maximum at zero;
    /// the endpoint hull plus that exact maximum gives a sound range. Outside
    /// this region, return the sound but coarse range [-1, 1].
    pub fn checked_cos(self) -> Result<Self, IntervalError> {
        if self.lower < -TRIG_SMALL_ARGUMENT_LIMIT
            || self.upper > TRIG_SMALL_ARGUMENT_LIMIT
        {
            return Ok(Self { lower: -1.0, upper: 1.0 });
        }

        let at_lower = cos_point_enclosure(self.lower)?;
        let at_upper = cos_point_enclosure(self.upper)?;
        let lower = at_lower.lower.min(at_upper.lower).max(-1.0);
        let upper = if self.contains_zero() {
            1.0
        } else {
            at_lower.upper.max(at_upper.upper).min(1.0)
        };
        if lower <= upper {
            Ok(Self { lower, upper })
        } else {
            Ok(Self { lower: -1.0, upper: 1.0 })
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

/// Enclose atan(1/d) with the alternating series and a symmetric remainder bound.
fn atan_reciprocal_enclosure(
    denominator: u32,
    terms: usize,
) -> Result<Interval, IntervalError> {
    if denominator == 0 || terms == 0 {
        return Err(IntervalError::InvalidBounds);
    }

    // Division in interval arithmetic encloses the exact rational 1/d; this
    // avoids treating its rounded f64 representation as exact.
    let z = Interval::point(1.0)?.checked_divide(Interval::point(denominator as f64)?)?;
    let negative_z_squared = z.checked_multiply(z).negate();
    let mut term = z;
    let mut sum = term;
    for n in 0..(terms - 1) {
        let numerator = Interval::point((2 * n + 1) as f64)?;
        let divisor = Interval::point((2 * n + 3) as f64)?;
        term = term
            .checked_multiply(negative_z_squared)
            .checked_multiply(numerator)
            .checked_divide(divisor)?;
        sum = sum.checked_add(term);
    }

    let remainder = Interval::new(
        -PI_ATAN_SERIES_TAIL_BOUND,
        PI_ATAN_SERIES_TAIL_BOUND,
    )?;
    Ok(sum.checked_add(remainder))
}

/// Machin's identity gives pi/2 = 8 atan(1/5) - 2 atan(1/239).
/// The interval series and remainder bounds are cached to avoid paying the
/// construction cost on subsequent evaluations. Failure is represented as
/// None, so callers can fail wide.
fn half_pi_enclosure() -> Option<Interval> {
    static HALF_PI: OnceLock<Option<Interval>> = OnceLock::new();
    *HALF_PI.get_or_init(|| {
        let atan_fifth = atan_reciprocal_enclosure(5, 25).ok()?;
        let atan_239th = atan_reciprocal_enclosure(239, 8).ok()?;
        let eight = Interval::point(8.0).ok()?;
        let minus_two = Interval::point(-2.0).ok()?;
        Some(
            atan_fifth
                .checked_multiply(eight)
                .checked_add(atan_239th.checked_multiply(minus_two)),
        )
    })
}

/// Compute one alternating Taylor term using interval division by exact small integers.
fn next_alternating_trig_term(
    term: Interval,
    x_squared: Interval,
    first_divisor: u32,
    second_divisor: u32,
) -> Result<Interval, IntervalError> {
    let numerator = term.checked_multiply(x_squared).negate();
    let first = Interval::point(first_divisor as f64)?;
    let second = Interval::point(second_divisor as f64)?;
    numerator.checked_divide(first)?.checked_divide(second)
}

/// Enclose sin(x) for |x| <= 2 using terms through degree 39.
fn sin_point_enclosure(value: f64) -> Result<Interval, IntervalError> {
    if value == 0.0 {
        return Interval::point(0.0);
    }
    let x = Interval::point(value)?;
    let x_squared = x.checked_multiply(x);
    let mut term = x;
    let mut sum = x;
    for n in 1..=19 {
        term = next_alternating_trig_term(term, x_squared, 2 * n, 2 * n + 1)?;
        sum = sum.checked_add(term);
    }
    let remainder = Interval::new(-TRIG_SERIES_TAIL_BOUND, TRIG_SERIES_TAIL_BOUND)?;
    Ok(sum.checked_add(remainder))
}

/// Enclose cos(x) for |x| <= 2 using terms through degree 38.
fn cos_point_enclosure(value: f64) -> Result<Interval, IntervalError> {
    if value == 0.0 {
        return Interval::point(1.0);
    }
    let x = Interval::point(value)?;
    let x_squared = x.checked_multiply(x);
    let mut term = Interval::point(1.0)?;
    let mut sum = term;
    for n in 1..=19 {
        term = next_alternating_trig_term(term, x_squared, 2 * n - 1, 2 * n)?;
        sum = sum.checked_add(term);
    }
    let remainder = Interval::new(-TRIG_SERIES_TAIL_BOUND, TRIG_SERIES_TAIL_BOUND)?;
    Ok(sum.checked_add(remainder))
}

/// Enclose ln(m) for the exact binary64 input 1 <= m <= 2 using
/// ln(m) = 2 * atanh((m-1)/(m+1)). All series arithmetic uses intervals.
fn ln_mantissa_enclosure(m: f64) -> Result<Interval, IntervalError> {
    if !(1.0..=2.0).contains(&m) {
        return Err(IntervalError::InvalidBounds);
    }
    if m == 1.0 {
        return Interval::point(0.0);
    }

    let numerator = Interval::point(m)?.checked_subtract(Interval::point(1.0)?);
    let denominator = Interval::point(m)?.checked_add(Interval::point(1.0)?);
    let z = numerator.checked_divide(denominator)?;
    let z_squared = z.checked_multiply(z);

    let mut power = z;
    let mut series = Interval::point(0.0)?;
    for index in 0..LN_SERIES_TERMS {
        let odd = (2 * index + 1) as f64;
        let term = power.checked_divide(Interval::point(odd)?)?;
        series = series.checked_add(term);
        power = power.checked_multiply(z_squared);
    }

    let twice_series = series.checked_multiply(Interval::point(2.0)?);
    let tail = Interval::point(LN_SERIES_TAIL_BOUND)?;
    let widened = twice_series.checked_add(tail);

    // The omitted terms are nonnegative, so retain the partial sum's lower
    // bound while using the tail-expanded upper bound.
    Interval::new(twice_series.lower, widened.upper)
}

/// Enclose ln(value) for a positive finite binary64 value, using its exact
/// significand and exponent. Scaling never passes through an approximate ln call.
fn ln_positive_point(value: f64, ln_two: Interval) -> Result<Interval, IntervalError> {
    debug_assert!(value.is_finite() && value > 0.0);
    if value == 1.0 {
        return Interval::point(0.0);
    }

    const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
    const EXPONENT_MASK: u64 = 0x7ff;

    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & EXPONENT_MASK) as i32;
    let fraction = bits & FRACTION_MASK;

    let (mantissa, exponent) = if exponent_bits != 0 {
        (
            f64::from_bits((1023_u64 << 52) | fraction),
            exponent_bits - 1023,
        )
    } else {
        // Normalize subnormals exactly using their integer significand.
        debug_assert_ne!(fraction, 0);
        let highest_bit = 63_i32 - fraction.leading_zeros() as i32;
        let shift = (52_i32 - highest_bit) as u32;
        let normalized = fraction << shift;
        (
            f64::from_bits((1023_u64 << 52) | (normalized & FRACTION_MASK)),
            highest_bit - 1074,
        )
    };

    let log_mantissa = ln_mantissa_enclosure(mantissa)?;
    let log_scale = ln_two.checked_multiply(Interval::point(exponent as f64)?);
    Ok(log_mantissa.checked_add(log_scale))
}

/// Enclose exp(r) by Taylor expansion. The input interval must lie inside
/// [-EXP_REDUCTION_LIMIT, EXP_REDUCTION_LIMIT].
fn exp_reduced_enclosure(r: Interval) -> Result<Interval, IntervalError> {
    if r.lower < -EXP_REDUCTION_LIMIT || r.upper > EXP_REDUCTION_LIMIT {
        return Ok(Interval::new(0.0, f64::INFINITY)?);
    }

    if r.is_exact_zero() {
        return Interval::point(1.0);
    }

    let mut sum = Interval::point(1.0)?;
    let mut term = Interval::point(1.0)?;
    for index in 1..EXP_SERIES_TERMS {
        term = term.checked_multiply(r);
        term = term.checked_divide(Interval::point(index as f64)?)?;
        sum = sum.checked_add(term);
    }

    // Taylor's theorem bounds the omitted remainder in magnitude. Widen both
    // sides because the reduced argument may be negative.
    let remainder = Interval::new(-EXP_SERIES_TAIL_BOUND, EXP_SERIES_TAIL_BOUND)?;
    let widened = sum.checked_add(remainder);
    if widened.upper < 0.0 {
        return Interval::new(0.0, f64::INFINITY);
    }
    Interval::new(widened.lower.max(0.0), widened.upper)
}

/// Reconstruct exp(x) for a finite endpoint without using f64::exp as a bound.
fn exp_positive_point(value: f64, ln_two: Interval) -> Result<Interval, IntervalError> {
    debug_assert!(value.is_finite());

    if value == 0.0 {
        return Interval::point(1.0);
    }
    if value < -745.0 {
        // Derive the subnormal cutoff from the same ln(2) enclosure used by
        // range reduction. If this check unexpectedly fails, widen safely.
        let log_min_subnormal = ln_two.checked_multiply(Interval::point(-1074.0)?);
        if value < log_min_subnormal.lower {
            return Interval::new(0.0, f64::from_bits(1));
        }
        return Interval::new(0.0, f64::INFINITY);
    }
    if value > 700.0 {
        // Broad but safe; do not manufacture a finite bound beyond this range.
        return Interval::new(0.0, f64::INFINITY);
    }

    let ln_two_mid = ln_two.lower + (ln_two.upper - ln_two.lower) * 0.5;
    let exponent_float = (value / ln_two_mid).round();
    if !exponent_float.is_finite() || exponent_float < -2000.0 || exponent_float > 2000.0 {
        return Interval::new(0.0, f64::INFINITY);
    }
    let exponent = exponent_float as i32;

    // Check the actual interval reduction before using any range-based fallback.
    let exponent_interval = Interval::point(exponent as f64)?;
    let scaled_ln_two = ln_two.checked_multiply(exponent_interval);
    let reduced = Interval::point(value)?.checked_subtract(scaled_ln_two);
    if reduced.lower < -EXP_REDUCTION_LIMIT || reduced.upper > EXP_REDUCTION_LIMIT {
        return Interval::new(0.0, f64::INFINITY);
    }

    // Once the checked reduction is known to satisfy |r| <= 0.36, exp(r) < 2;
    // this makes the underflow enclosure [0, min_subnormal] conservative.
    if exponent < -1074 {
        return Interval::new(0.0, f64::from_bits(1));
    }
    if exponent > 1023 {
        return Interval::new(0.0, f64::INFINITY);
    }

    let exp_reduced = exp_reduced_enclosure(reduced)?;
    let scale = exact_power_of_two(exponent)
        .ok_or(IntervalError::InvalidBounds)?;
    let scaled = exp_reduced.checked_multiply(Interval::point(scale)?);
    // Exact exp(x) is positive. Outward rounding of a subnormal product can
    // extend below zero; tightening that known mathematical invariant is safe.
    Interval::new(scaled.lower.max(0.0), scaled.upper)
}

/// Return 2^exponent exactly for representable normal or subnormal powers of two.
fn exact_power_of_two(exponent: i32) -> Option<f64> {
    if (-1022..=1023).contains(&exponent) {
        let encoded = ((exponent + 1023) as u64) << 52;
        Some(f64::from_bits(encoded))
    } else if (-1074..-1022).contains(&exponent) {
        Some(f64::from_bits(1_u64 << (exponent + 1074)))
    } else {
        None
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
    fn unbounded_endpoints_enclose_finite_values_not_infinity_as_a_real_member() {
        let whole = Interval::entire();
        assert!(whole.contains(-f64::MAX));
        assert!(whole.contains(0.0));
        assert!(whole.contains(f64::MAX));
        assert!(!whole.contains(f64::INFINITY));
        assert!(!whole.contains(f64::NEG_INFINITY));
        assert!(!whole.contains(f64::NAN));
    }

    #[test]
    fn rejects_nan_reversed_and_empty_infinite_bounds() {
        assert_eq!(Interval::new(f64::NAN, 1.0), Err(IntervalError::NaNBound));
        assert_eq!(Interval::new(0.0, f64::NAN), Err(IntervalError::NaNBound));
        assert_eq!(Interval::new(2.0, 1.0), Err(IntervalError::InvalidBounds));
        assert_eq!(Interval::new(f64::INFINITY, f64::INFINITY), Err(IntervalError::InvalidBounds));
        assert_eq!(
            Interval::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
            Err(IntervalError::InvalidBounds)
        );
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
    fn square_root_encloses_correctly_rounded_roots_and_preserves_zero() {
        let zero = interval(0.0, 0.0).checked_sqrt().unwrap();
        assert_eq!(zero, interval(0.0, 0.0));

        let exact_range = interval(4.0, 9.0).checked_sqrt().unwrap();
        assert!(exact_range.lower() <= 2.0);
        assert!(exact_range.upper() >= 3.0);
        assert!(exact_range.lower() >= 0.0);

        let irrational = interval(2.0, 2.0).checked_sqrt().unwrap();
        let rounded = 2.0_f64.sqrt();
        assert_eq!(irrational.lower(), rounded.next_down());
        assert_eq!(irrational.upper(), rounded.next_up());

        // Exercise extreme positive inputs, including subnormals and the
        // largest finite number; the output must remain ordered and nonnegative.
        for bits in [
            1_u64,
            2,
            (1_u64 << 52) - 1,
            1_u64 << 52,
            0x0010_0000_0000_0001,
            0x3ff0_0000_0000_0001,
            0x7fef_ffff_ffff_ffff,
        ] {
            let value = f64::from_bits(bits);
            let result = Interval::point(value).unwrap().checked_sqrt().unwrap();
            let rounded_root = value.sqrt();
            assert!(result.lower() <= rounded_root);
            assert!(result.upper() >= rounded_root);
            assert!(result.lower() >= 0.0);
            assert!(result.lower() <= result.upper());
        }
    }

    #[test]
    fn square_root_rejects_intervals_with_negative_values() {
        assert_eq!(
            interval(-1.0, 4.0).checked_sqrt(),
            Err(IntervalError::SquareRootOfNegativeInterval)
        );
        assert_eq!(
            Interval::new(f64::NEG_INFINITY, f64::INFINITY)
                .unwrap()
                .checked_sqrt(),
            Err(IntervalError::SquareRootOfNegativeInterval)
        );

        let unbounded = interval(1.0, f64::INFINITY).checked_sqrt().unwrap();
        assert!(unbounded.lower() <= 1.0);
        assert_eq!(unbounded.upper(), f64::INFINITY);
    }

    #[test]
    fn exponential_encloses_reference_values_across_supported_and_wide_ranges() {
        let samples = [
            -745.0,
            -744.7,
            -100.0,
            -10.0,
            -1.0,
            -0.1,
            0.0,
            0.1,
            0.5,
            1.0,
            2.0,
            10.0,
            100.0,
            700.0,
            701.0,
        ];

        for value in samples {
            let result = Interval::point(value).unwrap().checked_exp().unwrap();
            let reference = value.exp();
            assert!(
                result.contains(reference),
                "enclosure {result:?} should contain platform exp({value:?}) = {reference:?}"
            );
            assert!(result.lower() >= 0.0);
            assert!(result.lower() <= result.upper());
        }
        assert_eq!(
            Interval::point(0.0).unwrap().checked_exp().unwrap(),
            Interval::point(1.0).unwrap()
        );
    }

    #[test]
    fn exponential_is_monotone_and_handles_unbounded_domains() {
        let range = interval(-2.0, 3.0).checked_exp().unwrap();
        assert!(range.lower() <= (-2.0_f64).exp());
        assert!(range.upper() >= 3.0_f64.exp());
        assert!(range.lower() >= 0.0);

        let lower_unbounded = interval(f64::NEG_INFINITY, 0.0).checked_exp().unwrap();
        assert_eq!(lower_unbounded.lower(), 0.0);
        assert!(lower_unbounded.upper() >= 1.0);

        let upper_unbounded = interval(0.0, f64::INFINITY).checked_exp().unwrap();
        assert_eq!(upper_unbounded.lower(), 1.0);
        assert_eq!(upper_unbounded.upper(), f64::INFINITY);
    }

    #[test]
    fn natural_logarithm_encloses_reference_values_across_binary64_ranges() {
        let samples = [
            f64::from_bits(1),                    // smallest subnormal
            f64::from_bits(2),
            f64::from_bits((1_u64 << 52) - 1),   // largest subnormal
            f64::MIN_POSITIVE,
            f64::from_bits(1.0_f64.to_bits() - 1),
            1.0,
            f64::from_bits(1.0_f64.to_bits() + 1),
            0.5,
            2.0,
            3.0,
            10.0,
            f64::MAX,
        ];

        for value in samples {
            let result = Interval::point(value).unwrap().checked_ln().unwrap();
            assert!(
                result.contains(value.ln()),
                "computed enclosure {result:?} should contain platform ln({value:?}) = {:?}",
                value.ln()
            );
            assert!(result.lower() <= result.upper());
        }

        assert_eq!(
            Interval::point(1.0).unwrap().checked_ln().unwrap(),
            Interval::point(0.0).unwrap()
        );
        assert!(
            Interval::point(2.0)
                .unwrap()
                .checked_ln()
                .unwrap()
                .contains(std::f64::consts::LN_2)
        );
        assert!(
            Interval::point(0.5)
                .unwrap()
                .checked_ln()
                .unwrap()
                .contains(-std::f64::consts::LN_2)
        );
    }

    #[test]
    fn natural_logarithm_checks_domain_and_unbounded_upper_endpoint() {
        assert_eq!(
            interval(-1.0, 2.0).checked_ln(),
            Err(IntervalError::LogarithmOfNonPositiveInterval)
        );
        assert_eq!(
            interval(0.0, 2.0).checked_ln(),
            Err(IntervalError::LogarithmOfNonPositiveInterval)
        );
        assert_eq!(
            interval(f64::NEG_INFINITY, f64::INFINITY).checked_ln(),
            Err(IntervalError::LogarithmOfNonPositiveInterval)
        );

        let unbounded = interval(1.0, f64::INFINITY).checked_ln().unwrap();
        assert_eq!(unbounded.lower(), 0.0);
        assert_eq!(unbounded.upper(), f64::INFINITY);
    }

    #[test]
    fn natural_logarithm_interval_is_monotone_and_contains_sampled_values() {
        let input = interval(0.25, 16.0);
        let output = input.checked_ln().unwrap();
        assert!(output.contains(0.25_f64.ln()));
        assert!(output.contains(1.0_f64.ln()));
        assert!(output.contains(2.0_f64.ln()));
        assert!(output.contains(16.0_f64.ln()));
        assert!(output.lower() <= 0.0);
        assert!(output.upper() >= 16.0_f64.ln());
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
    fn bounded_trigonometric_series_enclose_point_and_interval_ranges() {
        let zero = Interval::point(0.0).unwrap();
        assert_eq!(zero.checked_sin().unwrap(), zero);
        assert_eq!(zero.checked_cos().unwrap(), Interval::point(1.0).unwrap());

        let half_pi = half_pi_enclosure().expect("Machin series should enclose pi/2");
        assert!(half_pi.contains(std::f64::consts::FRAC_PI_2));
        assert!(half_pi.upper() - half_pi.lower() < 1e-12);

        let edge_sine = Interval::point(2.0).unwrap().checked_sin().unwrap();
        assert!(edge_sine.contains(0.9092974268256817));
        assert!(edge_sine.upper() < 0.95);
        assert!(edge_sine.upper() - edge_sine.lower() < 1e-12);

        let edge_cosine = Interval::point(2.0).unwrap().checked_cos().unwrap();
        assert!(edge_cosine.contains(-0.4161468365471424));

        let crossing = interval(-2.0, 2.0);
        let sine = crossing.checked_sin().unwrap();
        assert_eq!(sine, interval(-1.0, 1.0));

        let cosine = crossing.checked_cos().unwrap();
        assert!(cosine.lower() <= -0.4161468365471424);
        assert_eq!(cosine.upper(), 1.0);
        assert!(cosine.lower() <= cosine.upper());

        let outer_positive = interval(1.6, 1.8).checked_sin().unwrap();
        assert!(outer_positive.lower() > 0.9);
        assert!(outer_positive.upper() < 1.0);
        let crosses_maximum = interval(1.55, 1.6).checked_sin().unwrap();
        assert_eq!(crosses_maximum.upper(), 1.0);
        let cosine_outer = interval(1.6, 2.0).checked_cos().unwrap();
        assert!(cosine_outer.lower() > -0.5);
        assert!(cosine_outer.upper() < 0.0);
    }

    #[test]
    fn trigonometric_functions_fail_wide_outside_the_certified_small_range() {
        let unit_range = interval(-1.0, 1.0);
        assert_eq!(interval(2.0, 2.0000000001).checked_sin().unwrap(), unit_range);
        assert_eq!(interval(-2.1, 2.0).checked_cos().unwrap(), unit_range);
        assert_eq!(interval(f64::NEG_INFINITY, f64::INFINITY).checked_sin().unwrap(), unit_range);
        assert_eq!(interval(1000.0, 1001.0).checked_cos().unwrap(), unit_range);
    }

    #[test]
    fn bounded_trigonometric_series_cross_check_representable_points() {
        for step in -200_i32..=200 {
            let value = step as f64 / 100.0;
            let point = Interval::point(value).unwrap();
            assert!(
                point.checked_sin().unwrap().contains(value.sin()),
                "sin enclosure should contain reference at {value}"
            );
            assert!(
                point.checked_cos().unwrap().contains(value.cos()),
                "cos enclosure should contain reference at {value}"
            );
        }
    }

    #[test]
    fn trigonometric_interval_ranges_enclose_deterministic_grid() {
        // All 153 ordered endpoints on a quarter-unit grid in [-2, 2], with
        // nine interior probes per interval. This exercises monotone segments,
        // zero-crossing cosine ranges, and the conservative +/-1 sine extrema.
        for lower_step in -8_i32..=8 {
            let lower = lower_step as f64 / 4.0;
            for upper_step in lower_step..=8 {
                let upper = upper_step as f64 / 4.0;
                let input = interval(lower, upper);
                let sine = input.checked_sin().unwrap();
                let cosine = input.checked_cos().unwrap();
                for probe in 0_i32..=8 {
                    let x = if probe == 0 {
                        lower
                    } else if probe == 8 {
                        upper
                    } else {
                        lower + (upper - lower) * (probe as f64 / 8.0)
                    };
                    assert!(
                        sine.contains(x.sin()),
                        "sine range {sine:?} failed for input [{lower}, {upper}] at x={x}"
                    );
                    assert!(
                        cosine.contains(x.cos()),
                        "cosine range {cosine:?} failed for input [{lower}, {upper}] at x={x}"
                    );
                }
            }
        }
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
