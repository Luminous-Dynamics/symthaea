// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded arbitrary-precision rational arithmetic.
//!
//! This is an opt-in companion to the legacy fixed-width Number::Rational.
//! It intentionally does not silently narrow or convert to f64. Callers choose
//! a RationalBudget and receive a typed error if an operand or result exceeds it.
//! The wire representation uses decimal strings so JSON consumers do not lose
//! integer precision through IEEE-754 number parsing.

use super::numeric_tower::Number;
use crate::hdc::binary_hv::BinaryHV;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Hard ceiling on configured numerator/denominator size, in bits.
pub const HARD_MAX_COMPONENT_BITS: u64 = 16_384;
/// Default numerator/denominator budget, in bits.
pub const DEFAULT_MAX_COMPONENT_BITS: u64 = 4_096;

/// Resource policy for parsing and computing with arbitrary-precision rationals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalBudget {
    max_component_bits: u64,
}

impl RationalBudget {
    /// Create a budget from 1 through HARD_MAX_COMPONENT_BITS bits per
    /// normalized numerator or denominator.
    pub fn new(max_component_bits: u64) -> Result<Self, ExactRationalError> {
        if max_component_bits == 0 || max_component_bits > HARD_MAX_COMPONENT_BITS {
            return Err(ExactRationalError::InvalidBudget);
        }
        Ok(Self { max_component_bits })
    }

    /// Maximum allowed bit length of each normalized numerator/denominator.
    pub fn max_component_bits(self) -> u64 {
        self.max_component_bits
    }

    fn validate_decimal_component(self, text: &str) -> Result<(), ExactRationalError> {
        let digits = text.trim_start_matches(|ch| ch == '-' || ch == '+');
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ExactRationalError::InvalidInteger);
        }

        // Bound parsing work before allocating a BigInt. The deliberately
        // conservative decimal digit ceiling allows a small guard band; the
        // exact bit length is checked immediately after parsing/normalizing.
        let max_decimal_digits = self
            .max_component_bits
            .saturating_mul(30_103)
            .checked_div(100_000)
            .unwrap_or(0)
            .saturating_add(2);
        if digits.len() as u64 > max_decimal_digits {
            return Err(ExactRationalError::ResourceLimitExceeded);
        }
        Ok(())
    }

    fn validate_value(self, value: &BigRational) -> Result<(), ExactRationalError> {
        if value.numer().bits() > self.max_component_bits
            || value.denom().bits() > self.max_component_bits
        {
            return Err(ExactRationalError::ResourceLimitExceeded);
        }
        Ok(())
    }
}

impl Default for RationalBudget {
    fn default() -> Self {
        Self {
            max_component_bits: DEFAULT_MAX_COMPONENT_BITS,
        }
    }
}

/// Errors from bounded arbitrary-precision rational operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactRationalError {
    /// The budget is zero or above the hard implementation ceiling.
    InvalidBudget,
    /// A numerator or denominator is not a valid signed decimal integer.
    InvalidInteger,
    /// A rational denominator is zero.
    ZeroDenominator,
    /// A division operation used a zero rational divisor.
    DivisionByZero,
    /// Parsing, an operand, or a result exceeded the configured resource budget.
    ResourceLimitExceeded,
    /// The exact value cannot be represented by the fixed-width numeric tower.
    ExactResultOutOfRange,
    /// An approximate floating-point input cannot be imported as an exact rational.
    ApproximateInput,
}

impl fmt::Display for ExactRationalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBudget => write!(f, "invalid rational bit budget"),
            Self::InvalidInteger => write!(f, "invalid decimal integer"),
            Self::ZeroDenominator => write!(f, "rational denominator cannot be zero"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::ResourceLimitExceeded => write!(f, "rational resource budget exceeded"),
            Self::ExactResultOutOfRange => {
                write!(f, "exact rational is outside the numeric-tower range")
            },
            Self::ApproximateInput => {
                write!(f, "approximate numeric input cannot be imported as an exact rational")
            },
        }
    }
}

impl std::error::Error for ExactRationalError {}

/// Stable wire representation for an exact rational.
/// Decimal strings avoid JSON number precision loss for large integers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactRationalWire {
    /// Canonical signed decimal numerator.
    pub numerator: String,
    /// Canonical positive decimal denominator.
    pub denominator: String,
}

/// A normalized arbitrary-precision rational with explicit budget checks.
///
/// The inner BigRational is private, so callers cannot mutate it around the
/// budget checks. Every operation rechecks both operands and the normalized result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactRational {
    value: BigRational,
}

impl ExactRational {
    /// Parse signed decimal numerator/denominator under an explicit resource budget.
    pub fn parse(
        numerator: &str,
        denominator: &str,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_decimal_component(numerator)?;
        budget.validate_decimal_component(denominator)?;

        let numerator = BigInt::from_str(numerator).map_err(|_| ExactRationalError::InvalidInteger)?;
        let denominator =
            BigInt::from_str(denominator).map_err(|_| ExactRationalError::InvalidInteger)?;
        if denominator.is_zero() {
            return Err(ExactRationalError::ZeroDenominator);
        }

        let value = BigRational::new(numerator, denominator);
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Construct an exact integer under an explicit resource budget.
    pub fn from_integer(
        value: &str,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        Self::parse(value, "1", budget)
    }

    /// Rehydrate a wire value with a caller-supplied budget.
    pub fn from_wire(
        wire: &ExactRationalWire,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        Self::parse(&wire.numerator, &wire.denominator, budget)
    }

    /// Export a normalized, deterministic wire value.
    pub fn to_wire(&self) -> ExactRationalWire {
        ExactRationalWire {
            numerator: self.value.numer().to_str_radix(10),
            denominator: self.value.denom().to_str_radix(10),
        }
    }

    /// Lossless, versioned, deterministic bytes for exact identity and transport.
    ///
    /// Unlike an HDC projection or a fixed-width seed, these bytes are an
    /// unambiguous representation: normalized decimal numerator and denominator
    /// components are length-prefixed and preceded by a format/version tag.
    pub fn canonical_encoding_bytes(&self) -> Vec<u8> {
        const PREFIX: &[u8] = b"SYMTHAEA\0EXACT-RATIONAL\0V1\0";
        let numerator = self.value.numer().to_str_radix(10);
        let denominator = self.value.denom().to_str_radix(10);
        let mut encoded = Vec::with_capacity(
            PREFIX.len() + 16 + numerator.len() + denominator.len(),
        );
        encoded.extend_from_slice(PREFIX);
        encoded.extend_from_slice(&(numerator.len() as u64).to_be_bytes());
        encoded.extend_from_slice(numerator.as_bytes());
        encoded.extend_from_slice(&(denominator.len() as u64).to_be_bytes());
        encoded.extend_from_slice(denominator.as_bytes());
        encoded
    }

    /// Deterministic HDC projection of this canonical rational value.
    ///
    /// This finite hypervector is a useful representation for similarity and
    /// binding, but is not a lossless identifier or proof object. Use the
    /// canonical_encoding_bytes method when exact identity matters.
    pub fn to_hdc(&self) -> BinaryHV {
        let canonical = self.canonical_encoding_bytes();
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"SYMTHAEA.EXACT_RATIONAL.HDC.V1\0");
        hasher.update(&canonical);
        let mut output = [0u8; BinaryHV::BYTES];
        hasher.finalize_xof().fill(&mut output);
        BinaryHV(output)
    }

    /// Convert into the legacy numeric tower only when the exact value fits.
    /// No approximation or overflow fallback is allowed on this path.
    pub fn try_to_number(&self) -> Result<Number, ExactRationalError> {
        if self.value.denom() == &BigInt::from(1u8) {
            if let Some(value) = self.value.numer().to_u64() {
                return Ok(Number::Natural(value));
            }
            if let Some(value) = self.value.numer().to_i64() {
                return Ok(Number::Integer(value));
            }
            return Err(ExactRationalError::ExactResultOutOfRange);
        }

        let numerator = self
            .value
            .numer()
            .to_i64()
            .ok_or(ExactRationalError::ExactResultOutOfRange)?;
        let denominator = self
            .value
            .denom()
            .to_i64()
            .filter(|value| *value > 0)
            .ok_or(ExactRationalError::ExactResultOutOfRange)?;
        Ok(Number::Rational { numerator, denominator })
    }

    /// Import an exact value from the legacy numeric tower.
    ///
    /// Natural, integer, and valid rational variants are imported without
    /// rounding. Number::Real is rejected even if it looks integral because
    /// the variant itself represents an approximation, not exact provenance.
    pub fn try_from_number_exact(
        value: &Number,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        match value {
            Number::Natural(value) => Self::from_integer(&value.to_string(), budget),
            Number::Integer(value) => Self::from_integer(&value.to_string(), budget),
            Number::Rational { numerator, denominator } => {
                Self::parse(&numerator.to_string(), &denominator.to_string(), budget)
            }
            Number::Real(_) => Err(ExactRationalError::ApproximateInput),
        }
    }

    /// Canonical numerator as a decimal string.
    pub fn numerator_str(&self) -> String {
        self.value.numer().to_str_radix(10)
    }

    /// Canonical positive denominator as a decimal string.
    pub fn denominator_str(&self) -> String {
        self.value.denom().to_str_radix(10)
    }

    /// Convert to a finite f64 approximation if representable.
    ///
    /// This is deliberately separate from exact arithmetic; None means that
    /// the rational cannot be represented as a finite floating-point value.
    pub fn to_f64_approx(&self) -> Option<f64> {
        self.value.to_f64().filter(|value| value.is_finite())
    }

    /// Add exactly, failing closed if operands or normalized result exceed budget.
    pub fn checked_add(
        &self,
        other: &Self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        budget.validate_value(&other.value)?;
        let value = &self.value + &other.value;
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Subtract exactly, failing closed if operands or normalized result exceed budget.
    pub fn checked_subtract(
        &self,
        other: &Self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        budget.validate_value(&other.value)?;
        let value = &self.value - &other.value;
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Multiply exactly, failing closed if operands or normalized result exceed budget.
    pub fn checked_multiply(
        &self,
        other: &Self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        budget.validate_value(&other.value)?;
        let value = &self.value * &other.value;
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Divide exactly, rejecting a zero divisor and enforcing the resource budget.
    pub fn checked_divide(
        &self,
        other: &Self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        budget.validate_value(&other.value)?;
        if other.value.numer().is_zero() {
            return Err(ExactRationalError::DivisionByZero);
        }
        let value = &self.value / &other.value;
        budget.validate_value(&value)?;
        Ok(Self { value })
    }
}

impl fmt::Display for ExactRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.value.numer(), self.value.denom())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(bits: u64) -> RationalBudget {
        RationalBudget::new(bits).expect("test budget must be valid")
    }

    #[test]
    fn normalizes_sign_and_greatest_common_divisor() {
        let value = ExactRational::parse("-18", "-24", budget(32)).unwrap();
        assert_eq!(value.to_string(), "3/4");
        assert_eq!(value.to_wire().numerator, "3");
        assert_eq!(value.to_wire().denominator, "4");
    }

    #[test]
    fn keeps_integer_precision_beyond_i64() {
        let limit = RationalBudget::default();
        let huge = ExactRational::from_integer("18446744073709551616", limit).unwrap();
        let one = ExactRational::from_integer("1", limit).unwrap();
        let sum = huge.checked_add(&one, limit).unwrap();
        assert_eq!(sum.to_string(), "18446744073709551617/1");
    }

    #[test]
    fn cross_cancels_large_intermediates_without_losing_exactness() {
        let limit = budget(512);
        let large = "1606938044258990275541962092341162602522202993782792835301376";
        let left = ExactRational::parse(large, "3", limit).unwrap();
        let right = ExactRational::parse("3", large, limit).unwrap();
        let product = left.checked_multiply(&right, limit).unwrap();
        assert_eq!(product.to_string(), "1/1");
    }

    #[test]
    fn rejects_zero_denominators_and_division_by_zero() {
        let limit = budget(128);
        assert_eq!(
            ExactRational::parse("1", "0", limit).unwrap_err(),
            ExactRationalError::ZeroDenominator
        );
        let one = ExactRational::from_integer("1", limit).unwrap();
        let zero = ExactRational::from_integer("0", limit).unwrap();
        assert_eq!(
            one.checked_divide(&zero, limit).unwrap_err(),
            ExactRationalError::DivisionByZero
        );
    }

    #[test]
    fn bounds_decimal_parsing_before_big_integer_allocation() {
        let error = ExactRational::from_integer(&"9".repeat(100), budget(16)).unwrap_err();
        assert_eq!(error, ExactRationalError::ResourceLimitExceeded);
    }

    #[test]
    fn rejects_result_that_exceeds_configured_bit_budget() {
        let limit = budget(8);
        let value = ExactRational::from_integer("255", limit).unwrap();
        let one = ExactRational::from_integer("1", limit).unwrap();
        assert_eq!(
            value.checked_add(&one, limit).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
    }

    #[test]
    fn wire_round_trip_preserves_large_exact_decimal_strings() {
        let limit = RationalBudget::default();
        let value = ExactRational::parse("123456789012345678901234567890", "7", limit).unwrap();
        let encoded = serde_json::to_string(&value.to_wire()).unwrap();
        assert!(encoded.contains("\"123456789012345678901234567890\""));
        let wire: ExactRationalWire = serde_json::from_str(&encoded).unwrap();
        assert_eq!(ExactRational::from_wire(&wire, limit).unwrap(), value);
    }

    #[test]
    fn rejects_invalid_budgets() {
        assert_eq!(RationalBudget::new(0).unwrap_err(), ExactRationalError::InvalidBudget);
        assert_eq!(
            RationalBudget::new(HARD_MAX_COMPONENT_BITS + 1).unwrap_err(),
            ExactRationalError::InvalidBudget
        );
    }
    #[test]
    fn canonical_encoding_is_normalized_versioned_and_unambiguous() {
        let limit = budget(128);
        let half_a = ExactRational::parse("1", "2", limit).unwrap();
        let half_b = ExactRational::parse("-3", "-6", limit).unwrap();
        let two_thirds = ExactRational::parse("2", "3", limit).unwrap();

        assert_eq!(half_a.canonical_encoding_bytes(), half_b.canonical_encoding_bytes());
        assert_ne!(half_a.canonical_encoding_bytes(), two_thirds.canonical_encoding_bytes());
        assert!(half_a.canonical_encoding_bytes().starts_with(b"SYMTHAEA\0EXACT-RATIONAL\0V1\0"));
    }

    #[test]
    fn hdc_projection_is_deterministic_for_equivalent_rationals() {
        let limit = budget(128);
        let a = ExactRational::parse("1", "2", limit).unwrap();
        let b = ExactRational::parse("50", "100", limit).unwrap();
        assert_eq!(a.to_hdc(), b.to_hdc());
    }

    #[test]
    fn exact_numeric_tower_adapter_preserves_values_without_fallback() {
        let limit = budget(128);
        let natural = ExactRational::from_integer("18446744073709551615", limit).unwrap();
        assert_eq!(natural.try_to_number().unwrap(), Number::Natural(u64::MAX));

        let negative = ExactRational::from_integer("-9223372036854775808", limit).unwrap();
        assert_eq!(negative.try_to_number().unwrap(), Number::Integer(i64::MIN));

        let rational = ExactRational::parse("-7", "9", limit).unwrap();
        assert_eq!(
            rational.try_to_number().unwrap(),
            Number::Rational { numerator: -7, denominator: 9 }
        );
        assert_eq!(
            ExactRational::try_from_number_exact(&Number::Rational { numerator: -7, denominator: 9 }, limit).unwrap(),
            rational
        );
    }

    #[test]
    fn exact_numeric_tower_adapter_rejects_approximation_and_narrowing() {
        let limit = budget(256);
        assert_eq!(
            ExactRational::try_from_number_exact(&Number::Real(0.5), limit).unwrap_err(),
            ExactRationalError::ApproximateInput
        );

        let too_large = ExactRational::from_integer("18446744073709551616", limit).unwrap();
        assert_eq!(too_large.try_to_number().unwrap_err(), ExactRationalError::ExactResultOutOfRange);

        let too_wide_denominator = ExactRational::parse("1", "9223372036854775808", limit).unwrap();
        assert_eq!(
            too_wide_denominator.try_to_number().unwrap_err(),
            ExactRationalError::ExactResultOutOfRange
        );
    }

}
