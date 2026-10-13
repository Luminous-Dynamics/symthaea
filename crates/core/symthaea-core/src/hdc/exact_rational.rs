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
use num_traits::{Signed, ToPrimitive, Zero};
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

    fn max_decimal_digits(self) -> u64 {
        self.max_component_bits
            .saturating_mul(30_103)
            .checked_div(100_000)
            .unwrap_or(0)
            .saturating_add(2)
    }

    fn validate_decimal_component(self, text: &str) -> Result<(), ExactRationalError> {
        // Strip at most one ASCII sign. Stripping every leading sign would let
        // malformed inputs such as a very long "-----1" run evade the size
        // preflight before BigInt parsing eventually rejects them.
        let digits = match text.as_bytes().first() {
            Some(&b'+') | Some(&b'-') => &text[1..],
            _ => text,
        };

        // Bound work before validating/scanning every byte and before allocating
        // a BigInt. The conservative digit ceiling allows a small guard band;
        // exact normalized component bits are checked after parsing.
        if digits.len() as u64 > self.max_decimal_digits() {
            return Err(ExactRationalError::ResourceLimitExceeded);
        }
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ExactRationalError::InvalidInteger);
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
    /// Canonical byte encoding has a malformed header, length, component, or suffix.
    InvalidCanonicalEncoding,
    /// A wire numerator/denominator is valid but not in canonical normalized form.
    InvalidCanonicalWire,
    /// A rational denominator is zero.
    ZeroDenominator,
    /// A division operation used a zero rational divisor.
    DivisionByZero,
    /// An operation requiring an integer received a non-integral rational.
    NonIntegerInput,
    /// Parsing, an operand, or a result exceeded the configured resource budget.
    ResourceLimitExceeded,
    /// A legacy numeric-tower rational violates its documented canonical form.
    InvalidNumericTowerRational,
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
            Self::InvalidCanonicalEncoding => write!(f, "invalid canonical rational encoding"),
            Self::InvalidCanonicalWire => write!(f, "noncanonical exact-rational wire value"),
            Self::InvalidNumericTowerRational => write!(f, "numeric-tower rational is not in canonical form"),
            Self::ZeroDenominator => write!(f, "rational denominator cannot be zero"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::NonIntegerInput => write!(f, "operation requires integer inputs"),
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

    /// Rehydrate a canonical wire value with a caller-supplied budget.
    ///
    /// The input strings must already be normalized decimal components;
    /// unlike the parse constructor, this transport boundary rejects alternate spellings.
    pub fn from_wire(
        wire: &ExactRationalWire,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        let value = Self::parse(&wire.numerator, &wire.denominator, budget)?;
        if value.numerator_str() != wire.numerator
            || value.denominator_str() != wire.denominator
        {
            return Err(ExactRationalError::InvalidCanonicalWire);
        }
        Ok(value)
    }

    /// Decode canonical bytes while enforcing resource limits before parsing.
    ///
    /// Non-canonical encodings (including leading zeroes, plus signs, unreduced
    /// fractions, invalid lengths, and trailing bytes) are rejected. This keeps
    /// byte identity aligned with normalized rational identity.
    pub fn from_canonical_encoding_bytes(
        input: &[u8],
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        const PREFIX: &[u8] = b"SYMTHAEA\0EXACT-RATIONAL\0V1\0";
        let payload = input
            .strip_prefix(PREFIX)
            .ok_or(ExactRationalError::InvalidCanonicalEncoding)?;
        let mut cursor = 0usize;

        fn read_component<'a>(
            payload: &'a [u8],
            cursor: &mut usize,
            budget: RationalBudget,
        ) -> Result<&'a str, ExactRationalError> {
            let length_end = (*cursor)
                .checked_add(8)
                .ok_or(ExactRationalError::InvalidCanonicalEncoding)?;
            let length_bytes = payload
                .get(*cursor..length_end)
                .ok_or(ExactRationalError::InvalidCanonicalEncoding)?;
            let encoded_length = u64::from_be_bytes(
                length_bytes
                    .try_into()
                    .map_err(|_| ExactRationalError::InvalidCanonicalEncoding)?,
            );
            *cursor = length_end;

            // Reject implausible lengths before UTF-8 validation or integer parsing.
            let max_chars = budget.max_decimal_digits().saturating_add(1);
            if encoded_length == 0 {
                return Err(ExactRationalError::InvalidCanonicalEncoding);
            }
            if encoded_length > max_chars {
                return Err(ExactRationalError::ResourceLimitExceeded);
            }
            let length = usize::try_from(encoded_length)
                .map_err(|_| ExactRationalError::ResourceLimitExceeded)?;
            let component_end = (*cursor)
                .checked_add(length)
                .ok_or(ExactRationalError::InvalidCanonicalEncoding)?;
            let component_bytes = payload
                .get(*cursor..component_end)
                .ok_or(ExactRationalError::InvalidCanonicalEncoding)?;
            *cursor = component_end;
            let component = std::str::from_utf8(component_bytes)
                .map_err(|_| ExactRationalError::InvalidCanonicalEncoding)?;
            budget
                .validate_decimal_component(component)
                .map_err(|error| match error {
                    ExactRationalError::ResourceLimitExceeded => error,
                    _ => ExactRationalError::InvalidCanonicalEncoding,
                })?;
            Ok(component)
        }

        let numerator = read_component(payload, &mut cursor, budget)?;
        let denominator = read_component(payload, &mut cursor, budget)?;
        if cursor != payload.len() {
            return Err(ExactRationalError::InvalidCanonicalEncoding);
        }

        let value = Self::parse(numerator, denominator, budget).map_err(|error| match error {
            ExactRationalError::ResourceLimitExceeded => error,
            _ => ExactRationalError::InvalidCanonicalEncoding,
        })?;
        if value.canonical_encoding_bytes().as_slice() != input {
            return Err(ExactRationalError::InvalidCanonicalEncoding);
        }
        Ok(value)
    }

    /// Export a normalized, deterministic wire value using the hard implementation cap.
    ///
    /// Call checked_to_wire when enforcing a stricter caller-selected budget.
    pub fn to_wire(&self) -> ExactRationalWire {
        ExactRationalWire {
            numerator: self.value.numer().to_str_radix(10),
            denominator: self.value.denom().to_str_radix(10),
        }
    }

    /// Export only if both normalized components fit the caller's bit budget.
    pub fn checked_to_wire(
        &self,
        budget: RationalBudget,
    ) -> Result<ExactRationalWire, ExactRationalError> {
        budget.validate_value(&self.value)?;
        Ok(self.to_wire())
    }

    /// Lossless, versioned, deterministic bytes for exact identity and transport.
    ///
    /// Unlike an HDC projection or a fixed-width seed, these bytes are an
    /// unambiguous representation: normalized decimal numerator and denominator
    /// components are length-prefixed and preceded by a format/version tag.
    /// This convenience form is bounded by HARD_MAX_COMPONENT_BITS. Use
    /// checked_canonical_encoding_bytes to enforce a stricter budget.
    pub fn canonical_encoding_bytes(&self) -> Vec<u8> {
        self.encode_canonical_bytes()
    }

    /// Encode the exact value only if it fits the caller's bit budget.
    pub fn checked_canonical_encoding_bytes(
        &self,
        budget: RationalBudget,
    ) -> Result<Vec<u8>, ExactRationalError> {
        budget.validate_value(&self.value)?;
        Ok(self.encode_canonical_bytes())
    }

    fn encode_canonical_bytes(&self) -> Vec<u8> {
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

    /// Deterministic categorical HDC projection of this canonical rational value.
    ///
    /// This hash-derived hypervector can serve as a stable symbol-like code for
    /// binding and lookup. It intentionally does not preserve numeric order,
    /// distance, or neighborhood similarity, and is not a lossless identifier
    /// or proof object. Use canonical_encoding_bytes when exact identity matters.
    /// This convenience form is bounded by HARD_MAX_COMPONENT_BITS.
    pub fn to_hdc(&self) -> BinaryHV {
        self.project_canonical_bytes(&self.encode_canonical_bytes())
    }

    /// Create an HDC projection only if the rational fits the caller's bit budget.
    pub fn checked_to_hdc(
        &self,
        budget: RationalBudget,
    ) -> Result<BinaryHV, ExactRationalError> {
        let canonical = self.checked_canonical_encoding_bytes(budget)?;
        Ok(self.project_canonical_bytes(&canonical))
    }

    fn project_canonical_bytes(&self, canonical: &[u8]) -> BinaryHV {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"SYMTHAEA.EXACT_RATIONAL.HDC.V1\0");
        hasher.update(canonical);
        let mut output = [0u8; BinaryHV::BYTES];
        hasher.finalize_xof().fill(&mut output);
        BinaryHV(output)
    }

    /// Convert into the legacy numeric tower only when the exact value fits.
    /// No approximation or overflow fallback is allowed on this path.
    pub fn try_to_number(&self) -> Result<Number, ExactRationalError> {
        self.try_to_number_with_budget(RationalBudget::new(HARD_MAX_COMPONENT_BITS)?)
    }

    /// Convert only if the exact value fits the caller-selected budget and
    /// the target fixed-width Number variant can represent it without loss.
    pub fn try_to_number_with_budget(
        &self,
        budget: RationalBudget,
    ) -> Result<Number, ExactRationalError> {
        budget.validate_value(&self.value)?;
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
                if *denominator == 0 {
                    return Err(ExactRationalError::ZeroDenominator);
                }
                let exact = Self::parse(
                    &numerator.to_string(),
                    &denominator.to_string(),
                    budget,
                )?;
                // Number::Rational promises a reduced numerator and positive
                // denominator. Reject malformed variants rather than silently
                // normalizing them across a strict provenance boundary.
                if *denominator < 0
                    || exact.numerator_str() != numerator.to_string()
                    || exact.denominator_str() != denominator.to_string()
                {
                    return Err(ExactRationalError::InvalidNumericTowerRational);
                }
                Ok(exact)
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

    /// Convert to a finite f64 approximation when it does not overflow or
    /// collapse a nonzero value all the way to zero.
    ///
    /// Subnormal nonzero approximations are retained. A nonzero exact rational
    /// that underflows to +/-0.0 returns None so callers can distinguish total
    /// underflow from a usable finite approximation. This convenience form is
    /// bounded by HARD_MAX_COMPONENT_BITS.
    pub fn to_f64_approx(&self) -> Option<f64> {
        self.value.to_f64().filter(|value| {
            value.is_finite() && (*value != 0.0 || self.value.numer().is_zero())
        })
    }

    /// Approximate conversion, only after validating the caller's resource budget.
    pub fn checked_to_f64_approx(
        &self,
        budget: RationalBudget,
    ) -> Result<Option<f64>, ExactRationalError> {
        budget.validate_value(&self.value)?;
        Ok(self.to_f64_approx())
    }

    /// True when the exact numerator is zero.
    pub fn is_zero(&self) -> bool {
        self.value.numer().is_zero()
    }

    /// True when the exact rational is strictly positive.
    pub fn is_positive(&self) -> bool {
        self.value.numer().is_positive()
    }

    /// True when the exact rational is strictly negative.
    pub fn is_negative(&self) -> bool {
        self.value.numer().is_negative()
    }

    /// Compare exact values without narrowing to floating point.
    /// Both operands must fit the caller-selected resource budget.
    pub fn checked_compare(
        &self,
        other: &Self,
        budget: RationalBudget,
    ) -> Result<std::cmp::Ordering, ExactRationalError> {
        budget.validate_value(&self.value)?;
        budget.validate_value(&other.value)?;
        Ok(self.value.cmp(&other.value))
    }

    /// Negate exactly, enforcing the caller-selected resource budget.
    pub fn checked_negate(
        &self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        let value = -self.value.clone();
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Return the exact absolute value under the caller-selected resource budget.
    pub fn checked_abs(
        &self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        let value = self.value.clone().abs();
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Whether this normalized rational is an integer.
    pub fn is_integer(&self) -> bool {
        self.value.denom() == &BigInt::from(1u8)
    }

    /// Exact mathematical floor, including for negative non-integers.
    /// The input and result must fit the caller-selected resource budget.
    pub fn checked_floor(
        &self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        let numerator = self.value.numer();
        let denominator = self.value.denom(); // BigRational always normalizes this positive.
        let quotient = numerator / denominator; // BigInt division truncates toward zero.
        let remainder = numerator % denominator;
        let floor = if numerator.is_negative() && !remainder.is_zero() {
            quotient - BigInt::from(1u8)
        } else {
            quotient
        };
        let value = BigRational::from_integer(floor);
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Exact mathematical ceiling, including for negative non-integers.
    /// The input and result must fit the caller-selected resource budget.
    pub fn checked_ceil(
        &self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        let numerator = self.value.numer();
        let denominator = self.value.denom();
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;
        let ceil = if numerator.is_positive() && !remainder.is_zero() {
            quotient + BigInt::from(1u8)
        } else {
            quotient
        };
        let value = BigRational::from_integer(ceil);
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Exact fractional part in [0, 1), including for negative inputs.
    /// For example, the fractional part of -1/3 is 2/3, not -1/3.
    pub fn checked_fractional_part(
        &self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        let floor = self.checked_floor(budget)?;
        self.checked_subtract(&floor, budget)
    }

    /// Greatest common divisor of two exact integers; result is non-negative.
    /// Non-integer rationals are rejected, and inputs/results obey the caller budget.
    pub fn checked_gcd(
        &self,
        other: &Self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        budget.validate_value(&other.value)?;
        if !self.is_integer() || !other.is_integer() {
            return Err(ExactRationalError::NonIntegerInput);
        }

        let mut a = self.value.numer().abs();
        let mut b = other.value.numer().abs();
        while !b.is_zero() {
            let remainder = &a % &b;
            a = b;
            b = remainder;
        }
        let value = BigRational::from_integer(a);
        budget.validate_value(&value)?;
        Ok(Self { value })
    }

    /// Least common multiple of two exact integers; result is non-negative.
    /// Zero has lcm zero with every integer. Non-integer rationals are rejected.
    pub fn checked_lcm(
        &self,
        other: &Self,
        budget: RationalBudget,
    ) -> Result<Self, ExactRationalError> {
        budget.validate_value(&self.value)?;
        budget.validate_value(&other.value)?;
        if !self.is_integer() || !other.is_integer() {
            return Err(ExactRationalError::NonIntegerInput);
        }

        let a = self.value.numer().abs();
        let b = other.value.numer().abs();
        if a.is_zero() || b.is_zero() {
            return Ok(Self { value: BigRational::from_integer(BigInt::from(0u8)) });
        }

        let gcd = self.checked_gcd(other, budget)?.value;
        let gcd_integer = gcd.numer();
        // Divide before multiplying to keep the intermediate no larger than
        // necessary; final normalized value must still fit the configured budget.
        let lcm = (a / gcd_integer) * b;
        let value = BigRational::from_integer(lcm);
        budget.validate_value(&value)?;
        Ok(Self { value })
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
    fn repeated_signs_cannot_bypass_decimal_length_preflight() {
        let malformed = format!("{}1", "-".repeat(1_000));
        assert_eq!(
            ExactRational::from_integer(&malformed, budget(16)).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );

        assert_eq!(
            ExactRational::from_integer("--1", budget(16)).unwrap_err(),
            ExactRationalError::InvalidInteger
        );
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
    fn wire_decoder_rejects_noncanonical_decimal_strings() {
        let limit = budget(128);
        for wire in [
            ExactRationalWire { numerator: "01".into(), denominator: "1".into() },
            ExactRationalWire { numerator: "2".into(), denominator: "4".into() },
            ExactRationalWire { numerator: "+1".into(), denominator: "2".into() },
            ExactRationalWire { numerator: "1".into(), denominator: "-2".into() },
        ] {
            assert_eq!(
                ExactRational::from_wire(&wire, limit).unwrap_err(),
                ExactRationalError::InvalidCanonicalWire
            );
        }
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
    fn canonical_bytes_round_trip_and_reject_noncanonical_or_malformed_input() {
        let limit = budget(256);
        let value = ExactRational::parse("-12345678901234567890", "77", limit).unwrap();
        let bytes = value.canonical_encoding_bytes();
        assert_eq!(ExactRational::from_canonical_encoding_bytes(&bytes, limit).unwrap(), value);

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            ExactRational::from_canonical_encoding_bytes(&trailing, limit).unwrap_err(),
            ExactRationalError::InvalidCanonicalEncoding
        );
        assert_eq!(
            ExactRational::from_canonical_encoding_bytes(b"not-an-exact-rational", limit).unwrap_err(),
            ExactRationalError::InvalidCanonicalEncoding
        );

        let leading_zero = {
            const PREFIX: &[u8] = b"SYMTHAEA\0EXACT-RATIONAL\0V1\0";
            let mut encoded = PREFIX.to_vec();
            encoded.extend_from_slice(&2u64.to_be_bytes());
            encoded.extend_from_slice(b"01");
            encoded.extend_from_slice(&1u64.to_be_bytes());
            encoded.extend_from_slice(b"1");
            encoded
        };
        assert_eq!(
            ExactRational::from_canonical_encoding_bytes(&leading_zero, limit).unwrap_err(),
            ExactRationalError::InvalidCanonicalEncoding
        );

        let non_reduced = {
            const PREFIX: &[u8] = b"SYMTHAEA\0EXACT-RATIONAL\0V1\0";
            let mut encoded = PREFIX.to_vec();
            encoded.extend_from_slice(&1u64.to_be_bytes());
            encoded.extend_from_slice(b"2");
            encoded.extend_from_slice(&1u64.to_be_bytes());
            encoded.extend_from_slice(b"4");
            encoded
        };
        assert_eq!(
            ExactRational::from_canonical_encoding_bytes(&non_reduced, limit).unwrap_err(),
            ExactRationalError::InvalidCanonicalEncoding
        );
    }

    #[test]
    fn canonical_encoding_round_trips_small_rational_grid() {
        let limit = budget(128);
        for numerator in -12..=12 {
            for denominator in 1..=9 {
                let value = ExactRational::parse(
                    &numerator.to_string(),
                    &denominator.to_string(),
                    limit,
                ).unwrap();
                let bytes = value.canonical_encoding_bytes();
                let decoded = ExactRational::from_canonical_encoding_bytes(&bytes, limit).unwrap();
                assert_eq!(&decoded, &value);
                assert_eq!(decoded.canonical_encoding_bytes(), bytes);
            }
        }
    }

    #[test]
    fn canonical_decoder_rejects_absurd_or_truncated_component_lengths() {
        let limit = budget(32);
        let mut absurd = b"SYMTHAEA\0EXACT-RATIONAL\0V1\0".to_vec();
        absurd.extend_from_slice(&u64::MAX.to_be_bytes());
        assert_eq!(
            ExactRational::from_canonical_encoding_bytes(&absurd, limit).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );

        let mut truncated = b"SYMTHAEA\0EXACT-RATIONAL\0V1\0".to_vec();
        truncated.extend_from_slice(&3u64.to_be_bytes());
        truncated.extend_from_slice(b"12");
        assert_eq!(
            ExactRational::from_canonical_encoding_bytes(&truncated, limit).unwrap_err(),
            ExactRationalError::InvalidCanonicalEncoding
        );
    }

    #[test]
    fn canonical_decoder_enforces_bit_budget_before_component_parsing() {
        let wide = budget(32);
        let narrow = budget(8);
        let value = ExactRational::from_integer("65535", wide).unwrap();
        let bytes = value.canonical_encoding_bytes();
        assert_eq!(
            ExactRational::from_canonical_encoding_bytes(&bytes, narrow).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
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
    fn numeric_tower_import_round_trips_fixed_width_integer_edges() {
        let limit = budget(128);
        let values = [
            Number::Natural(0),
            Number::Natural(u64::MAX),
            Number::Integer(i64::MIN),
            Number::Integer(-1),
            Number::Rational { numerator: i64::MIN, denominator: i64::MAX },
            Number::Rational { numerator: 7, denominator: 9 },
        ];

        for original in values {
            let exact = ExactRational::try_from_number_exact(&original, limit).unwrap();
            let narrowed = exact.try_to_number().unwrap();
            // Natural / Integer values preserve their exact variant where the
            // tower representation is canonical. Rational integer-like values
            // intentionally narrow to the narrower integer domain.
            match original {
                Number::Rational { numerator: 7, denominator: 9 } => {
                    assert_eq!(narrowed, Number::Rational { numerator: 7, denominator: 9 });
                }
                Number::Rational { numerator: i64::MIN, denominator: i64::MAX } => {
                    assert_eq!(narrowed, Number::Rational { numerator: i64::MIN, denominator: i64::MAX });
                }
                other => assert_eq!(narrowed, other),
            }
        }
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

    #[test]
    fn large_balanced_rational_has_finite_f64_approximation() {
        let limit = RationalBudget::default();
        let numerator = format!("1{}1", "0".repeat(899));
        let denominator = "9".repeat(900);
        let value = ExactRational::parse(&numerator, &denominator, limit).unwrap();
        let approx = value.to_f64_approx().expect("ratio near one should remain finite");
        assert!(approx.is_finite());
        assert!((approx - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn float_conversion_retains_smallest_positive_subnormal() {
        let limit = RationalBudget::default();
        let denominator = (BigInt::from(1u8) << 1074usize).to_str_radix(10);
        let smallest = ExactRational::parse("1", &denominator, limit).unwrap();
        assert_eq!(smallest.to_f64_approx(), Some(f64::from_bits(1)));
    }

    #[test]
    fn float_conversion_reports_overflow_and_total_underflow() {
        let limit = RationalBudget::default();
        let huge = format!("1{}", "0".repeat(1_100));
        let too_large = ExactRational::from_integer(&huge, limit).unwrap();
        assert_eq!(too_large.to_f64_approx(), None);

        let tiny = ExactRational::parse("1", &huge, limit).unwrap();
        assert_eq!(tiny.to_f64_approx(), None);

        let zero = ExactRational::from_integer("0", limit).unwrap();
        assert_eq!(zero.to_f64_approx(), Some(0.0));
    }

    #[test]
    fn exact_numeric_import_rejects_noncanonical_rational_variants() {
        let limit = budget(64);
        for malformed in [
            Number::Rational { numerator: 2, denominator: 4 },
            Number::Rational { numerator: 0, denominator: 5 },
            Number::Rational { numerator: 1, denominator: -2 },
        ] {
            assert_eq!(
                ExactRational::try_from_number_exact(&malformed, limit).unwrap_err(),
                ExactRationalError::InvalidNumericTowerRational
            );
        }
    }

    #[test]
    fn exact_numeric_import_rejects_malformed_rational_and_real_nan() {
        let limit = budget(64);
        assert_eq!(
            ExactRational::try_from_number_exact(
                &Number::Rational { numerator: 1, denominator: 0 },
                limit,
            ).unwrap_err(),
            ExactRationalError::ZeroDenominator
        );
        assert_eq!(
            ExactRational::try_from_number_exact(&Number::Real(f64::NAN), limit).unwrap_err(),
            ExactRationalError::ApproximateInput
        );
    }

    #[test]
    fn export_and_conversion_boundaries_enforce_caller_budget() {
        let wide_budget = budget(32);
        let narrow_budget = budget(8);
        let value = ExactRational::from_integer("65535", wide_budget).unwrap();

        assert_eq!(
            value.checked_to_wire(narrow_budget).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
        assert_eq!(
            value.checked_canonical_encoding_bytes(narrow_budget).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
        assert_eq!(
            value.checked_to_hdc(narrow_budget).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
        assert_eq!(
            value.try_to_number_with_budget(narrow_budget).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
        assert_eq!(
            value.checked_to_f64_approx(narrow_budget).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
    }

    #[test]
    fn exact_comparison_distinguishes_values_that_round_to_the_same_f64() {
        use std::cmp::Ordering;

        let limit = budget(512);
        let one = ExactRational::from_integer("1", limit).unwrap();
        let denominator = format!("1{}", "0".repeat(100));
        let epsilon = ExactRational::parse("1", &denominator, limit).unwrap();
        let one_plus_epsilon = one.checked_add(&epsilon, limit).unwrap();

        // Both values collapse to 1.0 in binary64, but exact ordering must not.
        assert_eq!(one.to_f64_approx(), Some(1.0));
        assert_eq!(one_plus_epsilon.to_f64_approx(), Some(1.0));
        assert_eq!(one.checked_compare(&one_plus_epsilon, limit).unwrap(), Ordering::Less);
        assert_eq!(one_plus_epsilon.checked_compare(&one, limit).unwrap(), Ordering::Greater);
        assert_eq!(one.checked_compare(&one.clone(), limit).unwrap(), Ordering::Equal);
    }

    #[test]
    fn exact_sign_negation_and_absolute_value_preserve_canonical_form() {
        let limit = budget(128);
        let negative = ExactRational::parse("-18", "24", limit).unwrap();
        assert!(negative.is_negative());
        assert!(!negative.is_positive());
        assert!(!negative.is_zero());

        let positive = negative.checked_abs(limit).unwrap();
        assert_eq!(positive.to_string(), "3/4");
        assert!(positive.is_positive());
        assert_eq!(negative.checked_negate(limit).unwrap(), positive);

        let zero = ExactRational::from_integer("0", limit).unwrap();
        assert!(zero.is_zero());
        assert!(!zero.is_positive());
        assert!(!zero.is_negative());
        assert_eq!(zero.checked_negate(limit).unwrap(), zero);
        assert_eq!(zero.checked_abs(limit).unwrap(), zero);
    }

    #[test]
    fn exact_comparison_and_unary_operations_enforce_caller_budget() {
        let wide = budget(32);
        let narrow = budget(8);
        let value = ExactRational::from_integer("65535", wide).unwrap();
        let one = ExactRational::from_integer("1", wide).unwrap();

        assert_eq!(
            value.checked_compare(&one, narrow).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
        assert_eq!(
            value.checked_negate(narrow).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
        assert_eq!(
            value.checked_abs(narrow).unwrap_err(),
            ExactRationalError::ResourceLimitExceeded
        );
    }

    #[test]
    fn exact_floor_ceil_and_fractional_part_cover_signed_values() {
        let limit = budget(128);
        let cases = [
            ("-7", "3", "-3", "-2", "2/3"),
            ("-1", "3", "-1", "0", "2/3"),
            ("-2", "1", "-2", "-2", "0/1"),
            ("0", "1", "0", "0", "0/1"),
            ("1", "3", "0", "1", "1/3"),
            ("7", "3", "2", "3", "1/3"),
        ];
        for (numerator, denominator, expected_floor, expected_ceil, expected_fraction) in cases {
            let value = ExactRational::parse(numerator, denominator, limit).unwrap();
            let floor = value.checked_floor(limit).unwrap();
            let ceil = value.checked_ceil(limit).unwrap();
            let fractional = value.checked_fractional_part(limit).unwrap();
            assert_eq!(floor.to_string(), format!("{expected_floor}/1"), "floor({value})");
            assert_eq!(ceil.to_string(), format!("{expected_ceil}/1"), "ceil({value})");
            assert_eq!(fractional.to_string(), expected_fraction, "fractional_part({value})");
            assert!(fractional.checked_compare(&ExactRational::from_integer("0", limit).unwrap(), limit).unwrap() != std::cmp::Ordering::Less);
            assert!(fractional.checked_compare(&ExactRational::from_integer("1", limit).unwrap(), limit).unwrap() == std::cmp::Ordering::Less);
        }
    }

    #[test]
    fn exact_integer_classification_and_floor_ceil_budget_are_explicit() {
        let wide = budget(32);
        let narrow = budget(8);
        let integer = ExactRational::parse("-12", "3", wide).unwrap();
        let fractional = ExactRational::parse("7", "3", wide).unwrap();
        assert!(integer.is_integer());
        assert!(!fractional.is_integer());
        assert!(ExactRational::from_integer("0", wide).unwrap().is_integer());

        let oversized = ExactRational::from_integer("65535", wide).unwrap();
        assert_eq!(oversized.checked_floor(narrow).unwrap_err(), ExactRationalError::ResourceLimitExceeded);
        assert_eq!(oversized.checked_ceil(narrow).unwrap_err(), ExactRationalError::ResourceLimitExceeded);
        assert_eq!(oversized.checked_fractional_part(narrow).unwrap_err(), ExactRationalError::ResourceLimitExceeded);
    }

    #[test]
    fn floor_and_ceil_obey_exact_ordering_on_a_signed_rational_grid() {
        let limit = budget(128);
        let one = ExactRational::from_integer("1", limit).unwrap();
        for numerator in -15..=15 {
            for denominator in 1..=7 {
                let value = ExactRational::parse(&numerator.to_string(), &denominator.to_string(), limit).unwrap();
                let floor = value.checked_floor(limit).unwrap();
                let ceil = value.checked_ceil(limit).unwrap();
                assert_ne!(floor.checked_compare(&value, limit).unwrap(), std::cmp::Ordering::Greater);
                assert_ne!(ceil.checked_compare(&value, limit).unwrap(), std::cmp::Ordering::Less);
                let floor_plus_one = floor.checked_add(&one, limit).unwrap();
                assert_eq!(floor_plus_one.checked_compare(&value, limit).unwrap(), std::cmp::Ordering::Greater);
                let ceil_minus_one = ceil.checked_subtract(&one, limit).unwrap();
                assert_eq!(ceil_minus_one.checked_compare(&value, limit).unwrap(), std::cmp::Ordering::Less);
            }
        }
    }

    #[test]
    fn exact_gcd_and_lcm_handle_signs_zero_and_large_integers() {
        let limit = budget(256);
        let from = |n: &str| ExactRational::from_integer(n, limit).unwrap();

        assert_eq!(from("-54").checked_gcd(&from("24"), limit).unwrap().to_string(), "6/1");
        assert_eq!(from("0").checked_gcd(&from("0"), limit).unwrap().to_string(), "0/1");
        assert_eq!(from("-21").checked_lcm(&from("6"), limit).unwrap().to_string(), "42/1");
        assert_eq!(from("-21").checked_lcm(&from("0"), limit).unwrap().to_string(), "0/1");

        let large = from("1606938044258990275541962092341162602522202993782792835301376"); // 2^200
        let multiple = from("4820814127976970826625886277023487807566608981348378505904128"); // 3 * 2^200
        assert_eq!(large.checked_gcd(&multiple, limit).unwrap(), large);
        assert_eq!(large.checked_lcm(&multiple, limit).unwrap(), multiple);
    }

    #[test]
    fn exact_gcd_and_lcm_reject_nonintegers_and_enforce_result_budget() {
        let wide = budget(32);
        let narrow = budget(8);
        let half = ExactRational::parse("1", "2", wide).unwrap();
        let one = ExactRational::from_integer("1", wide).unwrap();
        assert_eq!(half.checked_gcd(&one, wide).unwrap_err(), ExactRationalError::NonIntegerInput);
        assert_eq!(half.checked_lcm(&one, wide).unwrap_err(), ExactRationalError::NonIntegerInput);

        let a = ExactRational::from_integer("255", wide).unwrap();
        let b = ExactRational::from_integer("254", wide).unwrap();
        assert_eq!(a.checked_lcm(&b, narrow).unwrap_err(), ExactRationalError::ResourceLimitExceeded);
        assert_eq!(a.checked_gcd(&b, narrow).unwrap().to_string(), "1/1");

        let oversized = ExactRational::from_integer("256", wide).unwrap();
        assert_eq!(oversized.checked_gcd(&one, narrow).unwrap_err(), ExactRationalError::ResourceLimitExceeded);
    }

    #[test]
    fn gcd_lcm_identity_holds_for_a_small_signed_integer_grid() {
        let limit = budget(128);
        for a in -12..=12 {
            for b in -12..=12 {
                let lhs = ExactRational::from_integer(&a.to_string(), limit).unwrap();
                let rhs = ExactRational::from_integer(&b.to_string(), limit).unwrap();
                let gcd = lhs.checked_gcd(&rhs, limit).unwrap();
                let lcm = lhs.checked_lcm(&rhs, limit).unwrap();
                assert!(!gcd.is_negative());
                assert!(!lcm.is_negative());
                if a != 0 && b != 0 {
                    let product = lhs.checked_abs(limit).unwrap()
                        .checked_multiply(&rhs.checked_abs(limit).unwrap(), limit).unwrap();
                    assert_eq!(gcd.checked_multiply(&lcm, limit).unwrap(), product);
                } else {
                    assert!(lcm.is_zero());
                }
            }
        }
    }

    #[test]
    fn small_rational_grid_obeys_arithmetic_round_trip_identities() {
        let limit = budget(128);
        let mut values = Vec::new();
        for numerator in -3..=3 {
            for denominator in 1..=4 {
                values.push(
                    ExactRational::parse(
                        &numerator.to_string(),
                        &denominator.to_string(),
                        limit,
                    ).unwrap(),
                );
            }
        }

        for lhs in &values {
            for rhs in &values {
                let sum = lhs.checked_add(rhs, limit).unwrap();
                let recovered_lhs = sum.checked_subtract(rhs, limit).unwrap();
                assert_eq!(&recovered_lhs, lhs, "addition/subtraction identity");

                if rhs.numerator_str() != "0" {
                    let product = lhs.checked_multiply(rhs, limit).unwrap();
                    let recovered_lhs = product.checked_divide(rhs, limit).unwrap();
                    assert_eq!(&recovered_lhs, lhs, "multiplication/division identity");
                }
            }
        }
    }

    #[test]
    fn distinct_canonical_values_produce_distinct_expected_test_vectors() {
        let limit = budget(128);
        let half = ExactRational::parse("1", "2", limit).unwrap();
        let two_thirds = ExactRational::parse("2", "3", limit).unwrap();
        assert_ne!(half.to_hdc(), two_thirds.to_hdc());
    }

}
