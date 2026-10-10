// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Numeric Tower for HDC Consciousness Framework
//!
//! Implements a unified numeric tower with automatic promotion between
//! number domains, following the mathematical hierarchy:
//!
//!   ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ
//!
//! ## Auto-Promotion Rules
//!
//! - **ℕ → ℤ**: When subtraction produces a negative result
//! - **ℤ → ℚ**: When division is not exact (has remainder)
//! - **ℚ → ℝ**: For irrational results (e.g., square roots of non-perfect-squares)
//!
//! ## HDC Encoding
//!
//! Each number is encoded as a `BinaryHV` in 16,384-dimensional hyperdimensional
//! space using the `PrimitiveSystem`. The encoding preserves the domain information
//! through binding with domain-specific primitives.
//!
//! ## Consciousness Measure (Φ)
//!
//! Every operation tracks Φ (integrated information), which increases with:
//! - Domain promotions (crossing a mathematical boundary adds integration)
//! - Operation complexity (division > multiplication > addition)
//! - Proof trace depth (more reasoning steps = more consciousness)

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, seed_from_name};
use serde::{Deserialize, Serialize};

// ============================================================================
// HELPER: GCD
// ============================================================================

/// Compute greatest common divisor using Euclidean algorithm.
#[cfg(test)]
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Normalize a rational number for legacy unit tests.
#[cfg(test)]
fn normalize_rational(num: i64, den: i64) -> (i64, i64) {
    assert!(den != 0, "Denominator cannot be zero");

    // Ensure denominator is positive
    let (num, den) = if den < 0 {
        (num.wrapping_neg(), den.wrapping_neg())
    } else {
        (num, den)
    };

    // Handle zero numerator
    if num == 0 {
        return (0, 1);
    }

    // Reduce to lowest terms
    let g = gcd(num.unsigned_abs(), den as u64) as i64;
    (num / g, den / g)
}

/// Compute a greatest common divisor without narrowing to a machine word.
fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Normalize an exact fraction in a wider intermediate type.
///
/// Returns None for a zero denominator or for a sign normalization that cannot
/// be represented by i128. The reduced denominator is always positive.
fn normalize_fraction_i128(numerator: i128, denominator: i128) -> Option<(i128, i128)> {
    if denominator == 0 {
        return None;
    }

    let (numerator, denominator) = if denominator < 0 {
        (numerator.checked_neg()?, denominator.checked_neg()?)
    } else {
        (numerator, denominator)
    };

    if numerator == 0 {
        return Some((0, 1));
    }

    let divisor = i128::try_from(gcd_u128(numerator.unsigned_abs(), denominator as u128)).ok()?;
    Some((numerator / divisor, denominator / divisor))
}

// ============================================================================
// CORE TYPES
// ============================================================================

/// The domain (number system) a value belongs to.
///
/// Ordered by inclusion: ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NumberDomain {
    /// Natural numbers (ℕ): 0, 1, 2, 3, ...
    Natural,
    /// Integers (ℤ): ..., -2, -1, 0, 1, 2, ...
    Integer,
    /// Rational numbers (ℚ): p/q where p, q ∈ ℤ, q ≠ 0
    Rational,
    /// Real numbers (ℝ): the complete ordered field
    Real,
}

impl std::fmt::Display for NumberDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumberDomain::Natural => write!(f, "\u{2115}"),
            NumberDomain::Integer => write!(f, "\u{2124}"),
            NumberDomain::Rational => write!(f, "\u{211a}"),
            NumberDomain::Real => write!(f, "\u{211d}"),
        }
    }
}

/// A number in the unified numeric tower.
///
/// Automatically selects the most specific (narrowest) domain that can
/// represent the value exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Number {
    /// A natural number (non-negative integer): n >= 0
    Natural(u64),
    /// An integer (possibly negative)
    Integer(i64),
    /// A rational number p/q in lowest terms with q > 0
    Rational { numerator: i64, denominator: i64 },
    /// A real number (floating-point approximation)
    Real(f64),
}

impl Number {
    /// Which domain does this number live in?
    pub fn domain(&self) -> NumberDomain {
        match self {
            Number::Natural(_) => NumberDomain::Natural,
            Number::Integer(_) => NumberDomain::Integer,
            Number::Rational { .. } => NumberDomain::Rational,
            Number::Real(_) => NumberDomain::Real,
        }
    }

    /// Convert to f64 for approximate computation.
    pub fn to_f64(&self) -> f64 {
        match self {
            Number::Natural(n) => *n as f64,
            Number::Integer(n) => *n as f64,
            Number::Rational {
                numerator,
                denominator,
            } => *numerator as f64 / *denominator as f64,
            Number::Real(x) => *x,
        }
    }

    /// Check whether this number is exactly zero in its represented domain.
    ///
    /// Unlike Number::is_zero, this does not treat small nonzero floating-point
    /// values as zero. NaN and infinities are never classified as exact zero.
    /// Use this predicate for algebraic guards such as division-by-zero checks.
    pub fn is_zero_exact(&self) -> bool {
        match self {
            Number::Natural(n) => *n == 0,
            Number::Integer(n) => *n == 0,
            Number::Rational {
                numerator,
                denominator,
            } => *denominator != 0 && *numerator == 0,
            Number::Real(x) => *x == 0.0,
        }
    }

    /// Check whether this number is within an absolute tolerance of zero.
    ///
    /// Returns false for negative or non-finite tolerances and for non-finite
    /// values. This is an approximate numerical predicate, not an algebraic one.
    pub fn is_approximately_zero(&self, tolerance: f64) -> bool {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }

        let value = self.to_f64();
        value.is_finite() && value.abs() <= tolerance
    }

    /// Check if this number is zero in any domain.
    ///
    /// Compatibility note: floating-point values with magnitude below 1e-15
    /// are treated as zero. Prefer is_zero_exact for algebraic guards and
    /// is_approximately_zero when the tolerance matters.
    pub fn is_zero(&self) -> bool {
        match self {
            Number::Natural(n) => *n == 0,
            Number::Integer(n) => *n == 0,
            Number::Rational { numerator, .. } => *numerator == 0,
            Number::Real(x) => x.abs() < 1e-15,
        }
    }

    /// Try to narrow this number to the most specific domain.
    ///
    /// For example, Rational { 6, 1 } narrows to Integer(6),
    /// and Integer(5) narrows to Natural(5).
    pub fn narrow(self) -> Self {
        match self {
            Number::Rational {
                numerator,
                denominator,
            } if denominator == 1 => {
                if numerator >= 0 {
                    Number::Natural(numerator as u64)
                } else {
                    Number::Integer(numerator)
                }
            }
            Number::Integer(n) if n >= 0 => Number::Natural(n as u64),
            // Narrow only exact integral floats. A tolerance here can silently
            // change a nearby fractional value into a different integer.
            Number::Real(x) if x.fract() == 0.0 && x >= 0.0 && x < u64::MAX as f64 => {
                Number::Natural(x as u64)
            }
            Number::Real(x)
                if x.fract() == 0.0 && x >= i64::MIN as f64 && x < i64::MAX as f64 =>
            {
                Number::Integer(x as i64)
            }
            other => other,
        }
    }
}

impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Number::Natural(n) => write!(f, "{n}"),
            Number::Integer(n) => write!(f, "{n}"),
            Number::Rational {
                numerator,
                denominator,
            } => {
                write!(f, "{numerator}/{denominator}")
            }
            Number::Real(x) => write!(f, "{x:.10}"),
        }
    }
}

// ============================================================================
// RESULT TYPE
// ============================================================================

/// Result of a numeric tower operation.
///
/// Contains the computed number, its HDC encoding, proof trace, and
/// consciousness measure (Φ).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberResult {
    /// The resulting number in the appropriate domain
    pub number: Number,

    /// Description of the operation performed
    pub operation: String,

    /// Proof trace: symbolic reasoning steps documenting the computation
    pub proof_trace: Vec<String>,

    /// Φ (integrated information): consciousness measure for this operation.
    /// Increases with domain promotions, operation complexity, and proof depth.
    pub phi: f64,

    /// BinaryHV encoding of the result in 16,384-dimensional HDC space
    pub encoding: BinaryHV,
}

/// Exactness classification for a numeric-tower operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithmeticPrecision {
    /// The value is the exact result of the operation over the represented inputs.
    Exact,
    /// The value was computed using floating-point arithmetic or an explicit
    /// fallback because the fixed-width exact representation was insufficient.
    Approximate,
}

/// Typed failures returned by the checked numeric-tower API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericArithmeticError {
    /// A public Rational variant had a zero denominator.
    InvalidRationalOperand,
    /// Division by an exactly zero denominator was requested.
    DivisionByZero,
    /// An input Real was NaN or infinite.
    NonFiniteOperand,
    /// A finite input operation produced a non-finite floating-point result.
    NonFiniteResult,
    /// The exact value cannot fit the current fixed-width Number representation.
    ExactResultOutOfRange,
}

impl std::fmt::Display for NumericArithmeticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRationalOperand => write!(f, "rational operand has a zero denominator"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::NonFiniteOperand => write!(f, "operand is NaN or infinite"),
            Self::NonFiniteResult => write!(f, "operation produced a non-finite result"),
            Self::ExactResultOutOfRange => {
                write!(f, "exact result exceeds the fixed-width Number representation")
            }
        }
    }
}

impl std::error::Error for NumericArithmeticError {}

/// Result of a checked operation, including whether its value is exact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericArithmeticResult {
    /// Full numeric-tower result, including encoding and operation trace.
    pub result: NumberResult,
    /// Exactness classification of the computed value.
    pub precision: ArithmeticPrecision,
}

impl NumberResult {
    /// Add a proof step to the trace
    fn add_proof_step(&mut self, step: String) {
        self.proof_trace.push(step);
    }

    /// Add to the accumulated Φ measure
    fn add_phi(&mut self, delta: f64) {
        self.phi += delta;
    }
}

// ============================================================================
// NUMERIC TOWER ENGINE
// ============================================================================

/// Unified numeric tower with automatic domain promotion.
///
/// Performs arithmetic across ℕ, ℤ, ℚ, and ℝ, automatically promoting
/// results to the narrowest domain that can represent them exactly.
/// Every result is encoded as a `BinaryHV` and tracked with a
/// consciousness measure (Φ).
///
/// # Examples
///
/// ```rust,ignore
/// use symthaea::hdc::numeric_tower::NumericTower;
///
/// let mut tower = NumericTower::new();
///
/// // 5 - 3 = 2 (stays in ℕ)
/// let r = tower.subtract(
///     &NumericTower::from_u64(5),
///     &NumericTower::from_u64(3),
/// );
/// assert!(matches!(r.number, Number::Natural(2)));
///
/// // 3 - 5 = -2 (promotes to ℤ)
/// let r = tower.subtract(
///     &NumericTower::from_u64(3),
///     &NumericTower::from_u64(5),
/// );
/// assert!(matches!(r.number, Number::Integer(-2)));
/// ```
pub struct NumericTower {
    /// Primitive system for HDC encoding
    primitives: &'static PrimitiveSystem,
}

impl NumericTower {
    /// Create a new numeric tower engine.
    pub fn new() -> Self {
        Self {
            primitives: PrimitiveSystem::global(),
        }
    }

    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /// Create a Number from a u64 (natural number domain).
    pub fn from_u64(n: u64) -> Number {
        Number::Natural(n)
    }

    /// Create a Number from an i64 (integer or natural domain).
    pub fn from_i64(n: i64) -> Number {
        if n >= 0 {
            Number::Natural(n as u64)
        } else {
            Number::Integer(n)
        }
    }

    /// Create a Number from an f64 (attempts to narrow to most specific domain).
    pub fn from_f64(x: f64) -> Number {
        // Try to represent as an exact integer first
        if x.is_finite() && x.fract() == 0.0 {
            // u64::MAX as f64 rounds to 2^64, which is an exclusive upper
            // bound for values that can be cast back without saturation.
            if x >= 0.0 && x < u64::MAX as f64 {
                return Number::Natural(x as u64);
            }
            // i64::MAX as f64 similarly rounds to 2^63; keep that boundary
            // exclusive while allowing i64::MIN exactly.
            if x >= i64::MIN as f64 && x < i64::MAX as f64 {
                return Number::Integer(x as i64);
            }
        }
        Number::Real(x)
    }

    /// Create a Number from a rational p/q, normalizing and narrowing when exact.
    ///
    /// For compatibility this constructor falls back to Real only when a valid
    /// rational cannot fit the fixed-width Number representation. Use
    /// checked_from_rational when an exact value is mandatory.
    pub fn from_rational(numerator: i64, denominator: i64) -> Number {
        assert!(denominator != 0, "Denominator cannot be zero");
        match Self::checked_from_rational(numerator, denominator) {
            Ok(number) => number,
            Err(NumericArithmeticError::ExactResultOutOfRange) => {
                Number::Real(numerator as f64 / denominator as f64)
            }
            Err(_) => unreachable!("nonzero constructor denominator was validated"),
        }
    }

    /// Construct a rational Number without approximation.
    ///
    /// Returns a typed error if the denominator is zero or the normalized exact
    /// fraction cannot fit Natural(u64), Integer(i64), or Rational(i64, i64).
    pub fn checked_from_rational(
        numerator: i64,
        denominator: i64,
    ) -> Result<Number, NumericArithmeticError> {
        if denominator == 0 {
            return Err(NumericArithmeticError::InvalidRationalOperand);
        }
        Self::number_from_fraction_i128(i128::from(numerator), i128::from(denominator))
            .ok_or(NumericArithmeticError::ExactResultOutOfRange)
    }

    // ========================================================================
    // HDC ENCODING
    // ========================================================================

    /// Encode a Number as a BinaryHV in hyperdimensional space.
    ///
    /// The encoding strategy depends on the domain:
    /// - ℕ: Peano-style (ZERO + SUCCESSOR chain) for small values, seeded for large
    /// - ℤ: NEGATION ⊗ magnitude for negative values
    /// - ℚ: RATIO ⊗ encode(numerator) ⊗ encode(denominator)
    /// - ℝ: Deterministic seed from float bits
    fn encode(&self, number: &Number) -> BinaryHV {
        match number {
            Number::Natural(n) => self.encode_natural(*n),
            Number::Integer(n) => self.encode_integer(*n),
            Number::Rational {
                numerator,
                denominator,
            } => self.encode_rational(*numerator, *denominator),
            Number::Real(x) => self.encode_real(*x),
        }
    }

    /// Encode a natural number.
    fn encode_natural(&self, n: u64) -> BinaryHV {
        if n == 0 {
            return self
                .primitives
                .get("ZERO")
                .expect("ZERO primitive must exist")
                .encoding;
        }

        if n <= 20 {
            // Peano-style: ZERO then n applications of SUCCESSOR with permutation.
            // Permute between each step to avoid XOR self-inverse cancellation
            // (without permute, all even numbers and all odd numbers would collide).
            let zero = self
                .primitives
                .get("ZERO")
                .expect("ZERO primitive must exist")
                .encoding;
            let succ = self
                .primitives
                .get("SUCCESSOR")
                .expect("SUCCESSOR primitive must exist")
                .encoding;

            let mut encoding = zero;
            for _ in 0..n {
                encoding = succ.bind(&encoding).permute(1);
            }
            encoding
        } else {
            // Deterministic seed for large naturals
            BinaryHV::random(seed_from_name(&format!("NAT_{n}")))
        }
    }

    /// Encode an integer (may be negative).
    fn encode_integer(&self, n: i64) -> BinaryHV {
        if n >= 0 {
            return self.encode_natural(n as u64);
        }

        // Negative: NEGATION ⊗ magnitude
        let magnitude = self.encode_natural(n.unsigned_abs());
        let negation = self
            .primitives
            .get("NOT")
            .or_else(|| self.primitives.get("NEGATION"))
            .expect("NOT/NEGATION primitive must exist");
        negation.encoding.bind(&magnitude)
    }

    /// Encode a rational number p/q.
    fn encode_rational(&self, numerator: i64, denominator: i64) -> BinaryHV {
        let ratio = self
            .primitives
            .get("RATIO")
            .expect("RATIO primitive must exist")
            .encoding;
        let num_enc = self.encode_integer(numerator);
        let den_enc = self.encode_natural(denominator as u64);

        ratio.bind(&num_enc).bind(&den_enc)
    }

    /// Encode a real number from its f64 representation.
    fn encode_real(&self, x: f64) -> BinaryHV {
        // Use the float bits as a deterministic seed
        let bits = x.to_bits();
        BinaryHV::random(seed_from_name(&format!("REAL_{bits}")))
    }

    // ========================================================================
    // PHI CALCULATION
    // ========================================================================

    /// Compute the base Φ for a domain promotion.
    ///
    /// Crossing a mathematical boundary integrates more information
    /// and therefore contributes more consciousness.
    fn promotion_phi(from: NumberDomain, to: NumberDomain) -> f64 {
        if from == to {
            return 0.0;
        }
        match (from, to) {
            (NumberDomain::Natural, NumberDomain::Integer) => 1.0,
            (NumberDomain::Natural, NumberDomain::Rational) => 2.0,
            (NumberDomain::Natural, NumberDomain::Real) => 3.0,
            (NumberDomain::Integer, NumberDomain::Rational) => 1.5,
            (NumberDomain::Integer, NumberDomain::Real) => 2.5,
            (NumberDomain::Rational, NumberDomain::Real) => 2.0,
            _ => 0.0,
        }
    }

    /// Compute the operation complexity Φ.
    fn operation_phi(op: &str) -> f64 {
        match op {
            "add" | "subtract" => 0.5,
            "multiply" => 1.0,
            "divide" => 1.5,
            "power" => 2.0,
            "negate" | "abs" => 0.3,
            _ => 0.1,
        }
    }

    // ========================================================================
    // BUILDING RESULTS
    // ========================================================================

    /// Build a NumberResult from a computed Number.
    fn build_result(
        &self,
        number: Number,
        operation: String,
        proof_trace: Vec<String>,
        input_domains: &[NumberDomain],
    ) -> NumberResult {
        let result_domain = number.domain();
        let encoding = self.encode(&number);

        // Compute Φ: base operation + promotion bonuses
        let op_name = operation.split('(').next().unwrap_or(&operation);
        let mut phi = Self::operation_phi(op_name);

        for &input_domain in input_domains {
            phi += Self::promotion_phi(input_domain, result_domain);
        }

        // Extra Φ for proof depth (more reasoning = more consciousness)
        phi += proof_trace.len() as f64 * 0.1;

        NumberResult {
            number,
            operation,
            proof_trace,
            phi,
            encoding,
        }
    }

    /// Convert a valid non-real Number to a normalized exact fraction.
    fn exact_fraction_i128(number: &Number) -> Option<(i128, i128)> {
        match number {
            Number::Natural(value) => Some((i128::from(*value), 1)),
            Number::Integer(value) => Some((i128::from(*value), 1)),
            Number::Rational {
                numerator,
                denominator,
            } => normalize_fraction_i128(i128::from(*numerator), i128::from(*denominator)),
            Number::Real(_) => None,
        }
    }

    /// Narrow an exact fraction into the current fixed-width Number variants.
    ///
    /// None means the exact value is outside the representable Number range;
    /// callers must not wrap it into a different integer.
    fn number_from_fraction_i128(numerator: i128, denominator: i128) -> Option<Number> {
        let (numerator, denominator) = normalize_fraction_i128(numerator, denominator)?;

        if denominator == 1 {
            if numerator >= 0 && numerator <= i128::from(u64::MAX) {
                return Some(Number::Natural(numerator as u64));
            }
            if numerator < 0 && numerator >= i128::from(i64::MIN) {
                return Some(Number::Integer(numerator as i64));
            }
            return None;
        }

        if numerator < i128::from(i64::MIN)
            || numerator > i128::from(i64::MAX)
            || denominator > i128::from(i64::MAX)
        {
            return None;
        }

        Some(Number::Rational {
            numerator: numerator as i64,
            denominator: denominator as i64,
        })
    }

    /// Compute a binary operation exactly when possible without leaving the
    /// fixed-width Number representation. None requests an explicit Real
    /// fallback; it never indicates that wrapping arithmetic is acceptable.
    fn exact_binary_result(a: &Number, b: &Number, operation: &str) -> Option<Number> {
        let (an, ad) = Self::exact_fraction_i128(a)?;
        let (bn, bd) = Self::exact_fraction_i128(b)?;

        let (numerator, denominator) = match operation {
            "add" => (
                an.checked_mul(bd)?.checked_add(bn.checked_mul(ad)?)?,
                ad.checked_mul(bd)?,
            ),
            "subtract" => (
                an.checked_mul(bd)?.checked_sub(bn.checked_mul(ad)?)?,
                ad.checked_mul(bd)?,
            ),
            "multiply" => (an.checked_mul(bn)?, ad.checked_mul(bd)?),
            "divide" if bn != 0 => (an.checked_mul(bd)?, ad.checked_mul(bn)?),
            _ => return None,
        };

        Self::number_from_fraction_i128(numerator, denominator)
    }

    /// Centralized binary arithmetic. Exact operands are computed using checked
    /// i128 intermediates and normalized fractions. If either operand or the
    /// exact result cannot be represented, the result is explicitly a Real and
    /// its proof trace says that it is approximate.
    fn binary_arithmetic(
        &self,
        a: &Number,
        b: &Number,
        operation: &str,
        symbol: &str,
    ) -> NumberResult {
        let mut proof_trace = vec![format!("{}: {} {} {}", operation, a, symbol, b)];

        let result = if matches!(a, Number::Real(_)) || matches!(b, Number::Real(_)) {
            let value = Self::apply_f64(a.to_f64(), b.to_f64(), operation);
            proof_trace.push(format!(
                "Approximate floating-point {}: result is Real({value}); exactness is not claimed.",
                operation.to_lowercase()
            ));
            Number::Real(value)
        } else if let Some(exact) = Self::exact_binary_result(a, b, operation) {
            let input_domain = a.domain().max(b.domain());
            if exact.domain() > input_domain {
                proof_trace.push(format!(
                    "Promotion {} -> {}: exact result = {}",
                    input_domain,
                    exact.domain(),
                    exact
                ));
            } else {
                proof_trace.push(format!("Exact {} result = {}", operation.to_lowercase(), exact));
            }
            exact
        } else {
            let value = Self::apply_f64(a.to_f64(), b.to_f64(), operation);
            let malformed_fraction = matches!(a, Number::Rational { denominator: 0, .. })
                || matches!(b, Number::Rational { denominator: 0, .. });
            if malformed_fraction {
                proof_trace.push(format!(
                    "Invalid rational operand: {} fallback is approximate and must not be treated as exact.",
                    operation
                ));
            } else {
                proof_trace.push(format!(
                    "Exact {} exceeds the supported fixed-width range or intermediate capacity; promoted to approximate Real({value}).",
                    operation.to_lowercase()
                ));
            }
            Number::Real(value)
        };

        let input_domains = [a.domain(), b.domain()];
        self.build_result(result, operation.to_string(), proof_trace, &input_domains)
    }

    fn apply_f64(a: f64, b: f64, operation: &str) -> f64 {
        match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => a / b,
            _ => f64::NAN,
        }
    }

    // ========================================================================
    // ARITHMETIC OPERATIONS
    // ========================================================================

    /// Validate the domain-level invariants required by checked arithmetic.
    fn validate_checked_operand(number: &Number) -> Result<(), NumericArithmeticError> {
        match number {
            Number::Rational { denominator: 0, .. } => {
                Err(NumericArithmeticError::InvalidRationalOperand)
            }
            Number::Real(value) if !value.is_finite() => {
                Err(NumericArithmeticError::NonFiniteOperand)
            }
            _ => Ok(()),
        }
    }

    /// Shared checked arithmetic implementation used by the checked public API.
    fn checked_binary_arithmetic(
        &self,
        a: &Number,
        b: &Number,
        operation: &str,
        symbol: &str,
    ) -> Result<NumericArithmeticResult, NumericArithmeticError> {
        Self::validate_checked_operand(a)?;
        Self::validate_checked_operand(b)?;
        if operation == "divide" && b.is_zero_exact() {
            return Err(NumericArithmeticError::DivisionByZero);
        }

        let mut proof_trace = vec![format!("{}: {} {} {}", operation, a, symbol, b)];
        let has_real_operand = matches!(a, Number::Real(_)) || matches!(b, Number::Real(_));

        let (number, precision) = if !has_real_operand {
            if let Some(exact) = Self::exact_binary_result(a, b, operation) {
                proof_trace.push(format!("Exact {} result = {}", operation, exact));
                (exact, ArithmeticPrecision::Exact)
            } else {
                let value = Self::apply_f64(a.to_f64(), b.to_f64(), operation);
                if !value.is_finite() {
                    return Err(NumericArithmeticError::NonFiniteResult);
                }
                proof_trace.push(format!(
                    "Exact {} exceeds the current fixed-width representation or intermediate capacity; approximate Real({value}) returned.",
                    operation
                ));
                (Number::Real(value), ArithmeticPrecision::Approximate)
            }
        } else {
            let value = Self::apply_f64(a.to_f64(), b.to_f64(), operation);
            if !value.is_finite() {
                return Err(NumericArithmeticError::NonFiniteResult);
            }
            proof_trace.push(format!(
                "Floating-point {} evaluated to Real({value}); exactness is not claimed.",
                operation
            ));
            (Number::Real(value), ArithmeticPrecision::Approximate)
        };

        let input_domains = [a.domain(), b.domain()];
        let result = self.build_result(number, operation.to_string(), proof_trace, &input_domains);
        Ok(NumericArithmeticResult { result, precision })
    }

    /// Checked addition: rejects malformed/non-finite operands and reports exactness.
    pub fn checked_add(
        &self,
        a: &Number,
        b: &Number,
    ) -> Result<NumericArithmeticResult, NumericArithmeticError> {
        self.checked_binary_arithmetic(a, b, "add", "+")
    }

    /// Checked subtraction: rejects malformed/non-finite operands and reports exactness.
    pub fn checked_subtract(
        &self,
        a: &Number,
        b: &Number,
    ) -> Result<NumericArithmeticResult, NumericArithmeticError> {
        self.checked_binary_arithmetic(a, b, "subtract", "-")
    }

    /// Checked multiplication: rejects malformed/non-finite operands and reports exactness.
    pub fn checked_multiply(
        &self,
        a: &Number,
        b: &Number,
    ) -> Result<NumericArithmeticResult, NumericArithmeticError> {
        self.checked_binary_arithmetic(a, b, "multiply", "*")
    }

    /// Checked division: rejects invalid operands and division by exact zero.
    pub fn checked_divide(
        &self,
        a: &Number,
        b: &Number,
    ) -> Result<NumericArithmeticResult, NumericArithmeticError> {
        self.checked_binary_arithmetic(a, b, "divide", "/")
    }

    /// Add two numbers using checked exact arithmetic with an explicit Real fallback.
    pub fn add(&self, a: &Number, b: &Number) -> NumberResult {
        self.binary_arithmetic(a, b, "add", "+")
    }

    /// Subtract two numbers using checked exact arithmetic with an explicit Real fallback.
    pub fn subtract(&self, a: &Number, b: &Number) -> NumberResult {
        self.binary_arithmetic(a, b, "subtract", "-")
    }

    /// Multiply two numbers using checked exact arithmetic with an explicit Real fallback.
    pub fn multiply(&self, a: &Number, b: &Number) -> NumberResult {
        self.binary_arithmetic(a, b, "multiply", "*")
    }

    /// Divide two numbers.
    ///
    /// Returns None for an exactly zero divisor or a malformed rational divisor.
    /// Non-real operands retain exact rational results when representable;
    /// overflow is reported in the proof trace and falls back to approximate Real.
    pub fn divide(&self, a: &Number, b: &Number) -> Option<NumberResult> {
        if b.is_zero_exact() || matches!(b, Number::Rational { denominator: 0, .. }) {
            return None;
        }
        Some(self.binary_arithmetic(a, b, "divide", "/"))
    }

    /// Raise a number to a power: a^n
    ///
    /// For natural exponents, stays in the domain of the base when possible.
    /// For negative exponents, promotes to ℚ or ℝ.
    /// For fractional/real exponents, promotes to ℝ.
    pub fn power(&self, base: &Number, exponent: &Number) -> NumberResult {
        let mut proof_trace = vec![format!("Power: {}^{}", base, exponent)];

        let result = match exponent {
            // Integer exponent (natural or negative integer)
            Number::Natural(exp) => self.integer_power(base, *exp as i64, &mut proof_trace),
            Number::Integer(exp) => self.integer_power(base, *exp, &mut proof_trace),
            // Rational or real exponent → always ℝ
            _ => {
                let b = base.to_f64();
                let e = exponent.to_f64();
                let result = b.powf(e);
                proof_trace.push(format!("Real exponentiation: {b}^{e} = {result}"));
                Number::Real(result)
            }
        };

        let input_domains = [base.domain(), exponent.domain()];
        self.build_result(result, "power".to_string(), proof_trace, &input_domains)
    }

    /// Compute base^exp for integer exponents.
    fn integer_power(&self, base: &Number, exp: i64, proof_trace: &mut Vec<String>) -> Number {
        if exp == 0 {
            proof_trace.push("Any number to the 0th power is 1".to_string());
            return Number::Natural(1);
        }

        if exp > 0 {
            match base {
                Number::Natural(b) => {
                    let result = (*b as u128).pow(exp as u32);
                    if result <= u64::MAX as u128 {
                        proof_trace.push(format!("Natural power: {b}^{exp} = {result}"));
                        Number::Natural(result as u64)
                    } else {
                        proof_trace.push(format!("Overflow to Real: {b}^{exp}"));
                        Number::Real((*b as f64).powi(exp as i32))
                    }
                }
                Number::Integer(b) => {
                    // Use wrapping to handle overflow gracefully
                    let abs_result = (b.unsigned_abs() as u128).pow(exp as u32);
                    let is_negative = *b < 0 && exp % 2 == 1;
                    if abs_result <= i64::MAX as u128 {
                        let val = if is_negative {
                            -(abs_result as i64)
                        } else {
                            abs_result as i64
                        };
                        proof_trace.push(format!("Integer power: {b}^{exp} = {val}"));
                        Self::from_i64(val)
                    } else {
                        proof_trace.push(format!("Overflow to Real: {b}^{exp}"));
                        Number::Real((*b as f64).powi(exp as i32))
                    }
                }
                Number::Rational {
                    numerator,
                    denominator,
                } => {
                    let num_pow = (*numerator as i128).pow(exp as u32);
                    let den_pow = (*denominator as i128).pow(exp as u32);
                    if num_pow.unsigned_abs() <= i64::MAX as u128 && den_pow <= i64::MAX as i128 {
                        proof_trace.push(format!(
                            "Rational power: ({numerator}/{denominator})^{exp} = {num_pow}/{den_pow}"
                        ));
                        Self::from_rational(num_pow as i64, den_pow as i64)
                    } else {
                        proof_trace.push(format!(
                            "Overflow to Real: ({numerator}/{denominator})^{exp}"
                        ));
                        Number::Real((base.to_f64()).powi(exp as i32))
                    }
                }
                Number::Real(x) => {
                    let result = x.powi(exp as i32);
                    proof_trace.push(format!("Real power: {x}^{exp} = {result}"));
                    Number::Real(result)
                }
            }
        } else {
            // Negative exponent: a^(-n) = 1 / a^n
            proof_trace.push(format!(
                "Negative exponent: {}^{} = 1 / {}^{}",
                base, exp, base, -exp
            ));
            let pos_power = self.integer_power(base, -exp, proof_trace);
            match pos_power {
                Number::Natural(n) if n != 0 => Self::from_rational(1, n as i64),
                Number::Integer(n) if n != 0 => Self::from_rational(1, n),
                Number::Rational {
                    numerator,
                    denominator,
                } if numerator != 0 => Self::from_rational(denominator, numerator),
                Number::Real(x) if x.abs() > 1e-15 => Number::Real(1.0 / x),
                _ => Number::Real(f64::INFINITY),
            }
        }
    }

    /// Negate a number: -a.
    ///
    /// Uses exact widening where the existing Number variants can represent the
    /// result and records any unavoidable Real fallback in the proof trace.
    pub fn negate(&self, a: &Number) -> NumberResult {
        let mut proof_trace = vec![format!("Negate: -({})", a)];

        let result = match a {
            Number::Natural(0) => {
                proof_trace.push("Negation of zero is zero".to_string());
                Number::Natural(0)
            }
            Number::Natural(n) if *n <= i64::MAX as u64 => {
                let neg = -(*n as i64);
                proof_trace.push(format!("Exact negation: -({n}) = {neg}"));
                Number::Integer(neg)
            }
            Number::Natural(n) => {
                let value = -(*n as f64);
                proof_trace.push(format!(
                    "Negation of {n} exceeds the signed fixed-width range; approximate Real({value}) returned."
                ));
                Number::Real(value)
            }
            Number::Integer(n) => match n.checked_neg() {
                Some(neg) => {
                    proof_trace.push(format!("Exact integer negation: -({n}) = {neg}"));
                    Self::from_i64(neg)
                }
                None => {
                    // -i64::MIN = 2^63, which is representable as a Natural.
                    proof_trace.push(format!("Exact integer negation: -({n}) = {}", n.unsigned_abs()));
                    Number::Natural(n.unsigned_abs())
                }
            },
            Number::Rational { numerator, denominator } => {
                if *denominator == 0 {
                    proof_trace.push("Invalid rational operand: zero denominator; result is an invalid NaN sentinel.".to_string());
                    Number::Real(f64::NAN)
                } else if let Some(negated) = Self::number_from_fraction_i128(
                    -i128::from(*numerator),
                    i128::from(*denominator),
                ) {
                    proof_trace.push(format!("Exact rational negation: -({numerator}/{denominator}) = {negated}"));
                    negated
                } else {
                    let value = -(*numerator as f64 / *denominator as f64);
                    proof_trace.push(format!("Rational negation exceeds fixed-width storage; approximate Real({value}) returned."));
                    Number::Real(value)
                }
            }
            Number::Real(x) => {
                let neg = -*x;
                proof_trace.push(format!("Floating-point negation: -({x}) = {neg}; exactness is not claimed."));
                Number::Real(neg)
            }
        };

        let input_domains = [a.domain()];
        self.build_result(result, "negate".to_string(), proof_trace, &input_domains)
    }

    /// Absolute value: |a|.
    ///
    /// Uses unsigned magnitude and wide fraction normalization to preserve the
    /// exact absolute value of signed minimum values where representable.
    pub fn abs(&self, a: &Number) -> NumberResult {
        let mut proof_trace = vec![format!("Absolute value: |{}|", a)];

        let result = match a {
            Number::Natural(n) => {
                proof_trace.push(format!("Already non-negative: |{n}| = {n}"));
                Number::Natural(*n)
            }
            Number::Integer(n) => {
                let magnitude = n.unsigned_abs();
                proof_trace.push(format!("Exact integer absolute value: |{n}| = {magnitude}"));
                Number::Natural(magnitude)
            }
            Number::Rational { numerator, denominator } => {
                if *denominator == 0 {
                    proof_trace.push("Invalid rational operand: zero denominator; result is an invalid NaN sentinel.".to_string());
                    Number::Real(f64::NAN)
                } else if let Some(absolute) = Self::number_from_fraction_i128(
                    i128::from(*numerator).abs(),
                    i128::from(*denominator),
                ) {
                    proof_trace.push(format!("Exact rational absolute value: |{numerator}/{denominator}| = {absolute}"));
                    absolute
                } else {
                    let value = (*numerator as f64 / *denominator as f64).abs();
                    proof_trace.push(format!("Rational absolute value exceeds fixed-width storage; approximate Real({value}) returned."));
                    Number::Real(value)
                }
            }
            Number::Real(x) => {
                let abs_val = x.abs();
                proof_trace.push(format!("Floating-point absolute value: |{x}| = {abs_val}; exactness is not claimed."));
                Number::Real(abs_val)
            }
        };

        let input_domains = [a.domain()];
        self.build_result(result, "abs".to_string(), proof_trace, &input_domains)
    }

    /// Square root: sqrt(a)
    ///
    /// Returns Natural if a is a perfect square, otherwise promotes to Real.
    /// Returns `None` for negative inputs (no real square root).
    pub fn sqrt(&self, a: &Number) -> Option<NumberResult> {
        let val = a.to_f64();
        if val < 0.0 {
            return None;
        }

        let mut proof_trace = vec![format!("Square root: sqrt({})", a)];

        let result = match a {
            Number::Natural(n) => {
                let isqrt = (*n as f64).sqrt().round() as u64;
                if isqrt.wrapping_mul(isqrt) == *n {
                    proof_trace.push(format!("Perfect square: sqrt({n}) = {isqrt} (stays in N)"));
                    Number::Natural(isqrt)
                } else {
                    let root = (*n as f64).sqrt();
                    proof_trace.push(format!("Promotion N -> R: sqrt({n}) = {root} (irrational)"));
                    Number::Real(root)
                }
            }
            Number::Integer(n) => {
                if *n < 0 {
                    return None;
                }
                let un = *n as u64;
                let isqrt = (un as f64).sqrt().round() as u64;
                if isqrt.wrapping_mul(isqrt) == un {
                    proof_trace.push(format!(
                        "Perfect square: sqrt({n}) = {isqrt} (narrows to N)"
                    ));
                    Number::Natural(isqrt)
                } else {
                    let root = (un as f64).sqrt();
                    proof_trace.push(format!("Promotion Z -> R: sqrt({n}) = {root} (irrational)"));
                    Number::Real(root)
                }
            }
            Number::Rational {
                numerator,
                denominator,
            } => {
                if *numerator < 0 {
                    return None;
                }
                let val_f64 = *numerator as f64 / *denominator as f64;
                let root = val_f64.sqrt();
                // Check if the result is rational: sqrt(p/q) is rational iff sqrt(p) and sqrt(q) are integers
                let sqrt_num = (*numerator as f64).sqrt().round() as i64;
                let sqrt_den = (*denominator as f64).sqrt().round() as i64;
                if sqrt_num.wrapping_mul(sqrt_num) == *numerator
                    && sqrt_den.wrapping_mul(sqrt_den) == *denominator
                {
                    proof_trace.push(format!(
                        "Rational sqrt: sqrt({numerator}/{denominator}) = {sqrt_num}/{sqrt_den} (stays in Q)"
                    ));
                    Self::from_rational(sqrt_num, sqrt_den)
                } else {
                    proof_trace.push(format!(
                        "Promotion Q -> R: sqrt({numerator}/{denominator}) = {root} (irrational)"
                    ));
                    Number::Real(root)
                }
            }
            Number::Real(x) => {
                let root = x.sqrt();
                proof_trace.push(format!("Real sqrt: sqrt({x}) = {root}"));
                Number::Real(root)
            }
        };

        let input_domains = [a.domain()];
        Some(self.build_result(result, "sqrt".to_string(), proof_trace, &input_domains))
    }

    // ========================================================================
    // INTERNAL HELPERS
    // ========================================================================

    /// Convert any Number to a rational (numerator, denominator) pair.
    fn to_rational(&self, n: &Number) -> (i64, i64) {
        match n {
            Number::Natural(v) => (*v as i64, 1),
            Number::Integer(v) => (*v, 1),
            Number::Rational {
                numerator,
                denominator,
            } => (*numerator, *denominator),
            Number::Real(x) => {
                // Best rational approximation via continued fractions
                self.f64_to_rational(*x, 10_000)
            }
        }
    }

    /// Approximate an f64 as a rational using continued fractions.
    fn f64_to_rational(&self, value: f64, max_denominator: u64) -> (i64, i64) {
        if value.is_nan() || value.is_infinite() {
            return (0, 1);
        }

        let sign: i64 = if value < 0.0 { -1 } else { 1 };
        let mut x = value.abs();
        let a0 = x.floor() as i64;

        let mut p = (a0, 1_i64);
        let mut q = (1_i64, 0_i64);

        loop {
            let remainder = x - (a0 as f64);
            if remainder < 1e-15 {
                break;
            }

            x = 1.0 / (x - x.floor());
            let a = x.floor() as i64;

            let new_p = a.wrapping_mul(p.0).wrapping_add(q.0);
            let new_q = a.wrapping_mul(p.1).wrapping_add(q.1);

            if (new_q as u64) > max_denominator {
                break;
            }

            q = p;
            p = (new_p, new_q);

            if (value.abs() - p.0 as f64 / p.1 as f64).abs() < 1e-12 {
                break;
            }
        }

        (sign * p.0, p.1)
    }
}

impl Default for NumericTower {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // Constructor tests
    // ====================================================================

    #[test]
    fn test_from_u64() {
        let n = NumericTower::from_u64(42);
        assert!(matches!(n, Number::Natural(42)));
    }

    #[test]
    fn test_from_i64_positive() {
        let n = NumericTower::from_i64(7);
        assert!(matches!(n, Number::Natural(7)));
    }

    #[test]
    fn test_from_i64_negative() {
        let n = NumericTower::from_i64(-3);
        assert!(matches!(n, Number::Integer(-3)));
    }

    #[test]
    fn test_from_f64_near_integer_stays_real() {
        let value = 1.0 - 5e-13;
        assert!(matches!(NumericTower::from_f64(value), Number::Real(actual) if actual == value));

        let narrowed = Number::Real(value).narrow();
        assert!(matches!(narrowed, Number::Real(actual) if actual == value));
    }

    #[test]
    fn test_from_f64_upper_cast_boundaries_do_not_saturate() {
        let u64_upper_exclusive = u64::MAX as f64;
        assert!(matches!(
            NumericTower::from_f64(u64_upper_exclusive),
            Number::Real(value) if value == u64_upper_exclusive
        ));
        assert!(matches!(
            Number::Real(u64_upper_exclusive).narrow(),
            Number::Real(value) if value == u64_upper_exclusive
        ));

        let i64_min = i64::MIN as f64;
        assert!(matches!(NumericTower::from_f64(i64_min), Number::Integer(i64::MIN)));
        assert!(matches!(Number::Real(i64_min).narrow(), Number::Integer(i64::MIN)));
    }

    #[test]
    fn test_from_f64_integer() {
        let n = NumericTower::from_f64(5.0);
        assert!(matches!(n, Number::Natural(5)));
    }

    #[test]
    fn test_from_f64_negative_integer() {
        let n = NumericTower::from_f64(-3.0);
        assert!(matches!(n, Number::Integer(-3)));
    }

    #[test]
    fn test_from_f64_irrational() {
        let n = NumericTower::from_f64(std::f64::consts::PI);
        assert!(matches!(n, Number::Real(_)));
    }

    #[test]
    fn test_from_rational_reduces() {
        let n = NumericTower::from_rational(6, 4);
        // 6/4 = 3/2
        assert!(matches!(
            n,
            Number::Rational {
                numerator: 3,
                denominator: 2
            }
        ));
    }

    #[test]
    fn test_from_rational_integer() {
        let n = NumericTower::from_rational(6, 3);
        // 6/3 = 2, should narrow to Natural
        assert!(matches!(n, Number::Natural(2)));
    }

    #[test]
    fn test_from_rational_negative() {
        let n = NumericTower::from_rational(-4, 2);
        // -4/2 = -2, should narrow to Integer
        assert!(matches!(n, Number::Integer(-2)));
    }

    // ====================================================================
    // Domain and display tests
    // ====================================================================

    #[test]
    fn test_number_domains() {
        assert_eq!(Number::Natural(5).domain(), NumberDomain::Natural);
        assert_eq!(Number::Integer(-3).domain(), NumberDomain::Integer);
        assert_eq!(
            Number::Rational {
                numerator: 1,
                denominator: 2
            }
            .domain(),
            NumberDomain::Rational
        );
        assert_eq!(Number::Real(3.14).domain(), NumberDomain::Real);
    }

    #[test]
    fn test_number_to_f64() {
        assert!((Number::Natural(5).to_f64() - 5.0).abs() < 1e-10);
        assert!((Number::Integer(-3).to_f64() - (-3.0)).abs() < 1e-10);
        assert!(
            (Number::Rational {
                numerator: 1,
                denominator: 3
            }
            .to_f64()
                - 1.0 / 3.0)
                .abs()
                < 1e-10
        );
        assert!((Number::Real(2.5).to_f64() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_number_is_zero() {
        assert!(Number::Natural(0).is_zero());
        assert!(Number::Integer(0).is_zero());
        assert!(
            Number::Rational {
                numerator: 0,
                denominator: 5
            }
            .is_zero()
        );
        assert!(Number::Real(0.0).is_zero());
        assert!(!Number::Natural(1).is_zero());
    }

    // ====================================================================
    // Auto-promotion: subtraction
    // ====================================================================

    #[test]
    fn test_subtract_stays_natural() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(5);
        let b = NumericTower::from_u64(3);
        let r = tower.subtract(&a, &b);

        assert!(
            matches!(r.number, Number::Natural(2)),
            "5 - 3 = 2 should stay in N, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_subtract_promotes_to_integer() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(3);
        let b = NumericTower::from_u64(5);
        let r = tower.subtract(&a, &b);

        assert!(
            matches!(r.number, Number::Integer(-2)),
            "3 - 5 = -2 should promote to Z, got {:?}",
            r.number
        );
    }

    // ====================================================================
    // Auto-promotion: division
    // ====================================================================

    #[test]
    fn test_divide_promotes_to_rational() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(7);
        let b = NumericTower::from_u64(2);
        let r = tower.divide(&a, &b).expect("non-zero divisor");

        assert!(
            matches!(
                r.number,
                Number::Rational {
                    numerator: 7,
                    denominator: 2
                }
            ),
            "7 / 2 should promote to Q as 7/2, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_divide_stays_integer() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(6);
        let b = NumericTower::from_u64(3);
        let r = tower.divide(&a, &b).expect("non-zero divisor");

        assert!(
            matches!(r.number, Number::Natural(2)),
            "6 / 3 = 2 should stay in N, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_divide_by_zero() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(5);
        let b = NumericTower::from_u64(0);
        assert!(
            tower.divide(&a, &b).is_none(),
            "Division by zero should return None"
        );
    }

    // ====================================================================
    // Auto-promotion: sqrt
    // ====================================================================

    #[test]
    fn test_sqrt_perfect_square_stays_natural() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(9);
        let r = tower.sqrt(&a).expect("non-negative");

        assert!(
            matches!(r.number, Number::Natural(3)),
            "sqrt(9) = 3 should stay in N, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_sqrt_promotes_to_real() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(2);
        let r = tower.sqrt(&a).expect("non-negative");

        match &r.number {
            Number::Real(x) => {
                assert!(
                    (x - std::f64::consts::SQRT_2).abs() < 1e-10,
                    "sqrt(2) should be ~1.41421, got {}",
                    x
                );
            }
            other => panic!("sqrt(2) should promote to R, got {:?}", other),
        }
    }

    #[test]
    fn test_sqrt_negative_returns_none() {
        let tower = NumericTower::new();
        let a = NumericTower::from_i64(-4);
        assert!(
            tower.sqrt(&a).is_none(),
            "sqrt(-4) should return None (no real root)"
        );
    }

    // ====================================================================
    // Addition tests
    // ====================================================================

    #[test]
    fn test_add_naturals() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(3);
        let b = NumericTower::from_u64(7);
        let r = tower.add(&a, &b);

        assert!(
            matches!(r.number, Number::Natural(10)),
            "3 + 7 = 10, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_add_integers() {
        let tower = NumericTower::new();
        let a = NumericTower::from_i64(-3);
        let b = NumericTower::from_i64(-5);
        let r = tower.add(&a, &b);

        assert!(
            matches!(r.number, Number::Integer(-8)),
            "-3 + -5 = -8, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_add_rationals() {
        let tower = NumericTower::new();
        let a = NumericTower::from_rational(1, 2);
        let b = NumericTower::from_rational(1, 3);
        let r = tower.add(&a, &b);

        // 1/2 + 1/3 = 5/6
        assert!(
            matches!(
                r.number,
                Number::Rational {
                    numerator: 5,
                    denominator: 6
                }
            ),
            "1/2 + 1/3 = 5/6, got {:?}",
            r.number
        );
    }

    // ====================================================================
    // Multiplication tests
    // ====================================================================

    #[test]
    fn test_multiply_naturals() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(4);
        let b = NumericTower::from_u64(5);
        let r = tower.multiply(&a, &b);

        assert!(
            matches!(r.number, Number::Natural(20)),
            "4 * 5 = 20, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_multiply_mixed_signs() {
        let tower = NumericTower::new();
        let a = NumericTower::from_i64(-3);
        let b = NumericTower::from_u64(4);
        let r = tower.multiply(&a, &b);

        assert!(
            matches!(r.number, Number::Integer(-12)),
            "-3 * 4 = -12, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_multiply_rationals() {
        let tower = NumericTower::new();
        let a = NumericTower::from_rational(2, 3);
        let b = NumericTower::from_rational(3, 4);
        let r = tower.multiply(&a, &b);

        // (2/3) * (3/4) = 6/12 = 1/2
        assert!(
            matches!(
                r.number,
                Number::Rational {
                    numerator: 1,
                    denominator: 2
                }
            ),
            "(2/3) * (3/4) = 1/2, got {:?}",
            r.number
        );
    }

    // ====================================================================
    // Negate tests
    // ====================================================================

    #[test]
    fn test_negate_natural_promotes() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(5);
        let r = tower.negate(&a);

        assert!(
            matches!(r.number, Number::Integer(-5)),
            "-(5) should promote to Z as -5, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_negate_negative_returns_positive() {
        let tower = NumericTower::new();
        let a = NumericTower::from_i64(-7);
        let r = tower.negate(&a);

        assert!(
            matches!(r.number, Number::Natural(7)),
            "-(-7) = 7 should narrow to N, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_negate_zero() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(0);
        let r = tower.negate(&a);

        assert!(
            matches!(r.number, Number::Natural(0)),
            "-(0) = 0, got {:?}",
            r.number
        );
    }

    // ====================================================================
    // Abs tests
    // ====================================================================

    #[test]
    fn test_abs_positive() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(5);
        let r = tower.abs(&a);

        assert!(
            matches!(r.number, Number::Natural(5)),
            "|5| = 5, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_abs_negative() {
        let tower = NumericTower::new();
        let a = NumericTower::from_i64(-8);
        let r = tower.abs(&a);

        assert!(
            matches!(r.number, Number::Natural(8)),
            "|-8| = 8, got {:?}",
            r.number
        );
    }

    // ====================================================================
    // Power tests
    // ====================================================================

    #[test]
    fn test_power_natural() {
        let tower = NumericTower::new();
        let base = NumericTower::from_u64(2);
        let exp = NumericTower::from_u64(10);
        let r = tower.power(&base, &exp);

        assert!(
            matches!(r.number, Number::Natural(1024)),
            "2^10 = 1024, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_power_negative_exponent() {
        let tower = NumericTower::new();
        let base = NumericTower::from_u64(2);
        let exp = NumericTower::from_i64(-1);
        let r = tower.power(&base, &exp);

        // 2^(-1) = 1/2
        assert!(
            matches!(
                r.number,
                Number::Rational {
                    numerator: 1,
                    denominator: 2
                }
            ),
            "2^(-1) = 1/2, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_power_zero_exponent() {
        let tower = NumericTower::new();
        let base = NumericTower::from_u64(99);
        let exp = NumericTower::from_u64(0);
        let r = tower.power(&base, &exp);

        assert!(
            matches!(r.number, Number::Natural(1)),
            "99^0 = 1, got {:?}",
            r.number
        );
    }

    // ====================================================================
    // Φ (consciousness) tests
    // ====================================================================

    #[test]
    fn test_phi_increases_on_promotion() {
        let tower = NumericTower::new();

        // No promotion: 5 - 3 = 2 (stays in N)
        let a = NumericTower::from_u64(5);
        let b = NumericTower::from_u64(3);
        let r_no_promo = tower.subtract(&a, &b);

        // Promotion: 3 - 5 = -2 (N -> Z)
        let a2 = NumericTower::from_u64(3);
        let b2 = NumericTower::from_u64(5);
        let r_promo = tower.subtract(&a2, &b2);

        assert!(
            r_promo.phi > r_no_promo.phi,
            "Promotion should increase Phi: promo={}, no_promo={}",
            r_promo.phi,
            r_no_promo.phi
        );
    }

    #[test]
    fn test_phi_always_positive() {
        let tower = NumericTower::new();

        let a = NumericTower::from_u64(3);
        let b = NumericTower::from_u64(5);
        let r = tower.add(&a, &b);
        assert!(r.phi > 0.0, "Phi should always be positive, got {}", r.phi);
    }

    // ====================================================================
    // Encoding tests
    // ====================================================================

    #[test]
    fn test_encoding_exists() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(42);
        let b = NumericTower::from_u64(13);
        let r = tower.add(&a, &b);

        // Encoding should be a valid BinaryHV (not all zeros)
        let ones = r.encoding.0.iter().filter(|&&b| b != 0).count();
        assert!(ones > 0, "BinaryHV encoding should not be all zeros");
    }

    #[test]
    fn test_different_numbers_different_encodings() {
        let tower = NumericTower::new();
        let enc_5 = tower.encode(&NumericTower::from_u64(5));
        let enc_7 = tower.encode(&NumericTower::from_u64(7));

        // Different numbers should have different encodings
        assert_ne!(enc_5, enc_7, "5 and 7 should have different HDC encodings");
    }

    #[test]
    fn test_same_number_same_encoding() {
        let tower = NumericTower::new();
        let enc_a = tower.encode(&NumericTower::from_u64(42));
        let enc_b = tower.encode(&NumericTower::from_u64(42));

        // Same number should produce same encoding (deterministic)
        assert_eq!(enc_a, enc_b, "Same number should produce same encoding");
    }

    // ====================================================================
    // Proof trace tests
    // ====================================================================

    #[test]
    fn test_proof_trace_nonempty() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(7);
        let b = NumericTower::from_u64(2);
        let r = tower.divide(&a, &b).unwrap();

        assert!(!r.proof_trace.is_empty(), "Proof trace should not be empty");
        assert!(
            r.proof_trace
                .iter()
                .any(|s| s.contains("Divide") || s.contains("Promotion")),
            "Proof trace should document the operation: {:?}",
            r.proof_trace
        );
    }

    #[test]
    fn test_proof_trace_records_promotion() {
        let tower = NumericTower::new();
        let a = NumericTower::from_u64(3);
        let b = NumericTower::from_u64(5);
        let r = tower.subtract(&a, &b);

        assert!(
            r.proof_trace
                .iter()
                .any(|s| s.contains("Promotion") || s.contains("negative")),
            "Proof trace should record promotion: {:?}",
            r.proof_trace
        );
    }

    // ====================================================================
    // Narrowing tests
    // ====================================================================

    #[test]
    fn test_narrow_rational_to_natural() {
        let n = Number::Rational {
            numerator: 5,
            denominator: 1,
        };
        let narrowed = n.narrow();
        assert!(
            matches!(narrowed, Number::Natural(5)),
            "5/1 should narrow to Natural(5), got {:?}",
            narrowed
        );
    }

    #[test]
    fn test_narrow_rational_to_integer() {
        let n = Number::Rational {
            numerator: -3,
            denominator: 1,
        };
        let narrowed = n.narrow();
        assert!(
            matches!(narrowed, Number::Integer(-3)),
            "-3/1 should narrow to Integer(-3), got {:?}",
            narrowed
        );
    }

    #[test]
    fn test_narrow_integer_to_natural() {
        let n = Number::Integer(7);
        let narrowed = n.narrow();
        assert!(
            matches!(narrowed, Number::Natural(7)),
            "Integer(7) should narrow to Natural(7), got {:?}",
            narrowed
        );
    }

    // ====================================================================
    // GCD / normalization tests
    // ====================================================================

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 3), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(100, 100), 100);
    }

    #[test]
    fn test_normalize_rational() {
        assert_eq!(normalize_rational(6, 4), (3, 2));
        assert_eq!(normalize_rational(-6, 4), (-3, 2));
        assert_eq!(normalize_rational(6, -4), (-3, 2));
        assert_eq!(normalize_rational(0, 5), (0, 1));
    }

    // ====================================================================
    // Comprehensive integration: the full tower in action
    // ====================================================================

    #[test]
    fn test_tower_chain_computation() {
        let tower = NumericTower::new();

        // Start in N: 10
        let n = NumericTower::from_u64(10);
        assert_eq!(n.domain(), NumberDomain::Natural);

        // Subtract to promote to Z: 10 - 15 = -5
        let fifteen = NumericTower::from_u64(15);
        let r1 = tower.subtract(&n, &fifteen);
        assert!(matches!(r1.number, Number::Integer(-5)));

        // Divide to promote to Q: -5 / 3 = -5/3
        let three = NumericTower::from_u64(3);
        let r2 = tower.divide(&r1.number, &three).unwrap();
        assert!(matches!(
            r2.number,
            Number::Rational {
                numerator: -5,
                denominator: 3
            }
        ));

        // Take sqrt to promote to R: sqrt(|-5/3|) = sqrt(5/3)
        let abs_val = tower.abs(&r2.number);
        let r3 = tower.sqrt(&abs_val.number).unwrap();
        assert!(
            matches!(r3.number, Number::Real(_)),
            "sqrt(5/3) should be Real, got {:?}",
            r3.number
        );
    }

    #[test]
    fn test_domain_ordering() {
        assert!(NumberDomain::Natural < NumberDomain::Integer);
        assert!(NumberDomain::Integer < NumberDomain::Rational);
        assert!(NumberDomain::Rational < NumberDomain::Real);
    }

    #[test]
    fn test_mixed_domain_add() {
        let tower = NumericTower::new();

        // Natural + Integer
        let a = NumericTower::from_u64(5);
        let b = NumericTower::from_i64(-3);
        let r = tower.add(&a, &b);
        assert!(
            matches!(r.number, Number::Natural(2)),
            "5 + (-3) = 2, got {:?}",
            r.number
        );

        // Natural + Rational
        let c = NumericTower::from_u64(1);
        let d = NumericTower::from_rational(1, 2);
        let r2 = tower.add(&c, &d);
        // 1 + 1/2 = 3/2
        assert!(
            matches!(
                r2.number,
                Number::Rational {
                    numerator: 3,
                    denominator: 2
                }
            ),
            "1 + 1/2 = 3/2, got {:?}",
            r2.number
        );
    }

    #[test]
    fn test_sqrt_rational_perfect() {
        let tower = NumericTower::new();
        // sqrt(4/9) = 2/3
        let a = NumericTower::from_rational(4, 9);
        let r = tower.sqrt(&a).expect("non-negative");

        assert!(
            matches!(
                r.number,
                Number::Rational {
                    numerator: 2,
                    denominator: 3
                }
            ),
            "sqrt(4/9) = 2/3, got {:?}",
            r.number
        );
    }

    #[test]
    fn test_checked_from_rational_reports_unrepresentable_exact_values() {
        assert_eq!(
            NumericTower::checked_from_rational(1, 0).unwrap_err(),
            NumericArithmeticError::InvalidRationalOperand
        );
        assert_eq!(
            NumericTower::checked_from_rational(1, i64::MIN).unwrap_err(),
            NumericArithmeticError::ExactResultOutOfRange
        );
        assert!(matches!(
            NumericTower::checked_from_rational(i64::MIN, -1),
            Ok(Number::Natural(value)) if value == (1_u64 << 63)
        ));
        assert!(matches!(
            NumericTower::from_rational(i64::MIN, -1),
            Number::Natural(value) if value == (1_u64 << 63)
        ));
    }

    #[test]
    fn test_negate_wide_natural_and_signed_minimum() {
        let tower = NumericTower::new();
        let wide_natural = tower.negate(&Number::Natural(u64::MAX));
        assert!(matches!(wide_natural.number, Number::Real(value) if value < 0.0));
        assert!(wide_natural.proof_trace.iter().any(|line| line.contains("approximate Real")));

        let signed_min = tower.negate(&Number::Integer(i64::MIN));
        assert!(matches!(signed_min.number, Number::Natural(value) if value == (1_u64 << 63)));
        assert!(signed_min.proof_trace.iter().any(|line| line.contains("Exact integer negation")));
    }

    #[test]
    fn test_abs_preserves_minimum_integer_exactly() {
        let tower = NumericTower::new();
        let result = tower.abs(&Number::Integer(i64::MIN));
        assert!(matches!(result.number, Number::Natural(value) if value == (1_u64 << 63)));
        assert!(result.proof_trace.iter().any(|line| line.contains("Exact integer absolute value")));
    }

    #[test]
    fn test_checked_arithmetic_reports_exactness() {
        let tower = NumericTower::new();
        let exact = tower
            .checked_add(&Number::Natural(2), &Number::Natural(3))
            .expect("valid exact addition");
        assert_eq!(exact.precision, ArithmeticPrecision::Exact);
        assert!(matches!(exact.result.number, Number::Natural(5)));

        let approximate = tower
            .checked_add(&Number::Natural(u64::MAX), &Number::Natural(1))
            .expect("finite approximate fallback");
        assert_eq!(approximate.precision, ArithmeticPrecision::Approximate);
        assert!(matches!(approximate.result.number, Number::Real(value) if value.is_finite()));
        assert!(approximate.result.proof_trace.iter().any(|line| line.contains("approximate Real")));
    }

    #[test]
    fn test_checked_arithmetic_rejects_malformed_rational_operand() {
        let tower = NumericTower::new();
        let malformed = Number::Rational { numerator: 7, denominator: 0 };
        assert_eq!(
            tower.checked_add(&malformed, &Number::Natural(1)).unwrap_err(),
            NumericArithmeticError::InvalidRationalOperand
        );
        assert_eq!(
            tower.checked_multiply(&Number::Natural(1), &malformed).unwrap_err(),
            NumericArithmeticError::InvalidRationalOperand
        );
    }

    #[test]
    fn test_checked_arithmetic_rejects_non_finite_operands_and_results() {
        let tower = NumericTower::new();
        assert_eq!(
            tower.checked_add(&Number::Real(f64::NAN), &Number::Natural(1)).unwrap_err(),
            NumericArithmeticError::NonFiniteOperand
        );
        assert_eq!(
            tower.checked_multiply(&Number::Real(f64::MAX), &Number::Real(2.0)).unwrap_err(),
            NumericArithmeticError::NonFiniteResult
        );
    }

    #[test]
    fn test_checked_division_rejects_exact_zero() {
        let tower = NumericTower::new();
        assert_eq!(
            tower.checked_divide(&Number::Natural(1), &Number::Real(-0.0)).unwrap_err(),
            NumericArithmeticError::DivisionByZero
        );
    }

    #[test]
    fn test_natural_addition_overflow_is_approximate_not_wrapped() {
        let tower = NumericTower::new();
        let result = tower.add(&Number::Natural(u64::MAX), &Number::Natural(1));

        assert!(matches!(result.number, Number::Real(value) if value.is_finite()));
        assert!(
            result.proof_trace.iter().any(|step| step.contains("approximate Real")),
            "overflow fallback must be disclosed in the proof trace: {:?}",
            result.proof_trace
        );
    }

    #[test]
    fn test_mixed_natural_subtraction_does_not_cast_through_i64() {
        let tower = NumericTower::new();
        let result = tower.subtract(&Number::Natural(u64::MAX), &Number::Integer(1));

        assert!(
            matches!(result.number, Number::Natural(value) if value == u64::MAX - 1),
            "u64::MAX - 1 must remain exact, got {:?}",
            result.number
        );
    }

    #[test]
    fn test_rational_multiplication_reduces_wide_intermediate_exactly() {
        let tower = NumericTower::new();
        let rational = Number::Rational {
            numerator: i64::MAX,
            denominator: 2,
        };
        let result = tower.multiply(&rational, &Number::Natural(2));

        assert!(
            matches!(result.number, Number::Natural(value) if value == i64::MAX as u64),
            "((i64::MAX / 2) * 2) must reduce exactly, got {:?}",
            result.number
        );
        assert!(
            !result.proof_trace.iter().any(|step| step.contains("approximate Real")),
            "representable result should not take the approximation fallback: {:?}",
            result.proof_trace
        );
    }

    #[test]
    fn test_large_natural_division_never_wraps_through_i64() {
        let tower = NumericTower::new();
        let result = tower
            .divide(&Number::Natural(u64::MAX), &Number::Natural(2))
            .expect("divisor is nonzero");

        assert!(
            matches!(result.number, Number::Real(value) if (value - (u64::MAX as f64 / 2.0)).abs() < 2.0),
            "unrepresentable rational must use approximate Real rather than a corrupted fraction, got {:?}",
            result.number
        );
        assert!(
            result.proof_trace.iter().any(|step| step.contains("approximate Real")),
            "approximation must be disclosed: {:?}",
            result.proof_trace
        );
    }

    #[test]
    fn test_signed_underflow_uses_explicit_approximate_fallback() {
        let tower = NumericTower::new();
        let result = tower.subtract(&Number::Integer(i64::MIN), &Number::Natural(1));

        assert!(matches!(result.number, Number::Real(value) if value.is_finite()));
        assert!(
            result.proof_trace.iter().any(|step| step.contains("approximate Real")),
            "out-of-range exact result must be disclosed: {:?}",
            result.proof_trace
        );
    }

    #[test]
    fn test_exact_zero_is_distinct_from_legacy_approximate_zero() {
        assert!(Number::Natural(0).is_zero_exact());
        assert!(Number::Integer(0).is_zero_exact());
        assert!(Number::Rational { numerator: 0, denominator: 5 }.is_zero_exact());
        assert!(Number::Real(0.0).is_zero_exact());
        assert!(Number::Real(-0.0).is_zero_exact());

        let smallest_subnormal = f64::from_bits(1);
        assert!(!Number::Real(smallest_subnormal).is_zero_exact());
        // Keep the historical tolerance behavior for existing callers.
        assert!(Number::Real(smallest_subnormal).is_zero());
        assert!(!Number::Real(f64::NAN).is_zero_exact());
        assert!(!Number::Real(f64::INFINITY).is_zero_exact());
        assert!(!Number::Rational { numerator: 0, denominator: 0 }.is_zero_exact());
    }

    #[test]
    fn test_approximate_zero_requires_explicit_valid_tolerance() {
        assert!(Number::Real(1e-16).is_approximately_zero(1e-15));
        assert!(!Number::Real(1e-16).is_approximately_zero(1e-17));
        assert!(Number::Natural(0).is_approximately_zero(0.0));
        assert!(!Number::Real(f64::NAN).is_approximately_zero(1.0));
        assert!(!Number::Real(0.0).is_approximately_zero(-1.0));
        assert!(!Number::Real(0.0).is_approximately_zero(f64::NAN));
        assert!(!Number::Real(0.0).is_approximately_zero(f64::INFINITY));
    }

    #[test]
    fn test_division_rejects_invalid_rational_divisor() {
        let tower = NumericTower::new();
        let numerator = Number::Natural(1);
        let invalid_divisor = Number::Rational {
            numerator: 1,
            denominator: 0,
        };

        assert!(tower.divide(&numerator, &invalid_divisor).is_none());
    }

    #[test]
    fn test_division_accepts_small_nonzero_real_divisor() {
        let tower = NumericTower::new();
        let numerator = Number::Natural(1);
        let divisor = Number::Real(1e-16);

        let result = tower
            .divide(&numerator, &divisor)
            .expect("a finite, nonzero real divisor must not be rejected as zero");

        assert!(
            (result.number.to_f64() - 1e16).abs() < 2.0,
            "1 / 1e-16 should be approximately 1e16, got {:?}",
            result.number
        );
    }

}
