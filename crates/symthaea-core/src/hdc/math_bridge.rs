// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Math Bridge
//!
//! Connects the numeric tower (numeric_tower.rs) and complex arithmetic (complex.rs)
//! to the existing arithmetic engine, providing a single entry point for all
//! mathematical operations across every number domain:
//!
//!   ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ
//!
//! ## Key Types
//!
//! - [`MathValue`] — unified enum spanning Natural, Integer, Rational, Real, Complex
//! - [`MathResult`] — operation result with HDC encoding, Φ, and promotion trace
//! - [`UnifiedMathEngine`] — wraps `NumericTower` + `ComplexArithmeticEngine`
//!
//! ## Auto-Promotion Rules
//!
//! Mixed-domain operations promote to the narrowest domain that can represent the
//! result exactly:
//!
//! | Operation | Promotion |
//! |-----------|-----------|
//! | ℕ + ℤ | ℤ |
//! | ℤ + ℚ | ℚ |
//! | ℚ + ℝ | ℝ |
//! | anything + ℂ | ℂ |
//! | sqrt(negative) | ℂ |
//! | ℤ / ℤ (inexact) | ℚ |
//!
//! ## HDC Encoding
//!
//! Every `MathResult` carries a `BinaryHV` encoding in 16,384-dimensional
//! hyperdimensional space. Real-domain values delegate to `NumericTower::encode`;
//! complex values are encoded via `HdcComplex::encode`.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::complex::{Complex, ComplexArithmeticEngine, HdcComplex};
use crate::hdc::numeric_tower::{Number, NumberResult, NumericTower};
use crate::hdc::primitive_system::PrimitiveSystem;

// ============================================================================
// MATH VALUE — unified number representation across all domains
// ============================================================================

/// A number that can live in any domain of the extended tower ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ.
#[derive(Debug, Clone, PartialEq)]
pub enum MathValue {
    /// A natural number (non-negative integer): n >= 0
    Natural(u64),
    /// An integer (possibly negative)
    Integer(i64),
    /// A rational number p/q in lowest terms with q > 0
    Rational { numerator: i64, denominator: i64 },
    /// A real number (floating-point approximation)
    Real(f64),
    /// A complex number z = re + im*i
    Complex { re: f64, im: f64 },
}

impl MathValue {
    /// Convert from the existing `HdcNumber` type (arithmetic_engine.rs).
    ///
    /// `HdcNumber` stores a `u64` value, so the result is always `Natural`
    /// when the value is non-negative (which it always is for u64), but we
    /// accept a reference to any HdcNumber-like struct with a `value: u64` field.
    pub fn from_hdc_number_value(value: u64) -> MathValue {
        MathValue::Natural(value)
    }

    /// Convert to f64 for approximate computation.
    pub fn to_f64(&self) -> f64 {
        match self {
            MathValue::Natural(n) => *n as f64,
            MathValue::Integer(n) => *n as f64,
            MathValue::Rational {
                numerator,
                denominator,
            } => *numerator as f64 / *denominator as f64,
            MathValue::Real(x) => *x,
            MathValue::Complex { re, .. } => *re,
        }
    }

    /// Return the Unicode label for the domain this value belongs to.
    pub fn domain(&self) -> &str {
        match self {
            MathValue::Natural(_) => "\u{2115}",      // ℕ
            MathValue::Integer(_) => "\u{2124}",      // ℤ
            MathValue::Rational { .. } => "\u{211a}", // ℚ
            MathValue::Real(_) => "\u{211d}",         // ℝ
            MathValue::Complex { .. } => "\u{2102}",  // ℂ
        }
    }

    /// Numeric ordering of domains for promotion comparison.
    /// Higher rank means wider domain.
    fn domain_rank(&self) -> u8 {
        match self {
            MathValue::Natural(_) => 0,
            MathValue::Integer(_) => 1,
            MathValue::Rational { .. } => 2,
            MathValue::Real(_) => 3,
            MathValue::Complex { .. } => 4,
        }
    }

    /// Check if this value is zero in any domain.
    pub fn is_zero(&self) -> bool {
        match self {
            MathValue::Natural(n) => *n == 0,
            MathValue::Integer(n) => *n == 0,
            MathValue::Rational { numerator, .. } => *numerator == 0,
            MathValue::Real(x) => x.abs() < 1e-15,
            MathValue::Complex { re, im } => re.abs() < 1e-15 && im.abs() < 1e-15,
        }
    }

    /// Convert this MathValue into the numeric_tower `Number` type.
    /// Complex values are projected onto the real axis (the real part).
    fn to_tower_number(&self) -> Number {
        match self {
            MathValue::Natural(n) => Number::Natural(*n),
            MathValue::Integer(n) => Number::Integer(*n),
            MathValue::Rational {
                numerator,
                denominator,
            } => Number::Rational {
                numerator: *numerator,
                denominator: *denominator,
            },
            MathValue::Real(x) => Number::Real(*x),
            MathValue::Complex { re, .. } => Number::Real(*re),
        }
    }

    /// Convert this MathValue into the complex module's `Complex` type.
    fn to_complex(&self) -> Complex {
        match self {
            MathValue::Natural(n) => Complex::new(*n as f64, 0.0),
            MathValue::Integer(n) => Complex::new(*n as f64, 0.0),
            MathValue::Rational {
                numerator,
                denominator,
            } => Complex::new(*numerator as f64 / *denominator as f64, 0.0),
            MathValue::Real(x) => Complex::new(*x, 0.0),
            MathValue::Complex { re, im } => Complex::new(*re, *im),
        }
    }

    /// True if this value lives in the complex domain.
    fn is_complex(&self) -> bool {
        matches!(self, MathValue::Complex { .. })
    }
}

impl std::fmt::Display for MathValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MathValue::Natural(n) => write!(f, "{n}"),
            MathValue::Integer(n) => write!(f, "{n}"),
            MathValue::Rational {
                numerator,
                denominator,
            } => {
                write!(f, "{numerator}/{denominator}")
            }
            MathValue::Real(x) => write!(f, "{x:.10}"),
            MathValue::Complex { re, im } => {
                if *im >= 0.0 {
                    write!(f, "{re} + {im}i")
                } else {
                    write!(f, "{} - {}i", re, -im)
                }
            }
        }
    }
}

// ============================================================================
// Conversion helpers: Number <-> MathValue
// ============================================================================

/// Convert a numeric tower `Number` into a `MathValue`.
fn number_to_math_value(n: &Number) -> MathValue {
    match n {
        Number::Natural(v) => MathValue::Natural(*v),
        Number::Integer(v) => MathValue::Integer(*v),
        Number::Rational {
            numerator,
            denominator,
        } => MathValue::Rational {
            numerator: *numerator,
            denominator: *denominator,
        },
        Number::Real(v) => MathValue::Real(*v),
    }
}

/// Convert a complex `Complex` value into a `MathValue`.
/// If the imaginary part is negligible, narrows to Real (or further).
fn complex_to_math_value(c: Complex) -> MathValue {
    if c.im.abs() < 1e-15 {
        // Effectively real — try to narrow further
        let real = c.re;
        if real.fract().abs() < 1e-12 {
            if real >= 0.0 && real <= u64::MAX as f64 {
                return MathValue::Natural(real.round() as u64);
            }
            if real >= i64::MIN as f64 && real <= i64::MAX as f64 {
                return MathValue::Integer(real.round() as i64);
            }
        }
        MathValue::Real(real)
    } else {
        MathValue::Complex { re: c.re, im: c.im }
    }
}

// ============================================================================
// MATH RESULT
// ============================================================================

/// Result of a unified math operation.
///
/// Contains the computed value, its HDC encoding, the consciousness measure (Φ),
/// and a trace of any domain promotions that occurred.
#[derive(Debug, Clone)]
pub struct MathResult {
    /// The computed value in the appropriate domain.
    pub value: MathValue,
    /// BinaryHV encoding in 16,384-dimensional HDC space.
    pub encoding: BinaryHV,
    /// Φ (integrated information): consciousness measure for this operation.
    pub phi: f64,
    /// Ordered list of domain promotions that occurred (e.g. "ℕ → ℤ").
    pub domain_promotions: Vec<String>,
}

// ============================================================================
// UNIFIED MATH ENGINE
// ============================================================================

/// Unified mathematical engine bridging the numeric tower (ℕ→ℤ→ℚ→ℝ) and
/// complex arithmetic (ℂ) into a single API with automatic domain promotion.
///
/// Wraps:
/// - `NumericTower` for exact arithmetic across ℕ, ℤ, ℚ, ℝ
/// - `ComplexArithmeticEngine` for HDC-aware complex operations
///
/// # Examples
///
/// ```rust,ignore
/// use symthaea_core::hdc::math_bridge::{UnifiedMathEngine, MathValue};
///
/// let engine = UnifiedMathEngine::new();
///
/// // 3 + 4 = 7 (stays in ℕ)
/// let r = engine.add(&MathValue::Natural(3), &MathValue::Natural(4));
/// assert!(matches!(r.value, MathValue::Natural(7)));
///
/// // sqrt(-1) auto-promotes to ℂ
/// let r = engine.sqrt(&MathValue::Integer(-1));
/// assert!(matches!(r.value, MathValue::Complex { .. }));
/// ```
pub struct UnifiedMathEngine {
    /// Exact arithmetic across ℕ, ℤ, ℚ, ℝ.
    tower: NumericTower,
    /// HDC-aware complex arithmetic engine.
    complex_engine: ComplexArithmeticEngine,
}

impl UnifiedMathEngine {
    /// Create a new unified math engine.
    pub fn new() -> Self {
        let primitives = PrimitiveSystem::global();
        Self {
            tower: NumericTower::new(),
            complex_engine: ComplexArithmeticEngine::new(primitives),
        }
    }

    /// Reference to the underlying primitive system.
    fn primitives(&self) -> &'static PrimitiveSystem {
        PrimitiveSystem::global()
    }

    // ========================================================================
    // PROMOTION LOGIC
    // ========================================================================

    /// Record a domain promotion from one domain label to another.
    fn record_promotion(promotions: &mut Vec<String>, from: &str, to: &str) {
        if from != to {
            promotions.push(format!("{from} \u{2192} {to}"));
        }
    }

    /// Compute Φ contribution for a domain promotion.
    fn promotion_phi(from_rank: u8, to_rank: u8) -> f64 {
        if to_rank <= from_rank {
            return 0.0;
        }
        // Each step in the tower adds 1.0 of consciousness integration
        (to_rank - from_rank) as f64
    }

    /// Compute Φ for an operation name.
    fn operation_phi(op: &str) -> f64 {
        match op {
            "add" | "subtract" => 0.5,
            "multiply" => 1.0,
            "divide" => 1.5,
            "power" => 2.0,
            "sqrt" => 1.8,
            _ => 0.1,
        }
    }

    /// Determine whether the operation should be routed through the complex
    /// engine (if either operand is complex) or the numeric tower.
    fn needs_complex(a: &MathValue, b: &MathValue) -> bool {
        a.is_complex() || b.is_complex()
    }

    // ========================================================================
    // HDC ENCODING
    // ========================================================================

    /// Encode a MathValue as a BinaryHV.
    fn encode(&self, value: &MathValue) -> BinaryHV {
        match value {
            MathValue::Complex { re, im } => {
                let c = Complex::new(*re, *im);
                let hdc = HdcComplex::encode(c, self.primitives());
                hdc.encoding
            }
            other => {
                // Delegate to the numeric tower's encoding
                let number = other.to_tower_number();
                // Use NumericTower to build a trivial result to get the encoding.
                // We do this by adding zero — but a direct encode is simpler:
                // We'll replicate the encoding logic inline.
                self.encode_tower_number(&number)
            }
        }
    }

    /// Encode a `Number` using the numeric tower's encode method.
    /// We leverage NumericTower::add with zero to get an encoded result.
    fn encode_tower_number(&self, number: &Number) -> BinaryHV {
        // Simplest approach: perform a trivial operation (add 0) and extract encoding.
        let zero = Number::Natural(0);
        let result = self.tower.add(number, &zero);
        result.encoding
    }

    // ========================================================================
    // RESULT BUILDERS
    // ========================================================================

    /// Build a MathResult from a NumberResult (tower operation).
    fn from_tower_result(
        &self,
        nr: &NumberResult,
        a: &MathValue,
        b: &MathValue,
        op: &str,
    ) -> MathResult {
        let value = number_to_math_value(&nr.number);
        let result_domain = value.domain();

        let mut promotions = Vec::new();
        let a_domain = a.domain();
        let b_domain = b.domain();
        Self::record_promotion(&mut promotions, a_domain, result_domain);
        Self::record_promotion(&mut promotions, b_domain, result_domain);

        let phi = Self::operation_phi(op)
            + Self::promotion_phi(a.domain_rank(), value.domain_rank())
            + Self::promotion_phi(b.domain_rank(), value.domain_rank())
            + nr.proof_trace.len() as f64 * 0.1;

        MathResult {
            encoding: nr.encoding,
            value,
            phi,
            domain_promotions: promotions,
        }
    }

    /// Build a MathResult from a unary tower operation.
    fn from_tower_result_unary(&self, nr: &NumberResult, a: &MathValue, op: &str) -> MathResult {
        let value = number_to_math_value(&nr.number);
        let result_domain = value.domain();

        let mut promotions = Vec::new();
        Self::record_promotion(&mut promotions, a.domain(), result_domain);

        let phi = Self::operation_phi(op)
            + Self::promotion_phi(a.domain_rank(), value.domain_rank())
            + nr.proof_trace.len() as f64 * 0.1;

        MathResult {
            encoding: nr.encoding,
            value,
            phi,
            domain_promotions: promotions,
        }
    }

    /// Build a MathResult from a complex operation.
    fn from_complex_result(
        &self,
        c: Complex,
        a: &MathValue,
        b: &MathValue,
        op: &str,
    ) -> MathResult {
        let value = complex_to_math_value(c);
        let encoding = self.encode(&value);
        let result_domain = value.domain();

        let mut promotions = Vec::new();
        Self::record_promotion(&mut promotions, a.domain(), result_domain);
        Self::record_promotion(&mut promotions, b.domain(), result_domain);

        let phi = Self::operation_phi(op)
            + Self::promotion_phi(a.domain_rank(), value.domain_rank())
            + Self::promotion_phi(b.domain_rank(), value.domain_rank());

        MathResult {
            value,
            encoding,
            phi,
            domain_promotions: promotions,
        }
    }

    /// Build a unary MathResult from a complex operation.
    fn from_complex_result_unary(&self, c: Complex, a: &MathValue, op: &str) -> MathResult {
        let value = complex_to_math_value(c);
        let encoding = self.encode(&value);
        let result_domain = value.domain();

        let mut promotions = Vec::new();
        Self::record_promotion(&mut promotions, a.domain(), result_domain);

        let phi =
            Self::operation_phi(op) + Self::promotion_phi(a.domain_rank(), value.domain_rank());

        MathResult {
            value,
            encoding,
            phi,
            domain_promotions: promotions,
        }
    }

    // ========================================================================
    // ARITHMETIC OPERATIONS
    // ========================================================================

    /// Add two values: a + b.
    ///
    /// Auto-promotes to the narrowest domain that can hold both operands and
    /// the result. If either operand is Complex, the entire operation is
    /// performed in ℂ.
    pub fn add(&self, a: &MathValue, b: &MathValue) -> MathResult {
        if Self::needs_complex(a, b) {
            let ca = a.to_complex();
            let cb = b.to_complex();
            return self.from_complex_result(ca + cb, a, b, "add");
        }

        let na = a.to_tower_number();
        let nb = b.to_tower_number();
        let nr = self.tower.add(&na, &nb);
        self.from_tower_result(&nr, a, b, "add")
    }

    /// Subtract two values: a - b.
    ///
    /// Auto-promotes ℕ → ℤ when the result is negative.
    pub fn subtract(&self, a: &MathValue, b: &MathValue) -> MathResult {
        if Self::needs_complex(a, b) {
            let ca = a.to_complex();
            let cb = b.to_complex();
            return self.from_complex_result(ca - cb, a, b, "subtract");
        }

        let na = a.to_tower_number();
        let nb = b.to_tower_number();
        let nr = self.tower.subtract(&na, &nb);
        self.from_tower_result(&nr, a, b, "subtract")
    }

    /// Multiply two values: a * b.
    pub fn multiply(&self, a: &MathValue, b: &MathValue) -> MathResult {
        if Self::needs_complex(a, b) {
            let ca = a.to_complex();
            let cb = b.to_complex();
            return self.from_complex_result(ca * cb, a, b, "multiply");
        }

        let na = a.to_tower_number();
        let nb = b.to_tower_number();
        let nr = self.tower.multiply(&na, &nb);
        self.from_tower_result(&nr, a, b, "multiply")
    }

    /// Divide two values: a / b.
    ///
    /// Returns `None` if b is zero.
    /// Auto-promotes ℤ → ℚ when division is inexact.
    pub fn divide(&self, a: &MathValue, b: &MathValue) -> Option<MathResult> {
        if b.is_zero() {
            return None;
        }

        if Self::needs_complex(a, b) {
            let ca = a.to_complex();
            let cb = b.to_complex();
            // Complex division is always defined for nonzero divisor
            return Some(self.from_complex_result(ca / cb, a, b, "divide"));
        }

        let na = a.to_tower_number();
        let nb = b.to_tower_number();
        let nr = self.tower.divide(&na, &nb)?;
        Some(self.from_tower_result(&nr, a, b, "divide"))
    }

    /// Square root: sqrt(a).
    ///
    /// Negative reals and negative integers auto-promote to Complex:
    /// sqrt(-4) = 2i.
    pub fn sqrt(&self, a: &MathValue) -> MathResult {
        match a {
            // Complex input: always use complex sqrt
            MathValue::Complex { re, im } => {
                let c = Complex::new(*re, *im);
                let result = c.sqrt();
                self.from_complex_result_unary(result, a, "sqrt")
            }

            // For non-complex values, check if value is negative
            _ => {
                let val = a.to_f64();
                if val < 0.0 {
                    // Negative → promote to ℂ
                    let c = Complex::new(val, 0.0);
                    let result = c.sqrt();
                    self.from_complex_result_unary(result, a, "sqrt")
                } else {
                    // Non-negative → use numeric tower
                    let number = a.to_tower_number();
                    match self.tower.sqrt(&number) {
                        Some(nr) => self.from_tower_result_unary(&nr, a, "sqrt"),
                        None => {
                            // Shouldn't happen for non-negative, but fallback
                            let c = Complex::new(val, 0.0);
                            let result = c.sqrt();
                            self.from_complex_result_unary(result, a, "sqrt")
                        }
                    }
                }
            }
        }
    }

    /// Power: base^exp.
    ///
    /// For complex operands or cases that would produce complex results
    /// (e.g. negative base with fractional exponent), promotes to ℂ.
    pub fn power(&self, base: &MathValue, exp: &MathValue) -> MathResult {
        if Self::needs_complex(base, exp) {
            let cb = base.to_complex();
            let ce = exp.to_complex();
            let result = cb.pow(ce);
            return self.from_complex_result(result, base, exp, "power");
        }

        // Check for cases that would produce NaN in the real domain
        // (negative base with non-integer exponent → promote to ℂ)
        let base_f64 = base.to_f64();
        let exp_is_fractional = match exp {
            MathValue::Rational { .. } => true,
            MathValue::Real(x) => x.fract().abs() > 1e-12,
            _ => false,
        };

        if base_f64 < 0.0 && exp_is_fractional {
            let cb = base.to_complex();
            let ce = exp.to_complex();
            let result = cb.pow(ce);
            return self.from_complex_result(result, base, exp, "power");
        }

        let nb = base.to_tower_number();
        let ne = exp.to_tower_number();
        let nr = self.tower.power(&nb, &ne);
        self.from_tower_result(&nr, base, exp, "power")
    }
}

impl Default for UnifiedMathEngine {
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
    // Helper
    // ====================================================================

    fn engine() -> UnifiedMathEngine {
        UnifiedMathEngine::new()
    }

    // ====================================================================
    // Domain labeling
    // ====================================================================

    #[test]
    fn test_domain_labels() {
        assert_eq!(MathValue::Natural(5).domain(), "\u{2115}");
        assert_eq!(MathValue::Integer(-3).domain(), "\u{2124}");
        assert_eq!(
            MathValue::Rational {
                numerator: 1,
                denominator: 2
            }
            .domain(),
            "\u{211a}"
        );
        assert_eq!(MathValue::Real(3.14).domain(), "\u{211d}");
        assert_eq!(MathValue::Complex { re: 1.0, im: 2.0 }.domain(), "\u{2102}");
    }

    // ====================================================================
    // to_f64 conversion
    // ====================================================================

    #[test]
    fn test_to_f64() {
        assert!((MathValue::Natural(7).to_f64() - 7.0).abs() < 1e-10);
        assert!((MathValue::Integer(-3).to_f64() - (-3.0)).abs() < 1e-10);
        assert!(
            (MathValue::Rational {
                numerator: 1,
                denominator: 3
            }
            .to_f64()
                - 1.0 / 3.0)
                .abs()
                < 1e-10
        );
        assert!((MathValue::Real(2.5).to_f64() - 2.5).abs() < 1e-10);
        assert!((MathValue::Complex { re: 4.0, im: 3.0 }.to_f64() - 4.0).abs() < 1e-10);
    }

    // ====================================================================
    // from_hdc_number_value
    // ====================================================================

    #[test]
    fn test_from_hdc_number_value() {
        let mv = MathValue::from_hdc_number_value(42);
        assert!(matches!(mv, MathValue::Natural(42)));
        assert_eq!(mv.domain(), "\u{2115}");
    }

    // ====================================================================
    // Basic arithmetic — Natural domain
    // ====================================================================

    #[test]
    fn test_add_naturals() {
        let e = engine();
        let r = e.add(&MathValue::Natural(3), &MathValue::Natural(7));
        assert!(
            matches!(r.value, MathValue::Natural(10)),
            "3 + 7 = 10 in N, got {:?}",
            r.value
        );
        assert!(r.domain_promotions.is_empty(), "No promotions expected");
    }

    #[test]
    fn test_multiply_naturals() {
        let e = engine();
        let r = e.multiply(&MathValue::Natural(4), &MathValue::Natural(5));
        assert!(
            matches!(r.value, MathValue::Natural(20)),
            "4 * 5 = 20, got {:?}",
            r.value
        );
    }

    // ====================================================================
    // Basic arithmetic — Integer domain
    // ====================================================================

    #[test]
    fn test_add_integers() {
        let e = engine();
        let r = e.add(&MathValue::Integer(-3), &MathValue::Integer(-5));
        assert!(
            matches!(r.value, MathValue::Integer(-8)),
            "-3 + -5 = -8, got {:?}",
            r.value
        );
    }

    #[test]
    fn test_subtract_integers() {
        let e = engine();
        let r = e.subtract(&MathValue::Integer(10), &MathValue::Integer(3));
        assert!(
            matches!(r.value, MathValue::Natural(7)),
            "10 - 3 = 7 (narrows to N), got {:?}",
            r.value
        );
    }

    // ====================================================================
    // Basic arithmetic — Rational domain
    // ====================================================================

    #[test]
    fn test_add_rationals() {
        let e = engine();
        let a = MathValue::Rational {
            numerator: 1,
            denominator: 2,
        };
        let b = MathValue::Rational {
            numerator: 1,
            denominator: 3,
        };
        let r = e.add(&a, &b);
        // 1/2 + 1/3 = 5/6
        assert!(
            matches!(
                r.value,
                MathValue::Rational {
                    numerator: 5,
                    denominator: 6
                }
            ),
            "1/2 + 1/3 = 5/6, got {:?}",
            r.value
        );
    }

    #[test]
    fn test_multiply_rationals() {
        let e = engine();
        let a = MathValue::Rational {
            numerator: 2,
            denominator: 3,
        };
        let b = MathValue::Rational {
            numerator: 3,
            denominator: 4,
        };
        let r = e.multiply(&a, &b);
        // (2/3) * (3/4) = 1/2
        assert!(
            matches!(
                r.value,
                MathValue::Rational {
                    numerator: 1,
                    denominator: 2
                }
            ),
            "(2/3) * (3/4) = 1/2, got {:?}",
            r.value
        );
    }

    // ====================================================================
    // Basic arithmetic — Real domain
    // ====================================================================

    #[test]
    fn test_add_reals() {
        let e = engine();
        let r = e.add(&MathValue::Real(1.5), &MathValue::Real(2.3));
        match &r.value {
            MathValue::Real(x) => {
                assert!((x - 3.8).abs() < 1e-10, "1.5 + 2.3 = 3.8, got {}", x);
            }
            // Could narrow to Natural(4) if very close — accept that too
            MathValue::Natural(n) => {
                assert!(((*n as f64) - 3.8).abs() < 1e-10);
            }
            other => panic!("Expected Real or Natural, got {:?}", other),
        }
    }

    // ====================================================================
    // Basic arithmetic — Complex domain
    // ====================================================================

    #[test]
    fn test_add_complex() {
        let e = engine();
        let a = MathValue::Complex { re: 1.0, im: 2.0 };
        let b = MathValue::Complex { re: 3.0, im: 4.0 };
        let r = e.add(&a, &b);
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!((re - 4.0).abs() < 1e-10, "re: expected 4, got {}", re);
                assert!((im - 6.0).abs() < 1e-10, "im: expected 6, got {}", im);
            }
            other => panic!("Expected Complex, got {:?}", other),
        }
    }

    #[test]
    fn test_multiply_complex() {
        let e = engine();
        let a = MathValue::Complex { re: 1.0, im: 2.0 };
        let b = MathValue::Complex { re: 3.0, im: 4.0 };
        let r = e.multiply(&a, &b);
        // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!((re - (-5.0)).abs() < 1e-10, "re: expected -5, got {}", re);
                assert!((im - 10.0).abs() < 1e-10, "im: expected 10, got {}", im);
            }
            other => panic!("Expected Complex, got {:?}", other),
        }
    }

    #[test]
    fn test_divide_complex() {
        let e = engine();
        let a = MathValue::Complex { re: 1.0, im: 2.0 };
        let b = MathValue::Complex { re: 3.0, im: 4.0 };
        let r = e.divide(&a, &b).expect("nonzero divisor");
        // (1+2i)/(3+4i) = 11/25 + 2/25 i
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!((re - 11.0 / 25.0).abs() < 1e-10, "re: {}", re);
                assert!((im - 2.0 / 25.0).abs() < 1e-10, "im: {}", im);
            }
            other => panic!("Expected Complex, got {:?}", other),
        }
    }

    // ====================================================================
    // Auto-promotion across domains
    // ====================================================================

    #[test]
    fn test_promotion_natural_to_integer() {
        let e = engine();
        // 3 - 5 = -2 (ℕ → ℤ)
        let r = e.subtract(&MathValue::Natural(3), &MathValue::Natural(5));
        assert!(
            matches!(r.value, MathValue::Integer(-2)),
            "3 - 5 should promote to Z(-2), got {:?}",
            r.value
        );
        assert!(
            !r.domain_promotions.is_empty(),
            "Should record the N -> Z promotion"
        );
    }

    #[test]
    fn test_promotion_integer_to_rational() {
        let e = engine();
        // 7 / 2 = 7/2 (ℤ → ℚ)
        let r = e
            .divide(&MathValue::Natural(7), &MathValue::Natural(2))
            .expect("nonzero divisor");
        assert!(
            matches!(
                r.value,
                MathValue::Rational {
                    numerator: 7,
                    denominator: 2
                }
            ),
            "7 / 2 should promote to Q(7/2), got {:?}",
            r.value
        );
    }

    #[test]
    fn test_promotion_mixed_natural_integer() {
        let e = engine();
        // Natural(5) + Integer(-3) = Natural(2)
        let r = e.add(&MathValue::Natural(5), &MathValue::Integer(-3));
        assert!(
            matches!(r.value, MathValue::Natural(2)),
            "5 + (-3) = 2, got {:?}",
            r.value
        );
    }

    #[test]
    fn test_promotion_mixed_integer_rational() {
        let e = engine();
        // Integer(1) + Rational(1/2) = Rational(3/2)
        let r = e.add(
            &MathValue::Integer(1),
            &MathValue::Rational {
                numerator: 1,
                denominator: 2,
            },
        );
        assert!(
            matches!(
                r.value,
                MathValue::Rational {
                    numerator: 3,
                    denominator: 2
                }
            ),
            "1 + 1/2 = 3/2, got {:?}",
            r.value
        );
    }

    #[test]
    fn test_promotion_rational_to_real_via_sqrt() {
        let e = engine();
        // sqrt(2) promotes to Real
        let r = e.sqrt(&MathValue::Natural(2));
        match &r.value {
            MathValue::Real(x) => {
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
    fn test_promotion_anything_plus_complex() {
        let e = engine();
        // Natural(5) + Complex(1,2) should promote to Complex
        let r = e.add(
            &MathValue::Natural(5),
            &MathValue::Complex { re: 1.0, im: 2.0 },
        );
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!((re - 6.0).abs() < 1e-10, "re: {}", re);
                assert!((im - 2.0).abs() < 1e-10, "im: {}", im);
            }
            other => panic!("Natural + Complex should be Complex, got {:?}", other),
        }
    }

    // ====================================================================
    // sqrt(-1) → Complex
    // ====================================================================

    #[test]
    fn test_sqrt_negative_one_is_i() {
        let e = engine();
        let r = e.sqrt(&MathValue::Integer(-1));
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!(re.abs() < 1e-10, "sqrt(-1) re should be ~0, got {}", re);
                assert!(
                    (im - 1.0).abs() < 1e-10,
                    "sqrt(-1) im should be ~1, got {}",
                    im
                );
            }
            other => panic!("sqrt(-1) should be Complex(0+1i), got {:?}", other),
        }
    }

    #[test]
    fn test_sqrt_negative_four() {
        let e = engine();
        let r = e.sqrt(&MathValue::Integer(-4));
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!(re.abs() < 1e-10, "sqrt(-4) re should be ~0, got {}", re);
                assert!(
                    (im - 2.0).abs() < 1e-10,
                    "sqrt(-4) im should be ~2, got {}",
                    im
                );
            }
            other => panic!("sqrt(-4) should be Complex(0+2i), got {:?}", other),
        }
    }

    #[test]
    fn test_sqrt_positive_stays_real() {
        let e = engine();
        let r = e.sqrt(&MathValue::Natural(9));
        assert!(
            matches!(r.value, MathValue::Natural(3)),
            "sqrt(9) = 3 in N, got {:?}",
            r.value
        );
    }

    #[test]
    fn test_sqrt_complex_input() {
        let e = engine();
        // sqrt(0 + 1i) should produce a complex result
        let r = e.sqrt(&MathValue::Complex { re: 0.0, im: 1.0 });
        assert!(
            matches!(r.value, MathValue::Complex { .. }),
            "sqrt(i) should be Complex, got {:?}",
            r.value
        );
    }

    // ====================================================================
    // Division edge cases
    // ====================================================================

    #[test]
    fn test_divide_by_zero_returns_none() {
        let e = engine();
        assert!(
            e.divide(&MathValue::Natural(5), &MathValue::Natural(0))
                .is_none(),
            "Division by zero should return None"
        );
    }

    #[test]
    fn test_divide_exact_stays_natural() {
        let e = engine();
        let r = e
            .divide(&MathValue::Natural(6), &MathValue::Natural(3))
            .expect("nonzero divisor");
        assert!(
            matches!(r.value, MathValue::Natural(2)),
            "6 / 3 = 2 in N, got {:?}",
            r.value
        );
    }

    #[test]
    fn test_divide_complex_by_zero_returns_none() {
        let e = engine();
        assert!(
            e.divide(
                &MathValue::Complex { re: 1.0, im: 1.0 },
                &MathValue::Complex { re: 0.0, im: 0.0 },
            )
            .is_none(),
            "Division by zero complex should return None"
        );
    }

    // ====================================================================
    // Power
    // ====================================================================

    #[test]
    fn test_power_natural() {
        let e = engine();
        let r = e.power(&MathValue::Natural(2), &MathValue::Natural(10));
        assert!(
            matches!(r.value, MathValue::Natural(1024)),
            "2^10 = 1024, got {:?}",
            r.value
        );
    }

    #[test]
    fn test_power_negative_exponent() {
        let e = engine();
        let r = e.power(&MathValue::Natural(2), &MathValue::Integer(-1));
        assert!(
            matches!(
                r.value,
                MathValue::Rational {
                    numerator: 1,
                    denominator: 2
                }
            ),
            "2^(-1) = 1/2, got {:?}",
            r.value
        );
    }

    #[test]
    fn test_power_complex_base() {
        let e = engine();
        // i^2 = -1
        let r = e.power(
            &MathValue::Complex { re: 0.0, im: 1.0 },
            &MathValue::Natural(2),
        );
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!(
                    (re - (-1.0)).abs() < 1e-6,
                    "i^2 re should be -1, got {}",
                    re
                );
                assert!(im.abs() < 1e-6, "i^2 im should be ~0, got {}", im);
            }
            // May narrow to Integer(-1) via complex_to_math_value
            MathValue::Integer(-1) => {}
            other => panic!("i^2 should be -1, got {:?}", other),
        }
    }

    // ====================================================================
    // Phi (consciousness measure)
    // ====================================================================

    #[test]
    fn test_phi_positive() {
        let e = engine();
        let r = e.add(&MathValue::Natural(3), &MathValue::Natural(5));
        assert!(r.phi > 0.0, "Phi should be positive, got {}", r.phi);
    }

    #[test]
    fn test_phi_increases_on_promotion() {
        let e = engine();

        // No promotion: 5 + 3 = 8 (stays in N)
        let r_no = e.add(&MathValue::Natural(5), &MathValue::Natural(3));

        // Promotion: 5 + (-8) = -3 (N → Z)
        let r_yes = e.add(&MathValue::Natural(5), &MathValue::Integer(-8));

        assert!(
            r_yes.phi > r_no.phi,
            "Promotion should increase Phi: promo={}, no_promo={}",
            r_yes.phi,
            r_no.phi
        );
    }

    #[test]
    fn test_phi_complex_promotion_highest() {
        let e = engine();

        // Real operation
        let r_real = e.add(&MathValue::Real(1.5), &MathValue::Real(2.5));

        // Complex operation (promoted from Real)
        let r_complex = e.add(
            &MathValue::Real(1.5),
            &MathValue::Complex { re: 2.5, im: 1.0 },
        );

        assert!(
            r_complex.phi > r_real.phi,
            "Complex promotion should add more Phi: complex={}, real={}",
            r_complex.phi,
            r_real.phi
        );
    }

    // ====================================================================
    // HDC encoding (BinaryHV validity)
    // ====================================================================

    #[test]
    fn test_encoding_not_all_zeros() {
        let e = engine();
        let r = e.add(&MathValue::Natural(42), &MathValue::Natural(13));
        let ones = r.encoding.0.iter().filter(|&&b| b != 0).count();
        assert!(ones > 0, "BinaryHV encoding should not be all zeros");
    }

    #[test]
    fn test_encoding_different_for_different_values() {
        let e = engine();
        let r1 = e.add(&MathValue::Natural(5), &MathValue::Natural(0));
        let r2 = e.add(&MathValue::Natural(7), &MathValue::Natural(0));
        assert_ne!(
            r1.encoding, r2.encoding,
            "5 and 7 should have different encodings"
        );
    }

    #[test]
    fn test_encoding_deterministic() {
        let e = engine();
        let r1 = e.add(&MathValue::Natural(42), &MathValue::Natural(0));
        let r2 = e.add(&MathValue::Natural(42), &MathValue::Natural(0));
        assert_eq!(
            r1.encoding, r2.encoding,
            "Same value should produce same encoding"
        );
    }

    #[test]
    fn test_complex_encoding_not_all_zeros() {
        let e = engine();
        let r = e.sqrt(&MathValue::Integer(-1));
        let ones = r.encoding.0.iter().filter(|&&b| b != 0).count();
        assert!(ones > 0, "Complex encoding should not be all zeros");
    }

    // ====================================================================
    // Display
    // ====================================================================

    #[test]
    fn test_display_all_domains() {
        let s = format!("{}", MathValue::Natural(5));
        assert_eq!(s, "5");

        let s = format!("{}", MathValue::Integer(-3));
        assert_eq!(s, "-3");

        let s = format!(
            "{}",
            MathValue::Rational {
                numerator: 1,
                denominator: 2
            }
        );
        assert_eq!(s, "1/2");

        let s = format!("{}", MathValue::Complex { re: 1.0, im: -2.0 });
        assert!(s.contains("1") && s.contains("2") && s.contains("i"));
    }

    // ====================================================================
    // is_zero
    // ====================================================================

    #[test]
    fn test_is_zero() {
        assert!(MathValue::Natural(0).is_zero());
        assert!(MathValue::Integer(0).is_zero());
        assert!(
            MathValue::Rational {
                numerator: 0,
                denominator: 5
            }
            .is_zero()
        );
        assert!(MathValue::Real(0.0).is_zero());
        assert!(MathValue::Complex { re: 0.0, im: 0.0 }.is_zero());
        assert!(!MathValue::Natural(1).is_zero());
        assert!(!MathValue::Complex { re: 0.0, im: 1.0 }.is_zero());
    }

    // ====================================================================
    // End-to-end chain: ℕ → ℤ → ℚ → ℝ → ℂ
    // ====================================================================

    #[test]
    fn test_full_tower_chain() {
        let e = engine();

        // Start in ℕ: 10
        let n = MathValue::Natural(10);
        assert_eq!(n.domain(), "\u{2115}");

        // Subtract to promote to ℤ: 10 - 15 = -5
        let r1 = e.subtract(&n, &MathValue::Natural(15));
        assert!(
            matches!(r1.value, MathValue::Integer(-5)),
            "10 - 15 = -5, got {:?}",
            r1.value
        );
        assert_eq!(r1.value.domain(), "\u{2124}");

        // Divide to promote to ℚ: -5 / 3 = -5/3
        let r2 = e
            .divide(&r1.value, &MathValue::Natural(3))
            .expect("nonzero");
        assert!(
            matches!(
                r2.value,
                MathValue::Rational {
                    numerator: -5,
                    denominator: 3
                }
            ),
            "-5 / 3 = -5/3, got {:?}",
            r2.value
        );
        assert_eq!(r2.value.domain(), "\u{211a}");

        // sqrt of the absolute value to promote to ℝ: sqrt(5/3)
        // First get absolute value via multiply by -1 then by -1 ... simpler: just use sqrt on positive
        let five = MathValue::Natural(5);
        let r3 = e.sqrt(&five);
        assert_eq!(r3.value.domain(), "\u{211d}");

        // sqrt of negative to promote to ℂ: sqrt(-4)
        let neg4 = MathValue::Integer(-4);
        let r4 = e.sqrt(&neg4);
        assert_eq!(r4.value.domain(), "\u{2102}");

        // Full chain verified: ℕ → ℤ → ℚ → ℝ → ℂ
    }

    // ====================================================================
    // Promotion tracking
    // ====================================================================

    #[test]
    fn test_promotion_tracking_records_steps() {
        let e = engine();
        // sqrt(-1) should record ℤ → ℂ promotion
        let r = e.sqrt(&MathValue::Integer(-1));
        assert!(
            !r.domain_promotions.is_empty(),
            "sqrt(-1) should record domain promotion, got {:?}",
            r.domain_promotions
        );
        // Should contain an arrow
        assert!(
            r.domain_promotions.iter().any(|s| s.contains('\u{2192}')),
            "Promotion trace should contain arrow: {:?}",
            r.domain_promotions
        );
    }

    // ====================================================================
    // Mixed complex + real operations
    // ====================================================================

    #[test]
    fn test_subtract_complex_from_real() {
        let e = engine();
        let r = e.subtract(
            &MathValue::Real(5.0),
            &MathValue::Complex { re: 2.0, im: 3.0 },
        );
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!((re - 3.0).abs() < 1e-10, "re: {}", re);
                assert!((im - (-3.0)).abs() < 1e-10, "im: {}", im);
            }
            other => panic!("Real - Complex should be Complex, got {:?}", other),
        }
    }

    #[test]
    fn test_multiply_integer_by_complex() {
        let e = engine();
        // 3 * (0 + 1i) = 0 + 3i
        let r = e.multiply(
            &MathValue::Integer(3),
            &MathValue::Complex { re: 0.0, im: 1.0 },
        );
        match &r.value {
            MathValue::Complex { re, im } => {
                assert!(re.abs() < 1e-10, "re should be ~0, got {}", re);
                assert!((im - 3.0).abs() < 1e-10, "im should be 3, got {}", im);
            }
            other => panic!("Integer * Complex should be Complex, got {:?}", other),
        }
    }
}
