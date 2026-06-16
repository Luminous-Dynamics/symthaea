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
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Normalize a rational number: ensure denominator > 0 and coprime with |numerator|.
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

    /// Check if this number is zero in any domain.
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
            Number::Real(x) if x.fract().abs() < 1e-12 && x >= 0.0 && x <= u64::MAX as f64 => {
                Number::Natural(x.round() as u64)
            }
            Number::Real(x)
                if x.fract().abs() < 1e-12 && x >= i64::MIN as f64 && x <= i64::MAX as f64 =>
            {
                Number::Integer(x.round() as i64)
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
        if x.is_finite() && x.fract().abs() < 1e-12 {
            if x >= 0.0 && x <= u64::MAX as f64 {
                return Number::Natural(x.round() as u64);
            }
            if x >= i64::MIN as f64 && x <= i64::MAX as f64 {
                return Number::Integer(x.round() as i64);
            }
        }
        Number::Real(x)
    }

    /// Create a Number from a rational p/q (normalizes and narrows).
    pub fn from_rational(numerator: i64, denominator: i64) -> Number {
        assert!(denominator != 0, "Denominator cannot be zero");
        let (num, den) = normalize_rational(numerator, denominator);
        if den == 1 {
            if num >= 0 {
                Number::Natural(num as u64)
            } else {
                Number::Integer(num)
            }
        } else {
            Number::Rational {
                numerator: num,
                denominator: den,
            }
        }
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

    // ========================================================================
    // ARITHMETIC OPERATIONS
    // ========================================================================

    /// Add two numbers: a + b
    ///
    /// Stays in the narrowest domain that contains both operands and the result.
    pub fn add(&self, a: &Number, b: &Number) -> NumberResult {
        let mut proof_trace = vec![format!("Add: {} + {}", a, b)];

        let result = match (a, b) {
            // ℕ + ℕ → ℕ (always closed)
            (Number::Natural(x), Number::Natural(y)) => {
                let sum = x.wrapping_add(*y);
                proof_trace.push(format!("Natural addition: {x} + {y} = {sum}"));
                Number::Natural(sum)
            }

            // ℤ + ℤ → ℤ (or ℕ if non-negative)
            (Number::Integer(x), Number::Integer(y)) => {
                let sum = x.wrapping_add(*y);
                proof_trace.push(format!("Integer addition: {x} + {y} = {sum}"));
                Self::from_i64(sum)
            }

            // Mixed ℕ/ℤ: lift to ℤ
            (Number::Natural(x), Number::Integer(y)) | (Number::Integer(y), Number::Natural(x)) => {
                let sum = (*x as i64).wrapping_add(*y);
                proof_trace.push(format!("Lift to Z: {x} + {y} = {sum}"));
                Self::from_i64(sum)
            }

            // ℚ + ℚ → ℚ: (a/b) + (c/d) = (ad + bc) / bd
            (
                Number::Rational {
                    numerator: an,
                    denominator: ad,
                },
                Number::Rational {
                    numerator: bn,
                    denominator: bd,
                },
            ) => {
                let num = an.wrapping_mul(*bd).wrapping_add(bn.wrapping_mul(*ad));
                let den = ad.wrapping_mul(*bd);
                proof_trace.push(format!(
                    "Rational addition: ({an} x {bd} + {bn} x {ad}) / ({ad} x {bd}) = {num}/{den}"
                ));
                Self::from_rational(num, den)
            }

            // Mixed with ℚ: lift other operand to ℚ
            (
                Number::Rational {
                    numerator: rn,
                    denominator: rd,
                },
                other,
            )
            | (
                other,
                Number::Rational {
                    numerator: rn,
                    denominator: rd,
                },
            ) => {
                let (on, od) = self.to_rational(other);
                let num = rn.wrapping_mul(od).wrapping_add(on.wrapping_mul(*rd));
                let den = rd.wrapping_mul(od);
                proof_trace.push(format!("Lift to Q then add: {rn}/{rd} + {on}/{od}"));
                Self::from_rational(num, den)
            }

            // ℝ + anything → ℝ
            (Number::Real(x), other) | (other, Number::Real(x)) => {
                let sum = x + other.to_f64();
                proof_trace.push(format!(
                    "Real addition: {} + {} = {}",
                    x,
                    other.to_f64(),
                    sum
                ));
                Self::from_f64(sum)
            }
        };

        let input_domains = [a.domain(), b.domain()];
        self.build_result(result, "add".to_string(), proof_trace, &input_domains)
    }

    /// Subtract two numbers: a - b
    ///
    /// Auto-promotes ℕ → ℤ when the result is negative.
    pub fn subtract(&self, a: &Number, b: &Number) -> NumberResult {
        let mut proof_trace = vec![format!("Subtract: {} - {}", a, b)];

        let result = match (a, b) {
            // ℕ - ℕ: may promote to ℤ
            (Number::Natural(x), Number::Natural(y)) => {
                if *x >= *y {
                    let diff = x - y;
                    proof_trace.push(format!(
                        "Natural subtraction: {x} - {y} = {diff} (stays in N)"
                    ));
                    Number::Natural(diff)
                } else {
                    let diff = (*x as i64).wrapping_sub(*y as i64);
                    proof_trace.push(format!(
                        "Promotion N -> Z: {x} - {y} = {diff} (negative result)"
                    ));
                    Number::Integer(diff)
                }
            }

            // ℤ - ℤ → ℤ (or ℕ)
            (Number::Integer(x), Number::Integer(y)) => {
                let diff = x.wrapping_sub(*y);
                proof_trace.push(format!("Integer subtraction: {x} - {y} = {diff}"));
                Self::from_i64(diff)
            }

            // Mixed ℕ/ℤ
            (Number::Natural(x), Number::Integer(y)) => {
                let diff = (*x as i64).wrapping_sub(*y);
                proof_trace.push(format!("Lift to Z: {x} - {y} = {diff}"));
                Self::from_i64(diff)
            }
            (Number::Integer(x), Number::Natural(y)) => {
                let diff = x.wrapping_sub(*y as i64);
                proof_trace.push(format!("Lift to Z: {x} - {y} = {diff}"));
                Self::from_i64(diff)
            }

            // ℚ - ℚ
            (
                Number::Rational {
                    numerator: an,
                    denominator: ad,
                },
                Number::Rational {
                    numerator: bn,
                    denominator: bd,
                },
            ) => {
                let num = an.wrapping_mul(*bd).wrapping_sub(bn.wrapping_mul(*ad));
                let den = ad.wrapping_mul(*bd);
                proof_trace.push(format!(
                    "Rational subtraction: ({an} x {bd} - {bn} x {ad}) / ({ad} x {bd})"
                ));
                Self::from_rational(num, den)
            }

            // Mixed with ℚ
            (
                Number::Rational {
                    numerator: rn,
                    denominator: rd,
                },
                other,
            ) => {
                let (on, od) = self.to_rational(other);
                let num = rn.wrapping_mul(od).wrapping_sub(on.wrapping_mul(*rd));
                let den = rd.wrapping_mul(od);
                proof_trace.push(format!("Lift to Q: {rn}/{rd} - {on}/{od}"));
                Self::from_rational(num, den)
            }
            (
                other,
                Number::Rational {
                    numerator: rn,
                    denominator: rd,
                },
            ) => {
                let (on, od) = self.to_rational(other);
                let num = on.wrapping_mul(*rd).wrapping_sub(rn.wrapping_mul(od));
                let den = od.wrapping_mul(*rd);
                proof_trace.push(format!("Lift to Q: {on}/{od} - {rn}/{rd}"));
                Self::from_rational(num, den)
            }

            // ℝ - anything → ℝ
            (Number::Real(x), other) => {
                let diff = x - other.to_f64();
                proof_trace.push(format!(
                    "Real subtraction: {} - {} = {}",
                    x,
                    other.to_f64(),
                    diff
                ));
                Self::from_f64(diff)
            }
            (other, Number::Real(x)) => {
                let diff = other.to_f64() - x;
                proof_trace.push(format!(
                    "Real subtraction: {} - {} = {}",
                    other.to_f64(),
                    x,
                    diff
                ));
                Self::from_f64(diff)
            }
        };

        let input_domains = [a.domain(), b.domain()];
        self.build_result(result, "subtract".to_string(), proof_trace, &input_domains)
    }

    /// Multiply two numbers: a * b
    ///
    /// Stays in the narrowest closed domain.
    pub fn multiply(&self, a: &Number, b: &Number) -> NumberResult {
        let mut proof_trace = vec![format!("Multiply: {} * {}", a, b)];

        let result = match (a, b) {
            // ℕ * ℕ → ℕ
            (Number::Natural(x), Number::Natural(y)) => {
                let prod = x.wrapping_mul(*y);
                proof_trace.push(format!("Natural multiplication: {x} * {y} = {prod}"));
                Number::Natural(prod)
            }

            // ℤ * ℤ → ℤ (or ℕ)
            (Number::Integer(x), Number::Integer(y)) => {
                let prod = x.wrapping_mul(*y);
                proof_trace.push(format!("Integer multiplication: {x} * {y} = {prod}"));
                Self::from_i64(prod)
            }

            // Mixed ℕ/ℤ
            (Number::Natural(x), Number::Integer(y)) | (Number::Integer(y), Number::Natural(x)) => {
                let prod = (*x as i64).wrapping_mul(*y);
                proof_trace.push(format!("Lift to Z: {x} * {y} = {prod}"));
                Self::from_i64(prod)
            }

            // ℚ * ℚ → ℚ
            (
                Number::Rational {
                    numerator: an,
                    denominator: ad,
                },
                Number::Rational {
                    numerator: bn,
                    denominator: bd,
                },
            ) => {
                let num = an.wrapping_mul(*bn);
                let den = ad.wrapping_mul(*bd);
                proof_trace.push(format!(
                    "Rational multiplication: ({an} * {bn}) / ({ad} * {bd})"
                ));
                Self::from_rational(num, den)
            }

            // Mixed with ℚ
            (
                Number::Rational {
                    numerator: rn,
                    denominator: rd,
                },
                other,
            )
            | (
                other,
                Number::Rational {
                    numerator: rn,
                    denominator: rd,
                },
            ) => {
                let (on, od) = self.to_rational(other);
                let num = rn.wrapping_mul(on);
                let den = rd.wrapping_mul(od);
                proof_trace.push(format!("Lift to Q: ({rn}/{rd}) * ({on}/{od})"));
                Self::from_rational(num, den)
            }

            // ℝ * anything → ℝ
            (Number::Real(x), other) | (other, Number::Real(x)) => {
                let prod = x * other.to_f64();
                proof_trace.push(format!(
                    "Real multiplication: {} * {} = {}",
                    x,
                    other.to_f64(),
                    prod
                ));
                Self::from_f64(prod)
            }
        };

        let input_domains = [a.domain(), b.domain()];
        self.build_result(result, "multiply".to_string(), proof_trace, &input_domains)
    }

    /// Divide two numbers: a / b
    ///
    /// Auto-promotes ℤ → ℚ when division is not exact.
    /// Returns `None` if b is zero.
    pub fn divide(&self, a: &Number, b: &Number) -> Option<NumberResult> {
        if b.is_zero() {
            return None;
        }

        let mut proof_trace = vec![format!("Divide: {} / {}", a, b)];

        let result = match (a, b) {
            // ℕ / ℕ: exact → ℕ, otherwise → ℚ
            (Number::Natural(x), Number::Natural(y)) => {
                if *y != 0 && *x % *y == 0 {
                    let quot = x / y;
                    proof_trace.push(format!(
                        "Exact natural division: {x} / {y} = {quot} (stays in N)"
                    ));
                    Number::Natural(quot)
                } else {
                    let (num, den) = normalize_rational(*x as i64, *y as i64);
                    proof_trace.push(format!(
                        "Promotion N -> Q: {x} / {y} = {num}/{den} (not exact)"
                    ));
                    Self::from_rational(num, den)
                }
            }

            // ℤ / ℤ: exact → ℤ (or ℕ), otherwise → ℚ
            (Number::Integer(x), Number::Integer(y)) => {
                if *y != 0 && *x % *y == 0 {
                    let quot = x / y;
                    proof_trace.push(format!(
                        "Exact integer division: {x} / {y} = {quot} (stays in Z)"
                    ));
                    Self::from_i64(quot)
                } else {
                    proof_trace.push(format!("Promotion Z -> Q: {x} / {y} (not exact)"));
                    Self::from_rational(*x, *y)
                }
            }

            // Mixed ℕ/ℤ: lift to ℤ then decide
            (Number::Natural(x), Number::Integer(y)) => {
                let xi = *x as i64;
                if *y != 0 && xi % *y == 0 {
                    let quot = xi / *y;
                    proof_trace.push(format!("Exact division after lift: {x} / {y} = {quot}"));
                    Self::from_i64(quot)
                } else {
                    proof_trace.push(format!("Promotion to Q: {x} / {y}"));
                    Self::from_rational(xi, *y)
                }
            }
            (Number::Integer(x), Number::Natural(y)) => {
                let yi = *y as i64;
                if yi != 0 && *x % yi == 0 {
                    let quot = *x / yi;
                    proof_trace.push(format!("Exact division after lift: {x} / {y} = {quot}"));
                    Self::from_i64(quot)
                } else {
                    proof_trace.push(format!("Promotion to Q: {x} / {y}"));
                    Self::from_rational(*x, yi)
                }
            }

            // ℚ / ℚ: multiply by reciprocal
            (
                Number::Rational {
                    numerator: an,
                    denominator: ad,
                },
                Number::Rational {
                    numerator: bn,
                    denominator: bd,
                },
            ) => {
                // (an/ad) / (bn/bd) = (an * bd) / (ad * bn)
                let num = an.wrapping_mul(*bd);
                let den = ad.wrapping_mul(*bn);
                proof_trace.push(format!(
                    "Rational division: ({an}/{ad}) / ({bn}/{bd}) = ({an} * {bd}) / ({ad} * {bn})"
                ));
                Self::from_rational(num, den)
            }

            // Mixed with ℚ
            (
                Number::Rational {
                    numerator: rn,
                    denominator: rd,
                },
                other,
            ) => {
                let (on, od) = self.to_rational(other);
                // (rn/rd) / (on/od) = (rn * od) / (rd * on)
                let num = rn.wrapping_mul(od);
                let den = rd.wrapping_mul(on);
                proof_trace.push(format!("Lift to Q: ({rn}/{rd}) / ({on}/{od})"));
                Self::from_rational(num, den)
            }
            (
                other,
                Number::Rational {
                    numerator: rn,
                    denominator: rd,
                },
            ) => {
                let (on, od) = self.to_rational(other);
                // (on/od) / (rn/rd) = (on * rd) / (od * rn)
                let num = on.wrapping_mul(*rd);
                let den = od.wrapping_mul(*rn);
                proof_trace.push(format!("Lift to Q: ({on}/{od}) / ({rn}/{rd})"));
                Self::from_rational(num, den)
            }

            // ℝ / anything → ℝ
            (Number::Real(x), other) => {
                let quot = x / other.to_f64();
                proof_trace.push(format!(
                    "Real division: {} / {} = {}",
                    x,
                    other.to_f64(),
                    quot
                ));
                Self::from_f64(quot)
            }
            (other, Number::Real(x)) => {
                let quot = other.to_f64() / x;
                proof_trace.push(format!(
                    "Real division: {} / {} = {}",
                    other.to_f64(),
                    x,
                    quot
                ));
                Self::from_f64(quot)
            }
        };

        let input_domains = [a.domain(), b.domain()];
        Some(self.build_result(result, "divide".to_string(), proof_trace, &input_domains))
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

    /// Negate a number: -a
    pub fn negate(&self, a: &Number) -> NumberResult {
        let mut proof_trace = vec![format!("Negate: -({})", a)];

        let result = match a {
            Number::Natural(0) => {
                proof_trace.push("Negation of zero is zero".to_string());
                Number::Natural(0)
            }
            Number::Natural(n) => {
                let neg = -(*n as i64);
                proof_trace.push(format!("Promotion N -> Z: -({n}) = {neg}"));
                Number::Integer(neg)
            }
            Number::Integer(n) => {
                let neg = n.wrapping_neg();
                proof_trace.push(format!("Integer negation: -({n}) = {neg}"));
                Self::from_i64(neg)
            }
            Number::Rational {
                numerator,
                denominator,
            } => {
                let neg_num = numerator.wrapping_neg();
                proof_trace.push(format!(
                    "Rational negation: -({numerator}/{denominator}) = {neg_num}/{denominator}"
                ));
                Self::from_rational(neg_num, *denominator)
            }
            Number::Real(x) => {
                let neg = -x;
                proof_trace.push(format!("Real negation: -({x}) = {neg}"));
                Number::Real(neg)
            }
        };

        let input_domains = [a.domain()];
        self.build_result(result, "negate".to_string(), proof_trace, &input_domains)
    }

    /// Absolute value: |a|
    pub fn abs(&self, a: &Number) -> NumberResult {
        let mut proof_trace = vec![format!("Absolute value: |{}|", a)];

        let result = match a {
            Number::Natural(n) => {
                proof_trace.push(format!("Already non-negative: |{n}| = {n}"));
                Number::Natural(*n)
            }
            Number::Integer(n) => {
                let abs_val = n.checked_abs().unwrap_or(i64::MAX);
                proof_trace.push(format!("Integer absolute value: |{n}| = {abs_val}"));
                Number::Natural(abs_val as u64)
            }
            Number::Rational {
                numerator,
                denominator,
            } => {
                let abs_num = numerator.checked_abs().unwrap_or(i64::MAX);
                proof_trace.push(format!(
                    "Rational absolute value: |{numerator}/{denominator}| = {abs_num}/{denominator}"
                ));
                Self::from_rational(abs_num, *denominator)
            }
            Number::Real(x) => {
                let abs_val = x.abs();
                proof_trace.push(format!("Real absolute value: |{x}| = {abs_val}"));
                Self::from_f64(abs_val)
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
}
