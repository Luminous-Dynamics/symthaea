// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Integer Arithmetic (Z) for HDC Consciousness Framework
//!
//! Extends the natural number (ℕ) arithmetic to integers (ℤ) by introducing
//! sign and negation concepts through the primitive system.
//!
//! ## Core Approach
//!
//! - **Sign Representation**: NEGATION primitive bound with magnitude encoding
//! - **Positive Integers**: Use natural number encoding directly
//! - **Negative Integers**: NEGATION ⊗ magnitude_encoding
//! - **Zero**: Special case (neither positive nor negative)
//!
//! ## Operations
//!
//! All operations are sign-aware and use the HybridArithmeticEngine for
//! magnitude computations, then apply sign rules for integer semantics.

use crate::hdc::arithmetic_engine::HybridArithmeticEngine;
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, seed_from_name};
use serde::{Deserialize, Serialize};

// ============================================================================
// CORE TYPES
// ============================================================================

/// Sign of an integer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sign {
    /// Positive integer (> 0)
    Positive,
    /// Zero
    Zero,
    /// Negative integer (< 0)
    Negative,
}

impl Sign {
    /// Get sign from i64 value
    pub fn from_value(n: i64) -> Self {
        match n.cmp(&0) {
            std::cmp::Ordering::Greater => Sign::Positive,
            std::cmp::Ordering::Equal => Sign::Zero,
            std::cmp::Ordering::Less => Sign::Negative,
        }
    }

    /// Multiply two signs
    pub fn multiply(&self, other: &Sign) -> Sign {
        match (self, other) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (Sign::Positive, Sign::Positive) | (Sign::Negative, Sign::Negative) => Sign::Positive,
            (Sign::Positive, Sign::Negative) | (Sign::Negative, Sign::Positive) => Sign::Negative,
        }
    }

    /// Negate a sign
    pub fn negate(&self) -> Sign {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Zero => Sign::Zero,
            Sign::Negative => Sign::Positive,
        }
    }
}

/// An integer represented in Hyperdimensional space
///
/// Integers extend natural numbers with sign:
/// - Positive: magnitude encoding
/// - Negative: NEGATION ⊗ magnitude encoding
/// - Zero: ZERO primitive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcInteger {
    /// The hypervector encoding of this integer (with sign)
    pub encoding: BinaryHV,

    /// The numeric value (for verification and display)
    pub value: i64,

    /// The sign of this integer
    pub sign: Sign,

    /// The magnitude encoding (absolute value)
    pub magnitude_encoding: BinaryHV,

    /// Φ accumulated during construction
    pub construction_phi: f64,
}

impl HdcInteger {
    /// Create zero - the neutral element
    pub fn zero(primitives: &PrimitiveSystem) -> Self {
        let zero_prim = primitives.get("ZERO").expect("ZERO primitive must exist");

        Self {
            encoding: zero_prim.encoding,
            value: 0,
            sign: Sign::Zero,
            magnitude_encoding: zero_prim.encoding,
            construction_phi: 0.0,
        }
    }

    /// Encode an integer value
    ///
    /// - Positive: uses deterministic encoding from seed
    /// - Negative: NEGATION ⊗ positive_encoding
    /// - Zero: ZERO primitive
    fn encode(n: i64, primitives: &PrimitiveSystem) -> Self {
        let sign = Sign::from_value(n);
        let magnitude = n.unsigned_abs();

        // Get magnitude encoding
        let magnitude_encoding = if magnitude == 0 {
            primitives
                .get("ZERO")
                .expect("ZERO primitive must exist")
                .encoding
        } else if magnitude <= 20 {
            // Small numbers: Peano-style construction
            let zero_prim = primitives.get("ZERO").expect("ZERO primitive must exist");
            let succ_prim = primitives
                .get("SUCCESSOR")
                .expect("SUCCESSOR primitive must exist");

            let mut encoding = zero_prim.encoding;
            for _ in 0..magnitude {
                encoding = succ_prim.encoding.bind(&encoding);
            }
            encoding
        } else {
            // Large numbers: deterministic random encoding
            BinaryHV::random(seed_from_name(&format!("NUM_{magnitude}")))
        };

        // Apply sign
        let (encoding, construction_phi) = match sign {
            Sign::Zero => (magnitude_encoding, 0.0),
            Sign::Positive => (magnitude_encoding, magnitude as f64 * 0.1),
            Sign::Negative => {
                let negation_prim = primitives
                    .get("NOT")
                    .or_else(|| primitives.get("NEGATION"))
                    .expect("NOT/NEGATION primitive must exist");
                let encoding = negation_prim.encoding.bind(&magnitude_encoding);
                // Add phi for negation binding
                (encoding, magnitude as f64 * 0.1 + 1.0)
            }
        };

        Self {
            encoding,
            value: n,
            sign,
            magnitude_encoding,
            construction_phi,
        }
    }

    /// Get the absolute value (returns u64 to handle i64::MIN correctly)
    pub fn unsigned_abs(&self) -> u64 {
        self.value.unsigned_abs()
    }

    /// Get the signum (-1, 0, or 1)
    pub fn signum(&self) -> i64 {
        match self.sign {
            Sign::Positive => 1,
            Sign::Zero => 0,
            Sign::Negative => -1,
        }
    }
}

/// Result of an integer arithmetic operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegerResult {
    /// The resulting integer
    pub integer: HdcInteger,

    /// Description of the operation performed
    pub operation: String,

    /// Proof trace (symbolic reasoning steps)
    pub proof_trace: Vec<String>,

    /// Accumulated Φ (integrated information measure)
    pub phi: f64,
}

impl IntegerResult {
    /// Create a result from an integer and operation description
    fn new(integer: HdcInteger, operation: String, proof_trace: Vec<String>) -> Self {
        let phi = integer.construction_phi;
        Self {
            integer,
            operation,
            proof_trace,
            phi,
        }
    }

    /// Add a proof step
    fn add_proof_step(&mut self, step: String) {
        self.proof_trace.push(step);
    }

    /// Add to accumulated Φ
    fn add_phi(&mut self, delta: f64) {
        self.phi += delta;
    }
}

// ============================================================================
// INTEGER ARITHMETIC ENGINE
// ============================================================================

/// Arithmetic engine for integers (ℤ)
///
/// Wraps HybridArithmeticEngine for natural number operations and extends
/// with sign-aware integer semantics.
pub struct IntegerArithmeticEngine {
    /// Natural number engine for magnitude operations
    hybrid_engine: HybridArithmeticEngine,

    /// Primitive system for sign operations
    primitives: &'static PrimitiveSystem,
}

impl IntegerArithmeticEngine {
    /// Create a new integer arithmetic engine
    pub fn new() -> Self {
        Self {
            hybrid_engine: HybridArithmeticEngine::new(),
            primitives: PrimitiveSystem::global(),
        }
    }

    /// Encode an integer value
    pub fn encode(&self, value: i64) -> HdcInteger {
        HdcInteger::encode(value, self.primitives)
    }

    /// Add two integers: a + b
    ///
    /// Sign rules:
    /// - pos + pos = pos
    /// - neg + neg = neg
    /// - pos + neg = depends on magnitudes
    pub fn add(&mut self, a: i64, b: i64) -> IntegerResult {
        let result_value = a.wrapping_add(b);
        let hdc_a = self.encode(a);
        let hdc_b = self.encode(b);

        let mut proof_trace = vec![format!("Add: {} + {}", a, b)];

        // Determine sign and magnitude computation
        let phi_operation = match (hdc_a.sign, hdc_b.sign) {
            (Sign::Zero, _) => {
                proof_trace.push(format!("Identity: 0 + {b} = {b}"));
                0.0
            }
            (_, Sign::Zero) => {
                proof_trace.push(format!("Identity: {a} + 0 = {a}"));
                0.0
            }
            (Sign::Positive, Sign::Positive) => {
                let result = self.hybrid_engine.add(a as u64, b as u64);
                proof_trace.push(format!("Positive sum: {} + {} = {}", a, b, result.value));
                result.phi
            }
            (Sign::Negative, Sign::Negative) => {
                let result = self
                    .hybrid_engine
                    .add(hdc_a.unsigned_abs(), hdc_b.unsigned_abs());
                proof_trace.push(format!(
                    "Negative sum: -({} + {}) = -{}",
                    hdc_a.unsigned_abs(),
                    hdc_b.unsigned_abs(),
                    result.value
                ));
                result.phi + 1.0 // Extra phi for sign handling
            }
            (Sign::Positive, Sign::Negative) | (Sign::Negative, Sign::Positive) => {
                proof_trace.push("Mixed signs: compute difference of magnitudes".to_string());
                let mag_a = hdc_a.unsigned_abs();
                let mag_b = hdc_b.unsigned_abs();
                if mag_a >= mag_b {
                    let result = self.hybrid_engine.subtract(mag_a, mag_b);
                    result.map(|r| r.phi).unwrap_or(0.0) + 1.5 // Phi for subtraction and sign determination
                } else {
                    let result = self.hybrid_engine.subtract(mag_b, mag_a);
                    result.map(|r| r.phi).unwrap_or(0.0) + 1.5
                }
            }
        };

        let hdc_result = self.encode(result_value);
        let mut result = IntegerResult::new(hdc_result, format!("{a} + {b}"), proof_trace);
        result.add_phi(phi_operation);
        result
    }

    /// Subtract two integers: a - b
    ///
    /// Implemented as a + (-b)
    pub fn subtract(&mut self, a: i64, b: i64) -> IntegerResult {
        let result_value = a.wrapping_sub(b);
        let mut proof_trace = vec![
            format!("Subtract: {} - {}", a, b),
            format!("Rewrite as: {} + (-{})", a, b),
        ];

        // Use addition with negated second operand
        let add_result = self.add(a, -b);
        proof_trace.extend(add_result.proof_trace);

        IntegerResult {
            integer: self.encode(result_value),
            operation: format!("{a} - {b}"),
            proof_trace,
            phi: add_result.phi + 0.5, // Extra phi for negation
        }
    }

    /// Multiply two integers: a × b
    ///
    /// Sign rules:
    /// - pos × pos = pos
    /// - neg × neg = pos
    /// - pos × neg = neg
    pub fn multiply(&mut self, a: i64, b: i64) -> IntegerResult {
        let result_value = a.wrapping_mul(b);
        let hdc_a = self.encode(a);
        let hdc_b = self.encode(b);

        let mut proof_trace = vec![format!("Multiply: {} × {}", a, b)];

        // Compute magnitude
        let mag_a = hdc_a.unsigned_abs();
        let mag_b = hdc_b.unsigned_abs();

        let phi_operation = if mag_a == 0 || mag_b == 0 {
            proof_trace.push(format!("Zero product: {a} × {b} = 0"));
            0.0
        } else {
            let result = self.hybrid_engine.multiply(mag_a, mag_b);
            let result_sign = hdc_a.sign.multiply(&hdc_b.sign);
            proof_trace.push(format!(
                "Magnitude: {} × {} = {}",
                mag_a, mag_b, result.value
            ));
            proof_trace.push(format!(
                "Sign rule: {:?} × {:?} = {:?}",
                hdc_a.sign, hdc_b.sign, result_sign
            ));
            result.phi + 1.0 // Extra phi for sign rule
        };

        let hdc_result = self.encode(result_value);
        let mut result = IntegerResult::new(hdc_result, format!("{a} × {b}"), proof_trace);
        result.add_phi(phi_operation);
        result
    }

    /// Divide two integers: a ÷ b (truncating toward zero)
    ///
    /// Returns None if b = 0
    pub fn divide(&mut self, a: i64, b: i64) -> Option<IntegerResult> {
        if b == 0 {
            return None;
        }

        let result_value = a / b; // Truncates toward zero
        let hdc_a = self.encode(a);
        let hdc_b = self.encode(b);

        let mut proof_trace = vec![
            format!("Divide: {} ÷ {}", a, b),
            format!("Truncating division toward zero"),
        ];

        // Compute magnitude division
        let mag_a = hdc_a.unsigned_abs();
        let mag_b = hdc_b.unsigned_abs();
        let mag_result = mag_a / mag_b;

        proof_trace.push(format!("Magnitude: {mag_a} ÷ {mag_b} = {mag_result}"));

        // Apply sign rule (same as multiplication)
        let result_sign = hdc_a.sign.multiply(&hdc_b.sign);
        proof_trace.push(format!(
            "Sign rule: {:?} ÷ {:?} = {:?}",
            hdc_a.sign, hdc_b.sign, result_sign
        ));

        let hdc_result = self.encode(result_value);
        let mut result = IntegerResult::new(hdc_result, format!("{a} ÷ {b}"), proof_trace);
        result.add_phi(mag_result as f64 * 0.5 + 2.0); // Phi for division operation

        Some(result)
    }

    /// Modulo operation: a mod b
    ///
    /// Returns the remainder with sign of divisor (mathematical modulo).
    /// Returns None if b = 0
    pub fn modulo(&mut self, a: i64, b: i64) -> Option<IntegerResult> {
        if b == 0 {
            return None;
        }

        // Rust's rem_euclid gives mathematical modulo
        let result_value = a.rem_euclid(b);

        let mut proof_trace = vec![
            format!("Modulo: {} mod {}", a, b),
            format!("Mathematical modulo (Euclidean)"),
        ];

        // Compute using: a mod b = a - b * floor(a/b)
        let quotient = a / b;
        proof_trace.push(format!("Quotient: {a} ÷ {b} = {quotient}"));
        proof_trace.push(format!(
            "Remainder: {a} - {b} × {quotient} = {result_value}"
        ));

        let hdc_result = self.encode(result_value);
        let mut result = IntegerResult::new(hdc_result, format!("{a} mod {b}"), proof_trace);
        result.add_phi(2.5); // Phi for modulo computation

        Some(result)
    }

    /// Absolute value: |a|
    pub fn abs(&mut self, a: i64) -> IntegerResult {
        let result_value = a.checked_abs().unwrap_or(i64::MAX);
        let hdc_a = self.encode(a);

        let proof_trace = vec![
            format!("Absolute value: |{}|", a),
            match hdc_a.sign {
                Sign::Zero => "Result: 0".to_string(),
                Sign::Positive => format!("Already positive: {a}"),
                Sign::Negative => format!("Negate: -({a}) = {result_value}"),
            },
        ];

        let hdc_result = self.encode(result_value);
        let phi = if hdc_a.sign == Sign::Negative {
            1.0
        } else {
            0.0
        };

        let mut result = IntegerResult::new(hdc_result, format!("|{a}|"), proof_trace);
        result.add_phi(phi);
        result
    }

    /// Signum function: sgn(a)
    ///
    /// Returns -1, 0, or 1 depending on sign
    pub fn signum(&mut self, a: i64) -> IntegerResult {
        let result_value = a.signum();
        let hdc_a = self.encode(a);

        let proof_trace = vec![
            format!("Signum: sgn({})", a),
            format!("Sign: {:?} → {}", hdc_a.sign, result_value),
        ];

        let hdc_result = self.encode(result_value);
        let mut result = IntegerResult::new(hdc_result, format!("sgn({a})"), proof_trace);
        result.add_phi(0.5); // Simple sign extraction

        result
    }

    /// Negate an integer: -a
    pub fn negate(&mut self, a: i64) -> IntegerResult {
        let result_value = a.wrapping_neg();
        let hdc_a = self.encode(a);

        let proof_trace = vec![
            format!("Negate: -({})", a),
            format!("Sign: {:?} → {:?}", hdc_a.sign, hdc_a.sign.negate()),
            format!("Result: {}", result_value),
        ];

        let hdc_result = self.encode(result_value);
        let mut result = IntegerResult::new(hdc_result, format!("-({a})"), proof_trace);
        result.add_phi(1.0); // Phi for negation binding

        result
    }
}

impl Default for IntegerArithmeticEngine {
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

    #[test]
    fn test_sign_from_value() {
        assert_eq!(Sign::from_value(5), Sign::Positive);
        assert_eq!(Sign::from_value(0), Sign::Zero);
        assert_eq!(Sign::from_value(-5), Sign::Negative);
    }

    #[test]
    fn test_sign_multiply() {
        assert_eq!(Sign::Positive.multiply(&Sign::Positive), Sign::Positive);
        assert_eq!(Sign::Positive.multiply(&Sign::Negative), Sign::Negative);
        assert_eq!(Sign::Negative.multiply(&Sign::Positive), Sign::Negative);
        assert_eq!(Sign::Negative.multiply(&Sign::Negative), Sign::Positive);
        assert_eq!(Sign::Zero.multiply(&Sign::Positive), Sign::Zero);
    }

    #[test]
    fn test_sign_negate() {
        assert_eq!(Sign::Positive.negate(), Sign::Negative);
        assert_eq!(Sign::Negative.negate(), Sign::Positive);
        assert_eq!(Sign::Zero.negate(), Sign::Zero);
    }

    #[test]
    fn test_encode_zero() {
        let engine = IntegerArithmeticEngine::new();
        let zero = engine.encode(0);
        assert_eq!(zero.value, 0);
        assert_eq!(zero.sign, Sign::Zero);
    }

    #[test]
    fn test_encode_positive() {
        let engine = IntegerArithmeticEngine::new();
        let pos = engine.encode(5);
        assert_eq!(pos.value, 5);
        assert_eq!(pos.sign, Sign::Positive);
    }

    #[test]
    fn test_encode_negative() {
        let engine = IntegerArithmeticEngine::new();
        let neg = engine.encode(-5);
        assert_eq!(neg.value, -5);
        assert_eq!(neg.sign, Sign::Negative);
    }

    #[test]
    fn test_integer_add() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.add(-3, 5);
        assert_eq!(r.integer.value, 2);
        assert_eq!(r.integer.sign, Sign::Positive);
    }

    #[test]
    fn test_integer_add_positive() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.add(3, 5);
        assert_eq!(r.integer.value, 8);
        assert_eq!(r.integer.sign, Sign::Positive);
    }

    #[test]
    fn test_integer_add_negative() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.add(-3, -5);
        assert_eq!(r.integer.value, -8);
        assert_eq!(r.integer.sign, Sign::Negative);
    }

    #[test]
    fn test_integer_subtract() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.subtract(5, 3);
        assert_eq!(r.integer.value, 2);
    }

    #[test]
    fn test_integer_multiply_positives() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.multiply(3, 4);
        assert_eq!(r.integer.value, 12);
        assert_eq!(r.integer.sign, Sign::Positive);
    }

    #[test]
    fn test_integer_multiply_negatives() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.multiply(-2, -3);
        assert_eq!(r.integer.value, 6);
        assert_eq!(r.integer.sign, Sign::Positive);
    }

    #[test]
    fn test_integer_multiply_mixed() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.multiply(-2, 3);
        assert_eq!(r.integer.value, -6);
        assert_eq!(r.integer.sign, Sign::Negative);
    }

    #[test]
    fn test_negate_negate() {
        let mut engine = IntegerArithmeticEngine::new();
        let a = engine.encode(5);
        let neg_a = engine.negate(5);
        let neg_neg_a = engine.negate(-5);
        // negate(negate(x)) should have same value
        assert_eq!(neg_neg_a.integer.value, 5);
        assert_eq!(neg_a.integer.value, -5);
        assert_eq!(a.value, 5);
    }

    #[test]
    fn test_divide_truncates() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.divide(7, 2).unwrap();
        assert_eq!(r.integer.value, 3);
        let r = engine.divide(-7, 2).unwrap();
        assert_eq!(r.integer.value, -3);
    }

    #[test]
    fn test_divide_negative_by_negative() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.divide(-8, -2).unwrap();
        assert_eq!(r.integer.value, 4);
        assert_eq!(r.integer.sign, Sign::Positive);
    }

    #[test]
    fn test_divide_by_zero() {
        let mut engine = IntegerArithmeticEngine::new();
        assert!(engine.divide(5, 0).is_none());
    }

    #[test]
    fn test_modulo() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.modulo(7, 3).unwrap();
        assert_eq!(r.integer.value, 1);
    }

    #[test]
    fn test_modulo_negative() {
        let mut engine = IntegerArithmeticEngine::new();
        // -7 mod 3 = 2 (mathematical modulo)
        let r = engine.modulo(-7, 3).unwrap();
        assert_eq!(r.integer.value, 2);
    }

    #[test]
    fn test_modulo_by_zero() {
        let mut engine = IntegerArithmeticEngine::new();
        assert!(engine.modulo(5, 0).is_none());
    }

    #[test]
    fn test_abs() {
        let mut engine = IntegerArithmeticEngine::new();
        assert_eq!(engine.abs(5).integer.value, 5);
        assert_eq!(engine.abs(-5).integer.value, 5);
        assert_eq!(engine.abs(0).integer.value, 0);
    }

    #[test]
    fn test_signum() {
        let mut engine = IntegerArithmeticEngine::new();
        assert_eq!(engine.signum(5).integer.value, 1);
        assert_eq!(engine.signum(-5).integer.value, -1);
        assert_eq!(engine.signum(0).integer.value, 0);
    }

    #[test]
    fn test_phi_accumulation() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.add(3, 5);
        // Should have some positive Φ from the operation
        assert!(r.phi > 0.0);
    }

    #[test]
    fn test_proof_traces() {
        let mut engine = IntegerArithmeticEngine::new();
        let r = engine.multiply(-2, 3);
        // Should have proof steps documenting the operation
        assert!(!r.proof_trace.is_empty());
        assert!(r.proof_trace.iter().any(|s| s.contains("Multiply")));
    }

    #[test]
    fn test_large_numbers() {
        let engine = IntegerArithmeticEngine::new();
        // Test encoding of large numbers (> 20, should use deterministic random)
        let large = engine.encode(1000);
        assert_eq!(large.value, 1000);
        assert_eq!(large.sign, Sign::Positive);

        let large_neg = engine.encode(-1000);
        assert_eq!(large_neg.value, -1000);
        assert_eq!(large_neg.sign, Sign::Negative);
    }

    #[test]
    fn test_identity_operations() {
        let mut engine = IntegerArithmeticEngine::new();

        // Addition identity: a + 0 = a
        let r = engine.add(5, 0);
        assert_eq!(r.integer.value, 5);

        // Multiplication by one (using actual arithmetic since no direct multiply-by-one test)
        let r = engine.multiply(5, 1);
        assert_eq!(r.integer.value, 5);

        // Multiplication by zero: a × 0 = 0
        let r = engine.multiply(5, 0);
        assert_eq!(r.integer.value, 0);
    }
}
