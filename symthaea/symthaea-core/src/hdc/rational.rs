#![allow(dead_code)]

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, seed_from_name};
use std::cmp::Ordering;

/// A rational number encoded in HDC space
#[derive(Debug, Clone)]
pub struct HdcRational {
    /// The HDC encoding of this rational number
    pub encoding: BinaryHV,
    /// Numerator (can be negative)
    pub numerator: i64,
    /// Denominator (always > 0, coprime with abs(numerator))
    pub denominator: i64,
    /// Construction phi (similarity measure)
    pub construction_phi: f64,
}

/// Result of a rational arithmetic operation
#[derive(Debug, Clone)]
pub struct RationalResult {
    /// The resulting rational number
    pub rational: HdcRational,
    /// Description of the operation performed
    pub operation: String,
    /// Proof trace showing the steps
    pub proof_trace: Vec<String>,
    /// Phi value (similarity measure)
    pub phi: f64,
}

/// Compute greatest common divisor using Euclidean algorithm
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Normalize a rational number: ensure denominator > 0 and coprime with numerator
fn normalize(num: i64, den: i64) -> (i64, i64) {
    if den == 0 {
        panic!("Denominator cannot be zero");
    }

    // Ensure denominator is positive
    let (num, den) = if den < 0 {
        (-num, -den)
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

/// Engine for rational arithmetic in HDC space
pub struct RationalArithmeticEngine {
    primitives: &'static PrimitiveSystem,
}

impl RationalArithmeticEngine {
    /// Create a new rational arithmetic engine
    pub fn new() -> Self {
        Self {
            primitives: PrimitiveSystem::global(),
        }
    }

    /// Encode an integer into HDC space using the natural number encoding scheme
    fn encode_int(&self, n: i64) -> BinaryHV {
        if n == 0 {
            return self.primitives.get("ZERO").unwrap().encoding;
        }

        let (is_negative, abs_n) = if n < 0 {
            (true, (-n) as u64)
        } else {
            (false, n as u64)
        };

        // Build up from ZERO using SUCCESSOR
        let zero = self.primitives.get("ZERO").unwrap().encoding;
        let successor = self.primitives.get("SUCCESSOR").unwrap().encoding;

        let mut encoding = zero;
        for _ in 0..abs_n {
            encoding = encoding.bind(&successor);
        }

        // Apply negation if needed
        if is_negative {
            if let Some(negation) = self.primitives.get("NEGATION") {
                encoding = encoding.bind(&negation.encoding);
            } else {
                // Fallback: use a deterministic seed for negation
                let neg_hv = BinaryHV::random(seed_from_name("NEGATION"));
                encoding = encoding.bind(&neg_hv);
            }
        }

        encoding
    }

    /// Encode a rational number as numerator/denominator
    pub fn encode(&self, numerator: i64, denominator: i64) -> HdcRational {
        let (num, den) = normalize(numerator, denominator);

        let ratio = self.primitives.get("RATIO").unwrap().encoding;
        let num_encoding = self.encode_int(num);
        let den_encoding = self.encode_int(den);

        // encoding = RATIO ⊗ encode(num) ⊗ encode(den)
        let encoding = ratio.bind(&num_encoding).bind(&den_encoding);

        // Compute construction phi (similarity to RATIO primitive)
        let construction_phi = ratio.similarity(&encoding) as f64;

        HdcRational {
            encoding,
            numerator: num,
            denominator: den,
            construction_phi,
        }
    }

    /// Add two rational numbers: a/b + c/d = (ad + bc) / bd
    pub fn add(&self, a: &HdcRational, b: &HdcRational) -> RationalResult {
        let num = a.numerator * b.denominator + b.numerator * a.denominator;
        let den = a.denominator * b.denominator;

        let rational = self.encode(num, den);

        let operation = format!(
            "{}/{} + {}/{} = {}/{}",
            a.numerator, a.denominator,
            b.numerator, b.denominator,
            rational.numerator, rational.denominator
        );

        let proof_trace = vec![
            format!("Add: ({} × {}) + ({} × {}) = {}",
                a.numerator, b.denominator, b.numerator, a.denominator, num),
            format!("Denominator: {} × {} = {}", a.denominator, b.denominator, den),
            format!("Normalized: {}/{}", rational.numerator, rational.denominator),
        ];

        let phi = rational.construction_phi;

        RationalResult {
            rational,
            operation,
            proof_trace,
            phi,
        }
    }

    /// Subtract two rational numbers: a/b - c/d = (ad - bc) / bd
    pub fn subtract(&self, a: &HdcRational, b: &HdcRational) -> RationalResult {
        let num = a.numerator * b.denominator - b.numerator * a.denominator;
        let den = a.denominator * b.denominator;

        let rational = self.encode(num, den);

        let operation = format!(
            "{}/{} - {}/{} = {}/{}",
            a.numerator, a.denominator,
            b.numerator, b.denominator,
            rational.numerator, rational.denominator
        );

        let proof_trace = vec![
            format!("Subtract: ({} × {}) - ({} × {}) = {}",
                a.numerator, b.denominator, b.numerator, a.denominator, num),
            format!("Denominator: {} × {} = {}", a.denominator, b.denominator, den),
            format!("Normalized: {}/{}", rational.numerator, rational.denominator),
        ];

        let phi = rational.construction_phi;

        RationalResult {
            rational,
            operation,
            proof_trace,
            phi,
        }
    }

    /// Multiply two rational numbers: (a/b) × (c/d) = (ac) / (bd)
    pub fn multiply(&self, a: &HdcRational, b: &HdcRational) -> RationalResult {
        let num = a.numerator * b.numerator;
        let den = a.denominator * b.denominator;

        let rational = self.encode(num, den);

        let operation = format!(
            "{}/{} × {}/{} = {}/{}",
            a.numerator, a.denominator,
            b.numerator, b.denominator,
            rational.numerator, rational.denominator
        );

        let proof_trace = vec![
            format!("Numerator: {} × {} = {}", a.numerator, b.numerator, num),
            format!("Denominator: {} × {} = {}", a.denominator, b.denominator, den),
            format!("Normalized: {}/{}", rational.numerator, rational.denominator),
        ];

        let phi = rational.construction_phi;

        RationalResult {
            rational,
            operation,
            proof_trace,
            phi,
        }
    }

    /// Divide two rational numbers: (a/b) ÷ (c/d) = (ad) / (bc)
    /// Returns None if divisor numerator is zero
    pub fn divide(&self, a: &HdcRational, b: &HdcRational) -> Option<RationalResult> {
        if b.numerator == 0 {
            return None;
        }

        let num = a.numerator * b.denominator;
        let den = a.denominator * b.numerator;

        let rational = self.encode(num, den);

        let operation = format!(
            "{}/{} ÷ {}/{} = {}/{}",
            a.numerator, a.denominator,
            b.numerator, b.denominator,
            rational.numerator, rational.denominator
        );

        let proof_trace = vec![
            format!("Numerator: {} × {} = {}", a.numerator, b.denominator, num),
            format!("Denominator: {} × {} = {}", a.denominator, b.numerator, den),
            format!("Normalized: {}/{}", rational.numerator, rational.denominator),
        ];

        let phi = rational.construction_phi;

        Some(RationalResult {
            rational,
            operation,
            proof_trace,
            phi,
        })
    }

    /// Compute reciprocal: 1 / (a/b) = b/a
    /// Returns None if numerator is zero
    pub fn reciprocal(&self, a: &HdcRational) -> Option<RationalResult> {
        if a.numerator == 0 {
            return None;
        }

        let rational = self.encode(a.denominator, a.numerator);

        let operation = format!(
            "1 / ({}/{}) = {}/{}",
            a.numerator, a.denominator,
            rational.numerator, rational.denominator
        );

        let proof_trace = vec![
            format!("Reciprocal: swap numerator and denominator"),
            format!("Result: {}/{}", rational.numerator, rational.denominator),
        ];

        let phi = rational.construction_phi;

        Some(RationalResult {
            rational,
            operation,
            proof_trace,
            phi,
        })
    }

    /// Compare two rational numbers using cross-multiplication
    pub fn compare(&self, a: &HdcRational, b: &HdcRational) -> Ordering {
        let left = a.numerator * b.denominator;
        let right = b.numerator * a.denominator;
        left.cmp(&right)
    }

    /// Compute the mediant of two rationals (Stern-Brocot tree)
    /// mediant(a/b, c/d) = (a+c)/(b+d)
    pub fn mediant(&self, a: &HdcRational, b: &HdcRational) -> RationalResult {
        let num = a.numerator + b.numerator;
        let den = a.denominator + b.denominator;

        let rational = self.encode(num, den);

        let operation = format!(
            "mediant({}/{}, {}/{}) = {}/{}",
            a.numerator, a.denominator,
            b.numerator, b.denominator,
            rational.numerator, rational.denominator
        );

        let proof_trace = vec![
            format!("Numerator: {} + {} = {}", a.numerator, b.numerator, num),
            format!("Denominator: {} + {} = {}", a.denominator, b.denominator, den),
            format!("Mediant (Stern-Brocot): {}/{}", rational.numerator, rational.denominator),
        ];

        let phi = rational.construction_phi;

        RationalResult {
            rational,
            operation,
            proof_trace,
            phi,
        }
    }

    /// Create a rational from an integer n (represented as n/1)
    pub fn from_integer(&self, n: i64) -> HdcRational {
        self.encode(n, 1)
    }

    /// Convert rational to floating-point
    pub fn to_f64(&self, a: &HdcRational) -> f64 {
        a.numerator as f64 / a.denominator as f64
    }

    /// Check if rational is an integer (denominator == 1)
    pub fn is_integer(&self, a: &HdcRational) -> bool {
        a.denominator == 1
    }
}

impl Default for RationalArithmeticEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_fractions() {
        let engine = RationalArithmeticEngine::new();
        let a = engine.encode(1, 2);
        let b = engine.encode(1, 3);
        let r = engine.add(&a, &b);
        assert_eq!(r.rational.numerator, 5);
        assert_eq!(r.rational.denominator, 6);
    }

    #[test]
    fn test_normalization() {
        let engine = RationalArithmeticEngine::new();
        let r = engine.encode(2, 4);
        assert_eq!(r.numerator, 1);
        assert_eq!(r.denominator, 2);
    }

    #[test]
    fn test_negative_denominator() {
        let engine = RationalArithmeticEngine::new();
        let r = engine.encode(1, -2);
        assert_eq!(r.numerator, -1);
        assert_eq!(r.denominator, 2);
    }

    #[test]
    fn test_multiply() {
        let engine = RationalArithmeticEngine::new();
        let a = engine.encode(2, 3);
        let b = engine.encode(3, 4);
        let r = engine.multiply(&a, &b);
        assert_eq!(r.rational.numerator, 1);
        assert_eq!(r.rational.denominator, 2);
    }

    #[test]
    fn test_divide_by_zero() {
        let engine = RationalArithmeticEngine::new();
        let a = engine.encode(1, 2);
        let b = engine.encode(0, 1);
        assert!(engine.divide(&a, &b).is_none());
    }

    #[test]
    fn test_from_integer() {
        let engine = RationalArithmeticEngine::new();
        let r = engine.from_integer(5);
        assert_eq!(r.numerator, 5);
        assert_eq!(r.denominator, 1);
    }

    #[test]
    fn test_to_f64() {
        let engine = RationalArithmeticEngine::new();
        let r = engine.encode(1, 3);
        let f = engine.to_f64(&r);
        assert!((f - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare() {
        let engine = RationalArithmeticEngine::new();
        let a = engine.encode(1, 3);
        let b = engine.encode(1, 2);
        assert_eq!(engine.compare(&a, &b), std::cmp::Ordering::Less);
    }
}
