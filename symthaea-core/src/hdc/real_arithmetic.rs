// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use crate::hdc::unified_hv::ContinuousHV;

/// Real number representation with both rational and floating-point components,
/// along with HDC encodings for symbolic reasoning.
#[derive(Debug, Clone)]
pub struct HdcReal {
    pub rational_numerator: i64,
    pub rational_denominator: i64,
    pub error_bound: f64,
    pub f64_value: f64,
    pub binary_encoding: BinaryHV,
    pub continuous_encoding: ContinuousHV,
    pub phi: f64,
}

impl HdcReal {
    /// Create a new HdcReal from a rational approximation and exact f64 value.
    fn new(num: i64, den: i64, exact_value: f64, seed: u64) -> Self {
        let rational_approx = num as f64 / den as f64;
        let error_bound = (exact_value - rational_approx).abs();

        let binary_encoding = BinaryHV::random(seed);
        let continuous_encoding = binary_encoding.to_continuous();

        // Phi as measure of encoding coherence
        let phi = 1.0 / (1.0 + error_bound);

        Self {
            rational_numerator: num,
            rational_denominator: den,
            error_bound,
            f64_value: exact_value,
            binary_encoding,
            continuous_encoding,
            phi,
        }
    }
}

/// Result of dual-domain computation (exact rational + approximate f64).
#[derive(Debug, Clone)]
pub struct DualResult {
    pub exact_value: Option<(i64, i64)>,
    pub approximate_value: f64,
    pub encoding: BinaryHV,
    pub coherence: f64,
    pub phi: f64,
}

/// Mathematical constants represented as HdcReal.
pub struct MathConstants;

impl MathConstants {
    /// Pi (π) using rational approximation 355/113.
    pub fn pi() -> HdcReal {
        HdcReal::new(355, 113, std::f64::consts::PI, seed_from_name("PI"))
    }

    /// Euler's number (e) using rational approximation 2721/1001.
    pub fn e() -> HdcReal {
        HdcReal::new(2721, 1001, std::f64::consts::E, seed_from_name("E"))
    }

    /// Golden ratio (φ) using Fibonacci approximation 987/610.
    pub fn phi_golden() -> HdcReal {
        let exact_phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        HdcReal::new(987, 610, exact_phi, seed_from_name("PHI"))
    }

    /// Square root of 2 using rational approximation 1393/985.
    pub fn sqrt2() -> HdcReal {
        HdcReal::new(1393, 985, std::f64::consts::SQRT_2, seed_from_name("SQRT2"))
    }
}

/// Bridge between Peano arithmetic (integers) and real number physics.
pub struct PeanoPhysicsBridge;

impl Default for PeanoPhysicsBridge {
    fn default() -> Self {
        Self
    }
}

impl PeanoPhysicsBridge {
    pub fn new() -> Self {
        Self
    }

    /// Convert an integer to f64 with HDC encoding.
    pub fn integer_to_f64(&self, n: i64) -> (f64, BinaryHV) {
        let value = n as f64;
        let seed = seed_from_name(&format!("INT_{n}"));
        let encoding = BinaryHV::random(seed);
        (value, encoding)
    }

    /// Convert a rational number to HdcReal.
    pub fn rational_to_f64(&self, num: i64, den: i64) -> HdcReal {
        let exact_value = num as f64 / den as f64;
        let seed = seed_from_name(&format!("RAT_{num}_{den}"));
        HdcReal::new(num, den, exact_value, seed)
    }

    /// Convert f64 to best rational approximation using continued fractions.
    pub fn f64_to_rational(&self, value: f64, max_denominator: u64) -> (i64, i64) {
        if value.is_nan() || value.is_infinite() {
            return (0, 1);
        }

        let sign = if value < 0.0 { -1 } else { 1 };
        let mut x = value.abs();
        let mut a0 = x.floor() as i64;

        // Initialize convergents
        let mut p = (a0, 1i64);
        let mut q = (1i64, 0i64);

        loop {
            let remainder = x - a0 as f64;
            if remainder < 1e-15 {
                break;
            }

            x = 1.0 / remainder;
            let a = x.floor() as i64;

            let new_p = a * p.0 + q.0;
            let new_q = a * p.1 + q.1;

            if new_q.unsigned_abs() > max_denominator {
                break;
            }

            q = p;
            p = (new_p, new_q);
            a0 = a;

            if (value.abs() - p.0 as f64 / p.1 as f64).abs() < 1e-15 {
                break;
            }
        }

        (sign * p.0, p.1)
    }

    /// Perform dual computation: exact rational (if possible) and f64 approximation.
    pub fn dual_compute(&self, a: f64, b: f64, op: &str) -> DualResult {
        // Check if inputs are small integers for exact computation
        let a_is_int = (a.fract().abs() < 1e-10) && (a.abs() < 1e10);
        let b_is_int = (b.fract().abs() < 1e-10) && (b.abs() < 1e10);

        let exact_value = if a_is_int && b_is_int {
            let a_int = a.round() as i64;
            let b_int = b.round() as i64;

            match op {
                "add" => Some((a_int + b_int, 1)),
                "subtract" => Some((a_int - b_int, 1)),
                "multiply" => Some((a_int * b_int, 1)),
                "divide" if b_int != 0 => {
                    // Simplify if possible
                    let num = a_int;
                    let den = b_int;
                    Some((num, den))
                }
                _ => None,
            }
        } else {
            None
        };

        // Compute approximate value
        let approximate_value = match op {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" if b.abs() > 1e-10 => a / b,
            _ => 0.0,
        };

        // Create encodings for both exact and approximate results
        let approx_encoding =
            BinaryHV::random(seed_from_name(&format!("APPROX_{approximate_value}")));

        let coherence = if let Some((num, den)) = exact_value {
            let exact_result_value = num as f64 / den as f64;
            let exact_encoding = BinaryHV::random(seed_from_name(&format!("EXACT_{num}_{den}")));

            // Measure coherence as similarity between exact and approximate encodings
            let similarity = exact_encoding.similarity(&approx_encoding);

            // Also factor in numerical agreement
            let numerical_agreement = if (exact_result_value - approximate_value).abs() < 1e-10 {
                1.0
            } else {
                0.0
            };

            (similarity as f64 + numerical_agreement) / 2.0
        } else {
            // No exact computation possible, coherence is based on encoding stability
            0.5_f64
        };

        // Phi as measure of computational coherence
        let phi: f64 = if coherence > 0.0 {
            1.0 + coherence
        } else {
            1.0 / (1.0 - coherence.min(0.999))
        };

        DualResult {
            exact_value,
            approximate_value,
            encoding: approx_encoding,
            coherence,
            phi,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pi_constant() {
        let pi = MathConstants::pi();
        assert!((pi.f64_value - std::f64::consts::PI).abs() < 1e-10);
        assert!(pi.error_bound < 0.001); // 355/113 is very good
    }

    #[test]
    fn test_f64_to_rational_third() {
        let bridge = PeanoPhysicsBridge::new();
        let (num, den) = bridge.f64_to_rational(1.0 / 3.0, 1000);
        assert_eq!(num, 1);
        assert_eq!(den, 3);
    }

    #[test]
    fn test_f64_to_rational_pi() {
        let bridge = PeanoPhysicsBridge::new();
        let (num, den) = bridge.f64_to_rational(std::f64::consts::PI, 1000);
        // Should find 355/113 or similar good approximation
        let approx = num as f64 / den as f64;
        assert!((approx - std::f64::consts::PI).abs() < 0.001);
    }

    #[test]
    fn test_dual_compute_add() {
        let bridge = PeanoPhysicsBridge::new();
        let result = bridge.dual_compute(3.0, 5.0, "add");
        assert!((result.approximate_value - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_integer_to_f64() {
        let bridge = PeanoPhysicsBridge::new();
        let (val, _enc) = bridge.integer_to_f64(42);
        assert!((val - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_compute_coherence() {
        let bridge = PeanoPhysicsBridge::new();
        let result = bridge.dual_compute(2.0, 3.0, "multiply");
        assert!((result.approximate_value - 6.0).abs() < 1e-10);
        // Coherence should be positive (exact and approx agree)
        assert!(result.coherence > 0.0);
    }
}
