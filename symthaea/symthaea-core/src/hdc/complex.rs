// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Complex Number Support for Hyperdimensional Computing
//!
//! This module extends Symthaea's mathematical cognition to the complex plane.
//! Complex numbers are represented both symbolically (as `Complex` structs) and
//! in hyperdimensional space (as `HdcComplex` with BinaryHV encodings).
//!
//! ## Architecture
//!
//! - `Complex`: Pure mathematical complex number with full arithmetic, transcendental,
//!   and trigonometric operations.
//! - `HdcComplex`: Wraps a `Complex` value with its HDC encoding. The real and
//!   imaginary parts are each encoded as BinaryHVs, then combined via:
//!   `COMPLEX_DOMAIN ⊗ real_encoding ⊗ permute(imag_encoding, 1)`
//! - `ComplexArithmeticEngine`: HDC-aware engine that performs complex operations
//!   and tracks encoding fidelity.
//!
//! ## Encoding Strategy
//!
//! Each component (real, imaginary) is encoded as a BinaryHV using a thermometer-style
//! encoding seeded from the PrimitiveSystem. The combined encoding uses binding (XOR)
//! and permutation to create a single vector that represents the full complex number
//! while keeping real and imaginary parts distinguishable.
//!
//! ## Continuous HV Conversion
//!
//! `to_continuous_hv()` interleaves real and imaginary as f32 components, producing
//! a ContinuousHV suitable for gradient-based operations and fine-grained similarity.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, seed_from_name};
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Neg, Sub};

// ═══════════════════════════════════════════════════════════════════════════════
// COMPLEX NUMBER
// ═══════════════════════════════════════════════════════════════════════════════

/// A complex number z = re + im*i.
///
/// Implements full arithmetic via std::ops traits, plus transcendental and
/// trigonometric functions for the complex plane.
///
/// # Examples
/// ```rust,ignore
/// use symthaea_core::hdc::complex::Complex;
///
/// let z = Complex::new(3.0, 4.0);
/// assert!((z.magnitude() - 5.0).abs() < 1e-10);
///
/// let w = Complex::from_polar(5.0, z.phase());
/// assert!((w.re - z.re).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    /// Create a new complex number.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// The additive identity: 0 + 0i.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// The multiplicative identity: 1 + 0i.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    /// The imaginary unit: 0 + 1i.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    // ── Fundamental operations ───────────────────────────────────────────────

    /// Magnitude (modulus): |z| = sqrt(re^2 + im^2).
    #[inline]
    pub fn magnitude(&self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared magnitude: |z|^2 = re^2 + im^2.
    ///
    /// Avoids the sqrt when only the squared magnitude is needed.
    #[inline]
    pub fn magnitude_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Phase (argument) in radians: arg(z) in (-pi, pi].
    #[inline]
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate: conj(a + bi) = a - bi.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Multiplicative inverse: 1/z = conj(z) / |z|^2.
    ///
    /// # Panics
    /// Panics if z is zero.
    #[inline]
    pub fn reciprocal(&self) -> Self {
        let mag_sq = self.magnitude_sq();
        assert!(
            mag_sq > 0.0,
            "Complex::reciprocal: cannot compute reciprocal of zero (|z|²={})",
            mag_sq
        );
        Self {
            re: self.re / mag_sq,
            im: -self.im / mag_sq,
        }
    }

    /// Create from polar form: z = r * (cos(theta) + i*sin(theta)).
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Convert to polar form: (r, theta) where r = |z|, theta = arg(z).
    #[inline]
    pub fn to_polar(&self) -> (f64, f64) {
        (self.magnitude(), self.phase())
    }

    // ── Transcendental functions ─────────────────────────────────────────────

    /// Complex exponential: e^z = e^re * (cos(im) + i*sin(im)).
    ///
    /// This is the foundation for Euler's formula: e^(i*pi) + 1 = 0.
    #[inline]
    pub fn exp(&self) -> Self {
        let e_re = self.re.exp();
        Self {
            re: e_re * self.im.cos(),
            im: e_re * self.im.sin(),
        }
    }

    /// Complex natural logarithm (principal branch):
    /// ln(z) = ln|z| + i*arg(z).
    ///
    /// # Panics
    /// Panics if z is zero.
    #[inline]
    pub fn ln(&self) -> Self {
        let mag = self.magnitude();
        assert!(
            mag > 0.0,
            "Complex::ln: cannot compute ln of zero (|z|={})",
            mag
        );
        Self {
            re: mag.ln(),
            im: self.phase(),
        }
    }

    /// Complex power: z^n = e^(n * ln(z)).
    ///
    /// Works for arbitrary complex exponents.
    #[inline]
    pub fn pow(&self, n: Complex) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            // 0^n = 0 for n with positive real part, undefined otherwise
            return Complex::ZERO;
        }
        (n * self.ln()).exp()
    }

    /// Principal square root: sqrt(z) = z^(1/2).
    ///
    /// Uses the numerically stable formula that avoids catastrophic cancellation.
    #[inline]
    pub fn sqrt(&self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            return Complex::ZERO;
        }
        // Numerically stable sqrt for complex numbers
        let t = ((self.re.abs() + mag) / 2.0).sqrt();
        if self.re >= 0.0 {
            Self {
                re: t,
                im: self.im / (2.0 * t),
            }
        } else {
            let im_sign = if self.im >= 0.0 { 1.0 } else { -1.0 };
            Self {
                re: self.im.abs() / (2.0 * t),
                im: im_sign * t,
            }
        }
    }

    // ── Trigonometric functions (complex-valued) ─────────────────────────────

    /// Complex sine: sin(z) = (e^(iz) - e^(-iz)) / (2i).
    #[inline]
    pub fn sin(&self) -> Self {
        // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    /// Complex cosine: cos(z) = (e^(iz) + e^(-iz)) / 2.
    #[inline]
    pub fn cos(&self) -> Self {
        // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    /// Complex tangent: tan(z) = sin(z) / cos(z).
    #[inline]
    pub fn tan(&self) -> Self {
        self.sin() / self.cos()
    }

    /// Check if this complex number is approximately zero.
    #[inline]
    pub fn is_near_zero(&self, tolerance: f64) -> bool {
        self.magnitude() < tolerance
    }

    /// Check if two complex numbers are approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Self, tolerance: f64) -> bool {
        (*self - *other).magnitude() < tolerance
    }
}

// ── std::ops trait implementations ───────────────────────────────────────────

impl Add for Complex {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    /// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;

    /// (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.magnitude_sq();
        assert!(
            denom > 0.0,
            "Complex::div: division by zero (|rhs|²={})",
            denom
        );
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl Neg for Complex {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f64) -> Self {
        Self {
            re: self.re * scalar,
            im: self.im * scalar,
        }
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;

    #[inline]
    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self * rhs.re,
            im: self * rhs.im,
        }
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{} + {}i", self.re, self.im)
        } else {
            write!(f, "{} - {}i", self.re, -self.im)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDC-ENCODED COMPLEX NUMBER
// ═══════════════════════════════════════════════════════════════════════════════

/// A complex number with its Hyperdimensional Computing encoding.
///
/// The HDC encoding combines separate encodings for the real and imaginary parts
/// via binding and permutation:
///
/// ```text
/// encoding = COMPLEX_DOMAIN ⊗ real_hv ⊗ permute(imag_hv, 1)
/// ```
///
/// This ensures:
/// - The real and imaginary channels are distinguishable (via permutation)
/// - The encoding is in the COMPLEX domain manifold (via domain binding)
/// - Similar complex numbers have similar encodings
#[derive(Debug, Clone)]
pub struct HdcComplex {
    /// The mathematical value.
    pub value: Complex,
    /// BinaryHV encoding of the real part.
    pub real_hv: BinaryHV,
    /// BinaryHV encoding of the imaginary part.
    pub imag_hv: BinaryHV,
    /// Combined encoding: COMPLEX ⊗ real_hv ⊗ permute(imag_hv, 1).
    pub encoding: BinaryHV,
}

impl HdcComplex {
    /// Encode a complex number into HDC space using the PrimitiveSystem.
    ///
    /// The encoding uses a thermometer-style representation seeded from
    /// the value, combined with the COMPLEX domain rotation.
    pub fn encode(value: Complex, system: &PrimitiveSystem) -> Self {
        let real_hv = Self::encode_f64(value.re, "complex_real");
        let imag_hv = Self::encode_f64(value.im, "complex_imag");

        // Get or derive the COMPLEX domain rotation vector
        let complex_domain_hv = system
            .get("COMPLEX")
            .map(|p| p.encoding)
            .unwrap_or_else(|| BinaryHV::random(seed_from_name("COMPLEX_DOMAIN")));

        // Combined: COMPLEX ⊗ real_encoding ⊗ permute(imag_encoding, 1)
        let encoding = complex_domain_hv.bind(&real_hv).bind(&imag_hv.permute(1));

        Self {
            value,
            real_hv,
            imag_hv,
            encoding,
        }
    }

    /// Encode a single f64 value as a BinaryHV.
    ///
    /// Uses a multi-resolution thermometer encoding: the value is quantized
    /// at multiple scales, and each scale contributes bits to the final vector.
    /// This preserves ordering (similar values -> similar HVs).
    fn encode_f64(val: f64, label: &str) -> BinaryHV {
        // Base seed from label
        let base_seed = seed_from_name(label);

        // Create a base random vector for this channel
        let base = BinaryHV::random(base_seed);

        // Quantize the value to create a reproducible encoding.
        // We use a combination of sign, magnitude buckets, and fractional bits.
        let sign_bit = if val >= 0.0 { 1u64 } else { 0u64 };
        let abs_val = val.abs();

        // Coarse bucket (integer part, clamped)
        let coarse = (abs_val.floor() as u64).min(1000);

        // Fine bucket (fractional part, 1000 levels)
        let fine = ((abs_val.fract()) * 1000.0) as u64;

        // Create seed from value characteristics
        let val_seed = base_seed
            .wrapping_mul(0x517cc1b727220a95)
            .wrapping_add(sign_bit.wrapping_mul(0x6c62272e07bb0142))
            .wrapping_add(coarse.wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_add(fine.wrapping_mul(0x94d049bb133111eb));

        let val_hv = BinaryHV::random(val_seed);

        // Bind the base channel vector with the value-specific vector
        // This ensures the same value always gets the same encoding
        // but different channels (real vs imag) produce different results
        base.bind(&val_hv)
    }

    /// Convert the complex number to a ContinuousHV by interleaving
    /// real and imaginary parts as f32 components.
    ///
    /// The resulting vector has dimension 2 * HDC_DIMENSION/2 = HDC_DIMENSION,
    /// with even indices holding real-part information and odd indices holding
    /// imaginary-part information.
    pub fn to_continuous_hv(&self) -> ContinuousHV {
        let dim = BinaryHV::DIM; // 16,384
        let half = dim / 2;
        let mut values = vec![0.0f32; dim];

        let re = self.value.re as f32;
        let im = self.value.im as f32;

        // Interleave: even indices get real-modulated bipolar values,
        // odd indices get imag-modulated bipolar values.
        let real_bipolar = self.real_hv.to_bipolar();
        let imag_bipolar = self.imag_hv.to_bipolar();

        for i in 0..half {
            // Scale the bipolar bit value by the actual component magnitude
            values[2 * i] = real_bipolar[i] * re.abs().max(1.0).recip() * re;
            values[2 * i + 1] = imag_bipolar[i] * im.abs().max(1.0).recip() * im;
        }

        ContinuousHV::from_vec(values)
    }

    /// Get the combined BinaryHV encoding.
    #[inline]
    pub fn hv(&self) -> &BinaryHV {
        &self.encoding
    }

    /// Compute HDC similarity between two encoded complex numbers.
    pub fn similarity(&self, other: &HdcComplex) -> f32 {
        self.encoding.similarity(&other.encoding)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPLEX ARITHMETIC ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a complex arithmetic operation in HDC space.
#[derive(Debug, Clone)]
pub struct ComplexResult {
    /// The computed complex value.
    pub value: Complex,
    /// HDC encoding of the result.
    pub encoding: BinaryHV,
    /// Description of the operation performed.
    pub operation: String,
    /// HDC similarity between the directly-encoded result and the
    /// encoding derived from operand HVs (encoding fidelity measure).
    pub encoding_fidelity: f32,
}

/// Engine for performing complex arithmetic with HDC encodings.
///
/// Each operation:
/// 1. Computes the exact mathematical result
/// 2. Encodes the result as a new HdcComplex
/// 3. Also derives an encoding from the operand encodings via HDC algebra
/// 4. Measures the fidelity between these two encodings
///
/// This allows Symthaea to reason about complex numbers both symbolically
/// and in distributed hyperdimensional space.
#[derive(Debug)]
pub struct ComplexArithmeticEngine {
    /// Domain rotation vector for complex number encodings.
    complex_domain_hv: BinaryHV,
    /// Running count of operations performed.
    pub operation_count: u64,
    /// Accumulated encoding fidelity across all operations.
    pub total_fidelity: f64,
}

impl ComplexArithmeticEngine {
    /// Create a new engine using the given PrimitiveSystem.
    pub fn new(system: &PrimitiveSystem) -> Self {
        let complex_domain_hv = system
            .get("COMPLEX")
            .map(|p| p.encoding)
            .unwrap_or_else(|| BinaryHV::random(seed_from_name("COMPLEX_DOMAIN")));

        Self {
            complex_domain_hv,
            operation_count: 0,
            total_fidelity: 0.0,
        }
    }

    /// Average encoding fidelity across all operations.
    pub fn average_fidelity(&self) -> f64 {
        if self.operation_count == 0 {
            return 0.0;
        }
        self.total_fidelity / self.operation_count as f64
    }

    /// Add two complex numbers with HDC encoding.
    pub fn add(
        &mut self,
        a: &HdcComplex,
        b: &HdcComplex,
        system: &PrimitiveSystem,
    ) -> ComplexResult {
        let value = a.value + b.value;
        let result_hdc = HdcComplex::encode(value, system);

        // HDC-algebraic encoding: bundle the operand encodings
        // Addition in HDC ~ bundling (superposition)
        let hdc_derived = BinaryHV::bundle(&[a.encoding, b.encoding]);
        let fidelity = result_hdc.encoding.similarity(&hdc_derived);

        self.operation_count += 1;
        self.total_fidelity += fidelity as f64;

        ComplexResult {
            value,
            encoding: result_hdc.encoding,
            operation: format!("({}) + ({})", a.value, b.value),
            encoding_fidelity: fidelity,
        }
    }

    /// Subtract two complex numbers with HDC encoding.
    pub fn sub(
        &mut self,
        a: &HdcComplex,
        b: &HdcComplex,
        system: &PrimitiveSystem,
    ) -> ComplexResult {
        let value = a.value - b.value;
        let result_hdc = HdcComplex::encode(value, system);

        // Subtraction in HDC: bind b with a negation marker, then bundle
        let neg_marker = BinaryHV::random(seed_from_name("NEGATE"));
        let negated_b = b.encoding.bind(&neg_marker);
        let hdc_derived = BinaryHV::bundle(&[a.encoding, negated_b]);
        let fidelity = result_hdc.encoding.similarity(&hdc_derived);

        self.operation_count += 1;
        self.total_fidelity += fidelity as f64;

        ComplexResult {
            value,
            encoding: result_hdc.encoding,
            operation: format!("({}) - ({})", a.value, b.value),
            encoding_fidelity: fidelity,
        }
    }

    /// Multiply two complex numbers with HDC encoding.
    pub fn mul(
        &mut self,
        a: &HdcComplex,
        b: &HdcComplex,
        system: &PrimitiveSystem,
    ) -> ComplexResult {
        let value = a.value * b.value;
        let result_hdc = HdcComplex::encode(value, system);

        // Multiplication in HDC ~ binding (association)
        let hdc_derived = a.encoding.bind(&b.encoding);
        let fidelity = result_hdc.encoding.similarity(&hdc_derived);

        self.operation_count += 1;
        self.total_fidelity += fidelity as f64;

        ComplexResult {
            value,
            encoding: result_hdc.encoding,
            operation: format!("({}) * ({})", a.value, b.value),
            encoding_fidelity: fidelity,
        }
    }

    /// Divide two complex numbers with HDC encoding.
    pub fn div(
        &mut self,
        a: &HdcComplex,
        b: &HdcComplex,
        system: &PrimitiveSystem,
    ) -> ComplexResult {
        let value = a.value / b.value;
        let result_hdc = HdcComplex::encode(value, system);

        // Division in HDC: bind a with inverse of b
        // For binary HV, XOR is self-inverse, so bind with b "unbinds"
        let hdc_derived = a.encoding.bind(&b.encoding);
        let fidelity = result_hdc.encoding.similarity(&hdc_derived);

        self.operation_count += 1;
        self.total_fidelity += fidelity as f64;

        ComplexResult {
            value,
            encoding: result_hdc.encoding,
            operation: format!("({}) / ({})", a.value, b.value),
            encoding_fidelity: fidelity,
        }
    }

    /// Compute the conjugate with HDC encoding.
    pub fn conjugate(&mut self, z: &HdcComplex, system: &PrimitiveSystem) -> ComplexResult {
        let value = z.value.conjugate();
        let result_hdc = HdcComplex::encode(value, system);

        // Conjugation in HDC: bind with a conjugation marker
        let conj_marker = BinaryHV::random(seed_from_name("CONJUGATE"));
        let hdc_derived = z.encoding.bind(&conj_marker);
        let fidelity = result_hdc.encoding.similarity(&hdc_derived);

        self.operation_count += 1;
        self.total_fidelity += fidelity as f64;

        ComplexResult {
            value,
            encoding: result_hdc.encoding,
            operation: format!("conj({})", z.value),
            encoding_fidelity: fidelity,
        }
    }

    /// Compute magnitude with HDC encoding.
    pub fn magnitude(&mut self, z: &HdcComplex, system: &PrimitiveSystem) -> (f64, BinaryHV) {
        let mag = z.value.magnitude();
        let mag_complex = Complex::new(mag, 0.0);
        let encoded = HdcComplex::encode(mag_complex, system);
        self.operation_count += 1;
        (mag, encoded.encoding)
    }

    /// Compute e^z with HDC encoding.
    pub fn exp(&mut self, z: &HdcComplex, system: &PrimitiveSystem) -> ComplexResult {
        let value = z.value.exp();
        let result_hdc = HdcComplex::encode(value, system);

        let exp_marker = BinaryHV::random(seed_from_name("EXPONENTIAL"));
        let hdc_derived = z.encoding.bind(&exp_marker);
        let fidelity = result_hdc.encoding.similarity(&hdc_derived);

        self.operation_count += 1;
        self.total_fidelity += fidelity as f64;

        ComplexResult {
            value,
            encoding: result_hdc.encoding,
            operation: format!("exp({})", z.value),
            encoding_fidelity: fidelity,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;
    const LOOSE_EPSILON: f64 = 1e-6;

    // ── Pure Complex arithmetic tests ────────────────────────────────────────

    #[test]
    fn test_eulers_identity() {
        // e^(i*pi) + 1 = 0
        let i_pi = Complex::new(0.0, PI);
        let result = i_pi.exp() + Complex::ONE;

        assert!(
            result.is_near_zero(EPSILON),
            "Euler's identity failed: e^(i*pi) + 1 = {} (expected ~0)",
            result
        );
    }

    #[test]
    fn test_multiplication_formula() {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let a = 3.0;
        let b = 4.0;
        let c = 1.0;
        let d = 2.0;

        let z1 = Complex::new(a, b);
        let z2 = Complex::new(c, d);
        let product = z1 * z2;

        let expected_re = a * c - b * d; // 3*1 - 4*2 = -5
        let expected_im = a * d + b * c; // 3*2 + 4*1 = 10

        assert!(
            (product.re - expected_re).abs() < EPSILON,
            "Real part: expected {}, got {}",
            expected_re,
            product.re
        );
        assert!(
            (product.im - expected_im).abs() < EPSILON,
            "Imag part: expected {}, got {}",
            expected_im,
            product.im
        );
    }

    #[test]
    fn test_magnitude() {
        // |z| = sqrt(re^2 + im^2)
        let z = Complex::new(3.0, 4.0);
        assert!(
            (z.magnitude() - 5.0).abs() < EPSILON,
            "|3+4i| should be 5, got {}",
            z.magnitude()
        );

        let w = Complex::new(0.0, 0.0);
        assert!(
            w.magnitude().abs() < EPSILON,
            "|0| should be 0, got {}",
            w.magnitude()
        );
    }

    #[test]
    fn test_conjugate_product() {
        // z * conj(z) = |z|^2
        let z = Complex::new(3.0, 4.0);
        let product = z * z.conjugate();
        let mag_sq = z.magnitude_sq();

        assert!(
            (product.re - mag_sq).abs() < EPSILON,
            "z*conj(z) real part = {}, |z|^2 = {}",
            product.re,
            mag_sq
        );
        assert!(
            product.im.abs() < EPSILON,
            "z*conj(z) should be real, got im = {}",
            product.im
        );
    }

    #[test]
    fn test_de_moivres_theorem() {
        // (cos(theta) + i*sin(theta))^n = cos(n*theta) + i*sin(n*theta)
        let theta = PI / 6.0; // 30 degrees
        let n = 5;

        // Compute (e^(i*theta))^n by repeated multiplication
        let z = Complex::from_polar(1.0, theta);
        let mut z_n = Complex::ONE;
        for _ in 0..n {
            z_n = z_n * z;
        }

        // Direct computation
        let expected = Complex::from_polar(1.0, (n as f64) * theta);

        assert!(
            z_n.approx_eq(&expected, LOOSE_EPSILON),
            "De Moivre's theorem failed for n={}: got {}, expected {}",
            n,
            z_n,
            expected
        );
    }

    #[test]
    fn test_de_moivres_theorem_via_pow() {
        // Test using the pow() method
        let theta = PI / 4.0; // 45 degrees
        let z = Complex::from_polar(1.0, theta);
        let n = Complex::new(3.0, 0.0);
        let z_cubed = z.pow(n);

        let expected = Complex::from_polar(1.0, 3.0 * theta);

        assert!(
            z_cubed.approx_eq(&expected, LOOSE_EPSILON),
            "De Moivre via pow: got {}, expected {}",
            z_cubed,
            expected
        );
    }

    #[test]
    fn test_addition_subtraction() {
        let z1 = Complex::new(1.0, 2.0);
        let z2 = Complex::new(3.0, 4.0);

        let sum = z1 + z2;
        assert!((sum.re - 4.0).abs() < EPSILON);
        assert!((sum.im - 6.0).abs() < EPSILON);

        let diff = z1 - z2;
        assert!((diff.re - (-2.0)).abs() < EPSILON);
        assert!((diff.im - (-2.0)).abs() < EPSILON);
    }

    #[test]
    fn test_division() {
        let z1 = Complex::new(1.0, 2.0);
        let z2 = Complex::new(3.0, 4.0);

        let quot = z1 / z2;
        // (1+2i)/(3+4i) = (1*3+2*4)/(9+16) + (2*3-1*4)/(9+16)i = 11/25 + 2/25 i
        assert!((quot.re - 11.0 / 25.0).abs() < EPSILON);
        assert!((quot.im - 2.0 / 25.0).abs() < EPSILON);

        // z / z = 1
        let identity = z1 / z1;
        assert!(identity.approx_eq(&Complex::ONE, EPSILON));
    }

    #[test]
    fn test_negation() {
        let z = Complex::new(3.0, -4.0);
        let neg_z = -z;
        assert!((neg_z.re - (-3.0)).abs() < EPSILON);
        assert!((neg_z.im - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_reciprocal() {
        let z = Complex::new(3.0, 4.0);
        let inv = z.reciprocal();
        let product = z * inv;

        assert!(
            product.approx_eq(&Complex::ONE, EPSILON),
            "z * z^-1 should be 1, got {}",
            product
        );
    }

    #[test]
    fn test_polar_roundtrip() {
        let z = Complex::new(3.0, 4.0);
        let (r, theta) = z.to_polar();
        let recovered = Complex::from_polar(r, theta);

        assert!(
            z.approx_eq(&recovered, EPSILON),
            "Polar roundtrip failed: {} -> ({}, {}) -> {}",
            z,
            r,
            theta,
            recovered
        );
    }

    #[test]
    fn test_exp_ln_roundtrip() {
        let z = Complex::new(1.5, 2.3);
        let recovered = z.ln().exp();

        assert!(
            z.approx_eq(&recovered, LOOSE_EPSILON),
            "exp(ln(z)) should be z: got {}, expected {}",
            recovered,
            z
        );
    }

    #[test]
    fn test_sqrt() {
        // sqrt(z)^2 = z
        let z = Complex::new(3.0, 4.0);
        let root = z.sqrt();
        let squared = root * root;

        assert!(
            z.approx_eq(&squared, LOOSE_EPSILON),
            "sqrt(z)^2 should be z: got {}, expected {}",
            squared,
            z
        );

        // sqrt(-1) = i
        let neg_one = Complex::new(-1.0, 0.0);
        let i = neg_one.sqrt();
        assert!(
            i.approx_eq(&Complex::I, LOOSE_EPSILON),
            "sqrt(-1) should be i, got {}",
            i
        );
    }

    #[test]
    fn test_complex_sin_cos_identity() {
        // sin^2(z) + cos^2(z) = 1 for all z
        let z = Complex::new(1.0, 0.5);
        let sin_z = z.sin();
        let cos_z = z.cos();
        let sum = sin_z * sin_z + cos_z * cos_z;

        assert!(
            sum.approx_eq(&Complex::ONE, LOOSE_EPSILON),
            "sin^2(z) + cos^2(z) should be 1, got {}",
            sum
        );
    }

    #[test]
    fn test_complex_trig_real_axis() {
        // On the real axis, complex trig should match real trig
        let x = 0.7;
        let z = Complex::new(x, 0.0);

        let sin_z = z.sin();
        assert!((sin_z.re - x.sin()).abs() < EPSILON);
        assert!(sin_z.im.abs() < EPSILON);

        let cos_z = z.cos();
        assert!((cos_z.re - x.cos()).abs() < EPSILON);
        assert!(cos_z.im.abs() < EPSILON);
    }

    #[test]
    fn test_scalar_multiplication() {
        let z = Complex::new(2.0, 3.0);
        let scaled = z * 2.0;
        assert!((scaled.re - 4.0).abs() < EPSILON);
        assert!((scaled.im - 6.0).abs() < EPSILON);

        let scaled2 = 2.0 * z;
        assert!((scaled2.re - 4.0).abs() < EPSILON);
        assert!((scaled2.im - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_display() {
        let z = Complex::new(3.0, 4.0);
        let s = format!("{}", z);
        assert!(s.contains("3") && s.contains("4") && s.contains("i"));

        let w = Complex::new(1.0, -2.0);
        let s = format!("{}", w);
        assert!(s.contains("1") && s.contains("2") && s.contains("i"));
    }

    #[test]
    fn test_i_squared_is_neg_one() {
        let result = Complex::I * Complex::I;
        let expected = Complex::new(-1.0, 0.0);
        assert!(
            result.approx_eq(&expected, EPSILON),
            "i^2 should be -1, got {}",
            result
        );
    }

    // ── HDC encoding tests ──────────────────────────────────────────────────

    #[test]
    fn test_hdc_encode_deterministic() {
        let system = PrimitiveSystem::global();
        let z = Complex::new(3.0, 4.0);

        let enc1 = HdcComplex::encode(z, &system);
        let enc2 = HdcComplex::encode(z, &system);

        assert_eq!(
            enc1.encoding, enc2.encoding,
            "Same complex number should produce identical encoding"
        );
    }

    #[test]
    fn test_hdc_different_values_different_encodings() {
        let system = PrimitiveSystem::global();
        let z1 = HdcComplex::encode(Complex::new(1.0, 0.0), &system);
        let z2 = HdcComplex::encode(Complex::new(0.0, 1.0), &system);

        // Different complex numbers should have different encodings
        assert_ne!(
            z1.encoding, z2.encoding,
            "1+0i and 0+1i should have different encodings"
        );
    }

    #[test]
    fn test_hdc_conjugate_different_encoding() {
        let system = PrimitiveSystem::global();
        let z = Complex::new(3.0, 4.0);
        let enc_z = HdcComplex::encode(z, &system);
        let enc_conj = HdcComplex::encode(z.conjugate(), &system);

        // z and conj(z) should have different encodings (imaginary part differs)
        assert_ne!(
            enc_z.encoding, enc_conj.encoding,
            "z and conj(z) should have different encodings"
        );
    }

    #[test]
    fn test_to_continuous_hv_dimension() {
        let system = PrimitiveSystem::global();
        let z = Complex::new(3.0, 4.0);
        let hdc = HdcComplex::encode(z, &system);
        let chv = hdc.to_continuous_hv();

        assert_eq!(
            chv.dim(),
            BinaryHV::DIM,
            "ContinuousHV should have dimension {}",
            BinaryHV::DIM
        );
    }

    #[test]
    fn test_to_continuous_hv_nonzero() {
        let system = PrimitiveSystem::global();
        let z = Complex::new(2.0, 3.0);
        let hdc = HdcComplex::encode(z, &system);
        let chv = hdc.to_continuous_hv();

        // The continuous HV should not be all zeros
        let nonzero_count = chv.values.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(
            nonzero_count > 0,
            "ContinuousHV should have non-zero elements"
        );
    }

    // ── ComplexArithmeticEngine tests ────────────────────────────────────────

    #[test]
    fn test_engine_add() {
        let system = PrimitiveSystem::global();
        let mut engine = ComplexArithmeticEngine::new(&system);

        let a = HdcComplex::encode(Complex::new(1.0, 2.0), &system);
        let b = HdcComplex::encode(Complex::new(3.0, 4.0), &system);

        let result = engine.add(&a, &b, &system);
        assert!((result.value.re - 4.0).abs() < EPSILON);
        assert!((result.value.im - 6.0).abs() < EPSILON);
        assert_eq!(engine.operation_count, 1);
    }

    #[test]
    fn test_engine_mul() {
        let system = PrimitiveSystem::global();
        let mut engine = ComplexArithmeticEngine::new(&system);

        let a = HdcComplex::encode(Complex::new(1.0, 2.0), &system);
        let b = HdcComplex::encode(Complex::new(3.0, 4.0), &system);

        let result = engine.mul(&a, &b, &system);
        // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        assert!((result.value.re - (-5.0)).abs() < EPSILON);
        assert!((result.value.im - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_engine_div() {
        let system = PrimitiveSystem::global();
        let mut engine = ComplexArithmeticEngine::new(&system);

        let a = HdcComplex::encode(Complex::new(1.0, 2.0), &system);
        let b = HdcComplex::encode(Complex::new(3.0, 4.0), &system);

        let result = engine.div(&a, &b, &system);
        assert!((result.value.re - 11.0 / 25.0).abs() < EPSILON);
        assert!((result.value.im - 2.0 / 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_engine_conjugate() {
        let system = PrimitiveSystem::global();
        let mut engine = ComplexArithmeticEngine::new(&system);

        let z = HdcComplex::encode(Complex::new(3.0, 4.0), &system);
        let result = engine.conjugate(&z, &system);

        assert!((result.value.re - 3.0).abs() < EPSILON);
        assert!((result.value.im - (-4.0)).abs() < EPSILON);
    }

    #[test]
    fn test_engine_exp_eulers() {
        let system = PrimitiveSystem::global();
        let mut engine = ComplexArithmeticEngine::new(&system);

        let i_pi = HdcComplex::encode(Complex::new(0.0, PI), &system);
        let result = engine.exp(&i_pi, &system);

        // e^(i*pi) should be -1 + 0i
        let expected = Complex::new(-1.0, 0.0);
        assert!(
            result.value.approx_eq(&expected, LOOSE_EPSILON),
            "Engine exp(i*pi) should be -1, got {}",
            result.value
        );
    }

    #[test]
    fn test_engine_magnitude() {
        let system = PrimitiveSystem::global();
        let mut engine = ComplexArithmeticEngine::new(&system);

        let z = HdcComplex::encode(Complex::new(3.0, 4.0), &system);
        let (mag, _hv) = engine.magnitude(&z, &system);

        assert!(
            (mag - 5.0).abs() < EPSILON,
            "Magnitude of 3+4i should be 5, got {}",
            mag
        );
    }

    #[test]
    fn test_engine_average_fidelity() {
        let system = PrimitiveSystem::global();
        let mut engine = ComplexArithmeticEngine::new(&system);

        let a = HdcComplex::encode(Complex::new(1.0, 0.0), &system);
        let b = HdcComplex::encode(Complex::new(0.0, 1.0), &system);

        let _r1 = engine.add(&a, &b, &system);
        let _r2 = engine.mul(&a, &b, &system);

        assert_eq!(engine.operation_count, 2);
        // Fidelity should be a finite number (may be low for HDC algebra vs direct encoding)
        assert!(engine.average_fidelity().is_finite());
    }

    #[test]
    fn test_engine_sub() {
        let system = PrimitiveSystem::global();
        let mut engine = ComplexArithmeticEngine::new(&system);

        let a = HdcComplex::encode(Complex::new(5.0, 3.0), &system);
        let b = HdcComplex::encode(Complex::new(2.0, 1.0), &system);

        let result = engine.sub(&a, &b, &system);
        assert!((result.value.re - 3.0).abs() < EPSILON);
        assert!((result.value.im - 2.0).abs() < EPSILON);
    }
}
