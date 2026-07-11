// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! A `f64` complex number with the standard field operations and elementary
//! functions.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number `re + im·i`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    /// `re + im·i`.
    pub fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    /// A purely real number.
    pub fn real(re: f64) -> Complex {
        Complex { re, im: 0.0 }
    }

    /// The imaginary unit `i`.
    pub fn i() -> Complex {
        Complex { re: 0.0, im: 1.0 }
    }

    /// Build from polar form `r·e^{iθ}`.
    pub fn from_polar(r: f64, theta: f64) -> Complex {
        Complex {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// The complex conjugate `re − im·i`.
    pub fn conj(self) -> Complex {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// The modulus (absolute value) `|z|`.
    pub fn modulus(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// The squared modulus `|z|²` (no square root).
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// The argument (phase) in `(−π, π]`.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// The complex exponential `e^z`.
    pub fn exp(self) -> Complex {
        Complex::from_polar(self.re.exp(), self.im)
    }

    /// The principal complex logarithm.
    pub fn ln(self) -> Complex {
        Complex {
            re: self.modulus().ln(),
            im: self.arg(),
        }
    }

    /// The principal square root.
    pub fn sqrt(self) -> Complex {
        Complex::from_polar(self.modulus().sqrt(), self.arg() / 2.0)
    }

    /// The `n`-th roots of unity, `e^{2πik/n}` for `k = 0..n`.
    pub fn roots_of_unity(n: usize) -> Vec<Complex> {
        (0..n)
            .map(|k| Complex::from_polar(1.0, 2.0 * std::f64::consts::PI * k as f64 / n as f64))
            .collect()
    }

    /// Element-wise closeness (for tests / approximate comparison).
    pub fn approx_eq(self, other: Complex, tol: f64) -> bool {
        (self.re - other.re).abs() < tol && (self.im - other.im).abs() < tol
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl Div for Complex {
    type Output = Complex;
    fn div(self, o: Complex) -> Complex {
        let d = o.norm_sqr();
        Complex::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;
    fn mul(self, s: f64) -> Complex {
        Complex::new(self.re * s, self.im * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        assert_eq!(a + b, Complex::new(4.0, 6.0));
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = -5 + 10i.
        assert_eq!(a * b, Complex::new(-5.0, 10.0));
        // Division inverts multiplication.
        assert!((a * b / b).approx_eq(a, 1e-12));
    }

    #[test]
    fn modulus_and_conjugate() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.modulus() - 5.0).abs() < 1e-12);
        // z · conj(z) = |z|².
        assert!((z * z.conj()).approx_eq(Complex::real(25.0), 1e-12));
    }

    #[test]
    fn eulers_identity() {
        // e^{iπ} = −1.
        let z = (Complex::i() * Complex::real(PI)).exp();
        assert!(z.approx_eq(Complex::real(-1.0), 1e-12), "{z:?}");
    }

    #[test]
    fn fourth_roots_of_unity() {
        let r = Complex::roots_of_unity(4);
        assert!(r[0].approx_eq(Complex::new(1.0, 0.0), 1e-12));
        assert!(r[1].approx_eq(Complex::new(0.0, 1.0), 1e-12));
        assert!(r[2].approx_eq(Complex::new(-1.0, 0.0), 1e-12));
        assert!(r[3].approx_eq(Complex::new(0.0, -1.0), 1e-12));
        // Each is a genuine root: zⁿ = 1.
        for z in r {
            let z4 = z * z * z * z;
            assert!(z4.approx_eq(Complex::real(1.0), 1e-12));
        }
    }

    #[test]
    fn sqrt_of_negative_one_is_i() {
        assert!(Complex::real(-1.0).sqrt().approx_eq(Complex::i(), 1e-12));
    }
}
