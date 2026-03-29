// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Q16.16 Fixed-Point Arithmetic
//!
//! Provides deterministic fixed-point arithmetic for zkVM compatibility.
//!
//! # Format
//! - Q16.16: 16-bit signed integer part + 16-bit fractional part
//! - Range: -32768.0 to 32767.99998
//! - Precision: ~0.000015 (2^-16)
//!
//! # Example
//! ```
//! use vsv_core::Fixed;
//!
//! let a = Fixed::from_f32(3.5);
//! let b = Fixed::from_f32(2.0);
//! let c = a + b;
//! assert!((c.to_f32() - 5.5).abs() < 0.001);
//! ```

use std::ops::{Add, Div, Mul, Sub};

/// Q16.16 fixed-point number representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed(i32);

impl Fixed {
    /// Fractional bits (16 bits)
    pub const FRAC_BITS: u32 = 16;

    /// Scaling factor (2^16 = 65536)
    pub const SCALE: i32 = 1 << Self::FRAC_BITS;

    /// Zero constant
    pub const ZERO: Self = Fixed(0);

    /// One constant
    pub const ONE: Self = Fixed(Self::SCALE);

    /// Create from raw i32 representation
    #[inline]
    pub const fn from_raw(raw: i32) -> Self {
        Fixed(raw)
    }

    /// Create from i32 integer (scales by 2^16)
    #[inline]
    pub const fn from_i32(value: i32) -> Self {
        Fixed(value << Self::FRAC_BITS)
    }

    /// Create from usize integer (scales by 2^16)
    #[inline]
    pub fn from_usize(value: usize) -> Self {
        Fixed::from_i32(value as i32)
    }

    /// Get raw i32 representation
    #[inline]
    pub const fn to_raw(self) -> i32 {
        self.0
    }

    /// Convert from f32 to Q16.16
    ///
    /// # Example
    /// ```
    /// use vsv_core::Fixed;
    /// let x = Fixed::from_f32(3.14159);
    /// assert!((x.to_f32() - 3.14159).abs() < 0.001);
    /// ```
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        Fixed((f * Self::SCALE as f32) as i32)
    }

    /// Convert from Q16.16 to f32
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }

    /// ReLU activation: max(0, x)
    ///
    /// # Example
    /// ```
    /// use vsv_core::Fixed;
    /// assert_eq!(Fixed::from_f32(-1.0).relu(), Fixed::ZERO);
    /// assert_eq!(Fixed::from_f32(2.0).relu(), Fixed::from_f32(2.0));
    /// ```
    #[inline]
    pub fn relu(self) -> Self {
        if self.0 > 0 {
            self
        } else {
            Self::ZERO
        }
    }

    /// Maximum of two values
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.0 > other.0 {
            self
        } else {
            other
        }
    }

    /// Minimum of two values
    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.0 < other.0 {
            self
        } else {
            other
        }
    }

    /// Absolute value
    #[inline]
    pub fn abs(self) -> Self {
        Fixed(self.0.abs())
    }
}

// Addition: straightforward since both operands have same scaling
impl Add for Fixed {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Fixed(self.0.saturating_add(other.0))
    }
}

// Subtraction: straightforward since both operands have same scaling
impl Sub for Fixed {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Fixed(self.0.saturating_sub(other.0))
    }
}

// Multiplication: need to adjust for double scaling (Q16.16 * Q16.16 = Q32.32)
impl Mul for Fixed {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        // Use i64 to prevent overflow during multiplication
        let product = (self.0 as i64) * (other.0 as i64);
        // Shift right by FRAC_BITS to get back to Q16.16
        Fixed((product >> Self::FRAC_BITS) as i32)
    }
}

// Division: need to adjust scaling (Q16.16 / Q16.16 = Q0.32 without adjustment)
impl Div for Fixed {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        // Shift numerator left by FRAC_BITS to maintain Q16.16 format
        let numerator = (self.0 as i64) << Self::FRAC_BITS;
        let result = numerator / (other.0 as i64);
        Fixed(result as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.001;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_from_to_f32() {
        let test_cases = vec![
            0.0, 1.0, -1.0, 3.14159, -2.71828, 100.5, -100.5, 0.00001, 32767.0, -32768.0,
        ];

        for &val in &test_cases {
            let fixed = Fixed::from_f32(val);
            let recovered = fixed.to_f32();
            assert!(
                approx_eq(val, recovered, EPSILON),
                "Failed for {}: got {}",
                val,
                recovered
            );
        }
    }

    #[test]
    fn test_add() {
        let test_cases = vec![
            (1.0, 2.0, 3.0),
            (3.5, 2.5, 6.0),
            (-1.0, 1.0, 0.0),
            (100.0, 200.0, 300.0),
            (-50.0, 30.0, -20.0),
        ];

        for (a, b, expected) in test_cases {
            let fa = Fixed::from_f32(a);
            let fb = Fixed::from_f32(b);
            let result = fa + fb;
            assert!(
                approx_eq(result.to_f32(), expected, EPSILON),
                "Failed: {} + {} = {}, expected {}",
                a,
                b,
                result.to_f32(),
                expected
            );
        }
    }

    #[test]
    fn test_sub() {
        let test_cases = vec![
            (5.0, 3.0, 2.0),
            (3.5, 1.5, 2.0),
            (0.0, 1.0, -1.0),
            (100.0, 50.0, 50.0),
            (-10.0, -5.0, -5.0),
        ];

        for (a, b, expected) in test_cases {
            let fa = Fixed::from_f32(a);
            let fb = Fixed::from_f32(b);
            let result = fa - fb;
            assert!(
                approx_eq(result.to_f32(), expected, EPSILON),
                "Failed: {} - {} = {}, expected {}",
                a,
                b,
                result.to_f32(),
                expected
            );
        }
    }

    #[test]
    fn test_mul() {
        let test_cases = vec![
            (2.0, 3.0, 6.0),
            (1.5, 2.0, 3.0),
            (-2.0, 3.0, -6.0),
            (0.5, 0.5, 0.25),
            (10.0, 0.1, 1.0),
        ];

        for (a, b, expected) in test_cases {
            let fa = Fixed::from_f32(a);
            let fb = Fixed::from_f32(b);
            let result = fa * fb;
            assert!(
                approx_eq(result.to_f32(), expected, EPSILON),
                "Failed: {} * {} = {}, expected {}",
                a,
                b,
                result.to_f32(),
                expected
            );
        }
    }

    #[test]
    fn test_div() {
        let test_cases = vec![
            (6.0, 2.0, 3.0, EPSILON),
            (3.0, 1.5, 2.0, EPSILON),
            (-6.0, 3.0, -2.0, EPSILON),
            (1.0, 4.0, 0.25, EPSILON),
            (10.0, 0.1, 100.0, 0.05), // Wider tolerance for 0.1 (not exactly representable)
        ];

        for (a, b, expected, tol) in test_cases {
            let fa = Fixed::from_f32(a);
            let fb = Fixed::from_f32(b);
            let result = fa / fb;
            assert!(
                approx_eq(result.to_f32(), expected, tol),
                "Failed: {} / {} = {}, expected {}",
                a,
                b,
                result.to_f32(),
                expected
            );
        }
    }

    #[test]
    fn test_relu() {
        let test_cases = vec![
            (0.0, 0.0),
            (1.0, 1.0),
            (-1.0, 0.0),
            (5.5, 5.5),
            (-5.5, 0.0),
            (0.001, 0.001),
            (-0.001, 0.0),
        ];

        for (input, expected) in test_cases {
            let fixed = Fixed::from_f32(input);
            let result = fixed.relu();
            assert!(
                approx_eq(result.to_f32(), expected, EPSILON),
                "Failed: relu({}) = {}, expected {}",
                input,
                result.to_f32(),
                expected
            );
        }
    }

    #[test]
    fn test_max() {
        let test_cases = vec![
            (1.0, 2.0, 2.0),
            (5.0, 3.0, 5.0),
            (-1.0, -2.0, -1.0),
            (0.0, 0.0, 0.0),
        ];

        for (a, b, expected) in test_cases {
            let fa = Fixed::from_f32(a);
            let fb = Fixed::from_f32(b);
            let result = fa.max(fb);
            assert!(
                approx_eq(result.to_f32(), expected, EPSILON),
                "Failed: max({}, {}) = {}, expected {}",
                a,
                b,
                result.to_f32(),
                expected
            );
        }
    }

    #[test]
    fn test_constants() {
        assert_eq!(Fixed::ZERO.to_f32(), 0.0);
        assert!(approx_eq(Fixed::ONE.to_f32(), 1.0, EPSILON));
    }

    #[test]
    fn test_edge_cases() {
        // Test near limits of Q16.16 range
        let large_pos = Fixed::from_f32(32767.0);
        let large_neg = Fixed::from_f32(-32768.0);

        assert!(approx_eq(large_pos.to_f32(), 32767.0, 1.0));
        assert!(approx_eq(large_neg.to_f32(), -32768.0, 1.0));

        // Test saturation behavior
        let a = Fixed::from_f32(32000.0);
        let b = Fixed::from_f32(32000.0);
        let sum = a + b; // Should saturate

        // Result should be near max value
        assert!(sum.to_f32() > 32000.0);
    }

    #[test]
    fn test_precision() {
        // Test that precision is approximately 2^-16
        let small = Fixed::from_f32(0.00001);
        assert!(small.to_f32() < 0.0001);
    }

    #[test]
    fn test_from_integer_constructors() {
        let five = Fixed::from_i32(5);
        assert!(approx_eq(five.to_f32(), 5.0, EPSILON));

        let seven = Fixed::from_usize(7);
        assert!(approx_eq(seven.to_f32(), 7.0, EPSILON));
    }
}
