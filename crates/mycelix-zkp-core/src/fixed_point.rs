// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Q16.16 fixed-point arithmetic shared across ZKP backends.
//!
//! Both RISC0 guest programs and Winterfell AIR circuits use this
//! representation to avoid floating-point non-determinism in proofs.

use serde::{Deserialize, Serialize};

/// Q16.16 scale factor: 2^16 = 65536
pub const Q16_16_SCALE: u64 = 65_536;

/// A Q16.16 fixed-point value (16 integer bits, 16 fractional bits).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct FixedPoint(pub i64);

impl FixedPoint {
    /// Create from an f32 value.
    pub fn from_f32(v: f32) -> Self {
        Self((v * Q16_16_SCALE as f32) as i64)
    }

    /// Create from an f64 value.
    pub fn from_f64(v: f64) -> Self {
        Self((v * Q16_16_SCALE as f64) as i64)
    }

    /// Create from a raw scaled integer.
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    /// Convert back to f32.
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Q16_16_SCALE as f32
    }

    /// Convert back to f64.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Q16_16_SCALE as f64
    }

    /// Get the raw scaled integer.
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Fixed-point multiply: (a * b) >> 16
    pub fn mul(self, other: Self) -> Self {
        Self((self.0 as i128 * other.0 as i128 / Q16_16_SCALE as i128) as i64)
    }

    /// Fixed-point divide: (a << 16) / b
    pub fn div(self, other: Self) -> Self {
        if other.0 == 0 {
            return Self(i64::MAX); // Saturate on division by zero
        }
        Self((self.0 as i128 * Q16_16_SCALE as i128 / other.0 as i128) as i64)
    }

    /// Represent 1.0 in Q16.16.
    pub const ONE: Self = Self(Q16_16_SCALE as i64);

    /// Represent 0.0 in Q16.16.
    pub const ZERO: Self = Self(0);
}

impl std::ops::Add for FixedPoint {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl std::ops::Sub for FixedPoint {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

/// Convert a slice of f32 values to Q16.16 fixed-point for proof witnesses.
pub fn f32_slice_to_fixed(values: &[f32]) -> Vec<FixedPoint> {
    values.iter().map(|&v| FixedPoint::from_f32(v)).collect()
}

/// Convert Q16.16 fixed-point values back to f32.
pub fn fixed_to_f32_slice(values: &[FixedPoint]) -> Vec<f32> {
    values.iter().map(|v| v.to_f32()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_f32() {
        let v = 0.85f32;
        let fp = FixedPoint::from_f32(v);
        let back = fp.to_f32();
        assert!(
            (v - back).abs() < 0.001,
            "roundtrip: {} -> {} -> {}",
            v,
            fp.0,
            back
        );
    }

    #[test]
    fn test_fixed_multiply() {
        let a = FixedPoint::from_f32(0.5);
        let b = FixedPoint::from_f32(0.8);
        let c = a.mul(b);
        assert!((c.to_f32() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_fixed_divide() {
        let a = FixedPoint::from_f32(1.0);
        let b = FixedPoint::from_f32(2.0);
        let c = a.div(b);
        assert!((c.to_f32() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_one_and_zero() {
        assert_eq!(FixedPoint::ONE.to_f32(), 1.0);
        assert_eq!(FixedPoint::ZERO.to_f32(), 0.0);
    }

    #[test]
    fn test_slice_conversion() {
        let values = vec![0.1f32, 0.5, 0.9, 1.0];
        let fixed = f32_slice_to_fixed(&values);
        let back = fixed_to_f32_slice(&fixed);
        for (orig, conv) in values.iter().zip(back.iter()) {
            assert!((orig - conv).abs() < 0.001);
        }
    }
}
