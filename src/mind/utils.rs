// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Utility functions for the mind system.

use symthaea_core::hdc::ContinuousHV;

/// Floating point epsilon for comparisons
pub const EPSILON: f64 = 1e-10;
pub const EPSILON_F32: f32 = 1e-6;

/// Check if two f64 values are approximately equal within epsilon tolerance.
#[inline]
pub fn float_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

/// Check if two f32 values are approximately equal within epsilon tolerance.
#[inline]
pub fn float_eq_f32(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON_F32
}

/// Check if an f64 value is approximately zero.
#[inline]
pub fn is_zero(v: f64) -> bool {
    v.abs() < EPSILON
}

/// Check if an f32 value is approximately zero.
#[inline]
pub fn is_zero_f32(v: f32) -> bool {
    v.abs() < EPSILON_F32
}

/// Check if an f64 value is non-zero (greater than epsilon).
#[inline]
pub fn is_nonzero(v: f64) -> bool {
    v.abs() >= EPSILON
}

/// Check if an f32 value is non-zero (greater than epsilon).
#[inline]
pub fn is_nonzero_f32(v: f32) -> bool {
    v.abs() >= EPSILON_F32
}

/// Permute (circular shift) a ContinuousHV vector to create variation.
///
/// Used in dream processing to generate creative insights
/// by rotating the vector's dimensions.
pub(crate) fn permute_hv(hv: &ContinuousHV, shift: usize) -> ContinuousHV {
    let n = hv.values.len();
    if n == 0 || shift == 0 {
        return hv.clone();
    }
    let effective_shift = shift % n;
    let mut new_values = vec![0.0f32; n];
    for i in 0..n {
        new_values[(i + effective_shift) % n] = hv.values[i];
    }
    ContinuousHV::from_values(new_values)
}
