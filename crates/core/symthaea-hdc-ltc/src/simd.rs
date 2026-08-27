// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Opt-in SIMD acceleration for [`ContinuousHV`](crate::ContinuousHV).
//!
//! This module deliberately does **not** replace the crate's existing scalar
//! methods. The scalar implementation remains the reference path used by frozen
//! experiments. Callers opt in explicitly through [`ContinuousHvSimdExt`].
//!
//! On x86_64, AVX2 is selected at runtime and uses unaligned loads/stores. Other
//! targets use the portable scalar fallback in this module. FMA is deliberately
//! not used in state-changing kernels so the accelerated path stays as close as
//! practical to the scalar IEEE-754 operation structure.

use crate::ContinuousHV;

/// Runtime-selected implementation for the accelerated extension methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// Portable scalar fallback.
    Scalar,
    /// x86_64 AVX2 (8 x f32 lanes).
    Avx2,
}

/// Return the backend used by the accelerated extension methods.
#[inline]
pub fn simd_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return SimdBackend::Avx2;
    }
    SimdBackend::Scalar
}

/// Explicit accelerated operations for continuous hypervectors.
///
/// Enabling the `simd` Cargo feature alone cannot alter an existing experiment:
/// a caller must explicitly invoke these methods.
pub trait ContinuousHvSimdExt {
    /// Element-wise HDC binding using the selected accelerated kernel.
    fn bind_simd(&self, other: &ContinuousHV) -> ContinuousHV;

    /// Raw dot product using the selected accelerated reduction kernel.
    fn dot_simd(&self, other: &ContinuousHV) -> f32;

    /// Cosine similarity using a fused dot/norm reduction.
    fn similarity_simd(&self, other: &ContinuousHV) -> f32;

    /// Scale every component in place.
    fn scale_in_place_simd(&mut self, factor: f32);

    /// `self += other * scale` in one pass.
    fn add_scaled_simd(&mut self, other: &ContinuousHV, scale: f32);

    /// `self = (1 - alpha) * self + alpha * target` in one pass.
    fn lerp_in_place_simd(&mut self, target: &ContinuousHV, alpha: f32);
}

impl ContinuousHvSimdExt for ContinuousHV {
    #[inline]
    fn bind_simd(&self, other: &ContinuousHV) -> ContinuousHV {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        let mut out = vec![0.0f32; self.values.len()];
        bind_into(&self.values, &other.values, &mut out);
        ContinuousHV::from_values(out)
    }

    #[inline]
    fn dot_simd(&self, other: &ContinuousHV) -> f32 {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        dot(&self.values, &other.values)
    }

    #[inline]
    fn similarity_simd(&self, other: &ContinuousHV) -> f32 {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        let (dot, norm_a_sq, norm_b_sq) = cosine_stats(&self.values, &other.values);
        let denom = (norm_a_sq * norm_b_sq).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }

    #[inline]
    fn scale_in_place_simd(&mut self, factor: f32) {
        scale_in_place(&mut self.values, factor);
    }

    #[inline]
    fn add_scaled_simd(&mut self, other: &ContinuousHV, scale: f32) {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        add_scaled_in_place(&mut self.values, &other.values, scale);
    }

    #[inline]
    fn lerp_in_place_simd(&mut self, target: &ContinuousHV, alpha: f32) {
        assert_eq!(self.values.len(), target.values.len(), "Dimension mismatch");
        lerp_in_place(&mut self.values, &target.values, alpha);
    }
}

#[inline]
fn bind_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected at runtime; all slices are valid and equal length.
        unsafe { return bind_avx2(a, b, out) };
    }

    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

#[inline]
fn scale_in_place(values: &mut [f32], factor: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected at runtime and `values` is valid.
        unsafe { return scale_avx2(values, factor) };
    }

    for value in values {
        *value *= factor;
    }
}

#[inline]
fn add_scaled_in_place(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected at runtime; slices are valid and equal length.
        unsafe { return add_scaled_avx2(dst, src, scale) };
    }

    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s * scale;
    }
}

#[inline]
fn lerp_in_place(dst: &mut [f32], target: &[f32], alpha: f32) {
    debug_assert_eq!(dst.len(), target.len());
    let one_minus = 1.0 - alpha;

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected at runtime; slices are valid and equal length.
        unsafe { return lerp_avx2(dst, target, alpha, one_minus) };
    }

    for (d, &t) in dst.iter_mut().zip(target) {
        *d = one_minus * *d + alpha * t;
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected at runtime; slices are valid and equal length.
        unsafe { return dot_avx2(a, b) };
    }

    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn cosine_stats(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected at runtime; slices are valid and equal length.
        unsafe { return cosine_stats_avx2(a, b) };
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    (dot, norm_a, norm_b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bind_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let simd_len = a.len() / 8 * 8;
        let mut i = 0;
        while i < simd_len {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(av, bv));
            i += 8;
        }
        while i < a.len() {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(values: &mut [f32], factor: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let factor_vec = _mm256_set1_ps(factor);
        let simd_len = values.len() / 8 * 8;
        let mut i = 0;
        while i < simd_len {
            let v = _mm256_loadu_ps(values.as_ptr().add(i));
            _mm256_storeu_ps(values.as_mut_ptr().add(i), _mm256_mul_ps(v, factor_vec));
            i += 8;
        }
        while i < values.len() {
            values[i] *= factor;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_avx2(dst: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let scale_vec = _mm256_set1_ps(scale);
        let simd_len = dst.len() / 8 * 8;
        let mut i = 0;
        while i < simd_len {
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let s = _mm256_loadu_ps(src.as_ptr().add(i));
            let next = _mm256_add_ps(d, _mm256_mul_ps(s, scale_vec));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), next);
            i += 8;
        }
        while i < dst.len() {
            dst[i] += src[i] * scale;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lerp_avx2(dst: &mut [f32], target: &[f32], alpha: f32, one_minus: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let alpha_vec = _mm256_set1_ps(alpha);
        let one_minus_vec = _mm256_set1_ps(one_minus);
        let simd_len = dst.len() / 8 * 8;
        let mut i = 0;
        while i < simd_len {
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let t = _mm256_loadu_ps(target.as_ptr().add(i));
            let next = _mm256_add_ps(
                _mm256_mul_ps(d, one_minus_vec),
                _mm256_mul_ps(t, alpha_vec),
            );
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), next);
            i += 8;
        }
        while i < dst.len() {
            dst[i] = one_minus * dst[i] + alpha * target[i];
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let simd_len = a.len() / 8 * 8;
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        while i < simd_len {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
            i += 8;
        }
        let mut total = reduce_m256(sum);
        while i < a.len() {
            total += a[i] * b[i];
            i += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cosine_stats_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;
    unsafe {
        let simd_len = a.len() / 8 * 8;
        let mut dot = _mm256_setzero_ps();
        let mut norm_a = _mm256_setzero_ps();
        let mut norm_b = _mm256_setzero_ps();
        let mut i = 0;
        while i < simd_len {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            dot = _mm256_add_ps(dot, _mm256_mul_ps(av, bv));
            norm_a = _mm256_add_ps(norm_a, _mm256_mul_ps(av, av));
            norm_b = _mm256_add_ps(norm_b, _mm256_mul_ps(bv, bv));
            i += 8;
        }
        let mut dot_total = reduce_m256(dot);
        let mut norm_a_total = reduce_m256(norm_a);
        let mut norm_b_total = reduce_m256(norm_b);
        while i < a.len() {
            let x = a[i];
            let y = b[i];
            dot_total += x * y;
            norm_a_total += x * x;
            norm_b_total += y * y;
            i += 1;
        }
        (dot_total, norm_a_total, norm_b_total)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn reduce_m256(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::_mm256_storeu_ps;
    unsafe {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), value);
        lanes.into_iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tolerance: f32) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= tolerance * scale,
            "{a} != {b} (tol={tolerance}, backend={:?})",
            simd_backend()
        );
    }

    #[test]
    fn pointwise_kernels_match_reference() {
        for dim in [1, 7, 8, 31, 32, 257, 1024] {
            let a = ContinuousHV::new_random(dim, 11);
            let b = ContinuousHV::new_random(dim, 22);

            assert_eq!(a.bind_simd(&b), a.bind(&b));

            let mut scalar = a.clone();
            let mut accelerated = a.clone();
            scalar.scale_in_place(0.37);
            accelerated.scale_in_place_simd(0.37);
            assert_eq!(accelerated, scalar);

            let mut scalar = a.clone();
            let mut accelerated = a.clone();
            scalar.add_scaled(&b, -0.21);
            accelerated.add_scaled_simd(&b, -0.21);
            assert_eq!(accelerated, scalar);

            let mut scalar = a.clone();
            let mut accelerated = a.clone();
            scalar.lerp_in_place(&b, 0.125);
            accelerated.lerp_in_place_simd(&b, 0.125);
            assert_eq!(accelerated, scalar);
        }
    }

    #[test]
    fn reduction_kernels_preserve_numeric_and_ranking_semantics() {
        for dim in [8, 31, 256, 1024, 8192] {
            let query = ContinuousHV::new_random(dim, 7);
            let a = ContinuousHV::new_random(dim, 101);
            let b = ContinuousHV::new_random(dim, 202);

            assert_close(query.dot_simd(&a), query.dot(&a), 2e-5);
            assert_close(query.similarity_simd(&a), query.similarity(&a), 2e-5);

            let scalar_order = query.similarity(&a) > query.similarity(&b);
            let simd_order = query.similarity_simd(&a) > query.similarity_simd(&b);
            assert_eq!(simd_order, scalar_order);
        }
    }

    #[test]
    fn zero_vector_similarity_matches_reference() {
        let zero = ContinuousHV::new(1024);
        let random = ContinuousHV::new_random(1024, 99);
        assert_eq!(zero.similarity_simd(&random), zero.similarity(&random));
    }
}
