// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Prepared batch retrieval and fused update kernels for continuous HDC.
//!
//! This module is opt-in behind the crate's `simd` feature. It does not alter
//! [`ContinuousHV`](crate::ContinuousHV)'s reference scalar methods.
//!
//! The main optimization is to prepare immutable candidate hypervectors once in
//! contiguous row-major storage and cache their squared norms. A query then
//! computes its norm once and scores all candidates without rebuilding vectors,
//! labels, or candidate norms on every lookup.

use crate::ContinuousHV;

/// Immutable row-major candidate set for repeated cosine retrieval.
#[derive(Debug, Clone)]
pub struct PreparedContinuousHvSet {
    dim: usize,
    rows: Vec<f32>,
    norms_sq: Vec<f32>,
}

impl PreparedContinuousHvSet {
    /// Copy candidates into contiguous row-major storage and cache squared norms.
    ///
    /// Empty candidate sets are valid and have dimension zero.
    pub fn new(candidates: &[ContinuousHV]) -> Self {
        if candidates.is_empty() {
            return Self {
                dim: 0,
                rows: Vec::new(),
                norms_sq: Vec::new(),
            };
        }

        let dim = candidates[0].values.len();
        let mut rows = Vec::with_capacity(candidates.len() * dim);
        let mut norms_sq = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            assert_eq!(candidate.values.len(), dim, "Candidate dimension mismatch");
            norms_sq.push(dot_dispatch(&candidate.values, &candidate.values));
            rows.extend_from_slice(&candidate.values);
        }

        Self {
            dim,
            rows,
            norms_sq,
        }
    }

    /// Number of prepared candidates.
    #[inline]
    pub fn len(&self) -> usize {
        self.norms_sq.len()
    }

    /// Whether the prepared set contains no candidates.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.norms_sq.is_empty()
    }

    /// Hypervector dimension shared by all candidates, or zero when empty.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bytes owned by the prepared numeric payload, excluding allocator metadata.
    #[inline]
    pub fn payload_bytes(&self) -> usize {
        self.rows.len() * std::mem::size_of::<f32>()
            + self.norms_sq.len() * std::mem::size_of::<f32>()
    }

    /// Score one query against every prepared candidate into caller-owned output.
    ///
    /// This performs no heap allocation. Candidate norms are reused and the query
    /// norm is computed once for the whole candidate set.
    pub fn similarities_into(&self, query: &ContinuousHV, out: &mut [f32]) {
        assert_eq!(out.len(), self.len(), "Output length mismatch");
        if self.is_empty() {
            return;
        }
        assert_eq!(query.values.len(), self.dim, "Query dimension mismatch");

        let query_norm_sq = dot_dispatch(&query.values, &query.values);
        if query_norm_sq < 1e-20 {
            out.fill(0.0);
            return;
        }

        for (index, score) in out.iter_mut().enumerate() {
            let start = index * self.dim;
            let candidate = &self.rows[start..start + self.dim];
            let denom = (query_norm_sq * self.norms_sq[index]).sqrt();
            *score = if denom < 1e-10 {
                0.0
            } else {
                (dot_dispatch(&query.values, candidate) / denom).clamp(-1.0, 1.0)
            };
        }
    }

    /// Convenience allocation for callers that do not already own a score buffer.
    pub fn similarities(&self, query: &ContinuousHV) -> Vec<f32> {
        let mut out = vec![0.0; self.len()];
        self.similarities_into(query, &mut out);
        out
    }

    /// Return the row-major candidate vector at `index` without allocating.
    pub fn row(&self, index: usize) -> &[f32] {
        let start = index
            .checked_mul(self.dim)
            .expect("Candidate row offset overflow");
        &self.rows[start..start + self.dim]
    }
}

/// Fused state update operations that avoid temporary bound/scaled vectors.
pub trait ContinuousHvFusedSimdExt {
    /// Compute `self += (a * b) * scale` in one traversal.
    ///
    /// Binding is element-wise multiplication. The kernel deliberately avoids
    /// FMA so the per-component operation structure remains close to the scalar
    /// reference path.
    fn bind_add_scaled_simd(&mut self, a: &ContinuousHV, b: &ContinuousHV, scale: f32);
}

impl ContinuousHvFusedSimdExt for ContinuousHV {
    fn bind_add_scaled_simd(&mut self, a: &ContinuousHV, b: &ContinuousHV, scale: f32) {
        assert_eq!(self.values.len(), a.values.len(), "Dimension mismatch");
        assert_eq!(self.values.len(), b.values.len(), "Dimension mismatch");
        bind_add_scaled_dispatch(&mut self.values, &a.values, &b.values, scale);
    }
}

#[inline]
fn dot_dispatch(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected at runtime and the slices are equal length.
        unsafe { return dot_avx2(a, b) };
    }

    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn bind_add_scaled_dispatch(dst: &mut [f32], a: &[f32], b: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected at runtime and all slices are equal length.
        unsafe { return bind_add_scaled_avx2(dst, a, b, scale) };
    }

    for i in 0..dst.len() {
        dst[i] += (a[i] * b[i]) * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let simd_len = a.len() / 8 * 8;
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut i = 0;

        // Two independent accumulators reduce the dependency chain while keeping
        // deterministic lane assignment for a fixed AVX2 implementation.
        while i + 16 <= simd_len {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, b0));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a1, b1));
            i += 16;
        }
        if i < simd_len {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(av, bv));
            i += 8;
        }

        let sum = _mm256_add_ps(sum0, sum1);
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        let mut total = lanes.into_iter().sum::<f32>();

        while i < a.len() {
            total += a[i] * b[i];
            i += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bind_add_scaled_avx2(dst: &mut [f32], a: &[f32], b: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let scale_vec = _mm256_set1_ps(scale);
        let simd_len = dst.len() / 8 * 8;
        let mut i = 0;
        while i < simd_len {
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            let bound = _mm256_mul_ps(av, bv);
            let scaled = _mm256_mul_ps(bound, scale_vec);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(d, scaled));
            i += 8;
        }
        while i < dst.len() {
            dst[i] += (a[i] * b[i]) * scale;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tolerance: f32) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= tolerance * scale,
            "{a} != {b} with tolerance {tolerance}"
        );
    }

    #[test]
    fn prepared_batch_matches_scalar_winner_and_scores() {
        for dim in [1, 7, 8, 31, 256, 1024] {
            let query = ContinuousHV::new_random(dim, 7);
            let candidates: Vec<_> = (0..17)
                .map(|i| ContinuousHV::new_random(dim, 100 + i))
                .collect();
            let prepared = PreparedContinuousHvSet::new(&candidates);
            let accelerated = prepared.similarities(&query);
            let reference: Vec<_> = candidates.iter().map(|c| query.similarity(c)).collect();

            assert_eq!(accelerated.len(), reference.len());
            for (&fast, &slow) in accelerated.iter().zip(&reference) {
                assert_close(fast, slow, 2e-5);
            }

            let fast_winner = accelerated
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i);
            let slow_winner = reference
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i);
            assert_eq!(fast_winner, slow_winner);
        }
    }

    #[test]
    fn prepared_batch_reuses_output_buffer() {
        let candidates: Vec<_> = (0..8)
            .map(|i| ContinuousHV::new_random(64, 200 + i))
            .collect();
        let prepared = PreparedContinuousHvSet::new(&candidates);
        let query = ContinuousHV::new_random(64, 77);
        let mut out = vec![f32::NAN; candidates.len()];
        prepared.similarities_into(&query, &mut out);
        assert!(out.iter().all(|score| score.is_finite()));
    }

    #[test]
    fn fused_bind_accumulate_matches_scalar_exactly() {
        for dim in [1, 7, 8, 31, 32, 257, 1024] {
            let mut reference = ContinuousHV::new_random(dim, 1);
            let mut accelerated = reference.clone();
            let a = ContinuousHV::new_random(dim, 2);
            let b = ContinuousHV::new_random(dim, 3);
            let scale = 0.137f32;

            for i in 0..dim {
                reference.values[i] += (a.values[i] * b.values[i]) * scale;
            }
            accelerated.bind_add_scaled_simd(&a, &b, scale);
            assert_eq!(accelerated, reference);
        }
    }

    #[test]
    fn empty_prepared_set_is_valid() {
        let prepared = PreparedContinuousHvSet::new(&[]);
        let query = ContinuousHV::new_random(32, 1);
        let mut out = [];
        prepared.similarities_into(&query, &mut out);
        assert!(prepared.is_empty());
        assert_eq!(prepared.payload_bytes(), 0);
    }
}
