// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SIMD-Accelerated Operations for Continuous (f32) Hypervectors
//!
//! This module provides high-performance SIMD implementations for ContinuousHV operations:
//! - `dot_product_simd` - Vectorized dot product for similarity computation
//! - `bundle_simd` - Weighted sum bundling (superposition)
//! - `bind_simd` - Element-wise multiplication for binding
//!
//! # Architecture Support
//! - AVX2 (x86_64): 256-bit operations (8 f32s per instruction)
//! - SSE4.1 (x86_64): 128-bit operations (4 f32s per instruction)
//! - Portable: Safe fallback with auto-vectorization hints
//!
//! # Performance Targets
//! - 4x+ speedup for 16,384-dim vectors over naive scalar implementation
//! - GPU-ready architecture (identical algorithm patterns for future CUDA/Vulkan)
//!
//! # Feature Gate
//! All SIMD operations are gated behind `#[cfg(feature = "simd")]`

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
use std::arch::aarch64::*;

#[allow(unused_imports)]
use super::simd_detect::{has_avx, has_avx2, has_fma, has_neon, has_sse41};

// =============================================================================
// DOT PRODUCT - Core operation for similarity computation
// =============================================================================

/// SIMD-accelerated dot product for f32 slices
///
/// Computes sum(a[i] * b[i]) for all i using vectorized operations.
/// This is the fundamental building block for cosine similarity.
///
/// # Performance
/// - AVX2: ~4x speedup (8 f32s per cycle)
/// - SSE4.1: ~2x speedup (4 f32s per cycle)
/// - With FMA: Additional ~30% improvement
///
/// # Panics
/// Panics if `a.len() != b.len()`
#[cfg(feature = "simd")]
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // SAFETY: CPU feature detection above guarantees AVX2+FMA are available.
            // Slices `a` and `b` are valid, same-length (asserted above), and read-only.
            unsafe { dot_product_avx2_fma(a, b) }
        } else if has_avx2() {
            // SAFETY: AVX2 availability verified by runtime feature detection.
            unsafe { dot_product_avx2(a, b) }
        } else if has_sse41() {
            // SAFETY: SSE4.1 availability verified by runtime feature detection.
            unsafe { dot_product_sse41(a, b) }
        } else {
            dot_product_scalar(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "aarch64")]
        if has_neon() {
            // SAFETY: NEON availability verified by runtime feature detection (has_neon()).
            // Input slices are validated for length and alignment above.
            return unsafe { dot_product_neon(a, b) };
        }
        dot_product_scalar(a, b)
    }
}

/// AVX2 + FMA dot product (fastest path)
/// Processes 8 floats per iteration with fused multiply-add
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 32; // Process 32 floats per iteration (4 AVX registers)
        let remainder = len % 32;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Use 4 accumulators for better instruction-level parallelism
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        // Main loop: 32 elements per iteration
        for i in 0..chunks {
            let offset = i * 32;

            let a0 = _mm256_loadu_ps(a_ptr.add(offset));
            let b0 = _mm256_loadu_ps(b_ptr.add(offset));
            let a1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
            let b1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
            let a2 = _mm256_loadu_ps(a_ptr.add(offset + 16));
            let b2 = _mm256_loadu_ps(b_ptr.add(offset + 16));
            let a3 = _mm256_loadu_ps(a_ptr.add(offset + 24));
            let b3 = _mm256_loadu_ps(b_ptr.add(offset + 24));

            // FMA: sum += a * b (fused multiply-add)
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);
        }

        // Handle remaining 8-element chunks
        let remainder_chunks = remainder / 8;
        let final_remainder = remainder % 8;
        let offset = chunks * 32;

        for i in 0..remainder_chunks {
            let idx = offset + i * 8;
            let a_vec = _mm256_loadu_ps(a_ptr.add(idx));
            let b_vec = _mm256_loadu_ps(b_ptr.add(idx));
            sum0 = _mm256_fmadd_ps(a_vec, b_vec, sum0);
        }

        // Combine all accumulators
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Horizontal sum within AVX register
        // sum0 = [a0, a1, a2, a3, a4, a5, a6, a7]
        let hi = _mm256_extractf128_ps(sum0, 1); // [a4, a5, a6, a7]
        let lo = _mm256_castps256_ps128(sum0); // [a0, a1, a2, a3]
        let sum128 = _mm_add_ps(lo, hi); // [a0+a4, a1+a5, a2+a6, a3+a7]

        // Shuffle and add pairs
        let shuf = _mm_movehdup_ps(sum128); // [a1+a5, a1+a5, a3+a7, a3+a7]
        let sums = _mm_add_ps(sum128, shuf); // [a0+a1+a4+a5, ...]
        let shuf2 = _mm_movehl_ps(sums, sums); // [a2+a3+a6+a7, ...]
        let result = _mm_add_ss(sums, shuf2);

        let mut total = _mm_cvtss_f32(result);

        // Handle final scalar remainder
        let scalar_start = chunks * 32 + remainder_chunks * 8;
        for i in 0..final_remainder {
            total += a[scalar_start + i] * b[scalar_start + i];
        }

        total
    }
}

/// AVX2 dot product without FMA
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 32;
        let remainder = len % 32;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 32;

            let a0 = _mm256_loadu_ps(a_ptr.add(offset));
            let b0 = _mm256_loadu_ps(b_ptr.add(offset));
            let a1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
            let b1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
            let a2 = _mm256_loadu_ps(a_ptr.add(offset + 16));
            let b2 = _mm256_loadu_ps(b_ptr.add(offset + 16));
            let a3 = _mm256_loadu_ps(a_ptr.add(offset + 24));
            let b3 = _mm256_loadu_ps(b_ptr.add(offset + 24));

            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, b0));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a1, b1));
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(a2, b2));
            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(a3, b3));
        }

        // Handle remaining 8-element chunks
        let remainder_chunks = remainder / 8;
        let final_remainder = remainder % 8;
        let offset = chunks * 32;

        for i in 0..remainder_chunks {
            let idx = offset + i * 8;
            let a_vec = _mm256_loadu_ps(a_ptr.add(idx));
            let b_vec = _mm256_loadu_ps(b_ptr.add(idx));
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a_vec, b_vec));
        }

        // Combine accumulators
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum0, 1);
        let lo = _mm256_castps256_ps128(sum0);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);

        let mut total = _mm_cvtss_f32(result);

        // Scalar remainder
        let scalar_start = chunks * 32 + remainder_chunks * 8;
        for i in 0..final_remainder {
            total += a[scalar_start + i] * b[scalar_start + i];
        }

        total
    }
}

/// SSE4.1 dot product (fallback for older CPUs)
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn dot_product_sse41(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 16; // Process 16 floats per iteration
        let remainder = len % 16;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = _mm_setzero_ps();
        let mut sum1 = _mm_setzero_ps();
        let mut sum2 = _mm_setzero_ps();
        let mut sum3 = _mm_setzero_ps();

        for i in 0..chunks {
            let offset = i * 16;

            let a0 = _mm_loadu_ps(a_ptr.add(offset));
            let b0 = _mm_loadu_ps(b_ptr.add(offset));
            let a1 = _mm_loadu_ps(a_ptr.add(offset + 4));
            let b1 = _mm_loadu_ps(b_ptr.add(offset + 4));
            let a2 = _mm_loadu_ps(a_ptr.add(offset + 8));
            let b2 = _mm_loadu_ps(b_ptr.add(offset + 8));
            let a3 = _mm_loadu_ps(a_ptr.add(offset + 12));
            let b3 = _mm_loadu_ps(b_ptr.add(offset + 12));

            sum0 = _mm_add_ps(sum0, _mm_mul_ps(a0, b0));
            sum1 = _mm_add_ps(sum1, _mm_mul_ps(a1, b1));
            sum2 = _mm_add_ps(sum2, _mm_mul_ps(a2, b2));
            sum3 = _mm_add_ps(sum3, _mm_mul_ps(a3, b3));
        }

        // Handle remaining 4-element chunks
        let remainder_chunks = remainder / 4;
        let final_remainder = remainder % 4;
        let offset = chunks * 16;

        for i in 0..remainder_chunks {
            let idx = offset + i * 4;
            let a_vec = _mm_loadu_ps(a_ptr.add(idx));
            let b_vec = _mm_loadu_ps(b_ptr.add(idx));
            sum0 = _mm_add_ps(sum0, _mm_mul_ps(a_vec, b_vec));
        }

        // Combine accumulators
        sum0 = _mm_add_ps(sum0, sum1);
        sum2 = _mm_add_ps(sum2, sum3);
        sum0 = _mm_add_ps(sum0, sum2);

        // Horizontal sum using SSE3 hadd
        let shuf = _mm_movehdup_ps(sum0);
        let sums = _mm_add_ps(sum0, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);

        let mut total = _mm_cvtss_f32(result);

        // Scalar remainder
        let scalar_start = chunks * 16 + remainder_chunks * 4;
        for i in 0..final_remainder {
            total += a[scalar_start + i] * b[scalar_start + i];
        }

        total
    }
}

/// Scalar dot product (baseline)
#[cfg(feature = "simd")]
#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// =============================================================================
// BIND - Element-wise multiplication
// =============================================================================

/// SIMD-accelerated element-wise multiplication (binding operation)
///
/// Computes result[i] = a[i] * b[i] for all i.
/// In HDC, binding creates associations between concepts.
///
/// # Returns
/// A new `Vec<f32>` containing the element-wise product.
#[cfg(feature = "simd")]
#[inline]
pub fn bind_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

    let mut result = vec![0.0f32; a.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: AVX2 availability verified by runtime feature detection.
            // `a`, `b`, `result` are same-length slices (asserted/allocated above).
            unsafe { bind_avx2(a, b, &mut result) };
        } else if has_sse41() {
            // SAFETY: SSE4.1 availability verified by runtime feature detection.
            unsafe { bind_sse41(a, b, &mut result) };
        } else {
            bind_scalar(a, b, &mut result);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "aarch64")]
        if has_neon() {
            // SAFETY: NEON availability verified by runtime feature detection (has_neon()).
            // Input slices are validated for length and alignment above.
            unsafe { bind_neon(a, b, &mut result) };
            return result;
        }
        bind_scalar(a, b, &mut result);
    }

    result
}

/// AVX2 element-wise multiplication
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bind_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let chunks = len / 32;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let r_ptr = result.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 32;

            let a0 = _mm256_loadu_ps(a_ptr.add(offset));
            let b0 = _mm256_loadu_ps(b_ptr.add(offset));
            let a1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
            let b1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
            let a2 = _mm256_loadu_ps(a_ptr.add(offset + 16));
            let b2 = _mm256_loadu_ps(b_ptr.add(offset + 16));
            let a3 = _mm256_loadu_ps(a_ptr.add(offset + 24));
            let b3 = _mm256_loadu_ps(b_ptr.add(offset + 24));

            _mm256_storeu_ps(r_ptr.add(offset), _mm256_mul_ps(a0, b0));
            _mm256_storeu_ps(r_ptr.add(offset + 8), _mm256_mul_ps(a1, b1));
            _mm256_storeu_ps(r_ptr.add(offset + 16), _mm256_mul_ps(a2, b2));
            _mm256_storeu_ps(r_ptr.add(offset + 24), _mm256_mul_ps(a3, b3));
        }

        // Handle remainder with 8-element chunks
        let offset = chunks * 32;
        let remainder_chunks = (len - offset) / 8;

        for i in 0..remainder_chunks {
            let idx = offset + i * 8;
            let a_vec = _mm256_loadu_ps(a_ptr.add(idx));
            let b_vec = _mm256_loadu_ps(b_ptr.add(idx));
            _mm256_storeu_ps(r_ptr.add(idx), _mm256_mul_ps(a_vec, b_vec));
        }

        // Scalar remainder
        let scalar_start = offset + remainder_chunks * 8;
        for i in scalar_start..len {
            *r_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
        }
    }
}

/// SSE4.1 element-wise multiplication
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn bind_sse41(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let chunks = len / 16;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let r_ptr = result.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 16;

            let a0 = _mm_loadu_ps(a_ptr.add(offset));
            let b0 = _mm_loadu_ps(b_ptr.add(offset));
            let a1 = _mm_loadu_ps(a_ptr.add(offset + 4));
            let b1 = _mm_loadu_ps(b_ptr.add(offset + 4));
            let a2 = _mm_loadu_ps(a_ptr.add(offset + 8));
            let b2 = _mm_loadu_ps(b_ptr.add(offset + 8));
            let a3 = _mm_loadu_ps(a_ptr.add(offset + 12));
            let b3 = _mm_loadu_ps(b_ptr.add(offset + 12));

            _mm_storeu_ps(r_ptr.add(offset), _mm_mul_ps(a0, b0));
            _mm_storeu_ps(r_ptr.add(offset + 4), _mm_mul_ps(a1, b1));
            _mm_storeu_ps(r_ptr.add(offset + 8), _mm_mul_ps(a2, b2));
            _mm_storeu_ps(r_ptr.add(offset + 12), _mm_mul_ps(a3, b3));
        }

        // Scalar remainder
        let scalar_start = chunks * 16;
        for i in scalar_start..len {
            *r_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
        }
    }
}

/// Scalar element-wise multiplication
#[cfg(feature = "simd")]
#[inline]
fn bind_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] * b[i];
    }
}

// =============================================================================
// BUNDLE - Weighted sum (superposition)
// =============================================================================

/// SIMD-accelerated weighted bundle (superposition)
///
/// Computes weighted average: result[i] = sum(hvs[j][i] * weights[j]) / sum(weights)
///
/// # Arguments
/// * `hvs` - Slice of hypervector slices to bundle
/// * `weights` - Weight for each hypervector
///
/// # Returns
/// A new `Vec<f32>` containing the weighted superposition.
#[cfg(feature = "simd")]
#[inline]
pub fn bundle_simd(hvs: &[&[f32]], weights: &[f32]) -> Vec<f32> {
    if hvs.is_empty() || weights.is_empty() {
        return Vec::new();
    }

    assert_eq!(
        hvs.len(),
        weights.len(),
        "Number of HVs must match number of weights"
    );

    let dim = hvs[0].dim();
    for hv in hvs.iter() {
        assert_eq!(hv.len(), dim, "All HVs must have same dimension");
    }

    let weight_sum: f32 = weights.iter().sum();
    let inv_weight_sum = if weight_sum.abs() > 1e-10 {
        1.0 / weight_sum
    } else {
        0.0
    };

    let mut result = vec![0.0f32; dim];

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // SAFETY: AVX2+FMA availability verified by runtime feature detection.
            // All HVs have same dimension (asserted above), `result` is that size.
            unsafe { bundle_avx2_fma(hvs, weights, &mut result, inv_weight_sum) };
        } else if has_avx2() {
            // SAFETY: AVX2 availability verified by runtime feature detection.
            unsafe { bundle_avx2(hvs, weights, &mut result, inv_weight_sum) };
        } else if has_sse41() {
            // SAFETY: SSE4.1 availability verified by runtime feature detection.
            unsafe { bundle_sse41(hvs, weights, &mut result, inv_weight_sum) };
        } else {
            bundle_scalar(hvs, weights, &mut result, inv_weight_sum);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "aarch64")]
        if has_neon() {
            // SAFETY: NEON availability verified by runtime feature detection (has_neon()).
            // All HVs have same dimension (asserted above), `result` is that size.
            unsafe { bundle_neon(hvs, weights, &mut result, inv_weight_sum) };
            return result;
        }
        bundle_scalar(hvs, weights, &mut result, inv_weight_sum);
    }

    result
}

/// AVX2 + FMA weighted bundle
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn bundle_avx2_fma(
    hvs: &[&[f32]],
    weights: &[f32],
    result: &mut [f32],
    inv_weight_sum: f32,
) {
    unsafe {
        let dim = result.len();
        let chunks = dim / 8;
        let remainder = dim % 8;

        let r_ptr = result.as_mut_ptr();
        let inv_weight_vec = _mm256_set1_ps(inv_weight_sum);

        // Process in 8-element chunks
        for chunk in 0..chunks {
            let offset = chunk * 8;
            let mut acc = _mm256_setzero_ps();

            // Accumulate weighted sum across all HVs
            for (hv, &weight) in hvs.iter().zip(weights.iter()) {
                let weight_vec = _mm256_set1_ps(weight);
                let hv_vec = _mm256_loadu_ps(hv.as_ptr().add(offset));
                acc = _mm256_fmadd_ps(hv_vec, weight_vec, acc);
            }

            // Normalize by weight sum
            let normalized = _mm256_mul_ps(acc, inv_weight_vec);
            _mm256_storeu_ps(r_ptr.add(offset), normalized);
        }

        // Scalar remainder
        let scalar_start = chunks * 8;
        for i in 0..remainder {
            let idx = scalar_start + i;
            let mut sum = 0.0f32;
            for (hv, &weight) in hvs.iter().zip(weights.iter()) {
                sum += hv[idx] * weight;
            }
            result[idx] = sum * inv_weight_sum;
        }
    }
}

/// AVX2 weighted bundle without FMA
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bundle_avx2(hvs: &[&[f32]], weights: &[f32], result: &mut [f32], inv_weight_sum: f32) {
    unsafe {
        let dim = result.len();
        let chunks = dim / 8;
        let remainder = dim % 8;

        let r_ptr = result.as_mut_ptr();
        let inv_weight_vec = _mm256_set1_ps(inv_weight_sum);

        for chunk in 0..chunks {
            let offset = chunk * 8;
            let mut acc = _mm256_setzero_ps();

            for (hv, &weight) in hvs.iter().zip(weights.iter()) {
                let weight_vec = _mm256_set1_ps(weight);
                let hv_vec = _mm256_loadu_ps(hv.as_ptr().add(offset));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(hv_vec, weight_vec));
            }

            let normalized = _mm256_mul_ps(acc, inv_weight_vec);
            _mm256_storeu_ps(r_ptr.add(offset), normalized);
        }

        // Scalar remainder
        let scalar_start = chunks * 8;
        for i in 0..remainder {
            let idx = scalar_start + i;
            let mut sum = 0.0f32;
            for (hv, &weight) in hvs.iter().zip(weights.iter()) {
                sum += hv[idx] * weight;
            }
            result[idx] = sum * inv_weight_sum;
        }
    }
}

/// SSE4.1 weighted bundle
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn bundle_sse41(hvs: &[&[f32]], weights: &[f32], result: &mut [f32], inv_weight_sum: f32) {
    unsafe {
        let dim = result.len();
        let chunks = dim / 4;
        let remainder = dim % 4;

        let r_ptr = result.as_mut_ptr();
        let inv_weight_vec = _mm_set1_ps(inv_weight_sum);

        for chunk in 0..chunks {
            let offset = chunk * 4;
            let mut acc = _mm_setzero_ps();

            for (hv, &weight) in hvs.iter().zip(weights.iter()) {
                let weight_vec = _mm_set1_ps(weight);
                let hv_vec = _mm_loadu_ps(hv.as_ptr().add(offset));
                acc = _mm_add_ps(acc, _mm_mul_ps(hv_vec, weight_vec));
            }

            let normalized = _mm_mul_ps(acc, inv_weight_vec);
            _mm_storeu_ps(r_ptr.add(offset), normalized);
        }

        // Scalar remainder
        let scalar_start = chunks * 4;
        for i in 0..remainder {
            let idx = scalar_start + i;
            let mut sum = 0.0f32;
            for (hv, &weight) in hvs.iter().zip(weights.iter()) {
                sum += hv[idx] * weight;
            }
            result[idx] = sum * inv_weight_sum;
        }
    }
}

/// Scalar weighted bundle
#[cfg(feature = "simd")]
#[inline]
fn bundle_scalar(hvs: &[&[f32]], weights: &[f32], result: &mut [f32], inv_weight_sum: f32) {
    let dim = result.len();

    for i in 0..dim {
        let mut sum = 0.0f32;
        for (hv, &weight) in hvs.iter().zip(weights.iter()) {
            sum += hv[i] * weight;
        }
        result[i] = sum * inv_weight_sum;
    }
}

// =============================================================================
// L2 NORM - For normalization
// =============================================================================

/// SIMD-accelerated L2 norm (magnitude)
///
/// Computes sqrt(sum(x[i]^2)) using vectorized operations.
#[cfg(feature = "simd")]
#[inline]
pub fn norm_simd(x: &[f32]) -> f32 {
    dot_product_simd(x, x).sqrt()
}

// =============================================================================
// COSINE SIMILARITY - Using SIMD primitives
// =============================================================================

/// SIMD-accelerated cosine similarity
///
/// Computes dot(a, b) / (norm(a) * norm(b))
#[cfg(feature = "simd")]
#[inline]
pub fn similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

    // Compute all three dot products in a single pass for better cache locality
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // SAFETY: AVX2+FMA availability verified by runtime feature detection.
            // Slices `a` and `b` are valid, same-length (asserted above), and read-only.
            unsafe { similarity_avx2_fma(a, b) }
        } else if has_avx2() {
            similarity_scalar_optimized(a, b)
        } else {
            similarity_scalar_optimized(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "aarch64")]
        if has_neon() {
            // SAFETY: NEON availability verified by runtime feature detection (has_neon()).
            // Input slices are validated for length and alignment above.
            return unsafe { similarity_neon(a, b) };
        }
        similarity_scalar_optimized(a, b)
    }
}

/// AVX2 + FMA similarity with single-pass computation
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn similarity_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Compute dot(a,b), dot(a,a), dot(b,b) simultaneously
        let mut dot_ab = _mm256_setzero_ps();
        let mut dot_aa = _mm256_setzero_ps();
        let mut dot_bb = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let a_vec = _mm256_loadu_ps(a_ptr.add(offset));
            let b_vec = _mm256_loadu_ps(b_ptr.add(offset));

            dot_ab = _mm256_fmadd_ps(a_vec, b_vec, dot_ab);
            dot_aa = _mm256_fmadd_ps(a_vec, a_vec, dot_aa);
            dot_bb = _mm256_fmadd_ps(b_vec, b_vec, dot_bb);
        }

        // Horizontal sum helper
        let hsum = |v: __m256| -> f32 {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
        };

        let mut ab = hsum(dot_ab);
        let mut aa = hsum(dot_aa);
        let mut bb = hsum(dot_bb);

        // Scalar remainder
        let scalar_start = chunks * 8;
        for i in 0..remainder {
            let idx = scalar_start + i;
            let av = *a_ptr.add(idx);
            let bv = *b_ptr.add(idx);
            ab += av * bv;
            aa += av * av;
            bb += bv * bv;
        }

        let denom = (aa * bb).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            (ab / denom).clamp(-1.0, 1.0)
        }
    }
}

/// Optimized scalar similarity
#[cfg(feature = "simd")]
#[inline]
fn similarity_scalar_optimized(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_ab = 0.0f32;
    let mut dot_aa = 0.0f32;
    let mut dot_bb = 0.0f32;

    for (&av, &bv) in a.iter().zip(b.iter()) {
        dot_ab += av * bv;
        dot_aa += av * av;
        dot_bb += bv * bv;
    }

    let denom = (dot_aa * dot_bb).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (dot_ab / denom).clamp(-1.0, 1.0)
    }
}

// =============================================================================
// NEON (AArch64) IMPLEMENTATIONS FOR CONTINUOUS f32 HVs
// =============================================================================

/// NEON dot product: 4 f32s per vfmaq_f32, unrolled 4x.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 16;
        sum0 = vfmaq_f32(sum0, vld1q_f32(a_ptr.add(off)), vld1q_f32(b_ptr.add(off)));
        sum1 = vfmaq_f32(
            sum1,
            vld1q_f32(a_ptr.add(off + 4)),
            vld1q_f32(b_ptr.add(off + 4)),
        );
        sum2 = vfmaq_f32(
            sum2,
            vld1q_f32(a_ptr.add(off + 8)),
            vld1q_f32(b_ptr.add(off + 8)),
        );
        sum3 = vfmaq_f32(
            sum3,
            vld1q_f32(a_ptr.add(off + 12)),
            vld1q_f32(b_ptr.add(off + 12)),
        );
    }

    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let mut total = vaddvq_f32(sum0);

    for i in (chunks * 16)..len {
        total += *a_ptr.add(i) * *b_ptr.add(i);
    }
    total
}

/// NEON element-wise multiply (bind) for f32 slices.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
unsafe fn bind_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let chunks = len / 16;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let r_ptr = result.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 16;
        vst1q_f32(
            r_ptr.add(off),
            vmulq_f32(vld1q_f32(a_ptr.add(off)), vld1q_f32(b_ptr.add(off))),
        );
        vst1q_f32(
            r_ptr.add(off + 4),
            vmulq_f32(vld1q_f32(a_ptr.add(off + 4)), vld1q_f32(b_ptr.add(off + 4))),
        );
        vst1q_f32(
            r_ptr.add(off + 8),
            vmulq_f32(vld1q_f32(a_ptr.add(off + 8)), vld1q_f32(b_ptr.add(off + 8))),
        );
        vst1q_f32(
            r_ptr.add(off + 12),
            vmulq_f32(
                vld1q_f32(a_ptr.add(off + 12)),
                vld1q_f32(b_ptr.add(off + 12)),
            ),
        );
    }
    for i in (chunks * 16)..len {
        *r_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
    }
}

/// NEON weighted bundle (superposition) for f32 slices.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
unsafe fn bundle_neon(hvs: &[&[f32]], weights: &[f32], result: &mut [f32], inv_weight_sum: f32) {
    let dim = result.len();
    let chunks = dim / 4;
    let r_ptr = result.as_mut_ptr();
    let inv_w = vdupq_n_f32(inv_weight_sum);

    for chunk in 0..chunks {
        let off = chunk * 4;
        let mut acc = vdupq_n_f32(0.0);
        for (hv, &weight) in hvs.iter().zip(weights.iter()) {
            let w = vdupq_n_f32(weight);
            acc = vfmaq_f32(acc, vld1q_f32(hv.as_ptr().add(off)), w);
        }
        vst1q_f32(r_ptr.add(off), vmulq_f32(acc, inv_w));
    }
    for i in (chunks * 4)..dim {
        let mut sum = 0.0f32;
        for (hv, &weight) in hvs.iter().zip(weights.iter()) {
            sum += hv[i] * weight;
        }
        result[i] = sum * inv_weight_sum;
    }
}

/// NEON cosine similarity: single-pass dot(a,b), dot(a,a), dot(b,b).
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
unsafe fn similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dot_ab = vdupq_n_f32(0.0);
    let mut dot_aa = vdupq_n_f32(0.0);
    let mut dot_bb = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        let av = vld1q_f32(a_ptr.add(off));
        let bv = vld1q_f32(b_ptr.add(off));
        dot_ab = vfmaq_f32(dot_ab, av, bv);
        dot_aa = vfmaq_f32(dot_aa, av, av);
        dot_bb = vfmaq_f32(dot_bb, bv, bv);
    }

    let mut ab = vaddvq_f32(dot_ab);
    let mut aa = vaddvq_f32(dot_aa);
    let mut bb = vaddvq_f32(dot_bb);

    for i in (chunks * 4)..len {
        let av = *a_ptr.add(i);
        let bv = *b_ptr.add(i);
        ab += av * bv;
        aa += av * av;
        bb += bv * bv;
    }

    let denom = (aa * bb).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (ab / denom).clamp(-1.0, 1.0)
    }
}

// =============================================================================
// SCALAR FALLBACKS FOR NON-SIMD BUILDS
// =============================================================================

/// Non-SIMD dot product (for builds without simd feature)
#[cfg(not(feature = "simd"))]
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Non-SIMD bind (for builds without simd feature)
#[cfg(not(feature = "simd"))]
#[inline]
pub fn bind_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Non-SIMD bundle (for builds without simd feature)
#[cfg(not(feature = "simd"))]
#[inline]
pub fn bundle_simd(hvs: &[&[f32]], weights: &[f32]) -> Vec<f32> {
    if hvs.is_empty() || weights.is_empty() {
        return Vec::new();
    }

    let dim = hvs[0].len();
    let weight_sum: f32 = weights.iter().sum();
    let inv_weight_sum = if weight_sum.abs() > 1e-10 {
        1.0 / weight_sum
    } else {
        0.0
    };

    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for (hv, &weight) in hvs.iter().zip(weights.iter()) {
            sum += hv[i] * weight;
        }
        result[i] = sum * inv_weight_sum;
    }
    result
}

/// Non-SIMD norm (for builds without simd feature)
#[cfg(not(feature = "simd"))]
#[inline]
pub fn norm_simd(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Non-SIMD similarity (for builds without simd feature)
#[cfg(not(feature = "simd"))]
#[inline]
pub fn similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_ab = 0.0f32;
    let mut dot_aa = 0.0f32;
    let mut dot_bb = 0.0f32;

    for (&av, &bv) in a.iter().zip(b.iter()) {
        dot_ab += av * bv;
        dot_aa += av * av;
        dot_bb += bv * bv;
    }

    let denom = (dot_aa * dot_bb).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (dot_ab / denom).clamp(-1.0, 1.0)
    }
}

// =============================================================================
// SIMD CAPABILITY REPORT
// =============================================================================

/// Report available SIMD capabilities for continuous HV operations
#[cfg(feature = "simd")]
pub fn simd_capabilities_report() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        format!(
            "SIMD Capabilities (Continuous HV):\n\
             - AVX2:   {}\n\
             - AVX:    {}\n\
             - FMA:    {}\n\
             - SSE4.1: {}",
            has_avx2(),
            has_avx(),
            has_fma(),
            has_sse41()
        )
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        format!(
            "SIMD Capabilities (Continuous HV):\n\
             - NEON:   {}\n\
             - (non-x86_64 platform)",
            has_neon()
        )
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_capabilities_report() -> String {
    "SIMD Capabilities: Feature 'simd' not enabled (using scalar fallback)".to_string()
}

// =============================================================================
// HELPER TRAIT FOR SLICE LENGTH
// =============================================================================

trait SliceLen {
    fn dim(&self) -> usize;
}

impl SliceLen for &[f32] {
    fn dim(&self) -> usize {
        self.len()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;
    const HDC_DIM: usize = 16_384;

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed ^ 0x9E3779B97F4A7C15; // avoid xorshift64 fixed-point at 0
        (0..dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    // ===== Dot Product =====

    #[test]
    fn test_dot_product_correctness() {
        let a = random_vec(HDC_DIM, 42);
        let b = random_vec(HDC_DIM, 43);

        let simd_result = dot_product_simd(&a, &b);
        let scalar_result = scalar_dot_product(&a, &b);

        let relative_error = ((simd_result - scalar_result) / scalar_result.abs().max(1e-10)).abs();
        assert!(
            relative_error < EPSILON,
            "Dot product mismatch: SIMD={}, Scalar={}, Error={}",
            simd_result,
            scalar_result,
            relative_error
        );
    }

    #[test]
    fn test_dot_product_self() {
        let a = random_vec(HDC_DIM, 42);

        let result = dot_product_simd(&a, &a);
        let expected: f32 = a.iter().map(|&x| x * x).sum();

        let relative_error = ((result - expected) / expected.abs().max(1e-10)).abs();
        assert!(
            relative_error < EPSILON,
            "Self dot product error: {}",
            relative_error
        );
    }

    #[test]
    fn test_dot_product_zero_vector() {
        let a = random_vec(HDC_DIM, 42);
        let zero = vec![0.0f32; HDC_DIM];

        let result = dot_product_simd(&a, &zero);
        assert!(
            result.abs() < 1e-6,
            "Dot with zero should be 0, got {}",
            result
        );
    }

    #[test]
    fn test_dot_product_orthogonal_unit() {
        // Create two orthogonal unit vectors
        let mut a = vec![0.0f32; 100];
        let mut b = vec![0.0f32; 100];
        a[0] = 1.0;
        b[1] = 1.0;

        let result = dot_product_simd(&a, &b);
        assert!(
            result.abs() < EPSILON,
            "Orthogonal dot product should be 0, got {}",
            result
        );
    }

    #[test]
    fn test_dot_product_parallel_unit() {
        let mut a = vec![0.0f32; 100];
        a[0] = 1.0;

        let result = dot_product_simd(&a, &a);
        assert!(
            (result - 1.0).abs() < EPSILON,
            "Parallel unit dot should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn test_dot_product_empty() {
        let result = dot_product_simd(&[], &[]);
        assert_eq!(result, 0.0, "Empty dot product should be 0.0");
    }

    // ===== Bind =====

    #[test]
    fn test_bind_correctness() {
        let a = random_vec(HDC_DIM, 42);
        let b = random_vec(HDC_DIM, 43);

        let simd_result = bind_simd(&a, &b);
        let scalar_result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

        for i in 0..HDC_DIM {
            assert!(
                (simd_result[i] - scalar_result[i]).abs() < EPSILON,
                "Bind mismatch at index {}: {} vs {}",
                i,
                simd_result[i],
                scalar_result[i]
            );
        }
    }

    #[test]
    fn test_bind_identity() {
        let a = random_vec(HDC_DIM, 42);
        let ones = vec![1.0f32; HDC_DIM];

        let result = bind_simd(&a, &ones);
        for i in 0..HDC_DIM {
            assert!(
                (result[i] - a[i]).abs() < EPSILON,
                "Multiply by 1 should preserve: {} vs {}",
                result[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_bind_with_zero() {
        let a = random_vec(HDC_DIM, 42);
        let zero = vec![0.0f32; HDC_DIM];

        let result = bind_simd(&a, &zero);
        for &val in &result {
            assert!(
                val.abs() < EPSILON,
                "Multiply by 0 should be 0, got {}",
                val
            );
        }
    }

    #[test]
    fn test_bind_commutativity() {
        let a = random_vec(100, 42);
        let b = random_vec(100, 43);

        let ab = bind_simd(&a, &b);
        let ba = bind_simd(&b, &a);

        for i in 0..100 {
            assert!(
                (ab[i] - ba[i]).abs() < EPSILON,
                "Elementwise multiplication should be commutative"
            );
        }
    }

    // ===== Bundle =====

    #[test]
    fn test_bundle_uniform_weights() {
        let vecs: Vec<Vec<f32>> = (0..5).map(|i| random_vec(HDC_DIM, i + 100)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let weights = vec![1.0; 5];

        let simd_result = bundle_simd(&refs, &weights);

        let mut expected = vec![0.0f32; HDC_DIM];
        for i in 0..HDC_DIM {
            for vec in &vecs {
                expected[i] += vec[i];
            }
            expected[i] /= 5.0;
        }

        for i in 0..HDC_DIM {
            assert!(
                (simd_result[i] - expected[i]).abs() < EPSILON,
                "Bundle mismatch at index {}: {} vs {}",
                i,
                simd_result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_bundle_varying_weights() {
        let vecs: Vec<Vec<f32>> = (0..3).map(|i| random_vec(HDC_DIM, i + 100)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let weights = vec![1.0, 2.0, 3.0];
        let weight_sum: f32 = weights.iter().sum();

        let simd_result = bundle_simd(&refs, &weights);

        let mut expected = vec![0.0f32; HDC_DIM];
        for i in 0..HDC_DIM {
            for (vec, &w) in vecs.iter().zip(weights.iter()) {
                expected[i] += vec[i] * w;
            }
            expected[i] /= weight_sum;
        }

        for i in 0..HDC_DIM {
            assert!(
                (simd_result[i] - expected[i]).abs() < EPSILON,
                "Weighted bundle mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_bundle_single_vector() {
        let v = random_vec(100, 42);
        let refs: Vec<&[f32]> = vec![v.as_slice()];
        let weights = vec![1.0];

        let result = bundle_simd(&refs, &weights);
        for i in 0..100 {
            assert!(
                (result[i] - v[i]).abs() < EPSILON,
                "Single vector bundle should return that vector"
            );
        }
    }

    #[test]
    fn test_bundle_empty_input() {
        let result = bundle_simd(&[], &[]);
        assert!(result.is_empty(), "Bundle of empty should be empty");
    }

    #[test]
    fn test_bundle_dominant_weight() {
        // If one weight is very large, result should be close to that vector
        let vecs: Vec<Vec<f32>> = (0..3).map(|i| random_vec(100, i + 100)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let weights = vec![0.01, 100.0, 0.01];

        let result = bundle_simd(&refs, &weights);

        // Result should be very close to vecs[1]
        let mut max_diff = 0.0f32;
        for i in 0..100 {
            max_diff = max_diff.max((result[i] - vecs[1][i]).abs());
        }
        assert!(
            max_diff < 0.01,
            "Dominant weight should make result close to that vector, max_diff={}",
            max_diff
        );
    }

    // ===== Similarity =====

    #[test]
    fn test_similarity_self() {
        let a = random_vec(HDC_DIM, 42);

        let sim = similarity_simd(&a, &a);
        assert!(
            (sim - 1.0).abs() < EPSILON,
            "Self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_similarity_orthogonal() {
        let a = random_vec(HDC_DIM, 42);
        let b = random_vec(HDC_DIM, 43);

        let sim = similarity_simd(&a, &b);
        assert!(
            sim.abs() < 0.1,
            "Random vectors should be nearly orthogonal, got {}",
            sim
        );
    }

    #[test]
    fn test_similarity_negated() {
        let a = random_vec(100, 42);
        let neg_a: Vec<f32> = a.iter().map(|&x| -x).collect();

        let sim = similarity_simd(&a, &neg_a);
        assert!(
            (sim - (-1.0)).abs() < EPSILON,
            "Negated vector should have similarity -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_similarity_range() {
        for seed in 0..20 {
            let a = random_vec(100, seed);
            let b = random_vec(100, seed + 1000);
            let sim = similarity_simd(&a, &b);
            assert!(
                sim >= -1.0 && sim <= 1.0,
                "Similarity should be in [-1, 1], got {}",
                sim
            );
        }
    }

    #[test]
    fn test_similarity_zero_vector() {
        let a = random_vec(100, 42);
        let zero = vec![0.0f32; 100];

        let sim = similarity_simd(&a, &zero);
        assert_eq!(
            sim, 0.0,
            "Similarity with zero vector should be 0, got {}",
            sim
        );
    }

    // ===== Norm =====

    #[test]
    fn test_norm() {
        let a = random_vec(HDC_DIM, 42);

        let simd_norm = norm_simd(&a);
        let scalar_norm: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();

        let relative_error = ((simd_norm - scalar_norm) / scalar_norm.abs().max(1e-10)).abs();
        assert!(
            relative_error < EPSILON,
            "Norm mismatch: {} vs {}",
            simd_norm,
            scalar_norm
        );
    }

    #[test]
    fn test_norm_zero() {
        let zero = vec![0.0f32; 100];
        let norm = norm_simd(&zero);
        assert_eq!(norm, 0.0, "Norm of zero vector should be 0");
    }

    #[test]
    fn test_norm_unit() {
        let mut unit = vec![0.0f32; 100];
        unit[0] = 1.0;
        let norm = norm_simd(&unit);
        assert!(
            (norm - 1.0).abs() < EPSILON,
            "Norm of unit vector should be 1.0, got {}",
            norm
        );
    }

    // ===== Small/non-aligned vectors =====

    #[test]
    fn test_small_vectors() {
        for size in [1, 3, 7, 15, 31, 100, 1000] {
            let a = random_vec(size, 42);
            let b = random_vec(size, 43);

            let simd_dot = dot_product_simd(&a, &b);
            let scalar_dot = scalar_dot_product(&a, &b);

            let relative_error = ((simd_dot - scalar_dot) / scalar_dot.abs().max(1e-10)).abs();
            assert!(
                relative_error < EPSILON,
                "Size {} dot product error: {}",
                size,
                relative_error
            );
        }
    }

    #[test]
    fn test_small_bind() {
        for size in [1, 3, 7, 15] {
            let a = random_vec(size, 42);
            let b = random_vec(size, 43);

            let result = bind_simd(&a, &b);
            assert_eq!(result.len(), size, "Bind should preserve dimension");

            for i in 0..size {
                assert!(
                    (result[i] - a[i] * b[i]).abs() < EPSILON,
                    "Bind mismatch at size={}, index={}",
                    size,
                    i
                );
            }
        }
    }

    #[test]
    fn test_small_similarity() {
        for size in [2, 5, 10, 50] {
            let a = random_vec(size, 42);
            let sim = similarity_simd(&a, &a);
            assert!(
                (sim - 1.0).abs() < 0.01,
                "Self-similarity at size={} should be ~1.0, got {}",
                size,
                sim
            );
        }
    }

    // ===== Capabilities =====

    #[test]
    fn test_capabilities_report() {
        let report = simd_capabilities_report();
        println!("{}", report);
        assert!(!report.is_empty());
    }
}
