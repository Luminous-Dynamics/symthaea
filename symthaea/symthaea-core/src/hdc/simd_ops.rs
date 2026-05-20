// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SIMD-Optimized Operations for Binary Hypervectors
//!
//! This module provides high-performance implementations of BinaryHV operations
//! using explicit SIMD intrinsics for maximum throughput.
//!
//! # Performance Targets
//! - `bind` (XOR): 5-10ns (vs ~80ns scalar)
//! - `similarity` (popcount): 10-20ns (vs ~160ns scalar)
//! - `bundle` (majority vote): 50-100ns (vs ~1000ns scalar)
//!
//! # Architecture Support
//! - AVX-512 (x86_64): 512-bit operations (when available)
//! - AVX2 (x86_64): 256-bit operations
//! - SSE4.1 (x86_64): 128-bit operations (fallback)
//! - NEON (AArch64): 128-bit operations
//! - Portable: Safe fallback using auto-vectorization hints
//!
//! Feature detection is centralized in [`super::simd_detect`].

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[allow(unused_imports)]
use super::simd_detect::{
    has_avx2, has_avx512_vpopcntdq, has_avx512bw, has_avx512f, has_popcnt, has_sse41,
};

/// SIMD-optimized XOR (bind) operation for 2048 bytes
///
/// Uses AVX-512 or AVX2 when available for maximum throughput.
/// Feature detection is cached for 2-3x improvement on hot paths.
///
/// # Performance Hierarchy
/// - AVX-512: 512-bit operations (32 iterations for 2048 bytes)
/// - AVX2: 256-bit operations (64 iterations)
/// - SSE4.1: 128-bit operations (128 iterations)
/// - Scalar: 64-bit operations (256 iterations)
///
/// # Safety
/// Requires proper alignment and assumes input arrays are exactly 2048 bytes.
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn bind_simd(a: &[u8; 2048], b: &[u8; 2048]) -> [u8; 2048] {
    let mut result = [0u8; 2048];

    // Try AVX-512 first (512-bit = 64 bytes per operation)
    if has_avx512f() {
        unsafe { bind_avx512(a, b, &mut result) };
    }
    // Try AVX2 (256-bit = 32 bytes per operation)
    else if has_avx2() {
        unsafe { bind_avx2(a, b, &mut result) };
    }
    // Fall back to SSE4.1 (128-bit = 16 bytes per operation)
    else if has_sse41() {
        unsafe { bind_sse41(a, b, &mut result) };
    }
    // Scalar fallback with manual unrolling
    else {
        bind_scalar_unrolled(a, b, &mut result);
    }

    result
}

/// AVX-512 implementation of XOR (64 bytes per iteration = 32 iterations for 2048 bytes)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn bind_avx512(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = result.as_mut_ptr() as *mut __m512i;

        // 2048 bytes / 64 bytes = 32 iterations
        // Unroll by 2 for better instruction-level parallelism
        for i in (0..32).step_by(2) {
            let a0 = _mm512_loadu_si512(a_ptr.add(i));
            let b0 = _mm512_loadu_si512(b_ptr.add(i));
            let a1 = _mm512_loadu_si512(a_ptr.add(i + 1));
            let b1 = _mm512_loadu_si512(b_ptr.add(i + 1));

            let r0 = _mm512_xor_si512(a0, b0);
            let r1 = _mm512_xor_si512(a1, b1);

            _mm512_storeu_si512(r_ptr.add(i), r0);
            _mm512_storeu_si512(r_ptr.add(i + 1), r1);
        }
    }
}

/// AVX2 implementation of XOR (32 bytes per iteration = 64 iterations for 2048 bytes)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bind_avx2(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let b_ptr = b.as_ptr() as *const __m256i;
        let r_ptr = result.as_mut_ptr() as *mut __m256i;

        // 2048 bytes / 32 bytes = 64 iterations
        // Unroll by 4 for better instruction-level parallelism
        // Source arrays are align(32) from BinaryHV, so use aligned loads
        for i in (0..64).step_by(4) {
            let a0 = _mm256_load_si256(a_ptr.add(i));
            let b0 = _mm256_load_si256(b_ptr.add(i));
            let a1 = _mm256_load_si256(a_ptr.add(i + 1));
            let b1 = _mm256_load_si256(b_ptr.add(i + 1));
            let a2 = _mm256_load_si256(a_ptr.add(i + 2));
            let b2 = _mm256_load_si256(b_ptr.add(i + 2));
            let a3 = _mm256_load_si256(a_ptr.add(i + 3));
            let b3 = _mm256_load_si256(b_ptr.add(i + 3));

            let r0 = _mm256_xor_si256(a0, b0);
            let r1 = _mm256_xor_si256(a1, b1);
            let r2 = _mm256_xor_si256(a2, b2);
            let r3 = _mm256_xor_si256(a3, b3);

            _mm256_storeu_si256(r_ptr.add(i), r0);
            _mm256_storeu_si256(r_ptr.add(i + 1), r1);
            _mm256_storeu_si256(r_ptr.add(i + 2), r2);
            _mm256_storeu_si256(r_ptr.add(i + 3), r3);
        }
    }
}

/// SSE4.1 implementation of XOR (16 bytes per iteration = 128 iterations for 2048 bytes)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn bind_sse41(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m128i;
        let b_ptr = b.as_ptr() as *const __m128i;
        let r_ptr = result.as_mut_ptr() as *mut __m128i;

        // 2048 bytes / 16 bytes = 128 iterations
        // Unroll by 4
        for i in (0..128).step_by(4) {
            let a0 = _mm_loadu_si128(a_ptr.add(i));
            let b0 = _mm_loadu_si128(b_ptr.add(i));
            let a1 = _mm_loadu_si128(a_ptr.add(i + 1));
            let b1 = _mm_loadu_si128(b_ptr.add(i + 1));
            let a2 = _mm_loadu_si128(a_ptr.add(i + 2));
            let b2 = _mm_loadu_si128(b_ptr.add(i + 2));
            let a3 = _mm_loadu_si128(a_ptr.add(i + 3));
            let b3 = _mm_loadu_si128(b_ptr.add(i + 3));

            let r0 = _mm_xor_si128(a0, b0);
            let r1 = _mm_xor_si128(a1, b1);
            let r2 = _mm_xor_si128(a2, b2);
            let r3 = _mm_xor_si128(a3, b3);

            _mm_storeu_si128(r_ptr.add(i), r0);
            _mm_storeu_si128(r_ptr.add(i + 1), r1);
            _mm_storeu_si128(r_ptr.add(i + 2), r2);
            _mm_storeu_si128(r_ptr.add(i + 3), r3);
        }
    }
}

/// Scalar fallback with chunked u64 ops for auto-vectorization
#[inline]
fn bind_scalar_unrolled(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    for (r_chunk, (a_chunk, b_chunk)) in result
        .chunks_exact_mut(8)
        .zip(a.chunks_exact(8).zip(b.chunks_exact(8)))
    {
        let av = u64::from_ne_bytes([
            a_chunk[0], a_chunk[1], a_chunk[2], a_chunk[3], a_chunk[4], a_chunk[5], a_chunk[6],
            a_chunk[7],
        ]);
        let bv = u64::from_ne_bytes([
            b_chunk[0], b_chunk[1], b_chunk[2], b_chunk[3], b_chunk[4], b_chunk[5], b_chunk[6],
            b_chunk[7],
        ]);
        r_chunk.copy_from_slice(&(av ^ bv).to_ne_bytes());
    }
}

/// SIMD-optimized AND (intersection) operation for 2048 bytes
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn intersection_simd(a: &[u8; 2048], b: &[u8; 2048]) -> [u8; 2048] {
    let mut result = [0u8; 2048];
    if has_avx512f() {
        unsafe { intersection_avx512(a, b, &mut result) };
    } else if has_avx2() {
        unsafe { intersection_avx2(a, b, &mut result) };
    } else if has_sse41() {
        unsafe { intersection_sse41(a, b, &mut result) };
    } else {
        intersection_scalar_unrolled(a, b, &mut result);
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn intersection_avx512(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = result.as_mut_ptr() as *mut __m512i;
        for i in (0..32).step_by(2) {
            let r0 = _mm512_and_si512(
                _mm512_loadu_si512(a_ptr.add(i)),
                _mm512_loadu_si512(b_ptr.add(i)),
            );
            let r1 = _mm512_and_si512(
                _mm512_loadu_si512(a_ptr.add(i + 1)),
                _mm512_loadu_si512(b_ptr.add(i + 1)),
            );
            _mm512_storeu_si512(r_ptr.add(i), r0);
            _mm512_storeu_si512(r_ptr.add(i + 1), r1);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn intersection_avx2(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let b_ptr = b.as_ptr() as *const __m256i;
        let r_ptr = result.as_mut_ptr() as *mut __m256i;
        // Source arrays are align(32) from BinaryHV, so use aligned loads
        for i in (0..64).step_by(4) {
            _mm256_storeu_si256(
                r_ptr.add(i),
                _mm256_and_si256(
                    _mm256_load_si256(a_ptr.add(i)),
                    _mm256_load_si256(b_ptr.add(i)),
                ),
            );
            _mm256_storeu_si256(
                r_ptr.add(i + 1),
                _mm256_and_si256(
                    _mm256_load_si256(a_ptr.add(i + 1)),
                    _mm256_load_si256(b_ptr.add(i + 1)),
                ),
            );
            _mm256_storeu_si256(
                r_ptr.add(i + 2),
                _mm256_and_si256(
                    _mm256_load_si256(a_ptr.add(i + 2)),
                    _mm256_load_si256(b_ptr.add(i + 2)),
                ),
            );
            _mm256_storeu_si256(
                r_ptr.add(i + 3),
                _mm256_and_si256(
                    _mm256_load_si256(a_ptr.add(i + 3)),
                    _mm256_load_si256(b_ptr.add(i + 3)),
                ),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn intersection_sse41(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m128i;
        let b_ptr = b.as_ptr() as *const __m128i;
        let r_ptr = result.as_mut_ptr() as *mut __m128i;
        for i in (0..128).step_by(4) {
            _mm_storeu_si128(
                r_ptr.add(i),
                _mm_and_si128(_mm_loadu_si128(a_ptr.add(i)), _mm_loadu_si128(b_ptr.add(i))),
            );
            _mm_storeu_si128(
                r_ptr.add(i + 1),
                _mm_and_si128(
                    _mm_loadu_si128(a_ptr.add(i + 1)),
                    _mm_loadu_si128(b_ptr.add(i + 1)),
                ),
            );
            _mm_storeu_si128(
                r_ptr.add(i + 2),
                _mm_and_si128(
                    _mm_loadu_si128(a_ptr.add(i + 2)),
                    _mm_loadu_si128(b_ptr.add(i + 2)),
                ),
            );
            _mm_storeu_si128(
                r_ptr.add(i + 3),
                _mm_and_si128(
                    _mm_loadu_si128(a_ptr.add(i + 3)),
                    _mm_loadu_si128(b_ptr.add(i + 3)),
                ),
            );
        }
    }
}

#[inline]
fn intersection_scalar_unrolled(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    for (r_chunk, (a_chunk, b_chunk)) in result
        .chunks_exact_mut(8)
        .zip(a.chunks_exact(8).zip(b.chunks_exact(8)))
    {
        let av = u64::from_ne_bytes([
            a_chunk[0], a_chunk[1], a_chunk[2], a_chunk[3], a_chunk[4], a_chunk[5], a_chunk[6],
            a_chunk[7],
        ]);
        let bv = u64::from_ne_bytes([
            b_chunk[0], b_chunk[1], b_chunk[2], b_chunk[3], b_chunk[4], b_chunk[5], b_chunk[6],
            b_chunk[7],
        ]);
        r_chunk.copy_from_slice(&(av & bv).to_ne_bytes());
    }
}

/// SIMD-optimized OR (union) operation for 2048 bytes
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn union_simd(a: &[u8; 2048], b: &[u8; 2048]) -> [u8; 2048] {
    let mut result = [0u8; 2048];
    if has_avx512f() {
        unsafe { union_avx512(a, b, &mut result) };
    } else if has_avx2() {
        unsafe { union_avx2(a, b, &mut result) };
    } else if has_sse41() {
        unsafe { union_sse41(a, b, &mut result) };
    } else {
        union_scalar_unrolled(a, b, &mut result);
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn union_avx512(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = result.as_mut_ptr() as *mut __m512i;
        for i in (0..32).step_by(2) {
            let r0 = _mm512_or_si512(
                _mm512_loadu_si512(a_ptr.add(i)),
                _mm512_loadu_si512(b_ptr.add(i)),
            );
            let r1 = _mm512_or_si512(
                _mm512_loadu_si512(a_ptr.add(i + 1)),
                _mm512_loadu_si512(b_ptr.add(i + 1)),
            );
            _mm512_storeu_si512(r_ptr.add(i), r0);
            _mm512_storeu_si512(r_ptr.add(i + 1), r1);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn union_avx2(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let b_ptr = b.as_ptr() as *const __m256i;
        let r_ptr = result.as_mut_ptr() as *mut __m256i;
        // Source arrays are align(32) from BinaryHV, so use aligned loads
        for i in (0..64).step_by(4) {
            _mm256_storeu_si256(
                r_ptr.add(i),
                _mm256_or_si256(
                    _mm256_load_si256(a_ptr.add(i)),
                    _mm256_load_si256(b_ptr.add(i)),
                ),
            );
            _mm256_storeu_si256(
                r_ptr.add(i + 1),
                _mm256_or_si256(
                    _mm256_load_si256(a_ptr.add(i + 1)),
                    _mm256_load_si256(b_ptr.add(i + 1)),
                ),
            );
            _mm256_storeu_si256(
                r_ptr.add(i + 2),
                _mm256_or_si256(
                    _mm256_load_si256(a_ptr.add(i + 2)),
                    _mm256_load_si256(b_ptr.add(i + 2)),
                ),
            );
            _mm256_storeu_si256(
                r_ptr.add(i + 3),
                _mm256_or_si256(
                    _mm256_load_si256(a_ptr.add(i + 3)),
                    _mm256_load_si256(b_ptr.add(i + 3)),
                ),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn union_sse41(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m128i;
        let b_ptr = b.as_ptr() as *const __m128i;
        let r_ptr = result.as_mut_ptr() as *mut __m128i;
        for i in (0..128).step_by(4) {
            _mm_storeu_si128(
                r_ptr.add(i),
                _mm_or_si128(_mm_loadu_si128(a_ptr.add(i)), _mm_loadu_si128(b_ptr.add(i))),
            );
            _mm_storeu_si128(
                r_ptr.add(i + 1),
                _mm_or_si128(
                    _mm_loadu_si128(a_ptr.add(i + 1)),
                    _mm_loadu_si128(b_ptr.add(i + 1)),
                ),
            );
            _mm_storeu_si128(
                r_ptr.add(i + 2),
                _mm_or_si128(
                    _mm_loadu_si128(a_ptr.add(i + 2)),
                    _mm_loadu_si128(b_ptr.add(i + 2)),
                ),
            );
            _mm_storeu_si128(
                r_ptr.add(i + 3),
                _mm_or_si128(
                    _mm_loadu_si128(a_ptr.add(i + 3)),
                    _mm_loadu_si128(b_ptr.add(i + 3)),
                ),
            );
        }
    }
}

#[inline]
fn union_scalar_unrolled(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    for (r_chunk, (a_chunk, b_chunk)) in result
        .chunks_exact_mut(8)
        .zip(a.chunks_exact(8).zip(b.chunks_exact(8)))
    {
        let av = u64::from_ne_bytes([
            a_chunk[0], a_chunk[1], a_chunk[2], a_chunk[3], a_chunk[4], a_chunk[5], a_chunk[6],
            a_chunk[7],
        ]);
        let bv = u64::from_ne_bytes([
            b_chunk[0], b_chunk[1], b_chunk[2], b_chunk[3], b_chunk[4], b_chunk[5], b_chunk[6],
            b_chunk[7],
        ]);
        r_chunk.copy_from_slice(&(av | bv).to_ne_bytes());
    }
}

/// SIMD-optimized population count (Hamming weight) for similarity calculation
///
/// Returns the number of matching bits between two 2048-byte arrays.
/// Feature detection is cached for 2-3x improvement on hot paths.
///
/// Uses AVX-512 VPOPCNTDQ when available, else AVX2 with POPCNT.
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn matching_bits_simd(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    // Try AVX-512 with native VPOPCNTDQ (fastest path)
    if has_avx512f() && has_avx512_vpopcntdq() {
        unsafe { matching_bits_avx512_vpopcntdq(a, b) }
    }
    // AVX-512 without VPOPCNTDQ
    else if has_avx512f() && has_popcnt() {
        unsafe { matching_bits_avx512_popcnt(a, b) }
    }
    // AVX2 with POPCNT
    else if has_avx2() && has_popcnt() {
        unsafe { matching_bits_avx2_popcnt(a, b) }
    }
    // POPCNT only
    else if has_popcnt() {
        matching_bits_popcnt(a, b)
    }
    // Scalar fallback
    else {
        matching_bits_scalar(a, b)
    }
}

/// AVX-512 with native VPOPCNTDQ instruction (fastest path)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
#[inline]
unsafe fn matching_bits_avx512_vpopcntdq(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;

        let mut total = _mm512_setzero_si512();

        // Process 64 bytes at a time
        for i in 0..32 {
            let va = _mm512_loadu_si512(a_ptr.add(i));
            let vb = _mm512_loadu_si512(b_ptr.add(i));
            let diff = _mm512_xor_si512(va, vb);

            // Native 64-bit popcount on each of 8 qwords
            let popcnt = _mm512_popcnt_epi64(diff);
            total = _mm512_add_epi64(total, popcnt);
        }

        // Horizontal sum of 8 64-bit values
        let differing = _mm512_reduce_add_epi64(total) as u64;

        // Total bits - differing bits = matching bits
        (16_384 - differing) as u32
    }
}

/// AVX-512 with scalar POPCNT fallback
/// SAFETY: BinaryHV is #[repr(align(32))] so u64 reads are naturally aligned.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "popcnt")]
#[inline]
unsafe fn matching_bits_avx512_popcnt(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    unsafe {
        let a_ptr = a.as_ptr() as *const u64;
        let b_ptr = b.as_ptr() as *const u64;

        let mut total: u64 = 0;

        // 2048 bytes / 8 bytes = 256 u64s
        // Process 8 at a time for better ILP
        for i in (0..256).step_by(8) {
            let xor0 = *a_ptr.add(i) ^ *b_ptr.add(i);
            let xor1 = *a_ptr.add(i + 1) ^ *b_ptr.add(i + 1);
            let xor2 = *a_ptr.add(i + 2) ^ *b_ptr.add(i + 2);
            let xor3 = *a_ptr.add(i + 3) ^ *b_ptr.add(i + 3);
            let xor4 = *a_ptr.add(i + 4) ^ *b_ptr.add(i + 4);
            let xor5 = *a_ptr.add(i + 5) ^ *b_ptr.add(i + 5);
            let xor6 = *a_ptr.add(i + 6) ^ *b_ptr.add(i + 6);
            let xor7 = *a_ptr.add(i + 7) ^ *b_ptr.add(i + 7);

            total += _popcnt64(xor0 as i64) as u64;
            total += _popcnt64(xor1 as i64) as u64;
            total += _popcnt64(xor2 as i64) as u64;
            total += _popcnt64(xor3 as i64) as u64;
            total += _popcnt64(xor4 as i64) as u64;
            total += _popcnt64(xor5 as i64) as u64;
            total += _popcnt64(xor6 as i64) as u64;
            total += _popcnt64(xor7 as i64) as u64;
        }

        (16_384 - total) as u32
    }
}

/// AVX2 + POPCNT implementation
/// XOR bytes together, then popcount to find differing bits.
/// SAFETY: BinaryHV is #[repr(align(32))] so u64 reads are naturally aligned.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "popcnt")]
#[inline]
unsafe fn matching_bits_avx2_popcnt(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    unsafe {
        let a_ptr = a.as_ptr() as *const u64;
        let b_ptr = b.as_ptr() as *const u64;

        let mut total: u64 = 0;

        // 2048 bytes / 8 bytes = 256 u64s, process 4 at a time for ILP
        for i in (0..256).step_by(4) {
            let xor0 = *a_ptr.add(i) ^ *b_ptr.add(i);
            let xor1 = *a_ptr.add(i + 1) ^ *b_ptr.add(i + 1);
            let xor2 = *a_ptr.add(i + 2) ^ *b_ptr.add(i + 2);
            let xor3 = *a_ptr.add(i + 3) ^ *b_ptr.add(i + 3);

            // Count DIFFERING bits (popcount of XOR)
            // Matching = total bits - differing
            total += _popcnt64(xor0 as i64) as u64;
            total += _popcnt64(xor1 as i64) as u64;
            total += _popcnt64(xor2 as i64) as u64;
            total += _popcnt64(xor3 as i64) as u64;
        }

        // Total bits - differing bits = matching bits
        (16_384 - total) as u32
    }
}

/// POPCNT-only implementation (fallback when AVX2 not available)
#[cfg(target_arch = "x86_64")]
fn matching_bits_popcnt(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    let mut differing: u64 = 0;

    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let av = u64::from_ne_bytes([
            a_chunk[0], a_chunk[1], a_chunk[2], a_chunk[3], a_chunk[4], a_chunk[5], a_chunk[6],
            a_chunk[7],
        ]);
        let bv = u64::from_ne_bytes([
            b_chunk[0], b_chunk[1], b_chunk[2], b_chunk[3], b_chunk[4], b_chunk[5], b_chunk[6],
            b_chunk[7],
        ]);
        differing += (av ^ bv).count_ones() as u64;
    }

    (16_384 - differing) as u32
}

/// Scalar fallback implementation
#[inline]
fn matching_bits_scalar(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (!(x ^ y)).count_ones())
        .sum()
}

/// SIMD-optimized NOT (invert) operation
/// Feature detection is cached for 2-3x improvement on hot paths.
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn invert_simd(a: &[u8; 2048]) -> [u8; 2048] {
    let mut result = [0u8; 2048];

    // Using cached feature detection for 2-3x speedup
    if has_avx2() {
        unsafe { invert_avx2(a, &mut result) };
    } else if has_sse41() {
        unsafe { invert_sse41(a, &mut result) };
    } else {
        invert_scalar(a, &mut result);
    }

    result
}

/// AVX2 implementation of NOT
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn invert_avx2(a: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = result.as_mut_ptr() as *mut __m256i;
        let ones = _mm256_set1_epi8(-1i8); // All 1s

        // Source array is align(32) from BinaryHV, so use aligned loads
        for i in (0..64).step_by(4) {
            let a0 = _mm256_load_si256(a_ptr.add(i));
            let a1 = _mm256_load_si256(a_ptr.add(i + 1));
            let a2 = _mm256_load_si256(a_ptr.add(i + 2));
            let a3 = _mm256_load_si256(a_ptr.add(i + 3));

            // XOR with all 1s = NOT
            let r0 = _mm256_xor_si256(a0, ones);
            let r1 = _mm256_xor_si256(a1, ones);
            let r2 = _mm256_xor_si256(a2, ones);
            let r3 = _mm256_xor_si256(a3, ones);

            _mm256_storeu_si256(r_ptr.add(i), r0);
            _mm256_storeu_si256(r_ptr.add(i + 1), r1);
            _mm256_storeu_si256(r_ptr.add(i + 2), r2);
            _mm256_storeu_si256(r_ptr.add(i + 3), r3);
        }
    }
}

/// SSE4.1 implementation of NOT
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn invert_sse41(a: &[u8; 2048], result: &mut [u8; 2048]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m128i;
        let r_ptr = result.as_mut_ptr() as *mut __m128i;
        let ones = _mm_set1_epi8(-1i8);

        for i in (0..128).step_by(4) {
            let a0 = _mm_loadu_si128(a_ptr.add(i));
            let a1 = _mm_loadu_si128(a_ptr.add(i + 1));
            let a2 = _mm_loadu_si128(a_ptr.add(i + 2));
            let a3 = _mm_loadu_si128(a_ptr.add(i + 3));

            let r0 = _mm_xor_si128(a0, ones);
            let r1 = _mm_xor_si128(a1, ones);
            let r2 = _mm_xor_si128(a2, ones);
            let r3 = _mm_xor_si128(a3, ones);

            _mm_storeu_si128(r_ptr.add(i), r0);
            _mm_storeu_si128(r_ptr.add(i + 1), r1);
            _mm_storeu_si128(r_ptr.add(i + 2), r2);
            _mm_storeu_si128(r_ptr.add(i + 3), r3);
        }
    }
}

/// Scalar fallback for NOT
#[inline]
fn invert_scalar(a: &[u8; 2048], result: &mut [u8; 2048]) {
    for (r_chunk, a_chunk) in result.chunks_exact_mut(8).zip(a.chunks_exact(8)) {
        let av = u64::from_ne_bytes([
            a_chunk[0], a_chunk[1], a_chunk[2], a_chunk[3], a_chunk[4], a_chunk[5], a_chunk[6],
            a_chunk[7],
        ]);
        r_chunk.copy_from_slice(&(!av).to_ne_bytes());
    }
}

/// SIMD-optimized Hamming distance
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn hamming_distance_simd(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    // Matching bits + hamming distance = total bits
    // So hamming = total - matching
    16_384 - matching_bits_simd(a, b)
}

// =============================================================================
// SIMD-OPTIMIZED BUNDLE (MAJORITY VOTING)
// =============================================================================

/// SIMD-optimized bundle (majority vote) operation
///
/// This is the most compute-intensive HDC operation. For N vectors of 16,384 bits:
/// - Scalar: O(N * 16384) with poor cache locality
/// - SIMD: O(N * 16384 / 256) = O(N * 64) with AVX2
///
/// # Algorithm
/// For each bit position, count how many vectors have 1.
/// If count > N/2, result bit = 1, else 0.
///
/// # Note on SIMD
/// Majority-vote bundle on packed binary HVs is inherently hard to beat with
/// hand-written SIMD intrinsics because the per-bit extraction (shift+mask)
/// is the bottleneck, and LLVM auto-vectorizes it well. The scalar implementation
/// below is the fastest path on all tested hardware.
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn bundle_simd(vectors: &[&[u8; 2048]]) -> [u8; 2048] {
    if vectors.is_empty() {
        return [0u8; 2048];
    }
    if vectors.len() == 1 {
        return *vectors[0];
    }

    bundle_scalar_optimized(vectors)
}

/// Bundle using scalar bit counting with LLVM auto-vectorization.
///
/// Note: Hand-written AVX2 intrinsics for majority-vote bundle on packed
/// binary HVs do NOT outperform well-optimized scalar code. The per-bit
/// counting pattern (extract 8 bits per byte, count across N vectors) is
/// inherently hard to beat with SIMD when the data is bit-packed. LLVM
/// auto-vectorizes the shift+mask+add inner loop effectively.
///
/// A future optimization would be to transpose the bit layout at the HV
/// storage level (all bit-0s contiguous, then all bit-1s, etc.), enabling
/// true SIMD accumulation. That's a data structure change, not an
/// algorithm change.
#[cfg(target_arch = "x86_64")]
#[inline]
fn bundle_avx2_friendly(vectors: &[&[u8; 2048]]) -> [u8; 2048] {
    // Reuse the scalar implementation which LLVM auto-vectorizes well
    bundle_scalar_optimized(vectors)
}

/// Optimized scalar bundle with better cache locality
#[cfg(target_arch = "x86_64")]
#[inline]
fn bundle_scalar_optimized(vectors: &[&[u8; 2048]]) -> [u8; 2048] {
    let n = vectors.len();
    let threshold = n / 2;
    let mut result = [0u8; 2048];

    // Process byte by byte for cache efficiency
    for byte_idx in 0..2048 {
        let mut bit_counts = [0i16; 8];

        // Accumulate counts from all vectors
        for vec in vectors {
            let byte = vec[byte_idx];
            // Unrolled bit extraction
            bit_counts[0] += (byte & 1) as i16;
            bit_counts[1] += ((byte >> 1) & 1) as i16;
            bit_counts[2] += ((byte >> 2) & 1) as i16;
            bit_counts[3] += ((byte >> 3) & 1) as i16;
            bit_counts[4] += ((byte >> 4) & 1) as i16;
            bit_counts[5] += ((byte >> 5) & 1) as i16;
            bit_counts[6] += ((byte >> 6) & 1) as i16;
            bit_counts[7] += ((byte >> 7) & 1) as i16;
        }

        // Threshold comparison
        let mut result_byte = 0u8;
        for bit in 0..8 {
            if bit_counts[bit] as usize > threshold {
                result_byte |= 1 << bit;
            }
        }
        result[byte_idx] = result_byte;
    }

    result
}

/// Bundle multiple BinaryHV vectors (convenience function for slices)
#[cfg(target_arch = "x86_64")]
pub fn bundle_simd_slice(vectors: &[[u8; 2048]]) -> [u8; 2048] {
    let refs: Vec<&[u8; 2048]> = vectors.iter().collect();
    bundle_simd(&refs)
}

// =============================================================================
// NEON (AArch64) IMPLEMENTATIONS
// =============================================================================

/// NEON XOR bind: 128-bit veorq_u8, unrolled 4x (32 iterations for 2048 bytes)
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn bind_neon(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let r_ptr = result.as_mut_ptr();
    for i in (0..128).step_by(4) {
        let off0 = i * 16;
        let off1 = (i + 1) * 16;
        let off2 = (i + 2) * 16;
        let off3 = (i + 3) * 16;
        let r0 = veorq_u8(vld1q_u8(a_ptr.add(off0)), vld1q_u8(b_ptr.add(off0)));
        let r1 = veorq_u8(vld1q_u8(a_ptr.add(off1)), vld1q_u8(b_ptr.add(off1)));
        let r2 = veorq_u8(vld1q_u8(a_ptr.add(off2)), vld1q_u8(b_ptr.add(off2)));
        let r3 = veorq_u8(vld1q_u8(a_ptr.add(off3)), vld1q_u8(b_ptr.add(off3)));
        vst1q_u8(r_ptr.add(off0), r0);
        vst1q_u8(r_ptr.add(off1), r1);
        vst1q_u8(r_ptr.add(off2), r2);
        vst1q_u8(r_ptr.add(off3), r3);
    }
}

/// NEON AND intersection: vandq_u8
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn intersection_neon(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let r_ptr = result.as_mut_ptr();
    for i in (0..128).step_by(4) {
        let off0 = i * 16;
        let off1 = (i + 1) * 16;
        let off2 = (i + 2) * 16;
        let off3 = (i + 3) * 16;
        vst1q_u8(
            r_ptr.add(off0),
            vandq_u8(vld1q_u8(a_ptr.add(off0)), vld1q_u8(b_ptr.add(off0))),
        );
        vst1q_u8(
            r_ptr.add(off1),
            vandq_u8(vld1q_u8(a_ptr.add(off1)), vld1q_u8(b_ptr.add(off1))),
        );
        vst1q_u8(
            r_ptr.add(off2),
            vandq_u8(vld1q_u8(a_ptr.add(off2)), vld1q_u8(b_ptr.add(off2))),
        );
        vst1q_u8(
            r_ptr.add(off3),
            vandq_u8(vld1q_u8(a_ptr.add(off3)), vld1q_u8(b_ptr.add(off3))),
        );
    }
}

/// NEON OR union: vorrq_u8
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn union_neon(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let r_ptr = result.as_mut_ptr();
    for i in (0..128).step_by(4) {
        let off0 = i * 16;
        let off1 = (i + 1) * 16;
        let off2 = (i + 2) * 16;
        let off3 = (i + 3) * 16;
        vst1q_u8(
            r_ptr.add(off0),
            vorrq_u8(vld1q_u8(a_ptr.add(off0)), vld1q_u8(b_ptr.add(off0))),
        );
        vst1q_u8(
            r_ptr.add(off1),
            vorrq_u8(vld1q_u8(a_ptr.add(off1)), vld1q_u8(b_ptr.add(off1))),
        );
        vst1q_u8(
            r_ptr.add(off2),
            vorrq_u8(vld1q_u8(a_ptr.add(off2)), vld1q_u8(b_ptr.add(off2))),
        );
        vst1q_u8(
            r_ptr.add(off3),
            vorrq_u8(vld1q_u8(a_ptr.add(off3)), vld1q_u8(b_ptr.add(off3))),
        );
    }
}

/// NEON NOT invert: vmvnq_u8
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn invert_neon(a: &[u8; 2048], result: &mut [u8; 2048]) {
    let a_ptr = a.as_ptr();
    let r_ptr = result.as_mut_ptr();
    for i in (0..128).step_by(4) {
        let off0 = i * 16;
        let off1 = (i + 1) * 16;
        let off2 = (i + 2) * 16;
        let off3 = (i + 3) * 16;
        vst1q_u8(r_ptr.add(off0), vmvnq_u8(vld1q_u8(a_ptr.add(off0))));
        vst1q_u8(r_ptr.add(off1), vmvnq_u8(vld1q_u8(a_ptr.add(off1))));
        vst1q_u8(r_ptr.add(off2), vmvnq_u8(vld1q_u8(a_ptr.add(off2))));
        vst1q_u8(r_ptr.add(off3), vmvnq_u8(vld1q_u8(a_ptr.add(off3))));
    }
}

/// NEON popcount via vcntq_u8 + widening horizontal adds.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn matching_bits_neon(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut differing: u64 = 0;

    for i in 0..128 {
        let off = i * 16;
        let xor = veorq_u8(vld1q_u8(a_ptr.add(off)), vld1q_u8(b_ptr.add(off)));
        let cnt8 = vcntq_u8(xor);
        let cnt16 = vpaddlq_u8(cnt8);
        let cnt32 = vpaddlq_u16(cnt16);
        let cnt64 = vpaddlq_u32(cnt32);
        differing += vgetq_lane_u64(cnt64, 0) + vgetq_lane_u64(cnt64, 1);
    }

    (16_384 - differing) as u32
}

// =============================================================================
// NON-x86_64 DISPATCH FUNCTIONS (NEON on aarch64, scalar elsewhere)
// =============================================================================

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn bind_simd(a: &[u8; 2048], b: &[u8; 2048]) -> [u8; 2048] {
    let mut result = [0u8; 2048];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is guaranteed available on AArch64.
    // Input arrays are fixed-size [u8; 2048] so length and alignment are compile-time verified.
    unsafe {
        bind_neon(a, b, &mut result);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        bind_scalar_unrolled(a, b, &mut result);
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn intersection_simd(a: &[u8; 2048], b: &[u8; 2048]) -> [u8; 2048] {
    let mut result = [0u8; 2048];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is guaranteed available on AArch64.
    // Input arrays are fixed-size [u8; 2048] so length and alignment are compile-time verified.
    unsafe {
        intersection_neon(a, b, &mut result);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        intersection_scalar_unrolled(a, b, &mut result);
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn union_simd(a: &[u8; 2048], b: &[u8; 2048]) -> [u8; 2048] {
    let mut result = [0u8; 2048];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is guaranteed available on AArch64.
    // Input arrays are fixed-size [u8; 2048] so length and alignment are compile-time verified.
    unsafe {
        union_neon(a, b, &mut result);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        union_scalar_unrolled(a, b, &mut result);
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn matching_bits_simd(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is guaranteed available on AArch64.
    // Input arrays are fixed-size [u8; 2048] so length and alignment are compile-time verified.
    unsafe {
        return matching_bits_neon(a, b);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        matching_bits_scalar(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn invert_simd(a: &[u8; 2048]) -> [u8; 2048] {
    let mut result = [0u8; 2048];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is guaranteed available on AArch64.
    // Input arrays are fixed-size [u8; 2048] so length and alignment are compile-time verified.
    unsafe {
        invert_neon(a, &mut result);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        invert_scalar(a, &mut result);
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn hamming_distance_simd(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    16_384 - matching_bits_simd(a, b)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn bundle_simd(vectors: &[&[u8; 2048]]) -> [u8; 2048] {
    if vectors.is_empty() {
        return [0u8; 2048];
    }
    if vectors.len() == 1 {
        return *vectors[0];
    }
    bundle_scalar_portable(vectors)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn bundle_simd_slice(vectors: &[[u8; 2048]]) -> [u8; 2048] {
    let refs: Vec<&[u8; 2048]> = vectors.iter().collect();
    bundle_simd(&refs)
}

/// Scalar bundle with good cache locality (used on all non-x86 platforms).
fn bundle_scalar_portable(vectors: &[&[u8; 2048]]) -> [u8; 2048] {
    if vectors.is_empty() {
        return [0u8; 2048];
    }
    if vectors.len() == 1 {
        return *vectors[0];
    }

    let n = vectors.len();
    let threshold = n / 2;
    let mut result = [0u8; 2048];

    for byte_idx in 0..2048 {
        let mut bit_counts = [0i16; 8];

        for vec in vectors {
            let byte = vec[byte_idx];
            bit_counts[0] += (byte & 1) as i16;
            bit_counts[1] += ((byte >> 1) & 1) as i16;
            bit_counts[2] += ((byte >> 2) & 1) as i16;
            bit_counts[3] += ((byte >> 3) & 1) as i16;
            bit_counts[4] += ((byte >> 4) & 1) as i16;
            bit_counts[5] += ((byte >> 5) & 1) as i16;
            bit_counts[6] += ((byte >> 6) & 1) as i16;
            bit_counts[7] += ((byte >> 7) & 1) as i16;
        }

        let mut result_byte = 0u8;
        for bit in 0..8 {
            if bit_counts[bit] as usize > threshold {
                result_byte |= 1 << bit;
            }
        }
        result[byte_idx] = result_byte;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::binary_hv::BinaryHV;

    #[test]
    fn test_bind_simd_matches_scalar() {
        let a = BinaryHV::random(42);
        let b = BinaryHV::random(43);

        // SIMD version
        let simd_result = bind_simd(&a.0, &b.0);

        // Scalar version (original)
        let scalar_result = a.bind(&b);

        assert_eq!(
            simd_result, scalar_result.0,
            "SIMD bind must match scalar bind"
        );
    }

    #[test]
    fn test_matching_bits_simd_matches_scalar() {
        let a = BinaryHV::random(42);
        let b = BinaryHV::random(43);

        // SIMD version
        let simd_matching = matching_bits_simd(&a.0, &b.0);

        // Scalar version (via similarity * DIM)
        let scalar_similarity = a.similarity(&b);
        let scalar_matching = (scalar_similarity * BinaryHV::DIM as f32) as u32;

        // Allow small rounding difference
        let diff = (simd_matching as i32 - scalar_matching as i32).abs();
        assert!(
            diff <= 1,
            "SIMD matching bits must match scalar: {} vs {}",
            simd_matching,
            scalar_matching
        );
    }

    #[test]
    fn test_invert_simd_matches_scalar() {
        let a = BinaryHV::random(42);

        // SIMD version
        let simd_result = invert_simd(&a.0);

        // Scalar version
        let scalar_result = a.invert();

        assert_eq!(
            simd_result, scalar_result.0,
            "SIMD invert must match scalar invert"
        );
    }

    #[test]
    fn test_hamming_distance_simd_matches_scalar() {
        let a = BinaryHV::random(42);
        let b = BinaryHV::random(43);

        // SIMD version
        let simd_dist = hamming_distance_simd(&a.0, &b.0);

        // Scalar version
        let scalar_dist = a.hamming_distance(&b);

        assert_eq!(
            simd_dist, scalar_dist,
            "SIMD hamming distance must match scalar"
        );
    }

    #[test]
    fn test_simd_self_similarity() {
        let a = BinaryHV::random(42);

        let matching = matching_bits_simd(&a.0, &a.0);
        assert_eq!(matching, 16_384, "Self-matching should be all bits");

        let distance = hamming_distance_simd(&a.0, &a.0);
        assert_eq!(distance, 0, "Self-distance should be zero");
    }

    #[test]
    fn test_simd_inverse_properties() {
        let a = BinaryHV::random(42);
        let inv = invert_simd(&a.0);

        // XOR with inverse should be all 1s
        let xor_result = bind_simd(&a.0, &inv);
        for byte in xor_result.iter() {
            assert_eq!(*byte, 0xFF, "XOR with inverse should be all 1s");
        }

        // Hamming distance to inverse should be maximum
        let dist = hamming_distance_simd(&a.0, &inv);
        assert_eq!(dist, 16_384, "Distance to inverse should be maximum");
    }

    #[test]
    #[ignore = "benchmark test - run with cargo test --release -- --ignored"]
    fn bench_simd_vs_scalar() {
        use std::hint::black_box;
        use std::time::Instant;

        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        let iterations = 1_000_000;

        // Benchmark SIMD bind
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(bind_simd(black_box(&a.0), black_box(&b.0)));
        }
        let simd_bind_ns = start.elapsed().as_nanos() / iterations;

        // Benchmark scalar bind (using explicit scalar method)
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(a.bind_scalar(black_box(&b)));
        }
        let scalar_bind_ns = start.elapsed().as_nanos() / iterations;

        // Benchmark SIMD similarity
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(matching_bits_simd(black_box(&a.0), black_box(&b.0)));
        }
        let simd_sim_ns = start.elapsed().as_nanos() / iterations;

        // Benchmark scalar similarity (using explicit scalar method)
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(a.similarity_scalar(black_box(&b)));
        }
        let scalar_sim_ns = start.elapsed().as_nanos() / iterations;

        println!("\n📊 SIMD vs Scalar Performance:");
        println!(
            "  Bind:       SIMD {}ns vs Scalar {}ns ({:.1}x speedup)",
            simd_bind_ns,
            scalar_bind_ns,
            scalar_bind_ns as f64 / simd_bind_ns.max(1) as f64
        );
        println!(
            "  Similarity: SIMD {}ns vs Scalar {}ns ({:.1}x speedup)",
            simd_sim_ns,
            scalar_sim_ns,
            scalar_sim_ns as f64 / simd_sim_ns.max(1) as f64
        );

        // Assert meaningful speedup in release mode
        #[cfg(not(debug_assertions))]
        {
            assert!(
                simd_bind_ns < scalar_bind_ns,
                "SIMD bind should be faster than scalar"
            );
            assert!(
                simd_sim_ns < scalar_sim_ns,
                "SIMD similarity should be faster than scalar"
            );
        }
    }

    #[test]
    fn test_bundle_simd_matches_scalar() {
        let vectors: Vec<BinaryHV> = (0..10).map(|i| BinaryHV::random(i + 100)).collect();
        let refs: Vec<&[u8; 2048]> = vectors.iter().map(|v| &v.0).collect();

        // SIMD version
        let simd_result = bundle_simd(&refs);

        // Scalar version (via BinaryHV::bundle)
        let scalar_result = BinaryHV::bundle(&vectors);

        // Results should match exactly
        assert_eq!(
            simd_result, scalar_result.0,
            "SIMD bundle must match scalar bundle"
        );
    }

    #[test]
    fn test_bundle_simd_single_vector() {
        let v = BinaryHV::random(42);
        let refs: Vec<&[u8; 2048]> = vec![&v.0];

        let result = bundle_simd(&refs);
        assert_eq!(
            result, v.0,
            "Bundle of single vector should return that vector"
        );
    }

    #[test]
    fn test_bundle_simd_empty() {
        let refs: Vec<&[u8; 2048]> = vec![];
        let result = bundle_simd(&refs);
        assert_eq!(result, [0u8; 2048], "Bundle of empty should return zeros");
    }

    #[test]
    fn test_bundle_simd_majority_vote() {
        // Create 3 all-ones and 2 all-zeros vectors
        // Majority should be ones
        let ones = BinaryHV::ones();
        let zeros = BinaryHV::zero();

        let vectors = vec![&ones.0, &ones.0, &ones.0, &zeros.0, &zeros.0];
        let result = bundle_simd(&vectors);

        // Result should be all ones (3 > 2)
        for byte in result.iter() {
            assert_eq!(*byte, 0xFF, "Majority vote should produce ones");
        }
    }

    #[test]
    fn test_bundle_simd_similarity_preservation() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        let c = BinaryHV::random(3);

        let refs: Vec<&[u8; 2048]> = vec![&a.0, &b.0, &c.0];
        let bundled = BinaryHV(bundle_simd(&refs));

        // Bundled vector should be similar to all inputs
        assert!(
            bundled.similarity(&a) > 0.5,
            "Bundle should be similar to input A"
        );
        assert!(
            bundled.similarity(&b) > 0.5,
            "Bundle should be similar to input B"
        );
        assert!(
            bundled.similarity(&c) > 0.5,
            "Bundle should be similar to input C"
        );
    }

    #[test]
    #[ignore = "benchmark test - run with cargo test --release -- --ignored"]
    fn bench_bundle_simd_vs_scalar() {
        use std::hint::black_box;
        use std::time::Instant;

        let vectors: Vec<BinaryHV> = (0..100).map(|i| BinaryHV::random(i)).collect();
        let refs: Vec<&[u8; 2048]> = vectors.iter().map(|v| &v.0).collect();
        let iterations = 10_000;

        // Benchmark SIMD bundle
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(bundle_simd(black_box(&refs)));
        }
        let simd_ns = start.elapsed().as_nanos() / iterations;

        // Benchmark scalar bundle (via BinaryHV::bundle)
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(BinaryHV::bundle(black_box(&vectors)));
        }
        let scalar_ns = start.elapsed().as_nanos() / iterations;

        println!("\n📊 Bundle SIMD vs Scalar Performance (100 vectors):");
        println!("  SIMD:   {}ns", simd_ns);
        println!("  Scalar: {}ns", scalar_ns);
        println!(
            "  Speedup: {:.1}x",
            scalar_ns as f64 / simd_ns.max(1) as f64
        );
    }

    #[test]
    fn test_intersection_simd_correctness() {
        let a = BinaryHV::random(42);
        let b = BinaryHV::random(43);
        let simd_result = intersection_simd(&a.0, &b.0);
        let mut scalar = [0u8; 2048];
        for i in 0..2048 {
            scalar[i] = a.0[i] & b.0[i];
        }
        assert_eq!(
            simd_result, scalar,
            "SIMD intersection must match scalar AND"
        );
    }

    #[test]
    fn test_union_simd_correctness() {
        let a = BinaryHV::random(42);
        let b = BinaryHV::random(43);
        let simd_result = union_simd(&a.0, &b.0);
        let mut scalar = [0u8; 2048];
        for i in 0..2048 {
            scalar[i] = a.0[i] | b.0[i];
        }
        assert_eq!(simd_result, scalar, "SIMD union must match scalar OR");
    }

    #[test]
    fn test_simd_capabilities_report() {
        #[cfg(target_arch = "x86_64")]
        {
            println!("\n📊 CPU SIMD Capabilities:");
            println!("  AVX-512F:        {}", has_avx512f());
            println!("  AVX-512BW:       {}", has_avx512bw());
            println!("  AVX-512VPOPCNTDQ: {}", has_avx512_vpopcntdq());
            println!("  AVX2:            {}", has_avx2());
            println!("  SSE4.1:          {}", has_sse41());
            println!("  POPCNT:          {}", has_popcnt());
        }
    }
}
