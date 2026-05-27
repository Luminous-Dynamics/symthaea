// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified SIMD Feature Detection
//!
//! Single source of truth for CPU feature detection across all SIMD modules.
//! All detection results are cached via `OnceLock` for 2-3x improvement
//! over repeated `is_x86_feature_detected!` calls on hot paths.
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea_core::hdc::simd_detect::*;
//!
//! if has_avx2() && has_fma() {
//!     unsafe { fast_path_avx2_fma(...) }
//! } else if has_neon() {
//!     unsafe { fast_path_neon(...) }
//! } else {
//!     scalar_fallback(...)
//! }
//! ```

use std::sync::OnceLock;

// =============================================================================
// x86_64 FEATURE DETECTION
// =============================================================================

#[cfg(target_arch = "x86_64")]
static HAS_AVX512F: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static HAS_AVX512BW: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static HAS_AVX512VPOPCNTDQ: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static HAS_AVX2: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static HAS_AVX: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static HAS_SSE41: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static HAS_POPCNT: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static HAS_FMA: OnceLock<bool> = OnceLock::new();

/// AVX-512 Foundation (512-bit integer/float ops)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_avx512f() -> bool {
    *HAS_AVX512F.get_or_init(|| is_x86_feature_detected!("avx512f"))
}

/// AVX-512 Byte/Word (512-bit byte/word integer ops)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_avx512bw() -> bool {
    *HAS_AVX512BW.get_or_init(|| is_x86_feature_detected!("avx512bw"))
}

/// AVX-512 VPOPCNTDQ (hardware 512-bit popcount)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_avx512_vpopcntdq() -> bool {
    *HAS_AVX512VPOPCNTDQ.get_or_init(|| is_x86_feature_detected!("avx512vpopcntdq"))
}

/// AVX2 (256-bit integer ops — primary SIMD path)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_avx2() -> bool {
    *HAS_AVX2.get_or_init(|| is_x86_feature_detected!("avx2"))
}

/// AVX (256-bit float ops)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_avx() -> bool {
    *HAS_AVX.get_or_init(|| is_x86_feature_detected!("avx"))
}

/// SSE4.1 (128-bit ops with rounding/blending)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_sse41() -> bool {
    *HAS_SSE41.get_or_init(|| is_x86_feature_detected!("sse4.1"))
}

/// Hardware POPCNT instruction
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_popcnt() -> bool {
    *HAS_POPCNT.get_or_init(|| is_x86_feature_detected!("popcnt"))
}

/// FMA (Fused Multiply-Add — ~30% speedup for f32 dot products)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_fma() -> bool {
    *HAS_FMA.get_or_init(|| is_x86_feature_detected!("fma"))
}

// =============================================================================
// aarch64 FEATURE DETECTION
// =============================================================================

// NEON is baseline on AArch64 — always available, no runtime check needed.

/// NEON SIMD (128-bit, baseline on AArch64)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn has_neon() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn has_neon() -> bool {
    false
}

// =============================================================================
// CROSS-PLATFORM STUBS (for non-x86_64 builds)
// =============================================================================

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn has_avx512f() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn has_avx512bw() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn has_avx512_vpopcntdq() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn has_avx2() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn has_avx() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn has_sse41() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn has_popcnt() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn has_fma() -> bool {
    false
}

// =============================================================================
// SIMD LEVEL — coarse dispatch helper
// =============================================================================

/// Coarse SIMD capability level for dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    Neon,
    Sse41,
    Avx2,
    Avx2Fma,
    Avx512,
}

/// Best available SIMD level for integer operations (XOR, popcount, etc.)
pub fn best_integer_level() -> SimdLevel {
    if has_avx512f() {
        SimdLevel::Avx512
    } else if has_avx2() {
        SimdLevel::Avx2
    } else if has_sse41() {
        SimdLevel::Sse41
    } else if has_neon() {
        SimdLevel::Neon
    } else {
        SimdLevel::Scalar
    }
}

/// Best available SIMD level for float operations (dot product, FMA, etc.)
pub fn best_float_level() -> SimdLevel {
    if has_avx2() && has_fma() {
        SimdLevel::Avx2Fma
    } else if has_avx2() {
        SimdLevel::Avx2
    } else if has_sse41() {
        SimdLevel::Sse41
    } else if has_neon() {
        SimdLevel::Neon
    } else {
        SimdLevel::Scalar
    }
}

// =============================================================================
// SIMD UTILITY: Horizontal sum (shared by continuous HV and cosine helpers)
// =============================================================================

/// Horizontal sum of 8 f32 lanes in an AVX2 __m256 register.
///
/// Reduces [a0, a1, a2, a3, a4, a5, a6, a7] → a0+a1+...+a7.
///
/// # Safety
///
/// Caller must ensure AVX2 is available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

// =============================================================================
// CAPABILITIES REPORT
// =============================================================================

/// Comprehensive SIMD capability report for diagnostics.
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512_vpopcntdq: bool,
    pub avx2: bool,
    pub avx: bool,
    pub fma: bool,
    pub sse41: bool,
    pub popcnt: bool,
    pub neon: bool,
    pub best_integer: SimdLevel,
    pub best_float: SimdLevel,
}

impl SimdCapabilities {
    pub fn detect() -> Self {
        Self {
            avx512f: has_avx512f(),
            avx512bw: has_avx512bw(),
            avx512_vpopcntdq: has_avx512_vpopcntdq(),
            avx2: has_avx2(),
            avx: has_avx(),
            fma: has_fma(),
            sse41: has_sse41(),
            popcnt: has_popcnt(),
            neon: has_neon(),
            best_integer: best_integer_level(),
            best_float: best_float_level(),
        }
    }
}

impl std::fmt::Display for SimdCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SIMD: AVX-512F={} BW={} VPOPCNTDQ={} | AVX2={} FMA={} | SSE4.1={} POPCNT={} | NEON={} | best_int={:?} best_float={:?}",
            self.avx512f,
            self.avx512bw,
            self.avx512_vpopcntdq,
            self.avx2,
            self.fma,
            self.sse41,
            self.popcnt,
            self.neon,
            self.best_integer,
            self.best_float,
        )
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_is_consistent() {
        // Calling twice should return the same value (OnceLock caching)
        let a = has_avx2();
        let b = has_avx2();
        assert_eq!(a, b);
    }

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2Fma);
        assert!(SimdLevel::Avx2Fma > SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 > SimdLevel::Sse41);
        assert!(SimdLevel::Sse41 > SimdLevel::Neon);
        assert!(SimdLevel::Neon > SimdLevel::Scalar);
    }

    #[test]
    fn test_best_level_returns_valid() {
        let int_level = best_integer_level();
        let float_level = best_float_level();
        assert!(int_level >= SimdLevel::Scalar);
        assert!(float_level >= SimdLevel::Scalar);
    }

    #[test]
    fn test_capabilities_report() {
        let caps = SimdCapabilities::detect();
        let report = format!("{}", caps);
        assert!(report.contains("SIMD:"));
        assert!(report.contains("AVX2="));

        #[cfg(target_arch = "x86_64")]
        {
            // SSE4.1 should be available on any x86_64 CPU from 2008+
            assert!(caps.sse41 || !caps.avx2, "AVX2 implies SSE4.1");
        }
    }

    #[test]
    fn test_neon_detection() {
        let neon = has_neon();
        #[cfg(target_arch = "aarch64")]
        assert!(neon, "NEON must be true on aarch64");
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!neon, "NEON must be false on non-aarch64");
    }
}
