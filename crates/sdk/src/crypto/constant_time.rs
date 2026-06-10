// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Constant-Time Cryptographic Operations
//!
//! This module provides constant-time comparison functions to prevent timing
//! side-channel attacks on cryptographic verification paths.
//!
//! # Security Background
//!
//! Standard comparison operators like `==` can leak information through timing:
//! - They may short-circuit on the first differing byte
//! - Memory access patterns can reveal comparison results
//!
//! Constant-time comparisons ensure the comparison takes the same amount of time
//! regardless of whether the values match or where they differ.
//!
//! # Usage
//!
//! ```rust
//! use mycelix_sdk::crypto::{secure_compare, secure_compare_hash};
//!
//! // Compare arbitrary byte slices
//! let a = b"secret_value_1";
//! let b = b"secret_value_2";
//! let equal = secure_compare(a, b);
//!
//! // Compare 32-byte hashes/commitments
//! let hash1: [u8; 32] = [0x42; 32];
//! let hash2: [u8; 32] = [0x42; 32];
//! let equal = secure_compare_hash(&hash1, &hash2);
//! ```
//!
//! # Security Guarantees
//!
//! - Comparison time is independent of input values
//! - No early termination on mismatch
//! - Uses the audited `subtle` crate for underlying operations
//!
//! # FIND-004 Mitigation
//!
//! This module addresses FIND-004 from the security audit:
//! "Non-constant-time comparisons in cryptographic verification paths"

use subtle::ConstantTimeEq;

/// Compare two byte slices in constant time.
///
/// Returns `true` if the slices are equal, `false` otherwise.
///
/// # Security
///
/// - Length comparison is NOT constant-time (lengths are typically public)
/// - Content comparison IS constant-time when lengths match
///
/// # Example
///
/// ```rust
/// use mycelix_sdk::crypto::secure_compare;
///
/// let computed_mac = [0x01, 0x02, 0x03, 0x04];
/// let expected_mac = [0x01, 0x02, 0x03, 0x04];
///
/// if secure_compare(&computed_mac, &expected_mac) {
///     // MAC verified
/// }
/// ```
#[inline]
pub fn secure_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

/// Compare two 32-byte hashes/commitments in constant time.
///
/// This is a specialized version for the common case of comparing
/// SHA3-256 hashes, commitments, or other 32-byte values.
///
/// # Example
///
/// ```rust
/// use mycelix_sdk::crypto::secure_compare_hash;
///
/// let computed: [u8; 32] = [0x42; 32];
/// let expected: [u8; 32] = [0x42; 32];
///
/// if secure_compare_hash(&computed, &expected) {
///     // Hash verified
/// }
/// ```
#[inline]
pub fn secure_compare_hash(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.ct_eq(b).into()
}

/// Compare two 64-byte values (e.g., Ed25519 signatures) in constant time.
///
/// # Example
///
/// ```rust
/// use mycelix_sdk::crypto::secure_compare_64;
///
/// let sig1: [u8; 64] = [0x00; 64];
/// let sig2: [u8; 64] = [0x00; 64];
///
/// if secure_compare_64(&sig1, &sig2) {
///     // Signatures match
/// }
/// ```
#[inline]
pub fn secure_compare_64(a: &[u8; 64], b: &[u8; 64]) -> bool {
    a.ct_eq(b).into()
}

/// Constant-time conditional select for byte slices.
///
/// Copies bytes from `a` if `choice` is true, from `b` otherwise,
/// into the returned vector. Both slices must have the same length.
///
/// # Security
///
/// Uses bitwise masking instead of branching to prevent timing attacks.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
#[allow(dead_code)]
pub fn secure_select_bytes(choice: bool, a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len(), "slices must have equal length");
    // mask is 0xFF if choice is true, 0x00 if false — no branch
    let mask = (-(choice as i8)) as u8;
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x & mask) | (y & !mask))
        .collect()
}

/// Check if a byte slice is all zeros in constant time.
///
/// Useful for checking if a hash/key has been properly initialized.
#[inline]
pub fn is_zero(data: &[u8]) -> bool {
    let zeros = vec![0u8; data.len()];
    data.ct_eq(&zeros).into()
}

/// Check if a 32-byte array is all zeros in constant time.
#[inline]
pub fn is_zero_hash(hash: &[u8; 32]) -> bool {
    hash.ct_eq(&[0u8; 32]).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_compare_equal() {
        let a = b"test_secret_value";
        let b = b"test_secret_value";
        assert!(secure_compare(a, b));
    }

    #[test]
    fn test_secure_compare_different() {
        let a = b"test_secret_value_1";
        let b = b"test_secret_value_2";
        assert!(!secure_compare(a, b));
    }

    #[test]
    fn test_secure_compare_different_length() {
        let a = b"short";
        let b = b"longer_value";
        assert!(!secure_compare(a, b));
    }

    #[test]
    fn test_secure_compare_empty() {
        let a: &[u8] = b"";
        let b: &[u8] = b"";
        assert!(secure_compare(a, b));
    }

    #[test]
    fn test_secure_compare_hash_equal() {
        let a = [0x42u8; 32];
        let b = [0x42u8; 32];
        assert!(secure_compare_hash(&a, &b));
    }

    #[test]
    fn test_secure_compare_hash_different() {
        let a = [0x42u8; 32];
        let mut b = [0x42u8; 32];
        b[31] = 0x43; // Single byte difference
        assert!(!secure_compare_hash(&a, &b));
    }

    #[test]
    fn test_secure_compare_64_equal() {
        let a = [0x00u8; 64];
        let b = [0x00u8; 64];
        assert!(secure_compare_64(&a, &b));
    }

    #[test]
    fn test_secure_compare_64_different() {
        let a = [0x00u8; 64];
        let mut b = [0x00u8; 64];
        b[0] = 0x01;
        assert!(!secure_compare_64(&a, &b));
    }

    #[test]
    fn test_is_zero_true() {
        let zeros = [0u8; 32];
        assert!(is_zero(&zeros));
    }

    #[test]
    fn test_is_zero_false() {
        let mut data = [0u8; 32];
        data[15] = 0x01;
        assert!(!is_zero(&data));
    }

    #[test]
    fn test_is_zero_hash_true() {
        let zeros = [0u8; 32];
        assert!(is_zero_hash(&zeros));
    }

    #[test]
    fn test_is_zero_hash_false() {
        let data = [0x01u8; 32];
        assert!(!is_zero_hash(&data));
    }

    // Timing attack resistance test (basic sanity check)
    // Note: Real timing tests require statistical analysis
    #[test]
    fn test_timing_consistency() {
        use std::time::Instant;

        let a = [0x42u8; 32];
        let b_same = [0x42u8; 32];
        let mut b_diff_first = [0x42u8; 32];
        b_diff_first[0] = 0x00;
        let mut b_diff_last = [0x42u8; 32];
        b_diff_last[31] = 0x00;

        // Warm up
        for _ in 0..1000 {
            let _ = secure_compare_hash(&a, &b_same);
            let _ = secure_compare_hash(&a, &b_diff_first);
            let _ = secure_compare_hash(&a, &b_diff_last);
        }

        // This is a basic sanity check - real timing analysis would need
        // statistical methods and many more iterations
        let iterations = 10000;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = secure_compare_hash(&a, &b_same);
        }
        let time_same = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = secure_compare_hash(&a, &b_diff_first);
        }
        let time_diff_first = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = secure_compare_hash(&a, &b_diff_last);
        }
        let time_diff_last = start.elapsed();

        // Allow 50% variance (very loose - just sanity check)
        let max_time = time_same.max(time_diff_first).max(time_diff_last);
        let min_time = time_same.min(time_diff_first).min(time_diff_last);

        // All times should be within 10x of each other (very loose bound for CI).
        // Real constant-time validation requires statistical analysis over many runs.
        // This test just catches gross violations (e.g., early-exit on first byte).
        assert!(
            max_time.as_nanos() < min_time.as_nanos() * 10,
            "Timing variance too high: same={:?}, diff_first={:?}, diff_last={:?}",
            time_same,
            time_diff_first,
            time_diff_last
        );
    }
}
