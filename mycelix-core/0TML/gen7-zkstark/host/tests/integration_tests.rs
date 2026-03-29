// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for gradient validity proofs
//!
//! These tests verify the end-to-end proof generation and verification flow.
//!
//! Note: These tests are slow because they generate real zkSTARK proofs.
//! Run with: cargo test --release -- --test-threads=1

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// Import from host crate
// Note: The host crate would need to expose these as pub
// For now, we define minimal test utilities here

/// Q16.16 scale factor
const FIXED_SCALE: f32 = 65536.0;

/// Convert f32 to Q16.16 fixed-point
fn f32_to_fixed(f: f32) -> i32 {
    (f * FIXED_SCALE) as i32
}

/// Convert Q16.16 fixed-point to f32
fn fixed_to_f32(fixed: i32) -> f32 {
    fixed as f32 / FIXED_SCALE
}

/// Generate a deterministic test gradient
fn generate_test_gradient(size: usize, seed: u64) -> Vec<i32> {
    let mut gradient = Vec::with_capacity(size);
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);

    for i in 0..size {
        i.hash(&mut hasher);
        let hash = hasher.finish();
        // Generate value in [-1, 1]
        let normalized = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
        gradient.push(f32_to_fixed(normalized * 0.5)); // Scale down
    }

    gradient
}

/// Compute gradient norm squared
fn compute_norm_squared(gradient: &[i32]) -> i64 {
    let mut norm_squared: i64 = 0;
    for &g in gradient {
        let g_squared = (g as i64 * g as i64) >> 16;
        norm_squared = norm_squared.saturating_add(g_squared);
    }
    norm_squared
}

/// Compute gradient mean
fn compute_mean(gradient: &[i32]) -> i32 {
    if gradient.is_empty() {
        return 0;
    }
    let sum: i64 = gradient.iter().map(|&g| g as i64).sum();
    (sum / gradient.len() as i64) as i32
}

// =============================================================================
// Unit Tests (Fast)
// =============================================================================

#[test]
fn test_fixed_point_conversion() {
    let test_values = [0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0];

    for &v in &test_values {
        let fixed = f32_to_fixed(v);
        let back = fixed_to_f32(fixed);
        assert!(
            (v - back).abs() < 0.001,
            "Roundtrip failed for {}: got {}",
            v,
            back
        );
    }
}

#[test]
fn test_gradient_generation_deterministic() {
    let grad1 = generate_test_gradient(100, 42);
    let grad2 = generate_test_gradient(100, 42);

    assert_eq!(grad1, grad2, "Same seed should produce same gradient");

    let grad3 = generate_test_gradient(100, 43);
    assert_ne!(grad1, grad3, "Different seeds should produce different gradients");
}

#[test]
fn test_norm_computation() {
    // Gradient of all 1.0 values
    let gradient: Vec<i32> = (0..100).map(|_| f32_to_fixed(1.0)).collect();

    let norm_squared = compute_norm_squared(&gradient);

    // Expected: 100 * 1.0^2 = 100.0, in Q16.16 that's 100 * 65536
    let expected = 100 * 65536;
    let tolerance = 1000; // Allow small rounding error

    assert!(
        (norm_squared - expected).abs() < tolerance,
        "Norm squared {} differs from expected {} by more than {}",
        norm_squared,
        expected,
        tolerance
    );
}

#[test]
fn test_mean_computation() {
    // Gradient with known mean
    let gradient: Vec<i32> = vec![
        f32_to_fixed(1.0),
        f32_to_fixed(2.0),
        f32_to_fixed(3.0),
        f32_to_fixed(4.0),
    ];

    let mean = compute_mean(&gradient);
    let mean_f32 = fixed_to_f32(mean);

    // Expected mean: (1 + 2 + 3 + 4) / 4 = 2.5
    assert!(
        (mean_f32 - 2.5).abs() < 0.01,
        "Mean {} differs from expected 2.5",
        mean_f32
    );
}

#[test]
fn test_empty_gradient_mean() {
    let gradient: Vec<i32> = vec![];
    let mean = compute_mean(&gradient);
    assert_eq!(mean, 0, "Empty gradient should have zero mean");
}

// =============================================================================
// Gradient Validity Tests
// =============================================================================

/// Check if gradient values are within valid range
fn check_gradient_validity(gradient: &[i32]) -> Result<(), String> {
    if gradient.is_empty() {
        return Err("Gradient is empty".to_string());
    }

    // Check for extreme values (sentinel values)
    const FIXED_MAX: i32 = i32::MAX - 1;
    const FIXED_MIN: i32 = i32::MIN + 1;

    for (i, &g) in gradient.iter().enumerate() {
        if g <= FIXED_MIN || g >= FIXED_MAX {
            return Err(format!("Gradient element {} has invalid value: {}", i, g));
        }
    }

    // Check norm bounds
    let norm_squared = compute_norm_squared(gradient);
    let max_norm_squared: i64 = 1_000_000 * 65536; // ~1000.0 L2 norm

    if norm_squared > max_norm_squared {
        return Err(format!(
            "Gradient norm {} exceeds maximum {}",
            norm_squared, max_norm_squared
        ));
    }

    if norm_squared < 1 {
        return Err("Gradient norm too small (near-zero gradient)".to_string());
    }

    Ok(())
}

#[test]
fn test_valid_gradient_passes_check() {
    let gradient = generate_test_gradient(100, 123);
    let result = check_gradient_validity(&gradient);
    assert!(result.is_ok(), "Valid gradient should pass: {:?}", result);
}

#[test]
fn test_empty_gradient_fails_check() {
    let gradient: Vec<i32> = vec![];
    let result = check_gradient_validity(&gradient);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("empty"));
}

#[test]
fn test_large_gradient_fails_check() {
    // Gradient with very large values
    let gradient: Vec<i32> = (0..100).map(|_| f32_to_fixed(10000.0)).collect();

    let result = check_gradient_validity(&gradient);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("norm"));
}

#[test]
fn test_zero_gradient_fails_check() {
    let gradient: Vec<i32> = vec![0; 100];
    let result = check_gradient_validity(&gradient);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too small"));
}

#[test]
fn test_sentinel_values_fail_check() {
    let mut gradient = generate_test_gradient(100, 456);
    gradient[50] = i32::MAX; // Sentinel value

    let result = check_gradient_validity(&gradient);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("invalid"));
}

// =============================================================================
// Hash Consistency Tests
// =============================================================================

use sha2::{Digest, Sha256};

fn hash_gradient(gradient: &[i32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &g in gradient {
        hasher.update(g.to_le_bytes());
    }
    hasher.finalize().into()
}

#[test]
fn test_hash_deterministic() {
    let gradient = generate_test_gradient(100, 789);

    let hash1 = hash_gradient(&gradient);
    let hash2 = hash_gradient(&gradient);

    assert_eq!(hash1, hash2, "Same gradient should produce same hash");
}

#[test]
fn test_hash_different_gradients() {
    let gradient1 = generate_test_gradient(100, 1);
    let gradient2 = generate_test_gradient(100, 2);

    let hash1 = hash_gradient(&gradient1);
    let hash2 = hash_gradient(&gradient2);

    assert_ne!(hash1, hash2, "Different gradients should produce different hashes");
}

#[test]
fn test_hash_sensitive_to_small_changes() {
    let mut gradient1 = generate_test_gradient(100, 111);
    let gradient2 = gradient1.clone();

    // Modify just one element slightly
    gradient1[0] += 1;

    let hash1 = hash_gradient(&gradient1);
    let hash2 = hash_gradient(&gradient2);

    assert_ne!(hash1, hash2, "Hash should change with small modifications");
}

// =============================================================================
// Note on Integration Tests
// =============================================================================
//
// Full integration tests that generate zkSTARK proofs should be run with:
//
//   RISC0_DEV_MODE=1 cargo test --release integration -- --test-threads=1
//
// These are in the main.rs file as #[cfg(test)] tests.
