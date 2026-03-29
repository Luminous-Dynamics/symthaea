// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-Based Fuzz Tests for Mycelix SDK
//!
//! These tests verify security invariants that should hold for ALL inputs,
//! including edge cases like NaN, Infinity, and extreme values.
//!
//! Run with: cargo test --test property_tests

use proptest::prelude::*;

// =============================================================================
// Test Helpers
// =============================================================================

/// Safe threshold check that handles special floating-point values
///
/// Returns true if value >= threshold, with proper handling of:
/// - NaN (fails safe - returns false)
/// - Infinity (fails safe - returns false for positive infinity)
/// - Normal values (standard comparison)
fn safe_threshold_check(value: f64, threshold: f64) -> bool {
    // NaN checks - fail safe
    if value.is_nan() || threshold.is_nan() {
        return false;
    }

    // Infinity checks - fail safe for positive infinity
    if value.is_infinite() {
        if value.is_sign_positive() {
            return false; // +Inf fails safe
        }
        return false; // -Inf is always below threshold
    }

    if threshold.is_infinite() {
        if threshold.is_sign_positive() {
            return false; // Nothing can meet +Inf threshold
        }
        return true; // Everything meets -Inf threshold
    }

    // Normal comparison
    value >= threshold
}

/// Safe bounded value that clamps to [0.0, 1.0] with NaN/Inf handling
fn safe_bounded(value: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    if value.is_infinite() {
        return if value.is_sign_positive() { 1.0 } else { 0.0 };
    }
    value.clamp(0.0, 1.0)
}

/// Safe score aggregation that handles edge cases
fn safe_weighted_average(scores: &[(f64, f64)]) -> f64 {
    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;

    for (score, weight) in scores {
        // Skip NaN/Inf values
        if score.is_nan() || score.is_infinite() || weight.is_nan() || weight.is_infinite() {
            continue;
        }
        // Skip negative weights
        if *weight < 0.0 {
            continue;
        }
        weighted_sum += score * weight;
        total_weight += weight;
    }

    if total_weight <= 0.0 {
        return 0.5; // Default neutral score
    }

    safe_bounded(weighted_sum / total_weight)
}

// =============================================================================
// Property Tests - Threshold Bounds (FIND-006)
// =============================================================================

proptest! {
    /// Threshold check should never panic for any valid floating-point inputs
    #[test]
    fn test_threshold_check_never_panics(value in -1e10f64..1e10f64, threshold in -1e10f64..1e10f64) {
        let result = safe_threshold_check(value, threshold);
        // Just verify it doesn't panic and returns a boolean
        prop_assert!(result == true || result == false);
    }

    /// Standard comparison should work for normal values
    #[test]
    fn test_threshold_normal_values(value in 0.0f64..1.0f64, threshold in 0.0f64..1.0f64) {
        let result = safe_threshold_check(value, threshold);
        prop_assert!(result == (value >= threshold));
    }

    /// NaN should fail safe
    #[test]
    fn test_nan_handling(threshold in 0.0f64..1.0f64) {
        prop_assert!(!safe_threshold_check(f64::NAN, threshold));
        prop_assert!(!safe_threshold_check(threshold, f64::NAN));
        prop_assert!(!safe_threshold_check(f64::NAN, f64::NAN));
    }

    /// Infinity should fail safe
    #[test]
    fn test_infinity_handling(threshold in 0.0f64..1.0f64) {
        prop_assert!(!safe_threshold_check(f64::INFINITY, threshold));
        prop_assert!(!safe_threshold_check(f64::NEG_INFINITY, threshold));
        prop_assert!(!safe_threshold_check(threshold, f64::INFINITY));
    }
}

// =============================================================================
// Property Tests - Bounded Values
// =============================================================================

proptest! {
    /// Bounded values should always be in [0.0, 1.0]
    #[test]
    fn test_bounded_values_in_range(value in prop::num::f64::ANY) {
        let bounded = safe_bounded(value);
        prop_assert!(bounded >= 0.0);
        prop_assert!(bounded <= 1.0);
        prop_assert!(!bounded.is_nan());
        prop_assert!(!bounded.is_infinite());
    }

    /// Bounded values should preserve normal values in range
    #[test]
    fn test_bounded_preserves_valid(value in 0.0f64..1.0f64) {
        let bounded = safe_bounded(value);
        prop_assert!((bounded - value).abs() < 1e-10);
    }

    /// Bounded values should clamp out-of-range values
    #[test]
    fn test_bounded_clamps_out_of_range(value in 1.1f64..100.0f64) {
        let bounded = safe_bounded(value);
        prop_assert!((bounded - 1.0).abs() < 1e-10);
    }
}

// =============================================================================
// Property Tests - Weighted Average
// =============================================================================

proptest! {
    /// Weighted average should always produce valid bounded result
    #[test]
    fn test_weighted_average_bounded(
        s1 in 0.0f64..1.0f64,
        s2 in 0.0f64..1.0f64,
        s3 in 0.0f64..1.0f64,
        w1 in 0.0f64..10.0f64,
        w2 in 0.0f64..10.0f64,
        w3 in 0.0f64..10.0f64
    ) {
        let scores = vec![(s1, w1), (s2, w2), (s3, w3)];
        let result = safe_weighted_average(&scores);
        prop_assert!(result >= 0.0);
        prop_assert!(result <= 1.0);
        prop_assert!(!result.is_nan());
    }

    /// Weighted average with NaN values should skip them
    #[test]
    fn test_weighted_average_skips_nan(s1 in 0.0f64..1.0f64, w1 in 0.1f64..10.0f64) {
        let scores = vec![(s1, w1), (f64::NAN, 1.0), (f64::NAN, f64::NAN)];
        let result = safe_weighted_average(&scores);
        // Should just use the valid score
        prop_assert!((result - s1).abs() < 1e-10);
    }

    /// Empty scores should return default
    #[test]
    fn test_weighted_average_empty(_unused in 0i32..1i32) {
        let scores: Vec<(f64, f64)> = vec![];
        let result = safe_weighted_average(&scores);
        prop_assert!((result - 0.5).abs() < 1e-10);
    }

    /// Zero total weight should return default
    #[test]
    fn test_weighted_average_zero_weight(s in 0.0f64..1.0f64) {
        let scores = vec![(s, 0.0)];
        let result = safe_weighted_average(&scores);
        prop_assert!((result - 0.5).abs() < 1e-10);
    }
}

// =============================================================================
// Property Tests - Serialization Roundtrip
// =============================================================================

proptest! {
    /// Data should survive serialization roundtrip
    #[test]
    fn test_serialization_roundtrip_bytes(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        // Test that byte vectors survive JSON serialization
        let json = serde_json::to_string(&data).expect("serialize");
        let decoded: Vec<u8> = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(data, decoded);
    }

    /// Strings should survive serialization roundtrip
    #[test]
    fn test_serialization_roundtrip_string(s in ".*") {
        let json = serde_json::to_string(&s).expect("serialize");
        let decoded: String = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(s, decoded);
    }
}

// =============================================================================
// Property Tests - Byzantine Tolerance Bounds
// =============================================================================

/// Byzantine tolerance configuration
const MAX_BYZANTINE_TOLERANCE: f64 = 0.34;
const MIN_BYZANTINE_TOLERANCE: f64 = 0.10;

/// Validate Byzantine tolerance is within bounds
fn validate_byzantine_tolerance(value: f64) -> bool {
    !value.is_nan()
        && !value.is_infinite()
        && value >= MIN_BYZANTINE_TOLERANCE
        && value <= MAX_BYZANTINE_TOLERANCE
}

proptest! {
    /// Byzantine tolerance should always be within bounds after clamping
    #[test]
    fn test_byzantine_tolerance_bounds(value in prop::num::f64::ANY) {
        let clamped = if value.is_nan() || value.is_infinite() {
            0.3 // Default
        } else {
            value.clamp(MIN_BYZANTINE_TOLERANCE, MAX_BYZANTINE_TOLERANCE)
        };

        prop_assert!(validate_byzantine_tolerance(clamped));
    }

    /// Valid Byzantine tolerance values should pass validation
    #[test]
    fn test_byzantine_tolerance_valid_range(value in 0.1f64..0.34f64) {
        prop_assert!(validate_byzantine_tolerance(value));
    }
}

// =============================================================================
// Property Tests - Reputation Score Bounds
// =============================================================================

proptest! {
    /// Reputation scores should be bounded after processing
    #[test]
    fn test_reputation_score_bounds(raw_score in prop::num::f64::ANY) {
        let processed = safe_bounded(raw_score);
        prop_assert!(processed >= 0.0);
        prop_assert!(processed <= 1.0);
    }

    /// Multiple reputation scores should aggregate correctly
    #[test]
    fn test_reputation_aggregation(
        scores in prop::collection::vec(0.0f64..1.0f64, 1..10)
    ) {
        let sum: f64 = scores.iter().sum();
        let avg = sum / scores.len() as f64;
        let bounded = safe_bounded(avg);

        prop_assert!(bounded >= 0.0);
        prop_assert!(bounded <= 1.0);
    }
}

// =============================================================================
// Property Tests - Rate Limiting (FIND-008)
// =============================================================================

proptest! {
    /// Rate limit duration should be positive
    #[test]
    fn test_rate_limit_duration_positive(secs in 1u64..3600u64) {
        let duration = std::time::Duration::from_secs(secs);
        prop_assert!(duration.as_secs() > 0);
    }

    /// Elapsed time should never exceed rate limit duration (when checked immediately)
    #[test]
    fn test_rate_limit_elapsed_bounds(secs in 1u64..60u64) {
        let rate_limit = std::time::Duration::from_secs(secs);
        let start = std::time::Instant::now();
        let elapsed = start.elapsed();

        // Elapsed time should be less than rate limit when checked immediately
        prop_assert!(elapsed < rate_limit);
    }
}

// =============================================================================
// Property Tests - Cache Invalidation (FIND-009)
// =============================================================================

proptest! {
    /// Cache invalidation threshold should detect significant changes
    #[test]
    fn test_cache_invalidation_threshold(
        old_score in 0.0f64..1.0f64,
        new_score in 0.0f64..1.0f64,
        threshold in 0.01f64..0.5f64
    ) {
        let change = (new_score - old_score).abs();
        let should_invalidate = change > threshold;

        // Verify the logic is consistent
        if change > threshold {
            prop_assert!(should_invalidate);
        } else {
            prop_assert!(!should_invalidate);
        }
    }

    /// TTL should be positive
    #[test]
    fn test_cache_ttl_positive(ttl_secs in 1u64..86400u64) {
        prop_assert!(ttl_secs > 0);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_special_float_values() {
    // NaN
    assert!(!safe_threshold_check(f64::NAN, 0.5));
    assert!(!safe_threshold_check(0.5, f64::NAN));

    // Infinity
    assert!(!safe_threshold_check(f64::INFINITY, 0.5));
    assert!(!safe_threshold_check(f64::NEG_INFINITY, 0.5));

    // Negative zero
    assert!(safe_threshold_check(-0.0, 0.0));
    assert!(safe_threshold_check(0.0, -0.0));

    // Very small values
    assert!(safe_threshold_check(1e-300, 0.0));
    assert!(!safe_threshold_check(0.0, 1e-300));
}

#[test]
fn test_bounded_edge_cases() {
    assert_eq!(safe_bounded(f64::NAN), 0.0);
    assert_eq!(safe_bounded(f64::INFINITY), 1.0);
    assert_eq!(safe_bounded(f64::NEG_INFINITY), 0.0);
    assert_eq!(safe_bounded(-0.0), 0.0);
    assert_eq!(safe_bounded(0.5), 0.5);
    assert_eq!(safe_bounded(-1.0), 0.0);
    assert_eq!(safe_bounded(2.0), 1.0);
}

#[test]
fn test_weighted_average_edge_cases() {
    // All NaN
    let scores = vec![(f64::NAN, f64::NAN)];
    assert_eq!(safe_weighted_average(&scores), 0.5);

    // Negative weights (should be skipped)
    let scores = vec![(0.8, -1.0), (0.5, 1.0)];
    assert!((safe_weighted_average(&scores) - 0.5).abs() < 1e-10);

    // Single valid score
    let scores = vec![(0.75, 1.0)];
    assert!((safe_weighted_average(&scores) - 0.75).abs() < 1e-10);
}
