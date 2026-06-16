// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared mathematical utilities.
//!
//! This module provides numerically stable implementations of common
//! mathematical functions used across the Symthaea codebase.

/// Softmax with temperature scaling.
///
/// Converts a slice of real-valued scores into a probability distribution
/// using the softmax function with an adjustable temperature parameter.
///
/// - `temperature = 1.0` gives standard softmax.
/// - `temperature < 1.0` produces a sharper (more peaked) distribution.
/// - `temperature > 1.0` produces a flatter (more uniform) distribution.
///
/// The implementation is numerically stable: it subtracts the maximum value
/// before exponentiation to avoid overflow.
///
/// # Arguments
/// * `values` - Input logits / scores.
/// * `temperature` - Temperature parameter (clamped to >= 1e-10 to prevent division by zero).
///
/// # Returns
/// A `Vec<f32>` of probabilities that sum to 1.0, or an empty vec if the input is empty.
/// Falls back to a uniform distribution if all exponentials underflow.
///
/// # Examples
/// ```
/// use symthaea_core::math::softmax_with_temperature;
///
/// let probs = softmax_with_temperature(&[1.0, 2.0, 3.0], 1.0);
/// assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// assert!(probs[2] > probs[1] && probs[1] > probs[0]);
/// ```
pub fn softmax_with_temperature(values: &[f32], temperature: f32) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }

    let temp = temperature.max(1e-10); // Prevent division by zero

    // Numerical stability: subtract max before exp
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp_values: Vec<f32> = values
        .iter()
        .map(|&v| ((v - max_val) / temp).exp())
        .collect();

    let sum: f32 = exp_values.iter().sum();

    if sum < 1e-10 {
        // Uniform distribution fallback
        let n = values.len() as f32;
        return vec![1.0 / n; values.len()];
    }

    exp_values.iter().map(|&e| e / sum).collect()
}

/// Standard softmax (temperature = 1.0).
///
/// Convenience wrapper around [`softmax_with_temperature`] for the common
/// case where no temperature scaling is needed.
///
/// # Examples
/// ```
/// use symthaea_core::math::softmax;
///
/// let probs = softmax(&[1.0, 2.0, 3.0]);
/// assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// ```
pub fn softmax(values: &[f32]) -> Vec<f32> {
    softmax_with_temperature(values, 1.0)
}

/// Cosine similarity between two `f32` slices.
///
/// Returns a value in `[-1.0, 1.0]` measuring the cosine of the angle between
/// the two vectors.  Returns `0.0` for empty slices, zero-norm vectors, or
/// mismatched lengths.
///
/// # Examples
/// ```
/// use symthaea_core::math::cosine_similarity_f32;
///
/// let a = [1.0_f32, 0.0, 0.0];
/// let b = [0.0_f32, 1.0, 0.0];
/// assert!((cosine_similarity_f32(&a, &b) - 0.0).abs() < 1e-6);
///
/// assert!((cosine_similarity_f32(&a, &a) - 1.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

/// Cosine similarity between two `f64` slices.
///
/// Returns a value in `[-1.0, 1.0]` measuring the cosine of the angle between
/// the two vectors.  Returns `0.0` for empty slices, zero-norm vectors, or
/// mismatched lengths.
///
/// # Examples
/// ```
/// use symthaea_core::math::cosine_similarity_f64;
///
/// let a = [1.0_f64, 0.0, 0.0];
/// let b = [0.0_f64, 1.0, 0.0];
/// assert!((cosine_similarity_f64(&a, &b) - 0.0).abs() < 1e-12);
///
/// assert!((cosine_similarity_f64(&a, &a) - 1.0).abs() < 1e-12);
/// ```
pub fn cosine_similarity_f64(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

/// Exponential moving average (EMA) update.
///
/// Computes `alpha * current + (1.0 - alpha) * previous`, the standard
/// single-step EMA recurrence.
///
/// # Arguments
/// * `previous` - The previous EMA value.
/// * `current`  - The new observation.
/// * `alpha`    - Smoothing factor in `[0.0, 1.0]`. Higher values weight
///   the current observation more heavily.
///
/// # Examples
/// ```
/// use symthaea_core::math::ema_update;
///
/// // With alpha = 1.0, result equals current.
/// assert!((ema_update(10.0, 5.0, 1.0) - 5.0).abs() < 1e-6);
///
/// // With alpha = 0.0, result equals previous.
/// assert!((ema_update(10.0, 5.0, 0.0) - 10.0).abs() < 1e-6);
///
/// // Typical update
/// let result = ema_update(0.8, 0.6, 0.3);
/// assert!((result - 0.74).abs() < 1e-6); // 0.3*0.6 + 0.7*0.8 = 0.18 + 0.56 = 0.74
/// ```
#[inline]
pub fn ema_update(previous: f32, current: f32, alpha: f32) -> f32 {
    alpha * current + (1.0 - alpha) * previous
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_empty_input() {
        let result = softmax_with_temperature(&[], 1.0);
        assert!(result.is_empty(), "Empty input should produce empty output");
    }

    #[test]
    fn test_softmax_single_element() {
        let result = softmax(&[42.0]);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "Single element softmax should be 1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_softmax_known_values() {
        // For [1, 2, 3] with temperature=1: known analytical result
        let result = softmax(&[1.0, 2.0, 3.0]);

        assert_eq!(result.len(), 3);

        // Sum should be 1
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1.0, got {}",
            sum
        );

        // Ordering preserved
        assert!(
            result[2] > result[1],
            "Higher logit should get higher probability"
        );
        assert!(
            result[1] > result[0],
            "Middle logit should beat lower logit"
        );

        // Approximate known values: softmax([1,2,3]) ~ [0.0900, 0.2447, 0.6652]
        assert!(
            (result[0] - 0.0900).abs() < 0.01,
            "Expected ~0.09, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.2447).abs() < 0.01,
            "Expected ~0.245, got {}",
            result[1]
        );
        assert!(
            (result[2] - 0.6652).abs() < 0.01,
            "Expected ~0.665, got {}",
            result[2]
        );
    }

    #[test]
    fn test_softmax_near_zero_temperature() {
        // Very low temperature should concentrate all mass on the max element
        let result = softmax_with_temperature(&[1.0, 2.0, 3.0], 1e-10);

        assert!(
            result[2] > 0.99,
            "Near-zero temperature should yield near-1.0 for max element, got {}",
            result[2]
        );
    }

    #[test]
    fn test_softmax_high_temperature() {
        // Very high temperature should approach uniform distribution
        let result = softmax_with_temperature(&[1.0, 2.0, 3.0], 1000.0);

        let expected_uniform = 1.0 / 3.0;
        for (i, &p) in result.iter().enumerate() {
            assert!(
                (p - expected_uniform).abs() < 0.01,
                "High temperature should be near-uniform, element {} = {}",
                i,
                p
            );
        }
    }

    #[test]
    fn test_softmax_numerical_stability_large_inputs() {
        // Large values that would overflow naive exp()
        let result = softmax(&[1000.0, 1001.0, 1002.0]);

        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Should handle large inputs without overflow, sum = {}",
            sum
        );

        // Should still preserve ordering
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);

        // All values should be finite
        for (i, &p) in result.iter().enumerate() {
            assert!(p.is_finite(), "Element {} should be finite, got {}", i, p);
        }
    }

    #[test]
    fn test_softmax_numerical_stability_large_negative_inputs() {
        // Large negative values that would underflow naive exp()
        let result = softmax(&[-1000.0, -1001.0, -1002.0]);

        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Should handle large negative inputs, sum = {}",
            sum
        );

        // All values should be finite and positive
        for (i, &p) in result.iter().enumerate() {
            assert!(
                p.is_finite() && p >= 0.0,
                "Element {} should be finite and non-negative, got {}",
                i,
                p
            );
        }
    }

    #[test]
    fn test_softmax_equal_values() {
        // Equal inputs should produce uniform distribution
        let result = softmax(&[5.0, 5.0, 5.0, 5.0]);

        let expected = 0.25;
        for (i, &p) in result.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-6,
                "Equal inputs should give uniform distribution, element {} = {}",
                i,
                p
            );
        }
    }

    // -----------------------------------------------------------------------
    // cosine_similarity_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_similarity_f32_empty() {
        assert_eq!(cosine_similarity_f32(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_f32_zero_vector() {
        let zero = [0.0_f32; 3];
        let v = [1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity_f32(&zero, &v), 0.0);
        assert_eq!(cosine_similarity_f32(&v, &zero), 0.0);
        assert_eq!(cosine_similarity_f32(&zero, &zero), 0.0);
    }

    #[test]
    fn test_cosine_similarity_f32_identical() {
        let v = [1.0, 2.0, 3.0];
        let sim = cosine_similarity_f32(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_f32_orthogonal() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity_f32(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have similarity ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_f32_anti_parallel() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [-1.0_f32, 0.0, 0.0];
        let sim = cosine_similarity_f32(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "Anti-parallel vectors should have similarity -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_f32_different_lengths() {
        let a = [1.0_f32, 2.0];
        let b = [1.0_f32, 2.0, 3.0];
        assert_eq!(
            cosine_similarity_f32(&a, &b),
            0.0,
            "Mismatched lengths should return 0.0"
        );
    }

    #[test]
    fn test_cosine_similarity_f32_scaled_vectors() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [10.0_f32, 20.0, 30.0];
        let sim = cosine_similarity_f32(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Scaled vectors should have similarity ~1.0, got {}",
            sim
        );
    }

    // -----------------------------------------------------------------------
    // cosine_similarity_f64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_similarity_f64_empty() {
        assert_eq!(cosine_similarity_f64(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_f64_zero_vector() {
        let zero = [0.0_f64; 3];
        let v = [1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity_f64(&zero, &v), 0.0);
    }

    #[test]
    fn test_cosine_similarity_f64_identical() {
        let v = [1.0_f64, 2.0, 3.0];
        let sim = cosine_similarity_f64(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-12,
            "Identical vectors should have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_f64_orthogonal() {
        let a = [1.0_f64, 0.0, 0.0];
        let b = [0.0_f64, 1.0, 0.0];
        let sim = cosine_similarity_f64(&a, &b);
        assert!(
            sim.abs() < 1e-12,
            "Orthogonal vectors should have similarity ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_f64_anti_parallel() {
        let a = [1.0_f64, 0.0, 0.0];
        let b = [-1.0_f64, 0.0, 0.0];
        let sim = cosine_similarity_f64(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-12,
            "Anti-parallel vectors should have similarity -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_f64_different_lengths() {
        let a = [1.0_f64, 2.0];
        let b = [1.0_f64, 2.0, 3.0];
        assert_eq!(cosine_similarity_f64(&a, &b), 0.0);
    }

    // -----------------------------------------------------------------------
    // ema_update tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_update_alpha_one() {
        // alpha=1.0 means current value only
        assert!((ema_update(10.0, 5.0, 1.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_ema_update_alpha_zero() {
        // alpha=0.0 means previous value only
        assert!((ema_update(10.0, 5.0, 0.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_ema_update_typical() {
        // 0.3 * 0.6 + 0.7 * 0.8 = 0.18 + 0.56 = 0.74
        let result = ema_update(0.8, 0.6, 0.3);
        assert!(
            (result - 0.74).abs() < 1e-6,
            "Expected 0.74, got {}",
            result
        );
    }

    #[test]
    fn test_ema_update_half() {
        // 0.5 * 10 + 0.5 * 20 = 15
        assert!((ema_update(20.0, 10.0, 0.5) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_ema_update_equal_values() {
        // If previous == current, result should be the same regardless of alpha
        assert!((ema_update(7.0, 7.0, 0.3) - 7.0).abs() < 1e-6);
        assert!((ema_update(7.0, 7.0, 0.9) - 7.0).abs() < 1e-6);
    }
}
