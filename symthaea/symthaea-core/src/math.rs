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
        assert!(result[2] > result[1], "Higher logit should get higher probability");
        assert!(result[1] > result[0], "Middle logit should beat lower logit");

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
            assert!(p.is_finite() && p >= 0.0, "Element {} should be finite and non-negative, got {}", i, p);
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
}
