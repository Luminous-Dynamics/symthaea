// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! f32 <-> f64 Conversion Utilities
//!
//! The core crate uses f32 for memory efficiency and consistency with
//! Symthaea/Holochain. SDK layers that use f64 can convert at the boundary.

use crate::types::{AggregatedGradient, AggregationMethod, GradientMetadata, GradientUpdate};

/// Convert f64 gradient vector to f32
pub fn gradients_to_f32(gradients: &[f64]) -> Vec<f32> {
    gradients.iter().map(|&g| g as f32).collect()
}

/// Convert f32 gradient vector to f64
pub fn gradients_to_f64(gradients: &[f32]) -> Vec<f64> {
    gradients.iter().map(|&g| g as f64).collect()
}

/// Convert an f64-based GradientUpdate to f32
pub fn update_to_f32(
    participant_id: String,
    model_version: u64,
    gradients_f64: &[f64],
    batch_size: u32,
    loss: f64,
    accuracy: Option<f64>,
    timestamp: u64,
) -> GradientUpdate {
    GradientUpdate {
        participant_id,
        model_version,
        gradients: gradients_to_f32(gradients_f64),
        metadata: GradientMetadata {
            batch_size,
            loss: loss as f32,
            accuracy: accuracy.map(|a| a as f32),
            timestamp,
        },
    }
}

/// Convert an f32 AggregatedGradient to f64 gradient vector
pub fn aggregated_to_f64(result: &AggregatedGradient) -> Vec<f64> {
    gradients_to_f64(&result.gradients)
}

/// Convert AggregationMethod to/from string for cross-language compatibility
pub fn method_to_string(method: AggregationMethod) -> &'static str {
    match method {
        AggregationMethod::FedAvg => "fedavg",
        AggregationMethod::TrimmedMean => "trimmed_mean",
        AggregationMethod::Median => "median",
        AggregationMethod::Krum => "krum",
        AggregationMethod::MultiKrum => "multi_krum",
        AggregationMethod::GeometricMedian => "geometric_median",
        AggregationMethod::TrustWeighted => "trust_weighted",
    }
}

pub fn method_from_string(s: &str) -> Option<AggregationMethod> {
    match s {
        "fedavg" => Some(AggregationMethod::FedAvg),
        "trimmed_mean" => Some(AggregationMethod::TrimmedMean),
        "median" => Some(AggregationMethod::Median),
        "krum" => Some(AggregationMethod::Krum),
        "multi_krum" => Some(AggregationMethod::MultiKrum),
        "geometric_median" => Some(AggregationMethod::GeometricMedian),
        "trust_weighted" => Some(AggregationMethod::TrustWeighted),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_to_f32_roundtrip() {
        let original: Vec<f64> = vec![0.1, 0.2, 0.3, 1.5, -0.7];
        let f32_vec = gradients_to_f32(&original);
        let back = gradients_to_f64(&f32_vec);

        for (orig, back_val) in original.iter().zip(back.iter()) {
            let relative_error = if orig.abs() > 1e-10 {
                (orig - back_val).abs() / orig.abs()
            } else {
                (orig - back_val).abs()
            };
            assert!(
                relative_error < 1e-6,
                "Precision loss too high: {} vs {} (error: {})",
                orig,
                back_val,
                relative_error
            );
        }
    }

    #[test]
    fn test_update_conversion() {
        let update = update_to_f32(
            "p1".to_string(),
            1,
            &[0.1, 0.2, 0.3],
            100,
            0.5,
            Some(0.95),
            12345,
        );
        assert_eq!(update.gradients.len(), 3);
        assert!((update.gradients[0] - 0.1).abs() < 0.001);
        assert_eq!(update.metadata.batch_size, 100);
        assert_eq!(update.metadata.accuracy, Some(0.95));
        assert_eq!(update.metadata.timestamp, 12345);
    }

    #[test]
    fn test_method_strings() {
        assert_eq!(method_to_string(AggregationMethod::FedAvg), "fedavg");
        assert_eq!(
            method_from_string("trimmed_mean"),
            Some(AggregationMethod::TrimmedMean)
        );
        assert_eq!(method_from_string("unknown"), None);
    }

    #[test]
    fn test_f64_f32_precision_edge_cases() {
        // Typical gradient magnitudes (1e-6 to 1e2)
        let typical_gradients: Vec<f64> = vec![
            1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, -1e-6, -1e-4, -0.01, -0.5,
            -10.0, -100.0,
        ];
        let f32_vec = gradients_to_f32(&typical_gradients);
        let back = gradients_to_f64(&f32_vec);
        for (orig, back_val) in typical_gradients.iter().zip(back.iter()) {
            let relative_error = if orig.abs() > 1e-10 {
                (orig - back_val).abs() / orig.abs()
            } else {
                (orig - back_val).abs()
            };
            assert!(
                relative_error < 1e-6,
                "Typical gradient precision loss: {} vs {} (rel err: {:.2e})",
                orig,
                back_val,
                relative_error
            );
        }
    }

    #[test]
    fn test_f64_f32_precision_very_small_values() {
        // Very small values near f32 subnormal range
        let small: Vec<f64> = vec![1e-38, 1e-30, 1e-20, 1e-10, f64::MIN_POSITIVE];
        let f32_vec = gradients_to_f32(&small);
        let back = gradients_to_f64(&f32_vec);
        // Very small values may lose precision but should not become NaN/Inf
        for val in &back {
            assert!(val.is_finite(), "Small value became non-finite: {}", val);
        }
    }

    #[test]
    fn test_f64_f32_precision_large_values() {
        // Large values within f32 range (f32::MAX ≈ 3.4e38)
        let large: Vec<f64> = vec![1e10, 1e20, 1e30, 1e38, -1e38];
        let f32_vec = gradients_to_f32(&large);
        let back = gradients_to_f64(&f32_vec);
        for (orig, back_val) in large.iter().zip(back.iter()) {
            let relative_error = (orig - back_val).abs() / orig.abs();
            assert!(
                relative_error < 1e-6,
                "Large value precision loss: {:.2e} vs {:.2e} (rel err: {:.2e})",
                orig,
                back_val,
                relative_error
            );
        }
    }

    #[test]
    fn test_f64_f32_overflow_clamps_to_infinity() {
        // Values beyond f32 range become infinity
        let overflow: Vec<f64> = vec![1e39, -1e39, f64::MAX, f64::MIN];
        let f32_vec = gradients_to_f32(&overflow);
        assert!(f32_vec[0].is_infinite());
        assert!(f32_vec[1].is_infinite());
        assert!(f32_vec[2].is_infinite());
        assert!(f32_vec[3].is_infinite());
    }

    #[test]
    fn test_f64_f32_zero_and_negative_zero() {
        let zeros: Vec<f64> = vec![0.0, -0.0];
        let f32_vec = gradients_to_f32(&zeros);
        let back = gradients_to_f64(&f32_vec);
        assert_eq!(back[0], 0.0);
        assert_eq!(back[1], 0.0); // -0.0 == 0.0 in IEEE 754
    }

    #[test]
    fn test_f64_f32_nan_preserved() {
        let nans: Vec<f64> = vec![f64::NAN];
        let f32_vec = gradients_to_f32(&nans);
        assert!(f32_vec[0].is_nan());
        let back = gradients_to_f64(&f32_vec);
        assert!(back[0].is_nan());
    }

    #[test]
    fn test_sdk_aggregation_roundtrip_precision() {
        // Simulate the actual SDK workflow: f64 → f32 (core) → f32 result → f64
        let sdk_gradients: Vec<Vec<f64>> = vec![
            vec![0.001, -0.002, 0.003, -0.004, 0.005],
            vec![0.002, -0.001, 0.004, -0.003, 0.006],
            vec![0.003, -0.003, 0.002, -0.005, 0.004],
        ];

        // Convert to f32 (as the SDK does before calling core)
        let f32_updates: Vec<Vec<f32>> =
            sdk_gradients.iter().map(|g| gradients_to_f32(g)).collect();

        // Simulate FedAvg in f32 (what core does)
        let dim = f32_updates[0].len();
        let mut aggregated_f32 = vec![0.0f32; dim];
        for update in &f32_updates {
            for (i, &val) in update.iter().enumerate() {
                aggregated_f32[i] += val / f32_updates.len() as f32;
            }
        }

        // Convert result back to f64 (as SDK does after core returns)
        let result_f64 = gradients_to_f64(&aggregated_f32);

        // Compute expected result in f64 directly
        let mut expected_f64 = vec![0.0f64; dim];
        for g in &sdk_gradients {
            for (i, &val) in g.iter().enumerate() {
                expected_f64[i] += val / sdk_gradients.len() as f64;
            }
        }

        // The roundtrip error should be small enough for ML (< 1e-6 relative)
        for (i, (&expected, &actual)) in expected_f64.iter().zip(result_f64.iter()).enumerate() {
            let relative_error = if expected.abs() > 1e-10 {
                (expected - actual).abs() / expected.abs()
            } else {
                (expected - actual).abs()
            };
            assert!(
                relative_error < 1e-5,
                "SDK roundtrip dim {}: expected {:.8e}, got {:.8e} (rel err: {:.2e})",
                i,
                expected,
                actual,
                relative_error
            );
        }
    }
}
