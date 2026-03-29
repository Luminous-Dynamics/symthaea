// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for mycelix-fl using proptest.
//!
//! Covers:
//! - HyperFeel compression (always enabled)
//! - PoGQ-v4.1 Lite detection (feature-gated: pogq)
//! - CoherenceTimeSeries tracking (feature-gated: coherence-series | phi-series)

use proptest::prelude::*;

use mycelix_fl::compression::{EncodingConfig, HyperFeelCompressor};
use mycelix_fl::types::HV16_BYTES;

// =============================================================================
// Strategies
// =============================================================================

/// Generate a Vec<f32> with values in [-1.0, 1.0], length in [1, dim].
fn gradient_strategy(max_dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..=1.0f32, 1..=max_dim)
}

/// Generate a non-empty participant ID (1..16 alphanumeric chars).
fn participant_id_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9]{1,16}"
}

// =============================================================================
// HyperFeel Compression Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Compressed output is always exactly HV16_BYTES (2048).
    #[test]
    fn compress_output_is_always_hv16_bytes(
        gradient in gradient_strategy(4096),
        pid in participant_id_strategy(),
        round in 0u32..1000,
        quality in 0.0f32..=1.0f32,
    ) {
        let compressor = HyperFeelCompressor::default_seed();
        let result = compressor.compress(&pid, round, &gradient, quality).unwrap();
        prop_assert_eq!(result.hv_data.len(), HV16_BYTES);
    }

    /// Encode -> decode preserves sign of large components.
    ///
    /// Components with |value| > 0.5 should retain their sign after a
    /// round-trip through compress/decompress when the gradient is small
    /// enough that folding does not collide (dim <= HV16_BYTES).
    #[test]
    fn encode_decode_preserves_sign_of_large_components(
        gradient in prop::collection::vec(
            prop::sample::select(vec![-1.0f32, -0.8, -0.6, 0.6, 0.8, 1.0]),
            8..=64,
        ),
    ) {
        // Use no-seed compressor (modular folding) so position mapping is
        // trivial for small gradients where dim < HV16_BYTES.
        let compressor = HyperFeelCompressor::new(EncodingConfig::default());
        let compressed = compressor.compress("test", 1, &gradient, 0.9).unwrap();
        let decoded = compressor.decompress(&compressed.hv_data, gradient.len()).unwrap();

        prop_assert_eq!(decoded.len(), gradient.len());

        for (i, (&orig, &dec)) in gradient.iter().zip(decoded.iter()).enumerate() {
            if orig.abs() > 0.5 {
                prop_assert!(
                    orig.signum() == dec.signum(),
                    "Sign mismatch at index {}: original={}, decoded={}",
                    i, orig, dec,
                );
            }
        }
    }

    /// Deterministic: same seed + same input -> identical hv_data.
    #[test]
    fn compress_is_deterministic_with_same_seed(
        seed in 1u64..10_000,
        gradient in gradient_strategy(512),
    ) {
        let c1 = HyperFeelCompressor::with_seed(seed);
        let c2 = HyperFeelCompressor::with_seed(seed);
        let r1 = c1.compress("p1", 1, &gradient, 0.5).unwrap();
        let r2 = c2.compress("p1", 1, &gradient, 0.5).unwrap();
        prop_assert_eq!(r1.hv_data, r2.hv_data);
    }
}

// =============================================================================
// PoGQ Properties (feature-gated)
// =============================================================================

#[cfg(feature = "pogq")]
mod pogq_properties {
    use super::*;
    use mycelix_fl::pogq::{PoGQLiteConfig, PoGQLiteDetector};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// hybrid_score is in [0, infinity) but ema should stay finite.
        /// Since direction is post-ReLU (>=0) and utility is l2_norm (>=0),
        /// hybrid = lambda*direction + (1-lambda)*utility >= 0.
        #[test]
        fn pogq_hybrid_score_is_non_negative(
            gradient in gradient_strategy(128),
            reference in prop::collection::vec(0.1f32..=2.0, 128..=128),
        ) {
            let mut det = PoGQLiteDetector::new(PoGQLiteConfig::default());
            let result = det.score_gradient(&gradient, "c1", &reference, 0);
            prop_assert!(
                result.scores.hybrid >= 0.0,
                "hybrid score should be >= 0, got {}",
                result.scores.hybrid,
            );
            prop_assert!(
                result.scores.hybrid.is_finite(),
                "hybrid score should be finite, got {}",
                result.scores.hybrid,
            );
        }

        /// An honest gradient (close to reference) should score higher than a
        /// random/adversarial gradient (opposite direction) after warm-up.
        #[test]
        fn pogq_honest_scores_higher_than_adversarial(
            base in prop::collection::vec(0.5f32..=1.5, 64..=64),
            noise in prop::collection::vec(-0.1f32..=0.1, 64..=64),
        ) {
            let reference: Vec<f32> = base.clone();
            let honest: Vec<f32> = base.iter().zip(noise.iter())
                .map(|(b, n)| b + n)
                .collect();
            let adversarial: Vec<f32> = base.iter().map(|b| -b).collect();

            let config = PoGQLiteConfig {
                warmup_rounds: 0,
                ..Default::default()
            };

            let mut det_honest = PoGQLiteDetector::new(config.clone());
            let mut det_adv = PoGQLiteDetector::new(config);

            let res_honest = det_honest.score_gradient(&honest, "h1", &reference, 0);
            let res_adv = det_adv.score_gradient(&adversarial, "a1", &reference, 0);

            // Honest gradient has cosine ~1 => direction ~1, so hybrid is higher
            // than adversarial with cosine ~-1 => direction = 0 (post-ReLU).
            prop_assert!(
                res_honest.scores.hybrid > res_adv.scores.hybrid,
                "honest hybrid ({}) should exceed adversarial hybrid ({})",
                res_honest.scores.hybrid,
                res_adv.scores.hybrid,
            );
        }

        /// Deterministic: same inputs -> same PoGQResult fields.
        #[test]
        fn pogq_is_deterministic(
            gradient in prop::collection::vec(-1.0f32..=1.0, 32..=32),
            reference in prop::collection::vec(0.1f32..=2.0, 32..=32),
            round in 0u32..100,
        ) {
            let config = PoGQLiteConfig::default();

            let mut det1 = PoGQLiteDetector::new(config.clone());
            let mut det2 = PoGQLiteDetector::new(config);

            let r1 = det1.score_gradient(&gradient, "c1", &reference, round);
            let r2 = det2.score_gradient(&gradient, "c1", &reference, round);

            prop_assert_eq!(r1.scores.hybrid, r2.scores.hybrid);
            prop_assert_eq!(r1.scores.ema, r2.scores.ema);
            prop_assert_eq!(r1.scores.direction, r2.scores.direction);
            prop_assert_eq!(r1.scores.utility, r2.scores.utility);
            prop_assert_eq!(r1.detection.is_byzantine, r2.detection.is_byzantine);
            prop_assert_eq!(r1.detection.is_violation, r2.detection.is_violation);
        }
    }
}

// =============================================================================
// CoherenceTimeSeries Properties (feature-gated)
// =============================================================================

#[cfg(any(feature = "coherence-series", feature = "phi-series"))]
mod coherence_series_properties {
    use super::*;
    use mycelix_fl::coherence_series::{CoherenceTimeSeries, CoherenceTimeSeriesConfig};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Recording n points -> window contains min(n, window_size) points.
        #[test]
        fn window_size_is_bounded(
            window_size in 3usize..=50,
            n_points in 1usize..=100,
            values in prop::collection::vec(0.0f32..=1.0, 1..=100),
        ) {
            let n = n_points.min(values.len());
            let config = CoherenceTimeSeriesConfig::default()
                .with_window_size(window_size)
                .with_full_history(false);
            let mut series = CoherenceTimeSeries::new(config);

            for i in 0..n {
                series.record(i as u64, i as i64, values[i], 0.9, 0, 10);
            }

            let expected_len = n.min(window_size);
            prop_assert_eq!(
                series.len(),
                expected_len,
                "After recording {} points with window_size={}, len should be {}",
                n, window_size, expected_len,
            );
        }

        /// Statistics: mean is within [min, max] range.
        #[test]
        fn statistics_mean_within_min_max(
            values in prop::collection::vec(0.0f32..=1.0, 2..=50),
        ) {
            let config = CoherenceTimeSeriesConfig::default()
                .with_window_size(values.len());
            let mut series = CoherenceTimeSeries::new(config);

            for (i, &v) in values.iter().enumerate() {
                series.record(i as u64, i as i64, v, 0.9, 0, 10);
            }

            let stats = series.get_statistics();
            prop_assert!(
                stats.mean >= stats.min - 1e-6,
                "mean ({}) should be >= min ({})",
                stats.mean, stats.min,
            );
            prop_assert!(
                stats.mean <= stats.max + 1e-6,
                "mean ({}) should be <= max ({})",
                stats.mean, stats.max,
            );
            prop_assert_eq!(stats.sample_count, values.len());
            prop_assert!(stats.std_dev >= 0.0, "std_dev should be non-negative");
        }

        /// Anomaly detection: extreme outlier (z-score > threshold) triggers anomaly.
        #[test]
        fn anomaly_detected_for_extreme_outlier(
            baseline in 0.4f32..=0.6,
            window_size in 10usize..=30,
        ) {
            let threshold = 2.0f32;
            let config = CoherenceTimeSeriesConfig::default()
                .with_window_size(window_size)
                .with_anomaly_threshold(threshold);
            let mut series = CoherenceTimeSeries::new(config);

            // Record stable baseline values to build statistics
            for i in 0..window_size {
                series.record(i as u64, i as i64, baseline, 0.9, 0, 10);
            }

            // Inject a massive outlier (well beyond any threshold)
            let outlier = baseline - 10.0;
            series.record(window_size as u64, window_size as i64, outlier, 0.5, 2, 10);

            let anomaly = series.detect_anomaly();
            prop_assert!(
                anomaly.is_some(),
                "An extreme outlier ({} vs baseline {}) should trigger anomaly detection",
                outlier, baseline,
            );

            let anomaly = anomaly.unwrap();
            prop_assert!(
                anomaly.z_score.abs() >= threshold,
                "z_score ({}) should exceed threshold ({})",
                anomaly.z_score.abs(), threshold,
            );
        }
    }
}

// =============================================================================
// HV16 Bipolar Conversion Properties
//
// Tests the byte→bipolar conversion logic used by the FL coordinator zome.
// The algorithm is: for each byte, MSB-first, bit=1 → +1.0, bit=0 → -1.0.
// This is tested here because the zome targets wasm32 and can't run unit tests.
// =============================================================================

/// Convert HV16 binary bytes to bipolar f32 (same algorithm as pipeline.rs:hv16_to_bipolar)
fn hv16_to_bipolar(hv_bytes: &[u8]) -> Vec<f32> {
    hv_bytes
        .iter()
        .flat_map(|byte| {
            (0..8).rev().map(move |i| {
                if (byte >> i) & 1 == 1 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Output length is always 8x input length.
    #[test]
    fn hv16_bipolar_output_length(
        bytes in prop::collection::vec(0u8..=255, 1..=256)
    ) {
        let result = hv16_to_bipolar(&bytes);
        prop_assert_eq!(result.len(), bytes.len() * 8);
    }

    /// All output values are exactly +1.0 or -1.0.
    #[test]
    fn hv16_bipolar_values_are_unit(
        bytes in prop::collection::vec(0u8..=255, 1..=128)
    ) {
        let result = hv16_to_bipolar(&bytes);
        for (i, &v) in result.iter().enumerate() {
            prop_assert!(
                v == 1.0 || v == -1.0,
                "Value at index {} is {}, expected ±1.0", i, v,
            );
        }
    }

    /// Deterministic: same input always produces same output.
    #[test]
    fn hv16_bipolar_deterministic(
        bytes in prop::collection::vec(0u8..=255, 1..=64)
    ) {
        let r1 = hv16_to_bipolar(&bytes);
        let r2 = hv16_to_bipolar(&bytes);
        prop_assert_eq!(r1, r2);
    }
}

#[cfg(test)]
mod hv16_bipolar_unit_tests {
    use super::hv16_to_bipolar;

    #[test]
    fn all_zeros_byte() {
        let result = hv16_to_bipolar(&[0x00]);
        assert_eq!(result, vec![-1.0; 8]);
    }

    #[test]
    fn all_ones_byte() {
        let result = hv16_to_bipolar(&[0xFF]);
        assert_eq!(result, vec![1.0; 8]);
    }

    #[test]
    fn known_pattern() {
        // 0b10100101 = 0xA5
        let result = hv16_to_bipolar(&[0xA5]);
        assert_eq!(
            result,
            vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]
        );
    }

    #[test]
    fn empty_input() {
        let result = hv16_to_bipolar(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn hv16_full_size() {
        // HV16 = 2048 bytes = 16384 bipolar values
        let bytes = vec![0xAA; 2048]; // alternating 1010...
        let result = hv16_to_bipolar(&bytes);
        assert_eq!(result.len(), 16384);
        // 0xAA = 0b10101010 → [1, -1, 1, -1, 1, -1, 1, -1]
        for chunk in result.chunks(8) {
            assert_eq!(chunk, &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        }
    }
}
