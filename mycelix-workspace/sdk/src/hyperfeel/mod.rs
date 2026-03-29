// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HyperFeel Gradient Compression Module
//!
//! Revolutionary gradient compression using hyperdimensional computing:
//! - 2000x compression (10M parameters → 2KB hypervector)
//! - Preserves gradient semantics (cosine similarity)
//! - Binary hypervectors for efficient storage and comparison
//! - Compatible with Symthaea-HLB and MATL via `HyperGradient`
//!
//! # Mathematical Foundation
//!
//! Uses Johnson-Lindenstrauss lemma: random projections preserve
//! pairwise distances with high probability. For n points in R^d,
//! projecting to R^k where k = O(log(n)/ε²) preserves distances
//! within (1±ε) factor.
//!
//! # Example
//!
//! ```rust
//! use mycelix_sdk::hyperfeel::{HyperFeelEncoder, EncodingConfig, HyperGradient};
//!
//! let config = EncodingConfig::default();
//! let mut encoder = HyperFeelEncoder::new(config);
//!
//! // Encode 1M parameter gradient to 2KB
//! let gradient: Vec<f32> = vec![0.1; 1_000_000];
//! let hg = encoder.encode_gradient(&gradient, 1, "node-1");
//!
//! println!("Compression: {}x", hg.compression_ratio);
//! ```

mod encoder;
mod types;

pub use encoder::HyperFeelEncoder;
pub use types::{EncodingConfig, HyperGradient, HV16_BYTES, HV16_DIMENSION};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_similarity() {
        let config = EncodingConfig::default();
        let mut encoder = HyperFeelEncoder::new(config);

        // Two similar gradients (small perturbation)
        let gradient1: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.001).sin()).collect();
        let gradient2: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.001).sin() + 0.01).collect();

        // Completely different gradient (different frequency and shape)
        // This creates a truly different distribution, not just inverted
        let gradient3: Vec<f32> = (0..5000)
            .map(|i| {
                let x = i as f32;
                // Mix of different patterns: spiky, exponential decay
                ((x * 0.1).sin() * (x * 0.05).cos()).powi(3) + (x / 5000.0).exp() - 1.0
            })
            .collect();

        let hg1 = encoder.encode_gradient(&gradient1, 1, "node-1");
        let hg2 = encoder.encode_gradient(&gradient2, 1, "node-2");
        let hg3 = encoder.encode_gradient(&gradient3, 1, "node-3");

        let sim_similar = HyperFeelEncoder::cosine_similarity(&hg1.hypervector, &hg2.hypervector);
        let sim_different = HyperFeelEncoder::cosine_similarity(&hg1.hypervector, &hg3.hypervector);

        // Similar gradients should have high similarity
        assert!(
            sim_similar > 0.7,
            "Similar gradients should be similar: {}",
            sim_similar
        );
        // Different distribution should be detectable
        // Note: HyperFeel with JL projection preserves L2 distance, so different shapes should differ
        assert!(
            sim_similar > sim_different,
            "Similar grads ({}) should be more similar than different ones ({})",
            sim_similar,
            sim_different
        );
    }

    #[test]
    fn test_compression_ratio() {
        let config = EncodingConfig::default();
        let mut encoder = HyperFeelEncoder::new(config);

        // Use 100K elements (400KB) for fast test
        let gradient: Vec<f32> = vec![0.1; 100_000];
        let hg = encoder.encode_gradient(&gradient, 1, "node-1");

        assert_eq!(hg.original_size, 400_000);
        assert_eq!(hg.hypervector.len(), HV16_BYTES);
        // 400KB / 2KB = 195x compression (not the full 2000x but proves the mechanism)
        assert!(
            hg.compression_ratio > 150.0,
            "Compression ratio: {}",
            hg.compression_ratio
        );
    }
}
