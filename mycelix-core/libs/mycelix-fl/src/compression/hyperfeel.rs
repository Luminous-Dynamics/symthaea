// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! HyperFeel V2 gradient compression using Johnson-Lindenstrauss random projection.
//!
//! The core idea: project high-dimensional gradients (100K+ dims) into a much smaller
//! space (e.g., 512D) while approximately preserving distances (JL lemma).
//!
//! Uses the Achlioptas (2003) sparse random projection scheme:
//! - Each entry: P(+1) = 1/6, P(0) = 2/3, P(-1) = 1/6
//! - Scale by sqrt(3/k) where k = compressed_dim
//! - Gives (1 +/- epsilon) distance preservation with high probability
//!   when k = O(log(n) / epsilon^2)
//!
//! # Example
//!
//! ```
//! use mycelix_fl::compression::hyperfeel::{HyperFeelEncoder, HyperFeelConfig};
//! use mycelix_fl::Gradient;
//!
//! let config = HyperFeelConfig::default();
//! let encoder = HyperFeelEncoder::new(1000, config);
//!
//! let g = Gradient::new("node-1", vec![0.1; 1000], 1);
//! let compressed = encoder.compress(&g).unwrap();
//! assert_eq!(compressed.values.len(), 512);
//! assert!(compressed.compression_ratio > 1.0);
//! ```

use crate::error::FlError;
use crate::types::Gradient;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sparse projection storage
// ---------------------------------------------------------------------------

/// A single row of the sparse Achlioptas projection matrix.
///
/// Only non-zero entries are stored: each has a column index and a sign (+1 or -1).
#[derive(Clone, Debug)]
struct SparseProjectionRow {
    /// Column indices of the non-zero entries.
    indices: Vec<usize>,
    /// Signs of the non-zero entries (+1 or -1).
    signs: Vec<i8>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the HyperFeel encoder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperFeelConfig {
    /// Target compressed dimension (default 512).
    pub compressed_dim: usize,
    /// Fraction of non-zero entries per row (default 1/3 per Achlioptas 2003).
    pub sparse_density: f32,
    /// Seed for reproducible projection matrix generation.
    pub seed: u64,
}

impl Default for HyperFeelConfig {
    fn default() -> Self {
        Self {
            compressed_dim: 512,
            sparse_density: 1.0 / 3.0,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Compressed gradient
// ---------------------------------------------------------------------------

/// A gradient after HyperFeel compression.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedGradient {
    /// The compressed vector in the projected space.
    pub values: Vec<f32>,
    /// Node that produced the original gradient.
    pub node_id: String,
    /// FL round number.
    pub round: u64,
    /// Dimensionality of the original (uncompressed) gradient.
    pub original_dim: usize,
    /// Ratio of original_dim / compressed_dim.
    pub compression_ratio: f32,
}

// ---------------------------------------------------------------------------
// Simple xorshift64 PRNG (no external dep needed)
// ---------------------------------------------------------------------------

/// Minimal xorshift64 PRNG for deterministic projection generation.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Avoid zero state.
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Returns a pseudo-random u64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a pseudo-random u32 in [0, bound).
    fn next_bounded(&mut self, bound: u32) -> u32 {
        (self.next_u64() % bound as u64) as u32
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// HyperFeel V2 gradient compression encoder.
///
/// Compresses high-dimensional gradients into a fixed low-dimensional space
/// using sparse Achlioptas random projection, preserving approximate pairwise
/// distances per the Johnson-Lindenstrauss lemma.
pub struct HyperFeelEncoder {
    /// Sparse projection rows: `[compressed_dim]` rows, each referencing columns in `[0, original_dim)`.
    rows: Vec<SparseProjectionRow>,
    /// Target compressed dimension.
    compressed_dim: usize,
    /// Original (input) gradient dimension.
    original_dim: usize,
    /// Scale factor: sqrt(3 / compressed_dim) for Achlioptas.
    scale: f32,
}

impl HyperFeelEncoder {
    /// Create a new encoder for gradients of dimension `original_dim`.
    ///
    /// Generates the sparse random projection matrix deterministically from
    /// `config.seed`.
    pub fn new(original_dim: usize, config: HyperFeelConfig) -> Self {
        let compressed_dim = config.compressed_dim;
        let scale = (3.0f32 / compressed_dim as f32).sqrt();

        let mut rng = Xorshift64::new(config.seed);

        // Achlioptas (2003): P(+1) = 1/6, P(0) = 2/3, P(-1) = 1/6.
        // We draw a random number in [0, 6) for each entry:
        //   0 => +1, 1..=4 => 0, 5 => -1
        let rows: Vec<SparseProjectionRow> = (0..compressed_dim)
            .map(|_| {
                let mut indices = Vec::new();
                let mut signs = Vec::new();
                for col in 0..original_dim {
                    let r = rng.next_bounded(6);
                    match r {
                        0 => {
                            indices.push(col);
                            signs.push(1);
                        }
                        5 => {
                            indices.push(col);
                            signs.push(-1);
                        }
                        _ => {} // 2/3 probability: zero entry
                    }
                }
                SparseProjectionRow { indices, signs }
            })
            .collect();

        Self {
            rows,
            compressed_dim,
            original_dim,
            scale,
        }
    }

    /// Compress a single gradient.
    ///
    /// Returns `Err(FlError::DimensionMismatch)` if the gradient dimension does
    /// not match `original_dim`.
    pub fn compress(&self, gradient: &Gradient) -> Result<CompressedGradient, FlError> {
        if gradient.values.len() != self.original_dim {
            return Err(FlError::DimensionMismatch {
                expected: self.original_dim,
                got: gradient.values.len(),
            });
        }

        let values = self.project(&gradient.values);

        Ok(CompressedGradient {
            values,
            node_id: gradient.node_id.clone(),
            round: gradient.round,
            original_dim: self.original_dim,
            compression_ratio: self.ratio(),
        })
    }

    /// Compress a batch of gradients.
    pub fn compress_batch(
        &self,
        gradients: &[Gradient],
    ) -> Result<Vec<CompressedGradient>, FlError> {
        gradients.iter().map(|g| self.compress(g)).collect()
    }

    /// Approximate cosine similarity between two compressed gradients.
    ///
    /// The JL lemma guarantees that this is within +/- epsilon of the true
    /// cosine similarity in the original space (with high probability).
    pub fn compressed_similarity(a: &CompressedGradient, b: &CompressedGradient) -> f64 {
        crate::types::cosine_similarity(&a.values, &b.values)
    }

    /// Compression ratio: original_dim / compressed_dim.
    pub fn ratio(&self) -> f32 {
        self.original_dim as f32 / self.compressed_dim as f32
    }

    /// The compressed dimension.
    pub fn compressed_dim(&self) -> usize {
        self.compressed_dim
    }

    /// The original (input) dimension.
    pub fn original_dim(&self) -> usize {
        self.original_dim
    }

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    /// Sparse matrix-vector product: y = scale * R * x
    fn project(&self, x: &[f32]) -> Vec<f32> {
        self.rows
            .iter()
            .map(|row| {
                let dot: f32 = row
                    .indices
                    .iter()
                    .zip(&row.signs)
                    .map(|(&idx, &sign)| x[idx] * sign as f32)
                    .sum();
                dot * self.scale
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::cosine_similarity;

    fn make_gradient(values: Vec<f32>, node: &str, round: u64) -> Gradient {
        Gradient::new(node, values, round)
    }

    #[test]
    fn test_compression_reduces_dimension() {
        let dim = 2000;
        let config = HyperFeelConfig {
            compressed_dim: 128,
            ..Default::default()
        };
        let encoder = HyperFeelEncoder::new(dim, config);
        let g = make_gradient(vec![1.0; dim], "n1", 0);
        let c = encoder.compress(&g).unwrap();
        assert_eq!(c.values.len(), 128);
        assert_eq!(c.original_dim, dim);
    }

    #[test]
    fn test_compression_ratio_correct() {
        let dim = 1024;
        let config = HyperFeelConfig {
            compressed_dim: 256,
            ..Default::default()
        };
        let encoder = HyperFeelEncoder::new(dim, config);
        assert!((encoder.ratio() - 4.0).abs() < 1e-6);

        let g = make_gradient(vec![0.5; dim], "n1", 1);
        let c = encoder.compress(&g).unwrap();
        assert!((c.compression_ratio - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_preservation() {
        // JL guarantee: cosine similarity should be approximately preserved.
        // Use a large enough compressed_dim for the bound to be meaningful.
        let dim = 5000;
        let compressed_dim = 512;
        let config = HyperFeelConfig {
            compressed_dim,
            seed: 123,
            ..Default::default()
        };
        let encoder = HyperFeelEncoder::new(dim, config);

        // Two vectors with known cosine similarity.
        let mut a_vals = vec![0.0f32; dim];
        let mut b_vals = vec![0.0f32; dim];
        // Make them correlated: first half shared, second half different.
        for i in 0..dim {
            a_vals[i] = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
            if i < dim / 2 {
                b_vals[i] = a_vals[i]; // Same
            } else {
                b_vals[i] = -(a_vals[i]); // Opposite
            }
        }

        let orig_sim = cosine_similarity(&a_vals, &b_vals);

        let ga = make_gradient(a_vals, "a", 0);
        let gb = make_gradient(b_vals, "b", 0);
        let ca = encoder.compress(&ga).unwrap();
        let cb = encoder.compress(&gb).unwrap();
        let comp_sim = HyperFeelEncoder::compressed_similarity(&ca, &cb);

        // Allow +/- 0.15 tolerance (JL epsilon for this dimension).
        assert!(
            (comp_sim - orig_sim).abs() < 0.15,
            "Distance not preserved: original={orig_sim:.4}, compressed={comp_sim:.4}"
        );
    }

    #[test]
    fn test_batch_compression() {
        let dim = 500;
        let config = HyperFeelConfig::default();
        let encoder = HyperFeelEncoder::new(dim, config);

        let gradients: Vec<Gradient> = (0..5)
            .map(|i| make_gradient(vec![(i as f32) * 0.1; dim], &format!("n{i}"), 1))
            .collect();
        let batch = encoder.compress_batch(&gradients).unwrap();
        assert_eq!(batch.len(), 5);
        for (i, c) in batch.iter().enumerate() {
            assert_eq!(c.node_id, format!("n{i}"));
            assert_eq!(c.values.len(), 512);
        }
    }

    #[test]
    fn test_deterministic_same_seed() {
        let dim = 500;
        let config1 = HyperFeelConfig {
            seed: 999,
            ..Default::default()
        };
        let config2 = HyperFeelConfig {
            seed: 999,
            ..Default::default()
        };
        let encoder1 = HyperFeelEncoder::new(dim, config1);
        let encoder2 = HyperFeelEncoder::new(dim, config2);

        let g = make_gradient(vec![0.42; dim], "n1", 0);
        let c1 = encoder1.compress(&g).unwrap();
        let c2 = encoder2.compress(&g).unwrap();
        assert_eq!(c1.values, c2.values);
    }

    #[test]
    fn test_zero_gradient_produces_zero() {
        let dim = 500;
        let encoder = HyperFeelEncoder::new(dim, HyperFeelConfig::default());
        let g = make_gradient(vec![0.0; dim], "n1", 0);
        let c = encoder.compress(&g).unwrap();
        for &v in &c.values {
            assert!(v.abs() < 1e-12, "Expected zero, got {v}");
        }
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let encoder = HyperFeelEncoder::new(100, HyperFeelConfig::default());
        let g = make_gradient(vec![1.0; 200], "n1", 0);
        let result = encoder.compress(&g);
        assert!(result.is_err());
        match result.unwrap_err() {
            FlError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 100);
                assert_eq!(got, 200);
            }
            other => panic!("Expected DimensionMismatch, got: {other:?}"),
        }
    }
}
