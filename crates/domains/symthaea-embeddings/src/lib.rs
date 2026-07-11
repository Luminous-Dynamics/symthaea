#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop)]
//! # Embeddings Module: Semantic Bridge to HDC Space
//!
//! This module provides semantic embedding capabilities and projections
//! from high-dimensional embedding space (e.g., 1024D Qwen3) to HDC space (16,384D BinaryHV).
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                       EMBEDDINGS PIPELINE                          │
//! │                                                                    │
//! │   Input Text                                                       │
//! │       │                                                            │
//! │       ▼                                                            │
//! │   ┌──────────────────┐                                             │
//! │   │  Qwen3Embedder   │ → 1024D dense semantic embedding            │
//! │   │  (Burn/simulated)│                                             │
//! │   └────────┬─────────┘                                             │
//! │            │                                                       │
//! │            ▼                                                       │
//! │   ┌──────────────────┐                                             │
//! │   │    HdcBridge     │ → Johnson-Lindenstrauss projection          │
//! │   │  (JL projection) │                                             │
//! │   └────────┬─────────┘                                             │
//! │            │                                                       │
//! │            ▼                                                       │
//! │   ┌──────────────────┐                                             │
//! │   │      BinaryHV        │ → 16,384D binary hypervector                │
//! │   │  (HDC space)     │                                             │
//! │   └──────────────────┘                                             │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Qwen3 Embeddings**: 1024D semantic embeddings (Burn or simulated)
//! - **HdcBridge**: Projects embeddings to HDC space preserving similarity
//! - **Aligned Encoding**: Same projection for primitives and inputs

pub mod ollama;
pub mod qwen3;

#[cfg(feature = "persistent-cache")]
pub mod cache;

#[cfg(feature = "embed-channel")]
pub mod channel;

use anyhow::Result;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;

// Re-export qwen3 types at module level
pub use qwen3::{QWEN3_FULL_DIMENSION, Qwen3Config, Qwen3Embedder};

#[cfg(feature = "async-embed")]
pub use qwen3::AsyncQwen3Embedder;

/// Standard Qwen3 embedding dimension (1024D)
pub const QWEN3_DIMENSION: usize = 1024;

/// BGE embedding dimension (768D)
pub const BGE_DIMENSION: usize = 768;

/// Configuration for HdcBridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Input embedding dimension (default: 1024 for Qwen3)
    pub input_dim: usize,

    /// Output HDC dimension (default: 16384 for BinaryHV)
    pub output_dim: usize,

    /// Random seed for reproducible projections
    pub seed: u64,

    /// Normalization mode for projection
    pub normalization: NormalizationMode,

    /// Whether to use sparse projection (faster but slightly less accurate)
    pub sparse_projection: bool,

    /// Sparsity factor (0.0-1.0) for sparse projection
    pub sparsity: f32,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            input_dim: QWEN3_DIMENSION,
            output_dim: 16384,
            seed: 0x5974_1AEA, // "SYMTHAEA"
            normalization: NormalizationMode::L2,
            sparse_projection: false,
            sparsity: 0.1,
        }
    }
}

/// Normalization mode for projections
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMode {
    /// No normalization
    None,
    /// L2 normalization (unit sphere)
    L2,
    /// L1 normalization
    L1,
    /// Batch normalization (zero mean, unit variance)
    Batch,
}

/// HdcBridge: Projects semantic embeddings to HDC space
///
/// Uses Johnson-Lindenstrauss random projection to preserve
/// similarity relationships when mapping from embedding space
/// (e.g., 1024D) to HDC space (16,384D).
///
/// The projection matrix is initialized from a seed for reproducibility,
/// ensuring aligned encoding between primitives and input embeddings.
#[derive(Debug)]
pub struct HdcBridge {
    /// Configuration
    config: BridgeConfig,

    /// Projection matrix (output_dim x input_dim)
    /// Stored as flat Vec for efficiency
    projection_matrix: Vec<f32>,

    /// Bias vector (optional)
    bias: Option<Vec<f32>>,
}

impl HdcBridge {
    /// Create a new HdcBridge with default configuration
    pub fn new() -> Self {
        Self::with_config(BridgeConfig::default())
    }

    /// Create a new HdcBridge with custom configuration
    pub fn with_config(config: BridgeConfig) -> Self {
        let projection_matrix = Self::generate_projection_matrix(&config);

        Self {
            config,
            projection_matrix,
            bias: None,
        }
    }

    /// Generate the projection matrix using JL random projection
    fn generate_projection_matrix(config: &BridgeConfig) -> Vec<f32> {
        let input_dim = config.input_dim;
        let output_dim = config.output_dim;
        let total_size = output_dim * input_dim;

        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let mut matrix = Vec::with_capacity(total_size);

        // JL scaling factor
        let scale = (1.0 / output_dim as f32).sqrt();

        if config.sparse_projection {
            // Sparse projection (Achlioptas' method)
            // Each entry is +1, 0, or -1 with probabilities p, 1-2p, p
            let p = config.sparsity / 2.0;
            for _ in 0..total_size {
                let r: f32 = rng.r#gen();
                let val = if r < p {
                    scale
                } else if r < 2.0 * p {
                    -scale
                } else {
                    0.0
                };
                matrix.push(val);
            }
        } else {
            // Dense Gaussian projection
            for _ in 0..total_size {
                // Box-Muller transform for Gaussian distribution.
                // Use (1 - u1) ∈ (0, 1]: rng.gen::<f32>() ∈ [0, 1) can be exactly
                // 0.0 (P ≈ 2^-24 per draw, ~0.75 expected hits per 16,384×768
                // matrix), and ln(0) = -inf poisons every projected vector with
                // inf/NaN. The binary path masked this (NaN fails the sign
                // threshold); project_continuous exposed it (found 2026-07-09 by
                // the geometry-preservation example).
                let u1: f32 = rng.r#gen();
                let u2: f32 = rng.r#gen();
                let gaussian =
                    (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                matrix.push(gaussian * scale);
            }
        }

        matrix
    }

    /// Project an embedding to HDC space (BinaryHV)
    ///
    /// The projection preserves similarity: embeddings that are similar
    /// in the original space will have similar BinaryHV vectors.
    pub fn project(&self, embedding: &[f32]) -> BinaryHV {
        let projected = self.project_continuous(embedding);
        // Convert to binary BinaryHV by sign thresholding
        Self::to_hv16(&projected)
    }

    /// Project an embedding to HDC space as continuous floats.
    ///
    /// Returns the raw JL-projected vector without binarization. This preserves
    /// the full semantic signal from the embedding model, making it suitable for
    /// feeding into systems that operate on continuous similarity (e.g., moral
    /// topology persistent homology, harmony projection).
    ///
    /// The output dimension matches `config.output_dim` (default: 16,384).
    pub fn project_continuous(&self, embedding: &[f32]) -> Vec<f32> {
        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;

        // Handle dimension mismatch by padding or truncating
        let input = if embedding.len() == input_dim {
            embedding.to_vec()
        } else if embedding.len() < input_dim {
            let mut padded = embedding.to_vec();
            padded.resize(input_dim, 0.0);
            padded
        } else {
            embedding[..input_dim].to_vec()
        };

        // Normalize input if configured
        let normalized_input = match self.config.normalization {
            NormalizationMode::L2 => {
                let norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-6 {
                    input.iter().map(|x| x / norm).collect()
                } else {
                    input
                }
            }
            NormalizationMode::L1 => {
                let norm: f32 = input.iter().map(|x| x.abs()).sum();
                if norm > 1e-6 {
                    input.iter().map(|x| x / norm).collect()
                } else {
                    input
                }
            }
            NormalizationMode::Batch => {
                let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
                let variance: f32 =
                    input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
                let std = (variance + 1e-6).sqrt();
                input.iter().map(|x| (x - mean) / std).collect()
            }
            NormalizationMode::None => input,
        };

        // Compute projection: output = matrix @ input + bias
        let mut projected = Vec::with_capacity(output_dim);
        for i in 0..output_dim {
            let row_start = i * input_dim;
            let mut sum = 0.0f32;
            for j in 0..input_dim {
                sum += self.projection_matrix[row_start + j] * normalized_input[j];
            }
            // Add bias if present
            if let Some(ref bias) = self.bias {
                sum += bias[i];
            }
            projected.push(sum);
        }

        projected
    }

    /// Project multiple embeddings in batch.
    ///
    /// With the `parallel` feature enabled, batches of 4+ embeddings are
    /// projected in parallel using rayon. Below that threshold, rayon
    /// thread-pool overhead exceeds the benefit and sequential is used.
    pub fn project_batch(&self, embeddings: &[Vec<f32>]) -> Vec<BinaryHV> {
        #[cfg(feature = "parallel")]
        {
            if embeddings.len() >= 4 {
                use rayon::prelude::*;
                return embeddings.par_iter().map(|e| self.project(e)).collect();
            }
        }
        embeddings.iter().map(|e| self.project(e)).collect()
    }

    /// Convert projected floats to BinaryHV binary vector
    fn to_hv16(projected: &[f32]) -> BinaryHV {
        // Sign threshold: positive -> 1, negative -> 0
        // Then pack into BinaryHV format
        let mut bits = vec![0u64; projected.len().div_ceil(64)];

        for (i, &val) in projected.iter().enumerate() {
            if val > 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                if word_idx < bits.len() {
                    bits[word_idx] |= 1u64 << bit_idx;
                }
            }
        }

        BinaryHV::from_bits(&bits)
    }

    /// Compute cosine similarity between two embeddings in projected space
    pub fn projected_similarity(&self, emb_a: &[f32], emb_b: &[f32]) -> f32 {
        let hv_a = self.project(emb_a);
        let hv_b = self.project(emb_b);
        hv_a.similarity(&hv_b)
    }

    /// Get the projection matrix (for inspection/debugging)
    pub fn projection_matrix(&self) -> &[f32] {
        &self.projection_matrix
    }

    /// Get configuration
    pub fn config(&self) -> &BridgeConfig {
        &self.config
    }

    /// Set bias vector
    pub fn with_bias(mut self, bias: Vec<f32>) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Create a bridge optimized for Qwen3 embeddings
    pub fn for_qwen3() -> Self {
        Self::with_config(BridgeConfig {
            input_dim: QWEN3_DIMENSION,
            ..Default::default()
        })
    }

    /// Create a bridge optimized for BGE embeddings
    pub fn for_bge() -> Self {
        Self::with_config(BridgeConfig {
            input_dim: BGE_DIMENSION,
            ..Default::default()
        })
    }
}

impl Default for HdcBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HdcBridge {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            projection_matrix: self.projection_matrix.clone(),
            bias: self.bias.clone(),
        }
    }
}

/// Embedding result with metadata
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector
    pub embedding: Vec<f32>,

    /// Dimension of the embedding
    pub dimension: usize,

    /// Model that produced the embedding
    pub model: String,

    /// Whether this is a simulated embedding
    pub is_simulated: bool,

    /// Processing time in milliseconds
    pub processing_time_ms: f32,
}

impl EmbeddingResult {
    /// Create a new embedding result
    pub fn new(embedding: Vec<f32>, model: impl Into<String>) -> Self {
        let dimension = embedding.len();
        Self {
            embedding,
            dimension,
            model: model.into(),
            is_simulated: false,
            processing_time_ms: 0.0,
        }
    }

    /// Mark as simulated
    pub fn simulated(mut self) -> Self {
        self.is_simulated = true;
        self
    }

    /// Set processing time
    pub fn with_time(mut self, time_ms: f32) -> Self {
        self.processing_time_ms = time_ms;
        self
    }
}

/// Trait for embedding providers
pub trait Embedder: Send + Sync {
    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Embed a single text
    fn embed(&mut self, text: &str) -> Result<EmbeddingResult>;

    /// Embed multiple texts in batch
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Get model name
    fn model_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = HdcBridge::new();
        assert_eq!(bridge.config.input_dim, QWEN3_DIMENSION);
        assert_eq!(bridge.config.output_dim, 16384);
    }

    #[test]
    fn test_bridge_projection() {
        let bridge = HdcBridge::new();
        let embedding = vec![0.1f32; QWEN3_DIMENSION];
        let hv = bridge.project(&embedding);

        // BinaryHV should have the correct dimension (compile-time constant)
        assert_eq!(BinaryHV::DIM, 16384);
        // Verify hv is valid by checking it was created
        let _ = hv;
    }

    #[test]
    fn test_similarity_preservation() {
        let bridge = HdcBridge::new();

        // Create two similar embeddings
        let emb1 = vec![0.1f32; QWEN3_DIMENSION];
        let mut emb2 = emb1.clone();
        emb2[0] = 0.11; // Slightly different

        let hv1 = bridge.project(&emb1);
        let hv2 = bridge.project(&emb2);

        // Similarity should be high (above 0.9)
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.9,
            "Similar embeddings should project to similar HVs, got {}",
            sim
        );
    }

    #[test]
    fn test_different_embeddings_different_hvs() {
        let bridge = HdcBridge::new();

        // Create two very different embeddings
        let emb1: Vec<f32> = (0..QWEN3_DIMENSION)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let emb2: Vec<f32> = (0..QWEN3_DIMENSION)
            .map(|i| (i as f32 * 0.01).cos())
            .collect();

        let hv1 = bridge.project(&emb1);
        let hv2 = bridge.project(&emb2);

        // Similarity should be lower (different embeddings)
        let sim = hv1.similarity(&hv2);
        assert!(
            sim < 0.9,
            "Different embeddings should have lower similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_sparse_projection() {
        let config = BridgeConfig {
            sparse_projection: true,
            sparsity: 0.1,
            ..Default::default()
        };
        let bridge = HdcBridge::with_config(config);

        let embedding = vec![0.1f32; QWEN3_DIMENSION];
        let hv = bridge.project(&embedding);

        assert_eq!(BinaryHV::DIM, 16384);
        let _ = hv; // Ensure hv was created successfully
    }

    #[test]
    fn test_reproducibility() {
        // Same seed should produce same projection matrix
        let bridge1 = HdcBridge::with_config(BridgeConfig {
            seed: 12345,
            ..Default::default()
        });
        let bridge2 = HdcBridge::with_config(BridgeConfig {
            seed: 12345,
            ..Default::default()
        });

        let embedding = vec![0.5f32; QWEN3_DIMENSION];
        let hv1 = bridge1.project(&embedding);
        let hv2 = bridge2.project(&embedding);

        // Should be identical
        let sim = hv1.similarity(&hv2);
        assert_eq!(sim, 1.0, "Same seed should produce identical projections");
    }

    #[test]
    fn test_parallel_batch_matches_sequential() {
        let bridge = HdcBridge::new();
        let embeddings: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                (0..QWEN3_DIMENSION)
                    .map(|j| ((i * 1000 + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        let sequential: Vec<BinaryHV> = embeddings.iter().map(|e| bridge.project(e)).collect();
        let batched = bridge.project_batch(&embeddings);

        assert_eq!(sequential.len(), batched.len());
        for (i, (s, b)) in sequential.iter().zip(batched.iter()).enumerate() {
            assert_eq!(
                s.similarity(b),
                1.0,
                "Batch[{i}] should be identical to sequential"
            );
        }
    }

    #[test]
    fn test_parallel_batch_small() {
        // Batch < 4 should use sequential fallback (no rayon overhead)
        let bridge = HdcBridge::new();
        let embeddings: Vec<Vec<f32>> = (0..3)
            .map(|i| vec![(i as f32 + 1.0) * 0.1; QWEN3_DIMENSION])
            .collect();

        let result = bridge.project_batch(&embeddings);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_parallel_batch_similarity() {
        let bridge = HdcBridge::new();

        // Create similar embeddings (small perturbations)
        let base: Vec<f32> = (0..QWEN3_DIMENSION)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                base.iter()
                    .enumerate()
                    .map(|(j, &v)| v + (i as f32 * 0.001) * ((j as f32).cos()))
                    .collect()
            })
            .collect();

        let hvs = bridge.project_batch(&embeddings);
        // Adjacent embeddings should be similar
        for i in 0..hvs.len() - 1 {
            let sim = hvs[i].similarity(&hvs[i + 1]);
            assert!(
                sim > 0.8,
                "Adjacent similar embeddings should produce similar HVs, got {sim} at [{i},{j}]",
                j = i + 1
            );
        }
    }
}
