//! # HDC-Embedding Bridge
//!
//! Projects dense embeddings (768D) to binary hypervectors (HV16, 2048-bit).
//!
//! ## Mathematical Foundation
//!
//! The Johnson-Lindenstrauss lemma guarantees that random projections
//! preserve distances in high-dimensional spaces:
//!
//! ```text
//! For any ε ∈ (0, 1) and n points in R^d, there exists a linear map
//! f: R^d → R^k with k = O(log(n)/ε²) such that for all pairs u, v:
//!
//!     (1-ε)||u-v||² ≤ ||f(u)-f(v)||² ≤ (1+ε)||u-v||²
//! ```
//!
//! We use this to project 768D BGE embeddings to 2048-bit HV16 vectors
//! while preserving semantic similarity relationships.
//!
//! ## Projection Methods
//!
//! 1. **Random Projection + Sign**: R × embedding → binary (sign function)
//! 2. **Locality Sensitive Hashing**: Multiple hash tables for similarity
//! 3. **Learned Projection**: Train projection for specific domain

use crate::hdc::binary_hv::HV16;
use crate::embeddings::bge::{BGEEmbedder, BGE_DIMENSION};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for HDC-Embedding bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Input dimension (BGE = 768)
    pub input_dim: usize,

    /// Output dimension (HV16 = 2048 bits)
    pub output_dim: usize,

    /// Random seed for projection matrix
    pub seed: u64,

    /// Projection method
    pub method: ProjectionMethod,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            input_dim: BGE_DIMENSION,
            output_dim: HV16::DIM,
            seed: 0x5974_1AEA,  // "SYMTHAEA" as hex
            method: ProjectionMethod::RandomSign,
        }
    }
}

/// Projection method for embedding to HDC conversion
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProjectionMethod {
    /// Random projection matrix + sign function
    RandomSign,

    /// Locality-sensitive hashing (SimHash)
    SimHash,

    /// Learned projection (requires training)
    Learned,
}

/// Bridge between dense embeddings and HDC space
#[derive(Debug)]
pub struct HdcBridge {
    /// Configuration
    config: BridgeConfig,

    /// Random projection matrix (2048 x 768)
    projection_matrix: Vec<Vec<f32>>,

    /// Statistics
    stats: BridgeStats,
}

/// Statistics for the bridge
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BridgeStats {
    /// Total projections
    pub total_projections: u64,

    /// Average projection time (μs)
    pub avg_projection_time_us: f64,
}

impl HdcBridge {
    /// Create a new HDC bridge with default configuration
    pub fn new() -> Self {
        Self::with_config(BridgeConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BridgeConfig) -> Self {
        let projection_matrix = Self::generate_projection_matrix(
            config.output_dim,
            config.input_dim,
            config.seed,
        );

        Self {
            config,
            projection_matrix,
            stats: BridgeStats::default(),
        }
    }

    /// Project dense embedding to HV16
    ///
    /// Uses random projection + sign function.
    pub fn project(&self, embedding: &[f32]) -> HV16 {
        match self.config.method {
            ProjectionMethod::RandomSign => self.random_sign_project(embedding),
            ProjectionMethod::SimHash => self.simhash_project(embedding),
            ProjectionMethod::Learned => self.random_sign_project(embedding), // Fallback
        }
    }

    /// Project text via embedder to HV16
    pub fn embed_to_hdc(&self, embedder: &BGEEmbedder, text: &str) -> Result<HV16> {
        let emb = embedder.embed(text)?;
        Ok(self.project(&emb.embedding))
    }

    /// Random sign projection
    ///
    /// 1. Multiply: y = R × x (where R is random Gaussian matrix)
    /// 2. Binarize: b = sign(y)
    fn random_sign_project(&self, embedding: &[f32]) -> HV16 {
        let mut projected = vec![0.0f32; self.config.output_dim];

        // Matrix-vector multiplication
        for (i, row) in self.projection_matrix.iter().enumerate() {
            let mut sum = 0.0f32;
            for (j, &val) in embedding.iter().enumerate() {
                if j < row.len() {
                    sum += row[j] * val;
                }
            }
            projected[i] = sum;
        }

        // Binarize using sign function
        self.binarize(&projected)
    }

    /// SimHash projection (locality-sensitive)
    fn simhash_project(&self, embedding: &[f32]) -> HV16 {
        // SimHash: for each bit position, compute weighted sum
        // Bit is 1 if sum > 0, else 0
        self.random_sign_project(embedding) // Same as random sign for now
    }

    /// Convert float vector to HV16 using sign function
    fn binarize(&self, projected: &[f32]) -> HV16 {
        let mut bytes = [0u8; HV16::BYTES];

        for (bit_idx, &val) in projected.iter().take(HV16::DIM).enumerate() {
            if val > 0.0 {
                let byte_idx = bit_idx / 8;
                let bit_pos = bit_idx % 8;
                bytes[byte_idx] |= 1 << bit_pos;
            }
        }

        HV16(bytes)
    }

    /// Generate random projection matrix
    ///
    /// Each element is drawn from N(0, 1/sqrt(k)) where k is output dimension.
    fn generate_projection_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f32>> {
        let scale = 1.0 / (rows as f32).sqrt();
        let mut matrix = Vec::with_capacity(rows);

        let mut state = seed;
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                // Simple LCG for deterministic pseudo-random values
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u1 = ((state >> 33) as f32) / (u32::MAX as f32);

                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = ((state >> 33) as f32) / (u32::MAX as f32);

                // Box-Muller transform for Gaussian
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                row.push(z * scale);
            }
            matrix.push(row);
        }

        matrix
    }

    /// Compute expected similarity preservation
    ///
    /// Given cosine similarity in embedding space, estimate similarity in HV16 space.
    pub fn expected_hdc_similarity(&self, embedding_similarity: f32) -> f32 {
        // For random projections with sign function:
        // P(sign(r·a) = sign(r·b)) = 1 - arccos(sim(a,b)) / π
        // HV16 similarity ≈ 1 - hamming_distance / dim
        1.0 - (embedding_similarity.acos() / std::f32::consts::PI)
    }

    /// Get bridge statistics
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }
}

impl Default for HdcBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to embed text directly to HV16
pub fn text_to_hv16(embedder: &BGEEmbedder, bridge: &HdcBridge, text: &str) -> Result<HV16> {
    bridge.embed_to_hdc(embedder, text)
}

/// Compute semantic similarity via embedding + projection
pub fn semantic_similarity_hdc(
    embedder: &BGEEmbedder,
    bridge: &HdcBridge,
    text_a: &str,
    text_b: &str,
) -> Result<f32> {
    let hv_a = text_to_hv16(embedder, bridge, text_a)?;
    let hv_b = text_to_hv16(embedder, bridge, text_b)?;
    Ok(hv_a.similarity(&hv_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = HdcBridge::new();
        assert_eq!(bridge.config.input_dim, BGE_DIMENSION);
        assert_eq!(bridge.config.output_dim, HV16::DIM);
    }

    #[test]
    fn test_projection_matrix_size() {
        let bridge = HdcBridge::new();
        assert_eq!(bridge.projection_matrix.len(), HV16::DIM);
        assert_eq!(bridge.projection_matrix[0].len(), BGE_DIMENSION);
    }

    #[test]
    fn test_project() {
        let bridge = HdcBridge::new();
        let embedding = vec![0.1f32; BGE_DIMENSION];

        let hv = bridge.project(&embedding);
        // Should produce valid HV16
        assert_eq!(hv.0.len(), HV16::BYTES);
    }

    #[test]
    fn test_deterministic() {
        let bridge = HdcBridge::new();
        let embedding = vec![0.5f32; BGE_DIMENSION];

        let hv1 = bridge.project(&embedding);
        let hv2 = bridge.project(&embedding);

        // Same input should give same output
        assert_eq!(hv1.0, hv2.0);
    }

    #[test]
    fn test_similarity_preservation() {
        let bridge = HdcBridge::new();

        // Similar embeddings should produce similar HV16s
        let emb1: Vec<f32> = (0..BGE_DIMENSION).map(|i| (i as f32) / 1000.0).collect();
        let emb2: Vec<f32> = (0..BGE_DIMENSION).map(|i| (i as f32) / 1000.0 + 0.01).collect();
        let emb3: Vec<f32> = (0..BGE_DIMENSION).map(|i| -(i as f32) / 1000.0).collect();

        let hv1 = bridge.project(&emb1);
        let hv2 = bridge.project(&emb2);
        let hv3 = bridge.project(&emb3);

        let sim_12 = hv1.similarity(&hv2);
        let sim_13 = hv1.similarity(&hv3);

        // emb1 and emb2 are similar, should have higher HV similarity
        // emb1 and emb3 are opposite, should have lower HV similarity
        assert!(sim_12 > sim_13);
    }

    #[test]
    fn test_expected_similarity() {
        let bridge = HdcBridge::new();

        // Perfect similarity should map to ~1.0
        let expected = bridge.expected_hdc_similarity(1.0);
        assert!((expected - 1.0).abs() < 0.01);

        // Zero similarity should map to ~0.5 (random chance)
        let expected_zero = bridge.expected_hdc_similarity(0.0);
        assert!((expected_zero - 0.5).abs() < 0.1);

        // Negative similarity should map to <0.5
        let expected_neg = bridge.expected_hdc_similarity(-1.0);
        assert!(expected_neg < 0.1);
    }
}
