// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HyperFeel Encoder Implementation
//!
//! Core gradient compression using hyperdimensional computing.

use super::types::{EncodingConfig, HyperGradient, SimilarityResult, TemporalEntry, HV16_BYTES};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::time::Instant;

/// HyperFeel Encoder v2.0
///
/// Encodes gradients to 2KB hypervectors while preserving semantic similarity.
///
/// # Algorithm
///
/// 1. Normalize gradient to unit length
/// 2. Quantize to 8-bit integers
/// 3. Random projection to hypervector dimension
/// 4. Binarize using sign threshold
/// 5. Pack bits to bytes
///
/// # Mathematical Guarantee
///
/// Johnson-Lindenstrauss lemma ensures that for any ε > 0 and any set of n points
/// in R^d, there exists a map f: R^d → R^k where k = O(log(n)/ε²) such that
/// for all u,v: (1-ε)||u-v||² ≤ ||f(u)-f(v)||² ≤ (1+ε)||u-v||²
pub struct HyperFeelEncoder {
    config: EncodingConfig,
    /// Cached projection matrices by input dimension
    projection_cache: HashMap<usize, Vec<f32>>,
    /// Temporal history per node
    temporal_history: HashMap<String, Vec<TemporalEntry>>,
    /// PRNG state for reproducibility
    rng_state: u64,
}

impl HyperFeelEncoder {
    /// Create new encoder with configuration
    pub fn new(config: EncodingConfig) -> Self {
        config.validate().expect("Invalid EncodingConfig");

        Self {
            rng_state: config.projection_seed,
            config,
            projection_cache: HashMap::new(),
            temporal_history: HashMap::new(),
        }
    }

    /// Encode gradient to HyperGradient
    ///
    /// # Arguments
    ///
    /// * `gradient` - Flattened gradient array
    /// * `round_num` - Federation round number
    /// * `node_id` - Node identifier
    ///
    /// # Returns
    ///
    /// Compressed HyperGradient with 2KB hypervector
    pub fn encode_gradient(
        &mut self,
        gradient: &[f32],
        round_num: u32,
        node_id: &str,
    ) -> HyperGradient {
        let start = Instant::now();

        // Record original size
        let original_size = gradient.len() * 4; // f32 = 4 bytes

        // 1. Compute gradient hash
        let gradient_hash = self.hash_gradient(gradient);

        // 2. Compute quality score (L2 norm)
        let quality_score = self.compute_l2_norm(gradient);

        // 3. Normalize gradient
        let normalized = self.normalize(gradient);

        // 4. Quantize to 8-bit
        let quantized = self.quantize(&normalized);

        // 5. Random projection
        let projected = self.project(&quantized);

        // 6. Binarize and pack
        let hypervector = self.binarize_and_pack(&projected);

        // 7. Update temporal history if enabled
        if self.config.use_temporal {
            self.update_temporal_history(node_id, round_num, &hypervector);
        }

        let encode_time_ms = start.elapsed().as_secs_f32() * 1000.0;

        HyperGradient::new(
            node_id.to_string(),
            round_num,
            hypervector,
            quality_score,
            original_size,
            encode_time_ms,
            gradient_hash,
        )
    }

    /// Hash gradient using SHA3-256
    fn hash_gradient(&self, gradient: &[f32]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        for val in gradient {
            hasher.update(val.to_le_bytes());
        }
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Compute L2 norm of gradient
    fn compute_l2_norm(&self, gradient: &[f32]) -> f32 {
        gradient.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize gradient to unit length
    fn normalize(&self, gradient: &[f32]) -> Vec<f32> {
        let norm = self.compute_l2_norm(gradient);
        if norm > 1e-10 {
            gradient.iter().map(|x| x / norm).collect()
        } else {
            gradient.to_vec()
        }
    }

    /// Quantize to 8-bit integers
    fn quantize(&self, gradient: &[f32]) -> Vec<i8> {
        let max_val = (1i16 << (self.config.quantize_bits - 1)) - 1;
        let max_f = max_val as f32;

        gradient
            .iter()
            .map(|x| (x * max_f).clamp(-max_f, max_f) as i8)
            .collect()
    }

    /// Get or create projection matrix
    fn get_projection_matrix(&mut self, input_dim: usize) -> &[f32] {
        let output_dim = self.config.dimension / 8;

        if !self.projection_cache.contains_key(&input_dim) {
            // Generate Gaussian random projection matrix
            let size = input_dim * output_dim;
            let mut matrix = Vec::with_capacity(size);

            // Simple LCG PRNG for reproducibility
            let mut state = self.rng_state;
            let scale = 1.0 / (output_dim as f32).sqrt();

            for _ in 0..size {
                // Box-Muller transform for Gaussian
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u1 = (state >> 33) as f32 / (1u64 << 31) as f32;
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u2 = (state >> 33) as f32 / (1u64 << 31) as f32;

                let z =
                    (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                matrix.push(z * scale);
            }

            self.projection_cache.insert(input_dim, matrix);
        }

        // Safety: we just inserted into the cache above if it didn't exist
        self.projection_cache
            .get(&input_dim)
            .expect("projection matrix was just inserted")
    }

    /// Project quantized gradient to hypervector dimension
    fn project(&mut self, quantized: &[i8]) -> Vec<f32> {
        let input_dim = quantized.len();
        let output_dim = self.config.dimension / 8;

        let proj_matrix = self.get_projection_matrix(input_dim);

        let mut result = vec![0.0f32; output_dim];

        // Matrix-vector multiply: result = proj_matrix.T @ quantized
        for (j, out_val) in result.iter_mut().enumerate() {
            for (i, &in_val) in quantized.iter().enumerate() {
                *out_val += proj_matrix[i * output_dim + j] * (in_val as f32);
            }
        }

        result
    }

    /// Binarize and pack to bytes
    fn binarize_and_pack(&self, projected: &[f32]) -> Vec<u8> {
        // Binarize: sign(x) -> 0 or 1
        let binary: Vec<u8> = projected
            .iter()
            .map(|&x| if x > 0.0 { 1 } else { 0 })
            .collect();

        // Pack bits to bytes
        let mut packed = Vec::with_capacity(HV16_BYTES);
        for chunk in binary.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= bit << (7 - i);
            }
            packed.push(byte);
        }

        // Ensure exact size
        packed.resize(HV16_BYTES, 0);
        packed
    }

    /// Update temporal history for a node
    fn update_temporal_history(&mut self, node_id: &str, round: u32, hypervector: &[u8]) {
        let entry = TemporalEntry {
            round,
            hypervector: hypervector.to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let history = self
            .temporal_history
            .entry(node_id.to_string())
            .or_default();
        history.push(entry);

        // Keep only last 50 entries
        if history.len() > 50 {
            history.drain(0..history.len() - 50);
        }
    }

    /// Compute cosine similarity between two hypervectors
    ///
    /// For binary hypervectors, this relates to Hamming similarity.
    pub fn cosine_similarity(hv1: &[u8], hv2: &[u8]) -> f32 {
        if hv1.len() != hv2.len() || hv1.is_empty() {
            return 0.0;
        }

        // Unpack to bipolar (-1, +1) representation
        let mut dot_product = 0i32;
        let total_bits = hv1.len() * 8;

        for (b1, b2) in hv1.iter().zip(hv2.iter()) {
            for i in 0..8 {
                let bit1 = (b1 >> (7 - i)) & 1;
                let bit2 = (b2 >> (7 - i)) & 1;

                // Convert 0/1 to -1/+1 and multiply
                let val1 = if bit1 == 1 { 1 } else { -1 };
                let val2 = if bit2 == 1 { 1 } else { -1 };
                dot_product += val1 * val2;
            }
        }

        // Cosine similarity for bipolar vectors
        dot_product as f32 / total_bits as f32
    }

    /// Compute Hamming distance between two hypervectors
    pub fn hamming_distance(hv1: &[u8], hv2: &[u8]) -> usize {
        hv1.iter()
            .zip(hv2.iter())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    /// Compute detailed similarity metrics
    pub fn compute_similarity(hv1: &[u8], hv2: &[u8]) -> SimilarityResult {
        let cosine = Self::cosine_similarity(hv1, hv2);
        let hamming = Self::hamming_distance(hv1, hv2);
        let total_bits = hv1.len() * 8;

        SimilarityResult::new(cosine, hamming, total_bits)
    }

    /// Decode hypervector back to approximate gradient
    ///
    /// Note: This is lossy due to random projection.
    /// Use for aggregation, not exact reconstruction.
    pub fn decode_hypervector(&mut self, hypervector: &[u8], output_dim: usize) -> Vec<f32> {
        // Unpack bytes to bits
        let mut bits = Vec::with_capacity(hypervector.len() * 8);
        for byte in hypervector {
            for i in 0..8 {
                let bit = (byte >> (7 - i)) & 1;
                bits.push(if bit == 1 { 1.0f32 } else { -1.0f32 });
            }
        }

        // Get projection matrix (will create if needed)
        let proj_dim = self.config.dimension / 8;
        let proj_matrix = self.get_projection_matrix(output_dim);

        // Pseudo-inverse: multiply by transpose
        let mut result = vec![0.0f32; output_dim];
        for i in 0..output_dim {
            for j in 0..proj_dim.min(bits.len()) {
                result[i] += proj_matrix[i * proj_dim + j] * bits[j];
            }
        }

        // Normalize output
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut result {
                *x /= norm;
            }
        }

        result
    }

    /// Get temporal trajectory for a node
    pub fn get_temporal_trajectory(&self, node_id: &str) -> Option<&Vec<TemporalEntry>> {
        self.temporal_history.get(node_id)
    }

    /// Compute trajectory smoothness (consistency over time)
    ///
    /// Returns 0.0-1.0 where 1.0 = perfectly smooth trajectory
    pub fn compute_trajectory_smoothness(&self, node_id: &str) -> f32 {
        let history = match self.temporal_history.get(node_id) {
            Some(h) if h.len() >= 2 => h,
            _ => return 1.0, // No history = assumed smooth
        };

        // Compute consecutive similarities
        let mut similarities = Vec::new();
        for window in history.windows(2) {
            let sim = Self::cosine_similarity(&window[0].hypervector, &window[1].hypervector);
            similarities.push(sim);
        }

        if similarities.is_empty() {
            return 1.0;
        }

        // Smoothness = 1 - std_dev(similarities)
        let mean: f32 = similarities.iter().sum::<f32>() / similarities.len() as f32;
        let variance: f32 = similarities.iter().map(|s| (s - mean).powi(2)).sum::<f32>()
            / similarities.len() as f32;
        let std_dev = variance.sqrt();

        (1.0 - std_dev).clamp(0.0, 1.0)
    }

    /// Detect anomalous gradients based on trajectory
    ///
    /// Returns Some(anomaly_type) if anomaly detected, None otherwise
    pub fn detect_trajectory_anomaly(&self, node_id: &str) -> Option<&'static str> {
        let history = match self.temporal_history.get(node_id) {
            Some(h) if h.len() >= 2 => h,
            _ => return None,
        };

        // Compare latest with average of recent history
        let latest = match history.last() {
            Some(entry) => &entry.hypervector,
            None => return None,
        };
        let recent: Vec<_> = history.iter().rev().skip(1).take(5).collect();

        if recent.is_empty() {
            return None;
        }

        // Compute average similarity to recent history
        let avg_sim: f32 = recent
            .iter()
            .map(|entry| Self::cosine_similarity(latest, &entry.hypervector))
            .sum::<f32>()
            / recent.len() as f32;

        if avg_sim < 0.3 {
            Some("sudden_change")
        } else if avg_sim > 0.99 {
            Some("stale_gradient")
        } else {
            None
        }
    }

    /// Aggregate multiple hypergradients
    ///
    /// Uses element-wise majority voting for binary hypervectors.
    pub fn aggregate_hypergradients(gradients: &[&HyperGradient]) -> Vec<u8> {
        if gradients.is_empty() {
            return vec![0u8; HV16_BYTES];
        }
        if gradients.len() == 1 {
            return gradients[0].hypervector.clone();
        }

        let n = gradients.len();
        let threshold = n / 2;

        let mut result = vec![0u8; HV16_BYTES];

        for (byte_idx, res) in result.iter_mut().enumerate() {
            let mut result_byte = 0u8;
            for bit_idx in 0..8 {
                let mask = 1u8 << (7 - bit_idx);
                let ones: usize = gradients
                    .iter()
                    .map(|g| {
                        if g.hypervector.get(byte_idx).is_some_and(|&b| b & mask != 0) {
                            1
                        } else {
                            0
                        }
                    })
                    .sum();

                if ones > threshold {
                    result_byte |= mask;
                }
            }
            *res = result_byte;
        }

        result
    }

    /// Clear projection cache (for memory management)
    pub fn clear_cache(&mut self) {
        self.projection_cache.clear();
    }

    /// Clear temporal history for a node
    pub fn clear_temporal_history(&mut self, node_id: &str) {
        self.temporal_history.remove(node_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let encoder = HyperFeelEncoder::new(EncodingConfig::default());
        let gradient = vec![3.0, 4.0]; // norm = 5
        let normalized = encoder.normalize(&gradient);

        assert!((normalized[0] - 0.6).abs() < 1e-5);
        assert!((normalized[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_trajectory_tracking() {
        let config = EncodingConfig::default();
        let mut encoder = HyperFeelEncoder::new(config);

        // Encode multiple gradients from same node
        for i in 0..5 {
            let gradient: Vec<f32> = (0..1000).map(|x| (x as f32 + i as f32).sin()).collect();
            encoder.encode_gradient(&gradient, i, "node-1");
        }

        // Check temporal history
        assert!(encoder.get_temporal_trajectory("node-1").is_some());
        let history = encoder.get_temporal_trajectory("node-1").unwrap();
        assert_eq!(history.len(), 5);

        // Check smoothness
        let smoothness = encoder.compute_trajectory_smoothness("node-1");
        assert!(
            smoothness > 0.5,
            "Trajectory should be smooth: {}",
            smoothness
        );
    }

    #[test]
    fn test_aggregation() {
        let config = EncodingConfig::default();
        let mut encoder = HyperFeelEncoder::new(config);

        // Create similar gradients
        let g1: Vec<f32> = (0..1000).map(|x| (x as f32).sin()).collect();
        let g2: Vec<f32> = (0..1000).map(|x| (x as f32).sin() + 0.01).collect();
        let g3: Vec<f32> = (0..1000).map(|x| (x as f32).sin() - 0.01).collect();

        let hg1 = encoder.encode_gradient(&g1, 1, "node-1");
        let hg2 = encoder.encode_gradient(&g2, 1, "node-2");
        let hg3 = encoder.encode_gradient(&g3, 1, "node-3");

        let aggregated = HyperFeelEncoder::aggregate_hypergradients(&[&hg1, &hg2, &hg3]);

        // Aggregated should be similar to inputs
        let sim1 = HyperFeelEncoder::cosine_similarity(&aggregated, &hg1.hypervector);
        assert!(
            sim1 > 0.7,
            "Aggregated should be similar to inputs: {}",
            sim1
        );
    }
}
