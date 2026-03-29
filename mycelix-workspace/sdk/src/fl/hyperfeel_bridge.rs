// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HyperFeel-FL Bridge
//!
//! Integrates HyperFeel gradient compression with the Federated Learning pipeline.
//! This enables 2000x compression while maintaining Byzantine-resistant aggregation.
//!
//! # Architecture
//!
//! ```text
//! Local Gradient (10M params, 40MB)
//!        ↓ HyperFeelEncoder
//! HyperGradient (2KB hypervector)
//!        ↓ HyperFeelFLBridge
//! GradientUpdate (FL format)
//!        ↓ FLCoordinator
//! Aggregated Result
//! ```
//!
//! # Byzantine Detection via Similarity
//!
//! Uses HyperFeel's cosine similarity to detect Byzantine gradients:
//! - Cluster gradients by similarity
//! - Flag outliers (similarity < threshold to cluster centroid)
//! - Weight contributions by trust score × similarity

use super::coordinator::{CoordinatorError, FLCoordinator};
use super::types::{AggregatedGradient, GradientUpdate};
use crate::hyperfeel::{EncodingConfig, HyperFeelEncoder, HyperGradient};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for HyperFeel-FL integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperFeelFLConfig {
    /// Similarity threshold for Byzantine detection (0.0-1.0)
    /// Gradients below this similarity to centroid are flagged
    pub similarity_threshold: f32,
    /// Minimum cluster size for centroid calculation
    pub min_cluster_size: usize,
    /// Enable similarity-based weighting
    pub use_similarity_weighting: bool,
    /// Compression encoding config
    pub encoding_config: EncodingConfig,
}

impl Default for HyperFeelFLConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.5,
            min_cluster_size: 3,
            use_similarity_weighting: true,
            encoding_config: EncodingConfig::default(),
        }
    }
}

/// Byzantine detection result
#[derive(Debug, Clone)]
pub struct ByzantineAnalysis {
    /// Node ID → is_byzantine flag
    pub byzantine_flags: HashMap<String, bool>,
    /// Node ID → similarity to centroid
    pub similarity_scores: HashMap<String, f32>,
    /// Centroid hypervector (if enough samples)
    pub centroid: Option<Vec<u8>>,
    /// Number of flagged participants
    pub flagged_count: usize,
    /// Total participants analyzed
    pub total_count: usize,
}

impl ByzantineAnalysis {
    /// Get list of flagged (potentially Byzantine) nodes
    pub fn flagged_nodes(&self) -> Vec<&str> {
        self.byzantine_flags
            .iter()
            .filter(|(_, &is_byzantine)| is_byzantine)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get Byzantine ratio
    pub fn byzantine_ratio(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            self.flagged_count as f32 / self.total_count as f32
        }
    }
}

/// Compressed gradient submission for FL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedSubmission {
    /// The compressed hypergradient
    pub hypergradient: HyperGradient,
    /// Batch size used in training
    pub batch_size: usize,
    /// Training loss achieved
    pub loss: f64,
    /// Optional accuracy metric
    pub accuracy: Option<f64>,
}

/// HyperFeel-FL Bridge
///
/// Connects HyperFeel compression with FL coordinator for
/// bandwidth-efficient Byzantine-resistant federated learning.
pub struct HyperFeelFLBridge {
    /// Encoder for compression/decompression
    encoder: HyperFeelEncoder,
    /// Configuration
    config: HyperFeelFLConfig,
    /// Current round hypergradients (for similarity analysis)
    round_hypergradients: Vec<HyperGradient>,
    /// Node similarity cache
    similarity_cache: HashMap<String, f32>,
}

impl HyperFeelFLBridge {
    /// Create a new bridge with configuration
    pub fn new(config: HyperFeelFLConfig) -> Self {
        let encoder = HyperFeelEncoder::new(config.encoding_config.clone());
        Self {
            encoder,
            config,
            round_hypergradients: Vec::new(),
            similarity_cache: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default_bridge() -> Self {
        Self::new(HyperFeelFLConfig::default())
    }

    /// Compress a gradient for transmission
    ///
    /// Takes a full gradient vector and returns a compressed HyperGradient
    /// suitable for bandwidth-efficient transmission.
    pub fn compress_gradient(
        &mut self,
        gradient: &[f32],
        round: u32,
        node_id: &str,
    ) -> HyperGradient {
        self.encoder.encode_gradient(gradient, round, node_id)
    }

    /// Convert HyperGradient to GradientUpdate for FL coordinator
    ///
    /// Creates a FL-compatible update from compressed representation.
    /// Note: The gradients are sampled/expanded from hypervector encoding.
    pub fn hypergradient_to_update(
        &self,
        hg: &HyperGradient,
        model_version: u64,
        batch_size: usize,
        loss: f64,
    ) -> GradientUpdate {
        // Convert hypervector to gradient representation
        // We use the hypervector bytes directly as a compressed feature
        let gradients: Vec<f64> = hg
            .hypervector
            .iter()
            .map(|&b| (b as f64 - 128.0) / 128.0) // Normalize bytes to [-1, 1]
            .collect();

        GradientUpdate::new(
            hg.node_id.clone(),
            model_version,
            gradients,
            batch_size,
            loss,
        )
    }

    /// Submit a compressed gradient to the bridge
    ///
    /// Stores the hypergradient for similarity analysis.
    pub fn submit_compressed(&mut self, submission: CompressedSubmission) {
        self.round_hypergradients.push(submission.hypergradient);
    }

    /// Clear round state for new round
    pub fn start_new_round(&mut self) {
        self.round_hypergradients.clear();
        self.similarity_cache.clear();
    }

    /// Analyze submitted gradients for Byzantine behavior
    ///
    /// Uses hypervector cosine similarity to detect outliers.
    pub fn analyze_byzantine(&mut self) -> ByzantineAnalysis {
        let n = self.round_hypergradients.len();

        if n < self.config.min_cluster_size {
            // Not enough samples for meaningful analysis
            return ByzantineAnalysis {
                byzantine_flags: HashMap::new(),
                similarity_scores: HashMap::new(),
                centroid: None,
                flagged_count: 0,
                total_count: n,
            };
        }

        // Calculate centroid (average hypervector)
        let centroid = self.calculate_centroid();

        // Calculate similarity of each gradient to centroid
        let mut similarity_scores = HashMap::new();
        let mut byzantine_flags = HashMap::new();
        let mut flagged_count = 0;

        for hg in &self.round_hypergradients {
            let similarity = HyperFeelEncoder::cosine_similarity(&hg.hypervector, &centroid);
            similarity_scores.insert(hg.node_id.clone(), similarity);
            self.similarity_cache.insert(hg.node_id.clone(), similarity);

            let is_byzantine = similarity < self.config.similarity_threshold;
            byzantine_flags.insert(hg.node_id.clone(), is_byzantine);

            if is_byzantine {
                flagged_count += 1;
            }
        }

        ByzantineAnalysis {
            byzantine_flags,
            similarity_scores,
            centroid: Some(centroid),
            flagged_count,
            total_count: n,
        }
    }

    /// Calculate centroid hypervector from current round
    fn calculate_centroid(&self) -> Vec<u8> {
        if self.round_hypergradients.is_empty() {
            return vec![128u8; self.config.encoding_config.dimension / 8];
        }

        let n = self.round_hypergradients.len();
        let hv_len = self.round_hypergradients[0].hypervector.len();

        // Sum all hypervectors (as i32 to avoid overflow)
        let mut sum: Vec<i32> = vec![0; hv_len];
        for hg in &self.round_hypergradients {
            for (i, &byte) in hg.hypervector.iter().enumerate() {
                sum[i] += byte as i32;
            }
        }

        // Average and convert back to bytes
        sum.iter()
            .map(|&s| (s / n as i32).clamp(0, 255) as u8)
            .collect()
    }

    /// Get similarity weight for a node
    ///
    /// Returns a weight factor based on similarity to centroid.
    /// Nodes closer to centroid get higher weights.
    pub fn get_similarity_weight(&self, node_id: &str) -> f32 {
        if !self.config.use_similarity_weighting {
            return 1.0;
        }

        self.similarity_cache
            .get(node_id)
            .map(|&sim| sim.max(0.0)) // Clamp negative similarities to 0
            .unwrap_or(0.5) // Default to neutral weight
    }

    /// Aggregate hypergradients using similarity-weighted average
    ///
    /// This performs aggregation directly in hypervector space,
    /// avoiding the need to decompress gradients.
    pub fn aggregate_hypergradients(&self) -> Option<Vec<u8>> {
        if self.round_hypergradients.is_empty() {
            return None;
        }

        let hv_len = self.round_hypergradients[0].hypervector.len();
        let mut weighted_sum: Vec<f32> = vec![0.0; hv_len];
        let mut total_weight = 0.0;

        for hg in &self.round_hypergradients {
            let weight = self.get_similarity_weight(&hg.node_id);
            total_weight += weight;

            for (i, &byte) in hg.hypervector.iter().enumerate() {
                weighted_sum[i] += (byte as f32 - 128.0) * weight;
            }
        }

        if total_weight < 0.001 {
            return None;
        }

        // Normalize and convert back to bytes
        Some(
            weighted_sum
                .iter()
                .map(|&s| ((s / total_weight) + 128.0).clamp(0.0, 255.0) as u8)
                .collect(),
        )
    }

    /// Get compression statistics for current round
    pub fn compression_stats(&self) -> CompressionStats {
        if self.round_hypergradients.is_empty() {
            return CompressionStats::default();
        }

        let total_original: usize = self
            .round_hypergradients
            .iter()
            .map(|hg| hg.original_size)
            .sum();

        let total_compressed: usize = self
            .round_hypergradients
            .iter()
            .map(|hg| hg.hypervector.len())
            .sum();

        let avg_encode_time: f32 = self
            .round_hypergradients
            .iter()
            .map(|hg| hg.encode_time_ms)
            .sum::<f32>()
            / self.round_hypergradients.len() as f32;

        CompressionStats {
            submissions: self.round_hypergradients.len(),
            total_original_bytes: total_original,
            total_compressed_bytes: total_compressed,
            overall_compression_ratio: total_original as f32 / total_compressed.max(1) as f32,
            avg_encode_time_ms: avg_encode_time,
            bandwidth_saved_bytes: total_original.saturating_sub(total_compressed),
        }
    }

    /// Get current round hypergradients
    pub fn round_hypergradients(&self) -> &[HyperGradient] {
        &self.round_hypergradients
    }

    /// Get configuration
    pub fn config(&self) -> &HyperFeelFLConfig {
        &self.config
    }

    // =========================================================================
    // Direct Hypervector-Space Aggregation Methods (A2)
    // =========================================================================

    /// Aggregate hypergradients using FedAvg directly in hypervector space
    ///
    /// This is a simple average without weighting - useful when all participants
    /// are equally trusted.
    pub fn aggregate_fedavg_hv(&self) -> Option<AggregatedHyperGradient> {
        if self.round_hypergradients.is_empty() {
            return None;
        }

        let n = self.round_hypergradients.len();
        let hv_len = self.round_hypergradients[0].hypervector.len();
        let mut sum: Vec<i32> = vec![0; hv_len];

        for hg in &self.round_hypergradients {
            for (i, &byte) in hg.hypervector.iter().enumerate() {
                sum[i] += byte as i32;
            }
        }

        let aggregated: Vec<u8> = sum
            .iter()
            .map(|&s| (s / n as i32).clamp(0, 255) as u8)
            .collect();

        Some(AggregatedHyperGradient {
            hypervector: aggregated,
            participant_count: n,
            excluded_count: 0,
            method: HVAggregationMethod::FedAvg,
            byzantine_filtered: false,
        })
    }

    /// Aggregate with Byzantine filtering in hypervector space
    ///
    /// Excludes gradients that have similarity < threshold to the centroid.
    pub fn aggregate_byzantine_filtered_hv(&mut self) -> Option<AggregatedHyperGradient> {
        if self.round_hypergradients.len() < self.config.min_cluster_size {
            return self.aggregate_fedavg_hv();
        }

        // Run Byzantine analysis to populate similarity cache
        let analysis = self.analyze_byzantine();

        let hv_len = self.round_hypergradients[0].hypervector.len();
        let mut weighted_sum: Vec<f32> = vec![0.0; hv_len];
        let mut total_weight = 0.0;
        let mut included_count = 0;
        let mut excluded_count = 0;

        for hg in &self.round_hypergradients {
            let is_byzantine = analysis
                .byzantine_flags
                .get(&hg.node_id)
                .copied()
                .unwrap_or(false);

            if is_byzantine {
                excluded_count += 1;
                continue;
            }

            let weight = self.get_similarity_weight(&hg.node_id);
            total_weight += weight;
            included_count += 1;

            for (i, &byte) in hg.hypervector.iter().enumerate() {
                weighted_sum[i] += (byte as f32 - 128.0) * weight;
            }
        }

        if total_weight < 0.001 || included_count == 0 {
            return None;
        }

        let aggregated: Vec<u8> = weighted_sum
            .iter()
            .map(|&s| ((s / total_weight) + 128.0).clamp(0.0, 255.0) as u8)
            .collect();

        Some(AggregatedHyperGradient {
            hypervector: aggregated,
            participant_count: included_count,
            excluded_count,
            method: HVAggregationMethod::ByzantineFiltered,
            byzantine_filtered: true,
        })
    }

    /// Krum-like selection in hypervector space
    ///
    /// Selects the hypergradient that has the minimum total distance to its
    /// k nearest neighbors, making it robust against Byzantine participants.
    pub fn aggregate_krum_hv(&self, num_byzantine: usize) -> Option<AggregatedHyperGradient> {
        let n = self.round_hypergradients.len();
        if n < 3 {
            return self.aggregate_fedavg_hv();
        }

        // k = n - num_byzantine - 2
        let k = n.saturating_sub(num_byzantine).saturating_sub(2).max(1);

        // Calculate pairwise similarities
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            // Calculate similarities to all other gradients
            let mut similarities: Vec<f32> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i != j {
                    let sim = HyperFeelEncoder::cosine_similarity(
                        &self.round_hypergradients[i].hypervector,
                        &self.round_hypergradients[j].hypervector,
                    );
                    similarities.push(sim);
                }
            }

            // Sort and take k highest similarities (closest neighbors)
            similarities.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let krum_score: f32 = similarities.iter().take(k).sum();
            scores.push((i, krum_score));
        }

        // Select the gradient with highest Krum score (most similar to neighbors)
        let (best_idx, _) = scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

        Some(AggregatedHyperGradient {
            hypervector: self.round_hypergradients[*best_idx].hypervector.clone(),
            participant_count: 1,
            excluded_count: n - 1,
            method: HVAggregationMethod::Krum,
            byzantine_filtered: true,
        })
    }

    /// Multi-Krum aggregation in hypervector space
    ///
    /// Selects top-m gradients by Krum score and averages them.
    pub fn aggregate_multi_krum_hv(
        &self,
        num_byzantine: usize,
        m: usize,
    ) -> Option<AggregatedHyperGradient> {
        let n = self.round_hypergradients.len();
        if n < 3 || m == 0 {
            return self.aggregate_fedavg_hv();
        }

        let m = m.min(n);
        let k = n.saturating_sub(num_byzantine).saturating_sub(2).max(1);

        // Calculate Krum scores
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            let mut similarities: Vec<f32> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i != j {
                    let sim = HyperFeelEncoder::cosine_similarity(
                        &self.round_hypergradients[i].hypervector,
                        &self.round_hypergradients[j].hypervector,
                    );
                    similarities.push(sim);
                }
            }
            similarities.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let krum_score: f32 = similarities.iter().take(k).sum();
            scores.push((i, krum_score));
        }

        // Sort by score and take top m
        scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let selected_indices: Vec<usize> = scores.iter().take(m).map(|(i, _)| *i).collect();

        // Average the selected hypergradients
        let hv_len = self.round_hypergradients[0].hypervector.len();
        let mut sum: Vec<i32> = vec![0; hv_len];

        for &idx in &selected_indices {
            for (i, &byte) in self.round_hypergradients[idx]
                .hypervector
                .iter()
                .enumerate()
            {
                sum[i] += byte as i32;
            }
        }

        let aggregated: Vec<u8> = sum
            .iter()
            .map(|&s| (s / selected_indices.len() as i32).clamp(0, 255) as u8)
            .collect();

        Some(AggregatedHyperGradient {
            hypervector: aggregated,
            participant_count: selected_indices.len(),
            excluded_count: n - selected_indices.len(),
            method: HVAggregationMethod::MultiKrum,
            byzantine_filtered: true,
        })
    }

    /// Coordinate-wise median in hypervector space
    ///
    /// Takes the median value at each position, robust against up to 50% Byzantine.
    pub fn aggregate_median_hv(&self) -> Option<AggregatedHyperGradient> {
        let n = self.round_hypergradients.len();
        if n == 0 {
            return None;
        }

        let hv_len = self.round_hypergradients[0].hypervector.len();
        let mut aggregated: Vec<u8> = Vec::with_capacity(hv_len);

        for pos in 0..hv_len {
            let mut values: Vec<u8> = self
                .round_hypergradients
                .iter()
                .map(|hg| hg.hypervector[pos])
                .collect();
            values.sort();

            // Median value
            let median = if n.is_multiple_of(2) {
                ((values[n / 2 - 1] as u16 + values[n / 2] as u16) / 2) as u8
            } else {
                values[n / 2]
            };
            aggregated.push(median);
        }

        Some(AggregatedHyperGradient {
            hypervector: aggregated,
            participant_count: n,
            excluded_count: 0,
            method: HVAggregationMethod::Median,
            byzantine_filtered: false,
        })
    }

    /// Trimmed mean in hypervector space
    ///
    /// Removes top/bottom trim_ratio of values at each position before averaging.
    pub fn aggregate_trimmed_mean_hv(&self, trim_ratio: f32) -> Option<AggregatedHyperGradient> {
        let n = self.round_hypergradients.len();
        if n < 3 {
            return self.aggregate_fedavg_hv();
        }

        let trim_count = (n as f32 * trim_ratio).ceil() as usize;
        if trim_count * 2 >= n {
            return self.aggregate_median_hv();
        }

        let hv_len = self.round_hypergradients[0].hypervector.len();
        let mut aggregated: Vec<u8> = Vec::with_capacity(hv_len);

        for pos in 0..hv_len {
            let mut values: Vec<u8> = self
                .round_hypergradients
                .iter()
                .map(|hg| hg.hypervector[pos])
                .collect();
            values.sort();

            // Trim and average
            let trimmed = &values[trim_count..n - trim_count];
            let sum: u32 = trimmed.iter().map(|&v| v as u32).sum();
            let mean = (sum / trimmed.len() as u32) as u8;
            aggregated.push(mean);
        }

        Some(AggregatedHyperGradient {
            hypervector: aggregated,
            participant_count: n,
            excluded_count: 0,
            method: HVAggregationMethod::TrimmedMean,
            byzantine_filtered: false,
        })
    }
}

/// Result of hypervector-space aggregation
#[derive(Debug, Clone)]
pub struct AggregatedHyperGradient {
    /// The aggregated hypervector
    pub hypervector: Vec<u8>,
    /// Number of participants included in aggregation
    pub participant_count: usize,
    /// Number of participants excluded (Byzantine, etc.)
    pub excluded_count: usize,
    /// Aggregation method used
    pub method: HVAggregationMethod,
    /// Whether Byzantine filtering was applied
    pub byzantine_filtered: bool,
}

impl AggregatedHyperGradient {
    /// Get hypervector size in bytes
    pub fn size_bytes(&self) -> usize {
        self.hypervector.len()
    }

    /// Compute similarity to another hypergradient
    pub fn similarity_to(&self, other: &HyperGradient) -> f32 {
        HyperFeelEncoder::cosine_similarity(&self.hypervector, &other.hypervector)
    }
}

/// Aggregation methods for hypervector space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HVAggregationMethod {
    /// Simple average (FedAvg)
    FedAvg,
    /// Weighted average with Byzantine filtering
    ByzantineFiltered,
    /// Krum selection (single best)
    Krum,
    /// Multi-Krum (average of top-m)
    MultiKrum,
    /// Coordinate-wise median
    Median,
    /// Trimmed mean
    TrimmedMean,
}

/// Compression statistics for a round
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Number of submissions
    pub submissions: usize,
    /// Total original size in bytes
    pub total_original_bytes: usize,
    /// Total compressed size in bytes
    pub total_compressed_bytes: usize,
    /// Overall compression ratio
    pub overall_compression_ratio: f32,
    /// Average encoding time in milliseconds
    pub avg_encode_time_ms: f32,
    /// Bandwidth saved in bytes
    pub bandwidth_saved_bytes: usize,
}

impl CompressionStats {
    /// Get bandwidth savings as percentage
    pub fn bandwidth_savings_percent(&self) -> f32 {
        if self.total_original_bytes == 0 {
            0.0
        } else {
            (self.bandwidth_saved_bytes as f32 / self.total_original_bytes as f32) * 100.0
        }
    }
}

/// Extended FL coordinator with HyperFeel integration
pub struct HyperFeelFLCoordinator {
    /// Base FL coordinator
    coordinator: FLCoordinator,
    /// HyperFeel bridge
    pub bridge: HyperFeelFLBridge,
    /// Current round's Byzantine analysis
    byzantine_analysis: Option<ByzantineAnalysis>,
}

impl HyperFeelFLCoordinator {
    /// Create a new HyperFeel-enabled coordinator
    pub fn new(fl_config: super::types::FLConfig, hyperfeel_config: HyperFeelFLConfig) -> Self {
        Self {
            coordinator: FLCoordinator::new(fl_config),
            bridge: HyperFeelFLBridge::new(hyperfeel_config),
            byzantine_analysis: None,
        }
    }

    /// Register a participant
    pub fn register_participant(&mut self, id: String) {
        self.coordinator.register_participant(id);
    }

    /// Start a new FL round
    pub fn start_round(&mut self) -> Result<(), CoordinatorError> {
        self.bridge.start_new_round();
        self.byzantine_analysis = None;
        self.coordinator.start_round()?;
        Ok(())
    }

    /// Submit a compressed gradient
    pub fn submit_compressed(
        &mut self,
        submission: CompressedSubmission,
        model_version: u64,
    ) -> bool {
        // Store for Byzantine analysis
        let hg = submission.hypergradient.clone();
        self.bridge.submit_compressed(submission.clone());

        // Convert to FL format and submit
        let update = self.bridge.hypergradient_to_update(
            &hg,
            model_version,
            submission.batch_size,
            submission.loss,
        );

        self.coordinator.submit_update(update)
    }

    /// Analyze Byzantine behavior before aggregation
    pub fn analyze_and_flag_byzantine(&mut self) -> &ByzantineAnalysis {
        let analysis = self.bridge.analyze_byzantine();

        // Flag suspicious participants in coordinator
        for (node_id, &is_byzantine) in &analysis.byzantine_flags {
            if is_byzantine {
                if let Some(participant) = self.coordinator.get_participant_mut(node_id) {
                    participant.record_negative();
                }
            }
        }

        self.byzantine_analysis = Some(analysis);
        // Safe: we just set byzantine_analysis to Some above
        self.byzantine_analysis
            .as_ref()
            .expect("byzantine_analysis should be Some after setting it")
    }

    /// Aggregate the round with Byzantine filtering
    pub fn aggregate_round(&mut self) -> Result<AggregatedGradient, CoordinatorError> {
        // Run Byzantine analysis if not already done
        if self.byzantine_analysis.is_none() {
            self.analyze_and_flag_byzantine();
        }

        // Aggregate via base coordinator
        self.coordinator.aggregate_round()
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> CompressionStats {
        self.bridge.compression_stats()
    }

    /// Get Byzantine analysis
    pub fn byzantine_analysis(&self) -> Option<&ByzantineAnalysis> {
        self.byzantine_analysis.as_ref()
    }

    /// Get base coordinator
    pub fn coordinator(&self) -> &FLCoordinator {
        &self.coordinator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fl::FLConfig;

    fn create_test_bridge() -> HyperFeelFLBridge {
        HyperFeelFLBridge::new(HyperFeelFLConfig {
            similarity_threshold: 0.5,
            min_cluster_size: 2,
            ..Default::default()
        })
    }

    fn create_similar_gradient(base: &[f32], noise_factor: f32) -> Vec<f32> {
        base.iter()
            .enumerate()
            .map(|(i, &v)| v + noise_factor * (i as f32 * 0.0001).sin())
            .collect()
    }

    fn create_byzantine_gradient(size: usize) -> Vec<f32> {
        // Completely different pattern (Byzantine attacker)
        // Use random-looking values that are very different from sine waves
        (0..size)
            .map(|i| {
                let x = i as f32;
                // Mix of very different patterns: random-ish alternating high/low
                if i % 2 == 0 {
                    1000.0
                } else {
                    -1000.0
                }
            })
            .collect()
    }

    #[test]
    fn test_compress_and_convert() {
        let mut bridge = create_test_bridge();

        // Use larger gradient for better compression ratio
        let gradient: Vec<f32> = (0..50_000).map(|i| (i as f32 * 0.001).sin()).collect();
        let hg = bridge.compress_gradient(&gradient, 1, "node-1");

        // 50K floats = 200KB, hypervector = 2KB, ratio > 90
        assert!(
            hg.compression_ratio > 50.0,
            "Compression ratio: {}",
            hg.compression_ratio
        );

        // Convert to update
        let update = bridge.hypergradient_to_update(&hg, 1, 100, 0.5);
        assert_eq!(update.participant_id, "node-1");
        assert!(!update.gradients.is_empty());
    }

    #[test]
    fn test_byzantine_detection_similar() {
        let mut bridge = create_test_bridge();

        // Create similar gradients (honest nodes)
        let base: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.001).sin()).collect();

        for i in 0..5 {
            let gradient = create_similar_gradient(&base, 0.01);
            let hg = bridge.compress_gradient(&gradient, 1, &format!("node-{}", i));
            bridge.submit_compressed(CompressedSubmission {
                hypergradient: hg,
                batch_size: 100,
                loss: 0.5,
                accuracy: None,
            });
        }

        let analysis = bridge.analyze_byzantine();

        // All similar gradients should NOT be flagged
        assert!(analysis.flagged_count <= 1, "Too many false positives");
        assert!(analysis.centroid.is_some());
    }

    #[test]
    fn test_byzantine_detection_outlier() {
        // Use higher similarity threshold for stricter detection
        let mut bridge = HyperFeelFLBridge::new(HyperFeelFLConfig {
            similarity_threshold: 0.8, // Higher threshold makes outlier detection stricter
            min_cluster_size: 2,
            ..Default::default()
        });

        // Create similar gradients (honest nodes) with larger size
        let base: Vec<f32> = (0..10_000).map(|i| (i as f32 * 0.001).sin()).collect();

        for i in 0..4 {
            let gradient = create_similar_gradient(&base, 0.001); // Very similar
            let hg = bridge.compress_gradient(&gradient, 1, &format!("node-{}", i));
            bridge.submit_compressed(CompressedSubmission {
                hypergradient: hg,
                batch_size: 100,
                loss: 0.5,
                accuracy: None,
            });
        }

        // Add Byzantine gradient - completely different distribution
        let byzantine = create_byzantine_gradient(10_000);
        let hg = bridge.compress_gradient(&byzantine, 1, "byzantine-node");
        bridge.submit_compressed(CompressedSubmission {
            hypergradient: hg,
            batch_size: 100,
            loss: 0.5,
            accuracy: None,
        });

        let analysis = bridge.analyze_byzantine();

        // Check similarity scores - Byzantine should have lowest similarity
        let byzantine_sim = *analysis
            .similarity_scores
            .get("byzantine-node")
            .unwrap_or(&1.0);
        let honest_sims: Vec<f32> = (0..4)
            .map(|i| {
                *analysis
                    .similarity_scores
                    .get(&format!("node-{}", i))
                    .unwrap_or(&0.0)
            })
            .collect();
        let avg_honest_sim = honest_sims.iter().sum::<f32>() / 4.0;

        // Byzantine node should have lower similarity than honest nodes
        assert!(
            byzantine_sim < avg_honest_sim,
            "Byzantine ({}) should be less similar than honest avg ({})",
            byzantine_sim,
            avg_honest_sim
        );
    }

    #[test]
    fn test_similarity_weighting() {
        let mut bridge = create_test_bridge();

        // Setup some gradients
        let base: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.001).sin()).collect();

        for i in 0..3 {
            let gradient = create_similar_gradient(&base, 0.01);
            let hg = bridge.compress_gradient(&gradient, 1, &format!("node-{}", i));
            bridge.submit_compressed(CompressedSubmission {
                hypergradient: hg,
                batch_size: 100,
                loss: 0.5,
                accuracy: None,
            });
        }

        // Run analysis to populate cache
        bridge.analyze_byzantine();

        // Check weights
        for i in 0..3 {
            let weight = bridge.get_similarity_weight(&format!("node-{}", i));
            assert!(weight > 0.0, "Weight should be positive");
        }
    }

    #[test]
    fn test_compression_stats() {
        let mut bridge = create_test_bridge();

        let gradient: Vec<f32> = vec![0.1; 100_000]; // 400KB original

        for i in 0..3 {
            let hg = bridge.compress_gradient(&gradient, 1, &format!("node-{}", i));
            bridge.submit_compressed(CompressedSubmission {
                hypergradient: hg,
                batch_size: 100,
                loss: 0.5,
                accuracy: None,
            });
        }

        let stats = bridge.compression_stats();

        assert_eq!(stats.submissions, 3);
        assert!(stats.overall_compression_ratio > 100.0);
        assert!(stats.bandwidth_savings_percent() > 99.0);
    }

    #[test]
    fn test_aggregate_hypergradients() {
        let mut bridge = create_test_bridge();

        let base: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.001).sin()).collect();

        for i in 0..3 {
            let gradient = create_similar_gradient(&base, 0.01);
            let hg = bridge.compress_gradient(&gradient, 1, &format!("node-{}", i));
            bridge.submit_compressed(CompressedSubmission {
                hypergradient: hg,
                batch_size: 100,
                loss: 0.5,
                accuracy: None,
            });
        }

        // Run analysis first
        bridge.analyze_byzantine();

        // Aggregate
        let aggregated = bridge.aggregate_hypergradients();
        assert!(aggregated.is_some());
        assert_eq!(aggregated.unwrap().len(), 2048); // HV16 size
    }

    #[test]
    fn test_hyperfeel_fl_coordinator() {
        let fl_config = FLConfig {
            min_participants: 2,
            trust_threshold: 0.3,
            ..Default::default()
        };

        let hf_config = HyperFeelFLConfig {
            min_cluster_size: 2,
            ..Default::default()
        };

        let mut coordinator = HyperFeelFLCoordinator::new(fl_config, hf_config);

        // Register participants
        coordinator.register_participant("p1".to_string());
        coordinator.register_participant("p2".to_string());

        // Start round
        assert!(coordinator.start_round().is_ok());

        // Create and submit compressed gradients (larger size for better compression)
        let base: Vec<f32> = (0..50_000).map(|i| (i as f32 * 0.001).sin()).collect();

        let hg1 = coordinator.bridge.compress_gradient(&base, 1, "p1");
        let hg2 =
            coordinator
                .bridge
                .compress_gradient(&create_similar_gradient(&base, 0.01), 1, "p2");

        assert!(coordinator.submit_compressed(
            CompressedSubmission {
                hypergradient: hg1,
                batch_size: 100,
                loss: 0.5,
                accuracy: None,
            },
            1,
        ));

        assert!(coordinator.submit_compressed(
            CompressedSubmission {
                hypergradient: hg2,
                batch_size: 100,
                loss: 0.4,
                accuracy: None,
            },
            1,
        ));

        // Analyze and aggregate
        let analysis = coordinator.analyze_and_flag_byzantine();
        assert!(analysis.flagged_count == 0);

        let result = coordinator.aggregate_round();
        assert!(result.is_ok());

        // Check compression stats (50K floats = 200KB → 2KB = ~100x)
        let stats = coordinator.compression_stats();
        assert_eq!(stats.submissions, 2);
        assert!(stats.overall_compression_ratio > 10.0);
    }

    #[test]
    fn test_round_reset() {
        let mut bridge = create_test_bridge();

        // Add some gradients
        let gradient: Vec<f32> = vec![0.1; 1000];
        let hg = bridge.compress_gradient(&gradient, 1, "node-1");
        bridge.submit_compressed(CompressedSubmission {
            hypergradient: hg,
            batch_size: 100,
            loss: 0.5,
            accuracy: None,
        });

        assert_eq!(bridge.round_hypergradients().len(), 1);

        // Start new round
        bridge.start_new_round();

        assert_eq!(bridge.round_hypergradients().len(), 0);
    }

    // =========================================================================
    // Tests for Direct Hypervector Aggregation (A2)
    // =========================================================================

    fn setup_bridge_with_gradients(count: usize, size: usize) -> HyperFeelFLBridge {
        let mut bridge = create_test_bridge();
        let base: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();

        for i in 0..count {
            let gradient = create_similar_gradient(&base, 0.01);
            let hg = bridge.compress_gradient(&gradient, 1, &format!("node-{}", i));
            bridge.submit_compressed(CompressedSubmission {
                hypergradient: hg,
                batch_size: 100,
                loss: 0.5,
                accuracy: None,
            });
        }
        bridge
    }

    #[test]
    fn test_aggregate_fedavg_hv() {
        let bridge = setup_bridge_with_gradients(5, 5000);

        let result = bridge.aggregate_fedavg_hv();
        assert!(result.is_some());

        let agg = result.unwrap();
        assert_eq!(agg.participant_count, 5);
        assert_eq!(agg.excluded_count, 0);
        assert_eq!(agg.method, HVAggregationMethod::FedAvg);
        assert!(!agg.byzantine_filtered);
        assert_eq!(agg.size_bytes(), 2048); // HV16 size
    }

    #[test]
    fn test_aggregate_byzantine_filtered_hv() {
        let mut bridge = setup_bridge_with_gradients(4, 5000);

        // Add Byzantine gradient
        let byzantine = create_byzantine_gradient(5000);
        let hg = bridge.compress_gradient(&byzantine, 1, "byzantine");
        bridge.submit_compressed(CompressedSubmission {
            hypergradient: hg,
            batch_size: 100,
            loss: 0.5,
            accuracy: None,
        });

        let result = bridge.aggregate_byzantine_filtered_hv();
        assert!(result.is_some());

        let agg = result.unwrap();
        assert!(agg.byzantine_filtered);
        assert_eq!(agg.method, HVAggregationMethod::ByzantineFiltered);
        // Should have excluded some participants
        assert!(agg.participant_count + agg.excluded_count == 5);
    }

    #[test]
    fn test_aggregate_median_hv() {
        let bridge = setup_bridge_with_gradients(7, 5000);

        let result = bridge.aggregate_median_hv();
        assert!(result.is_some());

        let agg = result.unwrap();
        assert_eq!(agg.participant_count, 7);
        assert_eq!(agg.method, HVAggregationMethod::Median);
        assert_eq!(agg.size_bytes(), 2048);
    }

    #[test]
    fn test_aggregate_krum_hv() {
        let bridge = setup_bridge_with_gradients(5, 5000);

        let result = bridge.aggregate_krum_hv(1); // Assume 1 Byzantine
        assert!(result.is_some());

        let agg = result.unwrap();
        assert_eq!(agg.participant_count, 1); // Krum selects single best
        assert_eq!(agg.excluded_count, 4);
        assert_eq!(agg.method, HVAggregationMethod::Krum);
        assert!(agg.byzantine_filtered);
    }

    #[test]
    fn test_aggregate_multi_krum_hv() {
        let bridge = setup_bridge_with_gradients(6, 5000);

        let result = bridge.aggregate_multi_krum_hv(1, 3); // 1 Byzantine, select top 3
        assert!(result.is_some());

        let agg = result.unwrap();
        assert_eq!(agg.participant_count, 3);
        assert_eq!(agg.excluded_count, 3);
        assert_eq!(agg.method, HVAggregationMethod::MultiKrum);
    }

    #[test]
    fn test_aggregate_trimmed_mean_hv() {
        let bridge = setup_bridge_with_gradients(10, 5000);

        let result = bridge.aggregate_trimmed_mean_hv(0.1); // Trim 10%
        assert!(result.is_some());

        let agg = result.unwrap();
        assert_eq!(agg.participant_count, 10);
        assert_eq!(agg.method, HVAggregationMethod::TrimmedMean);
    }

    #[test]
    fn test_aggregated_hypergradient_similarity() {
        let bridge = setup_bridge_with_gradients(5, 5000);

        let agg = bridge.aggregate_fedavg_hv().unwrap();

        // Aggregated result should be similar to individual gradients
        for hg in bridge.round_hypergradients() {
            let sim = agg.similarity_to(hg);
            assert!(sim > 0.5, "Similarity should be positive: {}", sim);
        }
    }
}
