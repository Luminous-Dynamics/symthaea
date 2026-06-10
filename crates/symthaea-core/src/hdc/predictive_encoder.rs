// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Predictive HDC Encoder - Attention-Modulated Semantic Encoding
//!
//! This module implements the **bidirectional HDC↔LTC loop** by adding
//! prediction-driven attention to HDC encoding.
//!
//! ## The Core Innovation
//!
//! Traditional HDC encoding is instant and feedforward:
//! ```text
//! Input → HDC Encode → Done
//! ```
//!
//! PredictiveHdcEncoder creates a feedback loop:
//! ```text
//! Input → HDC Encode (with attention) → Output
//!           ↑                              │
//!           │    LTC Prediction ←──────────┘
//!           │           │
//!           └───── Prediction Error
//! ```
//!
//! ## How Attention Emerges
//!
//! 1. LTC predicts what the next HDC state should be
//! 2. Actual HDC encoding creates prediction error
//! 3. Error modulates attention weights per primitive
//! 4. High-error primitives get more attention (surprisal-based learning)
//!
//! This is biologically inspired by predictive coding in the cortex.

use crate::hdc::HDC_DIMENSION;
use crate::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};
use crate::hdc::text_encoder::{TextEncoder, TextEncoderConfig};
use crate::hdc::unified_hv::ContinuousHV;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::mem;
use std::sync::Arc;

/// Configuration for the predictive encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveEncoderConfig {
    /// HDC dimension (default: HDC_DIMENSION)
    pub dimension: usize,

    /// Learning rate for attention weights
    pub attention_lr: f32,

    /// Minimum attention weight (prevents complete suppression)
    pub min_attention: f32,

    /// Maximum attention weight (prevents explosion)
    pub max_attention: f32,

    /// Error history window size for smoothing
    pub error_window_size: usize,

    /// Threshold for significant prediction error
    pub error_threshold: f32,

    /// Initial attention weight for all primitives
    pub initial_attention: f32,

    /// Attention decay rate (slight decay to uniform)
    pub attention_decay: f32,
}

impl Default for PredictiveEncoderConfig {
    fn default() -> Self {
        Self {
            dimension: HDC_DIMENSION,
            attention_lr: 0.1,
            min_attention: 0.1,
            max_attention: 3.0,
            error_window_size: 50,
            error_threshold: 0.1,
            initial_attention: 1.0,
            attention_decay: 0.001,
        }
    }
}

/// Result of encoding with prediction
#[derive(Debug, Clone)]
pub struct EncodingResult {
    /// The encoded HDV (attention-modulated)
    pub hdv: ContinuousHV,

    /// Prediction error from this cycle
    pub prediction_error: f32,

    /// Primitives that were detected in input
    pub detected_primitives: Vec<String>,

    /// Peak attention weight this cycle (for MCE consciousness calculation)
    pub peak_attention: f32,
}

/// Statistics for monitoring the encoder
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncoderStats {
    /// Total encoding cycles
    pub total_cycles: usize,

    /// Average prediction error (EMA)
    pub avg_prediction_error: f32,

    /// Attention weight variance (emergence metric)
    pub attention_variance: f32,

    /// Number of primitives with non-uniform attention
    pub diverged_primitives: usize,

    /// Total prediction errors (sum)
    pub cumulative_error: f32,

    /// Cycles with significant error (above threshold)
    pub high_error_cycles: usize,
}

/// Predictive HDC Encoder with attention modulation
pub struct PredictiveHdcEncoder {
    /// Configuration
    config: PredictiveEncoderConfig,

    /// Base primitive system for semantic encoding
    primitive_system: &'static PrimitiveSystem,

    /// Text encoder for input processing
    text_encoder: TextEncoder,

    /// Attention weights per primitive (learned from prediction error)
    attention_weights: HashMap<String, f32>,

    /// Running prediction from LTC (compressed HDV)
    predicted_hdv: Option<Vec<f32>>,

    /// Prediction error history for smoothing
    error_history: VecDeque<f32>,

    /// Statistics
    stats: EncoderStats,

    /// Primitive name cache for fast lookup
    primitive_names: Vec<String>,

    /// Pre-lowercased primitive names (avoids 200x .to_lowercase() per cycle)
    primitive_names_lower: Vec<String>,

    /// Peak attention weight (cached to avoid HashMap iteration)
    peak_attention: f32,

    /// Pre-allocated buffer for bipolar→real conversion (avoids 64KB alloc per cycle)
    conversion_buffer: Vec<f32>,

    /// Pre-allocated buffer for detected primitives (avoids `Vec<String>` alloc per cycle)
    detected_buffer: Vec<String>,

    /// Pre-computed i8 bipolar encodings for all primitives (keyed by lowercase name).
    /// Avoids calling `to_bipolar_i8()` (16KB allocation) on every primitive match per cycle.
    /// Built once at construction; all values are `Arc` so cache lookups are O(1).
    primitive_i8_cache: HashMap<String, Arc<Vec<i8>>>,
}

impl PredictiveHdcEncoder {
    /// Create a new predictive encoder.
    ///
    /// Returns `Err` if the underlying `TextEncoder` fails to initialize.
    pub fn new(config: PredictiveEncoderConfig) -> anyhow::Result<Self> {
        let primitive_system = PrimitiveSystem::global();
        let text_encoder = TextEncoder::new(TextEncoderConfig {
            dimension: config.dimension,
            ..Default::default()
        })?;

        // Initialize attention weights for all primitives
        let mut attention_weights = HashMap::new();
        let mut primitive_names = Vec::new();

        // Collect all primitive names across all tiers
        for tier in [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ] {
            for prim in primitive_system.get_tier(tier) {
                attention_weights.insert(prim.name.clone(), config.initial_attention);
                primitive_names.push(prim.name.clone());
            }
        }

        let primitive_names_lower: Vec<String> =
            primitive_names.iter().map(|n| n.to_lowercase()).collect();
        let initial_attention = config.initial_attention;
        let dimension = config.dimension;

        // Pre-compute i8 bipolar encodings for ALL primitives (keyed by lowercase name).
        // This is done once at construction so encode() never calls to_bipolar_i8()
        // (which allocates 16KB per call) on the hot path.
        let primitive_i8_cache: HashMap<String, Arc<Vec<i8>>> = primitive_system
            .all_primitives()
            .map(|prim| {
                let key = prim.name.to_lowercase();
                let encoding = Arc::new(prim.encoding.to_bipolar_i8());
                (key, encoding)
            })
            .collect();

        Ok(Self {
            config,
            primitive_system,
            text_encoder,
            attention_weights,
            predicted_hdv: None,
            error_history: VecDeque::with_capacity(50),
            stats: EncoderStats::default(),
            primitive_names,
            primitive_names_lower,
            peak_attention: initial_attention,
            conversion_buffer: vec![0.0f32; dimension],
            detected_buffer: Vec::with_capacity(32),
            primitive_i8_cache,
        })
    }

    /// Encode input with attention modulation from prediction
    pub fn encode(&mut self, input: &str) -> EncodingResult {
        self.stats.total_cycles += 1;

        // 1. Get base encoding via text encoder (uses pre-cached primitive i8 encodings)
        let base_encoding = match self
            .text_encoder
            .encode_with_cached_primitives(input, &self.primitive_i8_cache)
        {
            Ok(enc) => enc,
            Err(_) => {
                // Fallback to sentence encoding if primitive encoding fails
                self.text_encoder
                    .encode_sentence(input)
                    .unwrap_or_else(|_| vec![0i8; self.config.dimension])
            }
        };

        // Convert bipolar to real-valued (reuses pre-allocated buffer)
        let base_hdv = self.bipolar_to_real(&base_encoding);

        // 2. Detect which primitives are in the input (reuses pre-allocated buffer)
        let detected_primitives = self.detect_primitives(input);

        // 3. Apply attention modulation (takes ownership, mutates in place to avoid alloc)
        let attended_hdv = self.apply_attention_in_place(base_hdv, &detected_primitives);

        // 4. Compute prediction error if we have a prediction
        let prediction_error = self.compute_prediction_error(&attended_hdv);

        // 5. Update attention weights based on error
        self.update_attention(&detected_primitives, prediction_error);

        // 6. Update statistics
        self.update_stats(prediction_error);

        EncodingResult {
            hdv: attended_hdv,
            prediction_error,
            detected_primitives,
            peak_attention: self.peak_attention,
        }
    }

    /// Get the full attention weights map (expensive clone — use only for debugging/monitoring).
    pub fn attention_weights_snapshot(&self) -> HashMap<String, f32> {
        self.attention_weights.clone()
    }

    /// Get the peak attention weight (cheap — cached per cycle).
    pub fn peak_attention(&self) -> f32 {
        self.peak_attention
    }

    /// Receive prediction from LTC for next cycle
    pub fn set_prediction(&mut self, predicted: Vec<f32>) {
        self.predicted_hdv = Some(predicted);
    }

    /// Clear the current prediction (for reset)
    pub fn clear_prediction(&mut self) {
        self.predicted_hdv = None;
    }

    /// Get current attention weights
    pub fn attention_weights(&self) -> &HashMap<String, f32> {
        &self.attention_weights
    }

    /// Get encoder statistics
    pub fn stats(&self) -> &EncoderStats {
        &self.stats
    }

    /// Reset attention to uniform
    pub fn reset_attention(&mut self) {
        for (_, weight) in self.attention_weights.iter_mut() {
            *weight = self.config.initial_attention;
        }
        self.stats.attention_variance = 0.0;
        self.stats.diverged_primitives = 0;
    }

    // ========== Internal Methods ==========

    /// Convert bipolar encoding to ContinuousHV
    ///
    /// Uses a pre-allocated conversion buffer to avoid a 64KB allocation per cycle.
    /// The buffer is moved into ContinuousHV via `mem::take` and immediately
    /// re-allocated; the allocator typically services this from the block freed
    /// by the previous cycle's ContinuousHV, making it effectively free.
    fn bipolar_to_real(&mut self, bipolar: &[i8]) -> ContinuousHV {
        let dim = self.config.dimension;
        let len = bipolar.len().min(dim);

        // Write bipolar values directly into the pre-allocated buffer (no zeroing needed)
        for i in 0..len {
            self.conversion_buffer[i] = bipolar[i] as f32;
        }
        // Zero any trailing elements (usually a no-op since len == dim)
        for i in len..dim {
            self.conversion_buffer[i] = 0.0;
        }

        // Move the filled buffer into ContinuousHV (zero-cost move, no allocation)
        let values = mem::take(&mut self.conversion_buffer);

        // Re-create the buffer for the next cycle. The allocator will typically
        // reuse the block freed when the *previous* cycle's ContinuousHV was dropped,
        // making this a hot-path allocation that hits allocator cache.
        self.conversion_buffer = vec![0.0f32; dim];

        ContinuousHV { values }
    }

    /// Detect which primitives are relevant to the input
    ///
    /// Reuses a pre-allocated buffer to avoid per-cycle Vec<String> allocation.
    /// Returns an owned Vec via `mem::take` + capacity preservation pattern:
    /// the buffer is swapped out, returned, and re-acquired next cycle.
    fn detect_primitives(&mut self, input: &str) -> Vec<String> {
        let input_lower = input.to_lowercase();
        self.detected_buffer.clear();

        // Check each primitive for presence in input
        // Uses pre-lowercased names (computed once at construction, not 200x per cycle)
        for (name, name_lower) in self
            .primitive_names
            .iter()
            .zip(self.primitive_names_lower.iter())
        {
            if input_lower.contains(name_lower.as_str()) {
                self.detected_buffer.push(name.clone());
            }
        }

        // Also check for semantic patterns
        self.detected_buffer
            .extend(self.detect_semantic_patterns(&input_lower));

        // Deduplicate
        self.detected_buffer.sort();
        self.detected_buffer.dedup();

        // Swap out the buffer (zero-cost move) instead of cloning.
        // The taken Vec keeps its heap allocation; we give ourselves
        // a fresh Vec that will reuse the allocation returned to us
        // on the next cycle's `clear()`.
        mem::take(&mut self.detected_buffer)
    }

    /// Detect primitives from semantic patterns (not just string matching)
    fn detect_semantic_patterns(&self, input: &str) -> Vec<String> {
        let mut detected = Vec::new();

        // Causal patterns
        if input.contains("cause") || input.contains("because") || input.contains("→") {
            detected.push("CAUSE".to_string());
            detected.push("EFFECT".to_string());
        }

        // Action patterns
        if input.contains("do") || input.contains("act") || input.contains("perform") {
            detected.push("ACTION".to_string());
        }

        // Temporal patterns
        if input.contains("before") || input.contains("after") || input.contains("then") {
            detected.push("BEFORE".to_string());
            detected.push("AFTER".to_string());
        }

        // Logical patterns
        if input.contains("if") || input.contains("then") || input.contains("implies") {
            detected.push("IMPLICATION".to_string());
        }

        // Quantity patterns
        if input.contains("more") || input.contains("less") || input.contains("equal") {
            detected.push("GREATER_THAN".to_string());
            detected.push("LESS_THAN".to_string());
        }

        detected
    }

    /// Apply attention weights to HDV (in-place, avoids extra 64KB allocation)
    ///
    /// Takes ownership of `base_hdv` and scales it in place, eliminating
    /// the allocation that `scale()` would create.
    fn apply_attention_in_place(
        &self,
        mut base_hdv: ContinuousHV,
        detected_primitives: &[String],
    ) -> ContinuousHV {
        if detected_primitives.is_empty() {
            return base_hdv;
        }

        // Compute composite attention weight from detected primitives
        let total_attention: f32 = detected_primitives
            .iter()
            .filter_map(|name| self.attention_weights.get(name))
            .sum();

        let avg_attention = if !detected_primitives.is_empty() {
            total_attention / detected_primitives.len() as f32
        } else {
            1.0
        };

        // Scale the HDV by attention in-place (no allocation)
        base_hdv.scale_in_place(avg_attention);
        base_hdv
    }

    /// Compute prediction error between current HDV and LTC's prediction
    ///
    /// IMPORTANT: Comparison happens in the COMPRESSED space (LTC output dimension)
    /// to ensure we're comparing apples to apples.
    fn compute_prediction_error(&self, current_hdv: &ContinuousHV) -> f32 {
        match &self.predicted_hdv {
            Some(predicted) => {
                // Compress current HDV to same dimension as prediction (NOT expand prediction!)
                // This ensures we compare in the same space the LTC operates in
                let compressed_current = self.compress_for_ltc(current_hdv, predicted.len());

                // Compute normalized L2 distance in compressed space
                let diff: f32 = compressed_current
                    .iter()
                    .zip(predicted.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                let norm_current: f32 =
                    compressed_current.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_pred: f32 = predicted.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_current > 0.0 && norm_pred > 0.0 {
                    // Normalized error in [0, 1] range
                    (diff.sqrt() / (norm_current + norm_pred)).clamp(0.0, 1.0)
                } else {
                    1.0 // Maximum error if norms are zero
                }
            }
            None => 1.0, // No prediction = maximum surprise
        }
    }

    /// Expand compressed prediction to full dimension
    fn expand_prediction(&self, compressed: &[f32]) -> Vec<f32> {
        if compressed.is_empty() {
            return vec![0.0; self.config.dimension];
        }
        if compressed.len() >= self.config.dimension {
            return compressed[..self.config.dimension].to_vec();
        }

        // Simple expansion: repeat values to fill dimension
        let repeat_factor = self.config.dimension / compressed.len().max(1);
        let mut expanded = Vec::with_capacity(self.config.dimension);

        for &val in compressed {
            for _ in 0..repeat_factor {
                expanded.push(val);
            }
        }

        // Pad remainder if needed
        while expanded.len() < self.config.dimension {
            expanded.push(0.0);
        }

        expanded
    }

    /// Update attention weights based on prediction error
    fn update_attention(&mut self, detected_primitives: &[String], error: f32) {
        // Only update if error is significant
        if error < self.config.error_threshold {
            // Apply small decay toward uniform
            for weight in self.attention_weights.values_mut() {
                *weight = *weight * (1.0 - self.config.attention_decay)
                    + self.config.initial_attention * self.config.attention_decay;
            }
            return;
        }

        // High error → increase attention on detected primitives
        // This is surprisal-based: surprising inputs get more attention
        for name in detected_primitives {
            if let Some(weight) = self.attention_weights.get_mut(name) {
                // Increase attention proportional to error
                let delta = self.config.attention_lr * error;
                *weight =
                    (*weight + delta).clamp(self.config.min_attention, self.config.max_attention);
            }
        }

        // Slightly decrease attention on non-detected primitives
        // This creates competition between primitives
        for name in &self.primitive_names {
            if !detected_primitives.contains(name) {
                if let Some(weight) = self.attention_weights.get_mut(name) {
                    let delta = self.config.attention_lr * error * 0.1; // Smaller decrease
                    *weight = (*weight - delta)
                        .clamp(self.config.min_attention, self.config.max_attention);
                }
            }
        }

        // Update cached peak attention (avoids HashMap iteration in hot path)
        self.peak_attention = self
            .attention_weights
            .values()
            .copied()
            .fold(0.0_f32, f32::max);
    }

    /// Update statistics
    fn update_stats(&mut self, error: f32) {
        // Update error history
        if self.error_history.len() >= self.config.error_window_size {
            self.error_history.pop_front();
        }
        self.error_history.push_back(error);

        // Compute average error (EMA)
        let alpha = 0.1;
        self.stats.avg_prediction_error =
            self.stats.avg_prediction_error * (1.0 - alpha) + error * alpha;

        // Compute attention variance (emergence metric) — iterate in-place, no Vec allocation
        let n = self.attention_weights.len() as f32;
        if n < 1.0 {
            self.stats.attention_variance = 0.0;
        } else {
            let sum: f32 = self.attention_weights.values().sum();
            let mean = sum / n;
            let variance: f32 = self
                .attention_weights
                .values()
                .map(|w| (w - mean).powi(2))
                .sum::<f32>()
                / n;
            self.stats.attention_variance = variance;
        }

        // Count diverged primitives
        let initial = self.config.initial_attention;
        self.stats.diverged_primitives = self
            .attention_weights
            .values()
            .filter(|&&w| (w - initial).abs() > 0.1)
            .count();

        // Update cumulative and high-error counts
        self.stats.cumulative_error += error;
        if error > self.config.error_threshold {
            self.stats.high_error_cycles += 1;
        }
    }

    /// Get compressed representation for LTC input via sparse random projection.
    ///
    /// Uses sparse Rademacher projection (Achlioptas 2003) with K=8 non-zeros
    /// per output dimension. Each non-zero is ±1/√K, selected deterministically
    /// from the input. This preserves pairwise distances (Johnson-Lindenstrauss
    /// lemma) while maintaining magnitude (no near-zero collapse from averaging).
    ///
    /// Prior approach (average pooling) collapsed values to near-zero via Law of
    /// Large Numbers, triggering excessive CfC backward passes (2x throughput hit).
    /// Sparse projection preserves variance: output σ ≈ input σ.
    pub fn compress_for_ltc(&self, hdv: &ContinuousHV, output_dim: usize) -> Vec<f32> {
        if output_dim == 0 {
            return Vec::new();
        }
        let input_len = hdv.values.len();
        if input_len <= output_dim {
            return hdv.values[..output_dim.min(input_len)].to_vec();
        }
        // Sparse Rademacher: K non-zeros per output dimension.
        // K=8 gives good accuracy at O(256×8) = O(2048) operations.
        // Deterministic selection via hash: reproducible across calls.
        let k = 8usize;
        let scale = 1.0 / (k as f32).sqrt();
        (0..output_dim)
            .map(|i| {
                let mut sum = 0.0f32;
                for j in 0..k {
                    // Deterministic pseudo-random index selection via mixing
                    let hash = ((i as u64).wrapping_mul(2654435761)
                        ^ (j as u64).wrapping_mul(2246822519))
                        as usize;
                    let idx = hash % input_len;
                    // Deterministic sign: ±1 based on hash bit
                    let sign = if (hash >> 16) & 1 == 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    sum += sign * hdv.values[idx];
                }
                sum * scale
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).unwrap();
        assert!(encoder.attention_weights.len() > 0);
        assert_eq!(encoder.stats.total_cycles, 0);
    }

    #[test]
    fn test_encoding_produces_result() {
        let mut encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).unwrap();
        let result = encoder.encode("The cause leads to effect");

        assert_eq!(result.hdv.values.len(), HDC_DIMENSION);
        assert!(result.prediction_error >= 0.0);
        assert!(result.prediction_error <= 1.0);
    }

    #[test]
    fn test_prediction_reduces_error() {
        let mut encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).unwrap();

        // First encoding (no prediction)
        let result1 = encoder.encode("test input");
        let error1 = result1.prediction_error;

        // Set prediction based on first encoding
        let compressed = encoder.compress_for_ltc(&result1.hdv, 64);
        encoder.set_prediction(compressed);

        // Second encoding (with prediction)
        let result2 = encoder.encode("test input");
        let error2 = result2.prediction_error;

        // Same input should have lower error with self-prediction
        assert!(error2 < error1, "Error should decrease with prediction");
    }

    #[test]
    fn test_attention_weights_diverge() {
        let mut encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig {
            error_threshold: 0.0, // Always update
            attention_lr: 0.5,    // High learning rate for test
            ..Default::default()
        })
        .unwrap();

        // Encode several times with high error
        for _ in 0..10 {
            encoder.encode("cause effect action");
        }

        // Attention should have diverged from uniform
        assert!(
            encoder.stats.attention_variance > 0.0,
            "Attention should diverge from uniform"
        );
    }

    #[test]
    fn test_primitive_detection() {
        let mut encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).unwrap();
        let detected = encoder.detect_primitives("The cause leads to an effect");

        // Should detect causal primitives
        assert!(
            detected.contains(&"CAUSE".to_string()) || detected.contains(&"EFFECT".to_string()),
            "Should detect causal primitives"
        );
    }

    #[test]
    fn test_stats_accumulate() {
        let mut encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).unwrap();

        for _ in 0..5 {
            encoder.encode("test");
        }

        assert_eq!(encoder.stats.total_cycles, 5);
        assert!(encoder.stats.cumulative_error > 0.0);
    }
}
