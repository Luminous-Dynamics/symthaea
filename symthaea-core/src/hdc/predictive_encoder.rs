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

use crate::hdc::real_hv::RealHV;
use crate::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};
use crate::hdc::text_encoder::{TextEncoder, TextEncoderConfig};
use crate::hdc::HDC_DIMENSION;

use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};

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
    pub hdv: RealHV,

    /// Prediction error from this cycle
    pub prediction_error: f32,

    /// Primitives that were detected in input
    pub detected_primitives: Vec<String>,

    /// Current attention weights (for monitoring)
    pub attention_snapshot: HashMap<String, f32>,
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
    primitive_system: PrimitiveSystem,

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
}

impl PredictiveHdcEncoder {
    /// Create a new predictive encoder
    pub fn new(config: PredictiveEncoderConfig) -> Self {
        let primitive_system = PrimitiveSystem::new();
        let text_encoder = TextEncoder::new(TextEncoderConfig {
            dimension: config.dimension,
            ..Default::default()
        }).expect("Failed to create text encoder for predictive encoder");

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

        Self {
            config,
            primitive_system,
            text_encoder,
            attention_weights,
            predicted_hdv: None,
            error_history: VecDeque::with_capacity(50),
            stats: EncoderStats::default(),
            primitive_names,
        }
    }

    /// Encode input with attention modulation from prediction
    pub fn encode(&mut self, input: &str) -> EncodingResult {
        self.stats.total_cycles += 1;

        // 1. Get base encoding via text encoder (uses primitives internally)
        let base_encoding = match self.text_encoder.encode_with_primitives(input, &self.primitive_system) {
            Ok(enc) => enc,
            Err(_) => {
                // Fallback to sentence encoding if primitive encoding fails
                self.text_encoder.encode_sentence(input)
                    .unwrap_or_else(|_| vec![0i8; self.config.dimension])
            }
        };

        // Convert bipolar to real-valued
        let base_hdv = self.bipolar_to_real(&base_encoding);

        // 2. Detect which primitives are in the input
        let detected_primitives = self.detect_primitives(input);

        // 3. Apply attention modulation
        let attended_hdv = self.apply_attention(&base_hdv, &detected_primitives);

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
            attention_snapshot: self.attention_weights.clone(),
        }
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

    /// Convert bipolar encoding to RealHV
    fn bipolar_to_real(&self, bipolar: &[i8]) -> RealHV {
        let values: Vec<f32> = bipolar.iter()
            .map(|&b| b as f32)
            .collect();

        // Pad or truncate to match dimension
        let mut result = vec![0.0f32; self.config.dimension];
        let len = values.len().min(self.config.dimension);
        result[..len].copy_from_slice(&values[..len]);

        RealHV { values: result }
    }

    /// Detect which primitives are relevant to the input
    fn detect_primitives(&self, input: &str) -> Vec<String> {
        let input_lower = input.to_lowercase();
        let mut detected = Vec::new();

        // Check each primitive for presence in input
        for name in &self.primitive_names {
            // Simple heuristic: check if primitive name is in input
            // In production, this would use semantic similarity
            if input_lower.contains(&name.to_lowercase()) {
                detected.push(name.clone());
            }
        }

        // Also check for semantic patterns
        detected.extend(self.detect_semantic_patterns(&input_lower));

        // Deduplicate
        detected.sort();
        detected.dedup();

        detected
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

    /// Apply attention weights to HDV
    fn apply_attention(&self, base_hdv: &RealHV, detected_primitives: &[String]) -> RealHV {
        if detected_primitives.is_empty() {
            return base_hdv.clone();
        }

        // Compute composite attention weight from detected primitives
        let total_attention: f32 = detected_primitives.iter()
            .filter_map(|name| self.attention_weights.get(name))
            .sum();

        let avg_attention = if !detected_primitives.is_empty() {
            total_attention / detected_primitives.len() as f32
        } else {
            1.0
        };

        // Scale the HDV by attention (this modulates magnitude)
        base_hdv.scale(avg_attention)
    }

    /// Compute prediction error between current HDV and LTC's prediction
    ///
    /// IMPORTANT: Comparison happens in the COMPRESSED space (LTC output dimension)
    /// to ensure we're comparing apples to apples.
    fn compute_prediction_error(&self, current_hdv: &RealHV) -> f32 {
        match &self.predicted_hdv {
            Some(predicted) => {
                // Compress current HDV to same dimension as prediction (NOT expand prediction!)
                // This ensures we compare in the same space the LTC operates in
                let compressed_current = self.compress_for_ltc(current_hdv, predicted.len());

                // Compute normalized L2 distance in compressed space
                let diff: f32 = compressed_current.iter()
                    .zip(predicted.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                let norm_current: f32 = compressed_current.iter().map(|x| x * x).sum::<f32>().sqrt();
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
        if compressed.len() >= self.config.dimension {
            return compressed[..self.config.dimension].to_vec();
        }

        // Simple expansion: repeat values to fill dimension
        let repeat_factor = self.config.dimension / compressed.len();
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
                *weight = (*weight + delta).clamp(
                    self.config.min_attention,
                    self.config.max_attention
                );
            }
        }

        // Slightly decrease attention on non-detected primitives
        // This creates competition between primitives
        for name in &self.primitive_names {
            if !detected_primitives.contains(name) {
                if let Some(weight) = self.attention_weights.get_mut(name) {
                    let delta = self.config.attention_lr * error * 0.1; // Smaller decrease
                    *weight = (*weight - delta).clamp(
                        self.config.min_attention,
                        self.config.max_attention
                    );
                }
            }
        }
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

        // Compute attention variance (emergence metric)
        let weights: Vec<f32> = self.attention_weights.values().cloned().collect();
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance: f32 = weights.iter()
            .map(|w| (w - mean).powi(2))
            .sum::<f32>() / weights.len() as f32;
        self.stats.attention_variance = variance;

        // Count diverged primitives
        self.stats.diverged_primitives = weights.iter()
            .filter(|&&w| (w - self.config.initial_attention).abs() > 0.1)
            .count();

        // Update cumulative and high-error counts
        self.stats.cumulative_error += error;
        if error > self.config.error_threshold {
            self.stats.high_error_cycles += 1;
        }
    }

    /// Get compressed representation for LTC input
    pub fn compress_for_ltc(&self, hdv: &RealHV, output_dim: usize) -> Vec<f32> {
        // Downsample by taking evenly spaced values
        let step = hdv.values.len() / output_dim;
        hdv.values.iter()
            .step_by(step)
            .take(output_dim)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default());
        assert!(encoder.attention_weights.len() > 0);
        assert_eq!(encoder.stats.total_cycles, 0);
    }

    #[test]
    fn test_encoding_produces_result() {
        let mut encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default());
        let result = encoder.encode("The cause leads to effect");

        assert_eq!(result.hdv.values.len(), HDC_DIMENSION);
        assert!(result.prediction_error >= 0.0);
        assert!(result.prediction_error <= 1.0);
    }

    #[test]
    fn test_prediction_reduces_error() {
        let mut encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default());

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
            attention_lr: 0.5, // High learning rate for test
            ..Default::default()
        });

        // Encode several times with high error
        for _ in 0..10 {
            encoder.encode("cause effect action");
        }

        // Attention should have diverged from uniform
        assert!(encoder.stats.attention_variance > 0.0,
            "Attention should diverge from uniform");
    }

    #[test]
    fn test_primitive_detection() {
        let encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default());
        let detected = encoder.detect_primitives("The cause leads to an effect");

        // Should detect causal primitives
        assert!(detected.contains(&"CAUSE".to_string()) ||
                detected.contains(&"EFFECT".to_string()),
            "Should detect causal primitives");
    }

    #[test]
    fn test_stats_accumulate() {
        let mut encoder = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default());

        for _ in 0..5 {
            encoder.encode("test");
        }

        assert_eq!(encoder.stats.total_cycles, 5);
        assert!(encoder.stats.cumulative_error > 0.0);
    }
}
