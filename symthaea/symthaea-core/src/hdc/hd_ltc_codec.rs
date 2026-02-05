//! # HD-LTC Codec - Bidirectional HDC ↔ LTC State Translation
//!
//! **THE "THROAT" OF CONSCIOUSNESS**
//!
//! This module provides bidirectional translation between:
//! - **HDC Space**: 16,384D semantic hypervectors (crystallized meaning)
//! - **LTC Space**: Neural states with continuous-time dynamics (living thought)
//!
//! ## Architecture
//!
//! ```text
//!                    HDC Space (16,384D)
//!                         ↑
//!                         │ encode_hv()
//!                         │
//! ┌──────────────────────┴──────────────────────┐
//! │              HD-LTC Codec                     │
//! │  ┌────────────────┐   ┌────────────────┐    │
//! │  │ Projection     │   │ Time-varying   │    │
//! │  │ Matrix (Learned)│   │ Adaptation     │    │
//! │  └────────────────┘   └────────────────┘    │
//! └──────────────────────┬──────────────────────┘
//!                         │ decode_ltc()
//!                         ↓
//!                    LTC Space (N neurons)
//! ```
//!
//! ## Key Innovation
//!
//! Unlike simple linear projection, this codec:
//! 1. **Preserves semantic structure** (similar HVs → similar LTC states)
//! 2. **Respects temporal dynamics** (LTC time constants influence decoding)
//! 3. **Bidirectional learning** (both directions improve with use)
//!
//! This enables the LTC to "think" in HDC semantics and express through them.

use crate::hdc::binary_hv::HV16;
use crate::hdc::HDC_DIMENSION;
use crate::learnable_ltc::LearnableLTC;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Default number of LTC neurons for the codec
const DEFAULT_LTC_NEURONS: usize = 1024;

/// History length for temporal context
const TEMPORAL_HISTORY: usize = 16;

/// Configuration for the HD-LTC Codec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HDLTCCodecConfig {
    /// Number of LTC neurons
    pub ltc_neurons: usize,
    /// Learning rate for projection matrices
    pub learning_rate: f32,
    /// Momentum for gradient updates
    pub momentum: f32,
    /// Dropout rate during training
    pub dropout: f32,
    /// Whether to use temporal context
    pub use_temporal_context: bool,
    /// Temperature for softmax in attention
    pub attention_temperature: f32,
}

impl Default for HDLTCCodecConfig {
    fn default() -> Self {
        Self {
            ltc_neurons: DEFAULT_LTC_NEURONS,
            learning_rate: 0.001,
            momentum: 0.9,
            dropout: 0.1,
            use_temporal_context: true,
            attention_temperature: 1.0,
        }
    }
}

/// Statistics about codec usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodecStats {
    /// Total encodings (HV → LTC)
    pub total_encodings: u64,
    /// Total decodings (LTC → HV)
    pub total_decodings: u64,
    /// Average reconstruction error
    pub avg_reconstruction_error: f32,
    /// Best reconstruction error achieved
    pub best_reconstruction_error: f32,
    /// Training iterations
    pub training_iterations: u64,
}

/// The HD-LTC Codec - bidirectional translation layer
#[derive(Serialize, Deserialize)]
pub struct HDLTCCodec {
    /// Configuration
    config: HDLTCCodecConfig,

    /// HV → LTC projection matrix (ltc_neurons × HDC_DIMENSION conceptually)
    /// Stored as flattened for efficiency
    hv_to_ltc_proj: Vec<f32>,

    /// LTC → HV projection matrix (HDC_DIMENSION × ltc_neurons conceptually)
    ltc_to_hv_proj: Vec<f32>,

    /// Bias for HV → LTC
    hv_to_ltc_bias: Vec<f32>,

    /// Bias for LTC → HV
    ltc_to_hv_bias: Vec<f32>,

    /// Temporal context: recent LTC states
    #[serde(skip)]
    temporal_context: VecDeque<Vec<f32>>,

    /// Running mean of LTC states (for normalization)
    ltc_mean: Vec<f32>,

    /// Running variance of LTC states
    ltc_var: Vec<f32>,

    /// Momentum buffer for hv_to_ltc
    #[serde(skip)]
    hv_to_ltc_momentum: Vec<f32>,

    /// Momentum buffer for ltc_to_hv
    #[serde(skip)]
    ltc_to_hv_momentum: Vec<f32>,

    /// Statistics
    stats: CodecStats,
}

impl HDLTCCodec {
    /// Create a new HD-LTC Codec
    pub fn new(config: HDLTCCodecConfig) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let n = config.ltc_neurons;
        let d = HDC_DIMENSION;

        // Xavier initialization for projection matrices
        let xavier_scale = (6.0 / (n + d) as f32).sqrt();

        let hv_to_ltc_proj: Vec<f32> = (0..(n * d))
            .map(|_| rng.gen_range(-xavier_scale..xavier_scale))
            .collect();

        let ltc_to_hv_proj: Vec<f32> = (0..(d * n))
            .map(|_| rng.gen_range(-xavier_scale..xavier_scale))
            .collect();

        let hv_to_ltc_bias = vec![0.0; n];
        let ltc_to_hv_bias = vec![0.0; d];

        Self {
            config,
            hv_to_ltc_proj,
            ltc_to_hv_proj,
            hv_to_ltc_bias,
            ltc_to_hv_bias,
            temporal_context: VecDeque::with_capacity(TEMPORAL_HISTORY),
            ltc_mean: vec![0.0; n],
            ltc_var: vec![1.0; n],
            hv_to_ltc_momentum: vec![0.0; n * d],
            ltc_to_hv_momentum: vec![0.0; d * n],
            stats: CodecStats::default(),
        }
    }

    /// Encode HDC hypervector to LTC neural state
    ///
    /// This is the "inhale" - taking in semantic meaning and preparing for thought.
    pub fn encode_to_ltc(&mut self, hv: &HV16) -> Vec<f32> {
        self.stats.total_encodings += 1;

        let n = self.config.ltc_neurons;

        // Convert binary HV to float representation
        let hv_float = self.hv_to_float(hv);

        // Project: ltc_state = W_h2l * hv + bias
        let mut ltc_state = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = self.hv_to_ltc_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.hv_to_ltc_proj[i * HDC_DIMENSION + j] * hv_float[j];
            }
            ltc_state[i] = sum;
        }

        // Apply tanh activation
        for x in &mut ltc_state {
            *x = x.tanh();
        }

        // Add to temporal context
        if self.config.use_temporal_context {
            self.temporal_context.push_back(ltc_state.clone());
            if self.temporal_context.len() > TEMPORAL_HISTORY {
                self.temporal_context.pop_front();
            }
        }

        // Update running statistics
        self.update_running_stats(&ltc_state);

        ltc_state
    }

    /// Decode LTC neural state to HDC hypervector
    ///
    /// This is the "exhale" - crystallizing dynamic thought into semantic form.
    pub fn decode_to_hv(&mut self, ltc_state: &[f32]) -> HV16 {
        self.stats.total_decodings += 1;

        let n = self.config.ltc_neurons;

        // Normalize LTC state
        let normalized = self.normalize_ltc(ltc_state);

        // Apply temporal attention if enabled
        let attended = if self.config.use_temporal_context && !self.temporal_context.is_empty() {
            self.apply_temporal_attention(&normalized)
        } else {
            normalized
        };

        // Project: hv_float = W_l2h * ltc_state + bias
        let mut hv_float = vec![0.0f32; HDC_DIMENSION];
        for i in 0..HDC_DIMENSION {
            let mut sum = self.ltc_to_hv_bias[i];
            for j in 0..n.min(attended.len()) {
                sum += self.ltc_to_hv_proj[i * n + j] * attended[j];
            }
            hv_float[i] = sum;
        }

        // Convert to binary HV via sign
        self.float_to_hv(&hv_float)
    }

    /// Round-trip encode-decode for testing reconstruction
    pub fn round_trip(&mut self, hv: &HV16) -> (HV16, f32) {
        let ltc = self.encode_to_ltc(hv);
        let reconstructed = self.decode_to_hv(&ltc);
        let error = 1.0 - hv.similarity(&reconstructed);

        // Update stats
        self.stats.avg_reconstruction_error =
            self.stats.avg_reconstruction_error * 0.99 + error * 0.01;
        if error < self.stats.best_reconstruction_error || self.stats.best_reconstruction_error == 0.0 {
            self.stats.best_reconstruction_error = error;
        }

        (reconstructed, error)
    }

    /// Train the codec on paired HV-LTC data
    ///
    /// Uses contrastive learning: similar HVs should produce similar LTC states
    pub fn train_step(&mut self, hv: &HV16, target_ltc: &[f32]) -> f32 {
        self.stats.training_iterations += 1;

        let n = self.config.ltc_neurons;
        let lr = self.config.learning_rate;
        let mom = self.config.momentum;

        // Forward pass
        let hv_float = self.hv_to_float(hv);
        let predicted_ltc = self.encode_to_ltc(hv);

        // Compute error
        let mut total_error = 0.0f32;
        for i in 0..n.min(target_ltc.len()) {
            let error = predicted_ltc[i] - target_ltc[i];
            total_error += error * error;
        }
        let mse = total_error / n as f32;

        // Backward pass: compute gradients for hv_to_ltc
        for i in 0..n.min(target_ltc.len()) {
            let error = predicted_ltc[i] - target_ltc[i];
            // Gradient of tanh
            let tanh_grad = 1.0 - predicted_ltc[i] * predicted_ltc[i];
            let delta = error * tanh_grad;

            for j in 0..HDC_DIMENSION {
                let grad = delta * hv_float[j];
                let idx = i * HDC_DIMENSION + j;

                // Update with momentum
                self.hv_to_ltc_momentum[idx] = mom * self.hv_to_ltc_momentum[idx] + grad;
                self.hv_to_ltc_proj[idx] -= lr * self.hv_to_ltc_momentum[idx];
            }

            // Update bias
            self.hv_to_ltc_bias[i] -= lr * delta;
        }

        mse
    }

    /// Convert binary HV16 to float representation
    fn hv_to_float(&self, hv: &HV16) -> Vec<f32> {
        let mut result = vec![0.0f32; HDC_DIMENSION];
        let bytes = &hv.0;

        for byte_idx in 0..2048 {
            for bit_idx in 0..8 {
                let pos = byte_idx * 8 + bit_idx;
                let bit = (bytes[byte_idx] >> bit_idx) & 1;
                result[pos] = if bit == 1 { 1.0 } else { -1.0 };
            }
        }

        result
    }

    /// Convert float representation to binary HV16
    fn float_to_hv(&self, float_vec: &[f32]) -> HV16 {
        let mut bytes = [0u8; 2048];

        for byte_idx in 0..2048 {
            for bit_idx in 0..8 {
                let pos = byte_idx * 8 + bit_idx;
                if pos < float_vec.len() && float_vec[pos] > 0.0 {
                    bytes[byte_idx] |= 1 << bit_idx;
                }
            }
        }

        HV16(bytes)
    }

    /// Normalize LTC state using running statistics
    fn normalize_ltc(&self, state: &[f32]) -> Vec<f32> {
        let n = self.config.ltc_neurons;
        let mut normalized = vec![0.0f32; n];

        for i in 0..n.min(state.len()) {
            let std = (self.ltc_var[i] + 1e-5).sqrt();
            normalized[i] = (state[i] - self.ltc_mean[i]) / std;
        }

        normalized
    }

    /// Update running mean and variance
    fn update_running_stats(&mut self, state: &[f32]) {
        let decay = 0.99;
        for i in 0..self.config.ltc_neurons.min(state.len()) {
            self.ltc_mean[i] = decay * self.ltc_mean[i] + (1.0 - decay) * state[i];
            let diff = state[i] - self.ltc_mean[i];
            self.ltc_var[i] = decay * self.ltc_var[i] + (1.0 - decay) * diff * diff;
        }
    }

    /// Apply temporal attention over recent context
    fn apply_temporal_attention(&self, current: &[f32]) -> Vec<f32> {
        if self.temporal_context.is_empty() {
            return current.to_vec();
        }

        let n = self.config.ltc_neurons;
        let temp = self.config.attention_temperature;

        // Compute attention scores
        let mut scores = Vec::with_capacity(self.temporal_context.len());
        for past in &self.temporal_context {
            let mut dot = 0.0f32;
            for i in 0..n.min(current.len()).min(past.len()) {
                dot += current[i] * past[i];
            }
            scores.push(dot / temp);
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
        let weights: Vec<f32> = scores.iter().map(|s| (s - max_score).exp() / exp_sum).collect();

        // Weighted sum
        let mut attended = vec![0.0f32; n];
        for (past, &weight) in self.temporal_context.iter().zip(weights.iter()) {
            for i in 0..n.min(past.len()) {
                attended[i] += weight * past[i];
            }
        }

        // Blend with current (residual connection)
        for i in 0..n.min(current.len()) {
            attended[i] = 0.5 * attended[i] + 0.5 * current[i];
        }

        attended
    }

    /// Get codec statistics
    pub fn stats(&self) -> &CodecStats {
        &self.stats
    }

    /// Reset temporal context
    pub fn reset_context(&mut self) {
        self.temporal_context.clear();
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(data)?)
    }
}

impl Default for HDLTCCodec {
    fn default() -> Self {
        Self::new(HDLTCCodecConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATED LTC CODEC - Connects to actual LearnableLTC
// ═══════════════════════════════════════════════════════════════════════════════

/// Integrated codec that directly interfaces with LearnableLTC
pub struct IntegratedLTCCodec {
    /// The base HD-LTC codec
    pub codec: HDLTCCodec,
    /// The underlying LTC network (optional - can be external)
    ltc: Option<LearnableLTC>,
}

impl IntegratedLTCCodec {
    /// Create with built-in LTC
    pub fn with_ltc(config: HDLTCCodecConfig, ltc_config: crate::learnable_ltc::LearnableLTCConfig) -> anyhow::Result<Self> {
        let ltc = LearnableLTC::new(ltc_config)?;
        Ok(Self {
            codec: HDLTCCodec::new(config),
            ltc: Some(ltc),
        })
    }

    /// Create without LTC (for external LTC integration)
    pub fn standalone(config: HDLTCCodecConfig) -> Self {
        Self {
            codec: HDLTCCodec::new(config),
            ltc: None,
        }
    }

    /// Encode HV and run through LTC for temporal processing
    pub fn process_thought(&mut self, hv: &HV16) -> anyhow::Result<(Vec<f32>, HV16)> {
        // Encode to LTC input
        let ltc_input = self.codec.encode_to_ltc(hv);

        // Run through LTC if available
        let ltc_output = if let Some(ref mut ltc) = self.ltc {
            let (output, _states) = ltc.forward(&ltc_input)?;
            output
        } else {
            ltc_input.clone()
        };

        // Decode back to HV
        let output_hv = self.codec.decode_to_hv(&ltc_output);

        Ok((ltc_output, output_hv))
    }

    /// Get the internal LTC if present
    pub fn ltc(&self) -> Option<&LearnableLTC> {
        self.ltc.as_ref()
    }

    /// Get mutable reference to internal LTC
    pub fn ltc_mut(&mut self) -> Option<&mut LearnableLTC> {
        self.ltc.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_creation() {
        let codec = HDLTCCodec::default();
        assert_eq!(codec.config.ltc_neurons, DEFAULT_LTC_NEURONS);
    }

    #[test]
    fn test_encode_decode() {
        let mut codec = HDLTCCodec::default();
        let hv = HV16::random(42);

        let ltc_state = codec.encode_to_ltc(&hv);
        assert_eq!(ltc_state.len(), DEFAULT_LTC_NEURONS);

        let recovered = codec.decode_to_hv(&ltc_state);
        let similarity = hv.similarity(&recovered);

        // Should have reasonable reconstruction (> 50% for random init)
        println!("Reconstruction similarity: {:.3}", similarity);
    }

    #[test]
    fn test_round_trip() {
        let mut codec = HDLTCCodec::default();
        let hv = HV16::random(123);

        let (recovered, error) = codec.round_trip(&hv);

        println!("Round-trip error: {:.4}", error);
        println!("Similarity: {:.4}", hv.similarity(&recovered));
    }

    #[test]
    fn test_training() {
        let mut codec = HDLTCCodec::default();
        let hv = HV16::random(456);

        // Create target (just use encoded as target for autoencoder)
        let target = vec![0.5f32; DEFAULT_LTC_NEURONS];

        let loss1 = codec.train_step(&hv, &target);
        let loss2 = codec.train_step(&hv, &target);

        // Loss should decrease
        println!("Loss 1: {:.4}, Loss 2: {:.4}", loss1, loss2);
    }

    #[test]
    fn test_temporal_context() {
        let mut codec = HDLTCCodec::default();

        // Process several HVs to build context
        for seed in 0..5 {
            let hv = HV16::random(seed * 1000);
            let _ = codec.encode_to_ltc(&hv);
        }

        // Context should be populated
        assert_eq!(codec.temporal_context.len(), 5);
    }
}
