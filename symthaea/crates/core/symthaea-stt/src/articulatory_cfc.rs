// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learned Articulatory CfC Model
//!
//! Loads trained weights from Python and runs inference.
//! The model has 3 parallel heads:
//! - Voicing: 2 classes (Voiced/Voiceless)
//! - Manner: 7 classes (Stop, Fricative, Affricate, Nasal, Liquid, Glide, Vowel)
//! - Place: 10 classes (Bilabial, Labiodental, Dental, Alveolar, Palatal, Velar, Glottal, Front, Central, Back)

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::articulatory::{ArticulatoryHDC, Manner, Place, Voicing};
use crate::hdc::{BundleAccumulator, HV16};

/// Loaded CfC weights for articulatory detection
#[derive(Debug, Clone)]
pub struct CfCWeights {
    // CfC backbone
    pub w_in_weight: Vec<Vec<f32>>,  // [hidden_size, input_size]
    pub w_in_bias: Vec<f32>,         // [hidden_size]
    pub w_rec_weight: Vec<Vec<f32>>, // [hidden_size, hidden_size]
    pub w_rec_bias: Vec<f32>,        // [hidden_size]
    pub log_tau: Vec<f32>,           // [hidden_size]

    // Voicing head
    pub voicing_layer0_weight: Vec<Vec<f32>>, // [32, hidden_size]
    pub voicing_layer0_bias: Vec<f32>,        // [32]
    pub voicing_layer2_weight: Vec<Vec<f32>>, // [2, 32]
    pub voicing_layer2_bias: Vec<f32>,        // [2]

    // Manner head
    pub manner_layer0_weight: Vec<Vec<f32>>, // [32, hidden_size]
    pub manner_layer0_bias: Vec<f32>,        // [32]
    pub manner_layer2_weight: Vec<Vec<f32>>, // [7, 32]
    pub manner_layer2_bias: Vec<f32>,        // [7]

    // Place head
    pub place_layer0_weight: Vec<Vec<f32>>, // [32, hidden_size]
    pub place_layer0_bias: Vec<f32>,        // [32]
    pub place_layer2_weight: Vec<Vec<f32>>, // [10, 32]
    pub place_layer2_bias: Vec<f32>,        // [10]

    // Config
    pub input_size: usize,
    pub hidden_size: usize,
    pub tau_min: f32,
    pub tau_max: f32,
}

impl CfCWeights {
    /// Load weights from a binary file exported by Python training script
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open weights: {e}"))?;
        let mut reader = BufReader::new(file);

        // Read metadata length
        let mut len_bytes = [0u8; 4];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| format!("Failed to read header: {e}"))?;
        let metadata_len = u32::from_le_bytes(len_bytes) as usize;

        // Read metadata JSON
        let mut metadata_bytes = vec![0u8; metadata_len];
        reader
            .read_exact(&mut metadata_bytes)
            .map_err(|e| format!("Failed to read metadata: {e}"))?;

        // Remove null terminator and parse JSON
        let metadata_str = String::from_utf8_lossy(&metadata_bytes)
            .trim_end_matches('\0')
            .to_string();

        let metadata: serde_json::Value = serde_json::from_str(&metadata_str)
            .map_err(|e| format!("Failed to parse metadata: {e}"))?;

        let input_size = metadata["input_size"].as_u64().unwrap_or(40) as usize;
        let hidden_size = metadata["hidden_size"].as_u64().unwrap_or(64) as usize;
        let tau_min = metadata["tau_min"].as_f64().unwrap_or(0.005) as f32;
        let tau_max = metadata["tau_max"].as_f64().unwrap_or(0.1) as f32;

        // Read all weights as float32
        let mut all_weights = Vec::new();
        loop {
            let mut float_bytes = [0u8; 4];
            match reader.read_exact(&mut float_bytes) {
                Ok(_) => all_weights.push(f32::from_le_bytes(float_bytes)),
                Err(_) => break,
            }
        }

        // Parse weights based on metadata
        let weights_meta = &metadata["weights"];

        let parse_2d = |name: &str| -> Result<Vec<Vec<f32>>, String> {
            let meta = &weights_meta[name];
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .ok_or_else(|| format!("Missing shape for {name}"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let offset = meta["offset"].as_u64().unwrap_or(0) as usize;
            let _size = meta["size"].as_u64().unwrap_or(0) as usize;

            if shape.len() != 2 {
                return Err(format!("Expected 2D array for {name}"));
            }

            let mut result = Vec::with_capacity(shape[0]);
            for i in 0..shape[0] {
                let start = offset + i * shape[1];
                let end = start + shape[1];
                if end > all_weights.len() {
                    return Err(format!("Weight data out of bounds for {name}"));
                }
                result.push(all_weights[start..end].to_vec());
            }
            Ok(result)
        };

        let parse_1d = |name: &str| -> Result<Vec<f32>, String> {
            let meta = &weights_meta[name];
            let offset = meta["offset"].as_u64().unwrap_or(0) as usize;
            let size = meta["size"].as_u64().unwrap_or(0) as usize;

            if offset + size > all_weights.len() {
                return Err(format!("Weight data out of bounds for {name}"));
            }
            Ok(all_weights[offset..offset + size].to_vec())
        };

        Ok(Self {
            w_in_weight: parse_2d("cfc_W_in_weight")?,
            w_in_bias: parse_1d("cfc_W_in_bias")?,
            w_rec_weight: parse_2d("cfc_W_rec_weight")?,
            w_rec_bias: parse_1d("cfc_W_rec_bias")?,
            log_tau: parse_1d("cfc_log_tau")?,

            voicing_layer0_weight: parse_2d("voicing_layer0_weight")?,
            voicing_layer0_bias: parse_1d("voicing_layer0_bias")?,
            voicing_layer2_weight: parse_2d("voicing_layer2_weight")?,
            voicing_layer2_bias: parse_1d("voicing_layer2_bias")?,

            manner_layer0_weight: parse_2d("manner_layer0_weight")?,
            manner_layer0_bias: parse_1d("manner_layer0_bias")?,
            manner_layer2_weight: parse_2d("manner_layer2_weight")?,
            manner_layer2_bias: parse_1d("manner_layer2_bias")?,

            place_layer0_weight: parse_2d("place_layer0_weight")?,
            place_layer0_bias: parse_1d("place_layer0_bias")?,
            place_layer2_weight: parse_2d("place_layer2_weight")?,
            place_layer2_bias: parse_1d("place_layer2_bias")?,

            input_size,
            hidden_size,
            tau_min,
            tau_max,
        })
    }
}

/// Learned CfC-based articulatory detector
pub struct LearnedArticulatoryDetector {
    weights: CfCWeights,
    hidden_state: Vec<f32>,
    hdc: ArticulatoryHDC,
}

impl LearnedArticulatoryDetector {
    /// Create detector with loaded weights
    pub fn new(weights: CfCWeights) -> Self {
        let hidden_size = weights.hidden_size;
        Self {
            weights,
            hidden_state: vec![0.0; hidden_size],
            hdc: ArticulatoryHDC::new(),
        }
    }

    /// Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let weights = CfCWeights::load(path)?;
        Ok(Self::new(weights))
    }

    /// Reset hidden state
    pub fn reset(&mut self) {
        for h in &mut self.hidden_state {
            *h = 0.0;
        }
    }

    /// Process one frame of mel features
    ///
    /// Returns (voicing_probs, manner_probs, place_probs)
    pub fn forward(&mut self, mel_features: &[f32], dt: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // CfC forward pass
        let h_new = self.cfc_forward(mel_features, dt);
        self.hidden_state = h_new.clone();

        // Head forward passes
        let voicing_logits = self.head_forward(
            &h_new,
            &self.weights.voicing_layer0_weight,
            &self.weights.voicing_layer0_bias,
            &self.weights.voicing_layer2_weight,
            &self.weights.voicing_layer2_bias,
        );
        let manner_logits = self.head_forward(
            &h_new,
            &self.weights.manner_layer0_weight,
            &self.weights.manner_layer0_bias,
            &self.weights.manner_layer2_weight,
            &self.weights.manner_layer2_bias,
        );
        let place_logits = self.head_forward(
            &h_new,
            &self.weights.place_layer0_weight,
            &self.weights.place_layer0_bias,
            &self.weights.place_layer2_weight,
            &self.weights.place_layer2_bias,
        );

        // Softmax
        (
            Self::softmax(&voicing_logits),
            Self::softmax(&manner_logits),
            Self::softmax(&place_logits),
        )
    }

    /// CfC forward pass with closed-form solution
    fn cfc_forward(&self, x: &[f32], dt: f32) -> Vec<f32> {
        let hidden_size = self.weights.hidden_size;
        let mut h_new = vec![0.0; hidden_size];

        // Compute tau (clamped)
        let tau: Vec<f32> = self
            .weights
            .log_tau
            .iter()
            .map(|lt| lt.exp().clamp(self.weights.tau_min, self.weights.tau_max))
            .collect();

        // Compute decay: exp(-dt/tau)
        let decay: Vec<f32> = tau.iter().map(|t| (-dt / t).exp()).collect();

        // Compute target = tanh(W_in * x + W_rec * h + bias)
        let mut pre_activation = vec![0.0; hidden_size];

        // W_in * x
        for i in 0..hidden_size {
            for j in 0..x.len().min(self.weights.w_in_weight[i].len()) {
                pre_activation[i] += self.weights.w_in_weight[i][j] * x[j];
            }
            pre_activation[i] += self.weights.w_in_bias[i];
        }

        // W_rec * h
        for i in 0..hidden_size {
            for j in 0..hidden_size {
                pre_activation[i] += self.weights.w_rec_weight[i][j] * self.hidden_state[j];
            }
            pre_activation[i] += self.weights.w_rec_bias[i];
        }

        // Tanh activation
        let target: Vec<f32> = pre_activation.iter().map(|x| x.tanh()).collect();

        // Closed-form update: h_new = h * decay + target * (1 - decay)
        for i in 0..hidden_size {
            h_new[i] = self.hidden_state[i] * decay[i] + target[i] * (1.0 - decay[i]);
        }

        h_new
    }

    /// Head forward pass (Linear -> ReLU -> Linear)
    fn head_forward(
        &self,
        h: &[f32],
        w0: &[Vec<f32>],
        b0: &[f32],
        w2: &[Vec<f32>],
        b2: &[f32],
    ) -> Vec<f32> {
        // Layer 0: Linear
        let mut hidden = vec![0.0; w0.len()];
        for i in 0..w0.len() {
            for j in 0..h.len().min(w0[i].len()) {
                hidden[i] += w0[i][j] * h[j];
            }
            hidden[i] += b0[i];
        }

        // ReLU
        for x in &mut hidden {
            *x = x.max(0.0);
        }

        // Layer 2: Linear
        let mut output = vec![0.0; w2.len()];
        for i in 0..w2.len() {
            for j in 0..hidden.len().min(w2[i].len()) {
                output[i] += w2[i][j] * hidden[j];
            }
            output[i] += b2[i];
        }

        output
    }

    /// Softmax normalization.
    ///
    /// NOTE: The canonical implementation lives in `symthaea_core::math::softmax`.
    /// This crate does not depend on `symthaea-core`, so we keep a local copy.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|x| x / sum).collect()
    }

    /// Detect articulatory features and return bound HV
    pub fn detect(&mut self, mel_features: &[f32], dt: f32) -> HV16 {
        let (voicing_probs, manner_probs, place_probs) = self.forward(mel_features, dt);

        // Get argmax for each stream
        let voicing = if voicing_probs[1] > voicing_probs[0] {
            Voicing::Voiced
        } else {
            Voicing::Voiceless
        };

        let manner = Self::argmax(&manner_probs);
        let manner = match manner {
            0 => Manner::Stop,
            1 => Manner::Fricative,
            2 => Manner::Affricate,
            3 => Manner::Nasal,
            4 => Manner::Liquid,
            5 => Manner::Glide,
            _ => Manner::Vowel,
        };

        let place = Self::argmax(&place_probs);
        let place = match place {
            0 => Place::Bilabial,
            1 => Place::Labiodental,
            2 => Place::Dental,
            3 => Place::Alveolar,
            4 => Place::Palatal,
            5 => Place::Velar,
            6 => Place::Glottal,
            7 => Place::Front,
            8 => Place::Central,
            _ => Place::Back,
        };

        // Bind features with roles and bundle
        let voicing_hv = self.hdc.voicing_hv(voicing);
        let manner_hv = self.hdc.manner_hv(manner);
        let place_hv = self.hdc.place_hv(place);

        let bound_v = self.hdc.role_voicing.bind(voicing_hv);
        let bound_m = self.hdc.role_manner.bind(manner_hv);
        let bound_p = self.hdc.role_place.bind(place_hv);

        let mut acc = BundleAccumulator::new();
        acc.add(&bound_v);
        acc.add(&bound_m);
        acc.add(&bound_p);

        acc.finalize()
    }

    /// Detect with soft (probabilistic) HV output
    ///
    /// Instead of hard argmax, creates weighted bundle of all possible features
    pub fn detect_soft(&mut self, mel_features: &[f32], dt: f32) -> HV16 {
        let (voicing_probs, manner_probs, place_probs) = self.forward(mel_features, dt);

        let mut acc = BundleAccumulator::new();

        // Weighted voicing
        let voiced_hv = self
            .hdc
            .role_voicing
            .bind(self.hdc.voicing_hv(Voicing::Voiced));
        let voiceless_hv = self
            .hdc
            .role_voicing
            .bind(self.hdc.voicing_hv(Voicing::Voiceless));

        let voiced_weight = (voicing_probs[1] * 10.0) as usize;
        let voiceless_weight = (voicing_probs[0] * 10.0) as usize;

        for _ in 0..voiced_weight {
            acc.add(&voiced_hv);
        }
        for _ in 0..voiceless_weight {
            acc.add(&voiceless_hv);
        }

        // Weighted manner (simplified - just top 2)
        let mut manner_indices: Vec<(usize, f32)> = manner_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        manner_indices.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (idx, prob) in manner_indices.iter().take(2) {
            let manner = match idx {
                0 => Manner::Stop,
                1 => Manner::Fricative,
                2 => Manner::Affricate,
                3 => Manner::Nasal,
                4 => Manner::Liquid,
                5 => Manner::Glide,
                _ => Manner::Vowel,
            };
            let hv = self.hdc.role_manner.bind(self.hdc.manner_hv(manner));
            let weight = (*prob * 10.0) as usize;
            for _ in 0..weight {
                acc.add(&hv);
            }
        }

        // Weighted place (top 2)
        let mut place_indices: Vec<(usize, f32)> = place_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        place_indices.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (idx, prob) in place_indices.iter().take(2) {
            let place = match idx {
                0 => Place::Bilabial,
                1 => Place::Labiodental,
                2 => Place::Dental,
                3 => Place::Alveolar,
                4 => Place::Palatal,
                5 => Place::Velar,
                6 => Place::Glottal,
                7 => Place::Front,
                8 => Place::Central,
                _ => Place::Back,
            };
            let hv = self.hdc.role_place.bind(self.hdc.place_hv(place));
            let weight = (*prob * 10.0) as usize;
            for _ in 0..weight {
                acc.add(&hv);
            }
        }

        acc.finalize()
    }

    fn argmax(probs: &[f32]) -> usize {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Softmax with temperature scaling.
    /// Temperature > 1 flattens the distribution (more uniform).
    /// Temperature < 1 sharpens it (more confident).
    ///
    /// NOTE: The canonical implementation lives in `symthaea_core::math::softmax_with_temperature`.
    /// This crate does not depend on `symthaea-core`, so we keep a local copy.
    fn softmax_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
        let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
        Self::softmax(&scaled)
    }

    /// Forward pass with temperature scaling
    /// Higher temperature = more uniform predictions
    pub fn forward_temperature(
        &mut self,
        mel_features: &[f32],
        dt: f32,
        temperature: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Update hidden state with CfC dynamics
        let h_new = self.cfc_forward(mel_features, dt);
        self.hidden_state = h_new.clone();

        // Compute logits from classification heads
        let voicing_logits = self.head_forward(
            &h_new,
            &self.weights.voicing_layer0_weight,
            &self.weights.voicing_layer0_bias,
            &self.weights.voicing_layer2_weight,
            &self.weights.voicing_layer2_bias,
        );
        let manner_logits = self.head_forward(
            &h_new,
            &self.weights.manner_layer0_weight,
            &self.weights.manner_layer0_bias,
            &self.weights.manner_layer2_weight,
            &self.weights.manner_layer2_bias,
        );
        let place_logits = self.head_forward(
            &h_new,
            &self.weights.place_layer0_weight,
            &self.weights.place_layer0_bias,
            &self.weights.place_layer2_weight,
            &self.weights.place_layer2_bias,
        );

        // Softmax with temperature
        (
            Self::softmax_temperature(&voicing_logits, temperature),
            Self::softmax_temperature(&manner_logits, temperature),
            Self::softmax_temperature(&place_logits, temperature),
        )
    }

    /// Detect with temperature scaling for Place classifier
    /// Uses higher temperature to reduce overconfidence
    pub fn detect_balanced(
        &mut self,
        mel_features: &[f32],
        dt: f32,
        place_temperature: f32,
    ) -> HV16 {
        // Normal forward for voicing/manner, temperature-scaled for place
        let h_new = self.cfc_forward(mel_features, dt);
        self.hidden_state = h_new.clone();

        let voicing_logits = self.head_forward(
            &h_new,
            &self.weights.voicing_layer0_weight,
            &self.weights.voicing_layer0_bias,
            &self.weights.voicing_layer2_weight,
            &self.weights.voicing_layer2_bias,
        );
        let manner_logits = self.head_forward(
            &h_new,
            &self.weights.manner_layer0_weight,
            &self.weights.manner_layer0_bias,
            &self.weights.manner_layer2_weight,
            &self.weights.manner_layer2_bias,
        );
        let place_logits = self.head_forward(
            &h_new,
            &self.weights.place_layer0_weight,
            &self.weights.place_layer0_bias,
            &self.weights.place_layer2_weight,
            &self.weights.place_layer2_bias,
        );

        let voicing_probs = Self::softmax(&voicing_logits);
        let manner_probs = Self::softmax(&manner_logits);
        let place_probs = Self::softmax_temperature(&place_logits, place_temperature);

        // Get argmax
        let voicing = if voicing_probs[1] > voicing_probs[0] {
            Voicing::Voiced
        } else {
            Voicing::Voiceless
        };

        let manner = Self::argmax(&manner_probs);
        let manner = match manner {
            0 => Manner::Stop,
            1 => Manner::Fricative,
            2 => Manner::Affricate,
            3 => Manner::Nasal,
            4 => Manner::Liquid,
            5 => Manner::Glide,
            _ => Manner::Vowel,
        };

        let place = Self::argmax(&place_probs);
        let place = match place {
            0 => Place::Bilabial,
            1 => Place::Labiodental,
            2 => Place::Dental,
            3 => Place::Alveolar,
            4 => Place::Palatal,
            5 => Place::Velar,
            6 => Place::Glottal,
            7 => Place::Front,
            8 => Place::Central,
            _ => Place::Back,
        };

        // Bind features
        let voicing_hv = self.hdc.voicing_hv(voicing);
        let manner_hv = self.hdc.manner_hv(manner);
        let place_hv = self.hdc.place_hv(place);

        let bound_v = self.hdc.role_voicing.bind(voicing_hv);
        let bound_m = self.hdc.role_manner.bind(manner_hv);
        let bound_p = self.hdc.role_place.bind(place_hv);

        let mut acc = BundleAccumulator::new();
        acc.add(&bound_v);
        acc.add(&bound_m);
        acc.add(&bound_p);

        acc.finalize()
    }

    /// Get the underlying HDC system
    pub fn hdc(&self) -> &ArticulatoryHDC {
        &self.hdc
    }

    /// Get raw classifier outputs (for analysis/debugging)
    pub fn get_raw_outputs(&mut self, mel_features: &[f32], dt: f32) -> (usize, usize, usize) {
        let (voicing_probs, manner_probs, place_probs) = self.forward(mel_features, dt);

        let voicing = if voicing_probs[1] > voicing_probs[0] {
            1
        } else {
            0
        };
        let manner = Self::argmax(&manner_probs);
        let place = Self::argmax(&place_probs);

        (voicing, manner, place)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = LearnedArticulatoryDetector::softmax(&logits);

        // Sum should be 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Last should be highest
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let probs = LearnedArticulatoryDetector::softmax(&logits);

        // All equal inputs should produce uniform distribution
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        for p in &probs {
            assert!((*p - 0.25).abs() < 0.001, "expected 0.25, got {p}");
        }
    }

    #[test]
    fn test_softmax_single_element() {
        let logits = vec![5.0];
        let probs = LearnedArticulatoryDetector::softmax(&logits);

        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_large_values_numerical_stability() {
        // Large logits should not overflow due to max subtraction
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = LearnedArticulatoryDetector::softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum = {sum}");
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
        // No NaN or Inf
        for p in &probs {
            assert!(p.is_finite(), "softmax produced non-finite value");
        }
    }

    #[test]
    fn test_softmax_negative_logits() {
        let logits = vec![-3.0, -2.0, -1.0];
        let probs = LearnedArticulatoryDetector::softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_temperature_high() {
        // High temperature should flatten the distribution
        let logits = vec![1.0, 2.0, 3.0];
        let probs_normal = LearnedArticulatoryDetector::softmax(&logits);
        let probs_high_temp = LearnedArticulatoryDetector::softmax_temperature(&logits, 10.0);

        // High temperature should make distribution more uniform
        let max_normal = probs_normal
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_normal = probs_normal.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_temp = probs_high_temp
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_temp = probs_high_temp
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);

        let range_normal = max_normal - min_normal;
        let range_temp = max_temp - min_temp;
        assert!(
            range_temp < range_normal,
            "high temp range {range_temp} should be < normal range {range_normal}"
        );
    }

    #[test]
    fn test_softmax_temperature_low() {
        // Low temperature should sharpen the distribution
        let logits = vec![1.0, 2.0, 3.0];
        let probs_normal = LearnedArticulatoryDetector::softmax(&logits);
        let probs_low_temp = LearnedArticulatoryDetector::softmax_temperature(&logits, 0.1);

        // Low temperature should make winner more dominant
        assert!(
            probs_low_temp[2] > probs_normal[2],
            "low temp peak {} should be > normal peak {}",
            probs_low_temp[2],
            probs_normal[2]
        );
    }

    #[test]
    fn test_argmax_basic() {
        assert_eq!(LearnedArticulatoryDetector::argmax(&[0.1, 0.5, 0.3]), 1);
        assert_eq!(LearnedArticulatoryDetector::argmax(&[0.9, 0.1, 0.0]), 0);
        assert_eq!(LearnedArticulatoryDetector::argmax(&[0.0, 0.0, 1.0]), 2);
    }

    #[test]
    fn test_argmax_empty() {
        assert_eq!(LearnedArticulatoryDetector::argmax(&[]), 0);
    }

    fn make_test_weights(input_size: usize, hidden_size: usize) -> CfCWeights {
        CfCWeights {
            w_in_weight: vec![vec![0.01; input_size]; hidden_size],
            w_in_bias: vec![0.0; hidden_size],
            w_rec_weight: vec![vec![0.01; hidden_size]; hidden_size],
            w_rec_bias: vec![0.0; hidden_size],
            log_tau: vec![-2.0; hidden_size], // tau = exp(-2) ~ 0.135
            voicing_layer0_weight: vec![vec![0.1; hidden_size]; 32],
            voicing_layer0_bias: vec![0.0; 32],
            voicing_layer2_weight: vec![vec![0.1; 32]; 2],
            voicing_layer2_bias: vec![0.0; 2],
            manner_layer0_weight: vec![vec![0.1; hidden_size]; 32],
            manner_layer0_bias: vec![0.0; 32],
            manner_layer2_weight: vec![vec![0.1; 32]; 7],
            manner_layer2_bias: vec![0.0; 7],
            place_layer0_weight: vec![vec![0.1; hidden_size]; 32],
            place_layer0_bias: vec![0.0; 32],
            place_layer2_weight: vec![vec![0.1; 32]; 10],
            place_layer2_bias: vec![0.0; 10],
            input_size,
            hidden_size,
            tau_min: 0.005,
            tau_max: 0.1,
        }
    }

    #[test]
    fn test_detector_creation_and_reset() {
        let weights = make_test_weights(40, 8);
        let mut detector = LearnedArticulatoryDetector::new(weights);

        // Initial hidden state should be zeros
        assert!(detector.hidden_state.iter().all(|&h| h == 0.0));

        // Forward pass should change hidden state
        let mel = vec![0.5; 40];
        detector.forward(&mel, 0.01);
        assert!(detector.hidden_state.iter().any(|&h| h != 0.0));

        // Reset should zero it again
        detector.reset();
        assert!(detector.hidden_state.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn test_forward_produces_valid_probabilities() {
        let weights = make_test_weights(40, 8);
        let mut detector = LearnedArticulatoryDetector::new(weights);

        let mel = vec![0.5; 40];
        let (voicing, manner, place) = detector.forward(&mel, 0.01);

        // Check voicing probabilities
        assert_eq!(voicing.len(), 2);
        let v_sum: f32 = voicing.iter().sum();
        assert!((v_sum - 1.0).abs() < 0.01, "voicing sum = {v_sum}");

        // Check manner probabilities
        assert_eq!(manner.len(), 7);
        let m_sum: f32 = manner.iter().sum();
        assert!((m_sum - 1.0).abs() < 0.01, "manner sum = {m_sum}");

        // Check place probabilities
        assert_eq!(place.len(), 10);
        let p_sum: f32 = place.iter().sum();
        assert!((p_sum - 1.0).abs() < 0.01, "place sum = {p_sum}");
    }

    #[test]
    fn test_forward_temperature_produces_valid_probabilities() {
        let weights = make_test_weights(40, 8);
        let mut detector = LearnedArticulatoryDetector::new(weights);

        let mel = vec![0.5; 40];
        let (voicing, manner, place) = detector.forward_temperature(&mel, 0.01, 2.0);

        let v_sum: f32 = voicing.iter().sum();
        assert!((v_sum - 1.0).abs() < 0.01);
        let m_sum: f32 = manner.iter().sum();
        assert!((m_sum - 1.0).abs() < 0.01);
        let p_sum: f32 = place.iter().sum();
        assert!((p_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_detect_returns_valid_hv() {
        let weights = make_test_weights(40, 8);
        let mut detector = LearnedArticulatoryDetector::new(weights);

        let mel = vec![0.5; 40];
        let hv = detector.detect(&mel, 0.01);

        // HV should not be all zeros (it's built from random bases)
        let popcount: u32 = hv.words.iter().map(|w| w.count_ones()).sum();
        assert!(popcount > 0, "detect produced all-zero HV");
    }

    #[test]
    fn test_detect_soft_returns_valid_hv() {
        let weights = make_test_weights(40, 8);
        let mut detector = LearnedArticulatoryDetector::new(weights);

        let mel = vec![0.5; 40];
        let hv = detector.detect_soft(&mel, 0.01);

        let popcount: u32 = hv.words.iter().map(|w| w.count_ones()).sum();
        assert!(popcount > 0, "detect_soft produced all-zero HV");
    }

    #[test]
    fn test_get_raw_outputs() {
        let weights = make_test_weights(40, 8);
        let mut detector = LearnedArticulatoryDetector::new(weights);

        let mel = vec![0.5; 40];
        let (voicing, manner, place) = detector.get_raw_outputs(&mel, 0.01);

        assert!(voicing <= 1, "voicing index out of range: {voicing}");
        assert!(manner <= 6, "manner index out of range: {manner}");
        assert!(place <= 9, "place index out of range: {place}");
    }
}
