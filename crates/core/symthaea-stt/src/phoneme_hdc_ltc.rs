// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phoneme HDC-LTC — Multi-Neuron Network for Speech Recognition
//!
//! Uses HdcLtcUnifiedNetwork (3 layers × 8 neurons = 24 neurons, same as Broca)
//! with weight-tied cosine decoding against phoneme prototypes.
//!
//! Architecture:
//! ```text
//! mel → AudioHdcEncoder → 16,384D input_hv
//!    → HdcLtcUnifiedNetwork (24 neurons, closed-form evolution)
//!    → output = network.output().normalize()
//!    → logits[p] = scale × cosine(output, prototype[p])
//!    → argmax → predicted phoneme
//! ```
//!
//! GPU-first: logit computation uses candle CUDA when available,
//! falls back to CPU cosine similarity.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

#[cfg(feature = "gpu")]
use candle_core::{DType, Device, Tensor};

/// Multi-neuron phoneme recognizer using HdcLtcUnifiedNetwork.
///
/// Same architecture as Broca's language generation pipeline:
/// 3 layers × 8 neurons = 24 neurons with layer bindings and skip connections.
pub struct PhonemeHdcLtc {
    /// Multi-layer unified network (24 neurons)
    network: HdcLtcUnifiedNetwork,
    /// Phoneme prototypes: (label, target_hv)
    prototypes: Vec<(String, ContinuousHV)>,
    /// Logit scaling (CLIP-style, default 20.0)
    logit_scale: f32,
    /// GPU prototype matrix for accelerated logit computation
    #[cfg(feature = "gpu")]
    gpu_protos: Option<GpuProtoCache>,
}

/// GPU-resident prototype matrix for batched logit computation.
#[cfg(feature = "gpu")]
struct GpuProtoCache {
    /// Pre-normalized prototype matrix [n_phonemes, HDC_DIMENSION] on device
    matrix: Tensor,
    device: Device,
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for GpuProtoCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuProtoCache")
            .field("shape", &self.matrix.shape())
            .finish()
    }
}

impl PhonemeHdcLtc {
    /// Create a new multi-neuron phoneme recognizer.
    ///
    /// Network: 3 layers × 8 neurons = 24 neurons (same as Broca).
    /// Prototypes initialized as random unit vectors from genesis seed.
    pub fn new(genesis: &GenesisSeed, phoneme_labels: &[String], tau_base: f32) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base,
            backbone_tau: 0.3,
            dimension: HDC_DIMENSION,
            activation: UnifiedActivation::Tanh,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            gating_steepness: 1.0,
            interp_bias: 0.0,
            fourier_frequencies: Vec::new(),
            fourier_amplitude: 0.1,
        };

        let network_config = UnifiedNetworkConfig {
            layer_sizes: vec![8, 8, 8], // 24 neurons total
            neuron_config,
            use_layer_binding: true,
            skip_connections: true,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(network_config, genesis);

        // Create phoneme prototypes (random unit vectors, quasi-orthogonal in 16,384D)
        let prototypes: Vec<(String, ContinuousHV)> = phoneme_labels
            .iter()
            .map(|label| {
                let hv = ContinuousHV::from_genesis(
                    genesis,
                    &format!("phoneme_proto::{label}"),
                    HDC_DIMENSION,
                );
                (label.clone(), hv.normalize())
            })
            .collect();

        // Try to build GPU cache
        #[cfg(feature = "gpu")]
        let gpu_protos = Self::build_gpu_cache(&prototypes);

        let result = Self {
            network,
            prototypes,
            logit_scale: 20.0,
            #[cfg(feature = "gpu")]
            gpu_protos,
        };

        #[cfg(feature = "gpu")]
        if result.gpu_protos.is_some() {
            eprintln!("  Device: CUDA (GPU-accelerated logits)");
        } else {
            eprintln!("  Device: CPU (CUDA not available)");
        }

        #[cfg(not(feature = "gpu"))]
        eprintln!("  Device: CPU (gpu feature not enabled)");

        result
    }

    /// Build GPU prototype cache for accelerated logit computation.
    #[cfg(feature = "gpu")]
    fn build_gpu_cache(prototypes: &[(String, ContinuousHV)]) -> Option<GpuProtoCache> {
        let device = Device::cuda_if_available(0).ok()?;
        if !matches!(device, Device::Cuda(_)) {
            return None;
        }

        let n = prototypes.len();
        let flat: Vec<f32> = prototypes
            .iter()
            .flat_map(|(_, hv)| hv.values.iter().copied())
            .collect();

        let matrix = Tensor::from_vec(flat, (n, HDC_DIMENSION), &device).ok()?;
        Some(GpuProtoCache { matrix, device })
    }

    /// Evolve the network with one frame of audio input.
    pub fn step(&mut self, input_hv: &ContinuousHV, dt: f32) {
        self.network.evolve_closed_form(dt, input_hv);
    }

    /// Get the network output HV (bundled last-layer states, normalized).
    pub fn output_hv(&self) -> ContinuousHV {
        self.network.output().normalize()
    }

    /// Compute logits against all phoneme prototypes.
    /// GPU-first with CPU fallback.
    pub fn compute_logits(&self, output: &ContinuousHV) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(ref cache) = self.gpu_protos {
            if let Ok(logits) = self.compute_logits_gpu(output, cache) {
                return logits;
            }
        }

        // CPU fallback: individual cosine similarities
        self.prototypes
            .iter()
            .map(|(_, proto)| self.logit_scale * output.similarity(proto))
            .collect()
    }

    /// GPU-accelerated logit computation via batched matmul.
    #[cfg(feature = "gpu")]
    fn compute_logits_gpu(
        &self,
        output: &ContinuousHV,
        cache: &GpuProtoCache,
    ) -> Result<Vec<f32>, candle_core::Error> {
        let query = Tensor::from_vec(output.values.clone(), (1, HDC_DIMENSION), &cache.device)?;
        // logits = proto_matrix @ query^T → [n_phonemes, 1]
        let logits = cache.matrix.matmul(&query.t()?)?;
        let logits = logits.squeeze(1)?.affine(self.logit_scale as f64, 0.0)?;
        logits.to_vec1::<f32>()
    }

    /// Decode: find the most similar phoneme prototype.
    pub fn decode(&self) -> (String, f32) {
        let output = self.output_hv();
        let logits = self.compute_logits(&output);

        let mut best_idx = 0;
        let mut best_logit = f32::NEG_INFINITY;
        for (i, &logit) in logits.iter().enumerate() {
            if logit > best_logit {
                best_logit = logit;
                best_idx = i;
            }
        }

        let confidence = (best_logit / self.logit_scale + 1.0) / 2.0; // normalize to [0, 1]
        (
            self.prototypes[best_idx].0.clone(),
            confidence.clamp(0.0, 1.0),
        )
    }

    /// Train one step: evolve network, then update each neuron via backward.
    ///
    /// For each neuron in each layer, computes gradients against the target
    /// phoneme prototype and updates weights. This distributes learning
    /// across all 24 neurons.
    ///
    /// Returns cosine similarity of output to target prototype.
    pub fn train_step(
        &mut self,
        input_hv: &ContinuousHV,
        target_label: &str,
        dt: f32,
        lr: f32,
    ) -> f32 {
        // Find target prototype
        let target_idx = self.prototypes.iter().position(|(l, _)| l == target_label);
        let target_idx = match target_idx {
            Some(idx) => idx,
            None => return 0.0,
        };
        let target_proto = self.prototypes[target_idx].1.clone();

        // Forward: evolve entire network
        self.network.evolve_closed_form(dt, input_hv);

        // Compute output similarity before update (for metrics)
        let output = self.network.output().normalize();
        let sim_before = output.similarity(&target_proto);

        // Backward: update each neuron in each layer
        let n_layers = self.network.n_layers();
        for layer_idx in 0..n_layers {
            // Get the input this layer received
            let layer_input = self.network.layer_input(layer_idx, input_hv);

            // Update each neuron in this layer
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let grads = neuron.backward(&layer_input, &target_proto, dt);
                    neuron.apply_gradients_no_decay(&grads, lr);
                }
            }
        }

        sim_before
    }

    /// Reset all neuron states.
    pub fn reset(&mut self) {
        // Reset each neuron in each layer
        let n_layers = self.network.n_layers();
        for layer_idx in 0..n_layers {
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    neuron.reset();
                }
            }
        }
    }

    /// Number of phoneme prototypes.
    pub fn num_phonemes(&self) -> usize {
        self.prototypes.len()
    }

    /// Number of neurons in the network.
    pub fn num_neurons(&self) -> usize {
        (0..self.network.n_layers())
            .filter_map(|i| self.network.layer(i))
            .map(|l| l.len())
            .sum()
    }

    /// Access prototypes.
    pub fn prototypes(&self) -> &[(String, ContinuousHV)] {
        &self.prototypes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_neuron_creation() {
        let genesis = GenesisSeed::from_phrase("test_multi");
        let labels = vec!["AA".to_string(), "IH".to_string(), "SIL".to_string()];
        let ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        assert_eq!(ltc.num_phonemes(), 3);
        assert_eq!(ltc.num_neurons(), 24); // 3 × 8
    }

    #[test]
    fn test_step_and_decode() {
        let genesis = GenesisSeed::from_phrase("test_step_decode");
        let labels = vec!["AA".to_string(), "IH".to_string(), "SIL".to_string()];
        let mut ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        let input = ContinuousHV::from_genesis(&genesis, "test_input", HDC_DIMENSION);
        ltc.step(&input, 0.010);

        let (label, confidence) = ltc.decode();
        assert!(labels.contains(&label));
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_logits_correct_size() {
        let genesis = GenesisSeed::from_phrase("test_logits");
        let labels: Vec<String> = (0..69).map(|i| format!("P{i}")).collect();
        let mut ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        let input = ContinuousHV::from_genesis(&genesis, "test_input", HDC_DIMENSION);
        ltc.step(&input, 0.010);

        let output = ltc.output_hv();
        let logits = ltc.compute_logits(&output);
        assert_eq!(logits.len(), 69);
        assert!(logits.iter().all(|l| l.is_finite()));
    }

    #[test]
    fn test_train_step_returns_finite() {
        let genesis = GenesisSeed::from_phrase("test_train");
        let labels = vec!["AA".to_string(), "IH".to_string()];
        let mut ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        let input = ContinuousHV::from_genesis(&genesis, "test_input", HDC_DIMENSION);
        let sim = ltc.train_step(&input, "AA", 0.010, 0.001);

        assert!(sim.is_finite(), "Similarity should be finite: {}", sim);
    }

    #[test]
    fn test_training_changes_output() {
        let genesis = GenesisSeed::from_phrase("test_changes");
        let labels = vec!["AA".to_string(), "IH".to_string()];
        let mut ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        let input = ContinuousHV::from_genesis(&genesis, "train_input", HDC_DIMENSION);

        // Get output before training
        ltc.step(&input, 0.010);
        let output_before = ltc.output_hv();

        // Train several steps
        ltc.reset();
        for _ in 0..10 {
            ltc.train_step(&input, "AA", 0.010, 0.01);
        }

        // Get output after training
        ltc.reset();
        ltc.step(&input, 0.010);
        let output_after = ltc.output_hv();

        let sim = output_before.similarity(&output_after);
        assert!(
            sim < 0.99,
            "Output should change after training: similarity={}",
            sim
        );
    }
}
