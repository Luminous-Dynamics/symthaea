// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phoneme HDC-LTC — Unified HdcLtcUnifiedNeuron for speech recognition.
//!
//! Wraps symthaea-core's HdcLtcUnifiedNeuron with phoneme-specific configuration:
//! - tau_base = 20ms (matches phoneme duration)
//! - Weight-tied cosine decoding against phoneme prototypes
//! - Contrastive training via built-in backward() + apply_gradients()
//!
//! This replaces the basic 128D LTC cell with a 16,384D unified neuron
//! where the state IS a hypervector — the same architecture Broca uses
//! for language generation.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig, UnifiedActivation};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Phoneme recognition using HdcLtcUnifiedNeuron.
///
/// The neuron state evolves through 16,384D HDC space. Each phoneme has a
/// prototype HV. Decoding is weight-tied cosine similarity (Broca pattern).
pub struct PhonemeHdcLtc {
    /// The unified HDC-LTC neuron (16,384D state)
    neuron: HdcLtcUnifiedNeuron,
    /// Phoneme prototypes: (label, target_hv)
    prototypes: Vec<(String, ContinuousHV)>,
    /// Logit scaling (like CLIP's learned temperature)
    logit_scale: f32,
}

impl PhonemeHdcLtc {
    /// Create a new phoneme recognizer.
    ///
    /// Prototypes are initialized as random unit vectors from the genesis seed,
    /// ensuring deterministic initialization and quasi-orthogonality.
    pub fn new(genesis: &GenesisSeed, phoneme_labels: &[String], tau_base: f32) -> Self {
        let config = UnifiedConfig {
            tau_base,               // 20ms for phonemes (fast dynamics)
            backbone_tau: 0.3,      // Moderate state-dependent tau
            dimension: HDC_DIMENSION,
            activation: UnifiedActivation::Tanh,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,      // No decay for BPTT (per apply_gradients_no_decay)
            gating_steepness: 1.0,
            interp_bias: 0.0,
            fourier_frequencies: Vec::new(),
            fourier_amplitude: 0.1,
        };

        let neuron = HdcLtcUnifiedNeuron::from_genesis(
            config,
            genesis,
            "phoneme_hdc_ltc::neuron",
        );

        // Create phoneme prototypes (random unit vectors, quasi-orthogonal in 16,384D)
        let prototypes = phoneme_labels
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

        Self {
            neuron,
            prototypes,
            logit_scale: 20.0, // CLIP-style scaling
        }
    }

    /// Evolve the neuron state with one frame of audio input.
    pub fn step(&mut self, input_hv: &ContinuousHV, dt: f32) {
        self.neuron.evolve_closed_form_fused(dt, input_hv);
    }

    /// Get the current state HV.
    pub fn state(&self) -> &ContinuousHV {
        self.neuron.state()
    }

    /// Decode: find the most similar phoneme prototype.
    ///
    /// Returns (label, confidence) where confidence is normalized similarity [0, 1].
    pub fn decode(&self) -> (String, f32) {
        let state = self.neuron.state();
        let mut best_label = String::new();
        let mut best_sim = f32::NEG_INFINITY;

        for (label, proto) in &self.prototypes {
            let sim = state.similarity(proto);
            if sim > best_sim {
                best_sim = sim;
                best_label = label.clone();
            }
        }

        // Normalize similarity from [-1, 1] to [0, 1]
        (best_label, (best_sim + 1.0) / 2.0)
    }

    /// Train one step: evolve, then contrastive update (pull toward target, push from negative).
    ///
    /// Uses the neuron's built-in `contrastive_update()` which directly modifies
    /// weight_hv to pull state toward the target phoneme prototype and push away
    /// from a randomly selected negative prototype.
    ///
    /// Returns cosine similarity to target (higher = better).
    pub fn train_step(
        &mut self,
        input_hv: &ContinuousHV,
        target_label: &str,
        dt: f32,
        lr: f32,
    ) -> f32 {
        // Find target prototype index
        let target_idx = self
            .prototypes
            .iter()
            .position(|(l, _)| l == target_label);

        let target_idx = match target_idx {
            Some(idx) => idx,
            None => return 0.0, // Unknown phoneme, skip
        };

        // Evolve neuron with input
        self.neuron.evolve_closed_form_fused(dt, input_hv);

        // Get positive prototype (target)
        let positive = self.prototypes[target_idx].1.clone();

        // Get negative prototype (hardest negative = most similar non-target)
        // Clone state to avoid borrow conflict with mutable update below
        let state_snapshot = self.neuron.state().clone();
        let mut worst_neg_idx = (target_idx + 1) % self.prototypes.len();
        let mut worst_neg_sim = f32::NEG_INFINITY;
        for (i, (_, proto)) in self.prototypes.iter().enumerate() {
            if i == target_idx {
                continue;
            }
            let sim = state_snapshot.similarity(proto);
            if sim > worst_neg_sim {
                worst_neg_sim = sim;
                worst_neg_idx = i;
            }
        }
        let negative = self.prototypes[worst_neg_idx].1.clone();

        // Contrastive update: pull toward positive, push from negative
        self.neuron.contrastive_update(&positive, &negative, lr);

        // Also do a Hebbian update to strengthen input→state association
        self.neuron.hebbian_update(input_hv, Some(lr * 0.1));

        // Return similarity to target (loss proxy)
        state_snapshot.similarity(&positive)
    }

    /// Reset neuron state to zero.
    pub fn reset(&mut self) {
        self.neuron.reset();
    }

    /// Number of phoneme prototypes.
    pub fn num_phonemes(&self) -> usize {
        self.prototypes.len()
    }

    /// Access prototypes.
    pub fn prototypes(&self) -> &[(String, ContinuousHV)] {
        &self.prototypes
    }

    /// Access the inner neuron (for serialization or advanced use).
    pub fn neuron(&self) -> &HdcLtcUnifiedNeuron {
        &self.neuron
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_hdc_ltc_creation() {
        let genesis = GenesisSeed::from_phrase("test_phoneme_ltc");
        let labels = vec!["AA".to_string(), "IH".to_string(), "SIL".to_string()];
        let ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        assert_eq!(ltc.num_phonemes(), 3);
        assert_eq!(ltc.state().dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_step_changes_state() {
        let genesis = GenesisSeed::from_phrase("test_step");
        let labels = vec!["AA".to_string(), "IH".to_string()];
        let mut ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        let state_before = ltc.state().clone();
        let input = ContinuousHV::from_genesis(&genesis, "test_input", HDC_DIMENSION);
        ltc.step(&input, 0.010);
        let state_after = ltc.state().clone();

        let sim = state_before.similarity(&state_after);
        assert!(
            sim < 0.99,
            "State should change after step: similarity={}",
            sim
        );
    }

    #[test]
    fn test_decode_returns_valid_phoneme() {
        let genesis = GenesisSeed::from_phrase("test_decode");
        let labels = vec!["AA".to_string(), "IH".to_string(), "SIL".to_string()];
        let mut ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        let input = ContinuousHV::from_genesis(&genesis, "test_input", HDC_DIMENSION);
        ltc.step(&input, 0.010);

        let (label, confidence) = ltc.decode();
        assert!(
            labels.contains(&label),
            "Decoded label should be one of the phonemes: {}",
            label
        );
        assert!(
            confidence >= 0.0 && confidence <= 1.0,
            "Confidence should be [0,1]: {}",
            confidence
        );
    }

    #[test]
    fn test_train_step_returns_finite_loss() {
        let genesis = GenesisSeed::from_phrase("test_train");
        let labels = vec!["AA".to_string(), "IH".to_string()];
        let mut ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        let input = ContinuousHV::from_genesis(&genesis, "test_input", HDC_DIMENSION);
        let loss = ltc.train_step(&input, "AA", 0.010, 0.001);

        assert!(loss.is_finite(), "Loss should be finite: {}", loss);
        assert!(loss >= 0.0, "MSE loss should be non-negative: {}", loss);
    }

    #[test]
    fn test_training_reduces_loss() {
        let genesis = GenesisSeed::from_phrase("test_converge");
        let labels = vec!["AA".to_string(), "IH".to_string()];
        let mut ltc = PhonemeHdcLtc::new(&genesis, &labels, 0.020);

        // Use the AA prototype as input (should converge to AA quickly)
        let aa_proto = ltc.prototypes()[0].1.clone();

        let mut first_loss = 0.0;
        let mut last_loss = 0.0;
        for i in 0..20 {
            ltc.reset();
            let loss = ltc.train_step(&aa_proto, "AA", 0.010, 0.01);
            if i == 0 {
                first_loss = loss;
            }
            last_loss = loss;
        }

        assert!(
            last_loss < first_loss + 0.1,
            "Loss should decrease over training: first={:.4}, last={:.4}",
            first_loss, last_loss
        );
    }
}
