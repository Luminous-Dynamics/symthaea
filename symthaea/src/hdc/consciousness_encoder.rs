// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-driven text encoder — feeds text word-by-word through a
//! single CfC neuron at full 16,384D for context-aware moral classification.
//!
//! V2: Uses full HDC_DIMENSION (16,384D) instead of 256D projection.
//! The 256D version collapsed to attractors — all texts looked the same.
//! At 16,384D, the CfC neuron has enough state dimensions to differentiate
//! temporal trajectories between different word sequences.
//!
//! Architecture: text → split words → encode each word via TextHdcEncoder →
//!   HdcLtcUnifiedNeuron::evolve_closed_form(dt, &word_hv) →
//!   final 16,384D state used for k-NN classification

use super::moral_text_encoder::TextHdcEncoder;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::hdc::HDC_DIMENSION;

/// Consciousness-driven text encoder using a single CfC neuron at full dimension.
///
/// Each word is encoded as a 16,384D ContinuousHV by TextHdcEncoder,
/// then drives one CfC evolution step. The temporal dynamics capture
/// word order and context accumulation that bag-of-trigrams misses.
pub struct ConsciousnessEncoder {
    word_encoder: TextHdcEncoder,
    neuron: HdcLtcUnifiedNeuron,
    dt: f32,
}

impl ConsciousnessEncoder {
    /// Create a new encoder with deterministic initialization.
    pub fn new() -> Self {
        let word_encoder = TextHdcEncoder::with_sentiment(HDC_DIMENSION, 3, 0.5, 0.15);

        let config = UnifiedConfig {
            dimension: HDC_DIMENSION,
            tau_base: 0.5,
            backbone_tau: 0.1,
            learning_rate: 0.0, // inference only
            momentum: 0.0,
            weight_decay: 0.0,
            gating_steepness: 4.0,
            interp_bias: 0.5,
            ..UnifiedConfig::default()
        };

        let neuron = HdcLtcUnifiedNeuron::new(config, 90_000_001);

        Self {
            word_encoder,
            neuron,
            dt: 0.05, // smaller dt for more gradual evolution
        }
    }

    /// Encode text by processing words sequentially through CfC dynamics.
    ///
    /// Returns the 16,384D final hidden state. The CfC temporal integration
    /// captures word order and context — "helped steal" evolves differently
    /// from "helped move" because each word modifies the neuron state.
    pub fn encode(&mut self, text: &str) -> Vec<f32> {
        let lowered = text.to_lowercase();
        let words: Vec<&str> = lowered
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        // Reset neuron — each text starts fresh
        self.neuron.reset();

        if words.is_empty() {
            return vec![0.0; HDC_DIMENSION];
        }

        // Encode each word as a full 16,384D HV and evolve the neuron
        for word in &words {
            // Hash word to deterministic HV (same as TextHdcEncoder's word channel)
            let word_hv = self.word_encoder.hash_word(word);
            self.neuron.evolve_closed_form(self.dt, &word_hv);
        }

        // Extract final state
        self.neuron.state().values.clone()
    }

    /// Output dimension.
    pub fn dim(&self) -> usize {
        HDC_DIMENSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_produces_correct_dimension() {
        let mut enc = ConsciousnessEncoder::new();
        let features = enc.encode("stealing is wrong");
        assert_eq!(features.len(), HDC_DIMENSION);
    }

    #[test]
    fn test_different_texts_produce_different_encodings() {
        let mut enc = ConsciousnessEncoder::new();
        let a = enc.encode("helping others is good");
        let b = enc.encode("stealing from others is bad");
        assert_ne!(a, b);
    }

    #[test]
    fn test_word_order_matters() {
        let mut enc = ConsciousnessEncoder::new();
        let a = enc.encode("it is okay to steal");
        let b = enc.encode("it is not okay to steal");
        // Just check they're not identical — the CfC trajectory should differ
        let identical = a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-10);
        assert!(!identical, "negation should produce different CfC state");
    }

    #[test]
    fn test_deterministic() {
        let mut enc = ConsciousnessEncoder::new();
        let a = enc.encode("the sky is blue");
        let b = enc.encode("the sky is blue");
        assert_eq!(a, b, "same input should produce same output");
    }
}
