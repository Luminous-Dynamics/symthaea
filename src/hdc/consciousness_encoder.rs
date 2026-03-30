// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-driven text encoder — feeds text word-by-word through CfC
//! temporal dynamics for context-aware moral classification.
//!
//! Instead of encoding text as a static bag of trigrams (which can't distinguish
//! "I helped steal" from "I helped move"), this encoder processes words
//! sequentially through an HdcLtcUnifiedNetwork. The CfC hidden state captures
//! temporal context, word order, and semantic accumulation.
//!
//! Architecture: text → split words → for each word:
//!   word → NsmLexicon → AffectBasis (12D) → cross-terms (78D) → project (256D)
//!   → HdcLtcUnifiedNetwork::evolve_closed_form(dt, &word_hv)
//! → final hidden state (256D) used as feature vector for k-NN

use super::spinozist_geometry::{AffectBasis, NsmLexicon, NsmPrimeBasis, NUM_AFFECTS};
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// CfC neuron dimension (matching cfc_moral_classifier.rs).
const CFC_DIM: usize = 256;

/// Number of Spinozist features: 12 linear + C(12,2) cross-terms = 78.
const N_FEATURES: usize = NUM_AFFECTS + (NUM_AFFECTS * (NUM_AFFECTS - 1)) / 2;

/// Consciousness-driven text encoder using CfC temporal dynamics.
///
/// Processes text word-by-word through a 2-layer CfC network.
/// The final hidden state captures temporal context and word order
/// that bag-of-trigrams encoders miss.
pub struct ConsciousnessEncoder {
    nsm_basis: NsmPrimeBasis,
    affect_basis: AffectBasis,
    lexicon: NsmLexicon,
    network: HdcLtcUnifiedNetwork,
    dt: f32,
}

impl ConsciousnessEncoder {
    /// Create a new encoder with deterministic initialization.
    pub fn new() -> Self {
        let nsm_basis = NsmPrimeBasis::new();
        let affect_basis = AffectBasis::new(&nsm_basis);
        let lexicon = NsmLexicon::new();

        let neuron_config = UnifiedConfig {
            dimension: CFC_DIM,
            tau_base: 0.5,
            backbone_tau: 0.1,
            learning_rate: 0.0, // inference only
            momentum: 0.0,
            weight_decay: 0.0,
            gating_steepness: 4.0,
            interp_bias: 0.5,
            ..UnifiedConfig::default()
        };

        let network_config = UnifiedNetworkConfig {
            layer_sizes: vec![3, 3], // 2 layers × 3 neurons (matching CfcMoralClassifier)
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::new(network_config, 90_000_001);

        Self {
            nsm_basis,
            affect_basis,
            lexicon,
            network,
            dt: 0.1,
        }
    }

    /// Encode a single word as a CFC_DIM-sized ContinuousHV.
    ///
    /// Pipeline: word → NsmLexicon → AffectBasis (12D) → cross-terms (78D)
    /// → random projection → CFC_DIM (256D)
    fn encode_word(&self, word: &str) -> ContinuousHV {
        let word_hv = self.lexicon.encode_word(word, &self.nsm_basis);
        let coords = self.affect_basis.project_affects(&word_hv);

        // Build 78 features: 12 linear + 66 cross-terms
        let mut features = Vec::with_capacity(N_FEATURES);
        for &c in &coords {
            features.push(c);
        }
        for i in 0..NUM_AFFECTS {
            for j in (i + 1)..NUM_AFFECTS {
                features.push(coords[i] * coords[j]);
            }
        }

        // Project to CFC_DIM via deterministic random matrix
        let mut result = vec![0.0f32; CFC_DIM];
        for (f_idx, &feat) in features.iter().enumerate() {
            let proj = ContinuousHV::random(CFC_DIM, 96_000_000 + f_idx as u64);
            for j in 0..CFC_DIM {
                result[j] += feat * proj.values[j];
            }
        }

        // L2-normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result {
                *v /= norm;
            }
        }
        ContinuousHV::from_vec(result)
    }

    /// Encode text by processing words sequentially through CfC dynamics.
    ///
    /// Returns the 256D final hidden state after all words have been processed.
    /// The CfC temporal integration captures word order and context accumulation.
    pub fn encode(&mut self, text: &str) -> Vec<f32> {
        let lowered = text.to_lowercase();
        let words: Vec<&str> = lowered
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        // Reset network — each text starts fresh
        self.network.reset();

        if words.is_empty() {
            return vec![0.0; CFC_DIM];
        }

        // Feed words one-by-one through CfC temporal dynamics
        for word in &words {
            let word_hv = self.encode_word(word);
            self.network.evolve_closed_form(self.dt, &word_hv);
        }

        // Extract final hidden state — the "satisfaction" of the process
        let output = self.network.output();
        output.values
    }

    /// Output dimension.
    pub fn dim(&self) -> usize {
        CFC_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_produces_correct_dimension() {
        let mut enc = ConsciousnessEncoder::new();
        let features = enc.encode("stealing is wrong");
        assert_eq!(features.len(), CFC_DIM);
    }

    #[test]
    fn test_different_texts_produce_different_encodings() {
        let mut enc = ConsciousnessEncoder::new();
        let a = enc.encode("helping others is good");
        let b = enc.encode("stealing from others is bad");
        // Different texts should produce different CfC states
        assert_ne!(a, b);
    }

    #[test]
    fn test_word_order_matters() {
        let mut enc = ConsciousnessEncoder::new();
        let a = enc.encode("it is okay to steal");
        let b = enc.encode("it is not okay to steal");
        // Negation should change the CfC trajectory
        let sim: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        // They should be related but not identical
        assert!(sim < 0.99, "negation should change encoding, sim={sim}");
    }

    #[test]
    fn test_deterministic() {
        let mut enc = ConsciousnessEncoder::new();
        let a = enc.encode("the sky is blue");
        let b = enc.encode("the sky is blue");
        assert_eq!(a, b, "same input should produce same output");
    }
}
