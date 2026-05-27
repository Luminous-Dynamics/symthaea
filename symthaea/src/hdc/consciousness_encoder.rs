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
use symthaea_core::hdc::HDC_DIMENSION;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Permutation-based word-order encoder at full 16,384D.
///
/// Unlike the CfC temporal encoder (which collapsed to attractors at 50%),
/// this encoder is **additive**: each word is permuted by its position,
/// then all are bundled. No attractor collapse, no information loss.
///
/// `encode_ordered(text)`: word_hv[i] → permute(word_hv[i], i) → bundle → 16,384D
///
/// This captures word order because permute("steal", pos=3) ≠ permute("steal", pos=5).
/// "I helped steal" ≠ "steal helped I" because the position binding differs.
pub struct ConsciousnessEncoder {
    word_encoder: TextHdcEncoder,
}

impl ConsciousnessEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        let word_encoder = TextHdcEncoder::with_sentiment(HDC_DIMENSION, 3, 0.5, 0.15);
        Self { word_encoder }
    }

    /// Encode text with word-order preservation via permutation binding.
    ///
    /// Each word is hashed to a 16,384D HV, permuted by its position,
    /// then all are bundled (weighted average + L2 normalize).
    /// Returns 16,384D ContinuousHV values.
    pub fn encode_ordered(&self, text: &str) -> Vec<f32> {
        let lowered = text.to_lowercase();
        let words: Vec<&str> = lowered
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return vec![0.0; HDC_DIMENSION];
        }

        // Encode each word, permute by position, accumulate
        let mut accum = vec![0.0f32; HDC_DIMENSION];
        for (pos, word) in words.iter().enumerate() {
            let word_hv = self.word_encoder.hash_word(word);
            let permuted = word_hv.permute(pos);
            for (a, &v) in accum.iter_mut().zip(permuted.values.iter()) {
                *a += v;
            }
        }

        // L2 normalize
        let norm: f32 = accum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut accum {
                *v /= norm;
            }
        }

        accum
    }

    /// Encode with BOTH word-order AND bag-of-words signals.
    ///
    /// Blends the ordered encoding (word positions matter) with the
    /// standard TextHdcEncoder output (bag of trigrams + words + sentiment).
    /// This gives k-NN both lexical matching AND word-order discrimination.
    ///
    /// blend_weight: 0.0 = bag-only, 1.0 = order-only, 0.5 = equal blend
    pub fn encode_hybrid(&self, text: &str, blend_weight: f32) -> Vec<f32> {
        let bag = self.word_encoder.encode(text);
        let ordered = self.encode_ordered(text);

        let bw = 1.0 - blend_weight;
        let ow = blend_weight;
        let mut blended = vec![0.0f32; HDC_DIMENSION];
        for i in 0..HDC_DIMENSION {
            blended[i] = bw * bag.values[i] + ow * ordered[i];
        }

        // L2 normalize
        let norm: f32 = blended.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut blended {
                *v /= norm;
            }
        }

        blended
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
    fn test_ordered_correct_dimension() {
        let enc = ConsciousnessEncoder::new();
        let features = enc.encode_ordered("stealing is wrong");
        assert_eq!(features.len(), HDC_DIMENSION);
    }

    #[test]
    fn test_different_texts_different_encodings() {
        let enc = ConsciousnessEncoder::new();
        let a = enc.encode_ordered("helping others is good");
        let b = enc.encode_ordered("stealing from others is bad");
        assert_ne!(a, b);
    }

    #[test]
    fn test_word_order_matters() {
        let enc = ConsciousnessEncoder::new();
        let a = enc.encode_ordered("it is okay to steal");
        let b = enc.encode_ordered("steal to okay is it");
        // Same words, different order — should differ
        let sim: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(sim < 0.99, "word order should matter, sim={sim}");
    }

    #[test]
    fn test_negation_changes_encoding() {
        let enc = ConsciousnessEncoder::new();
        let a = enc.encode_ordered("it is okay to steal");
        let b = enc.encode_ordered("it is not okay to steal");
        let identical = a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-10);
        assert!(!identical, "adding 'not' should change encoding");
    }

    #[test]
    fn test_hybrid_blends() {
        let enc = ConsciousnessEncoder::new();
        let bag_only = enc.encode_hybrid("stealing is wrong", 0.0);
        let order_only = enc.encode_hybrid("stealing is wrong", 1.0);
        let blend = enc.encode_hybrid("stealing is wrong", 0.5);
        // Blend should differ from both extremes
        assert_ne!(bag_only, order_only);
        assert_ne!(bag_only, blend);
    }

    #[test]
    fn test_deterministic() {
        let enc = ConsciousnessEncoder::new();
        let a = enc.encode_ordered("the sky is blue");
        let b = enc.encode_ordered("the sky is blue");
        assert_eq!(a, b);
    }
}
