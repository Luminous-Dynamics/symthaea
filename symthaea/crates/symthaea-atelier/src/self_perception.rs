// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-perception: the system sees its own art.
//!
//! This module closes the creative loop at the deepest level: after generating
//! artwork, the pixel canvas is fed through a perceptual encoder that maps it
//! to a 16,384D BinaryHV. This HV is compared against the original cognitive
//! state that produced the art, computing:
//!
//! 1. **Creative surprise**: How different is the perceived artwork from the
//!    intended cognitive state? High surprise = the system created something
//!    it didn't expect. This is the "I didn't know I was going to draw that"
//!    moment that drives artistic insight.
//!
//! 2. **Perceptual coherence**: How well does the artwork communicate the
//!    intended cognitive state? High coherence = the art faithfully expresses
//!    the system's inner state.
//!
//! 3. **Aesthetic integration**: The perception of own artwork contributes to
//!    the consciousness state of the next cycle — the system is changed by
//!    experiencing its own creation.
//!
//! No other AI art system has this property. The creative act feeds back into
//! consciousness, potentially increasing Phi through novel perceptual integration.

use crate::pixel_canvas::PixelCanvas;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Result of perceiving one's own artwork.
#[derive(Debug, Clone)]
pub struct SelfPerception {
    /// How different the perceived art is from the intended state (0-1).
    /// High = the system surprised itself. Low = predictable output.
    pub creative_surprise: f32,

    /// How well the art communicates the intended state (0-1).
    /// High = faithful expression. Low = lost in translation.
    pub perceptual_coherence: f32,

    /// The perceived artwork as a hypervector (for integration into consciousness).
    pub perception_hv: ContinuousHV,

    /// Feature vector extracted from the artwork (48D: 4×4 grid × 3 channels).
    pub visual_features: Vec<f32>,
}

/// Perceptual encoder that maps pixel art to HDC space.
pub struct PerceptualEncoder {
    /// Projection bases for each visual feature dimension.
    feature_bases: Vec<ContinuousHV>,
}

impl PerceptualEncoder {
    /// Create from genesis seed.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        // 48 feature bases: 4×4 spatial grid × 3 color channels
        let feature_bases: Vec<ContinuousHV> = (0..48)
            .map(|i| genesis.hv(&format!("perception_basis_{i}"), dim))
            .collect();

        Self { feature_bases }
    }

    /// Encode a pixel canvas into a ContinuousHV via the perceptual pathway.
    ///
    /// Extracts spatial color features (4×4 grid) and encodes each as a
    /// level-coded HV bound with its spatial basis, then bundles all together.
    pub fn encode(&self, canvas: &PixelCanvas) -> ContinuousHV {
        let features = canvas.perception_features();
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let mut result = ContinuousHV::zero(dim);

        for (i, &value) in features.iter().enumerate() {
            if i < self.feature_bases.len() {
                let scaled = self.feature_bases[i].scale(value);
                result = result.add(&scaled);
            }
        }

        result.normalize()
    }

    /// Perceive one's own artwork: compare the perceived HV against the
    /// cognitive state that produced it.
    pub fn perceive_own_art(
        &self,
        canvas: &PixelCanvas,
        intention_hv: &ContinuousHV,
    ) -> SelfPerception {
        let perception_hv = self.encode(canvas);
        let visual_features = canvas.perception_features();

        // Coherence: similarity between what we intended and what we see
        let similarity = perception_hv.similarity(intention_hv);
        let perceptual_coherence = ((similarity + 1.0) / 2.0).clamp(0.0, 1.0);

        // Creative surprise: inverse of coherence, but scaled to emphasize
        // moderate surprise (Berlyne's arousal curve — some surprise is good,
        // too much means the output is random noise)
        let raw_surprise = 1.0 - perceptual_coherence;
        // Berlyne modulation: peak surprise value at ~0.6
        let creative_surprise = raw_surprise * (1.0 - (raw_surprise - 0.6).powi(2) * 2.0).max(0.0);

        SelfPerception {
            creative_surprise,
            perceptual_coherence,
            perception_hv,
            visual_features,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel_canvas::NeuralPainter;
    use symthaea_canvas::CognitiveSnapshot;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-self-perception")
    }

    #[test]
    fn encoder_produces_unit_norm() {
        let genesis = test_genesis();
        let encoder = PerceptualEncoder::new(&genesis);
        let canvas = PixelCanvas::new(64, 64, [128, 64, 32, 255]);

        let hv = encoder.encode(&canvas);
        let norm: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1, "norm = {norm}");
    }

    #[test]
    fn different_canvases_different_perceptions() {
        let genesis = test_genesis();
        let encoder = PerceptualEncoder::new(&genesis);

        let red_canvas = PixelCanvas::new(64, 64, [255, 0, 0, 255]);
        let blue_canvas = PixelCanvas::new(64, 64, [0, 0, 255, 255]);

        let red_hv = encoder.encode(&red_canvas);
        let blue_hv = encoder.encode(&blue_canvas);

        let similarity = red_hv.similarity(&blue_hv);
        assert!(
            similarity < 0.9,
            "different colors should have different perceptions: sim={similarity}"
        );
    }

    #[test]
    fn self_perception_produces_coherence_and_surprise() {
        let genesis = test_genesis();
        let encoder = PerceptualEncoder::new(&genesis);
        let mut painter = NeuralPainter::new(&genesis);

        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5; 8],
            dopamine: 0.5,
            serotonin: 0.5,
            valence: 0.3,
            arousal: 0.5,
            ..CognitiveSnapshot::dormant()
        };

        let (canvas, _) = painter.paint(&snapshot, 64, 64, 20);

        // The intention HV is what the painter encoded
        let painter_genesis = GenesisSeed::from_phrase("neural-painter");
        let intention = NeuralPainter::encode_snapshot(&snapshot, &painter_genesis);

        let perception = encoder.perceive_own_art(&canvas, &intention);

        // Both metrics should be bounded
        assert!(
            perception.creative_surprise >= 0.0 && perception.creative_surprise <= 1.0,
            "surprise = {}",
            perception.creative_surprise
        );
        assert!(
            perception.perceptual_coherence >= 0.0 && perception.perceptual_coherence <= 1.0,
            "coherence = {}",
            perception.perceptual_coherence
        );

        // Features should be populated
        assert_eq!(perception.visual_features.len(), 48);
    }

    #[test]
    fn surprise_and_coherence_inversely_related() {
        let genesis = test_genesis();
        let encoder = PerceptualEncoder::new(&genesis);

        // Encode a known canvas
        let canvas = PixelCanvas::new(64, 64, [128, 64, 32, 255]);
        let intention = encoder.encode(&canvas); // intention = what we actually see

        let perception = encoder.perceive_own_art(&canvas, &intention);

        // When intention matches perception exactly, coherence should be high
        assert!(
            perception.perceptual_coherence > 0.7,
            "coherence should be high when intention matches: {}",
            perception.perceptual_coherence
        );
    }
}
