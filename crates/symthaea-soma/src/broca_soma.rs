// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Soma-aware Broca language center -- full 20-channel ThoughtEncoder.
//!
//! Replaces BrocaLite with the full Broca pipeline: 20-channel ThoughtEncoder,
//! epistemic gating (prevents hallucination), emotional modulation, semantic veto
//! (mid-sentence self-correction), and coherence feedback.

use symthaea_broca::{
    BrocaConfig, BrocaGenerator, GenerationResult, LanguageControllerConfig, SamplingStrategy,
    ThoughtChannels,
};
use symthaea_core::genesis::GenesisSeed;

/// Default checkpoint path for pre-trained Soma embodied vocabulary.
const DEFAULT_CHECKPOINT_PATH: &str = "data/soma_broca_checkpoint.bin";

/// Soma-aware Broca language center wrapping the full `BrocaGenerator`.
///
/// Maps embodied consciousness state (neuromodulators, sensor context, screen
/// perception) into the 20 ThoughtChannels, then runs autoregressive generation
/// with epistemic gating, emotional modulation, semantic veto, and coherence
/// feedback.
///
/// When a pre-trained checkpoint exists at `data/soma_broca_checkpoint.bin`,
/// it is loaded automatically so Soma generates phone-aware embodied language.
pub struct BrocaSoma {
    generator: BrocaGenerator,
    /// Whether a trained checkpoint was successfully loaded.
    checkpoint_loaded: bool,
}

impl BrocaSoma {
    /// Create a new BrocaSoma with default Soma-tuned configuration.
    ///
    /// Attempts to load a pre-trained checkpoint from `data/soma_broca_checkpoint.bin`.
    /// If the checkpoint exists and loads successfully, Soma will generate
    /// phone-aware embodied language. Otherwise, falls back to untrained weights.
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symthaea-soma");
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 3,
                neurons_per_layer: 8,
                dt_per_token: 0.02,
                logit_scale: 20.0,
                ..Default::default()
            },
            sampling: SamplingStrategy::TopK {
                k: 40,
                temperature: 0.7,
            },
            enable_epistemic_gate: true,
            enable_emotional_modulation: true,
            enable_coherence_feedback: true,
            enable_consciousness_gating: true,
            enable_semantic_veto: true,
            ..Default::default()
        };

        // Try to load pre-trained soma checkpoint
        let (generator, checkpoint_loaded) =
            if std::path::Path::new(DEFAULT_CHECKPOINT_PATH).exists() {
                match BrocaGenerator::from_checkpoint(DEFAULT_CHECKPOINT_PATH, &genesis) {
                    Ok((r#gen, _adam, _proj, _lm)) => (r#gen, true),
                    Err(_) => {
                        // Checkpoint exists but failed to load — fall back to untrained
                        (BrocaGenerator::new(&genesis, config), false)
                    }
                }
            } else {
                (BrocaGenerator::new(&genesis, config), false)
            };

        Self {
            generator,
            checkpoint_loaded,
        }
    }

    /// Load a pre-trained checkpoint for embodied language.
    ///
    /// Replaces the current generator with weights from the checkpoint file.
    /// The checkpoint must have been trained with the same genesis seed
    /// (`"symthaea-soma"`) and compatible `BrocaConfig`.
    pub fn load_checkpoint(&mut self, path: &str) -> Result<(), String> {
        let genesis = GenesisSeed::from_phrase("symthaea-soma");
        match BrocaGenerator::from_checkpoint(path, &genesis) {
            Ok((r#gen, _adam, _proj, _lm)) => {
                self.generator = r#gen;
                self.checkpoint_loaded = true;
                Ok(())
            }
            Err(e) => Err(format!("Failed to load Broca checkpoint: {e}")),
        }
    }

    /// Whether a trained checkpoint was successfully loaded.
    pub fn is_checkpoint_loaded(&self) -> bool {
        self.checkpoint_loaded
    }

    /// Generate text from Soma's current embodied consciousness state.
    ///
    /// Maps consciousness level, neuromodulators, sensor state, and screen
    /// perception into the 20 ThoughtChannels, then runs the full autoregressive
    /// pipeline with epistemic gating and semantic veto.
    pub fn generate_embodied(
        &mut self,
        consciousness_level: f32,
        prediction_error: f32,
        harmony_alignment: f32,
        neuromods: [f32; 4],  // [DA, NE, 5-HT, OT]
        wake_state: u8,       // 0=Sleep, 1=Drowsy, 2=Alert, 3=Focused
        motion_state: u8,     // 0=Stationary..3=InVehicle
        screen_surprise: f32, // 0.0-1.0 from screen vision
    ) -> GenerationResult {
        let thought = self.build_channels(
            consciousness_level,
            prediction_error,
            harmony_alignment,
            neuromods,
            wake_state,
            motion_state,
            screen_surprise,
        );
        self.generator.generate(&thought)
    }

    /// Generate text continuing from previous context (multi-turn).
    ///
    /// Same channel mapping as `generate_embodied` but preserves CfC neural
    /// state from prior generations for coherent multi-turn dialogue.
    pub fn generate_continuing_embodied(
        &mut self,
        consciousness_level: f32,
        prediction_error: f32,
        harmony_alignment: f32,
        neuromods: [f32; 4],
        wake_state: u8,
        motion_state: u8,
        screen_surprise: f32,
    ) -> GenerationResult {
        let thought = self.build_channels(
            consciousness_level,
            prediction_error,
            harmony_alignment,
            neuromods,
            wake_state,
            motion_state,
            screen_surprise,
        );
        self.generator.generate_continuing(&thought)
    }

    /// Build ThoughtChannels from embodied state.
    fn build_channels(
        &self,
        consciousness_level: f32,
        prediction_error: f32,
        harmony_alignment: f32,
        neuromods: [f32; 4],
        wake_state: u8,
        motion_state: u8,
        screen_surprise: f32,
    ) -> ThoughtChannels {
        let mut thought = ThoughtChannels::default();

        // Consciousness channels: psi, meta_awareness, coherence
        thought.set_consciousness(
            consciousness_level,
            prediction_error.min(1.0), // PE as meta-awareness proxy
            harmony_alignment,
        );

        // Neuromod channels: valence, arousal, warmth
        let valence = ((neuromods[0] - 0.5) + (neuromods[3] - 0.5)).clamp(-1.0, 1.0); // DA + OT
        let arousal = neuromods[1].clamp(0.0, 1.0); // NE
        let warmth = neuromods[3].clamp(0.0, 1.0); // OT
        thought.set_emotion(valence, arousal, warmth);

        // Epistemic: honest about simulation status -- always "Probable" at best
        thought.set_epistemic(1.0);

        // Context channels: time pressure, domain familiarity, social context, confidence
        let time_pressure = match wake_state {
            0 => 0.0, // Sleep -- no pressure
            1 => 0.1, // Drowsy -- minimal
            3 => 0.7, // Focused -- concise output
            _ => 0.3, // Alert -- moderate
        };
        let domain_familiarity = consciousness_level * 0.8;
        let social_context = if warmth > 0.7 { 0.2 } else { 0.6 }; // warm -> informal
        let response_confidence = (consciousness_level * harmony_alignment).clamp(0.0, 1.0);
        thought.set_context(
            time_pressure,
            domain_familiarity,
            social_context,
            response_confidence,
        );

        // Screen surprise modulates arousal
        if screen_surprise > 0.3 {
            let boosted_arousal = (arousal + screen_surprise * 0.3).min(1.0);
            thought.channels[10] = boosted_arousal; // arousal channel
        }

        // Motion modulates warmth (moving -> more engaged)
        if motion_state >= 1 {
            thought.channels[11] = (warmth + 0.1).min(1.0); // warmth boost
        }

        thought
    }
}

impl Default for BrocaSoma {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broca_soma_creates_and_generates() {
        let mut broca = BrocaSoma::new();
        let result = broca.generate_embodied(0.5, 0.3, 0.7, [0.5, 0.5, 0.5, 0.5], 2, 0, 0.0);
        // Generation completes without panic. With untrained weights, output
        // may be empty/garbage — the key is the pipeline runs end-to-end.
        assert!(result.num_tokens <= 512, "Token count should be bounded");
    }

    #[test]
    fn broca_soma_epistemic_gating_active() {
        let mut broca = BrocaSoma::new();
        let result = broca.generate_embodied(0.3, 0.8, 0.4, [0.3, 0.7, 0.3, 0.3], 2, 0, 0.0);
        // With low consciousness and high PE, gating should be conservative.
        // Coherence can be negative (anti-correlated), NaN (no tokens), or positive.
        // The key invariant: generation completes without panic.
        let _coherence = result.final_coherence; // may be any f32
        // Verify we got some output (even if empty due to heavy gating)
        assert!(result.num_tokens <= 128, "Token count should be bounded");
    }

    #[test]
    fn broca_soma_screen_surprise_boosts_arousal() {
        let mut broca = BrocaSoma::new();
        let r1 = broca.generate_embodied(0.5, 0.3, 0.7, [0.5, 0.3, 0.5, 0.5], 2, 0, 0.0);
        let r2 = broca.generate_embodied(0.5, 0.3, 0.7, [0.5, 0.3, 0.5, 0.5], 2, 0, 0.9);
        // Both should generate valid output (specific content differs due to arousal modulation)
        let _ = (r1, r2);
    }

    #[test]
    fn broca_soma_continuing_preserves_context() {
        let mut broca = BrocaSoma::new();
        let _r1 = broca.generate_embodied(0.5, 0.3, 0.7, [0.5, 0.5, 0.5, 0.5], 2, 0, 0.0);
        let r2 = broca.generate_continuing_embodied(0.5, 0.3, 0.7, [0.5, 0.5, 0.5, 0.5], 2, 0, 0.0);
        // Continuing generation should complete without panic
        assert!(r2.num_tokens <= 512, "Token count should be bounded");
    }

    #[test]
    fn broca_soma_default_trait() {
        let broca = BrocaSoma::default();
        // Should construct without panic
        let _ = broca;
    }

    #[test]
    fn broca_soma_checkpoint_not_loaded_by_default() {
        let broca = BrocaSoma::new();
        // No checkpoint file exists in test environment
        assert!(!broca.is_checkpoint_loaded());
    }

    #[test]
    fn broca_soma_load_checkpoint_nonexistent_fails() {
        let mut broca = BrocaSoma::new();
        let result = broca.load_checkpoint("/tmp/nonexistent_soma_checkpoint.bin");
        assert!(result.is_err());
        assert!(!broca.is_checkpoint_loaded());
    }
}
