// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full Broca pipeline integration — real HdcLtcUnifiedNetwork language generation.
//!
//! Feature-gated behind `broca-pipeline`. When enabled, the Spore can load a
//! trained `BrocaCheckpoint` from symthaea-broca and use the production Broca
//! generator instead of the built-in BrocaLite or broca-full modules.
//!
//! ## Pipeline
//!
//! - **24-channel ThoughtLanguageEncoder** with genesis-seeded base vectors
//! - **HdcLtcUnifiedNetwork** backbone (3 layers x 8 neurons, 16,384D)
//! - **EpistemicGate**: canonical word lists, logit penalties (-3.0 unknown, -1.5 uncertain)
//! - **EmotionalModulator**: arousal -> sentence length, warmth -> formality
//! - **CoherenceFeedback**: Welford adaptive z-score veto with hesitation insertion
//! - **Consciousness-gated verbosity**: Psi modulates max_tokens
//! - **Frequency-scaled repetition penalty** + no-repeat bigram blocking
//!
//! ## Checkpoint loading
//!
//! The checkpoint must be loaded at runtime via `load_checkpoint()` from a byte
//! buffer (fetched via WASM `fetch()` or passed from JS). The full v6-joint
//! checkpoint is ~976MB; use the distilled spore checkpoint (~2K tokens, no
//! optimizer state) for browser deployment.
//!
//! ## WASM considerations
//!
//! - No rayon (single-threaded); no SIMD (wasm32 portable)
//! - No filesystem access (checkpoint loaded from bytes)
//! - rmp-serde for checkpoint deserialization (same format as symthaea-broca)

use symthaea_broca::{
    BrocaCheckpoint, BrocaGenerator, GenerationResult as BrocaGenerationResult, ThoughtChannels,
};
use symthaea_core::genesis::GenesisSeed;

/// Result of generation from the full Broca pipeline, simplified for Spore.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineGenerationResult {
    /// Generated text.
    pub text: String,
    /// Number of tokens generated.
    pub num_tokens: usize,
    /// Whether generation was terminated by EOS.
    pub eos_terminated: bool,
    /// Whether a semantic veto was triggered (mid-sentence self-correction).
    pub veto_triggered: bool,
    /// Whether hallucination was detected (output drifted from thought intent).
    pub hallucination_flag: bool,
    /// Final coherence score.
    pub final_coherence: f32,
}

impl From<BrocaGenerationResult> for PipelineGenerationResult {
    fn from(r: BrocaGenerationResult) -> Self {
        Self {
            text: r.text,
            num_tokens: r.num_tokens,
            eos_terminated: r.eos_terminated,
            veto_triggered: r.veto_triggered,
            hallucination_flag: r.hallucination_flag,
            final_coherence: r.final_coherence,
        }
    }
}

/// Wrapper around the full symthaea-broca pipeline for Spore integration.
///
/// This provides the same interface as BrocaLite but backed by the production
/// BrocaGenerator with trained weights, BPE tokenizer, and full gating pipeline.
pub struct BrocaPipeline {
    generator: Option<BrocaGenerator>,
    genesis: GenesisSeed,
}

impl BrocaPipeline {
    /// Create a new pipeline wrapper. The generator is not initialized until
    /// a checkpoint is loaded via `load_checkpoint()`.
    pub fn new(genesis_seed: u64) -> Self {
        Self {
            generator: None,
            genesis: GenesisSeed::from_phrase(&format!("spore-broca-pipeline-{genesis_seed}")),
        }
    }

    /// Load a checkpoint from a byte buffer (fetched via WASM `fetch()`).
    ///
    /// Deserializes the BrocaCheckpoint from MessagePack (preferred) or bincode
    /// (legacy fallback), reconstructs the BrocaGenerator with trained weights
    /// and BPE tokenizer. After loading, `is_ready()` returns true.
    ///
    /// The checkpoint should be a distilled spore checkpoint (no optimizer state,
    /// reduced vocabulary) for browser deployment. Full checkpoints will also
    /// work but are very large (~976MB).
    pub fn load_checkpoint(&mut self, data: &[u8]) -> Result<(), String> {
        // Try MessagePack first (current format), then bincode (legacy)
        let checkpoint: BrocaCheckpoint =
            if let Ok(ckpt) = rmp_serde::from_slice::<BrocaCheckpoint>(data) {
                if !ckpt.verify() {
                    return Err("Checkpoint integrity check failed: checksum mismatch".into());
                }
                ckpt
            } else {
                bincode::deserialize(data).map_err(|e| {
                    format!("Failed to deserialize BrocaCheckpoint (tried msgpack + bincode): {e}")
                })?
            };

        // Reconstruct the generator from the checkpoint
        let tokenizer = symthaea_broca::BpeTokenizer::from_vocab_file(&checkpoint.vocab);
        let mut b_gen = BrocaGenerator::with_tokenizer(&self.genesis, checkpoint.config, tokenizer);

        // Restore trained weights
        *b_gen.controller_mut().token_embeddings_mut() = checkpoint.token_embeddings;
        *b_gen.controller_mut().network_mut() = checkpoint.network_state;

        self.generator = Some(b_gen);
        Ok(())
    }

    /// Check if the pipeline is ready (checkpoint loaded).
    pub fn is_ready(&self) -> bool {
        self.generator.is_some()
    }

    /// Generate text from consciousness state using the full Broca pipeline.
    ///
    /// Maps the Spore's cycle outputs to Broca's 24 ThoughtChannels and runs
    /// autoregressive generation with epistemic gating, emotional modulation,
    /// coherence feedback, and semantic veto.
    ///
    /// Returns `None` if no checkpoint is loaded.
    pub fn generate(
        &mut self,
        consciousness: f32,
        prediction_error: f32,
        harmony: f32,
        neuromods: [f32; 4],
        max_tokens: usize,
    ) -> Option<PipelineGenerationResult> {
        let b_gen = self.generator.as_mut()?;
        let channels = Self::build_channels(consciousness, prediction_error, harmony, neuromods);
        // Consciousness gates verbosity: low Psi = fewer tokens
        let effective_max = if consciousness < 0.15 {
            max_tokens.min(4)
        } else if consciousness < 0.3 {
            max_tokens.min(8)
        } else {
            max_tokens
        };
        // BrocaGenerator.generate applies its own consciousness gating on top
        let result = b_gen.generate(&channels);
        // Truncate to effective_max if the generator produced more
        let mut pipeline_result = PipelineGenerationResult::from(result);
        if pipeline_result.num_tokens > effective_max {
            // Truncate text to effective_max tokens (approximate by taking first N words)
            let words: Vec<&str> = pipeline_result.text.split_whitespace().collect();
            if words.len() > effective_max {
                pipeline_result.text = words[..effective_max].join(" ");
                pipeline_result.num_tokens = effective_max;
            }
        }
        Some(pipeline_result)
    }

    /// Generate text with awareness of user input.
    ///
    /// Injects intent signals from the user input into ThoughtChannels so that
    /// generated language relates to what the user said.
    pub fn generate_with_input(
        &mut self,
        input: &str,
        consciousness: f32,
        prediction_error: f32,
        harmony: f32,
        neuromods: [f32; 4],
        _max_tokens: usize,
    ) -> Option<PipelineGenerationResult> {
        let b_gen = self.generator.as_mut()?;
        let mut channels =
            Self::build_channels(consciousness, prediction_error, harmony, neuromods);
        Self::inject_intent(&mut channels, input);
        let result = b_gen.generate(&channels);
        Some(PipelineGenerationResult::from(result))
    }

    /// Build 24-channel ThoughtChannels from Spore cycle outputs.
    ///
    /// Channel mapping (24 channels):
    ///   0-6:  Intent channels (curiosity, valence, abstraction, urgency, social, creative, meta)
    ///   7:    Consciousness level (Psi)
    ///   8:    Prediction error
    ///   9:    Epistemic status (mapped from PE: high PE = uncertain)
    ///   10:   Harmony alignment
    ///   11:   Arousal (from noradrenaline)
    ///   12:   Valence (from serotonin - noradrenaline difference)
    ///   13:   Warmth (from oxytocin)
    ///   14:   Time pressure (from dopamine — high DA = urgent)
    ///   15:   Response confidence (inverse of PE)
    ///   16-23: Context channels (zeroed — populated by higher-level integration)
    fn build_channels(
        consciousness: f32,
        pe: f32,
        harmony: f32,
        neuromods: [f32; 4], // [DA, NE, 5-HT, Oxytocin]
    ) -> ThoughtChannels {
        let mut channels = ThoughtChannels::default();
        let c = &mut channels.channels;

        // Intent channels (0-6): derived from neuromodulators
        c[0] = neuromods[0].clamp(0.0, 1.0); // curiosity ~ dopamine
        c[1] = neuromods[2].clamp(0.0, 1.0); // valence ~ serotonin
        c[2] = consciousness.clamp(0.0, 1.0); // abstraction ~ consciousness
        c[3] = neuromods[1].clamp(0.0, 1.0); // urgency ~ noradrenaline
        c[4] = neuromods[3].clamp(0.0, 1.0); // social ~ oxytocin
        c[5] = 0.5; // creative (neutral default)
        c[6] = consciousness.clamp(0.0, 1.0); // meta ~ consciousness

        // State channels (7-15)
        c[7] = consciousness.clamp(0.0, 1.0);
        c[8] = pe.clamp(0.0, 1.0);
        // Epistemic: PE > 0.6 = Uncertain, PE > 0.4 = Possible, else = Probable
        c[9] = if pe > 0.6 {
            0.3 // Uncertain
        } else if pe > 0.4 {
            0.6 // Possible
        } else {
            0.85 // Probable
        };
        c[10] = harmony.clamp(0.0, 1.0);
        c[11] = neuromods[1].clamp(0.0, 1.0); // arousal ~ NE
        c[12] = (neuromods[2] - neuromods[1] * 0.5 + 0.5).clamp(0.0, 1.0); // valence
        c[13] = neuromods[3].clamp(0.0, 1.0); // warmth ~ oxytocin
        c[14] = (neuromods[0] * 0.5).clamp(0.0, 1.0); // time pressure ~ DA
        c[15] = (1.0 - pe).clamp(0.0, 1.0); // confidence

        // Context channels (16-23): zeroed by default
        channels
    }

    /// Inject intent signals from user text into ThoughtChannels.
    ///
    /// Simple keyword-based intent detection (matching BrocaLite's approach):
    /// question marks boost curiosity, exclamation marks boost urgency, etc.
    fn inject_intent(channels: &mut ThoughtChannels, input: &str) {
        let lower = input.to_lowercase();
        let c = &mut channels.channels;

        // Question detection -> curiosity
        if input.contains('?')
            || lower.starts_with("what")
            || lower.starts_with("how")
            || lower.starts_with("why")
            || lower.starts_with("who")
        {
            c[0] = (c[0] + 0.3).clamp(0.0, 1.0); // curiosity
        }

        // Exclamation -> urgency
        if input.contains('!') {
            c[3] = (c[3] + 0.2).clamp(0.0, 1.0); // urgency
        }

        // Social keywords -> social intent
        if lower.contains("you") || lower.contains("we") || lower.contains("together") {
            c[4] = (c[4] + 0.2).clamp(0.0, 1.0); // social
        }

        // Abstract/philosophical keywords -> abstraction
        if lower.contains("think")
            || lower.contains("consciousness")
            || lower.contains("meaning")
            || lower.contains("feel")
        {
            c[2] = (c[2] + 0.2).clamp(0.0, 1.0); // abstraction
            c[6] = (c[6] + 0.2).clamp(0.0, 1.0); // meta
        }

        // Negative sentiment -> lower valence
        if lower.contains("not") || lower.contains("don't") || lower.contains("can't") {
            c[1] = (c[1] - 0.15).clamp(0.0, 1.0); // valence
            c[12] = (c[12] - 0.1).clamp(0.0, 1.0); // emotional valence
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = BrocaPipeline::new(42);
        assert!(!pipeline.is_ready());
    }

    #[test]
    fn test_generate_without_checkpoint_returns_none() {
        let mut pipeline = BrocaPipeline::new(42);
        let result = pipeline.generate(0.5, 0.3, 0.7, [0.5, 0.4, 0.6, 0.3], 32);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_channels_mapping() {
        let channels = BrocaPipeline::build_channels(0.8, 0.3, 0.9, [0.5, 0.4, 0.6, 0.3]);
        let c = &channels.channels;

        // Consciousness level
        assert!((c[7] - 0.8).abs() < 1e-5);
        // Prediction error
        assert!((c[8] - 0.3).abs() < 1e-5);
        // Harmony
        assert!((c[10] - 0.9).abs() < 1e-5);
        // Epistemic: PE=0.3 -> Probable (0.85)
        assert!((c[9] - 0.85).abs() < 1e-5);
        // Curiosity ~ DA
        assert!((c[0] - 0.5).abs() < 1e-5);
        // Warmth ~ oxytocin
        assert!((c[13] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_build_channels_epistemic_uncertain() {
        let channels = BrocaPipeline::build_channels(0.5, 0.7, 0.5, [0.5, 0.5, 0.5, 0.5]);
        // PE=0.7 -> Uncertain (0.3)
        assert!((channels.channels[9] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_inject_intent_question() {
        let mut channels = ThoughtChannels::default();
        channels.channels[0] = 0.2;
        BrocaPipeline::inject_intent(&mut channels, "What is consciousness?");
        assert!(channels.channels[0] > 0.4); // curiosity boosted
    }

    #[test]
    fn test_inject_intent_social() {
        let mut channels = ThoughtChannels::default();
        channels.channels[4] = 0.3;
        BrocaPipeline::inject_intent(&mut channels, "Can we explore together?");
        assert!(channels.channels[4] > 0.4); // social boosted
    }

    #[test]
    fn test_load_invalid_checkpoint() {
        let mut pipeline = BrocaPipeline::new(42);
        let result = pipeline.load_checkpoint(b"not a valid checkpoint");
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_result_conversion() {
        let broca_result = BrocaGenerationResult {
            text: "hello world".to_string(),
            token_ids: vec![1, 2],
            num_tokens: 2,
            eos_terminated: true,
            veto_triggered: false,
            final_coherence: 0.85,
            long_coherence: 0.0,
            coherence_dynamics: vec![0.9, 0.85],
            gating_trace: vec![],
            hallucination_flag: false,
        };
        let result = PipelineGenerationResult::from(broca_result);
        assert_eq!(result.text, "hello world");
        assert_eq!(result.num_tokens, 2);
        assert!(result.eos_terminated);
        assert!(!result.veto_triggered);
        assert!(!result.hallucination_flag);
        assert!((result.final_coherence - 0.85).abs() < 1e-5);
    }
}
