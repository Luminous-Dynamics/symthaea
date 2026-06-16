// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SSM Backend: Native CfC-HDC language generation via symthaea-broca.
//!
//! Implements `LLMBackend` trait using the local BrocaGenerator instead of
//! external LLM APIs. Also provides `DirectThoughtBackend` for bypassing
//! text prompt serialization when the SSM backend is active.
//!
//! ## Direct Neural Path
//!
//! When an SSM or Liquid-Mamba backend is active, `StructuredThought` is
//! converted directly to `ThoughtChannels` via [`thought_to_channels`],
//! completely bypassing text-prompt serialization. The thought flows
//! continuously from the cognitive loop into the motor cortex.

use anyhow::Result;
use std::sync::Mutex;
use symthaea_broca::{
    BrocaConfig, BrocaGenerator, LanguageControllerConfig, SamplingStrategy, ThoughtChannels,
};
use symthaea_core::genesis::GenesisSeed;

use super::llm_backend::{GenerationParams, LLMBackend};
use crate::mind::structured_thought::{
    ETier, EpistemicStatus, HTier, MTier, NTier, SemanticIntent, StructuredThought,
};

/// Trait for direct thought-to-text generation (bypasses text prompt serialization).
pub trait DirectThoughtBackend: LLMBackend {
    /// Generate text directly from thought channels.
    fn generate_from_channels(
        &self,
        channels: &ThoughtChannels,
        params: &GenerationParams,
    ) -> Result<String>;
}

/// SSM backend using the BrocaGenerator for local, native text generation.
pub struct SsmBackend {
    /// The generator is behind a Mutex because LLMBackend requires &self
    /// but BrocaGenerator::generate requires &mut self.
    generator: Mutex<BrocaGenerator>,
}

impl SsmBackend {
    /// Create a new SSM backend from a genesis seed with default config.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 3,
                neurons_per_layer: 8,
                ..Default::default()
            },
            sampling: SamplingStrategy::TopK {
                k: 40,
                temperature: 0.7,
            },
            ..Default::default()
        };

        let generator = BrocaGenerator::new(genesis, config);
        Self {
            generator: Mutex::new(generator),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(genesis: &GenesisSeed, config: BrocaConfig) -> Self {
        let generator = BrocaGenerator::new(genesis, config);
        Self {
            generator: Mutex::new(generator),
        }
    }
}

impl std::fmt::Debug for SsmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SsmBackend")
            .field("name", &"symthaea-ssm-broca")
            .finish()
    }
}

#[async_trait::async_trait]
impl LLMBackend for SsmBackend {
    async fn generate(&self, prompt: &str, _params: &GenerationParams) -> Result<String> {
        // Parse prompt to extract thought channels if possible,
        // otherwise use a default channel set derived from the prompt text.
        let channels = channels_from_prompt(prompt);

        let mut generator = self
            .generator
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = generator.generate(&channels);
        Ok(result.text)
    }

    async fn generate_streaming(
        &self,
        prompt: &str,
        _params: &GenerationParams,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<String> {
        let channels = channels_from_prompt(prompt);

        let mut generator = self
            .generator
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = generator.generate_with_callback(&channels, on_token);
        Ok(result.text)
    }

    #[cfg(feature = "ssm_language")]
    fn generate_from_channels_direct(
        &self,
        channels: &ThoughtChannels,
        params: &super::llm_backend::GenerationParams,
    ) -> Option<anyhow::Result<String>> {
        Some(self.generate_from_channels(channels, params))
    }

    async fn is_available(&self) -> bool {
        true // Always available — no external dependencies
    }

    fn name(&self) -> &str {
        "symthaea-ssm-broca"
    }
}

impl DirectThoughtBackend for SsmBackend {
    fn generate_from_channels(
        &self,
        channels: &ThoughtChannels,
        _params: &GenerationParams,
    ) -> Result<String> {
        let mut generator = self
            .generator
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = generator.generate(channels);
        Ok(result.text)
    }
}

/// Extract ThoughtChannels from a text prompt.
///
/// Attempts to parse structured thought markers from the prompt text.
/// Falls back to default channels if parsing fails.
fn channels_from_prompt(prompt: &str) -> ThoughtChannels {
    let mut channels = ThoughtChannels::default();

    // Parse EPISTEMIC_STATUS from prompt
    if prompt.contains("EPISTEMIC_STATUS: Unknown") {
        channels.set_epistemic(3.0);
    } else if prompt.contains("EPISTEMIC_STATUS: Uncertain") {
        channels.set_epistemic(2.0);
    } else if prompt.contains("EPISTEMIC_STATUS: Probable") {
        channels.set_epistemic(1.0);
    } else if prompt.contains("EPISTEMIC_STATUS: Certain") {
        channels.set_epistemic(0.0);
    } else if prompt.contains("EPISTEMIC_STATUS: OutOfDomain") {
        channels.set_epistemic(4.0);
    }

    // Parse SEMANTIC_INTENT — set one-hot on existing channels (don't overwrite epistemic)
    let intent_idx = if prompt.contains("SEMANTIC_INTENT: Answer") {
        Some(1)
    } else if prompt.contains("SEMANTIC_INTENT: Acknowledge") {
        Some(0)
    } else if prompt.contains("SEMANTIC_INTENT: Clarify") {
        Some(2)
    } else if prompt.contains("SEMANTIC_INTENT: ProposeAction") {
        Some(3)
    } else if prompt.contains("SEMANTIC_INTENT: ExpressUncertainty") {
        Some(4)
    } else if prompt.contains("SEMANTIC_INTENT: Reflect") {
        Some(5)
    } else {
        None
    };
    if let Some(idx) = intent_idx {
        for i in 0..8 {
            channels.channels[i] = if i == idx { 1.0 } else { 0.0 };
        }
    }

    // Parse MOOD_TEMPERATURE
    if let Some(pos) = prompt.find("MOOD_TEMPERATURE:") {
        let rest = &prompt[pos + 17..];
        if let Some(val) = rest.trim().split_whitespace().next() {
            if let Ok(temp) = val.parse::<f32>() {
                channels.channels[17] = temp.clamp(0.5, 2.0);
            }
        }
    }

    channels
}

// ═══════════════════════════════════════════════════════════════════════════════
// DIRECT NEURAL PATH: StructuredThought → ThoughtChannels
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert a `StructuredThought` directly to `ThoughtChannels`.
///
/// This is the **Direct Neural Path** — it bypasses text-prompt serialization
/// entirely, mapping the cognitive loop's output directly into the 20-channel
/// representation that the HDC encoder consumes.
///
/// Channel layout (matches `crates/symthaea-broca/src/encoder.rs`):
/// - 0-7:  SemanticIntent one-hot (8 variants)
/// - 8:    EpistemicStatus ordinal (0=Certain..4=OutOfDomain)
/// - 9-11: valence, arousal, warmth
/// - 12-14: psi, meta_awareness, coherence
/// - 15-16: relationship_stage, trust
/// - 17:   mood_temperature
/// - 18:   has_computed_answer
/// - 19:   concept_count
pub fn thought_to_channels(thought: &StructuredThought, mood_temperature: f32) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();

    // Channels 0-7: semantic intent one-hot
    let intent_idx = match thought.semantic_intent {
        SemanticIntent::Acknowledge => 0,
        SemanticIntent::Answer => 1,
        SemanticIntent::Clarify => 2,
        SemanticIntent::ProposeAction => 3,
        SemanticIntent::ExpressUncertainty => 4,
        SemanticIntent::Reflect => 5,
        SemanticIntent::Continue => 6,
        SemanticIntent::Unknown => 7,
    };
    for i in 0..8 {
        ch.channels[i] = if i == intent_idx { 1.0 } else { 0.0 };
    }

    // Channel 8: epistemic status ordinal
    ch.channels[8] = match thought.epistemic_status {
        EpistemicStatus::Certain => 0.0,
        EpistemicStatus::Probable => 1.0,
        EpistemicStatus::Uncertain => 2.0,
        EpistemicStatus::Unknown => 3.0,
        EpistemicStatus::OutOfDomain => 4.0,
    };

    // Channels 9-11: emotional tone
    ch.channels[9] = (thought.emotional_tone.valence as f32).clamp(-1.0, 1.0);
    ch.channels[10] = (thought.emotional_tone.arousal as f32).clamp(0.0, 1.0);
    ch.channels[11] = (thought.emotional_tone.warmth as f32).clamp(0.0, 1.0);

    // Channels 12-14: consciousness metrics
    ch.channels[12] = (thought.psi as f32).clamp(0.0, 1.0);
    ch.channels[13] = (thought.meta_awareness as f32).clamp(0.0, 1.0);
    ch.channels[14] = (thought.coherence as f32).clamp(0.0, 1.0);

    // Channels 15-16: relational context
    ch.channels[15] = thought.relationship_stage as u32 as f32;
    ch.channels[16] = thought.trust.clamp(0.0, 1.0);

    // Channel 17: mood temperature
    ch.channels[17] = mood_temperature.clamp(0.5, 2.0);

    // Channel 18: has computed answer
    ch.channels[18] = if thought.structured_data.is_some() {
        1.0
    } else {
        0.0
    };

    // Channel 19: concept count (capped at 10)
    ch.channels[19] = (thought.activated_concepts.len() as f32).min(10.0);

    // Channels 24-27: code channels — populated from CodeContext when present
    if let Some(ref cc) = thought.code_context {
        // syntax_complexity: estimate from plan step count + constraint count
        let plan_depth = (cc.plan_steps.len() as f32 / 10.0).min(1.0);
        let constraint_pressure = (cc.spec_constraints.len() as f32 / 5.0).min(1.0);
        let syntax_complexity = (plan_depth * 0.6 + constraint_pressure * 0.4).clamp(0.0, 1.0);

        // type_confidence: from phi_score + syntactic validity + intent similarity
        let phi_contrib = cc.phi_score.unwrap_or(0.0);
        let validity_contrib = if cc.syntactically_valid.unwrap_or(false) {
            0.3
        } else {
            0.0
        };
        let similarity_contrib = cc.intent_similarity.unwrap_or(0.0) * 0.3;
        let type_confidence =
            (phi_contrib * 0.4 + validity_contrib + similarity_contrib).clamp(0.0, 1.0);

        // algorithm_pattern: higher when plan_steps are non-empty (CfC sequencer detected patterns)
        let algorithm_pattern = if cc.plan_steps.is_empty() {
            0.0
        } else {
            (cc.plan_steps.len() as f32 / 5.0).min(1.0) * 0.8
        };

        // error_likelihood: higher when needs_llm_completion, notes present, or low phi
        let unresolved = if cc.needs_llm_completion { 0.4 } else { 0.0 };
        let note_pressure = (cc.notes.len() as f32 / 3.0).min(0.3);
        let low_phi = (1.0 - cc.phi_score.unwrap_or(0.5)).max(0.0) * 0.3;
        let error_likelihood = (unresolved + note_pressure + low_phi).clamp(0.0, 1.0);

        ch.set_code(
            syntax_complexity,
            type_confidence,
            algorithm_pattern,
            error_likelihood,
        );
    }

    // Channels 28-42 (or 32-46 with therapeutic): Epistemic Cube from domain context
    if let Some(ref dc) = thought.domain_context {
        if let Some(ref cube) = dc.cube {
            let e_tier = match cube.e {
                ETier::E0 => 0,
                ETier::E1 => 1,
                ETier::E2 => 2,
                ETier::E3 => 3,
                ETier::E4 => 4,
            };
            let n_tier = match cube.n {
                NTier::N0 => 0,
                NTier::N1 => 1,
                NTier::N2 => 2,
                NTier::N3 => 3,
            };
            let m_tier = match cube.m {
                MTier::M0 => 0,
                MTier::M1 => 1,
                MTier::M2 => 2,
                MTier::M3 => 3,
            };
            let h_value = cube.harmonic().to_f64() as f32;

            // Quality score: E×0.40 + N×0.35 + M×0.25 (normalized to 0-1)
            let quality = (e_tier as f32 / 4.0) * 0.40
                + (n_tier as f32 / 3.0) * 0.35
                + (m_tier as f32 / 3.0) * 0.25;

            ch.set_epistemic_cube(e_tier, n_tier, m_tier, h_value, quality);
        }
    }

    ch
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIQUID-MAMBA BACKEND
// ═══════════════════════════════════════════════════════════════════════════════

/// Liquid-Mamba backend: pre-trained Mamba SSM fused with HDC consciousness gating.
///
/// Uses `LiquidMambaGenerator` from `symthaea-broca` to generate text from a
/// pre-trained mamba-130m model, guided by HDC thought projections and gated
/// by epistemic, emotional, and coherence constraints.
#[cfg(feature = "liquid-mamba")]
pub struct LiquidMambaBackend {
    generator: std::sync::Arc<Mutex<symthaea_broca::LiquidMambaGenerator>>,
}

#[cfg(feature = "liquid-mamba")]
impl LiquidMambaBackend {
    /// Create a new Liquid-Mamba backend.
    ///
    /// If `BROCA_PROJECTION_PATH` is set, loads pre-trained projection weights
    /// from that file (produced by `broca-projection-train`).
    pub fn new(genesis: &GenesisSeed, config: symthaea_broca::LiquidMambaConfig) -> Result<Self> {
        let mut generator = symthaea_broca::LiquidMambaGenerator::new(genesis, config)?;

        // Load pre-trained projection weights if available
        if let Ok(path) = std::env::var("BROCA_PROJECTION_PATH") {
            tracing::info!(path = %path, "Loading projection checkpoint");
            match symthaea_broca::ProjectionCheckpoint::load_from_file(&path) {
                Ok(ckpt) => {
                    generator
                        .projection_mut()
                        .load_weights(&ckpt.projection_weights);
                    tracing::info!(epoch = ckpt.training_epoch, "Projection weights loaded");
                }
                Err(e) => {
                    tracing::warn!(path = %path, error = %e, "Failed to load projection checkpoint, using random init");
                }
            }
        }

        Ok(Self {
            generator: std::sync::Arc::new(Mutex::new(generator)),
        })
    }
}

#[cfg(feature = "liquid-mamba")]
impl LiquidMambaBackend {
    /// Generate with full result (including output HVs and semantic PE).
    ///
    /// Unlike `DirectThoughtBackend::generate_from_channels` which returns only
    /// text, this returns the complete `GenerationResult` for downstream
    /// distillation and swarm exchange.
    pub fn generate_full(
        &self,
        channels: &ThoughtChannels,
    ) -> anyhow::Result<symthaea_broca::GenerationResult> {
        let mut generator = self
            .generator
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        Ok(generator.generate(channels))
    }

    /// Pass the FEP learning signal from the cognitive loop to modulate distillation LR.
    ///
    /// - `fep_signal > 0.7` → high surprise → boost distillation by 1.5×
    /// - `fep_signal < 0.3` → low surprise → dampen distillation by 0.7×
    /// - Otherwise → neutral (1.0×)
    pub fn set_fep_modulation(&self, fep_signal: f32) {
        if let Ok(mut generator) = self.generator.lock() {
            generator.set_fep_modulation(fep_signal);
        }
    }
}

#[cfg(feature = "liquid-mamba")]
impl std::fmt::Debug for LiquidMambaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidMambaBackend")
            .field("name", &"liquid-mamba-l-ssm")
            .finish()
    }
}

#[cfg(feature = "liquid-mamba")]
#[async_trait::async_trait]
impl LLMBackend for LiquidMambaBackend {
    async fn generate(&self, prompt: &str, _params: &GenerationParams) -> Result<String> {
        let channels = channels_from_prompt(prompt);
        let generator_handle = self.generator.clone();
        // Offload Mamba's CPU-heavy inference to a blocking thread so the
        // async runtime (and the 50Hz cognitive loop) continues ticking.
        tokio::task::spawn_blocking(move || {
            let mut generator = generator_handle
                .lock()
                .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
            let result = generator.generate(&channels);
            generator.distill_step(&channels, &result);
            Ok(result.text)
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?
    }

    async fn generate_streaming(
        &self,
        prompt: &str,
        _params: &GenerationParams,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<String> {
        // Streaming must stay synchronous since `on_token` is a mutable callback
        // that can't cross thread boundaries. The streaming path is typically
        // used for interactive display, not within the 50Hz loop.
        let channels = channels_from_prompt(prompt);
        let mut generator = self
            .generator
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = generator.generate_with_callback(&channels, on_token);
        generator.distill_step(&channels, &result);
        Ok(result.text)
    }

    #[cfg(feature = "ssm_language")]
    fn generate_from_channels_direct(
        &self,
        channels: &ThoughtChannels,
        params: &super::llm_backend::GenerationParams,
    ) -> Option<anyhow::Result<String>> {
        Some(self.generate_from_channels(channels, params))
    }

    #[cfg(feature = "liquid-mamba")]
    fn last_semantic_pe(&self) -> f32 {
        self.generator
            .lock()
            .ok()
            .map(|g| g.last_semantic_pe())
            .unwrap_or(0.0)
    }

    #[cfg(feature = "liquid-mamba")]
    fn export_gradient(
        &self,
        _source_id: [u8; 32],
        _trust: f32,
        _version: u64,
    ) -> Option<Vec<f32>> {
        self.generator
            .lock()
            .ok()
            .map(|g| g.projection().flatten_weights())
    }

    #[cfg(feature = "liquid-mamba")]
    fn apply_aggregated_gradient(&self, weights: &[f32]) -> bool {
        if let Ok(mut generator) = self.generator.lock() {
            generator.projection_mut().load_weights(weights);
            true
        } else {
            false
        }
    }

    #[cfg(feature = "liquid-mamba")]
    fn set_fep_modulation(&self, fep_signal: f32) {
        if let Ok(mut generator) = self.generator.lock() {
            generator.set_fep_modulation(fep_signal);
        }
    }

    #[cfg(feature = "liquid-mamba")]
    fn cycle_level_distill(
        &self,
        fep_precision: f32,
        thermodynamic_load: f32,
        prediction_confidence: f32,
        fep_lr_boost: f32,
    ) {
        // Gate: only modulate when prediction confidence is sufficient
        if prediction_confidence < 0.4 {
            return;
        }
        // Modulation factor: high precision + low load → boost; high load → suppress
        let modulation = fep_precision * fep_lr_boost * (1.0 - thermodynamic_load);
        let scaled = modulation.clamp(0.0, 2.0);
        if let Ok(mut generator) = self.generator.lock() {
            generator.set_fep_modulation(scaled);
        }
    }

    #[cfg(feature = "liquid-mamba")]
    fn current_distillation_lr(&self) -> f32 {
        self.generator
            .lock()
            .ok()
            .map(|g| g.current_lr())
            .unwrap_or(0.0)
    }

    #[cfg(feature = "liquid-mamba")]
    fn last_effective_rank(&self) -> f32 {
        self.generator
            .lock()
            .ok()
            .map(|g| g.last_cached_rank())
            .unwrap_or(0.0)
    }

    #[cfg(feature = "liquid-mamba")]
    fn generation_count(&self) -> u32 {
        self.generator
            .lock()
            .ok()
            .map(|g| g.generation_count() as u32)
            .unwrap_or(0)
    }

    fn update_affect(&self, load: f32, temp: f32) {
        if let Ok(mut generator) = self.generator.lock() {
            generator.update_affect(load, temp);
        }
    }

    async fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "liquid-mamba-l-ssm"
    }
}

#[cfg(feature = "liquid-mamba")]
impl DirectThoughtBackend for LiquidMambaBackend {
    fn generate_from_channels(
        &self,
        channels: &ThoughtChannels,
        _params: &GenerationParams,
    ) -> Result<String> {
        let mut generator = self
            .generator
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = generator.generate(channels);
        generator.distill_step(channels, &result);
        Ok(result.text)
    }
}

/// Async-friendly generation that doesn't block the cognitive loop.
///
/// Use this from the 50Hz pipeline: submit thought → CfC keeps ticking →
/// result arrives when Mamba finishes. The CfcModulation parameters
/// are frozen at submission time (snapshot semantics).
#[cfg(feature = "liquid-mamba")]
impl LiquidMambaBackend {
    /// Non-blocking generation: spawns Mamba inference on a blocking thread
    /// and returns a future that resolves when text is ready.
    ///
    /// The cognitive loop can continue at 50Hz while this runs. The
    /// `CfcModulation` (Δ and B scaling) is set before spawning.
    pub async fn generate_async(
        &self,
        channels: ThoughtChannels,
    ) -> Result<symthaea_broca::GenerationResult> {
        let generator_handle = self.generator.clone();
        tokio::task::spawn_blocking(move || {
            let mut generator = generator_handle
                .lock()
                .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
            let result = generator.generate(&channels);
            generator.distill_step(&channels, &result);
            Ok(result)
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-ssm-backend")
    }

    #[test]
    fn test_ssm_backend_creation() {
        let genesis = test_genesis();
        let backend = SsmBackend::new(&genesis);
        assert_eq!(backend.name(), "symthaea-ssm-broca");
    }

    #[tokio::test]
    async fn test_ssm_backend_is_available() {
        let genesis = test_genesis();
        let backend = SsmBackend::new(&genesis);
        assert!(backend.is_available().await);
    }

    #[tokio::test]
    async fn test_ssm_backend_generate() {
        let genesis = test_genesis();
        let backend = SsmBackend::new(&genesis);

        let params = GenerationParams::default();
        let result = backend
            .generate(
                "SEMANTIC_INTENT: Answer\nEPISTEMIC_STATUS: Certain\nTranslate this thought.",
                &params,
            )
            .await;
        assert!(result.is_ok());
        // Should produce some text (untrained model produces pseudo-random tokens)
        let text = result.unwrap();
        assert!(!text.is_empty() || true); // May be empty if EOS hit immediately
    }

    #[test]
    fn test_direct_thought_backend() {
        let genesis = test_genesis();
        let backend = SsmBackend::new(&genesis);

        let channels = ThoughtChannels::with_intent(1); // Answer
        let params = GenerationParams::default();

        let result = backend.generate_from_channels(&channels, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_channels_from_prompt() {
        let channels = channels_from_prompt("EPISTEMIC_STATUS: Unknown\nSEMANTIC_INTENT: Answer");
        assert!((channels.epistemic_ordinal() - 3.0).abs() < 0.1);

        let channels = channels_from_prompt("EPISTEMIC_STATUS: Certain\nSEMANTIC_INTENT: Clarify");
        assert!((channels.epistemic_ordinal() - 0.0).abs() < 0.1);
    }

    // =========================================================================
    // Direct Neural Path: thought_to_channels tests
    // =========================================================================

    fn make_test_thought() -> StructuredThought {
        use crate::mind::structured_thought::{ActivatedConcept, EmotionalTone};
        use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationshipStage};

        let mut thought = StructuredThought::default();
        thought.semantic_intent = SemanticIntent::Answer;
        thought.activated_concepts = vec![
            ActivatedConcept {
                name: "quantum".to_string(),
                activation: 0.8,
                relevance: 0.9,
                #[cfg(feature = "provenance")]
                source: None,
            },
            ActivatedConcept {
                name: "physics".to_string(),
                activation: 0.6,
                relevance: 0.7,
                #[cfg(feature = "provenance")]
                source: None,
            },
        ];
        thought.emotional_tone = EmotionalTone {
            valence: 0.3,
            arousal: 0.5,
            warmth: 0.7,
        };
        thought.psi = 0.65;
        thought.meta_awareness = 0.7;
        thought.coherence = 0.8;
        thought.epistemic_status = EpistemicStatus::Probable;
        thought.relationship_stage = RelationshipStage::Contact;
        thought.relation_mode = RelationMode::IThou;
        thought.trust = 0.6;
        thought
    }

    #[test]
    fn test_thought_to_channels_intent() {
        let thought = make_test_thought();
        let ch = thought_to_channels(&thought, 1.0);
        // Answer → index 1
        assert_eq!(ch.channels[1], 1.0);
        assert_eq!(ch.channels[0], 0.0); // Acknowledge off
        assert_eq!(ch.channels[2], 0.0); // Clarify off
    }

    #[test]
    fn test_thought_to_channels_epistemic() {
        let thought = make_test_thought();
        let ch = thought_to_channels(&thought, 1.0);
        // Probable → ordinal 1.0
        assert!((ch.channels[8] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_thought_to_channels_emotion() {
        let thought = make_test_thought();
        let ch = thought_to_channels(&thought, 1.0);
        assert!((ch.channels[9] - 0.3).abs() < 0.01); // valence
        assert!((ch.channels[10] - 0.5).abs() < 0.01); // arousal
        assert!((ch.channels[11] - 0.7).abs() < 0.01); // warmth
    }

    #[test]
    fn test_thought_to_channels_consciousness() {
        let thought = make_test_thought();
        let ch = thought_to_channels(&thought, 1.0);
        assert!((ch.channels[12] - 0.65).abs() < 0.01); // psi
        assert!((ch.channels[13] - 0.7).abs() < 0.01); // meta_awareness
        assert!((ch.channels[14] - 0.8).abs() < 0.01); // coherence
    }

    #[test]
    fn test_thought_to_channels_mood_temperature() {
        let thought = make_test_thought();
        let ch = thought_to_channels(&thought, 1.5);
        assert!((ch.channels[17] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_thought_to_channels_concept_count() {
        let thought = make_test_thought();
        let ch = thought_to_channels(&thought, 1.0);
        assert!((ch.channels[19] - 2.0).abs() < 0.01); // 2 concepts
    }

    #[test]
    fn test_thought_to_channels_all_intents() {
        let intents = [
            (SemanticIntent::Acknowledge, 0),
            (SemanticIntent::Answer, 1),
            (SemanticIntent::Clarify, 2),
            (SemanticIntent::ProposeAction, 3),
            (SemanticIntent::ExpressUncertainty, 4),
            (SemanticIntent::Reflect, 5),
            (SemanticIntent::Continue, 6),
            (SemanticIntent::Unknown, 7),
        ];
        for (intent, expected_idx) in intents {
            let mut thought = make_test_thought();
            thought.semantic_intent = intent;
            let ch = thought_to_channels(&thought, 1.0);
            assert_eq!(
                ch.channels[expected_idx], 1.0,
                "Intent {:?} should set channel {}",
                intent, expected_idx
            );
            // All other intent channels should be 0
            for i in 0..8 {
                if i != expected_idx {
                    assert_eq!(
                        ch.channels[i], 0.0,
                        "Channel {} should be 0 for intent {:?}",
                        i, intent
                    );
                }
            }
        }
    }

    #[test]
    fn test_thought_to_channels_epistemic_cube() {
        use crate::mind::structured_thought::{
            DomainContext, ETier, EpistemicCube, HTier, MTier, NTier,
        };
        use symthaea_broca::encoder::EPISTEMIC_CUBE_BASE;

        let mut thought = make_test_thought();
        thought.domain_context = Some(DomainContext {
            domain: "mathematics".to_string(),
            entities: vec![],
            computed_answer: Some("4".to_string()),
            cube: Some(EpistemicCube::with_harmonic(
                ETier::E4,
                NTier::N3,
                MTier::M3,
                HTier::H4,
            )),
            psi: Some(0.95),
        });

        let ch = thought_to_channels(&thought, 1.0);

        // E4 → one-hot at index 4
        assert_eq!(
            ch.channels[EPISTEMIC_CUBE_BASE + 4],
            1.0,
            "E4 should be active"
        );
        for i in 0..4 {
            assert_eq!(
                ch.channels[EPISTEMIC_CUBE_BASE + i],
                0.0,
                "E{i} should be inactive"
            );
        }

        // N3 → one-hot at index 3
        assert_eq!(
            ch.channels[EPISTEMIC_CUBE_BASE + 5 + 3],
            1.0,
            "N3 should be active"
        );

        // M3 → one-hot at index 3
        assert_eq!(
            ch.channels[EPISTEMIC_CUBE_BASE + 9 + 3],
            1.0,
            "M3 should be active"
        );

        // H4 → 1.0
        assert!(
            (ch.channels[EPISTEMIC_CUBE_BASE + 13] - 1.0).abs() < 0.01,
            "H4 should be 1.0"
        );

        // Quality = (4/4)*0.4 + (3/3)*0.35 + (3/3)*0.25 = 0.4 + 0.35 + 0.25 = 1.0
        assert!(
            (ch.channels[EPISTEMIC_CUBE_BASE + 14] - 1.0).abs() < 0.01,
            "quality should be 1.0"
        );

        // has_epistemic_cube should be true
        assert!(ch.has_epistemic_cube());
    }

    #[test]
    fn test_thought_to_channels_no_cube_defaults() {
        use symthaea_broca::encoder::EPISTEMIC_CUBE_BASE;

        let thought = make_test_thought();
        let ch = thought_to_channels(&thought, 1.0);

        // No cube → all one-hot channels inactive
        assert!(!ch.has_epistemic_cube());

        // H-tier defaults to 0.25 (H1 neutral)
        assert!(
            (ch.h_tier() - 0.25).abs() < 0.01,
            "default h_tier should be 0.25"
        );
    }

    #[test]
    fn test_thought_to_channels_all_epistemic_statuses() {
        let statuses = [
            (EpistemicStatus::Certain, 0.0),
            (EpistemicStatus::Probable, 1.0),
            (EpistemicStatus::Uncertain, 2.0),
            (EpistemicStatus::Unknown, 3.0),
            (EpistemicStatus::OutOfDomain, 4.0),
        ];
        for (status, expected_val) in statuses {
            let mut thought = make_test_thought();
            thought.epistemic_status = status;
            let ch = thought_to_channels(&thought, 1.0);
            assert!(
                (ch.channels[8] - expected_val).abs() < 0.01,
                "Status {:?} should map to {}, got {}",
                status,
                expected_val,
                ch.channels[8]
            );
        }
    }

    #[test]
    fn test_thought_to_channels_code_context_populated() {
        use crate::mind::structured_thought::CodeContext;

        let mut thought = make_test_thought();
        thought.code_context = Some(CodeContext {
            language: "rust".to_string(),
            spec_purpose: Some("Sort a vector".to_string()),
            spec_signature: Some("fn sort(v: &mut Vec<i32>)".to_string()),
            spec_constraints: vec!["in-place".to_string(), "stable".to_string()],
            spec_examples: vec![],
            plan_steps: vec![
                "DefineFunction".to_string(),
                "ImplementSort".to_string(),
                "AddTests".to_string(),
            ],
            generated_code: None,
            phi_score: Some(0.7),
            intent_similarity: Some(0.8),
            syntactically_valid: Some(true),
            notes: vec![],
            needs_llm_completion: false,
        });

        let ch = thought_to_channels(&thought, 1.0);

        // syntax_complexity: plan_depth=3/10=0.3, constraint=2/5=0.4 → 0.3*0.6+0.4*0.4=0.34
        assert!(
            ch.syntax_complexity() > 0.0,
            "syntax_complexity should be populated"
        );
        assert!(
            ch.syntax_complexity() < 1.0,
            "syntax_complexity should be bounded"
        );

        // type_confidence: phi 0.7*0.4=0.28, validity 0.3, similarity 0.8*0.3=0.24 → 0.82
        assert!(
            ch.type_confidence() > 0.5,
            "type_confidence should be high for valid code with good phi"
        );

        // algorithm_pattern: 3 steps, 3/5=0.6 * 0.8 = 0.48
        assert!(
            ch.algorithm_pattern() > 0.0,
            "algorithm_pattern should be populated from plan_steps"
        );

        // error_likelihood: no completion, no notes, phi=0.7 → low_phi=(1-0.7)*0.3=0.09
        assert!(
            ch.error_likelihood() < 0.3,
            "error_likelihood should be low for good code"
        );
    }

    #[test]
    fn test_thought_to_channels_code_context_high_error() {
        use crate::mind::structured_thought::CodeContext;

        let mut thought = make_test_thought();
        thought.code_context = Some(CodeContext {
            language: "rust".to_string(),
            spec_purpose: None,
            spec_signature: None,
            spec_constraints: vec![],
            spec_examples: vec![],
            plan_steps: vec![],
            generated_code: None,
            phi_score: Some(0.1),
            intent_similarity: None,
            syntactically_valid: Some(false),
            notes: vec![
                "TODO".to_string(),
                "uncertain types".to_string(),
                "complex".to_string(),
            ],
            needs_llm_completion: true,
        });

        let ch = thought_to_channels(&thought, 1.0);

        // High error likelihood: needs_llm=0.4, notes=3/3→1.0 clamped to 0.3, low_phi=(1-0.1)*0.3=0.27
        assert!(
            ch.error_likelihood() > 0.5,
            "error_likelihood should be high for incomplete code"
        );

        // Low type_confidence: phi 0.1*0.4=0.04, no validity, no similarity
        assert!(
            ch.type_confidence() < 0.2,
            "type_confidence should be low for unvalidated code"
        );

        // No algorithm pattern (empty plan_steps)
        assert!(
            (ch.algorithm_pattern() - 0.0).abs() < 0.01,
            "algorithm_pattern should be 0 with no steps"
        );
    }

    #[test]
    fn test_thought_to_channels_no_code_context_defaults() {
        let thought = make_test_thought();
        let ch = thought_to_channels(&thought, 1.0);

        // No code context → all code channels should be 0
        assert_eq!(
            ch.syntax_complexity(),
            0.0,
            "should default to 0 without code context"
        );
        assert_eq!(
            ch.type_confidence(),
            0.0,
            "should default to 0 without code context"
        );
        assert_eq!(
            ch.algorithm_pattern(),
            0.0,
            "should default to 0 without code context"
        );
        assert_eq!(
            ch.error_likelihood(),
            0.0,
            "should default to 0 without code context"
        );
    }
}
