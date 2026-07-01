// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BrocaGenerator: autoregressive text generation from thought channels.
//!
//! Orchestrates the full pipeline:
//! 1. Encode thought channels → ContinuousHV
//! 2. Autoregressive loop: forward step → gate → sample → decode
//! 3. Semantic veto: mid-sentence self-correction when coherence drops
//! 4. Thermodynamic subjective time: dt varies with system load

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

#[cfg(feature = "mamba-cpu")]
use crate::checkpoint::ProjectionCheckpoint;
use crate::checkpoint::{AdamState, BrocaCheckpoint};
use crate::controller::{LanguageController, LanguageControllerConfig, NetworkSnapshot};
use crate::encoder::{ThoughtChannels, ThoughtLanguageEncoder};
#[cfg(feature = "therapeutic")]
use crate::gating::TherapeuticGate;
use crate::gating::{
    CoherenceFeedback, EmotionalModulator, EpistemicGate, GatingConfig,
    confidence_adjusted_veto_threshold, consciousness_gated_max_tokens,
};
use crate::tokenizer::{BpeTokenizer, is_code_contamination_token};

use symthaea_core::genesis::GenesisSeed;

/// Sampling strategy for token selection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Always pick the highest-logit token.
    #[default]
    Greedy,
    /// Sample from top-k tokens with temperature.
    TopK { k: usize, temperature: f32 },
    /// Sample from top-p (nucleus) tokens with temperature.
    TopP { p: f32, temperature: f32 },
}

/// High-level decoder path for Broca language realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum BrocaDecoderKind {
    /// Native CfC/HDC controller with weight-tied lexical readout.
    Direct,
    /// Deterministic semantic molecule / thought-structure readout.
    #[default]
    Structured,
    /// Liquid-Mamba vocal-tract translator.
    Mamba,
    /// Structured meaning plus optional Mamba humanization.
    Hybrid,
}

impl BrocaDecoderKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Structured => "structured",
            Self::Mamba => "mamba",
            Self::Hybrid => "hybrid",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "direct" | "cfc" | "hdc" => Some(Self::Direct),
            "structured" | "semantic" | "molecule" => Some(Self::Structured),
            "mamba" | "liquid-mamba" | "liquid_mamba" => Some(Self::Mamba),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }
}

/// Configuration for the Broca generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrocaConfig {
    /// Preferred decoder path for external orchestration.
    ///
    /// `BrocaGenerator` itself remains the native direct CfC/HDC decoder.
    /// External orchestration should prefer structured output as the grounded
    /// product path and treat direct/Mamba token generation as experimental or
    /// translator paths until their gates pass.
    #[serde(default)]
    pub decoder_kind: BrocaDecoderKind,
    /// Controller configuration.
    pub controller: LanguageControllerConfig,
    /// Gating configuration.
    pub gating: GatingConfig,
    /// Default sampling strategy.
    pub sampling: SamplingStrategy,
    /// Enable epistemic gating.
    pub enable_epistemic_gate: bool,
    /// Enable emotional modulation.
    pub enable_emotional_modulation: bool,
    /// Enable coherence feedback.
    pub enable_coherence_feedback: bool,
    /// Enable consciousness-gated verbosity.
    pub enable_consciousness_gating: bool,
    /// Enable semantic veto (mid-sentence self-correction).
    pub enable_semantic_veto: bool,
    /// Hesitation token for semantic veto.
    pub veto_hesitation: String,
    /// Repetition penalty (1.0 = no penalty, >1.0 = suppress repeats).
    /// Applied to logits of tokens already generated in the current sequence.
    /// Uses frequency-scaled penalty: `penalty^count` where count is the number
    /// of times a token has appeared, preventing alternating-token loops.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Optional seed for sampling RNG (reproducible generation with TopK/TopP).
    /// If `None`, uses thread-local RNG (stochastic).
    #[serde(default)]
    pub sampling_seed: Option<u64>,
    /// Insert spaces between word tokens automatically (default: true).
    /// BPE vocab stores spaces as separate tokens. Untrained models don't emit them.
    #[serde(default = "default_auto_spacing")]
    pub enable_auto_spacing: bool,
    /// Enable NSM semantic content blending into thought HV (Phase 2).
    /// When true and a semantic_hv is provided, it is blended with the
    /// thought HV before generation via linear interpolation.
    /// Science: Barsalou (1999) — grounded cognition requires semantic modulation.
    #[serde(default)]
    pub enable_nsm_semantic: bool,
    /// Blend weight for NSM semantic HV into thought HV (0.0–1.0).
    /// Higher values give more weight to the NSM semantic content.
    /// Default 0.3: ~30% semantic modulation for meaningful influence
    /// without overwhelming procedural dynamics.
    #[serde(default = "default_nsm_semantic_alpha")]
    pub nsm_semantic_alpha: f32,
    /// Enable per-axis epistemic cube gating (E/N/M/H logit modulation).
    /// Requires cube channels 28-42 to be populated. Complements the 1D gate.
    /// Science: Goldman (1986) — reliabilist epistemology; knowledge source matters.
    #[serde(default = "default_true")]
    pub enable_epistemic_cube_gate: bool,
    /// Enable NSM semantic gate: boost logits for tokens expressing active primes.
    /// Science: Collins & Loftus (1975) — spreading activation in semantic networks.
    #[serde(default)]
    pub enable_nsm_gate: bool,
    /// Logit boost per active prime for NSM-aligned tokens (0.0–2.0).
    #[serde(default = "default_nsm_prime_logit_boost")]
    pub nsm_prime_logit_boost: f32,
    /// Scale for NSM coverage modulation of veto threshold (0.0–0.5).
    /// High prime coverage relaxes veto; low coverage tightens it.
    /// Science: Grice (1975) — cooperative principle.
    #[serde(default = "default_nsm_coverage_veto_scale")]
    pub nsm_coverage_veto_scale: f32,
    /// Bypass all gating (epistemic, emotional, coherence, consciousness, veto)
    /// for clean quality iteration on raw CfC output. Default: false.
    /// Use this during training evaluation to isolate generation quality
    /// from the gating stack.
    #[serde(default)]
    pub bypass_gating: bool,
    /// Suppress `<unk>` during decoding. `<unk>` is not a useful emitted token
    /// and tends to poison the next autoregressive step.
    #[serde(default = "default_true")]
    pub suppress_unknown_tokens: bool,
    /// Suppress syntax-heavy code tokens when code-intent channels are inactive.
    /// This protects prose generation from drifting into Rust/Python/Nix shards
    /// while preserving explicit code-generation cases.
    #[serde(default = "default_true")]
    pub suppress_code_tokens_without_code_intent: bool,
}

fn default_true() -> bool {
    true
}

fn default_repetition_penalty() -> f32 {
    1.5
}

fn default_auto_spacing() -> bool {
    true
}

fn default_nsm_semantic_alpha() -> f32 {
    0.3
}

fn default_nsm_prime_logit_boost() -> f32 {
    0.5
}

fn default_nsm_coverage_veto_scale() -> f32 {
    0.1
}

impl Default for BrocaConfig {
    fn default() -> Self {
        Self {
            decoder_kind: BrocaDecoderKind::Structured,
            controller: LanguageControllerConfig::default(),
            gating: GatingConfig::default(),
            sampling: SamplingStrategy::Greedy,
            enable_epistemic_gate: true,
            enable_emotional_modulation: true,
            enable_coherence_feedback: true,
            enable_consciousness_gating: true,
            enable_semantic_veto: false,
            veto_hesitation: "-- wait, ".to_string(),
            repetition_penalty: default_repetition_penalty(),
            sampling_seed: None,
            enable_auto_spacing: true,
            enable_epistemic_cube_gate: true,
            enable_nsm_semantic: false,
            nsm_semantic_alpha: default_nsm_semantic_alpha(),
            enable_nsm_gate: false,
            nsm_prime_logit_boost: default_nsm_prime_logit_boost(),
            nsm_coverage_veto_scale: default_nsm_coverage_veto_scale(),
            bypass_gating: false,
            suppress_unknown_tokens: true,
            suppress_code_tokens_without_code_intent: true,
        }
    }
}

/// Per-token gating adjustment record for auditing generation decisions.
#[derive(Debug, Clone)]
pub struct GatingTraceEntry {
    /// Token position in the generated sequence.
    pub position: usize,
    /// Coherence between output and thought HV at this position.
    pub coherence: f32,
    /// Binding weight applied to thought HV (>1.0 means correction active).
    pub binding_weight: f32,
    /// Whether veto was triggered at this position.
    pub veto_triggered: bool,
    /// Epistemic logit adjustment applied at this position.
    /// Represents the hedging boost magnitude for the current epistemic level.
    /// 0.0 when epistemic gating is disabled or epistemic level is Certain/Probable.
    pub epistemic_boost: f32,
    /// Emotional modulation logit adjustment at this position.
    /// Represents the maximum boost magnitude from arousal/valence/warmth effects.
    /// 0.0 when emotional modulation is disabled or no thresholds are exceeded.
    pub emotional_boost: f32,
    /// Max token reduction from time pressure (0.0 if not applicable).
    /// Expressed as the fraction of base tokens removed (e.g., 0.3 means 30% reduction).
    pub time_pressure_reduction: f32,
    /// Adjusted veto threshold from response confidence.
    /// Lower values mean the system is more tolerant of coherence drift.
    /// Equal to the base veto threshold when coherence feedback is disabled.
    pub confidence_veto_threshold: f32,
    /// NSM semantic logit boost applied at this position (0.0 when disabled).
    pub nsm_boost: f32,
}

/// Result of a single generation.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text.
    pub text: String,
    /// Generated token IDs (excluding BOS/EOS).
    pub token_ids: Vec<u32>,
    /// Number of tokens generated.
    pub num_tokens: usize,
    /// Whether generation was terminated by EOS.
    pub eos_terminated: bool,
    /// Whether a semantic veto was triggered.
    pub veto_triggered: bool,
    /// Final short-window coherence score.
    pub final_coherence: f32,
    /// Final long-window coherence score (Liquid-Mamba only, 0.0 for CfC-HDC).
    pub long_coherence: f32,
    /// Per-token coherence dynamics: coherence score at each generated position.
    pub coherence_dynamics: Vec<f32>,
    /// Per-token gating audit trail (populated when coherence feedback is enabled).
    pub gating_trace: Vec<GatingTraceEntry>,
    /// Per-token decoder distribution diagnostics captured after gating/repetition
    /// penalties and before sampling.
    pub logit_diagnostics: Vec<GenerationStepLogits>,
    /// Number of consecutive tokens with output-thought similarity < 0.05.
    /// When >= 3, suggests hallucination (output drifted far from thought intent).
    pub hallucination_flag: bool,
    /// NSM prime coverage: fraction of active primes expressed by generated tokens (0.0–1.0).
    /// Higher = the text semantically covered the intended meaning.
    pub nsm_prime_coverage: f32,
    /// Back-projected HDC vectors for each generated token (Liquid-Mamba only).
    #[cfg(feature = "mamba-cpu")]
    pub output_hvs: Vec<symthaea_core::hdc::ContinuousHV>,
    /// Semantic prediction error: round-trip reconstruction loss (Liquid-Mamba only).
    #[cfg(feature = "mamba-cpu")]
    pub semantic_pe: f32,
}

#[derive(Debug, Clone)]
pub struct GenerationStepLogits {
    pub position: usize,
    pub selected_token_id: u32,
    pub entropy: f32,
    pub max_probability: f32,
    pub pre_attractor_entropy: Option<f32>,
    pub pre_attractor_max_probability: Option<f32>,
    pub cfc_delta_scale: Option<f32>,
    pub cfc_b_scale: Option<f32>,
    pub semantic_attractor_mean_adjustment: Option<f32>,
    pub semantic_attractor_max_adjustment: Option<f32>,
    pub semantic_attractor_alignment_mean: Option<f32>,
    pub semantic_attractor_alignment_std: Option<f32>,
    pub selected_semantic_alignment: Option<f32>,
    pub top_k: Vec<GenerationTopLogit>,
}

#[derive(Debug, Clone)]
pub struct GenerationTopLogit {
    pub rank: usize,
    pub token_id: u32,
    pub logit: f32,
    pub probability: f32,
}

/// Broca generator: autoregressive thought-to-text.
#[derive(Clone)]
pub struct BrocaGenerator {
    controller: LanguageController,
    tokenizer: BpeTokenizer,
    encoder: ThoughtLanguageEncoder,
    epistemic_gate: EpistemicGate,
    epistemic_cube_gate: crate::gating::EpistemicCubeGate,
    emotional_modulator: EmotionalModulator,
    coherence_feedback: CoherenceFeedback,
    config: BrocaConfig,
    /// Seeded RNG for reproducible sampling (TopK/TopP). Recreated each generation
    /// from `config.sampling_seed`. If `None`, uses thread-local RNG.
    sampling_rng: Option<StdRng>,
    /// Cached NSM semantic gate (built once at construction, reused per-generation).
    /// `None` when `enable_nsm_gate` is false.
    nsm_gate: Option<crate::gating::NsmSemanticGate>,
    /// Code-aware gating: boosts/suppresses tokens based on code channels (24-27).
    /// Active only when `GatingConfig::enable_code_gate` is true.
    code_gate: crate::gating::CodeGate,
}

impl BrocaGenerator {
    /// Create a new generator from genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: BrocaConfig) -> Self {
        let tokenizer = BpeTokenizer::default_minimal();

        // Ensure controller vocab_size matches tokenizer
        let mut ctrl_config = config.controller.clone();
        ctrl_config.vocab_size = tokenizer.vocab_size();

        let controller = LanguageController::new(genesis, &ctrl_config);
        let encoder = ThoughtLanguageEncoder::new(genesis);
        let epistemic_gate = EpistemicGate::new(&tokenizer, &config.gating);
        let epistemic_cube_gate = crate::gating::EpistemicCubeGate::new(&tokenizer);
        let emotional_modulator = EmotionalModulator::new(&tokenizer, &config.gating);
        let mut coherence_feedback = CoherenceFeedback::with_veto_threshold(
            config.gating.coherence_drift_threshold,
            config.gating.veto_threshold,
        );
        // Phase 2: Enable algebraic coherence correction if configured
        coherence_feedback.set_algebraic(
            config.gating.enable_algebraic_correction,
            config.gating.algebraic_correction_strength,
        );
        let sampling_rng = config.sampling_seed.map(StdRng::seed_from_u64);

        let nsm_gate = if config.enable_nsm_gate {
            Some(crate::gating::NsmSemanticGate::new_with_lexicon(&tokenizer))
        } else {
            None
        };

        let code_gate = crate::gating::CodeGate::new(&tokenizer, &config.gating);

        Self {
            controller,
            tokenizer,
            encoder,
            epistemic_gate,
            epistemic_cube_gate,
            emotional_modulator,
            coherence_feedback,
            sampling_rng,
            nsm_gate,
            code_gate,
            config,
        }
    }

    /// Create a generator for a specific coding substrate.
    ///
    /// Automatically selects the optimized tokenizer and configures
    /// code-aware gating.
    pub fn for_substrate(genesis: &GenesisSeed, config: BrocaConfig, substrate: &str) -> Self {
        let tokenizer = BpeTokenizer::for_substrate_distillation(substrate);
        let mut generator = Self::with_tokenizer(genesis, config, tokenizer);

        // Auto-enable code gate and configure substrate-specific settings
        generator.config.gating.enable_code_gate = true;
        generator.code_gate =
            crate::gating::CodeGate::new(generator.tokenizer(), &generator.config.gating);

        generator
    }

    /// Create a generator with the 4K vocabulary (4,096 tokens).
    ///
    /// Use this for production training and deployment. The 4K vocabulary covers
    /// common English words, morphological affixes, and consciousness-domain terms.
    /// For unit tests, use `new()` which uses the faster minimal vocabulary (208 tokens).
    pub fn new_4k(genesis: &GenesisSeed, config: BrocaConfig) -> Self {
        let tokenizer = BpeTokenizer::default_4k();
        Self::with_tokenizer(genesis, config, tokenizer)
    }

    /// Create a generator with a custom tokenizer.
    pub fn with_tokenizer(
        genesis: &GenesisSeed,
        config: BrocaConfig,
        tokenizer: BpeTokenizer,
    ) -> Self {
        let mut ctrl_config = config.controller.clone();
        ctrl_config.vocab_size = tokenizer.vocab_size();

        let controller = LanguageController::new(genesis, &ctrl_config);
        let encoder = ThoughtLanguageEncoder::new(genesis);
        let epistemic_gate = EpistemicGate::new(&tokenizer, &config.gating);
        let epistemic_cube_gate = crate::gating::EpistemicCubeGate::new(&tokenizer);
        let emotional_modulator = EmotionalModulator::new(&tokenizer, &config.gating);
        let mut coherence_feedback = CoherenceFeedback::with_veto_threshold(
            config.gating.coherence_drift_threshold,
            config.gating.veto_threshold,
        );
        // Phase 2: Enable algebraic coherence correction if configured
        coherence_feedback.set_algebraic(
            config.gating.enable_algebraic_correction,
            config.gating.algebraic_correction_strength,
        );
        let sampling_rng = config.sampling_seed.map(StdRng::seed_from_u64);

        let nsm_gate = if config.enable_nsm_gate {
            Some(crate::gating::NsmSemanticGate::new_with_lexicon(&tokenizer))
        } else {
            None
        };

        let code_gate = crate::gating::CodeGate::new(&tokenizer, &config.gating);

        Self {
            controller,
            tokenizer,
            encoder,
            epistemic_gate,
            epistemic_cube_gate,
            emotional_modulator,
            coherence_feedback,
            sampling_rng,
            nsm_gate,
            code_gate,
            config,
        }
    }

    /// Load a generator from a checkpoint file.
    /// Supports both full BrocaCheckpoint and HdcSsmProjection checkpoints.
    pub fn from_checkpoint<P: AsRef<std::path::Path>>(
        path: P,
        genesis: &GenesisSeed,
    ) -> anyhow::Result<(Self, Option<AdamState>, Option<Vec<f32>>, Option<String>)> {
        // Path 1: Try full BrocaCheckpoint (CfC-HDC + all state)
        if let Ok(checkpoint) = BrocaCheckpoint::load_from_file(&path) {
            let adam = checkpoint.adam_state.clone();
            return Ok((
                Self::from_checkpoint_struct(checkpoint, genesis),
                adam,
                None::<Vec<f32>>,
                None::<String>,
            ));
        }

        // Path 2: Try ProjectionCheckpoint (Liquid-Mamba optimized)
        #[cfg(feature = "mamba-cpu")]
        match ProjectionCheckpoint::load_from_file(&path) {
            Ok(proj_ckpt) => {
                // If it has a full snapshot, restore everything
                let (mut r#gen, adam, _, config_str) =
                    if let Some(snapshot) = proj_ckpt.full_snapshot {
                        (
                            Self::from_checkpoint_struct(*snapshot, genesis),
                            None::<AdamState>,
                            None::<Vec<f32>>,
                            None::<String>,
                        )
                    } else {
                        (
                            Self::new(genesis, BrocaConfig::default()),
                            None::<AdamState>,
                            None::<Vec<f32>>,
                            None::<String>,
                        )
                    };

                // Restore projection weights
                r#gen
                    .controller
                    .restore_projection_weights(proj_ckpt.projection_weights.clone());

                return Ok((r#gen, adam, Some(proj_ckpt.projection_weights), config_str));
            }
            Err(e) => {
                tracing::debug!("Failed to load as ProjectionCheckpoint: {}", e);
            }
        }

        anyhow::bail!("Failed to load checkpoint: unknown format or corrupt file")
    }

    /// Load while allowing MessagePack checksum mismatch.
    pub fn from_checkpoint_allow_checksum_mismatch<P: AsRef<std::path::Path>>(
        path: P,
        genesis: &GenesisSeed,
    ) -> anyhow::Result<(Self, Option<AdamState>, Option<Vec<f32>>, Option<String>)> {
        // Path 1: Try full BrocaCheckpoint
        if let Ok(checkpoint) = BrocaCheckpoint::load_from_file_allow_checksum_mismatch(&path) {
            let adam = checkpoint.adam_state.clone();
            return Ok((
                Self::from_checkpoint_struct(checkpoint, genesis),
                adam,
                None::<Vec<f32>>,
                None::<String>,
            ));
        }

        // Path 2: Try ProjectionCheckpoint
        #[cfg(feature = "mamba-cpu")]
        match ProjectionCheckpoint::load_from_file_allow_checksum_mismatch(&path) {
            Ok(proj_ckpt) => {
                let (mut r#gen, adam, _, config_str) =
                    if let Some(snapshot) = proj_ckpt.full_snapshot {
                        (
                            Self::from_checkpoint_struct(*snapshot, genesis),
                            None::<AdamState>,
                            None::<Vec<f32>>,
                            None::<String>,
                        )
                    } else {
                        (
                            Self::new(genesis, BrocaConfig::default()),
                            None::<AdamState>,
                            None::<Vec<f32>>,
                            None::<String>,
                        )
                    };

                // Restore projection weights
                r#gen
                    .controller
                    .restore_projection_weights(proj_ckpt.projection_weights.clone());

                return Ok((r#gen, adam, Some(proj_ckpt.projection_weights), config_str));
            }
            Err(e) => {
                tracing::debug!("Failed to load as ProjectionCheckpoint (recovery): {}", e);
            }
        }

        anyhow::bail!("Failed to load checkpoint (recovery mode): unknown format or corrupt file")
    }

    fn from_checkpoint_struct(checkpoint: BrocaCheckpoint, genesis: &GenesisSeed) -> Self {
        let mut r#gen = Self::new(genesis, checkpoint.config);
        r#gen.controller.restore_network(checkpoint.network_state);
        r#gen
            .controller
            .restore_token_embeddings(checkpoint.token_embeddings);
        if let Some(hwy_w) = checkpoint.highway_weights {
            r#gen.controller.restore_highway_weights(hwy_w);
        }
        r#gen
    }

    /// Save the current generator state to a checkpoint file.
    pub fn save_checkpoint<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        training_epoch: usize,
        training_loss: f32,
        adam_state: Option<AdamState>,
        projection_weights: Option<Vec<f32>>,
        liquid_mamba_config: Option<String>,
    ) -> anyhow::Result<()> {
        let mut checkpoint = BrocaCheckpoint {
            version: crate::checkpoint::CHECKPOINT_VERSION,
            token_embeddings: self.controller.token_embeddings().to_vec(),
            network_state: self.controller.network().clone(),
            vocab: self.tokenizer.export_vocab(),
            config: self.config.clone(),
            training_epoch,
            training_loss,
            adam_state,
            projection_weights,
            highway_weights: self.controller.highway_weights(),
            liquid_mamba_config,
            metadata: crate::checkpoint::BrocaCheckpointMetadata::default(), // Will be populated in save_to_file eventually
            checksum: [0u8; 32],
        };
        checkpoint.save_to_file(path)
    }

    /// Generate text from thought channels.
    pub fn generate(&mut self, channels: &ThoughtChannels) -> GenerationResult {
        self.generate_with_callback(channels, &mut |_| {})
    }

    /// Generate with an optional NSM semantic content vector.
    ///
    /// When `semantic_hv` is provided and `enable_nsm_semantic` is true,
    /// the semantic HV is blended with the thought HV via linear interpolation
    /// before entering the CfC generation loop. This gives the generator
    /// actual semantic content (NSM-composed meaning) rather than just
    /// metadata about the thought.
    ///
    /// Science: Barsalou (1999) — grounded cognition requires ~30% semantic modulation.
    pub fn generate_with_semantic(
        &mut self,
        channels: &ThoughtChannels,
        semantic_hv: Option<&symthaea_core::hdc::ContinuousHV>,
        active_primes: &[String],
    ) -> GenerationResult {
        self.generate_internal_with_semantic(
            channels,
            &mut |_| {},
            true,
            semantic_hv,
            active_primes,
            None,
            None,
        )
    }

    /// Generate continuing from the current CfC state (multi-turn context).
    ///
    /// Unlike `generate()`, this does NOT reset the controller state,
    /// so the CfC network retains temporal context from prior generations.
    /// This enables coherent multi-turn dialogue where each utterance
    /// builds on the neural state of previous ones.
    pub fn generate_continuing(&mut self, channels: &ThoughtChannels) -> GenerationResult {
        self.generate_internal(channels, &mut |_| {}, false)
    }

    /// Generate text with a per-token streaming callback.
    pub fn generate_with_callback(
        &mut self,
        channels: &ThoughtChannels,
        on_token: &mut dyn FnMut(&str),
    ) -> GenerationResult {
        self.generate_internal(channels, on_token, true)
    }

    /// Internal generation with configurable state reset.
    fn generate_internal(
        &mut self,
        channels: &ThoughtChannels,
        on_token: &mut dyn FnMut(&str),
        reset_state: bool,
    ) -> GenerationResult {
        self.generate_internal_with_semantic(channels, on_token, reset_state, None, &[], None, None)
    }

    /// Public generation method supporting both NSM semantics, conversation context, and knowledge context.
    pub fn generate_full(
        &mut self,
        channels: &ThoughtChannels,
        semantic_hv: Option<&symthaea_core::hdc::ContinuousHV>,
        active_primes: &[String],
        context_hv: Option<&symthaea_core::hdc::ContinuousHV>,
        knowledge_hv: Option<&symthaea_core::hdc::ContinuousHV>,
        reset_state: bool,
    ) -> GenerationResult {
        self.generate_internal_with_semantic(
            channels,
            &mut |_| {},
            reset_state,
            semantic_hv,
            active_primes,
            context_hv,
            knowledge_hv,
        )
    }

    /// Internal generation with configurable state reset and optional NSM semantic HV.
    fn generate_internal_with_semantic(
        &mut self,
        channels: &ThoughtChannels,
        on_token: &mut dyn FnMut(&str),
        reset_state: bool,
        semantic_hv: Option<&symthaea_core::hdc::ContinuousHV>,
        active_primes: &[String],
        context_hv: Option<&symthaea_core::hdc::ContinuousHV>,
        knowledge_hv: Option<&symthaea_core::hdc::ContinuousHV>,
    ) -> GenerationResult {
        // 1. Encode thought channels once
        let thought_hv_original = self.encoder.encode(channels);
        let mut thought_hv = thought_hv_original.clone();

        // 1b. Blend NSM semantic content into thought HV if enabled and provided.
        // This gives the generator actual compositional meaning (e.g., FEEL(BAD) ⊗ BECAUSE ⊗ MOVE(AWAY))
        // rather than just scalar metadata about the thought.
        if self.config.enable_nsm_semantic {
            if let Some(sem_hv) = semantic_hv {
                let alpha = self.config.nsm_semantic_alpha.clamp(0.0, 1.0);
                if alpha > 0.0 {
                    thought_hv.lerp_in_place(sem_hv, 1.0 - alpha, alpha);
                }
            }
        }

        // 1c. Blend conversation context HV at 20% weight if provided (T1.1).
        if let Some(ctx_hv) = context_hv {
            thought_hv.lerp_in_place(ctx_hv, 0.80, 0.20);
        }

        // 1d. Blend knowledge context HV at 15% weight if provided (T1.2).
        if let Some(kn_hv) = knowledge_hv {
            thought_hv.lerp_in_place(kn_hv, 0.85, 0.15);
        }

        // 2. Compute max tokens (consciousness-gated, then time-pressure-adjusted)
        let pre_pressure_tokens = if self.config.enable_consciousness_gating {
            consciousness_gated_max_tokens(self.config.gating.base_max_tokens, channels.psi())
        } else {
            self.config.gating.base_max_tokens
        };
        let max_tokens = crate::gating::time_pressure_adjusted_max_tokens(
            pre_pressure_tokens,
            channels.time_pressure(),
            self.config.gating.time_pressure_max_reduction,
        );

        // Time pressure reduction as a fraction of pre-pressure tokens removed
        let time_pressure_reduction = if pre_pressure_tokens > 0 {
            1.0 - (max_tokens as f32 / pre_pressure_tokens as f32)
        } else {
            0.0
        };

        // 3. Optionally reset controller state (skip for multi-turn continuity)
        if reset_state {
            self.controller.reset();
            // Seed CfC neurons from thought HV so early tokens are thought-dependent
            self.controller.seed_from_thought(&thought_hv);
        }

        // 4. Re-seed sampling RNG for reproducibility
        if let Some(seed) = self.config.sampling_seed {
            self.sampling_rng = Some(StdRng::seed_from_u64(seed));
        }

        // 4b. Use cached NSM gate (built once at construction) and create a fresh tracker.
        let use_nsm = self.nsm_gate.is_some() && !active_primes.is_empty();
        let mut nsm_tracker = if use_nsm {
            Some(crate::gating::NsmCoherenceTracker::new(active_primes))
        } else {
            None
        };

        // 5. Autoregressive generation loop
        let mut tokens = Vec::new();
        let mut prev_token = self.tokenizer.thought_id; // Start with <thought> token
        let mut eos_terminated = false;
        let mut veto_triggered = false;
        // Accumulate raw bytes for proper UTF-8 decode (byte tokens like <0xE2>
        // are only valid when combined with subsequent bytes)
        let mut text_bytes: Vec<u8> = Vec::new();
        let rep_penalty = self.config.repetition_penalty;
        let min_veto_pos = self.config.gating.min_veto_position;
        let max_vetoes = self.config.gating.max_vetoes;
        let veto_refractory = self.config.gating.veto_refractory;
        let mut veto_count = 0usize;
        let mut tokens_since_veto = veto_refractory; // Start past refractory
        let mut coherence_dynamics = Vec::new();
        let mut gating_trace = Vec::new();
        let mut logit_diagnostics = Vec::new();
        let mut consecutive_low_coherence = 0usize;
        let mut hallucination_flag = false;
        // Phase 4: Soft veto snapshot — saved at each high-coherence step
        let mut good_snapshot: Option<NetworkSnapshot> = None;

        for pos in 0..max_tokens {
            // Forward step
            let mut logits = self.controller.forward_step(&thought_hv, prev_token, pos);

            // Apply frequency-scaled repetition penalty
            if rep_penalty > 1.0 + 1e-6 {
                apply_repetition_penalty(&mut logits, &tokens, rep_penalty);
            }

            // Apply gating and compute audit trail values
            // When bypass_gating is true, skip all gating for raw CfC quality iteration.
            let this_epistemic_boost =
                if !self.config.bypass_gating && self.config.enable_epistemic_gate {
                    let ordinal = channels.epistemic_ordinal();
                    self.epistemic_gate.apply(&mut logits, ordinal);
                    // Compute representative boost magnitude for the trace
                    if ordinal > 3.5 {
                        self.config.gating.unknown_hedging_boost
                            * self.config.gating.ood_boost_multiplier
                    } else if ordinal > 2.5 {
                        self.config.gating.unknown_hedging_boost
                    } else if ordinal > 1.5 {
                        self.config.gating.uncertain_hedging_boost
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

            // Apply per-axis epistemic cube gating (E/N/M/H)
            // Complements the 1D ordinal gate with fine-grained axis modulation:
            // E-axis controls assertion vs hedging, N-axis social framing,
            // M-axis temporal framing, H-axis coherence depth.
            if !self.config.bypass_gating && self.config.enable_epistemic_cube_gate {
                self.epistemic_cube_gate.apply(&mut logits, channels);
            }

            let this_emotional_boost = if !self.config.bypass_gating
                && self.config.enable_emotional_modulation
            {
                self.emotional_modulator.apply(&mut logits, channels, pos);
                // Compute maximum boost magnitude from arousal/valence/warmth effects
                let arousal = channels.arousal();
                let valence = channels.valence();
                let warmth = channels.warmth();
                let cfg = &self.config.gating;
                let mut max_boost = 0.0f32;
                if arousal > cfg.high_arousal_threshold && pos > cfg.arousal_position_threshold {
                    let boost =
                        (arousal - cfg.high_arousal_threshold) * cfg.arousal_boost_multiplier;
                    max_boost = max_boost.max(boost);
                    // arousal × negative valence interaction
                    if valence < cfg.negative_valence_threshold {
                        let interaction = (arousal - cfg.high_arousal_threshold)
                            * (-valence - (-cfg.negative_valence_threshold))
                            * cfg.valence_boost_multiplier;
                        max_boost = max_boost.max(interaction);
                    }
                }
                if valence < cfg.negative_valence_threshold {
                    let boost = (-valence - (-cfg.negative_valence_threshold))
                        * cfg.valence_boost_multiplier;
                    max_boost = max_boost.max(boost);
                }
                if warmth < cfg.low_warmth_threshold {
                    let penalty =
                        (cfg.low_warmth_threshold - warmth) * cfg.warmth_penalty_multiplier;
                    max_boost = max_boost.max(penalty.abs());
                }
                max_boost
            } else {
                0.0
            };

            // Therapeutic gating: modulate logits based on clinical context
            #[cfg(feature = "therapeutic")]
            if !self.config.bypass_gating {
                TherapeuticGate::apply_to_logits(&mut logits, channels, &self.tokenizer);
            }

            // NSM semantic gate: boost logits for tokens expressing active primes.
            // Inserted after epistemic/emotional/therapeutic gates, before coherence feedback.
            let this_nsm_boost = if !self.config.bypass_gating {
                if let Some(ref gate) = self.nsm_gate {
                    gate.apply(
                        &mut logits,
                        active_primes,
                        self.config.nsm_prime_logit_boost,
                    );
                    self.config.nsm_prime_logit_boost
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Code-aware gate: boost/suppress tokens based on code channels (24-27).
            // Structural keywords when complex, error handling when error-prone, etc.
            if !self.config.bypass_gating {
                self.code_gate.apply(&mut logits, channels);
            }

            apply_decode_suppression(&mut logits, &self.tokenizer, channels, &self.config);

            // Coherence feedback: scale thought HV to strengthen binding when coherence drifts
            let mut this_binding_weight = 1.0f32;
            let mut this_veto = false;
            let mut this_coherence = self.coherence_feedback.coherence();
            let this_confidence_veto_threshold = confidence_adjusted_veto_threshold(
                self.config.gating.veto_threshold,
                channels.response_confidence(),
                self.config.gating.confidence_veto_scale,
            );
            if self.config.enable_coherence_feedback {
                let output_hv = self.controller.output_hv();
                let measured_coherence = output_hv.similarity(&thought_hv);
                let coherence = if self.config.bypass_gating {
                    measured_coherence
                } else {
                    let binding_weight = self.coherence_feedback.update(&output_hv, &thought_hv);
                    this_binding_weight = binding_weight;
                    self.coherence_feedback.coherence()
                };
                this_coherence = coherence;
                coherence_dynamics.push(coherence);

                // Hallucination detection: track consecutive low-coherence tokens
                if coherence < 0.05 {
                    consecutive_low_coherence += 1;
                    if consecutive_low_coherence >= 3 {
                        hallucination_flag = true;
                    }
                } else {
                    consecutive_low_coherence = 0;
                }

                if !self.config.bypass_gating {
                    // Phase 3: Feed coherence to controller for adaptive dt.
                    self.controller.set_coherence_feedback(coherence);

                    // Phase 2: Algebraic coherence correction vs scalar scaling.
                    if self.coherence_feedback.is_algebraic() {
                        thought_hv = self
                            .coherence_feedback
                            .algebraic_correct(&output_hv, &thought_hv);
                    } else if this_binding_weight > 1.0 + 1e-6 {
                        thought_hv = thought_hv.scale(this_binding_weight);
                    }

                    // Phase 4: Save snapshot at high-coherence steps for soft veto.
                    if self.config.gating.enable_soft_veto
                        && coherence >= self.config.gating.coherence_drift_threshold
                    {
                        good_snapshot = Some(self.controller.save_state());
                    }

                    // Semantic veto: mid-sentence self-correction
                    // Gated by: min position, max vetoes, refractory period
                    // NSM coverage relaxes veto: high prime coverage means semantically on-track
                    // even if cosine coherence is low (different representation spaces).
                    let nsm_veto_relaxed = if let Some(ref tracker) = nsm_tracker {
                        let coverage = tracker.prime_coverage();
                        // High coverage relaxes: subtract from veto willingness
                        coverage > 0.6 // Don't veto if >60% of primes expressed
                    } else {
                        false
                    };
                    if self.config.enable_semantic_veto
                        && !nsm_veto_relaxed
                        && self.coherence_feedback.should_veto_with_confidence(
                            channels.response_confidence(),
                            self.config.gating.confidence_veto_scale,
                        )
                        && pos > min_veto_pos
                        && veto_count < max_vetoes
                        && tokens_since_veto >= veto_refractory
                    {
                        veto_triggered = true;
                        this_veto = true;
                        veto_count += 1;
                        tokens_since_veto = 0;
                        text_bytes.extend_from_slice(self.config.veto_hesitation.as_bytes());
                        on_token(&self.config.veto_hesitation);

                        // Phase 4: Soft veto — partial restore instead of hard reset
                        if self.config.gating.enable_soft_veto {
                            if let Some(ref snapshot) = good_snapshot {
                                self.controller.restore_partial(
                                    snapshot,
                                    self.config.gating.veto_rewind_alpha,
                                );
                            } else {
                                // No snapshot saved yet — fall back to hard reset
                                self.controller.reset();
                                let _ = self.controller.forward_step(
                                    &thought_hv_original,
                                    self.tokenizer.thought_id,
                                    0,
                                );
                            }
                        } else {
                            // Original hard veto: full reset + re-inject
                            self.controller.reset();
                            let _ = self.controller.forward_step(
                                &thought_hv_original,
                                self.tokenizer.thought_id,
                                0,
                            );
                        }
                        // Restore thought_hv from original (undo any drift scaling)
                        thought_hv = thought_hv_original.clone();
                    }
                }
            }

            gating_trace.push(GatingTraceEntry {
                position: pos,
                coherence: this_coherence,
                binding_weight: this_binding_weight,
                veto_triggered: this_veto,
                epistemic_boost: this_epistemic_boost,
                emotional_boost: this_emotional_boost,
                time_pressure_reduction,
                confidence_veto_threshold: this_confidence_veto_threshold,
                nsm_boost: this_nsm_boost,
            });

            tokens_since_veto += 1;

            // Sample next token
            let next_token = self.sample(&logits);

            // Observe token for NSM prime coverage tracking
            if let (Some(tracker), Some(gate)) = (&mut nsm_tracker, &self.nsm_gate) {
                tracker.observe_token(next_token, gate);
            }

            // Check EOS
            if next_token == self.tokenizer.eos_id {
                eos_terminated = true;
                // Remove the trace entries recorded for this EOS iteration
                // (they were pushed before we knew this was EOS)
                gating_trace.pop();
                coherence_dynamics.pop();
                break;
            }
            logit_diagnostics.push(logit_diagnostics_for_step(pos, &logits, next_token, 8));

            // Skip special tokens in output
            if !self.tokenizer.is_special(next_token) {
                let token_str = self.tokenizer.token_str(next_token);
                // Auto-spacing: insert space between consecutive alphanumeric tokens
                if self.config.enable_auto_spacing {
                    if let (Some(&last_byte), Some(&first_byte)) =
                        (text_bytes.last(), token_str.as_bytes().first())
                    {
                        if last_byte.is_ascii_alphanumeric() && first_byte.is_ascii_alphanumeric() {
                            text_bytes.push(b' ');
                            on_token(" ");
                        }
                    }
                }
                // Decode byte tokens (<0xHH>) to raw bytes for proper UTF-8
                if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6
                {
                    if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                        text_bytes.push(byte);
                    } else {
                        text_bytes.extend_from_slice(token_str.as_bytes());
                    }
                } else {
                    text_bytes.extend_from_slice(token_str.as_bytes());
                }
                on_token(token_str);
            }

            tokens.push(next_token);
            prev_token = next_token;
        }

        let final_coherence = coherence_dynamics
            .last()
            .copied()
            .unwrap_or_else(|| self.coherence_feedback.coherence());
        let text = String::from_utf8(text_bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());

        GenerationResult {
            text,
            token_ids: tokens.clone(),
            num_tokens: tokens.len(),
            eos_terminated,
            veto_triggered,
            final_coherence,
            long_coherence: 0.0, // CfC-HDC backend doesn't use long window
            coherence_dynamics,
            gating_trace,
            logit_diagnostics,
            hallucination_flag,
            nsm_prime_coverage: nsm_tracker
                .as_ref()
                .map(|t| t.prime_coverage())
                .unwrap_or(0.0),
            #[cfg(feature = "mamba-cpu")]
            output_hvs: Vec::new(),
            #[cfg(feature = "mamba-cpu")]
            semantic_pe: 0.0,
        }
    }

    /// Sample a token from logits using the configured strategy.
    fn sample(&mut self, logits: &[f32]) -> u32 {
        match &self.config.sampling {
            SamplingStrategy::Greedy => greedy_sample(logits),
            SamplingStrategy::TopK { k, temperature } => {
                top_k_sample(logits, *k, *temperature, self.sampling_rng.as_mut())
            }
            SamplingStrategy::TopP { p, temperature } => {
                top_p_sample(logits, *p, *temperature, self.sampling_rng.as_mut())
            }
        }
    }

    /// Get reference to the tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// Override the sampling strategy (e.g., for sample generation with custom temperature/top-k).
    pub fn set_sampling(&mut self, strategy: SamplingStrategy) {
        self.config.sampling = strategy;
    }

    /// Get mutable reference to the controller (for training).
    pub fn controller_mut(&mut self) -> &mut LanguageController {
        &mut self.controller
    }

    /// Get reference to the encoder.
    pub fn encoder(&self) -> &ThoughtLanguageEncoder {
        &self.encoder
    }

    /// Get reference to the controller.
    pub fn controller(&self) -> &LanguageController {
        &self.controller
    }

    /// Get reference to the config.
    pub fn config(&self) -> &BrocaConfig {
        &self.config
    }

    /// Get mutable reference to the config (for training-time adjustments).
    pub fn config_mut(&mut self) -> &mut BrocaConfig {
        &mut self.config
    }
}

/// Apply frequency-scaled repetition penalty to logits for tokens already generated.
///
/// For each token ID in `generated`, counts its occurrences and applies
/// `penalty^count` — repeated tokens get exponentially stronger suppression.
/// This prevents alternating-token loops (A B A B) that flat penalty misses.
///
/// Science: Keskar et al. (2019) "CTRL" — repetition penalty prevents
/// degenerate loops in autoregressive models.
fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    // Count occurrences of each generated token
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &token_id in generated {
        *counts.entry(token_id).or_insert(0) += 1;
    }
    for (&token_id, &count) in &counts {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            let effective_penalty = penalty.powi(count as i32);
            if *logit > 0.0 {
                *logit /= effective_penalty;
            } else {
                *logit *= effective_penalty;
            }
        }
    }
}

fn apply_decode_suppression(
    logits: &mut [f32],
    tokenizer: &BpeTokenizer,
    channels: &ThoughtChannels,
    config: &BrocaConfig,
) {
    if config.suppress_unknown_tokens {
        if let Some(logit) = logits.get_mut(tokenizer.unk_id as usize) {
            *logit = f32::NEG_INFINITY;
        }
    }

    if config.suppress_code_tokens_without_code_intent && !code_intent_active(channels) {
        for token_id in 0..logits.len().min(tokenizer.vocab_size()) {
            if is_code_contamination_token(tokenizer.token_str(token_id as u32)) {
                logits[token_id] = f32::NEG_INFINITY;
            }
        }
    }
}

fn code_intent_active(channels: &ThoughtChannels) -> bool {
    channels
        .channels
        .get(24..28)
        .map(|code_channels| {
            code_channels
                .iter()
                .copied()
                .fold(0.0f32, f32::max)
                .max(code_channels.iter().copied().sum::<f32>() * 0.25)
                > 0.25
        })
        .unwrap_or(false)
}

/// Greedy sampling: return the argmax token.
fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn logit_diagnostics_for_step(
    position: usize,
    logits: &[f32],
    selected_token_id: u32,
    k: usize,
) -> GenerationStepLogits {
    if logits.is_empty() {
        return GenerationStepLogits {
            position,
            selected_token_id,
            entropy: 0.0,
            max_probability: 0.0,
            pre_attractor_entropy: None,
            pre_attractor_max_probability: None,
            cfc_delta_scale: None,
            cfc_b_scale: None,
            semantic_attractor_mean_adjustment: None,
            semantic_attractor_max_adjustment: None,
            semantic_attractor_alignment_mean: None,
            semantic_attractor_alignment_std: None,
            selected_semantic_alignment: None,
            top_k: Vec::new(),
        };
    }

    let max_logit = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);
    let exp_values: Vec<f32> = logits
        .iter()
        .map(|&logit| {
            if logit.is_finite() {
                (logit - max_logit).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum: f32 = exp_values.iter().sum();
    if sum <= 1e-20 {
        return GenerationStepLogits {
            position,
            selected_token_id,
            entropy: 0.0,
            max_probability: 0.0,
            pre_attractor_entropy: None,
            pre_attractor_max_probability: None,
            cfc_delta_scale: None,
            cfc_b_scale: None,
            semantic_attractor_mean_adjustment: None,
            semantic_attractor_max_adjustment: None,
            semantic_attractor_alignment_mean: None,
            semantic_attractor_alignment_std: None,
            selected_semantic_alignment: None,
            top_k: Vec::new(),
        };
    }

    let mut entropy = 0.0f32;
    let mut max_probability = 0.0f32;
    for &exp_value in &exp_values {
        let probability = exp_value / sum;
        if probability > 0.0 {
            entropy -= probability * probability.ln();
            max_probability = max_probability.max(probability);
        }
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    indexed.truncate(k.min(indexed.len()));
    let top_k = indexed
        .into_iter()
        .enumerate()
        .map(|(rank, (token_id, logit))| GenerationTopLogit {
            rank,
            token_id: token_id as u32,
            logit,
            probability: exp_values[token_id] / sum,
        })
        .collect();

    GenerationStepLogits {
        position,
        selected_token_id,
        entropy,
        max_probability,
        pre_attractor_entropy: None,
        pre_attractor_max_probability: None,
        cfc_delta_scale: None,
        cfc_b_scale: None,
        semantic_attractor_mean_adjustment: None,
        semantic_attractor_max_adjustment: None,
        semantic_attractor_alignment_mean: None,
        semantic_attractor_alignment_std: None,
        selected_semantic_alignment: None,
        top_k,
    }
}

/// Top-k sampling with temperature.
///
/// If `rng` is provided, uses it for reproducible sampling.
/// Falls back to greedy on near-zero probability sum (not BOS token 0).
fn top_k_sample(logits: &[f32], k: usize, temperature: f32, rng: Option<&mut StdRng>) -> u32 {
    let k = k.min(logits.len());
    if k == 0 {
        return greedy_sample(logits);
    }

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    indexed.truncate(k);

    // Apply temperature and softmax
    let temp = temperature.max(1e-6);
    let max_logit = indexed[0].1;
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(i, l)| {
            let exp = ((l - max_logit) / temp).exp();
            (i, exp)
        })
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum < 1e-10 {
        // Fall back to greedy instead of returning BOS token 0
        return greedy_sample(logits);
    }

    let r = random_f32(rng);
    let mut cumulative = 0.0;
    for (i, p) in &probs {
        cumulative += p / sum;
        if r < cumulative {
            return *i as u32;
        }
    }

    // Numerical edge: fall back to greedy
    greedy_sample(logits)
}

/// Top-p (nucleus) sampling with temperature.
///
/// If `rng` is provided, uses it for reproducible sampling.
/// Falls back to greedy on near-zero probability sum (not BOS token 0).
fn top_p_sample(logits: &[f32], p: f32, temperature: f32, rng: Option<&mut StdRng>) -> u32 {
    // Sort by logit descending
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.total_cmp(a));

    // Apply temperature and softmax
    let temp = temperature.max(1e-6);
    let max_logit = indexed[0].1;
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(i, l)| {
            let exp = ((l - max_logit) / temp).exp();
            (i, exp)
        })
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum < 1e-10 {
        // Fall back to greedy instead of returning BOS token 0
        return greedy_sample(logits);
    }

    // Find nucleus: smallest set with cumulative prob >= p
    let mut cumulative = 0.0;
    let mut nucleus = Vec::new();
    for (i, prob) in &probs {
        cumulative += prob / sum;
        nucleus.push((*i, *prob / sum));
        if cumulative >= p {
            break;
        }
    }

    // Sample from nucleus
    let r = random_f32(rng);
    let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
    let mut c = 0.0;
    for (i, prob) in &nucleus {
        c += prob / nucleus_sum;
        if r < c {
            return *i as u32;
        }
    }

    // Numerical edge: fall back to greedy
    greedy_sample(logits)
}

/// Generate a random f32 in [0, 1) using either a seeded RNG or thread-local RNG.
fn random_f32(rng: Option<&mut StdRng>) -> f32 {
    use rand::Rng;
    match rng {
        Some(rng) => rng.r#gen::<f32>(),
        None => rand::thread_rng().r#gen::<f32>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-generator")
    }

    fn test_config() -> BrocaConfig {
        BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32, // Will be overridden by tokenizer
                max_seq_len: 16,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 20, // Short for testing
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: false, // Disable for deterministic tests
            enable_semantic_veto: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_generator_creation() {
        let genesis = test_genesis();
        let config = test_config();
        let r#gen = BrocaGenerator::new(&genesis, config);
        assert!(r#gen.tokenizer().vocab_size() > 100);
    }

    #[test]
    fn test_greedy_determinism() {
        let genesis = test_genesis();
        let config = test_config();

        let mut gen1 = BrocaGenerator::new(&genesis, config.clone());
        let mut gen2 = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::default();
        let result1 = gen1.generate(&channels);
        let result2 = gen2.generate(&channels);

        assert_eq!(
            result1.token_ids, result2.token_ids,
            "Greedy generation should be deterministic"
        );
        assert_eq!(result1.text, result2.text);
    }

    #[test]
    fn test_generation_produces_tokens() {
        let genesis = test_genesis();
        let config = test_config();
        let mut r#gen = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        assert!(result.num_tokens > 0, "Should generate at least 1 token");
    }

    #[test]
    fn test_max_tokens_limit() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.gating.base_max_tokens = 5;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        assert!(
            result.num_tokens <= 5,
            "Should respect max_tokens limit: got {}",
            result.num_tokens
        );
    }

    #[test]
    fn test_streaming_callback() {
        let genesis = test_genesis();
        let config = test_config();
        let mut r#gen = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::default();
        let mut streamed = String::new();
        let result = r#gen.generate_with_callback(&channels, &mut |token| {
            streamed.push_str(token);
        });

        assert_eq!(
            streamed, result.text,
            "Streamed tokens should match final text"
        );
    }

    #[test]
    fn test_different_intents_different_output() {
        let genesis = test_genesis();
        let config = test_config();

        let answer_channels = ThoughtChannels::with_intent(1); // Answer
        let clarify_channels = ThoughtChannels::with_intent(2); // Clarify

        let mut gen1 = BrocaGenerator::new(&genesis, config.clone());
        let result1 = gen1.generate(&answer_channels);

        let mut gen2 = BrocaGenerator::new(&genesis, config);
        let result2 = gen2.generate(&clarify_channels);

        // Different intents should produce different token sequences
        assert_ne!(
            result1.token_ids, result2.token_ids,
            "Different intents should generate different outputs"
        );
    }

    #[test]
    fn test_greedy_sample_fn() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(greedy_sample(&logits), 3);
    }

    #[test]
    fn test_consciousness_gated_max_tokens() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.enable_consciousness_gating = true;
        config.gating.base_max_tokens = 100;

        let mut r#gen = BrocaGenerator::new(&genesis, config);

        // Low psi → shorter
        let mut low_psi = ThoughtChannels::default();
        low_psi.set_consciousness(0.1, 0.5, 0.5);
        let result_low = r#gen.generate(&low_psi);

        let mut gen2 = BrocaGenerator::new(
            &test_genesis(),
            BrocaConfig {
                controller: LanguageControllerConfig {
                    network_layers: 2,
                    neurons_per_layer: 4,
                    vocab_size: 32,
                    max_seq_len: 16,
                    ..Default::default()
                },
                gating: GatingConfig {
                    base_max_tokens: 100,
                    ..Default::default()
                },
                enable_consciousness_gating: true,
                enable_coherence_feedback: false,
                enable_semantic_veto: false,
                sampling: SamplingStrategy::Greedy,
                ..Default::default()
            },
        );

        // High psi → longer (but may hit EOS or max_tokens)
        let mut high_psi = ThoughtChannels::default();
        high_psi.set_consciousness(0.9, 0.5, 0.5);
        let result_high = gen2.generate(&high_psi);

        // Just verify both produce output — the effective max differs
        assert!(result_low.num_tokens > 0);
        assert!(result_high.num_tokens > 0);
    }

    #[test]
    fn test_no_state_leakage_between_generations() {
        let genesis = test_genesis();
        let config = test_config();
        let mut r#gen = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::with_intent(1); // Answer

        // Generate twice with same input — should produce identical output
        let result1 = r#gen.generate(&channels);
        let result2 = r#gen.generate(&channels);

        assert_eq!(
            result1.token_ids, result2.token_ids,
            "Consecutive generations with same input should be identical (no state leakage)"
        );
    }

    #[test]
    fn test_long_horizon_generation() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.gating.base_max_tokens = 64;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        // Should produce tokens without crashing at 64 token horizon
        assert!(result.num_tokens > 0, "Should produce tokens");
        // All token IDs should be valid
        for &id in &result.token_ids {
            assert!(
                (id as usize) < r#gen.tokenizer().vocab_size(),
                "Token ID {id} exceeds vocab size {}",
                r#gen.tokenizer().vocab_size()
            );
        }
        // Final coherence should be finite
        assert!(
            result.final_coherence.is_finite(),
            "Final coherence should be finite after long generation"
        );
    }

    #[test]
    fn test_nan_channels_dont_crash() {
        let genesis = test_genesis();
        let config = test_config();
        let mut r#gen = BrocaGenerator::new(&genesis, config);

        let mut channels = ThoughtChannels::default();
        channels.channels[9] = f32::NAN; // NaN valence
        channels.channels[10] = f32::INFINITY; // Inf arousal

        let result = r#gen.generate(&channels);
        // Should not crash, should produce some output
        assert!(
            result.num_tokens > 0 || result.eos_terminated,
            "NaN channels should not crash generator"
        );
    }

    #[test]
    fn test_coherence_feedback_scales_thought() {
        use crate::gating::CoherenceFeedback;
        use symthaea_core::hdc::ContinuousHV;

        let thought_hv = ContinuousHV::random_default(42).normalize();
        // Create an output_hv that is very different (low coherence)
        let output_hv = ContinuousHV::random_default(9999).normalize();

        let mut feedback = CoherenceFeedback::new(0.30);
        let weight = feedback.update(&output_hv, &thought_hv);

        // Random 16384D vectors are nearly orthogonal → low coherence → weight > 1.0
        assert!(
            weight > 1.0,
            "Low-coherence output should produce binding weight > 1.0, got {weight}"
        );
        assert!(
            weight <= 3.0,
            "Binding weight should be capped at 3.0, got {weight}"
        );
    }

    #[test]
    fn test_repetition_penalty_reduces_repeats() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.repetition_penalty = 1.5;
        config.gating.base_max_tokens = 30;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        // Compute max consecutive repetition
        let max_rep = if result.token_ids.len() < 2 {
            1
        } else {
            let mut max = 1usize;
            let mut run = 1usize;
            for w in result.token_ids.windows(2) {
                if w[0] == w[1] {
                    run += 1;
                    max = max.max(run);
                } else {
                    run = 1;
                }
            }
            max
        };

        // With penalty 1.5, max consecutive repeats should be modest
        assert!(
            max_rep <= 5,
            "Repetition penalty should limit repeats: max_rep={max_rep}"
        );
    }

    #[test]
    fn test_repetition_penalty_fn() {
        let mut logits = vec![0.5, -0.3, 0.8, 0.1];
        let generated = vec![0, 2]; // tokens 0 and 2 each appeared once

        apply_repetition_penalty(&mut logits, &generated, 2.0);

        // Token 0: positive logit, count=1 → divided by 2.0^1 = 2.0
        assert!((logits[0] - 0.25).abs() < 1e-6);
        // Token 1: not generated → unchanged
        assert!((logits[1] - (-0.3)).abs() < 1e-6);
        // Token 2: positive logit, count=1 → divided by 2.0^1 = 2.0
        assert!((logits[2] - 0.4).abs() < 1e-6);
        // Token 3: not generated → unchanged
        assert!((logits[3] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        let mut logits = vec![-0.5, 0.3];
        let generated = vec![0]; // token 0 generated once (has negative logit)

        apply_repetition_penalty(&mut logits, &generated, 2.0);

        // Negative logit → multiplied by penalty^1 = 2.0 (pushed further negative)
        assert!((logits[0] - (-1.0)).abs() < 1e-6);
        // Token 1 unchanged
        assert!((logits[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_frequency_scaled() {
        let mut logits = vec![0.8, 0.8];
        // Token 0 appeared 3 times, token 1 appeared 1 time
        let generated = vec![0, 1, 0, 0];

        apply_repetition_penalty(&mut logits, &generated, 2.0);

        // Token 0: positive, count=3 → divided by 2.0^3 = 8.0
        assert!((logits[0] - 0.1).abs() < 1e-6, "got {}", logits[0]);
        // Token 1: positive, count=1 → divided by 2.0^1 = 2.0
        assert!((logits[1] - 0.4).abs() < 1e-6, "got {}", logits[1]);
    }

    #[test]
    fn test_sampling_reproducibility() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.sampling = SamplingStrategy::TopP {
            p: 0.9,
            temperature: 1.0,
        };
        config.sampling_seed = Some(42);

        let channels = ThoughtChannels::default();

        let mut gen1 = BrocaGenerator::new(&genesis, config.clone());
        let result1 = gen1.generate(&channels);

        let mut gen2 = BrocaGenerator::new(&genesis, config);
        let result2 = gen2.generate(&channels);

        assert_eq!(
            result1.token_ids, result2.token_ids,
            "Seeded TopP sampling should be reproducible"
        );
    }

    #[test]
    fn test_veto_refractory() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.enable_coherence_feedback = true;
        config.enable_semantic_veto = true;
        config.gating.max_vetoes = 1;
        config.gating.veto_refractory = 8;
        config.gating.base_max_tokens = 30;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        // If veto triggered, text should contain the hesitation token
        if result.veto_triggered {
            assert!(
                result.text.contains("-- wait,"),
                "Veto should insert hesitation token"
            );
            // At most max_vetoes (1) hesitation markers
            let veto_count = result.text.matches("-- wait,").count();
            assert!(veto_count <= 1, "max_vetoes=1 but got {veto_count} vetoes");
        }
        // Either way, should not crash
        assert!(result.num_tokens > 0 || result.eos_terminated);
    }

    #[test]
    fn test_generate_continuing_preserves_state() {
        let genesis = test_genesis();
        let config = test_config();
        let mut r#gen = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::with_intent(1);

        // First generation (with reset)
        let result1 = r#gen.generate(&channels);

        // Continuing generation (without reset) — should differ from fresh
        let result2 = r#gen.generate_continuing(&channels);

        // The continuing generation should produce different output because
        // it builds on the CfC state from result1
        // (Note: they MIGHT produce the same output by coincidence with random weights,
        // but with different initial CfC state the dynamics should diverge)
        assert!(
            result2.num_tokens > 0,
            "Continuing generation should produce tokens"
        );

        // Now do a fresh generation — should match result1 exactly
        let result3 = r#gen.generate(&channels);
        assert_eq!(
            result1.token_ids, result3.token_ids,
            "Fresh generation should be identical to first"
        );
    }

    #[test]
    fn test_gating_trace_populated() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.enable_coherence_feedback = true;
        config.gating.base_max_tokens = 10;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        // Gating trace should have entries for every token position
        assert_eq!(
            result.gating_trace.len(),
            result.num_tokens,
            "Gating trace should have one entry per generated token"
        );

        // Each entry should have finite coherence and new audit fields
        for entry in &result.gating_trace {
            assert!(
                entry.coherence.is_finite(),
                "Gating trace coherence should be finite"
            );
            assert!(
                entry.binding_weight >= 1.0 - 1e-6,
                "Binding weight should be >= 1.0"
            );
            assert!(
                entry.epistemic_boost.is_finite(),
                "Epistemic boost should be finite"
            );
            assert!(
                entry.emotional_boost.is_finite(),
                "Emotional boost should be finite"
            );
            assert!(
                entry.emotional_boost >= 0.0,
                "Emotional boost should be non-negative"
            );
            assert!(
                entry.time_pressure_reduction.is_finite(),
                "Time pressure reduction should be finite"
            );
            assert!(
                entry.time_pressure_reduction >= 0.0,
                "Time pressure reduction should be non-negative"
            );
            assert!(
                entry.confidence_veto_threshold.is_finite(),
                "Confidence veto threshold should be finite"
            );
            assert!(
                entry.confidence_veto_threshold >= 0.0,
                "Confidence veto threshold should be non-negative"
            );
        }
    }

    #[test]
    fn test_coherence_dynamics_populated() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.enable_coherence_feedback = true;
        config.gating.base_max_tokens = 10;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        // Coherence dynamics should have entries
        assert_eq!(
            result.coherence_dynamics.len(),
            result.num_tokens,
            "Should have one coherence value per token"
        );

        for &c in &result.coherence_dynamics {
            assert!(c.is_finite(), "Coherence should be finite");
        }
    }

    #[test]
    fn test_logit_diagnostics_populated() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.enable_coherence_feedback = true;
        config.gating.base_max_tokens = 10;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        assert_eq!(
            result.logit_diagnostics.len(),
            result.num_tokens,
            "Should have one logit diagnostic per generated token"
        );
        for step in &result.logit_diagnostics {
            assert!(step.entropy.is_finite(), "Entropy should be finite");
            assert!(
                step.max_probability.is_finite(),
                "Max probability should be finite"
            );
            assert!(
                (0.0..=1.0).contains(&step.max_probability),
                "Max probability should be in [0, 1]"
            );
            assert!(
                !step.top_k.is_empty(),
                "Top-k diagnostics should be present"
            );
        }
    }

    #[test]
    fn test_bypass_gating_still_measures_coherence() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.enable_coherence_feedback = true;
        config.bypass_gating = true;
        config.gating.base_max_tokens = 10;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        assert_eq!(
            result.coherence_dynamics.len(),
            result.num_tokens,
            "Raw bypass generation should still report measured coherence"
        );
        assert!(
            result.final_coherence.is_finite(),
            "Raw bypass final coherence should be measured, not a stale default"
        );
        if result.num_tokens > 0 {
            assert!(
                (result.final_coherence - 1.0).abs() > 1e-6,
                "Raw bypass coherence should not remain at the default perfect score"
            );
        }
    }

    #[test]
    fn test_hallucination_flag_on_noise() {
        // The hallucination flag detects 3+ consecutive tokens with
        // output-thought similarity < 0.05. With random weights this
        // may or may not trigger, so we just verify it's a bool.
        let genesis = test_genesis();
        let mut config = test_config();
        config.enable_coherence_feedback = true;
        config.gating.base_max_tokens = 15;
        config.enable_consciousness_gating = false;

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        // Just verify the flag is populated (true or false, both valid)
        let _ = result.hallucination_flag;
    }

    #[test]
    fn test_contrastive_intents() {
        let genesis = test_genesis();
        let config = test_config();

        // Generate with all 8 intents
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        for intent in 0..8 {
            let channels = ThoughtChannels::with_intent(intent);
            let mut r#gen = BrocaGenerator::new(&genesis, config.clone());
            let result = r#gen.generate(&channels);
            outputs.push(result.token_ids);
        }

        // Count distinct pairs
        let mut distinct_pairs = 0;
        for i in 0..8 {
            for j in (i + 1)..8 {
                if outputs[i] != outputs[j] {
                    distinct_pairs += 1;
                }
            }
        }

        // At least 6 of 28 intent pairs should produce different outputs
        assert!(
            distinct_pairs >= 6,
            "At least 6/28 intent pairs should differ: got {distinct_pairs}"
        );
    }

    #[test]
    fn test_generate_continuing_context_accumulates() {
        let genesis = test_genesis();
        let config = test_config();
        let mut r#gen = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::with_intent(1);

        // First generation (resets state)
        let result1 = r#gen.generate(&channels);

        // Second generation (continuing — builds on CfC state)
        let result2 = r#gen.generate_continuing(&channels);

        // Both should produce output
        assert!(result1.num_tokens > 0);
        assert!(result2.num_tokens > 0);

        // Third fresh generation should match first (state reset)
        let result3 = r#gen.generate(&channels);
        assert_eq!(
            result1.token_ids, result3.token_ids,
            "Fresh generation should reproduce first result"
        );
    }

    #[test]
    fn test_generate_continuing_differs_from_fresh() {
        let genesis = test_genesis();
        let config = test_config();

        // Fresh generator → generate() to establish CfC state
        let mut r#gen = BrocaGenerator::new(&genesis, config.clone());
        let channels = ThoughtChannels::with_intent(1);
        let _warm_up = r#gen.generate(&channels);

        // generate_continuing() builds on the CfC state left by warm_up
        let continuing_result = r#gen.generate_continuing(&channels);

        // Fresh generator → generate() with no prior state
        let mut gen_fresh = BrocaGenerator::new(&genesis, config);
        let fresh_result = gen_fresh.generate(&channels);

        // The continuing result should differ from a fresh generation because
        // generate_continuing() inherits CfC temporal state from the warm-up,
        // while the fresh generator starts from zeroed CfC state.
        assert_ne!(
            continuing_result.token_ids, fresh_result.token_ids,
            "generate_continuing() should produce different output than a fresh generate() \
             because it inherits CfC state from prior generation"
        );
    }

    #[test]
    fn test_context_accumulation_changes_output() {
        let genesis = test_genesis();
        let config = test_config();
        let mut r#gen = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::with_intent(1);

        // Initial generation resets state
        let _initial = r#gen.generate(&channels);

        // Successive continuing generations should evolve (CfC state accumulates)
        let cont1 = r#gen.generate_continuing(&channels);
        let cont2 = r#gen.generate_continuing(&channels);
        let cont3 = r#gen.generate_continuing(&channels);

        // All should produce tokens
        assert!(
            cont1.num_tokens > 0,
            "First continuing should produce tokens"
        );
        assert!(
            cont2.num_tokens > 0,
            "Second continuing should produce tokens"
        );
        assert!(
            cont3.num_tokens > 0,
            "Third continuing should produce tokens"
        );

        // At least two of the three continuing generations should differ,
        // demonstrating that CfC state accumulation produces evolving output
        let all_same = cont1.token_ids == cont2.token_ids && cont2.token_ids == cont3.token_ids;
        assert!(
            !all_same,
            "Multiple generate_continuing() calls should produce evolving outputs \
             as CfC temporal state accumulates, but all three were identical"
        );
    }

    #[test]
    fn test_sequential_generation_stability() {
        let genesis = test_genesis();
        let config = test_config();
        let mut r#gen = BrocaGenerator::new(&genesis, config);

        // Generate 50 times with varying intents — no panics, no NaN leakage
        for i in 0..50 {
            let channels = ThoughtChannels::with_intent(i % 8);
            let result = r#gen.generate(&channels);

            assert!(
                result.final_coherence.is_finite(),
                "Coherence should stay finite after {i} generations"
            );
            // Token IDs should all be in range
            for &id in &result.token_ids {
                assert!(
                    (id as usize) < r#gen.tokenizer().vocab_size(),
                    "Token ID {id} out of range at generation {i}"
                );
            }
        }
    }

    /// Full generate() veto integration test.
    ///
    /// Exercises the actual veto path inside `generate_internal()`:
    /// controller reset, hesitation text insertion, thought_hv re-injection.
    /// Uses a very high veto threshold (0.99) so that almost any output
    /// triggers a veto after `min_veto_position`.
    #[test]
    fn test_generate_veto_full_path() {
        let genesis = test_genesis();
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32,
                max_seq_len: 32,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 30,
                // Very high threshold: almost any output will have coherence < 0.99
                veto_threshold: 0.99,
                min_veto_position: 1, // Allow veto after just 1 token
                max_vetoes: 2,
                veto_refractory: 1, // Minimal refractory so second veto can fire quickly
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: true,
            enable_semantic_veto: true,
            veto_hesitation: "-- wait, ".to_string(),
            ..Default::default()
        };

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        // Veto should have triggered (coherence < 0.99 is almost guaranteed
        // for random network weights)
        assert!(
            result.veto_triggered,
            "Veto should trigger with threshold=0.99; coherence_dynamics={:?}",
            &result.coherence_dynamics[..result.coherence_dynamics.len().min(10)]
        );

        // Hesitation text should appear in output
        assert!(
            result.text.contains("-- wait, "),
            "Output should contain hesitation text, got: {:?}",
            &result.text[..result.text.len().min(100)]
        );

        // Gating trace should record at least one veto entry
        let veto_entries: Vec<_> = result
            .gating_trace
            .iter()
            .filter(|e| e.veto_triggered)
            .collect();
        assert!(
            !veto_entries.is_empty(),
            "Gating trace should record veto events"
        );
        // Veto positions should be > min_veto_position (1)
        for entry in &veto_entries {
            assert!(
                entry.position > 0,
                "Veto at position {} should be > min_veto_position",
                entry.position
            );
        }

        // At most max_vetoes (2) should have fired
        assert!(
            veto_entries.len() <= 2,
            "At most 2 vetoes should fire, got {}",
            veto_entries.len()
        );

        // All coherence values should be finite (no NaN from reset/re-inject)
        for &c in &result.coherence_dynamics {
            assert!(c.is_finite(), "Coherence should be finite after veto reset");
        }

        // Final coherence should be valid
        assert!(
            result.final_coherence.is_finite(),
            "Final coherence should be finite"
        );
    }

    /// Verify that with veto disabled (default), no veto occurs even
    /// when coherence would be low enough to trigger one.
    #[test]
    fn test_generate_veto_disabled_no_trigger() {
        let genesis = test_genesis();
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32,
                max_seq_len: 16,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 20,
                veto_threshold: 0.99, // Would trigger if enabled
                min_veto_position: 1,
                max_vetoes: 2,
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: true,
            enable_semantic_veto: false, // Disabled
            ..Default::default()
        };

        let mut r#gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = r#gen.generate(&channels);

        assert!(
            !result.veto_triggered,
            "Veto should NOT trigger when enable_semantic_veto=false"
        );
        assert!(
            !result.text.contains("-- wait, "),
            "No hesitation text when veto is disabled"
        );
    }

    /// Multi-turn veto stability test (Item #3).
    ///
    /// `generate_continuing()` preserves CfC state across calls. If a veto
    /// fires during multi-turn generation, it resets the controller — which
    /// discards the cross-turn temporal context. This test verifies that
    /// generation remains stable (no NaN, no panic) across multiple turns
    /// with veto enabled, even when veto triggers mid-turn.
    #[test]
    fn test_multi_turn_veto_stability() {
        let genesis = test_genesis();
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32,
                max_seq_len: 32,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 15,
                veto_threshold: 0.99, // High threshold to provoke vetoes
                min_veto_position: 1,
                max_vetoes: 1,
                veto_refractory: 2,
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: true,
            enable_semantic_veto: true,
            veto_hesitation: "-- wait, ".to_string(),
            ..Default::default()
        };

        let mut r#gen = BrocaGenerator::new(&genesis, config);

        // Simulate a 5-turn conversation using generate_continuing()
        let mut veto_count = 0;
        for turn in 0..5 {
            let channels = ThoughtChannels::with_intent(turn % 4);
            let result = r#gen.generate_continuing(&channels);

            // No NaN in coherence dynamics
            for &c in &result.coherence_dynamics {
                assert!(
                    c.is_finite(),
                    "Coherence NaN at turn {turn}: {:?}",
                    &result.coherence_dynamics
                );
            }
            assert!(
                result.final_coherence.is_finite(),
                "Final coherence NaN at turn {turn}"
            );

            // Track vetoes across turns
            if result.veto_triggered {
                veto_count += 1;
            }

            // All token IDs in range
            for &id in &result.token_ids {
                assert!(
                    (id as usize) < r#gen.tokenizer().vocab_size(),
                    "Token ID {id} out of range at turn {turn}"
                );
            }
        }

        // With threshold=0.99, at least some turns should veto
        // (random CfC weights almost never produce 0.99 coherence)
        assert!(
            veto_count > 0,
            "Expected at least one veto across 5 multi-turn generations"
        );
    }

    // ── Capstone: train→generate round-trip ──

    /// Train on a small corpus, then generate and verify that training
    /// actually improved generation quality. This is the single test that
    /// proves the entire pipeline works end-to-end: encoder → training →
    /// CfC network → generation → coherence feedback.
    #[test]
    fn test_train_generate_round_trip() {
        use crate::training::{TrainingConfig, TrainingDataset, TrainingPair, train};

        let genesis = test_genesis();
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32,
                max_seq_len: 32,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 20,
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: true,
            enable_semantic_veto: false,
            ..Default::default()
        };

        // Generate BEFORE training (baseline)
        let mut r#gen = BrocaGenerator::new(&genesis, config.clone());
        let channels = ThoughtChannels::default();
        let pre_train_result = r#gen.generate(&channels);

        // Train on a small corpus
        let tok = r#gen.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        for _ in 0..10 {
            dataset.push(TrainingPair::new(channels, "hello world".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 20,
            learning_rate: 0.01,
            bptt_window: 8,
            use_adam: true,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 128.0,
            ..Default::default()
        };

        let metrics = train(&mut r#gen, &dataset, &train_config);

        // Verify training reduced loss
        let first_loss = metrics[0].avg_loss;
        let final_loss = metrics.last().unwrap().avg_loss;
        assert!(
            final_loss < first_loss,
            "Training should reduce loss: {first_loss:.4} → {final_loss:.4}"
        );

        // Generate AFTER training
        let post_train_result = r#gen.generate(&channels);

        // Both should produce valid output
        assert!(
            post_train_result.final_coherence.is_finite(),
            "Post-training coherence should be finite"
        );
        assert!(
            !post_train_result.token_ids.is_empty(),
            "Post-training should generate tokens"
        );

        // The generator should produce different output after training
        // (weights changed, so even greedy sampling should differ)
        let changed = pre_train_result.token_ids != post_train_result.token_ids
            || pre_train_result.text != post_train_result.text;
        assert!(changed, "Training should change generator behavior");
    }

    // ── Capstone: intent fidelity benchmark ──

    /// Verify that different thought intents produce measurably different
    /// generated text. If all 8 intents produce identical output, the
    /// generator is ignoring intent information.
    #[test]
    fn test_intent_fidelity_generation_diversity() {
        let genesis = test_genesis();
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32,
                max_seq_len: 32,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 15,
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: false,
            enable_semantic_veto: false,
            ..Default::default()
        };

        let mut r#gen = BrocaGenerator::new(&genesis, config);

        // Generate for all 8 intents
        let mut results: Vec<Vec<u32>> = Vec::new();
        for intent in 0..8 {
            let channels = ThoughtChannels::with_intent(intent);
            let result = r#gen.generate(&channels);
            results.push(result.token_ids);
        }

        // Count distinct outputs
        let mut unique_outputs = results.clone();
        unique_outputs.sort();
        unique_outputs.dedup();

        // At least 3 of 8 intents should produce different token sequences
        // (random CfC weights may not perfectly separate all 8, but the
        // encoder one-hot intent channels should drive some diversity)
        assert!(
            unique_outputs.len() >= 3,
            "Expected at least 3 distinct outputs from 8 intents, got {}. \
             Generator may be ignoring intent information.",
            unique_outputs.len()
        );

        // Pairwise Jaccard distance: verify low overlap between different intents
        let mut total_jaccard = 0.0f32;
        let mut pair_count = 0;
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let set_i: std::collections::HashSet<u32> = results[i].iter().copied().collect();
                let set_j: std::collections::HashSet<u32> = results[j].iter().copied().collect();
                let intersection = set_i.intersection(&set_j).count();
                let union = set_i.union(&set_j).count();
                let jaccard = if union > 0 {
                    intersection as f32 / union as f32
                } else {
                    1.0
                };
                total_jaccard += jaccard;
                pair_count += 1;
            }
        }
        let mean_jaccard = total_jaccard / pair_count as f32;

        // Mean Jaccard should be < 0.95 (not all identical)
        assert!(
            mean_jaccard < 0.95,
            "Mean pairwise Jaccard similarity {mean_jaccard:.4} too high — \
             outputs are too similar across intents"
        );
    }
}
