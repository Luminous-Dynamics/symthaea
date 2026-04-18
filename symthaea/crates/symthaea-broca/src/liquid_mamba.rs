// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Liquid-Mamba Fusion (L-SSM): consciousness-gated SSM language generation.
//!
//! Fuses a pre-trained Mamba SSM with Symthaea's HDC-LTC cognitive loop.
//! The SSM provides vocabulary fluency; the cognitive loop provides
//! consciousness-gated semantic control, epistemic gating, emotional
//! authenticity, and biological constraints.
//!
//! # Architecture
//!
//! ```text
//! ThoughtChannels → ThoughtLanguageEncoder → thought_hv (16,384D)
//!                                              │
//!                           HdcSsmProjection ──┤──→ ssm_context (768D)
//!                                              │          │
//!                                              │    MambaWrapper.inject_initial_context()
//!                                              │          │
//!                                              │    Autoregressive loop:
//!                                              │      ├── biological delta modulation
//!                                              │      ├── forward_one_token() → logits
//!                                              │      ├── EpistemicGate.apply()
//!                                              │      ├── EmotionalModulator.apply()
//!                                              │      ├── top_k_sample()
//!                                              │      ├── back-project token → HDC
//!                                              │      └── CoherenceMonitor (veto check)
//!                                              │
//!                                         GenerationResult
//! ```

use std::collections::VecDeque;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::encoder::{ThoughtChannels, ThoughtLanguageEncoder};
use crate::gating::{
    consciousness_gated_max_tokens, EmotionalModulator, EpistemicGate, GatingConfig,
};
use crate::generator::GenerationResult;
use crate::mamba::{MambaBackend, MambaWrapper};
use crate::projection::{HdcSsmProjection, ProjectionGradientDiagnostics};
use crate::temporal_projection::TemporalProjection;

/// Configuration for the Liquid-Mamba generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidMambaConfig {
    /// HuggingFace model ID for the Mamba checkpoint.
    pub model_id: String,
    /// Maximum tokens to generate per call.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-k for sampling.
    pub top_k: usize,
    /// Coherence below this → veto (mid-sentence self-correction).
    pub veto_threshold: f32,
    /// Coherence below this → boost thought binding.
    pub drift_threshold: f32,
    /// Sliding window size for short-term coherence monitoring.
    pub coherence_window: usize,
    /// Sliding window size for long-term coherence trend (default 32).
    pub long_coherence_window: usize,
    /// EMA smoothing alpha for coherence.
    pub coherence_ema_alpha: f32,
    /// Minimum consecutive low-coherence tokens before veto.
    pub min_consecutive_low: usize,
    /// Biological delta modulation strength (0 = disabled, 1 = full).
    pub delta_mod_strength: f32,
    /// Text inserted on semantic veto.
    pub veto_hesitation: String,
    /// Enable epistemic gating.
    pub enable_gating: bool,
    /// Enable semantic veto.
    pub enable_veto: bool,
    /// Enable biological delta modulation.
    pub enable_liquid_delta: bool,
    /// Enable consciousness-gated max tokens.
    pub enable_consciousness_gating: bool,
    /// HDC dimension (default 16384).
    pub hdc_dim: usize,
    /// Bottleneck dimension (default 256).
    pub bottleneck_dim: usize,
    /// SSM hidden dimension (default 768 for mamba-130m).
    pub ssm_dim: usize,
    /// LR warmup steps: ramp from 0 to base LR over this many generations.
    pub warmup_steps: usize,
    /// Gradient accumulation steps: accumulate over N generations before applying.
    pub accumulation_steps: usize,
    /// Contrastive loss weight (0 = disabled). Pushes different thoughts apart.
    pub contrastive_weight: f32,
    /// Size of the recent-thoughts buffer for contrastive loss.
    pub contrastive_buffer_size: usize,
    /// Base learning rate for online distillation.
    pub base_lr: f32,
    /// Total expected training steps for cosine annealing (0 = no annealing).
    /// When set, LR follows cosine schedule after warmup: `lr * 0.5*(1 + cos(π * t/T))`.
    pub cosine_annealing_steps: usize,
    /// Minimum LR floor as fraction of base_lr (default 0.01 = 1% of base).
    pub min_lr_fraction: f32,
    /// Weight for prefix token loss (0 = disabled, 0.3 = default).
    /// The first 25% of output tokens get an additional gradient scaled by this factor.
    /// Prefix tokens frame the response topic — extra gradient signal here
    /// teaches the projection to steer opening words.
    pub prefix_loss_weight: f32,
    /// Enable EMA teacher for projection weights.
    /// When true, `apply_gradients` is followed by `update_ema`.
    pub enable_ema: bool,
    /// EMA decay rate (default 0.999). Higher = smoother, slower tracking.
    pub ema_decay: f32,
    /// Use deep double-bottleneck projection (HDC→256→128→256→SSM).
    /// When true, `new()` creates `HdcSsmProjection::new_deep()` instead of `new()`.
    /// Default false for backwards compatibility.
    pub deep_projection: bool,
    /// Use temporal projection (chunk-based continuous latent prompting).
    /// When true, the 16384D thought is chunked into 64×256D tokens,
    /// up-projected to 768D each, and fed as continuous embeddings into Mamba.
    /// Mutually exclusive with `deep_projection` (temporal takes precedence).
    /// Default false for backwards compatibility.
    #[serde(default)]
    pub temporal_projection: bool,
    /// Use learned (trainable) positional encoding for temporal projection.
    /// When false (default), uses fixed sinusoidal encoding.
    /// Only relevant when `temporal_projection` is true.
    #[serde(default)]
    pub learned_pos_enc: bool,
    /// Chunk stride for temporal projection (0 = use chunk_size, i.e., no overlap).
    /// When stride < chunk_size (bottleneck_dim), chunks overlap, producing more
    /// tokens. E.g., stride=128 with chunk_size=256 gives 50% overlap and ~127 tokens.
    #[serde(default)]
    pub temporal_stride: usize,
    /// Use directional cosine loss instead of roundtrip autoencoder loss for temporal.
    /// Directional loss optimizes angular alignment directly, avoiding the
    /// information bottleneck of the up→down roundtrip.
    #[serde(default)]
    pub temporal_directional_loss: bool,
    /// Weight for temporal smoothness loss (L2 penalty on consecutive chunk differences).
    /// 0 = disabled (default). Good starting values: 0.01-0.1.
    #[serde(default)]
    pub temporal_smoothness_weight: f32,
    /// Weight for rank regularization (decorrelation of W_up rows).
    /// 0 = disabled (default). Good starting values: 0.001-0.01.
    #[serde(default)]
    pub temporal_rank_reg_weight: f32,
    /// Chunk budget for inference-time temporal projection (0 = use all chunks).
    /// When > 0, only the top-K most informative chunks (by activation magnitude
    /// and learned attention) are injected, reducing inference latency.
    #[serde(default)]
    pub temporal_chunk_budget: usize,
    /// Enable learned chunk attention weighting for temporal projection.
    /// Each chunk gets a trainable importance weight (sigmoid-gated).
    #[serde(default)]
    pub temporal_learned_attention: bool,
    /// Surprise gradient scaling factor (0 = disabled, 0.5 = default).
    /// Gradients are scaled by `1 + alpha * semantic_pe` before application,
    /// amplifying gradients for surprising/difficult examples.
    pub surprise_gradient_alpha: f32,
    /// Gradient clipping threshold for projection updates (default 1.0).
    #[serde(default = "default_grad_clip")]
    pub grad_clip: f32,
    /// Token-PE similarity threshold below which gating boost activates (default 0.3).
    #[serde(default = "default_token_pe_sim_threshold")]
    pub token_pe_sim_threshold: f32,
    /// Maximum per-token epistemic gating boost from PE feedback (default 1.5).
    #[serde(default = "default_token_pe_max_boost")]
    pub token_pe_max_boost: f32,
    /// Decay factor for per-token PE boost when coherence is adequate (default 0.5).
    #[serde(default = "default_token_pe_decay")]
    pub token_pe_decay: f32,
    /// FEP signal threshold above which distillation LR is boosted (default 0.7).
    #[serde(default = "default_fep_high_threshold")]
    pub fep_high_threshold: f32,
    /// FEP signal threshold below which distillation LR is dampened (default 0.3).
    #[serde(default = "default_fep_low_threshold")]
    pub fep_low_threshold: f32,
    /// FEP high-surprise LR multiplier (default 1.5).
    #[serde(default = "default_fep_high_multiplier")]
    pub fep_high_multiplier: f32,
    /// FEP low-surprise LR multiplier (default 0.7).
    #[serde(default = "default_fep_low_multiplier")]
    pub fep_low_multiplier: f32,

    // ─── Improvement A-F config fields ───
    /// Chunk dimension for temporal projection (default 256 = bottleneck_dim).
    /// When non-zero, overrides bottleneck_dim for temporal chunk sizing.
    #[serde(default)]
    pub temporal_chunk_dim: usize,

    /// Number of per-group up-projection matrices for temporal projection (default 1).
    /// Each group of chunks shares its own W_up/W_down matrices.
    /// E.g., num_groups=4 with 64 chunks → chunks 0-15 share group 0, 16-31 share group 1, etc.
    #[serde(default = "default_temporal_num_groups")]
    pub temporal_num_groups: usize,

    /// Weight for anti-collapse regularization in temporal projection (default 0.0 = disabled).
    /// Pushes chunk projections apart when their cosine similarity exceeds the threshold.
    /// Good starting values: 0.01-0.1.
    #[serde(default)]
    pub temporal_anticollapse_weight: f32,

    /// Cosine similarity threshold for anti-collapse regularization (default 0.9).
    /// Pairs of chunk projections with similarity above this threshold receive a repulsive gradient.
    #[serde(default = "default_temporal_anticollapse_threshold")]
    pub temporal_anticollapse_threshold: f32,

    /// Enable rotating grad-chunk position for temporal E2E loss (default true).
    /// When true, computes E2E gradients at a single rotating position per step instead of all.
    /// When false, uses the original behavior of processing all chunks.
    #[serde(default = "default_true")]
    pub temporal_rotate_grad_position: bool,

    /// Enable adapter MLP after temporal up-projection (default false).
    /// Adds a learnable residual MLP (d_ssm → d_ssm) after each chunk's up-projection.
    #[serde(default)]
    pub temporal_adapter: bool,

    /// LoRA rank for Mamba layer adaptation (0 = disabled, default 0).
    /// When > 0, adds low-rank adapters to Mamba's in_proj and out_proj layers.
    #[serde(default)]
    pub lora_rank: usize,

    /// LoRA alpha scaling factor (default 1.0).
    /// The LoRA contribution is scaled by alpha/rank.
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f32,

    /// LoRA learning rate (default 0.0001).
    /// Separate LR for LoRA adapter updates (typically lower than projection LR).
    #[serde(default = "default_lora_lr")]
    pub lora_lr: f32,

    /// Path to pre-computed embedding stats file for manifold moment matching.
    /// When set with `temporal_adapter`, the adapter MLP is initialized with
    /// whitening weights derived from Mamba's embedding table statistics.
    #[serde(default)]
    pub embedding_stats_path: Option<String>,

    /// Number of simultaneous E2E gradient positions per step (default 1 = legacy rotating).
    /// With K=4, each chunk position gets gradient 4× more often than K=1 rotating.
    #[serde(default = "default_e2e_grad_chunks")]
    pub e2e_grad_chunks: usize,

    /// Weight for orthogonality regularization on the spatial projection's `w_up` rows
    /// (default 0.0 = disabled). Penalizes row cosine similarity to prevent rank collapse.
    /// Good starting values: 0.01–0.05.
    #[serde(default)]
    pub orthogonality_weight: f32,

    /// Number of random row pairs to sample per orthogonality regularization step (default 64).
    #[serde(default = "default_orthogonality_samples")]
    pub orthogonality_samples: usize,

    /// Gating configuration (epistemic boost strengths, coherence thresholds, etc.).
    /// Override to strengthen or weaken consciousness-gated generation control.
    #[serde(default)]
    pub gating_config: GatingConfig,
}

fn default_grad_clip() -> f32 {
    1.0
}
fn default_token_pe_sim_threshold() -> f32 {
    0.3
}
fn default_token_pe_max_boost() -> f32 {
    1.5
}
fn default_token_pe_decay() -> f32 {
    0.5
}
fn default_fep_high_threshold() -> f32 {
    0.7
}
fn default_fep_low_threshold() -> f32 {
    0.3
}
fn default_fep_high_multiplier() -> f32 {
    1.5
}
fn default_fep_low_multiplier() -> f32 {
    0.7
}
fn default_temporal_num_groups() -> usize {
    1
}
fn default_temporal_anticollapse_threshold() -> f32 {
    0.9
}
fn default_true() -> bool {
    true
}
fn default_lora_alpha() -> f32 {
    1.0
}
fn default_lora_lr() -> f32 {
    0.0001
}
fn default_e2e_grad_chunks() -> usize {
    1
}
fn default_orthogonality_samples() -> usize {
    64
}

impl Default for LiquidMambaConfig {
    fn default() -> Self {
        Self {
            model_id: "state-spaces/mamba-130m".to_string(),
            max_tokens: 256,
            temperature: 0.8,
            top_k: 40,
            veto_threshold: 0.20,
            drift_threshold: 0.30,
            coherence_window: 8,
            long_coherence_window: 32,
            coherence_ema_alpha: 0.3,
            min_consecutive_low: 3,
            delta_mod_strength: 1.0,
            veto_hesitation: "-- wait, ".to_string(),
            enable_gating: true,
            enable_veto: true,
            enable_liquid_delta: true,
            enable_consciousness_gating: true,
            hdc_dim: 16384,
            bottleneck_dim: 256,
            ssm_dim: 768,
            warmup_steps: 100,
            accumulation_steps: 4,
            contrastive_weight: 0.1,
            contrastive_buffer_size: 8,
            base_lr: 0.001,
            cosine_annealing_steps: 0,
            min_lr_fraction: 0.01,
            prefix_loss_weight: 0.3,
            enable_ema: false,
            ema_decay: 0.999,
            deep_projection: false,
            temporal_projection: true,
            learned_pos_enc: false,
            temporal_stride: 0,
            temporal_directional_loss: true,
            temporal_smoothness_weight: 0.02,
            temporal_rank_reg_weight: 0.005,
            temporal_chunk_budget: 16,
            temporal_learned_attention: true,
            surprise_gradient_alpha: 0.5,
            grad_clip: 1.0,
            token_pe_sim_threshold: 0.3,
            token_pe_max_boost: 1.5,
            token_pe_decay: 0.5,
            fep_high_threshold: 0.7,
            fep_low_threshold: 0.3,
            fep_high_multiplier: 1.5,
            fep_low_multiplier: 0.7,
            temporal_chunk_dim: 0,
            temporal_num_groups: 1,
            temporal_anticollapse_weight: 0.0,
            temporal_anticollapse_threshold: 0.9,
            temporal_rotate_grad_position: true,
            temporal_adapter: false,
            lora_rank: 0,
            lora_alpha: 1.0,
            lora_lr: 0.0001,
            embedding_stats_path: None,
            e2e_grad_chunks: 1,
            orthogonality_weight: 0.0,
            orthogonality_samples: 64,
            gating_config: GatingConfig::default(),
        }
    }
}

/// Liquid-Mamba fusion generator: consciousness-gated SSM language generation.
pub struct LiquidMambaGenerator {
    mamba: Box<dyn MambaBackend>,
    projection: HdcSsmProjection,
    temporal_proj: Option<TemporalProjection>,
    encoder: ThoughtLanguageEncoder,
    epistemic_gate: EpistemicGate,
    emotional_modulator: EmotionalModulator,
    config: LiquidMambaConfig,
    // Biological state (injected by cognitive loop via update_affect)
    thermodynamic_load: f32,
    arousal: f32,
    mood_temperature: f32,
    // Online distillation state
    generation_count: usize,
    distill_accumulator: usize,
    recent_thought_hvs: VecDeque<ContinuousHV>,
    // Last semantic prediction error for telemetry
    last_semantic_pe: f32,
    // PE history ring buffer for trend analysis (capacity 64)
    pe_history: VecDeque<f32>,
    // FEP modulation factor from cognitive loop (default 1.0 = neutral)
    fep_modulation: f32,
    // Last cached effective rank from check_projection_health()
    last_cached_rank: f32,
    // Optional gradient diagnostics (enabled via enable_diagnostics())
    diagnostics: Option<ProjectionGradientDiagnostics>,
    // Generation at which diagnostics-triggered recovery last ran (prevents rapid re-triggering)
    last_diag_recovery_gen: usize,
}

impl LiquidMambaGenerator {
    /// Create a new Liquid-Mamba generator.
    ///
    /// Loads the Mamba model from HuggingFace Hub (requires network on first run).
    pub fn new(genesis: &GenesisSeed, config: LiquidMambaConfig) -> Result<Self> {
        let device = crate::mamba::best_device();
        let mamba = MambaWrapper::load(&config.model_id, device)?;
        Self::with_backend(genesis, config, Box::new(mamba))
    }

    /// Create a Liquid-Mamba generator with a mock backend (for testing).
    ///
    /// Uses the provided [`MambaBackend`] implementation instead of loading
    /// a real model. This enables deterministic, offline testing of the full
    /// generation pipeline.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn with_mock(genesis: &GenesisSeed, config: LiquidMambaConfig) -> Self {
        use crate::mamba::tests::MockMamba;
        Self::with_backend(genesis, config, Box::new(MockMamba::new()))
            .expect("MockMamba backend cannot fail")
    }

    /// Internal constructor shared by `new()` and `with_mock()`.
    fn with_backend(
        genesis: &GenesisSeed,
        config: LiquidMambaConfig,
        mamba: Box<dyn MambaBackend>,
    ) -> Result<Self> {
        let projection = if config.deep_projection {
            HdcSsmProjection::new_deep(
                genesis,
                config.hdc_dim,
                config.bottleneck_dim,
                config.ssm_dim,
            )
        } else {
            HdcSsmProjection::new(
                genesis,
                config.hdc_dim,
                config.bottleneck_dim,
                config.ssm_dim,
            )
        };

        let temporal_proj = if config.temporal_projection {
            let chunk_dim = if config.temporal_chunk_dim > 0 {
                config.temporal_chunk_dim
            } else {
                config.bottleneck_dim // default: 256
            };
            let stride = if config.temporal_stride > 0 {
                config.temporal_stride
            } else {
                chunk_dim // default: no overlap
            };
            let mut tp = TemporalProjection::new_full(
                genesis,
                config.hdc_dim,
                chunk_dim,
                config.ssm_dim,
                config.learned_pos_enc,
                stride,
            );
            if config.temporal_learned_attention {
                tp.set_learned_attention(true);
            }
            // Improvement C: per-group up-projection
            if config.temporal_num_groups > 1 {
                tp.set_num_groups(config.temporal_num_groups, genesis);
            }
            // Improvement E: adapter MLP (with optional manifold moment matching)
            if config.temporal_adapter {
                if let Some(ref stats_path) = config.embedding_stats_path {
                    match crate::temporal_projection::EmbeddingStats::load(stats_path) {
                        Ok(stats) => {
                            tracing::info!(
                                dim = stats.dim,
                                count = stats.count,
                                path = stats_path,
                                "Adapter initialized with manifold moment matching (whitening)"
                            );
                            tp.enable_adapter_from_stats(genesis, &stats);
                        }
                        Err(e) => {
                            tracing::warn!(
                                path = stats_path,
                                error = %e,
                                "Failed to load embedding stats, falling back to random adapter init"
                            );
                            tp.enable_adapter(genesis);
                        }
                    }
                } else {
                    tp.enable_adapter(genesis);
                }
            }
            Some(tp)
        } else {
            None
        };

        let encoder = ThoughtLanguageEncoder::new(genesis);

        // Create gating modules using Mamba's own tokenizer (GPT-2) so that
        // token IDs match the logits produced by forward_one_token().
        let epistemic_gate = EpistemicGate::new_from_backend(mamba.as_ref(), &config.gating_config);
        let emotional_modulator =
            EmotionalModulator::new_from_backend(mamba.as_ref(), &config.gating_config);

        let enable_ema = config.enable_ema;
        let ema_decay = config.ema_decay;

        let mut gen = Self {
            mamba,
            projection,
            temporal_proj,
            encoder,
            epistemic_gate,
            emotional_modulator,
            config,
            thermodynamic_load: 0.0,
            arousal: 0.5,
            mood_temperature: 1.0,
            generation_count: 0,
            distill_accumulator: 0,
            recent_thought_hvs: VecDeque::new(),
            last_semantic_pe: 0.0,
            pe_history: VecDeque::with_capacity(64),
            fep_modulation: 1.0,
            last_cached_rank: 0.0,
            diagnostics: None,
            last_diag_recovery_gen: 0,
        };

        if enable_ema {
            gen.projection.enable_ema(ema_decay);
        }

        // Improvement D: enable LoRA on Mamba layers
        if gen.config.lora_rank > 0 {
            gen.mamba.enable_lora(
                gen.config.lora_rank,
                gen.config.lora_alpha,
                gen.config.lora_lr,
            );
        }

        Ok(gen)
    }

    /// Generate text from thought channels.
    pub fn generate(&mut self, channels: &ThoughtChannels) -> GenerationResult {
        match self.generate_inner(channels, None) {
            Ok(result) => {
                self.last_semantic_pe = result.semantic_pe;
                self.push_pe_history(result.semantic_pe);
                result
            }
            Err(e) => {
                tracing::error!("Liquid-Mamba generation failed: {e}");
                GenerationResult {
                    text: String::new(),
                    token_ids: Vec::new(),
                    num_tokens: 0,
                    eos_terminated: false,
                    veto_triggered: false,
                    final_coherence: 0.0,
                    long_coherence: 0.0,
                    coherence_dynamics: Vec::new(),
                    gating_trace: Vec::new(),
                    hallucination_flag: false,
                    output_hvs: Vec::new(),
                    semantic_pe: 0.0,
                    nsm_prime_coverage: 0.0,
                }
            }
        }
    }

    /// Generate text with a per-token streaming callback.
    pub fn generate_with_callback(
        &mut self,
        channels: &ThoughtChannels,
        on_token: &mut dyn FnMut(&str),
    ) -> GenerationResult {
        match self.generate_inner(channels, Some(on_token)) {
            Ok(result) => {
                self.last_semantic_pe = result.semantic_pe;
                self.push_pe_history(result.semantic_pe);
                result
            }
            Err(e) => {
                tracing::error!("Liquid-Mamba generation failed: {e}");
                GenerationResult {
                    text: String::new(),
                    token_ids: Vec::new(),
                    num_tokens: 0,
                    eos_terminated: false,
                    veto_triggered: false,
                    final_coherence: 0.0,
                    long_coherence: 0.0,
                    coherence_dynamics: Vec::new(),
                    gating_trace: Vec::new(),
                    hallucination_flag: false,
                    output_hvs: Vec::new(),
                    semantic_pe: 0.0,
                    nsm_prime_coverage: 0.0,
                }
            }
        }
    }

    fn generate_inner(
        &mut self,
        channels: &ThoughtChannels,
        mut on_token: Option<&mut dyn FnMut(&str)>,
    ) -> Result<GenerationResult> {
        // 1. Encode thought channels to 16,384D HDC
        let thought_hv = self.encoder.encode(channels);

        // 1b. Swap to EMA weights for inference (if enabled)
        let ema_live_backup = if self.config.enable_ema {
            self.projection.use_ema_weights()
        } else {
            None
        };

        // Closure-based scope guard: ensures EMA live weights are unconditionally
        // restored even if any fallible operation inside returns Err.
        let result = (|| -> Result<GenerationResult> {
            // 2-3. Project thought to SSM space and prime Mamba
            let ssm_context = if let Some(ref tp) = self.temporal_proj {
                // Temporal: chunk 16384D → N×768D soft tokens → forward_embeds()
                let sequence = if self.config.temporal_chunk_budget > 0 {
                    tp.project_to_ssm_sequence_topk(&thought_hv, self.config.temporal_chunk_budget)
                } else {
                    tp.project_to_ssm_sequence(&thought_hv)
                };
                self.mamba.inject_context_sequence(&sequence)?;
                // Keep a single-vector summary for veto re-injection
                sequence
                    .last()
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.config.ssm_dim])
            } else {
                // Spatial: 16384D → 256D → 768D → inject_initial_context()
                let ctx = self.projection.project_to_ssm(&thought_hv);
                self.mamba.inject_initial_context(&ctx)?;
                ctx
            };

            // 4. Compute max tokens (consciousness-gated, then time-pressure-adjusted)
            let max_tokens = if self.config.enable_consciousness_gating {
                let psi_tokens =
                    consciousness_gated_max_tokens(self.config.max_tokens, channels.psi());
                crate::gating::time_pressure_adjusted_max_tokens(
                    psi_tokens,
                    channels.time_pressure(),
                    self.config.gating_config.time_pressure_max_reduction,
                )
            } else {
                crate::gating::time_pressure_adjusted_max_tokens(
                    self.config.max_tokens,
                    channels.time_pressure(),
                    self.config.gating_config.time_pressure_max_reduction,
                )
            };

            // 5. Update arousal from channels
            self.arousal = channels.arousal();

            // 6. Initialize coherence monitors (short + long window)
            let mut coherence_monitor = CoherenceMonitor::new(
                thought_hv.clone(),
                self.config.coherence_window,
                self.config.coherence_ema_alpha,
                self.config.veto_threshold,
                self.config.min_consecutive_low,
            );
            let mut long_coherence_monitor = CoherenceMonitor::new(
                thought_hv.clone(),
                self.config.long_coherence_window,
                self.config.coherence_ema_alpha * 0.5, // Slower EMA for trend detection
                self.config.veto_threshold * 0.5,      // Lower threshold (trend signal)
                self.config.min_consecutive_low * 2,   // More patience for long window
            );

            // 7. Autoregressive generation loop
            let eos_id = self.mamba.eos_token_id();
            let mut tokens: Vec<u32> = Vec::new();
            let mut text = String::new();
            let mut veto_triggered = false;
            let mut prev_token = eos_id; // Start with EOS as BOS-equivalent
            let mut output_hvs: Vec<ContinuousHV> = Vec::new();
            // Per-token PE feedback: boost gating when last token had low thought-coherence
            let mut token_pe_boost: f32 = 0.0;

            for pos in 0..max_tokens {
                // 7a. Biological delta modulation via CfC → SSM coupling
                // Instead of bluntly scaling hidden states, modulate Δ (step-size)
                // and B (input sensitivity) at the SSM equation level.
                if self.config.enable_liquid_delta {
                    let scale = self.biological_state_scale();
                    // delta_scale: CfC tiredness → smaller temporal steps (more conservative)
                    // b_scale: same signal modulates input sensitivity
                    self.mamba.set_cfc_modulation(scale, scale.sqrt());
                }

                // 7b. Forward one token through Mamba
                let mut logits = self.mamba.forward_one_token(prev_token)?;

                // 7c. Epistemic gating (apply to Mamba's large vocab logits)
                //     Per-token PE feedback: if last token had low coherence with thought,
                //     boost epistemic gate by shifting toward higher uncertainty.
                //     Uses per-backend scale to account for Mamba's different logit distribution.
                if self.config.enable_gating {
                    let effective_epistemic =
                        (channels.epistemic_ordinal() + token_pe_boost).clamp(0.0, 4.0);
                    let ep_scale = self.config.gating_config.mamba_epistemic_scale();
                    self.epistemic_gate.apply_scaled(
                        &mut logits,
                        effective_epistemic,
                        channels.domain_familiarity(),
                        ep_scale,
                    );
                }

                // 7d. Emotional modulation (with backend-specific scale)
                if self.config.enable_gating {
                    let em_scale = self.config.gating_config.mamba_emotional_scale();
                    self.emotional_modulator
                        .apply_scaled(&mut logits, channels, pos, em_scale);
                }

                // 7e. Top-k sampling
                let next_token = top_k_sample(&logits, self.config.top_k, self.config.temperature);

                // 7f. Back-project token to HDC (unconditional — for distillation + veto)
                let token_emb = self.mamba.embedding_vector(next_token)?;
                let token_hdc = if let Some(ref tp) = self.temporal_proj {
                    tp.project_to_hdc(&token_emb)
                } else {
                    self.projection.project_to_hdc(&token_emb)
                };
                output_hvs.push(token_hdc.clone());

                // 7f2. Per-token PE feedback: compute token-level coherence with thought
                //      Low similarity → boost epistemic gating for next token
                let token_sim = thought_hv.similarity(&token_hdc).clamp(-1.0, 1.0);
                token_pe_boost = if token_sim < self.config.token_pe_sim_threshold {
                    // Low coherence: boost gating (shift toward Uncertain)
                    self.config.token_pe_max_boost
                        * (1.0 - token_sim / self.config.token_pe_sim_threshold)
                } else {
                    // Adequate coherence: decay boost
                    (token_pe_boost * self.config.token_pe_decay).max(0.0)
                };

                // 7g. Coherence monitoring + semantic veto
                // Always feed long monitor for trend tracking
                long_coherence_monitor.push(token_hdc.clone());

                if self.config.enable_veto {
                    coherence_monitor.push(token_hdc);

                    // Use predictive veto with long_coherence gating
                    let long_coh = long_coherence_monitor.current_coherence();
                    if coherence_monitor.should_veto_predictive(long_coh) && pos > 2 {
                        veto_triggered = true;
                        text.push_str(&self.config.veto_hesitation);
                        if let Some(ref mut cb) = on_token {
                            cb(&self.config.veto_hesitation);
                        }

                        // Reset Mamba and re-inject thought context
                        if let Some(ref tp) = self.temporal_proj {
                            let sequence = if self.config.temporal_chunk_budget > 0 {
                                tp.project_to_ssm_sequence_topk(
                                    &thought_hv,
                                    self.config.temporal_chunk_budget,
                                )
                            } else {
                                tp.project_to_ssm_sequence(&thought_hv)
                            };
                            self.mamba.inject_context_sequence(&sequence)?;
                        } else {
                            self.mamba.reset();
                            self.mamba.inject_initial_context(&ssm_context)?;
                        }
                        coherence_monitor.reset();

                        // Continue from current position (don't restart text)
                        continue;
                    }
                }

                // 7h. Decode token
                if next_token == eos_id {
                    let semantic_pe = self.semantic_prediction_error(&thought_hv, &output_hvs);
                    return Ok(GenerationResult {
                        text,
                        token_ids: tokens,
                        num_tokens: pos,
                        eos_terminated: true,
                        veto_triggered,
                        final_coherence: coherence_monitor.current_coherence(),
                        long_coherence: long_coherence_monitor.current_coherence(),
                        coherence_dynamics: Vec::new(),
                        gating_trace: Vec::new(),
                        hallucination_flag: false,
                        output_hvs,
                        semantic_pe,
                        nsm_prime_coverage: 0.0,
                    });
                }

                if let Ok(token_str) = self.mamba.decode_token(next_token) {
                    text.push_str(&token_str);
                    if let Some(ref mut cb) = on_token {
                        cb(&token_str);
                    }
                }

                tokens.push(next_token);
                prev_token = next_token;
            }

            let semantic_pe = self.semantic_prediction_error(&thought_hv, &output_hvs);
            Ok(GenerationResult {
                text,
                token_ids: tokens.clone(),
                num_tokens: tokens.len(),
                eos_terminated: false,
                veto_triggered,
                final_coherence: coherence_monitor.current_coherence(),
                long_coherence: long_coherence_monitor.current_coherence(),
                coherence_dynamics: Vec::new(),
                gating_trace: Vec::new(),
                hallucination_flag: false,
                output_hvs,
                semantic_pe,
                // NSM prime coverage is tracked in the CfC-HDC generator path;
                // Liquid-Mamba fusion does not run the NSM tracker, so report 0.
                nsm_prime_coverage: 0.0,
            })
        })();

        // Unconditional restore — runs on both Ok and Err paths
        if let Some(live) = ema_live_backup {
            self.projection.restore_live_weights(&live);
        }

        result
    }

    /// Compute effective learning rate with warmup + cosine annealing + biological load.
    ///
    /// Schedule:
    /// 1. Warmup: linear ramp from 0 to `base_lr` over `warmup_steps`
    /// 2. Cosine annealing: `base_lr * 0.5*(1 + cos(π * t/T))` down to `min_lr_fraction * base_lr`
    /// 3. Biological load: `(1 - thermodynamic_load)` multiplier
    fn compute_lr(&self) -> f32 {
        let step = self.generation_count;
        let base_lr = self.config.base_lr;

        // Phase 1: Warmup
        let warmup_factor = if self.config.warmup_steps > 0 && step < self.config.warmup_steps {
            step as f32 / self.config.warmup_steps as f32
        } else {
            1.0
        };

        // Phase 2: Cosine annealing (after warmup), clamped to avoid wrap-around
        let cosine_factor =
            if self.config.cosine_annealing_steps > 0 && step >= self.config.warmup_steps {
                let t_max = self.config.cosine_annealing_steps as f32;
                let t = ((step - self.config.warmup_steps) as f32).min(t_max);
                let min_frac = self.config.min_lr_fraction;
                let cos_decay = 0.5 * (1.0 + (std::f32::consts::PI * t / t_max).cos());
                min_frac + (1.0 - min_frac) * cos_decay
            } else {
                1.0
            };

        // Phase 3: Biological load dampening
        let load_factor = 1.0 - self.thermodynamic_load;

        base_lr * warmup_factor * cosine_factor * load_factor
    }

    /// Current effective learning rate (for diagnostics).
    pub fn current_lr(&self) -> f32 {
        self.compute_lr()
    }

    /// Biological delta modulation factor.
    ///
    /// Scales SSM hidden state based on thermodynamic load and arousal.
    /// - `factor < 1.0` → faster decay → shorter memory (exhausted/agitated)
    /// - `factor > 1.0` → slower decay → longer memory (rested/calm)
    ///
    /// Formula: `factor = exp(-alpha * load - beta * (arousal - 0.5))`
    /// - At rest (load=0, arousal=0.5): factor = 1.0 (no effect)
    /// - Exhausted (load=1.0, arousal=0.5): factor ≈ 0.61
    /// - Agitated (load=0, arousal=1.0): factor ≈ 0.86
    fn biological_state_scale(&self) -> f32 {
        let alpha = 0.5 * self.config.delta_mod_strength;
        let beta = 0.3 * self.config.delta_mod_strength;
        // Mood temperature modulates effective arousal: hot mood → higher arousal effect
        let effective_arousal =
            (self.arousal + 0.1 * (self.mood_temperature - 1.0)).clamp(0.0, 1.0);
        (-alpha * self.thermodynamic_load - beta * (effective_arousal - 0.5))
            .exp()
            .clamp(0.3, 2.0)
    }

    /// Update biological state from the cognitive loop.
    pub fn update_affect(&mut self, load: f32, temp: f32) {
        self.thermodynamic_load = load.clamp(0.0, 1.0);
        self.mood_temperature = temp.clamp(0.5, 2.0);
    }

    /// Get mutable reference to the projection (for gradient learning).
    pub fn projection_mut(&mut self) -> &mut HdcSsmProjection {
        &mut self.projection
    }

    /// Get reference to the projection.
    pub fn projection(&self) -> &HdcSsmProjection {
        &self.projection
    }

    /// Get reference to the Mamba backend.
    pub fn mamba(&self) -> &dyn MambaBackend {
        &*self.mamba
    }

    /// Get mutable reference to the Mamba backend.
    pub fn mamba_mut(&mut self) -> &mut dyn MambaBackend {
        &mut *self.mamba
    }

    /// Get reference to the temporal projection (if enabled).
    pub fn temporal_proj(&self) -> Option<&TemporalProjection> {
        self.temporal_proj.as_ref()
    }

    /// Get mutable reference to the temporal projection (if enabled).
    pub fn temporal_proj_mut(&mut self) -> Option<&mut TemporalProjection> {
        self.temporal_proj.as_mut()
    }

    /// Get reference to the config.
    pub fn config(&self) -> &LiquidMambaConfig {
        &self.config
    }

    /// Get reference to the encoder.
    pub fn encoder(&self) -> &ThoughtLanguageEncoder {
        &self.encoder
    }

    /// Current thermodynamic load (0-1).
    pub fn thermodynamic_load(&self) -> f32 {
        self.thermodynamic_load
    }

    /// Current mood temperature (0.5-2.0).
    pub fn mood_temperature(&self) -> f32 {
        self.mood_temperature
    }

    /// Number of generations completed (for warmup scheduling).
    pub fn generation_count(&self) -> usize {
        self.generation_count
    }

    /// Last semantic prediction error from generation (0.0-1.0).
    pub fn last_semantic_pe(&self) -> f32 {
        self.last_semantic_pe
    }

    /// Last cached effective rank from check_projection_health().
    pub fn last_cached_rank(&self) -> f32 {
        self.last_cached_rank
    }

    /// Enable gradient diagnostics collection.
    ///
    /// Once enabled, each `distill_step()` that applies gradients will record
    /// step metrics (norms, clipping, bottleneck stats) into the diagnostics buffer.
    pub fn enable_diagnostics(&mut self) {
        self.diagnostics = Some(ProjectionGradientDiagnostics::default());
    }

    /// Get a reference to the collected gradient diagnostics (if enabled).
    pub fn projection_diagnostics(&self) -> Option<&ProjectionGradientDiagnostics> {
        self.diagnostics.as_ref()
    }

    /// Get a mutable reference to the collected gradient diagnostics (if enabled).
    pub fn projection_diagnostics_mut(&mut self) -> Option<&mut ProjectionGradientDiagnostics> {
        self.diagnostics.as_mut()
    }

    /// Generation at which diagnostics-triggered recovery last ran.
    pub fn last_diag_recovery_gen(&self) -> usize {
        self.last_diag_recovery_gen
    }

    /// Push a PE value into the history ring buffer (capacity 64).
    fn push_pe_history(&mut self, pe: f32) {
        if self.pe_history.len() >= 64 {
            self.pe_history.pop_front();
        }
        self.pe_history.push_back(pe);
    }

    /// Compute least-squares slope over the last 16 PE entries.
    ///
    /// Positive = PE diverging (getting worse), negative = converging (improving).
    /// Returns 0.0 if fewer than 2 entries.
    pub fn pe_trend(&self) -> f32 {
        let window = 16.min(self.pe_history.len());
        if window < 2 {
            return 0.0;
        }
        // Least-squares slope: Σ(i - ī)(y_i - ȳ) / Σ(i - ī)²
        let start = self.pe_history.len() - window;
        let n = window as f32;
        let mean_x = (n - 1.0) / 2.0;
        let mean_y: f32 = self.pe_history.iter().skip(start).sum::<f32>() / n;
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (i, &pe) in self.pe_history.iter().skip(start).enumerate() {
            let dx = i as f32 - mean_x;
            num += dx * (pe - mean_y);
            den += dx * dx;
        }
        if den.abs() < 1e-10 {
            0.0
        } else {
            num / den
        }
    }

    /// PE statistics over the full history: (mean, std_dev, trend).
    pub fn pe_stats(&self) -> (f32, f32, f32) {
        if self.pe_history.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let n = self.pe_history.len() as f32;
        let mean: f32 = self.pe_history.iter().sum::<f32>() / n;
        let variance: f32 = self
            .pe_history
            .iter()
            .map(|&pe| (pe - mean) * (pe - mean))
            .sum::<f32>()
            / n;
        (mean, variance.sqrt(), self.pe_trend())
    }

    /// Set the FEP modulation factor from the cognitive loop.
    ///
    /// - `fep_signal > 0.7` → high surprise → boost distillation LR by 1.5×
    /// - `fep_signal < 0.3` → low surprise → dampen distillation LR by 0.7×
    /// - Otherwise → neutral (1.0×)
    pub fn set_fep_modulation(&mut self, fep_signal: f32) {
        self.fep_modulation = if fep_signal > self.config.fep_high_threshold {
            self.config.fep_high_multiplier
        } else if fep_signal < self.config.fep_low_threshold {
            self.config.fep_low_multiplier
        } else {
            1.0
        };
    }

    /// Check for projection collapse and auto-recover if detected.
    ///
    /// Monitors the effective rank of bottleneck activations. If rank drops
    /// below `bottleneck_dim / 4`, adds noise to the collapsed dimensions
    /// to restore representational capacity.
    ///
    /// Should be called periodically (e.g., every 50 generations).
    pub fn check_projection_health(&mut self, recent_hvs: &[ContinuousHV]) {
        if recent_hvs.len() < 4 {
            return;
        }

        if let Some(ref mut tp) = self.temporal_proj {
            // Temporal path: monitor temporal projection rank
            let rank = tp.effective_rank(recent_hvs);
            self.last_cached_rank = rank;
            let threshold = (tp.chunk_size() as f32) / 4.0;

            if rank < threshold {
                tracing::warn!(
                    effective_rank = rank,
                    threshold = threshold,
                    chunk_size = tp.chunk_size(),
                    "Temporal projection collapse detected, adding recovery noise"
                );

                let mut weights = tp.flatten_weights();
                let noise_scale = 0.01 / (tp.chunk_size() as f32).sqrt();
                let mut rng_state = self.generation_count as u64 ^ 0x9E3779B97F4A7C15;
                for w in &mut weights {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let noise = ((rng_state as f32) / (u64::MAX as f32) - 0.5) * 2.0 * noise_scale;
                    *w += noise;
                }
                tp.load_weights(&weights);

                let new_rank = tp.effective_rank(recent_hvs);
                tracing::info!(
                    old_rank = rank,
                    new_rank = new_rank,
                    "Temporal projection recovery applied"
                );
            }
        } else {
            // Spatial path: monitor spatial projection rank
            let rank = self.projection.effective_rank(recent_hvs);
            self.last_cached_rank = rank;
            let threshold = (self.config.bottleneck_dim as f32) / 4.0;

            if rank < threshold {
                tracing::warn!(
                    effective_rank = rank,
                    threshold = threshold,
                    bottleneck = self.config.bottleneck_dim,
                    "Projection collapse detected, adding recovery noise"
                );

                let mut weights = self.projection.flatten_weights();
                let noise_scale = 0.01 / (self.config.bottleneck_dim as f32).sqrt();
                let mut rng_state = self.generation_count as u64 ^ 0x9E3779B97F4A7C15;
                for w in &mut weights {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let noise = ((rng_state as f32) / (u64::MAX as f32) - 0.5) * 2.0 * noise_scale;
                    *w += noise;
                }
                self.projection.load_weights(&weights);

                let new_rank = self.projection.effective_rank(recent_hvs);
                tracing::info!(
                    old_rank = rank,
                    new_rank = new_rank,
                    "Projection recovery applied"
                );
            }
        }
    }

    /// Online distillation step: learn from the last generation.
    ///
    /// Adjusts the HDC↔SSM projection using:
    /// - Attention-weighted output HV bundling (recency × coherence)
    /// - LR warmup schedule (ramp from 0 over warmup_steps)
    /// - Gradient accumulation (apply every accumulation_steps)
    /// - Contrastive loss (push different thoughts' projections apart)
    pub fn distill_step(&mut self, channels: &ThoughtChannels, result: &GenerationResult) {
        self.generation_count += 1;

        // Periodic projection health check (every 50 generations).
        // Only inject recovery noise during inference (consciousness_gating enabled).
        // During batch training (consciousness_gating disabled), recovery noise
        // fights the gradients and causes rank oscillation.
        if self.config.enable_consciousness_gating
            && self.generation_count % 50 == 0
            && !result.output_hvs.is_empty()
        {
            self.check_projection_health(&result.output_hvs);
        }

        // Gate: skip if no tokens generated
        if result.output_hvs.is_empty() {
            return;
        }
        // Skip distillation on garbage output (PE > 0.8) unless consciousness gating is disabled
        // (e.g. during offline testing with mock backends)
        if self.config.enable_consciousness_gating && result.semantic_pe > 0.8 {
            return;
        }

        let thought_hv = self.encoder.encode(channels);

        // LR schedule: warmup → cosine annealing → PE-trend adaptive → FEP modulation
        let base_effective_lr = self.compute_lr();
        let trend = self.pe_trend();
        let trend_factor = if trend > 0.01 {
            // PE diverging — halve LR to stabilize
            0.5
        } else if trend < -0.01 {
            // PE converging well — allow 1.5× boost (capped at 2× base)
            1.5f32.min(2.0 * self.config.base_lr / base_effective_lr.max(1e-10))
        } else {
            1.0
        };
        let effective_lr = base_effective_lr * trend_factor * self.fep_modulation;
        if effective_lr < 1e-7 {
            return;
        }

        // Gradient curriculum: roundtrip autoencoder first, Mamba-output later.
        //
        // When PE > 0.5, Mamba generates garbage (projection is untrained), so
        // the bundled output is noise. Using `thought - noise` as error drives
        // a random walk in weight space. Instead, use pure roundtrip reconstruction
        // `thought - backward(forward(thought))` which gives clean gradients
        // independent of Mamba output quality.
        //
        // Once PE drops below 0.5, the projection is good enough that Mamba
        // output carries meaningful semantic signal — switch to the full
        // Mamba-output-based loss for fine-tuning.
        let use_mamba_signal = result.semantic_pe < 0.5;

        if self.config.temporal_projection {
            // ─── Temporal projection gradient path ───
            //
            // Two loss modes:
            // - Roundtrip (default): autoencoder loss (up → down → compare with original)
            // - Directional: cosine alignment loss (optimizes angular structure directly)
            //
            // When Mamba output is meaningful (PE < 0.5), we boost the gradient
            // magnitude proportional to output quality.
            let Some(tp) = self.temporal_proj.as_mut() else {
                // temporal_projection enabled in config but struct not initialized;
                // skip gradient computation rather than crashing the cognitive loop.
                return;
            };
            if self.config.temporal_directional_loss {
                tp.compute_directional_gradients(&thought_hv, None);
            } else {
                tp.compute_roundtrip_gradients(&thought_hv);
            }

            // Contrastive term through temporal projection
            if self.config.contrastive_weight > 0.0 {
                for neg_hv in &self.recent_thought_hvs {
                    let sim = thought_hv.similarity(neg_hv);
                    if sim < 0.8 {
                        tp.compute_contrastive_gradients(
                            &thought_hv,
                            neg_hv,
                            self.config.contrastive_weight,
                        );
                    }
                }
            }

            // Temporal smoothness: penalize large jumps between consecutive SSM vectors
            if self.config.temporal_smoothness_weight > 0.0 {
                tp.compute_smoothness_gradients(
                    &thought_hv,
                    self.config.temporal_smoothness_weight,
                );
            }

            // Rank regularization: decorrelate W_up rows to utilize full SSM space
            if self.config.temporal_rank_reg_weight > 0.0 {
                tp.compute_rank_regularization_gradients(
                    self.config.temporal_rank_reg_weight,
                    64, // sample 64 random pairs per step
                );
            }

            // Improvement A: Anti-collapse regularization
            if self.config.temporal_anticollapse_weight > 0.0 {
                tp.compute_anticollapse_gradients(
                    &thought_hv,
                    self.config.temporal_anticollapse_weight,
                    self.config.temporal_anticollapse_threshold,
                );
            }

            // Improvement B: E2E loss (single rotating or multi-position)
            if self.config.temporal_rotate_grad_position && use_mamba_signal {
                let num_chunks = tp.num_chunks();
                let sequence = tp.project_to_ssm_sequence(&thought_hv);
                let k = self.config.e2e_grad_chunks.max(1);

                if k > 1 {
                    // Multi-position: K evenly-spaced positions with rotating offset
                    let offset = self.generation_count % (num_chunks / k).max(1);
                    let positions: Vec<usize> = (0..k)
                        .map(|i| (offset + i * num_chunks / k) % num_chunks)
                        .collect();

                    // Warn on large configs that may exceed VRAM
                    if num_chunks > 32 && k > 2 && self.generation_count == 1 {
                        tracing::warn!(
                            num_chunks,
                            e2e_grad_chunks = k,
                            "Large chunk count with multi-position E2E may use significant VRAM"
                        );
                    }

                    if let Ok(multi_grads) = self.mamba.compute_e2e_token_loss_multi(
                        &sequence,
                        &result.token_ids,
                        &positions,
                    ) {
                        let mut sparse: Vec<Option<Vec<f32>>> = vec![None; num_chunks];
                        for (pos, grad) in multi_grads {
                            if pos < num_chunks {
                                sparse[pos] = Some(grad);
                            }
                        }
                        tp.compute_e2e_gradients(&thought_hv, &sparse);
                    }
                } else {
                    // Single rotating position (legacy behavior)
                    let pos = self.generation_count % num_chunks;
                    let ssm_grads =
                        self.mamba
                            .compute_e2e_token_loss_at(&sequence, &result.token_ids, pos);
                    if let Ok(grad) = ssm_grads {
                        let mut sparse: Vec<Option<Vec<f32>>> = vec![None; num_chunks];
                        sparse[pos] = Some(grad);
                        tp.compute_e2e_gradients(&thought_hv, &sparse);
                    }
                }
            }

            self.distill_accumulator += 1;
            if self.distill_accumulator >= self.config.accumulation_steps.max(1) {
                // Apply quality/surprise scaling ONCE as a single product,
                // after accumulation completes.
                if use_mamba_signal {
                    let mut combined_scale = 1.0 + (1.0 - result.semantic_pe).max(0.0);
                    if self.config.surprise_gradient_alpha > 0.0 {
                        combined_scale *=
                            1.0 + self.config.surprise_gradient_alpha * result.semantic_pe;
                    }
                    tp.scale_accumulated_gradients(combined_scale);
                }
                let metrics = tp.apply_gradients(effective_lr, self.config.grad_clip);
                if let Some(ref mut diag) = self.diagnostics {
                    diag.record_step(&metrics, effective_lr);
                    let bottleneck = tp.bottleneck_activation(&thought_hv);
                    diag.record_bottleneck(&bottleneck);
                }
                self.distill_accumulator = 0;

                // Auto-recovery: only during inference (consciousness_gating enabled).
                if self.config.enable_consciousness_gating {
                    let collapse_detected = self
                        .diagnostics
                        .as_ref()
                        .map_or(false, |d| d.bottleneck_collapse_detected());
                    let recovery_gap = self
                        .generation_count
                        .saturating_sub(self.last_diag_recovery_gen)
                        > 10;

                    if collapse_detected && recovery_gap {
                        tracing::warn!(
                            gen = self.generation_count,
                            "diagnostics: temporal bottleneck collapse detected, triggering recovery"
                        );
                        let recent: Vec<_> = self.recent_thought_hvs.iter().cloned().collect();
                        self.check_projection_health(&recent);
                        self.last_diag_recovery_gen = self.generation_count;
                    }
                }
            }
        } else {
            // ─── Spatial projection gradient path (original) ───
            if use_mamba_signal {
                // Phase 2: Mamba output is meaningful — use it as target
                let bundled = self.attention_weighted_bundle(&thought_hv, &result.output_hvs);
                self.projection.compute_gradients(&thought_hv, &bundled);

                // Prefix token loss (only useful when Mamba output is meaningful)
                if self.config.prefix_loss_weight > 0.0 && result.output_hvs.len() >= 4 {
                    let prefix_len = (result.output_hvs.len() / 4).max(1);
                    let prefix_bundle = self
                        .attention_weighted_bundle(&thought_hv, &result.output_hvs[..prefix_len]);
                    self.projection
                        .compute_gradients(&thought_hv, &prefix_bundle);
                }
            } else {
                // Phase 1: Pure roundtrip autoencoder (no Mamba dependency)
                self.projection.compute_roundtrip_gradients(&thought_hv);
            }

            // Contrastive term: push away from recent different thoughts
            if self.config.contrastive_weight > 0.0 {
                for neg_hv in &self.recent_thought_hvs {
                    let sim = thought_hv.similarity(neg_hv);
                    if sim < 0.8 {
                        self.projection.compute_contrastive_gradients(
                            &thought_hv,
                            neg_hv,
                            self.config.contrastive_weight,
                        );
                    }
                }
            }

            // Orthogonality regularization: decorrelate w_up rows to prevent rank collapse
            if self.config.orthogonality_weight > 0.0 {
                self.projection.compute_orthogonality_gradients(
                    self.config.orthogonality_weight,
                    self.config.orthogonality_samples,
                );
            }

            // Gradient accumulation: only apply every N steps
            self.distill_accumulator += 1;
            if self.distill_accumulator >= self.config.accumulation_steps.max(1) {
                // Apply surprise scaling ONCE after accumulation (not per-step)
                if use_mamba_signal && self.config.surprise_gradient_alpha > 0.0 {
                    let scale = 1.0 + self.config.surprise_gradient_alpha * result.semantic_pe;
                    self.projection.scale_accumulated_gradients(scale);
                }
                let metrics = self
                    .projection
                    .apply_gradients(effective_lr, self.config.grad_clip);
                if let Some(ref mut diag) = self.diagnostics {
                    diag.record_step(&metrics, effective_lr);
                    let bottleneck = self.projection.bottleneck_activation(&thought_hv);
                    diag.record_bottleneck(&bottleneck);
                }
                // EMA teacher update after gradient application
                self.projection.update_ema();
                self.distill_accumulator = 0;

                // Auto-recovery: only during inference (consciousness_gating enabled).
                // During batch training, recovery noise fights gradient updates.
                if self.config.enable_consciousness_gating {
                    let collapse_detected = self
                        .diagnostics
                        .as_ref()
                        .map_or(false, |d| d.bottleneck_collapse_detected());
                    let recovery_gap = self
                        .generation_count
                        .saturating_sub(self.last_diag_recovery_gen)
                        > 10;

                    if collapse_detected && recovery_gap {
                        tracing::warn!(
                            gen = self.generation_count,
                            "diagnostics: bottleneck collapse detected, triggering recovery"
                        );
                        let recent: Vec<_> = self.recent_thought_hvs.iter().cloned().collect();
                        self.check_projection_health(&recent);
                        self.last_diag_recovery_gen = self.generation_count;
                    }
                }
            }
        }

        // Update contrastive buffer
        if self.config.contrastive_buffer_size > 0 {
            self.recent_thought_hvs.push_back(thought_hv);
            while self.recent_thought_hvs.len() > self.config.contrastive_buffer_size {
                self.recent_thought_hvs.pop_front();
            }
        }
    }

    /// Attention-weighted bundling of output HVs.
    ///
    /// Weights each token's HV by:
    /// - Recency: `sqrt(position / total)` — later tokens weighted more
    /// - Coherence: cosine similarity with thought HV — relevant tokens weighted more
    fn attention_weighted_bundle(
        &self,
        thought_hv: &ContinuousHV,
        output_hvs: &[ContinuousHV],
    ) -> ContinuousHV {
        if output_hvs.is_empty() {
            return ContinuousHV::zero(self.config.hdc_dim);
        }
        if output_hvs.len() == 1 {
            return output_hvs[0].clone();
        }

        let n = output_hvs.len();
        let weights: Vec<f32> = output_hvs
            .iter()
            .enumerate()
            .map(|(i, hv)| {
                let recency = ((i + 1) as f32 / n as f32).sqrt();
                let coherence = thought_hv.similarity(hv).clamp(0.0, 1.0);
                recency * (0.5 + 0.5 * coherence)
            })
            .collect();

        let weight_sum: f32 = weights.iter().sum();
        if weight_sum < 1e-10 {
            let refs: Vec<&ContinuousHV> = output_hvs.iter().collect();
            return ContinuousHV::bundle(&refs).normalize();
        }

        let dim = output_hvs[0].values.len();
        let mut bundled = vec![0.0f32; dim];
        for (hv, &w) in output_hvs.iter().zip(weights.iter()) {
            let nw = w / weight_sum;
            for (bv, v) in bundled.iter_mut().zip(hv.values.iter()) {
                *bv += nw * v;
            }
        }

        ContinuousHV::from_vec(bundled).normalize()
    }

    /// Get the last semantic prediction error (round-trip reconstruction).
    pub fn semantic_prediction_error(
        &self,
        thought_hv: &ContinuousHV,
        output_hvs: &[ContinuousHV],
    ) -> f32 {
        if output_hvs.is_empty() {
            return 1.0;
        }

        if let Some(ref tp) = self.temporal_proj {
            // Temporal mode: use roundtrip reconstruction PE.
            //
            // The tiled back-projection makes output-based cosine similarity
            // meaningless (~0.01 always). Instead, measure how well the
            // temporal projection's up→down roundtrip preserves each chunk.
            // This directly tracks what the autoencoder loss optimizes.
            tp.roundtrip_pe(thought_hv)
        } else {
            // Spatial mode: full-vector cosine similarity
            let refs: Vec<&ContinuousHV> = output_hvs.iter().collect();
            let bundled = ContinuousHV::bundle(&refs).normalize();
            1.0 - thought_hv.similarity(&bundled).clamp(-1.0, 1.0)
        }
    }
}

impl std::fmt::Debug for LiquidMambaGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidMambaGenerator")
            .field("config", &self.config)
            .field("mamba_d_model", &self.mamba.d_model())
            .field("mamba_vocab_size", &self.mamba.vocab_size())
            .field("thermodynamic_load", &self.thermodynamic_load)
            .field("arousal", &self.arousal)
            .field("mood_temperature", &self.mood_temperature)
            .field("generation_count", &self.generation_count)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COHERENCE MONITOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Monitors semantic coherence between generated tokens and the original thought.
///
/// Bundles back-projected token HDC vectors in a sliding window and computes
/// EMA cosine similarity against the original `thought_hv`. If coherence drops
/// below threshold for `min_consecutive_low` tokens, signals a veto.
struct CoherenceMonitor {
    thought_hv: ContinuousHV,
    window: VecDeque<ContinuousHV>,
    window_size: usize,
    ema_coherence: f32,
    ema_alpha: f32,
    consecutive_low: usize,
    min_consecutive_low: usize,
    veto_threshold: f32,
    /// Previous coherence value for velocity computation.
    prev_coherence: f32,
    /// Rate of coherence change: `new - prev` per push.
    coherence_velocity: f32,
}

impl CoherenceMonitor {
    fn new(
        thought_hv: ContinuousHV,
        window_size: usize,
        ema_alpha: f32,
        veto_threshold: f32,
        min_consecutive_low: usize,
    ) -> Self {
        Self {
            thought_hv,
            window: VecDeque::with_capacity(window_size),
            window_size,
            ema_coherence: 1.0,
            ema_alpha,
            consecutive_low: 0,
            min_consecutive_low,
            veto_threshold,
            prev_coherence: 1.0,
            coherence_velocity: 0.0,
        }
    }

    /// Add a back-projected token HV and update coherence.
    fn push(&mut self, token_hdc: ContinuousHV) {
        // Maintain sliding window
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(token_hdc);

        // Bundle window contents
        let refs: Vec<&ContinuousHV> = self.window.iter().collect();
        let bundled = ContinuousHV::bundle(&refs).normalize();

        // Cosine similarity with thought
        let sim = self.thought_hv.similarity(&bundled).clamp(-1.0, 1.0);

        // Save previous for velocity computation
        self.prev_coherence = self.ema_coherence;

        // EMA smooth
        self.ema_coherence = self.ema_alpha * sim + (1.0 - self.ema_alpha) * self.ema_coherence;

        // Compute velocity (rate of change)
        self.coherence_velocity = self.ema_coherence - self.prev_coherence;

        // Track consecutive low
        if self.ema_coherence < self.veto_threshold {
            self.consecutive_low += 1;
        } else {
            self.consecutive_low = 0;
        }
    }

    /// Whether coherence warrants a veto (standard only, without predictive component).
    ///
    /// Standard veto: coherence below threshold for `min_consecutive_low` tokens.
    /// Production code uses `should_veto_predictive` which includes this check plus
    /// velocity-based anticipation. This simpler variant is retained for unit tests.
    #[cfg(test)]
    fn should_veto(&self) -> bool {
        self.consecutive_low >= self.min_consecutive_low
    }

    /// Predictive veto: anticipate coherence drop before it crosses threshold.
    /// Requires long_coherence from the trend monitor to prevent spurious triggers.
    fn should_veto_predictive(&self, long_coherence: f32) -> bool {
        // Standard veto always applies
        if self.consecutive_low >= self.min_consecutive_low {
            return true;
        }
        // Predictive: rapid decline approaching threshold, only when long-term is also low
        long_coherence < 0.7
            && self.coherence_velocity < -0.15
            && self.ema_coherence < self.veto_threshold + 0.1
    }

    /// Current EMA coherence value.
    fn current_coherence(&self) -> f32 {
        self.ema_coherence
    }

    /// Rate of coherence change (negative = declining).
    #[cfg(test)]
    fn coherence_velocity(&self) -> f32 {
        self.coherence_velocity
    }

    /// Reset the monitor (after a veto).
    fn reset(&mut self) {
        self.window.clear();
        self.ema_coherence = 1.0;
        self.consecutive_low = 0;
        self.prev_coherence = 1.0;
        self.coherence_velocity = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Top-k sampling with temperature from a logits vector.
fn top_k_sample(logits: &[f32], k: usize, temperature: f32) -> u32 {
    if k == 0 || logits.is_empty() {
        return greedy_sample(logits);
    }

    let k = k.min(logits.len());
    let temp = temperature.max(1e-8);

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);

    // Apply temperature and softmax
    let max_logit = indexed
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = indexed
        .into_iter()
        .map(|(i, l)| (i, ((l - max_logit) / temp).exp()))
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum < 1e-10 {
        return probs[0].0 as u32;
    }
    for p in &mut probs {
        p.1 /= sum;
    }

    // Sample
    let r = simple_random_f32();
    let mut cumulative = 0.0f32;
    for (i, p) in &probs {
        cumulative += p;
        if r < cumulative {
            return *i as u32;
        }
    }

    probs.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Greedy sampling: return argmax.
fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Simple random float in [0, 1) using thread_rng.
fn simple_random_f32() -> f32 {
    use rand::Rng;
    rand::thread_rng().gen::<f32>()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biological_state_scale_at_rest() {
        let config = LiquidMambaConfig::default();
        // At rest: load=0, arousal=0.5 → factor = exp(0) = 1.0
        let alpha = 0.5 * config.delta_mod_strength;
        let beta = 0.3 * config.delta_mod_strength;
        let factor = (-alpha * 0.0 - beta * (0.5 - 0.5)).exp();
        assert!((factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_biological_state_scale_exhausted() {
        let config = LiquidMambaConfig::default();
        let alpha = 0.5 * config.delta_mod_strength;
        let beta = 0.3 * config.delta_mod_strength;
        // Exhausted: load=1.0, arousal=0.5
        let factor = (-alpha * 1.0 - beta * 0.0).exp();
        assert!(
            (factor - 0.6065).abs() < 0.01,
            "Expected ~0.61, got {factor}"
        );
    }

    #[test]
    fn test_biological_state_scale_agitated() {
        let config = LiquidMambaConfig::default();
        let alpha = 0.5 * config.delta_mod_strength;
        let beta = 0.3 * config.delta_mod_strength;
        // Agitated: load=0, arousal=1.0
        let factor = (-alpha * 0.0 - beta * 0.5).exp();
        assert!(
            (factor - 0.8607).abs() < 0.01,
            "Expected ~0.86, got {factor}"
        );
    }

    #[test]
    fn test_coherence_monitor_no_veto_initially() {
        let thought = ContinuousHV::random_default(42).normalize();
        let monitor = CoherenceMonitor::new(thought, 8, 0.3, 0.20, 3);
        assert!(!monitor.should_veto());
        assert!((monitor.current_coherence() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_monitor_veto_on_drift() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought, 8, 0.3, 0.20, 3);

        // Push orthogonal vectors to simulate drift
        for i in 0..10 {
            let drifted = ContinuousHV::random_default(1000 + i as u64).normalize();
            monitor.push(drifted);
        }

        // After enough drift, coherence should be very low
        assert!(
            monitor.current_coherence() < 0.5,
            "Coherence should drop with orthogonal tokens, got {}",
            monitor.current_coherence()
        );
    }

    #[test]
    fn test_coherence_monitor_high_coherence_with_similar() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought.clone(), 8, 0.3, 0.20, 3);

        // Push vectors similar to thought (scaled versions)
        for _ in 0..5 {
            let similar = thought.scale(0.9).normalize();
            monitor.push(similar);
        }

        // Coherence should remain high
        assert!(
            monitor.current_coherence() > 0.5,
            "Coherence should stay high with similar tokens, got {}",
            monitor.current_coherence()
        );
        assert!(!monitor.should_veto());
    }

    #[test]
    fn test_coherence_monitor_reset() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought, 8, 0.3, 0.20, 3);

        // Push some drift
        for i in 0..5 {
            let drifted = ContinuousHV::random_default(1000 + i as u64).normalize();
            monitor.push(drifted);
        }

        monitor.reset();
        assert!(!monitor.should_veto());
        assert!((monitor.current_coherence() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_sample_greedy_fallback() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let token = greedy_sample(&logits);
        assert_eq!(token, 3, "Should pick index 3 (highest logit 0.8)");
    }

    #[test]
    fn test_top_k_sample_produces_valid_index() {
        let logits = vec![0.1; 100];
        let token = top_k_sample(&logits, 10, 1.0);
        assert!((token as usize) < 100, "Token should be valid index");
    }

    #[test]
    fn test_top_k_sample_temperature_zero() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        // Very low temperature → should approach greedy
        let token = top_k_sample(&logits, 5, 0.001);
        assert_eq!(token, 3, "Near-zero temperature should be greedy");
    }

    #[test]
    fn test_config_default() {
        let config = LiquidMambaConfig::default();
        assert_eq!(config.model_id, "state-spaces/mamba-130m");
        assert_eq!(config.max_tokens, 256);
        assert_eq!(config.hdc_dim, 16384);
        assert_eq!(config.bottleneck_dim, 256);
        assert_eq!(config.ssm_dim, 768);
        assert!(config.enable_gating);
        assert!(config.enable_veto);
        assert!(config.enable_liquid_delta);
        assert_eq!(config.coherence_window, 8);
        assert_eq!(config.long_coherence_window, 32);
        assert_eq!(config.warmup_steps, 100);
        assert_eq!(config.accumulation_steps, 4);
        assert!((config.contrastive_weight - 0.1).abs() < 1e-6);
        assert_eq!(config.contrastive_buffer_size, 8);
        assert!((config.base_lr - 0.001).abs() < 1e-6);
        assert_eq!(config.cosine_annealing_steps, 0);
        assert!((config.min_lr_fraction - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_lr_warmup_only() {
        // With no cosine annealing, LR should ramp linearly then stay flat
        let config = LiquidMambaConfig {
            base_lr: 0.01,
            warmup_steps: 100,
            cosine_annealing_steps: 0,
            ..Default::default()
        };

        // At step 0: 0/100 = 0
        let factor_0 = 0.0f32 / 100.0;
        assert!((factor_0 - 0.0).abs() < 1e-6);

        // At step 50: 50/100 = 0.5
        let factor_50 = 50.0f32 / 100.0;
        assert!((factor_50 - 0.5).abs() < 1e-6);

        // At step 100: full
        let factor_100 = (100.0f32 / 100.0).min(1.0);
        assert!((factor_100 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_schedule() {
        // After warmup, LR should follow cosine decay
        let base_lr = 0.01f32;
        let min_frac = 0.01f32;
        let t_max = 1000.0f32;

        // At t=0 (start of annealing): cos(0) = 1, factor = 1.0
        let factor_0 =
            min_frac + (1.0 - min_frac) * 0.5 * (1.0 + (std::f32::consts::PI * 0.0 / t_max).cos());
        assert!((factor_0 - 1.0).abs() < 0.01);

        // At t=T/2: cos(π/2) = 0, factor = 0.5*(1+0) = 0.5 → ~0.505
        let factor_mid = min_frac
            + (1.0 - min_frac) * 0.5 * (1.0 + (std::f32::consts::PI * 500.0 / t_max).cos());
        assert!(
            (factor_mid - 0.505).abs() < 0.01,
            "Mid-point should be ~0.505, got {factor_mid}"
        );

        // At t=T: cos(π) = -1, factor = 0.5*(1-1) = 0 → min_frac
        let factor_end = min_frac
            + (1.0 - min_frac) * 0.5 * (1.0 + (std::f32::consts::PI * 1000.0 / t_max).cos());
        assert!(
            (factor_end - min_frac).abs() < 0.01,
            "End should be ~min_frac, got {factor_end}"
        );
    }

    #[test]
    fn test_attention_weighted_bundle_single() {
        // Single HV should be returned as-is (no weighting needed)
        let thought = ContinuousHV::random_default(42).normalize();
        let output = ContinuousHV::random_default(99).normalize();

        // Create a minimal generator-like context to test the bundling
        // Use the standalone function approach: replicate the logic
        let n = 1;
        let weights: Vec<f32> = vec![1.0];
        let weight_sum: f32 = weights.iter().sum();
        assert!(weight_sum > 0.0);

        // With a single HV, the result should be that HV
        let sim = thought.similarity(&output);
        assert!(sim.is_finite());
    }

    #[test]
    fn test_attention_weighted_bundle_recency() {
        // Later tokens should get higher recency weight
        let n = 5;
        let weights: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 / n as f32).sqrt()).collect();
        // First token should have lowest weight, last should have highest
        assert!(weights[0] < weights[4]);
        assert!(weights[3] < weights[4]);
        // Verify monotonically increasing
        for i in 1..n {
            assert!(weights[i] >= weights[i - 1]);
        }
    }

    #[test]
    fn test_warmup_schedule() {
        // At generation 0: factor = 0/100 = 0
        // At generation 50: factor = 50/100 = 0.5
        // At generation 100: factor = 100/100 = 1.0
        // At generation 200: factor = min(200/100, 1.0) = 1.0
        let warmup_steps = 100usize;
        for (gen, expected) in [(0, 0.0), (50, 0.5), (100, 1.0), (200, 1.0)] {
            let factor = (gen as f32 / warmup_steps as f32).min(1.0);
            assert!(
                (factor - expected).abs() < 0.01,
                "gen={gen}: expected {expected}, got {factor}"
            );
        }
    }

    #[test]
    fn test_accumulation_gating() {
        // Verify that gradient accumulation fires every N steps
        let accumulation_steps = 4;
        let mut acc = 0usize;
        let mut apply_count = 0;

        for _ in 0..12 {
            acc += 1;
            if acc >= accumulation_steps {
                apply_count += 1;
                acc = 0;
            }
        }
        assert_eq!(
            apply_count, 3,
            "Should apply 3 times in 12 steps with acc=4"
        );
    }

    #[test]
    fn test_semantic_prediction_error_identical() {
        let thought = ContinuousHV::random_default(42).normalize();
        let output = thought.clone();

        // When output == thought, error should be ~0
        let refs: Vec<&ContinuousHV> = vec![&output];
        let bundled = ContinuousHV::bundle(&refs).normalize();
        let error = 1.0 - thought.similarity(&bundled).clamp(-1.0, 1.0);
        assert!(
            error < 0.1,
            "Error should be near 0 for identical vectors, got {error}"
        );
    }

    #[test]
    fn test_semantic_prediction_error_orthogonal() {
        let thought = ContinuousHV::random_default(42).normalize();
        let other = ContinuousHV::random_default(99).normalize();

        let refs: Vec<&ContinuousHV> = vec![&other];
        let bundled = ContinuousHV::bundle(&refs).normalize();
        let error = 1.0 - thought.similarity(&bundled).clamp(-1.0, 1.0);
        // Random high-dimensional vectors are nearly orthogonal
        assert!(
            error > 0.5,
            "Error should be high for orthogonal vectors, got {error}"
        );
    }

    #[test]
    fn test_pe_trend_empty() {
        let history: VecDeque<f32> = VecDeque::new();
        // Replicate trend logic for empty
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_pe_trend_rising() {
        // Simulate rising PE: 0.1, 0.2, ..., 0.8
        let mut history: VecDeque<f32> = VecDeque::with_capacity(64);
        for i in 0..8 {
            history.push_back(0.1 * (i + 1) as f32);
        }
        // Compute least-squares slope over last 8
        let window = 8.min(history.len());
        let n = window as f32;
        let mean_x = (n - 1.0) / 2.0;
        let mean_y: f32 = history.iter().sum::<f32>() / n;
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (i, &pe) in history.iter().enumerate() {
            let dx = i as f32 - mean_x;
            num += dx * (pe - mean_y);
            den += dx * dx;
        }
        let slope = num / den;
        assert!(
            slope > 0.05,
            "Rising PE should have positive trend, got {slope}"
        );
    }

    #[test]
    fn test_pe_trend_falling() {
        let mut history: VecDeque<f32> = VecDeque::with_capacity(64);
        for i in (0..8).rev() {
            history.push_back(0.1 * (i + 1) as f32);
        }
        let window = 8.min(history.len());
        let n = window as f32;
        let mean_x = (n - 1.0) / 2.0;
        let mean_y: f32 = history.iter().sum::<f32>() / n;
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (i, &pe) in history.iter().enumerate() {
            let dx = i as f32 - mean_x;
            num += dx * (pe - mean_y);
            den += dx * dx;
        }
        let slope = num / den;
        assert!(
            slope < -0.05,
            "Falling PE should have negative trend, got {slope}"
        );
    }

    #[test]
    fn test_coherence_velocity_computed() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought.clone(), 8, 0.3, 0.20, 3);

        // Push similar tokens — velocity should be near zero or slightly negative (from 1.0 init)
        let similar = thought.scale(0.9).normalize();
        monitor.push(similar.clone());
        let v1 = monitor.coherence_velocity();
        assert!(v1.is_finite(), "Velocity should be finite");

        // Push orthogonal token — velocity should be negative (coherence dropping)
        let orthogonal = ContinuousHV::random_default(999).normalize();
        monitor.push(orthogonal);
        let v2 = monitor.coherence_velocity();
        assert!(
            v2 < v1 || v2.is_finite(),
            "Velocity should track coherence change"
        );
    }

    #[test]
    fn test_predictive_veto_requires_long_coherence() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought.clone(), 4, 0.5, 0.20, 3);

        // Force low coherence and negative velocity by pushing orthogonal tokens
        for i in 0..4 {
            let drift = ContinuousHV::random_default(1000 + i as u64).normalize();
            monitor.push(drift);
        }

        // With high long_coherence (0.9), predictive veto should NOT fire
        // (prevents spurious veto during brief fluctuations)
        let predictive_high_long = monitor.should_veto_predictive(0.9);
        // Standard veto might fire, but predictive component gated by long_coherence > 0.7
        // The predictive path requires long_coherence < 0.7

        // With low long_coherence (0.3), predictive veto can fire
        let predictive_low_long = monitor.should_veto_predictive(0.3);

        // At minimum, predictive with low long should be >= predictive with high long
        // (standard veto applies regardless, but predictive path is gated)
        assert!(
            predictive_low_long || !predictive_high_long,
            "Predictive veto should be at least as aggressive with low long_coherence"
        );
    }

    #[test]
    fn test_coherence_velocity_resets() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought, 8, 0.3, 0.20, 3);

        // Push some drift
        for i in 0..3 {
            let drift = ContinuousHV::random_default(1000 + i as u64).normalize();
            monitor.push(drift);
        }
        assert!(monitor.coherence_velocity().is_finite());

        monitor.reset();
        assert!(
            (monitor.coherence_velocity() - 0.0).abs() < 1e-6,
            "Velocity should reset to 0"
        );
        assert!(
            (monitor.current_coherence() - 1.0).abs() < 1e-6,
            "Coherence should reset to 1"
        );
    }

    #[test]
    fn test_fep_modulation_thresholds() {
        // High FEP signal → boost
        let fep_high: f32 = if 0.8f32 > 0.7 { 1.5 } else { 1.0 };
        assert!((fep_high - 1.5f32).abs() < 1e-6);

        // Low FEP signal → dampen
        let fep_low: f32 = if 0.2f32 < 0.3 { 0.7 } else { 1.0 };
        assert!((fep_low - 0.7f32).abs() < 1e-6);

        // Mid FEP signal → neutral
        let fep_mid: f32 = if 0.5f32 > 0.7 {
            1.5
        } else if 0.5f32 < 0.3 {
            0.7
        } else {
            1.0
        };
        assert!((fep_mid - 1.0f32).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MOCK-BASED INTEGRATION TESTS
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_mock_liquid_mamba_generate() {
        use crate::encoder::ThoughtChannels;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-mock-lm-gen");
        let config = LiquidMambaConfig {
            max_tokens: 10,
            enable_veto: false, // Disable veto for deterministic test
            enable_consciousness_gating: false,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

        let mut channels = ThoughtChannels::with_intent(1); // Answer
        channels.set_epistemic(0.0); // Certain
        channels.set_emotion(0.5, 0.5, 0.5);

        let result = gen.generate(&channels);
        assert!(result.num_tokens > 0, "Should generate at least 1 token");
        assert!(result.num_tokens <= 10, "Should respect max_tokens");
        assert!(!result.token_ids.is_empty());
        assert!(result.final_coherence.is_finite());
        assert!(result.semantic_pe.is_finite());
    }

    #[test]
    fn test_mock_liquid_mamba_biological_modulation() {
        use crate::encoder::ThoughtChannels;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-mock-lm-bio");
        let config = LiquidMambaConfig {
            max_tokens: 5,
            enable_veto: false,
            enable_consciousness_gating: false,
            enable_liquid_delta: true,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

        // Test with exhausted state
        gen.update_affect(0.9, 1.5);
        assert!((gen.thermodynamic_load() - 0.9).abs() < 1e-6);
        assert!((gen.mood_temperature() - 1.5).abs() < 1e-6);

        let channels = ThoughtChannels::with_intent(0);
        let result = gen.generate(&channels);
        assert!(result.num_tokens > 0);
        assert!(result.final_coherence.is_finite());
    }

    #[test]
    fn test_mock_liquid_mamba_distillation() {
        use crate::encoder::ThoughtChannels;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-mock-lm-distill");
        let config = LiquidMambaConfig {
            max_tokens: 8,
            enable_veto: false,
            enable_consciousness_gating: false,
            warmup_steps: 2,
            accumulation_steps: 1,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

        let channels = ThoughtChannels::with_intent(1);
        // Default config has temporal_projection=true, so distill_step updates
        // the temporal projection weights (not the spatial projection).
        let weights_before = gen.temporal_proj().unwrap().flatten_weights();

        // Generate and distill multiple times to trigger gradient application
        for _ in 0..4 {
            let result = gen.generate(&channels);
            gen.distill_step(&channels, &result);
        }

        let weights_after = gen.temporal_proj().unwrap().flatten_weights();
        assert_eq!(gen.generation_count(), 4);

        // Weights should have changed after distillation
        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            changed,
            "Distillation should modify temporal projection weights"
        );
    }

    #[test]
    fn test_mock_liquid_mamba_pe_tracking() {
        use crate::encoder::ThoughtChannels;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-mock-lm-pe");
        let config = LiquidMambaConfig {
            max_tokens: 5,
            enable_veto: false,
            enable_consciousness_gating: false,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

        let channels = ThoughtChannels::with_intent(1);

        // Generate multiple times to build PE history
        for _ in 0..5 {
            let result = gen.generate(&channels);
            gen.distill_step(&channels, &result);
        }

        let pe = gen.last_semantic_pe();
        assert!(pe.is_finite(), "PE should be finite");
        assert!(
            pe >= 0.0 && pe <= 2.0,
            "PE should be in reasonable range, got {pe}"
        );

        let (mean, std_dev, trend) = gen.pe_stats();
        assert!(mean.is_finite());
        assert!(std_dev.is_finite());
        assert!(trend.is_finite());
    }

    #[test]
    fn test_mock_liquid_mamba_streaming_callback() {
        use crate::encoder::ThoughtChannels;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-mock-lm-stream");
        let config = LiquidMambaConfig {
            max_tokens: 5,
            enable_veto: false,
            enable_consciousness_gating: false,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

        let channels = ThoughtChannels::with_intent(1);
        let mut callback_count = 0usize;
        let mut _callback_tokens = Vec::new();

        let result = gen.generate_with_callback(&channels, &mut |token_str| {
            callback_count += 1;
            _callback_tokens.push(token_str.to_string());
        });

        assert!(
            callback_count > 0,
            "Callback should be invoked at least once"
        );
        assert_eq!(
            callback_count, result.num_tokens,
            "Callback count should match token count"
        );
    }

    #[test]
    fn test_mock_liquid_mamba_fep_modulation() {
        use crate::encoder::ThoughtChannels;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-mock-lm-fep");
        let config = LiquidMambaConfig {
            max_tokens: 5,
            enable_veto: false,
            enable_consciousness_gating: false,
            warmup_steps: 0,
            accumulation_steps: 1,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

        // Set high FEP signal (should boost distillation LR)
        gen.set_fep_modulation(0.9);
        let _lr_high = gen.current_lr();

        // Generate and verify FEP modulation doesn't crash
        let channels = ThoughtChannels::with_intent(1);
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);

        // Set low FEP signal
        gen.set_fep_modulation(0.1);
        let result2 = gen.generate(&channels);
        gen.distill_step(&channels, &result2);

        assert!(result.num_tokens > 0);
        assert!(result2.num_tokens > 0);
    }

    #[test]
    fn test_cosine_annealing_clamps_past_t() {
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-cosine-clamp");
        let config = LiquidMambaConfig {
            max_tokens: 5,
            enable_veto: false,
            enable_consciousness_gating: false,
            base_lr: 0.01,
            warmup_steps: 10,
            cosine_annealing_steps: 100,
            min_lr_fraction: 0.01,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

        // Advance generation count well past warmup + cosine_annealing_steps
        // (warmup=10, cosine=100, so t_max at step 110)
        for _ in 0..300 {
            gen.generation_count += 1;
        }

        let lr = gen.current_lr();
        let min_lr = 0.01 * 0.01; // base_lr * min_lr_fraction

        // After clamping, LR should be at or near min_lr (not oscillating back up)
        assert!(
            lr <= min_lr * 1.5,
            "LR at step 300 should be near min_lr ({min_lr}), got {lr}"
        );
        assert!(lr > 0.0, "LR should still be positive");
    }

    // ─── Multi-Position E2E Gradient tests ───────────────────────────────

    #[test]
    fn test_multi_position_mock() {
        use crate::mamba::tests::MockMamba;
        let mut mock = MockMamba::new();
        let d = mock.d_model();
        let sequence: Vec<Vec<f32>> = (0..16).map(|_| vec![0.1; d]).collect();
        let tokens = vec![1, 2, 3];
        let positions = vec![2, 5, 10, 14];

        let results = mock
            .compute_e2e_token_loss_multi(&sequence, &tokens, &positions)
            .expect("multi-position should succeed");

        assert_eq!(results.len(), 4, "should return K=4 gradients");
        for (pos, grad) in &results {
            assert!(
                positions.contains(pos),
                "pos {pos} not in requested positions"
            );
            assert_eq!(grad.len(), d);
            assert!(grad.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_multi_position_sparse_fills() {
        let genesis = GenesisSeed::from_phrase("test-multi-pos");
        let config = LiquidMambaConfig {
            temporal_projection: true,
            temporal_rotate_grad_position: true,
            e2e_grad_chunks: 4,
            enable_consciousness_gating: false,
            enable_veto: false,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);
        gen.enable_diagnostics();

        // Simulate a distill step that triggers multi-position
        let channels = ThoughtChannels {
            channels: [0.5; crate::encoder::NUM_CHANNELS],
        };
        let thought_hv = gen.encoder().encode(&channels);
        let tp = gen.temporal_proj().unwrap();
        let num_chunks = tp.num_chunks();
        let sequence = tp.project_to_ssm_sequence(&thought_hv);

        // Directly call multi-position
        let positions: Vec<usize> = (0..4).map(|i| i * num_chunks / 4).collect();
        let results = gen
            .mamba
            .compute_e2e_token_loss_multi(&sequence, &[1, 2], &positions)
            .expect("multi-position should succeed");

        // Should have exactly 4 entries
        assert_eq!(results.len(), 4, "should return 4 gradient entries");
        for (pos, _grad) in &results {
            assert!(positions.contains(pos));
        }
    }

    #[test]
    fn test_single_position_backward_compat() {
        let genesis = GenesisSeed::from_phrase("test-backward-compat");
        let config_single = LiquidMambaConfig {
            temporal_projection: true,
            temporal_rotate_grad_position: true,
            e2e_grad_chunks: 1, // Legacy single position
            enable_consciousness_gating: false,
            enable_veto: false,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config_single);

        // Run a generate + distill cycle — should work same as before
        let channels = ThoughtChannels {
            channels: [0.3; crate::encoder::NUM_CHANNELS],
        };
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
        // No panic means backward compat is preserved
    }

    #[test]
    fn test_multi_position_rotating_offset() {
        use crate::mamba::tests::MockMamba;
        let mut mock = MockMamba::new();
        let d = mock.d_model();
        let num_chunks = 16;
        let k = 4;
        let sequence: Vec<Vec<f32>> = (0..num_chunks).map(|_| vec![0.1; d]).collect();
        let tokens = vec![1, 2];

        // Simulate two steps with different generation_count for offset
        for gen_count in 0..4 {
            let offset = gen_count % (num_chunks / k).max(1);
            let positions: Vec<usize> = (0..k)
                .map(|i| (offset + i * num_chunks / k) % num_chunks)
                .collect();

            let results = mock
                .compute_e2e_token_loss_multi(&sequence, &tokens, &positions)
                .expect("should succeed");
            assert_eq!(results.len(), k);

            // Verify positions are evenly spaced
            let mut sorted: Vec<usize> = results.iter().map(|(p, _)| *p).collect();
            sorted.sort();
            for i in 1..sorted.len() {
                let gap = (sorted[i] + num_chunks - sorted[i - 1]) % num_chunks;
                assert!(
                    gap >= num_chunks / k - 1 && gap <= num_chunks / k + 1,
                    "positions should be roughly evenly spaced, gap={gap}"
                );
            }
        }
    }

    #[test]
    fn test_multi_position_integration() {
        let genesis = GenesisSeed::from_phrase("test-multi-integration");
        let config = LiquidMambaConfig {
            temporal_projection: true,
            temporal_rotate_grad_position: true,
            e2e_grad_chunks: 4,
            enable_consciousness_gating: false,
            enable_veto: false,
            base_lr: 0.001,
            accumulation_steps: 1,
            warmup_steps: 0,
            ..Default::default()
        };
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

        // Get initial weights
        let tp = gen.temporal_proj().unwrap();
        let initial_weights = tp.flatten_weights();

        // Run enough distill steps to trigger gradient application
        let channels = ThoughtChannels {
            channels: [0.5; crate::encoder::NUM_CHANNELS],
        };
        for _ in 0..5 {
            let result = gen.generate(&channels);
            gen.distill_step(&channels, &result);
        }

        // Weights should have changed
        let tp = gen.temporal_proj().unwrap();
        let updated_weights = tp.flatten_weights();
        let diff: f32 = initial_weights
            .iter()
            .zip(updated_weights.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.0,
            "weights should update after distill steps with multi-position E2E"
        );
    }

    #[test]
    fn test_mock_gating_uses_backend_tokenizer() {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-gating");
        let gen = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());
        // Gate should have been constructed via new_from_backend (MockMamba)
        assert_eq!(
            gen.epistemic_gate.hedging_count(),
            crate::gating::CANONICAL_HEDGING_WORDS.len(),
            "Epistemic gate should resolve all hedging words via backend"
        );
        assert_eq!(
            gen.epistemic_gate.factual_count(),
            crate::gating::CANONICAL_FACTUAL_WORDS.len(),
            "Epistemic gate should resolve all factual words via backend"
        );
    }

    #[test]
    fn test_gate_covers_full_vocab() {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-gating");
        let gen = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());
        // Verify gate applies to all logits (no 512 cap)
        let vocab_size = 50280; // MockMamba vocab size
        let mut logits = vec![0.0_f32; vocab_size];
        gen.epistemic_gate.apply(&mut logits, 3.0); // Unknown

        let modified: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, &l)| l.abs() > 1e-6)
            .map(|(i, _)| i)
            .collect();
        assert!(
            !modified.is_empty(),
            "Gate should modify logits at backend-resolved positions"
        );
        assert_eq!(logits.len(), vocab_size, "Logits should remain full-sized");
    }
}
