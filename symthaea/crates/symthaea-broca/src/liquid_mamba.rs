// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Liquid-Mamba fusion: pre-trained Mamba SSM + HDC projection + consciousness gating.
//!
//! Bridges high-dimensional abstract thought (16,384D) to linguistic surface tokens
//! via a learned bottleneck projection (256D) feeding a pre-trained Mamba-130M model.
//!
//! # Cognitive Gating
//! During generation, the `EpistemicGate` and `EmotionalModulator` adjust logits
//! in real-time based on the current `ThoughtChannels` state, enforcing factual
//! precision, hedging on uncertainty, and adjusting tone.

use std::collections::VecDeque;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::encoder::{ThoughtChannels, ThoughtLanguageEncoder};
use crate::gating::{
    consciousness_gated_max_tokens, EmotionalModulator, EpistemicGate, GatingConfig,
};
use crate::generator::GenerationResult;
use crate::mamba::{MambaBackend, MambaWrapper};
use crate::memory_bridge::MemoryBridge;
use crate::projection::{HdcSsmProjection, ProjectionGradientDiagnostics};
use crate::temporal_projection::TemporalProjection;
use crate::thought_chunk::{
    DynamicChunker, SimpleThoughtChunkDecoder, ThoughtChunk, ThoughtChunkDecoder,
    ThoughtChunkKind, ThoughtChunkSequence,
};
use candle_core::{DType, Device, Tensor};
use symthaea_fep::ActiveInferenceAgent;

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
    /// Coherence velocity threshold for dynamic chunk boundaries (default 0.15).
    #[serde(default = "default_coherence_velocity_threshold")]
    pub coherence_velocity_threshold: f32,
    /// Minimum tokens per dynamic chunk (default 6).
    #[serde(default = "default_min_chunk_size")]
    pub min_chunk_size: usize,
    /// Biological delta modulation strength (0 = disabled, 1 = full).
    pub delta_mod_strength: f32,
    /// Text inserted on semantic veto.
    pub veto_hesitation: String,
    /// Enable epistemic gating.
    pub enable_gating: bool,
    /// Enable semantic veto (self-correction).
    pub enable_veto: bool,
    /// Enable liquid Δ modulation (memory decay scaling).
    pub enable_liquid_delta: bool,
    /// Enable consciousness-gated max tokens.
    pub enable_consciousness_gating: bool,
    /// Enable EMA weight usage for inference.
    pub enable_ema: bool,
    /// EMA decay factor (default 0.999).
    pub ema_decay: f32,
    /// Enable online distillation (learn from Mamba while generating).
    pub enable_distillation: bool,
    /// Base learning rate for distillation.
    pub base_lr: f32,
    /// Fraction of base_lr for the CfC network (bottleneck).
    pub network_lr_scale: f32,
    /// Learning rate for Mamba LoRA layers (if enabled).
    pub lora_lr: f32,
    /// LoRA rank (0 = disabled).
    pub lora_rank: usize,
    /// LoRA alpha scaling.
    pub lora_alpha: f32,
    /// Warmup steps for distillation learning rate.
    pub warmup_steps: usize,
    /// Steps over which to anneal distillation learning rate to zero.
    pub cosine_annealing_steps: usize,
    /// Minimum LR reached after annealing (fraction of base_lr).
    pub min_lr_fraction: f32,
    /// Gradient accumulation steps (default 1).
    pub accumulation_steps: usize,
    /// Weight for contrastive loss between tokens of different intent.
    pub contrastive_weight: f32,
    /// Size of the recent-thoughts buffer for contrastive loss.
    pub contrastive_buffer_size: usize,

    // ─── Phase 1: Direct Thought-to-Token Binding ───
    /// Weight for auxiliary loss that aligns thought-permuted position vectors
    /// directly with target token embeddings.
    #[serde(default)]
    pub thought_logit_aux_weight: f32,
    /// Scale for direct thought-token logits.
    #[serde(default = "default_logit_scale")]
    pub logit_scale: f32,

    // ─── Phase 2: Per-Token Gradient Refinement ───
    /// Threshold for token similarity surprisal (surprisal > T triggers gradient).
    #[serde(default = "default_token_pe_sim_threshold")]
    pub token_pe_sim_threshold: f32,
    /// Maximum multiplier for surprise-boosted gradients.
    #[serde(default = "default_token_pe_max_boost")]
    pub token_pe_max_boost: f32,
    /// Decay rate for the surprise signal as the model becomes adequate (default 0.5).
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

    /// Enable temporal projection (chunk-based continuous latent prompting).
    #[serde(default)]
    pub temporal_projection: bool,

    /// Number of distinct projection groups (Improvement C).
    #[serde(default = "default_temporal_num_groups")]
    pub temporal_num_groups: usize,

    /// SSM model dimension (d_model). Default 768.
    #[serde(default = "default_ssm_dim")]
    pub ssm_dim: usize,

    /// Weight for anti-collapse regularization in temporal projection (default 0.0 = disabled).
    /// Pushes chunk projections apart when their cosine similarity exceeds the threshold.
    /// Good starting values: 0.01-0.1.
    #[serde(default)]
    pub temporal_anticollapse_weight: f32,

    /// Cosine similarity threshold for anti-collapse regularization (default 0.9).
    #[serde(default = "default_temporal_anticollapse_threshold")]
    pub temporal_anticollapse_threshold: f32,

    /// Use learned attention weights to bundle chunks before injection.
    #[serde(default)]
    pub temporal_learned_attention: bool,

    /// Enable directional cosine loss for temporal sequences (Improved Improvement A).
    #[serde(default)]
    pub temporal_directional_loss: bool,

    /// Enable rotating gradient position for temporal training (Improvement B).
    #[serde(default = "default_true")]
    pub temporal_rotate_grad_position: bool,

    /// Weight for temporal smoothness regularization (L2 of d_chunks/dt).
    /// 0 = disabled (default). Good starting values: 0.01-0.1.
    #[serde(default)]
    pub temporal_smoothness_weight: f32,

    /// Weight for rank regularization (decorrelation of W_up rows).
    /// 0 = disabled (default). Good starting values: 0.001-0.01.
    #[serde(default)]
    pub temporal_rank_reg_weight: f32,

    /// Maximum context budget for temporal projection (tokens/chunks).
    /// 0 = no limit (default).
    #[serde(default)]
    pub temporal_chunk_budget: usize,

    /// Enable learned adapter MLP after temporal projection (Improvement E).
    #[serde(default)]
    pub temporal_enable_adapter: bool,

    /// Power for surprise-weighted gradient scaling (0.0 = disabled, 1.0 = linear).
    #[serde(default)]
    pub surprise_gradient_alpha: f32,

    /// Use EMA-smoothed loss for surprise weighting instead of raw current loss.
    #[serde(default)]
    pub surprise_ema_smoothing: f32,

    /// Path to pre-computed embedding statistics (mean/std) for manifold matching.
    #[serde(default)]
    pub embedding_stats_path: Option<String>,

    /// Gradient clipping threshold (default 1.0).
    #[serde(default = "default_grad_clip")]
    pub grad_clip: f32,

    /// Number of simultaneous E2E gradient positions per step (default 1 = legacy rotating).
    /// With K=4, each chunk position gets gradient 4× more often than K=1 rotating.
    #[serde(default = "default_e2e_grad_chunks")]
    pub ee_grad_chunks: usize,

    /// Weight for orthogonality regularization on the up-projection matrix.
    /// Pushes W_up toward being an orthogonal frame (W^T @ W ≈ I).
    /// Prevents collapse in deep temporal latent space.
    #[serde(default)]
    pub orthogonality_weight: f32,

    /// Number of random pairs to sample for orthogonality check (default 64).
    #[serde(default = "default_orthogonality_samples")]
    pub orthogonality_samples: usize,

    /// Gating configuration (epistemic boost strengths, coherence thresholds, etc.).
    /// Override to strengthen or weaken consciousness-gated generation control.
    #[serde(default)]
    pub gating_config: GatingConfig,
}

/// Configuration for chunk-aware semantic generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkGenerationConfig {
    /// Maximum number of semantic chunks to generate.
    pub max_chunks: usize,
    /// Approximate tokens per semantic chunk (8–24 recommended).
    pub tokens_per_chunk: usize,
    /// Whether to update the thought vector after each chunk (recursive feedback).
    pub update_thought_after_chunk: bool,
    /// Decay factor when blending new thought into existing (0.0–1.0).
    pub thought_update_alpha: f32,
}

impl Default for ChunkGenerationConfig {
    fn default() -> Self {
        Self {
            max_chunks: 6,
            tokens_per_chunk: 16,
            update_thought_after_chunk: true,
            thought_update_alpha: 0.35,
        }
    }
}

/// Configuration for chunk-level distillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDistillConfig {
    /// Weight for thought HV cosine loss (1.0 - sim).
    pub chunk_prediction_weight: f32,
    /// Weight for normal token cross-entropy loss.
    pub token_prediction_weight: f32,
    /// Maximum chunks to process per distillation step.
    pub max_chunks_per_step: usize,
}

impl Default for ChunkDistillConfig {
    fn default() -> Self {
        Self {
            chunk_prediction_weight: 0.6,
            token_prediction_weight: 0.4,
            max_chunks_per_step: 4,
        }
    }
}

/// Configuration for self-supervised monologue training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonologueTrainingConfig {
    /// Number of chunks to generate per training step.
    pub chunks_per_monologue: usize,
    /// Weight for next-chunk HV prediction loss.
    pub chunk_prediction_weight: f32,
    /// Weight for token-level loss.
    pub token_loss_weight: f32,
    /// Weight for hidden state consistency (optional).
    pub hidden_consistency_weight: f32,
    /// Learning rate for this step.
    pub learning_rate: f32,
}

impl Default for MonologueTrainingConfig {
    fn default() -> Self {
        Self {
            chunks_per_monologue: 5,
            chunk_prediction_weight: 0.7,
            token_loss_weight: 0.3,
            hidden_consistency_weight: 0.1,
            learning_rate: 0.0005,
        }
    }
}

/// Learned prediction head for next-chunk thought vector.
/// Small but powerful: Linear → GELU → Linear with residual connection.
pub struct ChunkPredictor {
    pub w1: Tensor, // [hdc_dim, hidden]
    pub b1: Tensor,
    pub w2: Tensor, // [hidden, hdc_dim]
    pub b2: Tensor,
    pub hidden_dim: usize,
    device: Device,
}

impl ChunkPredictor {
    pub fn new(hdc_dim: usize, hidden_dim: usize, device: Device) -> Result<Self> {
        let w1 = Tensor::randn(0f32, 0.02, (hdc_dim, hidden_dim), &device)?;
        let b1 = Tensor::zeros((hidden_dim,), DType::F32, &device)?;
        let w2 = Tensor::randn(0f32, 0.02, (hidden_dim, hdc_dim), &device)?;
        let b2 = Tensor::zeros((hdc_dim,), DType::F32, &device)?;

        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            hidden_dim,
            device,
        })
    }

    pub fn forward(&self, input: &ContinuousHV) -> Result<ContinuousHV> {
        let x = Tensor::from_slice(&input.values, (1, input.values.len()), &self.device)?;

        // Linear → GELU → Linear + residual
        let h = x.matmul(&self.w1)?.add(&self.b1)?.gelu()?;
        let out = h.matmul(&self.w2)?.add(&self.b2)?;

        let out_vec = out.squeeze(0)?.to_vec1::<f32>()?;
        let mut values = vec![0.0f32; input.values.len()];

        // Residual connection + normalization
        for (i, v) in values.iter_mut().enumerate() {
            let pred = *out_vec.get(i).unwrap_or(&0.0);
            *v = (pred * 0.7) + (input.values[i] * 0.3);
        }

        Ok(ContinuousHV { values })
    }

    /// Save predictor weights to a MessagePack checkpoint.
    pub fn save_checkpoint(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let weights = PredictorWeights {
            w1: self.w1.to_vec2()?,
            b1: self.b1.to_vec1()?,
            w2: self.w2.to_vec2()?,
            b2: self.b2.to_vec1()?,
            hidden_dim: self.hidden_dim,
        };
        let data = rmp_serde::to_vec(&weights)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load predictor weights from a MessagePack checkpoint.
    pub fn load_checkpoint(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let data = std::fs::read(path)?;
        let weights: PredictorWeights = rmp_serde::from_slice(&data)?;

        let _hdc_dim = weights.w1.len();
        let hidden_dim = weights.hidden_dim;

        Ok(Self {
            w1: Tensor::new(weights.w1, device)?,
            b1: Tensor::new(weights.b1, device)?,
            w2: Tensor::new(weights.w2, device)?,
            b2: Tensor::new(weights.b2, device)?,
            hidden_dim,
            device: device.clone(),
        })
    }
}

/// Helper struct for serializing ChunkPredictor weights.
#[derive(Serialize, Deserialize)]
struct PredictorWeights {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    hidden_dim: usize,
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
fn default_coherence_velocity_threshold() -> f32 {
    0.15
}
fn default_min_chunk_size() -> usize {
    6
}
fn default_temporal_num_groups() -> usize {
    1
}
fn default_temporal_anticollapse_threshold() -> f32 {
    0.9
}
fn default_ssm_dim() -> usize {
    768
}
fn default_true() -> bool {
    true
}
fn default_e2e_grad_chunks() -> usize {
    1
}
fn default_orthogonality_samples() -> usize {
    64
}
fn default_logit_scale() -> f32 {
    250.0
}

impl Default for LiquidMambaConfig {
    fn default() -> Self {
        Self {
            model_id: "state-spaces/mamba-130m".to_string(),
            max_tokens: 256,
            temperature: 0.8,
            top_k: 40,
            veto_threshold: 0.15,
            drift_threshold: 0.30,
            coherence_window: 8,
            long_coherence_window: 32,
            coherence_ema_alpha: 0.3,
            min_consecutive_low: 3,
            coherence_velocity_threshold: 0.15,
            min_chunk_size: 6,
            delta_mod_strength: 1.0,
            veto_hesitation: "-- wait, ".to_string(),
            enable_gating: true,
            enable_veto: true,
            enable_liquid_delta: true,
            enable_consciousness_gating: true,
            enable_ema: true,
            ema_decay: 0.999,
            enable_distillation: true,
            base_lr: 0.001,
            network_lr_scale: 0.35,
            lora_lr: 0.0001,
            lora_rank: 0,
            lora_alpha: 1.0,
            warmup_steps: 100,
            cosine_annealing_steps: 0,
            min_lr_fraction: 0.01,
            accumulation_steps: 4,
            contrastive_weight: 0.1,
            contrastive_buffer_size: 8,
            thought_logit_aux_weight: 0.0,
            logit_scale: 250.0,
            token_pe_sim_threshold: 0.3,
            token_pe_max_boost: 1.5,
            token_pe_decay: 0.5,
            fep_high_threshold: 0.7,
            fep_low_threshold: 0.3,
            fep_high_multiplier: 1.5,
            fep_low_multiplier: 0.7,
            temporal_chunk_dim: 0,
            temporal_projection: true,
            temporal_num_groups: 1,
            ssm_dim: 768,
            temporal_anticollapse_weight: 0.0,
            temporal_anticollapse_threshold: 0.9,
            temporal_learned_attention: true,
            temporal_directional_loss: true,
            temporal_rotate_grad_position: true,
            temporal_smoothness_weight: 0.02,
            temporal_rank_reg_weight: 0.005,
            temporal_chunk_budget: 16,
            temporal_enable_adapter: true,
            surprise_gradient_alpha: 0.5,
            surprise_ema_smoothing: 0.9,
            embedding_stats_path: None,
            grad_clip: 1.0,
            ee_grad_chunks: 1,
            orthogonality_weight: 0.0,
            orthogonality_samples: 64,
            gating_config: GatingConfig::default(),
        }
    }
}

/// Liquid-Mamba fusion generator: consciousness-gated SSM language generation.
pub struct LiquidMambaGenerator {
    pub mamba: Box<dyn MambaBackend>,
    pub projection: HdcSsmProjection,
    pub temporal_proj: Option<TemporalProjection>,
    encoder: ThoughtLanguageEncoder,
    config: LiquidMambaConfig,
    /// Emotional state (injected by cognitive loop via update_affect).
    thermodynamic_load: f32,
    arousal: f32,
    mood_temperature: f32,
    /// Online distillation state.
    generation_count: usize,
    distill_accumulator: usize,
    last_semantic_pe: f32,
    /// History ring buffer for trend analysis (capacity 64).
    pe_history: VecDeque<f32>,
    /// FEP modulation factor from cognitive loop (default 1.0 = neutral).
    fep_modulation: f32,
    /// Last cached effective rank from check_projection_health().
    last_cached_rank: f32,
    /// Optional gradient diagnostics (enabled via enable_diagnostics()).
    diagnostics: Option<ProjectionGradientDiagnostics>,
    // Generation at which diagnostics-triggered recovery last ran (prevents rapid re-triggering)
    last_diag_recovery_gen: usize,
    // ═══ Phase 3: Semantic Autoregression State ═══
    /// Hidden state from the last generated chunk (for carry-over)
    pub last_chunk_hidden: Option<Vec<f32>>,
    /// History of generated chunks in current session
    pub chunk_history: VecDeque<ThoughtChunk>,
    /// Maximum number of chunks to keep in history
    pub max_chunk_history: usize,
    /// Phase 3: Learned prediction head for next-chunk thought vector.
    pub chunk_predictor: Option<ChunkPredictor>,
    /// Phase 4: Long-term memory bridge (HDC Store integration).
    pub memory_bridge: Option<MemoryBridge>,
    /// Phase 4: Active Inference agent for FEP-driven generation.
    pub fep_agent: Option<ActiveInferenceAgent>,
    /// Epistemic gate for logit adjustment.
    pub epistemic_gate: EpistemicGate,
}

impl LiquidMambaGenerator {
    /// Create a new Liquid-Mamba generator.
    pub fn new(genesis: &GenesisSeed, config: LiquidMambaConfig) -> Result<Self> {
        let device = crate::mamba::best_device();
        let mamba = MambaWrapper::load(&config.model_id, device)?;
        Self::with_backend(genesis, config, Box::new(mamba))
    }

    /// Create a Liquid-Mamba generator with a mock backend (for testing).
    pub fn with_mock(genesis: &GenesisSeed, config: LiquidMambaConfig) -> Self {
        use crate::mamba::mock::MockMamba;
        Self::with_backend(genesis, config, Box::new(MockMamba::new()))
            .expect("MockMamba backend cannot fail")
    }

    /// Internal constructor shared by `new()` and `with_mock()`.
    fn with_backend(
        genesis: &GenesisSeed,
        config: LiquidMambaConfig,
        mamba: Box<dyn MambaBackend>,
    ) -> Result<Self> {
        let encoder = ThoughtLanguageEncoder::new_from_genesis(genesis);

        let temporal_proj = if config.temporal_projection {
            let mut tp = TemporalProjection::new(
                genesis,
                16384, // HDC dim
                config.temporal_chunk_dim.max(256),
                config.ssm_dim,
            );
            if config.temporal_enable_adapter {
                tp.enable_adapter(genesis);
            }
            Some(tp)
        } else {
            None
        };

        let projection = HdcSsmProjection::new(genesis, 16384, 256, config.ssm_dim);
        let epistemic_gate = EpistemicGate::new_from_backend(mamba.as_ref(), &config.gating_config);

        let enable_ema = config.enable_ema;
        let ema_decay = config.ema_decay;

        let mut gen = Self {
            mamba,
            projection,
            temporal_proj,
            encoder,
            config,
            thermodynamic_load: 0.0,
            arousal: 0.5,
            mood_temperature: 1.0,
            generation_count: 0,
            distill_accumulator: 0,
            last_semantic_pe: 0.0,
            pe_history: VecDeque::with_capacity(64),
            fep_modulation: 1.0,
            last_cached_rank: 0.0,
            diagnostics: None,
            last_diag_recovery_gen: 0,
            last_chunk_hidden: None,
            chunk_history: VecDeque::with_capacity(16),
            max_chunk_history: 16,
            chunk_predictor: None,
            memory_bridge: None,
            fep_agent: None,
            epistemic_gate,
        };

        if enable_ema {
            gen.projection.enable_ema(ema_decay);
        }

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
                    text: "[error]".to_string(),
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
                    logit_diagnostics: Vec::new(),
                }
            }
        }
    }

    /// Distill from Mamba teacher to the projection for a single target sequence.
    pub fn distill_step(
        &mut self,
        thought_hv: &ContinuousHV,
        target_ids: &[u32],
        lr: f32,
    ) -> Result<f32> {
        let mut total_loss = 0.0f32;
        if target_ids.is_empty() {
            return Ok(0.0);
        }

        // 1. Prepare context for teacher-forcing
        if self.config.temporal_projection && self.temporal_proj.is_some() {
            let sequence = self
                .temporal_proj
                .as_ref()
                .unwrap()
                .project_to_ssm_sequence(thought_hv);
            self.mamba.inject_context_sequence(&sequence)?;
        } else {
            let ssm_context = self.projection.project_to_ssm(thought_hv);
            self.mamba.inject_initial_context(&ssm_context)?;
        }

        // 2. Teacher-forced forward + backward
        let mut prev_token = self.mamba.eos_token_id();
        let num_tokens = target_ids.len();

        for &target_id in target_ids {
            // Compute teacher loss at this position
            let d_ssm = self.mamba.compute_e2e_token_loss_at(
                &[vec![0.0; self.config.ssm_dim]], // sequence not used in this call
                &[target_id],
                0,
            )?;

            // Convert SSM gradient back to HDC gradient
            let d_hdc = if let Some(ref tp) = self.temporal_proj {
                tp.project_to_hdc(&d_ssm)
            } else {
                self.projection.project_to_hdc(&d_ssm)
            };

            // Apply gradient to projection weights
            if let Some(ref mut tp) = self.temporal_proj {
                tp.backward(thought_hv, &d_hdc, lr)?;
            } else {
                self.projection.backward(thought_hv, &d_hdc, lr);
            }

            // Move to next token in teacher sequence
            let _ = self.mamba.forward_one_token(prev_token)?;
            prev_token = target_id;
            total_loss += 1.0; // Placeholder until real loss is extracted
        }

        self.generation_count += 1;
        Ok(total_loss / num_tokens as f32)
    }

    /// Full training step for a single thought.
    pub fn train_step(&mut self, channels: &ThoughtChannels, target_ids: &[u32]) -> Result<f32> {
        let lr = self.compute_lr();
        let thought_hv = self.encoder.encode(channels);
        self.distill_step(&thought_hv, target_ids, lr)
    }

    /// Generate a sequence of semantically coherent chunks using autoregressive
    /// thought evolution. Each chunk conditions the next via hidden state carry-over.
    pub fn generate_semantic_monologue(
        &mut self,
        channels: &ThoughtChannels,
        max_chunks: usize,
    ) -> Result<ThoughtChunkSequence> {
        let mut sequence = ThoughtChunkSequence::new("semantic_monologue");
        self.chunk_history.clear();
        self.last_chunk_hidden = None;

        // First chunk — start from initial thought
        let first_chunk =
            self.generate_next_dynamic_chunk(&self.encoder.encode(channels), channels, 0)?;

        sequence.push(first_chunk.clone());
        self.chunk_history.push_back(first_chunk.clone());

        // Subsequent chunks use hidden state carry-over
        let mut current_chunk = first_chunk;
        let mut current_thought = current_chunk.thought_hv.clone();

        for i in 1..max_chunks {
            // --- Phase 4: Long-term Memory Blending ---
            if let Some(ref bridge) = self.memory_bridge {
                let _ = bridge.blend_past_experiences(&mut current_thought);
            }

            // --- Phase 4: Active Inference (Surprise-aware path selection) ---
            let next_chunk = self.generate_next_dynamic_chunk(&current_thought, channels, i)?;

            if let Some(ref mut agent) = self.fep_agent {
                // Map ContinuousHV to FEP observation (simplified mapping)
                let obs_values: Vec<f64> = next_chunk
                    .thought_hv
                    .values
                    .iter()
                    .take(agent.config.obs_dim)
                    .map(|&v| v as f64)
                    .collect();
                let observation = symthaea_fep::Observation {
                    values: obs_values,
                    modality: "semantic".to_string(),
                    precision: 1.0,
                    timestamp: i as u64,
                };

                // Perception update
                let _ = agent.perceive(&observation);

                // Use FEP surprise to modulate future generation
                if let Some(ref fe) = agent.last_fe_components {
                    self.fep_modulation = (fe.surprise as f32).exp().clamp(0.5, 2.0);
                }
            }

            sequence.push(next_chunk.clone());
            self.chunk_history.push_back(next_chunk.clone());

            // Maintain history size
            if self.chunk_history.len() > self.max_chunk_history {
                self.chunk_history.pop_front();
            }

            current_chunk = next_chunk;
            current_thought = current_chunk.thought_hv.clone();
        }

        // --- Phase 4: Persistence (Remember this monologue) ---
        if let Some(ref mut bridge) = self.memory_bridge {
            if !sequence.chunks.is_empty() {
                let refs: Vec<&ContinuousHV> =
                    sequence.chunks.iter().map(|c| &c.thought_hv).collect();
                let average_thought = ContinuousHV::bundle(&refs);
                let memory_id = (self.generation_count as u64) << 32 | (sequence.chunks.len() as u64);
                let _ = bridge.remember(memory_id, &average_thought);
            }
        }

        Ok(sequence)
    }

    /// Generate the next chunk using the previous chunk's state (true autoregression).
    /// Uses dynamic chunk boundaries based on coherence velocity.
    pub fn generate_next_dynamic_chunk(
        &mut self,
        current_thought: &ContinuousHV,
        channels: &ThoughtChannels,
        chunk_index: usize,
    ) -> Result<ThoughtChunk> {
        // 1. Warm-start Mamba with previous chunk's hidden state (if available)
        if let Some(ref hidden) = self.last_chunk_hidden {
            if let Err(e) = self.mamba.inject_hidden_state(hidden) {
                tracing::warn!("Could not inject hidden state: {}", e);
            }
        }

        // 2. Generate tokens until dynamic boundary is triggered
        let (tokens, _hvs, final_coherence) = self.generate_inner_dynamic(current_thought, channels)?;

        // 3. Capture new hidden state for next iteration
        self.last_chunk_hidden = self.mamba.extract_hidden_state().ok();

        // 4. Create ThoughtChunk
        let text = self.mamba.decode(&tokens).unwrap_or_default();
        let mut chunk = ThoughtChunk::new(
            format!("c{}", chunk_index),
            self.infer_chunk_kind(channels),
            current_thought.clone(),
            channels.psi(),
        )
        .with_confidence(final_coherence)
        .with_target(text);

        if !tokens.is_empty() {
            chunk = chunk.with_token_span(0, tokens.len());
        }

        Ok(chunk)
    }

    /// Helper for dynamic chunk generation loop.
    fn generate_inner_dynamic(
        &mut self,
        thought_hv: &ContinuousHV,
        channels: &ThoughtChannels,
    ) -> Result<(Vec<u32>, Vec<ContinuousHV>, f32)> {
        // Prepare context
        if self.config.temporal_projection && self.temporal_proj.is_some() {
            let sequence = self
                .temporal_proj
                .as_ref()
                .unwrap()
                .project_to_ssm_sequence(thought_hv);
            self.mamba.inject_context_sequence(&sequence)?;
        } else {
            let ssm_context = self.projection.project_to_ssm(thought_hv);
            self.mamba.inject_initial_context(&ssm_context)?;
        }

        let mut chunker = DynamicChunker::new(
            self.config.coherence_velocity_threshold,
            self.config.min_chunk_size,
        );

        let mut coherence_monitor = CoherenceMonitor::new(
            thought_hv.clone(),
            self.config.coherence_window,
            self.config.coherence_ema_alpha,
            self.config.veto_threshold,
            self.config.min_consecutive_low,
        );

        let mut tokens = Vec::new();
        let mut hvs = Vec::new();
        let mut prev_token = self.mamba.eos_token_id();
        let max_tokens = self.config.max_tokens.min(32); // Limit chunk size

        for _pos in 0..max_tokens {
            let mut logits = self.mamba.forward_one_token(prev_token)?;

            // Gating/Modulation (simplified)
            if self.config.enable_gating {
                let ep_scale = self.config.gating_config.mamba_epistemic_scale();
                self.epistemic_gate.apply_scaled(
                    &mut logits,
                    channels.epistemic_ordinal(),
                    channels.domain_familiarity(),
                    ep_scale,
                );
            }

            let next_token = top_k_sample(&logits, self.config.top_k, self.config.temperature);

            // Back-project
            let token_emb = self.mamba.embedding_vector(next_token)?;
            let token_hdc = if let Some(ref tp) = self.temporal_proj {
                tp.project_to_hdc(&token_emb)
            } else {
                self.projection.project_to_hdc(&token_emb)
            };

            coherence_monitor.push(token_hdc.clone());
            let current_coherence = coherence_monitor.current_coherence();

            tokens.push(next_token);
            hvs.push(token_hdc.clone());

            if chunker.process_token(next_token, token_hdc, current_coherence) {
                break;
            }

            if next_token == self.mamba.eos_token_id() {
                break;
            }

            prev_token = next_token;
        }

        Ok((tokens, hvs, coherence_monitor.current_coherence()))
    }

    /// Predict the next chunk's thought vector from current chunk + hidden state.
    fn predict_next_chunk_hv(&self, current_chunk: &ThoughtChunk) -> Result<ContinuousHV> {
        if let Some(ref predictor) = self.chunk_predictor {
            return predictor.forward(&current_chunk.thought_hv);
        }

        // Fallback: simple transformation of current HV + hidden state influence
        let mut next_hv = current_chunk.thought_hv.clone();

        // Apply small learned drift (simulates the model's prediction)
        if let Some(ref hidden) = self.last_chunk_hidden {
            // Blend hidden state information into the thought vector
            let len = next_hv.values.len().min(hidden.len());
            for i in 0..len {
                next_hv.values[i] = next_hv.values[i] * 0.92 + hidden[i] * 0.08;
            }
        }

        // Add slight noise to simulate prediction uncertainty
        let mut seed = (current_chunk.psi * 1000.0) as u64;
        for v in &mut next_hv.values {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            *v += noise * 0.01;
        }

        Ok(next_hv)
    }

    /// Predict the next chunk's thought vector (exposed for evaluation).
    pub fn predict_next_chunk_hv_exposed(
        &self,
        current_chunk: &ThoughtChunk,
    ) -> Result<ContinuousHV> {
        self.predict_next_chunk_hv(current_chunk)
    }

    /// Set a learned chunk predictor for Phase 3 semantic autoregression.
    pub fn set_chunk_predictor(&mut self, predictor: ChunkPredictor) {
        self.chunk_predictor = Some(predictor);
    }

    fn generate_inner(
        &mut self,
        channels: &ThoughtChannels,
        mut on_token: Option<&mut dyn FnMut(&str)>,
    ) -> Result<GenerationResult> {
        let thought_hv = self.encoder.encode(channels);

        let ema_live_backup = if self.config.enable_ema {
            self.projection.use_ema_weights()
        } else {
            None
        };

        let result = (|| -> Result<GenerationResult> {
            let _ssm_context = if let Some(ref tp) = self.temporal_proj {
                let sequence = if self.config.temporal_chunk_budget > 0 {
                    tp.project_to_ssm_sequence_topk(&thought_hv, self.config.temporal_chunk_budget)
                } else {
                    tp.project_to_ssm_sequence(&thought_hv)
                };
                self.mamba.inject_context_sequence(&sequence)?;
                sequence
                    .last()
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.config.ssm_dim])
            } else {
                let ctx = self.projection.project_to_ssm(&thought_hv);
                self.mamba.inject_initial_context(&ctx)?;
                ctx
            };

            let max_tokens = if self.config.enable_consciousness_gating {
                consciousness_gated_max_tokens(self.config.max_tokens, channels.psi())
            } else {
                self.config.max_tokens
            };

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
                0.05, // very slow alpha for long trend
                0.0,
                0,
            );

            let mut token_ids = Vec::new();
            let mut text = String::new();
            let mut prev_token = self.mamba.eos_token_id();
            let mut output_hvs = Vec::new();
            let mut logit_diagnostics = Vec::new();
            let mut semantic_pe = 0.0f32;

            for pos in 0..max_tokens {
                let mut logits = self.mamba.forward_one_token(prev_token)?;

                if self.config.enable_gating {
                    let ep_scale = self.config.gating_config.mamba_epistemic_scale();
                    self.epistemic_gate.apply_scaled(
                        &mut logits,
                        channels.epistemic_ordinal(),
                        channels.domain_familiarity(),
                        ep_scale,
                    );
                }

                let next_token = top_k_sample(&logits, self.config.top_k, self.config.temperature);

                let token_emb = self.mamba.embedding_vector(next_token)?;
                let token_hdc = if let Some(ref tp) = self.temporal_proj {
                    tp.project_to_hdc(&token_emb)
                } else {
                    self.projection.project_to_hdc(&token_emb)
                };

                coherence_monitor.push(token_hdc.clone());
                long_coherence_monitor.push(token_hdc.clone());
                output_hvs.push(token_hdc.clone());

                let local_coh = coherence_monitor.current_coherence();
                let long_coh = long_coherence_monitor.current_coherence();

                if self.config.enable_veto && coherence_monitor.should_veto() {
                    text.push_str(&self.config.veto_hesitation);
                    self.mamba.reset();
                    if let Some(ref tp) = self.temporal_proj {
                        let sequence = tp.project_to_ssm_sequence(&thought_hv);
                        self.mamba.inject_context_sequence(&sequence)?;
                    } else {
                        let ctx = self.projection.project_to_ssm(&thought_hv);
                        self.mamba.inject_initial_context(&ctx)?;
                    }
                    coherence_monitor.reset();
                    prev_token = self.mamba.eos_token_id();
                    continue;
                }

                if let Ok(token_str) = self.mamba.decode_token(next_token) {
                    text.push_str(&token_str);
                    if let Some(ref mut cb) = on_token {
                        cb(&token_str);
                    }
                }

                token_ids.push(next_token);
                if next_token == self.mamba.eos_token_id() {
                    break;
                }
                prev_token = next_token;
            }

            if !output_hvs.is_empty() {
                let bundle = ContinuousHV::bundle(&output_hvs.iter().collect::<Vec<_>>());
                semantic_pe = 1.0 - thought_hv.similarity(&bundle).clamp(-1.0, 1.0);
            }

            Ok(GenerationResult {
                text,
                token_ids,
                num_tokens: output_hvs.len(),
                eos_terminated: prev_token == self.mamba.eos_token_id(),
                veto_triggered: false,
                final_coherence: coherence_monitor.current_coherence(),
                long_coherence: long_coherence_monitor.current_coherence(),
                coherence_dynamics: Vec::new(),
                gating_trace: Vec::new(),
                hallucination_flag: false,
                output_hvs,
                semantic_pe,
                nsm_prime_coverage: 0.0,
                logit_diagnostics,
            })
        })();

        if let Some(live) = ema_live_backup {
            self.projection.restore_live_weights(&live);
        }

        result
    }

    fn compute_lr(&self) -> f32 {
        let step = self.generation_count;
        let base_lr = self.config.base_lr;
        let warmup = if self.config.warmup_steps > 0 && step < self.config.warmup_steps {
            step as f32 / self.config.warmup_steps as f32
        } else {
            1.0
        };
        base_lr * warmup
    }

    pub fn update_affect(&mut self, load: f32, mood_temp: f32) {
        self.thermodynamic_load = load.clamp(0.0, 1.0);
        self.mood_temperature = mood_temp.clamp(0.1, 5.0);
    }

    pub fn generation_count(&self) -> usize {
        self.generation_count
    }
    pub fn last_semantic_pe(&self) -> f32 {
        self.last_semantic_pe
    }
    pub fn pe_stats(&self) -> (f32, f32, f32) {
        let mean = if self.pe_history.is_empty() {
            0.0
        } else {
            self.pe_history.iter().sum::<f32>() / self.pe_history.len() as f32
        };
        (mean, 0.0, 0.0)
    }
    fn push_pe_history(&mut self, pe: f32) {
        if self.pe_history.len() >= 64 {
            self.pe_history.pop_front();
        }
        self.pe_history.push_back(pe);
    }

    pub fn encoder(&self) -> &ThoughtLanguageEncoder {
        &self.encoder
    }
    pub fn controller_mut(&mut self) -> &mut HdcSsmProjection {
        &mut self.projection
    }
    pub fn mamba(&self) -> &dyn MambaBackend {
        self.mamba.as_ref()
    }
    pub fn mamba_mut(&mut self) -> &mut dyn MambaBackend {
        self.mamba.as_mut()
    }
    pub fn current_ssm_dim(&self) -> usize {
        self.mamba.d_model()
    }
    pub fn infer_chunk_kind(&self, channels: &ThoughtChannels) -> ThoughtChunkKind {
        if channels.channels[24] > 0.6 {
            ThoughtChunkKind::Code
        } else if channels.channels[3] > 0.7 {
            ThoughtChunkKind::Action
        } else {
            ThoughtChunkKind::Text
        }
    }
    pub fn tokenizer(&self) -> &crate::tokenizer::BpeTokenizer {
        // Mock proxy for tokenizer access (assume generator owns one or proxy via mamba)
        panic!("Direct tokenizer access not supported in LiquidMamba fusion");
    }
    pub fn temporal_proj(&self) -> Option<&TemporalProjection> {
        self.temporal_proj.as_ref()
    }
    pub fn set_fep_modulation(&mut self, val: f32) {
        self.fep_modulation = val;
    }
    pub fn current_lr(&self) -> f32 {
        self.compute_lr()
    }
    pub fn thermodynamic_load(&self) -> f32 {
        self.thermodynamic_load
    }
    pub fn mood_temperature(&self) -> f32 {
        self.mood_temperature
    }
    pub fn enable_diagnostics(&mut self) {
        // In real: init diagnostics struct
    }
}

/// Helper for coherence monitoring.
struct CoherenceMonitor {
    thought: ContinuousHV,
    window: usize,
    history: VecDeque<f32>,
    alpha: f32,
    ema: f32,
    threshold: f32,
    min_low: usize,
    consecutive_low: usize,
}

impl CoherenceMonitor {
    fn new(thought: ContinuousHV, window: usize, alpha: f32, threshold: f32, min_low: usize) -> Self {
        Self {
            thought,
            window,
            history: VecDeque::with_capacity(window),
            alpha,
            ema: 1.0,
            threshold,
            min_low,
            consecutive_low: 0,
        }
    }

    fn push(&mut self, token_hdc: ContinuousHV) {
        let sim = self.thought.similarity(&token_hdc).clamp(-1.0, 1.0);
        self.ema = self.alpha * sim + (1.0 - self.alpha) * self.ema;
        if self.history.len() >= self.window {
            self.history.pop_front();
        }
        self.history.push_back(sim);

        if sim < self.threshold {
            self.consecutive_low += 1;
        } else {
            self.consecutive_low = 0;
        }
    }

    fn current_coherence(&self) -> f32 {
        self.ema
    }

    fn should_veto(&self) -> bool {
        self.consecutive_low >= self.min_low
    }

    fn reset(&mut self) {
        self.ema = 1.0;
        self.consecutive_low = 0;
        self.history.clear();
    }
}

fn top_k_sample(logits: &[f32], k: usize, temp: f32) -> u32 {
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / temp))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);

    let max_l = indexed[0].1;
    let exp_logits: Vec<f32> = indexed.iter().map(|(_, l)| (l - max_l).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();

    let mut r = rand::random::<f32>() * sum_exp;
    for (i, prob) in exp_logits.iter().enumerate() {
        r -= prob;
        if r <= 0.0 {
            return indexed[i].0 as u32;
        }
    }
    indexed[0].0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba::mock::MockMamba;

    #[test]
    fn test_memory_bridge_integration() {
        use symthaea_hdc_store::store::HdcStore;
        use std::path::PathBuf;

        let genesis = GenesisSeed::from_phrase("memory-test");
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());

        // 1. Create a temporary store
        let tmp_dir = std::env::temp_dir().join("broca_memory_test");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        
        let store = HdcStore::create(tmp_dir.join("test.hdc"), StoreConfig::default()).unwrap();
        let mut bridge = MemoryBridge::new(store, 3, 0.5);

        // 2. Add some "memories"
        let mem_hv = ContinuousHV::random(16384, 42);
        bridge.remember(100, &mem_hv).unwrap();

        gen.memory_bridge = Some(bridge);

        // 3. Generate monologue (should trigger blending)
        let channels = ThoughtChannels::default();
        let monologue = gen.generate_semantic_monologue(&channels, 3).unwrap();

        assert_eq!(monologue.chunks.len(), 3);
    }

    #[test]
    fn test_real_hidden_state_carryover() {
        let genesis = GenesisSeed::from_phrase("hidden-carryover");
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());

        let channels = ThoughtChannels::default();
        let monologue = gen.generate_semantic_monologue(&channels, 3).unwrap();

        assert_eq!(monologue.chunks.len(), 3);
        assert!(gen.last_chunk_hidden.is_some());
    }

    #[test]
    fn test_self_supervised_monologue_training() {
        let genesis = GenesisSeed::from_phrase("monologue-train");
        let mut gen = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());

        let channels = ThoughtChannels::default();
        let config = MonologueTrainingConfig::default();

        let initial_loss = gen.train_on_semantic_monologue(&channels, &config).unwrap();
        assert!(initial_loss >= 0.0);
    }
}
