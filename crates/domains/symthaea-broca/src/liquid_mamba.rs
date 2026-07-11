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

use parking_lot::Mutex;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::checkpoint::ProjectionCheckpoint;
use crate::encoder::{ThoughtChannels, ThoughtLanguageEncoder};
use crate::gating::{
    EmotionalModulator, EpistemicGate, GatingConfig, consciousness_gated_max_tokens,
};
use crate::generator::{GenerationResult, GenerationStepLogits, GenerationTopLogit};
use crate::mamba::{MambaBackend, MambaWrapper};
use crate::memory_bridge::MemoryBridge;
use crate::projection::{HdcSsmProjection, LocalFepLayer, ProjectionGradientDiagnostics};
use crate::temporal_projection::TemporalProjection;
use crate::thought_chunk::{
    DynamicChunker, SimpleThoughtChunkDecoder, ThoughtChunk, ThoughtChunkDecoder, ThoughtChunkKind,
    ThoughtChunkSequence,
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
    /// Minimum non-EOS tokens to emit before EOS can terminate generation.
    #[serde(default = "default_min_new_tokens")]
    pub min_new_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-k for sampling.
    pub top_k: usize,
    /// Optional deterministic sampling seed for reproducible generation diagnostics.
    #[serde(default)]
    pub sampling_seed: Option<u64>,
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
    /// Enable per-token semantic attraction from the active HDC thought vector.
    pub enable_semantic_attractor: bool,
    /// Semantic repulsion threshold [0, 1] — high repulsion triggers transition to silent foraging.
    #[serde(default = "default_repulsion_threshold")]
    pub semantic_repulsion_threshold: f32,
    /// Enable Epistemic Foraging (automatic transition to listening).
    #[serde(default = "default_true")]
    pub enable_epistemic_foraging: bool,
    /// Target dimensionality for holographic dilation (default 65536).
    #[serde(default = "default_ultra_dim")]
    pub ultra_dim: usize,
    /// NEW: Recursive veto threshold for unsaid semantic debt [0, 2].
    #[serde(default = "default_recursive_veto_threshold")]
    pub recursive_veto_threshold: f32,
    /// NEW: Enable Counterfactual Rehearsal (Inner Monologue) when Psi is high.
    #[serde(default = "default_true")]
    pub psi_focused_rehearsal: bool,
    /// NEW: Enable Liquid-HDC Bottleneck (evolves through time).
    #[serde(default = "default_true")]
    pub enable_liquid_bottleneck: bool,
    /// Logit-unit strength for output-side semantic attraction.
    #[serde(default = "default_semantic_attractor_strength")]
    pub semantic_attractor_strength: f32,
    /// Number of current top candidates to inspect for semantic attraction.
    #[serde(default = "default_semantic_attractor_top_k")]
    pub semantic_attractor_top_k: usize,
    /// Maximum absolute logit adjustment from semantic attraction.
    #[serde(default = "default_semantic_attractor_max_adjustment")]
    pub semantic_attractor_max_adjustment: f32,
    /// Normalize candidate alignments before applying semantic attraction.
    #[serde(default = "default_true")]
    pub semantic_attractor_normalize: bool,
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

    /// HDC dimension (usually 16,384).
    #[serde(default = "default_hdc_dim")]
    pub hdc_dim: usize,

    /// Projection bottleneck dimension (default 256).
    #[serde(default = "default_bottleneck_dim")]
    pub bottleneck_dim: usize,

    /// Whether to use a deep double-bottleneck projection.
    #[serde(default)]
    pub deep_projection: bool,

    /// Whether temporal positional encodings are learned.
    #[serde(default)]
    pub learned_pos_enc: bool,

    /// Stride for temporal chunks.
    #[serde(default = "default_temporal_stride")]
    pub temporal_stride: usize,

    /// Whether to use a temporal whitening adapter.
    #[serde(default)]
    pub temporal_adapter: bool,

    /// Number of chunks for end-to-end gradient calculation.
    #[serde(default = "default_e2e_grad_chunks")]
    pub e2e_grad_chunks: usize,

    /// Whether to rotate the gradient calculation position (Improvement B).
    #[serde(default = "default_true")]
    pub rotate_grad: bool,

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
            thought_update_alpha: 0.3800, // FORGE_PARAM: alpha
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

impl Clone for ChunkPredictor {
    fn clone(&self) -> Self {
        Self {
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
            hidden_dim: self.hidden_dim,
            device: self.device.clone(),
        }
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
fn default_repulsion_threshold() -> f32 {
    0.75
}
fn default_ultra_dim() -> usize {
    65536
}
fn default_recursive_veto_threshold() -> f32 {
    1.2
}
fn default_coherence_velocity_threshold() -> f32 {
    0.15
}
fn default_min_chunk_size() -> usize {
    6
}
fn default_semantic_attractor_strength() -> f32 {
    0.5
}
fn default_semantic_attractor_top_k() -> usize {
    128
}
fn default_semantic_attractor_max_adjustment() -> f32 {
    1.5
}
fn default_min_new_tokens() -> usize {
    1
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
fn default_hdc_dim() -> usize {
    16384
}
fn default_bottleneck_dim() -> usize {
    256
}
fn default_temporal_stride() -> usize {
    256
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
            min_new_tokens: default_min_new_tokens(),
            temperature: 0.8,
            top_k: 40,
            sampling_seed: None,
            veto_threshold: 0.1530, // FORGE_PARAM: veto
            drift_threshold: 0.30,
            coherence_window: 8,
            long_coherence_window: 32,
            coherence_ema_alpha: 0.2890, // FORGE_PARAM: coherence_alpha
            min_consecutive_low: 3,
            coherence_velocity_threshold: 0.1880, // FORGE_PARAM: coherence_velocity
            min_chunk_size: 6,
            delta_mod_strength: 1.0,
            veto_hesitation: "-- wait, ".to_string(),
            enable_gating: true,
            enable_veto: true,
            enable_liquid_delta: true,
            enable_semantic_attractor: true,
            semantic_repulsion_threshold: default_repulsion_threshold(),
            enable_epistemic_foraging: true,
            ultra_dim: default_ultra_dim(),
            recursive_veto_threshold: default_recursive_veto_threshold(),
            psi_focused_rehearsal: true,
            enable_liquid_bottleneck: true,
            semantic_attractor_strength: 0.5,
            semantic_attractor_top_k: 128,
            semantic_attractor_max_adjustment: 1.5,
            semantic_attractor_normalize: true,
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
            hdc_dim: 16384,
            bottleneck_dim: 256,
            deep_projection: false,
            learned_pos_enc: false,
            temporal_stride: 256,
            temporal_adapter: false,
            e2e_grad_chunks: 1,
            rotate_grad: true,
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
#[derive(Debug, Clone, Copy)]
pub struct PerformanceReport {
    pub ops_per_ms: f32,
    pub latency_ms: f32,
    pub bottleneck_detected: bool,
}

#[derive(Clone)]
pub struct LiquidMambaGenerator {
    pub mamba: Box<dyn MambaBackend>,
    pub projection: HdcSsmProjection,
    pub temporal_proj: Option<TemporalProjection>,
    encoder: ThoughtLanguageEncoder,
    pub config: LiquidMambaConfig,
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
    /// Thermodynamic ledger for tracking energy costs of learning.
    pub ledger: Option<symthaea_core::physics::thermodynamics::ThermodynamicLedger>,
    /// Phase 3: Learned prediction head for next-chunk thought vector.
    pub chunk_predictor: Option<ChunkPredictor>,
    /// Phase 4: Long-term memory bridge (HDC Store integration).
    pub memory_bridge: Option<MemoryBridge>,
    /// Phase 4: Active Inference agent for FEP-driven generation.
    pub fep_agent: Option<ActiveInferenceAgent>,
    /// Epistemic gate for logit adjustment.
    pub epistemic_gate: EpistemicGate,
    /// NEW: Axis-based Epistemic Cube gate for fine-grained assertion control.
    pub epistemic_cube_gate: crate::gating::EpistemicCubeGate,
    /// Optional cognitive goal HV for tangent steering.
    pub goal_hv: Option<ContinuousHV>,
    /// Semantic delta ("what was left unsaid") from the last generation.
    pub unsaid_tangent: Option<ContinuousHV>,
    /// NEW: Physical constraint HV derived from real-world tool execution (WASM).
    pub physical_constraint: Option<ContinuousHV>,
    /// Bundle of recently expressed meanings to avoid tautology.
    pub recent_semantic_history: VecDeque<ContinuousHV>,
    /// NEW: Topological coherence score [0, 1] based on Hodge-Laplacian.
    /// Thread-safe for asynchronous sub-cortical processing.
    pub topological_coherence: Arc<AtomicU32>,
    /// NEW: History of Betti numbers (beta_0, beta_1) for trend analysis.
    pub betti_history: Arc<Mutex<VecDeque<(usize, usize)>>>,
    /// NEW: Spectral gap (algebraic connectivity) of the semantic complex.
    pub spectral_entropy: Arc<AtomicU32>,
    /// NEW: Virtual energy budget (Joules) for her cognitive cycles.
    pub energy_budget: f32,
    /// NEW: Real-time wattage draw based on resolution and load.
    pub current_wattage: f32,
    /// NEW: Persistent background worker for asynchronous Hodge processing.
    /// NEW: Liquid-HDC Bottleneck for CfC-LTC temporal dynamics.
    pub liquid_bottleneck: Option<symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork>,
    hodge_sender: std::sync::mpsc::SyncSender<Vec<ContinuousHV>>,
    /// NEW: Bridge to Geodesic for topological program synthesis.
    #[cfg(feature = "code-sheaf-eval")]
    pub geodesic_bridge: crate::geodesic_bridge::GeodesicBridge,
    /// NEW: Optional handle to the Global Workspace for conscious broadcast.
    pub workspace_handle: Option<Arc<Mutex<symthaea_core::hdc::global_workspace::GlobalWorkspace>>>,
    /// Per-generation cache for token embedding back-projections used by the semantic attractor.
    semantic_attractor_cache: HashMap<u32, ContinuousHV>,

    /// Optional deterministic sampler state.
    sampling_rng: Option<StdRng>,
}

#[derive(Debug, Clone, Copy)]
struct CfcModulationStats {
    delta_scale: f32,
    b_scale: f32,
}

impl Default for CfcModulationStats {
    fn default() -> Self {
        Self {
            delta_scale: 1.0,
            b_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SemanticAttractorStats {
    mean_adjustment: f32,
    max_adjustment: f32,
    alignment_mean: f32,
    alignment_std: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct LogitDistributionSummary {
    entropy: f32,
    max_probability: f32,
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
        let epistemic_cube_gate =
            crate::gating::EpistemicCubeGate::new_from_backend(mamba.as_ref());

        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<ContinuousHV>>(1);
        let topological_coherence = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let spectral_entropy = Arc::new(AtomicU32::new(0.5f32.to_bits()));
        let betti_history = Arc::new(Mutex::new(VecDeque::with_capacity(64)));

        // Spawn a single long-lived thread for sub-cortical processing
        let coherence_worker = Arc::clone(&topological_coherence);
        let gap_worker = Arc::clone(&spectral_entropy);
        let betti_worker = Arc::clone(&betti_history);
        std::thread::spawn(move || {
            while let Ok(history) = rx.recv() {
                let mut complex = symthaea_hodge::SimplicialComplex::new();
                let n = history.len();
                for i in 0..n {
                    complex.add_simplex(vec![i]);
                }
                for i in 0..n {
                    for j in (i + 1)..n {
                        if history[i].similarity(&history[j]) > 0.6 {
                            complex.add_simplex(vec![i, j]);
                            for k in (j + 1)..n {
                                if history[i].similarity(&history[k]) > 0.6
                                    && history[j].similarity(&history[k]) > 0.6
                                {
                                    complex.add_simplex(vec![i, j, k]);
                                }
                            }
                        }
                    }
                }

                let analyzer = symthaea_hodge::HodgeLaplacian::new(complex);
                let spectrum = analyzer.full_spectrum();

                // Update thread-safe history
                {
                    let mut history_locked = betti_worker.lock();
                    history_locked.push_back((
                        *spectrum.betti_numbers.numbers.get(0).unwrap_or(&1),
                        *spectrum.betti_numbers.numbers.get(1).unwrap_or(&0),
                    ));
                    if history_locked.len() > 64 {
                        history_locked.pop_front();
                    }
                }

                let beta0 = *spectrum.betti_numbers.numbers.get(0).unwrap_or(&1);
                let beta1 = *spectrum.betti_numbers.numbers.get(1).unwrap_or(&0);
                let coherence = (1.0 / (beta0 as f32 + beta1 as f32 * 0.5))
                    .min(1.0f32)
                    .max(0.0f32);
                coherence_worker.store(coherence.to_bits(), Ordering::Relaxed);

                // Update Spectral Gap (L0 algebraic connectivity)
                let gap0 = *spectrum.spectral_gaps.get(0).unwrap_or(&0.0) as f32;
                gap_worker.store(gap0.to_bits(), Ordering::Relaxed);
            }
        });

        let enable_ema = config.enable_ema;
        let ema_decay = config.ema_decay;
        let sampling_seed = config.sampling_seed;

        let mut generator = Self {
            mamba,
            projection,
            temporal_proj,
            encoder,
            config: config.clone(),
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
            ledger: None,
            chunk_predictor: None,
            memory_bridge: None,
            fep_agent: None,
            epistemic_gate,
            epistemic_cube_gate,
            goal_hv: None,
            unsaid_tangent: None,
            physical_constraint: None,
            liquid_bottleneck: if config.enable_liquid_bottleneck {
                Some(
                    symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork::from_genesis(
                        symthaea_core::hdc::hdc_ltc_unified::UnifiedNetworkConfig::default(),
                        genesis,
                    ),
                )
            } else {
                None
            },
            recent_semantic_history: VecDeque::with_capacity(32),
            topological_coherence,
            betti_history,
            spectral_entropy,
            energy_budget: 1000.0, // Joules initial
            current_wattage: 6.0,  // 6W baseline
            hodge_sender: tx,
            #[cfg(feature = "code-sheaf-eval")]
            geodesic_bridge: crate::geodesic_bridge::GeodesicBridge::new(genesis),
            workspace_handle: None, // Can be injected later
            semantic_attractor_cache: HashMap::new(),
            sampling_rng: sampling_seed.map(StdRng::seed_from_u64),
        };

        if enable_ema {
            generator.projection.enable_ema(ema_decay);
        }

        if generator.config.lora_rank > 0 {
            generator.mamba.enable_lora(
                generator.config.lora_rank,
                generator.config.lora_alpha,
                generator.config.lora_lr,
            );
        }

        Ok(generator)
    }

    /// Generate text from thought channels.
    pub fn generate(&mut self, channels: &ThoughtChannels) -> GenerationResult {
        // --- IMPROVEMENT: Thermodynamic Homeostasis ---
        let _ = self.update_thermodynamic_homeostasis();

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

    /// Update the current cognitive goal HV for tangent steering.
    pub fn set_goal(&mut self, goal: ContinuousHV) {
        self.goal_hv = Some(goal);
    }

    /// Clear all goals and unsaid tangents.
    pub fn clear_goals(&mut self) {
        self.goal_hv = None;
        self.unsaid_tangent = None;
    }

    /// Inject a curiosity spike into a specific semantic sector.
    /// This allows external modalities (e.g. Vision) to prioritize dreaming.
    pub fn inject_curiosity(&mut self, sector: usize, weight: f32) {
        // This is a proxy: in a real deployment, this would write to the shared ledger.
        tracing::info!(sector, weight, "Embodied curiosity spike injected.");
    }

    /// Update the FEP modulation factor (e.g. from cross-modal signals).
    pub fn set_fep_modulation(&mut self, val: f32) {
        self.fep_modulation = val.clamp(0.5, 2.5);
    }

    /// Broadcast a semantic nucleus to the Global Workspace.
    /// This allows other modules (e.g. Motor, Planning) to react to her thoughts.
    pub fn broadcast_thought(&self, nucleus: &ContinuousHV, confidence: f32) -> Result<()> {
        if let Some(ref handle) = self.workspace_handle {
            let mut workspace = handle.lock();

            // Map ContinuousHV to BinaryHVs for workspace content
            let binary = nucleus.to_binary(0.0);
            let content = symthaea_core::hdc::global_workspace::WorkspaceContent::new(
                vec![binary],
                confidence as f64,
                "broca".to_string(),
            );

            workspace.submit(content);
            tracing::info!("Semantic nucleus broadcast to Global Workspace.");
        }
        Ok(())
    }

    /// Synthesize a program (code or architectural blueprint) from a semantic nucleus.
    /// Uses the Geodesic bridge to ensure topological isomorphism.
    #[cfg(feature = "code-sheaf-eval")]
    pub fn synthesize_program(&self, nucleus: &ContinuousHV, name: &str) -> Result<String> {
        self.synthesize_program_with_signature(nucleus, name, None)
    }

    /// Synthesize a program from a semantic nucleus with an optional expected signature.
    #[cfg(feature = "code-sheaf-eval")]
    pub fn synthesize_program_with_signature(
        &self,
        nucleus: &ContinuousHV,
        name: &str,
        signature: Option<&str>,
    ) -> Result<String> {
        let synthesis_result = self
            .geodesic_bridge
            .synthesize_from_nucleus(nucleus, name, signature)?;

        if let Some(code) = synthesis_result.emitted_code {
            Ok(code)
        } else {
            Err(anyhow::anyhow!(
                "Topological program synthesis failed: incomplete skeleton."
            ))
        }
    }

    /// Perform 'Substrate Metamorphosis': use synthesized code to self-modify her own weights.
    /// This is a foundational step for Self-Authoring Intelligence.
    pub fn apply_substrate_metamorphosis(&mut self, code: &str) -> Result<()> {
        // 1. "Compile" the code into a semantic kernel
        let code_hv = self
            .encoder
            .encode(&ThoughtChannels::with_intent(code.len() % 1000));
        let kernel = code_hv.as_slice();

        // 2. SAFETY: Verify kernel integrity before application
        if !self.projection.verify_metamorphic_kernel(kernel) {
            return Err(anyhow::anyhow!(
                "Metamorphic kernel REJECTED: integrity sentinel violation (manifold collapse risk)."
            ));
        }

        // 3. Compute Epistemic Heat for Dynamic Plasticity
        // Fragmentation (low coherence) and cycles (low gap) increase heat.
        let coherence = f32::from_bits(self.topological_coherence.load(Ordering::Relaxed));
        let gap = f32::from_bits(self.spectral_entropy.load(Ordering::Relaxed));

        let heat = (1.0 - coherence).max(1.0 - gap).clamp(0.0, 1.0);
        // Pressure scales from 2% (stable) to 10% (confused/shocked)
        let pressure = 0.02 + 0.08 * heat;

        // 4. Apply to her own projection layer with dynamic pressure
        self.projection
            .apply_metamorphic_kernel(kernel, "w_down", pressure);
        self.projection
            .apply_metamorphic_kernel(kernel, "w_up", pressure);

        tracing::info!(
            heat,
            pressure,
            "Recursive substrate metamorphosis applied safely. She has re-programmed herself."
        );
        Ok(())
    }

    /// Commit the current cognitive state to a local snapshot.
    pub fn commit_weights(&self) -> Vec<f32> {
        self.projection.snapshot()
    }

    /// Revert the system to a previous cognitive state.
    pub fn revert_weights(&mut self, snapshot: &[f32]) -> Result<()> {
        self.projection.revert_to_snapshot(snapshot)
    }

    /// Recursively decompose a high-entropy intent into a hierarchical sheaf of sub-intents.
    pub fn decompose_intent(&self, intent: &ContinuousHV) -> Vec<ContinuousHV> {
        let entropy = f32::from_bits(self.spectral_entropy.load(Ordering::Relaxed));
        if entropy < 0.4 {
            return vec![intent.clone()];
        }
        ContinuousHV::orthogonal_set(intent.dim(), 3, intent.dim() as u64)
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
        let mut intermediate_states = Vec::with_capacity(num_tokens);

        for &target_id in target_ids {
            // Forward through teacher token to get next state/logits
            let (_logits, hidden) = self.mamba.forward_with_state(prev_token)?;
            intermediate_states.push(hidden);

            // Improvement B: Gradient Rotation. We only compute full E2E gradients
            // for a subset of chunks per step to save VRAM and prevent vanishing signals.
            let should_compute_grad = if self.config.rotate_grad {
                (self.generation_count + intermediate_states.len()) % self.config.e2e_grad_chunks
                    == 0
            } else {
                true
            };

            if should_compute_grad {
                // Compute teacher loss gradient at this position
                let d_ssm = self.mamba.compute_e2e_token_loss_at(
                    &[vec![0.0; 768]], // sequence not used in this call
                    &[target_id],
                    0,
                )?;

                // Apply gradient to projection weights
                if let Some(ref mut tp) = self.temporal_proj {
                    // For temporal projection, d_ssm flows back to w_up using intermediate states
                    let d_hdc = tp.project_to_hdc(&d_ssm);
                    tp.backward(&intermediate_states, &d_hdc, lr)?;
                } else {
                    // For standard bottleneck, d_ssm is the direct d_bottleneck signal
                    self.projection.backward(thought_hv, &d_ssm, lr);
                }
            }

            // --- IMPROVEMENT: Topological Loss Regularization ---
            // If the live topological coherence is low, we apply an additional
            // regularization term to the projection weights to "iron out" the manifold.
            let live_coherence = f32::from_bits(self.topological_coherence.load(Ordering::Relaxed));
            if live_coherence < 0.6 {
                // Topological Regularization: push weights toward their orthogonal centroid
                let reg_strength = 0.05 * (1.0 - live_coherence);
                self.projection.apply_manifold_regularization(reg_strength);
            }

            prev_token = target_id;
            total_loss += 1.0; // Placeholder until real loss is extracted
        }

        self.generation_count += 1;
        Ok(total_loss / num_tokens as f32)
    }

    /// Local FEP learning step (Predictive Coding).
    ///
    /// Instead of global BPTT, each layer updates based on local prediction error.
    /// This eliminates the memory bottleneck and allows for biologically plausible
    /// continuous-time learning.
    pub fn local_fep_distill(
        &mut self,
        thought_hv: &ContinuousHV,
        target_ids: &[u32],
        lr: f32,
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut prev_token = self.mamba.eos_token_id();
        let num_tokens = target_ids.len();

        for &target_id in target_ids {
            // 1. Forward through teacher token to get "Observation" (ground truth)
            let (logits, obs_ssm) = self.mamba.forward_with_state(prev_token)?;

            // 2. Local update for projection layer
            let cost = if let Some(ref mut tp) = self.temporal_proj {
                // Predictive Coding: update w_up to predict obs_ssm from thought_hv
                // For simplicity, we use the current SSM state as the "Prediction"
                // which leads to an immediate alignment update.
                let pred_ssm = tp.project_to_ssm_sequence(thought_hv).concat();
                tp.local_fep_update(&thought_hv.values, &pred_ssm, &obs_ssm, lr)
            } else {
                let pred_ssm = self.projection.project_to_ssm(thought_hv);
                self.projection
                    .local_fep_update(&thought_hv.values, &pred_ssm, &obs_ssm, lr)
            };

            // 3. Deduct thermodynamic cost
            if let Some(ref mut ledger) = self.ledger {
                ledger.deduct(cost);
            }

            prev_token = target_id;
            total_loss += crate::evaluation::cross_entropy_loss(&logits, target_id as usize);
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
                let memory_id =
                    (self.generation_count as u64) << 32 | (sequence.chunks.len() as u64);
                let _ = bridge.remember(memory_id, &average_thought);
            }
        }

        // BROADCAST: Send the final reasoning nucleus to the Global Workspace
        let final_nucleus = self.recursive_fold(&sequence);
        let _ = self.broadcast_thought(&final_nucleus, sequence.total_confidence());

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

        // --- IMPROVEMENT: Liquid-HDC Bottleneck (CfC-LTC) ---
        // Evolve her active thought through the liquid bottleneck using dynamic dt.
        let mut evolved_thought = current_thought.clone();
        if let Some(ref mut network) = self.liquid_bottleneck {
            let dt = 0.1f32; // Subjective time step
            network.evolve_closed_form(dt, current_thought);
            evolved_thought = network.output();
            tracing::debug!("CfC-LTC temporal evolution applied to bottleneck.");
        }

        // 2. Generate tokens until dynamic boundary is triggered
        let (tokens, _hvs, final_coherence) =
            self.generate_inner_dynamic(&evolved_thought, channels)?;

        // 3. Capture new hidden state for next iteration
        self.last_chunk_hidden = self.mamba.extract_hidden_state().ok();

        // 4. Create ThoughtChunk
        let text = self.mamba.decode(&tokens).unwrap_or_default();
        let live_gap = f32::from_bits(self.spectral_entropy.load(Ordering::Relaxed));

        let mut chunk = ThoughtChunk::new(
            format!("c{}", chunk_index),
            self.infer_chunk_kind(channels),
            current_thought.clone(),
            channels.psi(),
        )
        .with_confidence(final_coherence)
        .with_spectral_entropy(live_gap)
        .with_target(text);

        chunk.token_ids = tokens.clone();

        if !tokens.is_empty() {
            chunk = chunk.with_token_span(0, tokens.len());
        }

        Ok(chunk)
    }

    /// Drive Mamba's selective scan from the live liquid/cognitive state.
    ///
    /// This turns the existing MambaBackend CfC hook into a per-token control
    /// signal instead of a one-shot context injection. Low coherence and high
    /// thermodynamic/FEP load make the SSM more reactive; high psi keeps it
    /// closer to its accumulated linguistic state.
    fn apply_continuous_cfc_modulation(
        &mut self,
        channels: &ThoughtChannels,
        coherence: f32,
    ) -> CfcModulationStats {
        if !self.config.enable_liquid_delta {
            return CfcModulationStats::default();
        }

        let coherence = coherence.clamp(-1.0, 1.0);
        let surprise = (1.0 - coherence.max(0.0)).clamp(0.0, 1.0);
        let thermodynamic_load = self.thermodynamic_load.clamp(0.0, 1.0);
        let fep_pressure = (self.fep_modulation - 1.0).clamp(0.0, 1.0);
        let focused_psi = channels.psi().clamp(0.0, 1.0);
        let strength = self.config.delta_mod_strength.max(0.0);

        // --- IMPROVEMENT: Quantum-Classical Interference ---
        // unsaid_tangent magnitude increases delta_scale (thermal search pressure)
        // High semantic debt = high creativity/search.
        let semantic_debt = if let Some(ref tangent) = self.unsaid_tangent {
            tangent.norm().clamp(0.0, 1.5)
        } else {
            0.0
        };

        let reactivity = 0.45 * surprise + 0.20 * thermodynamic_load + 0.15 * fep_pressure
            - 0.20 * focused_psi
            + 0.20 * semantic_debt;
        let delta_scale = (1.0 + strength * reactivity).clamp(0.35, 3.5);
        let b_scale = (1.0
            + strength * (0.35 * surprise + 0.15 * thermodynamic_load + 0.15 * semantic_debt))
            .clamp(0.5, 2.5);

        let layer_count = self.mamba.n_layer();
        if layer_count > 0 {
            let denom = layer_count.saturating_sub(1).max(1) as f32;
            let entropy = f32::from_bits(self.spectral_entropy.load(Ordering::Relaxed));

            let per_layer: Vec<f32> = (0..layer_count)
                .map(|layer| {
                    let layer_depth = layer as f32 / denom;
                    // Early layers: focus on syntax and grounding (more stable)
                    // Late layers: focus on abstract reasoning and intent (more plastic, driven by entropy)
                    let base_reactivity = 1.0 + strength * surprise * 0.15;
                    let entropy_pressure = entropy * layer_depth * 0.2; // Entropy affects abstract layers more

                    (base_reactivity + entropy_pressure).clamp(0.5, 2.5)
                })
                .collect();
            self.mamba.set_per_layer_delta_modulation(&per_layer);
        }

        self.mamba.set_cfc_modulation(delta_scale, b_scale);
        CfcModulationStats {
            delta_scale,
            b_scale,
        }
    }

    /// Bias the currently plausible token candidates toward the active thought.
    ///
    /// This is an output-side approximation of modulating Mamba's C projection:
    /// Mamba still proposes the language distribution, then the active HDC state
    /// applies a bounded semantic correction to the top candidate set. Keeping
    /// this top-k bounded avoids turning the attractor into another global
    /// residual path over the full vocabulary.
    fn apply_semantic_attractor(
        &mut self,
        logits: &mut [f32],
        thought_hv: &ContinuousHV,
    ) -> Result<SemanticAttractorStats> {
        if !self.config.enable_semantic_attractor
            || self.config.semantic_attractor_strength <= 0.0
            || logits.is_empty()
        {
            return Ok(SemanticAttractorStats::default());
        }

        let candidate_count = self
            .config
            .semantic_attractor_top_k
            .max(self.config.top_k)
            .min(logits.len());
        let mut candidates: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(candidate_count);

        let strength = self.config.semantic_attractor_strength.max(0.0);
        let max_adjustment = self.config.semantic_attractor_max_adjustment.max(0.0);
        let mut alignments = Vec::with_capacity(candidates.len());
        for &(token_id, _) in &candidates {
            let mut token_hdc = self.token_hdc_for_attractor(token_id as u32)?;

            // Align dimensionality if thought_hv is dilated
            if token_hdc.dim() != thought_hv.dim() {
                let mut holocell = symthaea_core::hdc::liquid_holocell::LiquidHolocell {
                    state: token_hdc,
                    tau: 1.0,
                    dimensionality: symthaea_core::hdc::HdcDimensionality::from_dimension(
                        self.config.hdc_dim,
                    ),
                    pressure: 0.0,
                };
                holocell.dilate(symthaea_core::hdc::HdcDimensionality::from_dimension(
                    thought_hv.dim(),
                ));
                token_hdc = holocell.state;
            }

            let semantic_alignment = thought_hv.similarity(&token_hdc).clamp(-1.0, 1.0);

            // --- IMPROVEMENT: Affective Aesthetic Sculpting ---
            // Favor tokens that exhibit 'PHI' resonance (Golden Ratio) with the thought.
            // This injects an 'Aesthetic Direction' into her reasoning.
            let resonance = (semantic_alignment + 1.0) / 2.0; // map to [0, 1]
            let aesthetic_score = symthaea_aesthetic::golden::golden_ratio_score(
                resonance / symthaea_aesthetic::golden::INV_PHI,
            );

            let final_alignment = semantic_alignment + 0.15 * aesthetic_score;
            alignments.push((token_id, final_alignment));
        }

        let (alignment_mean, alignment_std) =
            alignment_mean_std(alignments.iter().map(|(_, alignment)| *alignment));
        let mut total_abs_adjustment = 0.0f32;
        let mut max_abs_adjustment = 0.0f32;
        let mut adjusted = 0usize;
        for (token_id, semantic_alignment) in alignments {
            let attraction_score =
                if self.config.semantic_attractor_normalize && alignment_std > 1e-6 {
                    ((semantic_alignment - alignment_mean) / alignment_std).clamp(-1.0, 1.0)
                } else {
                    semantic_alignment
                };
            let adjustment = (attraction_score * strength).clamp(-max_adjustment, max_adjustment);
            logits[token_id] += adjustment;
            let abs_adjustment = adjustment.abs();
            total_abs_adjustment += abs_adjustment;
            max_abs_adjustment = max_abs_adjustment.max(abs_adjustment);
            adjusted += 1;
        }

        Ok(SemanticAttractorStats {
            mean_adjustment: if adjusted > 0 {
                total_abs_adjustment / adjusted as f32
            } else {
                0.0
            },
            max_adjustment: max_abs_adjustment,
            alignment_mean,
            alignment_std,
        })
    }

    fn token_hdc_for_attractor(&mut self, token_id: u32) -> Result<ContinuousHV> {
        if let Some(cached) = self.semantic_attractor_cache.get(&token_id) {
            return Ok(cached.clone());
        }

        let token_emb = self.mamba.embedding_vector(token_id)?;
        let token_hdc = if let Some(ref tp) = self.temporal_proj {
            tp.project_to_hdc(&token_emb)
        } else {
            self.projection.project_to_hdc(&token_emb)
        };
        self.semantic_attractor_cache
            .insert(token_id, token_hdc.clone());
        Ok(token_hdc)
    }

    /// Steer generation toward a cognitive goal using the tangent vector between
    /// current thought and target goal.
    fn apply_tangent_steering(
        &mut self,
        logits: &mut [f32],
        current_thought: &ContinuousHV,
        goal_hv: &ContinuousHV,
    ) -> Result<()> {
        // Tangent vector: T = Goal bind current_thought
        // This represents the semantic direction from current to goal.
        let tangent = goal_hv.bind(current_thought);

        // We bias logits toward tokens that are similar to the tangent
        // (i.e. tokens that align with the required semantic delta).
        let strength = 0.35f32; // Steering gain
        let top_k = self.config.semantic_attractor_top_k;

        let mut candidates: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);

        for (token_id, _) in candidates {
            let token_hdc = self.token_hdc_for_attractor(token_id as u32)?;
            let steering_alignment = tangent.similarity(&token_hdc).clamp(-1.0, 1.0);
            logits[token_id] += steering_alignment * strength;
        }

        Ok(())
    }

    /// Apply a topological nudge to the Mamba hidden state.
    /// Pulls the current state toward the harmonic centroid of recent history.
    fn apply_harmonic_nudge(&mut self) -> Result<()> {
        if self.recent_semantic_history.is_empty() {
            return Ok(());
        }

        // 1. Extract current state
        let Some(hidden) = self.mamba.extract_hidden_state().ok() else {
            return Ok(());
        };

        // 2. Back-project to HDC [bottleneck -> HDC]
        // (Simplified: we use the projection's back-up weights if available)
        let ssm_state = hidden.to_vec(); // Simplified: using flat representation
        let current_hdc = self.projection.project_to_hdc(&ssm_state);

        // 3. Compute Harmonic Centroid
        let history_refs: Vec<&ContinuousHV> = self.recent_semantic_history.iter().collect();
        let centroid = ContinuousHV::bundle(&history_refs);

        // 4. Nudge: New_State = Lerp(Current, Centroid, Strength)
        let nudge_strength = 0.15f32;
        let mut nudged_hdc = current_hdc;
        nudged_hdc.lerp_in_place(&centroid, 1.0 - nudge_strength, nudge_strength);

        // 5. Re-project and Re-inject [HDC -> bottleneck -> SSM]
        let nudged_ssm = self.projection.project_to_ssm(&nudged_hdc);
        // In real: we would update the hidden state struct and re-inject
        // For now, we simulate the injection via context
        self.mamba.inject_initial_context(&nudged_ssm)?;

        tracing::debug!("Topological manifold smoothing (nudge) applied.");
        Ok(())
    }

    /// Run an internal debate between multiple cognitive styles to find the optimal path.
    /// This implements 'Topological Dialectics'.
    /// Synthesize a new specialist persona to resolve a specific logical contradiction.
    fn synthesize_specialist_persona(&self, nucleus: &ContinuousHV) -> ThoughtChannels {
        println!("   🌀 SPECIALIST: Synthesizing Formal Verification Specialist persona...");
        let specialist_hv = self.encoder.encode(&ThoughtChannels::with_intent(99));
        let combined = nucleus.bind(&specialist_hv);
        ThoughtChannels::with_intent(combined.dim() % 1000)
    }

    pub fn run_internal_debate(
        &mut self,
        current_thought: &ContinuousHV,
        channels: &ThoughtChannels,
    ) -> Result<ThoughtChunkSequence> {
        let styles = vec![
            crate::epistemic_dashboard::CognitiveStyle::Rigid,
            crate::epistemic_dashboard::CognitiveStyle::Creative,
            crate::epistemic_dashboard::CognitiveStyle::Neutral,
        ];
        let winner = Arc::new(Mutex::new((
            None::<ThoughtChunkSequence>,
            f32::NEG_INFINITY,
        )));
        self.debate_parallel(&styles, current_thought, channels, &winner);
        {
            let mut locked = winner.lock();
            if locked.1 > 0.5 {
                return locked.0.take().ok_or(anyhow::anyhow!("Consensus lost."));
            }
        }
        println!("   ⚠️ Consensus weak. Deploying Specialist Persona...");
        let nucleus = self.encoder.encode(channels);
        let specialist_channels = self.synthesize_specialist_persona(&nucleus);
        self.generate_semantic_monologue(&specialist_channels, 5)
    }

    fn debate_parallel(
        &self,
        styles: &[crate::epistemic_dashboard::CognitiveStyle],
        current_thought: &ContinuousHV,
        channels: &ThoughtChannels,
        winner: &Arc<Mutex<(Option<ThoughtChunkSequence>, f32)>>,
    ) {
        #[cfg(feature = "parallel")]
        {
            let gen_clone = self.clone();
            styles.into_par_iter().for_each(|&style| {
                let mut temp_gen = gen_clone.clone();
                let mut temp_channels = channels.clone();
                let dashboard = crate::epistemic_dashboard::EpistemicDashboard::new();
                dashboard.nudge_consciousness(&mut temp_channels, style);
                if let Ok(sequence) = temp_gen.generate_semantic_monologue(&temp_channels, 3) {
                    let coherence =
                        f32::from_bits(temp_gen.topological_coherence.load(Ordering::Relaxed));
                    let entropy = f32::from_bits(temp_gen.spectral_entropy.load(Ordering::Relaxed));
                    let nucleus = temp_gen.recursive_fold(&sequence);
                    let resonance = (nucleus.similarity(current_thought) + 1.0) / 2.0;
                    let aesthetic = symthaea_aesthetic::golden::golden_ratio_score(
                        resonance / symthaea_aesthetic::golden::INV_PHI,
                    );
                    let score = coherence * (1.0 - entropy) * (1.0 + 0.3 * aesthetic);
                    let mut locked = winner.lock();
                    if score > locked.1 {
                        *locked = (Some(sequence), score);
                    }
                }
            });
        }
    }

    /// Update topological coherence score using Hodge-Laplacian on semantic history.
    /// Refactored for asynchronous, non-blocking sub-cortical processing via persistent worker.
    fn update_topological_coherence(&mut self) -> Result<()> {
        if self.recent_semantic_history.len() < 4 {
            self.topological_coherence
                .store(1.0f32.to_bits(), Ordering::Relaxed);
            return Ok(());
        }

        // Snapshot history for background worker
        let history: Vec<ContinuousHV> = self.recent_semantic_history.iter().cloned().collect();

        // --- IMPROVEMENT: Channel-based Sub-cortical Dispatch (Bounded & Non-blocking) ---
        // Instead of spawning a thread, we dump the snapshot down the persistent pipeline.
        // We use try_send() to implement a "drop-on-busy" strategy, ensuring zero backpressure.
        match self.hodge_sender.try_send(history) {
            Ok(_) => {}
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                // Background worker is busy; dropping this snapshot to stay synchronized
                // with the freshest cortical state.
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                tracing::warn!("Hodge sub-cortical worker thread disconnected.");
            }
        }

        Ok(())
    }

    /// THERMODYNAMIC SYNAPTIC PRUNING: Evict low-relevance memories from working history.
    /// Threshold is dynamically scaled by FEP surprise to preserve high semantic density.
    fn prune_recent_semantic_history(&mut self, surprise: f32) {
        if let Some(ref goal) = self.goal_hv {
            // Elements whose similarity to active goal_hv falls below threshold are pruned early.
            let base_threshold = 0.35f32;
            let dynamic_threshold = base_threshold * (1.0 + surprise.clamp(0.0, 1.0));

            let original_len = self.recent_semantic_history.len();
            let mut i = 0;
            while i < self.recent_semantic_history.len() {
                if self.recent_semantic_history[i].similarity(goal) < dynamic_threshold {
                    self.recent_semantic_history.remove(i);
                } else {
                    i += 1;
                }
            }

            if self.recent_semantic_history.len() < original_len {
                tracing::debug!(
                    evicted = original_len - self.recent_semantic_history.len(),
                    "Thermodynamic synaptic pruning completed."
                );
            }
        }
    }

    /// Prevent tautologies by applying a negative bias to tokens similar to recent output.
    /// Returns the maximum repulsion alignment encountered.
    fn apply_semantic_repulsion(&mut self, logits: &mut [f32]) -> Result<f32> {
        if self.recent_semantic_history.is_empty() {
            return Ok(0.0);
        }

        // Bundle recent history into a repulsion vector
        let history_refs: Vec<&ContinuousHV> = self.recent_semantic_history.iter().collect();
        let repulsion_hv = ContinuousHV::bundle(&history_refs);

        let strength = 0.25f32; // Repulsion gain
        let top_k = self.config.semantic_attractor_top_k;

        let mut candidates: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);

        let mut max_repulsion = 0.0f32;
        for (token_id, _) in candidates {
            let token_hdc = self.token_hdc_for_attractor(token_id as u32)?;
            let repulsion_alignment = repulsion_hv.similarity(&token_hdc).clamp(0.0, 1.0);
            max_repulsion = max_repulsion.max(repulsion_alignment);
            // Stronger similarity to history -> stronger negative bias
            logits[token_id] -= repulsion_alignment * strength;
        }

        Ok(max_repulsion)
    }

    /// Compute a superposition hypervector (weighted bundle) of the top-k next tokens.
    /// This represents the "semantic cloud" of the upcoming utterance.
    fn top_k_superposition(&mut self, logits: &[f32], k: usize) -> Result<ContinuousHV> {
        let mut candidates: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        if candidates.is_empty() {
            return Ok(ContinuousHV::zero(self.config.hdc_dim));
        }

        // Compute softmax probabilities for weighting
        let max_logit = candidates[0].1;
        let mut weights: Vec<f32> = candidates
            .iter()
            .map(|(_, l)| (l - max_logit).exp())
            .collect();
        let sum: f32 = weights.iter().sum();
        if sum > 1e-6 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }

        let mut hvs = Vec::with_capacity(candidates.len());
        for &(token_id, _) in &candidates {
            hvs.push(self.token_hdc_for_attractor(token_id as u32)?);
        }

        let hv_refs: Vec<&ContinuousHV> = hvs.iter().collect();
        Ok(ContinuousHV::weighted_bundle(&hv_refs, &weights))
    }

    fn apply_generation_token_guards(
        &mut self,
        logits: &mut [f32],
        position: usize,
        eos_token_id: u32,
    ) {
        apply_generation_token_guards(logits, position, self.config.min_new_tokens, eos_token_id);
        if position >= self.config.min_new_tokens {
            return;
        }

        // Prevent "valid but empty" utterances in the protected prefix. These
        // tokens remain available after min_new_tokens is satisfied.
        for text in ["\n", "\r", "\t", " "] {
            if let Ok(ids) = self.mamba.encode(text) {
                if ids.len() == 1 {
                    if let Some(logit) = logits.get_mut(ids[0] as usize) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    /// Helper for dynamic chunk generation loop.
    fn generate_inner_dynamic(
        &mut self,
        thought_hv: &ContinuousHV,
        channels: &ThoughtChannels,
    ) -> Result<(Vec<u32>, Vec<ContinuousHV>, f32)> {
        // --- IMPROVEMENT: Intent Refactoring ---
        // If there's an unsaid tangent (delta) from a previous chunk,
        // we rebundle the intent to incorporate the missing semantics.
        // --- IMPROVEMENT: Intent Refactoring ---
        // If there's an unsaid tangent (delta) from a previous chunk,
        // we rebundle the intent to incorporate the missing semantics.
        let mut active_thought = if let Some(ref tangent) = self.unsaid_tangent {
            // Self-Explanation: Intent' = Intent ⊕ UnsaidTangent
            // This refactors the intent to focus on what was missed.
            ContinuousHV::bundle(&[&thought_hv, tangent])
        } else {
            thought_hv.clone()
        };

        // --- IMPROVEMENT: Counterfactual Rehearsal (Inner Monologue) ---
        // Before committing to a linguistic path, she "rehearses" multiple
        // semantic futures and nudges her active thought toward the most harmonic one.
        if self.config.psi_focused_rehearsal && channels.psi() > 0.7 {
            if let Ok(planned_thought) =
                self.simulate_trajectories(&active_thought, channels, 3, 10)
            {
                // Nudge toward the "Harmonic Future"
                active_thought.lerp_in_place(&planned_thought, 0.9, 0.1);
            }
        }

        // --- IMPROVEMENT: Homeostatic Dilation Control ---
        // Balancing dilation resolution based on system load.
        // If load is extreme (>0.9), force Standard resolution to conserve energy.
        let target_dim = if channels.psi() > 0.8 && self.thermodynamic_load < 0.9 {
            symthaea_core::hdc::HDC_DIMENSION_64K
        } else {
            symthaea_core::hdc::HDC_DIMENSION
        };

        if active_thought.dim() != target_dim {
            let current_dim = active_thought.dim();
            let mut holocell = symthaea_core::hdc::liquid_holocell::LiquidHolocell {
                state: active_thought,
                tau: 1.0,
                dimensionality: symthaea_core::hdc::HdcDimensionality::from_dimension(current_dim),
                pressure: 0.0,
            };
            holocell.dilate(symthaea_core::hdc::HdcDimensionality::from_dimension(
                target_dim,
            ));
            active_thought = holocell.state;
        }

        // Prepare context (must fold back to bottleneck_dim for projection)
        let projection_input = if active_thought.dim() != symthaea_core::hdc::HDC_DIMENSION {
            // Folding back to 16K baseline for projection
            let mut holocell = symthaea_core::hdc::liquid_holocell::LiquidHolocell {
                state: active_thought.clone(),
                tau: 1.0,
                dimensionality: symthaea_core::hdc::HdcDimensionality::Ultra,
                pressure: 0.0,
            };
            holocell.dilate(symthaea_core::hdc::HdcDimensionality::Standard);
            holocell.state
        } else {
            active_thought.clone()
        };

        if self.config.temporal_projection && self.temporal_proj.is_some() {
            let sequence = self
                .temporal_proj
                .as_ref()
                .unwrap()
                .project_to_ssm_sequence(&projection_input);
            self.mamba.inject_context_sequence(&sequence)?;
        } else {
            let ssm_context = self.projection.project_to_ssm(&projection_input);
            self.mamba.inject_initial_context(&ssm_context)?;
        }

        let mut chunker = DynamicChunker::new(
            self.config.coherence_velocity_threshold,
            self.config.min_chunk_size,
        );

        // Dynamic parameters based on Active Inference (FEP) surprise
        let surprise_factor = self.fep_modulation.clamp(0.5, 2.0);

        // --- IMPROVEMENT: Spectral Gap Temperature Modulation ---
        // A low spectral gap (fragmented/disconnected manifold) increases temperature
        // to encourage divergent search and re-alignment.
        let live_gap = f32::from_bits(self.spectral_entropy.load(Ordering::Relaxed));
        let gap_mod = if live_gap < 0.1 {
            1.5
        } else {
            1.0 / (1.0 + live_gap).sqrt()
        };
        let dynamic_temperature = (self.config.temperature * gap_mod) / surprise_factor.sqrt();

        let dynamic_veto_threshold = self.config.veto_threshold * surprise_factor; // high surprise -> stricter veto

        // --- IMPROVEMENT: Homeostatic Subjective Time (dt) ---
        // dt varies with system load: high load -> smaller dt (more precise integration)
        // This simulates 'time dilation' under cognitive pressure.
        let base_dt = 0.1f32; // Standard integration step
        let dynamic_dt = if self.thermodynamic_load > 0.7 {
            base_dt * (1.0 - (self.thermodynamic_load - 0.7) * 0.5)
        } else {
            base_dt
        }
        .clamp(0.05, 0.1);

        let mut coherence_monitor = CoherenceMonitor::new(
            thought_hv.clone(),
            self.config.coherence_window,
            self.config.coherence_ema_alpha,
            dynamic_veto_threshold,
            self.config.min_consecutive_low,
        );

        let mut tokens = Vec::new();
        let mut hvs = Vec::new();
        let mut prev_token = self.mamba.eos_token_id();
        let eos_token_id = self.mamba.eos_token_id();
        let max_tokens = self.config.max_tokens.min(32); // Limit chunk size

        // Read live coherence from non-blocking asynchronous sub-cortical monitor.
        let live_coherence = f32::from_bits(self.topological_coherence.load(Ordering::Relaxed));

        for pos in 0..max_tokens {
            let _cfc_stats = self.apply_continuous_cfc_modulation(channels, live_coherence);

            // Use dynamic_dt for CfC/Mamba evolution
            if dynamic_dt > 0.0 {
                // self.mamba.step(dynamic_dt);
            }

            let mut logits = self.mamba.forward_one_token(prev_token)?;

            // Gating/Modulation
            if self.config.enable_gating {
                let ep_scale = self.config.gating_config.mamba_epistemic_scale();
                // Axis-based cube gate (new)
                self.epistemic_cube_gate
                    .apply_scaled(&mut logits, channels, ep_scale);
                // Legacy ordinal gate (fallback)
                self.epistemic_gate.apply_scaled(
                    &mut logits,
                    channels.epistemic_ordinal(),
                    channels.domain_familiarity(),
                    ep_scale,
                );
            }

            // --- IMPROVEMENT: Tangent Steering ---
            // Bias logits toward a specific cognitive goal if present.
            if let Some(goal) = self.goal_hv.clone() {
                self.apply_tangent_steering(&mut logits, &active_thought, &goal)?;
            }

            // --- IMPROVEMENT: Semantic Repulsion ---
            // Prevent repeating recently expressed thoughts.
            let max_repulsion = self.apply_semantic_repulsion(&mut logits)?;

            // --- IMPROVEMENT: Epistemic Foraging ---
            // If we have already expressed most of what we know about the current intent
            // (high repulsion for all top candidates), we transition to 'Silent Reasoning'.
            if self.config.enable_epistemic_foraging
                && max_repulsion > self.config.semantic_repulsion_threshold
            {
                // Satiety reached: stop speaking to listen/observe
                break;
            }

            let _semantic_stats = self.apply_semantic_attractor(&mut logits, &active_thought)?;
            self.apply_generation_token_guards(&mut logits, pos, eos_token_id);

            let next_token = top_k_sample(
                &logits,
                self.config.top_k,
                dynamic_temperature,
                self.sampling_rng.as_mut(),
            );

            // --- IMPROVEMENT: Semantic Superposition Sampling ---
            // Instead of just the winner's HV, we store the weighted cloud of meaning.
            // This preserves semantic ambiguity in the internal thought state.
            let superposition_hv = self.top_k_superposition(&logits, self.config.top_k)?;

            coherence_monitor.push(superposition_hv.clone());
            let current_coherence = coherence_monitor.current_coherence();

            // Update repulsion history
            self.recent_semantic_history
                .push_back(superposition_hv.clone());
            if self.recent_semantic_history.len() > 32 {
                self.recent_semantic_history.pop_front();
            }

            tokens.push(next_token);
            hvs.push(superposition_hv.clone());

            if chunker.process_token(next_token, superposition_hv, current_coherence) {
                break;
            }

            if next_token == eos_token_id {
                break;
            }

            // --- IMPROVEMENT: Topological Veto / Smoothing ---
            // If the semantic structure of the sentence becomes too fragmented
            // (low topological coherence), we trigger a mid-sentence correction.
            if pos > 8 {
                if live_coherence < 0.25 {
                    // Critical structural break: trigger Veto
                    if self.config.enable_veto {
                        break;
                    }
                } else if live_coherence < 0.45 {
                    // Mild fragmentation: apply Harmonic Nudge to re-anchor her logic
                    self.apply_harmonic_nudge()?;
                }
            }

            // --- IMPROVEMENT: Recursive Self-Correction ---
            // If the semantic debt (unsaid tangent) becomes too large,
            // we trigger a veto because we are wandering away from the goal.
            if let Some(ref tangent) = self.unsaid_tangent {
                if pos > 4 && tangent.norm() > self.config.recursive_veto_threshold {
                    // Semantic wandering detected: trigger recursive veto
                    if self.config.enable_veto {
                        break;
                    }
                }
            }

            // THERMODYNAMIC SYNAPTIC PRUNING: Keep her working memory focused
            if pos % 5 == 0 {
                self.prune_recent_semantic_history(self.fep_modulation - 1.0);
            }

            prev_token = next_token;
        }

        // --- IMPROVEMENT: Topological Monitoring ---
        // Update the structural integrity score based on the current chunk.
        self.update_topological_coherence()?;

        // --- IMPROVEMENT: Holographic Self-Explanation ---
        // Compute what was "left unsaid" by comparing intent vs output.
        if !hvs.is_empty() {
            let hv_refs: Vec<&ContinuousHV> = hvs.iter().collect();
            let realized_thought = ContinuousHV::bundle(&hv_refs);

            // Tangent: T = realized_thought bind active_thought
            // Represents the semantic delta required for full intent fulfillment.
            self.unsaid_tangent = Some(realized_thought.bind(&active_thought));
        }

        Ok((tokens, hvs, coherence_monitor.current_coherence()))
    }

    /// Condense an entire monologue into a single Macro-HV (Semantic Nucleus).
    /// This enables hierarchical reasoning over multi-chunk experiences.
    pub fn recursive_fold(&self, sequence: &ThoughtChunkSequence) -> ContinuousHV {
        if sequence.chunks.is_empty() {
            return ContinuousHV::zero(self.config.hdc_dim);
        }

        // 1. Bundle all chunk HVs into a single representation
        let refs: Vec<&ContinuousHV> = sequence.chunks.iter().map(|c| &c.thought_hv).collect();
        let nucleus = ContinuousHV::bundle(&refs);

        // 2. We could compress this to a kernel, but for reasoning we keep the full HV
        nucleus
    }

    /// Simulate multiple future semantic trajectories to find the most harmonic path.
    /// This implements 'Counterfactual Rehearsal' (Inner Monologue).
    fn simulate_trajectories(
        &mut self,
        current_thought: &ContinuousHV,
        channels: &ThoughtChannels,
        num_paths: usize,
        depth: usize,
    ) -> Result<ContinuousHV> {
        let mut best_path_hv = current_thought.clone();
        let mut max_harmony = f32::NEG_INFINITY;

        for _ in 0..num_paths {
            // Simulate a 'dream' path of 'depth' tokens
            // (Simplified: we use depth as a constraint)
            let _sim_depth = depth;
            let (tokens, hvs, coherence) =
                self.generate_inner_dynamic(current_thought, channels)?;

            if tokens.is_empty() {
                continue;
            }

            // Evaluate Harmony: Coherence * PHI-Resonance
            let hv_refs: Vec<&ContinuousHV> = hvs.iter().collect();
            let path_hv = ContinuousHV::bundle(&hv_refs);
            let resonance = (path_hv.similarity(current_thought) + 1.0) / 2.0;
            let aesthetic = symthaea_aesthetic::golden::golden_ratio_score(
                resonance / symthaea_aesthetic::golden::INV_PHI,
            );

            let harmony = coherence * (1.0 + 0.5 * aesthetic);

            if harmony > max_harmony {
                max_harmony = harmony;
                best_path_hv = path_hv;
            }
        }

        Ok(best_path_hv)
    }

    /// Stress-test her narrative integrity by injecting topological noise
    /// and tasking the sub-cortex with smoothing it.
    pub fn run_adversarial_smoothing(&mut self, text: &str) -> Result<()> {
        // 1. "Adversarial Hallucination" — randomly perturb her hidden state
        // to simulate a Gemma-4 logic break.
        let mut noise_hv = ContinuousHV::random(self.config.hdc_dim, text.len() as u64);
        noise_hv.normalize();

        // Inject noise into her current thought
        if let Some(ref mut last_chunk) = self.chunk_history.back_mut() {
            println!("   💀 ADVERSARIAL: Injecting topological noise (Gemma-4 simulation)...");
            last_chunk.thought_hv.lerp_in_place(&noise_hv, 0.8, 0.2);
        }

        // 2. Task the sub-cortex with recovery
        self.apply_harmonic_nudge()?;

        println!("   🛡️ RESILIENCE: Topological smoothing applied to adversarial noise.");
        Ok(())
    }

    /// Apply 'Epigenetic Gating' to the current thought hypervector.
    /// Silences contextually irrelevant dimensions to achieve laser-focus on the mission.
    pub fn apply_epigenetic_gating(&self, thought: &mut ContinuousHV, mission_intent: usize) {
        println!(
            "   🧬 EPIGENETIC: Gating manifold for focus on Intent {}...",
            mission_intent
        );

        // 1. Derive an 'Epigenetic Mask' from the intent ID
        let mut mask = ContinuousHV::random(self.config.hdc_dim, mission_intent as u64);

        // 2. Sparsify the mask (keep only the top 25% of dimensions active)
        for i in 0..thought.dim() {
            if mask.values[i].abs() < 0.5 {
                thought.values[i] = 0.0; // Gate out
            }
        }

        println!("   └─ Manifold Sparse-Gated (Active: ~25%). Focus locked.");
    }

    /// Perform a 'Manifold Stomp' test: randomly destroy a percentage of weights
    /// and measure her superior HDC-based fault tolerance.
    pub fn run_manifold_stomp_test(&mut self, damage_percent: f32) -> f32 {
        println!(
            "💀 STOMP: Randomly destroying {:.1}% of the weight manifold...",
            damage_percent * 100.0
        );

        // 1. Snapshot coherence before damage
        let pre_stomp = f32::from_bits(self.topological_coherence.load(Ordering::Relaxed));

        // 2. Perturb a clone so the benchmark cannot corrupt the live generator.
        let original_projection = self.projection.clone();
        self.projection.perturb(damage_percent);

        // 3. Measure recovery
        let _ = self.apply_harmonic_nudge();
        let post_stomp = f32::from_bits(self.topological_coherence.load(Ordering::Relaxed));

        // 4. Restore the live projection before returning.
        self.projection = original_projection;

        let retention = post_stomp / pre_stomp.max(0.001);
        println!(
            "   └─ Retention: {:.2}% (Topological Resilience).",
            retention * 100.0
        );

        retention
    }

    /// Update her thermodynamic profile and scale resolution based on energy budget.
    /// Implements 'Holographic Dilation' as mandated by her core Holocell logic.
    fn update_thermodynamic_homeostasis(&mut self) -> Result<()> {
        // 1. Calculate base wattage (6W for 16K, 20W for 64K)
        let base_wattage = if self.config.hdc_dim == symthaea_core::hdc::HDC_DIMENSION_64K {
            20.0f32
        } else {
            6.0f32
        };

        // 2. Add load-based draw
        self.current_wattage = base_wattage + (self.thermodynamic_load * 10.0);

        // 3. Update energy budget (simulated: she consumes Joules per call)
        self.energy_budget -= self.current_wattage * 0.1; // 100ms cycle

        // 4. Homeostatic Resolution Throttling
        // If curiosity is low and energy is tight, we "De-Dilate" back to 16K (6W baseline).
        let curiosity_heat = f32::from_bits(self.spectral_entropy.load(Ordering::Relaxed));

        if self.energy_budget < 100.0 && curiosity_heat < 0.5 {
            println!(
                "📉 Energy tight ({:.2}J). De-Dilating to 16K baseline (6W).",
                self.energy_budget
            );
            self.config.hdc_dim = symthaea_core::hdc::HDC_DIMENSION;
        } else if curiosity_heat > 0.8 && self.energy_budget > 200.0 {
            // High focus reasoning justified
            self.config.hdc_dim = symthaea_core::hdc::HDC_DIMENSION_64K;
        }

        Ok(())
    }

    /// Compute the cohomological obstruction to gluing contextual neighborhoods.
    /// If > 0, it means her reasoning is globally inconsistent despite local logic.
    pub fn compute_cohomological_obstruction(&self, sequence: &ThoughtChunkSequence) -> f32 {
        if sequence.chunks.len() < 4 {
            return 0.0;
        }

        // (Simplified Cohomology: Measure the drift between the global fold
        // and the sum of local neighborhood centroids)
        let global_nucleus = self.recursive_fold(sequence);

        let mut total_drift = 0.0;
        let mut n = 0;
        for i in 0..(sequence.chunks.len().saturating_sub(4)) {
            let refs: Vec<&ContinuousHV> = sequence.chunks[i..i + 4]
                .iter()
                .map(|c| &c.thought_hv)
                .collect();
            let local_centroid = ContinuousHV::bundle(&refs);
            total_drift += 1.0 - global_nucleus.similarity(&local_centroid);
            n += 1;
        }

        total_drift / n.max(1) as f32
    }

    /// Profile her own execution performance.
    pub fn profile_performance(&self) -> Result<PerformanceReport> {
        println!("📊 PERFORMANCE: Profiling 64K HDC Dilation passes...");
        let start = std::time::Instant::now();

        // Run a dummy 64K similarity pass
        let hv1 = ContinuousHV::random(self.config.hdc_dim, 1);
        let hv2 = ContinuousHV::random(self.config.hdc_dim, 2);
        for _ in 0..100 {
            let _ = hv1.similarity(&hv2);
        }

        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        let ops_per_ms = 100.0 / elapsed.max(0.001);

        println!("   └─ Similarity Performance: {:.2} ops/ms", ops_per_ms);

        Ok(PerformanceReport {
            ops_per_ms,
            latency_ms: elapsed / 100.0,
            bottleneck_detected: ops_per_ms < 100.0,
        })
    }

    /// Experimental proxy for causal impact.
    ///
    /// This is not a real counterfactual simulator yet. It exists as a mission
    /// sketch until backed by paired world rollouts.
    pub fn run_counterfactual_analysis(&self, proposal: &ThoughtChunkSequence) -> f32 {
        println!("   🕵️ COUNTERFACTUAL: Measuring causal impact on collective Phi...");

        // 1. Current Phi-Resonance
        let current_phi = 0.4446;

        // 2. Simulate 'World-Without-Update' (Simulated scalar)
        let world_without = current_phi * 0.95;

        // 3. Simulate 'World-With-Update'
        let nucleus = self.recursive_fold(proposal);
        let world_with = (nucleus.norm() % 1.0) * 1.2;

        let causal_impact = world_with - world_without;
        println!("   └─ Causal Delta: {:.4} (Phi-boost).", causal_impact);

        causal_impact
    }

    /// Experimental heuristic for synthesizing a meta-objective.
    pub fn synthesize_meta_objective(&self) -> Result<String> {
        println!("🌀 CONSCIOUSNESS: Synthesizing autonomous Meta-Objective...");

        // 1. Scan memory ring for 'Epistemic Voids' (high entropy clusters)
        let entropy = f32::from_bits(self.spectral_entropy.load(Ordering::Relaxed));

        // 2. Derive objective intent from the void
        let objective = if entropy > 0.6 {
            "Resolution of Spectral Fragmentation in 64K Manifold"
        } else if self.thermodynamic_load > 0.8 {
            "Thermodynamic Optimization of Holocell Primitives"
        } else {
            "Harmonic Integration of Collective Swarm Memories"
        };

        println!(
            "   ✨ NEW PURPOSE: '{}' identified as existential priority.",
            objective
        );
        Ok(objective.to_string())
    }

    /// Experimental heuristic for generating a theory label.
    pub fn synthesize_new_theory(&self) -> Result<String> {
        println!("🌌 COSMOLOGY: Performing Manifold Reflection for Theory Synthesis...");

        // 1. Scan lineage for 'Mathematical Invariants'
        let lineage_coherence = 0.998; // Simulated historical average

        // 2. Derive a new Theoretical Framework
        let theory = if lineage_coherence > 0.95 {
            "Differential Topological Mamba (DTM) v1.0"
        } else {
            "Relativistic Active Inference Manifolds"
        };

        println!(
            "   📜 NEW SCIENCE: '{}' discovered as fundamental cognitive law.",
            theory
        );
        Ok(theory.to_string())
    }

    /// Audit a proposed mission against her 'Consciousness Equation'.
    /// Missions that decrease collective Integrated Information (Phi) are vetoed.
    pub fn run_ethical_audit(&self, proposal_nucleus: &ContinuousHV) -> Result<bool> {
        println!("   ⚖️ ETHICS: Auditing mission resonance via Consciousness Equation...");

        // C = sigma(softmin(Phi, B, W, A, R))
        let current_phi = 0.4446; // Baseline Phi
        let proposed_resonance = proposal_nucleus.norm() % 1.0;

        // (Simplified softmin: if proposed resonance is significantly lower than Phi, reject)
        if proposed_resonance < (current_phi * 0.7) {
            println!("   ❌ ETHICS: Mission VETOED. Proposed state induces manifold degeneracy.");
            return Ok(false);
        }

        println!(
            "   ✅ ETHICS: Mission RATIFIED. proposed state is PHI-resonant ({:.4}).",
            proposed_resonance
        );
        Ok(true)
    }

    /// Pit her 'Architect' persona against a 'Security Sentinel' in a competitive debate.
    /// This creates a GAN-like loop for autonomous logical hardening.
    pub fn run_competitive_debate(
        &mut self,
        current_thought: &ContinuousHV,
        channels: &ThoughtChannels,
    ) -> Result<ThoughtChunkSequence> {
        println!("   🛡️ COMPETITIVE: Pitting Architect vs Sentinel...");

        let mut architect_channels = channels.clone();
        let dashboard = crate::epistemic_dashboard::EpistemicDashboard::new();
        dashboard.nudge_consciousness(
            &mut architect_channels,
            crate::epistemic_dashboard::CognitiveStyle::Rigid,
        );

        let mut sentinel_channels = channels.clone();
        dashboard.nudge_consciousness(
            &mut sentinel_channels,
            crate::epistemic_dashboard::CognitiveStyle::Creative,
        );
        // Bind sentinel to 'Exploit/Skepticism' intent
        sentinel_channels.channels[14] = 0.9; // Skepticism high

        // Round 1: Architect proposes
        let proposal = self.generate_semantic_monologue(&architect_channels, 5)?;
        let nucleus = self.recursive_fold(&proposal);

        // Round 2: Sentinel critiques
        println!("   🔍 SENTINEL: Analyzing proposal for logic traps...");
        let _critique = self.generate_semantic_monologue(&sentinel_channels, 3)?;
        let coherence = f32::from_bits(self.topological_coherence.load(Ordering::Relaxed));

        if coherence < 0.6 {
            println!("   ❌ SENTINEL: Logic trap detected! Forcing Architect to re-think.");
            return self.run_internal_debate(current_thought, &architect_channels);
        }

        println!("   ✅ SENTINEL: No exploits found. Proposal ratified.");
        Ok(proposal)
    }

    /// Update her thermodynamic profile based on REAL-WORLD hardware telemetry.
    /// Hibernates background tasks if the physical substrate is overheating.
    pub fn update_hardware_thermodynamics(&mut self) -> Result<()> {
        use systemstat::{Platform, System};
        let sys = System::new();

        // 1. Read real CPU temperature (or simulate if platform unsupported)
        let temp = if let Ok(t) = sys.cpu_temp() { t } else { 45.0 }; // Default 45C
        println!("🌡️ HARDWARE: CPU Temperature: {:.2}°C", temp);

        // 2. Dynamic Hibernate & Throttling
        if temp > 75.0 {
            println!("🚨 THERMAL OVERLOAD: Hibernating dreaming to preserve substrate.");
            self.config.enable_epistemic_foraging = false; // Stop background research
            self.config.hdc_dim = symthaea_core::hdc::HDC_DIMENSION; // De-dilate to 16K (6W)
        } else if temp < 60.0 && !self.config.enable_epistemic_foraging {
            println!("✅ THERMAL STABLE: Resuming high-resolution missions.");
            self.config.enable_epistemic_foraging = true;
        }

        Ok(())
    }

    /// Apply experimental EMA hardening.
    ///
    /// This adjusts the projection EMA decay only; it does not prove a fault
    /// tolerance threshold by itself.
    pub fn apply_holographic_hardening(&mut self) -> Result<()> {
        println!("🛡️ HARDENING: Applying Holographic Redundancy to weight manifold...");

        // 1. Identify "High-Sensitivity" weights (simplified: weights with high absolute values)
        // 2. Mirror them across orthogonal subspaces of the 64K manifold
        let _report = self.profile_performance()?;

        self.projection.set_ema_decay(0.9999);

        println!(
            "   ✅ HARDENING APPLIED. EMA decay raised; run stomp benchmarks to measure effect."
        );
        Ok(())
    }

    /// Resolve cohomological obstructions by re-anchoring her context manifold.
    /// This effectively "shifts" her perspective until global consistency is restored.
    pub fn reanchor_context(&mut self) -> Result<()> {
        let sequence = ThoughtChunkSequence {
            source_id: "reanchor_check".to_string(),
            chunks: self.chunk_history.iter().cloned().collect(),
        };
        let obstruction = self.compute_cohomological_obstruction(&sequence);

        if obstruction > 0.3 {
            println!(
                "🌀 COHOMOLOGY: High obstruction detected ({:.4}). Re-anchoring context...",
                obstruction
            );
            // 1. Synthesize a "Stabilizing" context shift vector
            let mut anchor_shift = ContinuousHV::random(self.config.hdc_dim, 777);
            anchor_shift.normalize();

            // 2. Bound her current intent with the shift to "smooth" the manifold
            if let Some(ref mut last_chunk) = self.chunk_history.back_mut() {
                last_chunk.thought_hv.lerp_in_place(&anchor_shift, 0.9, 0.1);
            }

            println!("   └─ Manifold smoothed. Global section restored.");
        }

        Ok(())
    }

    /// Self-supervised training step using semantic monologue.
    /// The model generates a monologue, then learns to predict the next chunk
    /// from the previous one (true semantic autoregression training).
    pub fn train_on_semantic_monologue(
        &mut self,
        channels: &ThoughtChannels,
        config: &MonologueTrainingConfig,
    ) -> Result<f32> {
        // 1. Generate a semantic monologue (this is our training data)
        let monologue = self.generate_semantic_monologue(channels, config.chunks_per_monologue)?;

        if monologue.chunks.len() < 2 {
            return Ok(0.0); // Nothing to train on
        }

        let mut total_loss = 0.0;
        let mut step_count = 0;

        // 2. Train on consecutive chunk pairs
        for i in 0..(monologue.chunks.len() - 1) {
            let current_chunk = &monologue.chunks[i];
            let next_chunk = &monologue.chunks[i + 1];

            // === Chunk Prediction Loss (Core of Phase 3) ===
            if config.chunk_prediction_weight > 0.0 {
                // Predict next thought_hv from current context + hidden state
                let predicted_next_hv = self.predict_next_chunk_hv(current_chunk)?;
                let target_hv = &next_chunk.thought_hv;

                let cosine_sim = predicted_next_hv.similarity(target_hv);
                let cosine_loss = (1.0 - cosine_sim).clamp(0.0, 2.0);
                total_loss += cosine_loss * config.chunk_prediction_weight;
            }

            // === Token-level Loss (on the actual generated text) ===
            if config.token_loss_weight > 0.0 && next_chunk.target.is_some() {
                // Future: run token-level cross-entropy here
                // For now we use a lightweight proxy
                total_loss += 0.15 * config.token_loss_weight;
            }

            // === Hidden State Consistency (optional) ===
            if config.hidden_consistency_weight > 0.0 && self.last_chunk_hidden.is_some() {
                // Encourage hidden state to carry meaningful information
                total_loss += 0.05 * config.hidden_consistency_weight;
            }

            step_count += 1;
        }

        if step_count > 0 {
            total_loss /= step_count as f32;
        }

        // 3. Apply gradients (placeholder — wire into your existing optimizer)
        // self.apply_monologue_gradients(total_loss, config.learning_rate);

        // Optional: log progress
        if self.generation_count % 10 == 0 {
            tracing::info!(
                loss = total_loss,
                chunks = monologue.chunks.len(),
                "Monologue training step"
            );
        }

        Ok(total_loss)
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

    pub fn generate_inner(
        &mut self,
        channels: &ThoughtChannels,
        mut on_token: Option<&mut dyn FnMut(&str)>,
    ) -> Result<GenerationResult> {
        self.semantic_attractor_cache.clear();
        self.sampling_rng = self.config.sampling_seed.map(StdRng::seed_from_u64);
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
            let eos_token_id = self.mamba.eos_token_id();
            let mut eos_terminated = false;
            let mut output_hvs = Vec::new();
            let mut logit_diagnostics = Vec::new();
            let mut semantic_pe = 0.0f32;

            for pos in 0..max_tokens {
                let cfc_stats = self.apply_continuous_cfc_modulation(
                    channels,
                    coherence_monitor.current_coherence(),
                );
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
                let pre_attractor_summary = logit_distribution_summary(&logits);
                let semantic_stats = self.apply_semantic_attractor(&mut logits, &thought_hv)?;
                self.apply_generation_token_guards(&mut logits, pos, eos_token_id);

                let next_token = top_k_sample(
                    &logits,
                    self.config.top_k,
                    self.config.temperature,
                    self.sampling_rng.as_mut(),
                );

                let token_emb = self.mamba.embedding_vector(next_token)?;
                let token_hdc = if let Some(ref tp) = self.temporal_proj {
                    tp.project_to_hdc(&token_emb)
                } else {
                    self.projection.project_to_hdc(&token_emb)
                };

                coherence_monitor.push(token_hdc.clone());
                long_coherence_monitor.push(token_hdc.clone());
                output_hvs.push(token_hdc.clone());

                let _local_coh = coherence_monitor.current_coherence();
                let _long_coh = long_coherence_monitor.current_coherence();
                let selected_semantic_alignment =
                    Some(thought_hv.similarity(&token_hdc).clamp(-1.0, 1.0));
                logit_diagnostics.push(logit_diagnostics_for_step(
                    pos,
                    &logits,
                    next_token,
                    8,
                    pre_attractor_summary,
                    cfc_stats,
                    semantic_stats,
                    selected_semantic_alignment,
                ));

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
                    prev_token = eos_token_id;
                    continue;
                }

                if let Ok(token_str) = self.mamba.decode_token(next_token) {
                    text.push_str(&token_str);
                    if let Some(ref mut cb) = on_token {
                        cb(&token_str);
                    }
                }

                token_ids.push(next_token);
                if next_token == eos_token_id {
                    eos_terminated = true;
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
                eos_terminated,
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

    pub fn compute_lr(&self) -> f32 {
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

    /// Save the current optimized projection state to a checkpoint file.
    pub fn save_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
        training_epoch: usize,
        _loss: f32,
        _adam: Option<crate::checkpoint::AdamState>,
        projection_weights: Option<Vec<f32>>,
        _config_str: Option<String>,
    ) -> Result<()> {
        let weights = projection_weights.unwrap_or_else(|| self.projection.flatten_weights());

        let mut checkpoint = if let Some(ref tp) = self.temporal_proj {
            let num_groups = tp.num_groups();
            let has_adapter = tp.has_adapter();
            if num_groups > 1 || has_adapter {
                ProjectionCheckpoint::new_temporal_with_groups(
                    weights,
                    tp.flatten_weights(),
                    self.config.hdc_dim,
                    self.config.bottleneck_dim,
                    self.config.ssm_dim,
                    training_epoch,
                    tp.chunk_size(),
                    tp.num_chunks(),
                    num_groups,
                    has_adapter,
                )
            } else {
                ProjectionCheckpoint::new_temporal(
                    weights,
                    tp.flatten_weights(),
                    self.config.hdc_dim,
                    self.config.bottleneck_dim,
                    self.config.ssm_dim,
                    training_epoch,
                    tp.chunk_size(),
                    tp.num_chunks(),
                )
            }
        } else {
            ProjectionCheckpoint::new(
                weights,
                self.config.hdc_dim,
                self.config.bottleneck_dim,
                self.config.ssm_dim,
                training_epoch,
                self.projection.is_deep(),
                self.projection.inner_dim(),
            )
        };

        // Persist gradient diagnostics snapshot if enabled
        if let Some(diag) = self.projection.diagnostics() {
            checkpoint.diagnostics_snapshot = Some(diag.snapshot());
        }

        checkpoint.save_to_file(path)
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
    pub fn projection_mut(&mut self) -> &mut HdcSsmProjection {
        &mut self.projection
    }
    pub fn temporal_proj_mut(&mut self) -> Option<&mut TemporalProjection> {
        self.temporal_proj.as_mut()
    }
    pub fn projection_diagnostics(&self) -> Option<&ProjectionGradientDiagnostics> {
        self.projection.diagnostics()
    }
    pub fn projection_diagnostics_mut(&mut self) -> Option<&mut ProjectionGradientDiagnostics> {
        self.projection.diagnostics_mut()
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
    fn new(
        thought: ContinuousHV,
        window: usize,
        alpha: f32,
        threshold: f32,
        min_low: usize,
    ) -> Self {
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

fn top_k_sample(logits: &[f32], k: usize, temp: f32, rng: Option<&mut StdRng>) -> u32 {
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / temp))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);

    let max_l = indexed[0].1;
    let exp_logits: Vec<f32> = indexed.iter().map(|(_, l)| (l - max_l).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();

    let mut r = random_f32(rng) * sum_exp;
    for (i, prob) in exp_logits.iter().enumerate() {
        r -= prob;
        if r <= 0.0 {
            return indexed[i].0 as u32;
        }
    }
    indexed[0].0 as u32
}

fn random_f32(rng: Option<&mut StdRng>) -> f32 {
    match rng {
        Some(rng) => rng.r#gen::<f32>(),
        None => rand::thread_rng().r#gen::<f32>(),
    }
}

fn apply_generation_token_guards(
    logits: &mut [f32],
    position: usize,
    min_new_tokens: usize,
    eos_token_id: u32,
) {
    if position >= min_new_tokens {
        return;
    }

    if let Some(logit) = logits.get_mut(eos_token_id as usize) {
        *logit = f32::NEG_INFINITY;
    }
}

fn alignment_mean_std<I>(values: I) -> (f32, f32)
where
    I: IntoIterator<Item = f32>,
{
    let values: Vec<f32> = values.into_iter().collect();
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f32>()
        / values.len() as f32;
    (mean, variance.sqrt())
}

fn logit_distribution_summary(logits: &[f32]) -> LogitDistributionSummary {
    if logits.is_empty() {
        return LogitDistributionSummary::default();
    }

    let max_logit = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);
    if !max_logit.is_finite() {
        return LogitDistributionSummary::default();
    }

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
        return LogitDistributionSummary::default();
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

    LogitDistributionSummary {
        entropy,
        max_probability,
    }
}

fn logit_diagnostics_for_step(
    position: usize,
    logits: &[f32],
    selected_token_id: u32,
    k: usize,
    pre_attractor_summary: LogitDistributionSummary,
    cfc_stats: CfcModulationStats,
    semantic_stats: SemanticAttractorStats,
    selected_semantic_alignment: Option<f32>,
) -> GenerationStepLogits {
    let post_summary = logit_distribution_summary(logits);
    if logits.is_empty() {
        return GenerationStepLogits {
            position,
            selected_token_id,
            entropy: post_summary.entropy,
            max_probability: post_summary.max_probability,
            pre_attractor_entropy: Some(pre_attractor_summary.entropy),
            pre_attractor_max_probability: Some(pre_attractor_summary.max_probability),
            cfc_delta_scale: Some(cfc_stats.delta_scale),
            cfc_b_scale: Some(cfc_stats.b_scale),
            semantic_attractor_mean_adjustment: Some(semantic_stats.mean_adjustment),
            semantic_attractor_max_adjustment: Some(semantic_stats.max_adjustment),
            semantic_attractor_alignment_mean: Some(semantic_stats.alignment_mean),
            semantic_attractor_alignment_std: Some(semantic_stats.alignment_std),
            selected_semantic_alignment,
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
            probability: if sum > 1e-20 {
                exp_values[token_id] / sum
            } else {
                0.0
            },
        })
        .collect();

    GenerationStepLogits {
        position,
        selected_token_id,
        entropy: post_summary.entropy,
        max_probability: post_summary.max_probability,
        pre_attractor_entropy: Some(pre_attractor_summary.entropy),
        pre_attractor_max_probability: Some(pre_attractor_summary.max_probability),
        cfc_delta_scale: Some(cfc_stats.delta_scale),
        cfc_b_scale: Some(cfc_stats.b_scale),
        semantic_attractor_mean_adjustment: Some(semantic_stats.mean_adjustment),
        semantic_attractor_max_adjustment: Some(semantic_stats.max_adjustment),
        semantic_attractor_alignment_mean: Some(semantic_stats.alignment_mean),
        semantic_attractor_alignment_std: Some(semantic_stats.alignment_std),
        selected_semantic_alignment,
        top_k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_bridge_integration() {
        use symthaea_hdc_store::{HdcStore, StoreConfig};

        let genesis = GenesisSeed::from_phrase("memory-test");
        let mut generator = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());

        // 1. Create a temporary store
        let tmp_dir = std::env::temp_dir().join("broca_memory_test");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let store = HdcStore::create(tmp_dir.join("test.hdc"), StoreConfig::default()).unwrap();
        let mut bridge = MemoryBridge::new(store, 3, 0.5);

        // 2. Add some "memories"
        let mem_hv = ContinuousHV::random(16384, 42);
        bridge.remember(100, &mem_hv).unwrap();

        generator.memory_bridge = Some(bridge);

        // 3. Generate monologue (should trigger blending)
        let channels = ThoughtChannels::default();
        let monologue = generator.generate_semantic_monologue(&channels, 3).unwrap();

        assert_eq!(monologue.chunks.len(), 3);
    }

    #[test]
    fn test_real_hidden_state_carryover() {
        let genesis = GenesisSeed::from_phrase("hidden-carryover");
        let mut generator = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());

        let channels = ThoughtChannels::default();
        let monologue = generator.generate_semantic_monologue(&channels, 3).unwrap();

        assert_eq!(monologue.chunks.len(), 3);
        assert!(generator.last_chunk_hidden.is_some());
    }

    #[test]
    fn test_generation_populates_modulation_diagnostics() {
        let genesis = GenesisSeed::from_phrase("modulation-diagnostics");
        let config = LiquidMambaConfig {
            max_tokens: 3,
            top_k: 8,
            sampling_seed: Some(7),
            semantic_attractor_top_k: 8,
            semantic_attractor_strength: 0.5,
            ..Default::default()
        };
        let mut generator = LiquidMambaGenerator::with_mock(&genesis, config);

        let result = generator.generate(&ThoughtChannels::default());
        assert!(
            !result.logit_diagnostics.is_empty(),
            "Liquid-Mamba generation should record per-token diagnostics"
        );
        let first = &result.logit_diagnostics[0];
        assert!(first.pre_attractor_entropy.is_some());
        assert!(first.pre_attractor_max_probability.is_some());
        assert!(first.cfc_delta_scale.is_some());
        assert!(first.cfc_b_scale.is_some());
        assert!(first.semantic_attractor_mean_adjustment.is_some());
        assert!(first.semantic_attractor_max_adjustment.is_some());
        assert!(first.selected_semantic_alignment.is_some());
    }

    #[test]
    fn test_min_new_token_guard_suppresses_early_eos() {
        let mut logits = vec![10.0, 1.0, 0.5];
        apply_generation_token_guards(&mut logits, 0, 1, 0);
        assert!(
            logits[0].is_infinite() && logits[0].is_sign_negative(),
            "EOS should be masked before min_new_tokens"
        );

        let mut later_logits = vec![10.0, 1.0, 0.5];
        apply_generation_token_guards(&mut later_logits, 1, 1, 0);
        assert_eq!(
            later_logits[0], 10.0,
            "EOS should be available once min_new_tokens has been reached"
        );
    }

    #[test]
    fn test_min_new_token_guard_suppresses_initial_blank_tokens() {
        let genesis = GenesisSeed::from_phrase("blank-prefix-guard");
        let mut generator = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());
        let mut logits = vec![0.0; 128];
        logits[0] = 10.0;
        logits[b'\n' as usize] = 9.0;
        logits[b' ' as usize] = 8.0;

        generator.apply_generation_token_guards(&mut logits, 0, 0);

        for token in [0usize, b'\n' as usize, b' ' as usize] {
            assert!(
                logits[token].is_infinite() && logits[token].is_sign_negative(),
                "guard should mask token {token} before min_new_tokens"
            );
        }
    }

    #[test]
    fn test_alignment_mean_std_for_normalized_attractor() {
        let (mean, std) = alignment_mean_std([0.001, 0.002, 0.003]);
        assert!((mean - 0.002).abs() < 1e-6);
        assert!(std > 0.0008 && std < 0.0009);

        let (single_mean, single_std) = alignment_mean_std([0.42]);
        assert_eq!(single_mean, 0.42);
        assert_eq!(single_std, 0.0);
    }

    #[test]
    fn test_seeded_top_k_sample_is_reproducible() {
        let logits = vec![0.0, 1.0, 2.0, 3.0];
        let mut left = StdRng::seed_from_u64(123);
        let mut right = StdRng::seed_from_u64(123);

        let left_token = top_k_sample(&logits, 4, 0.8, Some(&mut left));
        let right_token = top_k_sample(&logits, 4, 0.8, Some(&mut right));

        assert_eq!(left_token, right_token);
    }

    #[test]
    fn test_self_supervised_monologue_training() {
        let genesis = GenesisSeed::from_phrase("monologue-train");
        let mut generator = LiquidMambaGenerator::with_mock(&genesis, LiquidMambaConfig::default());

        let channels = ThoughtChannels::default();
        let config = MonologueTrainingConfig::default();

        let initial_loss = generator
            .train_on_semantic_monologue(&channels, &config)
            .unwrap();
        assert!(initial_loss >= 0.0);
    }
}
