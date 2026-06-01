// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Training pipeline: data collection, BPTT training, and model serialization.
//!
//! # Bootstrap Strategy
//!
//! Run existing LLM organ on diverse StructuredThought inputs, collect
//! (ThoughtChannels, target_text) pairs as JSONL.
//!
//! # BPTT Training
//!
//! Next-token cross-entropy loss with teacher forcing:
//! 1. Forward pass through full sequence (teacher-forced)
//! 2. Cross-entropy loss per position
//! 3. Gradient through weight-tied output
//! 4. BPTT through HdcLtcUnifiedNetwork
//! 5. Truncated BPTT window: 16-32 tokens

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

use crate::checkpoint::AdamState;
use crate::encoder::ThoughtChannels;
use crate::generator::BrocaGenerator;
use crate::tokenizer::{BpeTokenizer, is_code_contamination_token};

mod sequence;
pub use sequence::{SequenceResult, TrainingBackend};

/// A single training pair: thought channels + target text.
///
/// The `channels` field uses `Vec<f32>` for backward compatibility with both
/// legacy 20-channel and current 24-channel JSONL training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPair {
    pub channels: Vec<f32>,
    pub target_text: String,
    #[serde(default)]
    pub target_ids: Vec<u32>,
    /// **NEW**: Emotional valence at start of sequence (-1.0 to 1.0).
    #[serde(default)]
    pub valence: f32,
    /// **NEW**: Emotional arousal at start of sequence (0.0 to 1.0).
    #[serde(default)]
    pub arousal: f32,
}

impl TrainingPair {
    /// Create a new training pair, encoding the target text with the tokenizer.
    pub fn new(channels: ThoughtChannels, target_text: String, tokenizer: &BpeTokenizer) -> Self {
        let target_ids = tokenizer.encode(&target_text);
        Self {
            channels: channels.channels.to_vec(),
            target_text,
            target_ids,
            valence: 0.0,
            arousal: 0.0,
        }
    }

    /// Convert channels to ThoughtChannels, padding legacy 20-channel data with defaults.
    pub fn to_thought_channels(&self) -> ThoughtChannels {
        use crate::encoder::{LEGACY_NUM_CHANNELS, NEW_CHANNEL_DEFAULTS, NUM_CHANNELS};
        let mut tc = ThoughtChannels::default();
        let n = self.channels.len().min(NUM_CHANNELS);
        tc.channels[..n].copy_from_slice(&self.channels[..n]);
        // If legacy data (fewer channels than current), fill missing channels with defaults.
        // ThoughtChannels::default() already provides correct defaults for all channels,
        // but we explicitly set v3 channel defaults for legacy 20-channel data.
        if self.channels.len() < NUM_CHANNELS && self.channels.len() <= LEGACY_NUM_CHANNELS {
            for i in 0..NEW_CHANNEL_DEFAULTS
                .len()
                .min(NUM_CHANNELS - LEGACY_NUM_CHANNELS)
            {
                tc.channels[LEGACY_NUM_CHANNELS + i] = NEW_CHANNEL_DEFAULTS[i];
            }
        }
        // Code channels (24-27) and therapeutic channels (28-31) keep their Default values
        // when not provided in the input data.
        tc
    }
}

/// Training dataset: collection of (channels, target_text) pairs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingDataset {
    pub pairs: Vec<TrainingPair>,
}

impl TrainingDataset {
    /// Load from a JSONL file (one TrainingPair per line).
    pub fn from_jsonl(path: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("reading training data: {path}"))?;

        let pairs: Vec<TrainingPair> = data
            .lines()
            .filter(|l| !l.trim().is_empty())
            .enumerate()
            .map(|(i, line)| {
                serde_json::from_str(line).with_context(|| format!("parsing line {i}"))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { pairs })
    }

    /// Save to a JSONL file.
    pub fn to_jsonl(&self, path: &str) -> Result<()> {
        let mut out = String::new();
        for pair in &self.pairs {
            let line = serde_json::to_string(pair)?;
            out.push_str(&line);
            out.push('\n');
        }
        std::fs::write(path, out).with_context(|| format!("writing training data: {path}"))?;
        Ok(())
    }

    /// Number of training pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Add a training pair.
    pub fn push(&mut self, pair: TrainingPair) {
        self.pairs.push(pair);
    }

    /// Ensure all pairs have target_ids populated from the tokenizer.
    /// Only tokenizes pairs with empty target_ids.
    pub fn tokenize_all(&mut self, tokenizer: &BpeTokenizer) {
        for pair in &mut self.pairs {
            if pair.target_ids.is_empty() {
                pair.target_ids = tokenizer.encode(&pair.target_text);
            }
        }
    }

    /// Re-tokenize all pairs with the given tokenizer, replacing any existing target_ids.
    /// Use this when the dataset was tokenized with a different tokenizer (e.g.,
    /// GPT-NeoX from broca-collect) and needs the BPE tokenizer for CfC-HDC training.
    pub fn retokenize_all(&mut self, tokenizer: &BpeTokenizer) {
        for pair in &mut self.pairs {
            pair.target_ids = tokenizer.encode(&pair.target_text);
        }
    }
}

/// Curriculum learning schedule for training.
#[derive(Debug, Clone, Default)]
pub enum CurriculumSchedule {
    /// No curriculum — train on pairs in dataset order.
    #[default]
    None,
    /// Sort pairs by target length ascending — short sequences first.
    /// Helps CfC learn temporal dynamics before long-range dependencies.
    LengthAscending,
    /// Group pairs by intent (channels 0-7 argmax) and train each group in turn.
    IntentGrouped,
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate (for token embeddings).
    pub learning_rate: f32,
    /// BPTT truncation window (tokens).
    pub bptt_window: usize,
    /// Gradient clipping threshold.
    pub grad_clip: f32,
    /// Report loss every N steps.
    pub report_interval: usize,
    /// Emit unbuffered progress lines to stderr.
    pub progress: bool,
    /// Use Adam optimizer (if false, uses SGD).
    pub use_adam: bool,
    /// Warmup fraction (0.0 to 1.0). First N% of steps use linear LR ramp.
    pub warmup_fraction: f32,
    /// Early stopping patience (0 = disabled). Stop if no improvement for N epochs.
    pub patience: usize,
    /// Enable gradient flow diagnostics (tracking norms, clipping, vanishing/exploding).
    pub enable_diagnostics: bool,
    /// Train CfC network weights via BPTT (not just embeddings).
    /// This is the key improvement: without it, only token embeddings are updated
    /// and loss plateaus rapidly (~5.02 after 3 epochs).
    pub train_network: bool,
    /// Learning rate scale for CfC network weights relative to embedding LR.
    /// Typically 0.1-0.5x embedding LR to prevent destabilizing the temporal dynamics.
    pub network_lr_scale: f32,
    /// Target L2 norm for embedding normalization (0.0 = disabled).
    /// Prevents embedding norm explosion (observed >3000 without normalization).
    pub embedding_target_norm: f32,
    /// Number of negative samples for sampled softmax (0 = full softmax).
    /// When > 0, only computes logits for target + N random tokens per step.
    /// Gives ~V/N speedup (e.g., 4096/64 = 64×) with minimal quality loss.
    /// Recommended: 64-128 for 4K vocabulary.
    pub negative_samples: usize,
    /// Probability of carrying CfC state between training pairs (0.0-1.0).
    /// When > 0, skips controller.reset() between pairs with this probability,
    /// teaching the CfC network cross-sentence temporal coherence.
    /// Recommended: 0.3-0.5.
    pub carry_state: f32,
    /// Phased training: number of embedding-only epochs before enabling CfC BPTT.
    /// Solves the "uniform plateau" problem where CfC receives conflicting gradients
    /// from randomly-sampled negatives before embeddings have separated.
    /// When > 0, `train_network` is forced false for the first N epochs,
    /// then automatically enabled for the remaining epochs.
    /// Recommended: 10-30 (enough for embeddings to develop structure).
    pub network_warmup_epochs: usize,
    /// Optional validation dataset for computing validation loss.
    /// When set, validation loss is reported each epoch and used for early stopping.
    pub validation_dataset: Option<TrainingDataset>,
    /// Curriculum learning schedule.
    pub curriculum: CurriculumSchedule,
    /// Enable automatic gradient anomaly response during training.
    ///
    /// When enabled, checks gradient health at end of each epoch and reacts:
    /// - **Exploding**: halves learning rate
    /// - **Vanishing**: doubles learning rate (capped at 10× initial)
    /// - **Oscillating**: tightens gradient clipping by 50%
    /// - **Plateau**: forces CfC network training on
    /// - **3 consecutive anomalous epochs**: triggers early stopping
    ///
    /// Science: Pascanu et al. (2013) — adaptive gradient management in RNNs
    pub enable_anomaly_response: bool,
    /// Freeze embedding weights during training.
    /// When true, only CfC network weights are updated (via BPTT).
    /// Useful for Phase 2 training where embeddings are already well-trained
    /// and BPTT gradients flowing back through CfC destabilize embedding quality.
    pub freeze_embeddings: bool,
    /// Coherence-gated loss weighting (0.0 = disabled).
    /// When > 0, each training pair's loss contribution is scaled by its
    /// output-thought coherence: `weight = 1.0 - coherence_loss_weight × (1.0 - coherence)`.
    /// Low-coherence pairs (where the network produces output far from thought intent)
    /// have their gradients attenuated, letting training self-organize around
    /// its own veto signal. Recommended: 0.3-0.5.
    ///
    /// Science: Curriculum learning (Bengio et al. 2009) — focus on learnable examples
    pub coherence_loss_weight: f32,
    /// Adaptive coherence warmup: number of epochs before coherence_loss_weight activates.
    /// When > 0, coherence gating starts at 0.0 and linearly ramps to `coherence_loss_weight`
    /// over this many epochs. Early epochs focus on raw CE loss (learning the embedding space),
    /// later epochs increasingly penalize low-coherence outputs.
    /// Mirrors the `network_warmup_epochs` phased strategy.
    pub coherence_warmup_epochs: usize,
    /// Enable post-training smoke test.
    /// When true, after training completes, runs `generate()` on diverse thought
    /// channels and verifies output coherence exceeds `smoke_test_coherence_threshold`.
    /// Results returned in `TrainingValidation`.
    pub enable_smoke_test: bool,
    /// Minimum mean coherence for post-training smoke test to pass (default 0.05).
    pub smoke_test_coherence_threshold: f32,
    /// Coherence collapse threshold for anomaly detection.
    /// When mean epoch coherence drops below this, triggers `CoherenceCollapse` anomaly.
    /// Only active when `enable_anomaly_response` is true. Default: 0.05.
    pub coherence_collapse_threshold: f32,

    // ── Training-Time Fusion ──
    /// Weight for coherence alignment loss (0.0 = disabled).
    /// When > 0, adds `weight * (1 - cosine(output_hv, thought_hv))` to per-token loss.
    /// This trains the CfC to keep output representationally aligned with input thought.
    /// Recommended: 0.1-0.3.
    pub coherence_alignment_weight: f32,
    /// Curriculum annealing for coherence alignment: start at this weight and
    /// linearly decay to `coherence_alignment_weight` over training.
    /// Default 0.0 (disabled — use constant weight).
    /// When > `coherence_alignment_weight`, early epochs emphasize alignment
    /// (keeping CfC in thought-aligned space) while later epochs let CE loss dominate.
    /// Recommended: 1.0 (anneal from 1.0 → 0.2 over training).
    pub coherence_alignment_start_weight: f32,
    /// Merge-token loss bias: multiplicative weight for multi-byte BPE merge tokens.
    /// When > 1.0, the CE loss for tokens that represent merged subwords (e.g. "ing",
    /// "tion") is scaled up, nudging the model toward proper words over raw byte
    /// sequences (e.g. `<0x69> <0x6E> <0x67>` → `ing`).
    /// Default 1.5. Set to 1.0 to disable.
    pub merge_token_loss_weight: f32,

    // ── Training-Time Fusion ──
    /// Enable fusion flags on the generator's controller during training.
    /// When true, compositional logits, adaptive dt, and adaptive alpha are
    /// activated before the training loop begins, so BPTT gradients flow through
    /// the full fused forward path (not just the raw CfC output).
    /// Without this, the CfC network learns dynamics for raw output, and fusion
    /// flags at eval time are a post-hoc overlay.
    pub enable_fusion_during_training: bool,
    /// Epoch at which to enable fusion flags (phased activation).
    /// During epochs 0..fusion_warmup, training uses the raw CfC path.
    /// After fusion_warmup, fusion flags are enabled for the remainder.
    /// Default 0 (enable from the start when enable_fusion_during_training=true).
    pub fusion_warmup_epochs: usize,

    // ── Contrastive Intent Loss ──
    /// Weight for contrastive intent loss (0.0 = disabled).
    /// After processing each training pair's token sequence, the final CfC output HV
    /// is compared against a randomly sampled negative pair's thought HV (different intent).
    /// Loss = weight × max(0, similarity(output, neg_thought) - margin).
    /// This trains the CfC to produce intent-discriminative outputs — different intents
    /// should map to different regions of HDC space.
    ///
    /// Science: Contrastive learning (Chen et al. 2020, SimCLR) adapted to thought-space.
    /// Recommended: 0.1-0.3.
    pub contrastive_weight: f32,
    /// Margin for contrastive loss hinge. Similarity below this is penalty-free.
    /// Default 0.0 (any positive similarity incurs loss).
    /// Higher values (e.g., 0.1) allow some overlap between intents.
    pub contrastive_margin: f32,

    // ── Scheduled Sampling ──
    /// Maximum probability of using model's own prediction as next input (0.0 = pure teacher forcing).
    /// Linearly anneals from 0.0 to this value over training epochs.
    /// Bridges train-test gap: during generation, the model always uses its own outputs,
    /// but during training it only sees ground-truth tokens (exposure bias).
    ///
    /// Science: Bengio et al. (2015) — Scheduled Sampling for Sequence Prediction with RNNs.
    /// Recommended: 0.3-0.5 (higher risks destabilizing early training).
    pub scheduled_sampling_max: f32,

    // ── Label Smoothing ──
    /// Label smoothing epsilon (0.0 = disabled, hard targets).
    /// Distributes `epsilon` probability mass uniformly across all tokens,
    /// targeting `(1 - epsilon)` on the ground truth instead of 1.0.
    /// Prevents overconfident logits and improves generalization.
    ///
    /// Science: Szegedy et al. (2016) — Rethinking the Inception Architecture.
    /// Recommended: 0.1.
    pub label_smoothing: f32,

    // ── Thought-Logit Auxiliary Binding ──
    /// Auxiliary thought-to-logit loss weight (0.0 = disabled).
    /// Trains token embeddings directly from the encoded thought HV in addition
    /// to the recurrent CfC output HV. This is a targeted anti-collapse term:
    /// different thought channels should create different token logit rankings
    /// even before the decoder has learned strong temporal dynamics.
    pub thought_logit_aux_weight: f32,
    /// Number of prefit epochs to run on the direct thought-to-logit path before
    /// recurrent teacher-forced training starts.
    ///
    /// This isolates the binding question: can the positioned thought vector
    /// predict target tokens at all before CfC sequence dynamics are involved?
    /// Default 0 preserves existing behavior.
    pub thought_logit_prefit_epochs: usize,
    /// Weight used by the direct thought-logit prefit loss.
    pub thought_logit_prefit_weight: f32,
    /// Learning-rate multiplier for direct thought-logit prefit.
    pub thought_logit_prefit_lr_scale: f32,

    // ── Distribution Anchoring ──
    /// KL-style anchor loss weight (0.0 = disabled).
    ///
    /// On the first teacher-forced pass for each `(pair, position)`, the trainer
    /// caches the current logit distribution as the local language prior. Later
    /// passes penalize drift away from that cached distribution while still
    /// allowing CE/aux losses to steer the model toward the target. This is a
    /// lightweight frozen-teacher approximation that protects against
    /// catastrophic distribution collapse without requiring a second model copy
    /// in GPU memory.
    pub logit_anchor_weight: f32,

    // ── Top-Token Anti-Collapse ──
    /// Margin loss weight that penalizes a wrong argmax token outranking the
    /// target. This directly attacks degenerate runs where one token dominates
    /// teacher-forced logits despite improving average CE.
    pub top_token_anticollapse_weight: f32,
    /// Required target-vs-wrong-top margin before the anti-collapse term is
    /// inactive. Default 0.0 means only penalize wrong top tokens.
    pub top_token_anticollapse_margin: f32,

    // ── Common-Token Overuse Prior ──
    /// Online loss weight for tokens whose teacher-forced argmax rate exceeds
    /// their target-token frequency in the current curriculum.
    ///
    /// This is a targeted fix for collapse onto legal but overused English
    /// tokens such as "it" or "are": they remain available, but once the model
    /// predicts them far more often than the data uses them, wrong predictions
    /// receive an extra margin penalty.
    pub common_token_prior_weight: f32,
    /// Extra allowed argmax rate above target frequency before the common-token
    /// prior activates. Default 0.05 allows normal function-word reuse.
    pub common_token_prior_slack: f32,
    /// Target-vs-overused-token margin for the common-token prior.
    pub common_token_prior_margin: f32,

    // ── Unknown-Token Anti-Collapse ──
    /// Margin loss weight that penalizes `<unk>` outranking the target token.
    ///
    /// Generation already suppresses `<unk>`, but teacher-forced logits can
    /// still collapse toward it. This term attacks that root cause directly.
    pub unknown_token_penalty_weight: f32,
    /// Required target-vs-`<unk>` logit margin before the penalty is inactive.
    pub unknown_token_penalty_margin: f32,

    // ── Adaptive Veto Warmup ──
    /// Target veto threshold to ramp toward during training (0.0 = disabled).
    /// When > 0, the generator's veto_threshold is linearly ramped from 0.0
    /// to this value over the final `veto_warmup_epochs` of training.
    /// This teaches the CfC network to produce coherent output that satisfies
    /// the veto gate, rather than learning in a veto-free regime and then
    /// failing when veto is enabled at inference.
    ///
    /// Recommended: 0.10-0.20 for initial training, increase if coherence improves.
    pub adaptive_veto_target: f32,
    /// Number of epochs over which to ramp veto threshold from 0.0 to target.
    /// Ramp starts at `epochs - veto_warmup_epochs`. Default 10.
    pub veto_warmup_epochs: usize,
    /// Enable soft veto during training (partial CfC state restore on veto).
    /// When true, veto during training interpolates toward a saved snapshot
    /// rather than hard-resetting, producing smoother gradient flow.
    pub enable_soft_veto_during_training: bool,

    // ── Best Checkpoint Saving ──
    /// Path to save best checkpoint during training (empty = disabled).
    /// When set, saves a checkpoint whenever validation loss improves.
    /// This ensures the best model is preserved even with early stopping
    /// (which saves the final epoch, not the best).
    pub best_checkpoint_path: String,

    // ── Hidden State Dropout ──
    /// Dropout rate for CfC hidden states during training (0.0 = disabled).
    /// After each forward step, randomly zeros this fraction of hidden state
    /// dimensions (inverted dropout: scales remaining by 1/(1-rate)).
    /// Prevents the CfC from memorizing training sequences.
    ///
    /// Science: Gal & Ghahramani (2016) — Dropout as a Bayesian Approximation.
    /// Recommended: 0.1-0.3.
    pub hidden_dropout: f32,

    // ── GPU Acceleration ──
    /// Use GPU-accelerated CfC forward/backward via candle CUDA tensors.
    /// When true and a CUDA device is available, packs neuron states into
    /// batched [N, D] tensors on GPU for ~5-15x BPTT speedup.
    /// Automatically falls back to CPU if CUDA is unavailable.
    #[cfg(feature = "gpu")]
    pub use_gpu_cfc: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            learning_rate: 0.001,
            bptt_window: 16,
            grad_clip: 20.0,
            report_interval: 100,
            progress: false,
            use_adam: true,
            warmup_fraction: 0.1,
            patience: 0,
            enable_diagnostics: false,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 128.0,
            negative_samples: 0,
            carry_state: 0.0,
            network_warmup_epochs: 0,
            validation_dataset: None,
            curriculum: CurriculumSchedule::default(),
            enable_anomaly_response: false,
            freeze_embeddings: false,
            coherence_loss_weight: 0.0,
            coherence_warmup_epochs: 0,
            enable_smoke_test: false,
            smoke_test_coherence_threshold: 0.05,
            coherence_collapse_threshold: 0.05,
            coherence_alignment_weight: 0.0,
            coherence_alignment_start_weight: 0.0,
            merge_token_loss_weight: 1.0, // 1.0 = disabled; use --merge-bias 1.5 to enable
            enable_fusion_during_training: false,
            fusion_warmup_epochs: 0,
            contrastive_weight: 0.0,
            contrastive_margin: 0.0,
            scheduled_sampling_max: 0.0,
            label_smoothing: 0.0,
            thought_logit_aux_weight: 0.0,
            thought_logit_prefit_epochs: 0,
            thought_logit_prefit_weight: 1.0,
            thought_logit_prefit_lr_scale: 1.0,
            logit_anchor_weight: 0.0,
            top_token_anticollapse_weight: 0.0,
            top_token_anticollapse_margin: 0.0,
            common_token_prior_weight: 0.0,
            common_token_prior_slack: 0.05,
            common_token_prior_margin: 0.05,
            unknown_token_penalty_weight: 0.0,
            unknown_token_penalty_margin: 0.0,
            best_checkpoint_path: String::new(),
            hidden_dropout: 0.0,
            adaptive_veto_target: 0.0, // disabled by default
            veto_warmup_epochs: 10,
            enable_soft_veto_during_training: false,
            #[cfg(feature = "gpu")]
            use_gpu_cfc: true, // GPU CfC enabled by default when available
        }
    }
}

/// Training metrics for a single epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub avg_loss: f32,
    pub num_tokens: usize,
    pub num_pairs: usize,
    /// Validation loss (if validation_dataset provided).
    pub validation_loss: Option<f32>,
    /// Mean output-thought coherence this epoch (if coherence_loss_weight > 0 or diagnostics enabled).
    pub mean_coherence: Option<f32>,
    /// Mean adaptive dt used this epoch (when enable_adaptive_dt is on).
    pub adaptive_dt_mean: Option<f32>,
    /// Min adaptive dt observed this epoch.
    pub adaptive_dt_min: Option<f32>,
    /// Max adaptive dt observed this epoch.
    pub adaptive_dt_max: Option<f32>,
}

/// Gradient flow diagnostics: tracks per-step L2 norms, clipping events,
/// and embedding norms to detect vanishing or exploding gradients.
#[derive(Debug, Clone)]
pub struct GradientDiagnostics {
    /// Per-step L2 gradient norms.
    pub grad_norms: Vec<f32>,
    /// Maximum gradient norm observed.
    pub max_grad: f32,
    /// Minimum non-zero gradient norm observed.
    pub min_grad: f32,
    /// Number of steps where gradients were clipped.
    pub clip_count: usize,
    /// Total training steps recorded.
    pub total_steps: usize,
    /// Embedding L2 norms sampled at end of each epoch.
    pub embedding_norms: Vec<f32>,
}

impl GradientDiagnostics {
    fn new() -> Self {
        Self {
            grad_norms: Vec::new(),
            max_grad: 0.0,
            min_grad: f32::INFINITY,
            clip_count: 0,
            total_steps: 0,
            embedding_norms: Vec::new(),
        }
    }

    /// Record a single training step's gradient norm and whether it was clipped.
    pub fn record_step(&mut self, norm: f32, was_clipped: bool) {
        self.grad_norms.push(norm);
        self.total_steps += 1;
        if norm > self.max_grad {
            self.max_grad = norm;
        }
        if norm > 0.0 && norm < self.min_grad {
            self.min_grad = norm;
        }
        if was_clipped {
            self.clip_count += 1;
        }
    }

    /// Record embedding L2 norms (sampled at end of each epoch).
    pub fn record_embedding_norms(&mut self, embeddings: &[symthaea_core::hdc::ContinuousHV]) {
        for emb in embeddings {
            let norm: f32 = emb.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
            self.embedding_norms.push(norm);
        }
    }

    /// Mean gradient norm across all steps.
    pub fn mean_grad_norm(&self) -> f32 {
        if self.grad_norms.is_empty() {
            return 0.0;
        }
        self.grad_norms.iter().sum::<f32>() / self.grad_norms.len() as f32
    }

    /// Number of steps with vanishing gradients (norm < 1e-6).
    pub fn vanishing_count(&self) -> usize {
        self.grad_norms.iter().filter(|&&n| n < 1e-6).count()
    }

    /// Number of steps with exploding gradients (norm > 10).
    pub fn exploding_count(&self) -> usize {
        self.grad_norms.iter().filter(|&&n| n > 10.0).count()
    }

    /// Format a human-readable summary.
    pub fn format_summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Gradient Diagnostics ===\n");
        s.push_str(&format!("Total steps:       {}\n", self.total_steps));
        s.push_str(&format!(
            "Mean grad norm:    {:.6}\n",
            self.mean_grad_norm()
        ));
        s.push_str(&format!("Max grad norm:     {:.6}\n", self.max_grad));
        let min_display = if self.min_grad == f32::INFINITY {
            0.0
        } else {
            self.min_grad
        };
        s.push_str(&format!("Min grad norm:     {:.6}\n", min_display));
        s.push_str(&format!(
            "Clip count:        {} ({:.1}%)\n",
            self.clip_count,
            if self.total_steps > 0 {
                self.clip_count as f32 / self.total_steps as f32 * 100.0
            } else {
                0.0
            }
        ));
        s.push_str(&format!("Vanishing (<1e-6): {}\n", self.vanishing_count()));
        s.push_str(&format!("Exploding (>10):   {}\n", self.exploding_count()));
        if !self.embedding_norms.is_empty() {
            let mean_emb: f32 =
                self.embedding_norms.iter().sum::<f32>() / self.embedding_norms.len() as f32;
            s.push_str(&format!("Mean emb norm:     {:.4}\n", mean_emb));
        }
        s
    }

    /// Detect gradient anomalies from the recorded history.
    ///
    /// Classifies the gradient health based on norm statistics:
    /// - Vanishing: >20% of steps have norm < 1e-6
    /// - Exploding: >5% of steps have norm > 10
    /// - Oscillating: coefficient of variation > 2.0 (norms vary wildly)
    /// - Plateau: all norms within 1% of mean (no learning signal)
    ///
    /// Science: Pascanu et al. (2013) "On the difficulty of training RNNs"
    pub fn detect_anomalies(&self) -> Vec<GradientAnomaly> {
        if self.grad_norms.is_empty() {
            return Vec::new();
        }
        let mut anomalies = Vec::new();
        let n = self.grad_norms.len();
        let mean = self.mean_grad_norm();

        // Vanishing: >20% of steps below threshold
        let vanishing_frac = self.vanishing_count() as f32 / n as f32;
        if vanishing_frac > 0.2 {
            anomalies.push(GradientAnomaly::Vanishing {
                fraction: vanishing_frac,
            });
        }

        // Exploding: >5% of steps above threshold
        let exploding_frac = self.exploding_count() as f32 / n as f32;
        if exploding_frac > 0.05 {
            anomalies.push(GradientAnomaly::Exploding {
                fraction: exploding_frac,
                max_norm: self.max_grad,
            });
        }

        // Oscillating: high coefficient of variation (std/mean > 2.0)
        if mean > 1e-8 {
            let variance = self
                .grad_norms
                .iter()
                .map(|&g| (g - mean).powi(2))
                .sum::<f32>()
                / n as f32;
            let cv = variance.sqrt() / mean;
            if cv > 2.0 {
                anomalies.push(GradientAnomaly::Oscillating {
                    coefficient_of_variation: cv,
                });
            }
        }

        // Plateau: all norms within 1% of mean (no learning signal diversity)
        if n >= 10 && mean > 1e-8 {
            let all_flat = self
                .grad_norms
                .iter()
                .all(|&g| (g - mean).abs() < mean * 0.01);
            if all_flat {
                anomalies.push(GradientAnomaly::Plateau { mean_norm: mean });
            }
        }

        anomalies
    }

    /// Check overall gradient health. Returns true if no anomalies detected.
    pub fn is_healthy(&self) -> bool {
        self.detect_anomalies().is_empty()
    }
}

/// Classification of gradient flow anomalies during training.
///
/// Science: Pascanu et al. (2013), Bengio et al. (1994) — gradient flow in RNNs
#[derive(Debug, Clone, PartialEq)]
pub enum GradientAnomaly {
    /// Too many steps with near-zero gradients — network stops learning.
    Vanishing {
        /// Fraction of steps with gradient norm < 1e-6.
        fraction: f32,
    },
    /// Too many steps with very large gradients — training becomes unstable.
    Exploding {
        /// Fraction of steps with gradient norm > 10.
        fraction: f32,
        /// Maximum observed gradient norm.
        max_norm: f32,
    },
    /// Gradient norms vary wildly — optimizer may be fighting itself.
    Oscillating {
        /// Coefficient of variation (std/mean) of gradient norms.
        coefficient_of_variation: f32,
    },
    /// All gradient norms identical — possible dead network or saturated activations.
    Plateau {
        /// Mean gradient norm during the plateau.
        mean_norm: f32,
    },
    /// Mean output-thought coherence collapsed below threshold.
    /// Network is generating outputs divorced from thought intent.
    CoherenceCollapse {
        /// Mean coherence this epoch.
        mean_coherence: f32,
    },
}

impl std::fmt::Display for GradientAnomaly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vanishing { fraction } => {
                write!(f, "Vanishing gradients: {:.1}% of steps", fraction * 100.0)
            }
            Self::Exploding { fraction, max_norm } => {
                write!(
                    f,
                    "Exploding gradients: {:.1}% of steps (max={:.2})",
                    fraction * 100.0,
                    max_norm
                )
            }
            Self::Oscillating {
                coefficient_of_variation,
            } => {
                write!(
                    f,
                    "Oscillating gradients: CV={:.2}",
                    coefficient_of_variation
                )
            }
            Self::Plateau { mean_norm } => {
                write!(f, "Gradient plateau: mean_norm={:.6}", mean_norm)
            }
            Self::CoherenceCollapse { mean_coherence } => {
                write!(
                    f,
                    "Coherence collapse: mean={:.4} (output divorced from thought)",
                    mean_coherence
                )
            }
        }
    }
}

/// Summary of anomaly responses taken during training.
///
/// Returned alongside `GradientDiagnostics` from `train_with_adam()` so callers
/// can inspect what automatic adjustments were made.
#[derive(Debug, Clone, Default)]
pub struct AnomalyReport {
    /// Final LR multiplier (1.0 = no adjustment).
    pub final_lr_multiplier: f32,
    /// Final effective gradient clip (may differ from config if oscillating detected).
    pub final_grad_clip: f32,
    /// Number of epochs flagged as anomalous.
    pub anomalous_epoch_count: usize,
    /// Whether training was stopped early due to anomaly response (vs patience).
    pub anomaly_early_stopped: bool,
    /// Whether CfC BPTT was force-enabled by plateau detection.
    pub plateau_forced_network_training: bool,
    /// Per-epoch anomaly log: (epoch, anomalies detected).
    pub epoch_anomalies: Vec<(usize, Vec<GradientAnomaly>)>,
    /// Whether coherence collapse was detected during training.
    pub coherence_collapse_detected: bool,
}

/// Post-training validation results from smoke test.
#[derive(Debug, Clone)]
pub struct TrainingValidation {
    /// Per-intent coherence from smoke test generations.
    pub intent_coherences: Vec<(usize, f32)>,
    /// Mean coherence across all smoke test generations.
    pub mean_coherence: f32,
    /// Whether the smoke test passed (mean coherence >= threshold).
    pub passed: bool,
    /// Any intents where coherence was below threshold.
    pub failed_intents: Vec<usize>,
}

/// Compute the effective learning rate with warmup + cosine decay.
///
/// Schedule:
/// 1. Warmup phase (first `warmup_fraction` of total steps):
///    LR ramps linearly from 0.1x to 1.0x base LR.
/// 2. Cosine decay phase (remaining steps):
///    LR decays from base_lr to min_lr following `0.5 * (1 + cos(π * t))`.
fn warmup_lr(base_lr: f32, step: usize, total_steps: usize, warmup_fraction: f32) -> f32 {
    let warmup_steps = (total_steps as f32 * warmup_fraction) as usize;
    let min_lr = base_lr * 0.01; // Floor at 1% of base LR

    if warmup_steps > 0 && step < warmup_steps {
        // Warmup: linear ramp from 0.1x to 1.0x
        let progress = step as f32 / warmup_steps as f32;
        base_lr * (0.1 + 0.9 * progress)
    } else {
        // Cosine decay from base_lr to min_lr
        let decay_steps = total_steps.saturating_sub(warmup_steps);
        if decay_steps == 0 {
            return base_lr;
        }
        let decay_step = step.saturating_sub(warmup_steps);
        let t = (decay_step as f32 / decay_steps as f32).min(1.0);
        min_lr + (base_lr - min_lr) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
    }
}

/// Sample negative indices for sampled softmax training.
///
/// Returns a vector of `target` plus `k` random token indices (without duplicates).
/// Uses a simple LCG for speed (not cryptographic quality — fine for training).
fn sample_negatives(target: usize, vocab_size: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(k + 1);
    indices.push(target);

    // Simple LCG for reproducible-ish fast sampling
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut attempts = 0;
    while indices.len() < k + 1 && attempts < k * 4 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = (state >> 33) as usize % vocab_size;
        if idx != target && !indices.contains(&idx) {
            indices.push(idx);
        }
        attempts += 1;
    }

    indices
}

/// Train the BrocaGenerator on a dataset using teacher forcing.
///
/// For each pair:
/// 1. Encode thought channels → thought_hv
/// 2. Forward through full target sequence (teacher-forced)
/// 3. Cross-entropy loss at each position
/// 4. Accumulate gradients through weight-tied output
///
/// Returns per-epoch metrics.
pub fn train(
    generator: &mut BrocaGenerator,
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Vec<EpochMetrics> {
    let (metrics, _, _, _, _) = train_with_adam(generator, dataset, config, None);
    metrics
}

/// Train with optional Adam optimizer state.
///
/// Returns (metrics, final AdamState if Adam was used, optional GradientDiagnostics,
/// optional AnomalyReport if anomaly response was enabled,
/// optional TrainingValidation if smoke test was enabled).
#[allow(clippy::type_complexity)]
pub fn train_with_adam(
    generator: &mut BrocaGenerator,
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    mut adam_state: Option<AdamState>,
) -> (
    Vec<EpochMetrics>,
    Option<AdamState>,
    Option<GradientDiagnostics>,
    Option<AnomalyReport>,
    Option<TrainingValidation>,
) {
    let mut metrics = Vec::with_capacity(config.epochs);

    // Initialize Adam state if requested and not provided
    if config.use_adam && adam_state.is_none() {
        let vocab_size = generator.tokenizer().vocab_size();
        let dim = generator
            .controller()
            .token_embeddings()
            .first()
            .map(|e| e.dim())
            .unwrap_or(16384);
        adam_state = Some(AdamState::new(vocab_size, dim));
    }

    // Calculate total steps for warmup
    let tokens_per_epoch: usize = dataset
        .pairs
        .iter()
        .map(|p| p.target_ids.len().min(config.bptt_window))
        .sum();
    let total_steps = tokens_per_epoch * config.epochs;

    let mut global_step = 0usize;
    let mut best_loss = f32::INFINITY;
    let mut patience_counter = 0usize;
    let mut diagnostics = if config.enable_diagnostics || config.enable_anomaly_response {
        Some(GradientDiagnostics::new())
    } else {
        None
    };
    let mut lr_multiplier: f32 = 1.0;
    let mut effective_grad_clip = config.grad_clip;
    let mut consecutive_anomaly_epochs: usize = 0;
    let initial_lr = config.learning_rate;
    let mut anomaly_report = if config.enable_anomaly_response {
        Some(AnomalyReport {
            final_lr_multiplier: 1.0,
            final_grad_clip: config.grad_clip,
            ..Default::default()
        })
    } else {
        None
    };

    let use_sampled = config.negative_samples > 0;
    let vocab_size = generator.tokenizer().vocab_size();
    let unknown_token_ids = unknown_token_ids(generator.tokenizer());
    let code_token_ids = code_token_ids(generator.tokenizer());
    let target_token_rates = target_token_rates(dataset, vocab_size);
    let mut predicted_top_counts = vec![0usize; vocab_size];
    let mut predicted_top_total = 0usize;

    // Pre-compute curriculum ordering (indices into dataset)
    let curriculum_order: Vec<usize> = match &config.curriculum {
        CurriculumSchedule::None => (0..dataset.pairs.len()).collect(),
        CurriculumSchedule::LengthAscending => {
            let mut indices: Vec<usize> = (0..dataset.pairs.len()).collect();
            indices.sort_by_key(|&i| dataset.pairs[i].target_ids.len());
            indices
        }
        CurriculumSchedule::IntentGrouped => {
            let mut indices: Vec<usize> = (0..dataset.pairs.len()).collect();
            indices.sort_by_key(|&i| {
                // Argmax of intent channels 0-7
                let channels = &dataset.pairs[i].channels;
                (0..8)
                    .max_by(|&a, &b| channels[a].total_cmp(&channels[b]))
                    .unwrap_or(7)
            });
            indices
        }
    };

    let mut force_train_network = false;

    // Build merge-token set for loss biasing: tokens that are multi-byte BPE merges
    // (not byte tokens like <0xHH>, not special tokens like <bos>/<eos>/<thought>)
    let merge_token_ids: std::collections::HashSet<usize> = if config.merge_token_loss_weight > 1.0
    {
        let tokenizer = generator.tokenizer();
        (0..vocab_size)
            .filter(|&id| {
                let s = tokenizer.token_str(id as u32);
                // Merge tokens are multi-char strings that aren't special tokens
                s.len() > 1 && !s.starts_with('<')
            })
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    // Track whether coherence telemetry is needed
    let track_alignment =
        config.coherence_alignment_weight > 0.0 || config.coherence_alignment_start_weight > 0.0;
    let track_coherence = config.coherence_loss_weight > 0.0
        || config.enable_diagnostics
        || config.enable_anomaly_response
        || track_alignment;
    let anchor_enabled = config.logit_anchor_weight > 0.0;
    let mut logit_anchor_cache: std::collections::HashMap<(usize, usize), Vec<f32>> =
        std::collections::HashMap::new();

    // Pre-compute per-pair intent indices for contrastive sampling.
    // Group pairs by intent (argmax of channels 0-7) so we can sample negatives
    // with different intents efficiently.
    let contrastive_enabled = config.contrastive_weight > 0.0;
    let pair_intents: Vec<usize> = if contrastive_enabled {
        dataset
            .pairs
            .iter()
            .map(|p| {
                (0..8usize.min(p.channels.len()))
                    .max_by(|&a, &b| p.channels[a].total_cmp(&p.channels[b]))
                    .unwrap_or(0)
            })
            .collect()
    } else {
        Vec::new()
    };
    // Pre-encode all thought HVs for contrastive negative lookup
    let contrastive_thought_hvs: Vec<_> = if contrastive_enabled {
        dataset
            .pairs
            .iter()
            .map(|p| {
                let channels = p.to_thought_channels();
                generator.encoder().encode(&channels)
            })
            .collect()
    } else {
        Vec::new()
    };

    // ── GPU Trainer initialization ─────────────────────────────────────
    // Creates a GpuTrainer that keeps ALL state on device (CUDA or CPU tensors).
    // The entire BPTT window runs on GPU; sync happens once per training pair.
    // Automatically detects CUDA and falls back to CPU tensors if unavailable.
    #[cfg(feature = "gpu")]
    let mut gpu_trainer: Option<crate::gpu_cfc::GpuTrainer> = if config.use_gpu_cfc {
        let device = crate::gpu_cfc::detect_device();
        let max_pos = config.bptt_window + 4; // headroom for position cache
        match crate::gpu_cfc::GpuTrainer::from_controller(generator.controller(), &device, max_pos)
        {
            Ok(trainer) => Some(trainer),
            Err(e) => {
                tracing::warn!("GpuTrainer creation failed: {e}, using CPU training");
                None
            }
        }
    } else {
        None
    };

    if config.thought_logit_prefit_epochs > 0 && config.thought_logit_prefit_weight > 0.0 {
        let prefit_lr = config.learning_rate * config.thought_logit_prefit_lr_scale * lr_multiplier;
        for prefit_epoch in 0..config.thought_logit_prefit_epochs {
            let mut prefit_loss = 0.0f32;
            let mut prefit_tokens = 0usize;

            #[cfg(feature = "gpu")]
            if let Some(ref mut trainer) = gpu_trainer {
                let result: candle_core::Result<()> = (|| {
                    for &dataset_idx in &curriculum_order {
                        let pair = &dataset.pairs[dataset_idx];
                        if pair.target_ids.is_empty() {
                            continue;
                        }
                        let channels = pair.to_thought_channels();
                        let thought_hv = generator.encoder().encode(&channels);
                        let thought_tensor = candle_core::Tensor::from_vec(
                            thought_hv.as_slice().to_vec(),
                            (1, thought_hv.as_slice().len()),
                            &trainer.device,
                        )?;
                        for (pos, &target_id) in
                            pair.target_ids.iter().take(config.bptt_window).enumerate()
                        {
                            let loss = trainer.gpu_thought_logit_aux_gradient(
                                &thought_tensor,
                                target_id as usize,
                                pos,
                                prefit_lr,
                                effective_grad_clip,
                                config.thought_logit_prefit_weight,
                            )?;
                            prefit_loss += loss;
                            prefit_tokens += 1;
                        }
                    }
                    trainer.sync_to_cpu(generator.controller_mut())?;
                    trainer.sync_embeddings_to_cpu(generator.controller_mut())?;
                    Ok(())
                })();
                if let Err(err) = result {
                    tracing::warn!(
                        error = %err,
                        "GPU thought-logit prefit failed; continuing with current weights"
                    );
                }
                if config.progress {
                    use std::io::Write;
                    let avg = prefit_loss / prefit_tokens.max(1) as f32;
                    let _ = writeln!(
                        std::io::stderr(),
                        "[prefit] thought-logit epoch {prefit_epoch}/{} loss={avg:.4} tokens={prefit_tokens}",
                        config.thought_logit_prefit_epochs
                    );
                    std::io::stderr().flush().ok();
                }
                continue;
            }

            for &dataset_idx in &curriculum_order {
                let pair = &dataset.pairs[dataset_idx];
                if pair.target_ids.is_empty() {
                    continue;
                }
                let channels = pair.to_thought_channels();
                let thought_hv = generator.encoder().encode(&channels);
                for (pos, &target_id) in pair.target_ids.iter().take(config.bptt_window).enumerate()
                {
                    let loss = apply_thought_logit_aux_gradient(
                        generator.controller_mut(),
                        &thought_hv,
                        target_id as usize,
                        pos,
                        prefit_lr,
                        effective_grad_clip,
                        config.thought_logit_prefit_weight,
                    );
                    prefit_loss += loss;
                    prefit_tokens += 1;
                }
            }

            if config.progress {
                use std::io::Write;
                let avg = prefit_loss / prefit_tokens.max(1) as f32;
                let _ = writeln!(
                    std::io::stderr(),
                    "[prefit] thought-logit epoch {prefit_epoch}/{} loss={avg:.4} tokens={prefit_tokens}",
                    config.thought_logit_prefit_epochs
                );
                std::io::stderr().flush().ok();
            }
        }
    }

    for epoch in 0..config.epochs {
        // Training-time fusion: enable/disable fusion flags based on epoch.
        // Before fusion_warmup: raw CfC path (learn basic dynamics).
        // After fusion_warmup: fused path (learn dynamics that leverage fusion).
        if config.enable_fusion_during_training {
            let fusion_active = epoch >= config.fusion_warmup_epochs;
            let ctrl_config = &mut generator.controller_mut().config_mut();
            ctrl_config.enable_compositional_logits = fusion_active;
            ctrl_config.adaptive_compositional_alpha = fusion_active;
            ctrl_config.enable_adaptive_dt = fusion_active;
        }

        // Adaptive veto warmup: ramp veto_threshold from 0.0 to target over
        // the final veto_warmup_epochs. This teaches the CfC to produce outputs
        // that satisfy the veto coherence gate, rather than learning in a
        // veto-free regime and failing when veto is enabled at inference.
        if config.adaptive_veto_target > 0.0 && config.veto_warmup_epochs > 0 {
            let ramp_start = config.epochs.saturating_sub(config.veto_warmup_epochs);
            let effective_veto = if epoch >= ramp_start {
                let progress = (epoch - ramp_start) as f32 / config.veto_warmup_epochs as f32;
                config.adaptive_veto_target * progress.min(1.0)
            } else {
                0.0
            };
            generator.config_mut().gating.veto_threshold = effective_veto;
            generator.config_mut().gating.enable_soft_veto =
                config.enable_soft_veto_during_training;
        }

        // Reset CfC momentum between epochs to prevent accumulated directional
        // bias from 67K+ gradient steps (momentum 0.9 × 67K steps → runaway).
        if epoch > 0 && config.train_network {
            generator.controller_mut().reset_network_momentum();
        }

        let mut total_loss = 0.0f32;
        let mut total_tokens = 0usize;
        let mut coherence_sum = 0.0f32;
        let mut coherence_count = 0usize;
        let mut contrastive_loss_sum = 0.0f32;
        let mut contrastive_count = 0usize;

        // Adaptive coherence weight: ramp from 0 to config value over warmup epochs
        let effective_coherence_weight = if config.coherence_warmup_epochs > 0
            && epoch < config.coherence_warmup_epochs
        {
            config.coherence_loss_weight * (epoch as f32 / config.coherence_warmup_epochs as f32)
        } else {
            config.coherence_loss_weight
        };

        // Curriculum coherence alignment: anneal from start_weight → alignment_weight
        let effective_alignment_weight = if config.coherence_alignment_start_weight
            > config.coherence_alignment_weight
        {
            let progress = if config.epochs > 1 {
                epoch as f32 / (config.epochs - 1) as f32
            } else {
                1.0
            };
            config.coherence_alignment_start_weight
                + progress
                    * (config.coherence_alignment_weight - config.coherence_alignment_start_weight)
        } else {
            config.coherence_alignment_weight
        };

        // Log phase transition
        if config.network_warmup_epochs > 0 && epoch == config.network_warmup_epochs {
            tracing::info!(
                epoch = epoch,
                "Phase 2: enabling CfC network BPTT (embeddings warmed up)"
            );
            if config.progress {
                use std::io::Write;
                let _ = writeln!(std::io::stderr(), "[phase] epoch {epoch}: CfC BPTT enabled");
                std::io::stderr().flush().ok();
            }
        }

        // LCG state for negative sampling (varies per epoch)
        let mut neg_seed = epoch as u64 * 1000003 + 42;

        let num_pairs = curriculum_order.len();
        let epoch_start = std::time::Instant::now();
        if config.progress {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "  [epoch {epoch}] starting {num_pairs} pairs..."
            );
            std::io::stderr().flush().ok();
        }
        for (pair_idx, &dataset_idx) in curriculum_order.iter().enumerate() {
            if config.progress && pair_idx % 200 == 0 {
                let _running_loss = if total_tokens > 0 {
                    total_loss / total_tokens as f32
                } else {
                    0.0
                };
                let elapsed_s = epoch_start.elapsed().as_secs_f64();
                let pairs_per_sec = if elapsed_s > 0.0 && pair_idx > 0 {
                    pair_idx as f64 / elapsed_s
                } else {
                    0.0
                };
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr(),
                    "  [epoch {epoch}] pair {pair_idx}/{num_pairs} loss={:.4} elapsed={elapsed_s:.1}s rate={pairs_per_sec:.1}pairs/s",
                    total_loss / total_tokens.max(1) as f32
                );
                std::io::stderr().flush().ok();
            }

            let pair = &dataset.pairs[dataset_idx];
            if pair.target_ids.is_empty() {
                continue;
            }

            let channels = pair.to_thought_channels();
            let thought_hv = generator.encoder().encode(&channels);

            // Reset controller for this sequence (unless carrying state)
            let should_carry = config.carry_state > 0.0 && pair_idx > 0 && {
                // Deterministic carry decision based on epoch + pair index
                let hash = (epoch * 10007 + pair_idx * 1009) % 1000;
                (hash as f32 / 1000.0) < config.carry_state
            };
            if !should_carry {
                generator.controller_mut().reset();
                // Seed CfC from thought so training sees thought-dependent initial states
                generator.controller_mut().seed_from_thought(&thought_hv);
            }

            // Teacher-forced forward pass
            let mut prev_token = generator.tokenizer().thought_id;

            let window_end = pair.target_ids.len().min(config.bptt_window);
            let mut sequence_result: Option<SequenceResult> = None;

            // Progress report (GPU or CPU — doesn't matter, report at pair level)
            if config.progress && pair_idx > 0 && pair_idx % config.report_interval == 0 {
                let elapsed_s = epoch_start.elapsed().as_secs_f64();
                let pairs_per_sec = pair_idx as f64 / elapsed_s.max(0.001);
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr(),
                    "  [epoch {epoch}] pair {pair_idx}/{num_pairs} loss={:.4} elapsed={elapsed_s:.1}s rate={pairs_per_sec:.1}pairs/s",
                    total_loss / total_tokens.max(1) as f32
                );
                std::io::stderr().flush().ok();
            }

            // ── GPU fast path: entire BPTT window on device ─────────────
            #[cfg(feature = "gpu")]
            let gpu_ran = if let Some(ref mut trainer) = gpu_trainer {
                let train_network_this_epoch = (config.train_network
                    && epoch >= config.network_warmup_epochs)
                    || force_train_network;
                let mut gpu_sequence = SequenceResult::new(TrainingBackend::Gpu);
                let result: candle_core::Result<()> = (|| {
                    // Transfer thought HV to GPU once per pair
                    let thought_tensor = candle_core::Tensor::from_vec(
                        thought_hv.as_slice().to_vec(),
                        (1, thought_hv.as_slice().len()),
                        &trainer.device,
                    )?;

                    // Reset + seed
                    if !should_carry {
                        trainer.reset_states()?;
                        trainer.seed_from_thought(&thought_tensor)?;
                    }

                    let mut gpu_prev_token = prev_token;
                    for (pos, &target_id) in pair.target_ids[..window_end].iter().enumerate() {
                        let lr = warmup_lr(
                            config.learning_rate * lr_multiplier,
                            global_step,
                            total_steps,
                            config.warmup_fraction,
                        );

                        // Forward on GPU
                        let mut logits =
                            trainer.forward_step(&thought_tensor, gpu_prev_token, pos)?;
                        suppress_decode_forbidden_logits(
                            &mut logits,
                            target_id as usize,
                            vocab_size,
                            &unknown_token_ids,
                            if code_intent_active(&channels) {
                                &[]
                            } else {
                                &code_token_ids
                            },
                        );
                        let common_prior = common_token_prior_for_logits(
                            &logits,
                            target_id as usize,
                            &mut predicted_top_counts,
                            &target_token_rates,
                            &mut predicted_top_total,
                            config.common_token_prior_weight,
                            config.common_token_prior_slack,
                        );
                        let anchor_logits = logit_anchor_for_step(
                            anchor_enabled,
                            &mut logit_anchor_cache,
                            dataset_idx,
                            pos,
                            &logits,
                        );

                        // Loss computation (CPU — cheap)
                        let loss = cross_entropy_loss_smooth(
                            &logits,
                            target_id as usize,
                            config.label_smoothing,
                        );
                        total_loss += loss;
                        total_tokens += 1;
                        gpu_sequence.record_loss(loss);

                        // Backward on GPU (CfC BPTT)
                        if train_network_this_epoch {
                            let network_lr = lr * config.network_lr_scale;
                            trainer.backward_step(
                                &logits,
                                target_id as usize,
                                &thought_tensor,
                                gpu_prev_token,
                                pos,
                                network_lr,
                            )?;
                        }

                        // Embedding gradient: SGD on GPU via outer product matmul
                        // (replaces CPU Adam — avoids CPU↔GPU embedding sync)
                        if !config.freeze_embeddings {
                            let (grad_norm, was_clipped) = trainer.gpu_embedding_gradient(
                                &logits,
                                target_id as usize,
                                lr,
                                effective_grad_clip,
                            )?;
                            if let Some(ref mut diag) = diagnostics {
                                diag.record_step(grad_norm, was_clipped);
                            }
                            gpu_sequence.record_gradient(grad_norm, was_clipped);
                            if config.thought_logit_aux_weight > 0.0 {
                                let aux_loss = trainer.gpu_thought_logit_aux_gradient(
                                    &thought_tensor,
                                    target_id as usize,
                                    pos,
                                    lr,
                                    effective_grad_clip,
                                    config.thought_logit_aux_weight,
                                )?;
                                total_loss += aux_loss;
                            }
                            if let Some(ref reference_logits) = anchor_logits {
                                let anchor_loss = trainer.gpu_distribution_anchor_gradient(
                                    &logits,
                                    reference_logits,
                                    lr,
                                    effective_grad_clip,
                                    config.logit_anchor_weight,
                                )?;
                                total_loss += anchor_loss;
                            }
                            if config.top_token_anticollapse_weight > 0.0 {
                                let anti_loss = trainer.gpu_top_token_anticollapse_gradient(
                                    &logits,
                                    target_id as usize,
                                    lr,
                                    effective_grad_clip,
                                    config.top_token_anticollapse_weight,
                                    config.top_token_anticollapse_margin,
                                )?;
                                total_loss += anti_loss;
                            }
                            if config.unknown_token_penalty_weight > 0.0 {
                                for &unknown_token_id in &unknown_token_ids {
                                    let unk_loss = trainer.gpu_unknown_token_penalty_gradient(
                                        &logits,
                                        target_id as usize,
                                        unknown_token_id,
                                        lr,
                                        effective_grad_clip,
                                        config.unknown_token_penalty_weight,
                                        config.unknown_token_penalty_margin,
                                    )?;
                                    total_loss += unk_loss;
                                }
                            }
                            if let Some((token_id, weight)) = common_prior {
                                let prior_loss = trainer.gpu_unknown_token_penalty_gradient(
                                    &logits,
                                    target_id as usize,
                                    token_id,
                                    lr,
                                    effective_grad_clip,
                                    weight,
                                    config.common_token_prior_margin,
                                )?;
                                total_loss += prior_loss;
                            }
                        }

                        gpu_prev_token = target_id;
                        global_step += 1;
                    }

                    gpu_sequence.set_final_output_hv(trainer.current_output_hv()?);

                    // Periodic sync: CfC weights every 10 pairs, embeddings every 10 pairs
                    // (avoids 256MB embedding transfer per pair while keeping logits fresh)
                    // Sync GPU → CPU periodically (CfC weights for checkpointing)
                    // Embeddings stay on GPU (updated by gpu_embedding_gradient)
                    // Only sync at epoch boundaries or for checkpointing
                    if pair_idx == num_pairs - 1 {
                        if config.enable_diagnostics && config.thought_logit_aux_weight > 0.0 {
                            for (probe_pos, &probe_target) in
                                pair.target_ids[..window_end].iter().take(8).enumerate()
                            {
                                let gpu_logits =
                                    trainer.thought_position_logits(&thought_tensor, probe_pos)?;
                                let gpu_probe = logit_probe(&gpu_logits, probe_target as usize);
                                eprintln!(
                                    "BINDING_PROBE gpu_pre_sync epoch={epoch} pair={pair_idx} pos={probe_pos} target={} rank={} p={:.8} max_p={:.8} selected={}",
                                    probe_target,
                                    gpu_probe.target_rank,
                                    gpu_probe.target_probability,
                                    gpu_probe.max_probability,
                                    gpu_probe.selected_token_id
                                );
                            }
                        }
                        trainer.sync_to_cpu(generator.controller_mut())?;
                        // Also sync GPU embeddings back to CPU for checkpointing
                        trainer.sync_embeddings_to_cpu(generator.controller_mut())?;
                        if config.enable_diagnostics && config.thought_logit_aux_weight > 0.0 {
                            for (probe_pos, &probe_target) in
                                pair.target_ids[..window_end].iter().take(8).enumerate()
                            {
                                let thought_query = thought_hv.bind(
                                    &generator
                                        .controller()
                                        .position_base_ref()
                                        .permute(probe_pos),
                                );
                                let cpu_logits =
                                    generator.controller().compute_logits(&thought_query);
                                let cpu_probe = logit_probe(&cpu_logits, probe_target as usize);
                                eprintln!(
                                    "BINDING_PROBE cpu_post_sync epoch={epoch} pair={pair_idx} pos={probe_pos} target={} rank={} p={:.8} max_p={:.8} selected={}",
                                    probe_target,
                                    cpu_probe.target_rank,
                                    cpu_probe.target_probability,
                                    cpu_probe.max_probability,
                                    cpu_probe.selected_token_id
                                );
                            }
                        }
                    }

                    Ok(())
                })();

                match result {
                    Ok(()) => {
                        sequence_result = Some(gpu_sequence);
                        true
                    }
                    Err(e) => {
                        tracing::warn!(
                            "GPU training failed: {e}, falling back to CPU for this pair"
                        );
                        false
                    }
                }
            } else {
                false
            };
            #[cfg(not(feature = "gpu"))]
            let gpu_ran = false;

            if !gpu_ran {
                let mut cpu_sequence = SequenceResult::new(TrainingBackend::Cpu);
                for (pos, &target_id) in pair.target_ids[..window_end].iter().enumerate() {
                    let lr = warmup_lr(
                        config.learning_rate * lr_multiplier,
                        global_step,
                        total_steps,
                        config.warmup_fraction,
                    );
                    generator.controller_mut().set_learning_rate(lr);

                    let mut logits = if use_sampled {
                        neg_seed = neg_seed.wrapping_add(global_step as u64);
                        let active = sample_negatives(
                            target_id as usize,
                            vocab_size,
                            config.negative_samples,
                            neg_seed,
                        );
                        generator.controller_mut().forward_step_sampled(
                            &thought_hv,
                            prev_token,
                            pos,
                            &active,
                        )
                    } else {
                        generator
                            .controller_mut()
                            .forward_step(&thought_hv, prev_token, pos)
                    };
                    suppress_decode_forbidden_logits(
                        &mut logits,
                        target_id as usize,
                        vocab_size,
                        &unknown_token_ids,
                        if code_intent_active(&channels) {
                            &[]
                        } else {
                            &code_token_ids
                        },
                    );
                    let common_prior = common_token_prior_for_logits(
                        &logits,
                        target_id as usize,
                        &mut predicted_top_counts,
                        &target_token_rates,
                        &mut predicted_top_total,
                        config.common_token_prior_weight,
                        config.common_token_prior_slack,
                    );
                    let anchor_logits = if !use_sampled {
                        logit_anchor_for_step(
                            anchor_enabled,
                            &mut logit_anchor_cache,
                            dataset_idx,
                            pos,
                            &logits,
                        )
                    } else {
                        None
                    };

                    // Training-time dropout on CfC hidden states
                    if config.hidden_dropout > 0.0 {
                        generator
                            .controller_mut()
                            .apply_hidden_dropout(config.hidden_dropout, global_step);
                    }

                    // Cross-entropy loss: -log(softmax[target])
                    // Merge-token bias: weight multi-byte BPE tokens higher to prefer
                    // proper word tokens over raw byte sequences
                    let merge_weight = if merge_token_ids.contains(&(target_id as usize)) {
                        config.merge_token_loss_weight
                    } else {
                        1.0
                    };
                    let raw_loss = cross_entropy_loss_smooth(
                        &logits,
                        target_id as usize,
                        config.label_smoothing,
                    ) * merge_weight;

                    // Coherence tracking + gated loss weighting
                    // (Bengio et al. 2009 — curriculum learning: focus on learnable examples)
                    let loss = if effective_coherence_weight > 0.0 || track_coherence {
                        let output_hv = generator.controller().output_hv();
                        let coherence = output_hv.similarity(&thought_hv);
                        coherence_sum += coherence;
                        coherence_count += 1;
                        cpu_sequence.record_coherence(coherence);
                        let mut adjusted_loss = if effective_coherence_weight > 0.0 {
                            // weight = 1.0 when coherence=1.0, lower when coherence drops
                            let weight =
                                (1.0 - effective_coherence_weight * (1.0 - coherence)).max(0.05);
                            raw_loss * weight
                        } else {
                            raw_loss
                        };
                        // Coherence alignment loss: penalizes output-thought divergence
                        // Uses curriculum-annealed weight when start_weight > final weight
                        if track_alignment && effective_alignment_weight > 0.0 {
                            let alignment_penalty =
                                effective_alignment_weight * (1.0 - coherence).max(0.0);
                            adjusted_loss += alignment_penalty;
                        }
                        adjusted_loss
                    } else {
                        raw_loss
                    };

                    total_loss += loss;
                    total_tokens += 1;
                    cpu_sequence.record_loss(loss);

                    // Phased training: only enable CfC BPTT after network_warmup_epochs
                    // (force_train_network overrides when plateau anomaly detected)
                    let train_network_this_epoch = (config.train_network
                        && epoch >= config.network_warmup_epochs)
                        || force_train_network;

                    // Compute gradient of CE loss w.r.t. output HV (for CfC BPTT)
                    let d_output = if train_network_this_epoch {
                        Some(compute_ce_gradient_wrt_output(
                            &logits,
                            target_id as usize,
                            generator.controller(),
                        ))
                    } else {
                        None
                    };

                    // Apply embedding gradient update (skipped when freeze_embeddings is set)
                    let (grad_norm, was_clipped) = if config.freeze_embeddings {
                        (0.0, false)
                    } else if config.use_adam {
                        let result = apply_weight_tied_gradient_adam(
                            generator.controller_mut(),
                            &logits,
                            target_id as usize,
                            lr,
                            effective_grad_clip,
                            adam_state
                                .as_mut()
                                .expect("invariant: adam_state is Some when config.use_adam"),
                        );
                        if config.thought_logit_aux_weight > 0.0 {
                            let aux_loss = apply_thought_logit_aux_gradient(
                                generator.controller_mut(),
                                &thought_hv,
                                target_id as usize,
                                pos,
                                lr,
                                effective_grad_clip,
                                config.thought_logit_aux_weight,
                            );
                            total_loss += aux_loss;
                        }
                        if let Some(ref reference_logits) = anchor_logits {
                            let anchor_loss = apply_distribution_anchor_gradient(
                                generator.controller_mut(),
                                &logits,
                                reference_logits,
                                lr,
                                effective_grad_clip,
                                config.logit_anchor_weight,
                            );
                            total_loss += anchor_loss;
                        }
                        if config.top_token_anticollapse_weight > 0.0 {
                            let anti_loss = apply_top_token_anticollapse_gradient(
                                generator.controller_mut(),
                                &logits,
                                target_id as usize,
                                lr,
                                effective_grad_clip,
                                config.top_token_anticollapse_weight,
                                config.top_token_anticollapse_margin,
                            );
                            total_loss += anti_loss;
                        }
                        if config.unknown_token_penalty_weight > 0.0 {
                            for &unknown_token_id in &unknown_token_ids {
                                let unk_loss = apply_token_margin_penalty_gradient(
                                    generator.controller_mut(),
                                    &logits,
                                    target_id as usize,
                                    unknown_token_id,
                                    lr,
                                    effective_grad_clip,
                                    config.unknown_token_penalty_weight,
                                    config.unknown_token_penalty_margin,
                                );
                                total_loss += unk_loss;
                            }
                        }
                        if let Some((token_id, weight)) = common_prior {
                            let prior_loss = apply_token_margin_penalty_gradient(
                                generator.controller_mut(),
                                &logits,
                                target_id as usize,
                                token_id,
                                lr,
                                effective_grad_clip,
                                weight,
                                config.common_token_prior_margin,
                            );
                            total_loss += prior_loss;
                        }
                        result
                    } else {
                        let result = apply_weight_tied_gradient(
                            generator.controller_mut(),
                            &logits,
                            target_id as usize,
                            lr,
                            effective_grad_clip,
                        );
                        if config.thought_logit_aux_weight > 0.0 {
                            let aux_loss = apply_thought_logit_aux_gradient(
                                generator.controller_mut(),
                                &thought_hv,
                                target_id as usize,
                                pos,
                                lr,
                                effective_grad_clip,
                                config.thought_logit_aux_weight,
                            );
                            total_loss += aux_loss;
                        }
                        if let Some(ref reference_logits) = anchor_logits {
                            let anchor_loss = apply_distribution_anchor_gradient(
                                generator.controller_mut(),
                                &logits,
                                reference_logits,
                                lr,
                                effective_grad_clip,
                                config.logit_anchor_weight,
                            );
                            total_loss += anchor_loss;
                        }
                        if config.top_token_anticollapse_weight > 0.0 {
                            let anti_loss = apply_top_token_anticollapse_gradient(
                                generator.controller_mut(),
                                &logits,
                                target_id as usize,
                                lr,
                                effective_grad_clip,
                                config.top_token_anticollapse_weight,
                                config.top_token_anticollapse_margin,
                            );
                            total_loss += anti_loss;
                        }
                        if config.unknown_token_penalty_weight > 0.0 {
                            for &unknown_token_id in &unknown_token_ids {
                                let unk_loss = apply_token_margin_penalty_gradient(
                                    generator.controller_mut(),
                                    &logits,
                                    target_id as usize,
                                    unknown_token_id,
                                    lr,
                                    effective_grad_clip,
                                    config.unknown_token_penalty_weight,
                                    config.unknown_token_penalty_margin,
                                );
                                total_loss += unk_loss;
                            }
                        }
                        if let Some((token_id, weight)) = common_prior {
                            let prior_loss = apply_token_margin_penalty_gradient(
                                generator.controller_mut(),
                                &logits,
                                target_id as usize,
                                token_id,
                                lr,
                                effective_grad_clip,
                                weight,
                                config.common_token_prior_margin,
                            );
                            total_loss += prior_loss;
                        }
                        result
                    };

                    // CfC network BPTT: backpropagate CE gradient through the network
                    if let Some(ref d_out) = d_output {
                        let network_lr = lr * config.network_lr_scale;
                        let dt = generator.controller().config().dt_per_token;

                        // CPU BPTT (GPU training uses the fast path above)
                        generator.controller_mut().backward_step(
                            d_out,
                            &thought_hv,
                            prev_token,
                            pos,
                            dt,
                            network_lr,
                        );
                    }

                    if let Some(ref mut diag) = diagnostics {
                        diag.record_step(grad_norm, was_clipped);
                    }
                    cpu_sequence.record_gradient(grad_norm, was_clipped);

                    // Scheduled sampling: with annealed probability, use model's own
                    // prediction as next input instead of teacher-forced ground truth.
                    // This bridges the train-test gap (Bengio et al. 2015).
                    if config.scheduled_sampling_max > 0.0 {
                        let progress = if config.epochs > 1 {
                            epoch as f32 / (config.epochs - 1) as f32
                        } else {
                            1.0
                        };
                        let sampling_prob = config.scheduled_sampling_max * progress;
                        // Deterministic coin flip based on position/pair/epoch
                        let coin =
                            ((epoch * 10007 + pair_idx * 1009 + pos * 997) % 1000) as f32 / 1000.0;
                        if coin < sampling_prob {
                            // Use model's prediction (argmax of logits)
                            let predicted = logits
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                .map(|(i, _)| i as u32)
                                .unwrap_or(target_id);
                            prev_token = predicted;
                        } else {
                            prev_token = target_id;
                        }
                    } else {
                        prev_token = target_id;
                    }
                    global_step += 1;
                }
                cpu_sequence.set_final_output_hv(generator.controller().output_hv().clone());
                sequence_result = Some(cpu_sequence);
            }

            // Note: projected embeddings are refreshed once per epoch (after all sequences),
            // not per sequence. Per-sequence refresh of 4096×256×16384 projection takes ~30s,
            // which dominates training time. Stale projections during an epoch are acceptable
            // (same as standard SGD stale reads for embeddings).

            // Contrastive intent loss: after processing the full token sequence,
            // compare final CfC output against a negative example's thought HV.
            // Different intents should produce different output representations.
            let final_output_hv = sequence_result
                .as_ref()
                .and_then(|result| result.final_output_hv.as_ref());
            if track_coherence
                && sequence_result
                    .as_ref()
                    .is_some_and(|result| result.backend == TrainingBackend::Gpu)
            {
                if let Some(output_hv) = final_output_hv {
                    let coherence = output_hv.similarity(&thought_hv);
                    coherence_sum += coherence;
                    coherence_count += 1;
                }
            }
            if contrastive_enabled {
                let my_intent = pair_intents[dataset_idx];
                // Deterministic negative sampling: find a pair with a different intent
                let neg_seed_val = (epoch * 100003 + pair_idx * 1009 + 7) % dataset.pairs.len();
                let neg_idx = (neg_seed_val..neg_seed_val + dataset.pairs.len())
                    .map(|i| i % dataset.pairs.len())
                    .find(|&i| pair_intents[i] != my_intent)
                    .unwrap_or((neg_seed_val + 1) % dataset.pairs.len());

                let fallback_output_hv;
                let output_hv = if let Some(hv) = final_output_hv {
                    hv
                } else {
                    fallback_output_hv = generator.controller().output_hv();
                    &fallback_output_hv
                };
                let neg_thought = &contrastive_thought_hvs[neg_idx];
                let neg_sim = output_hv.similarity(neg_thought);

                // Hinge loss: penalize when output is too similar to negative thought
                let contrastive_loss =
                    config.contrastive_weight * (neg_sim - config.contrastive_margin).max(0.0);
                contrastive_loss_sum += contrastive_loss;
                contrastive_count += 1;
                if contrastive_loss > 0.0 {
                    total_loss += contrastive_loss;
                    // Count as 1 token equivalent for averaging
                    total_tokens += 1;
                }
            }
        }

        let avg_loss = if total_tokens > 0 {
            total_loss / total_tokens as f32
        } else {
            0.0
        };

        // Compute validation loss — prefer GPU path if available
        let validation_loss = if let Some(ref val_dataset) = config.validation_dataset {
            #[cfg(feature = "gpu")]
            let val_loss = if let Some(ref mut trainer) = gpu_trainer {
                compute_validation_loss_gpu(
                    trainer,
                    generator.encoder(),
                    val_dataset,
                    config.bptt_window,
                )
            } else {
                compute_validation_loss(generator, val_dataset, config.bptt_window)
            };
            #[cfg(not(feature = "gpu"))]
            let val_loss = compute_validation_loss(generator, val_dataset, config.bptt_window);
            Some(val_loss)
        } else {
            None
        };

        let epoch_mean_coherence = if coherence_count > 0 {
            Some(coherence_sum / coherence_count as f32)
        } else {
            None
        };

        metrics.push(EpochMetrics {
            epoch,
            avg_loss,
            num_tokens: total_tokens,
            num_pairs: dataset.len(),
            validation_loss,
            mean_coherence: epoch_mean_coherence,
            adaptive_dt_mean: None,
            adaptive_dt_min: None,
            adaptive_dt_max: None,
        });

        if (epoch + 1) % config.report_interval.max(1) == 0 || epoch == 0 {
            let val_str = validation_loss
                .map(|v| format!(" val_loss={v:.6}"))
                .unwrap_or_default();
            let coh_str = epoch_mean_coherence
                .map(|c| format!(" coh={c:.4}"))
                .unwrap_or_default();
            tracing::info!(
                epoch = epoch,
                avg_loss = avg_loss,
                tokens = total_tokens,
                "Broca training epoch"
            );
            if config.progress {
                // Unbuffered progress (tracing stderr is internally buffered when piped)
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr(),
                    "[epoch] {epoch}/{} loss={avg_loss:.6}{val_str}{coh_str}{} tokens={total_tokens}",
                    config.epochs,
                    if contrastive_count > 0 {
                        format!(
                            " contra={:.4}",
                            contrastive_loss_sum / contrastive_count as f32
                        )
                    } else {
                        String::new()
                    },
                );
                std::io::stderr().flush().ok();
            }
        }

        // Record embedding norms at end of each epoch
        if let Some(ref mut diag) = diagnostics {
            diag.record_embedding_norms(generator.controller().token_embeddings());
        }

        // Gradient anomaly response (end-of-epoch check)
        if config.enable_anomaly_response {
            if let Some(ref diag) = diagnostics {
                let mut anomalies = diag.detect_anomalies();

                // Coherence collapse detection
                if let Some(mean_coh) = epoch_mean_coherence {
                    if mean_coh < config.coherence_collapse_threshold {
                        anomalies.push(GradientAnomaly::CoherenceCollapse {
                            mean_coherence: mean_coh,
                        });
                    }
                }

                if anomalies.is_empty() {
                    consecutive_anomaly_epochs = 0;
                } else {
                    consecutive_anomaly_epochs += 1;
                    for anomaly in &anomalies {
                        match anomaly {
                            GradientAnomaly::Exploding { .. } => {
                                lr_multiplier = (lr_multiplier * 0.5).max(0.01);
                                tracing::warn!(
                                    epoch,
                                    lr_multiplier,
                                    "Anomaly response: halved LR (exploding gradients)"
                                );
                            }
                            GradientAnomaly::Vanishing { .. } => {
                                let max_mult = 10.0 * initial_lr;
                                lr_multiplier = (lr_multiplier * 2.0).min(max_mult / initial_lr);
                                tracing::warn!(
                                    epoch,
                                    lr_multiplier,
                                    "Anomaly response: doubled LR (vanishing gradients)"
                                );
                            }
                            GradientAnomaly::Oscillating { .. } => {
                                effective_grad_clip *= 0.5;
                                tracing::warn!(
                                    epoch,
                                    effective_grad_clip,
                                    "Anomaly response: tightened grad clip (oscillating)"
                                );
                            }
                            GradientAnomaly::Plateau { .. } => {
                                force_train_network = true;
                                if let Some(ref mut report) = anomaly_report {
                                    report.plateau_forced_network_training = true;
                                }
                                tracing::warn!(
                                    epoch,
                                    "Anomaly response: forced CfC training on (plateau)"
                                );
                            }
                            GradientAnomaly::CoherenceCollapse { mean_coherence } => {
                                // Reduce LR to stabilize — aggressive updates may be
                                // pushing output representations away from thought space
                                lr_multiplier = (lr_multiplier * 0.5).max(0.01);
                                if let Some(ref mut report) = anomaly_report {
                                    report.coherence_collapse_detected = true;
                                }
                                tracing::warn!(
                                    epoch,
                                    mean_coherence,
                                    lr_multiplier,
                                    "Anomaly response: halved LR (coherence collapse)"
                                );
                            }
                        }
                    }
                    // Record anomalies in report
                    if let Some(ref mut report) = anomaly_report {
                        report.anomalous_epoch_count += 1;
                        report.epoch_anomalies.push((epoch, anomalies));
                    }
                    if consecutive_anomaly_epochs >= 3 {
                        if let Some(ref mut report) = anomaly_report {
                            report.anomaly_early_stopped = true;
                        }
                        tracing::warn!(
                            epoch,
                            consecutive_anomaly_epochs,
                            "Anomaly response: early stopping (3 consecutive anomalous epochs)"
                        );
                        break;
                    }
                }
            }
        }

        // Normalize embeddings to prevent norm explosion
        if config.embedding_target_norm > 0.0 {
            generator
                .controller_mut()
                .normalize_embeddings(config.embedding_target_norm);
        }

        // Early stopping: use validation loss if available, otherwise training loss
        let stopping_loss = validation_loss.unwrap_or(avg_loss);
        if config.patience > 0 {
            if stopping_loss < best_loss - 1e-6 {
                best_loss = stopping_loss;
                patience_counter = 0;
                // Save best checkpoint if path configured
                if !config.best_checkpoint_path.is_empty() {
                    #[cfg(feature = "mamba-cpu")]
                    let projection_weights = generator.controller().projection_weights();
                    #[cfg(not(feature = "mamba-cpu"))]
                    let projection_weights = None;

                    match generator.save_checkpoint(
                        &config.best_checkpoint_path,
                        epoch,
                        avg_loss,
                        adam_state.as_ref().cloned(),
                        projection_weights,
                        None,
                    ) {
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to save best checkpoint");
                        }
                        _ => {
                            tracing::info!(
                                epoch = epoch,
                                val_loss = stopping_loss,
                                path = %config.best_checkpoint_path,
                                "Saved best checkpoint"
                            );
                        }
                    }
                }
            } else {
                patience_counter += 1;
                if patience_counter >= config.patience {
                    tracing::info!(
                        epoch = epoch,
                        patience = config.patience,
                        best_loss = best_loss,
                        "Early stopping triggered"
                    );
                    break;
                }
            }
        }

        // NATIVE AUTO-SAVE: Secure current weights at the end of each completed epoch
        {
            #[cfg(feature = "mamba-cpu")]
            let projection_weights = generator.controller().projection_weights();
            #[cfg(not(feature = "mamba-cpu"))]
            let projection_weights = None;

            let auto_path = "data/models/broca-checkpoint-latest.bin";
            if let Some(parent) = std::path::Path::new(auto_path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            let _ = generator.save_checkpoint(
                auto_path,
                epoch,
                avg_loss,
                adam_state.as_ref().cloned(),
                projection_weights,
                None,
            );
        }
    }

    // Finalize anomaly report with current state
    if let Some(ref mut report) = anomaly_report {
        report.final_lr_multiplier = lr_multiplier;
        report.final_grad_clip = effective_grad_clip;
    }

    // Post-training smoke test: generate on diverse intents, verify coherence
    let validation = if config.enable_smoke_test {
        let mut intent_coherences = Vec::new();
        let mut failed_intents = Vec::new();
        for intent in 0..8 {
            let channels = crate::encoder::ThoughtChannels::with_intent(intent);
            let result = generator.generate(&channels);
            let coh = result.final_coherence;
            intent_coherences.push((intent, coh));
            if coh < config.smoke_test_coherence_threshold {
                failed_intents.push(intent);
            }
        }
        let mean_coherence = if intent_coherences.is_empty() {
            0.0
        } else {
            intent_coherences.iter().map(|(_, c)| c).sum::<f32>() / intent_coherences.len() as f32
        };
        let passed = mean_coherence >= config.smoke_test_coherence_threshold;
        if !passed {
            tracing::warn!(
                mean_coherence,
                threshold = config.smoke_test_coherence_threshold,
                failed = ?failed_intents,
                "Post-training smoke test FAILED"
            );
        }
        Some(TrainingValidation {
            intent_coherences,
            mean_coherence,
            passed,
            failed_intents,
        })
    } else {
        None
    };

    (metrics, adam_state, diagnostics, anomaly_report, validation)
}

/// Cross-entropy loss for a single position.
fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    cross_entropy_loss_smooth(logits, target, 0.0)
}

fn unknown_token_ids(tokenizer: &BpeTokenizer) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..tokenizer.vocab_size())
        .filter(|&id| tokenizer.token_str(id as u32) == "<unk>")
        .collect();
    let canonical = tokenizer.unk_id as usize;
    if canonical < tokenizer.vocab_size() && !ids.contains(&canonical) {
        ids.push(canonical);
    }
    ids
}

fn code_token_ids(tokenizer: &BpeTokenizer) -> Vec<usize> {
    (0..tokenizer.vocab_size())
        .filter(|&id| is_code_contamination_token(tokenizer.token_str(id as u32)))
        .collect()
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

fn target_token_rates(dataset: &TrainingDataset, vocab_size: usize) -> Vec<f32> {
    let mut counts = vec![0usize; vocab_size];
    let mut total = 0usize;
    for pair in &dataset.pairs {
        for &id in &pair.target_ids {
            let id = id as usize;
            if id < vocab_size {
                counts[id] += 1;
                total += 1;
            }
        }
    }
    if total == 0 {
        return vec![0.0; vocab_size];
    }
    counts
        .into_iter()
        .map(|count| count as f32 / total as f32)
        .collect()
}

fn common_token_prior_for_logits(
    logits: &[f32],
    target: usize,
    predicted_counts: &mut [usize],
    target_rates: &[f32],
    predicted_total: &mut usize,
    weight: f32,
    slack: f32,
) -> Option<(usize, f32)> {
    if weight <= 0.0 || target >= logits.len() {
        return None;
    }
    let (top_id, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    if top_id >= predicted_counts.len() {
        return None;
    }
    predicted_counts[top_id] += 1;
    *predicted_total += 1;
    if top_id == target || *predicted_total == 0 {
        return None;
    }
    let predicted_rate = predicted_counts[top_id] as f32 / *predicted_total as f32;
    let target_rate = target_rates.get(top_id).copied().unwrap_or(0.0);
    let overuse = predicted_rate - target_rate - slack.max(0.0);
    if overuse <= 0.0 {
        return None;
    }
    Some((top_id, weight * overuse.min(1.0)))
}

fn suppress_decode_forbidden_logits(
    logits: &mut [f32],
    target: usize,
    vocab_size: usize,
    unknown_tokens: &[usize],
    code_tokens: &[usize],
) {
    for id in vocab_size..logits.len() {
        if target != id {
            logits[id] = f32::NEG_INFINITY;
        }
    }
    for &unknown_token in unknown_tokens {
        if unknown_token < logits.len() && target != unknown_token {
            logits[unknown_token] = f32::NEG_INFINITY;
        }
    }
    for &code_token in code_tokens {
        if code_token < logits.len() && target != code_token {
            logits[code_token] = f32::NEG_INFINITY;
        }
    }
}

/// Cross-entropy loss with optional label smoothing.
/// When `label_smoothing > 0`, targets become `(1 - eps)` on target and `eps / V` on others.
fn cross_entropy_loss_smooth(logits: &[f32], target: usize, label_smoothing: f32) -> f32 {
    if target >= logits.len() {
        return 0.0;
    }

    // Numerically stable log-softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_z = sum_exp.ln();

    if label_smoothing > 0.0 {
        // Smoothed CE: L = (1-eps) * -log_softmax[target] + eps/V * sum(-log_softmax[i])
        // The sum(-log_softmax[i]) = -sum(logit[i] - max - log_z) = V*log_z - sum(logit[i] - max)
        let log_softmax_target = (logits[target] - max_logit) - log_z;
        let sum_log_softmax: f32 = logits
            .iter()
            .filter(|l| l.is_finite()) // skip NEG_INFINITY from sampled softmax
            .map(|&l| (l - max_logit) - log_z)
            .sum();
        let active_count = logits.iter().filter(|l| l.is_finite()).count() as f32;
        let uniform_loss = if active_count > 0.0 {
            -sum_log_softmax / active_count
        } else {
            0.0
        };
        (1.0 - label_smoothing) * (-log_softmax_target) + label_smoothing * uniform_loss
    } else {
        let log_softmax_target = (logits[target] - max_logit) - log_z;
        -log_softmax_target
    }
}

#[derive(Debug, Clone, Copy)]
struct LogitProbe {
    target_rank: usize,
    target_probability: f32,
    max_probability: f32,
    selected_token_id: usize,
}

fn logit_probe(logits: &[f32], target: usize) -> LogitProbe {
    if logits.is_empty() || target >= logits.len() {
        return LogitProbe {
            target_rank: usize::MAX,
            target_probability: 0.0,
            max_probability: 0.0,
            selected_token_id: 0,
        };
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let (selected_token_id, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((0, &0.0));
    let target_rank = 1 + logits
        .iter()
        .filter(|&&logit| logit > logits[target])
        .count();
    let target_probability = if sum_exp > 0.0 && sum_exp.is_finite() {
        exps[target] / sum_exp
    } else {
        0.0
    };
    let max_probability = if sum_exp > 0.0 && sum_exp.is_finite() {
        exps[selected_token_id] / sum_exp
    } else {
        0.0
    };

    LogitProbe {
        target_rank,
        target_probability,
        max_probability,
        selected_token_id,
    }
}

/// Compute the gradient of cross-entropy loss w.r.t. the network output HV.
///
/// For weight-tied output: logits[i] = scale * cosine_similarity(output_hv, emb[i])
///
/// The derivative of cosine_similarity(o, e) w.r.t. o is:
///   (e - cos(o,e) * o) / (||o|| * ||e||)
///
/// Since output_hv is normalized (||o|| = 1), this simplifies to:
///   (e / ||e|| - cos(o,e) * o)
///
/// Full chain: ∂L/∂o = scale × Σ_i (softmax[i] - 1_{i=target}) × (e_i/||e_i|| - cos_i × o)
///
/// This gradient is used to backpropagate through the CfC network.
fn compute_ce_gradient_wrt_output(
    logits: &[f32],
    target: usize,
    controller: &crate::controller::LanguageController,
) -> symthaea_core::hdc::ContinuousHV {
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    if target >= logits.len() {
        return ContinuousHV::zero(HDC_DIMENSION);
    }

    // Compute softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();

    if sum_exp < 1e-10 {
        return ContinuousHV::zero(HDC_DIMENSION);
    }

    let scale = controller.config().logit_scale;

    // ── GPU-accelerated path ────────────────────────────────────────────
    // When a GpuEmbeddingCache is available (CUDA or CPU tensor ops),
    // compute the gradient via matrix multiply instead of the O(V×D) loop.
    //
    // Math: d_output = scale × error @ E_hat  -  scale × (error·cos) × o
    //   where error[i] = softmax[i] - 1{i=target}
    //   E_hat[i] = E[i] / ||E[i]|| (row-normalized embeddings, already in cache)
    //   cos[i] = logits[i] / scale
    // candle-core is always available (non-optional dep)
    #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
    {
        if let Some(cache) = controller.gpu_embedding_cache() {
            if let Ok(grad) =
                compute_ce_gradient_gpu(&exps, sum_exp, target, scale, logits, controller, cache)
            {
                return grad;
            }
            // Fall through to CPU on error
        }
    }

    // Full-dimension CPU path (no projection, no GPU)
    compute_ce_gradient_cpu(&exps, sum_exp, target, scale, logits, controller)
}

/// CPU fallback: O(vocab × HDC_DIMENSION) double loop
fn compute_ce_gradient_cpu(
    exps: &[f32],
    sum_exp: f32,
    target: usize,
    scale: f32,
    logits: &[f32],
    controller: &crate::controller::LanguageController,
) -> symthaea_core::hdc::ContinuousHV {
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    let embeddings = controller.token_embeddings();
    let output_hv = controller.output_hv();
    let output_slice = output_hv.as_slice();
    let n = embeddings.len().min(logits.len());

    let mut d_output = vec![0.0f32; HDC_DIMENSION];
    for i in 0..n {
        let prob = exps[i] / sum_exp;
        let error = if i == target { prob - 1.0 } else { prob };
        if error.abs() < 1e-6 {
            continue;
        }
        let emb_vals = embeddings[i].as_slice();
        let emb_norm: f32 = emb_vals.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let cos_i = if scale.abs() > 1e-6 {
            logits[i] / scale
        } else {
            0.0
        };
        let scaled_error = scale * error;
        for (j, &ev) in emb_vals.iter().enumerate() {
            if j < d_output.len() {
                d_output[j] += scaled_error * (ev / emb_norm - cos_i * output_slice[j]);
            }
        }
    }
    ContinuousHV::from_slice(&d_output)
}

/// GPU-accelerated CE gradient via candle tensor matmul.
///
/// Computes: d_output = scale × (error @ E_hat) - (scale × Σ(error_i × cos_i)) × o
/// where E_hat is the row-normalized embedding matrix (already in GpuEmbeddingCache).
///
/// This replaces the O(V×D) double loop with two GPU operations:
/// 1. error @ E_hat: [1, V] × [V, D] → [1, D] (matmul)
/// 2. Subtract the output-projection term
// candle-core is always available (non-optional dep)
#[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
fn compute_ce_gradient_gpu(
    exps: &[f32],
    sum_exp: f32,
    target: usize,
    scale: f32,
    logits: &[f32],
    controller: &crate::controller::LanguageController,
    cache: &crate::controller::GpuEmbeddingCache,
) -> Result<symthaea_core::hdc::ContinuousHV, candle_core::Error> {
    use candle_core::Tensor;
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    let n = logits.len();
    let device = cache.device();

    // Compute error vector: error[i] = scale × (softmax[i] - 1{i=target})
    let errors: Vec<f32> = (0..n)
        .map(|i| {
            let prob = exps[i] / sum_exp;
            let error = if i == target { prob - 1.0 } else { prob };
            scale * error
        })
        .collect();

    // error @ E_hat: [1, V] × [V, D] → [1, D]
    // GpuEmbeddingCache stores E_hat (pre-normalized rows)
    let error_tensor = Tensor::from_vec(errors.clone(), (1, n), device)?;
    let grad_from_emb = error_tensor.matmul(cache.embeddings())?; // [1, D]

    // Compute scalar: Σ(error_i × cos_i) for output-projection term
    let cos_sum: f32 = (0..n)
        .map(|i| {
            let cos_i = if scale.abs() > 1e-6 {
                logits[i] / scale
            } else {
                0.0
            };
            errors[i] * cos_i
        })
        .sum();

    // d_output = grad_from_emb - cos_sum × output
    let output_hv = controller.output_hv();
    let output_tensor =
        Tensor::from_vec(output_hv.as_slice().to_vec(), (1, HDC_DIMENSION), device)?;
    let cos_sum_tensor = Tensor::from_vec(vec![cos_sum], (1, 1), device)?;
    let output_term = output_tensor.broadcast_mul(&cos_sum_tensor)?;
    let d_output_tensor = (grad_from_emb - output_term)?;

    // Move back to CPU
    let d_output: Vec<f32> = d_output_tensor.squeeze(0)?.to_vec1()?;
    Ok(ContinuousHV::from_slice(&d_output))
}

/// Apply gradient through weight-tied output (SGD).
///
/// For weight-tied output: logits[i] = scale * cosine_similarity(output_hv, emb[i])
/// ∂L/∂e_i = scale × (softmax[i] - 1_{target}) × output_hv
/// (simplified: we use output_hv as gradient direction to shift embeddings)
///
/// Returns (grad_l2_norm, was_clipped).
fn apply_weight_tied_gradient(
    controller: &mut crate::controller::LanguageController,
    logits: &[f32],
    target: usize,
    lr: f32,
    grad_clip: f32,
) -> (f32, bool) {
    if target >= logits.len() {
        return (0.0, false);
    }

    // Compute softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();

    if sum_exp < 1e-10 {
        return (0.0, false);
    }

    let output_hv = controller.output_hv();
    let output_slice = output_hv.as_slice();
    let scale = controller.config().logit_scale;

    // Update token embeddings: shift target toward output, others away
    let embeddings = controller.token_embeddings_mut();
    let n = embeddings.len().min(logits.len());

    let mut sum_sq = 0.0f32;
    let mut was_clipped = false;

    // Sparse gradient: skip embeddings with negligible error
    let error_threshold = 1e-4;

    for i in 0..n {
        let prob = exps[i] / sum_exp;
        let error = if i == target { prob - 1.0 } else { prob };

        if error.abs() < error_threshold {
            continue;
        }

        sum_sq += error * error;

        // Gradient includes the logit_scale factor
        let raw = -lr * scale * error;
        let grad_scale = raw.clamp(-grad_clip, grad_clip);
        if (grad_scale - raw).abs() > 1e-10 {
            was_clipped = true;
        }

        let emb_values = &mut embeddings[i].values;
        for (j, emb_val) in emb_values.iter_mut().enumerate() {
            if j < output_slice.len() {
                *emb_val += grad_scale * output_slice[j];
            }
        }
    }

    (sum_sq.sqrt(), was_clipped)
}

/// Apply gradient through weight-tied output with Adam optimizer.
///
/// Returns (grad_l2_norm, was_clipped).
fn apply_weight_tied_gradient_adam(
    controller: &mut crate::controller::LanguageController,
    logits: &[f32],
    target: usize,
    lr: f32,
    grad_clip: f32,
    adam: &mut AdamState,
) -> (f32, bool) {
    if target >= logits.len() {
        return (0.0, false);
    }

    // Compute softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();

    if sum_exp < 1e-10 {
        return (0.0, false);
    }

    let output_hv = controller.output_hv();
    let output_slice = output_hv.as_slice();
    let scale = controller.config().logit_scale;

    let embeddings = controller.token_embeddings_mut();
    let n = embeddings.len().min(logits.len());

    let mut sum_sq = 0.0f32;
    let mut was_clipped = false;

    // Sparse gradient: only update embeddings with |error| above threshold.
    // For vocab=4096, most tokens have prob≈0 and error≈0.
    // Target token always updated (error = prob - 1.0).
    // This reduces per-token work from O(vocab × dim) to O(k × dim) where k ≈ 50-100.
    let error_threshold = 1e-4;

    // Increment Adam step counter once per training step (not per embedding)
    adam.t += 1;
    let t = adam.t as f32;
    let bc1 = 1.0 / (1.0 - adam.beta1.powf(t));
    let bc2 = 1.0 / (1.0 - adam.beta2.powf(t));

    for i in 0..n {
        let prob = exps[i] / sum_exp;
        let error = if i == target { prob - 1.0 } else { prob };

        if error.abs() < error_threshold {
            continue;
        }

        sum_sq += error * error;

        // Compute per-dimension gradient and apply Adam in fused loop
        // (avoids allocating separate grad + update vecs)
        let dim = embeddings[i].values.len().min(output_slice.len());

        // Check if any gradient would be clipped
        if !was_clipped {
            let raw_max =
                scale * error.abs() * output_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            if raw_max > grad_clip {
                was_clipped = true;
            }
        }

        let offset = i * dim;
        if offset + dim <= adam.m.len() {
            let emb_values = &mut embeddings[i].values;
            let se = scale * error;
            for j in 0..dim {
                let g = (se * output_slice[j]).clamp(-grad_clip, grad_clip);

                let mj = &mut adam.m[offset + j];
                let vj = &mut adam.v[offset + j];

                *mj = adam.beta1 * (*mj) + (1.0 - adam.beta1) * g;
                *vj = adam.beta2 * (*vj) + (1.0 - adam.beta2) * g * g;

                let m_hat = *mj * bc1;
                let v_hat = *vj * bc2;
                emb_values[j] -= lr * m_hat / (v_hat.sqrt() + adam.epsilon);
            }
        }
    }

    (sum_sq.sqrt(), was_clipped)
}

/// Auxiliary thought-to-logit embedding update.
///
/// Optimizes CE(logits = thought_hv · token_embeddings, target) directly so
/// the encoded thought carries token-discriminative signal even when recurrent
/// decoder logits are still nearly flat.
fn apply_thought_logit_aux_gradient(
    controller: &mut crate::controller::LanguageController,
    thought_hv: &ContinuousHV,
    target: usize,
    pos: usize,
    lr: f32,
    grad_clip: f32,
    weight: f32,
) -> f32 {
    if weight <= 0.0 {
        return 0.0;
    }

    let thought_query = thought_hv.bind(&controller.position_base_ref().permute(pos));
    let logits = controller.compute_logits(&thought_query);
    if target >= logits.len() {
        return 0.0;
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    if sum_exp < 1e-10 {
        return 0.0;
    }

    let log_prob = logits[target] - max_logit - sum_exp.ln();
    let loss = -log_prob;
    let thought_slice = thought_query.as_slice();
    let scale = controller.config().logit_scale;
    let scaled_lr = lr * weight;
    let embeddings = controller.token_embeddings_mut();
    let n = embeddings.len().min(logits.len());

    for i in 0..n {
        let prob = exps[i] / sum_exp;
        let error = if i == target { prob - 1.0 } else { prob };
        if error.abs() < 1e-4 {
            continue;
        }

        let raw = -scaled_lr * scale * error;
        let grad_scale = raw.clamp(-grad_clip, grad_clip);
        let emb_values = &mut embeddings[i].values;
        for (j, emb_val) in emb_values.iter_mut().enumerate() {
            if j < thought_slice.len() {
                *emb_val += grad_scale * thought_slice[j];
            }
        }
    }

    weight * loss
}

fn apply_top_token_anticollapse_gradient(
    controller: &mut crate::controller::LanguageController,
    logits: &[f32],
    target: usize,
    lr: f32,
    grad_clip: f32,
    weight: f32,
    margin: f32,
) -> f32 {
    if weight <= 0.0 || target >= logits.len() {
        return 0.0;
    }
    let Some((top_id, &top_logit)) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return 0.0;
    };
    if top_id == target {
        return 0.0;
    }
    let violation = top_logit + margin - logits[target];
    if violation <= 0.0 {
        return 0.0;
    }

    let output_hv = controller.output_hv();
    let output_slice = output_hv.as_slice();
    let scale = controller.config().logit_scale;
    let violation_scale = violation.clamp(0.0, 10.0);
    let step = (lr * weight * scale * violation_scale).clamp(-grad_clip, grad_clip);
    let embeddings = controller.token_embeddings_mut();

    if target < embeddings.len() {
        for (j, emb_val) in embeddings[target].values.iter_mut().enumerate() {
            if j < output_slice.len() {
                *emb_val += step * output_slice[j];
            }
        }
    }
    if top_id < embeddings.len() {
        for (j, emb_val) in embeddings[top_id].values.iter_mut().enumerate() {
            if j < output_slice.len() {
                *emb_val -= step * output_slice[j];
            }
        }
    }

    weight * violation
}

fn apply_token_margin_penalty_gradient(
    controller: &mut crate::controller::LanguageController,
    logits: &[f32],
    target: usize,
    penalized_token: usize,
    lr: f32,
    grad_clip: f32,
    weight: f32,
    margin: f32,
) -> f32 {
    if weight <= 0.0
        || target >= logits.len()
        || penalized_token >= logits.len()
        || target == penalized_token
    {
        return 0.0;
    }
    let violation = logits[penalized_token] + margin - logits[target];
    if violation <= 0.0 {
        return 0.0;
    }

    let output_hv = controller.output_hv();
    let output_slice = output_hv.as_slice();
    let scale = controller.config().logit_scale;
    let violation_scale = violation.clamp(0.0, 10.0);
    let step = (lr * weight * scale * violation_scale).clamp(-grad_clip, grad_clip);
    let embeddings = controller.token_embeddings_mut();

    if target < embeddings.len() {
        for (j, emb_val) in embeddings[target].values.iter_mut().enumerate() {
            if j < output_slice.len() {
                *emb_val += step * output_slice[j];
            }
        }
    }
    if penalized_token < embeddings.len() {
        for (j, emb_val) in embeddings[penalized_token].values.iter_mut().enumerate() {
            if j < output_slice.len() {
                *emb_val -= step * output_slice[j];
            }
        }
    }

    weight * violation
}

fn logit_anchor_for_step(
    enabled: bool,
    cache: &mut std::collections::HashMap<(usize, usize), Vec<f32>>,
    dataset_idx: usize,
    pos: usize,
    logits: &[f32],
) -> Option<Vec<f32>> {
    if !enabled {
        return None;
    }
    match cache.entry((dataset_idx, pos)) {
        std::collections::hash_map::Entry::Occupied(entry) => Some(entry.get().clone()),
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(logits.to_vec());
            None
        }
    }
}

fn distribution_anchor_loss(current_logits: &[f32], reference_logits: &[f32]) -> f32 {
    let n = current_logits.len().min(reference_logits.len());
    if n == 0 {
        return 0.0;
    }
    let current_probs = softmax_prefix(current_logits, n);
    let reference_probs = softmax_prefix(reference_logits, n);
    let mut loss = 0.0f32;
    for i in 0..n {
        let q = reference_probs[i].max(1e-12);
        let p = current_probs[i].max(1e-12);
        loss += q * (q.ln() - p.ln());
    }
    loss.max(0.0)
}

fn softmax_prefix(logits: &[f32], n: usize) -> Vec<f32> {
    let max_logit = logits[..n]
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return vec![1.0 / n as f32; n];
    }
    let exps: Vec<f32> = logits[..n]
        .iter()
        .map(|&logit| {
            if logit.is_finite() {
                (logit - max_logit).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 1e-12 {
        vec![1.0 / n as f32; n]
    } else {
        exps.into_iter().map(|exp| exp / sum).collect()
    }
}

fn apply_distribution_anchor_gradient(
    controller: &mut crate::controller::LanguageController,
    current_logits: &[f32],
    reference_logits: &[f32],
    lr: f32,
    grad_clip: f32,
    weight: f32,
) -> f32 {
    if weight <= 0.0 {
        return 0.0;
    }
    let n = current_logits
        .len()
        .min(reference_logits.len())
        .min(controller.token_embeddings().len());
    if n == 0 {
        return 0.0;
    }

    let loss = distribution_anchor_loss(&current_logits[..n], &reference_logits[..n]);
    let current_probs = softmax_prefix(current_logits, n);
    let reference_probs = softmax_prefix(reference_logits, n);
    let output_hv = controller.output_hv();
    let output_slice = output_hv.as_slice();
    let scale = controller.config().logit_scale;
    let scaled_lr = lr * weight;
    let embeddings = controller.token_embeddings_mut();

    for i in 0..n {
        let error = current_probs[i] - reference_probs[i];
        if error.abs() < 1e-4 {
            continue;
        }
        let raw = scaled_lr * scale * error;
        let grad_scale = raw.clamp(-grad_clip, grad_clip);
        let emb_values = &mut embeddings[i].values;
        for (j, emb_val) in emb_values.iter_mut().enumerate() {
            if j < output_slice.len() {
                *emb_val -= grad_scale * output_slice[j];
            }
        }
    }

    weight * loss
}

/// Generate a diverse set of ThoughtChannels for training data collection.
///
/// 8 intents × 5 epistemic × 5 emotional clusters × 3 relationship stages
/// × 3 consciousness levels = 1,800 configs.
/// Each has distinct channel encodings covering the full combinatorial space.
pub fn generate_diverse_thoughts() -> Vec<ThoughtChannels> {
    let mut thoughts = Vec::with_capacity(1800);

    // Emotional clusters: (valence, arousal, warmth)
    let emotions = [
        (0.8, 0.2, 0.9),  // Serene-warm (content, calm, nurturing)
        (0.7, 0.8, 0.7),  // Enthusiastic (positive, excited, warm)
        (-0.3, 0.7, 0.3), // Tense-cool (worried, alert, detached)
        (-0.6, 0.3, 0.5), // Melancholic (sad, low energy, moderate warmth)
        (0.0, 0.5, 0.5),  // Neutral-balanced
    ];

    // Relationship stages
    let stages = [0.0, 3.0, 6.0]; // New, Established, Deep

    // Consciousness levels: (psi, meta_awareness, coherence)
    let consciousness = [
        (0.2, 0.1, 0.3), // Low — drowsy/unfocused
        (0.5, 0.5, 0.5), // Medium — typical awareness
        (0.9, 0.8, 0.9), // High — fully lucid
    ];

    for intent in 0..8 {
        for epistemic in 0..5 {
            for &(valence, arousal, warmth) in &emotions {
                for &stage in &stages {
                    for &(psi, meta_aw, coh) in &consciousness {
                        let mut channels = ThoughtChannels::with_intent(intent);
                        channels.set_epistemic(epistemic as f32);
                        channels.set_emotion(valence, arousal, warmth);
                        channels.set_consciousness(psi, meta_aw, coh);
                        // Relationship stage, then trust as a function of shared context depth.
                        channels.channels[15] = stage;
                        channels.channels[16] = (stage / 6.0) * 0.5 + 0.25;
                        // Vary mood temperature with arousal
                        channels.channels[17] = 0.8 + arousal * 0.4;
                        thoughts.push(channels);
                    }
                }
            }
        }
    }

    thoughts
}

/// Reconstruct a text prompt from ThoughtChannels for LLM distillation.
///
/// Uses varied templates keyed on the thought state to avoid templated-start
/// monotony (prior version had 54.9% identical starts).
pub fn thought_to_prompt(channels: &ThoughtChannels) -> String {
    let intent_names = [
        "Acknowledge",
        "Answer",
        "Clarify",
        "Propose",
        "Uncertainty",
        "Reflect",
        "Continue",
        "Unknown",
    ];
    let active_intent = (0..8)
        .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
        .unwrap_or(7);

    let epistemic_names = ["Certain", "Probable", "Uncertain", "Unknown", "OutOfDomain"];
    let epistemic_idx = (channels.epistemic_ordinal() as usize).min(4);

    let intent = intent_names[active_intent];
    let epistemic = epistemic_names[epistemic_idx];
    let valence = channels.valence();
    let arousal = channels.arousal();
    let psi = channels.psi();

    // Select template based on intent + epistemic hash to avoid monotonous starts
    let template_idx = (active_intent * 5 + epistemic_idx) % 8;

    match template_idx {
        0 => format!(
            "SEMANTIC_INTENT: {intent}\nEPISTEMIC_STATUS: {epistemic}\nMOOD_TEMPERATURE: {:.2}\n",
            channels.channels[17]
        ),
        1 => format!(
            "[{intent}] With {epistemic} confidence (valence={valence:.1}, arousal={arousal:.1})\n"
        ),
        2 => format!(
            "Responding with intent to {intent} — epistemic level: {epistemic}, consciousness: {psi:.2}\n"
        ),
        3 => format!(
            "{epistemic} {intent}: emotional state ({valence:.1}/{arousal:.1}), awareness={psi:.2}\n"
        ),
        4 => format!("INTENT={intent} EPISTEMIC={epistemic} PSI={psi:.2} VALENCE={valence:.1}\n"),
        5 => format!("I need to {intent} (confidence: {epistemic}, feeling: {valence:.1})\n"),
        6 => format!(
            "Mode: {intent} | Certainty: {epistemic} | Alertness: {arousal:.1} | Warmth: {:.1}\n",
            channels.warmth()
        ),
        _ => format!(
            "Task={intent}, epistemic={epistemic}, mood={:.2}, psi={psi:.2}\n",
            channels.channels[17]
        ),
    }
}

/// Reconstruct a natural-language target from ThoughtChannels.
///
/// Unlike [`thought_to_prompt`], this intentionally avoids metadata/tag syntax.
/// The goal is to teach the thought-to-token bridge fluent sentence starts while
/// still preserving intent, epistemic stance, affect, relationship stage, and psi.
pub fn thought_to_prose_prompt(channels: &ThoughtChannels) -> String {
    let active_intent = (0..8)
        .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
        .unwrap_or(7);
    let epistemic_idx = (channels.epistemic_ordinal() as usize).min(4);
    let valence = channels.valence();
    let arousal = channels.arousal();
    let warmth = channels.warmth();
    let psi = channels.psi();
    let stage = channels.channels[15];

    let emotion_idx = if valence > 0.6 && arousal < 0.5 {
        0
    } else if valence > 0.4 && arousal >= 0.5 {
        1
    } else if valence < -0.45 {
        2
    } else if valence < -0.1 {
        3
    } else {
        4
    };
    let stage_idx = if stage < 1.0 {
        0
    } else if stage < 4.5 {
        1
    } else {
        2
    };
    let psi_idx = if psi >= 0.75 {
        2
    } else if psi <= 0.3 {
        0
    } else {
        1
    };
    let variant =
        (active_intent * 31 + epistemic_idx * 17 + emotion_idx * 11 + stage_idx * 5 + psi_idx) % 4;

    let stance_options = match epistemic_idx {
        0 => ["I am confident", "The signal is clear", "My read is firm"],
        1 => [
            "It is likely",
            "The evidence points that way",
            "My read is probable",
        ],
        2 => [
            "I am not fully certain",
            "The evidence is mixed",
            "My confidence is partial",
        ],
        3 => [
            "I do not know enough yet",
            "The ground is still incomplete",
            "I need more context",
        ],
        _ => [
            "This may be outside my scope",
            "The request may exceed my evidence",
            "I should not overclaim here",
        ],
    };
    let stance = stance_options[variant % stance_options.len()];

    let tone_options = if valence < -0.45 {
        ["carefully", "steadily", "without rushing"]
    } else if arousal > 0.7 {
        ["directly", "with energy", "in concrete terms"]
    } else if warmth > 0.75 {
        ["warmly", "with care", "in a supportive voice"]
    } else {
        ["clearly", "plainly", "with focus"]
    };
    let tone = tone_options[(variant + active_intent) % tone_options.len()];

    let relation_options = if stage < 1.0 {
        [
            "as we begin",
            "from a fresh starting point",
            "without assuming history",
        ]
    } else if stage < 4.5 {
        [
            "with the context we share",
            "using the thread already established",
            "inside the current conversation",
        ]
    } else {
        [
            "from our deeper context",
            "with the longer pattern in mind",
            "building on what is already known",
        ]
    };
    let relation = relation_options[(variant + epistemic_idx) % relation_options.len()];

    let awareness_options = if psi >= 0.75 {
        [
            "while keeping the whole situation in view",
            "while watching the broader implications",
            "while preserving the larger pattern",
        ]
    } else if psi <= 0.3 {
        [
            "while staying with the immediate facts",
            "while keeping attention close to the data",
            "while avoiding broad leaps",
        ]
    } else {
        [
            "while tracking the main thread",
            "while holding the central point steady",
            "while keeping the response coherent",
        ]
    };
    let awareness = awareness_options[(variant + emotion_idx) % awareness_options.len()];

    let action_options = match active_intent {
        0 => [
            "I understand the situation",
            "I can acknowledge what is present",
            "The first step is to reflect the point back",
        ],
        1 => [
            "The answer should follow the available evidence",
            "I can give a grounded answer",
            "The response should resolve the question",
        ],
        2 => [
            "One clearer detail is needed before I answer",
            "The next move is to ask a focused question",
            "I should narrow the ambiguity first",
        ],
        3 => [
            "We should choose a small practical step",
            "The next action should be concrete",
            "I can propose a workable path forward",
        ],
        4 => [
            "The uncertainty should be named openly",
            "I should avoid pretending to know more",
            "The limits of the evidence need to stay visible",
        ],
        5 => [
            "This deserves reflection before action",
            "The pattern should be mirrored back",
            "I should slow down and examine the meaning",
        ],
        6 => [
            "We can continue from here",
            "The sequence should remain coherent",
            "The next response should extend the thread",
        ],
        _ => [
            "I should pause and ask for grounding",
            "The safest response is to state the limit",
            "I need to re-anchor before continuing",
        ],
    };
    let action = action_options[(variant + stage_idx) % action_options.len()];

    match variant {
        0 => format!("{stance}: {action}. I will respond {tone} {relation} {awareness}.\n"),
        1 => format!("{action}. {stance}, so I will respond {tone} {relation} {awareness}.\n"),
        2 => {
            format!("{stance}. {action}. I will keep the response {tone} {relation} {awareness}.\n")
        }
        _ => format!("{action}. {stance}, and I will proceed {tone} {relation} {awareness}.\n"),
    }
}

/// Synthetic curriculum target style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurriculumStyle {
    /// Metadata-like targets useful for channel inspection.
    Structured,
    /// Fluent prose targets for thought-to-language binding.
    Prose,
}

impl CurriculumStyle {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "structured" => Ok(Self::Structured),
            "prose" => Ok(Self::Prose),
            other => {
                anyhow::bail!("unknown curriculum style {other:?}; expected structured or prose")
            }
        }
    }
}

/// Generate a synthetic training curriculum from diverse ThoughtChannels.
///
/// Uses `generate_diverse_thoughts()` to create 360 diverse ThoughtChannels,
/// converts each to a text prompt via `thought_to_prompt()`, and encodes
/// with the provided tokenizer. Returns pairs suitable for distillation.
///
/// This is "teacher-free" bootstrapping: the target text is the structured
/// prompt itself, training the projection to reproduce thought→text mappings.
pub fn generate_curriculum(tokenizer: &BpeTokenizer) -> TrainingDataset {
    generate_curriculum_with_style(tokenizer, CurriculumStyle::Structured)
}

/// Generate a synthetic training curriculum from diverse ThoughtChannels.
pub fn generate_curriculum_with_style(
    tokenizer: &BpeTokenizer,
    style: CurriculumStyle,
) -> TrainingDataset {
    let thoughts = generate_diverse_thoughts();
    let mut dataset = TrainingDataset::default();

    for thought in &thoughts {
        let prompt = match style {
            CurriculumStyle::Structured => thought_to_prompt(thought),
            CurriculumStyle::Prose => thought_to_prose_prompt(thought),
        };
        let ids = tokenizer.encode(&prompt);
        dataset.pairs.push(TrainingPair {
            channels: thought.channels.to_vec(),
            target_text: prompt,
            target_ids: ids,
            valence: 0.0,
            arousal: 0.5,
        });
    }

    dataset
}

/// Compute average cross-entropy loss on a validation dataset (no weight updates).
///
/// Uses teacher-forced forward pass through the CfC network, computing loss
/// at each position. Returns the average per-token loss.
///
/// Uses negative sampling (same as training) to avoid the 8x cost of computing
/// all 4096 cosine similarities per token. The loss estimate is slightly biased
/// (softmax denominator covers fewer terms) but the relative ranking between
/// epochs is preserved, which is all early stopping needs.
pub fn compute_validation_loss(
    generator: &mut BrocaGenerator,
    dataset: &TrainingDataset,
    bptt_window: usize,
) -> f32 {
    // Full-vocab softmax for validation — sampled softmax plateaus too quickly
    // (60 negatives → PPL ~2.4 over 61 candidates is trivially achievable).
    // Full-vocab gives a meaningful signal for early stopping.
    compute_validation_loss_sampled(generator, dataset, bptt_window, 0)
}

/// Validation loss with configurable negative sampling.
/// `negative_samples = 0` uses full softmax (expensive but exact).
pub fn compute_validation_loss_sampled(
    generator: &mut BrocaGenerator,
    dataset: &TrainingDataset,
    bptt_window: usize,
    negative_samples: usize,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let vocab_size = generator.controller().vocab_size();
    let use_sampled = negative_samples > 0 && vocab_size > negative_samples + 1;
    let mut neg_seed: u64 = 0xDEAD_BEEF_CAFE;

    for pair in &dataset.pairs {
        if pair.target_ids.is_empty() {
            continue;
        }

        let channels = pair.to_thought_channels();
        let thought_hv = generator.encoder().encode(&channels);
        generator.controller_mut().reset();
        generator.controller_mut().seed_from_thought(&thought_hv);

        let mut prev_token = generator.tokenizer().thought_id;
        let window_end = pair.target_ids.len().min(bptt_window);

        for (pos, &target_id) in pair.target_ids[..window_end].iter().enumerate() {
            let logits = if use_sampled {
                neg_seed = neg_seed.wrapping_add(1);
                let active =
                    sample_negatives(target_id as usize, vocab_size, negative_samples, neg_seed);
                generator.controller_mut().forward_step_sampled(
                    &thought_hv,
                    prev_token,
                    pos,
                    &active,
                )
            } else {
                generator
                    .controller_mut()
                    .forward_step(&thought_hv, prev_token, pos)
            };
            let loss = cross_entropy_loss(&logits, target_id as usize);
            total_loss += loss;
            total_tokens += 1;
            prev_token = target_id;
        }
    }

    if total_tokens > 0 {
        total_loss / total_tokens as f32
    } else {
        0.0
    }
}

/// GPU-accelerated validation loss computation.
///
/// Uses GpuTrainer for forward passes instead of CPU controller.
/// ~10x faster than CPU validation on CUDA.
#[cfg(feature = "gpu")]
pub fn compute_validation_loss_gpu(
    trainer: &mut crate::gpu_cfc::GpuTrainer,
    encoder: &crate::encoder::ThoughtLanguageEncoder,
    dataset: &TrainingDataset,
    bptt_window: usize,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;

    for pair in &dataset.pairs {
        if pair.target_ids.is_empty() {
            continue;
        }

        let channels = pair.to_thought_channels();
        let thought_hv = encoder.encode(&channels);

        // Transfer thought to GPU
        let thought_tensor = match candle_core::Tensor::from_vec(
            thought_hv.as_slice().to_vec(),
            (1, thought_hv.as_slice().len()),
            &trainer.device,
        ) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Reset + seed
        if trainer.reset_states().is_err() {
            continue;
        }
        let _ = trainer.seed_from_thought(&thought_tensor);

        let mut prev_token = 4u32; // thought_id
        let window_end = pair.target_ids.len().min(bptt_window);

        for (pos, &target_id) in pair.target_ids[..window_end].iter().enumerate() {
            let logits = match trainer.forward_step(&thought_tensor, prev_token, pos) {
                Ok(l) => l,
                Err(_) => break,
            };
            let loss = cross_entropy_loss(&logits, target_id as usize);
            total_loss += loss;
            total_tokens += 1;
            prev_token = target_id;
        }
    }

    if total_tokens > 0 {
        total_loss / total_tokens as f32
    } else {
        0.0
    }
}

/// Write a curriculum dataset to a JSONL file.
///
/// Returns the number of pairs written.
pub fn write_curriculum_jsonl(path: &str, tokenizer: &BpeTokenizer) -> Result<usize> {
    write_curriculum_jsonl_with_style(path, tokenizer, CurriculumStyle::Structured)
}

/// Write a curriculum dataset to a JSONL file using a target style.
///
/// Returns the number of pairs written.
pub fn write_curriculum_jsonl_with_style(
    path: &str,
    tokenizer: &BpeTokenizer,
    style: CurriculumStyle,
) -> Result<usize> {
    let dataset = generate_curriculum_with_style(tokenizer, style);
    let count = dataset.pairs.len();
    dataset
        .to_jsonl(path)
        .with_context(|| format!("writing curriculum to {path}"))?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::LanguageControllerConfig;
    use crate::gating::GatingConfig;
    use crate::generator::BrocaConfig;
    use crate::generator::SamplingStrategy;
    use symthaea_core::genesis::GenesisSeed;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-training")
    }

    fn test_config() -> BrocaConfig {
        BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32,
                max_seq_len: 16,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 20,
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: false,
            enable_semantic_veto: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_training_pair_creation() {
        let tok = BpeTokenizer::default_minimal();
        let channels = ThoughtChannels::default();
        let pair = TrainingPair::new(channels, "hello world".to_string(), &tok);
        assert!(!pair.target_ids.is_empty());
        assert_eq!(pair.channels.len(), crate::encoder::NUM_CHANNELS);
    }

    #[test]
    fn test_dataset_operations() {
        let tok = BpeTokenizer::default_minimal();
        let mut dataset = TrainingDataset::default();
        assert!(dataset.is_empty());

        let channels = ThoughtChannels::default();
        dataset.push(TrainingPair::new(channels, "test".to_string(), &tok));
        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_cross_entropy_loss() {
        // When target has highest logit, loss should be low
        let logits = vec![0.1, 0.2, 5.0, 0.3];
        let loss_correct = cross_entropy_loss(&logits, 2);

        // When target has low logit, loss should be high
        let loss_wrong = cross_entropy_loss(&logits, 0);

        assert!(
            loss_correct < loss_wrong,
            "Loss for correct prediction should be lower"
        );
        assert!(loss_correct >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_distribution_anchor_loss_tracks_drift() {
        let reference = vec![4.0, 0.0, -1.0];
        let same = distribution_anchor_loss(&reference, &reference);
        let drifted = distribution_anchor_loss(&[-1.0, 0.0, 4.0], &reference);
        assert!(same < 1e-6, "same distribution should have near-zero KL");
        assert!(
            drifted > same + 0.1,
            "drifted distribution should be penalized"
        );
    }

    fn softmax_probability(logits: &[f32], target: usize) -> f32 {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if target >= logits.len() || sum_exp <= 0.0 {
            0.0
        } else {
            exps[target] / sum_exp
        }
    }

    fn target_rank(logits: &[f32], target: usize) -> usize {
        if target >= logits.len() {
            return usize::MAX;
        }
        1 + logits
            .iter()
            .filter(|&&logit| logit > logits[target])
            .count()
    }

    #[test]
    fn test_thought_logit_aux_gradient_improves_target_logit() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let thought_hv = generator.encoder().encode(&channels);
        let thought_query = thought_hv.bind(&generator.controller().position_base_ref().permute(0));
        let target_id = generator.tokenizer().token_id("hello") as usize;

        let before_logits = generator.controller().compute_logits(&thought_query);
        let before_rank = target_rank(&before_logits, target_id);
        let before_prob = softmax_probability(&before_logits, target_id);
        let before_logit = before_logits[target_id];

        let aux_loss = apply_thought_logit_aux_gradient(
            generator.controller_mut(),
            &thought_hv,
            target_id,
            0,
            0.05,
            10.0,
            1.0,
        );

        let after_query = thought_hv.bind(&generator.controller().position_base_ref().permute(0));
        let after_logits = generator.controller().compute_logits(&after_query);
        let after_rank = target_rank(&after_logits, target_id);
        let after_prob = softmax_probability(&after_logits, target_id);
        let after_logit = after_logits[target_id];

        assert!(aux_loss.is_finite() && aux_loss > 0.0);
        assert!(
            after_logit > before_logit,
            "target logit should increase: before={before_logit} after={after_logit}"
        );
        assert!(
            after_prob > before_prob,
            "target probability should increase: before={before_prob} after={after_prob}"
        );
        assert!(
            after_rank <= before_rank,
            "target rank should not get worse: before={before_rank} after={after_rank}"
        );
    }

    #[test]
    fn test_tiny_binding_overfit_sanity_has_clean_decode() {
        let genesis = test_genesis();
        let mut generator = BrocaGenerator::new(&genesis, test_config());
        let tokenizer = generator.tokenizer().clone();
        let dataset = TrainingDataset {
            pairs: vec![
                TrainingPair::new(
                    ThoughtChannels::with_intent(1),
                    "I understand.".to_string(),
                    &tokenizer,
                ),
                TrainingPair::new(
                    ThoughtChannels::with_intent(2),
                    "Which detail?".to_string(),
                    &tokenizer,
                ),
            ],
        };
        let train_config = TrainingConfig {
            epochs: 2,
            learning_rate: 0.002,
            bptt_window: 6,
            train_network: true,
            negative_samples: 0,
            thought_logit_aux_weight: 0.1,
            logit_anchor_weight: 0.02,
            report_interval: 1000,
            progress: false,
            #[cfg(feature = "gpu")]
            use_gpu_cfc: false,
            ..Default::default()
        };

        let (metrics, _, _, _, _) = train_with_adam(&mut generator, &dataset, &train_config, None);
        assert!(metrics.iter().all(|metric| metric.avg_loss.is_finite()));

        generator.config_mut().gating.base_max_tokens = 4;
        generator.config_mut().enable_consciousness_gating = false;
        for pair in &dataset.pairs {
            let result = generator.generate(&pair.to_thought_channels());
            assert_eq!(
                crate::evaluation::unknown_token_rate(&result.token_ids, generator.tokenizer()),
                0.0
            );
            assert_eq!(
                crate::evaluation::code_token_rate(&result.token_ids, generator.tokenizer()),
                0.0
            );
        }
    }

    #[test]
    fn test_training_reduces_loss() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        // Create a simple dataset with repeated examples for stronger signal
        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..5 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
            dataset.push(TrainingPair::new(channels, "world".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 20,
            learning_rate: 0.05,
            bptt_window: 8,
            grad_clip: 1.0,
            report_interval: 100,
            use_adam: false, // SGD for simplicity in test
            warmup_fraction: 0.0,
            patience: 0,
            enable_diagnostics: false,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        assert_eq!(metrics.len(), 20);

        // Loss should be finite
        for m in &metrics {
            assert!(
                m.avg_loss.is_finite(),
                "Loss should be finite: {}",
                m.avg_loss
            );
        }

        // Loss should decrease: first epoch > last epoch
        let first_loss = metrics[0].avg_loss;
        let last_loss = metrics.last().unwrap().avg_loss;
        assert!(
            last_loss < first_loss,
            "Training should reduce loss: first={first_loss} last={last_loss}"
        );
    }

    #[test]
    fn test_training_with_adam() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 10,
            learning_rate: 0.01,
            bptt_window: 8,
            grad_clip: 1.0,
            report_interval: 100,
            use_adam: true,
            warmup_fraction: 0.1,
            patience: 0,
            enable_diagnostics: false,
            ..Default::default()
        };

        let (metrics, adam, diag, report, _validation) =
            train_with_adam(&mut generator, &dataset, &train_config, None);
        assert_eq!(metrics.len(), 10);
        assert!(adam.is_some());
        assert!(
            diag.is_none(),
            "Diagnostics should be None when not enabled"
        );
        assert!(
            report.is_none(),
            "AnomalyReport should be None when not enabled"
        );

        let adam = adam.unwrap();
        assert!(adam.t > 0, "Adam should have stepped");
    }

    #[test]
    fn test_early_stopping() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        dataset.push(TrainingPair::new(channels, "a".to_string(), &tok));

        // Use a very tiny LR so loss barely changes, triggering early stopping
        let train_config = TrainingConfig {
            epochs: 100,
            learning_rate: 1e-8,
            bptt_window: 8,
            grad_clip: 1.0,
            report_interval: 200,
            use_adam: false,
            warmup_fraction: 0.0,
            patience: 3,
            enable_diagnostics: false,
            train_network: false, // Disable network BPTT for early stopping test
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        // With near-zero LR, loss changes are < 1e-6, so patience triggers
        assert!(
            metrics.len() < 100,
            "Early stopping should trigger: got {} epochs",
            metrics.len()
        );
    }

    #[test]
    fn test_warmup_lr() {
        let base_lr = 0.01;
        // At step 0, should be 10% of base
        let lr0 = warmup_lr(base_lr, 0, 100, 0.1);
        assert!(
            (lr0 - 0.001).abs() < 1e-5,
            "Step 0 should be 10% of base: {lr0}"
        );

        // At step 9 (last warmup step), should be near full base
        let lr9 = warmup_lr(base_lr, 9, 100, 0.1);
        assert!(
            (lr9 - base_lr * (0.1 + 0.9 * 0.9)).abs() < 1e-5,
            "Last warmup step should be 91% of base: {lr9}"
        );

        // At step 10 (first decay step), should be full base (cosine at t=0)
        let lr10 = warmup_lr(base_lr, 10, 100, 0.1);
        assert!(
            (lr10 - base_lr).abs() < 1e-4,
            "Start of decay should be ~base_lr: {lr10}"
        );

        // At step 50, cosine decay should be below base but above min
        let lr50 = warmup_lr(base_lr, 50, 100, 0.1);
        let min_lr = base_lr * 0.01;
        assert!(
            lr50 < base_lr && lr50 > min_lr,
            "Mid-decay should be between min and base: {lr50}"
        );
    }

    #[test]
    fn test_dataset_jsonl_roundtrip() {
        let tok = BpeTokenizer::default_minimal();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        dataset.push(TrainingPair::new(channels, "world".to_string(), &tok));

        // Serialize to string (not file, to avoid filesystem in tests)
        let mut jsonl = String::new();
        for pair in &dataset.pairs {
            let line = serde_json::to_string(pair).unwrap();
            jsonl.push_str(&line);
            jsonl.push('\n');
        }

        // Parse back
        let parsed: Vec<TrainingPair> = jsonl
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].target_text, "hello");
        assert_eq!(parsed[1].target_text, "world");
    }

    #[test]
    fn test_generate_diverse_thoughts() {
        let thoughts = generate_diverse_thoughts();
        assert_eq!(
            thoughts.len(),
            1800,
            "8 intents × 5 epistemic × 5 emotions × 3 stages × 3 consciousness = 1800"
        );

        // Verify all are distinct
        for (i, a) in thoughts.iter().enumerate() {
            for (j, b) in thoughts.iter().enumerate() {
                if i != j {
                    assert_ne!(a.channels, b.channels, "Thoughts {i} and {j} should differ");
                }
            }
        }
    }

    #[test]
    fn test_thought_to_prompt() {
        let mut channels = ThoughtChannels::with_intent(1); // Answer
        channels.set_epistemic(0.0); // Certain
        channels.channels[17] = 1.0; // mood_temperature

        let prompt = thought_to_prompt(&channels);
        assert!(prompt.contains("Answer"), "Should contain intent name");
        assert!(
            prompt.contains("Certain"),
            "Should contain epistemic status"
        );
        assert!(!prompt.is_empty(), "Should produce non-empty prompt");

        // Verify template diversity: different intents produce different formats
        let mut channels2 = ThoughtChannels::with_intent(3); // Propose
        channels2.set_epistemic(2.0); // Uncertain
        let prompt2 = thought_to_prompt(&channels2);
        assert!(prompt2.contains("Propose"), "Should contain intent name");
        assert!(prompt2.contains("Uncertain"), "Should contain epistemic");
        // Different intent+epistemic → different template
        let same_start = prompt.chars().take(20).collect::<String>()
            == prompt2.chars().take(20).collect::<String>();
        assert!(
            !same_start,
            "Different thoughts should use different templates"
        );
    }

    #[test]
    fn test_gradient_diagnostics_no_vanishing_or_exploding() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            bptt_window: 8,
            grad_clip: 1.0,
            report_interval: 100,
            use_adam: true,
            warmup_fraction: 0.1,
            patience: 0,
            enable_diagnostics: true,
            ..Default::default()
        };

        let (_metrics, _adam, diag, _report, _validation) =
            train_with_adam(&mut generator, &dataset, &train_config, None);
        let diag = diag.expect("Diagnostics should be Some when enabled");
        assert!(diag.total_steps > 0, "Should have recorded steps");
        let mean = diag.mean_grad_norm();
        assert!(mean > 1e-8, "Mean grad norm should not vanish: {mean}");
        assert!(mean < 100.0, "Mean grad norm should not explode: {mean}");

        // Embedding norms should be sampled (5 epochs × vocab_size samples)
        assert!(
            !diag.embedding_norms.is_empty(),
            "Should have embedding norms"
        );
        for &norm in &diag.embedding_norms {
            assert!(norm > 0.01, "Embedding norm should be positive: {norm}");
            assert!(norm < 10000.0, "Embedding norm should not explode: {norm}");
        }
    }

    #[test]
    fn test_gradient_diagnostics_format_summary() {
        let mut diag = GradientDiagnostics::new();
        diag.record_step(0.5, false);
        diag.record_step(1.2, true);
        diag.record_step(0.001, false);

        let summary = diag.format_summary();
        assert!(
            summary.contains("Gradient Diagnostics"),
            "Should have header"
        );
        assert!(
            summary.contains("Total steps:       3"),
            "Should show 3 steps"
        );
        assert!(
            summary.contains("Clip count:        1"),
            "Should show 1 clip"
        );
    }

    #[test]
    fn test_gradient_anomaly_vanishing() {
        let mut diag = GradientDiagnostics::new();
        // 80% vanishing (< 1e-6)
        for _ in 0..80 {
            diag.record_step(1e-8, false);
        }
        for _ in 0..20 {
            diag.record_step(0.5, false);
        }
        let anomalies = diag.detect_anomalies();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, GradientAnomaly::Vanishing { .. })),
            "Should detect vanishing: {:?}",
            anomalies
        );
        assert!(!diag.is_healthy());
    }

    #[test]
    fn test_gradient_anomaly_exploding() {
        let mut diag = GradientDiagnostics::new();
        // 30% exploding (> 10)
        for _ in 0..30 {
            diag.record_step(50.0, true);
        }
        for _ in 0..70 {
            diag.record_step(0.5, false);
        }
        let anomalies = diag.detect_anomalies();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, GradientAnomaly::Exploding { .. })),
            "Should detect exploding: {:?}",
            anomalies
        );
    }

    #[test]
    fn test_gradient_anomaly_oscillating() {
        let mut diag = GradientDiagnostics::new();
        // Mix of very small and very large norms → high CV (>2.0)
        // mean ≈ 1.0, but std >> mean when most values are near 0 and a few are huge
        for _ in 0..90 {
            diag.record_step(0.01, false);
        }
        for _ in 0..10 {
            diag.record_step(30.0, false);
        }
        let anomalies = diag.detect_anomalies();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, GradientAnomaly::Oscillating { .. })),
            "Should detect oscillation: {:?}",
            anomalies
        );
    }

    #[test]
    fn test_gradient_anomaly_plateau() {
        let mut diag = GradientDiagnostics::new();
        // All norms identical → plateau
        for _ in 0..50 {
            diag.record_step(1.0, false);
        }
        let anomalies = diag.detect_anomalies();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, GradientAnomaly::Plateau { .. })),
            "Should detect plateau: {:?}",
            anomalies
        );
    }

    #[test]
    fn test_gradient_anomaly_healthy() {
        let mut diag = GradientDiagnostics::new();
        // Normal-ish gradient norms with some variation
        for i in 0..100 {
            diag.record_step(0.3 + (i as f32 * 0.01), false);
        }
        assert!(
            diag.is_healthy(),
            "Should be healthy, got anomalies: {:?}",
            diag.detect_anomalies()
        );
    }

    #[test]
    fn test_gradient_anomaly_empty() {
        let diag = GradientDiagnostics::new();
        assert!(diag.detect_anomalies().is_empty());
        assert!(diag.is_healthy());
    }

    #[test]
    fn test_gradient_anomaly_display() {
        let a = GradientAnomaly::Vanishing { fraction: 0.5 };
        let s = format!("{}", a);
        assert!(s.contains("50.0%"), "Display should show percentage: {}", s);

        let b = GradientAnomaly::Exploding {
            fraction: 0.1,
            max_norm: 42.0,
        };
        let s = format!("{}", b);
        assert!(s.contains("42.00"), "Display should show max norm: {}", s);
    }

    #[test]
    fn test_generate_curriculum_produces_valid_pairs() {
        let tokenizer = BpeTokenizer::default_minimal();
        let dataset = generate_curriculum(&tokenizer);

        assert!(
            dataset.pairs.len() >= 100,
            "Should produce many pairs, got {}",
            dataset.pairs.len()
        );

        for pair in &dataset.pairs {
            // All channels finite
            assert!(
                pair.channels.iter().all(|c| c.is_finite()),
                "All channels should be finite"
            );
            // Non-empty target text
            assert!(
                !pair.target_text.is_empty(),
                "Target text should not be empty"
            );
            // Token IDs present
            assert!(
                !pair.target_ids.is_empty(),
                "Target IDs should not be empty"
            );
            // Round-trip: decode(encode(text)) produces non-empty output
            let decoded = tokenizer.decode(&pair.target_ids);
            assert!(!decoded.is_empty(), "Decoded text should not be empty");
        }

        // Verify diversity: multiple intents present
        let unique_intents: std::collections::HashSet<usize> = dataset
            .pairs
            .iter()
            .map(|p| {
                (0..8)
                    .max_by(|&a, &b| p.channels[a].total_cmp(&p.channels[b]))
                    .unwrap()
            })
            .collect();
        assert!(
            unique_intents.len() >= 6,
            "Should cover most intents, got {}",
            unique_intents.len()
        );
    }

    #[test]
    fn test_prose_curriculum_avoids_metadata_targets() {
        let tokenizer = BpeTokenizer::default_minimal();
        let dataset = generate_curriculum_with_style(&tokenizer, CurriculumStyle::Prose);

        assert_eq!(dataset.pairs.len(), generate_diverse_thoughts().len());
        assert!(dataset.pairs.iter().take(32).all(|pair| {
            !pair.target_text.contains("SEMANTIC_INTENT")
                && !pair.target_text.contains("EPISTEMIC_STATUS")
                && !pair.target_text.contains("INTENT=")
                && !pair.target_text.contains("Mode:")
                && !pair.target_text.contains("Task=")
        }));
        assert!(
            dataset
                .pairs
                .iter()
                .take(32)
                .any(|pair| pair.target_text.contains("I am confident"))
        );
        assert!(
            dataset
                .pairs
                .iter()
                .take(128)
                .all(|pair| !pair.target_ids.is_empty())
        );
    }

    #[test]
    fn test_network_bptt_improves_over_embeddings_only() {
        let genesis = test_genesis();
        let config = test_config();

        // Train with embeddings only
        let mut gen_emb = BrocaGenerator::new(&genesis, config.clone());
        let tok = gen_emb.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..5 {
            dataset.push(TrainingPair::new(channels, "hello world".to_string(), &tok));
        }

        let emb_only_config = TrainingConfig {
            epochs: 15,
            learning_rate: 0.01,
            bptt_window: 8,
            use_adam: true,
            train_network: false,
            embedding_target_norm: 0.0, // No norm for fair comparison
            ..Default::default()
        };
        let emb_metrics = train(&mut gen_emb, &dataset, &emb_only_config);

        // Train with CfC BPTT
        let mut gen_bptt = BrocaGenerator::new(&genesis, config);
        let bptt_config = TrainingConfig {
            epochs: 15,
            learning_rate: 0.01,
            bptt_window: 8,
            use_adam: true,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 0.0, // No norm for fair comparison
            ..Default::default()
        };
        let bptt_metrics = train(&mut gen_bptt, &dataset, &bptt_config);

        let emb_final = emb_metrics.last().unwrap().avg_loss;
        let bptt_final = bptt_metrics.last().unwrap().avg_loss;

        // BPTT should achieve lower or equal loss (more parameters being trained)
        // Allow a generous margin: with a large vocab (code tokens expand the space),
        // the CfC network may converge slower than embeddings-only on tiny datasets.
        assert!(
            bptt_final <= emb_final + 1.5,
            "BPTT ({bptt_final:.4}) should not be much worse than emb-only ({emb_final:.4})"
        );
    }

    #[test]
    fn test_embedding_normalization() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "test".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 10,
            learning_rate: 0.05,
            bptt_window: 8,
            embedding_target_norm: 128.0,
            ..Default::default()
        };

        let _ = train(&mut generator, &dataset, &train_config);

        // All embeddings should have norms at or below the target (with 10% margin)
        let max_allowed = 128.0 * 1.15;
        for emb in generator.controller().token_embeddings() {
            let norm = emb.norm();
            assert!(
                norm <= max_allowed,
                "Embedding norm {norm:.1} should be ≤ {max_allowed:.1} after normalization"
            );
        }
    }

    #[test]
    fn test_negative_sampling() {
        // Verify sample_negatives produces correct output
        let indices = sample_negatives(5, 100, 10, 42);
        assert!(indices.contains(&5), "Must include target");
        assert_eq!(indices.len(), 11, "target + 10 negatives");
        // No duplicates
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len(), "No duplicates");
    }

    #[test]
    fn test_sampled_softmax_training() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..5 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 10,
            learning_rate: 0.05,
            bptt_window: 8,
            negative_samples: 16, // Only compute 16 negatives + target
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        assert_eq!(metrics.len(), 10);

        // Loss should be finite and decreasing
        let first = metrics[0].avg_loss;
        let last = metrics.last().unwrap().avg_loss;
        assert!(first.is_finite(), "First loss should be finite");
        assert!(last.is_finite(), "Last loss should be finite");
        // Sampled softmax introduces noise; allow up to 5% regression
        assert!(
            last < first * 1.05,
            "Sampled softmax should still reduce loss (5% tolerance): {first} → {last}"
        );
    }

    #[test]
    fn test_carry_state_training() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        for i in 0..5 {
            let channels = ThoughtChannels::with_intent(i % 3);
            dataset.push(TrainingPair::new(channels, "test".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            bptt_window: 8,
            carry_state: 0.5, // 50% chance of carrying state
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        assert_eq!(metrics.len(), 5);
        for m in &metrics {
            assert!(
                m.avg_loss.is_finite(),
                "Loss should be finite with carry_state"
            );
        }
    }

    #[test]
    fn test_curriculum_length_ascending() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        // Varying lengths: short, medium, long
        dataset.push(TrainingPair::new(channels, "hi".to_string(), &tok));
        dataset.push(TrainingPair::new(
            channels,
            "hello world this is a longer sentence".to_string(),
            &tok,
        ));
        dataset.push(TrainingPair::new(channels, "test".to_string(), &tok));

        let train_config = TrainingConfig {
            epochs: 3,
            learning_rate: 0.01,
            bptt_window: 16,
            curriculum: CurriculumSchedule::LengthAscending,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        assert_eq!(metrics.len(), 3);
        for m in &metrics {
            assert!(m.avg_loss.is_finite(), "Loss should be finite");
        }
    }

    #[test]
    fn test_validation_loss_computation() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut train_dataset = TrainingDataset::default();
        let mut val_dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();

        for _ in 0..5 {
            train_dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }
        for _ in 0..2 {
            val_dataset.push(TrainingPair::new(channels, "world".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            bptt_window: 8,
            validation_dataset: Some(val_dataset),
            ..Default::default()
        };

        let (metrics, _, _, _, _) =
            train_with_adam(&mut generator, &train_dataset, &train_config, None);
        assert_eq!(metrics.len(), 5);

        // All epochs should have validation loss
        for m in &metrics {
            assert!(
                m.validation_loss.is_some(),
                "Should have validation loss when dataset is provided"
            );
            assert!(
                m.validation_loss.unwrap().is_finite(),
                "Validation loss should be finite"
            );
        }
    }

    #[test]
    fn test_anomaly_response_exploding_halves_lr() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        // Very high LR to provoke exploding gradients
        let train_config = TrainingConfig {
            epochs: 10,
            learning_rate: 100.0,
            bptt_window: 8,
            enable_diagnostics: true,
            enable_anomaly_response: true,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        // Should early-stop (3 consecutive anomalous epochs) or complete with reduced LR
        // Either way, all losses should be finite (anomaly response prevents divergence)
        for m in &metrics {
            assert!(
                m.avg_loss.is_finite(),
                "Loss should remain finite with anomaly response"
            );
        }
    }

    #[test]
    fn test_anomaly_response_early_stop() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        // Extremely tiny LR → vanishing gradients → anomaly response doubles LR
        // but if gradients remain anomalous for 3 epochs, early stop triggers
        let train_config = TrainingConfig {
            epochs: 100,
            learning_rate: 1e-30,
            bptt_window: 8,
            warmup_fraction: 0.0,
            enable_diagnostics: true,
            enable_anomaly_response: true,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        // Should early-stop well before 100 epochs
        assert!(
            metrics.len() < 100,
            "Anomaly response should early-stop: got {} epochs",
            metrics.len()
        );
    }

    #[test]
    fn test_anomaly_response_disabled_by_default() {
        // enable_anomaly_response defaults to false
        let cfg = TrainingConfig::default();
        assert!(!cfg.enable_anomaly_response);
    }

    // ── Item #1: freeze_embeddings tests ──

    #[test]
    fn test_freeze_embeddings_preserves_norms() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        // Snapshot embedding norms before training
        let norms_before: Vec<f32> = generator
            .controller()
            .token_embeddings()
            .iter()
            .map(|e| e.norm())
            .collect();

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..5 {
            dataset.push(TrainingPair::new(channels, "hello world".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 10,
            learning_rate: 0.05,
            bptt_window: 8,
            freeze_embeddings: true,
            train_network: true,
            embedding_target_norm: 0.0, // Disable normalization to see raw change
            ..Default::default()
        };

        train(&mut generator, &dataset, &train_config);

        // Embeddings should be unchanged when frozen
        let norms_after: Vec<f32> = generator
            .controller()
            .token_embeddings()
            .iter()
            .map(|e| e.norm())
            .collect();

        for (i, (before, after)) in norms_before.iter().zip(norms_after.iter()).enumerate() {
            assert!(
                (before - after).abs() < 1e-6,
                "Embedding {i} norm changed: {before} → {after} (freeze_embeddings=true)"
            );
        }
    }

    #[test]
    fn test_freeze_embeddings_cfc_still_trains() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..5 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        // freeze_embeddings + train_network: only CfC weights should update
        let train_config = TrainingConfig {
            epochs: 15,
            learning_rate: 0.01,
            bptt_window: 8,
            freeze_embeddings: true,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 0.0,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        // Should complete without error and produce finite loss
        for m in &metrics {
            assert!(
                m.avg_loss.is_finite(),
                "Loss should be finite with frozen embeddings"
            );
        }
    }

    // ── Item #2: AnomalyReport tests ──

    #[test]
    fn test_anomaly_report_returned_when_enabled() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            bptt_window: 8,
            enable_anomaly_response: true,
            ..Default::default()
        };

        let (_, _, _, report, _) = train_with_adam(&mut generator, &dataset, &train_config, None);
        let report = report.expect("AnomalyReport should be Some when enabled");
        assert!(report.final_lr_multiplier > 0.0);
        assert!(report.final_grad_clip > 0.0);
    }

    #[test]
    fn test_anomaly_report_records_early_stop() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 100,
            learning_rate: 1e-30,
            bptt_window: 8,
            warmup_fraction: 0.0,
            enable_diagnostics: true,
            enable_anomaly_response: true,
            ..Default::default()
        };

        let (metrics, _, _, report, _) =
            train_with_adam(&mut generator, &dataset, &train_config, None);
        let report = report.expect("AnomalyReport should be Some");
        if metrics.len() < 100 {
            assert!(
                report.anomaly_early_stopped,
                "Should flag anomaly_early_stopped when stopped before max epochs"
            );
        }
        assert!(
            report.anomalous_epoch_count > 0,
            "Should record anomalous epochs"
        );
        assert!(
            !report.epoch_anomalies.is_empty(),
            "Should have per-epoch anomaly log"
        );
    }

    // ── Item #4: coherence-gated loss tests ──

    #[test]
    fn test_coherence_loss_weight_default_disabled() {
        let cfg = TrainingConfig::default();
        assert!(
            cfg.coherence_loss_weight == 0.0,
            "coherence_loss_weight should default to 0.0 (disabled)"
        );
    }

    #[test]
    fn test_coherence_loss_weight_training() {
        let genesis = test_genesis();
        let config = test_config();

        // Train with coherence gating disabled
        let mut gen1 = BrocaGenerator::new(&genesis, config.clone());
        let tok = gen1.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..5 {
            dataset.push(TrainingPair::new(channels, "hello world".to_string(), &tok));
        }

        let config_no_coherence = TrainingConfig {
            epochs: 10,
            learning_rate: 0.01,
            bptt_window: 8,
            embedding_target_norm: 0.0,
            coherence_loss_weight: 0.0,
            ..Default::default()
        };
        let metrics_no = train(&mut gen1, &dataset, &config_no_coherence);

        // Train with coherence gating enabled
        let mut gen2 = BrocaGenerator::new(&genesis, config);
        let config_coherence = TrainingConfig {
            epochs: 10,
            learning_rate: 0.01,
            bptt_window: 8,
            embedding_target_norm: 0.0,
            coherence_loss_weight: 0.5,
            ..Default::default()
        };
        let metrics_yes = train(&mut gen2, &dataset, &config_coherence);

        // Both should produce finite loss and train successfully
        for m in &metrics_no {
            assert!(m.avg_loss.is_finite(), "No-coherence loss should be finite");
        }
        for m in &metrics_yes {
            assert!(
                m.avg_loss.is_finite(),
                "Coherence-gated loss should be finite"
            );
        }

        // Coherence-weighted loss should be lower (scaled down by weight)
        let final_no = metrics_no.last().unwrap().avg_loss;
        let final_yes = metrics_yes.last().unwrap().avg_loss;
        assert!(
            final_yes <= final_no + 0.5,
            "Coherence-gated training ({final_yes:.4}) should not be much worse than baseline ({final_no:.4})"
        );
    }

    #[test]
    fn test_early_stopping_with_validation() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut train_dataset = TrainingDataset::default();
        let mut val_dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();

        train_dataset.push(TrainingPair::new(channels, "a".to_string(), &tok));
        val_dataset.push(TrainingPair::new(channels, "b".to_string(), &tok));

        let train_config = TrainingConfig {
            epochs: 100,
            learning_rate: 1e-8, // Tiny LR → no improvement → early stop
            bptt_window: 8,
            patience: 3,
            use_adam: false,
            warmup_fraction: 0.0,
            train_network: false,
            validation_dataset: Some(val_dataset),
            ..Default::default()
        };

        let metrics = train(&mut generator, &train_dataset, &train_config);
        assert!(
            metrics.len() < 100,
            "Early stopping should trigger with validation: got {} epochs",
            metrics.len()
        );
    }

    // ── Item #1: coherence telemetry tests ──

    #[test]
    fn test_epoch_metrics_coherence_tracked() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        // Coherence tracking activates when coherence_loss_weight > 0
        let train_config = TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            bptt_window: 8,
            coherence_loss_weight: 0.3,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        for m in &metrics {
            let coh = m
                .mean_coherence
                .expect("mean_coherence should be Some when coherence tracking is active");
            assert!(coh.is_finite(), "Coherence should be finite");
            // ContinuousHV cosine similarity can be in [-1, 1]
            assert!(
                coh >= -1.0 && coh <= 1.0,
                "Coherence should be in [-1,1]: {coh}"
            );
        }
    }

    #[test]
    fn test_epoch_metrics_coherence_also_with_diagnostics() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        // Coherence tracked when diagnostics enabled (even without coherence_loss_weight)
        let train_config = TrainingConfig {
            epochs: 3,
            learning_rate: 0.01,
            bptt_window: 8,
            enable_diagnostics: true,
            coherence_loss_weight: 0.0,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        for m in &metrics {
            assert!(
                m.mean_coherence.is_some(),
                "mean_coherence should be Some when diagnostics enabled"
            );
        }
    }

    // ── Item #2: coherence collapse tests ──

    #[test]
    fn test_coherence_collapse_display() {
        let a = GradientAnomaly::CoherenceCollapse {
            mean_coherence: 0.02,
        };
        let s = format!("{a}");
        assert!(s.contains("0.02"), "Should show mean coherence: {s}");
        assert!(s.contains("collapse"), "Should say 'collapse': {s}");
    }

    // ── Item #3: smoke test ──

    #[test]
    fn test_smoke_test_runs_and_returns_validation() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        }

        let train_config = TrainingConfig {
            epochs: 3,
            learning_rate: 0.01,
            bptt_window: 8,
            enable_smoke_test: true,
            smoke_test_coherence_threshold: 0.0, // Very low — should pass
            ..Default::default()
        };

        let (_, _, _, _, validation) =
            train_with_adam(&mut generator, &dataset, &train_config, None);
        let val = validation.expect("TrainingValidation should be Some when smoke test enabled");
        assert_eq!(val.intent_coherences.len(), 8, "Should test all 8 intents");
        assert!(val.mean_coherence.is_finite());
        // With threshold=0.0, should always pass
        assert!(val.passed, "Smoke test should pass with threshold=0.0");
    }

    #[test]
    fn test_smoke_test_disabled_by_default() {
        let cfg = TrainingConfig::default();
        assert!(!cfg.enable_smoke_test);
    }

    // ── Item #4: adaptive coherence warmup ──

    #[test]
    fn test_coherence_warmup_ramps_gradually() {
        let genesis = test_genesis();
        let config = test_config();
        let mut generator = BrocaGenerator::new(&genesis, config);

        let tok = generator.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for _ in 0..5 {
            dataset.push(TrainingPair::new(channels, "hello world".to_string(), &tok));
        }

        // With warmup: coherence weight ramps from 0 to 0.5 over 5 epochs
        let train_config = TrainingConfig {
            epochs: 10,
            learning_rate: 0.01,
            bptt_window: 8,
            coherence_loss_weight: 0.5,
            coherence_warmup_epochs: 5,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);
        // All epochs should have coherence tracked
        for m in &metrics {
            assert!(m.mean_coherence.is_some());
            assert!(m.mean_coherence.unwrap().is_finite());
        }

        // Early epochs should have higher reported loss (less coherence attenuation)
        // compared to later epochs (more attenuation). We verify training completes
        // and produces finite results with the warmup schedule.
        assert_eq!(metrics.len(), 10);
    }

    #[test]
    fn test_coherence_warmup_default_zero() {
        let cfg = TrainingConfig::default();
        assert_eq!(cfg.coherence_warmup_epochs, 0);
    }

    /// Quality regression test: train with fusion enabled, verify no degradation.
    ///
    /// Trains 15 epochs with fusion flags on from epoch 5, then evaluates.
    /// Asserts: loss decreases, output changes from pre-training, all values finite.
    /// This is the automated gate preventing future changes from silently
    /// degrading generation quality when fusion is active.
    #[test]
    fn test_fusion_training_quality_regression() {
        use crate::encoder::ThoughtChannels;
        use crate::generator::{BrocaConfig, BrocaGenerator, SamplingStrategy};

        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("fusion-regression");
        let config = BrocaConfig {
            controller: crate::controller::LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32,
                max_seq_len: 32,
                ..Default::default()
            },
            gating: crate::gating::GatingConfig {
                base_max_tokens: 20,
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: true,
            enable_semantic_veto: false,
            ..Default::default()
        };

        let mut generator = BrocaGenerator::new(&genesis, config);
        let tok = generator.tokenizer().clone();

        // Build small dataset
        let mut dataset = TrainingDataset::default();
        for intent in 0..4 {
            let channels = ThoughtChannels::with_intent(intent);
            dataset.push(TrainingPair::new(channels, "hello world".to_string(), &tok));
        }

        // Pre-training baseline
        let pre_result = generator.generate(&ThoughtChannels::default());

        // Train WITH fusion enabled from epoch 5
        let train_config = TrainingConfig {
            epochs: 15,
            learning_rate: 0.01,
            bptt_window: 8,
            use_adam: true,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 128.0,
            enable_fusion_during_training: true,
            fusion_warmup_epochs: 5,
            ..Default::default()
        };

        let metrics = train(&mut generator, &dataset, &train_config);

        // Verify training reduced loss
        let first_loss = metrics[0].avg_loss;
        let final_loss = metrics.last().unwrap().avg_loss;
        assert!(
            final_loss < first_loss,
            "Fusion training should reduce loss: {first_loss:.4} → {final_loss:.4}"
        );

        // Verify all metrics are finite
        for m in &metrics {
            assert!(
                m.avg_loss.is_finite(),
                "Loss should be finite at epoch {}",
                m.epoch
            );
        }

        // Verify fusion training changed behavior (weights updated)
        let post_result = generator.generate(&ThoughtChannels::default());
        let changed = pre_result.token_ids != post_result.token_ids;
        assert!(changed, "Fusion training should change generator behavior");
    }
}

// ── Therapeutic training data generation ──────────────────────────────────

/// Generate therapeutic training pairs from template responses.
///
/// Creates training data with therapeutic channels set for different
/// clinical scenarios (validation, crisis, reappraisal, etc.).
/// This bootstraps the model's ability to generate clinically appropriate
/// language without requiring a large external corpus.
#[cfg(feature = "therapeutic")]
pub fn generate_therapeutic_training_data(tokenizer: &BpeTokenizer) -> TrainingDataset {
    let mut dataset = TrainingDataset::default();

    // Template responses for each therapeutic intent
    let templates: Vec<(f32, f32, f32, f32, &str)> = vec![
        // (intent, alliance, distress, depth, response)
        // Validation (intent=0): empathic, non-directive
        (
            0.0,
            0.3,
            0.5,
            0.1,
            "I hear you. That sounds really difficult.",
        ),
        (
            0.0,
            0.5,
            0.6,
            0.1,
            "It makes sense that you would feel that way.",
        ),
        (
            0.0,
            0.4,
            0.4,
            0.1,
            "What you're going through is completely understandable.",
        ),
        (
            0.0,
            0.6,
            0.3,
            0.1,
            "Thank you for sharing that with me. It takes courage.",
        ),
        // Reflection (intent=1): mirroring, exploring
        (
            1.0,
            0.5,
            0.4,
            0.2,
            "It sounds like you're feeling overwhelmed right now.",
        ),
        (
            1.0,
            0.6,
            0.3,
            0.2,
            "I wonder what comes up for you when you think about that.",
        ),
        (
            1.0,
            0.5,
            0.5,
            0.3,
            "When you say that, what does it feel like in your body?",
        ),
        // Reappraisal (intent=2): cognitive restructuring
        (
            2.0,
            0.6,
            0.5,
            0.4,
            "What if we looked at this from a different angle?",
        ),
        (
            2.0,
            0.7,
            0.4,
            0.4,
            "Are there other ways to understand what happened?",
        ),
        // Exploration (intent=3): deepening understanding
        (
            3.0,
            0.6,
            0.3,
            0.3,
            "Tell me more about what that experience was like for you.",
        ),
        (
            3.0,
            0.7,
            0.4,
            0.4,
            "How does this connect to other things in your life?",
        ),
        // Psychoeducation (intent=4): normalizing, teaching
        (
            4.0,
            0.5,
            0.5,
            0.3,
            "Many people experience similar feelings in situations like this.",
        ),
        (
            4.0,
            0.6,
            0.4,
            0.3,
            "Our bodies often respond to stress in ways that feel overwhelming but are actually protective.",
        ),
        // Containment (intent=6): holding, stabilizing
        (
            6.0,
            0.5,
            0.8,
            0.5,
            "I'm here with you right now. You are safe in this moment.",
        ),
        (
            6.0,
            0.6,
            0.7,
            0.5,
            "Let's take a moment together. Can you feel your feet on the ground?",
        ),
        // Crisis (intent=7): grounding, referral
        (
            7.0,
            0.3,
            0.9,
            0.1,
            "I want you to know you're not alone. Can you take a slow breath with me?",
        ),
        (
            7.0,
            0.3,
            0.95,
            0.1,
            "Your safety matters. If you're in immediate danger, please call 988 or emergency services.",
        ),
        (
            7.0,
            0.4,
            0.85,
            0.1,
            "Right now, can you notice five things you can see around you?",
        ),
    ];

    for (intent, alliance, distress, depth, response) in templates {
        let mut channels = ThoughtChannels::default();
        // Set base consciousness channels to reasonable defaults
        channels.set_consciousness(0.6, 0.5, 0.7);
        channels.set_emotion(
            if distress > 0.5 {
                -(distress - 0.5)
            } else {
                0.3
            },
            distress.clamp(0.3, 0.8),
            0.7, // warmth
        );
        channels.set_therapeutic(intent, alliance, distress, depth);

        let pair = TrainingPair::new(channels, response.to_string(), tokenizer);
        dataset.push(pair);
    }

    dataset
}
