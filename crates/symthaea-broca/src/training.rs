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

use crate::checkpoint::AdamState;
use crate::encoder::ThoughtChannels;
use crate::generator::BrocaGenerator;
use crate::tokenizer::BpeTokenizer;

/// A single training pair: thought channels + target text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPair {
    pub channels: [f32; 20],
    pub target_text: String,
    #[serde(default)]
    pub target_ids: Vec<u32>,
}

impl TrainingPair {
    /// Create a new training pair, encoding the target text with the tokenizer.
    pub fn new(channels: ThoughtChannels, target_text: String, tokenizer: &BpeTokenizer) -> Self {
        let target_ids = tokenizer.encode(&target_text);
        Self {
            channels: channels.channels,
            target_text,
            target_ids,
        }
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
    pub fn tokenize_all(&mut self, tokenizer: &BpeTokenizer) {
        for pair in &mut self.pairs {
            if pair.target_ids.is_empty() {
                pair.target_ids = tokenizer.encode(&pair.target_text);
            }
        }
    }
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// BPTT truncation window (tokens).
    pub bptt_window: usize,
    /// Gradient clipping threshold.
    pub grad_clip: f32,
    /// Report loss every N steps.
    pub report_interval: usize,
    /// Use Adam optimizer (if false, uses SGD).
    pub use_adam: bool,
    /// Warmup fraction (0.0 to 1.0). First N% of steps use linear LR ramp.
    pub warmup_fraction: f32,
    /// Early stopping patience (0 = disabled). Stop if no improvement for N epochs.
    pub patience: usize,
    /// Enable gradient flow diagnostics (tracking norms, clipping, vanishing/exploding).
    pub enable_diagnostics: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            learning_rate: 0.001,
            bptt_window: 16,
            grad_clip: 1.0,
            report_interval: 100,
            use_adam: true,
            warmup_fraction: 0.1,
            patience: 0,
            enable_diagnostics: false,
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
}

/// Compute the effective learning rate with warmup.
///
/// During the warmup phase (first `warmup_fraction` of total steps),
/// LR ramps linearly from 0.1x to 1.0x base LR.
fn warmup_lr(base_lr: f32, step: usize, total_steps: usize, warmup_fraction: f32) -> f32 {
    let warmup_steps = (total_steps as f32 * warmup_fraction) as usize;
    if warmup_steps == 0 || step >= warmup_steps {
        base_lr
    } else {
        let progress = step as f32 / warmup_steps as f32;
        base_lr * (0.1 + 0.9 * progress)
    }
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
    train_with_adam(generator, dataset, config, None).0
}

/// Train with optional Adam optimizer state.
///
/// Returns (metrics, final AdamState if Adam was used, optional GradientDiagnostics).
pub fn train_with_adam(
    generator: &mut BrocaGenerator,
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    mut adam_state: Option<AdamState>,
) -> (
    Vec<EpochMetrics>,
    Option<AdamState>,
    Option<GradientDiagnostics>,
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
    let mut diagnostics = if config.enable_diagnostics {
        Some(GradientDiagnostics::new())
    } else {
        None
    };

    for epoch in 0..config.epochs {
        let mut total_loss = 0.0f32;
        let mut total_tokens = 0usize;

        for pair in &dataset.pairs {
            if pair.target_ids.is_empty() {
                continue;
            }

            let channels = ThoughtChannels {
                channels: pair.channels,
            };
            let thought_hv = generator.encoder().encode(&channels);

            // Reset controller for this sequence
            generator.controller_mut().reset();

            // Teacher-forced forward pass
            let mut prev_token = generator.tokenizer().thought_id;

            let window_end = pair.target_ids.len().min(config.bptt_window);

            for (pos, &target_id) in pair.target_ids[..window_end].iter().enumerate() {
                let lr = warmup_lr(
                    config.learning_rate,
                    global_step,
                    total_steps,
                    config.warmup_fraction,
                );
                generator.controller_mut().set_learning_rate(lr);

                let logits = generator
                    .controller_mut()
                    .forward_step(&thought_hv, prev_token, pos);

                // Cross-entropy loss: -log(softmax[target])
                let loss = cross_entropy_loss(&logits, target_id as usize);
                total_loss += loss;
                total_tokens += 1;

                // Apply gradient update
                let (grad_norm, was_clipped) = if config.use_adam {
                    apply_weight_tied_gradient_adam(
                        generator.controller_mut(),
                        &logits,
                        target_id as usize,
                        lr,
                        config.grad_clip,
                        adam_state.as_mut().unwrap(),
                    )
                } else {
                    apply_weight_tied_gradient(
                        generator.controller_mut(),
                        &logits,
                        target_id as usize,
                        lr,
                        config.grad_clip,
                    )
                };

                if let Some(ref mut diag) = diagnostics {
                    diag.record_step(grad_norm, was_clipped);
                }

                prev_token = target_id;
                global_step += 1;
            }
        }

        let avg_loss = if total_tokens > 0 {
            total_loss / total_tokens as f32
        } else {
            0.0
        };

        metrics.push(EpochMetrics {
            epoch,
            avg_loss,
            num_tokens: total_tokens,
            num_pairs: dataset.len(),
        });

        if (epoch + 1) % config.report_interval.max(1) == 0 || epoch == 0 {
            tracing::info!(
                epoch = epoch,
                avg_loss = avg_loss,
                tokens = total_tokens,
                "Broca training epoch"
            );
        }

        // Record embedding norms at end of each epoch
        if let Some(ref mut diag) = diagnostics {
            diag.record_embedding_norms(generator.controller().token_embeddings());
        }

        // Early stopping check
        if config.patience > 0 {
            if avg_loss < best_loss - 1e-6 {
                best_loss = avg_loss;
                patience_counter = 0;
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
    }

    (metrics, adam_state, diagnostics)
}

/// Cross-entropy loss for a single position.
fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    if target >= logits.len() {
        return 0.0;
    }

    // Numerically stable softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_softmax_target = (logits[target] - max_logit) - sum_exp.ln();

    -log_softmax_target
}

/// Apply gradient through weight-tied output (SGD).
///
/// For weight-tied output: logits[i] = similarity(output_hv, emb[i])
/// Gradient of CE loss w.r.t. output_hv is: sum_i (softmax[i] - one_hot[target]) * emb[i]
/// We use this to shift the relevant token embeddings toward/away from the output.
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

    // Update token embeddings: shift target toward output, others away
    let embeddings = controller.token_embeddings_mut();
    let n = embeddings.len().min(logits.len());

    let mut sum_sq = 0.0f32;
    let mut was_clipped = false;

    for i in 0..n {
        let prob = exps[i] / sum_exp;
        let error = if i == target { prob - 1.0 } else { prob };

        // Skip tiny gradients
        if error.abs() < 1e-6 {
            continue;
        }

        sum_sq += error * error;

        // Gradient: error * output_hv (projected onto embedding dimension)
        let raw = -lr * error;
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

    let embeddings = controller.token_embeddings_mut();
    let n = embeddings.len().min(logits.len());

    let mut sum_sq = 0.0f32;
    let mut was_clipped = false;

    for i in 0..n {
        let prob = exps[i] / sum_exp;
        let error = if i == target { prob - 1.0 } else { prob };

        if error.abs() < 1e-6 {
            continue;
        }

        sum_sq += error * error;

        // Compute per-dimension gradient: error * output_hv
        let dim = embeddings[i].values.len().min(output_slice.len());
        let grad: Vec<f32> = (0..dim)
            .map(|j| {
                let raw = error * output_slice[j];
                let clamped = raw.clamp(-grad_clip, grad_clip);
                if (clamped - raw).abs() > 1e-10 {
                    // Note: can't mutate was_clipped in closure directly,
                    // but we'll check post-hoc
                }
                clamped
            })
            .collect();

        // Check if any gradient was actually clipped
        for j in 0..dim {
            let raw = error * output_slice[j];
            if raw.abs() > grad_clip {
                was_clipped = true;
                break;
            }
        }

        // Apply Adam step if state exists for this embedding
        if i < adam.m.len() {
            let update = adam.step(i, &grad, lr);
            let emb_values = &mut embeddings[i].values;
            for (j, delta) in update.iter().enumerate() {
                if j < emb_values.len() {
                    emb_values[j] -= delta;
                }
            }
        }
    }

    (sum_sq.sqrt(), was_clipped)
}

/// Generate a diverse set of ThoughtChannels for training data collection.
///
/// 8 intents x 5 epistemic x 3 emotional clusters x 3 relationship stages = 360 configs.
/// Each has distinct channel encodings covering the full combinatorial space.
pub fn generate_diverse_thoughts() -> Vec<ThoughtChannels> {
    let mut thoughts = Vec::with_capacity(360);

    // Emotional clusters: (valence, arousal, warmth)
    let emotions = [
        (0.7, 0.3, 0.8),  // Calm-warm
        (-0.3, 0.7, 0.4), // Tense-cool
        (0.5, 0.5, 0.6),  // Neutral-balanced
    ];

    // Relationship stages
    let stages = [0.0, 3.0, 6.0]; // New, Established, Deep

    for intent in 0..8 {
        for epistemic in 0..5 {
            for &(valence, arousal, warmth) in &emotions {
                for &stage in &stages {
                    let mut channels = ThoughtChannels::with_intent(intent);
                    channels.set_epistemic(epistemic as f32);
                    channels.set_emotion(valence, arousal, warmth);
                    channels.channels[15] = stage; // relationship_stage
                    channels.channels[16] = 0.5; // trust: mid
                    channels.channels[17] = 1.0; // mood_temperature: neutral
                    channels.channels[12] = 0.5; // psi: mid
                    channels.channels[13] = 0.5; // meta_awareness: mid
                    channels.channels[14] = 0.5; // coherence: mid
                    thoughts.push(channels);
                }
            }
        }
    }

    thoughts
}

/// Reconstruct a text prompt from ThoughtChannels for LLM distillation.
///
/// Produces markers in the format that `channels_from_prompt()` in ssm_backend.rs can parse.
pub fn thought_to_prompt(channels: &ThoughtChannels) -> String {
    // Find the active intent
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

    format!(
        "SEMANTIC_INTENT: {}\nEPISTEMIC_STATUS: {}\nMOOD_TEMPERATURE: {:.2}\n",
        intent_names[active_intent], epistemic_names[epistemic_idx], channels.channels[17],
    )
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
        assert_eq!(pair.channels.len(), 20);
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
    fn test_training_reduces_loss() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);

        // Create a simple dataset with repeated examples for stronger signal
        let tok = gen.tokenizer().clone();
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
        };

        let metrics = train(&mut gen, &dataset, &train_config);
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
        let mut gen = BrocaGenerator::new(&genesis, config);

        let tok = gen.tokenizer().clone();
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
        };

        let (metrics, adam, diag) = train_with_adam(&mut gen, &dataset, &train_config, None);
        assert_eq!(metrics.len(), 10);
        assert!(adam.is_some());
        assert!(
            diag.is_none(),
            "Diagnostics should be None when not enabled"
        );

        let adam = adam.unwrap();
        assert!(adam.t > 0, "Adam should have stepped");
    }

    #[test]
    fn test_early_stopping() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);

        let tok = gen.tokenizer().clone();
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
        };

        let metrics = train(&mut gen, &dataset, &train_config);
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

        // At step 10 (end of warmup), should be full base
        let lr10 = warmup_lr(base_lr, 10, 100, 0.1);
        assert!(
            (lr10 - base_lr).abs() < 1e-5,
            "After warmup should be full base: {lr10}"
        );

        // At step 50, should be full base
        let lr50 = warmup_lr(base_lr, 50, 100, 0.1);
        assert!((lr50 - base_lr).abs() < 1e-5);
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
            360,
            "8 intents x 5 epistemic x 3 emotions x 3 stages = 360"
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
        assert!(
            prompt.contains("MOOD_TEMPERATURE"),
            "Should contain mood temp marker"
        );
    }

    #[test]
    fn test_gradient_diagnostics_no_vanishing_or_exploding() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);

        let tok = gen.tokenizer().clone();
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
        };

        let (_metrics, _adam, diag) = train_with_adam(&mut gen, &dataset, &train_config, None);
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
}
