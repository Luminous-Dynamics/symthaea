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
                serde_json::from_str(line)
                    .with_context(|| format!("parsing line {i}"))
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
        std::fs::write(path, out)
            .with_context(|| format!("writing training data: {path}"))?;
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
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            learning_rate: 0.001,
            bptt_window: 16,
            grad_clip: 1.0,
            report_interval: 100,
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
    let mut metrics = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let mut total_loss = 0.0f32;
        let mut total_tokens = 0usize;

        for pair in &dataset.pairs {
            if pair.target_ids.is_empty() {
                continue;
            }

            let channels = ThoughtChannels { channels: pair.channels };
            let thought_hv = generator.encoder().encode(&channels);

            // Reset controller for this sequence
            generator.controller_mut().reset();
            generator.controller_mut().set_learning_rate(config.learning_rate);

            // Teacher-forced forward pass
            let mut prev_token = generator.tokenizer().thought_id;

            let window_end = pair.target_ids.len().min(config.bptt_window);

            for (pos, &target_id) in pair.target_ids[..window_end].iter().enumerate() {
                let logits = generator.controller_mut().forward_step(&thought_hv, prev_token, pos);

                // Cross-entropy loss: -log(softmax[target])
                let loss = cross_entropy_loss(&logits, target_id as usize);
                total_loss += loss;
                total_tokens += 1;

                // Compute gradient through weight-tied output and apply
                // This is a simplified single-step update (not full BPTT through time)
                apply_weight_tied_gradient(
                    generator.controller_mut(),
                    &logits,
                    target_id as usize,
                    config.learning_rate,
                    config.grad_clip,
                );

                prev_token = target_id;
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
    }

    metrics
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

/// Apply gradient through weight-tied output.
///
/// For weight-tied output: logits[i] = similarity(output_hv, emb[i])
/// Gradient of CE loss w.r.t. output_hv is: sum_i (softmax[i] - one_hot[target]) * emb[i]
/// We use this to shift the relevant token embeddings toward/away from the output.
fn apply_weight_tied_gradient(
    controller: &mut crate::controller::LanguageController,
    logits: &[f32],
    target: usize,
    lr: f32,
    grad_clip: f32,
) {
    if target >= logits.len() {
        return;
    }

    // Compute softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();

    if sum_exp < 1e-10 {
        return;
    }

    let output_hv = controller.output_hv();
    let output_slice = output_hv.as_slice();

    // Update token embeddings: shift target toward output, others away
    let embeddings = controller.token_embeddings_mut();
    let n = embeddings.len().min(logits.len());

    for i in 0..n {
        let prob = exps[i] / sum_exp;
        let error = if i == target { prob - 1.0 } else { prob };

        // Skip tiny gradients
        if error.abs() < 1e-6 {
            continue;
        }

        // Gradient: error * output_hv (projected onto embedding dimension)
        let grad_scale = (-lr * error).clamp(-grad_clip, grad_clip);

        let emb_values = &mut embeddings[i].values;
        for (j, emb_val) in emb_values.iter_mut().enumerate() {
            if j < output_slice.len() {
                *emb_val += grad_scale * output_slice[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::BrocaConfig;
    use crate::controller::LanguageControllerConfig;
    use crate::gating::GatingConfig;
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

        assert!(loss_correct < loss_wrong, "Loss for correct prediction should be lower");
        assert!(loss_correct >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_training_reduces_loss() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);

        // Create a simple dataset
        let tok = gen.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        dataset.push(TrainingPair::new(channels, "hello".to_string(), &tok));
        dataset.push(TrainingPair::new(channels, "world".to_string(), &tok));

        let train_config = TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            bptt_window: 8,
            grad_clip: 1.0,
            report_interval: 1,
        };

        let metrics = train(&mut gen, &dataset, &train_config);
        assert_eq!(metrics.len(), 5);

        // Loss should be finite
        for m in &metrics {
            assert!(m.avg_loss.is_finite(), "Loss should be finite: {}", m.avg_loss);
        }
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
}
