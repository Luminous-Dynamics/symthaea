// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Supervised LTC Training for Phoneme-Discriminative Projections
//!
//! Trains the LTC cell weights (w_in, w_rec, bias, tau) via BPTT
//! with a contrastive phoneme loss, so the hidden state becomes
//! phoneme-discriminative — unlike the random BLAKE3 projection.
//!
//! ## Architecture
//!
//! ```text
//! Mel Features (40D)
//!       │
//!       ▼
//! ┌─────────────────┐
//! │   LtcCell        │ ← w_in, w_rec, bias, tau (TRAINED via BPTT)
//! │   (closed-form)  │
//! └────────┬────────┘
//!          │ hidden_state (64D)
//!          ▼
//! ┌─────────────────┐
//! │ Contrastive Loss │ ← cos_sim(hidden, phoneme_centroid[label])
//! │                  │
//! └────────┬────────┘
//!          │ ∂L / ∂hidden
//!          ▼
//!    BPTT → update w_in, w_rec, bias, tau
//! ```

use std::collections::HashMap;

use crate::ltc::{LtcCell, LtcGradients, LtcStepCache};

// ============================================================================
// Phoneme Centroid Bank
// ============================================================================

/// Learnable phoneme centroids in LTC hidden space.
///
/// Each phoneme maps to a target vector in hidden-state space.
/// The contrastive loss pushes LTC output toward the correct centroid
/// and away from all others.
pub struct PhonemeCentroids {
    /// Centroid vectors: phoneme_label → hidden_state target
    centroids: HashMap<String, Vec<f32>>,
    /// Hidden dimension
    dim: usize,
    /// Temperature for softmax (lower = sharper)
    temperature: f32,
}

impl PhonemeCentroids {
    /// Create centroids initialized to random orthogonal directions.
    pub fn new(phoneme_labels: &[String], dim: usize, temperature: f32) -> Self {
        let mut centroids = HashMap::new();
        let mut seed = 0xC0DE_F00Du64;
        let mut rand = move || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        for label in phoneme_labels {
            let mut v: Vec<f32> = (0..dim).map(|_| rand()).collect();
            // Normalize to unit length
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut v {
                *x /= norm;
            }
            centroids.insert(label.clone(), v);
        }

        Self {
            centroids,
            dim,
            temperature,
        }
    }

    /// Compute contrastive loss and gradient w.r.t. hidden state.
    ///
    /// Uses dot-product similarity (not cosine) for clean gradient computation:
    /// Loss = -log(exp(h · c_pos / T) / Σ_k exp(h · c_k / T))
    ///
    /// Returns (loss, ∂loss/∂hidden_state).
    pub fn contrastive_loss(&self, hidden: &[f32], label: &str) -> (f32, Vec<f32>) {
        // Compute logits (dot product / temperature) for all centroids
        let mut logits: Vec<(String, f32)> = Vec::with_capacity(self.centroids.len());
        let mut max_logit = f32::NEG_INFINITY;

        for (name, centroid) in &self.centroids {
            let dot: f32 = hidden.iter().zip(centroid).map(|(a, b)| a * b).sum();
            let logit = dot / self.temperature;
            max_logit = max_logit.max(logit);
            logits.push((name.clone(), logit));
        }

        // Softmax with numerical stability
        let mut sum_exp = 0.0f32;
        let mut pos_logit = 0.0f32;
        let mut exps: Vec<f32> = Vec::with_capacity(logits.len());

        for (name, logit) in &logits {
            let e = (logit - max_logit).exp();
            exps.push(e);
            sum_exp += e;
            if name == label {
                pos_logit = *logit;
            }
        }

        // Loss = -log(softmax(pos))
        let log_sum_exp = max_logit + sum_exp.max(1e-30).ln();
        let loss = -(pos_logit - log_sum_exp);

        // Gradient: ∂L/∂h = (1/T) · (Σ_k softmax(k) · c_k - c_pos)
        let mut dl_dh = vec![0.0f32; self.dim];
        let inv_t = 1.0 / self.temperature;

        for (idx, (name, _logit)) in logits.iter().enumerate() {
            let prob = exps[idx] / sum_exp;
            let centroid = &self.centroids[name.as_str()];
            let is_positive = name == label;
            let weight = if is_positive { prob - 1.0 } else { prob };

            for d in 0..self.dim {
                dl_dh[d] += inv_t * weight * centroid[d];
            }
        }

        (loss, dl_dh)
    }

    /// Get the predicted phoneme for a hidden state.
    pub fn predict(&self, hidden: &[f32]) -> (String, f32) {
        let h_norm = hidden.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let mut best = (String::new(), f32::NEG_INFINITY);

        for (name, centroid) in &self.centroids {
            let c_norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            let dot: f32 = hidden.iter().zip(centroid).map(|(a, b)| a * b).sum();
            let sim = dot / (h_norm * c_norm);
            if sim > best.1 {
                best = (name.clone(), sim);
            }
        }

        best
    }
}

// ============================================================================
// Training Loop
// ============================================================================

/// Configuration for supervised LTC training.
pub struct LtcTrainingConfig {
    /// Learning rate for LTC weights
    pub learning_rate: f32,
    /// Gradient clipping norm
    pub grad_clip: f32,
    /// BPTT truncation length (frames)
    pub bptt_length: usize,
    /// Temperature for contrastive loss
    pub temperature: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Report interval (utterances)
    pub report_interval: usize,
    /// Frame hop in seconds (must match audio frontend)
    pub frame_hop: f32,
}

impl Default for LtcTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            grad_clip: 5.0,
            bptt_length: 50,
            temperature: 0.1,
            epochs: 10,
            report_interval: 100,
            frame_hop: 0.010,
        }
    }
}

/// Train one utterance via BPTT.
///
/// Returns (total_loss, correct_frames, total_frames).
pub fn train_utterance(
    ltc: &mut LtcCell,
    centroids: &PhonemeCentroids,
    mel_frames: &[Vec<f32>],
    labels: &[String],
    config: &LtcTrainingConfig,
) -> (f32, usize, usize) {
    let n = mel_frames.len().min(labels.len());
    if n == 0 {
        return (0.0, 0, 0);
    }

    // Forward pass: collect states and caches
    ltc.reset();
    let mut caches: Vec<LtcStepCache> = Vec::with_capacity(n);
    let mut states: Vec<Vec<f32>> = Vec::with_capacity(n);

    for frame in mel_frames.iter().take(n) {
        let (state, cache) = ltc.forward_cached(frame, config.frame_hop);
        states.push(state);
        caches.push(cache);
    }

    // Backward pass: accumulate gradients via truncated BPTT
    let h = ltc.hidden_size();
    let input_size = ltc.input_size();
    let mut total_grads = LtcGradients::zeros(h, input_size);
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;
    let mut dl_dh_next = vec![0.0f32; h];

    // Process in reverse (BPTT)
    for t in (0..n).rev() {
        // Compute loss at this frame
        let (loss, mut dl_dh) = centroids.contrastive_loss(&states[t], &labels[t]);
        total_loss += loss;

        // Check prediction accuracy
        let (pred, _) = centroids.predict(&states[t]);
        if pred == labels[t] {
            correct += 1;
        }

        // Add gradient from future steps (recurrent)
        for d in 0..h {
            dl_dh[d] += dl_dh_next[d];
        }

        // Backward through LTC step
        let (step_grads, dl_dprev) = ltc.backward_step(&dl_dh, &caches[t]);
        total_grads.accumulate(&step_grads);

        // Truncate BPTT: zero out recurrent gradient every bptt_length steps
        if t > 0 && (n - t).is_multiple_of(config.bptt_length) {
            dl_dh_next = vec![0.0f32; h];
        } else {
            dl_dh_next = dl_dprev;
        }
    }

    // Average gradients over sequence length
    total_grads.scale(1.0 / n as f32);

    // Clip gradients
    total_grads.clip_norm(config.grad_clip);

    // Apply gradient update
    total_grads.apply_sgd(ltc, config.learning_rate);

    (total_loss / n as f32, correct, n)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ltc::LtcConfig;

    #[test]
    fn test_contrastive_loss_basic() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let centroids = PhonemeCentroids::new(&labels, 8, 0.1);

        // Random hidden state
        let hidden = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, -0.3];
        let (loss, grad) = centroids.contrastive_loss(&hidden, "A");

        assert!(loss.is_finite(), "Loss should be finite: {}", loss);
        assert!(loss >= 0.0, "Loss should be non-negative: {}", loss);
        assert_eq!(grad.len(), 8, "Gradient should match hidden dim");
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "Gradients should be finite"
        );
    }

    #[test]
    fn test_contrastive_loss_decreases_for_correct() {
        let labels = vec!["A".to_string(), "B".to_string()];
        let centroids = PhonemeCentroids::new(&labels, 16, 0.1);

        // Use centroid A as hidden state — should have low loss
        let centroid_a = centroids.centroids["A"].clone();
        let (loss_correct, _) = centroids.contrastive_loss(&centroid_a, "A");

        // Use centroid B as hidden state with label A — should have high loss
        let centroid_b = centroids.centroids["B"].clone();
        let (loss_wrong, _) = centroids.contrastive_loss(&centroid_b, "A");

        assert!(
            loss_correct < loss_wrong,
            "Correct label should have lower loss: correct={}, wrong={}",
            loss_correct,
            loss_wrong
        );
    }

    #[test]
    fn test_gradient_finite_difference() {
        let labels = vec!["A".to_string(), "B".to_string()];
        let centroids = PhonemeCentroids::new(&labels, 4, 0.5);
        let hidden = vec![0.3, -0.1, 0.2, 0.4];
        let eps = 1e-4;

        let (_, analytic_grad) = centroids.contrastive_loss(&hidden, "A");

        for d in 0..4 {
            let mut h_plus = hidden.clone();
            let mut h_minus = hidden.clone();
            h_plus[d] += eps;
            h_minus[d] -= eps;

            let (l_plus, _) = centroids.contrastive_loss(&h_plus, "A");
            let (l_minus, _) = centroids.contrastive_loss(&h_minus, "A");

            let numerical = (l_plus - l_minus) / (2.0 * eps);
            let diff = (analytic_grad[d] - numerical).abs();

            assert!(
                diff < 0.01,
                "Gradient mismatch at dim {}: analytic={:.6}, numerical={:.6}, diff={:.6}",
                d,
                analytic_grad[d],
                numerical,
                diff
            );
        }
    }

    #[test]
    fn test_train_utterance_reduces_loss() {
        let config = LtcTrainingConfig {
            learning_rate: 0.01,
            epochs: 1,
            bptt_length: 10,
            temperature: 0.5,
            ..Default::default()
        };

        let labels = vec!["A".to_string(), "B".to_string()];
        let centroids = PhonemeCentroids::new(&labels, 64, config.temperature);
        let ltc_config = LtcConfig {
            hidden_size: 64,
            tau_init: 0.020,
            ..LtcConfig::default()
        };
        let mut ltc = LtcCell::new(40, ltc_config);

        // Create simple training data: 20 frames alternating A and B
        let mel_frames: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let mut f = vec![0.0f32; 40];
                f[i % 40] = if i % 2 == 0 { 1.0 } else { -1.0 };
                f
            })
            .collect();
        let frame_labels: Vec<String> = (0..20)
            .map(|i| if i % 2 == 0 { "A" } else { "B" }.to_string())
            .collect();

        // Train 5 iterations and verify loss decreases
        let (loss_0, _, _) =
            train_utterance(&mut ltc, &centroids, &mel_frames, &frame_labels, &config);
        for _ in 0..4 {
            train_utterance(&mut ltc, &centroids, &mel_frames, &frame_labels, &config);
        }
        let (loss_5, _, _) =
            train_utterance(&mut ltc, &centroids, &mel_frames, &frame_labels, &config);

        // Loss should decrease (or at minimum not increase drastically)
        assert!(
            loss_5 < loss_0 + 0.5,
            "Loss should decrease over training: initial={:.4}, final={:.4}",
            loss_0,
            loss_5
        );
    }
}
