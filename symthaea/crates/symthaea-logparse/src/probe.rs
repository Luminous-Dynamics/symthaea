//! Minimal one-vs-rest logistic regression over bipolar HDC hypervectors.
//!
//! Pure Rust, no autodiff, no ML framework. Used by the supervised probe to
//! measure the encoder's learned-linear ceiling — the honest upper bound that
//! nearest-centroid can't reach because it averages instead of learning
//! per-dimension weights.
//!
//! The encoder produces bipolar {-1, +1} vectors; we treat them as f32 inputs
//! and train a dense weight matrix `W ∈ R^{n_classes × D}` plus bias
//! `b ∈ R^{n_classes}`. One-vs-rest cross-entropy with L2 regularization.
//!
//! Design constraints (why no linfa, candle, etc.):
//!   - Phase 1 spike; we want zero new dependencies
//!   - 16,384D × 8 classes × 1000 events is small enough for plain SGD
//!   - We need to be able to read the gradient update in 40 lines of code,
//!     not trust a framework's defaults

use crate::encoder::Hdv;

/// Trained one-vs-rest logistic regression.
pub struct LogisticProbe {
    pub weights: Vec<Vec<f32>>, // [n_classes][D]
    pub biases: Vec<f32>,       // [n_classes]
    pub class_names: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub l2: f32,
    pub batch_size: usize,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            learning_rate: 0.05,
            l2: 1e-4,
            batch_size: 32,
            seed: 0xBEEFu64,
        }
    }
}

/// Xorshift for deterministic shuffling.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl LogisticProbe {
    /// Train one-vs-rest logistic regression.
    ///
    /// `hvs[i]` has label `labels[i]`. Labels are arbitrary strings; the
    /// probe learns one binary classifier per distinct label.
    pub fn train(
        hvs: &[Hdv],
        labels: &[String],
        cfg: TrainConfig,
    ) -> Self {
        assert_eq!(hvs.len(), labels.len());
        assert!(!hvs.is_empty());
        let dim = hvs[0].len();

        // Gather class names in sorted order (deterministic).
        let mut class_set: Vec<String> = {
            let mut s: std::collections::HashSet<String> = Default::default();
            for l in labels {
                s.insert(l.clone());
            }
            s.into_iter().collect()
        };
        class_set.sort();
        let n_classes = class_set.len();

        // One-hot labels per sample: 1.0 for its class, 0.0 for others.
        let class_idx: std::collections::HashMap<String, usize> = class_set
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        let targets: Vec<usize> = labels.iter().map(|l| class_idx[l]).collect();

        // Xavier-ish init: weights ~ U[-1/√D, 1/√D]
        let mut state = cfg.seed.max(1);
        let scale = 1.0 / (dim as f32).sqrt();
        let mut weights: Vec<Vec<f32>> = (0..n_classes)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        let r = xorshift(&mut state);
                        let u = (r as f32 / u64::MAX as f32) * 2.0 - 1.0;
                        u * scale
                    })
                    .collect()
            })
            .collect();
        let mut biases = vec![0.0f32; n_classes];

        // Simple SGD with mini-batches. Shuffle indices each epoch.
        let n = hvs.len();
        let mut order: Vec<usize> = (0..n).collect();

        for epoch in 0..cfg.epochs {
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = (xorshift(&mut state) as usize) % (i + 1);
                order.swap(i, j);
            }

            let mut epoch_loss = 0.0f32;
            let mut batch_start = 0;
            while batch_start < n {
                let batch_end = (batch_start + cfg.batch_size).min(n);
                let batch: &[usize] = &order[batch_start..batch_end];

                // Accumulate gradients across the batch, per class.
                let mut grad_w = vec![vec![0.0f32; dim]; n_classes];
                let mut grad_b = vec![0.0f32; n_classes];

                for &i in batch {
                    let x = &hvs[i];
                    let y = targets[i];

                    // Per-class logits, then sigmoid, then gradient.
                    for c in 0..n_classes {
                        // logit = W[c] · x + b[c]
                        // x is i8 bipolar, cast on the fly
                        let mut logit = biases[c];
                        let wc = &weights[c];
                        for (xi, &wi) in x.iter().zip(wc.iter()) {
                            logit += (*xi as f32) * wi;
                        }
                        let p = sigmoid(logit);
                        let target = if c == y { 1.0 } else { 0.0 };
                        let err = p - target; // ∂loss/∂logit for BCE
                        epoch_loss += if target > 0.5 {
                            -((p + 1e-9).ln())
                        } else {
                            -((1.0 - p + 1e-9).ln())
                        };

                        // ∂loss/∂W[c] = err * x
                        for (gi, &xi) in grad_w[c].iter_mut().zip(x.iter()) {
                            *gi += err * (xi as f32);
                        }
                        grad_b[c] += err;
                    }
                }

                // Apply averaged gradients with L2.
                let bs = (batch_end - batch_start) as f32;
                let lr = cfg.learning_rate;
                for c in 0..n_classes {
                    let wc = &mut weights[c];
                    let gc = &grad_w[c];
                    for (wi, gi) in wc.iter_mut().zip(gc.iter()) {
                        *wi -= lr * (gi / bs + cfg.l2 * *wi);
                    }
                    biases[c] -= lr * grad_b[c] / bs;
                }

                batch_start = batch_end;
            }

            if epoch == 0 || epoch == cfg.epochs - 1 || (epoch + 1) % 10 == 0 {
                eprintln!(
                    "  epoch {:>3}: loss={:.4}",
                    epoch + 1,
                    epoch_loss / (n * n_classes) as f32
                );
            }
        }

        Self {
            weights,
            biases,
            class_names: class_set,
        }
    }

    /// Predict the most-likely class index for a single hypervector.
    pub fn predict(&self, hv: &Hdv) -> usize {
        let n_classes = self.class_names.len();
        let mut best_c = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..n_classes {
            let mut logit = self.biases[c];
            for (xi, &wi) in hv.iter().zip(self.weights[c].iter()) {
                logit += (*xi as f32) * wi;
            }
            if logit > best_score {
                best_score = logit;
                best_c = c;
            }
        }
        best_c
    }

    /// Predict a batch, returning class indices (not names).
    pub fn predict_batch(&self, hvs: &[Hdv]) -> Vec<usize> {
        hvs.iter().map(|h| self.predict(h)).collect()
    }

    /// Accuracy on a labeled set.
    pub fn accuracy(&self, hvs: &[Hdv], labels: &[String]) -> f32 {
        assert_eq!(hvs.len(), labels.len());
        if hvs.is_empty() {
            return f32::NAN;
        }
        let class_idx: std::collections::HashMap<&str, usize> = self
            .class_names
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();
        let mut correct = 0usize;
        for (hv, lbl) in hvs.iter().zip(labels.iter()) {
            let pred = self.predict(hv);
            if class_idx.get(lbl.as_str()) == Some(&pred) {
                correct += 1;
            }
        }
        correct as f32 / hvs.len() as f32
    }
}

/// Stratified train/test split by label.
pub fn stratified_split(
    hvs: &[Hdv],
    labels: &[String],
    train_frac: f32,
    seed: u64,
) -> (Vec<Hdv>, Vec<String>, Vec<Hdv>, Vec<String>) {
    assert_eq!(hvs.len(), labels.len());
    let mut by_label: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for (i, l) in labels.iter().enumerate() {
        by_label.entry(l.clone()).or_default().push(i);
    }

    let mut state = seed.max(1);
    let mut train_hvs = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_hvs = Vec::new();
    let mut test_labels = Vec::new();

    for (label, mut indices) in by_label {
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = (xorshift(&mut state) as usize) % (i + 1);
            indices.swap(i, j);
        }
        let n_train = ((indices.len() as f32) * train_frac) as usize;
        for (k, &idx) in indices.iter().enumerate() {
            if k < n_train {
                train_hvs.push(hvs[idx].clone());
                train_labels.push(label.clone());
            } else {
                test_hvs.push(hvs[idx].clone());
                test_labels.push(label.clone());
            }
        }
    }
    (train_hvs, train_labels, test_hvs, test_labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode;
    use crate::fixtures::generate_synthetic_corpus;

    #[test]
    fn probe_learns_synthetic_classes() {
        let corpus = generate_synthetic_corpus(20, 0xC0FFEE);
        let hvs: Vec<Hdv> = corpus.iter().map(encode).collect();
        let labels: Vec<String> = corpus.iter().map(|e| e.label.clone().unwrap()).collect();

        let (train_x, train_y, test_x, test_y) =
            stratified_split(&hvs, &labels, 0.8, 0x1234);

        let cfg = TrainConfig {
            epochs: 30,
            learning_rate: 0.1,
            l2: 1e-4,
            batch_size: 16,
            seed: 0xFEED,
        };
        let probe = LogisticProbe::train(&train_x, &train_y, cfg);
        let acc = probe.accuracy(&test_x, &test_y);
        assert!(
            acc >= 0.80,
            "probe should learn synthetic classes with >=0.80 accuracy, got {acc}"
        );
    }
}
