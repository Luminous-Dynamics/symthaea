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
    pub fn train(hvs: &[Hdv], labels: &[String], cfg: TrainConfig) -> Self {
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

/// A reproducible split that keeps every provenance group entirely in one arm.
///
/// This is appropriate when each file/host/incident group has exactly one
/// ground-truth class. It refuses mixed-label groups rather than silently
/// leaking related events across train and test.
#[derive(Debug)]
pub struct GroupStratifiedSplit {
    pub train_hvs: Vec<Hdv>,
    pub train_labels: Vec<String>,
    pub train_group_ids: Vec<String>,
    pub test_hvs: Vec<Hdv>,
    pub test_labels: Vec<String>,
    pub test_group_ids: Vec<String>,
}

/// Split by class and provenance group, never by individual event.
///
/// Every group ID must belong to exactly one class and each class must have
/// at least two distinct groups. A class with only one source group cannot
/// support a group-disjoint train/test estimate and is reported as an error.
pub fn group_stratified_split(
    hvs: &[Hdv],
    labels: &[String],
    group_ids: &[String],
    train_frac: f32,
    seed: u64,
) -> Result<GroupStratifiedSplit, String> {
    if hvs.len() != labels.len() || labels.len() != group_ids.len() {
        return Err("hypervectors, labels, and group IDs must have equal length".into());
    }
    if hvs.is_empty() {
        return Err("cannot split an empty dataset".into());
    }
    if !train_frac.is_finite() || train_frac <= 0.0 || train_frac >= 1.0 {
        return Err("train fraction must be finite and strictly between 0 and 1".into());
    }

    let mut group_labels: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    let mut by_label: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, Vec<usize>>,
    > = Default::default();

    for (index, (label, group_id)) in labels.iter().zip(group_ids.iter()).enumerate() {
        if group_id.trim().is_empty() {
            return Err(format!("empty provenance group ID at sample {index}"));
        }
        if let Some(existing_label) = group_labels.get(group_id) {
            if existing_label != label {
                return Err(format!(
                    "provenance group {group_id:?} contains multiple labels"
                ));
            }
        } else {
            group_labels.insert(group_id.clone(), label.clone());
        }
        by_label
            .entry(label.clone())
            .or_default()
            .entry(group_id.clone())
            .or_default()
            .push(index);
    }

    let mut state = seed.max(1);
    let mut split = GroupStratifiedSplit {
        train_hvs: Vec::new(),
        train_labels: Vec::new(),
        train_group_ids: Vec::new(),
        test_hvs: Vec::new(),
        test_labels: Vec::new(),
        test_group_ids: Vec::new(),
    };

    for (label, groups) in by_label {
        if groups.len() < 2 {
            return Err(format!(
                "class {label:?} has {} provenance group(s); at least two are required",
                groups.len()
            ));
        }

        let mut groups: Vec<(String, Vec<usize>)> = groups.into_iter().collect();
        // Shuffle whole provenance groups, not individual events.
        for i in (1..groups.len()).rev() {
            let j = (xorshift(&mut state) as usize) % (i + 1);
            groups.swap(i, j);
        }
        let n_train = ((groups.len() as f32) * train_frac).round() as usize;
        let n_train = n_train.clamp(1, groups.len() - 1);

        for (group_index, (group_id, sample_indices)) in groups.into_iter().enumerate() {
            let is_train = group_index < n_train;
            for sample_index in sample_indices {
                if is_train {
                    split.train_hvs.push(hvs[sample_index].clone());
                    split.train_labels.push(label.clone());
                    split.train_group_ids.push(group_id.clone());
                } else {
                    split.test_hvs.push(hvs[sample_index].clone());
                    split.test_labels.push(label.clone());
                    split.test_group_ids.push(group_id.clone());
                }
            }
        }
    }

    if split.train_hvs.is_empty() || split.test_hvs.is_empty() {
        return Err("group split produced an empty train or test partition".into());
    }

    Ok(split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode;
    use crate::fixtures::generate_synthetic_corpus;

    #[test]
    fn grouped_split_keeps_each_provenance_group_in_one_partition() {
        use std::collections::HashSet;

        let hvs = vec![
            vec![1, 1],
            vec![1, -1],
            vec![1, 1],
            vec![-1, 1],
            vec![-1, -1],
            vec![1, -1],
            vec![-1, -1],
            vec![-1, 1],
        ];
        let labels = vec![
            "A".to_string(), "A".to_string(), "A".to_string(), "A".to_string(),
            "B".to_string(), "B".to_string(), "B".to_string(), "B".to_string(),
        ];
        let groups = vec![
            "a-file-1".to_string(), "a-file-1".to_string(),
            "a-file-2".to_string(), "a-file-2".to_string(),
            "b-file-1".to_string(), "b-file-1".to_string(),
            "b-file-2".to_string(), "b-file-2".to_string(),
        ];

        let split = group_stratified_split(&hvs, &labels, &groups, 0.5, 0xFACADE).unwrap();
        let train_groups: HashSet<_> = split.train_group_ids.iter().collect();
        let test_groups: HashSet<_> = split.test_group_ids.iter().collect();

        assert!(train_groups.is_disjoint(&test_groups));
        assert_eq!(split.train_hvs.len() + split.test_hvs.len(), hvs.len());
        assert!(split.train_labels.iter().any(|label| label == "A"));
        assert!(split.train_labels.iter().any(|label| label == "B"));
        assert!(split.test_labels.iter().any(|label| label == "A"));
        assert!(split.test_labels.iter().any(|label| label == "B"));
    }

    #[test]
    fn grouped_split_rejects_a_group_with_multiple_labels() {
        let hvs = vec![vec![1], vec![-1]];
        let labels = vec!["benign".to_string(), "malicious".to_string()];
        let groups = vec!["same-file".to_string(), "same-file".to_string()];

        let error = group_stratified_split(&hvs, &labels, &groups, 0.8, 7).unwrap_err();
        assert!(error.contains("multiple labels"));
    }

    #[test]
    fn grouped_split_rejects_single_group_classes() {
        let hvs = vec![vec![1], vec![-1], vec![1]];
        let labels = vec!["A".to_string(), "A".to_string(), "B".to_string()];
        let groups = vec!["a-1".to_string(), "a-2".to_string(), "b-1".to_string()];

        let error = group_stratified_split(&hvs, &labels, &groups, 0.8, 7).unwrap_err();
        assert!(error.contains("at least two are required"));
    }

    #[test]
    fn grouped_split_is_deterministic_for_a_fixed_seed() {
        let hvs = vec![vec![1], vec![-1], vec![1], vec![-1], vec![1], vec![-1], vec![1], vec![-1]];
        let labels = vec![
            "A".to_string(), "A".to_string(), "A".to_string(), "A".to_string(),
            "B".to_string(), "B".to_string(), "B".to_string(), "B".to_string(),
        ];
        let groups = vec![
            "a1".to_string(), "a1".to_string(), "a2".to_string(), "a2".to_string(),
            "b1".to_string(), "b1".to_string(), "b2".to_string(), "b2".to_string(),
        ];

        let first = group_stratified_split(&hvs, &labels, &groups, 0.5, 17).unwrap();
        let second = group_stratified_split(&hvs, &labels, &groups, 0.5, 17).unwrap();
        assert_eq!(first.train_group_ids, second.train_group_ids);
        assert_eq!(first.test_group_ids, second.test_group_ids);
    }

    #[test]
    fn probe_learns_synthetic_classes() {
        let corpus = generate_synthetic_corpus(20, 0xC0FFEE);
        let hvs: Vec<Hdv> = corpus.iter().map(encode).collect();
        let labels: Vec<String> = corpus.iter().map(|e| e.label.clone().unwrap()).collect();

        let (train_x, train_y, test_x, test_y) = stratified_split(&hvs, &labels, 0.8, 0x1234);

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
