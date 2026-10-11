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

/// Assign whole source groups to deterministic stratified folds.
///
/// Every group receives exactly one fold, and every label must have at least
/// `k` distinct groups so each fold has examples of every class.
/// This function does not examine feature vectors, making fold assignment
/// independently testable and reusable by benchmark runners.
pub fn grouped_fold_assignment(
    labels: &[String],
    groups: &[String],
    k: usize,
    seed: u64,
) -> Result<Vec<usize>, String> {
    if labels.len() != groups.len() {
        return Err(format!(
            "length mismatch: labels={}, groups={}",
            labels.len(),
            groups.len()
        ));
    }
    if labels.is_empty() {
        return Err("cannot assign folds for an empty corpus".to_string());
    }
    if k < 2 {
        return Err("fold count must be at least two".to_string());
    }
    if labels.iter().any(|label| label.is_empty()) || groups.iter().any(|group| group.is_empty()) {
        return Err("labels and group IDs must be non-empty".to_string());
    }

    let mut by_label: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, Vec<usize>>,
    > = Default::default();
    let mut group_labels: std::collections::BTreeMap<&str, &str> = Default::default();
    for (index, (label, group)) in labels.iter().zip(groups.iter()).enumerate() {
        if let Some(existing) = group_labels.get(group.as_str()) {
            if *existing != label.as_str() {
                return Err(format!(
                    "group {group:?} contains multiple labels: {existing:?} and {label:?}"
                ));
            }
        } else {
            group_labels.insert(group.as_str(), label.as_str());
        }
        by_label
            .entry(label.clone())
            .or_default()
            .entry(group.clone())
            .or_default()
            .push(index);
    }

    let mut state = seed.max(1);
    let mut fold_of = vec![usize::MAX; labels.len()];
    for (label, group_map) in by_label {
        let mut group_ids: Vec<String> = group_map.keys().cloned().collect();
        if group_ids.len() < k {
            return Err(format!(
                "label {label:?} has {} groups; need at least {k} for grouped {k}-fold CV",
                group_ids.len()
            ));
        }
        for i in (1..group_ids.len()).rev() {
            let j = (xorshift(&mut state) as usize) % (i + 1);
            group_ids.swap(i, j);
        }
        for (position, group_id) in group_ids.iter().enumerate() {
            for &index in &group_map[group_id] {
                fold_of[index] = position % k;
            }
        }
    }

    if fold_of.iter().any(|fold| *fold == usize::MAX) {
        return Err("fold assignment left one or more events unassigned".to_string());
    }
    Ok(fold_of)
}

/// Grouped stratified train/test split for leakage-resistant evaluation.
///
/// A group is indivisible: all events from one source file, capture, or scenario
/// are assigned to exactly one side. Every group must have a single label, and
/// every label must have at least two groups so both train and test contain that
/// class. Groups are sorted by their stable string ID before deterministic
/// seeded shuffling; input row order therefore does not affect group assignment.
pub fn grouped_stratified_split(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
    train_frac: f32,
    seed: u64,
) -> Result<(Vec<Hdv>, Vec<String>, Vec<Hdv>, Vec<String>), String> {
    if hvs.len() != labels.len() || labels.len() != groups.len() {
        return Err(format!(
            "length mismatch: hvs={}, labels={}, groups={}",
            hvs.len(),
            labels.len(),
            groups.len()
        ));
    }
    if hvs.is_empty() {
        return Err("cannot split an empty corpus".to_string());
    }
    if !train_frac.is_finite() || train_frac <= 0.0 || train_frac >= 1.0 {
        return Err("train_frac must be finite and strictly between 0 and 1".to_string());
    }
    if labels.iter().any(|label| label.is_empty()) || groups.iter().any(|group| group.is_empty()) {
        return Err("labels and group IDs must be non-empty".to_string());
    }

    // label -> group ID -> row indices; BTreeMap makes the seed's meaning
    // independent of HashMap iteration order.
    let mut by_label: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, Vec<usize>>,
    > = Default::default();
    let mut group_labels: std::collections::BTreeMap<&str, &str> = Default::default();
    for (index, (label, group)) in labels.iter().zip(groups.iter()).enumerate() {
        if let Some(existing) = group_labels.get(group.as_str()) {
            if *existing != label.as_str() {
                return Err(format!(
                    "group {group:?} contains multiple labels: {existing:?} and {label:?}"
                ));
            }
        } else {
            group_labels.insert(group.as_str(), label.as_str());
        }
        by_label
            .entry(label.clone())
            .or_default()
            .entry(group.clone())
            .or_default()
            .push(index);
    }

    let mut state = seed.max(1);
    let mut train_hvs = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_hvs = Vec::new();
    let mut test_labels = Vec::new();

    for (label, group_map) in by_label {
        let mut group_ids: Vec<String> = group_map.keys().cloned().collect();
        if group_ids.len() < 2 {
            return Err(format!(
                "label {label:?} has {} distinct groups; need at least two for a grouped train/test split",
                group_ids.len()
            ));
        }
        for i in (1..group_ids.len()).rev() {
            let j = (xorshift(&mut state) as usize) % (i + 1);
            group_ids.swap(i, j);
        }
        let n_train_groups = ((group_ids.len() as f32 * train_frac).floor() as usize)
            .clamp(1, group_ids.len() - 1);
        let train_group_count = n_train_groups;

        for (position, group_id) in group_ids.iter().enumerate() {
            let indices = &group_map[group_id];
            if position < train_group_count {
                for &index in indices {
                    train_hvs.push(hvs[index].clone());
                    train_labels.push(label.clone());
                }
            } else {
                for &index in indices {
                    test_hvs.push(hvs[index].clone());
                    test_labels.push(label.clone());
                }
            }
        }
    }

    if train_hvs.is_empty() || test_hvs.is_empty() {
        return Err("grouped split produced an empty train or test partition".to_string());
    }
    Ok((train_hvs, train_labels, test_hvs, test_labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode;
    use crate::fixtures::generate_synthetic_corpus;

    #[test]
    fn grouped_split_keeps_each_group_on_one_side_and_every_label_on_both() {
        let hvs = vec![vec![1], vec![2], vec![3], vec![4], vec![5], vec![6], vec![7], vec![8]];
        let labels = vec![
            "a".to_string(), "a".to_string(), "a".to_string(), "a".to_string(),
            "b".to_string(), "b".to_string(), "b".to_string(), "b".to_string(),
        ];
        let groups = vec![
            "a-1".to_string(), "a-1".to_string(), "a-2".to_string(), "a-2".to_string(),
            "b-1".to_string(), "b-1".to_string(), "b-2".to_string(), "b-2".to_string(),
        ];
        let (tx, ty, vx, vy) =
            grouped_stratified_split(&hvs, &labels, &groups, 0.5, 7).expect("valid grouped split");

        // Reconstruct group membership from unique sample values; every pair in a
        // group has adjacent IDs and must remain on the same side.
        let train_values: std::collections::BTreeSet<i8> = tx.iter().map(|x| x[0]).collect();
        let test_values: std::collections::BTreeSet<i8> = vx.iter().map(|x| x[0]).collect();
        for (left, right) in [(1, 2), (3, 4), (5, 6), (7, 8)] {
            assert!(
                (train_values.contains(&left) && train_values.contains(&right))
                    || (test_values.contains(&left) && test_values.contains(&right)),
                "group containing values {left} and {right} was split"
            );
        }
        assert!(train_values.is_disjoint(&test_values));
        assert!(
            ["a", "b"]
                .iter()
                .all(|label| ty.iter().any(|actual| actual.as_str() == *label))
        );
        assert!(
            ["a", "b"]
                .iter()
                .all(|label| vy.iter().any(|actual| actual.as_str() == *label))
        );
    }

    #[test]
    fn grouped_split_is_deterministic() {
        let hvs = vec![vec![1], vec![2], vec![3], vec![4], vec![5], vec![6]];
        let labels = vec![
            "a".to_string(),
            "a".into(),
            "a".into(),
            "a".into(),
            "b".into(),
            "b".into(),
        ];
        let groups = vec![
            "a1".into(),
            "a2".into(),
            "a3".into(),
            "a4".into(),
            "b1".into(),
            "b2".into(),
        ];
        let a = grouped_stratified_split(&hvs, &labels, &groups, 0.5, 11).unwrap();
        let b = grouped_stratified_split(&hvs, &labels, &groups, 0.5, 11).unwrap();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
        assert_eq!(a.3, b.3);
    }

    #[test]
    fn grouped_split_rejects_group_with_conflicting_labels() {
        let hvs = vec![vec![1], vec![2], vec![3], vec![4]];
        let labels = vec!["a".into(), "b".into(), "a".into(), "b".into()];
        let groups = vec!["same".into(), "same".into(), "a2".into(), "b2".into()];
        let error = grouped_stratified_split(&hvs, &labels, &groups, 0.5, 1).unwrap_err();
        assert!(error.contains("multiple labels"));
    }

    #[test]
    fn grouped_split_rejects_label_with_only_one_group() {
        let hvs = vec![vec![1], vec![2], vec![3]];
        let labels = vec!["a".into(), "a".into(), "b".into()];
        let groups = vec!["a1".into(), "a1".into(), "b1".into()];
        let error = grouped_stratified_split(&hvs, &labels, &groups, 0.5, 1).unwrap_err();
        assert!(error.contains("need at least two"));
    }

    #[test]
    fn grouped_split_rejects_invalid_parameters_and_lengths() {
        let hvs = vec![vec![1], vec![2]];
        let labels = vec!["a".into(), "a".into()];
        let groups = vec!["a1".into(), "a2".into()];
        assert!(grouped_stratified_split(&hvs, &labels, &groups[..1], 0.5, 1).is_err());
        assert!(grouped_stratified_split(&hvs, &labels, &groups, 1.0, 1).is_err());
        assert!(grouped_stratified_split(&hvs, &labels, &groups, f32::NAN, 1).is_err());
    }

    #[test]
    fn grouped_fold_assignment_keeps_groups_intact_and_stratifies_every_fold() {
        let labels = vec![
            "a".into(), "a".into(), "a".into(), "a".into(),
            "a".into(), "a".into(), "b".into(), "b".into(),
            "b".into(), "b".into(), "b".into(), "b".into(),
        ];
        let groups = vec![
            "a1".into(), "a1".into(), "a2".into(), "a2".into(), "a3".into(), "a3".into(),
            "b1".into(), "b1".into(), "b2".into(), "b2".into(), "b3".into(), "b3".into(),
        ];
        let folds = grouped_fold_assignment(&labels, &groups, 3, 0xC0FFEE).unwrap();
        for i in 0..groups.len() {
            for j in 0..groups.len() {
                if groups[i] == groups[j] {
                    assert_eq!(folds[i], folds[j], "group {} was split", groups[i]);
                }
            }
        }
        for label in ["a", "b"] {
            for fold in 0..3 {
                assert!(
                    labels.iter().zip(folds.iter()).any(|(actual, assigned)| {
                        actual.as_str() == label && *assigned == fold
                    }),
                    "label {label} missing from fold {fold}"
                );
            }
        }
    }

    #[test]
    fn grouped_fold_assignment_is_deterministic() {
        let labels = vec!["a".into(), "a".into(), "a".into(), "b".into(), "b".into(), "b".into()];
        let groups = vec![
            "a1".into(),
            "a2".into(),
            "a3".into(),
            "b1".into(),
            "b2".into(),
            "b3".into(),
        ];
        let first = grouped_fold_assignment(&labels, &groups, 3, 99).unwrap();
        let repeated = grouped_fold_assignment(&labels, &groups, 3, 99).unwrap();
        assert_eq!(first, repeated);

        let labels_reversed: Vec<String> = labels.iter().rev().cloned().collect();
        let groups_reversed: Vec<String> = groups.iter().rev().cloned().collect();
        let folds_reversed =
            grouped_fold_assignment(&labels_reversed, &groups_reversed, 3, 99).unwrap();
        let original_by_group: std::collections::BTreeMap<String, usize> = groups
            .iter()
            .cloned()
            .zip(first.iter().copied())
            .collect();
        let reversed_by_group: std::collections::BTreeMap<String, usize> = groups_reversed
            .iter()
            .cloned()
            .zip(folds_reversed.iter().copied())
            .collect();
        assert_eq!(original_by_group, reversed_by_group);
    }

    #[test]
    fn grouped_fold_assignment_rejects_ambiguous_or_undersized_groups() {
        let labels = vec!["a".into(), "b".into(), "a".into(), "b".into()];
        let mixed_groups = vec!["same".into(), "same".into(), "a2".into(), "b2".into()];
        assert!(
            grouped_fold_assignment(&labels, &mixed_groups, 2, 1)
                .unwrap_err()
                .contains("multiple labels")
        );

        let small_labels = vec!["a".into(), "a".into(), "b".into(), "b".into()];
        let small_groups = vec!["a1".into(), "a2".into(), "b1".into(), "b1".into()];
        assert!(
            grouped_fold_assignment(&small_labels, &small_groups, 2, 1)
                .unwrap_err()
                .contains("need at least")
        );
        assert!(grouped_fold_assignment(&labels, &mixed_groups, 1, 1).is_err());
        assert!(grouped_fold_assignment(&labels, &mixed_groups[..3], 2, 1).is_err());
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
