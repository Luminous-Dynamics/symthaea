// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Random Forest secondary structure predictor.
//!
//! Analogous to `ml_mass.rs` in symthaea-nuclear, but for classification:
//! - Split criterion: Gini impurity (not variance)
//! - Leaf prediction: majority class H/E/C (not mean value)
//! - Same bootstrap + feature subsampling strategy
//!
//! Training pipeline:
//! 1. Load CB513 subset
//! 2. For each residue: extract 15-D feature vector
//! 3. RF trains to predict actual SS class from features
//! 4. Features include Chou-Fasman propensity, which encodes the baseline
//!
//! Expected Q3: ~60-70% on held-out proteins (vs ~55% Chou-Fasman baseline).
//!
//! Reference: Breiman, L. (2001). Random forests. *Machine Learning* 45, 5-32.

use serde::{Deserialize, Serialize};

use crate::amino_acid::{AminoAcid, SecondaryStructure};
use crate::cb513::load_cb513_subset;
use crate::chou_fasman::{predict_chou_fasman, q3_accuracy};
use crate::features::{N_FEATURES, extract_features};

/// Number of SS classes (H, E, C).
const N_CLASSES: usize = 3;

/// A single node in a classification decision tree.
#[derive(Debug, Clone)]
enum TreeNode {
    Leaf {
        class: u8,
        counts: [usize; N_CLASSES],
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    fn predict(&self, features: &[f64; N_FEATURES]) -> u8 {
        match self {
            TreeNode::Leaf { class, .. } => *class,
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if features[*feature] <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }

    fn leaf_counts(&self, features: &[f64; N_FEATURES]) -> [usize; N_CLASSES] {
        match self {
            TreeNode::Leaf { counts, .. } => *counts,
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if features[*feature] <= *threshold {
                    left.leaf_counts(features)
                } else {
                    right.leaf_counts(features)
                }
            }
        }
    }
}

fn gini(counts: &[usize; N_CLASSES], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    1.0 - counts.iter().map(|&c| (c as f64 / t).powi(2)).sum::<f64>()
}

fn majority_class(counts: &[usize; N_CLASSES]) -> u8 {
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i as u8)
        .unwrap_or(2)
}

fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Random Forest classifier for 3-class secondary structure prediction.
#[derive(Debug, Clone)]
pub struct RandomForestClassifier {
    trees: Vec<TreeNode>,
}

impl RandomForestClassifier {
    /// Train a random forest on labelled feature vectors.
    pub fn train(
        features: &[[f64; N_FEATURES]],
        labels: &[u8],
        n_trees: usize,
        max_depth: usize,
    ) -> Self {
        let n = features.len();
        assert_eq!(n, labels.len());
        let min_samples = 3;
        let mut trees = Vec::with_capacity(n_trees);
        let mut rng_state: u64 = 42;

        for _ in 0..n_trees {
            let indices: Vec<usize> = (0..n)
                .map(|_| {
                    rng_state = lcg_next(rng_state);
                    (rng_state >> 33) as usize % n
                })
                .collect();

            let boot_features: Vec<_> = indices.iter().map(|&i| features[i]).collect();
            let boot_labels: Vec<_> = indices.iter().map(|&i| labels[i]).collect();

            let tree = Self::build_tree(
                &boot_features,
                &boot_labels,
                0,
                max_depth,
                min_samples,
                &mut rng_state,
            );
            trees.push(tree);
        }

        Self { trees }
    }

    fn build_tree(
        features: &[[f64; N_FEATURES]],
        labels: &[u8],
        depth: usize,
        max_depth: usize,
        min_samples: usize,
        rng: &mut u64,
    ) -> TreeNode {
        let n = features.len();
        let mut counts = [0usize; N_CLASSES];
        for &l in labels {
            counts[l as usize] += 1;
        }

        if n <= min_samples || depth >= max_depth {
            return TreeNode::Leaf {
                class: majority_class(&counts),
                counts,
            };
        }
        if counts.iter().filter(|&&c| c > 0).count() == 1 {
            return TreeNode::Leaf {
                class: majority_class(&counts),
                counts,
            };
        }

        let parent_gini = gini(&counts, n);
        let n_features_try = 4;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = -1.0f64;

        for _ in 0..n_features_try {
            *rng = lcg_next(*rng);
            let f_idx = (*rng >> 33) as usize % N_FEATURES;

            let mut values: Vec<f64> = features.iter().map(|row| row[f_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            let step = (values.len() / 10).max(1);
            for i in (0..values.len().saturating_sub(1)).step_by(step) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                let mut left_counts = [0usize; N_CLASSES];
                let mut right_counts = [0usize; N_CLASSES];

                for (feat, &label) in features.iter().zip(labels.iter()) {
                    let c = label as usize;
                    if feat[f_idx] <= threshold {
                        left_counts[c] += 1;
                    } else {
                        right_counts[c] += 1;
                    }
                }

                let left_n: usize = left_counts.iter().sum();
                let right_n: usize = right_counts.iter().sum();

                if left_n < min_samples || right_n < min_samples {
                    continue;
                }

                let child_gini = (left_n as f64 * gini(&left_counts, left_n)
                    + right_n as f64 * gini(&right_counts, right_n))
                    / n as f64;
                let gain = parent_gini - child_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = f_idx;
                    best_threshold = threshold;
                }
            }
        }

        if best_gain <= 0.0 {
            return TreeNode::Leaf {
                class: majority_class(&counts),
                counts,
            };
        }

        let (left_feat, left_lab): (Vec<_>, Vec<_>) = features
            .iter()
            .zip(labels.iter())
            .filter(|(f, _)| f[best_feature] <= best_threshold)
            .map(|(f, &l)| (*f, l))
            .unzip();
        let (right_feat, right_lab): (Vec<_>, Vec<_>) = features
            .iter()
            .zip(labels.iter())
            .filter(|(f, _)| f[best_feature] > best_threshold)
            .map(|(f, &l)| (*f, l))
            .unzip();

        if left_feat.is_empty() || right_feat.is_empty() {
            return TreeNode::Leaf {
                class: majority_class(&counts),
                counts,
            };
        }

        TreeNode::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(Self::build_tree(
                &left_feat,
                &left_lab,
                depth + 1,
                max_depth,
                min_samples,
                rng,
            )),
            right: Box::new(Self::build_tree(
                &right_feat,
                &right_lab,
                depth + 1,
                max_depth,
                min_samples,
                rng,
            )),
        }
    }

    /// Predict class label for a single feature vector.
    pub fn predict(&self, features: &[f64; N_FEATURES]) -> u8 {
        let mut votes = [0usize; N_CLASSES];
        for tree in &self.trees {
            votes[tree.predict(features) as usize] += 1;
        }
        majority_class(&votes)
    }

    /// Predict class probabilities: `[P(H), P(E), P(C)]`.
    pub fn predict_proba(&self, features: &[f64; N_FEATURES]) -> [f64; N_CLASSES] {
        let mut total_counts = [0.0f64; N_CLASSES];
        for tree in &self.trees {
            let counts = tree.leaf_counts(features);
            let leaf_total: usize = counts.iter().sum();
            if leaf_total > 0 {
                let lt = leaf_total as f64;
                for (i, &c) in counts.iter().enumerate() {
                    total_counts[i] += c as f64 / lt;
                }
            }
        }
        let sum: f64 = total_counts.iter().sum();
        if sum > 0.0 {
            [
                total_counts[0] / sum,
                total_counts[1] / sum,
                total_counts[2] / sum,
            ]
        } else {
            [0.0, 0.0, 1.0]
        }
    }
}

/// Result of predicting secondary structure with baseline comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Chou-Fasman baseline predictions.
    pub baseline_predictions: Vec<SecondaryStructure>,
    /// Random Forest predictions.
    pub rf_predictions: Vec<SecondaryStructure>,
    /// Baseline Q3 accuracy (only meaningful if actual SS is known).
    pub baseline_q3: Option<f64>,
    /// RF Q3 accuracy (only meaningful if actual SS is known).
    pub rf_q3: Option<f64>,
}

/// Secondary structure predictor: Chou-Fasman baseline + Random Forest correction.
pub struct SecondaryStructurePredictor {
    forest: RandomForestClassifier,
}

impl SecondaryStructurePredictor {
    /// Create and train the predictor on CB513 data.
    pub fn new() -> Self {
        let entries = load_cb513_subset();
        let mut all_features = Vec::new();
        let mut all_labels = Vec::new();

        for entry in &entries {
            for pos in 0..entry.sequence.len() {
                all_features.push(extract_features(&entry.sequence, pos));
                all_labels.push(entry.ss[pos].class_id());
            }
        }

        let forest = RandomForestClassifier::train(&all_features, &all_labels, 50, 10);
        Self { forest }
    }

    /// Predict secondary structure for a protein sequence.
    pub fn predict_sequence(&self, seq: &[AminoAcid]) -> Vec<SecondaryStructure> {
        seq.iter()
            .enumerate()
            .map(|(pos, _)| {
                let feats = extract_features(seq, pos);
                SecondaryStructure::from_class_id(self.forest.predict(&feats))
            })
            .collect()
    }

    /// Predict with Chou-Fasman baseline comparison.
    pub fn predict_with_baseline(
        &self,
        seq: &[AminoAcid],
        actual_ss: Option<&[SecondaryStructure]>,
    ) -> PredictionResult {
        let baseline = predict_chou_fasman(seq);
        let rf_pred = self.predict_sequence(seq);
        let baseline_q3 = actual_ss.map(|actual| q3_accuracy(&baseline, actual));
        let rf_q3 = actual_ss.map(|actual| q3_accuracy(&rf_pred, actual));

        PredictionResult {
            baseline_predictions: baseline,
            rf_predictions: rf_pred,
            baseline_q3,
            rf_q3,
        }
    }

    /// Access the underlying forest.
    pub fn forest(&self) -> &RandomForestClassifier {
        &self.forest
    }
}

impl Default for SecondaryStructurePredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amino_acid::parse_sequence;
    use crate::benchmark::ClassAccuracy;

    #[test]
    fn test_gini_pure() {
        assert!((gini(&[10, 0, 0], 10) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gini_uniform() {
        let g = gini(&[10, 10, 10], 30);
        assert!((g - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_majority_class() {
        assert_eq!(majority_class(&[5, 3, 2]), 0);
        assert_eq!(majority_class(&[1, 8, 1]), 1);
        assert_eq!(majority_class(&[0, 0, 5]), 2);
    }

    #[test]
    fn test_forest_trains_and_predicts() {
        let features = [
            [
                1.0, 0.0, 0.0, 0.0, 1.5, 0.8, 0.5, 1.8, 1.42, 0.83, 0.0, 0.5, 0.0, 0.0, 1.0,
            ],
            [
                0.5, 0.0, 0.0, 0.5, 0.8, 1.5, -0.5, 4.2, 1.06, 1.70, 0.0, 0.8, 0.0, 0.0, 0.0,
            ],
            [
                0.2, 0.0, 0.0, 0.3, 0.6, 0.6, 0.0, -0.4, 0.57, 0.75, 2.0, 0.2, 0.0, 0.0, 3.0,
            ],
        ];
        let labels = [0u8, 1, 2];
        let forest = RandomForestClassifier::train(&features, &labels, 10, 5);
        let pred = forest.predict(&features[0]);
        assert!(pred < 3);
        let proba = forest.predict_proba(&features[0]);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_predictor_trains_on_cb513() {
        let predictor = SecondaryStructurePredictor::new();
        let seq = parse_sequence("MQIFVKTLTGKTITLEVEPSDTIENVKAK");
        let pred = predictor.predict_sequence(&seq);
        assert_eq!(pred.len(), seq.len());
    }

    #[test]
    fn test_predict_with_baseline_comparison() {
        let predictor = SecondaryStructurePredictor::new();
        let seq = parse_sequence("AEAAAKEAAAKEAAAK");
        let result = predictor.predict_with_baseline(&seq, None);
        assert_eq!(result.baseline_predictions.len(), seq.len());
        assert_eq!(result.rf_predictions.len(), seq.len());
        assert!(result.baseline_q3.is_none());
    }

    #[test]
    fn test_rf_accuracy_on_holdout_protein() {
        let entries = load_cb513_subset();
        assert!(entries.len() >= 5);

        let holdout_idx = entries.len() - 1;
        let test_entry = &entries[holdout_idx];
        let train_entries = &entries[..holdout_idx];

        let mut all_features = Vec::new();
        let mut all_labels = Vec::new();
        for entry in train_entries {
            for pos in 0..entry.sequence.len() {
                all_features.push(extract_features(&entry.sequence, pos));
                all_labels.push(entry.ss[pos].class_id());
            }
        }

        let forest = RandomForestClassifier::train(&all_features, &all_labels, 50, 10);

        let rf_pred: Vec<SecondaryStructure> = (0..test_entry.sequence.len())
            .map(|pos| {
                let feats = extract_features(&test_entry.sequence, pos);
                SecondaryStructure::from_class_id(forest.predict(&feats))
            })
            .collect();

        let baseline_pred = predict_chou_fasman(&test_entry.sequence);
        let rf_q3 = q3_accuracy(&rf_pred, &test_entry.ss);
        let baseline_q3 = q3_accuracy(&baseline_pred, &test_entry.ss);

        let rf_acc = ClassAccuracy::compute(&rf_pred, &test_entry.ss);
        let baseline_acc = ClassAccuracy::compute(&baseline_pred, &test_entry.ss);

        eprintln!(
            "\n=== Holdout: {} ({} res) ===",
            test_entry.name,
            test_entry.sequence.len()
        );
        eprintln!("Chou-Fasman Q3: {:.1}%", baseline_q3 * 100.0);
        eprintln!("Random Forest Q3: {:.1}%", rf_q3 * 100.0);
        eprintln!(
            "  H: {:.1}% vs {:.1}%",
            baseline_acc.class_q(0) * 100.0,
            rf_acc.class_q(0) * 100.0
        );
        eprintln!(
            "  E: {:.1}% vs {:.1}%",
            baseline_acc.class_q(1) * 100.0,
            rf_acc.class_q(1) * 100.0
        );
        eprintln!(
            "  C: {:.1}% vs {:.1}%",
            baseline_acc.class_q(2) * 100.0,
            rf_acc.class_q(2) * 100.0
        );

        assert!(
            rf_q3 > 0.30,
            "RF Q3 = {:.1}% should be > 30%",
            rf_q3 * 100.0
        );
        assert_eq!(rf_pred.len(), test_entry.ss.len());
    }

    #[test]
    fn test_rf_full_dataset_accuracy() {
        let predictor = SecondaryStructurePredictor::new();
        let entries = load_cb513_subset();
        let mut total_residues = 0;
        let mut rf_correct = 0;
        let mut baseline_correct = 0;

        for entry in &entries {
            let rf_pred = predictor.predict_sequence(&entry.sequence);
            let baseline_pred = predict_chou_fasman(&entry.sequence);
            for i in 0..entry.ss.len() {
                total_residues += 1;
                if rf_pred[i] == entry.ss[i] {
                    rf_correct += 1;
                }
                if baseline_pred[i] == entry.ss[i] {
                    baseline_correct += 1;
                }
            }
        }

        let rf_q3 = rf_correct as f64 / total_residues as f64;
        let baseline_q3 = baseline_correct as f64 / total_residues as f64;

        eprintln!("\n=== Full Dataset ({} residues) ===", total_residues);
        eprintln!("Chou-Fasman Q3: {:.1}%", baseline_q3 * 100.0);
        eprintln!("Random Forest Q3: {:.1}%", rf_q3 * 100.0);
        eprintln!("Improvement: {:.1} pp", (rf_q3 - baseline_q3) * 100.0);

        assert!(
            rf_q3 >= baseline_q3,
            "RF ({:.1}%) should beat baseline ({:.1}%)",
            rf_q3 * 100.0,
            baseline_q3 * 100.0
        );
    }
}
