// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ML Bandgap Predictor — Random Forest on electronegativity-model residuals.
//!
//! Trains a simple Random Forest regressor on the residuals between the
//! electronegativity baseline and experimental bandgaps. Features include
//! composition descriptors: electronegativity stats, covalent radii,
//! block fractions, crystal system, and ionicity.
//!
//! Architecture mirrors `ml_mass.rs` from symthaea-nuclear:
//!   prediction = baseline(composition) + RF_correction(features)
//!
//! Expected performance:
//! - Electronegativity baseline: ~1.5-2.0 eV MAE (crude, composition-only)
//! - RF on training data: ~0.3-0.5 eV MAE
//! - RF cross-validation: ~0.5-1.0 eV MAE
//!
//! Reference: Zhuo et al. (2018), J. Phys. Chem. Lett. 9, 1668.

use crate::bandgap_baseline::electronegativity_bandgap;
use crate::features::{N_FEATURES, extract_features};
use crate::training_data::{CrystalSystem, load_training_data};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a bandgap prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandgapPrediction {
    /// Predicted experimental bandgap (eV).
    pub bandgap: f64,
    /// Electronegativity model prediction (eV).
    pub baseline: f64,
    /// RF correction term (eV).
    pub ml_correction: f64,
    /// Inter-tree standard deviation (eV) — epistemic uncertainty.
    pub uncertainty: f64,
}

/// ML-enhanced bandgap predictor.
///
/// Architecture: electronegativity baseline + Random Forest residual correction.
pub struct BandgapPredictor {
    forest: RandomForest,
}

impl BandgapPredictor {
    /// Create and train the predictor on the full training dataset.
    pub fn new() -> Self {
        let data = load_training_data();

        let mut features = Vec::with_capacity(data.len());
        let mut residuals = Vec::with_capacity(data.len());

        for d in &data {
            let baseline = electronegativity_bandgap(&d.composition);
            let residual = d.bandgap_exp() - baseline;
            features.push(extract_features(&d.composition, d.crystal()));
            residuals.push(residual);
        }

        let forest = RandomForest::train(&features, &residuals, 50, 8, 3);

        Self { forest }
    }

    /// Predict bandgap for a given composition and crystal system.
    pub fn predict(&self, composition: &[(u8, f64)], crystal: &CrystalSystem) -> BandgapPrediction {
        let features = extract_features(composition, crystal);
        let baseline = electronegativity_bandgap(composition);
        let (correction, uncertainty) = self.forest.predict(&features);

        BandgapPrediction {
            bandgap: (baseline + correction).max(0.0),
            baseline,
            ml_correction: correction,
            uncertainty,
        }
    }

    /// 5-fold cross-validation. Returns (MAE, RMSE) across all folds.
    pub fn cross_validate() -> (f64, f64) {
        let data = load_training_data();
        let n = data.len();
        let fold_size = n / 5;

        let mut total_abs_error = 0.0;
        let mut total_sq_error = 0.0;
        let mut total_count = 0;

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            // Train on everything except this fold
            let mut train_features = Vec::new();
            let mut train_targets = Vec::new();

            for (i, d) in data.iter().enumerate() {
                if i >= test_start && i < test_end {
                    continue;
                }
                let baseline = electronegativity_bandgap(&d.composition);
                train_features.push(extract_features(&d.composition, d.crystal()));
                train_targets.push(d.bandgap_exp() - baseline);
            }

            let forest = RandomForest::train(&train_features, &train_targets, 50, 8, 3);

            // Test on this fold
            for i in test_start..test_end {
                let d = &data[i];
                let features = extract_features(&d.composition, d.crystal());
                let baseline = electronegativity_bandgap(&d.composition);
                let (correction, _) = forest.predict(&features);
                let predicted = (baseline + correction).max(0.0);
                let error = predicted - d.bandgap_exp();
                total_abs_error += error.abs();
                total_sq_error += error * error;
                total_count += 1;
            }
        }

        let mae = total_abs_error / total_count as f64;
        let rmse = (total_sq_error / total_count as f64).sqrt();
        (mae, rmse)
    }
}

impl Default for BandgapPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Random Forest implementation (inline, same pattern as nuclear crate)
// ---------------------------------------------------------------------------

/// A single decision tree node.
#[derive(Debug, Clone)]
enum TreeNode {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    fn predict(&self, features: &[f64; N_FEATURES]) -> f64 {
        match self {
            TreeNode::Leaf(val) => *val,
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
}

/// Simple Random Forest regressor — bootstrap aggregation of depth-limited
/// decision trees with random feature subsampling.
struct RandomForest {
    trees: Vec<TreeNode>,
}

impl RandomForest {
    /// Train on (features, targets) pairs.
    ///
    /// * `n_trees` — number of bootstrap-aggregated trees
    /// * `max_depth` — maximum tree depth
    /// * `min_samples` — minimum samples per leaf
    fn train(
        features: &[[f64; N_FEATURES]],
        targets: &[f64],
        n_trees: usize,
        max_depth: usize,
        min_samples: usize,
    ) -> Self {
        let n = features.len();
        let mut trees = Vec::with_capacity(n_trees);
        let mut rng_state: u64 = 42;

        for _ in 0..n_trees {
            // Bootstrap sample
            let indices: Vec<usize> = (0..n)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (rng_state >> 33) as usize % n
                })
                .collect();

            let boot_features: Vec<_> = indices.iter().map(|&i| features[i]).collect();
            let boot_targets: Vec<_> = indices.iter().map(|&i| targets[i]).collect();

            let tree = Self::build_tree(
                &boot_features,
                &boot_targets,
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
        targets: &[f64],
        depth: usize,
        max_depth: usize,
        min_samples: usize,
        rng: &mut u64,
    ) -> TreeNode {
        let n = features.len();

        // Base case: leaf
        if n <= min_samples || depth >= max_depth {
            let mean = targets.iter().sum::<f64>() / n as f64;
            return TreeNode::Leaf(mean);
        }

        // Check if all targets are (nearly) identical
        let mean_target = targets.iter().sum::<f64>() / n as f64;
        let variance = targets
            .iter()
            .map(|t| (t - mean_target).powi(2))
            .sum::<f64>()
            / n as f64;
        if variance < 1e-10 {
            return TreeNode::Leaf(mean_target);
        }

        // Random feature subset: sqrt(N_FEATURES) ~ 4
        let n_features_try = 4;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = f64::INFINITY;

        for _ in 0..n_features_try {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f_idx = (*rng >> 33) as usize % N_FEATURES;

            // Collect and sort unique values for this feature
            let mut values: Vec<f64> = features.iter().map(|row| row[f_idx]).collect();
            values.sort_by(|a, b| a.total_cmp(b));
            values.dedup();

            // Try ~10 candidate thresholds evenly spaced
            let step = (values.len() / 10).max(1);
            for i in (0..values.len()).step_by(step) {
                let threshold = values[i];

                let (left_sum, left_sq, left_n, right_sum, right_sq, right_n) =
                    features.iter().zip(targets.iter()).fold(
                        (0.0f64, 0.0f64, 0usize, 0.0f64, 0.0f64, 0usize),
                        |(ls, lsq, ln, rs, rsq, rn), (feat, &tgt)| {
                            if feat[f_idx] <= threshold {
                                (ls + tgt, lsq + tgt * tgt, ln + 1, rs, rsq, rn)
                            } else {
                                (ls, lsq, ln, rs + tgt, rsq + tgt * tgt, rn + 1)
                            }
                        },
                    );

                if left_n < min_samples || right_n < min_samples {
                    continue;
                }

                // Weighted variance reduction
                let left_var = left_sq / left_n as f64 - (left_sum / left_n as f64).powi(2);
                let right_var = right_sq / right_n as f64 - (right_sum / right_n as f64).powi(2);
                let score = left_n as f64 * left_var.max(0.0) + right_n as f64 * right_var.max(0.0);

                if score < best_score {
                    best_score = score;
                    best_feature = f_idx;
                    best_threshold = threshold;
                }
            }
        }

        // No valid split found
        if best_score == f64::INFINITY {
            return TreeNode::Leaf(mean_target);
        }

        // Partition data
        let (left_feat, left_tgt): (Vec<_>, Vec<_>) = features
            .iter()
            .zip(targets.iter())
            .filter(|(f, _)| f[best_feature] <= best_threshold)
            .map(|(f, &t)| (*f, t))
            .unzip();
        let (right_feat, right_tgt): (Vec<_>, Vec<_>) = features
            .iter()
            .zip(targets.iter())
            .filter(|(f, _)| f[best_feature] > best_threshold)
            .map(|(f, &t)| (*f, t))
            .unzip();

        if left_feat.is_empty() || right_feat.is_empty() {
            return TreeNode::Leaf(mean_target);
        }

        TreeNode::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(Self::build_tree(
                &left_feat,
                &left_tgt,
                depth + 1,
                max_depth,
                min_samples,
                rng,
            )),
            right: Box::new(Self::build_tree(
                &right_feat,
                &right_tgt,
                depth + 1,
                max_depth,
                min_samples,
                rng,
            )),
        }
    }

    /// Predict: returns (mean prediction, inter-tree std dev).
    fn predict(&self, features: &[f64; N_FEATURES]) -> (f64, f64) {
        let predictions: Vec<f64> = self.trees.iter().map(|t| t.predict(features)).collect();
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance =
            predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / predictions.len() as f64;
        (mean, variance.sqrt())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_trains() {
        let predictor = BandgapPredictor::new();
        let pred = predictor.predict(&[(14, 1.0)], &CrystalSystem::Cubic);
        assert!(
            pred.bandgap > 0.0 && pred.bandgap < 5.0,
            "Si prediction = {} eV, expected ~1.1",
            pred.bandgap
        );
    }

    #[test]
    fn test_prediction_fields() {
        let predictor = BandgapPredictor::new();
        let pred = predictor.predict(&[(31, 0.5), (33, 0.5)], &CrystalSystem::Cubic);
        // bandgap = baseline + correction
        let reconstructed = pred.baseline + pred.ml_correction;
        assert!(
            (pred.bandgap - reconstructed.max(0.0)).abs() < 1e-10,
            "bandgap ({}) != baseline ({}) + correction ({})",
            pred.bandgap,
            pred.baseline,
            pred.ml_correction
        );
        assert!(pred.uncertainty >= 0.0);
        assert!(pred.uncertainty.is_finite());
    }

    #[test]
    fn test_rf_improves_over_baseline() {
        let predictor = BandgapPredictor::new();
        let data = load_training_data();

        let mut baseline_sq = 0.0;
        let mut rf_sq = 0.0;

        for d in &data {
            let baseline = electronegativity_bandgap(&d.composition);
            let pred = predictor.predict(&d.composition, d.crystal());
            baseline_sq += (baseline - d.bandgap_exp()).powi(2);
            rf_sq += (pred.bandgap - d.bandgap_exp()).powi(2);
        }

        let baseline_rmse = (baseline_sq / data.len() as f64).sqrt();
        let rf_rmse = (rf_sq / data.len() as f64).sqrt();

        eprintln!(
            "Baseline RMSE: {:.3} eV, RF RMSE: {:.3} eV (improvement: {:.1}x)",
            baseline_rmse,
            rf_rmse,
            baseline_rmse / rf_rmse
        );
        assert!(
            rf_rmse < baseline_rmse,
            "RF ({:.3}) should beat baseline ({:.3})",
            rf_rmse,
            baseline_rmse
        );
    }

    #[test]
    fn test_cross_validation() {
        let (mae, rmse) = BandgapPredictor::cross_validate();
        eprintln!("5-fold CV: MAE = {:.3} eV, RMSE = {:.3} eV", mae, rmse);
        assert!(mae.is_finite() && mae > 0.0);
        assert!(rmse.is_finite() && rmse > 0.0);
        // CV RMSE should be reasonable (< 5 eV for this diverse dataset)
        assert!(rmse < 5.0, "CV RMSE = {} is unreasonably high", rmse);
    }

    #[test]
    fn test_nonnegative_predictions() {
        let predictor = BandgapPredictor::new();
        let data = load_training_data();
        for d in &data {
            let pred = predictor.predict(&d.composition, d.crystal());
            assert!(
                pred.bandgap >= 0.0,
                "{}: predicted {} eV (negative!)",
                d.name(),
                pred.bandgap
            );
        }
    }
}
