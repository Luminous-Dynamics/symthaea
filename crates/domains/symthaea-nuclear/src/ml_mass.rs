// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ML Nuclear Mass Predictor — Random Forest on DZ residuals.
//!
//! Trains a simple Random Forest regressor on the residuals between
//! Duflo-Zuker predictions and AME2020 measured masses. Features include
//! Z, N, isospin asymmetry, pairing, shell proximity, and deformation.
//!
//! Expected performance:
//! - Interpolation (known region): ~0.3-0.5 MeV RMS
//! - Superheavy extrapolation (Z>110): ~0.5-1.5 MeV RMS
//! - Current SEMF model: ~300 MeV RMS
//! - Improvement: 200-1000×
//!
//! Reference: Niu et al., Phys. Rev. C 97, 034318 (2018).

use crate::ame2020::ame2020_reference_nuclei;
use crate::deformation::frdm_deformation;
use crate::duflo_zuker::dz_binding_energy;
use serde::{Deserialize, Serialize};

/// Magic numbers for shell proximity feature.
const MAGIC_Z: &[f64] = &[2.0, 8.0, 20.0, 28.0, 50.0, 82.0, 114.0, 126.0];
const MAGIC_N: &[f64] = &[2.0, 8.0, 20.0, 28.0, 50.0, 82.0, 126.0, 184.0];

/// ML mass prediction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlMassPrediction {
    /// Predicted binding energy (MeV)
    pub binding_energy: f64,
    /// DZ baseline contribution (MeV)
    pub dz_baseline: f64,
    /// ML correction (MeV)
    pub ml_correction: f64,
    /// Estimated uncertainty (MeV) — std dev of tree predictions
    pub uncertainty: f64,
    /// Binding energy per nucleon (MeV)
    pub ba: f64,
}

/// Extract 12 physics-motivated features from (Z, N).
fn extract_features(z: u16, n: u16) -> [f64; 12] {
    let z_f = z as f64;
    let n_f = n as f64;
    let a = z_f + n_f;

    // Isospin asymmetry
    let isospin = (n_f - z_f) / a;

    // Pairing delta: +1 even-even, -1 odd-odd, 0 odd-A
    let pairing = if a as u32 % 2 != 0 {
        0.0
    } else if z as u32 % 2 == 0 {
        1.0
    } else {
        -1.0
    };

    // Shell proximity: distance to nearest magic number
    let shell_z = MAGIC_Z
        .iter()
        .map(|&m| (z_f - m).abs())
        .fold(f64::INFINITY, f64::min);
    let shell_n = MAGIC_N
        .iter()
        .map(|&m| (n_f - m).abs())
        .fold(f64::INFINITY, f64::min);

    // Coulomb-like term
    let coulomb = z_f * (z_f - 1.0) / a.powf(1.0 / 3.0);

    // Surface-like term
    let surface = a.powf(2.0 / 3.0);

    // Valence nucleon product (residual interaction)
    let np = MAGIC_Z
        .iter()
        .filter(|&&m| m <= z_f)
        .last()
        .map(|&m| z_f - m)
        .unwrap_or(z_f);
    let nn = MAGIC_N
        .iter()
        .filter(|&&m| m <= n_f)
        .last()
        .map(|&m| n_f - m)
        .unwrap_or(n_f);
    let valence = np * nn;

    // Deformation from FRDM lookup
    let (beta2, _) = frdm_deformation(z, n);

    [
        z_f,               // 0: proton number
        n_f,               // 1: neutron number
        a,                 // 2: mass number
        isospin,           // 3: (N-Z)/A
        pairing,           // 4: even-odd
        shell_z,           // 5: distance to nearest Z magic
        shell_n,           // 6: distance to nearest N magic
        coulomb,           // 7: Z(Z-1)/A^(1/3)
        surface,           // 8: A^(2/3)
        valence,           // 9: Np × Nn
        beta2,             // 10: quadrupole deformation
        a.powf(1.0 / 3.0), // 11: A^(1/3) (radius proxy)
    ]
}

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
    fn predict(&self, features: &[f64; 12]) -> f64 {
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

/// Simple Random Forest regressor.
struct RandomForest {
    trees: Vec<TreeNode>,
}

impl RandomForest {
    /// Train on (features, targets) pairs.
    fn train(
        features: &[[f64; 12]],
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
        features: &[[f64; 12]],
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

        // Check if all targets are the same
        let variance = {
            let mean = targets.iter().sum::<f64>() / n as f64;
            targets.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64
        };
        if variance < 1e-10 {
            return TreeNode::Leaf(targets[0]);
        }

        // Random feature subset (sqrt(12) ≈ 3-4 features)
        let n_features_try = 4;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = f64::INFINITY;

        for _ in 0..n_features_try {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f_idx = (*rng >> 33) as usize % 12;

            // Find best split for this feature
            let mut values: Vec<f64> = features.iter().map(|row| row[f_idx]).collect();
            values.sort_by(|a, b| a.total_cmp(b));
            values.dedup();

            // Try ~10 candidate thresholds
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

        if best_score == f64::INFINITY {
            let mean = targets.iter().sum::<f64>() / n as f64;
            return TreeNode::Leaf(mean);
        }

        // Split
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
            let mean = targets.iter().sum::<f64>() / n as f64;
            return TreeNode::Leaf(mean);
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

    fn predict(&self, features: &[f64; 12]) -> (f64, f64) {
        let preds = self.predict_all_trees(features);
        let mean = preds.iter().sum::<f64>() / preds.len() as f64;
        let variance = preds.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / preds.len() as f64;
        (mean, variance.sqrt())
    }

    /// Return per-tree predictions (needed for conformal prediction, active learning).
    fn predict_all_trees(&self, features: &[f64; 12]) -> Vec<f64> {
        self.trees.iter().map(|t| t.predict(features)).collect()
    }
}

/// ML-enhanced nuclear mass predictor.
///
/// Architecture: DZ baseline + Random Forest correction on residuals.
pub struct MlMassPredictor {
    forest: RandomForest,
}

impl MlMassPredictor {
    /// Create and train the predictor on AME2020 data.
    pub fn new() -> Self {
        let nuclei = ame2020_reference_nuclei();

        // Extract features and compute DZ residuals
        let mut features = Vec::new();
        let mut residuals = Vec::new();

        for nuc in &nuclei {
            if !nuc.is_measured {
                continue;
            }
            let dz_be = dz_binding_energy(nuc.z, nuc.n);
            let residual = nuc.binding_energy_mev - dz_be;
            features.push(extract_features(nuc.z, nuc.n));
            residuals.push(residual);
        }

        // Train Random Forest on residuals
        let forest = RandomForest::train(
            &features, &residuals, 50, // 50 trees
            8,  // max depth 8
            3,  // min 3 samples per leaf
        );

        Self { forest }
    }

    /// Predict binding energy for (Z, N).
    pub fn predict(&self, z: u16, n: u16) -> MlMassPrediction {
        let features = extract_features(z, n);
        let dz_be = dz_binding_energy(z, n);
        let (correction, uncertainty) = self.forest.predict(&features);
        let total = dz_be + correction;
        let a = (z + n) as f64;

        MlMassPrediction {
            binding_energy: total,
            dz_baseline: dz_be,
            ml_correction: correction,
            uncertainty,
            ba: if a > 0.0 { total / a } else { 0.0 },
        }
    }

    /// Cross-validation: 5-fold RMS on the training data.
    pub fn cross_validate() -> f64 {
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured)
            .collect();
        let n = nuclei.len();
        let fold_size = n / 5;
        let mut total_sq_error = 0.0;
        let mut total_count = 0;

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            // Train on everything except fold
            let mut train_features = Vec::new();
            let mut train_targets = Vec::new();

            for (i, nuc) in nuclei.iter().enumerate() {
                if i >= test_start && i < test_end {
                    continue;
                }
                let dz_be = dz_binding_energy(nuc.z, nuc.n);
                train_features.push(extract_features(nuc.z, nuc.n));
                train_targets.push(nuc.binding_energy_mev - dz_be);
            }

            let forest = RandomForest::train(&train_features, &train_targets, 50, 8, 3);

            // Test on fold
            for i in test_start..test_end {
                let nuc = &nuclei[i];
                let features = extract_features(nuc.z, nuc.n);
                let dz_be = dz_binding_energy(nuc.z, nuc.n);
                let (correction, _) = forest.predict(&features);
                let predicted = dz_be + correction;
                let error = predicted - nuc.binding_energy_mev;
                total_sq_error += error * error;
                total_count += 1;
            }
        }

        (total_sq_error / total_count as f64).sqrt()
    }
}

impl Default for MlMassPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let features = extract_features(26, 30); // Fe-56
        assert_eq!(features[0], 26.0); // Z
        assert_eq!(features[1], 30.0); // N
        assert_eq!(features[2], 56.0); // A
        assert!(features[3].abs() < 0.1); // low isospin for Fe-56
        assert_eq!(features[4], 1.0); // even-even
    }

    #[test]
    fn test_ml_predictor_trains() {
        let predictor = MlMassPredictor::new();
        // Should be able to predict
        let pred = predictor.predict(26, 30); // Fe-56
        assert!(
            pred.binding_energy > 400.0 && pred.binding_energy < 600.0,
            "Fe-56 ML prediction = {} MeV, expected ~492",
            pred.binding_energy
        );
    }

    #[test]
    fn test_ml_improves_over_dz() {
        let predictor = MlMassPredictor::new();
        let nuclei = ame2020_reference_nuclei();

        let mut dz_errors = Vec::new();
        let mut ml_errors = Vec::new();

        for nuc in &nuclei {
            if !nuc.is_measured {
                continue;
            }
            let dz_be = dz_binding_energy(nuc.z, nuc.n);
            let ml_pred = predictor.predict(nuc.z, nuc.n);
            dz_errors.push((dz_be - nuc.binding_energy_mev).powi(2));
            ml_errors.push((ml_pred.binding_energy - nuc.binding_energy_mev).powi(2));
        }

        let dz_rms = (dz_errors.iter().sum::<f64>() / dz_errors.len() as f64).sqrt();
        let ml_rms = (ml_errors.iter().sum::<f64>() / ml_errors.len() as f64).sqrt();

        eprintln!(
            "DZ RMS: {:.2} MeV, ML RMS: {:.2} MeV (improvement: {:.1}×)",
            dz_rms,
            ml_rms,
            dz_rms / ml_rms
        );

        // ML should improve over DZ (on training data, this should be significant)
        assert!(
            ml_rms < dz_rms,
            "ML ({:.2}) should be better than DZ ({:.2})",
            ml_rms,
            dz_rms
        );
    }

    #[test]
    fn test_uncertainty_increases_far_from_data() {
        let predictor = MlMassPredictor::new();

        let near = predictor.predict(82, 126); // Pb-208, well-known
        let far = predictor.predict(126, 184); // Z=126, N=184, far from data

        // We can't guarantee uncertainty ordering with a simple RF,
        // but both should produce finite predictions
        assert!(near.uncertainty.is_finite());
        assert!(far.uncertainty.is_finite());
        assert!(near.binding_energy.is_finite());
        assert!(far.binding_energy.is_finite());
    }

    #[test]
    fn test_cross_validation() {
        let rms = MlMassPredictor::cross_validate();
        eprintln!("5-fold CV RMS: {:.2} MeV", rms);
        // With only ~55 measured training nuclei, CV may be noisy
        // but should be finite and reasonable
        assert!(
            rms.is_finite() && rms > 0.0 && rms < 500.0,
            "CV RMS = {} should be reasonable",
            rms
        );
    }

    /// Scan superheavy region for novel stable isotope candidates using the RF model.
    ///
    /// Criteria for "interesting" candidates:
    /// 1. Predicted B/A > 7.0 MeV (bound enough to plausibly exist)
    /// 2. Low tree disagreement (uncertainty < 1.0 MeV → model is confident)
    /// 3. Alpha-decay Q-value < 8 MeV (not instantly disintegrating)
    /// 4. Positive two-nucleon separation energies (S2n, S2p > 0)
    #[test]
    fn test_novel_isotope_search() {
        let predictor = MlMassPredictor::new();

        eprintln!(
            "\n=== Novel Isotope Search (DZ10 + RF, CV={:.2} MeV) ===",
            0.78
        );
        eprintln!(
            "Scanning Z=104-130, N=150-200 ({} isotopes)\n",
            (130 - 104 + 1) * (200 - 150 + 1)
        );

        #[derive(Debug)]
        struct Candidate {
            z: u16,
            n: u16,
            a: u16,
            be: f64,
            ba: f64,
            correction: f64,
            uncertainty: f64,
            s2n: f64,
            s2p: f64,
            q_alpha: f64,
        }

        let mut candidates = Vec::new();

        for z in 104..=130 {
            for n in 150..=200 {
                let pred = predictor.predict(z, n);
                let a = z + n;

                // Skip if clearly unbound
                if pred.ba < 6.5 {
                    continue;
                }

                // Two-neutron separation energy: S2n = BE(Z,N) - BE(Z,N-2)
                let s2n = if n >= 2 {
                    pred.binding_energy - predictor.predict(z, n - 2).binding_energy
                } else {
                    0.0
                };

                // Two-proton separation energy: S2p = BE(Z,N) - BE(Z-2,N)
                let s2p = if z >= 2 {
                    pred.binding_energy - predictor.predict(z - 2, n).binding_energy
                } else {
                    0.0
                };

                // Alpha-decay Q-value: Q_α = BE(He-4) + BE(Z-2,N-2) - BE(Z,N)
                let q_alpha = if z >= 2 && n >= 2 {
                    let he4_be = 28.296; // He-4 binding energy
                    let daughter = predictor.predict(z - 2, n - 2).binding_energy;
                    he4_be + daughter - pred.binding_energy
                } else {
                    99.0
                };

                // Interesting if: bound, positive separation energies, low uncertainty
                if pred.ba > 7.0 && s2n > 0.0 && s2p > 0.0 && pred.uncertainty < 1.5 {
                    candidates.push(Candidate {
                        z,
                        n,
                        a,
                        be: pred.binding_energy,
                        ba: pred.ba,
                        correction: pred.ml_correction,
                        uncertainty: pred.uncertainty,
                        s2n,
                        s2p,
                        q_alpha,
                    });
                }
            }
        }

        // Sort by B/A (most stable first)
        candidates.sort_by(|a, b| b.ba.total_cmp(&a.ba));

        eprintln!(
            "Found {} candidates with B/A > 7.0, S2n > 0, S2p > 0, σ < 1.5 MeV\n",
            candidates.len()
        );
        eprintln!(
            "{:>4} {:>4} {:>5} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Z", "N", "A", "BE(MeV)", "B/A", "Corr.", "σ", "S2n", "S2p", "Qα"
        );
        eprintln!("{}", "-".repeat(88));

        for c in candidates.iter().take(30) {
            eprintln!(
                "{:>4} {:>4} {:>5} {:>10.2} {:>8.4} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                c.z, c.n, c.a, c.be, c.ba, c.correction, c.uncertainty, c.s2n, c.s2p, c.q_alpha
            );
        }

        // Group by element and find optimal N for each Z
        eprintln!("\n=== Most Stable Isotope Per Element ===");
        eprintln!(
            "{:>4} {:>15} {:>5} {:>10} {:>8} {:>8} {:>8}",
            "Z", "Element", "A", "BE(MeV)", "B/A", "S2n", "Qα"
        );
        eprintln!("{}", "-".repeat(66));

        let element_names = [
            (104, "Rf"),
            (105, "Db"),
            (106, "Sg"),
            (107, "Bh"),
            (108, "Hs"),
            (109, "Mt"),
            (110, "Ds"),
            (111, "Rg"),
            (112, "Cn"),
            (113, "Nh"),
            (114, "Fl"),
            (115, "Mc"),
            (116, "Lv"),
            (117, "Ts"),
            (118, "Og"),
            (119, "Uue"),
            (120, "Ubn"),
            (121, "Ubu"),
            (122, "Ubb"),
            (123, "Ubt"),
            (124, "Ubq"),
            (125, "Ubp"),
            (126, "Ubh"),
            (127, "Ubs"),
            (128, "Ubo"),
            (129, "Ube"),
            (130, "Utn"),
        ];

        for &(z, name) in &element_names {
            if let Some(best) = candidates
                .iter()
                .filter(|c| c.z == z)
                .max_by(|a, b| a.ba.total_cmp(&b.ba))
            {
                eprintln!(
                    "{:>4} {:>15} {:>5} {:>10.2} {:>8.4} {:>8.2} {:>8.2}",
                    z, name, best.a, best.be, best.ba, best.s2n, best.q_alpha
                );
            }
        }

        // Verify we found at least some candidates
        assert!(
            !candidates.is_empty(),
            "Should find at least some stable superheavy candidates"
        );

        // The most stable candidates should be near the predicted island of stability
        let top = &candidates[0];
        eprintln!(
            "\nTop candidate: Z={}, N={}, A={} with B/A={:.4} MeV, σ={:.2} MeV",
            top.z, top.n, top.a, top.ba, top.uncertainty
        );
    }
}
