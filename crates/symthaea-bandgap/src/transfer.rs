// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Cross-Domain Transfer Experiment: Nuclear Physics → Bandgap Prediction
//!
//! Tests whether nuclear-derived features (binding energy per nucleon, magic
//! number shell proximity, d-electron count, metallic character) improve
//! semiconductor bandgap predictions when added to the standard 18D
//! composition feature vector.
//!
//! Motivation: Nuclear shell structure and binding energy encode deep
//! information about electron cloud organisation that may correlate with
//! solid-state electronic structure (bandgap). If so, this demonstrates
//! cross-domain transfer between nuclear physics and materials science.
//!
//! ## Method
//!
//! 1. Augment the 18D feature vector with 6 nuclear-derived features → 24D.
//! 2. Train 50-tree Random Forests on both 18D and 24D representations.
//! 3. 5-fold cross-validation RMSE comparison.
//! 4. Permutation importance for each of the 6 new features.

#[cfg(test)]
mod tests {
    use crate::bandgap_baseline::electronegativity_bandgap;
    use crate::features::*;
    use crate::training_data::*;

    // ── Nuclear-derived element properties ──────────────────────────────────

    /// Binding energy per nucleon of most stable isotope (MeV), Z=0..103.
    /// Source: AME2020 (Wang et al. 2021).
    const BE_PER_NUCLEON: &[f64] = &[
        0.0, 0.0, 0.0, 2.83, 6.46, 5.33, 6.68, 7.48, 7.68, 7.06, 7.77,
        8.03, 8.26, 8.33, 8.45, 8.48, 8.58, 8.52, 8.57, 8.56, 8.55,
        8.62, 8.66, 8.72, 8.76, 8.77, 8.79, 8.77, 8.77, 8.74, 8.76,
        8.72, 8.71, 8.65, 8.70, 8.64, 8.67, 8.60, 8.63, 8.57, 8.59,
        8.55, 8.58, 8.51, 8.55, 8.49, 8.53, 8.47, 8.50, 8.44, 8.49,
        8.42, 8.44, 8.40, 8.44, 8.37, 8.42, 8.36, 8.39, 8.35, 8.38,
        8.34, 8.35, 8.33, 8.34, 8.32, 8.33, 8.30, 8.32, 8.29, 8.30,
        8.28, 8.30, 8.27, 8.28, 8.26, 8.27, 8.24, 8.26, 8.23, 8.24,
        8.22, 8.23, 7.87, 7.85, 7.82, 7.80, 7.78, 7.76, 7.74, 7.72,
        7.70, 7.69, 7.60, 7.58, 7.56, 7.54, 7.52, 7.50, 7.48, 7.46,
        7.44, 7.42, 7.40, 7.38,
    ];

    /// Magic proton numbers (nuclear shell closures).
    const MAGIC_Z: &[u8] = &[2, 8, 20, 28, 50, 82];

    /// Distance to nearest magic number — encodes nuclear shell stability.
    fn z_shell_proximity(z: u8) -> f64 {
        MAGIC_Z
            .iter()
            .map(|&m| (z as i32 - m as i32).unsigned_abs() as f64)
            .fold(f64::MAX, f64::min)
    }

    /// Approximate d-electron count from electron configuration.
    fn d_electrons(z: u8) -> u8 {
        match z {
            21..=30 => z - 18 - 2, // 3d: Sc=1, Ti=2, ..., Zn=10
            39..=48 => z - 36 - 2, // 4d
            57 | 72..=80 => z.saturating_sub(54).saturating_sub(2).min(10), // 5d
            89 | 104..=112 => z.saturating_sub(86).saturating_sub(2).min(10), // 6d
            _ => 0,
        }
    }

    /// Binary metallic character: 0.0 (nonmetal), 0.5 (metalloid), 1.0 (metal).
    fn metallic_character(z: u8) -> f64 {
        match z {
            1..=2 => 0.0,
            5 | 14 | 32 | 33 | 51 | 52 => 0.5,                             // metalloids
            6..=10 | 15..=18 | 34..=36 | 53..=54 | 85..=86 => 0.0,         // nonmetals
            _ => 1.0,                                                        // metals
        }
    }

    // ── Augmented feature vector (24D) ──────────────────────────────────────

    const N_AUGMENTED: usize = 24;
    const N_NUCLEAR: usize = 6;

    /// Nuclear feature names for reporting.
    const NUCLEAR_FEATURE_NAMES: [&str; N_NUCLEAR] = [
        "BE/A (binding energy per nucleon)",
        "shell_min (min magic number proximity)",
        "shell_max (max magic number proximity)",
        "d_electrons (d-electron count)",
        "metallic_character",
        "unfilled_fraction (unfilled orbitals)",
    ];

    fn extract_augmented(composition: &[(u8, f64)], crystal: &CrystalSystem) -> [f64; N_AUGMENTED] {
        let base = extract_features(composition, crystal);
        let mut aug = [0.0f64; N_AUGMENTED];

        // Copy base 18 features
        for i in 0..18 {
            aug[i] = base[i];
        }

        // Add nuclear-derived features (weighted by composition)
        let mut be_a_sum = 0.0;
        let mut shell_min = f64::MAX;
        let mut shell_max = 0.0f64;
        let mut d_sum = 0.0;
        let mut metal_sum = 0.0;
        let mut unfilled_sum = 0.0;

        for &(z, frac) in composition {
            let be_a = BE_PER_NUCLEON.get(z as usize).copied().unwrap_or(7.5);
            let shell = z_shell_proximity(z);
            let d = d_electrons(z) as f64;
            let met = metallic_character(z);
            let unfilled = (18.0
                - crate::periodic_table::element(z).valence_electrons as f64)
                .max(0.0);

            be_a_sum += frac * be_a;
            shell_min = shell_min.min(shell);
            shell_max = shell_max.max(shell);
            d_sum += frac * d;
            metal_sum += frac * met;
            unfilled_sum += frac * unfilled;
        }

        aug[18] = be_a_sum / 8.8;          // normalised BE/A
        aug[19] = shell_min / 50.0;         // min shell proximity
        aug[20] = shell_max / 50.0;         // max shell proximity
        aug[21] = d_sum / 10.0;             // d-electron fraction
        aug[22] = metal_sum;                 // metallic character
        aug[23] = unfilled_sum / 18.0;      // unfilled fraction

        aug
    }

    // ── Generic Random Forest (works with any const N) ──────────────────────

    #[derive(Debug, Clone)]
    enum TreeNode<const N: usize> {
        Leaf(f64),
        Split {
            feature: usize,
            threshold: f64,
            left: Box<TreeNode<N>>,
            right: Box<TreeNode<N>>,
        },
    }

    impl<const N: usize> TreeNode<N> {
        fn predict(&self, features: &[f64; N]) -> f64 {
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

    struct RandomForest<const N: usize> {
        trees: Vec<TreeNode<N>>,
    }

    impl<const N: usize> RandomForest<N> {
        fn train(
            features: &[[f64; N]],
            targets: &[f64],
            n_trees: usize,
            max_depth: usize,
            min_samples: usize,
        ) -> Self {
            let n = features.len();
            let mut trees = Vec::with_capacity(n_trees);
            let mut rng_state: u64 = 42;

            for _ in 0..n_trees {
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
            features: &[[f64; N]],
            targets: &[f64],
            depth: usize,
            max_depth: usize,
            min_samples: usize,
            rng: &mut u64,
        ) -> TreeNode<N> {
            let n = features.len();

            if n <= min_samples || depth >= max_depth {
                let mean = targets.iter().sum::<f64>() / n as f64;
                return TreeNode::Leaf(mean);
            }

            let mean_target = targets.iter().sum::<f64>() / n as f64;
            let variance = targets
                .iter()
                .map(|t| (t - mean_target).powi(2))
                .sum::<f64>()
                / n as f64;
            if variance < 1e-10 {
                return TreeNode::Leaf(mean_target);
            }

            // Random feature subset: sqrt(N) candidates
            let n_features_try = (N as f64).sqrt().ceil() as usize;
            let mut best_feature = 0;
            let mut best_threshold = 0.0;
            let mut best_score = f64::INFINITY;

            for _ in 0..n_features_try {
                *rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let f_idx = (*rng >> 33) as usize % N;

                let mut values: Vec<f64> = features.iter().map(|row| row[f_idx]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                values.dedup();

                let step = (values.len() / 10).max(1);
                for i in (0..values.len()).step_by(step) {
                    let threshold = values[i];

                    let (left_sum, left_sq, left_n, right_sum, right_sq, right_n) = features
                        .iter()
                        .zip(targets.iter())
                        .fold(
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

                    let left_var =
                        left_sq / left_n as f64 - (left_sum / left_n as f64).powi(2);
                    let right_var =
                        right_sq / right_n as f64 - (right_sum / right_n as f64).powi(2);
                    let score =
                        left_n as f64 * left_var.max(0.0) + right_n as f64 * right_var.max(0.0);

                    if score < best_score {
                        best_score = score;
                        best_feature = f_idx;
                        best_threshold = threshold;
                    }
                }
            }

            if best_score == f64::INFINITY {
                return TreeNode::Leaf(mean_target);
            }

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

        fn predict(&self, features: &[f64; N]) -> (f64, f64) {
            let predictions: Vec<f64> = self.trees.iter().map(|t| t.predict(features)).collect();
            let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
            let variance = predictions
                .iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f64>()
                / predictions.len() as f64;
            (mean, variance.sqrt())
        }
    }

    // ── Cross-validation helper ─────────────────────────────────────────────

    /// 5-fold CV returning RMSE. Generic over feature dimension.
    fn cross_validate_rmse<const N: usize>(
        all_features: &[[f64; N]],
        all_targets: &[f64],
        baselines: &[f64],
    ) -> f64 {
        let n = all_features.len();
        let fold_size = n / 5;
        let mut total_sq_error = 0.0;
        let mut total_count = 0;

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            let mut train_features = Vec::new();
            let mut train_targets = Vec::new();

            for i in 0..n {
                if i >= test_start && i < test_end {
                    continue;
                }
                train_features.push(all_features[i]);
                train_targets.push(all_targets[i]);
            }

            let forest =
                RandomForest::<N>::train(&train_features, &train_targets, 50, 8, 3);

            for i in test_start..test_end {
                let (correction, _) = forest.predict(&all_features[i]);
                let predicted = (baselines[i] + correction).max(0.0);
                let actual = baselines[i] + all_targets[i]; // target = exp - baseline
                let error = predicted - actual;
                total_sq_error += error * error;
                total_count += 1;
            }
        }

        (total_sq_error / total_count as f64).sqrt()
    }

    // ── The experiment ──────────────────────────────────────────────────────

    #[test]
    fn test_cross_domain_transfer_nuclear_to_bandgap() {
        let data = load_training_data();
        let n = data.len();

        // Prepare baseline predictions and residual targets
        let mut baselines = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        let mut features_18: Vec<[f64; 18]> = Vec::with_capacity(n);
        let mut features_24: Vec<[f64; N_AUGMENTED]> = Vec::with_capacity(n);

        for d in &data {
            let bl = electronegativity_bandgap(&d.composition);
            baselines.push(bl);
            targets.push(d.bandgap_exp() - bl);
            features_18.push(extract_features(&d.composition, d.crystal()));
            features_24.push(extract_augmented(&d.composition, d.crystal()));
        }

        // ── 5-fold CV for both models ───────────────────────────────────────

        let rmse_18 = cross_validate_rmse::<18>(&features_18, &targets, &baselines);
        let rmse_24 = cross_validate_rmse::<N_AUGMENTED>(&features_24, &targets, &baselines);

        let improvement_pct = (rmse_18 - rmse_24) / rmse_18 * 100.0;

        eprintln!();
        eprintln!("=== CROSS-DOMAIN TRANSFER EXPERIMENT ===");
        eprintln!("Dataset: {} materials", n);
        eprintln!("Baseline (18D): CV RMSE = {:.3} eV", rmse_18);
        eprintln!(
            "Augmented (24D with nuclear features): CV RMSE = {:.3} eV",
            rmse_24
        );
        eprintln!("Improvement: {:.1}%", improvement_pct);
        eprintln!();

        // ── Permutation importance for the 6 nuclear features ───────────────

        eprintln!("--- Permutation Importance (nuclear features) ---");
        eprintln!(
            "Method: shuffle one feature at a time, measure RMSE increase"
        );
        eprintln!();

        let mut importances: Vec<(usize, &str, f64)> = Vec::new();

        for nf_idx in 0..N_NUCLEAR {
            let feature_col = 18 + nf_idx; // column index in 24D vector

            // Create a shuffled copy of features_24
            let mut shuffled = features_24.clone();
            let mut rng_state: u64 = 12345 + nf_idx as u64;

            // Fisher-Yates shuffle on the target column
            for i in (1..n).rev() {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                // Swap feature_col values between rows i and j
                let tmp = shuffled[i][feature_col];
                shuffled[i][feature_col] = shuffled[j][feature_col];
                shuffled[j][feature_col] = tmp;
            }

            let rmse_shuffled =
                cross_validate_rmse::<N_AUGMENTED>(&shuffled, &targets, &baselines);
            let delta = rmse_shuffled - rmse_24;
            let delta_pct = delta / rmse_24 * 100.0;

            importances.push((nf_idx, NUCLEAR_FEATURE_NAMES[nf_idx], delta_pct));

            eprintln!(
                "  [{}] {:40} -> RMSE {:.3} eV  (delta: {:+.3} eV, {:+.1}%)",
                nf_idx,
                NUCLEAR_FEATURE_NAMES[nf_idx],
                rmse_shuffled,
                delta,
                delta_pct,
            );
        }

        // Sort by importance (descending)
        importances.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        eprintln!();
        eprintln!("--- Ranked by importance ---");
        for (rank, (idx, name, pct)) in importances.iter().enumerate() {
            eprintln!("  #{}: [{}] {} ({:+.1}%)", rank + 1, idx, name, pct);
        }
        eprintln!();

        // ── Assertions ──────────────────────────────────────────────────────

        // Both models should produce finite, positive RMSE
        assert!(
            rmse_18.is_finite() && rmse_18 > 0.0,
            "18D RMSE should be finite positive: {}",
            rmse_18
        );
        assert!(
            rmse_24.is_finite() && rmse_24 > 0.0,
            "24D RMSE should be finite positive: {}",
            rmse_24
        );

        // The 24D model should not be catastrophically worse (allow up to 10% degradation)
        assert!(
            rmse_24 < rmse_18 * 1.10,
            "24D RMSE ({:.3}) is >10% worse than 18D ({:.3})",
            rmse_24,
            rmse_18
        );

        // RMSE should be reasonable for this dataset
        assert!(
            rmse_18 < 5.0,
            "18D CV RMSE = {} eV is unreasonably high",
            rmse_18
        );
        assert!(
            rmse_24 < 5.0,
            "24D CV RMSE = {} eV is unreasonably high",
            rmse_24
        );
    }
}
