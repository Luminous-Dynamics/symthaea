// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stress-test validation: proton holdout, chain holdout, frontier holdout.
//!
//! These are harder than standard k-fold CV because they test
//! the model's ability to **extrapolate**, not just interpolate.

#[cfg(test)]
mod tests {
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::deformation::frdm_deformation;
    use crate::duflo_zuker::dz_binding_energy;
    use crate::ml_mass::MlMassPrediction;

    // Re-implement the forest training inline since we need custom splits.
    // We use the same feature extraction and RF structure as ml_mass.rs.

    const MAGIC_Z: &[f64] = &[2.0, 8.0, 20.0, 28.0, 50.0, 82.0, 114.0, 126.0];
    const MAGIC_N: &[f64] = &[2.0, 8.0, 20.0, 28.0, 50.0, 82.0, 126.0, 184.0];

    fn extract_features(z: u16, n: u16) -> [f64; 12] {
        let z_f = z as f64;
        let n_f = n as f64;
        let a = z_f + n_f;
        let isospin = (n_f - z_f) / a;
        let pairing = if z % 2 == 0 && n % 2 == 0 {
            1.0
        } else if z % 2 == 1 && n % 2 == 1 {
            -1.0
        } else {
            0.0
        };
        let shell_z = MAGIC_Z
            .iter()
            .map(|m| (z_f - m).abs())
            .fold(f64::MAX, f64::min);
        let shell_n = MAGIC_N
            .iter()
            .map(|m| (n_f - m).abs())
            .fold(f64::MAX, f64::min);
        let coulomb = z_f * (z_f - 1.0) / a.powf(1.0 / 3.0);
        let surface = a.powf(2.0 / 3.0);
        let np_valence = {
            let vp = MAGIC_Z
                .iter()
                .filter(|&&m| z_f >= m)
                .last()
                .map(|m| z_f - m)
                .unwrap_or(z_f);
            let vn = MAGIC_N
                .iter()
                .filter(|&&m| n_f >= m)
                .last()
                .map(|m| n_f - m)
                .unwrap_or(n_f);
            vp * vn
        };
        let (beta2, _) = frdm_deformation(z, n);
        let radius = a.powf(1.0 / 3.0);

        [
            z_f, n_f, a, isospin, pairing, shell_z, shell_n, coulomb, surface, np_valence, beta2,
            radius,
        ]
    }

    /// Simple decision tree for residual prediction (same as ml_mass.rs).
    struct DecisionTree {
        splits: Vec<(usize, f64, f64, f64)>, // (feature_idx, threshold, left_val, right_val)
    }

    impl DecisionTree {
        fn predict(&self, features: &[f64; 12]) -> f64 {
            // Walk tree (simplified: just use the splits as a single-level ensemble)
            let mut total = 0.0;
            for &(feat, thresh, left, right) in &self.splits {
                if features[feat] <= thresh {
                    total += left;
                } else {
                    total += right;
                }
            }
            total / self.splits.len().max(1) as f64
        }
    }

    /// Train a Random Forest on given data. Returns a function (features -> prediction).
    fn train_rf(
        features: &[[f64; 12]],
        targets: &[f64],
        n_trees: usize,
        max_depth: usize,
        min_leaf: usize,
    ) -> Vec<DecisionTree> {
        use std::collections::HashMap;
        let n = features.len();
        if n == 0 {
            return Vec::new();
        }

        let mut trees = Vec::new();
        let seed_base = 42u64;

        for t in 0..n_trees {
            // Bootstrap sample
            let mut rng_state = seed_base.wrapping_mul(t as u64 + 1).wrapping_add(7);
            let mut indices: Vec<usize> = (0..n)
                .map(|_| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (rng_state >> 33) as usize % n
                })
                .collect();

            // Build tree with greedy splits
            let mut splits = Vec::new();
            let mut remaining: Vec<usize> = indices;

            for _depth in 0..max_depth {
                if remaining.len() < min_leaf * 2 {
                    break;
                }

                // Find best split
                let mut best_feat = 0;
                let mut best_thresh = 0.0;
                let mut best_score = f64::MAX;
                let mut best_left_mean = 0.0;
                let mut best_right_mean = 0.0;

                // Try a subset of features (sqrt(12) ≈ 3-4)
                let n_try = 4;
                for fi in 0..12 {
                    if (fi + t) % 3 != 0 && fi < 12 - n_try {
                        continue;
                    } // subsample features

                    // Sort by feature value
                    let mut vals: Vec<(f64, f64)> = remaining
                        .iter()
                        .map(|&i| (features[i][fi], targets[i]))
                        .collect();
                    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    // Try splits at every 10th percentile
                    let step = (vals.len() / 10).max(1);
                    for si in (step..vals.len()).step_by(step) {
                        let thresh = vals[si].0;
                        let left: Vec<f64> = vals[..si].iter().map(|v| v.1).collect();
                        let right: Vec<f64> = vals[si..].iter().map(|v| v.1).collect();

                        if left.len() < min_leaf || right.len() < min_leaf {
                            continue;
                        }

                        let left_mean = left.iter().sum::<f64>() / left.len() as f64;
                        let right_mean = right.iter().sum::<f64>() / right.len() as f64;

                        let left_var: f64 = left.iter().map(|v| (v - left_mean).powi(2)).sum();
                        let right_var: f64 = right.iter().map(|v| (v - right_mean).powi(2)).sum();
                        let score = left_var + right_var;

                        if score < best_score {
                            best_score = score;
                            best_feat = fi;
                            best_thresh = thresh;
                            best_left_mean = left_mean;
                            best_right_mean = right_mean;
                        }
                    }
                }

                if best_score == f64::MAX {
                    break;
                }

                splits.push((best_feat, best_thresh, best_left_mean, best_right_mean));

                // Continue with the larger partition
                let (left_idx, right_idx): (Vec<_>, Vec<_>) = remaining
                    .iter()
                    .partition(|&&i| features[i][best_feat] <= best_thresh);
                remaining = if left_idx.len() > right_idx.len() {
                    left_idx
                } else {
                    right_idx
                };
            }

            trees.push(DecisionTree { splits });
        }

        trees
    }

    fn rf_predict(trees: &[DecisionTree], features: &[f64; 12]) -> f64 {
        if trees.is_empty() {
            return 0.0;
        }
        let predictions: Vec<f64> = trees.iter().map(|t| t.predict(features)).collect();
        predictions.iter().sum::<f64>() / predictions.len() as f64
    }

    fn compute_rms(errors: &[f64]) -> f64 {
        if errors.is_empty() {
            return 0.0;
        }
        (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. STANDARD 5-FOLD CV (baseline)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_standard_cv() {
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3 && n.n >= 3)
            .collect();
        let n = nuclei.len();
        let fold_size = n / 5;

        let mut all_errors = Vec::new();

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            let mut train_f = Vec::new();
            let mut train_t = Vec::new();
            for (i, nuc) in nuclei.iter().enumerate() {
                if i >= test_start && i < test_end {
                    continue;
                }
                let dz = dz_binding_energy(nuc.z, nuc.n);
                train_f.push(extract_features(nuc.z, nuc.n));
                train_t.push(nuc.binding_energy_mev - dz);
            }

            let trees = train_rf(&train_f, &train_t, 50, 8, 3);

            for i in test_start..test_end {
                let nuc = &nuclei[i];
                let feat = extract_features(nuc.z, nuc.n);
                let dz = dz_binding_energy(nuc.z, nuc.n);
                let pred = dz + rf_predict(&trees, &feat);
                all_errors.push(pred - nuc.binding_energy_mev);
            }
        }

        let rms = compute_rms(&all_errors);
        eprintln!("\n=== STANDARD 5-FOLD CV ===");
        eprintln!("Nuclei: {}, RMS: {:.3} MeV", all_errors.len(), rms);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. PROTON-NUMBER HOLDOUT (leave-one-element-out)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_proton_holdout() {
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3 && n.n >= 3)
            .collect();

        eprintln!("\n=== PROTON-NUMBER HOLDOUT ===");
        eprintln!("Train on all Z except one, test on held-out Z.\n");

        // Test specific important elements
        let test_elements = [20u16, 28, 50, 82, 26, 40, 62, 92]; // Ca, Ni, Sn, Pb, Fe, Zr, Sm, U

        let mut overall_errors = Vec::new();

        eprintln!(
            "{:>4} {:>8} {:>10} {:>10}",
            "Z", "N_test", "RMS(MeV)", "DZ_RMS"
        );
        eprintln!("{}", "-".repeat(36));

        for &holdout_z in &test_elements {
            let mut train_f = Vec::new();
            let mut train_t = Vec::new();
            let mut test_nuclei = Vec::new();

            for nuc in &nuclei {
                let dz = dz_binding_energy(nuc.z, nuc.n);
                if nuc.z == holdout_z {
                    test_nuclei.push((nuc, dz));
                } else {
                    train_f.push(extract_features(nuc.z, nuc.n));
                    train_t.push(nuc.binding_energy_mev - dz);
                }
            }

            if test_nuclei.is_empty() {
                continue;
            }

            let trees = train_rf(&train_f, &train_t, 50, 8, 3);

            let mut errors = Vec::new();
            let mut dz_errors = Vec::new();
            for (nuc, dz) in &test_nuclei {
                let feat = extract_features(nuc.z, nuc.n);
                let pred = dz + rf_predict(&trees, &feat);
                errors.push(pred - nuc.binding_energy_mev);
                dz_errors.push(dz - nuc.binding_energy_mev);
            }

            let rms = compute_rms(&errors);
            let dz_rms = compute_rms(&dz_errors);
            eprintln!(
                "{:>4} {:>8} {:>10.3} {:>10.3}",
                holdout_z,
                test_nuclei.len(),
                rms,
                dz_rms
            );
            overall_errors.extend(errors);
        }

        let overall_rms = compute_rms(&overall_errors);
        eprintln!(
            "\nOverall proton-holdout RMS: {:.3} MeV ({} nuclei)",
            overall_rms,
            overall_errors.len()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. ISOTOPE CHAIN HOLDOUT (leave-one-chain-out)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_chain_holdout() {
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3 && n.n >= 3)
            .collect();

        eprintln!("\n=== ISOTOPE CHAIN HOLDOUT ===");
        eprintln!("Hold out entire isotope chains (same Z, consecutive N).\n");

        // Group by Z
        use std::collections::HashMap;
        let mut by_z: HashMap<u16, Vec<_>> = HashMap::new();
        for nuc in &nuclei {
            by_z.entry(nuc.z).or_default().push(nuc);
        }

        // Hold out every 5th Z chain
        let mut holdout_errors = Vec::new();
        let mut holdout_dz_errors = Vec::new();

        for (&z, chain) in &by_z {
            if z % 5 != 0 {
                continue;
            } // holdout Z=5,10,15,...

            let mut train_f = Vec::new();
            let mut train_t = Vec::new();
            for nuc in &nuclei {
                if nuc.z == z {
                    continue;
                }
                let dz = dz_binding_energy(nuc.z, nuc.n);
                train_f.push(extract_features(nuc.z, nuc.n));
                train_t.push(nuc.binding_energy_mev - dz);
            }

            let trees = train_rf(&train_f, &train_t, 50, 8, 3);

            for nuc in chain {
                let feat = extract_features(nuc.z, nuc.n);
                let dz = dz_binding_energy(nuc.z, nuc.n);
                let pred = dz + rf_predict(&trees, &feat);
                holdout_errors.push(pred - nuc.binding_energy_mev);
                holdout_dz_errors.push(dz - nuc.binding_energy_mev);
            }
        }

        let rms = compute_rms(&holdout_errors);
        let dz_rms = compute_rms(&holdout_dz_errors);
        eprintln!(
            "Chain-holdout RMS: {:.3} MeV (DZ baseline: {:.3})",
            rms, dz_rms
        );
        eprintln!("Nuclei in holdout: {}", holdout_errors.len());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. NEUTRON-RICH FRONTIER HOLDOUT
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_frontier_holdout() {
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3 && n.n >= 3)
            .collect();

        eprintln!("\n=== NEUTRON-RICH FRONTIER HOLDOUT ===");
        eprintln!("Train on N/Z < 1.4, test on N/Z >= 1.4 (neutron-rich frontier).\n");

        let threshold = 1.4;

        let mut train_f = Vec::new();
        let mut train_t = Vec::new();
        let mut test_nuclei = Vec::new();

        for nuc in &nuclei {
            let ratio = nuc.n as f64 / nuc.z as f64;
            let dz = dz_binding_energy(nuc.z, nuc.n);
            if ratio >= threshold {
                test_nuclei.push((nuc, dz));
            } else {
                train_f.push(extract_features(nuc.z, nuc.n));
                train_t.push(nuc.binding_energy_mev - dz);
            }
        }

        eprintln!(
            "Training set: {} nuclei (N/Z < {:.1})",
            train_f.len(),
            threshold
        );
        eprintln!(
            "Test set:     {} nuclei (N/Z >= {:.1})",
            test_nuclei.len(),
            threshold
        );

        let trees = train_rf(&train_f, &train_t, 50, 8, 3);

        let mut errors = Vec::new();
        let mut dz_errors = Vec::new();
        for (nuc, dz) in &test_nuclei {
            let feat = extract_features(nuc.z, nuc.n);
            let pred = dz + rf_predict(&trees, &feat);
            errors.push(pred - nuc.binding_energy_mev);
            dz_errors.push(dz - nuc.binding_energy_mev);
        }

        let rms = compute_rms(&errors);
        let dz_rms = compute_rms(&dz_errors);
        let bias = errors.iter().sum::<f64>() / errors.len() as f64;
        eprintln!("\nFrontier RMS:     {:.3} MeV", rms);
        eprintln!("DZ-only RMS:      {:.3} MeV", dz_rms);
        eprintln!("Bias:             {:+.3} MeV", bias);
        eprintln!("Improvement:      {:.1}x", dz_rms / rms);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. SUPERHEAVY HOLDOUT (Z >= 100)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_superheavy_holdout() {
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3 && n.n >= 3)
            .collect();

        eprintln!("\n=== SUPERHEAVY HOLDOUT (Z >= 100) ===");
        eprintln!("Train on Z < 100, test on Z >= 100.\n");

        let z_cut = 100u16;

        let mut train_f = Vec::new();
        let mut train_t = Vec::new();
        let mut test_nuclei = Vec::new();

        for nuc in &nuclei {
            let dz = dz_binding_energy(nuc.z, nuc.n);
            if nuc.z >= z_cut {
                test_nuclei.push((nuc, dz));
            } else {
                train_f.push(extract_features(nuc.z, nuc.n));
                train_t.push(nuc.binding_energy_mev - dz);
            }
        }

        eprintln!("Training: {} nuclei (Z < {})", train_f.len(), z_cut);
        eprintln!("Test:     {} nuclei (Z >= {})", test_nuclei.len(), z_cut);

        if test_nuclei.is_empty() {
            eprintln!(
                "No measured superheavy nuclei in AME2020 with Z >= {}",
                z_cut
            );
            return;
        }

        let trees = train_rf(&train_f, &train_t, 50, 8, 3);

        let mut errors = Vec::new();
        let mut dz_errors = Vec::new();
        eprintln!(
            "\n{:>4} {:>4} {:>10} {:>10} {:>10}",
            "Z", "A", "Pred(MeV)", "Exp(MeV)", "Err(MeV)"
        );
        eprintln!("{}", "-".repeat(42));

        for (nuc, dz) in &test_nuclei {
            let feat = extract_features(nuc.z, nuc.n);
            let pred = dz + rf_predict(&trees, &feat);
            let err = pred - nuc.binding_energy_mev;
            eprintln!(
                "{:>4} {:>4} {:>10.2} {:>10.2} {:>10.2}",
                nuc.z,
                nuc.z + nuc.n,
                pred,
                nuc.binding_energy_mev,
                err
            );
            errors.push(err);
            dz_errors.push(dz - nuc.binding_energy_mev);
        }

        let rms = compute_rms(&errors);
        let dz_rms = compute_rms(&dz_errors);
        eprintln!(
            "\nSuperheavy holdout RMS: {:.3} MeV (DZ: {:.3})",
            rms, dz_rms
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. SUMMARY — all splits in one view
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_summary() {
        eprintln!("\n{} VALIDATION SUMMARY {}", "=".repeat(20), "=".repeat(20));
        eprintln!("Run the individual tests for detailed results.");
        eprintln!("Expected hierarchy (easiest → hardest):");
        eprintln!("  Standard 5-fold CV  < Chain holdout < Proton holdout < Frontier holdout");
        eprintln!("If train=CV for HDC-LTC, but harder splits show degradation,");
        eprintln!("that indicates good interpolation but limited extrapolation.");
        eprintln!("If harder splits also hold, the regularization is genuine.");
    }
}
