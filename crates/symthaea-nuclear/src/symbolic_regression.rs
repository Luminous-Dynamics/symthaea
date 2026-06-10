// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Template-based symbolic regression on RF residuals.
//!
//! Enumerates power-law formula templates over the Random Forest residuals
//! and finds interpretable analytic corrections to the DZ+RF mass model.
//!
//! Templates:
//! - 1-term: `c0 + c1 * feat_a^p`
//! - 2-term: `c0 + c1 * feat_a^p1 * feat_b^p2`
//!
//! Fit via closed-form OLS, ranked by R^2.

#[cfg(test)]
mod tests {
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::deformation::frdm_deformation;
    use crate::ml_mass::MlMassPredictor;

    const FEATURE_NAMES: &[&str] = &[
        "Z", "N", "A", "isospin", "pairing", "shell_Z", "shell_N", "coulomb", "surface", "valence",
        "deform", "radius",
    ];

    const MAGIC_Z: &[f64] = &[2.0, 8.0, 20.0, 28.0, 50.0, 82.0, 114.0, 126.0];
    const MAGIC_N: &[f64] = &[2.0, 8.0, 20.0, 28.0, 50.0, 82.0, 126.0, 184.0];

    const POWERS: &[f64] = &[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0];

    fn extract_features(z: u16, n: u16) -> [f64; 12] {
        let z_f = z as f64;
        let n_f = n as f64;
        let a = z_f + n_f;

        let isospin = (n_f - z_f) / a;
        let pairing = if a as u32 % 2 != 0 {
            0.0
        } else if z as u32 % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let shell_z = MAGIC_Z
            .iter()
            .map(|&m| (z_f - m).abs())
            .fold(f64::INFINITY, f64::min);
        let shell_n = MAGIC_N
            .iter()
            .map(|&m| (n_f - m).abs())
            .fold(f64::INFINITY, f64::min);
        let coulomb = z_f * (z_f - 1.0) / a.powf(1.0 / 3.0);
        let surface = a.powf(2.0 / 3.0);
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
        let (beta2, _) = frdm_deformation(z, n);

        [
            z_f,
            n_f,
            a,
            isospin,
            pairing,
            shell_z,
            shell_n,
            coulomb,
            surface,
            valence,
            beta2,
            a.powf(1.0 / 3.0),
        ]
    }

    /// Format a power as a readable string.
    fn power_str(p: f64) -> &'static str {
        if (p - (-2.0)).abs() < 1e-9 {
            "^(-2)"
        } else if (p - (-1.0)).abs() < 1e-9 {
            "^(-1)"
        } else if (p - (-0.5)).abs() < 1e-9 {
            "^(-1/2)"
        } else if (p - 0.5).abs() < 1e-9 {
            "^(1/2)"
        } else if (p - 1.0).abs() < 1e-9 {
            ""
        } else if (p - 2.0).abs() < 1e-9 {
            "^2"
        } else {
            "^?"
        }
    }

    struct FormulaResult {
        description: String,
        r_squared: f64,
        c0: f64,
        c1: f64,
        complexity: u8, // 1 = single feature, 2 = product of two
    }

    /// Fit c0, c1 by OLS for y = c0 + c1 * x.
    /// Returns (c0, c1, R^2) or None if degenerate.
    fn ols_fit(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64)> {
        let n = x.len() as f64;
        if n < 3.0 {
            return None;
        }

        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-20 {
            return None;
        }

        let c1 = (n * sum_xy - sum_x * sum_y) / denom;
        let c0 = (sum_y - c1 * sum_x) / n;

        let y_mean = sum_y / n;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - c0 - c1 * xi).powi(2))
            .sum();

        if ss_tot < 1e-20 {
            return None;
        }

        let r2 = 1.0 - ss_res / ss_tot;
        Some((c0, c1, r2))
    }

    /// Compute column x_i = feat_a^p, returning None if any value is non-finite.
    fn compute_column(features: &[[f64; 12]], feat_idx: usize, power: f64) -> Option<Vec<f64>> {
        let col: Vec<f64> = features.iter().map(|f| f[feat_idx].powf(power)).collect();
        if col.iter().all(|v| v.is_finite()) {
            Some(col)
        } else {
            None
        }
    }

    /// Compute column x_i = feat_a^p1 * feat_b^p2.
    fn compute_column_2(
        features: &[[f64; 12]],
        a_idx: usize,
        p1: f64,
        b_idx: usize,
        p2: f64,
    ) -> Option<Vec<f64>> {
        let col: Vec<f64> = features
            .iter()
            .map(|f| f[a_idx].powf(p1) * f[b_idx].powf(p2))
            .collect();
        if col.iter().all(|v| v.is_finite()) {
            Some(col)
        } else {
            None
        }
    }

    /// Check if a template has a physics motivation.
    fn physics_motivation(desc: &str) -> Option<&'static str> {
        if desc.contains("coulomb") && desc.contains("^(-1)") {
            Some("Coulomb redistribution correction")
        } else if desc.contains("isospin") && desc.contains("^2") {
            Some("Symmetry energy (Wigner term)")
        } else if desc.contains("shell_Z") || desc.contains("shell_N") {
            Some("Shell correction (proximity to magic numbers)")
        } else if desc.contains("pairing") {
            Some("Pairing interaction")
        } else if desc.contains("valence") {
            Some("Residual proton-neutron interaction (Np*Nn)")
        } else if desc.contains("deform") {
            Some("Deformation energy correction")
        } else if desc.contains("surface") {
            Some("Surface energy correction")
        } else if desc.contains("radius") {
            Some("Nuclear radius scaling")
        } else {
            None
        }
    }

    #[test]
    fn symbolic_regression_on_residuals() {
        let pred = MlMassPredictor::new();
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3 && n.n >= 3)
            .collect();

        // Collect RF residuals and features
        let mut features_all = Vec::new();
        let mut residuals = Vec::new();
        for nuc in &nuclei {
            let p = pred.predict(nuc.z, nuc.n);
            let feat = extract_features(nuc.z, nuc.n);
            features_all.push(feat);
            residuals.push(nuc.binding_energy_mev - p.binding_energy);
        }

        let n_nuclei = nuclei.len();
        let y_mean: f64 = residuals.iter().sum::<f64>() / n_nuclei as f64;
        let ss_tot: f64 = residuals.iter().map(|r| (r - y_mean).powi(2)).sum();
        let rms_residual = (residuals.iter().map(|r| r * r).sum::<f64>() / n_nuclei as f64).sqrt();

        eprintln!("\n=== SYMBOLIC REGRESSION ON RF RESIDUALS ===");
        eprintln!("Nuclei: {} (measured, Z>=3, N>=3)", n_nuclei);
        eprintln!("Mean residual: {:.4} MeV", y_mean);
        eprintln!("RMS residual:  {:.4} MeV", rms_residual);
        eprintln!("SS_tot:        {:.2}", ss_tot);

        let mut results: Vec<FormulaResult> = Vec::new();

        // === 1-term templates: f = c0 + c1 * feat_a^p ===
        for (a_idx, &a_name) in FEATURE_NAMES.iter().enumerate() {
            for &p in POWERS {
                if let Some(col) = compute_column(&features_all, a_idx, p) {
                    if let Some((c0, c1, r2)) = ols_fit(&col, &residuals) {
                        if r2.is_finite() {
                            let desc = format!("{}{}", a_name, power_str(p));
                            results.push(FormulaResult {
                                description: desc,
                                r_squared: r2,
                                c0,
                                c1,
                                complexity: 1,
                            });
                        }
                    }
                }
            }
        }

        // === 2-term templates: f = c0 + c1 * feat_a^p1 * feat_b^p2 ===
        for (a_idx, &a_name) in FEATURE_NAMES.iter().enumerate() {
            for &p1 in POWERS {
                for (b_idx, &b_name) in FEATURE_NAMES.iter().enumerate() {
                    if b_idx <= a_idx {
                        continue; // avoid duplicates
                    }
                    for &p2 in POWERS {
                        if let Some(col) = compute_column_2(&features_all, a_idx, p1, b_idx, p2) {
                            if let Some((c0, c1, r2)) = ols_fit(&col, &residuals) {
                                if r2.is_finite() {
                                    let desc = format!(
                                        "{}{} * {}{}",
                                        a_name,
                                        power_str(p1),
                                        b_name,
                                        power_str(p2)
                                    );
                                    results.push(FormulaResult {
                                        description: desc,
                                        r_squared: r2,
                                        c0,
                                        c1,
                                        complexity: 2,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by R^2 descending
        results.sort_by(|a, b| {
            b.r_squared
                .partial_cmp(&a.r_squared)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // === Top 20 by R^2 ===
        eprintln!("\n--- TOP 20 FORMULAS (by R^2) ---");
        eprintln!(
            "{:>4} {:>6} {:>10} {:>12} {:>12}  {}",
            "Rank", "Compl", "R^2", "c0", "c1", "Formula: residual = c0 + c1 * <template>"
        );
        eprintln!("{}", "-".repeat(90));
        for (i, r) in results.iter().take(20).enumerate() {
            eprintln!(
                "{:>4} {:>6} {:>10.6} {:>12.4} {:>12.6}  {}",
                i + 1,
                r.complexity,
                r.r_squared,
                r.c0,
                r.c1,
                r.description
            );
        }

        // === Physics-motivated templates ===
        eprintln!("\n--- PHYSICS-MOTIVATED TEMPLATES ---");
        eprintln!(
            "{:>4} {:>10} {:>12} {:>12}  {:30}  {}",
            "Rank", "R^2", "c0", "c1", "Formula", "Motivation"
        );
        eprintln!("{}", "-".repeat(110));
        let mut phys_count = 0;
        for r in &results {
            if let Some(motivation) = physics_motivation(&r.description) {
                phys_count += 1;
                eprintln!(
                    "{:>4} {:>10.6} {:>12.4} {:>12.6}  {:30}  {}",
                    phys_count, r.r_squared, r.c0, r.c1, r.description, motivation
                );
                if phys_count >= 30 {
                    break;
                }
            }
        }

        // === Pareto front: complexity vs R^2 ===
        eprintln!("\n--- PARETO FRONT (complexity vs R^2) ---");
        let mut best_r2_by_complexity = [f64::NEG_INFINITY; 3]; // complexity 0,1,2
        let mut best_formula_by_complexity: [Option<&FormulaResult>; 3] = [None; 3];
        for r in &results {
            let c = r.complexity as usize;
            if c < 3 && r.r_squared > best_r2_by_complexity[c] {
                best_r2_by_complexity[c] = r.r_squared;
                best_formula_by_complexity[c] = Some(r);
            }
        }
        for c in 1..=2 {
            if let Some(r) = best_formula_by_complexity[c] {
                eprintln!(
                    "  Complexity {}: R^2 = {:.6}, formula: {} (c0={:.4}, c1={:.6})",
                    c, r.r_squared, r.description, r.c0, r.c1
                );
            }
        }

        // Summary statistics
        let total_templates = results.len();
        let positive_r2 = results.iter().filter(|r| r.r_squared > 0.0).count();
        let good_r2 = results.iter().filter(|r| r.r_squared > 0.1).count();
        eprintln!("\n--- SUMMARY ---");
        eprintln!("Total templates evaluated: {}", total_templates);
        eprintln!("Templates with R^2 > 0:   {}", positive_r2);
        eprintln!("Templates with R^2 > 0.1: {}", good_r2);

        if let Some(best) = results.first() {
            let rms_after = (residuals
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    // We don't have the x column cached, so just report the R^2-based estimate
                    let _ = (i, r);
                    0.0
                })
                .sum::<f64>())
            .sqrt();
            let _ = rms_after;
            eprintln!(
                "Best formula: {} (R^2={:.6})",
                best.description, best.r_squared
            );
            let estimated_rms_reduction = (1.0 - best.r_squared).sqrt();
            eprintln!(
                "Estimated RMS after correction: {:.4} MeV (factor {:.2}x from {:.4})",
                rms_residual * estimated_rms_reduction,
                1.0 / estimated_rms_reduction,
                rms_residual
            );
        }

        // Verify we found something meaningful
        assert!(
            !results.is_empty(),
            "Should enumerate at least some templates"
        );
        assert!(
            results[0].r_squared > -1.0,
            "Best R^2 should be finite: {}",
            results[0].r_squared
        );
    }
}
