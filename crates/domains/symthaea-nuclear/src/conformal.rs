// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Split Conformal Prediction for Nuclear Masses
//!
//! Implements locally-adaptive (normalized) nonconformity scores on top of
//! the Random Forest mass predictor. The calibration set is the last 20% of
//! the AME2020 measured nuclei (Z≥3, N≥3); the first 80% overlaps with the
//! training data.
//!
//! **Algorithm**: Split conformal prediction (Vovk et al., 2005) with
//! normalized residuals `|ŷ − y| / σ_RF`, where `σ_RF` is the per-prediction
//! uncertainty from the Random Forest ensemble.
//!
//! Reference: Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*. Springer.

#[cfg(test)]
mod tests {
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::ml_mass::MlMassPredictor;

    fn predictor() -> MlMassPredictor {
        MlMassPredictor::new()
    }

    #[test]
    fn conformal_calibration() {
        let pred = predictor();
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3 && n.n >= 3)
            .collect();

        // Split: first 80% train (already used by predictor), last 20% calibration
        let n = nuclei.len();
        let cal_start = n * 4 / 5;
        let calibration = &nuclei[cal_start..];

        // Compute normalized nonconformity scores
        let mut scores: Vec<f64> = calibration
            .iter()
            .map(|nuc| {
                let p = pred.predict(nuc.z, nuc.n);
                let residual = (p.binding_energy - nuc.binding_energy_mev).abs();
                let sigma = p.uncertainty.max(0.01); // avoid division by zero
                residual / sigma // normalized score
            })
            .collect();
        scores.sort_by(|a, b| a.total_cmp(b));

        // 95% quantile
        let n_cal = scores.len();
        let k_95 = ((n_cal as f64 + 1.0) * 0.95).ceil() as usize;
        let q_95 = scores[k_95.min(n_cal - 1)];

        // 90% quantile
        let k_90 = ((n_cal as f64 + 1.0) * 0.90).ceil() as usize;
        let q_90 = scores[k_90.min(n_cal - 1)];

        eprintln!("\n=== CONFORMAL PREDICTION CALIBRATION ===");
        eprintln!("Calibration set: {} nuclei", n_cal);
        eprintln!("95% quantile (q_hat): {:.3}", q_95);
        eprintln!("90% quantile: {:.3}", q_90);
        eprintln!(
            "\nInterpretation: 95% prediction interval = ŷ ± {:.3} × σ_RF",
            q_95
        );

        // Now verify coverage on the FULL dataset
        let mut covered_95 = 0usize;
        let mut covered_90 = 0usize;
        let mut total = 0usize;
        let mut interval_widths = Vec::new();

        for nuc in &nuclei {
            let p = pred.predict(nuc.z, nuc.n);
            let sigma = p.uncertainty.max(0.01);
            let residual = (p.binding_energy - nuc.binding_energy_mev).abs();
            let half_width_95 = q_95 * sigma;
            let half_width_90 = q_90 * sigma;

            if residual <= half_width_95 {
                covered_95 += 1;
            }
            if residual <= half_width_90 {
                covered_90 += 1;
            }
            interval_widths.push(2.0 * half_width_95);
            total += 1;
        }

        let cov_95 = covered_95 as f64 / total as f64 * 100.0;
        let cov_90 = covered_90 as f64 / total as f64 * 100.0;
        let mean_width = interval_widths.iter().sum::<f64>() / interval_widths.len() as f64;
        let median_width = {
            let mut sorted = interval_widths.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            sorted[sorted.len() / 2]
        };

        eprintln!("\n--- Coverage Verification (full dataset) ---");
        eprintln!(
            "95% target: actual coverage = {:.1}% ({}/{})",
            cov_95, covered_95, total
        );
        eprintln!(
            "90% target: actual coverage = {:.1}% ({}/{})",
            cov_90, covered_90, total
        );
        eprintln!("\n--- Interval Width Statistics ---");
        eprintln!("Mean 95% interval width: {:.3} MeV", mean_width);
        eprintln!("Median 95% interval width: {:.3} MeV", median_width);
        eprintln!(
            "Min: {:.3}, Max: {:.3} MeV",
            interval_widths.iter().cloned().fold(f64::MAX, f64::min),
            interval_widths.iter().cloned().fold(0.0f64, f64::max)
        );

        // Compare with naive σ-based intervals
        eprintln!("\n--- Comparison: Conformal vs Naive ---");
        eprintln!("Naive 2σ interval would give: ŷ ± 2×σ_RF");
        let mut naive_covered = 0;
        for nuc in &nuclei {
            let p = pred.predict(nuc.z, nuc.n);
            let residual = (p.binding_energy - nuc.binding_energy_mev).abs();
            if residual <= 2.0 * p.uncertainty {
                naive_covered += 1;
            }
        }
        eprintln!(
            "Naive 2σ coverage: {:.1}% (should be ~95% if calibrated, likely higher or lower)",
            naive_covered as f64 / total as f64 * 100.0
        );

        // Regional breakdown
        eprintln!("\n--- Regional Conformal Intervals ---");
        for (name, z_min, z_max) in [
            ("Light (Z<20)", 3u16, 19u16),
            ("Medium (20-50)", 20, 50),
            ("Heavy (50-82)", 50, 82),
            ("Actinides (Z>82)", 83, 120),
        ] {
            let region: Vec<f64> = nuclei
                .iter()
                .filter(|n| n.z >= z_min && n.z <= z_max)
                .map(|n| {
                    let p = pred.predict(n.z, n.n);
                    2.0 * q_95 * p.uncertainty.max(0.01)
                })
                .collect();
            if !region.is_empty() {
                let mean = region.iter().sum::<f64>() / region.len() as f64;
                eprintln!(
                    "  {}: mean 95% interval = {:.3} MeV ({} nuclei)",
                    name,
                    mean,
                    region.len()
                );
            }
        }
    }
}
