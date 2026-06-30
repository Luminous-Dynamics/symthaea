// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark tests for bandgap prediction: baseline vs RF, cross-validation,
//! and known-semiconductor spot checks.

#[cfg(test)]
mod tests {
    use crate::bandgap_baseline::electronegativity_bandgap;
    use crate::ml_bandgap::BandgapPredictor;
    use crate::training_data::{CrystalSystem, load_training_data};

    /// Compute MAE, RMSE, R^2 from parallel (predicted, actual) slices.
    fn metrics(predicted: &[f64], actual: &[f64]) -> (f64, f64, f64) {
        let n = predicted.len() as f64;
        let mae = predicted
            .iter()
            .zip(actual)
            .map(|(p, a)| (p - a).abs())
            .sum::<f64>()
            / n;
        let rmse = (predicted
            .iter()
            .zip(actual)
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let mean_actual = actual.iter().sum::<f64>() / n;
        let ss_res: f64 = predicted
            .iter()
            .zip(actual)
            .map(|(p, a)| (a - p).powi(2))
            .sum();
        let ss_tot: f64 = actual.iter().map(|a| (a - mean_actual).powi(2)).sum();
        let r2 = if ss_tot > 1e-10 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
        (mae, rmse, r2)
    }

    // -----------------------------------------------------------------------
    // Benchmark 1: Electronegativity baseline on all training data
    // -----------------------------------------------------------------------
    #[test]
    fn benchmark_baseline() {
        let data = load_training_data();

        let predicted: Vec<f64> = data
            .iter()
            .map(|d| electronegativity_bandgap(&d.composition))
            .collect();
        let actual: Vec<f64> = data.iter().map(|d| d.bandgap_exp()).collect();

        let (mae, rmse, r2) = metrics(&predicted, &actual);

        eprintln!("\n========================================");
        eprintln!("  Electronegativity Baseline (n={})", data.len());
        eprintln!("========================================");
        eprintln!("  MAE:  {:.3} eV", mae);
        eprintln!("  RMSE: {:.3} eV", rmse);
        eprintln!("  R^2:  {:.4}", r2);
        eprintln!("========================================\n");

        // Baseline is crude but should capture broad trends
        assert!(
            r2 > 0.0,
            "Baseline R^2 = {} should be positive (captures some variance)",
            r2
        );
    }

    // -----------------------------------------------------------------------
    // Benchmark 2: RF predictor on training data (interpolation)
    // -----------------------------------------------------------------------
    #[test]
    fn benchmark_rf_predictor() {
        let predictor = BandgapPredictor::new();
        let data = load_training_data();

        let baseline_pred: Vec<f64> = data
            .iter()
            .map(|d| electronegativity_bandgap(&d.composition))
            .collect();
        let rf_pred: Vec<f64> = data
            .iter()
            .map(|d| predictor.predict(&d.composition, d.crystal()).bandgap)
            .collect();
        let actual: Vec<f64> = data.iter().map(|d| d.bandgap_exp()).collect();

        let (b_mae, b_rmse, b_r2) = metrics(&baseline_pred, &actual);
        let (rf_mae, rf_rmse, rf_r2) = metrics(&rf_pred, &actual);

        eprintln!("\n=============================================");
        eprintln!("  RF Predictor — Training Set (n={})", data.len());
        eprintln!("=============================================");
        eprintln!("              Baseline        RF");
        eprintln!("  MAE:     {:>8.3} eV   {:>8.3} eV", b_mae, rf_mae);
        eprintln!("  RMSE:    {:>8.3} eV   {:>8.3} eV", b_rmse, rf_rmse);
        eprintln!("  R^2:     {:>8.4}      {:>8.4}", b_r2, rf_r2);
        eprintln!(
            "  RMSE improvement: {:.1}x",
            if rf_rmse > 1e-10 {
                b_rmse / rf_rmse
            } else {
                f64::INFINITY
            }
        );
        eprintln!("=============================================\n");

        // RF must improve over baseline on training data
        assert!(
            rf_rmse < b_rmse,
            "RF RMSE ({:.3}) should be less than baseline ({:.3})",
            rf_rmse,
            b_rmse
        );
        assert!(
            rf_r2 > b_r2,
            "RF R^2 ({:.4}) should exceed baseline ({:.4})",
            rf_r2,
            b_r2
        );
    }

    // -----------------------------------------------------------------------
    // Benchmark 3: 5-fold cross-validation
    // -----------------------------------------------------------------------
    #[test]
    fn benchmark_cross_validation() {
        let (cv_mae, cv_rmse) = BandgapPredictor::cross_validate();

        // Also compute baseline CV for comparison
        let data = load_training_data();
        let n = data.len();
        let fold_size = n / 5;
        let mut b_abs = 0.0;
        let mut b_sq = 0.0;
        let mut count = 0;

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };
            for i in test_start..test_end {
                let d = &data[i];
                let pred = electronegativity_bandgap(&d.composition);
                let err = pred - d.bandgap_exp();
                b_abs += err.abs();
                b_sq += err * err;
                count += 1;
            }
        }
        let b_mae = b_abs / count as f64;
        let b_rmse = (b_sq / count as f64).sqrt();

        eprintln!("\n=============================================");
        eprintln!("  5-Fold Cross-Validation (n={})", n);
        eprintln!("=============================================");
        eprintln!("              Baseline        RF (CV)");
        eprintln!("  MAE:     {:>8.3} eV   {:>8.3} eV", b_mae, cv_mae);
        eprintln!("  RMSE:    {:>8.3} eV   {:>8.3} eV", b_rmse, cv_rmse);
        eprintln!(
            "  RMSE improvement: {:.1}x",
            if cv_rmse > 1e-10 {
                b_rmse / cv_rmse
            } else {
                f64::INFINITY
            }
        );
        eprintln!("=============================================\n");

        // CV error should be higher than training but still reasonable
        assert!(
            cv_rmse < b_rmse * 1.5,
            "CV RMSE ({:.3}) should not be much worse than baseline ({:.3})",
            cv_rmse,
            b_rmse
        );
    }

    // -----------------------------------------------------------------------
    // Benchmark 4: Known semiconductor spot checks
    // -----------------------------------------------------------------------
    #[test]
    fn benchmark_known_semiconductors() {
        let predictor = BandgapPredictor::new();

        // (name, composition, crystal, experimental_bandgap_eV, tolerance_eV)
        let known = [
            ("Si", vec![(14u8, 1.0)], CrystalSystem::Cubic, 1.12, 0.8),
            (
                "GaAs",
                vec![(31, 0.5), (33, 0.5)],
                CrystalSystem::Cubic,
                1.42,
                0.8,
            ),
            (
                "GaN",
                vec![(31, 0.5), (7, 0.5)],
                CrystalSystem::Hexagonal,
                3.39,
                1.0,
            ),
            (
                "ZnO",
                vec![(30, 0.5), (8, 0.5)],
                CrystalSystem::Hexagonal,
                3.37,
                1.0,
            ),
            (
                "C (diamond)",
                vec![(6, 1.0)],
                CrystalSystem::Cubic,
                5.47,
                1.5,
            ),
            (
                "CdTe",
                vec![(48, 0.5), (52, 0.5)],
                CrystalSystem::Cubic,
                1.49,
                1.0,
            ),
            (
                "SiC (3C)",
                vec![(14, 0.5), (6, 0.5)],
                CrystalSystem::Cubic,
                2.36,
                1.0,
            ),
            (
                "NaCl",
                vec![(11, 0.5), (17, 0.5)],
                CrystalSystem::Cubic,
                8.50,
                2.0,
            ),
            (
                "MgO",
                vec![(12, 0.5), (8, 0.5)],
                CrystalSystem::Cubic,
                7.83,
                2.0,
            ),
            (
                "InP",
                vec![(49, 0.5), (15, 0.5)],
                CrystalSystem::Cubic,
                1.35,
                0.8,
            ),
        ];

        eprintln!("\n=============================================");
        eprintln!("  Known Semiconductor Spot Checks");
        eprintln!("=============================================");
        eprintln!(
            "{:<15} {:>8} {:>8} {:>8} {:>8}  {}",
            "Material", "Exp(eV)", "Base", "RF", "Err", "Status"
        );
        eprintln!("{}", "-".repeat(70));

        let mut pass_count = 0;
        let total = known.len();

        for (name, comp, crystal, exp, tol) in &known {
            let baseline = electronegativity_bandgap(comp);
            let pred = predictor.predict(comp, crystal);
            let err = (pred.bandgap - exp).abs();
            let status = if err < *tol { "PASS" } else { "WIDE" };
            if err < *tol {
                pass_count += 1;
            }
            eprintln!(
                "{:<15} {:>8.2} {:>8.2} {:>8.2} {:>8.2}  {}",
                name, exp, baseline, pred.bandgap, err, status
            );
        }

        eprintln!("\nPassed: {}/{} (within tolerance)\n", pass_count, total);

        // At least 70% should be within tolerance (generous tolerances)
        assert!(
            pass_count as f64 / total as f64 >= 0.7,
            "Only {}/{} known semiconductors within tolerance",
            pass_count,
            total
        );
    }
}
