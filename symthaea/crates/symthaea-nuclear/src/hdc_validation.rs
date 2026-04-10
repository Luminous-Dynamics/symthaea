// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hard validation splits for the HDC-LTC nuclear mass predictor.
//!
//! Five distinct train/test partitions that test generalization beyond
//! random CV:
//!
//! 1. **Standard 5-fold CV** — baseline with explicit RMS reporting
//! 2. **Proton holdout** — hold out magic/interesting Z values one at a time
//! 3. **Chain holdout** — hold out every 5th Z chain
//! 4. **Frontier holdout** — train on N/Z < 1.4, test on N/Z >= 1.4
//! 5. **Superheavy holdout** — train on Z < 100, test on Z >= 100

#[cfg(test)]
mod tests {
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::discovery::MeasuredNucleus;
    use crate::duflo_zuker::dz_binding_energy;
    use crate::hdc_mass::HdcMassPredictor;

    const EPOCHS: usize = 10;

    /// Load the standard measured nuclei set (Z >= 3).
    fn load_nuclei() -> Vec<MeasuredNucleus> {
        ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3)
            .collect()
    }

    /// Compute RMS error of HDC predictions vs measured BE on test set.
    fn hdc_rms(predictor: &HdcMassPredictor, test: &[MeasuredNucleus]) -> f64 {
        if test.is_empty() {
            return 0.0;
        }
        let sq: f64 = test
            .iter()
            .map(|nuc| {
                let pred = predictor.predict(nuc.z, nuc.n);
                (pred.binding_energy - nuc.binding_energy_mev).powi(2)
            })
            .sum();
        (sq / test.len() as f64).sqrt()
    }

    /// Compute RMS error of DZ-only baseline on test set.
    fn dz_rms(test: &[MeasuredNucleus]) -> f64 {
        if test.is_empty() {
            return 0.0;
        }
        let sq: f64 = test
            .iter()
            .map(|nuc| {
                let dz = dz_binding_energy(nuc.z, nuc.n);
                (dz - nuc.binding_energy_mev).powi(2)
            })
            .sum();
        (sq / test.len() as f64).sqrt()
    }

    // ── 1. Standard 5-fold CV ───────────────────────────────────────────

    #[test]
    fn hdc_validation_5fold_cv() {
        let nuclei = load_nuclei();
        let n = nuclei.len();
        let fold_size = n / 5;
        let mut total_sq = 0.0;
        let mut total_dz_sq = 0.0;
        let mut count = 0usize;

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            let train: Vec<_> = nuclei
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start || *i >= test_end)
                .map(|(_, nuc)| nuc.clone())
                .collect();
            let test: Vec<_> = nuclei[test_start..test_end].to_vec();

            let predictor = HdcMassPredictor::train_on(&train, EPOCHS, fold as u64);

            for nuc in &test {
                let pred = predictor.predict(nuc.z, nuc.n);
                let err = pred.binding_energy - nuc.binding_energy_mev;
                total_sq += err * err;

                let dz = dz_binding_energy(nuc.z, nuc.n);
                let dz_err = dz - nuc.binding_energy_mev;
                total_dz_sq += dz_err * dz_err;

                count += 1;
            }

            eprintln!(
                "  fold {}: train={}, test={}, HDC RMS={:.2}, DZ RMS={:.2}",
                fold,
                train.len(),
                test.len(),
                hdc_rms(&predictor, &test),
                dz_rms(&test),
            );
        }

        let rms = (total_sq / count as f64).sqrt();
        let dz = (total_dz_sq / count as f64).sqrt();
        eprintln!(
            "5-fold CV overall: HDC RMS={:.2} MeV, DZ RMS={:.2} MeV, N={}",
            rms, dz, count
        );

        assert!(
            rms.is_finite() && rms > 0.0 && rms < 500.0,
            "HDC CV RMS={}",
            rms
        );
    }

    // ── 2. Proton holdout ───────────────────────────────────────────────

    #[test]
    fn hdc_validation_proton_holdout() {
        let nuclei = load_nuclei();
        let holdout_zs: &[u16] = &[20, 28, 50, 82, 26, 40, 62, 92];

        for (i, &z_hold) in holdout_zs.iter().enumerate() {
            let train: Vec<_> = nuclei.iter().filter(|n| n.z != z_hold).cloned().collect();
            let test: Vec<_> = nuclei.iter().filter(|n| n.z == z_hold).cloned().collect();

            if test.is_empty() {
                eprintln!("  Z={}: no measured nuclei, skipping", z_hold);
                continue;
            }

            let predictor = HdcMassPredictor::train_on(&train, EPOCHS, 100 + i as u64);
            let rms = hdc_rms(&predictor, &test);
            let dz = dz_rms(&test);

            eprintln!(
                "  Z={}: train={}, test={}, HDC RMS={:.2} MeV, DZ RMS={:.2} MeV",
                z_hold,
                train.len(),
                test.len(),
                rms,
                dz,
            );

            assert!(rms.is_finite() && rms > 0.0, "Z={} HDC RMS={}", z_hold, rms);
        }
    }

    // ── 3. Chain holdout (every 5th Z) ──────────────────────────────────

    #[test]
    fn hdc_validation_chain_holdout() {
        let nuclei = load_nuclei();

        let train: Vec<_> = nuclei.iter().filter(|n| n.z % 5 != 0).cloned().collect();
        let test: Vec<_> = nuclei.iter().filter(|n| n.z % 5 == 0).cloned().collect();

        eprintln!(
            "Chain holdout (Z%%5==0): train={}, test={}",
            train.len(),
            test.len()
        );

        let predictor = HdcMassPredictor::train_on(&train, EPOCHS, 200);
        let rms = hdc_rms(&predictor, &test);
        let dz = dz_rms(&test);

        eprintln!("  HDC RMS={:.2} MeV, DZ RMS={:.2} MeV", rms, dz);

        assert!(
            rms.is_finite() && rms > 0.0,
            "Chain holdout HDC RMS={}",
            rms
        );
    }

    // ── 4. Frontier holdout (N/Z >= 1.4) ────────────────────────────────

    #[test]
    fn hdc_validation_frontier_holdout() {
        let nuclei = load_nuclei();

        let train: Vec<_> = nuclei
            .iter()
            .filter(|n| (n.n as f64 / n.z as f64) < 1.4)
            .cloned()
            .collect();
        let test: Vec<_> = nuclei
            .iter()
            .filter(|n| (n.n as f64 / n.z as f64) >= 1.4)
            .cloned()
            .collect();

        eprintln!(
            "Frontier holdout (N/Z >= 1.4): train={}, test={}",
            train.len(),
            test.len()
        );

        if test.is_empty() {
            eprintln!("  No neutron-rich frontier nuclei in AME2020, skipping");
            return;
        }

        let predictor = HdcMassPredictor::train_on(&train, EPOCHS, 300);
        let rms = hdc_rms(&predictor, &test);
        let dz = dz_rms(&test);

        eprintln!("  HDC RMS={:.2} MeV, DZ RMS={:.2} MeV", rms, dz);

        assert!(
            rms.is_finite() && rms > 0.0,
            "Frontier holdout HDC RMS={}",
            rms
        );
    }

    // ── 5. Superheavy holdout (Z >= 100) ────────────────────────────────

    #[test]
    fn hdc_validation_superheavy_holdout() {
        let nuclei = load_nuclei();

        let train: Vec<_> = nuclei.iter().filter(|n| n.z < 100).cloned().collect();
        let test: Vec<_> = nuclei.iter().filter(|n| n.z >= 100).cloned().collect();

        eprintln!(
            "Superheavy holdout (Z >= 100): train={}, test={}",
            train.len(),
            test.len()
        );

        if test.is_empty() {
            eprintln!("  No Z >= 100 measured nuclei in AME2020, skipping");
            return;
        }

        let predictor = HdcMassPredictor::train_on(&train, EPOCHS, 400);
        let rms = hdc_rms(&predictor, &test);
        let dz = dz_rms(&test);

        eprintln!("  HDC RMS={:.2} MeV, DZ RMS={:.2} MeV", rms, dz);

        assert!(
            rms.is_finite() && rms > 0.0,
            "Superheavy holdout HDC RMS={}",
            rms
        );
    }
}
