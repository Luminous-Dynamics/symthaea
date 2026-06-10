// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Figure data generators — write TSV files for gnuplot / matplotlib plotting.

#[cfg(test)]
mod tests {
    use crate::ml_mass::MlMassPredictor;
    use std::fs;
    use std::io::Write;

    const OUT_DIR: &str = "/tmp/nuclear_figures";

    /// Q_alpha heatmap for superheavy region Z=104-120, N=150-190.
    ///
    /// Q_alpha(Z,N) = BE(Z-2, N-2) + BE(4He) - BE(Z,N)
    /// where BE(4He) = 28.296 MeV.
    #[test]
    fn superheavy_qalpha_heatmap() {
        fs::create_dir_all(OUT_DIR).unwrap();
        let predictor = MlMassPredictor::new();
        let path = format!("{OUT_DIR}/superheavy_qalpha_heatmap.tsv");
        let mut f = fs::File::create(&path).unwrap();

        writeln!(f, "Z\tN\tA\tQ_alpha\tuncertainty").unwrap();

        for z in (104u16..=120).step_by(2) {
            for n in (150u16..=190).step_by(2) {
                let a = z + n;
                let parent = predictor.predict(z, n);
                let daughter = predictor.predict(z - 2, n - 2);
                let q_alpha = daughter.binding_energy + 28.296 - parent.binding_energy;
                let unc = (parent.uncertainty.powi(2) + daughter.uncertainty.powi(2)).sqrt();
                writeln!(f, "{z}\t{n}\t{a}\t{q_alpha:.4}\t{unc:.4}").unwrap();
            }
        }

        eprintln!("Wrote {path}");
    }

    /// Shell evolution around N=82 for Z=30-60.
    /// S_2n(Z,N) = BE(Z,N) - BE(Z, N-2).
    #[test]
    fn shell_evolution_n82() {
        fs::create_dir_all(OUT_DIR).unwrap();
        let predictor = MlMassPredictor::new();
        let path = format!("{OUT_DIR}/shell_evolution_n82.tsv");
        let mut f = fs::File::create(&path).unwrap();

        writeln!(f, "Z\tS2n_N80\tS2n_N82\tS2n_N84\tgap").unwrap();

        for z in (30u16..=60).step_by(2) {
            let s2n = |n: u16| -> f64 {
                let be_n = predictor.predict(z, n).binding_energy;
                let be_nm2 = predictor.predict(z, n - 2).binding_energy;
                be_n - be_nm2
            };
            let s2n_80 = s2n(80);
            let s2n_82 = s2n(82);
            let s2n_84 = s2n(84);
            let gap = s2n_82 - s2n_84;
            writeln!(f, "{z}\t{s2n_80:.4}\t{s2n_82:.4}\t{s2n_84:.4}\t{gap:.4}").unwrap();
        }

        eprintln!("Wrote {path}");
    }

    /// Shell evolution around N=50 for Z=20-42.
    #[test]
    fn shell_evolution_n50() {
        fs::create_dir_all(OUT_DIR).unwrap();
        let predictor = MlMassPredictor::new();
        let path = format!("{OUT_DIR}/shell_evolution_n50.tsv");
        let mut f = fs::File::create(&path).unwrap();

        writeln!(f, "Z\tS2n_N48\tS2n_N50\tS2n_N52\tgap").unwrap();

        for z in (20u16..=42).step_by(2) {
            let s2n = |n: u16| -> f64 {
                let be_n = predictor.predict(z, n).binding_energy;
                let be_nm2 = predictor.predict(z, n - 2).binding_energy;
                be_n - be_nm2
            };
            let s2n_48 = s2n(48);
            let s2n_50 = s2n(50);
            let s2n_52 = s2n(52);
            let gap = s2n_50 - s2n_52;
            writeln!(f, "{z}\t{s2n_48:.4}\t{s2n_50:.4}\t{s2n_52:.4}\t{gap:.4}").unwrap();
        }

        eprintln!("Wrote {path}");
    }

    /// Validation split summary (hardcoded from existing results).
    #[test]
    fn validation_splits() {
        fs::create_dir_all(OUT_DIR).unwrap();
        let path = format!("{OUT_DIR}/validation_splits.tsv");
        let mut f = fs::File::create(&path).unwrap();

        writeln!(f, "split_name\tn_test\trf_rms\tdz_rms").unwrap();
        writeln!(f, "standard_cv\t2530\t0.665\t0.668").unwrap();
        writeln!(f, "chain_holdout\t505\t0.712\t0.717").unwrap();
        writeln!(f, "proton_holdout\t215\t0.751\t0.730").unwrap();
        writeln!(f, "frontier\t1260\t0.677\t0.667").unwrap();
        writeln!(f, "superheavy\t47\t0.760\t0.755").unwrap();

        eprintln!("Wrote {path}");
    }

    /// Nuclear chart: full (Z,N) grid with BE, uncertainty, measured flag, S_n.
    ///
    /// Produces a TSV for the segre chart showing measured vs predicted nuclei
    /// with uncertainty coloring and magic number annotations.
    #[test]
    fn nuclear_chart_data() {
        use crate::ame2020::ame2020_reference_nuclei;
        use std::collections::HashSet;

        fs::create_dir_all(OUT_DIR).unwrap();
        let predictor = MlMassPredictor::new();
        let path = format!("{OUT_DIR}/nuclear_chart.tsv");
        let mut f = fs::File::create(&path).unwrap();

        // Build set of measured (Z,N) pairs
        let measured: HashSet<(u16, u16)> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|nuc| nuc.is_measured)
            .map(|nuc| (nuc.z, nuc.n))
            .collect();

        writeln!(f, "Z\tN\tBE\tuncertainty\tis_measured\tS_n").unwrap();

        let mut count = 0u32;
        for z in 2u16..=110 {
            for n in 2u16..=180 {
                let pred = predictor.predict(z, n);
                let a = (z + n) as f64;
                let ba = pred.binding_energy / a;

                // Filter: plausibly bound nuclei only
                if pred.binding_energy <= 0.0 || ba <= 4.0 {
                    continue;
                }

                let is_meas = if measured.contains(&(z, n)) { 1 } else { 0 };

                // Neutron separation energy: S_n = BE(Z,N) - BE(Z,N-1)
                let sn = if n > 2 {
                    let prev = predictor.predict(z, n - 1);
                    pred.binding_energy - prev.binding_energy
                } else {
                    0.0
                };

                writeln!(
                    f,
                    "{z}\t{n}\t{:.4}\t{:.4}\t{is_meas}\t{sn:.4}",
                    pred.binding_energy, pred.uncertainty
                )
                .unwrap();
                count += 1;
            }
        }

        eprintln!("Wrote {path} ({count} nuclei)");
    }

    /// Wigner energy for even-Z nuclei Z=10-50.
    ///
    /// W(Z) = BE(Z, N=Z) - 0.5 * [BE(Z, N=Z+2) + BE(Z, N=Z-2)]
    #[test]
    fn wigner_energy() {
        fs::create_dir_all(OUT_DIR).unwrap();
        let predictor = MlMassPredictor::new();
        let path = format!("{OUT_DIR}/wigner_energy.tsv");
        let mut f = fs::File::create(&path).unwrap();

        writeln!(f, "Z\tA\twigner_mev").unwrap();

        for z in (10u16..=50).step_by(2) {
            let n = z; // N = Z line
            let a = z + n;
            let be_nz = predictor.predict(z, n).binding_energy;
            let be_np2 = predictor.predict(z, n + 2).binding_energy;
            let be_nm2 = predictor.predict(z, n - 2).binding_energy;
            let wigner = be_nz - 0.5 * (be_np2 + be_nm2);
            writeln!(f, "{z}\t{a}\t{wigner:.4}").unwrap();
        }

        eprintln!("Wrote {path}");
    }
}
