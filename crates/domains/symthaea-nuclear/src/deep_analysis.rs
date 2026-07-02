// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deep analysis: extracting novel findings from DZ10+RF predictions.

#[cfg(test)]
mod tests {
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::ml_mass::MlMassPredictor;

    fn predictor() -> MlMassPredictor {
        MlMassPredictor::new()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // A. MODEL ACCURACY AUDIT — where does DZ10+RF excel vs struggle?
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deep_model_accuracy_by_region() {
        let pred = predictor();

        eprintln!(
            "\n{} MODEL ACCURACY BY NUCLEAR REGION {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // Compare against AME2020 measured values
        let ame = ame2020_reference_nuclei();

        let mut light_errors = Vec::new(); // A < 40
        let mut medium_errors = Vec::new(); // 40 <= A < 120
        let mut heavy_errors = Vec::new(); // 120 <= A < 210
        let mut actinide_errors = Vec::new(); // A >= 210
        let mut magic_errors = Vec::new(); // near shell closures
        let mut midshell_errors = Vec::new(); // between shells

        let magic_n = [8, 20, 28, 50, 82, 126];
        let magic_z = [8, 20, 28, 50, 82];

        for entry in &ame {
            if !entry.is_measured {
                continue;
            }
            let z = entry.z as u16;
            let n = entry.n as u16;
            if z < 2 || n < 2 {
                continue;
            }

            let p = pred.predict(z, n);
            let a = z + n;
            let exp_be = entry.binding_energy_mev;
            let err = (p.binding_energy - exp_be).abs();

            let bin = if a < 40 {
                &mut light_errors
            } else if a < 120 {
                &mut medium_errors
            } else if a < 210 {
                &mut heavy_errors
            } else {
                &mut actinide_errors
            };
            bin.push(err);

            // Check if near magic number
            let near_magic = magic_n
                .iter()
                .any(|&m| (n as i32 - m as i32).unsigned_abs() <= 2)
                || magic_z
                    .iter()
                    .any(|&m| (z as i32 - m as i32).unsigned_abs() <= 2);
            if near_magic {
                magic_errors.push(err);
            } else {
                midshell_errors.push(err);
            }
        }

        fn stats(v: &[f64]) -> (f64, f64, f64) {
            if v.is_empty() {
                return (0.0, 0.0, 0.0);
            }
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let rms = (v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64).sqrt();
            let max = v.iter().cloned().fold(0.0f64, f64::max);
            (mean, rms, max)
        }

        eprintln!(
            "\n{:<20} {:>6} {:>10} {:>10} {:>10}",
            "Region", "Count", "Mean(MeV)", "RMS(MeV)", "Max(MeV)"
        );
        eprintln!("{}", "-".repeat(58));
        for (name, errs) in [
            ("Light (A<40)", &light_errors),
            ("Medium (40-120)", &medium_errors),
            ("Heavy (120-210)", &heavy_errors),
            ("Actinides (A>210)", &actinide_errors),
            ("Near magic", &magic_errors),
            ("Midshell", &midshell_errors),
        ] {
            let (mean, rms, max) = stats(errs);
            eprintln!(
                "{:<20} {:>6} {:>10.3} {:>10.3} {:>10.3}",
                name,
                errs.len(),
                mean,
                rms,
                max
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // B. SHELL EVOLUTION — do new magic numbers appear far from stability?
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deep_shell_evolution() {
        let pred = predictor();

        eprintln!(
            "\n{} SHELL EVOLUTION FAR FROM STABILITY {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // S_2n across multiple Z values to see how N=82 evolves
        eprintln!("\n--- S_2n drop at N=82 shell closure across Z ---");
        eprintln!(
            "{:>4} {:>10} {:>10} {:>10} {:>10}",
            "Z", "S2n(N=80)", "S2n(N=82)", "S2n(N=84)", "Gap(MeV)"
        );
        eprintln!("{}", "-".repeat(48));

        for z in (30..=60).step_by(2) {
            let s2n_80 = pred.predict(z, 80).binding_energy - pred.predict(z, 78).binding_energy;
            let s2n_82 = pred.predict(z, 82).binding_energy - pred.predict(z, 80).binding_energy;
            let s2n_84 = pred.predict(z, 84).binding_energy - pred.predict(z, 82).binding_energy;
            let gap = s2n_82 - s2n_84;
            eprintln!(
                "{:>4} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                z, s2n_80, s2n_82, s2n_84, gap
            );
        }

        // S_2n across N=50 shell (relevant for r-process)
        eprintln!("\n--- S_2n drop at N=50 shell closure across Z ---");
        eprintln!(
            "{:>4} {:>10} {:>10} {:>10} {:>10}",
            "Z", "S2n(N=48)", "S2n(N=50)", "S2n(N=52)", "Gap(MeV)"
        );
        eprintln!("{}", "-".repeat(48));

        for z in (20..=42).step_by(2) {
            let s2n_48 = pred.predict(z, 48).binding_energy - pred.predict(z, 46).binding_energy;
            let s2n_50 = pred.predict(z, 50).binding_energy - pred.predict(z, 48).binding_energy;
            let s2n_52 = pred.predict(z, 52).binding_energy - pred.predict(z, 50).binding_energy;
            let gap = s2n_50 - s2n_52;
            eprintln!(
                "{:>4} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                z, s2n_48, s2n_50, s2n_52, gap
            );
        }

        // Check for possible N=40 subshell closure
        eprintln!("\n--- Possible N=40 subshell (important for Ni-68, Ca-60) ---");
        eprintln!(
            "{:>4} {:>10} {:>10} {:>10} {:>10}",
            "Z", "S2n(N=38)", "S2n(N=40)", "S2n(N=42)", "Gap(MeV)"
        );
        eprintln!("{}", "-".repeat(48));
        for z in (20..=32).step_by(2) {
            let s2n_38 = pred.predict(z, 38).binding_energy - pred.predict(z, 36).binding_energy;
            let s2n_40 = pred.predict(z, 40).binding_energy - pred.predict(z, 38).binding_energy;
            let s2n_42 = pred.predict(z, 42).binding_energy - pred.predict(z, 40).binding_energy;
            let gap = s2n_40 - s2n_42;
            eprintln!(
                "{:>4} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                z, s2n_38, s2n_40, s2n_42, gap
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // C. ALPHA DECAY LANDSCAPE — full Q_alpha surface for superheavies
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deep_alpha_landscape() {
        let pred = predictor();

        eprintln!(
            "\n{} SUPERHEAVY ALPHA DECAY LANDSCAPE {}",
            "=".repeat(20),
            "=".repeat(20)
        );
        eprintln!("Q_alpha (MeV) for Z=104-120, N=150-190");
        eprintln!("Lower Q = longer-lived. Island of stability = Q minimum.\n");

        // Header
        eprint!("{:>6}", "Z\\N");
        for n in (150..=190).step_by(4) {
            eprint!("{:>7}", n);
        }
        eprintln!();
        eprintln!("{}", "-".repeat(6 + 11 * 7));

        let mut min_q = f64::MAX;
        let mut min_z = 0u16;
        let mut min_n = 0u16;

        for z in (104..=120).step_by(2) {
            eprint!("{:>6}", z);
            for n in (150..=190).step_by(4) {
                if z > 2 && n > 2 {
                    let parent = pred.predict(z, n);
                    let daughter = pred.predict(z - 2, n - 2);
                    let q = daughter.binding_energy + 28.296 - parent.binding_energy;
                    eprint!("{:>7.2}", q);

                    if q > 0.0 && q < min_q {
                        min_q = q;
                        min_z = z;
                        min_n = n;
                    }
                } else {
                    eprint!("{:>7}", "-");
                }
            }
            eprintln!();
        }

        eprintln!(
            "\nMost alpha-stable superheavy: Z={} N={} A={} Q_α={:.3} MeV",
            min_z,
            min_n,
            min_z + min_n,
            min_q
        );

        // Geiger-Nuttall half-life estimate for the most stable
        // log10(t½/s) ≈ -36.6 + 1.61 * Z / sqrt(Q_α)
        let log_t = -36.6 + 1.61 * (min_z as f64) / min_q.sqrt();
        let t_sec = 10.0_f64.powf(log_t);
        eprintln!(
            "Estimated half-life: {:.2e} s ({:.2e} yr)",
            t_sec,
            t_sec / 3.156e7
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D. NEUTRON-RICH TERRA INCOGNITA — where we predict unmeasured nuclei
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deep_terra_incognita() {
        let pred = predictor();
        let ame = ame2020_reference_nuclei();

        eprintln!(
            "\n{} TERRA INCOGNITA: UNMEASURED PREDICTIONS {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // Build set of measured nuclei
        use std::collections::HashSet;
        let measured: HashSet<(u16, u16)> = ame
            .iter()
            .filter(|e| e.is_measured)
            .map(|e| (e.z as u16, e.n as u16))
            .collect();

        // Find nuclei with low uncertainty that are NOT in AME2020
        let mut unmeasured_predictions = Vec::new();

        for z in 20..=100u16 {
            for n in 20..=170u16 {
                if measured.contains(&(z, n)) {
                    continue;
                }

                let p = pred.predict(z, n);
                if p.uncertainty < 0.5 {
                    let a = z + n;
                    let ba = p.binding_energy / a as f64;

                    // Is it bound? (S_n > 0)
                    let sn = if n > 1 {
                        p.binding_energy - pred.predict(z, n - 1).binding_energy
                    } else {
                        0.0
                    };

                    if sn > 0.0 && ba > 6.0 {
                        unmeasured_predictions.push((
                            z,
                            n,
                            a,
                            p.binding_energy,
                            p.uncertainty,
                            sn,
                            ba,
                        ));
                    }
                }
            }
        }

        unmeasured_predictions.sort_by(|a, b| a.4.total_cmp(&b.4));

        eprintln!("\n--- Most Confident Unmeasured Predictions (σ < 0.5 MeV, bound) ---");
        eprintln!(
            "{:>4} {:>4} {:>4} {:>10} {:>8} {:>8} {:>8}",
            "Z", "N", "A", "BE(MeV)", "σ(MeV)", "S_n(MeV)", "B/A"
        );
        eprintln!("{}", "-".repeat(52));

        let count = unmeasured_predictions.len().min(30);
        for &(z, n, a, be, sigma, sn, ba) in unmeasured_predictions.iter().take(count) {
            eprintln!(
                "{:>4} {:>4} {:>4} {:>10.3} {:>8.3} {:>8.3} {:>8.4}",
                z, n, a, be, sigma, sn, ba
            );
        }

        eprintln!(
            "\nTotal unmeasured predictions with σ < 0.5 MeV: {}",
            unmeasured_predictions.len()
        );

        // How many are in r-process path? (Z < 50, N > 60)
        let rprocess_relevant: Vec<_> = unmeasured_predictions
            .iter()
            .filter(|&&(z, n, _, _, _, _, _)| z < 50 && n > 60)
            .collect();
        eprintln!(
            "Of which, r-process relevant (Z<50, N>60): {}",
            rprocess_relevant.len()
        );

        if !rprocess_relevant.is_empty() {
            eprintln!("\n--- R-process relevant unmeasured (top 15) ---");
            for &&(z, n, a, be, sigma, sn, _ba) in rprocess_relevant.iter().take(15) {
                eprintln!(
                    "  Z={:>3} N={:>3} A={:>3}: BE={:.1} ± {:.2} MeV, S_n={:.2} MeV",
                    z, n, a, be, sigma, sn
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // E. FISSION BARRIER LANDSCAPE — where can superheavies survive?
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deep_fission_barriers() {
        let pred = predictor();

        eprintln!(
            "\n{} FISSION BARRIER LANDSCAPE {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // Fission barrier ≈ shell correction + liquid drop barrier
        // Bf ≈ 0.38 * E_s (surface energy correction from shell model)
        // Simple estimate: Bf = Bf_LD × (1 + δE_shell / E_surface)

        eprintln!("\n--- Fissility Parameter & Estimated Barriers ---");
        eprintln!(
            "{:>4} {:>4} {:>4} {:>10} {:>10} {:>10} {:>12}",
            "Z", "N", "A", "Z²/A", "B/A(MeV)", "Q_f(MeV)", "Survivable?"
        );
        eprintln!("{}", "-".repeat(60));

        for z in (104..=120).step_by(2) {
            for n in [170, 174, 178, 182, 184, 186, 190] {
                let a = z + n;
                let p = pred.predict(z, n);
                let ba = p.binding_energy / a as f64;
                let z2_a = (z as f64 * z as f64) / a as f64;
                let fissility = z2_a / 50.883; // x = Z²/A / 2a_s (with a_s ≈ 25.44)

                // Q_fission ≈ BE(frag1) + BE(frag2) - BE(parent)
                // Symmetric fission: two fragments of A/2
                let half_a = a / 2;
                let half_z = z / 2;
                let half_n = half_a - half_z;
                let be_frag = pred.predict(half_z, half_n).binding_energy;
                let q_fission = 2.0 * be_frag - p.binding_energy;

                // Barrier estimate: Cohen-Plasil-Swiatecki
                // Bf ≈ 0.38 × E_surface × (1 - x)³ where x = fissility
                let e_surface = 17.23 * (a as f64).powf(2.0 / 3.0);
                let barrier = if fissility < 1.0 {
                    0.38 * e_surface * (1.0 - fissility).powi(3)
                } else {
                    0.0
                };

                let survivable = if barrier > 4.0 {
                    "YES"
                } else if barrier > 1.0 {
                    "marginal"
                } else {
                    "no"
                };

                eprintln!(
                    "{:>4} {:>4} {:>4} {:>10.2} {:>10.4} {:>10.1} {:>12} (Bf≈{:.1})",
                    z, n, a, z2_a, ba, q_fission, survivable, barrier
                );
            }
            eprintln!();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // F. BINDING ENERGY ANOMALIES — where does our model disagree most?
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deep_anomalies() {
        let pred = predictor();
        let ame = ame2020_reference_nuclei();

        eprintln!(
            "\n{} BINDING ENERGY ANOMALIES {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // Find the largest residuals
        let mut residuals: Vec<(u16, u16, f64, f64, f64)> = Vec::new(); // z, n, predicted, exp, error

        for entry in &ame {
            if !entry.is_measured {
                continue;
            }
            let z = entry.z as u16;
            let n = entry.n as u16;
            if z < 2 || n < 2 {
                continue;
            }

            let p = pred.predict(z, n);
            let error = p.binding_energy - entry.binding_energy_mev;
            residuals.push((z, n, p.binding_energy, entry.binding_energy_mev, error));
        }

        // Sort by absolute error descending
        residuals.sort_by(|a, b| b.4.abs().total_cmp(&a.4.abs()));

        eprintln!("\n--- Top 20 Largest Residuals (model - experiment) ---");
        eprintln!(
            "{:>4} {:>4} {:>4} {:>10} {:>10} {:>10} {:>10}",
            "Z", "N", "A", "Pred(MeV)", "Exp(MeV)", "Err(MeV)", "Err/A"
        );
        eprintln!("{}", "-".repeat(56));

        for &(z, n, pred_be, exp_be, err) in residuals.iter().take(20) {
            let a = z + n;
            eprintln!(
                "{:>4} {:>4} {:>4} {:>10.3} {:>10.3} {:>10.3} {:>10.4}",
                z,
                n,
                a,
                pred_be,
                exp_be,
                err,
                err / a as f64
            );
        }

        // Statistics by shell region
        eprintln!("\n--- Residual Statistics by Region ---");

        let near_doubly_magic: Vec<f64> = residuals
            .iter()
            .filter(|&&(z, n, _, _, _)| {
                let dm = [
                    (8, 8),
                    (20, 20),
                    (28, 28),
                    (50, 50),
                    (82, 82),
                    (82, 126),
                    (50, 82),
                ];
                dm.iter().any(|&(mz, mn)| {
                    (z as i32 - mz as i32).unsigned_abs() <= 3
                        && (n as i32 - mn as i32).unsigned_abs() <= 3
                })
            })
            .map(|r| r.4)
            .collect();

        let deformed: Vec<f64> = residuals
            .iter()
            .filter(|&&(z, n, _, _, _)| {
                // Rare earth deformed region: 60 < Z < 72, 88 < N < 104
                (60..=72).contains(&z) && (88..=104).contains(&n)
            })
            .map(|r| r.4)
            .collect();

        let actinide: Vec<f64> = residuals
            .iter()
            .filter(|&&(z, _, _, _, _)| z >= 89)
            .map(|r| r.4)
            .collect();

        for (name, errs) in [
            ("Doubly magic (±3)", &near_doubly_magic),
            ("Rare earth deformed", &deformed),
            ("Actinides (Z≥89)", &actinide),
        ] {
            if errs.is_empty() {
                continue;
            }
            let mean = errs.iter().sum::<f64>() / errs.len() as f64;
            let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt();
            let bias = mean; // systematic over/under-binding
            eprintln!(
                "  {:<25} n={:>4} RMS={:.3} MeV, bias={:+.3} MeV",
                name,
                errs.len(),
                rms,
                bias
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // G. WIGNER ENERGY — the T=0 np pairing in N=Z nuclei
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deep_wigner_energy() {
        let pred = predictor();
        let ame = ame2020_reference_nuclei();

        eprintln!(
            "\n{} WIGNER ENERGY (np PAIRING AT N=Z) {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // The Wigner energy is an extra binding for N=Z nuclei from np pairing.
        // It manifests as a cusp in binding energy vs (N-Z) at N=Z.
        // Measure: BE(N=Z) - interpolation from N≠Z neighbors.

        eprintln!("\n--- Wigner Energy Extraction ---");
        eprintln!(
            "{:>4} {:>4} {:>10} {:>10} {:>10} {:>10}",
            "Z", "A", "BE(N=Z)", "BE(N=Z+2)", "BE(N=Z-2)", "Wigner(MeV)"
        );
        eprintln!("{}", "-".repeat(56));

        for z in (10..=50).step_by(2) {
            let n = z; // N = Z
            let a = z + n;

            let be_nz = pred.predict(z, n).binding_energy;
            let be_np2 = pred.predict(z, n + 2).binding_energy; // N = Z + 2
            let be_nm2 = if n >= 2 {
                pred.predict(z, n - 2).binding_energy
            } else {
                continue;
            };

            // Wigner energy ≈ BE(N=Z) - 0.5 * [BE(N=Z+2) + BE(N=Z-2)]
            let wigner = be_nz - 0.5 * (be_np2 + be_nm2);

            eprintln!(
                "{:>4} {:>4} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                z, a, be_nz, be_np2, be_nm2, wigner
            );
        }
    }
}
