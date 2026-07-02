// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Advanced nuclear physics analyses — Coulomb corrections, cluster radioactivity,
//! model comparison, gross theory beta decay, nuclear statistical equilibrium,
//! and neutron star crust lattice energy.

#[cfg(test)]
mod tests {
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::duflo_zuker::dz_binding_energy;
    use crate::ml_mass::MlMassPredictor;
    use std::collections::HashMap;

    fn predictor() -> MlMassPredictor {
        MlMassPredictor::new()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. GLOBAL COULOMB CORRECTION — fit residual vs Z(Z-1)/A^{1/3}
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn global_coulomb_correction() {
        let pred = predictor();
        let nuclei = ame2020_reference_nuclei();

        eprintln!(
            "\n{} GLOBAL COULOMB CORRECTION {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        let mut sum_rx = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_r2_orig = 0.0_f64;
        let mut count = 0usize;

        let mut residuals: Vec<(u16, u16, f64, f64)> = Vec::new(); // (Z, N, x, residual)

        for nuc in &nuclei {
            if !nuc.is_measured {
                continue;
            }
            let z = nuc.z;
            let n = nuc.n;
            let a = (z + n) as f64;
            if a < 3.0 || z < 2 {
                continue;
            }

            let prediction = pred.predict(z, n);
            let be_pred = prediction.binding_energy;
            let be_exp = nuc.binding_energy_mev;
            let residual = be_exp - be_pred;

            let x = (z as f64) * ((z as f64) - 1.0) / a.powf(1.0 / 3.0);

            sum_rx += residual * x;
            sum_xx += x * x;
            sum_r2_orig += residual * residual;
            count += 1;
            residuals.push((z, n, x, residual));
        }

        let c = sum_rx / sum_xx;
        let rms_orig = (sum_r2_orig / count as f64).sqrt();

        let mut sum_r2_corr = 0.0_f64;
        for &(_, _, x, r) in &residuals {
            let r_corr = r - c * x;
            sum_r2_corr += r_corr * r_corr;
        }
        let rms_corr = (sum_r2_corr / count as f64).sqrt();
        let improvement = (1.0 - rms_corr / rms_orig) * 100.0;

        eprintln!("Nuclei analyzed: {}", count);
        eprintln!("Coulomb correction coefficient c = {:.6} MeV", c);
        eprintln!("Original RMS residual: {:.4} MeV", rms_orig);
        eprintln!("Corrected RMS residual: {:.4} MeV", rms_corr);
        eprintln!("Improvement: {:.2}%", improvement);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. CLUSTER RADIOACTIVITY — heavy-cluster emission Q-values
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn cluster_radioactivity() {
        let nuclei = ame2020_reference_nuclei();
        let measured: HashMap<(u16, u16), f64> = nuclei
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| ((n.z, n.n), n.binding_energy_mev))
            .collect();

        eprintln!(
            "\n{} CLUSTER RADIOACTIVITY {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // Clusters: (name, Z_c, N_c, BE_c)
        let clusters: &[(&str, u16, u16, f64)] = &[
            ("C-14", 6, 8, 105.285),
            ("O-16", 8, 8, 127.619),
            ("Ne-20", 10, 10, 160.645),
            ("Mg-24", 12, 12, 198.257),
            ("Si-28", 14, 14, 236.537),
        ];

        struct Candidate {
            parent_z: u16,
            parent_n: u16,
            cluster_name: &'static str,
            daughter_z: u16,
            daughter_n: u16,
            q_value: f64,
        }

        let mut candidates: Vec<Candidate> = Vec::new();

        for z_parent in 80..=100u16 {
            for n_parent in 120..=160u16 {
                let be_parent = match measured.get(&(z_parent, n_parent)) {
                    Some(&be) => be,
                    None => continue,
                };

                for &(name, z_c, n_c, be_cluster) in clusters {
                    if z_parent <= z_c || n_parent <= n_c {
                        continue;
                    }
                    let z_d = z_parent - z_c;
                    let n_d = n_parent - n_c;

                    let be_daughter = match measured.get(&(z_d, n_d)) {
                        Some(&be) => be,
                        None => continue,
                    };

                    let q = be_daughter + be_cluster - be_parent;
                    if q > 0.0 {
                        candidates.push(Candidate {
                            parent_z: z_parent,
                            parent_n: n_parent,
                            cluster_name: name,
                            daughter_z: z_d,
                            daughter_n: n_d,
                            q_value: q,
                        });
                    }
                }
            }
        }

        candidates.sort_by(|a, b| b.q_value.total_cmp(&a.q_value));

        eprintln!("Found {} candidates with Q > 0\n", candidates.len());
        eprintln!(
            "{:>6} {:>4} {:>4}  {:>8}  {:>6} {:>4} {:>4}  {:>10}",
            "Parent", "Z", "N", "Cluster", "Daught", "Z_d", "N_d", "Q (MeV)"
        );
        eprintln!("{}", "-".repeat(70));

        for c in candidates.iter().take(20) {
            let a_parent = c.parent_z + c.parent_n;
            let a_daughter = c.daughter_z + c.daughter_n;
            eprintln!(
                "A={:>3}  {:>4} {:>4}  {:>8}  A={:>3}  {:>4} {:>4}  {:>10.3}",
                a_parent,
                c.parent_z,
                c.parent_n,
                c.cluster_name,
                a_daughter,
                c.daughter_z,
                c.daughter_n,
                c.q_value
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. MODEL COMPARISON BY REGION — DZ10 vs DZ10+RF across mass regions
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn model_comparison_by_region() {
        let pred = predictor();
        let nuclei = ame2020_reference_nuclei();

        eprintln!(
            "\n{} MODEL COMPARISON BY REGION {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        let magic_numbers: &[u16] = &[2, 8, 20, 28, 50, 82, 126];

        fn is_near_magic(val: u16, magics: &[u16]) -> bool {
            magics
                .iter()
                .any(|&m| (val as i32 - m as i32).unsigned_abs() as u16 <= 3)
        }

        struct Entry {
            z: u16,
            n: u16,
            a: u16,
            dz_err: f64,
            rf_err: f64,
        }

        let mut entries: Vec<Entry> = Vec::new();

        for nuc in &nuclei {
            if !nuc.is_measured {
                continue;
            }
            let z = nuc.z;
            let n = nuc.n;
            let a = z + n;
            if a < 3 || z < 2 {
                continue;
            }

            let be_exp = nuc.binding_energy_mev;
            let be_dz = dz_binding_energy(z, n);
            let prediction = pred.predict(z, n);
            let be_rf = prediction.binding_energy;

            entries.push(Entry {
                z,
                n,
                a,
                dz_err: be_exp - be_dz,
                rf_err: be_exp - be_rf,
            });
        }

        // Group by mass region
        let regions: &[(&str, Box<dyn Fn(u16) -> bool>)] = &[
            ("Light (A<40)", Box::new(|a: u16| a < 40)),
            (
                "Medium (40-120)",
                Box::new(|a: u16| (40..=120).contains(&a)),
            ),
            (
                "Heavy (120-210)",
                Box::new(|a: u16| (120..=210).contains(&a)),
            ),
            ("Actinides (A>210)", Box::new(|a: u16| a > 210)),
        ];

        eprintln!("\n── By mass region ──");
        eprintln!(
            "{:<22} {:>5}  {:>10} {:>10}  {:>10} {:>10}",
            "Region", "Count", "DZ mean", "DZ RMS", "RF mean", "RF RMS"
        );
        eprintln!("{}", "-".repeat(75));

        for &(label, ref filter) in regions {
            let group: Vec<&Entry> = entries.iter().filter(|e| filter(e.a)).collect();
            if group.is_empty() {
                continue;
            }
            let n = group.len() as f64;
            let dz_mean = group.iter().map(|e| e.dz_err).sum::<f64>() / n;
            let rf_mean = group.iter().map(|e| e.rf_err).sum::<f64>() / n;
            let dz_rms = (group.iter().map(|e| e.dz_err * e.dz_err).sum::<f64>() / n).sqrt();
            let rf_rms = (group.iter().map(|e| e.rf_err * e.rf_err).sum::<f64>() / n).sqrt();
            eprintln!(
                "{:<22} {:>5}  {:>10.4} {:>10.4}  {:>10.4} {:>10.4}",
                label,
                group.len(),
                dz_mean,
                dz_rms,
                rf_mean,
                rf_rms
            );
        }

        // Group by shell proximity
        eprintln!("\n── By shell proximity ──");
        eprintln!(
            "{:<22} {:>5}  {:>10} {:>10}  {:>10} {:>10}",
            "Category", "Count", "DZ mean", "DZ RMS", "RF mean", "RF RMS"
        );
        eprintln!("{}", "-".repeat(75));

        for (label, filter_fn) in &[
            (
                "Near-magic",
                Box::new(|e: &&Entry| {
                    is_near_magic(e.z, magic_numbers) || is_near_magic(e.n, magic_numbers)
                }) as Box<dyn Fn(&&Entry) -> bool>,
            ),
            (
                "Mid-shell",
                Box::new(|e: &&Entry| {
                    !is_near_magic(e.z, magic_numbers) && !is_near_magic(e.n, magic_numbers)
                }) as Box<dyn Fn(&&Entry) -> bool>,
            ),
        ] {
            let group: Vec<&Entry> = entries.iter().filter(|e| filter_fn(e)).collect();
            if group.is_empty() {
                continue;
            }
            let n = group.len() as f64;
            let dz_mean = group.iter().map(|e| e.dz_err).sum::<f64>() / n;
            let rf_mean = group.iter().map(|e| e.rf_err).sum::<f64>() / n;
            let dz_rms = (group.iter().map(|e| e.dz_err * e.dz_err).sum::<f64>() / n).sqrt();
            let rf_rms = (group.iter().map(|e| e.rf_err * e.rf_err).sum::<f64>() / n).sqrt();
            eprintln!(
                "{:<22} {:>5}  {:>10.4} {:>10.4}  {:>10.4} {:>10.4}",
                label,
                group.len(),
                dz_mean,
                dz_rms,
                rf_mean,
                rf_rms
            );
        }

        // RF helps most / hurts most
        let mut sorted_by_improvement: Vec<&Entry> = entries.iter().collect();
        sorted_by_improvement.sort_by(|a, b| {
            let improve_a = a.dz_err.abs() - a.rf_err.abs();
            let improve_b = b.dz_err.abs() - b.rf_err.abs();
            improve_b.total_cmp(&improve_a)
        });

        eprintln!("\n── Top 10 where RF helps MOST ──");
        eprintln!(
            "{:>4} {:>4} {:>4}  {:>10} {:>10} {:>10}",
            "Z", "N", "A", "DZ err", "RF err", "Improve"
        );
        for e in sorted_by_improvement.iter().take(10) {
            let improve = e.dz_err.abs() - e.rf_err.abs();
            eprintln!(
                "{:>4} {:>4} {:>4}  {:>10.4} {:>10.4} {:>10.4}",
                e.z, e.n, e.a, e.dz_err, e.rf_err, improve
            );
        }

        eprintln!("\n── Top 10 where RF hurts MOST ──");
        eprintln!(
            "{:>4} {:>4} {:>4}  {:>10} {:>10} {:>10}",
            "Z", "N", "A", "DZ err", "RF err", "Degrade"
        );
        for e in sorted_by_improvement.iter().rev().take(10) {
            let degrade = e.rf_err.abs() - e.dz_err.abs();
            eprintln!(
                "{:>4} {:>4} {:>4}  {:>10.4} {:>10.4} {:>10.4}",
                e.z, e.n, e.a, e.dz_err, e.rf_err, degrade
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. GROSS THEORY BETA DECAY — simplified half-life estimation
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn gross_theory_beta_decay() {
        eprintln!(
            "\n{} GROSS THEORY BETA DECAY {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        let alpha_fs = 1.0 / 137.036; // fine structure constant
        let c_gross = 6000.0; // Sargent constant (seconds)

        // Approximate Fermi function: Coulomb correction
        fn fermi_f(z: f64, _a: f64, alpha: f64) -> f64 {
            1.0 + std::f64::consts::PI * alpha * z
        }

        // Known beta emitters: (name, Z_daughter, A, t_half_exp_seconds, Q_MeV)
        let beta_emitters: &[(&str, f64, f64, f64, f64)] = &[
            ("Co-60", 28.0, 60.0, 5.27 * 365.25 * 86400.0, 2.82),
            ("Sr-90", 39.0, 90.0, 28.8 * 365.25 * 86400.0, 0.546),
            ("Cs-137", 56.0, 137.0, 30.2 * 365.25 * 86400.0, 1.176),
            ("I-131", 54.0, 131.0, 8.02 * 86400.0, 0.971),
        ];

        eprintln!(
            "{:<10} {:>12} {:>12} {:>12} {:>8}",
            "Isotope", "t_pred (s)", "t_exp (s)", "Ratio", "Q (MeV)"
        );
        eprintln!("{}", "-".repeat(60));

        for &(name, z_d, a, t_exp, q) in beta_emitters {
            let f = fermi_f(z_d, a, alpha_fs);
            let t_pred = c_gross / (q.powi(5) * f);
            let ratio = t_pred / t_exp;
            eprintln!(
                "{:<10} {:>12.3e} {:>12.3e} {:>12.4} {:>8.3}",
                name, t_pred, t_exp, ratio, q
            );
        }

        eprintln!("\nNote: Gross theory gives order-of-magnitude estimates.");
        eprintln!("Ratios far from 1.0 reflect nuclear matrix element variations.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. NUCLEAR STATISTICAL EQUILIBRIUM — abundance at high temperatures
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn nuclear_statistical_equilibrium() {
        let nuclei = ame2020_reference_nuclei();
        let measured: HashMap<(u16, u16), f64> = nuclei
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| ((n.z, n.n), n.binding_energy_mev))
            .collect();

        eprintln!(
            "\n{} NUCLEAR STATISTICAL EQUILIBRIUM {}",
            "=".repeat(18),
            "=".repeat(18)
        );

        let temperatures_gk = [7.0, 8.0, 9.0, 10.0];

        for t_gk in &temperatures_gk {
            let kt = 0.0862 * t_gk; // MeV

            let mut abundances: Vec<(u16, u16, f64)> = Vec::new();

            for z in 1..=30u16 {
                for n in 1..=30u16 {
                    let be = match measured.get(&(z, n)) {
                        Some(&be) => be,
                        None => continue,
                    };
                    let a = (z + n) as f64;
                    // Y(Z,N) ∝ A^{3/2} * exp(BE / kT)
                    // Use log to avoid overflow, then exponentiate relative to max
                    let log_y = 1.5 * a.ln() + be / kt;
                    abundances.push((z, n, log_y));
                }
            }

            if abundances.is_empty() {
                continue;
            }

            // Normalize via log-sum-exp
            let max_log = abundances
                .iter()
                .map(|&(_, _, ly)| ly)
                .fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = abundances
                .iter()
                .map(|&(_, _, ly)| (ly - max_log).exp())
                .sum();
            let log_norm = max_log + sum_exp.ln();

            let mut normed: Vec<(u16, u16, f64)> = abundances
                .iter()
                .map(|&(z, n, ly)| (z, n, (ly - log_norm).exp()))
                .collect();
            normed.sort_by(|a, b| b.2.total_cmp(&a.2));

            eprintln!("\n── T = {} GK (kT = {:.3} MeV) ──", t_gk, kt);
            eprintln!("{:>4} {:>4} {:>4}  {:>12}", "Z", "N", "A", "Y (fraction)");
            for &(z, n, y) in normed.iter().take(5) {
                eprintln!("{:>4} {:>4} {:>4}  {:>12.6e}", z, n, z + n, y);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. NEUTRON STAR CRUST LATTICE — equilibrium nuclei at given densities
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn neutron_star_crust_lattice() {
        let nuclei = ame2020_reference_nuclei();
        let measured: HashMap<(u16, u16), f64> = nuclei
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| ((n.z, n.n), n.binding_energy_mev))
            .collect();

        eprintln!(
            "\n{} NEUTRON STAR CRUST LATTICE {}",
            "=".repeat(18),
            "=".repeat(18)
        );

        let e2 = 1.44_f64; // MeV·fm (e^2 in natural units)
        let pi = std::f64::consts::PI;

        let densities = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]; // fm^-3

        eprintln!(
            "{:>10} {:>4} {:>4} {:>4}  {:>12} {:>12} {:>12}",
            "n_b", "Z", "N", "A", "-BE/A", "E_lat/A", "E_tot/A"
        );
        eprintln!("{}", "-".repeat(75));

        for &n_b in &densities {
            let mut best: Option<(u16, u16, f64, f64, f64)> = None; // (Z, N, -BE/A, E_lat/A, E_tot/A)

            for z in 20..=50u16 {
                let n_max = 3 * z;
                for n in z..=n_max {
                    let be = match measured.get(&(z, n)) {
                        Some(&be) => be,
                        None => continue,
                    };
                    let a = (z + n) as f64;

                    // Wigner-Seitz cell radius
                    let r_ws = (3.0 * a / (4.0 * pi * n_b)).powf(1.0 / 3.0);

                    // Lattice energy per nucleon (Madelung)
                    let e_lattice_per_a = -1.82 * (z as f64 * e2).powi(2) / (r_ws * a);

                    let e_total_per_a = -be / a + e_lattice_per_a;

                    let is_better = match best {
                        None => true,
                        Some((_, _, _, _, best_e)) => e_total_per_a < best_e,
                    };
                    if is_better {
                        best = Some((z, n, -be / a, e_lattice_per_a, e_total_per_a));
                    }
                }
            }

            if let Some((z, n, neg_be_a, e_lat_a, e_tot_a)) = best {
                eprintln!(
                    "{:>10.1e} {:>4} {:>4} {:>4}  {:>12.4} {:>12.4} {:>12.4}",
                    n_b,
                    z,
                    n,
                    z + n,
                    neg_be_a,
                    e_lat_a,
                    e_tot_a
                );
            }
        }

        eprintln!("\nNote: At low density, symmetric nuclei dominate.");
        eprintln!("At higher density, more neutron-rich nuclei are favored.");
    }
}
