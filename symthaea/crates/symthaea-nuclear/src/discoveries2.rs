// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Extended nuclear structure analyses — 7 physics studies connecting
//! nuclear masses to FRIB predictions, isomers, drip lines, deformation,
//! Coulomb displacement, symmetry energy, and neutron star EOS.

#[cfg(test)]
mod tests {
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::deformation::frdm_deformation;
    use crate::duflo_zuker::dz_binding_energy;
    use crate::ml_mass::MlMassPredictor;
    use std::collections::HashSet;

    fn predictor() -> MlMassPredictor {
        MlMassPredictor::new()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. FRIB BLIND PREDICTIONS — neutron-rich nuclei near N=50, N=82
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn frib_predictions() {
        let pred = predictor();
        let nuclei = ame2020_reference_nuclei();
        let measured: HashSet<(u16, u16)> = nuclei
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| (n.z, n.n))
            .collect();

        eprintln!(
            "\n{} FRIB BLIND PREDICTIONS {}",
            "=".repeat(20),
            "=".repeat(20)
        );
        eprintln!("Neutron-rich nuclei FRIB is likely measuring (2023-2025).\n");

        // (label, Z, N_range)
        let chains: &[(&str, u16, std::ops::RangeInclusive<u16>)] = &[
            ("Ca", 20, 32..=40),
            ("Ni", 28, 40..=52),
            ("Sn", 50, 82..=90),
            ("Se", 34, 48..=56),
        ];

        for &(label, z, ref n_range) in chains {
            eprintln!("── {} isotopes (Z={}) ──", label, z);
            eprintln!(
                "{:>4} {:>4} {:>4}  {:>10} {:>8} {:>8} {:>8}  {}",
                "Z", "N", "A", "BE(MeV)", "B/A", "S_n", "σ(MeV)", "AME2020?"
            );
            let mut prev_be: Option<f64> = None;
            for n in n_range.clone() {
                let a = z as u32 + n as u32;
                let p = pred.predict(z, n);
                let in_ame = if measured.contains(&(z, n)) {
                    "YES"
                } else {
                    "NO"
                };
                let s_n = match prev_be {
                    Some(be_prev) => format!("{:8.3}", p.binding_energy - be_prev),
                    None => "     ---".to_string(),
                };
                eprintln!(
                    "{:4} {:4} {:4}  {:10.3} {:8.4} {} {:8.3}  {}",
                    z, n, a, p.binding_energy, p.ba, s_n, p.uncertainty, in_ame
                );
                prev_be = Some(p.binding_energy);
            }
            eprintln!();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. ISOMERIC CANDIDATES — large shell gaps in rare-earth region
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn isomeric_candidates() {
        let pred = predictor();

        eprintln!(
            "\n{} ISOMERIC CANDIDATES (Z=70-80, N=100-120) {}",
            "=".repeat(12),
            "=".repeat(12)
        );
        eprintln!("Looking for shell gaps > 2.5 MeV in S_n or S_p.\n");

        let mut candidates: Vec<(u16, u16, f64, &str)> = Vec::new();

        for z in 70..=80 {
            for n in 100..=120 {
                let be = pred.predict(z, n).binding_energy;

                // Neutron separation energy gap: |S_n(Z,N) - S_n(Z,N+1)|
                let s_n_here = be - pred.predict(z, n - 1).binding_energy;
                let s_n_next = pred.predict(z, n + 1).binding_energy - be;
                let gap_n = (s_n_here - s_n_next).abs();
                if gap_n > 2.5 {
                    candidates.push((z, n, gap_n, "neutron"));
                }

                // Proton separation energy gap
                let s_p_here = be - pred.predict(z - 1, n).binding_energy;
                let s_p_next = pred.predict(z + 1, n).binding_energy - be;
                let gap_p = (s_p_here - s_p_next).abs();
                if gap_p > 2.5 {
                    candidates.push((z, n, gap_p, "proton"));
                }
            }
        }

        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        eprintln!(
            "{:>4} {:>4} {:>4}  {:>10}  {}",
            "Z", "N", "A", "Gap(MeV)", "Type"
        );
        for &(z, n, gap, kind) in candidates.iter().take(30) {
            eprintln!(
                "{:4} {:4} {:4}  {:10.3}  {}",
                z,
                n,
                z as u32 + n as u32,
                gap,
                kind
            );
        }
        eprintln!(
            "\nTotal candidates with gap > 2.5 MeV: {}",
            candidates.len()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. BINDING ENERGY GRADIENT SURFACE — S_n across the chart
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn binding_energy_gradient_surface() {
        let pred = predictor();

        eprintln!(
            "\n{} NEUTRON SEPARATION ENERGY SURFACE {}",
            "=".repeat(15),
            "=".repeat(15)
        );
        eprintln!("S_n = BE(Z,N) - BE(Z,N-1) across Z=20-82, N=20-130 (step 2).\n");

        let mut all_sn: Vec<(u16, u16, f64)> = Vec::new();

        for z in (20..=82).step_by(2) {
            for n in (20..=130).step_by(2) {
                let be = pred.predict(z, n).binding_energy;
                let be_prev = pred.predict(z, n - 1).binding_energy;
                let s_n = be - be_prev;
                all_sn.push((z, n, s_n));
            }
        }

        all_sn.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        eprintln!("── Top 20 HIGHEST S_n (most bound neutrons) ──");
        eprintln!("{:>4} {:>4} {:>4}  {:>10}", "Z", "N", "A", "S_n(MeV)");
        for &(z, n, sn) in all_sn.iter().take(20) {
            eprintln!("{:4} {:4} {:4}  {:10.3}", z, n, z as u32 + n as u32, sn);
        }

        eprintln!("\n── Top 20 LOWEST S_n (approaching drip line) ──");
        eprintln!("{:>4} {:>4} {:>4}  {:>10}", "Z", "N", "A", "S_n(MeV)");
        for &(z, n, sn) in all_sn.iter().rev().take(20) {
            eprintln!("{:4} {:4} {:4}  {:10.3}", z, n, z as u32 + n as u32, sn);
        }

        // Count drip-line nuclei (S_n < 2 MeV)
        let drip = all_sn.iter().filter(|x| x.2 < 2.0).count();
        eprintln!(
            "\nNuclei with S_n < 2 MeV (near drip line): {} / {}",
            drip,
            all_sn.len()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. DEFORMATION LANDSCAPE — DZ residual vs FRDM β₂
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deformation_landscape_comparison() {
        let nuclei = ame2020_reference_nuclei();

        eprintln!(
            "\n{} DEFORMATION LANDSCAPE: DZ RESIDUAL vs FRDM β₂ {}",
            "=".repeat(10),
            "=".repeat(10)
        );
        eprintln!("Grouping measured AME2020 nuclei by |β₂| deformation.\n");

        struct Group {
            label: &'static str,
            residuals: Vec<f64>,
        }

        let mut spherical = Group {
            label: "Spherical (|β₂|<0.05)",
            residuals: Vec::new(),
        };
        let mut transitional = Group {
            label: "Transitional (0.05-0.15)",
            residuals: Vec::new(),
        };
        let mut deformed = Group {
            label: "Deformed (0.15-0.30)",
            residuals: Vec::new(),
        };
        let mut superdeformed = Group {
            label: "Superdeformed (>0.30)",
            residuals: Vec::new(),
        };

        for nuc in &nuclei {
            if !nuc.is_measured {
                continue;
            }
            if nuc.z < 8 || nuc.n < 8 {
                continue;
            } // skip very light

            let dz_be = dz_binding_energy(nuc.z, nuc.n);
            let residual = nuc.binding_energy_mev - dz_be;
            let (beta2, _) = frdm_deformation(nuc.z, nuc.n);
            let abs_b2 = beta2.abs();

            let group = if abs_b2 < 0.05 {
                &mut spherical
            } else if abs_b2 < 0.15 {
                &mut transitional
            } else if abs_b2 < 0.30 {
                &mut deformed
            } else {
                &mut superdeformed
            };
            group.residuals.push(residual);
        }

        eprintln!(
            "{:<30} {:>6} {:>12} {:>12}",
            "Group", "Count", "Mean(MeV)", "RMS(MeV)"
        );
        for g in [&spherical, &transitional, &deformed, &superdeformed] {
            let count = g.residuals.len();
            if count == 0 {
                eprintln!("{:<30} {:>6}       ---          ---", g.label, 0);
                continue;
            }
            let mean = g.residuals.iter().sum::<f64>() / count as f64;
            let rms = (g.residuals.iter().map(|r| r * r).sum::<f64>() / count as f64).sqrt();
            eprintln!("{:<30} {:>6} {:>12.4} {:>12.4}", g.label, count, mean, rms);
        }

        eprintln!("\nNote: FRDM table covers superheavy region primarily;");
        eprintln!("lighter nuclei get interpolated β₂ from nearest entry.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. MIRROR NUCLEI COULOMB DISPLACEMENT ENERGIES
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn mirror_nuclei_coulomb() {
        let pred = predictor();

        eprintln!(
            "\n{} MIRROR NUCLEI: COULOMB DISPLACEMENT ENERGIES {}",
            "=".repeat(10),
            "=".repeat(10)
        );
        eprintln!("ΔE_C = BE(Z_low,N_high) - BE(Z_high,N_low) for mirror pairs.");
        eprintln!("Nolen-Schiffer: ΔE_C ≈ 0.7 × (2Z_low-1) / A^(1/3) MeV.\n");

        // Mirror pairs: (A, Z_low, N_low, label)
        // Z_high = N_low, N_high = Z_low
        let pairs: &[(u32, u16, u16, &str)] = &[
            (11, 5, 6, "¹¹B / ¹¹C"),
            (13, 6, 7, "¹³C / ¹³N"),
            (15, 7, 8, "¹⁵N / ¹⁵O"),
            (17, 8, 9, "¹⁷O / ¹⁷F"),
            (27, 13, 14, "²⁷Al / ²⁷Si"),
            (39, 19, 20, "³⁹K / ³⁹Ca"),
            (41, 20, 21, "⁴¹Ca / ⁴¹Sc"),
        ];

        eprintln!(
            "{:<16} {:>4}  {:>10} {:>10} {:>10} {:>10}",
            "Pair", "A", "ΔE_pred", "ΔE_NS", "Diff", "% err"
        );

        for &(a, z_low, n_low, label) in pairs {
            let z_high = n_low; // swap
            let n_high = z_low;

            let be_high = pred.predict(z_high, n_high).binding_energy;
            let be_low = pred.predict(z_low, n_low).binding_energy;
            let delta_e_pred = be_low - be_high;

            // Nolen-Schiffer formula
            let delta_e_ns = 0.7 * (2.0 * z_low as f64 - 1.0) / (a as f64).powf(1.0 / 3.0);

            let diff = delta_e_pred - delta_e_ns;
            let pct = if delta_e_ns.abs() > 1e-6 {
                100.0 * diff / delta_e_ns
            } else {
                0.0
            };

            eprintln!(
                "{:<16} {:>4}  {:10.3} {:10.3} {:10.3} {:10.1}%",
                label, a, delta_e_pred, delta_e_ns, diff, pct
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. SYMMETRY ENERGY FROM ISOBARIC MASS PARABOLAS
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn symmetry_energy_from_isobaric_chains() {
        let nuclei = ame2020_reference_nuclei();
        let measured: HashSet<(u16, u16)> = nuclei
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| (n.z, n.n))
            .collect();
        let be_map: std::collections::HashMap<(u16, u16), f64> = nuclei
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| ((n.z, n.n), n.binding_energy_mev))
            .collect();

        eprintln!(
            "\n{} SYMMETRY ENERGY FROM ISOBARIC CHAINS {}",
            "=".repeat(12),
            "=".repeat(12)
        );
        eprintln!("Fit BE vs (N-Z)²/A for even A from 40-200 (step 10).");
        eprintln!("Liquid drop: BE ≈ const - a_sym × (N-Z)²/A\n");

        eprintln!(
            "{:>6} {:>8} {:>10} {:>8}",
            "A", "Isobars", "a_sym(MeV)", "A^(-1/3)"
        );

        let mut a_sym_values: Vec<(u32, f64)> = Vec::new();

        for a in (40..=200).step_by(10) {
            // Collect all measured isobars for this A
            let mut points: Vec<(f64, f64)> = Vec::new(); // (x=(N-Z)^2/A, y=BE)
            for z in 1..a {
                let n = a - z;
                if n < 1 {
                    continue;
                }
                let zz = z as u16;
                let nn = n as u16;
                if !measured.contains(&(zz, nn)) {
                    continue;
                }
                let be = be_map[&(zz, nn)];
                // Subtract Coulomb contribution before fitting symmetry term
                let a_c = 0.7; // Coulomb coefficient (MeV)
                let z_f = z as f64;
                let a_f = a as f64;
                let coulomb = a_c * z_f * (z_f - 1.0) / a_f.powf(1.0 / 3.0);
                let be_corrected = be + coulomb;
                let x = ((n as f64 - z as f64).powi(2)) / a as f64;
                points.push((x, be_corrected));
            }
            if points.len() < 3 {
                continue;
            }

            // Linear regression: BE = a0 - a_sym * x
            // => y = a0 + b*x where b = -a_sym
            let n_pts = points.len() as f64;
            let sum_x: f64 = points.iter().map(|p| p.0).sum();
            let sum_y: f64 = points.iter().map(|p| p.1).sum();
            let sum_xy: f64 = points.iter().map(|p| p.0 * p.1).sum();
            let sum_xx: f64 = points.iter().map(|p| p.0 * p.0).sum();
            let denom = n_pts * sum_xx - sum_x * sum_x;
            if denom.abs() < 1e-10 {
                continue;
            }

            let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
            let a_sym = -slope; // negative slope = positive symmetry energy

            let a_inv_third = 1.0 / (a as f64).powf(1.0 / 3.0);

            eprintln!(
                "{:6} {:8} {:10.3} {:8.4}",
                a,
                points.len(),
                a_sym,
                a_inv_third
            );
            a_sym_values.push((a, a_sym));
        }

        // Summary statistics
        let valid: Vec<f64> = a_sym_values
            .iter()
            .filter(|(_, v)| *v > 0.0 && *v < 100.0)
            .map(|(_, v)| *v)
            .collect();
        if !valid.is_empty() {
            let mean = valid.iter().sum::<f64>() / valid.len() as f64;
            let std =
                (valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len() as f64).sqrt();
            eprintln!(
                "\nMean a_sym = {:.2} ± {:.2} MeV (accepted ≈ 23 MeV)",
                mean, std
            );
        }

        // Surface symmetry: fit a_sym = J - K_ss * A^(-1/3)
        let fit_pts: Vec<(f64, f64)> = a_sym_values
            .iter()
            .filter(|(_, v)| *v > 0.0 && *v < 100.0)
            .map(|(a, v)| (1.0 / (*a as f64).powf(1.0 / 3.0), *v))
            .collect();
        if fit_pts.len() >= 2 {
            let n_f = fit_pts.len() as f64;
            let sx: f64 = fit_pts.iter().map(|p| p.0).sum();
            let sy: f64 = fit_pts.iter().map(|p| p.1).sum();
            let sxy: f64 = fit_pts.iter().map(|p| p.0 * p.1).sum();
            let sxx: f64 = fit_pts.iter().map(|p| p.0 * p.0).sum();
            let d = n_f * sxx - sx * sx;
            if d.abs() > 1e-10 {
                let slope_surf = (n_f * sxy - sx * sy) / d;
                let intercept_j = (sy - slope_surf * sx) / n_f;
                eprintln!(
                    "Surface fit: a_sym(A) ≈ {:.1} - {:.1} × A^(-1/3)",
                    intercept_j, -slope_surf
                );
                eprintln!(
                    "  → Volume symmetry J ≈ {:.1} MeV, Surface correction K_ss ≈ {:.1} MeV",
                    intercept_j, -slope_surf
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 7. NEUTRON STAR EQUATION OF STATE — symmetry energy & slope L
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn neutron_star_equation_of_state() {
        let nuclei = ame2020_reference_nuclei();
        let measured: HashSet<(u16, u16)> = nuclei
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| (n.z, n.n))
            .collect();
        let be_map: std::collections::HashMap<(u16, u16), f64> = nuclei
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| ((n.z, n.n), n.binding_energy_mev))
            .collect();

        eprintln!(
            "\n{} NEUTRON STAR EOS FROM NUCLEAR MASSES {}",
            "=".repeat(12),
            "=".repeat(12)
        );
        eprintln!("Extracting symmetry energy J and slope parameter L.");
        eprintln!("LIGO/Virgo GW170817 constraint: 40 < L < 120 MeV.\n");

        // Extract a_sym for many A values (finer grid)
        let mut a_sym_data: Vec<(f64, f64)> = Vec::new(); // (A^{-1/3}, a_sym)

        for a in (30u32..=240).step_by(2) {
            let mut points: Vec<(f64, f64)> = Vec::new();
            for z in 1..a {
                let n = a - z;
                if n < 1 {
                    continue;
                }
                if !measured.contains(&(z as u16, n as u16)) {
                    continue;
                }
                let be = be_map[&(z as u16, n as u16)];
                // Subtract Coulomb contribution before fitting symmetry term
                let a_c = 0.7; // Coulomb coefficient (MeV)
                let z_f = z as f64;
                let a_f = a as f64;
                let coulomb = a_c * z_f * (z_f - 1.0) / a_f.powf(1.0 / 3.0);
                let be_corrected = be + coulomb;
                let x = ((n as f64 - z as f64).powi(2)) / a as f64;
                points.push((x, be_corrected));
            }
            if points.len() < 4 {
                continue;
            }

            let n_pts = points.len() as f64;
            let sx: f64 = points.iter().map(|p| p.0).sum();
            let sy: f64 = points.iter().map(|p| p.1).sum();
            let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
            let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
            let d = n_pts * sxx - sx * sx;
            if d.abs() < 1e-10 {
                continue;
            }

            let slope = (n_pts * sxy - sx * sy) / d;
            let a_sym = -slope;
            if a_sym > 0.0 && a_sym < 100.0 {
                a_sym_data.push((1.0 / (a as f64).powf(1.0 / 3.0), a_sym));
            }
        }

        // Fit: a_sym(A) = J + K_ss * A^{-1/3}
        // where J = volume symmetry energy at saturation
        if a_sym_data.len() < 3 {
            eprintln!("Insufficient data for EOS extraction.");
            return;
        }

        let n_f = a_sym_data.len() as f64;
        let sx: f64 = a_sym_data.iter().map(|p| p.0).sum();
        let sy: f64 = a_sym_data.iter().map(|p| p.1).sum();
        let sxy: f64 = a_sym_data.iter().map(|p| p.0 * p.1).sum();
        let sxx: f64 = a_sym_data.iter().map(|p| p.0 * p.0).sum();
        let d = n_f * sxx - sx * sx;
        let k_ss = (n_f * sxy - sx * sy) / d;
        let j_vol = (sy - k_ss * sx) / n_f;

        eprintln!(
            "Extracted parameters from {} isobaric chains:",
            a_sym_data.len()
        );
        eprintln!(
            "  Volume symmetry energy  J = {:.2} MeV (accepted ≈ 30-34 MeV)",
            j_vol
        );
        eprintln!("  Surface symmetry coeff  K_ss = {:.2} MeV", k_ss);

        // Estimate L from the empirical correlation:
        // L ≈ (3/ρ₀) × d(a_sym)/d(ρ) ≈ ... from finite nucleus data
        // Empirical relation (Danielewicz 2004): L ≈ 3 × (J - a_sym(A=208))
        // Using Pb-208 as the reference heavy nucleus
        let a_sym_208 = a_sym_data
            .iter()
            .min_by(|(x, _), (x2, _)| {
                let d1 = (x - 1.0 / 208.0_f64.powf(1.0 / 3.0)).abs();
                let d2 = (x2 - 1.0 / 208.0_f64.powf(1.0 / 3.0)).abs();
                d1.partial_cmp(&d2).unwrap()
            })
            .map(|(_, v)| *v)
            .unwrap_or(23.0);

        let l_param = 3.0 * (j_vol - a_sym_208).abs();
        // Alternative: Centelles et al. (2009) correlation
        // L ≈ (9 * J^2) / (4 * a_sym_surface) when surface term known
        let l_centelles = if k_ss.abs() > 1e-6 {
            (9.0 * j_vol * j_vol) / (4.0 * k_ss.abs())
        } else {
            0.0
        };

        eprintln!("\nSlope parameter L estimates:");
        eprintln!(
            "  Danielewicz method:     L ≈ {:.1} MeV  (using a_sym(A~208) = {:.1} MeV)",
            l_param, a_sym_208
        );
        eprintln!("  Centelles correlation:  L ≈ {:.1} MeV", l_centelles);

        eprintln!("\nGW170817 constraint check (40 < L < 120 MeV):");
        for (name, l_val) in [("Danielewicz", l_param), ("Centelles", l_centelles)] {
            let status = if l_val > 40.0 && l_val < 120.0 {
                "CONSISTENT"
            } else if l_val > 20.0 && l_val < 160.0 {
                "MARGINAL"
            } else {
                "OUTSIDE"
            };
            eprintln!("  {} L = {:.1} MeV → {}", name, l_val, status);
        }

        // Neutron star radius estimate from L
        // R_1.4 ≈ (10.0 + 0.04 * L) km  (Lattimer & Prakash 2001 approximation)
        let r_14_d = 10.0 + 0.04 * l_param;
        let r_14_c = 10.0 + 0.04 * l_centelles;
        eprintln!("\nNeutron star radius estimates (1.4 M_sun):");
        eprintln!("  From Danielewicz L: R ≈ {:.1} km", r_14_d);
        eprintln!("  From Centelles L:   R ≈ {:.1} km", r_14_c);
        eprintln!("  NICER observation:  R = 12.4 ± 1.0 km (Miller et al. 2021)");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 8. COULOMB CORRECTION FOR DOUBLE-BETA DECAY Q-VALUES
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn coulomb_correction_for_qbb() {
        let pred = predictor();

        eprintln!(
            "\n{} COULOMB CORRECTION FOR ββ Q-VALUES {}",
            "=".repeat(12),
            "=".repeat(12)
        );
        eprintln!("Fitting a single Coulomb asymmetry parameter to reduce");
        eprintln!("the systematic bias in double-beta decay Q-value predictions.\n");

        // 13 double-beta decay candidates: (Z, N, name, Q_exp in keV)
        let candidates: &[(u16, u16, &str, f64)] = &[
            (20, 28, "Ca-48", 4268.0),
            (32, 44, "Ge-76", 2039.0),
            (34, 48, "Se-82", 2998.0),
            (36, 50, "Kr-86", 1257.0),
            (40, 56, "Zr-96", 3356.0),
            (42, 58, "Mo-100", 3034.0),
            (46, 64, "Pd-110", 2018.0),
            (48, 68, "Cd-116", 2814.0),
            (50, 74, "Sn-124", 2289.0),
            (52, 78, "Te-130", 2527.0),
            (54, 82, "Xe-136", 2458.0),
            (60, 90, "Nd-150", 3371.0),
            (92, 146, "U-238", 1145.0),
        ];

        let m_e_mev = 0.51100; // electron mass in MeV
        let two_m_e = 2.0 * m_e_mev; // 1.022 MeV

        // Compute predicted Q-values and Coulomb asymmetry variable
        struct QbbEntry {
            name: &'static str,
            z: u16,
            n: u16,
            q_exp: f64,    // keV
            q_pred: f64,   // keV
            residual: f64, // keV (exp - pred)
            x_coulomb: f64,
        }

        let mut entries: Vec<QbbEntry> = Vec::new();

        for &(z, n, name, q_exp_kev) in candidates {
            // Daughter: Z+2, N-2
            let be_parent = pred.predict(z, n).binding_energy; // MeV
            let be_daughter = pred.predict(z + 2, n - 2).binding_energy; // MeV

            // Q_bb = BE(daughter) - BE(parent) + 2*m_e - 2*m_e
            // More precisely: Q = [M(parent) - M(daughter) - 2*m_e] c^2
            //   = [BE(Z+2,N-2) - BE(Z,N)] + 2*0.78235 - 1.022 MeV
            // where 0.78235 MeV = (m_n - m_p) c^2
            let q_pred_mev = (be_daughter - be_parent) + 2.0 * 0.78235 - two_m_e;
            let q_pred_kev = q_pred_mev * 1000.0;
            let residual = q_exp_kev - q_pred_kev;

            // Coulomb asymmetry variable:
            // x_i = (Z+2)(Z+1)/A^{1/3} - Z(Z-1)/A^{1/3}
            let a = (z as f64) + (n as f64);
            let a_cbrt = a.powf(1.0 / 3.0);
            let z_f = z as f64;
            let x_coulomb = ((z_f + 2.0) * (z_f + 1.0) - z_f * (z_f - 1.0)) / a_cbrt;

            entries.push(QbbEntry {
                name,
                z,
                n,
                q_exp: q_exp_kev,
                q_pred: q_pred_kev,
                residual,
                x_coulomb,
            });
        }

        // Fit single parameter c via least squares:
        // c = sum(residual_i * x_i) / sum(x_i^2)
        let sum_rx: f64 = entries.iter().map(|e| e.residual * e.x_coulomb).sum();
        let sum_xx: f64 = entries.iter().map(|e| e.x_coulomb * e.x_coulomb).sum();
        let c_fit = sum_rx / sum_xx;

        // Compute RMS before and after correction
        let n_pts = entries.len() as f64;
        let rms_orig =
            (entries.iter().map(|e| e.residual * e.residual).sum::<f64>() / n_pts).sqrt();
        let rms_corr = (entries
            .iter()
            .map(|e| {
                let corr_resid = e.residual - c_fit * e.x_coulomb;
                corr_resid * corr_resid
            })
            .sum::<f64>()
            / n_pts)
            .sqrt();

        let mean_orig = entries.iter().map(|e| e.residual).sum::<f64>() / n_pts;
        let mean_corr = entries
            .iter()
            .map(|e| e.residual - c_fit * e.x_coulomb)
            .sum::<f64>()
            / n_pts;

        eprintln!(
            "Fitted Coulomb correction coefficient: c = {:.3} keV",
            c_fit
        );
        eprintln!();
        eprintln!(
            "{:<10} {:>4} {:>4} {:>4}  {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "Nucleus", "Z", "N", "A", "Q_exp", "Q_pred", "Resid", "x_C", "Q_corr", "Resid_c"
        );
        eprintln!(
            "{:<10} {:>4} {:>4} {:>4}  {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "", "", "", "", "(keV)", "(keV)", "(keV)", "", "(keV)", "(keV)"
        );

        for e in &entries {
            let q_corrected = e.q_pred + c_fit * e.x_coulomb;
            let resid_corrected = e.q_exp - q_corrected;
            eprintln!(
                "{:<10} {:4} {:4} {:4}  {:8.1} {:8.1} {:8.1} {:8.3} {:10.1} {:10.1}",
                e.name,
                e.z,
                e.n,
                e.z as u32 + e.n as u32,
                e.q_exp,
                e.q_pred,
                e.residual,
                e.x_coulomb,
                q_corrected,
                resid_corrected
            );
        }

        eprintln!();
        eprintln!("── Summary ──");
        eprintln!("  Original  mean bias: {:+.1} keV", mean_orig);
        eprintln!("  Corrected mean bias: {:+.1} keV", mean_corr);
        eprintln!("  Original  RMS:       {:.1} keV", rms_orig);
        eprintln!("  Corrected RMS:       {:.1} keV", rms_corr);
        eprintln!(
            "  Improvement:         {:.1}%",
            100.0 * (1.0 - rms_corr / rms_orig)
        );
        eprintln!("  Coulomb coeff c:     {:.3} keV", c_fit);
    }
}
