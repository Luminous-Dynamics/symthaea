// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Nuclear Physics Exploration Engine
//!
//! Comprehensive analysis driver that runs the DZ10+RF mass predictor across
//! all 6 application domains and extracts physics findings.

#[cfg(test)]
mod tests {
    use crate::fundamental::*;
    use crate::isotope_properties::*;
    use crate::medical_isotopes::*;
    use crate::ml_mass::MlMassPredictor;
    use crate::nuclear_forensics::*;
    use crate::reactor::*;
    use crate::rprocess::{RProcessConditions, RProcessConfig, RProcessNetwork};
    use crate::space_nuclear::*;
    use std::io::Write;

    fn predictor() -> MlMassPredictor {
        MlMassPredictor::new()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. R-PROCESS NUCLEOSYNTHESIS
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn explore_rprocess() {
        eprintln!(
            "\n{} R-PROCESS NUCLEOSYNTHESIS {}",
            "=".repeat(20),
            "=".repeat(20)
        );
        let mut net = RProcessNetwork::new();
        // "Hot dynamical" NSM ejecta: moderate density, tuned expansion
        // timescale to balance neutron supply against shell quenching.
        // The key physics: N=82 and N=126 shell closures suppress (n,γ)
        // by ~1000×, creating abundance peaks as material accumulates at
        // waiting points before beta-decaying through them.
        let config = RProcessConfig {
            conditions: RProcessConditions {
                temperature_gk: 2.0,
                neutron_density: 1.0e24,
                ye: 0.15,
                expansion_timescale: 0.85,
            },
            dt: 1.0e-3,
            max_time: 2.5,
            freezeout_steps: 5000,
            freezeout_dt: 5.0e-3,
            enable_equilibrium: true,
            waiting_point_sn: 2.5,
        };

        let result = net.simulate(&config);

        eprintln!("\n--- Simulation Summary ---");
        eprintln!(
            "Total mass numbers with material: {}",
            result.abundances.len()
        );
        eprintln!("Waiting points found: {}", result.waiting_points.len());
        eprintln!("Path length: {} steps", result.path.len());

        // Top 30 most abundant mass numbers
        let mut sorted: Vec<_> = result.abundances.iter().collect();
        sorted.sort_by(|a, b| b.abundance.total_cmp(&a.abundance));

        eprintln!("\n--- Top 30 Most Abundant Mass Numbers ---");
        eprintln!("{:>6} {:>12}", "A", "Abundance");
        eprintln!("{}", "-".repeat(20));
        for m in sorted.iter().take(30) {
            eprintln!("{:>6} {:>12.6e}", m.a, m.abundance);
        }

        // Shell closure peaks
        let n82: f64 = result
            .abundances
            .iter()
            .filter(|m| (125..=135).contains(&m.a))
            .map(|m| m.abundance)
            .sum();
        let n126: f64 = result
            .abundances
            .iter()
            .filter(|m| (190..=200).contains(&m.a))
            .map(|m| m.abundance)
            .sum();
        let rare_earth: f64 = result
            .abundances
            .iter()
            .filter(|m| (155..=170).contains(&m.a))
            .map(|m| m.abundance)
            .sum();

        eprintln!("\n--- Shell Closure Peaks ---");
        eprintln!("A~130 (N=82):      {:.12}", n82);
        eprintln!("A~165 (rare earth): {:.12}", rare_earth);
        eprintln!("A~195 (N=126):      {:.12}", n126);

        // Waiting points
        eprintln!("\n--- Waiting Points (beta-decay bottlenecks) ---");
        for wp in result.waiting_points.iter().take(15) {
            eprintln!("  Z={:>3} N={:>3} (A={})", wp.z, wp.n, wp.a());
        }

        // Heaviest element produced
        if let Some(heaviest) = result
            .abundances
            .iter()
            .filter(|m| m.abundance > 1.0e-6)
            .max_by_key(|m| m.a)
        {
            eprintln!("\nHeaviest element with Y > 1e-6: A={}", heaviest.a);
        }

        // Write TSV for plotting
        let tsv_dir = "/tmp/nuclear_figures";
        std::fs::create_dir_all(tsv_dir).expect("create /tmp/nuclear_figures");
        let tsv_path = format!("{}/rprocess_abundances.tsv", tsv_dir);
        let mut f = std::fs::File::create(&tsv_path).expect("create TSV");
        writeln!(f, "A\tabundance").unwrap();
        let mut by_a: Vec<_> = result
            .abundances
            .iter()
            .filter(|m| m.abundance > 1.0e-8)
            .collect();
        by_a.sort_by_key(|m| m.a);
        for m in &by_a {
            writeln!(f, "{}\t{:.12}", m.a, m.abundance).unwrap();
        }
        eprintln!("\nWrote {} entries to {}", by_a.len(), tsv_path);

        // Solar r-process comparison
        eprintln!("\n--- Solar R-Process Comparison (Arlandini 1999) ---");
        let solar = &result.solar_comparison;
        let solar_tsv = format!("{}/solar_comparison.tsv", tsv_dir);
        let mut sf = std::fs::File::create(&solar_tsv).expect("create solar TSV");
        writeln!(sf, "A\tpredicted\tsolar\tratio").unwrap();
        let mut chi2 = 0.0_f64;
        let mut n_points = 0usize;
        let mut log_resid_sq_sum = 0.0_f64;
        eprintln!(
            "{:>6} {:>12} {:>12} {:>8}",
            "A", "Predicted", "Solar", "Ratio"
        );
        eprintln!("{}", "-".repeat(44));
        for sc in solar {
            writeln!(
                sf,
                "{}\t{:.12}\t{:.12}\t{:.4}",
                sc.a, sc.predicted, sc.solar, sc.ratio
            )
            .unwrap();
            eprintln!(
                "{:>6} {:>12.4e} {:>12.4e} {:>8.3}",
                sc.a, sc.predicted, sc.solar, sc.ratio
            );
            if sc.solar > 0.0 && sc.predicted > 0.0 {
                let diff = sc.predicted - sc.solar;
                chi2 += (diff * diff) / sc.solar;
                let log_r = (sc.predicted / sc.solar).ln();
                log_resid_sq_sum += log_r * log_r;
                n_points += 1;
            }
        }
        let rms_log_ratio = if n_points > 0 {
            (log_resid_sq_sum / n_points as f64).sqrt()
        } else {
            0.0
        };
        eprintln!(
            "\nSolar comparison: {} points, chi2 = {:.4}, RMS(log ratio) = {:.4}",
            n_points, chi2, rms_log_ratio
        );
        eprintln!("Wrote solar comparison to {}", solar_tsv);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. REACTOR PHYSICS
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn explore_reactor() {
        let pred = predictor();

        eprintln!("\n{} REACTOR PHYSICS {}", "=".repeat(20), "=".repeat(20));

        // U-235 fission
        let u235 = fission_product_yields(&pred, 92, 143);
        eprintln!("\n--- U-235 Thermal Fission ---");
        eprintln!("Q_fission:  {:.1} MeV", u235.q_fission_mev);
        eprintln!("TKE (Viola): {:.1} MeV", u235.tke_viola_mev);
        eprintln!("Light peak:  A={}", u235.peak_light);
        eprintln!("Heavy peak:  A={}", u235.peak_heavy);
        eprintln!("\nTop fission fragments:");
        eprintln!("{:>4} {:>4} {:>8} {:>10}", "Z", "A", "Yield", "BE(MeV)");
        for fp in u235.fragments.iter().take(10) {
            eprintln!(
                "{:>4} {:>4} {:>8.4} {:>10.4}",
                fp.z, fp.a, fp.yield_fraction, fp.binding_energy
            );
        }

        // Pu-239 fission
        let pu239 = fission_product_yields(&pred, 94, 145);
        eprintln!("\n--- Pu-239 Fission ---");
        eprintln!("Q_fission:  {:.1} MeV", pu239.q_fission_mev);
        eprintln!("TKE (Viola): {:.1} MeV", pu239.tke_viola_mev);
        eprintln!("Light peak:  A={}", pu239.peak_light);
        eprintln!("Heavy peak:  A={}", pu239.peak_heavy);

        // Transmutation analysis for key waste isotopes
        eprintln!("\n--- Waste Transmutation Pathways ---");
        let targets = [
            (38, 52, "Sr-90"),
            (55, 82, "Cs-137"),
            (53, 76, "I-129"),
            (43, 56, "Tc-99"),
            (94, 145, "Pu-239"),
        ];
        for (z, n, name) in targets {
            let chain = transmutation_chain(&pred, z, n, 10);
            eprintln!(
                "{}: {} steps → {:?}",
                name,
                chain.steps.len(),
                chain
                    .steps
                    .last()
                    .map(|s| format!("Z={} N={}", s.z, s.n))
                    .unwrap_or("stable".into())
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. MEDICAL ISOTOPES
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn explore_medical() {
        let pred = predictor();

        eprintln!(
            "\n{} MEDICAL ISOTOPE DISCOVERY {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        let candidates = scan_novel_candidates(&pred);
        eprintln!("Total novel candidates: {}", candidates.len());

        // Group by modality
        let alpha: Vec<_> = candidates
            .iter()
            .filter(|c| c.suggested_modality == ClinicalModality::AlphaTherapy)
            .collect();
        let pet: Vec<_> = candidates
            .iter()
            .filter(|c| c.suggested_modality == ClinicalModality::PET)
            .collect();
        let beta: Vec<_> = candidates
            .iter()
            .filter(|c| c.suggested_modality == ClinicalModality::BetaTherapy)
            .collect();

        eprintln!("  Alpha therapy: {}", alpha.len());
        eprintln!("  PET imaging:   {}", pet.len());
        eprintln!("  Beta therapy:  {}", beta.len());

        // Most promising alpha candidates (lowest uncertainty, right Q range)
        let mut alpha_sorted: Vec<_> = alpha
            .iter()
            .filter(|c| c.q_value > 5.0 && c.q_value < 8.0)
            .collect();
        alpha_sorted.sort_by(|a, b| a.uncertainty.total_cmp(&b.uncertainty));

        eprintln!("\n--- Top Alpha Therapy Candidates (Q=5-8 MeV, low uncertainty) ---");
        eprintln!(
            "{:<8} {:>4} {:>4} {:>8} {:>8} {:>20}",
            "Symbol", "Z", "A", "Q(MeV)", "σ(MeV)", "Half-life"
        );
        for c in alpha_sorted.iter().take(15) {
            eprintln!(
                "{:<8} {:>4} {:>4} {:>8.3} {:>8.3} {:>20}",
                c.symbol, c.z, c.a, c.q_value, c.uncertainty, c.half_life_category
            );
        }

        // Most promising PET candidates
        let mut pet_sorted: Vec<_> = pet.iter().collect();
        pet_sorted.sort_by(|a, b| a.uncertainty.total_cmp(&b.uncertainty));

        eprintln!("\n--- Top PET Imaging Candidates (low uncertainty) ---");
        eprintln!(
            "{:<8} {:>4} {:>4} {:>8} {:>8} {:>20}",
            "Symbol", "Z", "A", "Q(MeV)", "σ(MeV)", "Half-life"
        );
        for c in pet_sorted.iter().take(10) {
            eprintln!(
                "{:<8} {:>4} {:>4} {:>8.3} {:>8.3} {:>20}",
                c.symbol, c.z, c.a, c.q_value, c.uncertainty, c.half_life_category
            );
        }

        // Ac-225 production routes
        let routes = analyze_ac225_production(&pred);
        eprintln!("\n--- Ac-225 Production Routes ---");
        for r in &routes {
            eprintln!(
                "  {}: Q={:.2} MeV (σ={:.2}), {}",
                r.route_description,
                r.q_value_ml,
                r.q_uncertainty,
                if r.is_exothermic {
                    "exothermic"
                } else {
                    "endothermic"
                }
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. NUCLEAR FORENSICS
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn explore_forensics() {
        let pred = predictor();

        eprintln!("\n{} NUCLEAR FORENSICS {}", "=".repeat(20), "=".repeat(20));

        // Trace the 4 natural decay series
        let series = [
            (92, 146, "U-238 (4n+2)"),
            (90, 142, "Th-232 (4n)"),
            (92, 143, "U-235 (4n+3)"),
            (93, 144, "Np-237 (4n+1)"),
        ];

        for (z, n, name) in series {
            let chain = trace_decay_chain(&pred, z, n, 30);
            eprintln!("\n--- {} Decay Chain ---", name);
            eprintln!(
                "{:>6} {:>4} {:>4} {:>10} {:>12}",
                "Step", "Z", "A", "Mode", "Q(MeV)"
            );
            for (i, step) in chain.steps.iter().enumerate().take(20) {
                eprintln!(
                    "{:>6} {:>4} {:>4} {:>10} {:>12.3}",
                    i,
                    step.z,
                    step.z + step.n,
                    format!("{:?}", step.decay_mode),
                    step.q_value
                );
            }
            if let Some(last) = chain.steps.last() {
                eprintln!("  → Terminates at Z={} A={}", last.z, last.z + last.n);
            }
        }

        // Enrichment signatures
        eprintln!("\n--- Uranium Enrichment Signatures ---");
        let enrichments = [0.007, 0.035, 0.05, 0.20, 0.90, 0.93];
        for e in enrichments {
            let sig = uranium_enrichment_indicator(e);
            eprintln!(
                "  {:.1}%: U234/U238={:.4e}, U235/U238={:.4e}",
                e * 100.0,
                sig.u234_u238_activity,
                sig.u235_u238_atom
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. SPACE NUCLEAR
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn explore_space() {
        let pred = predictor();

        eprintln!(
            "\n{} SPACE NUCLEAR APPLICATIONS {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        let report = generate_space_nuclear_report(&pred, 15.0);

        // RTG fuels ranked by specific power
        eprintln!("\n--- RTG Fuel Ranking (15-year mission) ---");
        eprintln!(
            "{:<12} {:>10} {:>10} {:>10} {:>8} {:>10}",
            "Fuel", "W/g", "t½(yr)", "Q(MeV)", "Mode", "15yr W/g"
        );
        for f in &report.rtg_known {
            let power_15yr = f.specific_power_wg * (-0.693 * 15.0 / f.half_life_years).exp();
            eprintln!(
                "{:<12} {:>10.4} {:>10.1} {:>10.2} {:>8} {:>10.4}",
                f.name,
                f.specific_power_wg,
                f.half_life_years,
                f.q_value_mev,
                f.decay_mode,
                power_15yr
            );
        }

        // Novel RTG candidates
        if !report.rtg_novel.is_empty() {
            eprintln!("\n--- Novel RTG Fuel Candidates ---");
            eprintln!(
                "{:<12} {:>4} {:>4} {:>10} {:>10}",
                "Name", "Z", "A", "W/g", "Q(MeV)"
            );
            for f in report.rtg_novel.iter().take(10) {
                eprintln!(
                    "{:<12} {:>4} {:>4} {:>10.4} {:>10.2}",
                    f.name, f.z, f.a, f.specific_power_wg, f.q_value_mev
                );
            }
        }

        // Shielding materials
        eprintln!("\n--- Cosmic Ray Shielding (ranked by shielding score) ---");
        eprintln!(
            "{:<20} {:>8} {:>8} {:>10}",
            "Material", "avg_Z", "avg_A", "Score"
        );
        for s in report.shielding.iter().take(8) {
            eprintln!(
                "{:<20} {:>8.1} {:>8.1} {:>10.3}",
                s.name, s.avg_z, s.avg_a, s.shielding_score
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. FUNDAMENTAL NUCLEAR SCIENCE
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn explore_fundamental() {
        let pred = predictor();

        eprintln!(
            "\n{} FUNDAMENTAL NUCLEAR SCIENCE {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // Magic number verification — shell_gap(pred, z, n) measures Δ_n = S_2n(N) - S_2n(N+2)
        // We use a representative Z=50 (tin chain) for neutron shell gaps
        eprintln!("\n--- Shell Closure Verification (Z=50 tin chain) ---");
        let magic = [2, 8, 20, 28, 50, 82, 126];
        for &m in &magic {
            let gap = shell_gap(&pred, 50, m);
            eprintln!("  N={:>3}: shell gap = {:.3} MeV", m, gap);
        }

        // Two-neutron separation energy across Sn chain (Z=50, magic proton)
        eprintln!("\n--- Tin (Z=50) Isotope Chain: S_2n ---");
        eprintln!("{:>6} {:>4} {:>10}", "A", "N", "S_2n(MeV)");
        for n in (50..=90).step_by(2) {
            let s2n = two_neutron_separation(&pred, 50, n);
            let marker = if n == 82 { " ← N=82 magic" } else { "" };
            eprintln!("{:>6} {:>4} {:>10.3}{}", 50 + n, n, s2n, marker);
        }

        // Pairing gap
        eprintln!("\n--- Pairing Gap Δ(3) for Ca chain (Z=20) ---");
        eprintln!("{:>6} {:>10}", "N", "Δ(MeV)");
        for n in 18..=32 {
            let gap = neutron_pairing_gap(&pred, 20, n);
            let marker = if n == 20 || n == 28 { " ← magic" } else { "" };
            eprintln!("{:>6} {:>10.3}{}", n, gap, marker);
        }

        // Drip line
        eprintln!("\n--- Neutron Drip Line (S_n < 0) ---");
        eprintln!("{:>6} {:>6} {:>6}", "Z", "N_drip", "A_drip");
        for z in (8..=82).step_by(2) {
            let mut n_drip = 0u16;
            for n in 1..=200u16 {
                let sn = one_neutron_separation(&pred, z, n);
                if sn < 0.0 {
                    n_drip = n - 1;
                    break;
                }
            }
            if n_drip > 0 {
                eprintln!("{:>6} {:>6} {:>6}", z, n_drip, z + n_drip);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 7. CROSS-DOMAIN: ISLAND OF STABILITY DEEP DIVE
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn explore_island_of_stability() {
        let pred = predictor();

        eprintln!(
            "\n{} ISLAND OF STABILITY DEEP CHARACTERIZATION {}",
            "=".repeat(20),
            "=".repeat(20)
        );

        // Scan Z=110-120, N=170-190 for the predicted island
        eprintln!("\n--- Binding Energy Landscape (B/A) ---");
        eprintln!(
            "{:>6} {:>6} {:>6} {:>10} {:>10} {:>10}",
            "Z", "N", "A", "B/A(MeV)", "σ(MeV)", "S_n(MeV)"
        );

        let mut best_ba = 0.0f64;
        let mut best_z = 0u16;
        let mut best_n = 0u16;

        for z in 110..=120 {
            for n in 170..=190 {
                let p = pred.predict(z, n);
                let a = z + n;
                let ba = p.binding_energy / a as f64;

                let sn = if n > 1 {
                    let p_prev = pred.predict(z, n - 1);
                    p.binding_energy - p_prev.binding_energy
                } else {
                    0.0
                };

                if ba > best_ba {
                    best_ba = ba;
                    best_z = z;
                    best_n = n;
                }

                // Only print select values
                if n % 4 == 0 && z % 2 == 0 {
                    eprintln!(
                        "{:>6} {:>6} {:>6} {:>10.4} {:>10.3} {:>10.3}",
                        z, n, a, ba, p.uncertainty, sn
                    );
                }
            }
        }

        eprintln!(
            "\nBest B/A in island: Z={} N={} A={} → {:.4} MeV",
            best_z,
            best_n,
            best_z + best_n,
            best_ba
        );

        // Alpha decay chains from Oganesson
        eprintln!("\n--- Alpha Decay Chain from Og-294 ---");
        let mut z = 118u16;
        let mut n = 176u16;
        for i in 0..10 {
            let p = pred.predict(z, n);
            let a = z + n;
            let ba = p.binding_energy / a as f64;

            let q_alpha = if z > 2 && n > 2 {
                let daughter = pred.predict(z - 2, n - 2);
                daughter.binding_energy + 28.296 - p.binding_energy
            } else {
                0.0
            };

            eprintln!(
                "  {:>2}. Z={:>3} N={:>3} A={:>3} B/A={:.4} Q_α={:.2} MeV",
                i, z, n, a, ba, q_alpha
            );

            if q_alpha <= 0.0 || z <= 82 {
                break;
            }
            z -= 2;
            n -= 2;
        }
    }
}
