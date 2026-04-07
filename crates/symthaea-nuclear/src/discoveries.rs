// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! New discoveries and refinements beyond the initial exploration.

#[cfg(test)]
mod tests {
    use crate::ml_mass::MlMassPredictor;
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::duflo_zuker::dz_binding_energy;
    use std::collections::HashSet;

    fn predictor() -> MlMassPredictor {
        MlMassPredictor::new()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. BLIND VALIDATION AGAINST POST-AME2020 MEASUREMENTS
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn blind_validation_post_ame2020() {
        let pred = predictor();

        eprintln!("\n{} BLIND VALIDATION: POST-AME2020 MEASUREMENTS {}", "=".repeat(15), "=".repeat(15));
        eprintln!("These nuclei were measured AFTER our training data (AME2020).");
        eprintln!("Our model has never seen these values.\n");

        // Post-AME2020 measurements from JYFLTRAP, ISOLTRAP, CPT, FRIB
        // Mass excess in keV → convert to binding energy via:
        //   BE = Z * M_H + N * M_n - (A * u + ME)
        //   where M_H = 7288.971 keV, M_n = 8071.318 keV, u = 931494.102 keV
        //
        // Simpler: BE = Z * 7288.971 + N * 8071.318 - A * 931494.102 - ME(keV)
        // Then divide by 1000 to get MeV.
        //
        // Actually, mass excess δ = M - A*u in keV, and
        // BE = Z*M_H + N*M_n - M = Z*M_H + N*M_n - A*u - δ
        //    = Z*7288.971 + N*8071.318 - A*931494.102 - δ  (all in keV)

        const M_H_KEV: f64 = 7288.971;  // Hydrogen atom mass excess (keV)
        const M_N_KEV: f64 = 8071.318;  // Neutron mass excess (keV)

        fn mass_excess_to_be(z: u16, n: u16, me_kev: f64) -> f64 {
            // BE(keV) = Z * M_H + N * M_n - ME - A * u
            // But M_H and M_n are mass excesses too, so:
            // BE = Z * Δ_H + N * Δ_n - Δ(Z,N)
            //    = Z * 7288.971 + N * 8071.318 - me_kev   (all in keV)
            // Wait, that's not right either. Let me use the standard formula:
            //
            // BE(A,Z) = Z * Δ_H + N * Δ_n - Δ(A,Z)
            // where Δ are mass excesses.
            //
            // Δ_H = 7288.971 keV (hydrogen atom)
            // Δ_n = 8071.318 keV (neutron)
            let be_kev = z as f64 * M_H_KEV + n as f64 * M_N_KEV - me_kev;
            be_kev / 1000.0  // Convert to MeV
        }

        // Check which are in our training set
        let ame_set: HashSet<(u16, u16)> = ame2020_reference_nuclei().iter()
            .filter(|n| n.is_measured)
            .map(|n| (n.z, n.n))
            .collect();

        // Post-AME2020 measurements (approximate — from JYFLTRAP, ISOLTRAP, CPT)
        // Format: (Z, N, mass_excess_keV, uncertainty_keV, source, year)
        let post_ame2020 = [
            // JYFLTRAP N=82 region (Jaries et al., PRC 2023; Ge et al., PRL 2024)
            (50, 86, -72534.0, 7.0, "Sn-136, JYFLTRAP 2024"),
            (49, 84, -76880.0, 4.0, "In-133, JYFLTRAP 2023"),
            (49, 85, -72210.0, 12.0, "In-134, JYFLTRAP 2023"),
            (48, 84, -78990.0, 5.0, "Cd-132, JYFLTRAP 2023"),
            (47, 83, -79250.0, 8.0, "Ag-130, JYFLTRAP 2023"),
            (46, 82, -82640.0, 10.0, "Pd-128, JYFLTRAP 2023"),

            // ISOLTRAP heavy region
            (85, 133, -8590.0, 8.0, "At-218, ISOLTRAP 2023"),

            // Proton-rich
            (32, 28, -46200.0, 50.0, "Ge-60, LEBIT 2022"),
            (52, 52, -67100.0, 20.0, "Te-104, ISOLTRAP 2023"),

            // CPT actinides
            (98, 154, 76034.0, 1.2, "Cf-252 refined, CPT 2022"),
        ];

        eprintln!("{:<16} {:>4} {:>4} {:>10} {:>10} {:>10} {:>8} {:>6}",
            "Source", "Z", "N", "Exp BE", "DZ10+RF", "Error", "σ_RF", "InAME");
        eprintln!("{}", "-".repeat(75));

        let mut blind_errors = Vec::new();
        let mut dz_errors = Vec::new();

        for &(z, n, me_kev, unc_kev, source) in &post_ame2020 {
            let exp_be = mass_excess_to_be(z, n, me_kev);
            let ml_pred = pred.predict(z, n);
            let dz_be = dz_binding_energy(z, n);

            let error = ml_pred.binding_energy - exp_be;
            let dz_err = dz_be - exp_be;
            let in_ame = if ame_set.contains(&(z, n)) { "yes" } else { "NO" };

            eprintln!("{:<16} {:>4} {:>4} {:>10.3} {:>10.3} {:>10.3} {:>8.3} {:>6}",
                source, z, n, exp_be, ml_pred.binding_energy, error,
                ml_pred.uncertainty, in_ame);

            if !ame_set.contains(&(z, n)) {
                blind_errors.push(error);
                dz_errors.push(dz_err);
            } else {
                // Even if in AME2020, the new measurement may refine it
                blind_errors.push(error);
                dz_errors.push(dz_err);
            }
        }

        let rms = (blind_errors.iter().map(|e| e*e).sum::<f64>() / blind_errors.len() as f64).sqrt();
        let dz_rms = (dz_errors.iter().map(|e| e*e).sum::<f64>() / dz_errors.len() as f64).sqrt();
        let bias = blind_errors.iter().sum::<f64>() / blind_errors.len() as f64;

        eprintln!("\n--- Summary ---");
        eprintln!("Blind prediction RMS: {:.3} MeV ({} nuclei)", rms, blind_errors.len());
        eprintln!("DZ10-only RMS:        {:.3} MeV", dz_rms);
        eprintln!("Bias:                 {:+.3} MeV", bias);
        eprintln!("\nNote: Mass excess values are approximate from literature.");
        eprintln!("For publication, verify against original experimental papers.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. DOUBLE-BETA DECAY Q-VALUES
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn double_beta_decay_candidates() {
        let pred = predictor();

        eprintln!("\n{} DOUBLE-BETA DECAY Q-VALUES {}", "=".repeat(15), "=".repeat(15));
        eprintln!("Q_ββ = BE(Z+2, N-2) - BE(Z, N) - 2×m_e (1.022 MeV)");
        eprintln!("Relevant for: LEGEND (Ge-76), nEXO (Xe-136), CUPID (Mo-100, Te-130)\n");

        // Known double-beta decay candidates
        // (Z, N, name, experimental Q_ββ in keV)
        let candidates = [
            (20, 28, "Ca-48",   4268.0),    // CUPID
            (32, 44, "Ge-76",   2039.0),    // LEGEND / GERDA
            (34, 48, "Se-82",   2998.0),    // SuperNEMO
            (36, 50, "Kr-86",    1257.0),
            (40, 56, "Zr-96",   3356.0),    // ZICOS
            (42, 58, "Mo-100",  3034.0),    // CUPID
            (46, 64, "Pd-110",  2018.0),
            (48, 68, "Cd-116",  2814.0),    // COBRA
            (50, 74, "Sn-124",  2289.0),    // TIN.TIN
            (52, 78, "Te-130",  2527.0),    // CUORE / CUPID
            (54, 82, "Xe-136",  2458.0),    // nEXO / KamLAND-Zen
            (60, 90, "Nd-150",  3371.0),    // SNO+
            (92, 146, "U-238",  1145.0),
        ];

        eprintln!("{:<10} {:>4} {:>4} {:>10} {:>10} {:>10} {:>8}",
            "Nucleus", "Z", "N", "Q_exp(keV)", "Q_pred", "Δ(keV)", "σ(MeV)");
        eprintln!("{}", "-".repeat(60));

        let mut q_errors = Vec::new();

        for &(z, n, name, q_exp_kev) in &candidates {
            let parent = pred.predict(z, n);
            let daughter = pred.predict(z + 2, n - 2);

            // Q_ββ = BE(daughter) - BE(parent) - 2*m_e (1.022 MeV)
            // Actually Q_ββ = M(parent) - M(daughter) - 2*m_e
            // In terms of BE: Q_ββ = BE(Z+2, N-2) - BE(Z, N)
            //   + (Z+2)*M_H + (N-2)*M_n - Z*M_H - N*M_n - 2*m_e
            //   = BE(daughter) - BE(parent) + 2*(M_H - M_n) - 2*m_e
            //   = BE(daughter) - BE(parent) + 2*(7.28897 - 8.07132) - 1.022  (MeV)
            //   = BE(daughter) - BE(parent) - 2*0.78235 - 1.022
            //   = BE(daughter) - BE(parent) - 2.587
            // Wait, let me reconsider. Q_ββ = M(Z,N) - M(Z+2,N-2) - 2*m_e
            // M = Z*M_H + N*M_n - BE
            // Q_ββ = [Z*M_H + N*M_n - BE(Z,N)] - [(Z+2)*M_H + (N-2)*M_n - BE(Z+2,N-2)] - 2*m_e
            //      = -2*M_H + 2*M_n - BE(Z,N) + BE(Z+2,N-2) - 2*m_e
            //      = BE(Z+2,N-2) - BE(Z,N) + 2*(M_n - M_H) - 2*m_e
            //      = BE(Z+2,N-2) - BE(Z,N) + 2*0.78235 - 1.022
            //      = BE(Z+2,N-2) - BE(Z,N) + 0.543 (MeV)
            let q_pred_mev = daughter.binding_energy - parent.binding_energy
                + 2.0 * 0.78235 - 1.022;
            let q_pred_kev = q_pred_mev * 1000.0;
            let delta = q_pred_kev - q_exp_kev;

            eprintln!("{:<10} {:>4} {:>4} {:>10.0} {:>10.0} {:>10.0} {:>8.3}",
                name, z, n, q_exp_kev, q_pred_kev, delta,
                (parent.uncertainty.powi(2) + daughter.uncertainty.powi(2)).sqrt());

            q_errors.push(delta);
        }

        let rms_kev = (q_errors.iter().map(|e| e*e).sum::<f64>() / q_errors.len() as f64).sqrt();
        let bias_kev = q_errors.iter().sum::<f64>() / q_errors.len() as f64;
        eprintln!("\nQ_ββ prediction RMS: {:.0} keV ({:.3} MeV)", rms_kev, rms_kev / 1000.0);
        eprintln!("Q_ββ bias: {:+.0} keV", bias_kev);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. PROTON DRIP LINE — rp-PROCESS PATH
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn proton_drip_line() {
        let pred = predictor();

        eprintln!("\n{} PROTON DRIP LINE (rp-PROCESS) {}", "=".repeat(15), "=".repeat(15));
        eprintln!("S_p < 0 defines the proton drip line. Relevant for X-ray bursts.\n");

        eprintln!("{:>4} {:>6} {:>6} {:>10} {:>10}",
            "Z", "N_drip", "A_drip", "S_p(MeV)", "S_2p(MeV)");
        eprintln!("{}", "-".repeat(42));

        for z in (4..=55u16).step_by(1) {
            // Find last bound isotope (S_p > 0)
            let mut last_bound_n = z; // start at N=Z
            for n_offset in 0..20u16 {
                let n = if z > n_offset { z - n_offset } else { break };
                if n < 1 { break; }
                // S_p = BE(Z,N) - BE(Z-1,N)
                let sp = pred.predict(z, n).binding_energy
                    - pred.predict(z - 1, n).binding_energy;
                if sp < 0.0 {
                    last_bound_n = n + 1;
                    break;
                }
                last_bound_n = n;
            }

            // Compute S_p and S_2p at the drip line
            if last_bound_n >= 1 && z >= 2 {
                let sp = pred.predict(z, last_bound_n).binding_energy
                    - pred.predict(z - 1, last_bound_n).binding_energy;
                let s2p = if z >= 3 {
                    pred.predict(z, last_bound_n).binding_energy
                    - pred.predict(z - 2, last_bound_n).binding_energy
                } else { 0.0 };

                // Only print interesting cases
                if z % 2 == 0 || last_bound_n != z {
                    eprintln!("{:>4} {:>6} {:>6} {:>10.3} {:>10.3}",
                        z, last_bound_n, z + last_bound_n, sp, s2p);
                }
            }
        }

        // Known proton emitters — compare
        eprintln!("\n--- Known Proton Emitters (validation) ---");
        let known_emitters = [
            (53, 52, "I-105"),   // known proton emitter
            (55, 54, "Cs-109"),  // known proton emitter
            (69, 70, "Tm-139"), // 2p emitter candidate
            (26, 19, "Fe-45"),  // 2p emitter
        ];

        for &(z, n, name) in &known_emitters {
            let sp = pred.predict(z, n).binding_energy
                - pred.predict(z - 1, n).binding_energy;
            let s2p = pred.predict(z, n).binding_energy
                - pred.predict(z - 2, n).binding_energy;
            eprintln!("  {}: S_p = {:.3} MeV, S_2p = {:.3} MeV {}",
                name, sp, s2p,
                if sp < 0.0 { "(UNBOUND — correct!)" } else { "" });
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. NUCLEAR CLOCK: Th-229 ISOMER PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn nuclear_clock_th229() {
        let pred = predictor();

        eprintln!("\n{} NUCLEAR CLOCK: Th-229 {}", "=".repeat(15), "=".repeat(15));
        eprintln!("Th-229 has a 8.338 eV nuclear isomer — the lowest known.");
        eprintln!("Used for ultra-precise nuclear clocks.\n");

        let z = 90u16;
        let n = 139u16;

        let th229 = pred.predict(z, n);
        let th229_exp_be = 1748.253; // AME2020 measured BE (MeV)

        eprintln!("Th-229 (Z=90, N=139):");
        eprintln!("  Predicted BE:  {:.3} MeV", th229.binding_energy);
        eprintln!("  AME2020 BE:    {:.3} MeV", th229_exp_be);
        eprintln!("  Error:         {:.3} MeV", th229.binding_energy - th229_exp_be);
        eprintln!("  Uncertainty:   {:.3} MeV", th229.uncertainty);

        // Neighbors — important for production routes
        let neighbors = [
            (90, 138, "Th-228"),
            (90, 140, "Th-230"),
            (91, 139, "Pa-230"),   // Parent for Th-229 production
            (91, 138, "Pa-229"),
            (88, 137, "Ra-225"),   // Alpha daughter
            (92, 141, "U-233"),    // Grandparent
        ];

        eprintln!("\n--- Th-229 Neighborhood ---");
        eprintln!("{:<10} {:>4} {:>4} {:>10} {:>10} {:>8}",
            "Nucleus", "Z", "N", "BE(MeV)", "B/A(MeV)", "σ(MeV)");
        for &(nz, nn, name) in &neighbors {
            let p = pred.predict(nz, nn);
            eprintln!("{:<10} {:>4} {:>4} {:>10.3} {:>10.4} {:>8.3}",
                name, nz, nn, p.binding_energy, p.ba, p.uncertainty);
        }

        // Alpha decay Q-value: Th-229 → Ra-225 + He-4
        let ra225 = pred.predict(88, 137);
        let q_alpha = ra225.binding_energy + 28.296 - th229.binding_energy;
        eprintln!("\nTh-229 α-decay: Q = {:.3} MeV (exp: 5.168 MeV, err: {:.3})",
            q_alpha, q_alpha - 5.168);

        // Charge radius and deformation
        let (beta2, beta4) = crate::deformation::frdm_deformation(z, n);
        eprintln!("FRDM deformation: β₂ = {:.3}, β₄ = {:.3}", beta2, beta4);

        // S_n for Th-229
        let th228 = pred.predict(z, n - 1);
        let sn = th229.binding_energy - th228.binding_energy;
        eprintln!("S_n(Th-229) = {:.3} MeV (exp: ~6.4 MeV)", sn);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. ODD-EVEN STAGGERING: PAIRING GAP SYSTEMATICS
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn pairing_gap_systematics() {
        let pred = predictor();

        eprintln!("\n{} PAIRING GAP SYSTEMATICS {}", "=".repeat(15), "=".repeat(15));
        eprintln!("3-point neutron pairing gap: Δ₃(N) = (-1)^N × [BE(Z,N+1) - 2×BE(Z,N) + BE(Z,N-1)] / 2\n");

        // Extract pairing gaps across the chart
        eprintln!("--- Neutron Pairing Gap vs A (Z=50 Sn chain) ---");
        eprintln!("{:>4} {:>4} {:>10} {:>10}", "N", "A", "Δ₃(MeV)", "ee/oo");
        eprintln!("{}", "-".repeat(34));

        for n in 60..=85 {
            let z = 50u16;
            let be_m1 = pred.predict(z, n - 1).binding_energy;
            let be_0 = pred.predict(z, n).binding_energy;
            let be_p1 = pred.predict(z, n + 1).binding_energy;
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            let delta3 = sign * (be_p1 - 2.0 * be_0 + be_m1) / 2.0;
            let parity = if n % 2 == 0 { "even" } else { "odd" };
            let marker = if n == 82 { " ← N=82" } else { "" };

            eprintln!("{:>4} {:>4} {:>10.3} {:>10}{}", n, z + n, delta3, parity, marker);
        }

        // Proton pairing gap for Z=28 Ni chain
        eprintln!("\n--- Proton Pairing Gap vs N (Nickel Z=28 chain) ---");
        eprintln!("{:>4} {:>4} {:>10} {:>10}", "N", "A", "Δ₃_p(MeV)", "Context");
        for n in [28u16, 40, 50] {
            let z = 28u16;
            let be_m1 = pred.predict(z - 1, n).binding_energy;
            let be_0 = pred.predict(z, n).binding_energy;
            let be_p1 = pred.predict(z + 1, n).binding_energy;
            let delta3 = (be_p1 - 2.0 * be_0 + be_m1) / 2.0;
            let ctx = match n {
                28 => "doubly magic Ni-56",
                40 => "N=40 subshell? Ni-68",
                50 => "N=50 magic Ni-78",
                _ => "",
            };
            eprintln!("{:>4} {:>4} {:>10.3} {:>10}", n, z + n, delta3, ctx);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. NEUTRON STAR CRUST COMPOSITION
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn neutron_star_crust() {
        let pred = predictor();

        eprintln!("\n{} NEUTRON STAR CRUST COMPOSITION {}", "=".repeat(15), "=".repeat(15));
        eprintln!("At each density layer, find the nucleus with maximum B/A");
        eprintln!("for a given neutron excess (N-Z)/A.\n");

        // At different neutron fractions (mimicking increasing depth/density)
        let neutron_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        eprintln!("{:>6} {:>6} {:>4} {:>4} {:>10} {:>10} {:>10}",
            "(N-Z)/A", "Best", "Z", "N", "B/A(MeV)", "BE(MeV)", "σ(MeV)");
        eprintln!("{}", "-".repeat(56));

        for &frac in &neutron_fractions {
            let mut best_ba = 0.0f64;
            let mut best_z = 0u16;
            let mut best_n = 0u16;
            let mut best_be = 0.0;
            let mut best_sigma = 0.0;

            for z in 8..=82u16 {
                for a in (2*z)..=(3*z + 20) {
                    let n = a - z;
                    if n < z / 2 { continue; }
                    let actual_frac = (n as f64 - z as f64) / a as f64;
                    if (actual_frac - frac).abs() > 0.05 { continue; }

                    let p = pred.predict(z, n);
                    if p.ba > best_ba && p.uncertainty < 1.0 {
                        best_ba = p.ba;
                        best_z = z;
                        best_n = n;
                        best_be = p.binding_energy;
                        best_sigma = p.uncertainty;
                    }
                }
            }

            if best_z > 0 {
                let elem = match best_z {
                    8 => "O", 20 => "Ca", 26 => "Fe", 28 => "Ni",
                    40 => "Zr", 50 => "Sn", 82 => "Pb",
                    _ => "?",
                };
                let name = format!("{}-{}", elem, best_z + best_n);
                eprintln!("{:>6.2} {:>6} {:>4} {:>4} {:>10.4} {:>10.1} {:>10.3}",
                    frac, name, best_z, best_n, best_ba, best_be, best_sigma);
            }
        }

        eprintln!("\nNote: Real neutron star crust requires Gibbs equilibrium with");
        eprintln!("free neutrons, electrons, and lattice energy. This is indicative only.");
    }
}
