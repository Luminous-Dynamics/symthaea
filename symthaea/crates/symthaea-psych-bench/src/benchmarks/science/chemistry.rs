// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chemistry Benchmark.
//!
//! Tests Symthaea's chemical reasoning against known values from NIST-JANAF
//! and standard physical chemistry references. Scope: stoichiometry, Hess's Law,
//! Gibbs free energy, ICE tables, and Arrhenius kinetics.
//!
//! ## Test Conditions
//!
//! 1. **Molar mass** — H₂O (18.015), CO₂ (44.009), CH₄ (16.043), NaCl (58.44) g/mol
//! 2. **Hess's Law** — ΔH°_rxn for methane combustion (−890.3 kJ/mol)
//! 3. **Gibbs Free Energy** — CH₄ combustion spontaneity, ATP hydrolysis (ΔG°≈−30.5 kJ)
//! 4. **ICE Table** — N₂O₄⇌2NO₂ equilibrium concentrations (K_eq from ΔG°)
//! 5. **Arrhenius Kinetics** — k ratio between 300K and 500K for Ea=80 kJ/mol
//!
//! Human baselines derived from general chemistry course performance:
//! - Molar mass: ~0.95 (routine calculation)
//! - Hess's Law: ~0.72 (requires stoichiometric care)
//! - Gibbs/spontaneity: ~0.70 (conceptual + numerical)
//! - Equilibrium ICE: ~0.60 (most students struggle)
//! - Arrhenius kinetics: ~0.65 (Ea handling, natural log)
//!
//! References: NIST-JANAF (Chase 1998), Atkins Physical Chemistry (11th ed.),
//! Zumdahl Chemistry (10th ed.).

use std::time::Instant;

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};

use symthaea_core::hdc::chemistry::{
    all_elements, arrhenius_k, gibbs_free_energy, hess_law, is_spontaneous, keq_from_gibbs,
    molar_mass, solve_ice, thermochemical_database,
};

/// Chemistry Assessment benchmark.
pub struct ChemistryBenchmark;

struct TrialResult {
    molar_mass_score: f64,
    hess_law_score: f64,
    gibbs_score: f64,
    ice_score: f64,
    arrhenius_score: f64,
}

// ── Scoring helpers ──────────────────────────────────────────────────────────

/// Score how close a computed value is to a reference (within tol%).
fn relative_score(computed: f64, reference: f64, tol_frac: f64) -> f64 {
    if reference.abs() < 1e-12 {
        if computed.abs() < 1e-6 {
            1.0
        } else {
            0.0
        }
    } else {
        let err = (computed - reference).abs() / reference.abs();
        // Full credit within tol_frac; linear decay out to 3×tol_frac; 0 beyond.
        if err <= tol_frac {
            1.0
        } else if err <= 3.0 * tol_frac {
            1.0 - (err - tol_frac) / (2.0 * tol_frac)
        } else {
            0.0
        }
    }
}

// ── Benchmark logic ──────────────────────────────────────────────────────────

fn run_trial() -> TrialResult {
    let elems = all_elements();
    let db = thermochemical_database();

    // ── 1. Molar mass ────────────────────────────────────────────────────────
    // H₂O: 2×1.008 + 15.999 = 18.015 g/mol
    // CO₂: 12.011 + 2×15.999 = 44.009 g/mol
    // CH₄: 12.011 + 4×1.008  = 16.043 g/mol
    // NaCl: 22.990 + 35.450  = 58.440 g/mol
    let mm_cases = [
        ("H2O", 18.015),
        ("CO2", 44.009),
        ("CH4", 16.043),
        ("NaCl", 58.440),
    ];
    let mm_score = mm_cases
        .iter()
        .map(|(formula, expected)| {
            match molar_mass(formula, &elems) {
                Ok(mm) => relative_score(mm, *expected, 0.005), // 0.5% tolerance
                Err(_) => 0.0,
            }
        })
        .sum::<f64>()
        / mm_cases.len() as f64;

    // ── 2. Hess's Law — methane combustion ──────────────────────────────────
    // CH₄(g) + 2O₂(g) → CO₂(g) + 2H₂O(l)
    // ΔH° = ΔHf°(CO₂) + 2·ΔHf°(H₂O,l) − ΔHf°(CH₄) − 2·ΔHf°(O₂)
    //     = −393.509 + 2×(−285.830) − (−74.870) − 0
    //     = −890.299 kJ/mol
    let rxn_combustion = [
        ("CO2(g)", 1.0_f64),
        ("H2O(l)", 2.0_f64),
        ("CH4(g)", -1.0_f64),
        ("O2(g)", -2.0_f64),
    ];
    let dh_combustion = hess_law(&rxn_combustion, &db);
    let hess_score = relative_score(dh_combustion, -890.3, 0.01); // 1% tolerance

    // ── 3. Gibbs Free Energy & spontaneity ──────────────────────────────────
    // CH₄ combustion at 298 K:
    // ΔS°_rxn = S°(CO₂) + 2·S°(H₂O,l) - S°(CH₄) - 2·S°(O₂)
    //         = 213.785 + 2×69.950 - 186.250 - 2×205.150
    //         = -242.865 J/mol·K → -0.242865 kJ/mol·K
    // ΔG° = ΔH° - TΔS° = -890.3 - 298×(-0.242865) = -890.3 + 72.37 = -817.93 kJ
    // Should be spontaneous (ΔG < 0)
    let ds_combustion_kj_k = (213.785 + 2.0 * 69.950 - 186.250 - 2.0 * 205.150) / 1000.0;
    let dg_combustion = gibbs_free_energy(dh_combustion, 298.15, ds_combustion_kj_k * 1000.0);
    // The function takes ΔS in J/mol·K
    let spontaneous_ok = if is_spontaneous(dg_combustion) {
        1.0
    } else {
        0.0
    };
    // Check ΔG is in the right ballpark (−800 to −900 kJ/mol)
    let dg_range_ok = if dg_combustion < -700.0 && dg_combustion > -1000.0 {
        1.0
    } else {
        0.0
    };

    // ATP hydrolysis: ATP + H₂O → ADP + Pi  ΔG°' ≈ −30.5 kJ/mol
    // Use approximate values: ΔH≈−20.9 kJ/mol, ΔS≈34 J/mol·K at 310K (body temp)
    // ΔG = -20.9 - 310 × 0.034 = -20.9 - 10.54 = -31.44 kJ/mol
    let dg_atp = gibbs_free_energy(-20.9, 310.0, 34.0);
    let atp_score = relative_score(dg_atp, -31.44, 0.15); // 15% tolerance — approximate values

    let gibbs_score = (spontaneous_ok + dg_range_ok + atp_score) / 3.0;

    // ── 4. ICE Table — N₂O₄ ⇌ 2NO₂ ─────────────────────────────────────────
    // ΔG° = 2×ΔGf°(NO₂) − ΔGf°(N₂O₄)
    //     = 2×51.310 − 97.890 = 4.73 kJ/mol
    // K_eq = exp(−ΔG°/RT) = exp(−4730/(8.314×298.15)) ≈ 0.149
    let dg_n2o4 = 2.0 * 51.310 - 97.890; // kJ/mol
    let keq = keq_from_gibbs(dg_n2o4, 298.15);

    // ICE: N₂O₄ starts at 1.0 M, NO₂ starts at 0.0 M
    // stoich: N₂O₄ coeff = -1, NO₂ coeff = +2
    let initial = vec![1.0, 0.0];
    let stoich = vec![-1_i32, 2_i32];
    let ice_score = match solve_ice(&initial, &stoich, keq) {
        Ok(table) => {
            // At equilibrium, K = [NO₂]²/[N₂O₄]
            if table.equilibrium.len() == 2 {
                let no2_eq = table.equilibrium[1];
                let n2o4_eq = table.equilibrium[0];
                if n2o4_eq > 1e-10 {
                    let k_check = (no2_eq * no2_eq) / n2o4_eq;
                    // Check K reconstructed from equilibrium concentrations
                    relative_score(k_check, keq, 0.05)
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    };

    // ── 5. Arrhenius kinetics ────────────────────────────────────────────────
    // For a reaction with Ea = 80,000 J/mol, A = 1e12 s⁻¹:
    // k(300K) = 1e12 × exp(−80000/(8.314×300))  ≈ 1e12 × exp(−32.07) ≈ 1.12e-2
    // k(500K) = 1e12 × exp(−80000/(8.314×500))  ≈ 1e12 × exp(−19.24) ≈ 4.47e3
    // k(500)/k(300) = exp(Ea/R × (1/T₁ − 1/T₂)) = exp(80000/8.314 × (1/300 − 1/500))
    //              = exp(9621.8 × (0.00333 − 0.002)) = exp(9621.8 × 0.00133) ≈ exp(12.80) ≈ 3.63e5
    let a_factor = 1e12_f64; // pre-exponential factor (s⁻¹)
    let ea = 80_000.0_f64; // J/mol
    let k_300 = arrhenius_k(a_factor, ea, 300.0);
    let k_500 = arrhenius_k(a_factor, ea, 500.0);
    let ratio = k_500 / k_300;
    let expected_ratio = (ea / 8.314 * (1.0 / 300.0 - 1.0 / 500.0)).exp();
    let arrhenius_score = relative_score(ratio, expected_ratio, 0.02); // 2% tolerance

    TrialResult {
        molar_mass_score: mm_score,
        hess_law_score: hess_score,
        gibbs_score,
        ice_score,
        arrhenius_score,
    }
}

// ── PsychBenchmark impl ──────────────────────────────────────────────────────

impl PsychBenchmark for ChemistryBenchmark {
    fn name(&self) -> &str {
        "Chemistry (Stoichiometry, Thermochemistry, Kinetics)"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = Instant::now();
        let n = config.trials_per_condition.max(1);

        let mut molar_mass_scores = Vec::with_capacity(n);
        let mut hess_law_scores = Vec::with_capacity(n);
        let mut gibbs_scores = Vec::with_capacity(n);
        let mut ice_scores = Vec::with_capacity(n);
        let mut arrhenius_scores = Vec::with_capacity(n);

        for _ in 0..n {
            let t = run_trial();
            molar_mass_scores.push(t.molar_mass_score);
            hess_law_scores.push(t.hess_law_score);
            gibbs_scores.push(t.gibbs_score);
            ice_scores.push(t.ice_score);
            arrhenius_scores.push(t.arrhenius_score);
        }

        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        result.insert(
            "molar_mass_accuracy",
            MetricValue::from_samples(&molar_mass_scores),
        );
        result.insert(
            "hess_law_accuracy",
            MetricValue::from_samples(&hess_law_scores),
        );
        result.insert(
            "gibbs_free_energy_accuracy",
            MetricValue::from_samples(&gibbs_scores),
        );
        result.insert(
            "ice_equilibrium_accuracy",
            MetricValue::from_samples(&ice_scores),
        );
        result.insert(
            "arrhenius_kinetics_accuracy",
            MetricValue::from_samples(&arrhenius_scores),
        );
        result.conditions = 5;
        result.trials_per_condition = n;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Physical Chemistry (Stoichiometry, Hess, Gibbs, ICE, Arrhenius)",
            citation:
                "NIST-JANAF (Chase 1998); Atkins Physical Chemistry 11e; Zumdahl Chemistry 10e",
            year: 1998,
            doi: Some("10.18434/T42S31"),
        })
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::chemistry::{all_elements, molar_mass, parse_formula};

    #[test]
    fn test_molar_mass_water() {
        let elems = all_elements();
        let mm = molar_mass("H2O", &elems).unwrap();
        assert!(
            (mm - 18.015).abs() < 0.1,
            "H₂O molar mass expected ~18.015, got {mm}"
        );
    }

    #[test]
    fn test_molar_mass_co2() {
        let elems = all_elements();
        let mm = molar_mass("CO2", &elems).unwrap();
        assert!(
            (mm - 44.009).abs() < 0.1,
            "CO₂ molar mass expected ~44.009, got {mm}"
        );
    }

    #[test]
    fn test_molar_mass_nacl() {
        let elems = all_elements();
        let mm = molar_mass("NaCl", &elems).unwrap();
        assert!(
            (mm - 58.44).abs() < 0.5,
            "NaCl molar mass expected ~58.44, got {mm}"
        );
    }

    #[test]
    fn test_parse_formula_h2so4() {
        let result = parse_formula("H2SO4");
        assert_eq!(*result.get("H").unwrap_or(&0), 2);
        assert_eq!(*result.get("S").unwrap_or(&0), 1);
        assert_eq!(*result.get("O").unwrap_or(&0), 4);
    }

    #[test]
    fn test_methane_combustion_hess() {
        let db = thermochemical_database();
        let rxn = [
            ("CO2(g)", 1.0_f64),
            ("H2O(l)", 2.0_f64),
            ("CH4(g)", -1.0_f64),
            ("O2(g)", -2.0_f64),
        ];
        let dh = hess_law(&rxn, &db);
        // NIST value: −890.3 kJ/mol
        assert!(
            (dh - (-890.3)).abs() < 10.0,
            "Methane combustion ΔH° expected ~−890.3 kJ/mol, got {dh}"
        );
    }

    #[test]
    fn test_methane_combustion_spontaneous() {
        let db = thermochemical_database();
        let rxn = [
            ("CO2(g)", 1.0_f64),
            ("H2O(l)", 2.0_f64),
            ("CH4(g)", -1.0_f64),
            ("O2(g)", -2.0_f64),
        ];
        let dh = hess_law(&rxn, &db);
        // ΔS° = -243 J/mol·K → -0.243 kJ/mol·K; function takes J/mol·K
        let ds_j_k = -242.865;
        let dg = gibbs_free_energy(dh, 298.15, ds_j_k);
        assert!(
            is_spontaneous(dg),
            "CH₄ combustion must be spontaneous at 298 K, ΔG = {dg}"
        );
        assert!(
            dg < -700.0,
            "CH₄ combustion ΔG° should be < −700 kJ, got {dg}"
        );
    }

    #[test]
    fn test_n2o4_keq_from_gibbs() {
        // ΔG° = 2×ΔGf°(NO₂) − ΔGf°(N₂O₄) = 2×51.31 − 97.89 = 4.73 kJ/mol
        let dg = 2.0 * 51.310 - 97.890;
        let keq = keq_from_gibbs(dg, 298.15);
        // K_eq should be < 1 (ΔG° > 0 → products not favored at standard state)
        assert!(
            keq > 0.01 && keq < 1.0,
            "N₂O₄ ⇌ 2NO₂ K_eq expected 0.01–1.0, got {keq}"
        );
    }

    #[test]
    fn test_ice_table_n2o4() {
        let dg = 2.0 * 51.310 - 97.890; // kJ/mol
        let keq = keq_from_gibbs(dg, 298.15);
        let initial = vec![1.0, 0.0];
        let stoich = vec![-1_i32, 2_i32];
        let table = solve_ice(&initial, &stoich, keq).expect("ICE should converge");
        // Verify K is reconstructed
        let no2 = table.equilibrium[1];
        let n2o4 = table.equilibrium[0];
        let k_check = (no2 * no2) / n2o4;
        assert!(
            (k_check - keq).abs() / keq < 0.05,
            "ICE K reconstruction error > 5%: expected {keq:.4}, got {k_check:.4}"
        );
    }

    #[test]
    fn test_arrhenius_ratio() {
        let ea = 80_000.0_f64;
        let a = 1e12_f64;
        let k1 = arrhenius_k(a, ea, 300.0);
        let k2 = arrhenius_k(a, ea, 500.0);
        assert!(
            k1 > 0.0 && k2 > k1,
            "Arrhenius k must increase with temperature"
        );
        let ratio = k2 / k1;
        // Expected ratio: exp(Ea/R × (1/300 - 1/500))
        let expected = (ea / 8.314 * (1.0 / 300.0 - 1.0 / 500.0)).exp();
        assert!(
            (ratio - expected).abs() / expected < 0.01,
            "Arrhenius ratio error > 1%: expected {expected:.2e}, got {ratio:.2e}"
        );
    }

    #[test]
    fn test_benchmark_runs() {
        let bench = ChemistryBenchmark;
        let config = BenchmarkConfig {
            trials_per_condition: 2,
            label: Some("test".to_string()),
            ..Default::default()
        };
        let result = bench.run(&config);
        assert!(
            !result.metrics.is_empty(),
            "Benchmark should produce metrics"
        );
        assert_eq!(result.conditions, 5);
    }

    #[test]
    fn test_relative_score_exact() {
        assert_eq!(relative_score(18.015, 18.015, 0.005), 1.0);
    }

    #[test]
    fn test_relative_score_within_tol() {
        // 0.3% error within 0.5% tol → full credit
        let score = relative_score(18.069, 18.015, 0.005);
        assert_eq!(score, 1.0, "0.3% error should give full credit at 0.5% tol");
    }

    #[test]
    fn test_relative_score_out_of_tol() {
        // 10% error at 0.5% tol → 0 credit
        let score = relative_score(19.8, 18.015, 0.005);
        assert_eq!(score, 0.0);
    }
}
