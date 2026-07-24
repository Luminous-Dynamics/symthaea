// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Theory Consciousness Comparison.
//!
//! Maps 10 consciousness theories to computable molecular quantities,
//! runs them on a set of molecules, and analyzes which theories agree,
//! which are independent, and whether consciousness is uni- or multi-dimensional.
//!
//! Follows the softmin-bottleneck + weighted-amplifier architecture from
//! Symthaea's ConsciousnessEquationV2.
//!
//! ## The 10 Theories
//!
//! 1. IIT (Tononi) — bipartition entanglement entropy
//! 2. GWT (Baars) — orbital delocalization across atoms
//! 3. FEP (Friston) — Helmholtz free energy (thermodynamic order)
//! 4. HOT (Rosenthal) — excitation gap (meta-representational access)
//! 5. Orch-OR (Penrose) — quantum coherence time
//! 6. Autopoiesis (Maturana) — electron correlation fraction
//! 7. Complexity (Crutchfield) — mutual information network density
//! 8. Info Geometry (Amari) — KL divergence from product state
//! 9. Dissipative (Prigogine) — entropy production proxy
//! 10. Quantum Darwinism (Zurek) — quantum-classical transition parameter

use crate::basis::BasisSetProvider;
use crate::basis::sto3g::Sto3g;
use crate::consciousness::{build_atom_basis_ranges, compute_orbital_phi};
use crate::integrals::eri::compute_eri_tensor;
use crate::molecule::Molecule;
use crate::post_hf::mp2::mp2_correlation_energy;
use crate::quantum_info::{bipartition_entanglement, orbital_mutual_information};
use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};
use crate::stat_mech;
use crate::time_dependent::cis_excitations;

/// Number of consciousness theories.
pub const N_THEORIES: usize = 10;

/// Theory names for display.
pub const THEORY_NAMES: [&str; N_THEORIES] = [
    "IIT",         // 0: Integration
    "GWT",         // 1: Global Workspace
    "FEP",         // 2: Free Energy Principle
    "HOT",         // 3: Higher-Order Thought
    "Orch-OR",     // 4: Orchestrated Objective Reduction
    "Autopoiesis", // 5: Self-Organization
    "Complexity",  // 6: Mutual Information
    "InfoGeom",    // 7: Information Geometry
    "Dissipative", // 8: Dissipative Structures
    "QDarwinism",  // 9: Quantum Darwinism
];

/// Short theory descriptions.
pub const THEORY_DESCRIPTIONS: [&str; N_THEORIES] = [
    "Entanglement across subsystems",
    "Orbital delocalization (global access)",
    "Thermodynamic free energy (order)",
    "Excitation gap (meta-representation)",
    "Quantum coherence duration",
    "Electron correlation (self-organization)",
    "Orbital mutual information density",
    "KL divergence from product state",
    "Entropy production (dissipative order)",
    "Quantum-classical transition ease",
];

/// Scores for one molecule across all 10 theories.
#[derive(Debug, Clone)]
pub struct TheoryScores {
    pub molecule_name: String,
    pub n_atoms: usize,
    pub n_electrons: usize,
    pub hf_energy: f64,
    pub raw: [f64; N_THEORIES],
    pub normalized: [f64; N_THEORIES],
    pub limiting_theory: usize,
    pub composite_score: f64,
}

/// Full survey results.
#[derive(Debug, Clone)]
pub struct MultiTheorySurvey {
    pub molecules: Vec<TheoryScores>,
    pub temperature: f64,
    pub correlations: [[f64; N_THEORIES]; N_THEORIES],
    pub clusters: Vec<Vec<usize>>,
}

/// Compute all 10 theory-grounded consciousness metrics for a molecule.
pub fn compute_theory_scores(molecule: &Molecule, name: &str, temperature: f64) -> TheoryScores {
    let basis = Sto3g::build(molecule);
    let n = basis.n_basis();
    let rhf = restricted_hartree_fock(molecule, &basis, &RhfConfig::default());
    let n_occ = rhf.n_occupied;
    let n_mo = rhf.n_independent;
    let eps = &rhf.orbital_energies;

    // Precompute shared data
    let ranges = build_atom_basis_ranges(&molecule.atoms, &basis.functions);
    let phi_meas = compute_orbital_phi(&rhf, &ranges);
    let (eri, _, _) = compute_eri_tensor(&basis.functions);
    let mp2 = mp2_correlation_energy(&rhf, &eri);

    let n_atoms = molecule.n_atoms();
    let n_electrons = molecule.n_electrons();

    // === Theory 0: IIT — Bipartition entanglement ===
    let iit_raw = if n_atoms >= 2 {
        let mut max_ent = 0.0_f64;
        for cut in 1..n_atoms {
            let part_a: Vec<usize> = (0..ranges[cut].0).collect();
            let s = bipartition_entanglement(&rhf, &part_a);
            if s > max_ent {
                max_ent = s;
            }
        }
        max_ent
    } else {
        0.0
    };
    let iit_norm = if n_atoms > 1 {
        (iit_raw / (n_atoms as f64).ln()).min(1.0)
    } else {
        0.0
    };

    // === Theory 1: GWT — Orbital delocalization ===
    let gwt_raw = phi_meas.molecular_phi;
    let gwt_norm = gwt_raw; // Already [0, 1)

    // === Theory 2: FEP — Free energy (thermodynamic order) ===
    let energies: Vec<f64> = eps.iter().cloned().collect();
    let degens = vec![2.0; energies.len()];
    let f_helmholtz = stat_mech::helmholtz_free_energy(&energies, &degens, temperature);
    let fep_raw = f_helmholtz;
    // Normalize: ratio of free energy to total energy (more negative = more ordered)
    let fep_norm = if rhf.total_energy.abs() > 1e-10 {
        (f_helmholtz / rhf.total_energy).clamp(0.0, 1.0)
    } else {
        0.5
    };

    // === Theory 3: HOT — Excitation gap (meta-representation capacity) ===
    let exc = cis_excitations(&rhf, &eri, 1);
    let first_exc_ev = if exc.is_empty() {
        100.0
    } else {
        exc[0].energy_ev
    };
    let hot_raw = first_exc_ev;
    // Higher score for SMALLER gap (easier meta-representation). Phase Q5
    // (2026-07-17): switched from the old rational form `1/(1+de/10)` to a
    // logistic -- that old form assumed `de` (the excitation gap) is always
    // positive, an assumption the previous Koopmans-only implementation
    // enforced by construction (it filtered `de > 0.0` before returning any
    // excitation at all). Real CIS has no such filter, correctly: a
    // negative CIS eigenvalue is a real, physically meaningful signal (RHF
    // reference instability -- found for N2/STO-3G here, a system Q0's own
    // audit already flagged as having an unresolved HF discrepancy). The
    // old rational form is unbounded and sign-flips for `de < -10`,
    // violating this module's own declared [0,1] invariant
    // (`test_theory_scores_bounded`). The logistic is bounded in (0,1) for
    // any real input, monotonic decreasing (preserving the intended
    // "smaller gap = more meta-representation capacity" ordering), and
    // saturates toward 1.0 for a large negative (unstable) gap -- a
    // defensible qualitative reading, not a claim of new precision on an
    // already-exploratory, disclosed-as-unvalidated proxy metric.
    let hot_norm = 1.0 / (1.0 + (first_exc_ev / 10.0).exp());

    // === Theory 4: Orch-OR — Coherence time ===
    let homo_lumo_gap = phi_meas.homo_lumo_gap;
    let coherence_time = if homo_lumo_gap > 1e-10 {
        1.0 / homo_lumo_gap
    } else {
        1000.0
    };
    let orch_raw = coherence_time;
    let orch_norm = (coherence_time / 100.0).tanh();

    // === Theory 5: Autopoiesis — Electron correlation fraction ===
    let corr_fraction = if rhf.total_energy.abs() > 1e-10 {
        mp2.correlation_energy.abs() / rhf.total_energy.abs()
    } else {
        0.0
    };
    let auto_raw = corr_fraction;
    // Scale: typical correlation fractions are 0.01-1% for small molecules
    let auto_norm = (corr_fraction * 100.0).min(1.0);

    // === Theory 6: Complexity — MI network density ===
    let mi_matrix = orbital_mutual_information(&rhf);
    let n_pairs = if n_mo > 1 { n_mo * (n_mo - 1) / 2 } else { 1 };
    let mi_total: f64 = mi_matrix.iter().flat_map(|row| row.iter()).sum::<f64>() / 2.0;
    let mi_density = mi_total / n_pairs as f64;
    let complexity_raw = mi_density;
    let complexity_norm = mi_density.min(1.0);

    // === Theory 7: Info Geometry — KL divergence from product state ===
    // Build joint distribution from atom populations of occupied orbitals
    let phi_ig = if n_atoms >= 2 && n_occ > 0 {
        // Use atom populations from orbital phi measurement
        // Build a joint distribution over (atom_of_electron, orbital)
        let n_a = n_atoms.min(4); // Limit for computational tractability
        let n_b = n_occ.min(4);
        let mut joint = vec![0.0; n_a * n_b];
        let total_weight = (n_a * n_b) as f64;

        for i in 0..n_b {
            for a in 0..n_a {
                if i < phi_meas.atom_populations.len() && a < phi_meas.atom_populations[i].len() {
                    joint[a * n_b + i] = phi_meas.atom_populations[i][a] / n_b as f64;
                } else {
                    joint[a * n_b + i] = 1.0 / total_weight;
                }
            }
        }
        // Normalize
        let sum: f64 = joint.iter().sum();
        if sum > 1e-15 {
            for j in &mut joint {
                *j /= sum;
            }
        }

        compute_phi_ig(&joint, n_a, n_b)
    } else {
        0.0
    };
    let ig_raw = phi_ig;
    let ig_norm = if n_atoms > 1 {
        (phi_ig / (n_atoms as f64).ln()).min(1.0)
    } else {
        0.0
    };

    // === Theory 8: Dissipative — Entropy production proxy ===
    let thermal_s = stat_mech::entropy(&energies, &degens, temperature);
    let dissipative_raw = temperature * thermal_s;
    let kt = stat_mech::K_BOLTZMANN_HARTREE * temperature;
    let dissipative_norm = if kt > 1e-20 {
        (dissipative_raw / kt).min(1.0).max(0.0)
    } else {
        0.0
    };

    // === Theory 9: Quantum Darwinism — Quantum-classical parameter ===
    // χ = n_basis × coupling² × n_env / gap²
    // Small χ = quantum (more "conscious"), large χ = classical
    let n_env = 1000; // Assume environment of ~1000 modes
    let coupling = 0.01; // Typical system-environment coupling
    let chi =
        n as f64 * coupling * coupling * n_env as f64 / (homo_lumo_gap * homo_lumo_gap + 1e-10);
    let qd_raw = chi;
    // Invert: quantum regime (small χ) gets high score
    let qd_norm = 1.0 / (1.0 + chi);

    // === Assemble raw and normalized scores ===
    let raw = [
        iit_raw,
        gwt_raw,
        fep_raw,
        hot_raw,
        orch_raw,
        auto_raw,
        complexity_raw,
        ig_raw,
        dissipative_raw,
        qd_raw,
    ];
    let normalized = [
        iit_norm,
        gwt_norm,
        fep_norm,
        hot_norm,
        orch_norm,
        auto_norm,
        complexity_norm,
        ig_norm,
        dissipative_norm,
        qd_norm,
    ];

    // === Composite: softmin(necessary) × mean(amplifiers) ===
    // Necessary conditions: IIT(0), GWT(1), Complexity(6)
    let necessary = [normalized[0], normalized[1], normalized[6]];
    let softmin_tau = 0.1;
    let softmin_val = softmin(&necessary, softmin_tau);

    // Amplifiers: FEP(2), HOT(3), Orch-OR(4), Autopoiesis(5), InfoGeom(7), Dissipative(8), QDarwinism(9)
    let amplifiers = [
        normalized[2],
        normalized[3],
        normalized[4],
        normalized[5],
        normalized[7],
        normalized[8],
        normalized[9],
    ];
    let amp_mean: f64 = amplifiers.iter().sum::<f64>() / amplifiers.len() as f64;

    let composite = softmin_val * (0.5 + 0.5 * amp_mean);

    // Limiting theory: which necessary condition is lowest?
    let limiting = if necessary[0] <= necessary[1] && necessary[0] <= necessary[2] {
        0 // IIT
    } else if necessary[1] <= necessary[2] {
        1 // GWT
    } else {
        6 // Complexity
    };

    TheoryScores {
        molecule_name: name.to_string(),
        n_atoms,
        n_electrons,
        hf_energy: rhf.total_energy,
        raw,
        normalized,
        limiting_theory: limiting,
        composite_score: composite,
    }
}

/// Run the full multi-theory survey.
pub fn multi_theory_survey(molecules: &[(&str, Molecule)], temperature: f64) -> MultiTheorySurvey {
    let results: Vec<TheoryScores> = molecules
        .iter()
        .map(|(name, mol)| compute_theory_scores(mol, name, temperature))
        .collect();

    let correlations = theory_correlation_matrix(&results);
    let clusters = find_theory_clusters(&correlations, 0.7);

    MultiTheorySurvey {
        molecules: results,
        temperature,
        correlations,
        clusters,
    }
}

/// 10×10 Pearson correlation matrix between theories across molecules.
pub fn theory_correlation_matrix(results: &[TheoryScores]) -> [[f64; N_THEORIES]; N_THEORIES] {
    let mut corr = [[0.0; N_THEORIES]; N_THEORIES];
    for i in 0..N_THEORIES {
        for j in i..N_THEORIES {
            let xi: Vec<f64> = results.iter().map(|r| r.normalized[i]).collect();
            let xj: Vec<f64> = results.iter().map(|r| r.normalized[j]).collect();
            let r = pearson(&xi, &xj);
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }

    corr
}

/// Find clusters of theories that agree (r > threshold).
pub fn find_theory_clusters(
    correlations: &[[f64; N_THEORIES]; N_THEORIES],
    threshold: f64,
) -> Vec<Vec<usize>> {
    let mut visited = [false; N_THEORIES];
    let mut clusters = Vec::new();

    for i in 0..N_THEORIES {
        if visited[i] {
            continue;
        }
        let mut cluster = vec![i];
        visited[i] = true;

        for j in (i + 1)..N_THEORIES {
            if !visited[j] && correlations[i][j] > threshold {
                cluster.push(j);
                visited[j] = true;
            }
        }

        if cluster.len() > 1 {
            clusters.push(cluster);
        }
    }

    // Collect independents
    let mut independents = Vec::new();
    for i in 0..N_THEORIES {
        if !visited[i] {
            independents.push(i);
        }
    }
    if !independents.is_empty() {
        clusters.push(independents); // Last cluster = independents
    }

    clusters
}

/// Softmin: differentiable minimum.
/// softmin(x; τ) = Σ x_i exp(-x_i/τ) / Σ exp(-x_i/τ)
fn softmin(values: &[f64], tau: f64) -> f64 {
    let weights: Vec<f64> = values.iter().map(|&x| (-x / tau).exp()).collect();
    let w_sum: f64 = weights.iter().sum();
    if w_sum < 1e-30 {
        return values.iter().cloned().fold(f64::INFINITY, f64::min);
    }
    values
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| x * w)
        .sum::<f64>()
        / w_sum
}

/// Pearson correlation coefficient. Returns 0.0 for constant inputs.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-15 || vy < 1e-15 {
        return 0.0;
    }
    let r = cov / (vx * vy).sqrt();
    if r.is_nan() { 0.0 } else { r.clamp(-1.0, 1.0) }
}

/// Compute information-geometric Phi (KL divergence from product state).
fn compute_phi_ig(joint: &[f64], n_a: usize, n_b: usize) -> f64 {
    let mut marginal_a = vec![0.0; n_a];
    let mut marginal_b = vec![0.0; n_b];
    for a in 0..n_a {
        for b in 0..n_b {
            let p = joint[a * n_b + b];
            marginal_a[a] += p;
            marginal_b[b] += p;
        }
    }
    let mut phi = 0.0;
    for a in 0..n_a {
        for b in 0..n_b {
            let p_joint = joint[a * n_b + b];
            let p_product = marginal_a[a] * marginal_b[b];
            if p_joint > 1e-15 && p_product > 1e-15 {
                phi += p_joint * (p_joint / p_product).ln();
            }
        }
    }
    phi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Atom;

    #[test]
    fn test_theory_scores_bounded() {
        let mol = Molecule::water();
        let scores = compute_theory_scores(&mol, "H2O", 300.0);
        for i in 0..N_THEORIES {
            assert!(
                scores.normalized[i] >= 0.0 && scores.normalized[i] <= 1.0,
                "Theory {} ({}) normalized = {:.4}, should be in [0,1]",
                i,
                THEORY_NAMES[i],
                scores.normalized[i]
            );
        }
    }

    #[test]
    fn test_composite_bounded() {
        let mol = Molecule::h2();
        let scores = compute_theory_scores(&mol, "H2", 300.0);
        assert!(
            scores.composite_score >= 0.0 && scores.composite_score <= 1.0,
            "Composite = {:.4}",
            scores.composite_score
        );
    }

    #[test]
    fn test_correlation_matrix_symmetric() {
        let molecules = vec![
            ("H2", Molecule::h2()),
            ("H2O", Molecule::water()),
            ("HeH+", Molecule::heh_plus()),
        ];
        let results: Vec<TheoryScores> = molecules
            .iter()
            .map(|(name, mol)| compute_theory_scores(mol, name, 300.0))
            .collect();
        let corr = theory_correlation_matrix(&results);

        for i in 0..N_THEORIES {
            // Diagonal is 1.0 for non-constant theories, 0.0 for constant
            assert!(
                (corr[i][i] - 1.0).abs() < 1e-8 || corr[i][i].abs() < 1e-8,
                "Diagonal[{}] = {:.6}, should be 1.0 or 0.0",
                i,
                corr[i][i]
            );
            for j in 0..N_THEORIES {
                assert!(
                    (corr[i][j] - corr[j][i]).abs() < 1e-8,
                    "Correlation matrix not symmetric at ({},{}): {} vs {}",
                    i,
                    j,
                    corr[i][j],
                    corr[j][i]
                );
            }
        }
    }

    #[test]
    fn test_heh_plus_low_composite() {
        let heh = compute_theory_scores(&Molecule::heh_plus(), "HeH+", 300.0);
        let water = compute_theory_scores(&Molecule::water(), "H2O", 300.0);
        assert!(
            heh.composite_score < water.composite_score,
            "HeH+ ({:.4}) should score lower than H2O ({:.4})",
            heh.composite_score,
            water.composite_score
        );
    }

    #[test]
    fn test_survey_runs() {
        let molecules = vec![("H2", Molecule::h2()), ("H2O", Molecule::water())];
        let survey = multi_theory_survey(&molecules, 300.0);
        assert_eq!(survey.molecules.len(), 2);
        assert!(!survey.clusters.is_empty() || survey.molecules.len() < 3);
    }
}
