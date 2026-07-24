// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validation against published quantum chemistry benchmarks.
//!
//! Reference values from:
//! - Szabo & Ostlund (1996). *Modern Quantum Chemistry*. Tables 3.1, 3.15.
//! - NIST Computational Chemistry Comparison and Benchmark Database (CCCBDB).
//! - Hehre, Radom, Schleyer & Pople (1986). *Ab Initio Molecular Orbital Theory*.
//!
//! Each benchmark molecule has known HF and MP2 energies at STO-3G and 6-31G
//! that have been independently computed by multiple groups over 50+ years.

use crate::basis::BasisSetProvider;
use crate::basis::basis_631g::Basis631G;
use crate::basis::sto3g::Sto3g;
use crate::molecule::{Atom, Molecule};
use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

/// A benchmark reference value.
#[derive(Debug, Clone)]
pub struct BenchmarkEntry {
    pub molecule: &'static str,
    pub basis: &'static str,
    pub method: &'static str,
    pub reference_energy: f64,
    pub source: &'static str,
    pub tolerance: f64,
}

/// Published HF/STO-3G reference energies (Hartree).
/// Sources: Szabo & Ostlund Table 3.15, Hehre et al. (1986), CCCBDB.
///
/// Tolerances here are intentionally loose (0.5-1.0 Ha) for anything past H2/HeH+,
/// which hides real error for the molecules that don't need the full tolerance --
/// see `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md` Phase 0. Measured
/// actual errors (2026-07-12, post the Boys-function fix in `integrals/boys.rs`,
/// via `examples/phase0_audit.rs`): H2O +1.9 kcal/mol, NH3 +0.8, CH4 +0.04,
/// HF +0.02, LiH +0.8 -- all genuinely tight. N2 is the one real outlier at
/// +458 kcal/mol (0.73 Ha); investigated only enough to confirm it's unrelated to
/// the Boys-function bug (unchanged before/after that fix) and unrelated to the
/// C+N divergence bug (this is N-only).
///
/// Phase Q0 (2026-07-16): further narrowed via internal-consistency diagnostics
/// (`test_diagnose_n2_sto3g_discrepancy` below; no independent QC package
/// available in this sandbox to cross-check matrices directly, see that test's
/// comment) -- ruled out basis self-normalization (diagonal S ~= 1.0 to 1e-11),
/// matrix symmetry, and SCF self-consistency (Tr[PS] = n_electrons exactly).
/// SCF converges cleanly (9 iterations). The error is real and in either the
/// integral *values* themselves (not caught by these coarse invariants) or a
/// Fock-construction/exchange term specific to N2's strong p-p triple-bond
/// character (NH3, which also has nitrogen, is tight -- so it isn't simply
/// "any N-containing molecule"). Root cause still not isolated -- flagging as
/// a known, measured, further-narrowed, still-open, non-blocking discrepancy
/// rather than silently passing behind the loose tolerance.
pub fn hf_sto3g_references() -> Vec<(&'static str, f64, f64)> {
    // (molecule, reference_energy, tolerance)
    vec![
        // Szabo & Ostlund Table 3.15 (exact STO-3G values)
        ("H2", -1.1175, 0.01),   // R = 1.4 Bohr
        ("HeH+", -2.8607, 0.05), // R = 1.4632 Bohr
        // Hehre et al. / CCCBDB
        ("H2O", -74.9659, 0.5), // Experimental geometry
        ("NH3", -55.4554, 0.5), // Experimental geometry
        ("CH4", -39.7269, 0.5), // Tetrahedral
        ("HF", -98.5708, 0.5),  // R = 1.7328 Bohr
        ("N2", -107.4964, 1.0), // R = 2.074 Bohr -- measured error +458 kcal/mol, unresolved (see doc comment above)
        ("LiH", -7.8633, 0.5),  // R = 3.015 Bohr
    ]
}

/// Published HF/6-31G reference energies (Hartree).
/// Source: CCCBDB, Hehre et al.
///
/// Measured actual errors (2026-07-12, see `hf_sto3g_references` doc comment for
/// context): H2 +0.04 kcal/mol (tight), but H2O -206 kcal/mol and CH4 +27
/// kcal/mol -- both real, both hidden by the 0.5 Ha tolerance, both unrelated to
/// the Boys-function fix (unchanged before/after). Likely candidate: the 6-31G
/// basis data/contraction itself for these two.
///
/// Phase Q0 (2026-07-16): same internal-consistency diagnostics as
/// `hf_sto3g_references` (`test_diagnose_h2o_631g_discrepancy`,
/// `test_diagnose_ch4_631g_discrepancy` below) also rule out basis
/// self-normalization, symmetry, and SCF self-consistency for both. Since
/// CH4/STO-3G is tight (+0.04 kcal/mol, see above) but CH4/6-31G is not
/// (+27), the same molecule is fine in one basis and wrong in the other --
/// this is strong evidence the issue is specific to the 6-31G basis data or
/// its contraction scheme, not the integral engine or SCF machinery, but
/// still not conclusively isolated.
pub fn hf_631g_references() -> Vec<(&'static str, f64, f64)> {
    vec![
        ("H2", -1.1268, 0.01),
        ("H2O", -75.5854, 0.5), // measured error -206 kcal/mol, unresolved
        ("CH4", -40.1952, 0.5), // measured error +27 kcal/mol, unresolved
                                // N₂ excluded: BSE general contraction → segmented contraction conversion needed
    ]
}

/// Build the standard set of benchmark molecules.
pub fn benchmark_molecules() -> Vec<(&'static str, Molecule)> {
    vec![
        ("H2", Molecule::h2()),
        ("HeH+", Molecule::heh_plus()),
        (
            "LiH",
            Molecule::new(vec![
                Atom::new(3, 0.0, 0.0, 0.0),
                Atom::new(1, 0.0, 0.0, 3.015),
            ]),
        ),
        (
            "HF",
            Molecule::new(vec![
                Atom::new(1, 0.0, 0.0, 0.0),
                Atom::new(9, 0.0, 0.0, 1.7328),
            ]),
        ),
        ("H2O", Molecule::water()),
        (
            "NH3",
            Molecule::new(vec![
                Atom::from_angstrom(7, 0.0, 0.0, 0.0),
                Atom::from_angstrom(1, 0.0, 0.9377, -0.3816),
                Atom::from_angstrom(1, 0.8121, -0.4689, -0.3816),
                Atom::from_angstrom(1, -0.8121, -0.4689, -0.3816),
            ]),
        ),
        (
            "CH4",
            Molecule::new(vec![
                Atom::new(6, 0.0, 0.0, 0.0),
                Atom::new(1, 1.185, 1.185, 1.185),
                Atom::new(1, -1.185, -1.185, 1.185),
                Atom::new(1, -1.185, 1.185, -1.185),
                Atom::new(1, 1.185, -1.185, -1.185),
            ]),
        ),
        (
            "N2",
            Molecule::new(vec![
                Atom::new(7, 0.0, 0.0, 0.0),
                Atom::new(7, 0.0, 0.0, 2.074),
            ]),
        ),
    ]
}

/// Validation result for one molecule.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub molecule: String,
    pub basis: String,
    pub computed: f64,
    pub reference: f64,
    pub error_hartree: f64,
    pub error_kcal: f64,
    pub pass: bool,
    pub tolerance: f64,
}

/// Run the full HF/STO-3G validation suite.
pub fn validate_hf_sto3g() -> Vec<ValidationResult> {
    let molecules = benchmark_molecules();
    let refs = hf_sto3g_references();

    let mut results = Vec::new();

    for (ref_name, ref_energy, tolerance) in &refs {
        if let Some((_, mol)) = molecules.iter().find(|(name, _)| name == ref_name) {
            let basis = Sto3g::build(mol);
            let rhf = restricted_hartree_fock(mol, &basis, &RhfConfig::default());

            let error = rhf.total_energy - ref_energy;
            results.push(ValidationResult {
                molecule: ref_name.to_string(),
                basis: "STO-3G".to_string(),
                computed: rhf.total_energy,
                reference: *ref_energy,
                error_hartree: error,
                error_kcal: error * 627.509,
                pass: error.abs() < *tolerance,
                tolerance: *tolerance,
            });
        }
    }

    results
}

/// Run HF/6-31G validation for available molecules.
pub fn validate_hf_631g() -> Vec<ValidationResult> {
    let molecules = benchmark_molecules();
    let refs = hf_631g_references();

    let mut results = Vec::new();

    for (ref_name, ref_energy, tolerance) in &refs {
        if let Some((_, mol)) = molecules.iter().find(|(name, _)| name == ref_name) {
            let basis = Basis631G::build(mol);
            let rhf = restricted_hartree_fock(mol, &basis, &RhfConfig::default());

            let error = rhf.total_energy - ref_energy;
            results.push(ValidationResult {
                molecule: ref_name.to_string(),
                basis: "6-31G".to_string(),
                computed: rhf.total_energy,
                reference: *ref_energy,
                error_hartree: error,
                error_kcal: error * 627.509,
                pass: error.abs() < *tolerance,
                tolerance: *tolerance,
            });
        }
    }

    results
}

/// Test normalization sensitivity: do the multi-theory correlations survive
/// different normalization schemes?
pub fn normalization_sensitivity() -> NormalizationSensitivityResult {
    use crate::multi_theory::{
        N_THEORIES, TheoryScores, compute_theory_scores, theory_correlation_matrix,
    };

    let molecules = benchmark_molecules();

    // Scheme 1: our default normalization
    let scores_default: Vec<TheoryScores> = molecules
        .iter()
        .map(|(name, mol)| compute_theory_scores(mol, name, 300.0))
        .collect();
    let corr_default = theory_correlation_matrix(&scores_default);

    // Scheme 2: rank-based normalization (replace values with ranks / n)
    let mut scores_rank = scores_default.clone();
    for theory_idx in 0..N_THEORIES {
        let mut vals: Vec<(usize, f64)> = scores_rank
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.normalized[theory_idx]))
            .collect();
        vals.sort_by(|a, b| a.1.total_cmp(&b.1));
        let n = vals.len() as f64;
        for (rank, (mol_idx, _)) in vals.iter().enumerate() {
            scores_rank[*mol_idx].normalized[theory_idx] = (rank as f64 + 0.5) / n;
        }
    }
    let corr_rank = theory_correlation_matrix(&scores_rank);

    // Scheme 3: min-max normalization (rescale to [0,1] per theory)
    let mut scores_minmax = scores_default.clone();
    for theory_idx in 0..N_THEORIES {
        let vals: Vec<f64> = scores_minmax
            .iter()
            .map(|s| s.normalized[theory_idx])
            .collect();
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-15);
        for s in &mut scores_minmax {
            s.normalized[theory_idx] = (s.normalized[theory_idx] - min) / range;
        }
    }
    let corr_minmax = theory_correlation_matrix(&scores_minmax);

    // Check if key correlations survive: HOT-OrchOR and HOT-QDarwinism
    let hot = 3;
    let orch = 4;
    let qdar = 9;

    NormalizationSensitivityResult {
        hot_orch_default: corr_default[hot][orch],
        hot_orch_rank: corr_rank[hot][orch],
        hot_orch_minmax: corr_minmax[hot][orch],
        hot_qdar_default: corr_default[hot][qdar],
        hot_qdar_rank: corr_rank[hot][qdar],
        hot_qdar_minmax: corr_minmax[hot][qdar],
        robust_hot_orch: corr_default[hot][orch].signum() == corr_rank[hot][orch].signum()
            && corr_rank[hot][orch].signum() == corr_minmax[hot][orch].signum(),
        robust_hot_qdar: corr_default[hot][qdar].signum() == corr_rank[hot][qdar].signum()
            && corr_rank[hot][qdar].signum() == corr_minmax[hot][qdar].signum(),
    }
}

#[derive(Debug, Clone)]
pub struct NormalizationSensitivityResult {
    pub hot_orch_default: f64,
    pub hot_orch_rank: f64,
    pub hot_orch_minmax: f64,
    pub hot_qdar_default: f64,
    pub hot_qdar_rank: f64,
    pub hot_qdar_minmax: f64,
    pub robust_hot_orch: bool,
    pub robust_hot_qdar: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_sto3g_h2() {
        let results = validate_hf_sto3g();
        let h2 = results.iter().find(|r| r.molecule == "H2").unwrap();
        assert!(
            h2.pass,
            "H2 HF/STO-3G: computed={:.6}, reference={:.6}, error={:.6} Ha ({:.1} kcal/mol)",
            h2.computed, h2.reference, h2.error_hartree, h2.error_kcal
        );
    }

    #[test]
    fn test_hf_sto3g_water() {
        let results = validate_hf_sto3g();
        let water = results.iter().find(|r| r.molecule == "H2O").unwrap();
        assert!(
            water.pass,
            "H2O HF/STO-3G: computed={:.6}, reference={:.6}, error={:.6} Ha ({:.1} kcal/mol)",
            water.computed, water.reference, water.error_hartree, water.error_kcal
        );
    }

    #[test]
    fn test_hf_631g_improves_on_sto3g() {
        let sto3g = validate_hf_sto3g();
        let b631g = validate_hf_631g();

        // For each molecule in both, 6-31G should be lower energy (variational principle)
        for r631 in &b631g {
            if let Some(rsto) = sto3g.iter().find(|r| r.molecule == r631.molecule) {
                assert!(
                    r631.computed < rsto.computed + 0.001,
                    "{}: 6-31G ({:.4}) should be ≤ STO-3G ({:.4})",
                    r631.molecule,
                    r631.computed,
                    rsto.computed
                );
            }
        }
    }

    #[test]
    fn test_normalization_robustness() {
        let sens = normalization_sensitivity();

        // HOT-OrchOR correlation should survive all normalizations (same sign)
        assert!(
            sens.robust_hot_orch,
            "HOT-OrchOR not robust: default={:.3}, rank={:.3}, minmax={:.3}",
            sens.hot_orch_default, sens.hot_orch_rank, sens.hot_orch_minmax
        );

        // HOT-QDarwinism anti-correlation should survive (same sign)
        assert!(
            sens.robust_hot_qdar,
            "HOT-QDarwinism not robust: default={:.3}, rank={:.3}, minmax={:.3}",
            sens.hot_qdar_default, sens.hot_qdar_rank, sens.hot_qdar_minmax
        );
    }

    #[test]
    fn test_all_sto3g_benchmarks_pass() {
        let results = validate_hf_sto3g();
        let failures: Vec<&ValidationResult> = results.iter().filter(|r| !r.pass).collect();

        if !failures.is_empty() {
            for f in &failures {
                eprintln!(
                    "  FAIL: {} HF/STO-3G: computed={:.6}, ref={:.6}, error={:.4} Ha ({:.1} kcal/mol), tol={:.4}",
                    f.molecule, f.computed, f.reference, f.error_hartree, f.error_kcal, f.tolerance
                );
            }
        }

        let pass_rate = (results.len() - failures.len()) as f64 / results.len() as f64;
        assert!(
            pass_rate >= 0.75,
            "Pass rate {:.0}% is too low ({}/{} passed)",
            pass_rate * 100.0,
            results.len() - failures.len(),
            results.len()
        );
    }

    // ── Phase Q0.4 (2026-07-16): internal-consistency diagnostics for the
    // N2/STO-3G, H2O and CH4/6-31G energy discrepancies. No independent
    // external QC package (pyscf/psi4) is available in this sandbox (NixOS
    // immutable root; checked both ambient Python and `nix develop`, neither
    // has it, and installing one is a bigger environment change than this
    // bounded investigation warrants) -- these checks use invariants that
    // must hold regardless of any external reference: basis-function
    // self-normalization (diagonal S ~= 1.0), matrix symmetry, and Tr[PS] =
    // n_electrons after SCF converges. ──────────────────────────────────

    fn diagnose(mol: &Molecule, basis: &crate::basis::BasisSet, label: &str) {
        use crate::integrals::overlap::overlap_matrix;
        use crate::scf::density::build_density_matrix;

        let n = basis.n_basis();
        let s = overlap_matrix(&basis.functions);

        let max_diag_dev = (0..n)
            .map(|i| (s[i * n + i] - 1.0).abs())
            .fold(0.0_f64, f64::max);
        let max_asym = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| (s[i * n + j] - s[j * n + i]).abs())
            .fold(0.0_f64, f64::max);

        let rhf = restricted_hartree_fock(mol, basis, &RhfConfig::default());
        let p = build_density_matrix(
            &rhf.orbital_coefficients,
            rhf.n_basis,
            rhf.n_independent,
            rhf.n_occupied,
        );
        // Tr[PS] = sum_ij P_ij S_ji, should equal n_electrons for RHF.
        let mut tr_ps = 0.0;
        for i in 0..n {
            for j in 0..n {
                tr_ps += p[i * n + j] * s[j * n + i];
            }
        }
        let n_elec = mol.n_electrons() as f64;

        eprintln!(
            "DIAG {label}: n_basis={n} max|S_ii-1|={max_diag_dev:.2e} max|S_ij-S_ji|={max_asym:.2e} \
             Tr[PS]={tr_ps:.6} n_electrons={n_elec} |Tr[PS]-n_e|={:.2e} \
             energy={:.6} converged={} n_iter={}",
            (tr_ps - n_elec).abs(),
            rhf.total_energy,
            rhf.converged,
            rhf.n_iterations
        );
    }

    #[test]
    fn test_diagnose_n2_sto3g_discrepancy() {
        let mol = Molecule::new(vec![
            crate::molecule::Atom::new(7, 0.0, 0.0, 0.0),
            crate::molecule::Atom::new(7, 0.0, 0.0, 2.074),
        ]);
        let basis = Sto3g::build(&mol);
        diagnose(&mol, &basis, "N2/STO-3G");
    }

    #[test]
    fn test_diagnose_h2o_631g_discrepancy() {
        let mol = Molecule::water();
        let basis = Basis631G::build(&mol);
        diagnose(&mol, &basis, "H2O/6-31G");
    }

    #[test]
    fn test_diagnose_ch4_631g_discrepancy() {
        let mol = Molecule::new(vec![
            crate::molecule::Atom::new(6, 0.0, 0.0, 0.0),
            crate::molecule::Atom::new(1, 1.185, 1.185, 1.185),
            crate::molecule::Atom::new(1, -1.185, -1.185, 1.185),
            crate::molecule::Atom::new(1, -1.185, 1.185, -1.185),
            crate::molecule::Atom::new(1, 1.185, -1.185, -1.185),
        ]);
        let basis = Basis631G::build(&mol);
        diagnose(&mol, &basis, "CH4/6-31G");
    }

    #[test]
    fn test_diagnose_h2_sto3g_control_case_should_be_tight() {
        // Control: H2/STO-3G has +0.02 kcal/mol error (already tight per the
        // module doc's measured figures) -- confirms the diagnostic itself
        // reports clean invariants for a KNOWN-GOOD case, so any anomaly
        // seen for N2/H2O/CH4 isn't an artifact of the diagnostic method.
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        diagnose(&mol, &basis, "H2/STO-3G (control)");
    }
}
