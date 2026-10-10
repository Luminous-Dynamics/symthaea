// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unrestricted Hartree-Fock (UHF) self-consistent field solver.
//!
//! Solves two coupled Roothaan-Hall equations, one per spin channel:
//! F^α C^α = S C^α ε^α, F^β C^β = S C^β ε^β
//!
//! where F^α and F^β depend on BOTH spin densities through the shared
//! Coulomb term (see `scf::fock::build_uhf_fock_matrices`). Unlike RHF,
//! alpha and beta electrons occupy independent sets of spatial orbitals --
//! this is what makes UHF able to represent open-shell systems (radicals,
//! triplets, doublets, any `multiplicity != 1` molecule) that RHF cannot
//! (see `scf::rhf::restricted_hartree_fock`'s Phase Q0 guard).
//!
//! Phase Q2, 2026-07-16. Built as the direct fix for the gap Q0's
//! multiplicity guard found but didn't close. Reuses RHF's one-electron
//! integrals, ERI tensor, and canonical-orthogonalization machinery
//! directly (all spin-independent); density-build and Fock-build are
//! spin-specific (see `scf::density::build_density_matrix_unrestricted`,
//! `scf::fock::build_uhf_fock_matrices`) since RHF's versions hardcode a
//! factor-of-2 double-occupancy convention that doesn't apply per spin
//! channel.
//!
//! References:
//! - Szabo & Ostlund (1996). *Modern Quantum Chemistry*. Chapter 3.
//! - Pople & Nesbet (1954). J. Chem. Phys. 22, 571 (UHF).

use crate::basis::BasisSet;
use crate::constants::{
    CANONICAL_ORTH_THRESHOLD, MAX_SCF_ITERATIONS, SCF_DENSITY_THRESHOLD,
    SCF_ENERGY_THRESHOLD,
};
use crate::integrals::eri::{compute_eri_tensor, compute_schwarz_bounds};
use crate::integrals::kinetic::kinetic_matrix;
use crate::integrals::nuclear::nuclear_matrix;
use crate::integrals::overlap::overlap_matrix;
use crate::molecule::Molecule;
use crate::scf::density::{build_density_matrix_unrestricted, density_rms_change};
use crate::scf::diis::Diis;
use crate::scf::fock::{
    build_uhf_fock_matrices, build_uhf_fock_matrices_direct, uhf_electronic_energy,
};
use crate::scf::generalized_eigen::{
    ScfSolveError, ScfSolverDiagnostics, canonical_orthogonalization_checked,
    solve_generalized_eigen_checked,
};

/// Configuration for the UHF solver. Same shape as `RhfConfig`.
#[derive(Debug, Clone)]
pub struct UhfConfig {
    pub max_iterations: usize,
    pub energy_convergence: f64,
    pub density_convergence: f64,
    pub use_diis: bool,
    /// Direct SCF (Phase Q3, 2026-07-16) -- see `RhfConfig::direct`'s doc
    /// comment for the memory/compute tradeoff. Default `false`.
    pub direct: bool,
}

impl Default for UhfConfig {
    fn default() -> Self {
        Self {
            max_iterations: MAX_SCF_ITERATIONS,
            energy_convergence: SCF_ENERGY_THRESHOLD,
            density_convergence: SCF_DENSITY_THRESHOLD,
            use_diis: true,
            direct: false,
        }
    }
}

/// Result of a UHF calculation.
#[derive(Debug, Clone)]
pub struct UhfResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion: f64,
    pub orbital_energies_alpha: Vec<f64>,
    pub orbital_energies_beta: Vec<f64>,
    /// C[μ * n_mo + i], same layout as `RhfResult::orbital_coefficients`.
    pub orbital_coefficients_alpha: Vec<f64>,
    pub orbital_coefficients_beta: Vec<f64>,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub n_iterations: usize,
    pub converged: bool,
    pub n_basis: usize,
    pub n_independent: usize,
    /// S(S+1) from the molecule's declared multiplicity.
    pub spin_squared_expected: f64,
    /// <S^2> actually computed from the converged alpha/beta orbitals --
    /// equals `spin_squared_expected` only for a spin-pure solution; real
    /// UHF solutions are usually somewhat spin-contaminated (higher than
    /// expected). See the module doc and `compute_spin_squared` below.
    pub spin_squared_computed: f64,
    /// Present when a required checked eigensolve or energy validation failed.
    /// Present when a required checked eigensolve or energy validation failed.
    /// A failed calculation always has converged=false and non-finite energies.
    pub solver_error: Option<String>,
    /// Residual receipt for every required checked eigensolve and overlap transform.
    /// None on numerical failure; a present receipt does not imply SCF fixed-point convergence.
    pub solver_diagnostics: Option<ScfSolverDiagnostics>,
}

/// Split `n_electrons`/`multiplicity` into (n_alpha, n_beta).
///
/// `multiplicity = 2S + 1`, so `n_alpha - n_beta = multiplicity - 1` and
/// `n_alpha + n_beta = n_electrons`. Panics (matching this crate's
/// established panic-based fail-closed convention -- see Q0's
/// `restricted_hartree_fock`/`Molecule::n_electrons` guards) if the
/// combination is inconsistent: a non-integer split (e.g. even electron
/// count with even multiplicity), or `n_beta` would be negative (an
/// unreachable multiplicity for that electron count). `n_alpha` is always
/// the majority-spin count by this crate's convention.
fn try_alpha_beta_split(
    n_electrons: usize,
    multiplicity: u32,
) -> Result<(usize, usize), ScfSolveError> {
    let electron_count = i64::try_from(n_electrons).map_err(|_| {
        ScfSolveError::InconsistentElectronicState {
            electron_count: n_electrons,
            multiplicity,
        }
    })?;
    if multiplicity == 0 {
        return Err(ScfSolveError::InvalidMultiplicity { multiplicity });
    }
    let unpaired = i64::from(multiplicity) - 1;
    let sum_minus_diff = electron_count - unpaired;
    if sum_minus_diff < 0 || sum_minus_diff % 2 != 0 {
        return Err(ScfSolveError::InconsistentElectronicState {
            electron_count: n_electrons,
            multiplicity,
        });
    }
    let n_beta = sum_minus_diff / 2;
    let n_alpha = n_beta + unpaired;
    Ok((n_alpha as usize, n_beta as usize))
}

/// Preserve the legacy panic contract for impossible spin states.
fn alpha_beta_split(n_electrons: usize, multiplicity: u32) -> (usize, usize) {
    match try_alpha_beta_split(n_electrons, multiplicity) {
        Ok(split) => split,
        Err(ScfSolveError::InvalidMultiplicity { multiplicity }) => {
            panic!("invalid multiplicity {multiplicity}: must be >= 1")
        }
        Err(ScfSolveError::InconsistentElectronicState {
            electron_count,
            multiplicity,
        }) => panic!(
            "inconsistent electronic state: {electron_count} electrons cannot have multiplicity {multiplicity}"
        ),
        Err(error) => panic!("{error}"),
    }
}

/// Cross spin-orbital overlap <ψ_i^α|ψ_j^β> = Σ_μν C^α_μi S_μν C^β_νj,
/// then <S²>_computed = S(S+1) + n_beta - Σ_ij |<ψ_i^α|ψ_j^β>|² (standard
/// UHF spin-contamination formula, e.g. Szabo & Ostlund).
fn compute_spin_squared(
    s_mat: &[f64],
    c_alpha: &[f64],
    c_beta: &[f64],
    n_basis: usize,
    n_mo: usize,
    n_alpha: usize,
    n_beta: usize,
) -> f64 {
    let s_exact = {
        let s = (n_alpha as f64 - n_beta as f64) / 2.0;
        s * (s + 1.0)
    };

    let mut overlap_sum_sq = 0.0;
    for i in 0..n_alpha {
        for j in 0..n_beta {
            let mut overlap_ij = 0.0;
            for mu in 0..n_basis {
                for nu in 0..n_basis {
                    overlap_ij +=
                        c_alpha[mu * n_mo + i] * s_mat[mu * n_basis + nu] * c_beta[nu * n_mo + j];
                }
            }
            overlap_sum_sq += overlap_ij * overlap_ij;
        }
    }

    s_exact + n_beta as f64 - overlap_sum_sq
}

/// Run an Unrestricted Hartree-Fock calculation.
///
/// Works for any `multiplicity` (including 1 -- see the closed-shell
/// reduction tests below, which verify UHF gives the exact same energy as
/// RHF when alpha and beta densities are forced equal). Panics on an
/// inconsistent electron-count/multiplicity combination (see
/// `alpha_beta_split`).
pub fn unrestricted_hartree_fock(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &UhfConfig,
) -> UhfResult {
    // Preserve panic-based validation for impossible spin declarations in the
    // legacy entry point. Numerical failures are returned as converged=false.
    let _ = alpha_beta_split(molecule.n_electrons(), molecule.multiplicity);
    match try_unrestricted_hartree_fock(molecule, basis, config) {
        Ok(result) => result,
        Err(error) => UhfResult {
            total_energy: f64::NAN,
            electronic_energy: f64::NAN,
            nuclear_repulsion: f64::NAN,
            orbital_energies_alpha: Vec::new(),
            orbital_energies_beta: Vec::new(),
            orbital_coefficients_alpha: Vec::new(),
            orbital_coefficients_beta: Vec::new(),
            n_alpha: 0,
            n_beta: 0,
            n_iterations: 0,
            converged: false,
            n_basis: basis.n_basis(),
            n_independent: 0,
            spin_squared_expected: f64::NAN,
            spin_squared_computed: f64::NAN,
            solver_error: Some(error.to_string()),
            solver_diagnostics: None,
        },
    }
}

/// Fallible UHF entry point. Every required alpha/beta generalized eigen solve
/// uses the checked path; a failed eigensolve aborts without claiming convergence.
pub fn try_unrestricted_hartree_fock(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &UhfConfig,
) -> Result<UhfResult, ScfSolveError> {
    try_unrestricted_hartree_fock_with_eigensolver_budget(molecule, basis, config, None)
}

/// Internal budget override makes alpha/beta iteration-exhaustion propagation testable
/// without changing production configuration or weakening numerical thresholds.
fn try_unrestricted_hartree_fock_with_eigensolver_budget(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &UhfConfig,
    eigensolver_iteration_budget: Option<usize>,
) -> Result<UhfResult, ScfSolveError> {
    let electron_count = super::checked_electron_count(molecule)?;
    let (n_alpha, n_beta) = try_alpha_beta_split(electron_count, molecule.multiplicity)?;
    let n = basis.n_basis();
    let v_nn = molecule.nuclear_repulsion_energy();
    if !v_nn.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "nuclear repulsion",
        });
    }

    // Step 1: One-electron integrals (spin-independent).
    let s_mat = overlap_matrix(&basis.functions);
    let t_mat = kinetic_matrix(&basis.functions);
    let v_mat = nuclear_matrix(&basis.functions, &molecule.atoms);
    let mut h_core = vec![0.0; n * n];
    for i in 0..n * n {
        h_core[i] = t_mat[i] + v_mat[i];
    }

    // Step 2: Check S before the expensive two-electron integral build.
    let overlap_iterations = n.saturating_mul(n).saturating_mul(100).max(1);
    let checked_orthogonalization = canonical_orthogonalization_checked(
        &s_mat,
        n,
        CANONICAL_ORTH_THRESHOLD,
        overlap_iterations,
    )?;
    let mut solver_diagnostics =
        ScfSolverDiagnostics::from_overlap_residual(checked_orthogonalization.orthogonality_residual);
    let x_mat = checked_orthogonalization.transformation;
    let n_ind = checked_orthogonalization.n_independent;
    let eigensolver_iterations = eigensolver_iteration_budget
        .unwrap_or_else(|| n_ind.saturating_mul(n_ind).saturating_mul(100).max(1));

    // Step 3: Two-electron integrals (spin-independent) -- dense-tensor
    // (default) or direct (Phase Q3, 2026-07-16) mode.
    let (eri, _eri_computed, _eri_screened) = if config.direct {
        (Vec::new(), 0, 0)
    } else {
        compute_eri_tensor(&basis.functions)
    };
    let schwarz = if config.direct {
        compute_schwarz_bounds(&basis.functions)
    } else {
        Vec::new()
    };
    let build_fock = |h_core: &[f64], p_a: &[f64], p_b: &[f64]| -> (Vec<f64>, Vec<f64>) {
        if config.direct {
            build_uhf_fock_matrices_direct(h_core, p_a, p_b, &basis.functions, &schwarz, n)
        } else {
            build_uhf_fock_matrices(h_core, p_a, p_b, &eri, n)
        }
    };

    // Step 4: Initial guess — one checked H_core generalized solve for both spins.
    let initial = solve_generalized_eigen_checked(
        &h_core,
        &s_mat,
        &x_mat,
        n,
        n_ind,
        eigensolver_iterations,
    )?;
    solver_diagnostics.record_generalized_solve(&initial);
    let mut c_alpha = initial.generalized.coefficients.clone();
    let mut c_beta = initial.generalized.coefficients;
    let mut eps_alpha = initial.generalized.eigenvalues.clone();
    let mut eps_beta = initial.generalized.eigenvalues;

    let mut p_alpha = build_density_matrix_unrestricted(&c_alpha, n, n_ind, n_alpha);
    let mut p_beta = build_density_matrix_unrestricted(&c_beta, n, n_ind, n_beta);

    let mut energy_old = 0.0;
    let mut converged = false;
    let mut n_iterations = 0;
    let mut diis_alpha = if config.use_diis {
        Some(Diis::new(n))
    } else {
        None
    };
    let mut diis_beta = if config.use_diis {
        Some(Diis::new(n))
    } else {
        None
    };

    for iter in 0..config.max_iterations {
        n_iterations = iter + 1;

        let (mut fock_alpha, mut fock_beta) = build_fock(&h_core, &p_alpha, &p_beta);

        if let Some(ref mut d) = diis_alpha {
            fock_alpha = d.extrapolate(&fock_alpha, &p_alpha, &s_mat);
        }
        if let Some(ref mut d) = diis_beta {
            fock_beta = d.extrapolate(&fock_beta, &p_beta, &s_mat);
        }

        let e_elec = uhf_electronic_energy(&p_alpha, &p_beta, &h_core, &fock_alpha, &fock_beta, n);
        let e_total = e_elec + v_nn;
        if !e_elec.is_finite() || !e_total.is_finite() {
            return Err(ScfSolveError::NonFiniteEnergy {
                stage: "UHF SCF iteration",
            });
        }

        let result_alpha = solve_generalized_eigen_checked(
            &fock_alpha,
            &s_mat,
            &x_mat,
            n,
            n_ind,
            eigensolver_iterations,
        )?;
        solver_diagnostics.record_generalized_solve(&result_alpha);
        c_alpha = result_alpha.generalized.coefficients;
        eps_alpha = result_alpha.generalized.eigenvalues;

        let result_beta = solve_generalized_eigen_checked(
            &fock_beta,
            &s_mat,
            &x_mat,
            n,
            n_ind,
            eigensolver_iterations,
        )?;
        solver_diagnostics.record_generalized_solve(&result_beta);
        c_beta = result_beta.generalized.coefficients;
        eps_beta = result_beta.generalized.eigenvalues;

        let p_alpha_new = build_density_matrix_unrestricted(&c_alpha, n, n_ind, n_alpha);
        let p_beta_new = build_density_matrix_unrestricted(&c_beta, n, n_ind, n_beta);

        let d_rms_alpha = density_rms_change(&p_alpha_new, &p_alpha, n);
        let d_rms_beta = density_rms_change(&p_beta_new, &p_beta, n);
        let de = (e_total - energy_old).abs();

        p_alpha = p_alpha_new;
        p_beta = p_beta_new;
        energy_old = e_total;

        if de < config.energy_convergence
            && d_rms_alpha < config.density_convergence
            && d_rms_beta < config.density_convergence
            && iter > 0
        {
            converged = true;
            break;
        }
    }

    let (fock_alpha_final, fock_beta_final) = build_fock(&h_core, &p_alpha, &p_beta);
    let e_elec = uhf_electronic_energy(
        &p_alpha,
        &p_beta,
        &h_core,
        &fock_alpha_final,
        &fock_beta_final,
        n,
    );

    let spin_squared_expected = {
        let s = (n_alpha as f64 - n_beta as f64) / 2.0;
        s * (s + 1.0)
    };
    let spin_squared_computed =
        compute_spin_squared(&s_mat, &c_alpha, &c_beta, n, n_ind, n_alpha, n_beta);
    let total_energy = e_elec + v_nn;
    if !e_elec.is_finite() || !total_energy.is_finite() || !spin_squared_computed.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "final UHF energy or spin diagnostic",
        });
    }

    Ok(UhfResult {
        total_energy,
        electronic_energy: e_elec,
        nuclear_repulsion: v_nn,
        orbital_energies_alpha: eps_alpha,
        orbital_energies_beta: eps_beta,
        orbital_coefficients_alpha: c_alpha,
        orbital_coefficients_beta: c_beta,
        n_alpha,
        n_beta,
        n_iterations,
        converged,
        n_basis: n,
        n_independent: n_ind,
        spin_squared_expected,
        spin_squared_computed,
        solver_error: None,
        solver_diagnostics: Some(solver_diagnostics),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

    #[test]
    fn test_alpha_beta_split_singlet() {
        assert_eq!(alpha_beta_split(10, 1), (5, 5));
    }

    #[test]
    fn test_alpha_beta_split_doublet() {
        // Li atom: 3 electrons, doublet ground state.
        assert_eq!(alpha_beta_split(3, 2), (2, 1));
    }

    #[test]
    fn test_alpha_beta_split_triplet() {
        // O atom in its triplet ground state: 8 electrons.
        assert_eq!(alpha_beta_split(8, 3), (5, 3));
    }

    #[test]
    #[should_panic(expected = "inconsistent electronic state")]
    fn test_alpha_beta_split_rejects_impossible_combination() {
        // 10 electrons (even) can't have multiplicity 2 (needs odd n_alpha-n_beta parity).
        alpha_beta_split(10, 2);
    }

    #[test]
    fn test_failed_checked_overlap_never_returns_converged_energy() {
        let molecule = Molecule::h2();
        let mut basis = Sto3g::build(&molecule);
        basis.functions[0].primitives[0].alpha = f64::NAN;
        let config = UhfConfig::default();

        let fallible = try_unrestricted_hartree_fock(&molecule, &basis, &config);
        assert!(matches!(
            fallible,
            Err(ScfSolveError::CanonicalOrthogonalization(_))
        ));

        let legacy = unrestricted_hartree_fock(&molecule, &basis, &config);
        assert!(!legacy.converged);
        assert!(legacy.total_energy.is_nan());
        assert!(legacy.electronic_energy.is_nan());
        assert!(legacy.solver_error.as_deref().is_some_and(|message| message.contains("non-finite")));
        assert!(legacy.solver_diagnostics.is_none());
    }

    #[test]
    fn test_fallible_uhf_rejects_invalid_charge_and_empty_molecule_without_panicking() {
        let hydrogen = crate::molecule::Atom::new(1, 0.0, 0.0, 0.0);
        let invalid_charge = Molecule::with_charge(vec![hydrogen], 2, 1);
        let invalid_charge_basis = Sto3g::build(&invalid_charge);
        assert!(matches!(
            try_unrestricted_hartree_fock(
                &invalid_charge,
                &invalid_charge_basis,
                &UhfConfig::default()
            ),
            Err(ScfSolveError::InvalidCharge {
                charge: 2,
                nuclear_charge: 1
            })
        ));

        let empty = Molecule::new(Vec::new());
        let empty_basis = crate::basis::BasisSet {
            name: "empty-test".to_string(),
            functions: Vec::new(),
        };
        assert!(matches!(
            try_unrestricted_hartree_fock(&empty, &empty_basis, &UhfConfig::default()),
            Err(ScfSolveError::EmptyMolecule)
        ));
    }

    #[test]
    #[should_panic(expected = "invalid molecule: charge")]
    fn test_legacy_uhf_keeps_invalid_charge_panic_contract() {
        let molecule = Molecule::with_charge(
            vec![crate::molecule::Atom::new(1, 0.0, 0.0, 0.0)],
            2,
            1,
        );
        let basis = Sto3g::build(&molecule);
        let _ = unrestricted_hartree_fock(&molecule, &basis, &UhfConfig::default());
    }

    #[test]
    fn test_fock_eigensolver_iteration_exhaustion_propagates_as_error() {
        let molecule = Molecule::water();
        let basis = Sto3g::build(&molecule);
        let error = try_unrestricted_hartree_fock_with_eigensolver_budget(
            &molecule,
            &basis,
            &UhfConfig::default(),
            Some(0),
        )
        .expect_err("a zero-rotation budget must not produce an unchecked UHF result");
        assert!(matches!(
            error,
            ScfSolveError::GeneralizedEigen(
                crate::scf::generalized_eigen::CheckedGeneralizedEigenError::EigensolverNotConverged(_)
            )
        ));
    }

    #[test]
    fn test_uhf_h2_matches_rhf_for_closed_shell() {
        // Phase Q2's primary correctness test: no external reference needed
        // -- for a genuinely closed-shell molecule, UHF's alpha and beta
        // densities converge to be identical, and the equations
        // mathematically collapse to RHF's. H2/STO-3G's RHF energy is
        // already validated against Szabo & Ostlund in rhf.rs's own tests.
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let uhf = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());

        assert!(uhf.converged);
        let diagnostics = uhf
            .solver_diagnostics
            .as_ref()
            .expect("successful UHF must expose a solver residual receipt");
        assert_eq!(
            diagnostics.generalized_eigensolve_count,
            1 + 2 * uhf.n_iterations,
            "UHF performs one shared core-guess solve plus alpha and beta solves per iteration"
        );
        assert!(diagnostics.total_jacobi_rotations < 1_000_000);
        assert!(diagnostics.max_transformed_off_diagonal_hartree.is_finite());
        assert!(diagnostics.max_transformed_eigenpair_residual <= 1e-10);
        assert!(diagnostics.max_eigenvector_orthogonality_residual <= 1e-10);
        assert!(diagnostics.max_generalized_eigenpair_residual <= 1e-8);
        assert!(diagnostics.overlap_orthogonality_residual <= 1e-8);
        assert_eq!(uhf.n_alpha, 1);
        assert_eq!(uhf.n_beta, 1);
        assert!(
            (uhf.total_energy - rhf.total_energy).abs() < 1e-8,
            "UHF energy {} should match RHF energy {} for closed-shell H2",
            uhf.total_energy,
            rhf.total_energy
        );
        // Spin-pure singlet: <S^2> should be exactly 0.
        assert!(
            uhf.spin_squared_computed.abs() < 1e-8,
            "closed-shell UHF should have <S^2>=0, got {}",
            uhf.spin_squared_computed
        );
    }

    #[test]
    fn test_uhf_water_matches_rhf_for_closed_shell() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let uhf = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());

        assert!(uhf.converged);
        assert!(
            (uhf.total_energy - rhf.total_energy).abs() < 1e-8,
            "UHF energy {} should match RHF energy {} for closed-shell water",
            uhf.total_energy,
            rhf.total_energy
        );
    }

    #[test]
    fn test_uhf_lithium_atom_open_shell() {
        // Li atom: 3 electrons, doublet ground state (1s^2 2s^1),
        // n_alpha=2, n_beta=1 -- a real open-shell system RHF now refuses
        // (Phase Q0) and this function can actually compute.
        let mol = Molecule::with_charge(vec![crate::molecule::Atom::new(3, 0.0, 0.0, 0.0)], 0, 2);
        let basis = Sto3g::build(&mol);
        let uhf = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());

        assert!(uhf.converged, "Li atom UHF should converge");
        assert_eq!(uhf.n_alpha, 2);
        assert_eq!(uhf.n_beta, 1);
        assert!(uhf.total_energy < 0.0, "Li atom energy should be negative");

        // Internal-consistency invariants (no external reference needed --
        // same discipline as Q0's HF-discrepancy investigation): Tr[P*S]
        // must equal the electron count for each spin.
        let p_alpha = build_density_matrix_unrestricted(
            &uhf.orbital_coefficients_alpha,
            uhf.n_basis,
            uhf.n_independent,
            uhf.n_alpha,
        );
        let p_beta = build_density_matrix_unrestricted(
            &uhf.orbital_coefficients_beta,
            uhf.n_basis,
            uhf.n_independent,
            uhf.n_beta,
        );
        let s_mat = overlap_matrix(&basis.functions);
        let n = uhf.n_basis;
        let mut tr_ps_alpha = 0.0;
        let mut tr_ps_beta = 0.0;
        for i in 0..n {
            for j in 0..n {
                tr_ps_alpha += p_alpha[i * n + j] * s_mat[j * n + i];
                tr_ps_beta += p_beta[i * n + j] * s_mat[j * n + i];
            }
        }
        assert!(
            (tr_ps_alpha - uhf.n_alpha as f64).abs() < 1e-6,
            "Tr[P_alpha*S]={tr_ps_alpha}, expected {}",
            uhf.n_alpha
        );
        assert!(
            (tr_ps_beta - uhf.n_beta as f64).abs() < 1e-6,
            "Tr[P_beta*S]={tr_ps_beta}, expected {}",
            uhf.n_beta
        );

        // Li atom is a well-behaved, textbook-low-contamination doublet --
        // <S^2>_computed should be close to the exact 0.75 (S=1/2). Measured
        // 2026-07-16: 0.750000 to 6 decimals -- Li/STO-3G's 1s core (beta)
        // and 2s valence (unpaired alpha) are well-separated, so this
        // system is essentially perfectly spin-pure. Tolerance kept looser
        // than the measured precision to avoid a brittle exact-equality
        // assertion across platforms/toolchains.
        assert_eq!(uhf.spin_squared_expected, 0.75);
        assert!(
            (uhf.spin_squared_computed - 0.75).abs() < 1e-4,
            "Li atom spin contamination unexpectedly high: <S^2>={}",
            uhf.spin_squared_computed
        );
    }

    #[test]
    fn test_uhf_direct_mode_matches_dense_mode_end_to_end() {
        // Phase Q3 (2026-07-16): full end-to-end direct-vs-dense identity
        // for UHF, using the real open-shell Li atom test case.
        let mol = Molecule::with_charge(vec![crate::molecule::Atom::new(3, 0.0, 0.0, 0.0)], 0, 2);
        let basis = Sto3g::build(&mol);

        let dense = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());
        let direct = unrestricted_hartree_fock(
            &mol,
            &basis,
            &UhfConfig {
                direct: true,
                ..Default::default()
            },
        );

        assert!(dense.converged && direct.converged);
        assert!(
            (dense.total_energy - direct.total_energy).abs() < 1e-8,
            "dense={:.10} direct={:.10} should match",
            dense.total_energy,
            direct.total_energy
        );
        assert!(
            (dense.spin_squared_computed - direct.spin_squared_computed).abs() < 1e-6,
            "spin contamination should match between modes"
        );
    }
}
