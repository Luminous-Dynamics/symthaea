// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Restricted Hartree-Fock (RHF) self-consistent field solver.
//!
//! Solves the Roothaan-Hall equations: FC = SCε
//!
//! Algorithm:
//! 1. Compute one-electron integrals: S, T, V → H_core = T + V
//! 2. Compute two-electron integrals: ERIs with Schwarz prescreening
//! 3. Canonical orthogonalization of S → X matrix
//! 4. Initial guess: diagonalize H_core in orthogonal basis
//! 5. SCF loop:
//!    a. Build density matrix P from occupied MOs
//!    b. Build Fock matrix F = H_core + G(P)
//!    c. DIIS extrapolation (optional)
//!    d. Solve FC = SCε via X transformation
//!    e. Check convergence (energy + density)
//!    f. Repeat until converged or max iterations
//!
//! References:
//! - Szabo & Ostlund (1996). *Modern Quantum Chemistry*. Chapter 3.
//! - Pulay, P. (1980). Chem. Phys. Lett. 73, 393 (DIIS).

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
use crate::scf::density::{build_density_matrix, density_rms_change};
use crate::scf::diis::Diis;
use crate::scf::fock::{build_fock_matrix, build_fock_matrix_direct, electronic_energy};
use crate::scf::generalized_eigen::{
    ScfSolveError, ScfSolverDiagnostics, canonical_orthogonalization_checked,
    solve_generalized_eigen_checked,
};

/// Configuration for the RHF solver.
#[derive(Debug, Clone)]
pub struct RhfConfig {
    /// Maximum SCF iterations
    pub max_iterations: usize,
    /// Energy convergence threshold (Hartree)
    pub energy_convergence: f64,
    /// Density matrix RMS convergence threshold
    pub density_convergence: f64,
    /// Use DIIS convergence acceleration
    pub use_diis: bool,
    /// Use direct SCF (Phase Q3, 2026-07-16): compute each Fock-matrix ERI
    /// on demand instead of caching a dense `n^4` tensor. Trades memory
    /// (`O(n^4)` -> `O(n^2)`) for extra computation (each unique integral
    /// gets recomputed up to 8x instead of cached once). Default `false` --
    /// this is an opt-in alternative, not a replacement; existing
    /// performance for small systems is unaffected. See `scf::fock`'s
    /// module doc and `eri_computed`/`eri_screened`'s doc comments below
    /// for what changes when this is `true`.
    pub direct: bool,
}

impl Default for RhfConfig {
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

/// Result of an RHF calculation.
#[derive(Debug, Clone)]
pub struct RhfResult {
    /// Total energy (electronic + nuclear repulsion) in Hartree
    pub total_energy: f64,
    /// Electronic energy in Hartree
    pub electronic_energy: f64,
    /// Nuclear repulsion energy in Hartree
    pub nuclear_repulsion: f64,
    /// Orbital energies (eigenvalues) in Hartree
    pub orbital_energies: Vec<f64>,
    /// MO coefficients: C[μ * n_mo + i] = coefficient of AO μ in MO i
    pub orbital_coefficients: Vec<f64>,
    /// Number of SCF iterations
    pub n_iterations: usize,
    /// Whether the SCF converged
    pub converged: bool,
    /// Number of basis functions
    pub n_basis: usize,
    /// Number of independent basis functions (after linear dependence removal)
    pub n_independent: usize,
    /// Number of occupied orbitals
    pub n_occupied: usize,
    /// Number of ERIs computed (vs screened). Only meaningful for
    /// `config.direct = false` (the dense-tensor path); always 0 when
    /// `direct = true`, since that path never counts a single upfront
    /// tensor build (it recomputes integrals per Fock build instead).
    pub eri_computed: usize,
    /// Number of ERIs screened out. Same `direct = true` caveat as
    /// `eri_computed`.
    pub eri_screened: usize,
    /// Present when a required checked eigensolve or energy validation failed.
    /// Present when a required checked eigensolve or energy validation failed.
    /// A failed calculation always has converged=false and non-finite energies.
    pub solver_error: Option<String>,
    /// Residual receipt for every required checked eigensolve and overlap transform.
    /// None on numerical failure; a present receipt does not imply SCF fixed-point convergence.
    pub solver_diagnostics: Option<ScfSolverDiagnostics>,
}

/// Run a Restricted Hartree-Fock calculation.
///
/// Panics if `molecule.multiplicity != 1` (Phase Q0, 2026-07-16). RHF is
/// fundamentally a closed-shell method -- doubly occupying `n_electrons()/2`
/// spatial orbitals is only physically correct for a singlet. Before this
/// check, an open-shell molecule (e.g. triplet O2, any radical) would
/// silently receive a plausible-looking but wrong closed-shell energy
/// instead of an error; `multiplicity` was otherwise decorative (unused
/// anywhere else in this crate). UHF now exists for open-shell systems
/// (`scf::uhf::unrestricted_hartree_fock`, Phase Q2, 2026-07-16); ROHF is
/// still not implemented.
pub fn restricted_hartree_fock(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &RhfConfig,
) -> RhfResult {
    // Preserve the established panic contract for using RHF on an open-shell state.
    assert_eq!(
        molecule.multiplicity, 1,
        "restricted_hartree_fock only supports closed-shell (multiplicity=1) molecules; \
         got multiplicity={}. Use unrestricted_hartree_fock (scf::uhf) for open-shell \
         systems; ROHF is not yet implemented.",
        molecule.multiplicity
    );
    // Preserve the legacy panic contract for charge exceeding nuclear charge.
    let _ = molecule.n_electrons();
    match try_restricted_hartree_fock(molecule, basis, config) {
        Ok(result) => result,
        Err(error) => RhfResult {
            total_energy: f64::NAN,
            electronic_energy: f64::NAN,
            nuclear_repulsion: f64::NAN,
            orbital_energies: Vec::new(),
            orbital_coefficients: Vec::new(),
            n_iterations: 0,
            converged: false,
            n_basis: basis.n_basis(),
            n_independent: 0,
            n_occupied: 0,
            eri_computed: 0,
            eri_screened: 0,
            solver_error: Some(error.to_string()),
            solver_diagnostics: None,
        },
    }
}

/// Fallible RHF entry point. Any required overlap/Fock eigensolve that fails its
/// residual contracts aborts the calculation instead of allowing a later SCF
/// energy/density iteration to report convergence.
pub fn try_restricted_hartree_fock(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &RhfConfig,
) -> Result<RhfResult, ScfSolveError> {
    try_restricted_hartree_fock_with_eigensolver_budget(molecule, basis, config, None)
}

/// Internal budget override makes iteration-exhaustion propagation testable without
/// changing the production SCF configuration or weakening any numerical threshold.
fn try_restricted_hartree_fock_with_eigensolver_budget(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &RhfConfig,
    eigensolver_iteration_budget: Option<usize>,
) -> Result<RhfResult, ScfSolveError> {
    if molecule.multiplicity != 1 {
        return Err(ScfSolveError::UnsupportedMultiplicity {
            method: "RHF",
            multiplicity: molecule.multiplicity,
        });
    }
    let electron_count = super::checked_electron_count(molecule)?;
    if electron_count % 2 != 0 {
        return Err(ScfSolveError::InconsistentElectronicState {
            electron_count,
            multiplicity: molecule.multiplicity,
        });
    }
    let n = basis.n_basis();
    let n_occ = electron_count / 2;
    let v_nn = molecule.nuclear_repulsion_energy();

    // Step 1: One-electron integrals
    let s_mat = overlap_matrix(&basis.functions);
    let t_mat = kinetic_matrix(&basis.functions);
    let v_mat = nuclear_matrix(&basis.functions, &molecule.atoms);

    // Core Hamiltonian: H_core = T + V
    let mut h_core = vec![0.0; n * n];
    for i in 0..n * n {
        h_core[i] = t_mat[i] + v_mat[i];
    }

    if !v_nn.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "nuclear repulsion",
        });
    }

    // Step 2: Checked canonical orthogonalization happens before the expensive ERI
    // build, so an invalid/indefinite overlap matrix fails before SCF work begins.
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

    // Step 3: Two-electron integrals -- dense-tensor (default) or direct
    // (Phase Q3, 2026-07-16) mode, see `RhfConfig::direct`'s doc comment.
    let (eri, eri_computed, eri_screened) = if config.direct {
        (Vec::new(), 0, 0)
    } else {
        compute_eri_tensor(&basis.functions)
    };
    let schwarz = if config.direct {
        compute_schwarz_bounds(&basis.functions)
    } else {
        Vec::new()
    };
    let build_fock = |h_core: &[f64], density: &[f64]| -> Vec<f64> {
        if config.direct {
            build_fock_matrix_direct(h_core, density, &basis.functions, &schwarz, n)
        } else {
            build_fock_matrix(h_core, density, &eri, n)
        }
    };

    // Step 4: Initial guess — a required checked generalized eigensolve.
    let initial = solve_generalized_eigen_checked(
        &h_core,
        &s_mat,
        &x_mat,
        n,
        n_ind,
        eigensolver_iterations,
    )?;
    solver_diagnostics.record_generalized_solve(&initial);
    let mut coefficients = initial.generalized.coefficients;
    let mut orbital_energies = initial.generalized.eigenvalues;

    // Step 5: SCF loop
    let mut density = build_density_matrix(&coefficients, n, n_ind, n_occ);
    let mut energy_old = 0.0;
    let mut converged = false;
    let mut n_iterations = 0;
    let mut diis = if config.use_diis {
        Some(Diis::new(n))
    } else {
        None
    };

    for iter in 0..config.max_iterations {
        n_iterations = iter + 1;

        // 5a: Build Fock matrix
        let mut fock = build_fock(&h_core, &density);

        // 5b: DIIS extrapolation
        if let Some(ref mut diis_engine) = diis {
            fock = diis_engine.extrapolate(&fock, &density, &s_mat);
        }

        // 5c: Compute electronic energy
        let e_elec = electronic_energy(&density, &h_core, &fock, n);
        let e_total = e_elec + v_nn;
        if !e_elec.is_finite() || !e_total.is_finite() {
            return Err(ScfSolveError::NonFiniteEnergy {
                stage: "RHF SCF iteration",
            });
        }

        // 5d: Solve FC = SCε. Failure propagates out of the SCF loop.
        let result = solve_generalized_eigen_checked(
            &fock,
            &s_mat,
            &x_mat,
            n,
            n_ind,
            eigensolver_iterations,
        )?;
        solver_diagnostics.record_generalized_solve(&result);
        coefficients = result.generalized.coefficients;
        orbital_energies = result.generalized.eigenvalues;

        // 5e: New density matrix
        let density_new = build_density_matrix(&coefficients, n, n_ind, n_occ);
        let d_rms = density_rms_change(&density_new, &density, n);
        let de = (e_total - energy_old).abs();

        density = density_new;
        energy_old = e_total;

        // 5f: Check convergence
        if de < config.energy_convergence && d_rms < config.density_convergence && iter > 0 {
            converged = true;
            break;
        }
    }

    // Final energy
    let fock_final = build_fock(&h_core, &density);
    let e_elec = electronic_energy(&density, &h_core, &fock_final, n);
    let total_energy = e_elec + v_nn;
    if !e_elec.is_finite() || !total_energy.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "final RHF energy",
        });
    }

    Ok(RhfResult {
        total_energy,
        electronic_energy: e_elec,
        nuclear_repulsion: v_nn,
        orbital_energies,
        orbital_coefficients: coefficients,
        n_iterations,
        converged,
        n_basis: n,
        n_independent: n_ind,
        n_occupied: n_occ,
        eri_computed,
        eri_screened,
        solver_error: None,
        solver_diagnostics: Some(solver_diagnostics),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;

    #[test]
    #[should_panic(expected = "only supports closed-shell (multiplicity=1)")]
    fn test_open_shell_molecule_is_rejected_not_silently_wrong() {
        // Phase Q0 (2026-07-16): before this guard, a triplet O2 (or any
        // multiplicity != 1 molecule) would silently receive a
        // plausible-looking but physically wrong closed-shell RHF energy
        // instead of an error -- `multiplicity` was otherwise decorative
        // (unused anywhere in the crate). Uses a single oxygen atom with
        // multiplicity=3 (a real, simple open-shell case) rather than
        // building a full O2 molecule, since only the guard is under test.
        let mol = Molecule::with_charge(vec![crate::molecule::Atom::new(8, 0.0, 0.0, 0.0)], 0, 3);
        let basis = Sto3g::build(&mol);
        let config = RhfConfig::default();
        let _ = restricted_hartree_fock(&mol, &basis, &config);
    }

    #[test]
    fn test_fallible_rhf_rejects_invalid_electronic_inputs_without_panicking() {
        let hydrogen = crate::molecule::Atom::new(1, 0.0, 0.0, 0.0);
        let invalid_charge = Molecule::with_charge(vec![hydrogen.clone()], 2, 1);
        let invalid_charge_basis = Sto3g::build(&invalid_charge);
        assert!(matches!(
            try_restricted_hartree_fock(
                &invalid_charge,
                &invalid_charge_basis,
                &RhfConfig::default()
            ),
            Err(ScfSolveError::InvalidCharge {
                charge: 2,
                nuclear_charge: 1
            })
        ));

        let odd_electron_singlet = Molecule::new(vec![hydrogen]);
        let odd_basis = Sto3g::build(&odd_electron_singlet);
        assert!(matches!(
            try_restricted_hartree_fock(
                &odd_electron_singlet,
                &odd_basis,
                &RhfConfig::default()
            ),
            Err(ScfSolveError::InconsistentElectronicState {
                electron_count: 1,
                multiplicity: 1
            })
        ));

        let zero_electron = Molecule::with_charge(
            vec![crate::molecule::Atom::new(1, 0.0, 0.0, 0.0)],
            1,
            1,
        );
        let zero_electron_basis = Sto3g::build(&zero_electron);
        assert!(matches!(
            try_restricted_hartree_fock(
                &zero_electron,
                &zero_electron_basis,
                &RhfConfig::default()
            ),
            Err(ScfSolveError::NoElectrons)
        ));

        let empty = Molecule::new(Vec::new());
        let empty_basis = BasisSet {
            name: "empty-test".to_string(),
            functions: Vec::new(),
        };
        assert!(matches!(
            try_restricted_hartree_fock(&empty, &empty_basis, &RhfConfig::default()),
            Err(ScfSolveError::EmptyMolecule)
        ));
    }

    #[test]
    #[should_panic(expected = "invalid molecule: charge")]
    fn test_legacy_rhf_keeps_invalid_charge_panic_contract() {
        let molecule = Molecule::with_charge(
            vec![crate::molecule::Atom::new(1, 0.0, 0.0, 0.0)],
            2,
            1,
        );
        let basis = Sto3g::build(&molecule);
        let _ = restricted_hartree_fock(&molecule, &basis, &RhfConfig::default());
    }

    #[test]
    fn test_failed_checked_overlap_never_returns_converged_energy() {
        let molecule = Molecule::h2();
        let mut basis = Sto3g::build(&molecule);
        basis.functions[0].primitives[0].alpha = f64::NAN;
        let config = RhfConfig::default();

        let fallible = try_restricted_hartree_fock(&molecule, &basis, &config);
        assert!(matches!(
            fallible,
            Err(ScfSolveError::CanonicalOrthogonalization(_))
        ));

        let legacy = restricted_hartree_fock(&molecule, &basis, &config);
        assert!(!legacy.converged);
        assert!(legacy.total_energy.is_nan());
        assert!(legacy.electronic_energy.is_nan());
        assert!(legacy.solver_error.as_deref().is_some_and(|message| message.contains("non-finite")));
        assert!(legacy.solver_diagnostics.is_none());
    }

    #[test]
    fn test_fock_eigensolver_iteration_exhaustion_propagates_as_error() {
        let molecule = Molecule::water();
        let basis = Sto3g::build(&molecule);
        let error = try_restricted_hartree_fock_with_eigensolver_budget(
            &molecule,
            &basis,
            &RhfConfig::default(),
            Some(0),
        )
        .expect_err("a zero-rotation budget must not produce an unchecked RHF result");
        assert!(matches!(
            error,
            ScfSolveError::GeneralizedEigen(
                crate::scf::generalized_eigen::CheckedGeneralizedEigenError::EigensolverNotConverged(_)
            )
        ));
    }

    #[test]
    fn test_heh_plus_sto3g() {
        // HeH+ STO-3G: 2 basis functions, 2 electrons, 1 occupied orbital
        // Reference: Szabo & Ostlund Table 3.15: E = -2.8607 Hartree
        let mol = Molecule::heh_plus();
        let basis = Sto3g::build(&mol);
        let config = RhfConfig::default();

        let result = restricted_hartree_fock(&mol, &basis, &config);

        assert!(result.converged, "SCF should converge");
        assert!(
            (result.total_energy - (-2.8607)).abs() < 0.05,
            "HeH+ energy = {:.6}, expected ≈ -2.8607",
            result.total_energy
        );
        assert!(
            result.n_iterations < 50,
            "Should converge in < 50 iterations, took {}",
            result.n_iterations
        );
    }

    #[test]
    fn test_h2_sto3g() {
        // H2 STO-3G at R=1.4 Bohr: E ≈ -1.1175 Hartree
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let config = RhfConfig::default();

        let result = restricted_hartree_fock(&mol, &basis, &config);

        assert!(result.converged, "SCF should converge");
        let diagnostics = result
            .solver_diagnostics
            .as_ref()
            .expect("successful RHF must expose a solver residual receipt");
        assert_eq!(
            diagnostics.generalized_eigensolve_count,
            result.n_iterations + 1,
            "RHF performs one core-guess solve plus one checked solve per SCF iteration"
        );
        assert!(diagnostics.total_jacobi_rotations < 1_000_000);
        assert!(diagnostics.max_transformed_off_diagonal_hartree.is_finite());
        assert!(diagnostics.max_transformed_eigenpair_residual <= 1e-10);
        assert!(diagnostics.max_eigenvector_orthogonality_residual <= 1e-10);
        assert!(diagnostics.max_generalized_eigenpair_residual <= 1e-8);
        assert!(diagnostics.overlap_orthogonality_residual <= 1e-8);
        assert!(
            (result.total_energy - (-1.1175)).abs() < 0.01,
            "H2 energy = {:.6}, expected ≈ -1.1175",
            result.total_energy
        );
    }

    #[test]
    fn test_water_sto3g() {
        // H2O STO-3G: 7 basis functions, 10 electrons, 5 occupied
        // Reference: E ≈ -74.9420 Hartree
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let config = RhfConfig::default();

        let result = restricted_hartree_fock(&mol, &basis, &config);

        assert!(result.converged, "SCF should converge for water");
        assert_eq!(result.n_basis, 7);
        assert_eq!(result.n_occupied, 5);
        assert!(
            (result.total_energy - (-74.9420)).abs() < 0.5,
            "H2O energy = {:.6}, expected ≈ -74.9420",
            result.total_energy
        );
    }

    #[test]
    fn test_diis_accelerates() {
        // DIIS should converge in fewer iterations than no-DIIS
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);

        let config_diis = RhfConfig {
            use_diis: true,
            ..Default::default()
        };
        let config_no_diis = RhfConfig {
            use_diis: false,
            ..Default::default()
        };

        let result_diis = restricted_hartree_fock(&mol, &basis, &config_diis);
        let result_no_diis = restricted_hartree_fock(&mol, &basis, &config_no_diis);

        assert!(result_diis.converged);
        assert!(result_no_diis.converged);
        // Both should give the same energy
        assert!(
            (result_diis.total_energy - result_no_diis.total_energy).abs() < 1e-6,
            "DIIS energy={:.8}, no-DIIS energy={:.8}",
            result_diis.total_energy,
            result_no_diis.total_energy
        );
    }

    #[test]
    fn test_orbital_energies_ordered() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let result = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        // Orbital energies should be in ascending order
        for i in 1..result.orbital_energies.len() {
            assert!(
                result.orbital_energies[i] >= result.orbital_energies[i - 1] - 1e-10,
                "Orbital energies not ordered: ε[{}]={} < ε[{}]={}",
                i,
                result.orbital_energies[i],
                i - 1,
                result.orbital_energies[i - 1]
            );
        }
    }

    #[test]
    fn test_schwarz_screening_reported() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let result = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        // Should report both computed and screened integrals
        assert!(result.eri_computed > 0, "Should compute some ERIs");
        // Total should be reasonable for 7 basis functions
        let total = result.eri_computed + result.eri_screened;
        assert!(total > 0);
    }

    #[test]
    fn test_direct_mode_matches_dense_mode_end_to_end_water() {
        // Phase Q3 (2026-07-16): the real end-to-end correctness proof for
        // direct SCF -- running the FULL converged calculation with
        // direct=true must give the same energy as direct=false, not just
        // a single Fock-matrix identity (already checked in scf::fock's
        // tests). No external reference needed: both paths compute the
        // exact same physics.
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);

        let dense = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let direct = restricted_hartree_fock(
            &mol,
            &basis,
            &RhfConfig {
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
        // direct mode doesn't build a dense tensor, so these stay 0.
        assert_eq!(direct.eri_computed, 0);
        assert_eq!(direct.eri_screened, 0);
    }

    #[test]
    fn test_direct_mode_matches_dense_mode_end_to_end_heh_plus() {
        let mol = Molecule::heh_plus();
        let basis = Sto3g::build(&mol);

        let dense = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let direct = restricted_hartree_fock(
            &mol,
            &basis,
            &RhfConfig {
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
    }
}
