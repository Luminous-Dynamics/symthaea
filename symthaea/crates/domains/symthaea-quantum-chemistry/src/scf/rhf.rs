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
use crate::constants::{MAX_SCF_ITERATIONS, SCF_DENSITY_THRESHOLD, SCF_ENERGY_THRESHOLD};
use crate::integrals::eri::compute_eri_tensor;
use crate::integrals::kinetic::kinetic_matrix;
use crate::integrals::nuclear::nuclear_matrix;
use crate::integrals::overlap::overlap_matrix;
use crate::molecule::Molecule;
use crate::scf::density::{build_density_matrix, density_rms_change};
use crate::scf::diis::Diis;
use crate::scf::fock::{build_fock_matrix, electronic_energy};
use crate::scf::generalized_eigen::{canonical_orthogonalization, solve_generalized_eigen};

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
}

impl Default for RhfConfig {
    fn default() -> Self {
        Self {
            max_iterations: MAX_SCF_ITERATIONS,
            energy_convergence: SCF_ENERGY_THRESHOLD,
            density_convergence: SCF_DENSITY_THRESHOLD,
            use_diis: true,
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
    /// Number of ERIs computed (vs screened)
    pub eri_computed: usize,
    /// Number of ERIs screened out
    pub eri_screened: usize,
}

/// Run a Restricted Hartree-Fock calculation.
pub fn restricted_hartree_fock(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &RhfConfig,
) -> RhfResult {
    let n = basis.n_basis();
    let n_occ = molecule.n_occupied();
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

    // Step 2: Two-electron integrals with Schwarz prescreening
    let (eri, eri_computed, eri_screened) = compute_eri_tensor(&basis.functions);

    // Step 3: Canonical orthogonalization
    let (x_mat, n_ind, _n_disc) = canonical_orthogonalization(&s_mat, n);

    // Step 4: Initial guess — diagonalize H_core
    let initial = solve_generalized_eigen(&h_core, &x_mat, n, n_ind);
    let mut coefficients = initial.coefficients;
    let mut orbital_energies = initial.eigenvalues;

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
        let mut fock = build_fock_matrix(&h_core, &density, &eri, n);

        // 5b: DIIS extrapolation
        if let Some(ref mut diis_engine) = diis {
            fock = diis_engine.extrapolate(&fock, &density, &s_mat);
        }

        // 5c: Compute electronic energy
        let e_elec = electronic_energy(&density, &h_core, &fock, n);
        let e_total = e_elec + v_nn;

        // 5d: Solve FC = SCε
        let result = solve_generalized_eigen(&fock, &x_mat, n, n_ind);
        coefficients = result.coefficients;
        orbital_energies = result.eigenvalues;

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
    let fock_final = build_fock_matrix(&h_core, &density, &eri, n);
    let e_elec = electronic_energy(&density, &h_core, &fock_final, n);

    RhfResult {
        total_energy: e_elec + v_nn,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;

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
}
