// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Kohn-Sham DFT solver.
//!
//! Replaces exact HF exchange with an exchange-correlation functional
//! evaluated on a numerical grid. The KS equations are otherwise identical
//! to Roothaan-Hall: F_KS C = S C ε.

use crate::basis::{BasisSet, ContractedGaussian};
use crate::constants::{MAX_SCF_ITERATIONS, SCF_DENSITY_THRESHOLD, SCF_ENERGY_THRESHOLD};
use crate::dft::grid::{DftGrid, GridQuality};
use crate::dft::lda::lda_exchange_correlation;
use crate::integrals::eri::compute_eri_tensor;
use crate::integrals::kinetic::kinetic_matrix;
use crate::integrals::nuclear::nuclear_matrix;
use crate::integrals::overlap::overlap_matrix;
use crate::molecule::Molecule;
use crate::scf::density::{build_density_matrix, density_rms_change};
use crate::scf::diis::Diis;
use crate::scf::generalized_eigen::{canonical_orthogonalization, solve_generalized_eigen};

/// Available exchange-correlation functionals.
#[derive(Debug, Clone, Copy)]
pub enum XcFunctional {
    /// Local Density Approximation (Slater + VWN)
    Lda,
}

/// DFT configuration.
#[derive(Debug, Clone)]
pub struct DftConfig {
    pub functional: XcFunctional,
    pub grid_quality: GridQuality,
    pub max_iterations: usize,
    pub energy_convergence: f64,
    pub density_convergence: f64,
    pub use_diis: bool,
}

impl Default for DftConfig {
    fn default() -> Self {
        Self {
            functional: XcFunctional::Lda,
            grid_quality: GridQuality::Medium,
            max_iterations: MAX_SCF_ITERATIONS,
            energy_convergence: SCF_ENERGY_THRESHOLD,
            density_convergence: SCF_DENSITY_THRESHOLD,
            use_diis: true,
        }
    }
}

/// Result of a DFT calculation.
#[derive(Debug, Clone)]
pub struct DftResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub xc_energy: f64,
    pub nuclear_repulsion: f64,
    pub orbital_energies: Vec<f64>,
    pub orbital_coefficients: Vec<f64>,
    pub n_iterations: usize,
    pub converged: bool,
    pub n_basis: usize,
    pub n_grid_points: usize,
}

/// Run a Kohn-Sham DFT calculation.
///
/// The KS Fock matrix is: F_KS = H_core + J + V_xc
/// where J is the Coulomb operator (from ERIs) and V_xc replaces exact exchange.
pub fn kohn_sham_dft(molecule: &Molecule, basis: &BasisSet, config: &DftConfig) -> DftResult {
    let n = basis.n_basis();
    let n_occ = molecule.n_occupied();
    let v_nn = molecule.nuclear_repulsion_energy();

    // Build grid
    let grid = DftGrid::build(&molecule.atoms, config.grid_quality);

    // One-electron integrals
    let s_mat = overlap_matrix(&basis.functions);
    let t_mat = kinetic_matrix(&basis.functions);
    let v_mat = nuclear_matrix(&basis.functions, &molecule.atoms);

    let mut h_core = vec![0.0; n * n];
    for i in 0..n * n {
        h_core[i] = t_mat[i] + v_mat[i];
    }

    // Two-electron integrals (for Coulomb J)
    let (eri, _, _) = compute_eri_tensor(&basis.functions);

    // Evaluate basis functions at grid points
    let basis_at_grid = evaluate_basis_on_grid(&basis.functions, &grid);

    // Canonical orthogonalization
    let (x_mat, n_ind, _) = canonical_orthogonalization(&s_mat, n);

    // Initial guess: diagonalize H_core
    let initial = solve_generalized_eigen(&h_core, &x_mat, n, n_ind);
    let mut coefficients = initial.coefficients;
    let mut orbital_energies = initial.eigenvalues;

    // SCF loop
    let mut density = build_density_matrix(&coefficients, n, n_ind, n_occ);
    let mut energy_old = 0.0;
    let mut converged = false;
    let mut n_iterations = 0;
    let mut xc_energy = 0.0;

    let mut diis = if config.use_diis {
        Some(Diis::new(n))
    } else {
        None
    };

    for iter in 0..config.max_iterations {
        n_iterations = iter + 1;

        // Compute electron density at grid points
        let rho = compute_density_at_grid(&density, &basis_at_grid, n, grid.n_points());

        // Compute XC energy and potential on grid
        // E_xc = Σ_g w_g × ε_xc(ρ_g) × ρ_g
        let (e_xc_unweighted, v_xc) = lda_exchange_correlation(&rho);
        // Weight the XC energy by grid weights
        xc_energy = 0.0;
        for (g, pt) in grid.points.iter().enumerate() {
            let rho_g = rho[g];
            if rho_g > 1e-20 {
                xc_energy += pt.weight * crate::dft::lda::SlaterExchange::energy_per_point(rho_g)
                    + pt.weight * crate::dft::lda::VwnCorrelation::energy_per_point(rho_g);
            }
        }

        // Build Coulomb matrix J_μν = Σ_λσ P_λσ (μν|λσ)
        let j_matrix = build_coulomb_matrix(&density, &eri, n);

        // Build XC potential matrix from grid
        let v_xc_matrix = build_xc_matrix(&v_xc, &basis_at_grid, &grid, n);

        // KS Fock matrix: F = H_core + J + V_xc (no exact exchange K)
        let mut fock = h_core.clone();
        for i in 0..n * n {
            fock[i] += j_matrix[i] + v_xc_matrix[i];
        }

        // DIIS
        if let Some(ref mut diis_engine) = diis {
            fock = diis_engine.extrapolate(&fock, &density, &s_mat);
        }

        // Electronic energy: E = Tr[P*H_core] + ½Tr[P*J] + E_xc
        let e_one: f64 = (0..n * n).map(|i| density[i] * h_core[i]).sum::<f64>();
        let e_j: f64 = (0..n * n).map(|i| density[i] * j_matrix[i]).sum::<f64>();
        let e_elec = e_one + 0.5 * e_j + xc_energy;
        let e_total = e_elec + v_nn;

        // Solve KS equations
        let result = solve_generalized_eigen(&fock, &x_mat, n, n_ind);
        coefficients = result.coefficients;
        orbital_energies = result.eigenvalues;

        // New density
        let density_new = build_density_matrix(&coefficients, n, n_ind, n_occ);
        let d_rms = density_rms_change(&density_new, &density, n);
        let de = (e_total - energy_old).abs();

        density = density_new;
        energy_old = e_total;

        if de < config.energy_convergence && d_rms < config.density_convergence && iter > 0 {
            converged = true;
            break;
        }
    }

    // Final energy with properly weighted XC
    let rho_final = compute_density_at_grid(&density, &basis_at_grid, n, grid.n_points());
    let mut e_xc_final = 0.0;
    for (g, pt) in grid.points.iter().enumerate() {
        let rho_g = rho_final[g];
        if rho_g > 1e-20 {
            e_xc_final += pt.weight * crate::dft::lda::SlaterExchange::energy_per_point(rho_g)
                + pt.weight * crate::dft::lda::VwnCorrelation::energy_per_point(rho_g);
        }
    }
    let j_final = build_coulomb_matrix(&density, &eri, n);
    let e_one: f64 = (0..n * n).map(|i| density[i] * h_core[i]).sum::<f64>();
    let e_j: f64 = (0..n * n).map(|i| density[i] * j_final[i]).sum::<f64>();
    let e_elec = e_one + 0.5 * e_j + e_xc_final;

    DftResult {
        total_energy: e_elec + v_nn,
        electronic_energy: e_elec,
        xc_energy: e_xc_final,
        nuclear_repulsion: v_nn,
        orbital_energies,
        orbital_coefficients: coefficients,
        n_iterations,
        converged,
        n_basis: n,
        n_grid_points: grid.n_points(),
    }
}

/// Evaluate all basis functions at all grid points.
/// Returns basis_at_grid[point_idx * n_basis + basis_idx].
fn evaluate_basis_on_grid(basis: &[ContractedGaussian], grid: &DftGrid) -> Vec<f64> {
    let n = basis.len();
    let n_pts = grid.n_points();
    let mut vals = vec![0.0; n_pts * n];

    for (g, pt) in grid.points.iter().enumerate() {
        for (mu, func) in basis.iter().enumerate() {
            let mut val = 0.0;
            for prim in &func.primitives {
                let dx = pt.x - prim.center[0];
                let dy = pt.y - prim.center[1];
                let dz = pt.z - prim.center[2];
                let r2 = dx * dx + dy * dy + dz * dz;

                let angular = dx.powi(prim.l as i32)
                    * dy.powi(prim.m as i32)
                    * dz.powi(prim.n as i32);

                val += prim.coeff * prim.normalization() * angular * (-prim.alpha * r2).exp();
            }
            vals[g * n + mu] = val;
        }
    }

    vals
}

/// Compute electron density at grid points from density matrix.
/// ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)
fn compute_density_at_grid(
    density: &[f64],
    basis_at_grid: &[f64],
    n: usize,
    n_points: usize,
) -> Vec<f64> {
    let mut rho = vec![0.0; n_points];

    for g in 0..n_points {
        let mut val = 0.0;
        for mu in 0..n {
            let phi_mu = basis_at_grid[g * n + mu];
            for nu in 0..n {
                let phi_nu = basis_at_grid[g * n + nu];
                val += density[mu * n + nu] * phi_mu * phi_nu;
            }
        }
        rho[g] = val.max(0.0); // Density must be non-negative
    }

    rho
}

/// Build the Coulomb matrix: J_μν = Σ_λσ P_λσ (μν|λσ)
fn build_coulomb_matrix(density: &[f64], eri: &[f64], n: usize) -> Vec<f64> {
    let n2 = n * n;
    let n3 = n2 * n;
    let mut j = vec![0.0; n * n];

    for mu in 0..n {
        for nu in mu..n {
            let mut val = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    val += density[lam * n + sig] * eri[mu * n3 + nu * n2 + lam * n + sig];
                }
            }
            j[mu * n + nu] = val;
            j[nu * n + mu] = val;
        }
    }

    j
}

/// Build the XC potential matrix: V_xc_μν = Σ_g w_g × v_xc(r_g) × φ_μ(r_g) × φ_ν(r_g)
fn build_xc_matrix(
    v_xc: &[f64],
    basis_at_grid: &[f64],
    grid: &DftGrid,
    n: usize,
) -> Vec<f64> {
    let mut mat = vec![0.0; n * n];

    for (g, pt) in grid.points.iter().enumerate() {
        let vxc_g = v_xc[g];
        let w_g = pt.weight;
        let factor = vxc_g * w_g;

        if factor.abs() < 1e-20 {
            continue;
        }

        for mu in 0..n {
            let phi_mu = basis_at_grid[g * n + mu];
            if phi_mu.abs() < 1e-15 {
                continue;
            }
            for nu in mu..n {
                let phi_nu = basis_at_grid[g * n + nu];
                let val = factor * phi_mu * phi_nu;
                mat[mu * n + nu] += val;
                if mu != nu {
                    mat[nu * n + mu] += val;
                }
            }
        }
    }

    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::sto3g::Sto3g;
    use crate::basis::BasisSetProvider;

    #[test]
    fn test_dft_h2_converges() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let config = DftConfig::default();

        let result = kohn_sham_dft(&mol, &basis, &config);

        assert!(result.converged, "KS-DFT should converge for H2");
        // With Coulomb J + XC, energy should be in a physical range
        // LDA typically overbinds slightly vs HF
        assert!(
            result.total_energy < 0.0 && result.total_energy > -3.0,
            "H2 DFT/LDA energy = {:.4}, expected in [-3, 0]",
            result.total_energy
        );
    }

    #[test]
    fn test_dft_water_converges() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let config = DftConfig {
            grid_quality: GridQuality::Coarse,
            ..Default::default()
        };

        let result = kohn_sham_dft(&mol, &basis, &config);

        assert!(result.converged, "KS-DFT should converge for H2O");
        // LDA/STO-3G water should be in [-80, -60] Hartree range
        assert!(
            result.total_energy < -50.0 && result.total_energy > -100.0,
            "H2O DFT/LDA energy = {:.4}, expected in [-100, -50]",
            result.total_energy
        );
    }

    #[test]
    fn test_dft_xc_energy_negative() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let result = kohn_sham_dft(&mol, &basis, &DftConfig::default());

        assert!(
            result.xc_energy < 0.0,
            "XC energy should be negative: {:.6}",
            result.xc_energy
        );
    }

    #[test]
    fn test_dft_grid_points_reported() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let config = DftConfig {
            grid_quality: GridQuality::Coarse,
            ..Default::default()
        };
        let result = kohn_sham_dft(&mol, &basis, &config);

        assert!(
            result.n_grid_points > 50,
            "Should have reasonable grid: {} points",
            result.n_grid_points
        );
    }
}
