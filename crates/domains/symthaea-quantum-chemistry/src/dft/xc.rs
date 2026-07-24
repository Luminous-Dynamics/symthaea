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
use crate::dft::pbe::PbeExchange;
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
    let mut xc_energy: f64;

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
        let (_e_xc_unweighted, v_xc) = lda_exchange_correlation(&rho);
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

                let angular =
                    dx.powi(prim.l as i32) * dy.powi(prim.m as i32) * dz.powi(prim.n as i32);

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

/// Evaluate the gradient of every basis function at every grid point
/// (Phase Q5d, 2026-07-17). Analytic derivative of the same
/// Cartesian-Gaussian primitive form `evaluate_basis_on_grid` uses:
/// `φ = C·N·dx^l·dy^m·dz^n·exp(-α r²)`, so
/// `∂φ/∂x = C·N·exp(-α r²)·dy^m·dz^n·[l·dx^(l-1) - 2α·dx^(l+1)]`
/// (and analogously for y, z; the `l·dx^(l-1)` term is dropped when
/// `l=0`, since `x^(-1)` isn't meaningful there and the derivative of a
/// constant angular factor is genuinely zero). Returns `(d/dx, d/dy, d/dz)`
/// in the same `[point_idx * n_basis + basis_idx]` layout as
/// `evaluate_basis_on_grid`. Verified against a finite-difference of
/// `evaluate_basis_on_grid` itself in tests.
pub(crate) fn evaluate_basis_gradient_on_grid(
    basis: &[ContractedGaussian],
    grid: &DftGrid,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = basis.len();
    let n_pts = grid.n_points();
    let mut dx_vals = vec![0.0; n_pts * n];
    let mut dy_vals = vec![0.0; n_pts * n];
    let mut dz_vals = vec![0.0; n_pts * n];

    for (g, pt) in grid.points.iter().enumerate() {
        for (mu, func) in basis.iter().enumerate() {
            let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
            for prim in &func.primitives {
                let dx = pt.x - prim.center[0];
                let dy = pt.y - prim.center[1];
                let dz = pt.z - prim.center[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let (l, m, k) = (prim.l as i32, prim.m as i32, prim.n as i32);

                let dx_l = dx.powi(l);
                let dy_m = dy.powi(m);
                let dz_k = dz.powi(k);
                let expo = (-prim.alpha * r2).exp();
                let c = prim.coeff * prim.normalization();

                let dangular_dx = if l > 0 {
                    l as f64 * dx.powi(l - 1) - 2.0 * prim.alpha * dx.powi(l + 1)
                } else {
                    -2.0 * prim.alpha * dx.powi(l + 1)
                };
                let dangular_dy = if m > 0 {
                    m as f64 * dy.powi(m - 1) - 2.0 * prim.alpha * dy.powi(m + 1)
                } else {
                    -2.0 * prim.alpha * dy.powi(m + 1)
                };
                let dangular_dz = if k > 0 {
                    k as f64 * dz.powi(k - 1) - 2.0 * prim.alpha * dz.powi(k + 1)
                } else {
                    -2.0 * prim.alpha * dz.powi(k + 1)
                };

                gx += c * expo * dy_m * dz_k * dangular_dx;
                gy += c * expo * dx_l * dz_k * dangular_dy;
                gz += c * expo * dx_l * dy_m * dangular_dz;
            }
            dx_vals[g * n + mu] = gx;
            dy_vals[g * n + mu] = gy;
            dz_vals[g * n + mu] = gz;
        }
    }

    (dx_vals, dy_vals, dz_vals)
}

/// Compute the electron density gradient at every grid point (Phase Q5d,
/// 2026-07-17): `∇ρ = 2 Σ_μν P_μν φ_μ ∇φ_ν` (density matrix is symmetric,
/// so the `μ↔ν` sum doubles rather than needing both `φ_μ∇φ_ν` and
/// `φ_ν∇φ_μ` computed separately). Verified against a finite-difference of
/// `compute_density_at_grid` itself in tests.
pub(crate) fn compute_density_gradient_at_grid(
    density: &[f64],
    basis_at_grid: &[f64],
    grad_x: &[f64],
    grad_y: &[f64],
    grad_z: &[f64],
    n: usize,
    n_points: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut drho_dx = vec![0.0; n_points];
    let mut drho_dy = vec![0.0; n_points];
    let mut drho_dz = vec![0.0; n_points];

    for g in 0..n_points {
        let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
        for mu in 0..n {
            let phi_mu = basis_at_grid[g * n + mu];
            for nu in 0..n {
                let p_munu = density[mu * n + nu];
                gx += p_munu * phi_mu * grad_x[g * n + nu];
                gy += p_munu * phi_mu * grad_y[g * n + nu];
                gz += p_munu * phi_mu * grad_z[g * n + nu];
            }
        }
        drho_dx[g] = 2.0 * gx;
        drho_dy[g] = 2.0 * gy;
        drho_dz[g] = 2.0 * gz;
    }

    (drho_dx, drho_dy, drho_dz)
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
fn build_xc_matrix(v_xc: &[f64], basis_at_grid: &[f64], grid: &DftGrid, n: usize) -> Vec<f64> {
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

/// Non-self-consistent ("post-hoc") PBE exchange energy on an
/// already-converged density (Phase Q5d, 2026-07-17) -- e.g. the density
/// matrix from a converged `kohn_sham_dft` LDA calculation or
/// `restricted_hartree_fock` result. Evaluates `PbeExchange::energy_density`
/// at every grid point using the real analytic density gradient
/// (`evaluate_basis_gradient_on_grid` + `compute_density_gradient_at_grid`),
/// not a self-consistent GGA SCF cycle -- see `dft::pbe`'s module doc for
/// why (the SCF Fock-matrix build would need an additional
/// gradient-coupling term, not implemented). The returned energy is a real
/// physical quantity (a genuine PBE-exchange energy evaluation on that
/// density), just not the energy of a density that has been
/// self-consistently relaxed under the PBE exchange potential itself.
pub fn pbe_exchange_energy_posthoc(
    molecule: &Molecule,
    basis: &BasisSet,
    density: &[f64],
    grid_quality: GridQuality,
) -> f64 {
    let n = basis.n_basis();
    let grid = DftGrid::build(&molecule.atoms, grid_quality);
    let n_points = grid.n_points();

    let basis_at_grid = evaluate_basis_on_grid(&basis.functions, &grid);
    let (grad_x, grad_y, grad_z) = evaluate_basis_gradient_on_grid(&basis.functions, &grid);

    let rho = compute_density_at_grid(density, &basis_at_grid, n, n_points);
    let (drho_dx, drho_dy, drho_dz) = compute_density_gradient_at_grid(
        density,
        &basis_at_grid,
        &grad_x,
        &grad_y,
        &grad_z,
        n,
        n_points,
    );

    let mut e_x = 0.0;
    for (g, pt) in grid.points.iter().enumerate() {
        let rho_g = rho[g];
        if rho_g > 1e-20 {
            let sigma_g =
                drho_dx[g] * drho_dx[g] + drho_dy[g] * drho_dy[g] + drho_dz[g] * drho_dz[g];
            e_x += pt.weight * PbeExchange::energy_per_point(rho_g, sigma_g);
        }
    }

    e_x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;

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

    // ── Phase Q5d (2026-07-17): gradient infrastructure, verified via
    // finite-difference cross-checks against the already-existing,
    // already-tested value-only functions ──────────────────────────────

    fn single_point_grid(x: f64, y: f64, z: f64) -> DftGrid {
        DftGrid {
            points: vec![crate::dft::grid::GridPoint {
                x,
                y,
                z,
                weight: 1.0,
            }],
        }
    }

    #[test]
    fn test_basis_gradient_matches_finite_difference() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let h = 1e-5;
        let n = basis.n_basis();

        for &center in &[
            [0.3, 0.1, -0.2],
            [1.2, 0.0, 0.0],
            [-0.5, 0.5, 0.8],
            [0.0, 0.0, 1.5],
        ] {
            let grid_c = single_point_grid(center[0], center[1], center[2]);
            let (gx, gy, gz) = evaluate_basis_gradient_on_grid(&basis.functions, &grid_c);

            for dim in 0..3 {
                let mut plus = center;
                let mut minus = center;
                plus[dim] += h;
                minus[dim] -= h;
                let phi_plus = evaluate_basis_on_grid(
                    &basis.functions,
                    &single_point_grid(plus[0], plus[1], plus[2]),
                );
                let phi_minus = evaluate_basis_on_grid(
                    &basis.functions,
                    &single_point_grid(minus[0], minus[1], minus[2]),
                );
                let analytic = match dim {
                    0 => &gx,
                    1 => &gy,
                    _ => &gz,
                };
                for mu in 0..n {
                    let fd = (phi_plus[mu] - phi_minus[mu]) / (2.0 * h);
                    assert!(
                        (fd - analytic[mu]).abs() < 1e-6,
                        "center={center:?} dim={dim} mu={mu}: analytic={}, finite-diff={fd}",
                        analytic[mu]
                    );
                }
            }
        }
    }

    #[test]
    fn test_density_gradient_matches_finite_difference() {
        use crate::scf::density::build_density_matrix;
        use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let density = build_density_matrix(
            &rhf.orbital_coefficients,
            rhf.n_basis,
            rhf.n_independent,
            rhf.n_occupied,
        );
        let n = basis.n_basis();
        let h = 1e-5;

        for &center in &[[0.3, 0.1, -0.2], [0.5, 0.5, 0.5]] {
            let grid_c = single_point_grid(center[0], center[1], center[2]);
            let basis_c = evaluate_basis_on_grid(&basis.functions, &grid_c);
            let (gx, gy, gz) = evaluate_basis_gradient_on_grid(&basis.functions, &grid_c);
            let (drho_dx, drho_dy, drho_dz) =
                compute_density_gradient_at_grid(&density, &basis_c, &gx, &gy, &gz, n, 1);
            let analytic = [drho_dx[0], drho_dy[0], drho_dz[0]];

            for dim in 0..3 {
                let mut plus = center;
                let mut minus = center;
                plus[dim] += h;
                minus[dim] -= h;
                let basis_plus = evaluate_basis_on_grid(
                    &basis.functions,
                    &single_point_grid(plus[0], plus[1], plus[2]),
                );
                let basis_minus = evaluate_basis_on_grid(
                    &basis.functions,
                    &single_point_grid(minus[0], minus[1], minus[2]),
                );
                let rho_plus = compute_density_at_grid(&density, &basis_plus, n, 1)[0];
                let rho_minus = compute_density_at_grid(&density, &basis_minus, n, 1)[0];
                let fd = (rho_plus - rho_minus) / (2.0 * h);
                assert!(
                    (fd - analytic[dim]).abs() < 1e-5,
                    "center={center:?} dim={dim}: analytic={}, finite-diff={fd}",
                    analytic[dim]
                );
            }
        }
    }

    #[test]
    fn test_pbe_exchange_posthoc_runs_and_is_negative() {
        use crate::scf::density::build_density_matrix;
        use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let density = build_density_matrix(
            &rhf.orbital_coefficients,
            rhf.n_basis,
            rhf.n_independent,
            rhf.n_occupied,
        );

        let e_pbe_x = pbe_exchange_energy_posthoc(&mol, &basis, &density, GridQuality::Coarse);
        assert!(
            e_pbe_x < 0.0,
            "PBE exchange energy should be negative: {e_pbe_x}"
        );

        // Compare against the existing LDA/Slater exchange on the same
        // converged density: |E_x^PBE| >= |E_x^LDA| always (real
        // inequality, see PbeExchange's own module-level tests).
        let grid = DftGrid::build(&mol.atoms, GridQuality::Coarse);
        let n = basis.n_basis();
        let basis_at_grid = evaluate_basis_on_grid(&basis.functions, &grid);
        let rho = compute_density_at_grid(&density, &basis_at_grid, n, grid.n_points());
        let mut e_lda_x = 0.0;
        for (g, pt) in grid.points.iter().enumerate() {
            e_lda_x += pt.weight * crate::dft::lda::SlaterExchange::energy_per_point(rho[g]);
        }
        assert!(
            e_pbe_x.abs() >= e_lda_x.abs(),
            "|E_x^PBE|={} should be >= |E_x^LDA|={}",
            e_pbe_x.abs(),
            e_lda_x.abs()
        );
    }
}
