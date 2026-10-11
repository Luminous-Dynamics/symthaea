// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Kohn-Sham DFT solver.
//!
//! Replaces exact HF exchange with an exchange-correlation functional
//! evaluated on a numerical grid. The KS equations are otherwise identical
//! to Roothaan-Hall: F_KS C = S C ε.

use crate::basis::{BasisSet, ContractedGaussian};
use crate::constants::{
    CANONICAL_ORTH_THRESHOLD, MAX_SCF_ITERATIONS, SCF_DENSITY_THRESHOLD,
    SCF_ENERGY_THRESHOLD,
};
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
use crate::scf::generalized_eigen::{
    ScfSolveError, ScfSolverDiagnostics, canonical_orthogonalization_checked,
    solve_generalized_eigen_checked,
};
#[cfg(test)]
use crate::scf::generalized_eigen::CheckedGeneralizedEigenError;

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
    /// Number of outer SCF iterations attempted.
    /// Zero in a fail-closed compatibility result with `solver_error = Some(...)`.
    pub n_iterations: usize,
    /// Whether the DFT energy/density fixed point met its convergence criteria.
    /// This does not imply chemical accuracy or replace the numerical receipt.
    pub converged: bool,
    pub n_basis: usize,
    /// Number of grid points used by a successful numerical result.
    /// Zero in a fail-closed compatibility result with `solver_error = Some(...)`.
    pub n_grid_points: usize,
    /// Error emitted by the fallible solver when the legacy wrapper must fail closed.
    /// If set, energies are NaN, orbital vectors are empty, `converged` is false,
    /// `n_iterations` and `n_grid_points` are zero, and diagnostics are absent.
    pub solver_error: Option<String>,
    /// Receipts for the checked overlap orthogonalization and every generalized eigensolve.
    /// Independent of the fixed-point convergence flag; absent for a fail-closed error result.
    pub solver_diagnostics: Option<ScfSolverDiagnostics>,
}

/// Reject non-finite inputs and intermediates before they can become a DFT result.
fn ensure_finite_values(stage: &'static str, values: &[f64]) -> Result<(), ScfSolveError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ScfSolveError::NonFiniteNumericValue { stage, index });
    }
    Ok(())
}

/// Validate the integration grid before any quadrature uses its coordinates or weights.
fn validate_dft_grid(grid: &DftGrid) -> Result<(), ScfSolveError> {
    if grid.points.is_empty() {
        return Err(ScfSolveError::EmptyIntegrationGrid);
    }

    for (point_index, point) in grid.points.iter().enumerate() {
        for (axis, value) in [point.x, point.y, point.z, point.weight].iter().enumerate() {
            if !value.is_finite() {
                return Err(ScfSolveError::NonFiniteNumericValue {
                    stage: "DFT integration grid",
                    index: point_index * 4 + axis,
                });
            }
        }

        // The generated Lebedev/radial/Becke grid uses non-negative weights.
        // A negative finite weight is a malformed quadrature rule, not a
        // harmless non-finite-value case.
        if point.weight < 0.0 {
            return Err(ScfSolveError::InvalidNumericInput {
                stage: "DFT integration-grid weight",
                index: point_index,
                value: point.weight,
            });
        }
    }
    Ok(())
}

/// Run a Kohn-Sham DFT calculation through the checked numerical path.
///
/// The legacy return type is retained for compatibility. On a typed numerical
/// failure the wrapper returns NaN energies, empty orbital arrays, and
/// converged=false; callers requiring error handling should use
/// try_kohn_sham_dft. A failed eigensolve can never be represented as a
/// successful-looking finite result.
pub fn kohn_sham_dft(molecule: &Molecule, basis: &BasisSet, config: &DftConfig) -> DftResult {
    // Preserve the legacy invalid-charge panic contract: the original DFT
    // entry point called n_occupied() before doing any integral work.
    let _ = molecule.n_occupied();
    kohn_sham_dft_with_eigensolver_budget(molecule, basis, config, None)
}

fn kohn_sham_dft_with_eigensolver_budget(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &DftConfig,
    eigensolver_iteration_budget: Option<usize>,
) -> DftResult {
    match try_kohn_sham_dft_with_eigensolver_budget(
        molecule,
        basis,
        config,
        eigensolver_iteration_budget,
    ) {
        Ok(result) => result,
        Err(error) => DftResult {
            total_energy: f64::NAN,
            electronic_energy: f64::NAN,
            xc_energy: f64::NAN,
            nuclear_repulsion: f64::NAN,
            orbital_energies: Vec::new(),
            orbital_coefficients: Vec::new(),
            n_iterations: 0,
            converged: false,
            n_basis: basis.n_basis(),
            n_grid_points: 0,
            solver_error: Some(error.to_string()),
            solver_diagnostics: None,
        },
    }
}

/// Fallible LDA Kohn-Sham DFT entry point.
///
/// A successful return means the overlap orthogonalization and every required
/// generalized eigensolve met their numerical contracts. The independent
/// DFT energy/density fixed point is represented by DftResult::converged and
/// is not implied by a successful numerical receipt.
pub fn try_kohn_sham_dft(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &DftConfig,
) -> Result<DftResult, ScfSolveError> {
    try_kohn_sham_dft_with_eigensolver_budget(molecule, basis, config, None)
}

/// Internal iteration-budget override makes eigensolver failure propagation
/// deterministic in tests without changing production tolerances or policy.
fn try_kohn_sham_dft_with_eigensolver_budget(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &DftConfig,
    eigensolver_iteration_budget: Option<usize>,
) -> Result<DftResult, ScfSolveError> {
    if molecule.multiplicity != 1 {
        return Err(ScfSolveError::UnsupportedMultiplicity {
            method: "LDA DFT",
            multiplicity: molecule.multiplicity,
        });
    }

    let electron_count = crate::scf::checked_electron_count(molecule)?;
    if electron_count % 2 != 0 {
        return Err(ScfSolveError::InconsistentElectronicState {
            electron_count,
            multiplicity: molecule.multiplicity,
        });
    }

    // Reject malformed inputs before grid generation or integral construction.
    for (atom_index, atom) in molecule.atoms.iter().enumerate() {
        if atom.atomic_number == 0 || atom.atomic_number > 118 {
            return Err(ScfSolveError::InvalidNumericInput {
                stage: "DFT atomic number",
                index: atom_index,
                value: f64::from(atom.atomic_number),
            });
        }
        for (axis, coordinate) in atom.position.iter().enumerate() {
            if !coordinate.is_finite() {
                return Err(ScfSolveError::NonFiniteNumericValue {
                    stage: "DFT molecular geometry",
                    index: atom_index * 3 + axis,
                });
            }
        }
    }

    let atom_count = molecule.atoms.len();
    for i in 0..atom_count {
        for j in (i + 1)..atom_count {
            let distance = molecule.atoms[i].distance_to(&molecule.atoms[j]);
            if !distance.is_finite() {
                return Err(ScfSolveError::NonFiniteNumericValue {
                    stage: "DFT internuclear distance",
                    index: i * atom_count + j,
                });
            }
        }
    }

    for (index, value) in [
        config.energy_convergence,
        config.density_convergence,
    ]
    .iter()
    .enumerate()
    {
        if !value.is_finite() || *value < 0.0 {
            return Err(ScfSolveError::InvalidNumericInput {
                stage: "DFT convergence tolerance",
                index,
                value: *value,
            });
        }
    }

    let mut primitive_index = 0usize;
    for function in &basis.functions {
        for primitive in &function.primitives {
            let fields = [
                primitive.alpha,
                primitive.coeff,
                primitive.center[0],
                primitive.center[1],
                primitive.center[2],
            ];
            for (field_index, value) in fields.iter().enumerate() {
                if !value.is_finite() {
                    return Err(ScfSolveError::NonFiniteNumericValue {
                        stage: "DFT basis primitive fields",
                        index: primitive_index * fields.len() + field_index,
                    });
                }
            }
            if primitive.alpha <= 0.0 {
                return Err(ScfSolveError::InvalidNumericInput {
                    stage: "DFT basis exponent",
                    index: primitive_index,
                    value: primitive.alpha,
                });
            }
            primitive_index = primitive_index.saturating_add(1);
        }
    }

    let n = basis.n_basis();
    let n_occ = electron_count / 2;
    let v_nn = molecule.nuclear_repulsion_energy();
    if !v_nn.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "DFT nuclear repulsion",
        });
    }

    // Build one-electron terms first. Fail before the expensive ERI build if
    // the basis or the overlap metric is malformed.
    let grid = DftGrid::build(&molecule.atoms, config.grid_quality);
    validate_dft_grid(&grid)?;
    let s_mat = overlap_matrix(&basis.functions);
    let t_mat = kinetic_matrix(&basis.functions);
    let v_mat = nuclear_matrix(&basis.functions, &molecule.atoms);

    let mut h_core = vec![0.0; n * n];
    for i in 0..n * n {
        h_core[i] = t_mat[i] + v_mat[i];
    }
    ensure_finite_values("DFT overlap matrix", &s_mat)?;
    ensure_finite_values("DFT core Hamiltonian", &h_core)?;

    let overlap_iterations = n.saturating_mul(n).saturating_mul(100).max(1);
    let checked_orthogonalization = canonical_orthogonalization_checked(
        &s_mat,
        n,
        CANONICAL_ORTH_THRESHOLD,
        overlap_iterations,
    )?;
    let x_mat = checked_orthogonalization.transformation;
    let n_ind = checked_orthogonalization.n_independent;
    if n_occ > n_ind {
        return Err(ScfSolveError::InsufficientBasisForOccupiedOrbitals {
            occupied_orbitals: n_occ,
            independent_basis_functions: n_ind,
        });
    }

    let eigensolver_iterations = eigensolver_iteration_budget
        .unwrap_or_else(|| n_ind.saturating_mul(n_ind).saturating_mul(100).max(1));
    let mut solver_diagnostics = ScfSolverDiagnostics::from_overlap_residual(
        checked_orthogonalization.orthogonality_residual,
    );

    // Initial guess: the core-Hamiltonian solve is checked independently.
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

    let mut density = build_density_matrix(&coefficients, n, n_ind, n_occ);
    ensure_finite_values("DFT initial density matrix", &density)?;

    // Solve the initial core Hamiltonian before expensive ERI construction.
    // A failed checked eigensolve should fail fast without doing Coulomb work.
    let (eri, _, _) = compute_eri_tensor(&basis.functions);
    ensure_finite_values("DFT ERI tensor", &eri)?;
    let basis_at_grid = evaluate_basis_on_grid(&basis.functions, &grid);
    ensure_finite_values("DFT basis values on grid", &basis_at_grid)?;

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

        let rho = compute_density_at_grid(&density, &basis_at_grid, n, grid.n_points());
        ensure_finite_values("DFT electron density on grid", &rho)?;

        let (_e_xc_unweighted, v_xc) = lda_exchange_correlation(&rho);
        ensure_finite_values("DFT XC potential on grid", &v_xc)?;
        let mut xc_energy = 0.0;
        for (g, pt) in grid.points.iter().enumerate() {
            let rho_g = rho[g];
            if rho_g > 1e-20 {
                xc_energy += pt.weight
                    * crate::dft::lda::SlaterExchange::energy_per_point(rho_g)
                    + pt.weight
                        * crate::dft::lda::VwnCorrelation::energy_per_point(rho_g);
            }
        }
        if !xc_energy.is_finite() {
            return Err(ScfSolveError::NonFiniteEnergy {
                stage: "DFT XC energy during SCF",
            });
        }

        let j_matrix = build_coulomb_matrix(&density, &eri, n);
        ensure_finite_values("DFT Coulomb matrix", &j_matrix)?;
        let v_xc_matrix = build_xc_matrix(&v_xc, &basis_at_grid, &grid, n);
        ensure_finite_values("DFT XC matrix", &v_xc_matrix)?;

        let mut fock = h_core.clone();
        for i in 0..n * n {
            fock[i] += j_matrix[i] + v_xc_matrix[i];
        }
        ensure_finite_values("DFT KS Fock matrix before DIIS", &fock)?;
        if let Some(ref mut diis_engine) = diis {
            fock = diis_engine.extrapolate(&fock, &density, &s_mat);
        }
        ensure_finite_values("DFT KS Fock matrix after DIIS", &fock)?;

        let e_one: f64 = (0..n * n)
            .map(|i| density[i] * h_core[i])
            .sum::<f64>();
        if !e_one.is_finite() {
            return Err(ScfSolveError::NonFiniteEnergy {
                stage: "DFT one-electron energy during SCF",
            });
        }
        let e_j: f64 = (0..n * n)
            .map(|i| density[i] * j_matrix[i])
            .sum::<f64>();
        if !e_j.is_finite() {
            return Err(ScfSolveError::NonFiniteEnergy {
                stage: "DFT Coulomb energy during SCF",
            });
        }
        let e_elec = e_one + 0.5 * e_j + xc_energy;
        if !e_elec.is_finite() {
            return Err(ScfSolveError::NonFiniteEnergy {
                stage: "DFT electronic energy during SCF",
            });
        }
        let e_total = e_elec + v_nn;
        if !e_total.is_finite() {
            return Err(ScfSolveError::NonFiniteEnergy {
                stage: "DFT total energy during SCF",
            });
        }

        // Every SCF Fock diagonalization must satisfy the checked contracts.
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

        let density_new = build_density_matrix(&coefficients, n, n_ind, n_occ);
        ensure_finite_values("DFT updated density matrix", &density_new)?;
        let d_rms = density_rms_change(&density_new, &density, n);
        let de = (e_total - energy_old).abs();
        if !d_rms.is_finite() || !de.is_finite() {
            return Err(ScfSolveError::NonFiniteNumericValue {
                stage: "DFT convergence metrics",
                index: 0,
            });
        }

        density = density_new;
        energy_old = e_total;

        if de < config.energy_convergence && d_rms < config.density_convergence && iter > 0 {
            converged = true;
            break;
        }
    }

    // Recompute the final energy from the returned density and independently
    // validate each component. Fixed-point convergence remains a separate flag.
    let rho_final = compute_density_at_grid(&density, &basis_at_grid, n, grid.n_points());
    ensure_finite_values("DFT final electron density on grid", &rho_final)?;
    let mut e_xc_final = 0.0;
    for (g, pt) in grid.points.iter().enumerate() {
        let rho_g = rho_final[g];
        if rho_g > 1e-20 {
            e_xc_final += pt.weight
                * crate::dft::lda::SlaterExchange::energy_per_point(rho_g)
                + pt.weight * crate::dft::lda::VwnCorrelation::energy_per_point(rho_g);
        }
    }
    if !e_xc_final.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "final DFT XC energy",
        });
    }
    let j_final = build_coulomb_matrix(&density, &eri, n);
    ensure_finite_values("DFT final Coulomb matrix", &j_final)?;
    let e_one: f64 = (0..n * n)
        .map(|i| density[i] * h_core[i])
        .sum::<f64>();
    let e_j: f64 = (0..n * n)
        .map(|i| density[i] * j_final[i])
        .sum::<f64>();
    if !e_one.is_finite() || !e_j.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "final DFT energy components",
        });
    }
    let e_elec = e_one + 0.5 * e_j + e_xc_final;
    if !e_elec.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "final DFT electronic energy",
        });
    }
    let total_energy = e_elec + v_nn;
    if !total_energy.is_finite() {
        return Err(ScfSolveError::NonFiniteEnergy {
            stage: "final DFT total energy",
        });
    }

    Ok(DftResult {
        total_energy,
        electronic_energy: e_elec,
        xc_energy: e_xc_final,
        nuclear_repulsion: v_nn,
        orbital_energies,
        orbital_coefficients: coefficients,
        n_iterations,
        converged,
        n_basis: n,
        n_grid_points: grid.n_points(),
        solver_error: None,
        solver_diagnostics: Some(solver_diagnostics),
    })
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
        // Clamp valid negative roundoff, but preserve NaN/Inf so fail-closed
        // callers can reject it instead of silently turning NaN into zero.
        rho[g] = if val.is_finite() { val.max(0.0) } else { val };
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
    fn try_dft_emits_a_separate_numerical_receipt() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let result = try_kohn_sham_dft(&mol, &basis, &DftConfig::default())
            .expect("H2 checked LDA should solve its numerical subproblems");
        assert!(result.total_energy.is_finite());
        assert!(result.electronic_energy.is_finite());
        assert!(result.xc_energy.is_finite());
        let receipt = result.solver_diagnostics.expect("checked receipt");
        assert_eq!(
            receipt.generalized_eigensolve_count,
            result.n_iterations + 1,
            "receipt must include the core guess and every KS-Fock diagonalization"
        );
        assert!(receipt.overlap_orthogonality_residual.is_finite());
        assert!(receipt.max_transformed_eigenpair_residual.is_finite());
        assert!(receipt.max_eigenvector_orthogonality_residual.is_finite());
        assert!(receipt.max_generalized_eigenpair_residual.is_finite());
        assert!(receipt.overlap_orthogonality_residual <= 1e-8);
        assert!(receipt.max_transformed_eigenpair_residual <= 1e-10);
        assert!(receipt.max_eigenvector_orthogonality_residual <= 1e-10);
        assert!(receipt.max_generalized_eigenpair_residual <= 1e-8);
    }

    #[test]
    fn try_dft_keeps_solver_success_separate_from_scf_convergence() {
        let molecule = Molecule::h2();
        let basis = Sto3g::build(&molecule);
        let config = DftConfig {
            max_iterations: 0,
            ..DftConfig::default()
        };

        let result = try_kohn_sham_dft(&molecule, &basis, &config)
            .expect("the initial checked core solve should succeed independently");
        assert!(!result.converged, "no SCF iterations were requested");
        assert_eq!(result.n_iterations, 0);
        assert!(result.total_energy.is_finite());
        assert!(result.solver_error.is_none());
        assert!(result.n_grid_points > 0);

        let receipt = result
            .solver_diagnostics
            .expect("a successful initial solve must produce a numerical receipt");
        assert_eq!(receipt.generalized_eigensolve_count, 1);
        assert!(receipt.max_generalized_eigenpair_residual.is_finite());
    }

    #[test]
    fn try_dft_rejects_non_finite_overlap_before_eri_or_scf() {
        let mol = Molecule::h2();
        let mut basis = Sto3g::build(&mol);
        // Keep primitive inputs finite and valid, but make the overlap product
        // overflow. This proves the overlap matrix gate, not the primitive
        // input preflight, is the rejection point.
        basis.functions[0].primitives[0].coeff = f64::MAX;

        assert!(matches!(
            try_kohn_sham_dft(&mol, &basis, &DftConfig::default()),
            Err(ScfSolveError::NonFiniteNumericValue {
                stage: "DFT overlap matrix",
                ..
            })
        ));
    }

    #[test]
    fn checked_dft_grid_rejects_empty_and_non_finite_points() {
        assert!(matches!(
            validate_dft_grid(&DftGrid { points: Vec::new() }),
            Err(ScfSolveError::EmptyIntegrationGrid)
        ));

        let bad_grid = DftGrid {
            points: vec![crate::dft::grid::GridPoint {
                x: 0.0,
                y: f64::NAN,
                z: 0.0,
                weight: 1.0,
            }],
        };
        assert!(matches!(
            validate_dft_grid(&bad_grid),
            Err(ScfSolveError::NonFiniteNumericValue {
                stage: "DFT integration grid",
                index: 1
            })
        ));
    }

    #[test]
    fn checked_dft_grid_rejects_negative_quadrature_weights() {
        let grid = DftGrid {
            points: vec![crate::dft::grid::GridPoint {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                weight: -1.0,
            }],
        };

        assert!(matches!(
            validate_dft_grid(&grid),
            Err(ScfSolveError::InvalidNumericInput {
                stage: "DFT integration-grid weight",
                index: 0,
                value,
            }) if value == -1.0
        ));
    }

    #[test]
    fn try_dft_rejects_overflowing_internuclear_distance_before_grid() {
        let mut molecule = Molecule::h2();
        molecule.atoms[0].position[0] = f64::MAX;
        let basis = Sto3g::build(&Molecule::h2());
        assert!(matches!(
            try_kohn_sham_dft(&molecule, &basis, &DftConfig::default()),
            Err(ScfSolveError::NonFiniteNumericValue {
                stage: "DFT internuclear distance",
                ..
            })
        ));
    }

    #[test]
    fn try_dft_rejects_non_finite_geometry_before_grid_generation() {
        let mut molecule = Molecule::h2();
        molecule.atoms[0].position[0] = f64::NAN;
        let basis = Sto3g::build(&Molecule::h2());

        assert!(matches!(
            try_kohn_sham_dft(&molecule, &basis, &DftConfig::default()),
            Err(ScfSolveError::NonFiniteNumericValue {
                stage: "DFT molecular geometry",
                index: 0
            })
        ));
    }

    #[test]
    fn try_dft_rejects_invalid_basis_exponent_and_convergence_tolerance() {
        let molecule = Molecule::h2();
        let mut basis = Sto3g::build(&molecule);
        basis.functions[0].primitives[0].alpha = 0.0;
        assert!(matches!(
            try_kohn_sham_dft(&molecule, &basis, &DftConfig::default()),
            Err(ScfSolveError::InvalidNumericInput {
                stage: "DFT basis exponent",
                index: 0,
                value,
            }) if value == 0.0
        ));

        let basis = Sto3g::build(&molecule);
        let config = DftConfig {
            energy_convergence: f64::NAN,
            ..DftConfig::default()
        };
        assert!(matches!(
            try_kohn_sham_dft(&molecule, &basis, &config),
            Err(ScfSolveError::InvalidNumericInput {
                stage: "DFT convergence tolerance",
                index: 0,
                value
            }) if value.is_nan()
        ));
    }

    #[test]
    fn try_dft_rejects_open_shell_and_invalid_charge() {
        let open_shell = Molecule::with_charge(
            vec![crate::molecule::Atom::new(1, 0.0, 0.0, 0.0)],
            0,
            2,
        );
        let open_basis = Sto3g::build(&open_shell);
        assert!(matches!(
            try_kohn_sham_dft(&open_shell, &open_basis, &DftConfig::default()),
            Err(ScfSolveError::UnsupportedMultiplicity { .. })
        ));

        let invalid_charge = Molecule::with_charge(
            vec![crate::molecule::Atom::new(1, 0.0, 0.0, 0.0)],
            2,
            1,
        );
        let invalid_basis = Sto3g::build(&invalid_charge);
        assert!(matches!(
            try_kohn_sham_dft(&invalid_charge, &invalid_basis, &DftConfig::default()),
            Err(ScfSolveError::InvalidCharge { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "invalid molecule: charge 2 exceeds total nuclear charge 1")]
    fn legacy_dft_preserves_invalid_charge_panic_contract() {
        let molecule = Molecule::with_charge(
            vec![crate::molecule::Atom::new(1, 0.0, 0.0, 0.0)],
            2,
            1,
        );
        let basis = Sto3g::build(&molecule);
        let _ = kohn_sham_dft(&molecule, &basis, &DftConfig::default());
    }

    #[test]
    fn try_dft_propagates_eigensolver_exhaustion_to_legacy_wrapper() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let config = DftConfig {
            grid_quality: GridQuality::Coarse,
            ..DftConfig::default()
        };
        let error = try_kohn_sham_dft_with_eigensolver_budget(&mol, &basis, &config, Some(0))
            .expect_err("zero rotation budget must not produce an unchecked DFT result");
        assert!(matches!(
            error,
            ScfSolveError::GeneralizedEigen(
                CheckedGeneralizedEigenError::EigensolverNotConverged(_)
            )
        ));

        let legacy = kohn_sham_dft_with_eigensolver_budget(&mol, &basis, &config, Some(0));
        assert!(!legacy.converged);
        assert!(legacy.total_energy.is_nan());
        assert!(legacy.electronic_energy.is_nan());
        assert!(legacy.xc_energy.is_nan());
        assert!(legacy.orbital_energies.is_empty());
        assert!(legacy.orbital_coefficients.is_empty());
        assert!(legacy.solver_error.is_some());
        assert!(legacy.solver_diagnostics.is_none());
    }

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
