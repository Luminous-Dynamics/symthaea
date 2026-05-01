// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Molecular geometry optimization.
//!
//! Finds equilibrium structures by minimizing the total energy as a function
//! of nuclear coordinates. Uses finite-difference gradients with a simple
//! steepest descent optimizer (L-BFGS planned for Phase 3b).
//!
//! The potential energy surface (PES) is the total energy E(R) where R is
//! the set of all nuclear coordinates.

use crate::basis::sto3g::Sto3g;
use crate::basis::BasisSetProvider;
use crate::molecule::Molecule;
use crate::scf::rhf::{restricted_hartree_fock, RhfConfig};

/// Configuration for geometry optimization.
#[derive(Debug, Clone)]
pub struct GeomOptConfig {
    /// Maximum optimization steps
    pub max_steps: usize,
    /// Gradient convergence threshold (Hartree/Bohr)
    pub gradient_threshold: f64,
    /// Energy convergence threshold (Hartree)
    pub energy_threshold: f64,
    /// Finite-difference step size (Bohr)
    pub fd_step: f64,
    /// Step size for coordinate update (Bohr)
    pub step_size: f64,
    /// SCF configuration for each energy evaluation
    pub scf_config: RhfConfig,
}

impl Default for GeomOptConfig {
    fn default() -> Self {
        Self {
            max_steps: 50,
            gradient_threshold: 1e-4,
            energy_threshold: 1e-6,
            fd_step: 0.005,
            step_size: 0.3,
            scf_config: RhfConfig::default(),
        }
    }
}

/// Result of a geometry optimization.
#[derive(Debug, Clone)]
pub struct GeomOptResult {
    /// Optimized molecule
    pub molecule: Molecule,
    /// Final energy (Hartree)
    pub energy: f64,
    /// Number of optimization steps
    pub n_steps: usize,
    /// Whether the optimization converged
    pub converged: bool,
    /// Final RMS gradient (Hartree/Bohr)
    pub rms_gradient: f64,
    /// Energy at each step
    pub energy_history: Vec<f64>,
}

/// Compute the energy of a molecule (single-point HF/STO-3G).
fn compute_energy(mol: &Molecule, config: &RhfConfig) -> f64 {
    let basis = Sto3g::build(mol);
    let result = restricted_hartree_fock(mol, &basis, config);
    result.total_energy
}

/// Compute the gradient of the energy with respect to nuclear coordinates
/// using central finite differences.
///
/// Returns a flat vector: [dE/dx₁, dE/dy₁, dE/dz₁, dE/dx₂, ...]
fn compute_gradient(mol: &Molecule, config: &RhfConfig, fd_step: f64) -> Vec<f64> {
    let n_atoms = mol.n_atoms();
    let mut gradient = vec![0.0; 3 * n_atoms];

    for atom_idx in 0..n_atoms {
        for coord in 0..3 {
            // Forward displacement
            let mut mol_plus = mol.clone();
            mol_plus.atoms[atom_idx].position[coord] += fd_step;
            let e_plus = compute_energy(&mol_plus, config);

            // Backward displacement
            let mut mol_minus = mol.clone();
            mol_minus.atoms[atom_idx].position[coord] -= fd_step;
            let e_minus = compute_energy(&mol_minus, config);

            // Central difference
            gradient[atom_idx * 3 + coord] = (e_plus - e_minus) / (2.0 * fd_step);
        }
    }

    gradient
}

/// Optimize molecular geometry using steepest descent with line search.
pub fn optimize_geometry(molecule: &Molecule, config: &GeomOptConfig) -> GeomOptResult {
    let mut mol = molecule.clone();
    let mut energy = compute_energy(&mol, &config.scf_config);
    let mut energy_history = vec![energy];
    let mut converged = false;
    let mut n_steps = 0;
    let mut rms_grad = f64::MAX;

    for step in 0..config.max_steps {
        n_steps = step + 1;

        // Compute gradient
        let grad = compute_gradient(&mol, &config.scf_config, config.fd_step);

        // RMS gradient
        let grad_sq_sum: f64 = grad.iter().map(|g| g * g).sum();
        rms_grad = (grad_sq_sum / grad.len() as f64).sqrt();

        // Check gradient convergence
        if rms_grad < config.gradient_threshold {
            converged = true;
            break;
        }

        // Steepest descent: x_new = x_old - step_size * gradient
        let mut step_size = config.step_size;

        // Simple backtracking line search
        for _ in 0..5 {
            let mut mol_trial = mol.clone();
            for atom_idx in 0..mol.n_atoms() {
                for coord in 0..3 {
                    mol_trial.atoms[atom_idx].position[coord] -=
                        step_size * grad[atom_idx * 3 + coord];
                }
            }

            let e_trial = compute_energy(&mol_trial, &config.scf_config);
            if e_trial < energy {
                mol = mol_trial;
                energy = e_trial;
                break;
            }
            step_size *= 0.5;
        }

        energy_history.push(energy);

        // Check energy convergence
        if energy_history.len() >= 2 {
            let de = (energy_history[energy_history.len() - 1]
                - energy_history[energy_history.len() - 2])
                .abs();
            if de < config.energy_threshold && rms_grad < config.gradient_threshold * 10.0 {
                converged = true;
                break;
            }
        }
    }

    GeomOptResult {
        molecule: mol,
        energy,
        n_steps,
        converged,
        rms_gradient: rms_grad,
        energy_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::BOHR_TO_ANGSTROM;

    #[test]
    fn test_h2_bond_optimization() {
        // Start H2 at a non-equilibrium distance (2.0 Bohr instead of ~1.4)
        let mol = Molecule::new(vec![
            Atom::new(1, 0.0, 0.0, 0.0),
            Atom::new(1, 0.0, 0.0, 2.0),
        ]);

        let config = GeomOptConfig {
            max_steps: 20,
            step_size: 0.5,
            ..Default::default()
        };

        let result = optimize_geometry(&mol, &config);

        // Energy should decrease
        assert!(
            result.energy < result.energy_history[0],
            "Energy should decrease: final={:.6}, initial={:.6}",
            result.energy,
            result.energy_history[0]
        );

        // Bond length should be closer to ~1.4 Bohr
        let r_final = result.molecule.atoms[0].distance_to(&result.molecule.atoms[1]);
        assert!(
            (r_final - 1.4).abs() < 0.5,
            "H2 bond length = {:.3} Bohr ({:.3} Å), expected ~1.4 Bohr",
            r_final,
            r_final * BOHR_TO_ANGSTROM
        );
    }

    #[test]
    fn test_gradient_near_equilibrium() {
        // At equilibrium, gradients should be small
        let mol = Molecule::h2(); // R = 1.4 Bohr (near equilibrium)
        let config = RhfConfig::default();

        let grad = compute_gradient(&mol, &config, 0.005);

        let rms: f64 = (grad.iter().map(|g| g * g).sum::<f64>() / grad.len() as f64).sqrt();
        assert!(
            rms < 0.1,
            "Gradient at equilibrium should be small: RMS = {:.6}",
            rms
        );
    }
}
