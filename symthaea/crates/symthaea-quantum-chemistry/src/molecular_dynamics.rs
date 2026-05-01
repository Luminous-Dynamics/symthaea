// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ab Initio Molecular Dynamics (AIMD).
//!
//! Propagates nuclei on the Born-Oppenheimer potential energy surface
//! using Velocity-Verlet integration. Forces are computed from finite-difference
//! gradients of the quantum chemical energy.
//!
//! References:
//! - Car & Parrinello (1985). Phys. Rev. Lett. 55, 2471.
//! - Marx & Hutter (2009). *Ab Initio Molecular Dynamics*. Cambridge UP.

use crate::basis::sto3g::Sto3g;
use crate::basis::BasisSetProvider;
use crate::molecule::Molecule;
use crate::scf::rhf::{restricted_hartree_fock, RhfConfig};

/// Atomic masses in atomic units (electron masses). 1 amu = 1822.888 a.u.
const AMU_TO_AU: f64 = 1822.888;

fn atomic_mass_au(z: u8) -> f64 {
    match z {
        1 => 1.008 * AMU_TO_AU,
        2 => 4.003 * AMU_TO_AU,
        6 => 12.011 * AMU_TO_AU,
        7 => 14.007 * AMU_TO_AU,
        8 => 15.999 * AMU_TO_AU,
        9 => 18.998 * AMU_TO_AU,
        _ => z as f64 * AMU_TO_AU,
    }
}

/// MD trajectory configuration.
#[derive(Debug, Clone)]
pub struct MdConfig {
    /// Time step in atomic units (1 a.u. ≈ 0.0242 fs)
    pub dt: f64,
    /// Number of steps
    pub n_steps: usize,
    /// Initial temperature (Kelvin) for velocity initialization
    pub temperature: f64,
    /// Finite-difference step for force computation (Bohr)
    pub fd_step: f64,
}

impl Default for MdConfig {
    fn default() -> Self {
        Self {
            dt: 20.0, // ~0.5 fs
            n_steps: 50,
            temperature: 300.0,
            fd_step: 0.005,
        }
    }
}

/// A snapshot of the MD trajectory.
#[derive(Debug, Clone)]
pub struct MdFrame {
    pub positions: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub energy: f64,
    pub kinetic_energy: f64,
    pub temperature: f64,
    pub time: f64,
}

/// Result of an MD simulation.
#[derive(Debug, Clone)]
pub struct MdResult {
    pub frames: Vec<MdFrame>,
    pub total_steps: usize,
}

/// Compute forces on all atoms via central finite differences.
fn compute_forces(mol: &Molecule, fd_step: f64) -> Vec<[f64; 3]> {
    let n_atoms = mol.n_atoms();
    let mut forces = vec![[0.0; 3]; n_atoms];
    let scf_config = RhfConfig::default();

    for atom_idx in 0..n_atoms {
        for coord in 0..3 {
            let mut mol_plus = mol.clone();
            mol_plus.atoms[atom_idx].position[coord] += fd_step;
            let basis_p = Sto3g::build(&mol_plus);
            let e_plus = restricted_hartree_fock(&mol_plus, &basis_p, &scf_config).total_energy;

            let mut mol_minus = mol.clone();
            mol_minus.atoms[atom_idx].position[coord] -= fd_step;
            let basis_m = Sto3g::build(&mol_minus);
            let e_minus = restricted_hartree_fock(&mol_minus, &basis_m, &scf_config).total_energy;

            // Force = -dE/dx
            forces[atom_idx][coord] = -(e_plus - e_minus) / (2.0 * fd_step);
        }
    }

    forces
}

/// Compute kinetic energy and instantaneous temperature.
fn kinetic_energy_and_temperature(velocities: &[[f64; 3]], masses: &[f64]) -> (f64, f64) {
    let mut ke = 0.0;
    for (v, &m) in velocities.iter().zip(masses.iter()) {
        ke += 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    let n_dof = 3 * velocities.len() - 3; // subtract 3 for CM translation
    let temp = if n_dof > 0 {
        2.0 * ke / (n_dof as f64 * crate::stat_mech::K_BOLTZMANN_HARTREE)
    } else {
        0.0
    };
    (ke, temp)
}

/// Run Born-Oppenheimer molecular dynamics using Velocity-Verlet.
pub fn run_md(molecule: &Molecule, config: &MdConfig) -> MdResult {
    let n_atoms = molecule.n_atoms();
    let masses: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| atomic_mass_au(a.atomic_number))
        .collect();

    let mut positions: Vec<[f64; 3]> = molecule.atoms.iter().map(|a| a.position).collect();
    let mut velocities = vec![[0.0; 3]; n_atoms];

    // Initialize velocities from Maxwell-Boltzmann (simplified: random uniform scaled)
    if config.temperature > 0.0 {
        let seed = 42u64;
        for i in 0..n_atoms {
            for c in 0..3 {
                // Simple deterministic pseudo-random
                let hash = ((seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407))
                .wrapping_mul((i * 3 + c + 1) as u64)) as f64
                    / u64::MAX as f64;
                let v_scale =
                    (crate::stat_mech::K_BOLTZMANN_HARTREE * config.temperature / masses[i]).sqrt();
                velocities[i][c] = (hash - 0.5) * 2.0 * v_scale;
            }
        }
    }

    // Initial forces
    let mut mol = molecule.clone();
    let mut forces = compute_forces(&mol, config.fd_step);
    let basis = Sto3g::build(&mol);
    let mut energy = restricted_hartree_fock(&mol, &basis, &RhfConfig::default()).total_energy;

    let mut frames = Vec::with_capacity(config.n_steps + 1);
    let (ke, temp) = kinetic_energy_and_temperature(&velocities, &masses);
    frames.push(MdFrame {
        positions: positions.clone(),
        velocities: velocities.clone(),
        energy,
        kinetic_energy: ke,
        temperature: temp,
        time: 0.0,
    });

    // Velocity-Verlet integration
    for step in 0..config.n_steps {
        let dt = config.dt;

        // Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
        for i in 0..n_atoms {
            for c in 0..3 {
                let a = forces[i][c] / masses[i];
                positions[i][c] += velocities[i][c] * dt + 0.5 * a * dt * dt;
            }
        }

        // Update molecule for force evaluation
        for i in 0..n_atoms {
            mol.atoms[i].position = positions[i];
        }

        // New forces
        let forces_new = compute_forces(&mol, config.fd_step);
        let basis = Sto3g::build(&mol);
        energy = restricted_hartree_fock(&mol, &basis, &RhfConfig::default()).total_energy;

        // Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        for i in 0..n_atoms {
            for c in 0..3 {
                let a_old = forces[i][c] / masses[i];
                let a_new = forces_new[i][c] / masses[i];
                velocities[i][c] += 0.5 * (a_old + a_new) * dt;
            }
        }

        forces = forces_new;

        let (ke, temp) = kinetic_energy_and_temperature(&velocities, &masses);
        frames.push(MdFrame {
            positions: positions.clone(),
            velocities: velocities.clone(),
            energy,
            kinetic_energy: ke,
            temperature: temp,
            time: (step + 1) as f64 * dt,
        });
    }

    MdResult {
        frames,
        total_steps: config.n_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md_h2_runs() {
        let mol = Molecule::h2();
        let config = MdConfig {
            n_steps: 3,
            dt: 10.0,
            temperature: 300.0,
            ..Default::default()
        };
        let result = run_md(&mol, &config);
        assert_eq!(result.frames.len(), 4); // initial + 3 steps
        assert_eq!(result.total_steps, 3);
    }

    #[test]
    fn test_md_energy_conservation_approximate() {
        // Total energy (PE + KE) should be roughly conserved
        let mol = Molecule::h2();
        let config = MdConfig {
            n_steps: 3,
            dt: 5.0,
            temperature: 100.0,
            ..Default::default()
        };
        let result = run_md(&mol, &config);

        let e_total_0 = result.frames[0].energy + result.frames[0].kinetic_energy;
        let e_total_last =
            result.frames.last().unwrap().energy + result.frames.last().unwrap().kinetic_energy;

        // Allow 10% drift for short trajectory with large dt
        let drift = (e_total_last - e_total_0).abs();
        assert!(
            drift < 0.1,
            "Energy drift = {:.6} Hartree (should be small)",
            drift
        );
    }

    #[test]
    fn test_forces_finite() {
        let mol = Molecule::h2();
        let forces = compute_forces(&mol, 0.005);
        for f in &forces {
            for &c in f {
                assert!(c.is_finite(), "Force should be finite");
            }
        }
    }
}
