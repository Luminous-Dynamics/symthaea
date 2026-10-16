// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ab Initio Molecular Dynamics (AIMD).
//!
//! Propagates nuclei on the Born-Oppenheimer potential energy surface
//! using Velocity-Verlet integration. Forces are computed from finite-difference
//! gradients of the quantum chemical energy.
//!
//! ## Initialization and current qualification limits
//!
//! Initial velocities use seeded pseudo-random Maxwell-Boltzmann Gaussian
//! sampling (Box-Muller), center-of-mass translation removal, and rescaling to
//! the requested initial kinetic temperature. `run_md` retains a deterministic
//! default seed; `run_md_with_seed` permits reproducible independent starts.
//!
//! This corrects the initial-condition sampler only. Forces remain central
//! finite differences rather than analytic gradients. The energy-drift
//! regression test (`test_md_energy_conservation_approximate`) allows drift up
//! to 0.1 Hartree (~63 kcal/mol) over a short trajectory -- a deliberately
//! loose bound, not a validated conservation guarantee. This molecular,
//! non-periodic HF/STO-3G implementation is not a qualified nanocrystal solver.
//! References:
//! - Car & Parrinello (1985). Phys. Rev. Lett. 55, 2471.
//! - Marx & Hutter (2009). *Ab Initio Molecular Dynamics*. Cambridge UP.

use crate::basis::BasisSetProvider;
use crate::basis::sto3g::Sto3g;
use crate::molecule::Molecule;
use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

/// Atomic masses in atomic units (electron masses). 1 amu = 1822.888 a.u.
pub(crate) const AMU_TO_AU: f64 = 1822.888;

/// Real atomic mass (a.u.) for element `z`, via Phase Q1's `element_data`
/// table (118 confirmed real elements). Fixed Phase Q4, 2026-07-17 -- this
/// used to hardcode real masses for only 6 elements (H/He/C/N/O/F) and
/// fall back to `z as f64 * AMU_TO_AU` for everything else, a bad
/// approximation (e.g. gave Li roughly half its real mass: 3 vs. the real
/// 6.94 amu).
pub(crate) fn atomic_mass_au(z: u8) -> f64 {
    crate::element_data::element_metadata(z)
        .unwrap_or_else(|| panic!("no element metadata for Z={z}"))
        .atomic_mass
        * AMU_TO_AU
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

/// Fixed default seed retained for backwards-reproducible run_md calls.
const DEFAULT_MD_SEED: u64 = 42;

/// Small deterministic SplitMix64 generator for dependency-free thermal sampling.
#[derive(Debug, Clone)]
struct NormalRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl NormalRng {
    fn new(seed: u64) -> Self {
        Self { state: seed, spare_normal: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Uniform variate strictly inside (0, 1), avoiding ln(0) in Box-Muller.
    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open();
        let (sin_angle, cos_angle) = angle.sin_cos();
        self.spare_normal = Some(radius * sin_angle);
        radius * cos_angle
    }
}

/// Construct a thermal initial velocity set in atomic units.
fn initialize_velocities(masses: &[f64], temperature: f64, seed: u64) -> Vec<[f64; 3]> {
    let mut velocities = vec![[0.0; 3]; masses.len()];
    if !temperature.is_finite() || temperature <= 0.0 || masses.len() <= 1 {
        return velocities;
    }

    let mut rng = NormalRng::new(seed);
    for (velocity, &mass) in velocities.iter_mut().zip(masses) {
        let sigma = (crate::stat_mech::K_BOLTZMANN_HARTREE * temperature / mass).sqrt();
        for component in velocity {
            *component = sigma * rng.standard_normal();
        }
    }

    // Remove net linear momentum so the thermal ensemble has no spurious
    // center-of-mass translation.
    let total_mass: f64 = masses.iter().sum();
    let mut total_momentum = [0.0; 3];
    for (velocity, &mass) in velocities.iter().zip(masses) {
        for axis in 0..3 {
            total_momentum[axis] += mass * velocity[axis];
        }
    }
    let center_velocity = total_momentum.map(|momentum| momentum / total_mass);
    for velocity in &mut velocities {
        for axis in 0..3 {
            velocity[axis] -= center_velocity[axis];
        }
    }

    // Match the requested initial kinetic temperature after COM removal, using
    // the same 3N-3 degrees of freedom as kinetic_energy_and_temperature.
    let (_, sampled_temperature) = kinetic_energy_and_temperature(&velocities, masses);
    if sampled_temperature.is_finite() && sampled_temperature > 0.0 {
        let scale = (temperature / sampled_temperature).sqrt();
        for velocity in &mut velocities {
            for component in velocity {
                *component *= scale;
            }
        }
    }

    velocities
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
///
/// Uses a fixed default seed to preserve deterministic behavior for existing callers.
pub fn run_md(molecule: &Molecule, config: &MdConfig) -> MdResult {
    run_md_with_seed(molecule, config, DEFAULT_MD_SEED)
}

/// Run molecular dynamics with an explicit seed for thermal initial velocities.
///
/// Each component is drawn with variance k_B T / m_i, net center-of-mass
/// translation is removed, and the initial kinetic temperature is rescaled
/// to config.temperature. This makes replicas reproducible; it does not
/// qualify the underlying electronic-structure model or trajectory physics.
pub fn run_md_with_seed(molecule: &Molecule, config: &MdConfig, seed: u64) -> MdResult {
    let n_atoms = molecule.n_atoms();
    let masses: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| atomic_mass_au(a.atomic_number))
        .collect();

    let mut positions: Vec<[f64; 3]> = molecule.atoms.iter().map(|a| a.position).collect();
    let mut velocities = initialize_velocities(&masses, config.temperature, seed);

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

    fn h2_masses() -> Vec<f64> {
        Molecule::h2()
            .atoms
            .iter()
            .map(|atom| atomic_mass_au(atom.atomic_number))
            .collect()
    }

    #[test]
    fn thermal_initialization_is_repeatable_for_same_seed_and_varies_by_seed() {
        let masses = h2_masses();
        let a = initialize_velocities(&masses, 300.0, 7);
        let b = initialize_velocities(&masses, 300.0, 7);
        let c = initialize_velocities(&masses, 300.0, 8);
        assert_eq!(a, b, "same seed must reproduce the same initial velocities");
        assert_ne!(a, c, "different seeds should produce different replicas");
    }

    #[test]
    fn thermal_initialization_removes_center_of_mass_momentum_and_hits_temperature() {
        let masses = h2_masses();
        let velocities = initialize_velocities(&masses, 300.0, 1234);
        let mut momentum = [0.0; 3];
        for (velocity, &mass) in velocities.iter().zip(&masses) {
            for axis in 0..3 {
                momentum[axis] += mass * velocity[axis];
            }
        }
        for component in momentum {
            assert!(component.abs() < 1e-9, "center-of-mass momentum was not removed: {component:e}");
        }
        let (_, temperature) = kinetic_energy_and_temperature(&velocities, &masses);
        assert!((temperature - 300.0).abs() < 1e-8, "initial temperature was {temperature}, expected 300 K");
    }

    #[test]
    fn zero_temperature_initialization_is_exactly_stationary() {
        let velocities = initialize_velocities(&h2_masses(), 0.0, 19);
        assert!(velocities.iter().flatten().all(|velocity| *velocity == 0.0));
    }

    #[test]
    fn box_muller_samples_have_approximately_zero_mean_and_unit_variance() {
        let mut rng = NormalRng::new(0xC0FF_EE);
        let count = 20_000;
        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        for _ in 0..count {
            let sample = rng.standard_normal();
            sum += sample;
            sum_squares += sample * sample;
        }
        let mean = sum / count as f64;
        let variance = sum_squares / count as f64 - mean * mean;
        assert!(mean.abs() < 0.03, "sample mean was {mean}");
        assert!((variance - 1.0).abs() < 0.05, "sample variance was {variance}");
    }
}
