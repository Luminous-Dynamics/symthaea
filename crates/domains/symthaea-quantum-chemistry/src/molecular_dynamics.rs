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
//!
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

/// Compatibility helper for callers that expect finite-difference forces directly.
/// The checked AIMD runner calls compute_forces_checked and returns typed failures.
fn compute_forces(mol: &Molecule, fd_step: f64) -> Vec<[f64; 3]> {
    compute_forces_checked(mol, fd_step, 0)
        .unwrap_or_else(|error| panic!("finite-difference force evaluation failed: {error}"))
}

/// Reject finite-difference shifts that overflow or round back to the original value.
fn validate_perturbed_coordinate(
    original: f64,
    perturbed: f64,
    step: usize,
    atom_index: usize,
    axis: usize,
    direction: i8,
) -> Result<(), MdRunError> {
    if !perturbed.is_finite() {
        return Err(MdRunError::NonFinitePosition { step, atom_index, axis });
    }
    if perturbed == original {
        return Err(MdRunError::UnrepresentablePerturbation {
            step,
            atom_index,
            axis,
            direction,
        });
    }
    Ok(())
}

/// Checked central-difference forces; perturbations and energies are validated
/// before they can reach subsequent integration work.
fn compute_forces_checked(
    mol: &Molecule,
    fd_step: f64,
    step: usize,
) -> Result<Vec<[f64; 3]>, MdRunError> {
    let n_atoms = mol.n_atoms();
    let mut forces = vec![[0.0; 3]; n_atoms];
    let scf_config = RhfConfig::default();

    for atom_idx in 0..n_atoms {
        for coord in 0..3 {
            let mut mol_plus = mol.clone();
            let original = mol.atoms[atom_idx].position[coord];
            mol_plus.atoms[atom_idx].position[coord] += fd_step;
            validate_perturbed_coordinate(
                original,
                mol_plus.atoms[atom_idx].position[coord],
                step,
                atom_idx,
                coord,
                1,
            )?;
            for (idx, atom) in mol_plus.atoms.iter().enumerate() {
                for (axis, value) in atom.position.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(MdRunError::NonFinitePosition {
                            step,
                            atom_index: idx,
                            axis,
                        });
                    }
                }
            }
            let basis_p = Sto3g::build(&mol_plus);
            let e_plus = restricted_hartree_fock(&mol_plus, &basis_p, &scf_config).total_energy;
            if !e_plus.is_finite() {
                return Err(MdRunError::NonFiniteForceEnergy {
                    step,
                    atom_index: atom_idx,
                    axis: coord,
                    direction: 1,
                });
            }

            let mut mol_minus = mol.clone();
            mol_minus.atoms[atom_idx].position[coord] -= fd_step;
            validate_perturbed_coordinate(
                original,
                mol_minus.atoms[atom_idx].position[coord],
                step,
                atom_idx,
                coord,
                -1,
            )?;
            for (idx, atom) in mol_minus.atoms.iter().enumerate() {
                for (axis, value) in atom.position.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(MdRunError::NonFinitePosition {
                            step,
                            atom_index: idx,
                            axis,
                        });
                    }
                }
            }
            let basis_m = Sto3g::build(&mol_minus);
            let e_minus = restricted_hartree_fock(&mol_minus, &basis_m, &scf_config).total_energy;
            if !e_minus.is_finite() {
                return Err(MdRunError::NonFiniteForceEnergy {
                    step,
                    atom_index: atom_idx,
                    axis: coord,
                    direction: -1,
                });
            }

            let force = -(e_plus - e_minus) / (2.0 * fd_step);
            if !force.is_finite() {
                return Err(MdRunError::NonFiniteForce {
                    step,
                    atom_index: atom_idx,
                    axis: coord,
                });
            }
            forces[atom_idx][coord] = force;
        }
    }

    Ok(forces)
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
    let n_dof = velocities
        .len()
        .saturating_mul(3)
        .saturating_sub(3); // subtract 3 for CM translation
    let temp = if n_dof > 0 {
        2.0 * ke / (n_dof as f64 * crate::stat_mech::K_BOLTZMANN_HARTREE)
    } else {
        0.0
    };
    (ke, temp)
}

/// Validation errors returned before molecular AIMD starts an electronic-structure evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MdInputError {
    EmptyMolecule,
    UnsupportedElement {
        atom_index: usize,
        atomic_number: u8,
    },
    NonFiniteCoordinate {
        atom_index: usize,
        axis: usize,
    },
    UnsupportedMultiplicity { multiplicity: u32 },
    InvalidElectronCount { electron_count: i64 },
    OddElectronCount { electron_count: i64 },
    NonFiniteTimeStep,
    NonPositiveTimeStep,
    NonFiniteTimeStepSquared,
    NonFiniteTemperature,
    NegativeTemperature,
    NonFiniteFiniteDifferenceStep,
    NonPositiveFiniteDifferenceStep,
    NonFiniteFiniteDifferenceDenominator,
    StepCountOverflow,
    NonFiniteTrajectoryDuration,
}

impl std::fmt::Display for MdInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMolecule => write!(f, "molecule has no atoms"),
            Self::UnsupportedElement {
                atom_index,
                atomic_number,
            } => write!(
                f,
                "atom {atom_index} has atomic number {atomic_number}, unsupported by STO-3G"
            ),
            Self::NonFiniteCoordinate { atom_index, axis } => {
                write!(f, "atom {atom_index} coordinate axis {axis} is non-finite")
            }
            Self::UnsupportedMultiplicity { multiplicity } => write!(
                f,
                "multiplicity {multiplicity} is unsupported by restricted Hartree-Fock AIMD; expected singlet (1)"
            ),
            Self::InvalidElectronCount { electron_count } => {
                write!(f, "electron count {electron_count} is not positive")
            }
            Self::OddElectronCount { electron_count } => write!(
                f,
                "electron count {electron_count} is odd; restricted closed-shell AIMD requires an even count"
            ),
            Self::NonFiniteTimeStep => write!(f, "time step dt must be finite"),
            Self::NonPositiveTimeStep => write!(f, "time step dt must be greater than zero"),
            Self::NonFiniteTimeStepSquared => {
                write!(f, "squared time step dt^2 must be finite for Velocity-Verlet integration")
            }
            Self::NonFiniteTemperature => write!(f, "temperature must be finite"),
            Self::NegativeTemperature => write!(f, "temperature must be non-negative"),
            Self::NonFiniteFiniteDifferenceStep => write!(f, "finite-difference step must be finite"),
            Self::NonPositiveFiniteDifferenceStep => {
                write!(f, "finite-difference step must be greater than zero")
            }
            Self::NonFiniteFiniteDifferenceDenominator => {
                write!(f, "twice the finite-difference step must be finite for the central-difference denominator")
            }
            Self::StepCountOverflow => {
                write!(f, "step count plus initial frame overflows usize")
            }
            Self::NonFiniteTrajectoryDuration => {
                write!(f, "total trajectory duration must be finite")
            }
        }
    }
}

impl std::error::Error for MdInputError {}

/// Numerical/runtime failures detected by the checked molecular AIMD integrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MdRunError {
    InvalidInput(MdInputError),
    NonFinitePosition { step: usize, atom_index: usize, axis: usize },
    UnrepresentablePerturbation {
        step: usize,
        atom_index: usize,
        axis: usize,
        direction: i8,
    },
    NonFiniteForce { step: usize, atom_index: usize, axis: usize },
    NonFiniteForceEnergy {
        step: usize,
        atom_index: usize,
        axis: usize,
        direction: i8,
    },
    NonFiniteVelocity { step: usize, atom_index: usize, axis: usize },
    NonFinitePotentialEnergy { step: usize },
    NonFiniteKineticEnergy { step: usize },
    NonFiniteTemperature { step: usize },
}

impl std::fmt::Display for MdRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(error) => write!(f, "invalid AIMD input: {error}"),
            Self::NonFinitePosition { step, atom_index, axis } => write!(
                f, "non-finite position at step {step}, atom {atom_index}, axis {axis}"
            ),
            Self::UnrepresentablePerturbation { step, atom_index, axis, direction } => write!(
                f, "finite-difference perturbation rounded away at step {step}, atom {atom_index}, axis {axis}, direction {direction}"
            ),
            Self::NonFiniteForce { step, atom_index, axis } => write!(
                f, "non-finite force at step {step}, atom {atom_index}, axis {axis}"
            ),
            Self::NonFiniteForceEnergy { step, atom_index, axis, direction } => write!(
                f,
                "non-finite perturbed energy at step {step}, atom {atom_index}, axis {axis}, direction {direction}"
            ),
            Self::NonFiniteVelocity { step, atom_index, axis } => write!(
                f, "non-finite velocity at step {step}, atom {atom_index}, axis {axis}"
            ),
            Self::NonFinitePotentialEnergy { step } => {
                write!(f, "non-finite potential energy at step {step}")
            }
            Self::NonFiniteKineticEnergy { step } => {
                write!(f, "non-finite kinetic energy at step {step}")
            }
            Self::NonFiniteTemperature { step } => {
                write!(f, "non-finite temperature at step {step}")
            }
        }
    }
}

impl std::error::Error for MdRunError {}

/// Validate molecular AIMD inputs before basis construction or SCF evaluation.
///
/// This preflight reflects the current molecular RHF/STO-3G implementation,
/// not a general periodic or open-shell molecular-dynamics capability.
pub fn validate_md_inputs(molecule: &Molecule, config: &MdConfig) -> Result<(), MdInputError> {
    if molecule.atoms.is_empty() {
        return Err(MdInputError::EmptyMolecule);
    }
    if config.n_steps.checked_add(1).is_none() {
        return Err(MdInputError::StepCountOverflow);
    }

    if !config.dt.is_finite() {
        return Err(MdInputError::NonFiniteTimeStep);
    }
    if config.dt <= 0.0 {
        return Err(MdInputError::NonPositiveTimeStep);
    }
    if !(config.dt * config.n_steps as f64).is_finite() {
        return Err(MdInputError::NonFiniteTrajectoryDuration);
    }
    // Velocity-Verlet multiplies by dt squared; reject overflow before the
    // force/SCF loop even when the total duration remains finite.
    if !(config.dt * config.dt).is_finite() {
        return Err(MdInputError::NonFiniteTimeStepSquared);
    }
    if !config.temperature.is_finite() {
        return Err(MdInputError::NonFiniteTemperature);
    }
    if config.temperature < 0.0 {
        return Err(MdInputError::NegativeTemperature);
    }
    if !config.fd_step.is_finite() {
        return Err(MdInputError::NonFiniteFiniteDifferenceStep);
    }
    if config.fd_step <= 0.0 {
        return Err(MdInputError::NonPositiveFiniteDifferenceStep);
    }
    // The central-difference formula divides by 2 * fd_step; reject overflow
    // before any basis construction or SCF evaluation.
    if !(2.0 * config.fd_step).is_finite() {
        return Err(MdInputError::NonFiniteFiniteDifferenceDenominator);
    }

    for (atom_index, atom) in molecule.atoms.iter().enumerate() {
        if !crate::basis::sto3g::Sto3g::supports_element(atom.atomic_number) {
            return Err(MdInputError::UnsupportedElement {
                atom_index,
                atomic_number: atom.atomic_number,
            });
        }
        for (axis, coordinate) in atom.position.iter().enumerate() {
            if !coordinate.is_finite() {
                return Err(MdInputError::NonFiniteCoordinate { atom_index, axis });
            }
        }
    }

    if molecule.multiplicity != 1 {
        return Err(MdInputError::UnsupportedMultiplicity {
            multiplicity: molecule.multiplicity,
        });
    }
    let nuclear_charge: i64 = molecule
        .atoms
        .iter()
        .map(|atom| atom.atomic_number as i64)
        .sum();
    let electron_count = nuclear_charge - i64::from(molecule.charge);
    if electron_count <= 0 {
        return Err(MdInputError::InvalidElectronCount { electron_count });
    }
    if electron_count % 2 != 0 {
        return Err(MdInputError::OddElectronCount { electron_count });
    }

    Ok(())
}

/// Checked molecular AIMD entry point using the fixed default seed.
pub fn run_md_checked(
    molecule: &Molecule,
    config: &MdConfig,
) -> Result<MdResult, MdRunError> {
    run_md_checked_with_seed(molecule, config, DEFAULT_MD_SEED)
}

/// Checked molecular AIMD entry point with a reproducible, explicit seed.
///
/// Input errors and non-finite numerical states are returned as typed errors.
pub fn run_md_checked_with_seed(
    molecule: &Molecule,
    config: &MdConfig,
    seed: u64,
) -> Result<MdResult, MdRunError> {
    validate_md_inputs(molecule, config).map_err(MdRunError::InvalidInput)?;
    run_md_validated_with_seed(molecule, config, seed)
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
    run_md_checked_with_seed(molecule, config, seed)
        .unwrap_or_else(|error| panic!("AIMD run failed: {error}"))
}

fn validate_finite_positions(step: usize, values: &[[f64; 3]]) -> Result<(), MdRunError> {
    for (atom_index, vector) in values.iter().enumerate() {
        for (axis, value) in vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(MdRunError::NonFinitePosition { step, atom_index, axis });
            }
        }
    }
    Ok(())
}

fn validate_finite_forces(step: usize, values: &[[f64; 3]]) -> Result<(), MdRunError> {
    for (atom_index, vector) in values.iter().enumerate() {
        for (axis, value) in vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(MdRunError::NonFiniteForce { step, atom_index, axis });
            }
        }
    }
    Ok(())
}

fn validate_finite_velocities(step: usize, values: &[[f64; 3]]) -> Result<(), MdRunError> {
    for (atom_index, vector) in values.iter().enumerate() {
        for (axis, value) in vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(MdRunError::NonFiniteVelocity { step, atom_index, axis });
            }
        }
    }
    Ok(())
}

fn validate_finite_observables(
    step: usize,
    potential_energy: f64,
    kinetic_energy: f64,
    temperature: f64,
) -> Result<(), MdRunError> {
    if !potential_energy.is_finite() {
        return Err(MdRunError::NonFinitePotentialEnergy { step });
    }
    if !kinetic_energy.is_finite() {
        return Err(MdRunError::NonFiniteKineticEnergy { step });
    }
    if !temperature.is_finite() {
        return Err(MdRunError::NonFiniteTemperature { step });
    }
    Ok(())
}

fn run_md_validated_with_seed(
    molecule: &Molecule,
    config: &MdConfig,
    seed: u64,
) -> Result<MdResult, MdRunError> {
    let n_atoms = molecule.n_atoms();
    let masses: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| atomic_mass_au(a.atomic_number))
        .collect();

    let mut positions: Vec<[f64; 3]> = molecule.atoms.iter().map(|a| a.position).collect();
    let mut velocities = initialize_velocities(&masses, config.temperature, seed);
    validate_finite_velocities(0, &velocities)?;

    // Initial forces are checked before the trajectory proceeds to energy and
    // integration; non-finite force estimates can otherwise poison positions.
    let mut mol = molecule.clone();
    let mut forces = compute_forces_checked(&mol, config.fd_step, 0)?;
    validate_finite_forces(0, &forces)?;
    let basis = Sto3g::build(&mol);
    let mut energy = restricted_hartree_fock(&mol, &basis, &RhfConfig::default()).total_energy;
    if !energy.is_finite() {
        return Err(MdRunError::NonFinitePotentialEnergy { step: 0 });
    }

    let mut frames = Vec::with_capacity(config.n_steps + 1);
    let (ke, temp) = kinetic_energy_and_temperature(&velocities, &masses);
    validate_finite_observables(0, energy, ke, temp)?;
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

        // Reject non-finite coordinates before basis construction or SCF.
        validate_finite_positions(step + 1, &positions)?;
        // Update molecule for force evaluation only after that check.
        for i in 0..n_atoms {
            mol.atoms[i].position = positions[i];
        }

        // New forces; check them before the follow-on energy evaluation.
        let forces_new = compute_forces_checked(&mol, config.fd_step, step + 1)?;
        validate_finite_forces(step + 1, &forces_new)?;
        let basis = Sto3g::build(&mol);
        energy = restricted_hartree_fock(&mol, &basis, &RhfConfig::default()).total_energy;
        if !energy.is_finite() {
            return Err(MdRunError::NonFinitePotentialEnergy { step: step + 1 });
        }

        // Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        for i in 0..n_atoms {
            for c in 0..3 {
                let a_old = forces[i][c] / masses[i];
                let a_new = forces_new[i][c] / masses[i];
                velocities[i][c] += 0.5 * (a_old + a_new) * dt;
            }
        }

        validate_finite_velocities(step + 1, &velocities)?;
        forces = forces_new;

        let (ke, temp) = kinetic_energy_and_temperature(&velocities, &masses);
        validate_finite_observables(step + 1, energy, ke, temp)?;
        frames.push(MdFrame {
            positions: positions.clone(),
            velocities: velocities.clone(),
            energy,
            kinetic_energy: ke,
            temperature: temp,
            time: (step + 1) as f64 * dt,
        });
    }

    Ok(MdResult {
        frames,
        total_steps: config.n_steps,
    })
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
            assert!(
                component.abs() < 1e-9,
                "center-of-mass momentum was not removed: {component:e}"
            );
        }
        let (_, temperature) = kinetic_energy_and_temperature(&velocities, &masses);
        assert!(
            (temperature - 300.0).abs() < 1e-8,
            "initial temperature was {temperature}, expected 300 K"
        );
    }

    #[test]
    fn zero_temperature_initialization_is_exactly_stationary() {
        let velocities = initialize_velocities(&h2_masses(), 0.0, 19);
        assert!(
            velocities.iter().flatten().all(|velocity| *velocity == 0.0)
        );
    }

    #[test]
    fn non_finite_temperature_does_not_create_non_finite_velocities() {
        let velocities = initialize_velocities(&h2_masses(), f64::NAN, 19);
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
    #[test]
    fn checked_force_rejects_perturbation_rounded_away_before_solver() {
        let mut molecule = Molecule::h2();
        molecule.atoms[0].position[0] = 1.0e300;
        let result = compute_forces_checked(&molecule, 0.005, 7);
        assert_eq!(
            result,
            Err(MdRunError::UnrepresentablePerturbation {
                step: 7,
                atom_index: 0,
                axis: 0,
                direction: 1,
            })
        );
    }

    #[test]
    fn checked_force_evaluation_rejects_overflowed_perturbation() {
        let mut molecule = Molecule::h2();
        molecule.atoms[0].position[0] = 1.0e308;
        let result = compute_forces_checked(&molecule, 1.0e308, 7);
        assert_eq!(
            result,
            Err(MdRunError::NonFinitePosition {
                step: 7,
                atom_index: 0,
                axis: 0,
            })
        );
    }

    #[test]
    fn runtime_guards_reject_non_finite_vector_states() {
        assert_eq!(
            validate_finite_positions(3, &[[f64::NAN, 0.0, 0.0]]),
            Err(MdRunError::NonFinitePosition { step: 3, atom_index: 0, axis: 0 })
        );
        assert_eq!(
            validate_finite_forces(4, &[[0.0, f64::INFINITY, 0.0]]),
            Err(MdRunError::NonFiniteForce { step: 4, atom_index: 0, axis: 1 })
        );
        assert_eq!(
            validate_finite_velocities(5, &[[0.0, 0.0, f64::NEG_INFINITY]]),
            Err(MdRunError::NonFiniteVelocity { step: 5, atom_index: 0, axis: 2 })
        );
    }

    #[test]
    fn runtime_guards_reject_non_finite_observables() {
        assert_eq!(
            validate_finite_observables(2, f64::NAN, 0.0, 300.0),
            Err(MdRunError::NonFinitePotentialEnergy { step: 2 })
        );
        assert_eq!(
            validate_finite_observables(2, -1.0, f64::INFINITY, 300.0),
            Err(MdRunError::NonFiniteKineticEnergy { step: 2 })
        );
        assert_eq!(
            validate_finite_observables(2, -1.0, 0.1, f64::NAN),
            Err(MdRunError::NonFiniteTemperature { step: 2 })
        );
        assert!(validate_finite_observables(0, -1.0, 0.1, 300.0).is_ok());
    }

    #[test]
    fn validate_default_h2_inputs() {
        let molecule = Molecule::h2();
        assert_eq!(validate_md_inputs(&molecule, &MdConfig::default()), Ok(()));
    }

    #[test]
    fn validation_rejects_empty_molecule() {
        let molecule = Molecule::new(vec![]);
        assert_eq!(
            validate_md_inputs(&molecule, &MdConfig::default()),
            Err(MdInputError::EmptyMolecule)
        );
    }

    #[test]
    fn checked_runner_rejects_empty_molecule_before_solver_invocation() {
        let molecule = Molecule::new(vec![]);
        assert!(matches!(
            run_md_checked(&molecule, &MdConfig::default()),
            Err(MdRunError::InvalidInput(MdInputError::EmptyMolecule))
        ));
    }

    #[test]
    fn validation_rejects_invalid_time_step_and_fd_step() {
        let molecule = Molecule::h2();
        let mut config = MdConfig::default();
        config.dt = 0.0;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NonPositiveTimeStep)
        );
        config.dt = f64::NAN;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NonFiniteTimeStep)
        );
        config.dt = 1.0;
        config.fd_step = 0.0;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NonPositiveFiniteDifferenceStep)
        );
        config.fd_step = f64::INFINITY;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NonFiniteFiniteDifferenceStep)
        );
        config.fd_step = 1.0e308;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NonFiniteFiniteDifferenceDenominator)
        );
    }

    #[test]
    fn validation_rejects_overflowing_step_count_or_duration() {
        let molecule = Molecule::h2();
        let mut config = MdConfig {
            n_steps: usize::MAX,
            ..MdConfig::default()
        };
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::StepCountOverflow)
        );
        config.n_steps = 2;
        config.dt = f64::MAX;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NonFiniteTrajectoryDuration)
        );

        // A one-step run can have finite duration but overflowing dt squared.
        config.n_steps = 1;
        config.dt = 1.0e200;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NonFiniteTimeStepSquared)
        );
    }

    #[test]
    fn validation_rejects_negative_and_non_finite_temperature() {
        let molecule = Molecule::h2();
        let mut config = MdConfig::default();
        config.temperature = -1.0;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NegativeTemperature)
        );
        config.temperature = f64::INFINITY;
        assert_eq!(
            validate_md_inputs(&molecule, &config),
            Err(MdInputError::NonFiniteTemperature)
        );
    }

    #[test]
    fn validation_rejects_non_finite_geometry_and_unsupported_elements() {
        let mut molecule = Molecule::h2();
        molecule.atoms[0].position[1] = f64::NAN;
        assert_eq!(
            validate_md_inputs(&molecule, &MdConfig::default()),
            Err(MdInputError::NonFiniteCoordinate {
                atom_index: 0,
                axis: 1,
            })
        );

        let unsupported = Molecule::new(vec![crate::molecule::Atom::new(55, 0.0, 0.0, 0.0)]);
        assert_eq!(
            validate_md_inputs(&unsupported, &MdConfig::default()),
            Err(MdInputError::UnsupportedElement {
                atom_index: 0,
                atomic_number: 55,
            })
        );
    }

    #[test]
    fn validation_rejects_open_shell_and_odd_electron_inputs() {
        let mut open_shell = Molecule::h2();
        open_shell.multiplicity = 3;
        assert_eq!(
            validate_md_inputs(&open_shell, &MdConfig::default()),
            Err(MdInputError::UnsupportedMultiplicity { multiplicity: 3 })
        );

        let odd_electron = Molecule::with_charge(Molecule::h2().atoms, 1, 1);
        assert_eq!(
            validate_md_inputs(&odd_electron, &MdConfig::default()),
            Err(MdInputError::OddElectronCount { electron_count: 1 })
        );
    }

    #[test]
    fn kinetic_temperature_handles_empty_velocity_slice() {
        let (kinetic_energy, temperature) = kinetic_energy_and_temperature(&[], &[]);
        assert_eq!(kinetic_energy, 0.0);
        assert_eq!(temperature, 0.0);
    }
}
