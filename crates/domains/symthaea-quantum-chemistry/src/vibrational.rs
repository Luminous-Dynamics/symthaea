// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vibrational analysis and thermochemistry (Phase Q4, 2026-07-17).
//!
//! Harmonic normal-mode analysis via a numerical Hessian, plus rigid-rotor
//! / harmonic-oscillator / ideal-gas thermochemistry. Standard, legitimate
//! practice even in production QC codes -- not a lesser substitute for
//! analytic Hessians. Both pieces reuse already-verified machinery rather
//! than re-deriving from scratch: the Hessian is a second layer of central
//! finite-differencing over `geometry_opt::compute_gradient` (itself already
//! tested), and thermochemistry reuses `stat_mech.rs`'s existing partition
//! function formulas.
//!
//! ## Status / honest scope
//!
//! - **Hessian is purely numerical** (finite-difference of a finite-difference
//!   gradient) -- no analytic second derivatives. This means noise scales with
//!   `fd_step` more aggressively than the gradient alone; the default step
//!   here (0.01 Bohr) is larger than `geometry_opt`'s gradient default (0.005)
//!   for exactly this reason.
//! - **Thermochemistry is scoped to linear molecules only**, matching
//!   `stat_mech::rotational_z_linear`'s own scope. General (asymmetric-top)
//!   rotational treatment is not implemented.
//! - **Translational entropy omits the Sackur-Tetrode indistinguishability
//!   correction** (`-k ln N!`): `S_trans` here is computed via the same
//!   `S = (U-F)/T` identity `stat_mech::entropy` already uses internally,
//!   generalized from discrete energy levels to a continuous partition
//!   function, treating the molecule as a single distinguishable particle.
//!   This under-counts absolute translational entropy relative to standard
//!   gas-phase thermochemistry tables (which include the correction) --
//!   disclosed here rather than silently assumed correct.
//! - **No independent Hessian/frequency reference tool is available in this
//!   sandbox** (same constraint as Phases Q0/Q2/Q3) -- verification below
//!   uses structural/physical invariants (zero-mode count, positivity,
//!   order-of-magnitude), not comparison to literature reference numbers.
//! - **Callers must pass an (approximately) optimized geometry** -- i.e.
//!   run `geometry_opt::optimize_geometry` first. This was discovered, not
//!   assumed: an early version of this module's tests ran `normal_mode_analysis`
//!   directly on `Molecule::h2()`'s fixed, not-quite-equilibrium bond length
//!   (1.4 Bohr, doc'd there as "equilibrium-ish") and found only 3 zero
//!   modes instead of 5, with the two bending/rotational directions showing
//!   a spurious ~1000 cm⁻¹ curvature instead of ~0. Root cause: translational
//!   zero modes are an *exact* Hessian null space at any geometry (energy is
//!   translation-invariant identically, so the second derivative along a
//!   uniform-translation direction is identically zero), but rotational zero
//!   modes are only exact **at a true stationary point** (zero gradient) --
//!   away from equilibrium they pick up a real "gradient-contamination" term
//!   (standard result, see Wilson/Decius/Cross, *Molecular Vibrations*, on
//!   the Eckart conditions). Fixed by optimizing geometry before analysis in
//!   every test below, not by loosening the zero-mode threshold.

use crate::constants::HARTREE_TO_CM1;
use crate::geometry_opt::compute_gradient;
use crate::molecular_dynamics::atomic_mass_au;
use crate::molecule::Molecule;
use crate::scf::generalized_eigen::symmetric_eigen;
use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

/// Below this magnitude (in cm⁻¹), a normal mode is treated as a
/// translation/rotation artifact of the numerical Hessian rather than a
/// real vibration. 50 cm⁻¹ is a common practical cutoff (real vibrations
/// of even weakly-bound systems are typically well above this; genuine
/// zero modes should land within a few cm⁻¹ of 0 for a converged SCF).
pub const ZERO_MODE_THRESHOLD_CM1: f64 = 50.0;

/// Configuration for vibrational normal-mode analysis.
#[derive(Debug, Clone)]
pub struct VibConfig {
    pub scf_config: RhfConfig,
    /// Finite-difference step for the Hessian (Bohr). Larger than a typical
    /// gradient-only step since this is a second layer of differencing.
    pub fd_step: f64,
    pub zero_mode_threshold_cm1: f64,
}

impl Default for VibConfig {
    fn default() -> Self {
        Self {
            scf_config: RhfConfig::default(),
            fd_step: 0.01,
            zero_mode_threshold_cm1: ZERO_MODE_THRESHOLD_CM1,
        }
    }
}

/// A single normal-mode frequency.
#[derive(Debug, Clone, Copy)]
pub struct Frequency {
    /// Angular frequency in Hartree (ħω, since ħ=1 in atomic units) --
    /// negative for an imaginary mode (unstable direction: the mass-weighted
    /// Hessian eigenvalue was negative), by the standard sign convention.
    pub angular_hartree: f64,
    /// Same quantity converted to wavenumbers (cm⁻¹) via `HARTREE_TO_CM1`.
    pub wavenumber_cm1: f64,
    /// Whether this mode's magnitude falls below `ZERO_MODE_THRESHOLD_CM1`
    /// -- i.e. it is judged to be a translation/rotation artifact, not a
    /// real vibration. This is a **verification invariant checked in
    /// tests**, not an assumption baked into production logic about how
    /// many zero modes there "should" be.
    pub is_zero_mode: bool,
}

/// Result of a normal-mode analysis.
#[derive(Debug, Clone)]
pub struct VibrationalResult {
    /// All 3N modes, sorted ascending by the underlying mass-weighted
    /// Hessian eigenvalue (most-imaginary/negative first).
    pub frequencies: Vec<Frequency>,
    pub n_zero_modes: usize,
    pub n_real_modes: usize,
}

/// Compute the energy of a molecule (single-point HF/STO-3G).
///
/// `pub(crate)` since Phase Q4 -- mirrors `geometry_opt::compute_energy`
/// (kept private there) so thermochemistry here doesn't need a second
/// SCF-energy helper; both do the same two-line `Sto3g::build` +
/// `restricted_hartree_fock` call.
pub(crate) fn compute_electronic_energy(mol: &Molecule, config: &RhfConfig) -> f64 {
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    let basis = Sto3g::build(mol);
    restricted_hartree_fock(mol, &basis, config).total_energy
}

/// Compute the 3N x 3N Cartesian Hessian via central finite differences of
/// the (already finite-difference) gradient, then symmetrize by averaging
/// with its transpose -- standard practice, cancels finite-difference
/// asymmetry noise.
pub fn compute_hessian(mol: &Molecule, scf_config: &RhfConfig, fd_step: f64) -> Vec<f64> {
    let n = 3 * mol.n_atoms();
    let mut hessian = vec![0.0; n * n];

    for i in 0..n {
        let atom_i = i / 3;
        let coord_i = i % 3;

        let mut mol_plus = mol.clone();
        mol_plus.atoms[atom_i].position[coord_i] += fd_step;
        let grad_plus = compute_gradient(&mol_plus, scf_config, fd_step);

        let mut mol_minus = mol.clone();
        mol_minus.atoms[atom_i].position[coord_i] -= fd_step;
        let grad_minus = compute_gradient(&mol_minus, scf_config, fd_step);

        for j in 0..n {
            hessian[i * n + j] = (grad_plus[j] - grad_minus[j]) / (2.0 * fd_step);
        }
    }

    let mut symmetrized = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            symmetrized[i * n + j] = 0.5 * (hessian[i * n + j] + hessian[j * n + i]);
        }
    }
    symmetrized
}

/// Mass-weight a Cartesian Hessian: `H_mw[i][j] = H[i][j] / sqrt(m_i * m_j)`,
/// where `atom_masses` has one entry per atom (each repeated across its 3
/// Cartesian coordinates internally).
pub fn mass_weight_hessian(hessian: &[f64], atom_masses: &[f64]) -> Vec<f64> {
    let n_atoms = atom_masses.len();
    let n = 3 * n_atoms;
    assert_eq!(
        hessian.len(),
        n * n,
        "hessian size {} doesn't match 3*n_atoms={}",
        hessian.len(),
        n
    );

    let mut mass_weighted = vec![0.0; n * n];
    for i in 0..n {
        let m_i = atom_masses[i / 3];
        for j in 0..n {
            let m_j = atom_masses[j / 3];
            mass_weighted[i * n + j] = hessian[i * n + j] / (m_i * m_j).sqrt();
        }
    }
    mass_weighted
}

/// Full harmonic normal-mode analysis: builds the mass-weighted Hessian,
/// diagonalizes it, and converts eigenvalues to frequencies.
pub fn normal_mode_analysis(mol: &Molecule, config: &VibConfig) -> VibrationalResult {
    let n = 3 * mol.n_atoms();
    let hessian = compute_hessian(mol, &config.scf_config, config.fd_step);
    let masses: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| atomic_mass_au(a.atomic_number))
        .collect();
    let mass_weighted = mass_weight_hessian(&hessian, &masses);

    let (eigenvalues, _eigenvectors) = symmetric_eigen(&mass_weighted, n);
    let mut sorted_eigenvalues = eigenvalues;
    sorted_eigenvalues.sort_by(f64::total_cmp);

    let frequencies: Vec<Frequency> = sorted_eigenvalues
        .into_iter()
        .map(|lambda| {
            // sqrt of a negative eigenvalue -> imaginary mode, reported by
            // the standard convention as a negative frequency.
            let angular_hartree = if lambda >= 0.0 {
                lambda.sqrt()
            } else {
                -(-lambda).sqrt()
            };
            let wavenumber_cm1 = angular_hartree * HARTREE_TO_CM1;
            Frequency {
                angular_hartree,
                wavenumber_cm1,
                is_zero_mode: wavenumber_cm1.abs() < config.zero_mode_threshold_cm1,
            }
        })
        .collect();

    let n_zero_modes = frequencies.iter().filter(|f| f.is_zero_mode).count();
    let n_real_modes = frequencies.len() - n_zero_modes;

    VibrationalResult {
        frequencies,
        n_zero_modes,
        n_real_modes,
    }
}

// ── Thermochemistry (linear molecules only) ─────────────────────────────────

/// Configuration for linear-molecule thermochemistry.
#[derive(Debug, Clone)]
pub struct ThermoConfig {
    pub vib_config: VibConfig,
    /// Kelvin. Standard conditions default (298.15 K), disclosed not
    /// silently assumed.
    pub temperature: f64,
    /// Atmospheres. Standard conditions default (1 atm).
    pub pressure_atm: f64,
    /// Rotational symmetry number: 2 for a homonuclear diatomic (or any
    /// symmetric linear molecule), 1 otherwise. Caller-supplied since this
    /// crate has no general point-group detection.
    pub symmetry_number: u32,
}

impl Default for ThermoConfig {
    fn default() -> Self {
        Self {
            vib_config: VibConfig::default(),
            temperature: 298.15,
            pressure_atm: 1.0,
            symmetry_number: 1,
        }
    }
}

/// Thermochemistry breakdown for a linear molecule at the configured
/// temperature/pressure.
#[derive(Debug, Clone)]
pub struct ThermoResult {
    pub temperature: f64,
    pub electronic_energy_hartree: f64,
    /// 0.5 * sum of real (non-zero-mode) vibrational frequencies.
    pub zpe_hartree: f64,
    /// B = 1/(2I), from the real moment of inertia about the center of mass.
    pub rotational_constant_hartree: f64,
    /// Electronic + translational + rotational + vibrational (vibrational
    /// term already includes ZPE).
    pub internal_energy_hartree: f64,
    /// H = U + kT (ideal-gas PV correction).
    pub enthalpy_hartree: f64,
    pub entropy_hartree_per_kelvin: f64,
    /// G = H - TS.
    pub gibbs_free_energy_hartree: f64,
    pub n_real_modes: usize,
    pub n_zero_modes: usize,
}

fn center_of_mass(mol: &Molecule, masses: &[f64]) -> [f64; 3] {
    let total_mass: f64 = masses.iter().sum();
    let mut com = [0.0; 3];
    for (atom, &m) in mol.atoms.iter().zip(masses.iter()) {
        for c in 0..3 {
            com[c] += m * atom.position[c];
        }
    }
    for c in com.iter_mut() {
        *c /= total_mass;
    }
    com
}

/// Moment of inertia of a linear molecule about its center of mass. Since
/// every atom lies on the bond axis, each atom's 3D distance from the COM
/// equals its perpendicular distance from the (perpendicular-to-bond-axis)
/// rotation axis, so `I = sum m_i * |r_i - r_com|^2` directly -- no axis
/// projection needed.
fn moment_of_inertia_linear(mol: &Molecule, masses: &[f64]) -> f64 {
    let com = center_of_mass(mol, masses);
    mol.atoms
        .iter()
        .zip(masses.iter())
        .map(|(atom, &m)| {
            let dx = atom.position[0] - com[0];
            let dy = atom.position[1] - com[1];
            let dz = atom.position[2] - com[2];
            m * (dx * dx + dy * dy + dz * dz)
        })
        .sum()
}

/// Ideal-gas per-molecule volume at the given temperature/pressure, via
/// V = kT/P (not a memorized "24.465 L/mol"-style constant): pressure is
/// converted from atm through `constants::ATM_TO_PASCAL` and
/// `constants::PRESSURE_AU_TO_PASCAL`, both themselves derived from
/// already-verified fundamental constants.
fn ideal_gas_volume_per_molecule_bohr3(temperature: f64, pressure_atm: f64) -> f64 {
    let pressure_pascal = pressure_atm * crate::constants::ATM_TO_PASCAL;
    let pressure_au = pressure_pascal / crate::constants::PRESSURE_AU_TO_PASCAL;
    crate::stat_mech::K_BOLTZMANN_HARTREE * temperature / pressure_au
}

/// Helmholtz free energy from a partition function: F = -kT ln(Z). Trivial
/// algebra, not a new physical claim -- factored out since it's used
/// identically for the translational, rotational, and vibrational pieces
/// below (partition functions factorize, so ln Z_total = sum of ln Z_x,
/// hence F_total = sum of F_x).
fn free_energy_from_z(z: f64, temperature: f64) -> f64 {
    -crate::stat_mech::K_BOLTZMANN_HARTREE * temperature * z.ln()
}

/// Full thermochemistry for a linear molecule at standard (or configured)
/// conditions.
///
/// Panics if any non-zero-mode frequency is negative (imaginary) -- that
/// indicates the input geometry is a saddle point, not a minimum, and
/// standard rigid-rotor/harmonic-oscillator thermochemistry is undefined
/// there.
pub fn compute_thermochemistry_linear(mol: &Molecule, config: &ThermoConfig) -> ThermoResult {
    let vib = normal_mode_analysis(mol, &config.vib_config);
    let masses: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| atomic_mass_au(a.atomic_number))
        .collect();
    let total_mass: f64 = masses.iter().sum();

    let real_frequencies: Vec<f64> = vib
        .frequencies
        .iter()
        .filter(|f| !f.is_zero_mode)
        .map(|f| f.angular_hartree)
        .collect();
    assert!(
        real_frequencies.iter().all(|&w| w > 0.0),
        "found an imaginary (negative) vibrational frequency -- geometry is \
         not a minimum on the PES, thermochemistry is undefined here"
    );

    let zpe: f64 = 0.5 * real_frequencies.iter().sum::<f64>();

    let moment_of_inertia = moment_of_inertia_linear(mol, &masses);
    let rotational_constant = 1.0 / (2.0 * moment_of_inertia);

    let t = config.temperature;
    let kt = crate::stat_mech::K_BOLTZMANN_HARTREE * t;

    // Translational.
    let volume_bohr3 = ideal_gas_volume_per_molecule_bohr3(t, config.pressure_atm);
    let z_trans = crate::stat_mech::translational_z(total_mass, volume_bohr3, t);
    let f_trans = free_energy_from_z(z_trans, t);
    let u_trans = 1.5 * kt;
    let s_trans = (u_trans - f_trans) / t;

    // Rotational (linear rigid rotor: 2 rotational degrees of freedom).
    let z_rot =
        crate::stat_mech::rotational_z_linear(rotational_constant, config.symmetry_number, t);
    let f_rot = free_energy_from_z(z_rot, t);
    let u_rot = kt;
    let s_rot = (u_rot - f_rot) / t;

    // Vibrational: product of per-mode partition functions (each already
    // includes its own ZPE contribution, absolute-energy-referenced -- see
    // `stat_mech::harmonic_oscillator_z`'s doc).
    let mut ln_z_vib_total = 0.0;
    let mut u_vib_total = 0.0;
    for &omega in &real_frequencies {
        let z_mode = crate::stat_mech::harmonic_oscillator_z(omega, t);
        ln_z_vib_total += z_mode.ln();
        let x = omega / kt;
        u_vib_total += omega * (0.5 + 1.0 / (x.exp() - 1.0));
    }
    let f_vib = -crate::stat_mech::K_BOLTZMANN_HARTREE * t * ln_z_vib_total;
    let s_vib = (u_vib_total - f_vib) / t;

    let electronic_energy = compute_electronic_energy(mol, &config.vib_config.scf_config);

    let internal_energy = electronic_energy + u_trans + u_rot + u_vib_total;
    let enthalpy = internal_energy + kt;
    let entropy = s_trans + s_rot + s_vib;
    let gibbs_free_energy = enthalpy - t * entropy;

    ThermoResult {
        temperature: t,
        electronic_energy_hartree: electronic_energy,
        zpe_hartree: zpe,
        rotational_constant_hartree: rotational_constant,
        internal_energy_hartree: internal_energy,
        enthalpy_hartree: enthalpy,
        entropy_hartree_per_kelvin: entropy,
        gibbs_free_energy_hartree: gibbs_free_energy,
        n_real_modes: vib.n_real_modes,
        n_zero_modes: vib.n_zero_modes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry_opt::{GeomOptConfig, optimize_geometry};

    /// Relax to an equilibrium geometry first -- see the module doc's
    /// "Callers must pass an (approximately) optimized geometry" note.
    ///
    /// Uses a tighter-than-default convergence threshold: `GeomOptConfig`'s
    /// default `gradient_threshold` (1e-4 Hartree/Bohr) is loose enough to
    /// leave real rotational-mode contamination (empirically ~40-180 cm⁻¹
    /// spurious curvature, occasionally a spuriously *imaginary* mode) --
    /// found while first writing these tests, see the module doc's status
    /// note. 1e-8 reliably reproduces the clean 5-zero/1-real invariant for
    /// both H2 (fully converges, ~31 steps) and HeH+ (doesn't fully
    /// converge within 200 steepest-descent steps but gets close enough,
    /// rms_gradient ~1.8e-6, for the invariant to hold).
    fn optimized(mol: &Molecule) -> Molecule {
        let config = GeomOptConfig {
            gradient_threshold: 1e-8,
            energy_threshold: 1e-12,
            max_steps: 200,
            ..Default::default()
        };
        optimize_geometry(mol, &config).molecule
    }

    /// Zero-mode count invariant: H2 (3N=6 total modes, linear diatomic)
    /// must show exactly 5 zero modes (3 translation + 2 rotation) and
    /// exactly 1 real vibrational mode.
    #[test]
    fn test_h2_zero_mode_count_invariant() {
        let mol = optimized(&Molecule::h2());
        let vib = normal_mode_analysis(&mol, &VibConfig::default());
        assert_eq!(vib.frequencies.len(), 6);
        assert_eq!(
            vib.n_zero_modes, 5,
            "expected 5 zero modes (3 trans + 2 rot) for H2, got {}: {:?}",
            vib.n_zero_modes, vib.frequencies
        );
        assert_eq!(vib.n_real_modes, 1);
    }

    /// Same invariant for HeH+ -- a different linear diatomic (heteronuclear,
    /// charged), confirming the invariant isn't an H2-specific coincidence.
    #[test]
    fn test_heh_plus_zero_mode_count_invariant() {
        let mol = optimized(&Molecule::heh_plus());
        let vib = normal_mode_analysis(&mol, &VibConfig::default());
        assert_eq!(vib.frequencies.len(), 6);
        assert_eq!(
            vib.n_zero_modes, 5,
            "expected 5 zero modes for HeH+, got {}: {:?}",
            vib.n_zero_modes, vib.frequencies
        );
        assert_eq!(vib.n_real_modes, 1);
    }

    /// Order-of-magnitude sanity: H2's HF/STO-3G harmonic stretch should
    /// land in the few-thousand cm^-1 range (real H-H stretch is ~4000
    /// cm^-1; HF systematically overestimates -- a well-known, disclosed
    /// qualitative effect, not a precise target).
    #[test]
    fn test_h2_stretch_frequency_order_of_magnitude() {
        let mol = optimized(&Molecule::h2());
        let vib = normal_mode_analysis(&mol, &VibConfig::default());
        let real_mode = vib
            .frequencies
            .iter()
            .find(|f| !f.is_zero_mode)
            .expect("H2 must have one real mode");
        assert!(
            real_mode.wavenumber_cm1 > 1000.0 && real_mode.wavenumber_cm1 < 10_000.0,
            "H2 stretch = {} cm^-1, expected a few thousand",
            real_mode.wavenumber_cm1
        );
    }

    #[test]
    fn test_hessian_is_symmetric() {
        let mol = optimized(&Molecule::h2());
        let config = VibConfig::default();
        let hessian = compute_hessian(&mol, &config.scf_config, config.fd_step);
        let n = 3 * mol.n_atoms();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (hessian[i * n + j] - hessian[j * n + i]).abs() < 1e-10,
                    "Hessian not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_mass_weight_hessian_dimensions_and_symmetry() {
        let hessian = vec![
            4.0, 1.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0,
            4.0, // fake 1-atom-equivalent 3x3 block won't be used
        ];
        // Use a real 2-atom (n=6) case instead for a meaningful check.
        let mol = optimized(&Molecule::h2());
        let masses: Vec<f64> = mol
            .atoms
            .iter()
            .map(|a| atomic_mass_au(a.atomic_number))
            .collect();
        let real_hessian = compute_hessian(&mol, &RhfConfig::default(), 0.01);
        let mw = mass_weight_hessian(&real_hessian, &masses);
        let n = 6;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (mw[i * n + j] - mw[j * n + i]).abs() < 1e-10,
                    "mass-weighted Hessian not symmetric at ({i},{j})"
                );
            }
        }
        let _ = hessian; // silence unused-fake-data warning; real check is above
    }

    /// Physical positivity: ZPE, all partition functions, and entropy must
    /// be positive at 298.15 K -- mirrors the exact pattern `stat_mech.rs`'s
    /// own existing tests already use.
    #[test]
    fn test_h2_thermochemistry_physical_positivity() {
        let mol = optimized(&Molecule::h2());
        let config = ThermoConfig {
            symmetry_number: 2, // H2 is homonuclear
            ..Default::default()
        };
        let thermo = compute_thermochemistry_linear(&mol, &config);

        assert!(thermo.zpe_hartree > 0.0, "ZPE should be positive");
        assert!(
            thermo.rotational_constant_hartree > 0.0,
            "rotational constant should be positive"
        );
        assert!(
            thermo.entropy_hartree_per_kelvin > 0.0,
            "entropy should be positive at 298.15K: {}",
            thermo.entropy_hartree_per_kelvin
        );
        assert!(
            thermo.enthalpy_hartree > thermo.electronic_energy_hartree,
            "enthalpy should exceed bare electronic energy (thermal + ZPE additions)"
        );
        assert!(
            thermo.gibbs_free_energy_hartree.is_finite(),
            "Gibbs free energy should be finite"
        );
        assert_eq!(thermo.n_real_modes, 1);
        assert_eq!(thermo.n_zero_modes, 5);
    }

    #[test]
    fn test_heh_plus_thermochemistry_physical_positivity() {
        let mol = optimized(&Molecule::heh_plus());
        let config = ThermoConfig {
            symmetry_number: 1, // heteronuclear
            ..Default::default()
        };
        let thermo = compute_thermochemistry_linear(&mol, &config);

        assert!(thermo.zpe_hartree > 0.0);
        assert!(thermo.rotational_constant_hartree > 0.0);
        assert!(thermo.entropy_hartree_per_kelvin > 0.0);
        assert!(thermo.gibbs_free_energy_hartree.is_finite());
    }

    #[test]
    #[should_panic(expected = "hessian size")]
    fn test_mass_weight_hessian_size_mismatch_panics() {
        let bad_hessian = vec![0.0; 4]; // not 3*n_atoms squared for any sane n_atoms
        let masses = vec![1.0, 2.0];
        mass_weight_hessian(&bad_hessian, &masses);
    }
}
