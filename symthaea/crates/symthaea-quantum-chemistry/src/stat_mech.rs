// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Statistical Mechanics from first principles.
//!
//! Computes thermodynamic properties from molecular energy levels via
//! partition functions. Bridges quantum mechanics → macroscopic observables.
//!
//! Z(T) = Σ_i g_i exp(-E_i / kT)
//! F = -kT ln(Z)  (Helmholtz free energy)
//! S = -∂F/∂T     (entropy)
//! C_v = -T ∂²F/∂T²  (heat capacity)
//!
//! References:
//! - McQuarrie, D. A. (2000). *Statistical Mechanics*. University Science Books.
//! - Hill, T. L. (1960). *An Introduction to Statistical Thermodynamics*. Dover.

use std::f64::consts::PI;

/// Boltzmann constant in Hartree/Kelvin
pub const K_BOLTZMANN_HARTREE: f64 = 3.166_811_563e-6;

/// Boltzmann constant in eV/Kelvin
pub const K_BOLTZMANN_EV: f64 = 8.617_333_262e-5;

/// Compute the canonical partition function Z(T) from energy levels.
///
/// Z = Σ_i g_i × exp(-E_i / kT)
///
/// `energies` are in Hartree, `degeneracies` are multiplicity factors,
/// `temperature` is in Kelvin.
pub fn partition_function(energies: &[f64], degeneracies: &[f64], temperature: f64) -> f64 {
    if temperature <= 0.0 {
        return degeneracies.first().copied().unwrap_or(1.0);
    }

    let beta = 1.0 / (K_BOLTZMANN_HARTREE * temperature);
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);

    energies
        .iter()
        .zip(degeneracies.iter())
        .map(|(&e, &g)| g * (-(e - e_min) * beta).exp())
        .sum()
}

/// Helmholtz free energy: F = -kT ln(Z) + E_0
pub fn helmholtz_free_energy(energies: &[f64], degeneracies: &[f64], temperature: f64) -> f64 {
    let z = partition_function(energies, degeneracies, temperature);
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    e_min - K_BOLTZMANN_HARTREE * temperature * z.ln()
}

/// Internal energy: U = -∂ln(Z)/∂β + E_0
pub fn internal_energy(energies: &[f64], degeneracies: &[f64], temperature: f64) -> f64 {
    if temperature <= 0.0 {
        return energies.first().copied().unwrap_or(0.0);
    }

    let beta = 1.0 / (K_BOLTZMANN_HARTREE * temperature);
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let z = partition_function(energies, degeneracies, temperature);

    let numerator: f64 = energies
        .iter()
        .zip(degeneracies.iter())
        .map(|(&e, &g)| g * (e - e_min) * (-(e - e_min) * beta).exp())
        .sum();

    e_min + numerator / z
}

/// Entropy: S = k[ln(Z) + β<E>] (in Hartree/Kelvin)
pub fn entropy(energies: &[f64], degeneracies: &[f64], temperature: f64) -> f64 {
    if temperature <= 0.0 {
        return 0.0;
    }
    let f = helmholtz_free_energy(energies, degeneracies, temperature);
    let u = internal_energy(energies, degeneracies, temperature);
    (u - f) / temperature
}

/// Heat capacity at constant volume: C_v = ∂U/∂T (numerical derivative)
pub fn heat_capacity(energies: &[f64], degeneracies: &[f64], temperature: f64) -> f64 {
    let dt = temperature * 1e-4;
    let u_plus = internal_energy(energies, degeneracies, temperature + dt);
    let u_minus = internal_energy(energies, degeneracies, temperature - dt);
    (u_plus - u_minus) / (2.0 * dt)
}

/// Quantum harmonic oscillator partition function.
/// Z_vib = 1 / (2 sinh(ℏω/2kT))
pub fn harmonic_oscillator_z(frequency_hartree: f64, temperature: f64) -> f64 {
    let x = frequency_hartree / (2.0 * K_BOLTZMANN_HARTREE * temperature);
    1.0 / (2.0 * x.sinh())
}

/// Translational partition function (ideal gas, 3D box of volume V).
/// Z_trans = V × (2πmkT/h²)^(3/2)
pub fn translational_z(mass_au: f64, volume_bohr3: f64, temperature: f64) -> f64 {
    let thermal_wavelength_sq = 2.0 * PI * mass_au * K_BOLTZMANN_HARTREE * temperature;
    volume_bohr3 * thermal_wavelength_sq.powf(1.5)
}

/// Rotational partition function (rigid rotor, linear molecule).
/// Z_rot = kT / (σ × B)  where B = ℏ²/(2I) is the rotational constant
pub fn rotational_z_linear(
    rotational_constant: f64,
    symmetry_number: u32,
    temperature: f64,
) -> f64 {
    K_BOLTZMANN_HARTREE * temperature / (symmetry_number as f64 * rotational_constant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_level_system() {
        // Two-level system: E = [0, ε], g = [1, 1]
        // Z = 1 + exp(-ε/kT)
        let eps = 0.01; // Hartree
        let t = 3000.0; // K
        let z = partition_function(&[0.0, eps], &[1.0, 1.0], t);
        let expected = 1.0 + (-eps / (K_BOLTZMANN_HARTREE * t)).exp();
        assert!((z - expected).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_zero_at_t0() {
        // Third law: S → 0 as T → 0 (non-degenerate ground state)
        let s = entropy(&[0.0, 0.1, 0.2], &[1.0, 1.0, 1.0], 1.0);
        assert!(s.abs() < 1e-10, "S(T→0) = {:.2e}, expected ~0", s);
    }

    #[test]
    fn test_entropy_positive() {
        let s = entropy(&[0.0, 0.01, 0.02], &[1.0, 1.0, 1.0], 5000.0);
        assert!(s > 0.0, "Entropy should be positive at finite T: {}", s);
    }

    #[test]
    fn test_heat_capacity_positive() {
        let cv = heat_capacity(&[0.0, 0.01, 0.02], &[1.0, 2.0, 1.0], 3000.0);
        assert!(cv > 0.0, "C_v should be positive: {}", cv);
    }

    #[test]
    fn test_harmonic_oscillator_high_t() {
        // At high T, Z_vib → kT/(ℏω) (classical limit)
        let omega = 0.01; // Hartree
        let t = 50000.0; // Very high T
        let z = harmonic_oscillator_z(omega, t);
        let classical = K_BOLTZMANN_HARTREE * t / omega;
        assert!(
            (z - classical).abs() / classical < 0.1,
            "Z_vib={:.4}, classical={:.4}",
            z,
            classical
        );
    }
}
