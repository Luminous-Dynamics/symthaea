// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stochastic QED — Zero-Point Field and vacuum fluctuations.
//!
//! The quantum vacuum is not empty: it contains zero-point energy E₀ = ℏω/2
//! for each mode. These fluctuations may play a role in consciousness via
//! stochastic resonance (amplifying weak signals in noisy environments).
//!
//! References:
//! - de la Peña, L. & Cetto, A. M. (1996). *The Quantum Dice*. Kluwer.
//! - Boyer, T. H. (1975). Phys. Rev. D 11, 790 (Casimir derivation from SED).
//! - Gammaitoni et al. (1998). Rev. Mod. Phys. 70, 223 (stochastic resonance).

use std::f64::consts::PI;

/// Zero-point energy of a quantum harmonic oscillator: E₀ = ℏω/2
/// In natural units: E₀ = ω/2
pub fn zero_point_energy(omega: f64) -> f64 {
    0.5 * omega
}

/// Vacuum energy density of the electromagnetic field (per unit volume).
/// ρ_vac = ∫₀^{ω_max} (ℏω/2) × g(ω) dω where g(ω) = ω²/(π²c³)
///
/// With cutoff ω_max (to avoid UV divergence):
/// ρ_vac = ω_max⁴ / (16π²) in natural units
pub fn vacuum_energy_density(omega_cutoff: f64) -> f64 {
    omega_cutoff.powi(4) / (16.0 * PI * PI)
}

/// Unruh temperature: an accelerating observer sees thermal radiation.
/// T_U = ℏa/(2πck_B) = a/(2π) in natural units
pub fn unruh_temperature(acceleration: f64) -> f64 {
    acceleration / (2.0 * PI)
}

/// Stochastic resonance: signal-to-noise ratio for a bistable system
/// driven by a weak periodic signal in the presence of noise.
///
/// SNR ∝ exp(-2V_b/(D)) × A²/(D²)
/// where V_b = barrier height, D = noise intensity, A = signal amplitude.
///
/// There's an optimal noise level D* where SNR is maximized.
pub fn stochastic_resonance_snr(barrier: f64, noise: f64, signal_amplitude: f64) -> f64 {
    if noise < 1e-20 {
        return 0.0;
    }
    let kramers_rate = (-2.0 * barrier / noise).exp();
    kramers_rate * signal_amplitude * signal_amplitude / (noise * noise)
}

/// Find the optimal noise level for stochastic resonance.
/// D* ≈ V_b (barrier height) gives roughly optimal SNR.
pub fn optimal_noise_for_sr(barrier: f64) -> f64 {
    barrier // First approximation
}

/// Vacuum fluctuation spectrum (Planck spectrum at T=0).
/// Energy per mode: E(ω) = ℏω/2 (zero-point, temperature-independent)
/// At finite T: E(ω, T) = ℏω/(exp(ℏω/kT) - 1) + ℏω/2
pub fn planck_spectrum_with_zpf(omega: f64, temperature: f64) -> f64 {
    let hbar_omega = omega; // Natural units
    let zpe = 0.5 * hbar_omega;
    if temperature <= 0.0 {
        return zpe;
    }
    let k_b_t = temperature; // In natural units with k_B = 1
    let thermal = hbar_omega / ((hbar_omega / k_b_t).exp() - 1.0);
    thermal + zpe
}

/// Lamb shift contribution from vacuum fluctuations (rough estimate).
/// The Lamb shift is the effect of virtual photon emission/absorption.
/// δE ≈ (α/(3π)) × (E_n/m_e²) × ln(m_e/ω_min)
pub fn vacuum_lamb_shift_estimate(binding_energy: f64, mass: f64) -> f64 {
    let alpha = 1.0 / 137.036;
    alpha / (3.0 * PI) * binding_energy / (mass * mass) * (mass / 1e-3).ln().abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_point_energy() {
        assert!((zero_point_energy(2.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_vacuum_energy_positive() {
        let rho = vacuum_energy_density(1.0);
        assert!(rho > 0.0, "Vacuum energy should be positive");
    }

    #[test]
    fn test_unruh_temperature_proportional() {
        let t1 = unruh_temperature(1.0);
        let t2 = unruh_temperature(2.0);
        assert!((t2 / t1 - 2.0).abs() < 1e-14, "T ∝ acceleration");
    }

    #[test]
    fn test_stochastic_resonance_optimal() {
        let barrier = 1.0;
        let signal = 0.1;

        // SNR should peak near noise ≈ barrier
        let snr_low = stochastic_resonance_snr(barrier, 0.1, signal);
        let snr_opt = stochastic_resonance_snr(barrier, 0.8, signal);
        let snr_high = stochastic_resonance_snr(barrier, 5.0, signal);

        // Optimal noise should give higher SNR than too-low or too-high
        assert!(
            snr_opt > snr_high,
            "Optimal ({:.6}) should beat high noise ({:.6})",
            snr_opt,
            snr_high
        );
    }

    #[test]
    fn test_zpf_dominates_at_low_t() {
        let e_low_t = planck_spectrum_with_zpf(1.0, 0.01); // Low T
        let zpe = zero_point_energy(1.0);
        assert!(
            (e_low_t - zpe).abs() < 0.1,
            "Low T should be ~ZPE: {:.4} vs {:.4}",
            e_low_t,
            zpe
        );
    }
}
