// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Continuum electrodynamics — dielectric response, conductivity, wave propagation.
//!
//! Computes material electromagnetic properties from first principles concepts:
//! permittivity, conductivity, skin depth, impedance matching, waveguide modes.
//!
//! References:
//! - Jackson, J. D. (1999). *Classical Electrodynamics*. Wiley.
//! - Griffiths, D. J. (2017). *Introduction to Electrodynamics*. Cambridge UP.

use crate::maxwell::{C_LIGHT, EPS_0, MU_0};

/// Drude model for metal conductivity.
/// σ(ω) = σ_0 / (1 - iωτ)  where σ_0 = ne²τ/m
///
/// Returns (real, imaginary) parts of complex conductivity.
pub fn drude_conductivity(sigma_dc: f64, omega: f64, tau: f64) -> (f64, f64) {
    let denom = 1.0 + omega * omega * tau * tau;
    let re = sigma_dc / denom;
    let im = sigma_dc * omega * tau / denom;
    (re, im)
}

/// Complex dielectric function from Drude model.
/// ε(ω) = ε_∞ - ω_p²/(ω² + iω/τ)
///
/// Returns (real, imaginary) parts of relative permittivity.
pub fn drude_permittivity(eps_inf: f64, omega_p: f64, omega: f64, gamma: f64) -> (f64, f64) {
    let denom = omega * omega + gamma * gamma;
    let _re = eps_inf
        - omega_p * omega_p * (omega * omega - gamma * gamma) / (denom * omega * omega + 1e-30);
    let _im = omega_p * omega_p * gamma / (omega * denom + 1e-30);

    // Simplified: ε(ω) = ε_∞ - ω_p²/(ω(ω + iγ))
    let eps_re = eps_inf - omega_p * omega_p / (omega * omega + gamma * gamma);
    let eps_im = omega_p * omega_p * gamma / (omega * (omega * omega + gamma * gamma) + 1e-30);
    (eps_re, eps_im)
}

/// Skin depth: δ = √(2/(ωμσ)) — how far EM fields penetrate a conductor.
pub fn skin_depth(omega: f64, mu: f64, sigma: f64) -> f64 {
    (2.0 / (omega * mu * sigma)).sqrt()
}

/// Electromagnetic wave impedance in a medium.
/// Z = √(μ/ε) for lossless, Z = √(iωμ/(σ + iωε)) for lossy.
pub fn wave_impedance(eps_r: f64, mu_r: f64) -> f64 {
    (MU_0 * mu_r / (EPS_0 * eps_r)).sqrt()
}

/// Fresnel reflection coefficient at normal incidence.
/// r = (n₁ - n₂)/(n₁ + n₂) where n = √(εμ)
pub fn fresnel_reflection(n1: f64, n2: f64) -> f64 {
    (n1 - n2) / (n1 + n2)
}

/// Reflectance (power): R = |r|²
pub fn reflectance(n1: f64, n2: f64) -> f64 {
    let r = fresnel_reflection(n1, n2);
    r * r
}

/// Brewster's angle: θ_B = arctan(n₂/n₁) — angle of zero p-polarization reflection.
pub fn brewster_angle(n1: f64, n2: f64) -> f64 {
    (n2 / n1).atan()
}

/// Critical angle for total internal reflection: θ_c = arcsin(n₂/n₁).
/// Only exists when n₁ > n₂.
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if n1 <= n2 {
        return None;
    }
    Some((n2 / n1).asin())
}

/// Plasma frequency: ω_p = √(ne²/(ε₀m))
/// Below this frequency, EM waves cannot propagate (are evanescent).
pub fn plasma_frequency(electron_density: f64, electron_mass: f64) -> f64 {
    let e = 1.602_176_634e-19; // Coulombs
    (electron_density * e * e / (EPS_0 * electron_mass)).sqrt()
}

/// Refractive index from permittivity: n = √ε_r (lossless, non-magnetic)
pub fn refractive_index(eps_r: f64) -> f64 {
    eps_r.sqrt()
}

/// Group velocity: v_g = c/n_g where n_g = n + ω(dn/dω)
/// For simple dispersion n(ω) = √(1 - ω_p²/ω²):
pub fn group_velocity_plasma(omega: f64, omega_p: f64) -> f64 {
    if omega <= omega_p {
        return 0.0; // Evanescent
    }
    C_LIGHT * (1.0 - omega_p * omega_p / (omega * omega)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_impedance() {
        // Z₀ = √(μ₀/ε₀) ≈ 377 Ω
        let z0 = wave_impedance(1.0, 1.0);
        assert!((z0 - 376.73).abs() < 1.0, "Z₀ = {:.2} Ω, expected ~377", z0);
    }

    #[test]
    fn test_glass_reflection() {
        // Air-glass (n=1.5): R ≈ 4%
        let r = reflectance(1.0, 1.5);
        assert!((r - 0.04).abs() < 0.01, "R = {:.4}, expected ~0.04", r);
    }

    #[test]
    fn test_total_internal_reflection() {
        // Glass → air: critical angle = arcsin(1/1.5) ≈ 41.8°
        let theta_c = critical_angle(1.5, 1.0).unwrap();
        assert!((theta_c.to_degrees() - 41.8).abs() < 0.5);
        // Air → glass: no TIR
        assert!(critical_angle(1.0, 1.5).is_none());
    }

    #[test]
    fn test_brewster_angle() {
        // Air → glass: θ_B = arctan(1.5) ≈ 56.3°
        let theta_b = brewster_angle(1.0, 1.5).to_degrees();
        assert!((theta_b - 56.3).abs() < 0.5);
    }

    #[test]
    fn test_skin_depth_copper() {
        // Copper at 1 GHz: δ ≈ 2.1 μm
        let omega = 2.0 * std::f64::consts::PI * 1e9;
        let sigma = 5.96e7; // S/m
        let delta = skin_depth(omega, MU_0, sigma);
        assert!(
            delta > 1e-6 && delta < 5e-6,
            "Cu skin depth at 1GHz = {:.2e} m",
            delta
        );
    }

    #[test]
    fn test_group_velocity_below_plasma() {
        let vg = group_velocity_plasma(0.5, 1.0); // ω < ω_p
        assert_eq!(vg, 0.0, "No propagation below plasma frequency");
    }
}
