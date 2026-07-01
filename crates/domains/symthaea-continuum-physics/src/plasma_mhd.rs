// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Magnetohydrodynamics (MHD) — plasma physics from first principles.
//!
//! Ideal MHD equations:
//! ∂ρ/∂t + ∇·(ρv) = 0           (mass)
//! ρ(∂v/∂t + v·∇v) = -∇p + J×B  (momentum)
//! ∂B/∂t = ∇×(v×B)               (induction)
//! ∇·B = 0                        (no monopoles)
//!
//! References:
//! - Freidberg, J. P. (2014). *Ideal MHD*. Cambridge UP.
//! - Goedbloed & Poedts (2004). *Principles of MHD*. Cambridge UP.

use crate::maxwell::MU_0;
use std::f64::consts::PI;

/// Boltzmann constant (J/K)
const K_B: f64 = 1.380_649e-23;
/// Electron charge (C)
const E_CHARGE: f64 = 1.602_176_634e-19;
/// Electron mass (kg)
const M_ELECTRON: f64 = 9.109_383_702e-31;
/// Proton mass (kg)
#[allow(dead_code)]
const M_PROTON: f64 = 1.672_621_924e-27;

/// Debye length: λ_D = √(ε₀ k_B T / (n e²))
/// The distance over which electric fields are screened in a plasma.
pub fn debye_length(temperature: f64, density: f64) -> f64 {
    let eps_0 = crate::maxwell::EPS_0;
    (eps_0 * K_B * temperature / (density * E_CHARGE * E_CHARGE)).sqrt()
}

/// Plasma frequency: ω_p = √(n e² / (ε₀ m_e))
pub fn plasma_frequency(density: f64) -> f64 {
    let eps_0 = crate::maxwell::EPS_0;
    (density * E_CHARGE * E_CHARGE / (eps_0 * M_ELECTRON)).sqrt()
}

/// Cyclotron frequency: ω_c = eB/m
pub fn cyclotron_frequency(b_field: f64, mass: f64) -> f64 {
    E_CHARGE * b_field / mass
}

/// Larmor radius: r_L = mv_⊥/(eB) = v_th/ω_c
pub fn larmor_radius(temperature: f64, b_field: f64, mass: f64) -> f64 {
    let v_thermal = (K_B * temperature / mass).sqrt();
    mass * v_thermal / (E_CHARGE * b_field)
}

/// Alfvén speed: v_A = B/√(μ₀ ρ) — speed of magnetic disturbances
pub fn alfven_speed(b_field: f64, mass_density: f64) -> f64 {
    b_field / (MU_0 * mass_density).sqrt()
}

/// Sound speed in plasma: c_s = √(γ k_B T / m_i)
pub fn plasma_sound_speed(temperature: f64, ion_mass: f64, gamma: f64) -> f64 {
    (gamma * K_B * temperature / ion_mass).sqrt()
}

/// Plasma beta: β = 2μ₀ n k_B T / B² = thermal pressure / magnetic pressure
pub fn plasma_beta(density: f64, temperature: f64, b_field: f64) -> f64 {
    2.0 * MU_0 * density * K_B * temperature / (b_field * b_field)
}

/// Magnetic pressure: p_B = B²/(2μ₀)
pub fn magnetic_pressure(b_field: f64) -> f64 {
    b_field * b_field / (2.0 * MU_0)
}

/// Magnetic Reynolds number: R_m = μ₀ σ L v
/// If R_m >> 1, field is "frozen in" to the plasma.
pub fn magnetic_reynolds_number(conductivity: f64, length: f64, velocity: f64) -> f64 {
    MU_0 * conductivity * length * velocity
}

/// Spitzer resistivity (collisional): η = π Z e² m_e^(1/2) ln(Λ) / (2(2πk_BT)^(3/2))
/// Simplified: η ∝ T^(-3/2)
pub fn spitzer_resistivity(temperature: f64, z_eff: f64, coulomb_log: f64) -> f64 {
    let coeff = PI * z_eff * E_CHARGE * E_CHARGE * M_ELECTRON.sqrt() * coulomb_log;
    let denom = 2.0 * (2.0 * PI * K_B * temperature).powf(1.5);
    coeff / denom
}

/// Coulomb logarithm: ln(Λ) ≈ ln(λ_D / b_min)
/// Approximate: ln(Λ) ≈ 23 - ln(n^(1/2) T^(-3/2)) for T in eV, n in cm⁻³
pub fn coulomb_logarithm(temperature_ev: f64, density_cm3: f64) -> f64 {
    (23.0 - 0.5 * density_cm3.ln() + 1.5 * temperature_ev.ln()).max(1.0)
}

/// Ideal MHD equilibrium for a Z-pinch: B_θ(r) = μ₀ I r / (2π a²)  for r < a
pub fn z_pinch_field(current: f64, radius: f64, pinch_radius: f64) -> f64 {
    if radius < pinch_radius {
        MU_0 * current * radius / (2.0 * PI * pinch_radius * pinch_radius)
    } else {
        MU_0 * current / (2.0 * PI * radius)
    }
}

/// Bennett condition for Z-pinch equilibrium: I² = (8π N k_B T) / μ₀
/// where N is the line density (particles per unit length).
pub fn bennett_current(line_density: f64, temperature: f64) -> f64 {
    (8.0 * PI * line_density * K_B * temperature / MU_0).sqrt()
}

/// Tokamak safety factor: q = r B_φ / (R B_θ)
/// For stability: q(edge) > 2 (Kruskal-Shafranov limit)
pub fn safety_factor(r: f64, major_radius: f64, b_toroidal: f64, b_poloidal: f64) -> f64 {
    r * b_toroidal / (major_radius * b_poloidal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debye_length_order() {
        // Tokamak: T ≈ 10 keV ≈ 1.16e8 K, n ≈ 10²⁰ m⁻³
        // λ_D ≈ 7.4 × 10⁻⁵ m ≈ 74 μm
        let lambda_d = debye_length(1.16e8, 1e20);
        assert!(
            lambda_d > 1e-5 && lambda_d < 1e-3,
            "Tokamak Debye length = {:.2e} m",
            lambda_d
        );
    }

    #[test]
    fn test_alfven_speed() {
        // Solar corona: B ≈ 10⁻³ T, ρ ≈ 10⁻¹² kg/m³
        // v_A ≈ 10⁶ m/s
        let va = alfven_speed(1e-3, 1e-12);
        assert!(
            va > 1e5 && va < 1e7,
            "Coronal Alfvén speed = {:.2e} m/s",
            va
        );
    }

    #[test]
    fn test_plasma_beta_low_field() {
        // High β (thermal dominated): β >> 1
        let beta = plasma_beta(1e20, 1e8, 0.01);
        assert!(beta > 1.0, "Weak field → high β = {:.2}", beta);
    }

    #[test]
    fn test_plasma_beta_high_field() {
        // Low β (magnetically dominated): β << 1
        let beta = plasma_beta(1e20, 1e6, 10.0);
        assert!(beta < 1.0, "Strong field → low β = {:.2e}", beta);
    }

    #[test]
    fn test_magnetic_pressure() {
        // B = 1 T → p_B = 1/(2×4π×10⁻⁷) ≈ 4×10⁵ Pa ≈ 4 atm
        let p = magnetic_pressure(1.0);
        assert!((p - 3.98e5).abs() < 1e4, "p_B(1T) = {:.2e} Pa", p);
    }

    #[test]
    fn test_safety_factor_kruskal_shafranov() {
        // Stable tokamak: q > 2 at the edge
        let q = safety_factor(0.5, 1.5, 5.0, 0.5);
        assert!(q > 2.0, "q = {:.2}, should be > 2 for stability", q);
    }

    #[test]
    fn test_cyclotron_frequency() {
        // Electron in B=1T: ω_ce = eB/m_e ≈ 1.76 × 10¹¹ rad/s
        let omega = cyclotron_frequency(1.0, M_ELECTRON);
        assert!(
            (omega - 1.76e11).abs() / 1.76e11 < 0.01,
            "ω_ce = {:.3e} rad/s",
            omega
        );
    }
}
