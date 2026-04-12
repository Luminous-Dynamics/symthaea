// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! General Relativity from first principles.
//!
//! Implements the Schwarzschild solution, geodesic equations, and
//! Friedmann cosmology. Gravity as spacetime curvature.
//!
//! References:
//! - Weinberg, S. (1972). *Gravitation and Cosmology*. Wiley.
//! - Misner, Thorne & Wheeler (1973). *Gravitation*. W.H. Freeman.
//! - Carroll, S. (2004). *Spacetime and Geometry*. Addison-Wesley.

use std::f64::consts::PI;

/// Gravitational constant in natural units (GeV⁻²)
pub const G_NEWTON_NATURAL: f64 = 6.708_83e-39;

/// Planck mass in GeV
pub const M_PLANCK: f64 = 1.220_890e19;

/// Planck length in GeV⁻¹
pub const L_PLANCK: f64 = 1.0 / M_PLANCK;

/// Schwarzschild radius for mass M (in natural units: r_s = 2GM)
pub fn schwarzschild_radius(mass_gev: f64) -> f64 {
    2.0 * G_NEWTON_NATURAL * mass_gev
}

/// Gravitational redshift: Δν/ν = GM/(rc²) = GM/r in natural units
pub fn gravitational_redshift(mass_gev: f64, radius_gev_inv: f64) -> f64 {
    G_NEWTON_NATURAL * mass_gev / radius_gev_inv
}

/// Schwarzschild metric component g_tt = -(1 - r_s/r)
pub fn schwarzschild_g_tt(r: f64, r_s: f64) -> f64 {
    -(1.0 - r_s / r)
}

/// Schwarzschild metric component g_rr = 1/(1 - r_s/r)
pub fn schwarzschild_g_rr(r: f64, r_s: f64) -> f64 {
    1.0 / (1.0 - r_s / r)
}

/// Orbital precession per orbit (Einstein's perihelion advance).
///
/// Δφ = 6πGM / (a(1-e²)c²) = 6πGM / (a(1-e²)) in natural units
///
/// `mass` in GeV, `semi_major` in GeV⁻¹, `eccentricity` dimensionless.
pub fn perihelion_precession(mass_gev: f64, semi_major_gev_inv: f64, eccentricity: f64) -> f64 {
    6.0 * PI * G_NEWTON_NATURAL * mass_gev
        / (semi_major_gev_inv * (1.0 - eccentricity * eccentricity))
}

/// Hawking temperature of a black hole: T_H = ℏc³/(8πGMk_B)
/// In natural units: T_H = 1/(8πGM)
pub fn hawking_temperature(mass_gev: f64) -> f64 {
    1.0 / (8.0 * PI * G_NEWTON_NATURAL * mass_gev)
}

/// Bekenstein-Hawking entropy: S_BH = A/(4G) = 4πG M² (in natural units)
pub fn black_hole_entropy(mass_gev: f64) -> f64 {
    4.0 * PI * G_NEWTON_NATURAL * mass_gev * mass_gev
}

// ── Friedmann Cosmology ─────────────────────────────────────────────────────

/// Friedmann equation: H² = (8πG/3)ρ - k/a² + Λ/3
///
/// Returns the Hubble parameter H for given energy density ρ,
/// curvature k, scale factor a, and cosmological constant Λ.
pub fn hubble_parameter(rho_gev4: f64, k: f64, a: f64, lambda_gev2: f64) -> f64 {
    let h_sq = 8.0 * PI * G_NEWTON_NATURAL * rho_gev4 / 3.0 - k / (a * a) + lambda_gev2 / 3.0;
    if h_sq > 0.0 {
        h_sq.sqrt()
    } else {
        0.0
    }
}

/// Critical density: ρ_c = 3H²/(8πG)
pub fn critical_density(h_gev: f64) -> f64 {
    3.0 * h_gev * h_gev / (8.0 * PI * G_NEWTON_NATURAL)
}

/// Scale factor evolution for matter-dominated universe: a(t) ∝ t^(2/3)
pub fn matter_dominated_scale_factor(t: f64, t0: f64, a0: f64) -> f64 {
    a0 * (t / t0).powf(2.0 / 3.0)
}

/// Scale factor for radiation-dominated universe: a(t) ∝ t^(1/2)
pub fn radiation_dominated_scale_factor(t: f64, t0: f64, a0: f64) -> f64 {
    a0 * (t / t0).sqrt()
}

/// Geodesic deviation equation for radial free-fall in Schwarzschild spacetime.
/// Returns the proper acceleration at distance r from mass M.
/// a = GM/r² = G_N M / r² (Newtonian limit)
pub fn radial_acceleration(mass_gev: f64, r_gev_inv: f64) -> f64 {
    G_NEWTON_NATURAL * mass_gev / (r_gev_inv * r_gev_inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarzschild_radius_proportional_to_mass() {
        let r1 = schwarzschild_radius(1.0);
        let r2 = schwarzschild_radius(2.0);
        assert!((r2 / r1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hawking_temperature_inverse_mass() {
        // T ∝ 1/M
        let t1 = hawking_temperature(1e10);
        let t2 = hawking_temperature(2e10);
        assert!((t1 / t2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_black_hole_entropy_positive() {
        let s = black_hole_entropy(M_PLANCK);
        assert!(s > 0.0, "BH entropy should be positive: {}", s);
    }

    #[test]
    fn test_metric_signature() {
        // g_tt < 0 outside horizon, g_rr > 0
        let r_s = 1.0;
        let r = 2.0 * r_s; // Outside horizon

        assert!(schwarzschild_g_tt(r, r_s) < 0.0, "g_tt should be negative");
        assert!(schwarzschild_g_rr(r, r_s) > 0.0, "g_rr should be positive");
    }

    #[test]
    fn test_metric_at_horizon() {
        let r_s = 1.0;
        // At horizon: g_tt → 0, g_rr → ∞
        let g_tt = schwarzschild_g_tt(r_s, r_s);
        assert!(g_tt.abs() < 1e-10, "g_tt at horizon should be 0");
    }

    #[test]
    fn test_friedmann_flat_universe() {
        // For flat universe (k=0, Λ=0): H² = 8πGρ/3
        let rho = 1.0;
        let h = hubble_parameter(rho, 0.0, 1.0, 0.0);
        let expected = (8.0 * PI * G_NEWTON_NATURAL * rho / 3.0).sqrt();
        assert!((h - expected).abs() < 1e-30);
    }

    #[test]
    fn test_matter_dominated_scaling() {
        // a(2t₀) / a(t₀) = 2^(2/3)
        let a1 = matter_dominated_scale_factor(1.0, 1.0, 1.0);
        let a2 = matter_dominated_scale_factor(2.0, 1.0, 1.0);
        assert!((a2 / a1 - 2.0_f64.powf(2.0 / 3.0)).abs() < 1e-10);
    }
}
