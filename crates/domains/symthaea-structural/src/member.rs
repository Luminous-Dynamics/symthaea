// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Axially loaded members: stress/strain/elongation and Euler buckling.

use std::f64::consts::PI;

/// Axial (normal) stress σ = F/A (Pa). Tension positive.
pub fn axial_stress(force: f64, area: f64) -> f64 {
    force / area
}

/// Axial strain ε = σ/E (dimensionless), Hooke's law.
pub fn axial_strain(stress: f64, youngs_modulus: f64) -> f64 {
    stress / youngs_modulus
}

/// Axial elongation δ = F·L/(A·E) (m).
pub fn axial_elongation(force: f64, length: f64, area: f64, youngs_modulus: f64) -> f64 {
    force * length / (area * youngs_modulus)
}

/// Euler critical buckling load P_cr = π²·E·I / (K·L)² (N).
///
/// `k_factor` is the effective-length factor: 1.0 pinned-pinned, 0.5
/// fixed-fixed, 0.699 fixed-pinned, 2.0 fixed-free (cantilever column).
pub fn euler_buckling_load(
    youngs_modulus: f64,
    moment_of_inertia: f64,
    length: f64,
    k_factor: f64,
) -> f64 {
    let effective = k_factor * length;
    PI * PI * youngs_modulus * moment_of_inertia / (effective * effective)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axial_rod_hand_calc() {
        // F=10 kN, A=1 cm²=1e-4 m², L=1 m, E=200 GPa.
        // σ = 100 MPa, ε = 5e-4, δ = 0.5 mm.
        let sigma = axial_stress(10_000.0, 1e-4);
        assert!((sigma - 100.0e6).abs() < 1.0);
        assert!((axial_strain(sigma, 200e9) - 5e-4).abs() < 1e-9);
        assert!((axial_elongation(10_000.0, 1.0, 1e-4, 200e9) - 5e-4).abs() < 1e-9);
    }

    #[test]
    fn pinned_column_buckling() {
        // E=200 GPa, I=4.16667e-6 m⁴, L=3 m, K=1 → P_cr ≈ 913.9 kN.
        let p_cr = euler_buckling_load(200e9, 4.166_666_67e-6, 3.0, 1.0);
        assert!((p_cr - 913_857.0).abs() < 500.0, "P_cr={}", p_cr);
    }

    #[test]
    fn shorter_column_buckles_at_higher_load() {
        let long = euler_buckling_load(200e9, 4e-6, 4.0, 1.0);
        let short = euler_buckling_load(200e9, 4e-6, 2.0, 1.0);
        assert!(short > long);
        assert!((short / long - 4.0).abs() < 1e-6); // ∝ 1/L²
    }
}
