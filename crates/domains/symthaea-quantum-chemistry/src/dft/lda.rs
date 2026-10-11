// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local Density Approximation (LDA) exchange-correlation functional.
//!
//! - **Slater exchange**: ε_x(ρ) = -3/4 × (3ρ/π)^(1/3)
//! - **VWN correlation**: Vosko-Wilk-Nusair (1980) parametrization of
//!   the homogeneous electron gas correlation energy.
//!
//! References:
//! - Slater, J. C. (1951). Phys. Rev. 81, 385.
//! - Vosko, Wilk & Nusair (1980). Can. J. Phys. 58, 1200.

use std::f64::consts::PI;

/// Slater exchange energy density ε_x(ρ) and potential v_x(ρ).
pub struct SlaterExchange;

impl SlaterExchange {
    /// Exchange energy density per electron: ε_x = -3/4 × (3ρ/π)^(1/3)
    pub fn energy_density(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }
        -0.75 * (3.0 * rho / PI).powf(1.0 / 3.0)
    }

    /// Exchange potential: v_x = dε_x/dρ × ρ + ε_x = (4/3) ε_x
    pub fn potential(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }
        (4.0 / 3.0) * Self::energy_density(rho)
    }

    /// Total exchange energy for given density: E_x = ∫ ε_x(ρ) × ρ dr
    /// (per grid point contribution: ε_x × ρ × weight)
    pub fn energy_per_point(rho: f64) -> f64 {
        Self::energy_density(rho) * rho
    }
}

/// Unpolarized VWN5 (Vosko-Wilk-Nusair) correlation functional.
///
/// The energy-per-particle formula follows Libxc's unpolarized VWN5
/// parameterization (functional ID LDA_C_VWN=7), adapted from the published
/// expression of Vosko, Wilk & Nusair, Can. J. Phys. 58, 1200 (1980).
/// Reference implementation and constants:
/// https://github.com/molpro/libxc/blob/470df8e851c412ce8062b040204f11d038b5ab91/maple/vwn.mpl
///
/// Densities are in atomic units and energies are in Hartree. This is the
/// unpolarized/paramagnetic branch; spin-polarized VWN is not implemented.
pub struct VwnCorrelation;

impl VwnCorrelation {
    // Libxc's VWN5 parameters are divided by two from the original
    // Rydberg convention to produce Hartree energy-per-particle values.
    const A: f64 = 0.0310907;
    const X0: f64 = -0.10498;
    const B: f64 = 3.72744;
    const C: f64 = 12.9352;

    /// Correlation energy per electron, ε_c(r_s), with
    /// r_s = (3 / (4πρ))^(1/3).
    pub fn energy_density(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }

        let rs = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);
        Self::vwn_energy_and_rs_derivative(rs.sqrt()).0
    }

    /// Unpolarized VWN5 energy and dε_c/dr_s, evaluated from the same
    /// analytic expression so the potential can use the exact derivative.
    fn vwn_energy_and_rs_derivative(x: f64) -> (f64, f64) {
        let rs = x * x;
        let a = Self::A;
        let b = Self::B;
        let c = Self::C;
        let x0 = Self::X0;
        let q = (4.0 * c - b * b).sqrt();
        let fx = rs + b * x + c;
        let f1 = 2.0 * b / q;
        let f2 = b * x0 / (x0 * x0 + b * x0 + c);
        let f3 = 2.0 * (2.0 * x0 + b) / q;
        let angular = (q / (2.0 * x + b)).atan();

        // Equivalent to ln(r_s / f_x) and
        // ln((sqrt(r_s)-x_0)^2 / f_x), written as ln(1 + δ) with
        // simplified δ expressions to reduce cancellation at large r_s.
        let log_rs_over_fx = ((-b * x - c) / fx).ln_1p();
        let log_shifted_over_fx =
            ((-(2.0 * x0 + b) * x + x0 * x0 - c) / fx).ln_1p();
        let angular_coefficient = f1 - f2 * f3;
        let epsilon = a
            * (log_rs_over_fx + angular_coefficient * angular
                - f2 * log_shifted_over_fx);

        // d/dr_s atan(Q/(2 sqrt(r_s)+b)) = -Q/(4 sqrt(r_s) f_x).
        // Differentiating the same energy expression avoids a noisy forward
        // finite difference in the Kohn-Sham potential.
        let dfx_drs = 1.0 + b / (2.0 * x);
        let d_epsilon_drs = a
            * (1.0 / rs
                - (1.0 - f2) * dfx_drs / fx
                - angular_coefficient * q / (4.0 * x * fx)
                - f2 / (x * (x - x0)));

        (epsilon, d_epsilon_drs)
    }

    /// Correlation potential v_c = ε_c + ρ·dε_c/dρ
    /// = ε_c - (r_s/3)·dε_c/dr_s.
    pub fn potential(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }

        let rs = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);
        let (epsilon, d_epsilon_drs) = Self::vwn_energy_and_rs_derivative(rs.sqrt());
        epsilon - (rs / 3.0) * d_epsilon_drs
    }

    /// Total correlation energy contribution per grid point: ε_c × ρ.
    pub fn energy_per_point(rho: f64) -> f64 {
        Self::energy_density(rho) * rho
    }
}

/// Compute total LDA exchange-correlation energy and potential matrix contribution.
///
/// Returns (E_xc, V_xc_diagonal) where V_xc is the XC potential contribution
/// evaluated at each grid point.
pub fn lda_exchange_correlation(rho_at_points: &[f64]) -> (f64, Vec<f64>) {
    let mut e_xc = 0.0;
    let mut v_xc = Vec::with_capacity(rho_at_points.len());

    for &rho in rho_at_points {
        // Exchange
        let ex = SlaterExchange::energy_per_point(rho);
        let vx = SlaterExchange::potential(rho);

        // Correlation
        let ec = VwnCorrelation::energy_per_point(rho);
        let vc = VwnCorrelation::potential(rho);

        e_xc += ex + ec;
        v_xc.push(vx + vc);
    }

    (e_xc, v_xc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slater_exchange_sign() {
        // Exchange energy should be negative
        let rho = 0.1; // typical density
        let ex = SlaterExchange::energy_density(rho);
        assert!(ex < 0.0, "Exchange should be negative: {}", ex);
    }

    #[test]
    fn test_slater_exchange_scaling() {
        // ε_x scales as ρ^(1/3)
        let rho1 = 0.1;
        let rho2 = 0.8; // 8× density
        let ex1 = SlaterExchange::energy_density(rho1);
        let ex2 = SlaterExchange::energy_density(rho2);
        let ratio = ex2 / ex1;
        let expected = (rho2 / rho1).powf(1.0 / 3.0);
        assert!(
            (ratio - expected).abs() < 0.01,
            "Scaling: ratio={:.4}, expected={:.4}",
            ratio,
            expected
        );
    }

    #[test]
    fn test_vwn_correlation_sign() {
        // Correlation energy should be negative (stabilizing)
        let rho = 0.1;
        let ec = VwnCorrelation::energy_density(rho);
        assert!(ec < 0.0, "Correlation should be negative: {}", ec);
    }

    #[test]
    fn test_vwn_correlation_magnitude() {
        // |ε_c| << |ε_x| (correlation is ~10% of exchange)
        let rho = 0.1;
        let ex = SlaterExchange::energy_density(rho).abs();
        let ec = VwnCorrelation::energy_density(rho).abs();
        assert!(ec < ex, "|ε_c| = {:.6} should be < |ε_x| = {:.6}", ec, ex);
    }

    #[test]
    fn test_lda_zero_density() {
        let (e_xc, v_xc) = lda_exchange_correlation(&[0.0, 0.0, 0.0]);
        assert_eq!(e_xc, 0.0);
        for v in v_xc {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_vwn5_matches_independent_reference_values() {
        // Frozen values evaluated independently from the published unpolarized
        // VWN5 expression (Libxc LDA_C_VWN=7; energies in Hartree per electron).
        // These are formula fixtures, not yet a Libxc runtime/API comparison.
        let cases = [
            (0.01_f64, -0.037_645_190_262_171_425, -0.043_872_656_447_393_65),
            (0.1_f64, -0.053_397_289_185_949_826, -0.060_812_030_331_261_556),
            (1.0_f64, -0.071_592_612_306_790_65, -0.079_938_383_175_985_62),
        ];

        for (rho, expected_epsilon, expected_potential) in cases {
            let epsilon = VwnCorrelation::energy_density(rho);
            let potential = VwnCorrelation::potential(rho);
            assert!(
                (epsilon - expected_epsilon).abs() < 2e-12,
                "rho={rho}: epsilon={epsilon:.15e}, expected={expected_epsilon:.15e}"
            );
            assert!(
                (potential - expected_potential).abs() < 2e-12,
                "rho={rho}: potential={potential:.15e}, expected={expected_potential:.15e}"
            );
        }
    }

    #[test]
    fn test_vwn5_analytic_potential_matches_density_derivative() {
        for rho in [1e-6_f64, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0] {
            let h = rho * 1e-5;
            let epsilon = VwnCorrelation::energy_density(rho);
            let epsilon_plus = VwnCorrelation::energy_density(rho + h);
            let epsilon_minus = VwnCorrelation::energy_density(rho - h);
            let finite_difference = epsilon
                + rho * (epsilon_plus - epsilon_minus) / (2.0 * h);
            let analytic = VwnCorrelation::potential(rho);
            assert!(
                (analytic - finite_difference).abs() < 2e-9,
                "rho={rho}: analytic={analytic:.15e}, finite difference={finite_difference:.15e}"
            );
        }
    }

}
