// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local Density Approximation (LDA) exchange-correlation functional.
//!
//! - **Slater exchange**: ε_x(ρ) = -3/4 × (3ρ/π)^(1/3)
//! - **VWN5 correlation**: Vosko-Wilk-Nusair's unpolarized homogeneous-electron-gas fit.
//!
//! References:
//! - Slater, J. C. (1951). Phys. Rev. 81, 385.
//! - Vosko, Wilk & Nusair (1980). Can. J. Phys. 58, 1200.
//! - The VWN5 parameterization is Libxc LDA_C_VWN (id 7), used by PySCF's
//!   `lda,vwn` alias. Its canonical expression is documented in the source
//!   references noted above; the full logarithmic-rational form matters.

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

    /// Exchange potential: v_x = d(ρε_x)/dρ = (4/3) ε_x
    pub fn potential(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }
        (4.0 / 3.0) * Self::energy_density(rho)
    }

    /// Total exchange energy integrand before multiplication by the grid weight: ρ ε_x.
    pub fn energy_per_point(rho: f64) -> f64 {
        Self::energy_density(rho) * rho
    }
}

/// VWN5 (Vosko-Wilk-Nusair, 1980) unpolarized correlation functional.
pub struct VwnCorrelation;

impl VwnCorrelation {
    // Paramagnetic VWN5 fit. These are the standard constants used by
    // Libxc's LDA_C_VWN (functional ID 7).
    const A: f64 = 0.031_090_7;
    const X0: f64 = -0.104_98;
    const B: f64 = 3.727_44;
    const C: f64 = 12.935_2;

    /// Correlation energy per electron ε_c(ρ), where r_s=(3/(4πρ))^(1/3).
    pub fn energy_density(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }

        let rs = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);
        let x = rs.sqrt();
        Self::vwn5_formula_and_derivative(x).0
    }

    /// VWN5 correlation potential v_c = d[ρ ε_c(ρ)]/dρ.
    ///
    /// Since x = sqrt(r_s) ∝ ρ^(-1/6), this is ε_c - (x/6) dε_c/dx.
    /// Differentiate the canonical VWN5 expression analytically instead of
    /// using the previous forward finite difference, which made the XC
    /// potential inconsistent with the energy functional.
    pub fn potential(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }

        let rs = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);
        let x = rs.sqrt();
        let (epsilon, d_epsilon_dx) = Self::vwn5_formula_and_derivative(x);
        epsilon - (x / 6.0) * d_epsilon_dx
    }

    /// Canonical VWN5 equation and its analytic derivative with respect to
    /// x=sqrt(r_s). Returns (ε_c, dε_c/dx).
    fn vwn5_formula_and_derivative(x: f64) -> (f64, f64) {
        let a = Self::A;
        let x0 = Self::X0;
        let b = Self::B;
        let c = Self::C;

        let xx = x * x + b * x + c;
        let x0x0 = x0 * x0 + b * x0 + c;
        let q = (4.0 * c - b * b).sqrt();
        let atan_term = (q / (2.0 * x + b)).atan();
        let reference_factor = b * x0 / x0x0;

        // ln(x²/X(x)) and ln((x-x0)²/X(x)), written in a form that avoids
        // squaring the positive logarithm arguments.
        let log_term_1 = 2.0 * x.ln() - xx.ln();
        let log_term_2 = 2.0 * (x - x0).ln() - xx.ln();
        let atan_term_1 = (2.0 * b / q) * atan_term;
        let atan_term_2 = (2.0 * (b + 2.0 * x0) / q) * atan_term;

        let bracket = log_term_1 + atan_term_1 - reference_factor * (log_term_2 + atan_term_2);

        // d/dx atan(Q/(2x+b)) = -Q/(2 X(x)).
        let d_log_term_1 = 2.0 / x - (2.0 * x + b) / xx;
        let d_atan_term_1 = -b / xx;
        let d_log_term_2 = 2.0 / (x - x0) - (2.0 * x + b) / xx;
        let d_atan_term_2 = -(b + 2.0 * x0) / xx;
        let d_bracket =
            d_log_term_1 + d_atan_term_1 - reference_factor * (d_log_term_2 + d_atan_term_2);

        (a * bracket, a * d_bracket)
    }

    /// Total correlation-energy integrand before multiplication by grid weight: ρ ε_c.
    pub fn energy_per_point(rho: f64) -> f64 {
        Self::energy_density(rho) * rho
    }
}

/// Compute total LDA exchange-correlation energy and potential contributions.
///
/// Returns (sum of per-point XC energies, XC potential at each point). The
/// energy sum is unweighted; callers integrating a grid must multiply each
/// point's contribution by its own quadrature weight.
pub fn lda_exchange_correlation(rho_at_points: &[f64]) -> (f64, Vec<f64>) {
    let mut e_xc = 0.0;
    let mut v_xc = Vec::with_capacity(rho_at_points.len());

    for &rho in rho_at_points {
        let ex = SlaterExchange::energy_per_point(rho);
        let vx = SlaterExchange::potential(rho);
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
        assert!(SlaterExchange::energy_density(0.1) < 0.0);
    }

    #[test]
    fn test_slater_exchange_scaling() {
        let rho1 = 0.1;
        let rho2 = 0.8;
        let ratio = SlaterExchange::energy_density(rho2) / SlaterExchange::energy_density(rho1);
        let expected = (rho2 / rho1).powf(1.0 / 3.0);
        assert!((ratio - expected).abs() < 0.01);
    }

    #[test]
    fn test_vwn5_energy_and_potential_reference_values() {
        // Reference values from evaluating the canonical VWN5 equation with
        // its published unpolarized parameter set. These guard the full
        // logarithmic/rational expression, not just sign and magnitude.
        let references = [
            (1e-6, -0.004_776_617_504_447_303, -0.006_099_907_725_695_197),
            (1e-3, -0.024_864_794_928_981_93, -0.029_718_194_274_025_9),
            (1e-1, -0.053_397_289_185_949_83, -0.060_812_030_331_261_56),
            (1.0, -0.071_592_612_306_790_66, -0.079_938_383_175_985_62),
        ];

        for (rho, expected_epsilon, expected_potential) in references {
            let epsilon = VwnCorrelation::energy_density(rho);
            let potential = VwnCorrelation::potential(rho);
            assert!(
                (epsilon - expected_epsilon).abs() < 2e-12,
                "VWN5 ε_c({rho})={epsilon:.16e}, expected {expected_epsilon:.16e}"
            );
            assert!(
                (potential - expected_potential).abs() < 2e-12,
                "VWN5 v_c({rho})={potential:.16e}, expected {expected_potential:.16e}"
            );
        }
    }

    #[test]
    fn test_vwn5_analytic_potential_matches_energy_derivative() {
        for rho in [1e-6, 1e-3, 1e-1, 1.0] {
            let delta = rho * 1e-5;
            let epsilon = VwnCorrelation::energy_density(rho);
            let derivative = (VwnCorrelation::energy_density(rho + delta)
                - VwnCorrelation::energy_density(rho - delta))
                / (2.0 * delta);
            let finite_difference_potential = epsilon + rho * derivative;
            let analytic_potential = VwnCorrelation::potential(rho);
            assert!(
                (analytic_potential - finite_difference_potential).abs() < 1e-8,
                "VWN5 potential derivative mismatch at rho={rho}: analytic={analytic_potential}, FD={finite_difference_potential}"
            );
        }
    }

    #[test]
    fn test_vwn_correlation_sign() {
        assert!(VwnCorrelation::energy_density(0.1) < 0.0);
    }

    #[test]
    fn test_vwn_correlation_magnitude() {
        let ex = SlaterExchange::energy_density(0.1).abs();
        let ec = VwnCorrelation::energy_density(0.1).abs();
        assert!(ec < ex);
    }

    #[test]
    fn test_lda_zero_density() {
        let (e_xc, v_xc) = lda_exchange_correlation(&[0.0, 0.0, 0.0]);
        assert_eq!(e_xc, 0.0);
        for v in v_xc {
            assert_eq!(v, 0.0);
        }
    }
}
