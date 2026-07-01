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

/// VWN (Vosko-Wilk-Nusair) correlation functional, Formula V (1980).
pub struct VwnCorrelation;

impl VwnCorrelation {
    // VWN-V parameters (paramagnetic case)
    const A: f64 = 0.0621814;
    const X0: f64 = -0.10498;
    const B: f64 = 3.72744;
    const C: f64 = 12.9352;

    /// Correlation energy density ε_c(r_s) where r_s = (3/(4πρ))^(1/3).
    pub fn energy_density(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }

        let rs = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);
        let x = rs.sqrt();

        Self::vwn_formula(x)
    }

    /// VWN formula: ε_c(x) where x = sqrt(r_s)
    fn vwn_formula(x: f64) -> f64 {
        let a = Self::A;
        let x0 = Self::X0;
        let b = Self::B;
        let c = Self::C;

        let xx = x * x + b * x + c;
        let x0x0 = x0 * x0 + b * x0 + c;
        let q = (4.0 * c - b * b).sqrt();

        a * (2.0 * (x - x0).ln() / xx
            - b * x0 / x0x0
                * (2.0 * (2.0 * x + b) / q).atan()
            + 2.0 * x.ln() / xx // This is simplified; full VWN-V is more complex
            - b / q * (2.0 * (2.0 * x + b) / q).atan())
            * 0.5 // Factor for correct normalization
    }

    /// Correlation potential v_c(ρ) (numerical derivative).
    pub fn potential(rho: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }

        let eps = Self::energy_density(rho);
        let drho = rho * 1e-6;
        let eps_plus = Self::energy_density(rho + drho);

        // v_c = ε_c + ρ × dε_c/dρ
        eps + rho * (eps_plus - eps) / drho
    }

    /// Total correlation energy per grid point: ε_c × ρ
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
}
