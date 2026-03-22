// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Order of phase transition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionOrder {
    /// First order: Discontinuous jump (like ice to water)
    FirstOrder,
    /// Second order: Continuous but singular (like ferromagnetism)
    SecondOrder,
    /// Crossover: Smooth transition (no true phase boundary)
    Crossover,
}

/// Critical exponents for second-order transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalExponents {
    /// Heat capacity exponent (alpha)
    pub alpha: f64,
    /// Order parameter exponent (beta)
    pub beta: f64,
    /// Susceptibility exponent (gamma)
    pub gamma: f64,
    /// Correlation length exponent (nu)
    pub nu: f64,
    /// Correlation function exponent (eta)
    pub eta: f64,
}

impl Default for CriticalExponents {
    fn default() -> Self {
        // Mean-field (Landau) values
        Self {
            alpha: 0.0,
            beta: 0.5,
            gamma: 1.0,
            nu: 0.5,
            eta: 0.0,
        }
    }
}

/// Fluctuation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluctuationStats {
    /// Mean fluctuation amplitude
    pub mean_amplitude: f64,

    /// Variance of fluctuations
    pub variance: f64,

    /// Autocorrelation time
    pub autocorrelation_time: f64,

    /// Critical slowing down indicator
    pub slowing_down: f64,

    /// Susceptibility (response to perturbation)
    pub susceptibility: f64,

    /// Fluctuation-dissipation ratio
    pub fdr: f64,
}

impl Default for FluctuationStats {
    fn default() -> Self {
        Self {
            mean_amplitude: 0.1,
            variance: 0.01,
            autocorrelation_time: 1.0,
            slowing_down: 0.0,
            susceptibility: 1.0,
            fdr: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_order_variants() {
        let first = TransitionOrder::FirstOrder;
        let second = TransitionOrder::SecondOrder;
        let crossover = TransitionOrder::Crossover;

        // All three are distinct
        assert_ne!(first, second);
        assert_ne!(first, crossover);
        assert_ne!(second, crossover);
    }

    #[test]
    fn test_critical_exponents_default() {
        // Mean-field (Landau) values
        let exp = CriticalExponents::default();
        assert!((exp.alpha - 0.0).abs() < f64::EPSILON);
        assert!((exp.beta - 0.5).abs() < f64::EPSILON);
        assert!((exp.gamma - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_critical_exponents_hyperscaling() {
        // Hyperscaling relation: 2 - alpha = d * nu
        // At the mean-field upper critical dimension d=4:
        // 2 - 0.0 = 4 * 0.5 = 2.0
        let exp = CriticalExponents::default();
        let d = 4.0_f64; // upper critical dimension for mean-field
        let lhs = 2.0 - exp.alpha;
        let rhs = d * exp.nu;
        assert!(
            (lhs - rhs).abs() < f64::EPSILON,
            "Hyperscaling 2-alpha ({}) should equal d*nu ({})",
            lhs,
            rhs,
        );
    }

    #[test]
    fn test_fluctuation_stats_default() {
        let stats = FluctuationStats::default();
        assert!((stats.mean_amplitude - 0.1).abs() < f64::EPSILON);
        assert!((stats.variance - 0.01).abs() < f64::EPSILON);
        assert!((stats.fdr - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fluctuation_stats_fdr_equilibrium() {
        // In thermal equilibrium, the fluctuation-dissipation ratio = 1.0
        let stats = FluctuationStats::default();
        assert!(
            (stats.fdr - 1.0).abs() < f64::EPSILON,
            "FDR should be 1.0 at equilibrium, got {}",
            stats.fdr,
        );
    }
}
