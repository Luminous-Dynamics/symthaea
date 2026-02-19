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
