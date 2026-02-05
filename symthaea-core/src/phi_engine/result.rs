//! # PhiResult and PhiUncertainty
//!
//! Standard result structures for Φ (integrated information) measurements.

use std::time::Duration;
use super::calculator::PartitionInfo;

/// Result of a Φ calculation
///
/// Contains the Φ value along with metadata about the computation.
#[derive(Debug, Clone)]
pub struct PhiResult {
    /// The computed Φ value in range [0, 1]
    ///
    /// - 0 = No integration (disconnected/reducible)
    /// - 1 = Perfect integration (maximally conscious)
    pub phi: f64,

    /// Name of the calculation method used
    pub method: &'static str,

    /// Time taken for computation
    pub computation_time: Duration,

    /// Number of nodes in the topology
    pub n_nodes: usize,

    /// Information about the limiting partition (if computed)
    pub limiting_partition: Option<PartitionInfo>,
}

impl PhiResult {
    /// Create a new PhiResult
    pub fn new(phi: f64, method: &'static str, n_nodes: usize) -> Self {
        Self {
            phi,
            method,
            computation_time: Duration::ZERO,
            n_nodes,
            limiting_partition: None,
        }
    }

    /// Create with timing
    pub fn with_timing(phi: f64, method: &'static str, n_nodes: usize, time: Duration) -> Self {
        Self {
            phi,
            method,
            computation_time: time,
            n_nodes,
            limiting_partition: None,
        }
    }

    /// Check if this Φ indicates high integration
    ///
    /// Based on empirical findings: Φ > 0.49 indicates high integration
    /// (typical of uniform k-regular topologies like Ring, Torus, Hypercube)
    pub fn is_highly_integrated(&self) -> bool {
        self.phi > 0.49
    }

    /// Check if this Φ indicates consciousness emergence
    ///
    /// Very conservative threshold based on research findings
    pub fn indicates_consciousness(&self) -> bool {
        self.phi > 0.45
    }

    /// Get percentage of theoretical maximum
    ///
    /// Based on asymptotic limit Φ → 0.5 for optimal topologies
    pub fn percent_of_maximum(&self) -> f64 {
        (self.phi / 0.5) * 100.0
    }
}

impl std::fmt::Display for PhiResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Φ = {:.4} ({}, n={}, {:.2}ms)",
            self.phi,
            self.method,
            self.n_nodes,
            self.computation_time.as_secs_f64() * 1000.0
        )
    }
}

/// Statistical uncertainty for Φ measurements
///
/// Provides confidence intervals and variance estimates for Φ values
/// computed via resampling or multiple measurements.
#[derive(Debug, Clone)]
pub struct PhiUncertainty {
    /// Standard deviation of Φ measurements
    pub std_dev: f64,

    /// 95% confidence interval (lower, upper)
    pub confidence_interval_95: (f64, f64),

    /// Number of samples used for uncertainty estimation
    pub n_samples: usize,
}

impl PhiUncertainty {
    /// Create uncertainty from a set of Φ samples
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n < 2 {
            return Self {
                std_dev: 0.0,
                confidence_interval_95: (samples.first().copied().unwrap_or(0.0),
                                         samples.first().copied().unwrap_or(0.0)),
                n_samples: n,
            };
        }

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();

        // 95% CI with t-distribution approximation
        let z = 1.96;
        let margin = z * std_dev / (n as f64).sqrt();

        Self {
            std_dev,
            confidence_interval_95: (mean - margin, mean + margin),
            n_samples: n,
        }
    }

    /// Check if two Φ results are significantly different
    ///
    /// Uses non-overlapping confidence intervals as criterion
    pub fn is_significantly_different(&self, other: &PhiUncertainty) -> bool {
        self.confidence_interval_95.1 < other.confidence_interval_95.0 ||
        other.confidence_interval_95.1 < self.confidence_interval_95.0
    }

    /// Get coefficient of variation (relative uncertainty)
    pub fn coefficient_of_variation(&self, mean: f64) -> f64 {
        if mean.abs() < 1e-10 {
            return 0.0;
        }
        self.std_dev / mean.abs() * 100.0
    }
}

impl std::fmt::Display for PhiUncertainty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "σ = {:.4}, 95% CI: [{:.4}, {:.4}], n = {}",
            self.std_dev,
            self.confidence_interval_95.0,
            self.confidence_interval_95.1,
            self.n_samples
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_result_display() {
        let result = PhiResult::with_timing(
            0.4976,
            "Continuous",
            16,
            Duration::from_millis(150),
        );

        let display = format!("{}", result);
        assert!(display.contains("0.4976"));
        assert!(display.contains("Continuous"));
        assert!(display.contains("n=16"));
    }

    #[test]
    fn test_phi_result_classification() {
        let high = PhiResult::new(0.495, "test", 8);
        assert!(high.is_highly_integrated());
        assert!(high.indicates_consciousness());

        let low = PhiResult::new(0.35, "test", 8);
        assert!(!low.is_highly_integrated());
        assert!(!low.indicates_consciousness());
    }

    #[test]
    fn test_percent_of_maximum() {
        let result = PhiResult::new(0.4976, "test", 16);
        let percent = result.percent_of_maximum();
        assert!((percent - 99.52).abs() < 0.1);
    }

    #[test]
    fn test_uncertainty_from_samples() {
        let samples = vec![0.495, 0.497, 0.496, 0.498, 0.494];
        let uncertainty = PhiUncertainty::from_samples(&samples);

        assert!(uncertainty.std_dev > 0.0);
        assert!(uncertainty.std_dev < 0.01);
        assert_eq!(uncertainty.n_samples, 5);
    }

    #[test]
    fn test_significant_difference() {
        let high = PhiUncertainty {
            std_dev: 0.001,
            confidence_interval_95: (0.493, 0.497),
            n_samples: 10,
        };

        let low = PhiUncertainty {
            std_dev: 0.001,
            confidence_interval_95: (0.430, 0.440),
            n_samples: 10,
        };

        assert!(high.is_significantly_different(&low));
    }
}
