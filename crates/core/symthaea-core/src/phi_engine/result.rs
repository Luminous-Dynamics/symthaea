// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # PhiResult and PhiUncertainty
//!
//! Standard result structures for Φ (integrated information) measurements.

use super::calculator::PartitionInfo;
use std::time::Duration;

/// Classifies what a Φ-like measurement actually computes.
///
/// Symthaea uses three layers of consciousness measurement, each with
/// a distinct mathematical basis and computational cost:
///
/// | Layer | Symbol | Category | Cost |
/// |-------|--------|----------|------|
/// | 1 | Ψ | `ConsciousnessEstimate` | O(1) |
/// | 2 | λ₂ | `SpectralConnectivity` | O(n³) |
/// | 3 | Σ | `SynergisticIntegration` | O(n²) |
/// | 4 | Φ | `IntegratedInformation` | O(2ⁿ) |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhiCategory {
    /// Ψ — Fast consciousness estimate from composite soft signals.
    /// Computed every cycle (50Hz). NOT IIT Φ.
    ConsciousnessEstimate,

    /// λ₂ — Spectral connectivity (Fiedler value of the cosine-similarity
    /// graph Laplacian). Measures network integration structure.
    SpectralConnectivity,

    /// Σ — Synergistic integration via information decomposition.
    /// Measures how much information exists in the whole that isn't
    /// in any partition (PhiR-inspired, Mediano et al. 2021).
    SynergisticIntegration,

    /// Φ — True IIT Integrated Information via MIP search.
    /// Exact or heuristic approximation of Tononi's Φ.
    IntegratedInformation,
}

impl std::fmt::Display for PhiCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConsciousnessEstimate => write!(f, "Ψ (consciousness estimate)"),
            Self::SpectralConnectivity => write!(f, "λ₂ (spectral connectivity)"),
            Self::SynergisticIntegration => write!(f, "Σ (synergistic integration)"),
            Self::IntegratedInformation => write!(f, "Φ (integrated information)"),
        }
    }
}

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

    /// What this measurement actually computes (see [`PhiCategory`]).
    pub category: PhiCategory,

    /// Time taken for computation
    pub computation_time: Duration,

    /// Number of nodes in the topology
    pub n_nodes: usize,

    /// Information about the limiting partition (if computed)
    pub limiting_partition: Option<PartitionInfo>,
}

impl PhiResult {
    /// Create a new PhiResult with default category [`PhiCategory::SpectralConnectivity`].
    pub fn new(phi: f64, method: &'static str, n_nodes: usize) -> Self {
        Self {
            phi,
            method,
            category: PhiCategory::SpectralConnectivity,
            computation_time: Duration::ZERO,
            n_nodes,
            limiting_partition: None,
        }
    }

    /// Create with timing and default category.
    pub fn with_timing(phi: f64, method: &'static str, n_nodes: usize, time: Duration) -> Self {
        Self {
            phi,
            method,
            category: PhiCategory::SpectralConnectivity,
            computation_time: time,
            n_nodes,
            limiting_partition: None,
        }
    }

    /// Set the measurement category.
    pub fn with_category(mut self, category: PhiCategory) -> Self {
        self.category = category;
        self
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
            "{} = {:.4} ({}, n={}, {:.2}ms)",
            self.category,
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
                confidence_interval_95: (
                    samples.first().copied().unwrap_or(0.0),
                    samples.first().copied().unwrap_or(0.0),
                ),
                n_samples: n,
            };
        }

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
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
        self.confidence_interval_95.1 < other.confidence_interval_95.0
            || other.confidence_interval_95.1 < self.confidence_interval_95.0
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
    use proptest::prelude::*;

    #[test]
    fn test_phi_result_display() {
        let result = PhiResult::with_timing(0.4976, "Continuous", 16, Duration::from_millis(150));

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

    // =====================================================================
    // Property-based tests
    // =====================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_phi_classification_consistent(phi in 0.0f64..=1.0) {
            let result = PhiResult::new(phi, "test", 8);
            // highly_integrated implies indicates_consciousness
            if result.is_highly_integrated() {
                prop_assert!(result.indicates_consciousness(),
                    "Highly integrated (phi={}) should also indicate consciousness", phi);
            }
        }

        #[test]
        fn prop_phi_percent_bounded(phi in 0.0f64..=1.0) {
            let result = PhiResult::new(phi, "test", 8);
            let pct = result.percent_of_maximum();
            prop_assert!(pct >= 0.0, "Percent should be non-negative, got {} for phi={}", pct, phi);
            prop_assert!(pct <= 200.0, "Percent should be ≤200%, got {} for phi={}", pct, phi);
        }

        #[test]
        fn prop_uncertainty_stddev_nonneg(
            s1 in -1.0f64..1.0,
            s2 in -1.0f64..1.0,
            s3 in -1.0f64..1.0,
            s4 in -1.0f64..1.0,
        ) {
            let samples = vec![s1, s2, s3, s4];
            let u = PhiUncertainty::from_samples(&samples);
            prop_assert!(u.std_dev >= 0.0,
                "Standard deviation must be non-negative, got {}", u.std_dev);
            prop_assert!(!u.std_dev.is_nan(),
                "Standard deviation must not be NaN");
        }

        #[test]
        fn prop_uncertainty_ci_ordered(
            s1 in 0.0f64..1.0,
            s2 in 0.0f64..1.0,
            s3 in 0.0f64..1.0,
        ) {
            let samples = vec![s1, s2, s3];
            let u = PhiUncertainty::from_samples(&samples);
            prop_assert!(u.confidence_interval_95.0 <= u.confidence_interval_95.1,
                "CI lower {} should be ≤ upper {}", u.confidence_interval_95.0, u.confidence_interval_95.1);
        }

        #[test]
        fn prop_uncertainty_symmetric(
            s1 in 0.0f64..1.0,
            s2 in 0.0f64..1.0,
            s3 in 0.0f64..1.0,
        ) {
            let a_samples = vec![s1, s2, s3];
            let b_samples = vec![s1 + 2.0, s2 + 2.0, s3 + 2.0];
            let a = PhiUncertainty::from_samples(&a_samples);
            let b = PhiUncertainty::from_samples(&b_samples);
            // is_significantly_different should be symmetric
            prop_assert_eq!(
                a.is_significantly_different(&b),
                b.is_significantly_different(&a),
                "Significance test should be symmetric"
            );
        }

        #[test]
        fn prop_single_sample_zero_stddev(phi in 0.0f64..1.0) {
            let u = PhiUncertainty::from_samples(&[phi]);
            prop_assert_eq!(u.std_dev, 0.0, "Single sample should have zero std_dev");
            prop_assert_eq!(u.n_samples, 1);
        }

        #[test]
        fn prop_identical_samples_zero_stddev(phi in 0.0f64..1.0) {
            let u = PhiUncertainty::from_samples(&[phi, phi, phi, phi]);
            prop_assert!(u.std_dev < 1e-10,
                "Identical samples should have near-zero std_dev, got {}", u.std_dev);
        }

        #[test]
        fn prop_cv_nonneg(
            s1 in 0.1f64..1.0,
            s2 in 0.1f64..1.0,
            s3 in 0.1f64..1.0,
        ) {
            let samples = vec![s1, s2, s3];
            let mean = (s1 + s2 + s3) / 3.0;
            let u = PhiUncertainty::from_samples(&samples);
            let cv = u.coefficient_of_variation(mean);
            prop_assert!(cv >= 0.0, "CV should be non-negative, got {}", cv);
        }
    }
}
