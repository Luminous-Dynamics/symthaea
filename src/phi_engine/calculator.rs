//! # PhiCalculator Trait
//!
//! Unified interface for all Φ (integrated information) calculation methods.
//!
//! ## Purpose
//! Provides a common trait that all Φ calculators must implement, enabling
//! interchangeable use of different calculation methods with consistent interfaces.
//!
//! ## Implementations
//! - `ContinuousPhiCalculator` - Uses cosine similarity on RealHV
//! - `TieredPhi` - Binary with multiple approximation tiers
//! - `ResonatorPhi` - O(n log N) resonator-based approximation

use crate::hdc::unified_hv::ContinuousHV;
use super::result::{PhiResult, PhiUncertainty};
use std::time::{Duration, Instant};

/// Computational complexity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    /// O(1) - Constant time (mock/cached)
    Constant,

    /// O(n) - Linear in number of nodes
    Linear,

    /// O(n log n) - Log-linear (resonator)
    LogLinear,

    /// O(n²) - Quadratic (similarity matrix)
    Quadratic,

    /// O(n³) - Cubic (eigenvalue decomposition)
    Cubic,

    /// O(2^n) - Exponential (exact MIP search)
    Exponential,
}

impl Complexity {
    /// Get human-readable complexity string
    pub fn as_str(&self) -> &'static str {
        match self {
            Complexity::Constant => "O(1)",
            Complexity::Linear => "O(n)",
            Complexity::LogLinear => "O(n log n)",
            Complexity::Quadratic => "O(n²)",
            Complexity::Cubic => "O(n³)",
            Complexity::Exponential => "O(2^n)",
        }
    }

    /// Check if suitable for a given size
    pub fn is_tractable_for(&self, n: usize) -> bool {
        match self {
            Complexity::Constant => true,
            Complexity::Linear => true,
            Complexity::LogLinear => true,
            Complexity::Quadratic => n <= 10_000,
            Complexity::Cubic => n <= 1_000,
            Complexity::Exponential => n <= 15,
        }
    }
}

/// Unified trait for Φ (integrated information) calculators
///
/// All Φ calculation methods implement this trait to provide a consistent
/// interface for consciousness measurement.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::phi_engine::{PhiCalculator, PhiResult};
///
/// fn measure_consciousness(calc: &dyn PhiCalculator, hvs: &[ContinuousHV]) -> f64 {
///     calc.compute_from_hvs(hvs).phi
/// }
/// ```
pub trait PhiCalculator: Send + Sync {
    /// Compute Φ for a set of hypervector node representations
    ///
    /// # Arguments
    /// * `node_representations` - Vector of hypervectors representing network nodes
    ///
    /// # Returns
    /// `PhiResult` containing the Φ value and metadata
    fn compute_from_hvs(&self, node_representations: &[ContinuousHV]) -> PhiResult;

    /// Compute Φ with uncertainty estimate via resampling
    ///
    /// # Arguments
    /// * `node_representations` - Vector of hypervectors representing network nodes
    /// * `n_samples` - Number of bootstrap samples for uncertainty estimation
    ///
    /// # Returns
    /// Tuple of (PhiResult, PhiUncertainty) with statistical confidence
    fn compute_with_uncertainty(
        &self,
        node_representations: &[ContinuousHV],
        n_samples: usize,
    ) -> (PhiResult, PhiUncertainty) {
        // Default implementation using simple resampling
        let mut phi_values = Vec::with_capacity(n_samples);
        let start = Instant::now();

        for _ in 0..n_samples {
            let result = self.compute_from_hvs(node_representations);
            phi_values.push(result.phi);
        }

        let mean: f64 = phi_values.iter().sum::<f64>() / n_samples as f64;
        let variance: f64 = phi_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n_samples - 1) as f64;
        let std_dev = variance.sqrt();

        // 95% confidence interval (assuming normal distribution)
        let z = 1.96;
        let margin = z * std_dev / (n_samples as f64).sqrt();

        let result = PhiResult {
            phi: mean,
            method: self.method_name(),
            computation_time: start.elapsed(),
            n_nodes: node_representations.len(),
            limiting_partition: None,
        };

        let uncertainty = PhiUncertainty {
            std_dev,
            confidence_interval_95: (mean - margin, mean + margin),
            n_samples,
        };

        (result, uncertainty)
    }

    /// Get the method name for reporting
    fn method_name(&self) -> &'static str;

    /// Get the computational complexity class
    fn complexity(&self) -> Complexity;

    /// Check if this method is suitable for a given topology size
    fn is_suitable_for(&self, n_nodes: usize) -> bool {
        self.complexity().is_tractable_for(n_nodes)
    }
}

/// Information about the limiting partition (MIP)
#[derive(Debug, Clone)]
pub struct PartitionInfo {
    /// The cut that minimizes information loss
    pub cut: Vec<Vec<usize>>,

    /// Information lost by this partition
    pub information_loss: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_string() {
        assert_eq!(Complexity::Constant.as_str(), "O(1)");
        assert_eq!(Complexity::Exponential.as_str(), "O(2^n)");
    }

    #[test]
    fn test_complexity_tractability() {
        assert!(Complexity::Linear.is_tractable_for(1_000_000));
        assert!(!Complexity::Exponential.is_tractable_for(100));
        assert!(Complexity::Exponential.is_tractable_for(10));
    }
}
