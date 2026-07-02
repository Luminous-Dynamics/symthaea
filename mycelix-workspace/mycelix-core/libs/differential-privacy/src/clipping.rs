// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient Clipping for Differential Privacy
//!
//! Implements gradient clipping to bound sensitivity before adding noise.
//! This is essential for DP-SGD and federated learning with DP guarantees.

use crate::{DpError, DpResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

/// Methods for gradient clipping
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ClippingMethod {
    /// Clip by L2 norm (Euclidean)
    L2,
    /// Clip by L1 norm (Manhattan)
    L1,
    /// Clip by L-infinity norm (max absolute value)
    LInf,
    /// Per-coordinate clipping
    PerCoordinate,
}

/// Gradient clipper for bounding sensitivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientClipper {
    /// Maximum norm for clipping
    max_norm: f64,
    /// Clipping method to use
    method: ClippingMethod,
    /// Statistics tracking
    #[serde(skip)]
    stats: ClippingStats,
}

/// Statistics about clipping operations
#[derive(Debug, Clone, Default)]
pub struct ClippingStats {
    /// Number of gradients processed
    pub total_processed: usize,
    /// Number of gradients that were clipped
    pub total_clipped: usize,
    /// Sum of original norms
    pub sum_original_norms: f64,
    /// Sum of clipped norms
    pub sum_clipped_norms: f64,
    /// Maximum original norm seen
    pub max_norm_seen: f64,
}

impl GradientClipper {
    /// Create a new gradient clipper
    ///
    /// # Arguments
    /// * `max_norm` - Maximum allowed norm (clipping threshold C)
    /// * `method` - Clipping method to use
    pub fn new(max_norm: f64, method: ClippingMethod) -> DpResult<Self> {
        if max_norm <= 0.0 {
            return Err(DpError::InvalidClipNorm(max_norm));
        }

        Ok(Self {
            max_norm,
            method,
            stats: ClippingStats::default(),
        })
    }

    /// Create an L2 clipper (most common for DP-SGD)
    pub fn l2(max_norm: f64) -> DpResult<Self> {
        Self::new(max_norm, ClippingMethod::L2)
    }

    /// Create an L1 clipper
    pub fn l1(max_norm: f64) -> DpResult<Self> {
        Self::new(max_norm, ClippingMethod::L1)
    }

    /// Clip a single gradient vector
    ///
    /// # Arguments
    /// * `gradient` - The gradient to clip
    ///
    /// # Returns
    /// The clipped gradient with norm ≤ max_norm
    #[instrument(skip(self, gradient), fields(len = gradient.len()))]
    pub fn clip(&mut self, gradient: &[f64]) -> Vec<f64> {
        let original_norm = self.compute_norm(gradient);

        self.stats.total_processed += 1;
        self.stats.sum_original_norms += original_norm;
        self.stats.max_norm_seen = self.stats.max_norm_seen.max(original_norm);

        if original_norm <= self.max_norm {
            self.stats.sum_clipped_norms += original_norm;
            return gradient.to_vec();
        }

        self.stats.total_clipped += 1;

        let clipped: Vec<f64> = match self.method {
            ClippingMethod::L2 | ClippingMethod::L1 => {
                let scale = self.max_norm / original_norm;
                gradient.iter().map(|&x| x * scale).collect()
            }
            ClippingMethod::LInf => {
                gradient
                    .iter()
                    .map(|&x| x.clamp(-self.max_norm, self.max_norm))
                    .collect()
            }
            ClippingMethod::PerCoordinate => {
                gradient
                    .iter()
                    .map(|&x| x.clamp(-self.max_norm, self.max_norm))
                    .collect()
            }
        };

        let clipped_norm = self.compute_norm(&clipped);
        self.stats.sum_clipped_norms += clipped_norm;

        debug!(
            original_norm = original_norm,
            clipped_norm = clipped_norm,
            "Clipped gradient"
        );

        clipped
    }

    /// Clip a batch of gradients
    pub fn clip_batch(&mut self, gradients: &[Vec<f64>]) -> Vec<Vec<f64>> {
        gradients.iter().map(|g| self.clip(g)).collect()
    }

    /// Clip and aggregate gradients (for federated learning)
    ///
    /// Clips each gradient, then computes the mean.
    pub fn clip_and_aggregate(&mut self, gradients: &[Vec<f64>]) -> Vec<f64> {
        if gradients.is_empty() {
            return Vec::new();
        }

        let n = gradients.len() as f64;
        let dim = gradients[0].len();

        let mut aggregated = vec![0.0; dim];

        for grad in gradients {
            let clipped = self.clip(grad);
            for (agg, &val) in aggregated.iter_mut().zip(clipped.iter()) {
                *agg += val / n;
            }
        }

        aggregated
    }

    /// Compute the norm of a gradient based on the clipping method
    fn compute_norm(&self, gradient: &[f64]) -> f64 {
        match self.method {
            ClippingMethod::L2 => {
                gradient.iter().map(|x| x * x).sum::<f64>().sqrt()
            }
            ClippingMethod::L1 => {
                gradient.iter().map(|x| x.abs()).sum()
            }
            ClippingMethod::LInf | ClippingMethod::PerCoordinate => {
                gradient
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0_f64, f64::max)
            }
        }
    }

    /// Get the maximum norm (clipping threshold)
    pub fn max_norm(&self) -> f64 {
        self.max_norm
    }

    /// Get the clipping method
    pub fn method(&self) -> ClippingMethod {
        self.method
    }

    /// Get clipping statistics
    pub fn stats(&self) -> &ClippingStats {
        &self.stats
    }

    /// Get the fraction of gradients that were clipped
    pub fn clip_fraction(&self) -> f64 {
        if self.stats.total_processed == 0 {
            0.0
        } else {
            self.stats.total_clipped as f64 / self.stats.total_processed as f64
        }
    }

    /// Get average compression ratio (clipped_norm / original_norm)
    pub fn average_compression(&self) -> f64 {
        if self.stats.sum_original_norms == 0.0 {
            1.0
        } else {
            self.stats.sum_clipped_norms / self.stats.sum_original_norms
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ClippingStats::default();
    }
}

/// Adaptive gradient clipper that adjusts the clipping norm
///
/// Uses the median or quantile of recent gradient norms to set the threshold.
#[derive(Debug, Clone)]
pub struct AdaptiveClipper {
    /// Base clipper
    clipper: GradientClipper,
    /// Target quantile (e.g., 0.5 for median)
    target_quantile: f64,
    /// Recent gradient norms for adaptation
    norm_history: Vec<f64>,
    /// Maximum history size
    max_history: usize,
    /// Minimum allowed clip norm
    min_clip_norm: f64,
    /// Maximum allowed clip norm
    max_clip_norm: f64,
}

impl AdaptiveClipper {
    /// Create a new adaptive clipper
    pub fn new(
        initial_norm: f64,
        method: ClippingMethod,
        target_quantile: f64,
        min_clip_norm: f64,
        max_clip_norm: f64,
    ) -> DpResult<Self> {
        Ok(Self {
            clipper: GradientClipper::new(initial_norm, method)?,
            target_quantile,
            norm_history: Vec::new(),
            max_history: 1000,
            min_clip_norm,
            max_clip_norm,
        })
    }

    /// Clip a gradient and update the adaptive threshold
    pub fn clip(&mut self, gradient: &[f64]) -> Vec<f64> {
        let norm = self.compute_norm(gradient);
        self.norm_history.push(norm);

        // Trim history if too large
        if self.norm_history.len() > self.max_history {
            self.norm_history.remove(0);
        }

        // Adapt threshold every 100 samples
        if self.norm_history.len() >= 100 && self.norm_history.len() % 100 == 0 {
            self.adapt_threshold();
        }

        self.clipper.clip(gradient)
    }

    /// Adapt the clipping threshold based on recent norms
    fn adapt_threshold(&mut self) {
        let mut sorted = self.norm_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((self.target_quantile * sorted.len() as f64) as usize)
            .min(sorted.len().saturating_sub(1));

        let new_norm = sorted[idx].clamp(self.min_clip_norm, self.max_clip_norm);

        debug!(
            old_norm = self.clipper.max_norm,
            new_norm = new_norm,
            "Adapted clipping threshold"
        );

        self.clipper.max_norm = new_norm;
    }

    fn compute_norm(&self, gradient: &[f64]) -> f64 {
        match self.clipper.method {
            ClippingMethod::L2 => gradient.iter().map(|x| x * x).sum::<f64>().sqrt(),
            ClippingMethod::L1 => gradient.iter().map(|x| x.abs()).sum(),
            ClippingMethod::LInf | ClippingMethod::PerCoordinate => {
                gradient.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
            }
        }
    }

    /// Get current clip norm
    pub fn current_norm(&self) -> f64 {
        self.clipper.max_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipper_creation() {
        let clipper = GradientClipper::l2(1.0).unwrap();
        assert_eq!(clipper.max_norm(), 1.0);
        assert_eq!(clipper.method(), ClippingMethod::L2);
    }

    #[test]
    fn test_invalid_clip_norm() {
        assert!(GradientClipper::l2(-1.0).is_err());
        assert!(GradientClipper::l2(0.0).is_err());
    }

    #[test]
    fn test_no_clipping_needed() {
        let mut clipper = GradientClipper::l2(10.0).unwrap();
        let gradient = vec![1.0, 2.0, 2.0]; // norm = 3.0

        let clipped = clipper.clip(&gradient);
        assert_eq!(clipped, gradient);
        assert_eq!(clipper.stats().total_clipped, 0);
    }

    #[test]
    fn test_l2_clipping() {
        let mut clipper = GradientClipper::l2(1.0).unwrap();
        let gradient = vec![3.0, 4.0]; // norm = 5.0

        let clipped = clipper.clip(&gradient);

        // Clipped norm should be 1.0
        let clipped_norm: f64 = clipped.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-10);

        // Direction should be preserved
        assert!((clipped[0] / clipped[1] - 0.75).abs() < 1e-10);
        assert_eq!(clipper.stats().total_clipped, 1);
    }

    #[test]
    fn test_linf_clipping() {
        let mut clipper = GradientClipper::new(1.0, ClippingMethod::LInf).unwrap();
        let gradient = vec![0.5, 2.0, -3.0];

        let clipped = clipper.clip(&gradient);

        assert_eq!(clipped[0], 0.5); // Not clipped
        assert_eq!(clipped[1], 1.0); // Clipped to 1.0
        assert_eq!(clipped[2], -1.0); // Clipped to -1.0
    }

    #[test]
    fn test_clip_and_aggregate() {
        let mut clipper = GradientClipper::l2(10.0).unwrap();
        let gradients = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];

        let aggregated = clipper.clip_and_aggregate(&gradients);

        assert_eq!(aggregated.len(), 2);
        assert!((aggregated[0] - 3.0).abs() < 1e-10); // (1+3+5)/3
        assert!((aggregated[1] - 4.0).abs() < 1e-10); // (2+4+6)/3
    }

    #[test]
    fn test_clip_fraction() {
        let mut clipper = GradientClipper::l2(5.0).unwrap();

        // This won't be clipped (norm = 3)
        clipper.clip(&[1.0, 2.0, 2.0]);
        // This will be clipped (norm = 10)
        clipper.clip(&[6.0, 8.0]);

        assert!((clipper.clip_fraction() - 0.5).abs() < 1e-10);
    }
}
