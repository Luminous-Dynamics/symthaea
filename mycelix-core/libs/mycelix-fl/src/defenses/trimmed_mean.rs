// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trimmed Mean: Coordinate-wise robust aggregation.
//!
//! For each coordinate, sort values across all nodes, trim `trim_ratio` fraction
//! from both ends, then average the remaining. Byzantine-robust for f < (1-2β)n.
//!
//! Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards
//! Optimal Statistical Rates", ICML 2018.

use crate::error::FlError;
use crate::types::{AggregationResult, DefenseConfig, Gradient};

use super::{validate_gradients, Defense};

/// Coordinate-wise trimmed mean aggregation.
pub struct TrimmedMean;

impl Defense for TrimmedMean {
    fn aggregate(
        &self,
        gradients: &[Gradient],
        config: &DefenseConfig,
    ) -> Result<AggregationResult, FlError> {
        let dim = validate_gradients(gradients)?;
        let n = gradients.len();

        if config.trim_ratio < 0.0 || config.trim_ratio >= 0.5 {
            return Err(FlError::InvalidParameter(format!(
                "trim_ratio must be in [0, 0.5), got {}",
                config.trim_ratio
            )));
        }

        let n_trim = (n as f32 * config.trim_ratio) as usize;

        // If nothing to trim, fall back to simple mean
        if n_trim == 0 {
            let mut sum = vec![0.0f64; dim];
            for g in gradients {
                for (i, &v) in g.values.iter().enumerate() {
                    sum[i] += v as f64;
                }
            }
            let gradient: Vec<f32> = sum.iter().map(|&s| (s / n as f64) as f32).collect();
            return Ok(AggregationResult {
                gradient,
                included_nodes: gradients.iter().map(|g| g.node_id.clone()).collect(),
                excluded_nodes: Vec::new(),
                scores: gradients.iter().map(|g| (g.node_id.clone(), 1.0)).collect(),
            });
        }

        let remaining = n - 2 * n_trim;
        if remaining == 0 {
            return Err(FlError::InvalidParameter(format!(
                "trim_ratio {} trims all {} gradients (2 × {} = {})",
                config.trim_ratio,
                n,
                n_trim,
                2 * n_trim
            )));
        }

        let mut result = vec![0.0f32; dim];
        let mut col = Vec::with_capacity(n);

        for d in 0..dim {
            col.clear();
            for g in gradients {
                col.push(g.values[d]);
            }
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

            let trimmed = &col[n_trim..n - n_trim];
            let sum: f64 = trimmed.iter().map(|&v| v as f64).sum();
            result[d] = (sum / remaining as f64) as f32;
        }

        Ok(AggregationResult {
            gradient: result,
            included_nodes: gradients.iter().map(|g| g.node_id.clone()).collect(),
            excluded_nodes: Vec::new(),
            scores: gradients.iter().map(|g| (g.node_id.clone(), 1.0)).collect(),
        })
    }

    fn name(&self) -> &str {
        "trimmed_mean"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grad(values: Vec<f32>, id: &str) -> Gradient {
        Gradient {
            values,
            node_id: id.into(),
            round: 0,
        }
    }

    #[test]
    fn test_outlier_excluded() {
        // 5 nodes: 4 honest at ~1.0, 1 outlier at 100.0
        let gradients = vec![
            grad(vec![1.0], "a"),
            grad(vec![1.0], "b"),
            grad(vec![1.0], "c"),
            grad(vec![1.0], "d"),
            grad(vec![100.0], "evil"),
        ];
        let mut config = DefenseConfig::default();
        config.trim_ratio = 0.2; // Trim 1 from each end (20% of 5 = 1)

        let result = TrimmedMean.aggregate(&gradients, &config).unwrap();
        // After trimming: [1.0, 1.0, 1.0] — the outlier and one honest are trimmed
        assert!((result.gradient[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_trimming_when_ratio_zero() {
        let gradients = vec![
            grad(vec![1.0, 2.0], "a"),
            grad(vec![3.0, 4.0], "b"),
        ];
        let mut config = DefenseConfig::default();
        config.trim_ratio = 0.0;

        let result = TrimmedMean.aggregate(&gradients, &config).unwrap();
        assert!((result.gradient[0] - 2.0).abs() < 1e-6);
        assert!((result.gradient[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_trimming_when_n_too_small() {
        // 2 nodes with trim_ratio=0.1 → n_trim = int(2*0.1) = 0
        let gradients = vec![
            grad(vec![1.0], "a"),
            grad(vec![3.0], "b"),
        ];
        let config = DefenseConfig::default(); // trim_ratio = 0.1

        let result = TrimmedMean.aggregate(&gradients, &config).unwrap();
        assert!((result.gradient[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_gradient() {
        let g = grad(vec![5.0, 10.0], "x");
        let result = TrimmedMean
            .aggregate(&[g], &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let result = TrimmedMean.aggregate(&[], &DefenseConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = grad(vec![1.0], "a");
        let b = grad(vec![1.0, 2.0], "b");
        let result = TrimmedMean.aggregate(&[a, b], &DefenseConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_trim_ratio() {
        let g = grad(vec![1.0], "a");
        let mut config = DefenseConfig::default();
        config.trim_ratio = 0.5;
        let result = TrimmedMean.aggregate(&[g], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_name() {
        assert_eq!(TrimmedMean.name(), "trimmed_mean");
    }
}
