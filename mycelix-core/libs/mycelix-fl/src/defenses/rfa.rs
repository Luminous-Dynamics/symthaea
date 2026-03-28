// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! RFA (Robust Federated Averaging): Geometric median via Weiszfeld algorithm.
//!
//! Computes the geometric median of client gradients, which minimizes the sum of
//! L2 distances to all points. Provably robust to Byzantine outliers.
//!
//! Reference: Pillutla et al., "Robust Aggregation for Federated Learning", 2022.

use crate::error::FlError;
use crate::types::{AggregationResult, DefenseConfig, Gradient};

use super::{validate_gradients, Defense};

/// Robust Federated Averaging — geometric median via Weiszfeld's algorithm.
pub struct Rfa;

/// Small epsilon to avoid division by zero in Weiszfeld iterations.
const WEISZFELD_EPS: f64 = 1e-8;

impl Rfa {
    /// Compute the geometric median using Weiszfeld's iteratively reweighted
    /// least squares algorithm.
    fn geometric_median(
        gradients: &[Gradient],
        dim: usize,
        max_iter: usize,
        tol: f64,
    ) -> Vec<f32> {
        let n = gradients.len();

        // Initialize at the arithmetic mean
        let mut median = vec![0.0f64; dim];
        for g in gradients {
            for (i, &v) in g.values.iter().enumerate() {
                median[i] += v as f64;
            }
        }
        for v in &mut median {
            *v /= n as f64;
        }

        for _iter in 0..max_iter {
            // Compute weights = 1 / ||median - gradient_i||
            let mut weights = Vec::with_capacity(n);
            for g in gradients {
                let dist: f64 = g
                    .values
                    .iter()
                    .zip(&median)
                    .map(|(&v, &m)| {
                        let d = v as f64 - m;
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt();
                weights.push(1.0 / (dist + WEISZFELD_EPS));
            }

            // Normalize weights
            let weight_sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= weight_sum;
            }

            // Compute new median as weighted average
            let mut new_median = vec![0.0f64; dim];
            for (g, &w) in gradients.iter().zip(&weights) {
                for (i, &v) in g.values.iter().enumerate() {
                    new_median[i] += w * v as f64;
                }
            }

            // Check convergence
            let delta: f64 = new_median
                .iter()
                .zip(&median)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();

            median = new_median;

            if delta < tol {
                break;
            }
        }

        median.iter().map(|&v| v as f32).collect()
    }
}

impl Defense for Rfa {
    fn aggregate(
        &self,
        gradients: &[Gradient],
        config: &DefenseConfig,
    ) -> Result<AggregationResult, FlError> {
        let dim = validate_gradients(gradients)?;

        // Single gradient: return it directly
        if gradients.len() == 1 {
            return Ok(AggregationResult {
                gradient: gradients[0].values.clone(),
                included_nodes: vec![gradients[0].node_id.clone()],
                excluded_nodes: Vec::new(),
                scores: vec![(gradients[0].node_id.clone(), 1.0)],
            });
        }

        let gradient =
            Self::geometric_median(gradients, dim, config.weiszfeld_max_iter, config.weiszfeld_tol);

        Ok(AggregationResult {
            gradient,
            included_nodes: gradients.iter().map(|g| g.node_id.clone()).collect(),
            excluded_nodes: Vec::new(),
            scores: gradients.iter().map(|g| (g.node_id.clone(), 1.0)).collect(),
        })
    }

    fn name(&self) -> &str {
        "rfa"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::l2_distance;

    fn grad(values: Vec<f32>, id: &str) -> Gradient {
        Gradient {
            values,
            node_id: id.into(),
            round: 0,
        }
    }

    #[test]
    fn test_geometric_median_closer_to_cluster() {
        // 3 honest nodes near [1,1], 1 outlier at [100, 100]
        let gradients = vec![
            grad(vec![1.0, 1.0], "a"),
            grad(vec![1.1, 0.9], "b"),
            grad(vec![0.9, 1.1], "c"),
            grad(vec![100.0, 100.0], "evil"),
        ];
        let result = Rfa
            .aggregate(&gradients, &DefenseConfig::default())
            .unwrap();

        // Should be much closer to [1, 1] than to [100, 100]
        let dist_honest = l2_distance(&result.gradient, &[1.0, 1.0]);
        let dist_outlier = l2_distance(&result.gradient, &[100.0, 100.0]);
        assert!(
            dist_honest < dist_outlier,
            "geometric median should be closer to honest cluster: honest={dist_honest}, outlier={dist_outlier}"
        );
        // Should be within reasonable range of the honest cluster
        assert!(dist_honest < 2.0, "should be near honest cluster: {dist_honest}");
    }

    #[test]
    fn test_identical_gradients() {
        let g = grad(vec![3.0, 4.0, 5.0], "a");
        let result = Rfa
            .aggregate(
                &[g.clone(), g.clone(), g],
                &DefenseConfig::default(),
            )
            .unwrap();
        assert!((result.gradient[0] - 3.0).abs() < 1e-4);
        assert!((result.gradient[1] - 4.0).abs() < 1e-4);
        assert!((result.gradient[2] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_single_gradient() {
        let g = grad(vec![5.0, 10.0], "x");
        let result = Rfa
            .aggregate(&[g], &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 5.0).abs() < 1e-6);
        assert!((result.gradient[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_two_gradients() {
        // With 2 points, the geometric median is on the line segment between them.
        let a = grad(vec![0.0, 0.0], "a");
        let b = grad(vec![10.0, 0.0], "b");
        let result = Rfa
            .aggregate(&[a, b], &DefenseConfig::default())
            .unwrap();
        // Should be at or very near the midpoint for 2 equidistant points
        assert!((result.gradient[0] - 5.0).abs() < 0.5);
        assert!(result.gradient[1].abs() < 0.5);
    }

    #[test]
    fn test_empty_input() {
        let result = Rfa.aggregate(&[], &DefenseConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = grad(vec![1.0], "a");
        let b = grad(vec![1.0, 2.0], "b");
        let result = Rfa.aggregate(&[a, b], &DefenseConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_convergence_with_custom_config() {
        let gradients = vec![
            grad(vec![1.0], "a"),
            grad(vec![2.0], "b"),
            grad(vec![3.0], "c"),
        ];
        let config = DefenseConfig {
            weiszfeld_max_iter: 200,
            weiszfeld_tol: 1e-10,
            ..DefenseConfig::default()
        };
        let result = Rfa.aggregate(&gradients, &config).unwrap();
        // Geometric median of [1, 2, 3] in 1D is 2.0 (the actual median)
        assert!((result.gradient[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_name() {
        assert_eq!(Rfa.name(), "rfa");
    }
}
