// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Coordinate-wise Median: simple baseline defense.
//!
//! Computes the median of each gradient coordinate independently.
//! Byzantine-robust up to 50% Byzantine fraction.
//!
//! Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards
//! Optimal Statistical Rates", ICML 2018.

use crate::error::FlError;
use crate::types::{AggregationResult, DefenseConfig, Gradient};

use super::{validate_gradients, Defense};

/// Coordinate-wise median aggregation.
pub struct CoordinateMedian;

/// Compute the median of a mutable slice of f32 values.
/// The slice will be sorted in place.
fn median_of(vals: &mut [f32]) -> f32 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

impl Defense for CoordinateMedian {
    fn aggregate(
        &self,
        gradients: &[Gradient],
        _config: &DefenseConfig,
    ) -> Result<AggregationResult, FlError> {
        let dim = validate_gradients(gradients)?;
        let n = gradients.len();

        let mut result = vec![0.0f32; dim];
        let mut col = Vec::with_capacity(n);

        for d in 0..dim {
            col.clear();
            for g in gradients {
                col.push(g.values[d]);
            }
            result[d] = median_of(&mut col);
        }

        Ok(AggregationResult {
            gradient: result,
            included_nodes: gradients.iter().map(|g| g.node_id.clone()).collect(),
            excluded_nodes: Vec::new(),
            scores: gradients.iter().map(|g| (g.node_id.clone(), 1.0)).collect(),
        })
    }

    fn name(&self) -> &str {
        "coordinate_median"
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
    fn test_median_robust_to_outlier() {
        // Median of [1, 2, 100] = 2
        let gradients = vec![
            grad(vec![1.0], "a"),
            grad(vec![2.0], "b"),
            grad(vec![100.0], "evil"),
        ];
        let result = CoordinateMedian
            .aggregate(&gradients, &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_median_even_count() {
        // Median of [1, 2, 3, 4] = 2.5
        let gradients = vec![
            grad(vec![1.0], "a"),
            grad(vec![2.0], "b"),
            grad(vec![3.0], "c"),
            grad(vec![4.0], "d"),
        ];
        let result = CoordinateMedian
            .aggregate(&gradients, &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_single_gradient() {
        let g = grad(vec![7.0, 8.0], "x");
        let result = CoordinateMedian
            .aggregate(&[g], &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 7.0).abs() < 1e-6);
        assert!((result.gradient[1] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_dimensional() {
        let gradients = vec![
            grad(vec![1.0, 10.0], "a"),
            grad(vec![2.0, 20.0], "b"),
            grad(vec![3.0, 30.0], "c"),
        ];
        let result = CoordinateMedian
            .aggregate(&gradients, &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 2.0).abs() < 1e-6);
        assert!((result.gradient[1] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let result = CoordinateMedian.aggregate(&[], &DefenseConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = grad(vec![1.0], "a");
        let b = grad(vec![1.0, 2.0], "b");
        let result = CoordinateMedian.aggregate(&[a, b], &DefenseConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_name() {
        assert_eq!(CoordinateMedian.name(), "coordinate_median");
    }
}
