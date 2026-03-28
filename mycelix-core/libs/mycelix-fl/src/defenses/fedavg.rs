// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! FedAvg: Vanilla Federated Averaging (baseline).
//!
//! Simple unweighted mean of all client gradients. NOT Byzantine-robust.
//!
//! Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
//! from Decentralized Data", AISTATS 2017.

use crate::error::FlError;
use crate::types::{AggregationResult, DefenseConfig, Gradient};

use super::{validate_gradients, Defense};

/// Vanilla Federated Averaging — simple element-wise mean.
pub struct FedAvg;

impl Defense for FedAvg {
    fn aggregate(
        &self,
        gradients: &[Gradient],
        _config: &DefenseConfig,
    ) -> Result<AggregationResult, FlError> {
        let dim = validate_gradients(gradients)?;
        let n = gradients.len() as f64;

        let mut sum = vec![0.0f64; dim];
        for g in gradients {
            for (i, &v) in g.values.iter().enumerate() {
                sum[i] += v as f64;
            }
        }

        let gradient: Vec<f32> = sum.iter().map(|&s| (s / n) as f32).collect();
        let included: Vec<String> = gradients.iter().map(|g| g.node_id.clone()).collect();

        Ok(AggregationResult {
            gradient,
            included_nodes: included,
            excluded_nodes: Vec::new(),
            scores: gradients.iter().map(|g| (g.node_id.clone(), 1.0)).collect(),
        })
    }

    fn name(&self) -> &str {
        "fedavg"
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
    fn test_mean_of_identical() {
        let g = grad(vec![1.0, 2.0, 3.0], "a");
        let result = FedAvg
            .aggregate(&[g.clone(), g.clone(), g], &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 1.0).abs() < 1e-6);
        assert!((result.gradient[1] - 2.0).abs() < 1e-6);
        assert!((result.gradient[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_of_two() {
        let a = grad(vec![1.0, 0.0], "a");
        let b = grad(vec![0.0, 1.0], "b");
        let result = FedAvg
            .aggregate(&[a, b], &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 0.5).abs() < 1e-6);
        assert!((result.gradient[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_single_gradient() {
        let g = grad(vec![5.0, 10.0], "x");
        let result = FedAvg
            .aggregate(&[g], &DefenseConfig::default())
            .unwrap();
        assert!((result.gradient[0] - 5.0).abs() < 1e-6);
        assert!((result.gradient[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let result = FedAvg.aggregate(&[], &DefenseConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = grad(vec![1.0, 2.0], "a");
        let b = grad(vec![1.0, 2.0, 3.0], "b");
        let result = FedAvg.aggregate(&[a, b], &DefenseConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_all_nodes_included() {
        let a = grad(vec![1.0], "a");
        let b = grad(vec![2.0], "b");
        let result = FedAvg
            .aggregate(&[a, b], &DefenseConfig::default())
            .unwrap();
        assert_eq!(result.included_nodes.len(), 2);
        assert!(result.excluded_nodes.is_empty());
    }

    #[test]
    fn test_name() {
        assert_eq!(FedAvg.name(), "fedavg");
    }
}
