// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Holochain integration module -- WASM-compatible subset of mycelix-fl.
//!
//! This module provides the FL algorithms without tokio/std dependencies,
//! suitable for use inside Holochain WASM zomes. It re-exports the core
//! defense algorithms behind a simple dispatch interface.
//!
//! # Usage from a Holochain zome
//!
//! ```rust,ignore
//! use mycelix_fl::holochain::{aggregate_gradients, validate_gradient};
//! use mycelix_fl::types::{Gradient, DefenseConfig};
//!
//! let gradients: Vec<Gradient> = collect_from_dht();
//! let config = DefenseConfig::default();
//! let result = aggregate_gradients(&gradients, "coordinate_median", &config)?;
//! ```

use crate::defenses::coordinate_median::CoordinateMedian;
use crate::defenses::fedavg::FedAvg;
use crate::defenses::rfa::Rfa;
use crate::defenses::trimmed_mean::TrimmedMean;
use crate::defenses::Defense;
use crate::error::FlError;
use crate::types::{AggregationResult, DefenseConfig, Gradient};

/// Aggregate gradients using the specified defense algorithm.
///
/// This is the main entry point for Holochain zomes. It dispatches to
/// the appropriate stateless defense algorithm by name.
///
/// # Supported defenses
///
/// - `"fedavg"` -- Vanilla averaging (no BFT)
/// - `"trimmed_mean"` -- Coordinate-wise trimmed mean
/// - `"coordinate_median"` -- Coordinate-wise median (50% BFT)
/// - `"rfa"` -- Geometric median via Weiszfeld algorithm
///
/// # Errors
///
/// Returns [`FlError::InvalidParameter`] for unknown defense names.
/// Propagates errors from the underlying defense algorithm.
pub fn aggregate_gradients(
    gradients: &[Gradient],
    defense: &str,
    config: &DefenseConfig,
) -> Result<AggregationResult, FlError> {
    match defense {
        "fedavg" => FedAvg.aggregate(gradients, config),
        "trimmed_mean" => TrimmedMean.aggregate(gradients, config),
        "coordinate_median" => CoordinateMedian.aggregate(gradients, config),
        "rfa" => Rfa.aggregate(gradients, config),
        _ => Err(FlError::InvalidParameter(format!(
            "unknown defense: {}",
            defense
        ))),
    }
}

/// Validate a gradient submission (basic structural checks, no auth).
///
/// Checks that the gradient is non-empty and contains only finite values.
/// This is suitable for use in Holochain validation callbacks.
///
/// # Errors
///
/// Returns [`FlError::EmptyGradient`] if the gradient vector is empty.
/// Returns [`FlError::InvalidInput`] if any value is NaN or infinite.
pub fn validate_gradient(gradient: &Gradient) -> Result<(), FlError> {
    if gradient.values.is_empty() {
        return Err(FlError::EmptyGradient);
    }
    for (i, v) in gradient.values.iter().enumerate() {
        if !v.is_finite() {
            return Err(FlError::InvalidInput(format!(
                "non-finite value at index {}",
                i
            )));
        }
    }
    Ok(())
}

/// Compute reputation-weighted Byzantine power.
///
/// Byzantine power = sum(rep^2) for flagged nodes. This metric is used
/// in MATL to quantify the influence of suspected Byzantine actors.
/// Squaring ensures low-reputation attackers have minimal impact.
pub fn compute_byzantine_power(reputations: &[(String, f64)], flagged: &[String]) -> f64 {
    reputations
        .iter()
        .filter(|(id, _)| flagged.contains(id))
        .map(|(_, rep)| rep * rep)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grad(node_id: &str, values: Vec<f32>) -> Gradient {
        Gradient::new(node_id, values, 1)
    }

    #[test]
    fn test_aggregate_fedavg() {
        let gs = vec![
            grad("a", vec![1.0, 2.0]),
            grad("b", vec![3.0, 4.0]),
        ];
        let result = aggregate_gradients(&gs, "fedavg", &DefenseConfig::default()).unwrap();
        assert!((result.gradient[0] - 2.0).abs() < 1e-6);
        assert!((result.gradient[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_coordinate_median() {
        let gs = vec![
            grad("a", vec![1.0]),
            grad("b", vec![3.0]),
            grad("c", vec![100.0]), // outlier
        ];
        let result =
            aggregate_gradients(&gs, "coordinate_median", &DefenseConfig::default()).unwrap();
        // Median of [1, 3, 100] = 3
        assert!((result.gradient[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_unknown_defense() {
        let gs = vec![grad("a", vec![1.0])];
        let result = aggregate_gradients(&gs, "nonexistent", &DefenseConfig::default());
        assert!(matches!(result, Err(FlError::InvalidParameter(_))));
    }

    #[test]
    fn test_validate_gradient_ok() {
        let g = grad("a", vec![1.0, 2.0, 3.0]);
        assert!(validate_gradient(&g).is_ok());
    }

    #[test]
    fn test_validate_gradient_empty() {
        let g = grad("a", vec![]);
        assert!(matches!(validate_gradient(&g), Err(FlError::EmptyGradient)));
    }

    #[test]
    fn test_validate_gradient_nan() {
        let g = grad("a", vec![1.0, f32::NAN, 3.0]);
        assert!(matches!(validate_gradient(&g), Err(FlError::InvalidInput(_))));
    }

    #[test]
    fn test_validate_gradient_inf() {
        let g = grad("a", vec![f32::INFINITY]);
        assert!(matches!(validate_gradient(&g), Err(FlError::InvalidInput(_))));
    }

    #[test]
    fn test_byzantine_power_no_flagged() {
        let reps = vec![("a".into(), 0.9), ("b".into(), 0.8)];
        assert_eq!(compute_byzantine_power(&reps, &[]), 0.0);
    }

    #[test]
    fn test_byzantine_power_one_flagged() {
        let reps = vec![
            ("a".to_string(), 0.9),
            ("b".to_string(), 0.5),
        ];
        let flagged = vec!["b".to_string()];
        let power = compute_byzantine_power(&reps, &flagged);
        assert!((power - 0.25).abs() < 1e-9); // 0.5^2
    }

    #[test]
    fn test_byzantine_power_all_flagged() {
        let reps = vec![
            ("a".to_string(), 0.9),
            ("b".to_string(), 0.5),
        ];
        let flagged = vec!["a".to_string(), "b".to_string()];
        let power = compute_byzantine_power(&reps, &flagged);
        assert!((power - (0.81 + 0.25)).abs() < 1e-9); // 0.9^2 + 0.5^2
    }
}
