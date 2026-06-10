// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hybrid Reputation-Weighted Trimmed Mean for Byzantine Fault Tolerance
//!
//! Combines Symthaea's per-dimension trimmed-mean outlier detection with
//! Mycelix's reputation² weighting to achieve stronger Byzantine tolerance
//! than either approach alone.
//!
//! ## Implementation
//!
//! This module delegates to `mycelix_fl_core::hybrid_bft` for the core algorithm
//! and provides Symthaea-specific wrappers that work with `GradientMessage`.
//!
//! ## Algorithm
//!
//! 1. **Reputation Gate**: Drop contributions from nodes below minimum reputation
//! 2. **Reputation-Weighted Outlier Score**: Per-dimension outlier counting weighted
//!    by inverse reputation (low-rep nodes penalized more for being outliers)
//! 3. **Trimmed Mean**: Remove top outlier-scoring contributions
//! 4. **Reputation² Aggregation**: Weight remaining contributions by reputation²

use super::federated_cfc::GradientMessage;

// Re-export core types for convenience
pub use mycelix_fl_core::hybrid_bft::{HybridAggregationResult, HybridBftConfig};

/// A contribution with its associated reputation score
///
/// This is the Symthaea-specific wrapper that uses `GradientMessage`
/// (with source_id, trust_score, etc.) instead of the core crate's
/// `GradientUpdate` type.
#[derive(Debug, Clone)]
pub struct ReputationGradient {
    /// The gradient data
    pub gradient: GradientMessage,
    /// Reputation score from Holochain/MATL (0.0 to 1.0)
    pub reputation: f32,
}

/// Perform hybrid reputation-weighted trimmed mean aggregation.
///
/// This wraps `mycelix_fl_core::hybrid_bft::hybrid_trimmed_mean` by converting
/// `GradientMessage`-based `ReputationGradient` to the core crate's types.
///
/// # Arguments
/// * `contributions` - Gradient contributions with reputation scores
/// * `config` - Hybrid BFT configuration
///
/// # Returns
/// * `Some(HybridAggregationResult)` if aggregation succeeded
/// * `None` if not enough valid contributions
pub fn hybrid_trimmed_mean(
    contributions: &[ReputationGradient],
    config: &HybridBftConfig,
) -> Option<HybridAggregationResult> {
    if contributions.is_empty() {
        return None;
    }

    // Convert Symthaea's ReputationGradient to core's ReputationGradient,
    // filtering out invalid contributions
    let core_contributions: Vec<mycelix_fl_core::hybrid_bft::ReputationGradient> = contributions
        .iter()
        .filter(|c| {
            c.reputation.is_finite()
                && c.reputation >= 0.0
                && c.reputation <= 1.0
                && c.gradient.gradient_data.len() < 1_000_000
        })
        .map(|c| {
            let sample_count = (c.gradient.sample_count as u64).min(u32::MAX as u64) as u32;
            let update = mycelix_fl_core::types::GradientUpdate::new(
                hex::encode(c.gradient.source_id),
                c.gradient.model_version,
                c.gradient.gradient_data.clone(),
                sample_count,
                0.5, // loss not tracked in GradientMessage
            );
            mycelix_fl_core::hybrid_bft::ReputationGradient {
                update,
                reputation: c.reputation,
            }
        })
        .collect();

    mycelix_fl_core::hybrid_bft::hybrid_trimmed_mean(&core_contributions, config)
}

/// Compute the effective Byzantine fraction after reputation gating.
///
/// Delegates to `mycelix_fl_core::hybrid_bft::effective_byzantine_fraction`.
pub fn effective_byzantine_fraction(
    total_nodes: usize,
    byzantine_nodes: usize,
    avg_byzantine_reputation: f32,
    avg_honest_reputation: f32,
    reputation_exponent: f32,
) -> f32 {
    mycelix_fl_core::hybrid_bft::effective_byzantine_fraction(
        total_nodes,
        byzantine_nodes,
        avg_byzantine_reputation,
        avg_honest_reputation,
        reputation_exponent,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient(id: u8, values: Vec<f32>, reputation: f32) -> ReputationGradient {
        ReputationGradient {
            gradient: GradientMessage::new([id; 32], values, reputation),
            reputation,
        }
    }

    #[test]
    fn test_basic_hybrid_aggregation() {
        let contributions = vec![
            make_gradient(1, vec![0.10, 0.20, 0.30], 0.9),
            make_gradient(2, vec![0.12, 0.18, 0.28], 0.85),
            make_gradient(3, vec![0.11, 0.22, 0.32], 0.88),
        ];

        let config = HybridBftConfig::default();
        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();

        assert_eq!(result.gated_count, 3);
        assert_eq!(result.surviving_count, 3);
        assert_eq!(result.aggregated.len(), 3);

        for val in &result.aggregated {
            assert!(*val > 0.0);
        }
    }

    #[test]
    fn test_reputation_gate_filters_low_rep() {
        let contributions = vec![
            make_gradient(1, vec![0.1, 0.2], 0.9),
            make_gradient(2, vec![0.1, 0.2], 0.8),
            make_gradient(3, vec![0.1, 0.2], 0.1),
            make_gradient(4, vec![0.1, 0.2], 0.05),
        ];

        let config = HybridBftConfig {
            min_reputation: 0.3,
            ..Default::default()
        };
        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();

        assert_eq!(result.gated_count, 2);
    }

    #[test]
    fn test_byzantine_detection_via_trimming() {
        let contributions = vec![
            make_gradient(1, vec![0.10, 0.20, 0.30], 0.9),
            make_gradient(2, vec![0.12, 0.18, 0.28], 0.85),
            make_gradient(3, vec![0.11, 0.22, 0.32], 0.88),
            make_gradient(4, vec![0.09, 0.21, 0.29], 0.87),
            make_gradient(5, vec![0.13, 0.19, 0.31], 0.86),
            make_gradient(6, vec![10.0, -5.0, 100.0], 0.4),
            make_gradient(7, vec![-8.0, 50.0, -30.0], 0.35),
        ];

        let config = HybridBftConfig {
            trim_fraction: 0.3,
            reputation_outlier_weight: 0.8,
            ..Default::default()
        };

        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();

        assert!(
            result.trimmed_indices.contains(&5) || result.trimmed_indices.contains(&6),
            "At least one Byzantine node should be trimmed, trimmed: {:?}",
            result.trimmed_indices
        );
        assert_eq!(result.surviving_count, 5);
        assert!((result.aggregated[0] - 0.11).abs() < 0.05);
    }

    #[test]
    fn test_reputation_weighting_reduces_low_rep_influence() {
        let contributions = vec![
            make_gradient(1, vec![1.0, 1.0], 0.9),
            make_gradient(2, vec![0.0, 0.0], 0.3),
        ];

        let config = HybridBftConfig {
            min_reputation: 0.1,
            trim_fraction: 0.0,
            ..Default::default()
        };

        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();

        assert!(
            (result.aggregated[0] - 0.9).abs() < 0.01,
            "High-rep node should dominate: expected ~0.9, got {}",
            result.aggregated[0]
        );
    }

    #[test]
    fn test_effective_byzantine_fraction_low_rep() {
        let frac = effective_byzantine_fraction(100, 34, 0.3, 0.9, 2.0);
        assert!(
            frac < 0.06,
            "34% Byzantine at low rep should have <6% effective power, got {}",
            frac
        );
    }

    #[test]
    fn test_effective_byzantine_fraction_same_rep() {
        let frac = effective_byzantine_fraction(100, 45, 0.9, 0.9, 2.0);
        assert!(
            (frac - 0.45).abs() < 0.01,
            "Same rep should give no improvement"
        );
    }

    #[test]
    fn test_34_percent_byzantine_with_low_rep_converges() {
        let mut contributions = Vec::new();

        for i in 0..66 {
            let val = 0.5 + (i as f32 * 0.001);
            contributions.push(make_gradient(i, vec![val; 10], 0.85 + (i as f32 * 0.001)));
        }

        for i in 0..20 {
            let val = if i % 2 == 0 { 100.0 } else { -100.0 };
            contributions.push(make_gradient(66 + i, vec![val; 10], 0.15));
        }
        for i in 0..14 {
            let val = if i % 2 == 0 { 100.0 } else { -100.0 };
            contributions.push(make_gradient(86 + i, vec![val; 10], 0.4));
        }

        let config = HybridBftConfig {
            min_reputation: 0.3,
            trim_fraction: 0.2,
            reputation_outlier_weight: 0.7,
            ..Default::default()
        };

        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();

        assert_eq!(result.gated_count, 80, "20 should be gated out");

        for (i, val) in result.aggregated.iter().enumerate() {
            assert!(
                (*val - 0.5).abs() < 0.15,
                "Dim {} should be ~0.5, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_empty_contributions() {
        let result = hybrid_trimmed_mean(&[], &HybridBftConfig::default());
        assert!(result.is_none());
    }

    #[test]
    fn test_all_below_min_reputation() {
        let contributions = vec![
            make_gradient(1, vec![0.1], 0.1),
            make_gradient(2, vec![0.2], 0.2),
        ];
        let result = hybrid_trimmed_mean(&contributions, &HybridBftConfig::default());
        assert!(result.is_none(), "All below min rep should return None");
    }

    #[test]
    fn test_invalid_contributions_filtered() {
        // 3 valid contributions
        let valid1 = make_gradient(1, vec![0.10, 0.20, 0.30], 0.9);
        let valid2 = make_gradient(7, vec![0.12, 0.21, 0.29], 0.85);
        let valid3 = make_gradient(8, vec![0.11, 0.19, 0.31], 0.88);

        // Invalid: NaN reputation
        let nan_rep = make_gradient(2, vec![0.1, 0.2, 0.3], f32::NAN);
        // Invalid: negative reputation
        let neg_rep = make_gradient(3, vec![0.1, 0.2, 0.3], -0.5);
        // Invalid: reputation > 1.0
        let high_rep = make_gradient(4, vec![0.1, 0.2, 0.3], 1.5);
        // Invalid: infinity reputation
        let inf_rep = make_gradient(5, vec![0.1, 0.2, 0.3], f32::INFINITY);

        let contributions = vec![valid1, valid2, valid3, nan_rep, neg_rep, high_rep, inf_rep];

        let config = HybridBftConfig {
            min_reputation: 0.0,
            ..Default::default()
        };

        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();
        // Only the 3 valid contributions should pass our pre-filter
        assert_eq!(
            result.gated_count, 3,
            "Only 3 valid contributions should pass validation, got {}",
            result.gated_count
        );
    }
}
