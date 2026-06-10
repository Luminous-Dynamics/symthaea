// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hybrid Reputation-Weighted Trimmed Mean for Byzantine Fault Tolerance
//!
//! Combines per-dimension trimmed-mean outlier detection with reputation²
//! weighting to achieve stronger Byzantine tolerance than either alone.
//!
//! ## Algorithm
//!
//! 1. **Reputation Gate**: Drop contributions below minimum reputation
//! 2. **Reputation-Weighted Outlier Score**: Per-dimension outlier counting
//!    weighted by inverse reputation (low-rep nodes penalized more)
//! 3. **Trimmed Mean**: Remove top outlier-scoring contributions
//! 4. **Reputation² Aggregation**: Weight remaining by reputation^exponent

use crate::types::GradientUpdate;

/// Configuration for hybrid reputation-weighted trimmed mean
#[derive(Debug, Clone)]
pub struct HybridBftConfig {
    /// Minimum reputation to participate (below this = dropped)
    pub min_reputation: f32,
    /// Fraction of contributions to trim as outliers (0.0 to 0.5)
    pub trim_fraction: f32,
    /// Number of dimensions to sample for outlier detection (0 = ~10%)
    pub sample_dims: usize,
    /// Reputation exponent for weighting (2.0 = quadratic)
    pub reputation_exponent: f32,
    /// Weight factor for reputation in outlier scoring
    pub reputation_outlier_weight: f32,
}

impl Default for HybridBftConfig {
    fn default() -> Self {
        Self {
            min_reputation: 0.3,
            trim_fraction: 0.1,
            sample_dims: 0,
            reputation_exponent: 2.0,
            reputation_outlier_weight: 0.5,
        }
    }
}

/// A gradient contribution with its associated reputation score
#[derive(Debug, Clone)]
pub struct ReputationGradient {
    /// The gradient update data
    pub update: GradientUpdate,
    /// Reputation score (0.0 to 1.0)
    pub reputation: f32,
}

/// Result of hybrid BFT aggregation
#[derive(Debug)]
pub struct HybridAggregationResult {
    /// Aggregated gradient values
    pub aggregated: Vec<f32>,
    /// Number of contributions that passed the reputation gate
    pub gated_count: usize,
    /// Number of contributions that survived trimming
    pub surviving_count: usize,
    /// Total reputation-squared weight of surviving contributions
    pub total_weight: f32,
    /// Indices of contributions that were trimmed (potential Byzantines)
    pub trimmed_indices: Vec<usize>,
}

/// Perform hybrid reputation-weighted trimmed mean aggregation.
///
/// Returns `None` if not enough valid contributions.
pub fn hybrid_trimmed_mean(
    contributions: &[ReputationGradient],
    config: &HybridBftConfig,
) -> Option<HybridAggregationResult> {
    if contributions.is_empty() {
        return None;
    }

    // Phase 1: Reputation Gate
    let gated: Vec<(usize, &ReputationGradient)> = contributions
        .iter()
        .enumerate()
        .filter(|(_, c)| c.reputation >= config.min_reputation)
        .collect();

    let gated_count = gated.len();
    if gated_count < 2 {
        return None;
    }

    let dim = gated[0].1.update.gradients.len();
    if dim == 0 {
        return None;
    }

    // Phase 2: Reputation-Weighted Outlier Detection
    let trim_count = ((gated_count as f32 * config.trim_fraction) as usize).min(gated_count / 2);

    let (surviving, trimmed_indices) = if trim_count > 0 && gated_count >= 4 {
        let sample_dims = if config.sample_dims > 0 && config.sample_dims < dim {
            config.sample_dims
        } else {
            (dim / 10).clamp(1, 100)
        };
        let step = (dim / sample_dims).max(1);
        let sampled: Vec<usize> = (0..dim).step_by(step).collect();
        let num_sampled = sampled.len();

        let mut outlier_scores: Vec<f64> = vec![0.0; gated_count];

        for &d in &sampled {
            let mut values: Vec<(usize, f32)> = gated
                .iter()
                .enumerate()
                .map(|(local_idx, (_, c))| (local_idx, c.update.gradients[d]))
                .collect();
            values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..trim_count.min(values.len()) {
                let low_idx = values[i].0;
                let high_idx = values[values.len() - 1 - i].0;

                let low_rep = gated[low_idx].1.reputation;
                let high_rep = gated[high_idx].1.reputation;

                let rep_weight = config.reputation_outlier_weight;
                let low_penalty =
                    1.0 + rep_weight * (1.0 - low_rep.powf(config.reputation_exponent));
                let high_penalty =
                    1.0 + rep_weight * (1.0 - high_rep.powf(config.reputation_exponent));

                outlier_scores[low_idx] += low_penalty as f64;
                outlier_scores[high_idx] += high_penalty as f64;
            }
        }

        let max_possible = num_sampled as f64 * (1.0 + config.reputation_outlier_weight as f64);
        for score in outlier_scores.iter_mut() {
            *score /= max_possible.max(1.0);
        }

        let mut indexed_scores: Vec<(usize, f64)> = outlier_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let trimmed_local_indices: Vec<usize> = indexed_scores
            .iter()
            .take(trim_count)
            .map(|(i, _)| *i)
            .collect();

        let trimmed_original_indices: Vec<usize> = trimmed_local_indices
            .iter()
            .map(|&local_idx| gated[local_idx].0)
            .collect();

        let surviving: Vec<&ReputationGradient> = gated
            .iter()
            .enumerate()
            .filter(|(i, _)| !trimmed_local_indices.contains(i))
            .map(|(_, (_, c))| *c)
            .collect();

        (surviving, trimmed_original_indices)
    } else {
        let surviving: Vec<&ReputationGradient> = gated.iter().map(|(_, c)| *c).collect();
        (surviving, vec![])
    };

    let surviving_count = surviving.len();
    if surviving_count == 0 {
        return None;
    }

    // Phase 3: Reputation² Weighted Aggregation
    let weights: Vec<f32> = surviving
        .iter()
        .map(|c| c.reputation.powf(config.reputation_exponent))
        .collect();

    let total_weight: f32 = weights.iter().sum();
    if total_weight <= 0.0 {
        return None;
    }

    let mut aggregated = vec![0.0f32; dim];
    for (contrib, &weight) in surviving.iter().zip(weights.iter()) {
        let normalized = weight / total_weight;
        for (i, &g) in contrib.update.gradients.iter().enumerate() {
            aggregated[i] += g * normalized;
        }
    }

    Some(HybridAggregationResult {
        aggregated,
        gated_count,
        surviving_count,
        total_weight,
        trimmed_indices,
    })
}

/// Compute effective Byzantine fraction after reputation gating.
///
/// Shows how reputation disparity reduces Byzantine voting power.
pub fn effective_byzantine_fraction(
    total_nodes: usize,
    byzantine_nodes: usize,
    avg_byzantine_reputation: f32,
    avg_honest_reputation: f32,
    reputation_exponent: f32,
) -> f32 {
    let honest_nodes = total_nodes - byzantine_nodes;

    let byz_power = byzantine_nodes as f32 * avg_byzantine_reputation.powf(reputation_exponent);
    let honest_power = honest_nodes as f32 * avg_honest_reputation.powf(reputation_exponent);

    let total_power = byz_power + honest_power;
    if total_power <= 0.0 {
        return 0.0;
    }

    byz_power / total_power
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rep_gradient(id: &str, values: Vec<f32>, reputation: f32) -> ReputationGradient {
        ReputationGradient {
            update: GradientUpdate::new(id.to_string(), 1, values, 100, 0.5),
            reputation,
        }
    }

    #[test]
    fn test_basic_hybrid() {
        let contributions = vec![
            make_rep_gradient("p1", vec![0.10, 0.20, 0.30], 0.9),
            make_rep_gradient("p2", vec![0.12, 0.18, 0.28], 0.85),
            make_rep_gradient("p3", vec![0.11, 0.22, 0.32], 0.88),
        ];
        let result = hybrid_trimmed_mean(&contributions, &HybridBftConfig::default()).unwrap();
        assert_eq!(result.gated_count, 3);
        assert_eq!(result.surviving_count, 3);
        assert_eq!(result.aggregated.len(), 3);
    }

    #[test]
    fn test_reputation_gate() {
        let contributions = vec![
            make_rep_gradient("p1", vec![0.1, 0.2], 0.9),
            make_rep_gradient("p2", vec![0.1, 0.2], 0.8),
            make_rep_gradient("p3", vec![0.1, 0.2], 0.1),
            make_rep_gradient("p4", vec![0.1, 0.2], 0.05),
        ];
        let config = HybridBftConfig {
            min_reputation: 0.3,
            ..Default::default()
        };
        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();
        assert_eq!(result.gated_count, 2);
    }

    #[test]
    fn test_byzantine_detection() {
        let contributions = vec![
            make_rep_gradient("p1", vec![0.10, 0.20, 0.30], 0.9),
            make_rep_gradient("p2", vec![0.12, 0.18, 0.28], 0.85),
            make_rep_gradient("p3", vec![0.11, 0.22, 0.32], 0.88),
            make_rep_gradient("p4", vec![0.09, 0.21, 0.29], 0.87),
            make_rep_gradient("p5", vec![0.13, 0.19, 0.31], 0.86),
            make_rep_gradient("byz1", vec![10.0, -5.0, 100.0], 0.4),
            make_rep_gradient("byz2", vec![-8.0, 50.0, -30.0], 0.35),
        ];
        let config = HybridBftConfig {
            trim_fraction: 0.3,
            reputation_outlier_weight: 0.8,
            ..Default::default()
        };
        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();
        assert!(
            result.trimmed_indices.contains(&5) || result.trimmed_indices.contains(&6),
            "At least one Byzantine should be trimmed: {:?}",
            result.trimmed_indices
        );
        assert_eq!(result.surviving_count, 5);
        assert!((result.aggregated[0] - 0.11).abs() < 0.05);
    }

    #[test]
    fn test_reputation_weighting() {
        let contributions = vec![
            make_rep_gradient("p1", vec![1.0, 1.0], 0.9),
            make_rep_gradient("p2", vec![0.0, 0.0], 0.3),
        ];
        let config = HybridBftConfig {
            min_reputation: 0.1,
            trim_fraction: 0.0,
            ..Default::default()
        };
        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();
        assert!(
            (result.aggregated[0] - 0.9).abs() < 0.01,
            "High-rep should dominate: got {}",
            result.aggregated[0]
        );
    }

    #[test]
    fn test_effective_byzantine_fraction_low_rep() {
        let frac = effective_byzantine_fraction(100, 34, 0.3, 0.9, 2.0);
        assert!(
            frac < 0.06,
            "34% Byzantine at low rep should have <6% power, got {}",
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
    fn test_34_percent_converges() {
        let mut contributions = Vec::new();
        for i in 0..66 {
            let val = 0.5 + (i as f32 * 0.001);
            contributions.push(make_rep_gradient(
                &format!("h{}", i),
                vec![val; 10],
                0.85 + (i as f32 * 0.001),
            ));
        }
        for i in 0..20 {
            let val = if i % 2 == 0 { 100.0 } else { -100.0 };
            contributions.push(make_rep_gradient(&format!("bg{}", i), vec![val; 10], 0.15));
        }
        for i in 0..14 {
            let val = if i % 2 == 0 { 100.0 } else { -100.0 };
            contributions.push(make_rep_gradient(&format!("bt{}", i), vec![val; 10], 0.4));
        }

        let config = HybridBftConfig {
            min_reputation: 0.3,
            trim_fraction: 0.2,
            reputation_outlier_weight: 0.7,
            ..Default::default()
        };
        let result = hybrid_trimmed_mean(&contributions, &config).unwrap();
        assert_eq!(result.gated_count, 80);
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
        assert!(hybrid_trimmed_mean(&[], &HybridBftConfig::default()).is_none());
    }

    #[test]
    fn test_all_below_min_reputation() {
        let contributions = vec![
            make_rep_gradient("p1", vec![0.1], 0.1),
            make_rep_gradient("p2", vec![0.2], 0.2),
        ];
        assert!(hybrid_trimmed_mean(&contributions, &HybridBftConfig::default()).is_none());
    }
}
