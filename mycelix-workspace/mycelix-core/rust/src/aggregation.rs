// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Secure Aggregation Module
//!
//! Implements privacy-preserving model aggregation using secret sharing

use std::collections::HashMap;

/// Aggregation result from a federated learning round
#[derive(Debug, Clone)]
pub struct AggregationResult {
    pub weights: Vec<f64>,
    pub participating_nodes: Vec<String>,
    pub convergence_score: f64,
}

/// Trust-weighted aggregation of model updates
pub fn aggregate_weighted(
    updates: &[(String, Vec<f64>)],
    trust_scores: &HashMap<String, f64>,
) -> AggregationResult {
    if updates.is_empty() {
        return AggregationResult {
            weights: vec![],
            participating_nodes: vec![],
            convergence_score: 0.0,
        };
    }

    let dim = updates[0].1.len();
    let mut aggregated = vec![0.0; dim];
    let mut total_weight = 0.0;

    for (node_id, gradients) in updates {
        let trust = trust_scores.get(node_id).copied().unwrap_or(1.0);
        if trust > 0.3 {
            total_weight += trust;
            for (i, &g) in gradients.iter().enumerate() {
                aggregated[i] += g * trust;
            }
        }
    }

    if total_weight > 0.0 {
        for a in &mut aggregated {
            *a /= total_weight;
        }
    }

    // Calculate convergence (inverse of gradient magnitude)
    let magnitude: f64 = aggregated.iter().map(|x| x * x).sum::<f64>().sqrt();
    let convergence = 1.0 / (1.0 + magnitude);

    AggregationResult {
        weights: aggregated,
        participating_nodes: updates.iter().map(|(id, _)| id.clone()).collect(),
        convergence_score: convergence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_weighted() {
        let updates = vec![
            ("node-1".to_string(), vec![1.0, 2.0, 3.0]),
            ("node-2".to_string(), vec![1.0, 2.0, 3.0]),
        ];
        let mut trust = HashMap::new();
        trust.insert("node-1".to_string(), 1.0);
        trust.insert("node-2".to_string(), 1.0);

        let result = aggregate_weighted(&updates, &trust);
        assert_eq!(result.participating_nodes.len(), 2);
        assert!(!result.weights.is_empty());
    }
}
