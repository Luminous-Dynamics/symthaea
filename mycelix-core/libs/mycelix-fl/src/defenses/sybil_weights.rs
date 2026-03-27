// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sybil weight computation via reputation tracking.
//!
//! Tracks per-node contributions, validations, and rejections to compute
//! reputation scores. Weight is reputation squared (quadratic), so
//! low-reputation nodes have near-zero influence on aggregation.
//!
//! This module also implements cluster-penalty weighting from cosine
//! similarity analysis, which can be combined with reputation weights.
//!
//! Reference: Fung et al., "The Limitations of Federated Learning in
//! Sybil Settings", NeurIPS 2020 Workshop.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::FlError;
use crate::types::{cosine_similarity, AggregationResult, Gradient};

use super::validate_gradients;

/// Per-node reputation record.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeReputation {
    /// Total number of gradient contributions.
    pub contributions: u64,
    /// Number of contributions that passed validation.
    pub successful_validations: u64,
    /// Number of contributions that were rejected.
    pub rejections: u64,
    /// Computed reputation score in [0.0, 1.0].
    pub reputation_score: f64,
}

impl NodeReputation {
    /// Recompute the reputation score from contribution history.
    ///
    /// Formula: `score = successful / (successful + 2 * rejections + 1)`.
    /// The `+1` in the denominator provides a mild prior that starts new
    /// nodes at moderate reputation rather than 1.0.
    fn recompute(&mut self) {
        let s = self.successful_validations as f64;
        let r = self.rejections as f64;
        self.reputation_score = s / (s + 2.0 * r + 1.0);
    }
}

/// Sybil weight computer based on reputation tracking.
///
/// Maintains a persistent reputation record per node. Weights are derived
/// from reputation scores via quadratic mapping, so low-reputation nodes
/// (likely Sybils or Byzantine) contribute almost nothing.
pub struct SybilWeightComputer {
    /// Per-node reputation tracking.
    reputation: HashMap<String, NodeReputation>,
    /// Cosine threshold for cluster detection penalty.
    cosine_threshold: f64,
}

impl SybilWeightComputer {
    /// Create a new SybilWeightComputer with default parameters.
    pub fn new() -> Self {
        Self {
            reputation: HashMap::new(),
            cosine_threshold: 0.95,
        }
    }

    /// Create with a custom cosine threshold for cluster detection.
    pub fn with_cosine_threshold(cosine_threshold: f64) -> Self {
        Self {
            reputation: HashMap::new(),
            cosine_threshold,
        }
    }

    /// Update reputation for a node after a validation decision.
    pub fn update_reputation(&mut self, node_id: &str, accepted: bool) {
        let rep = self.reputation.entry(node_id.to_string()).or_default();
        rep.contributions += 1;
        if accepted {
            rep.successful_validations += 1;
        } else {
            rep.rejections += 1;
        }
        rep.recompute();
    }

    /// Get the current reputation for a node (None if never seen).
    pub fn get_reputation(&self, node_id: &str) -> Option<&NodeReputation> {
        self.reputation.get(node_id)
    }

    /// Compute reputation-based weights for a set of nodes.
    ///
    /// Weight = reputation^2, normalized to sum to 1. Unknown nodes
    /// receive a small default reputation (0.1).
    pub fn compute_weights(&self, node_ids: &[String]) -> Vec<f64> {
        let default_rep = 0.1;

        let mut weights: Vec<f64> = node_ids
            .iter()
            .map(|id| {
                let score = self
                    .reputation
                    .get(id)
                    .map(|r| r.reputation_score)
                    .unwrap_or(default_rep);
                score * score
            })
            .collect();

        let total: f64 = weights.iter().sum();
        if total > 1e-12 {
            for w in &mut weights {
                *w /= total;
            }
        } else if !weights.is_empty() {
            let uniform = 1.0 / weights.len() as f64;
            for w in &mut weights {
                *w = uniform;
            }
        }

        weights
    }

    /// Compute cluster penalty based on cosine similarity in this round.
    ///
    /// Nodes in clusters of size k > 1 (cosine > threshold) receive a
    /// penalty of `1 / sqrt(k)`.
    pub fn compute_cluster_penalty(&self, gradients: &[Gradient]) -> Vec<f64> {
        let n = gradients.len();
        let mut penalty = vec![1.0; n];

        if n < 2 {
            return penalty;
        }

        // Compute pairwise cosine similarity.
        let mut sim = vec![0.0_f64; n * n];
        for i in 0..n {
            sim[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let cos = cosine_similarity(&gradients[i].values, &gradients[j].values);
                sim[i * n + j] = cos;
                sim[j * n + i] = cos;
            }
        }

        // Greedy single-linkage clustering.
        let mut assigned = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let mut cluster = vec![i];
            assigned[i] = true;
            for j in (i + 1)..n {
                if !assigned[j] && sim[i * n + j] > self.cosine_threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            clusters.push(cluster);
        }

        for cluster in &clusters {
            let k = cluster.len();
            if k > 1 {
                let p = 1.0 / (k as f64).sqrt();
                for &idx in cluster {
                    penalty[idx] = p;
                }
            }
        }

        penalty
    }

    /// Aggregate gradients using reputation weights combined with cluster penalty.
    pub fn weighted_aggregate(&self, gradients: &[Gradient]) -> Result<AggregationResult, FlError> {
        let dim = validate_gradients(gradients)?;

        let node_ids: Vec<String> = gradients.iter().map(|g| g.node_id.clone()).collect();

        let rep_weights = self.compute_weights(&node_ids);
        let cluster_penalty = self.compute_cluster_penalty(gradients);

        // Combined weights = reputation * cluster_penalty, re-normalized.
        let mut weights: Vec<f64> = rep_weights
            .iter()
            .zip(cluster_penalty.iter())
            .map(|(&r, &c)| r * c)
            .collect();

        let total: f64 = weights.iter().sum();
        if total > 1e-12 {
            for w in &mut weights {
                *w /= total;
            }
        }

        // Weighted aggregation.
        let mut agg = vec![0.0_f64; dim];
        for (g, &w) in gradients.iter().zip(weights.iter()) {
            for (a, &v) in agg.iter_mut().zip(g.values.iter()) {
                *a += w * v as f64;
            }
        }

        let mut included = Vec::new();
        let mut excluded = Vec::new();
        for (g, &w) in gradients.iter().zip(weights.iter()) {
            if w > 1e-12 {
                included.push(g.node_id.clone());
            } else {
                excluded.push(g.node_id.clone());
            }
        }

        let scores: Vec<(String, f64)> = node_ids
            .into_iter()
            .zip(weights.iter().copied())
            .collect();

        Ok(AggregationResult {
            gradient: agg.iter().map(|&v| v as f32).collect(),
            included_nodes: included,
            excluded_nodes: excluded,
            scores,
        })
    }

    /// Return the number of nodes with reputation records.
    pub fn num_tracked(&self) -> usize {
        self.reputation.len()
    }

    /// Clear all reputation history.
    pub fn reset(&mut self) {
        self.reputation.clear();
    }
}

impl Default for SybilWeightComputer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grad(id: &str, values: Vec<f32>) -> Gradient {
        Gradient { node_id: id.into(), values, round: 0 }
    }

    #[test]
    fn test_new_node_low_reputation() {
        let computer = SybilWeightComputer::new();
        let weights = computer.compute_weights(&["unknown".to_string()]);
        assert_eq!(weights.len(), 1);
        // Single node normalizes to 1.0 regardless.
        assert!((weights[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reputation_grows_with_acceptance() {
        let mut computer = SybilWeightComputer::new();
        for _ in 0..10 {
            computer.update_reputation("good_node", true);
        }
        let rep = computer.get_reputation("good_node").unwrap();
        assert!(
            rep.reputation_score > 0.8,
            "10 acceptances should yield high reputation, got {}",
            rep.reputation_score
        );
    }

    #[test]
    fn test_reputation_drops_with_rejections() {
        let mut computer = SybilWeightComputer::new();
        for _ in 0..5 {
            computer.update_reputation("bad_node", false);
        }
        let rep = computer.get_reputation("bad_node").unwrap();
        assert!(
            rep.reputation_score < 0.1,
            "5 rejections should yield low reputation, got {}",
            rep.reputation_score
        );
    }

    #[test]
    fn test_high_rep_gets_more_weight() {
        let mut computer = SybilWeightComputer::new();
        for _ in 0..10 {
            computer.update_reputation("good", true);
        }
        for _ in 0..10 {
            computer.update_reputation("bad", false);
        }
        let weights = computer.compute_weights(&["good".to_string(), "bad".to_string()]);
        assert!(
            weights[0] > weights[1],
            "good ({:.4}) should have more weight than bad ({:.4})",
            weights[0], weights[1]
        );
    }

    #[test]
    fn test_cluster_penalty_detects_identical_gradients() {
        let computer = SybilWeightComputer::new();
        let gradients = vec![
            grad("a", vec![1.0, 0.0, 0.0]),
            grad("b", vec![1.0, 0.0, 0.0]),
            grad("c", vec![0.0, 1.0, 0.0]),
        ];
        let penalty = computer.compute_cluster_penalty(&gradients);
        assert!(penalty[0] < 1.0, "clustered node should be penalized");
        assert!((penalty[0] - penalty[1]).abs() < 1e-6);
        assert!((penalty[2] - 1.0).abs() < 1e-6, "unique node gets no penalty");
    }

    #[test]
    fn test_weighted_aggregate_basic() {
        let mut computer = SybilWeightComputer::new();
        for _ in 0..10 {
            computer.update_reputation("high", true);
        }
        for _ in 0..10 {
            computer.update_reputation("low", false);
        }

        let gradients = vec![
            grad("high", vec![1.0, 0.0]),
            grad("low", vec![0.0, 1.0]),
        ];
        let result = computer.weighted_aggregate(&gradients).unwrap();
        assert!(result.gradient[0] > result.gradient[1]);
    }

    #[test]
    fn test_empty_gradients() {
        let computer = SybilWeightComputer::new();
        assert!(computer.weighted_aggregate(&[]).is_err());
    }

    #[test]
    fn test_quadratic_weighting() {
        let mut computer = SybilWeightComputer::new();
        computer.update_reputation("a", true);  // score = 1/(1+0+1) = 0.5
        computer.update_reputation("b", true);
        computer.update_reputation("b", true);  // score = 2/(2+0+1) = 0.667

        let rep_a = computer.get_reputation("a").unwrap().reputation_score;
        let rep_b = computer.get_reputation("b").unwrap().reputation_score;

        let weights = computer.compute_weights(&["a".to_string(), "b".to_string()]);

        let raw_a = rep_a * rep_a;
        let raw_b = rep_b * rep_b;
        let total = raw_a + raw_b;

        assert!((weights[0] - raw_a / total).abs() < 1e-6);
        assert!((weights[1] - raw_b / total).abs() < 1e-6);
    }
}
