// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! FoolsGold: Anti-Sybil defense via historical gradient cosine similarity.
//!
//! Detects Sybil attacks by tracking per-node gradient history and computing
//! pairwise cosine similarities. Nodes that consistently submit similar
//! gradients (likely Sybils sharing training data) are down-weighted.
//!
//! Reference: Fung et al., "The Limitations of Federated Learning in Sybil
//! Settings", NeurIPS 2020 Workshop.

use std::collections::HashMap;

use crate::error::FlError;
use crate::types::{cosine_similarity, AggregationResult, DefenseConfig, Gradient};

use super::validate_gradients;

/// FoolsGold anti-Sybil defense.
///
/// Maintains a running history of gradient submissions per node and computes
/// pairwise cosine similarities to detect colluding Sybil nodes. Nodes with
/// high historical similarity to others receive reduced aggregation weight.
pub struct FoolsGold {
    /// Historical gradient accumulator per node_id (running EMA of gradients).
    history: HashMap<String, Vec<f32>>,
    /// Number of rounds tracked per node (for EMA decay).
    round_counts: HashMap<String, usize>,
    /// Maximum history window (older contributions decay via EMA).
    history_window: usize,
    /// Controls severity of down-weighting (higher = more aggressive).
    alpha: f64,
}

impl FoolsGold {
    /// Create a new FoolsGold instance with default parameters.
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
            round_counts: HashMap::new(),
            history_window: 10,
            alpha: 1.0,
        }
    }

    /// Create a FoolsGold instance from a shared defense configuration.
    pub fn from_config(config: &DefenseConfig) -> Self {
        Self {
            history: HashMap::new(),
            round_counts: HashMap::new(),
            history_window: config.history_window,
            alpha: 1.0,
        }
    }

    /// Update the gradient history for a node via exponential moving average.
    fn update_history(&mut self, node_id: &str, gradient: &[f32]) {
        let count = self.round_counts.entry(node_id.to_string()).or_insert(0);
        *count += 1;

        let decay = if *count > self.history_window {
            1.0_f32 / self.history_window as f32
        } else {
            1.0_f32 / *count as f32
        };

        let entry = self
            .history
            .entry(node_id.to_string())
            .or_insert_with(|| vec![0.0; gradient.len()]);

        if entry.len() == gradient.len() {
            for (h, &g) in entry.iter_mut().zip(gradient.iter()) {
                *h = (1.0 - decay) * *h + decay * g;
            }
        } else {
            // Dimension changed — reset history for this node.
            *entry = gradient.to_vec();
        }
    }

    /// Compute pairwise cosine similarity matrix from accumulated histories.
    fn similarity_matrix(&self, node_ids: &[String]) -> Vec<f64> {
        let n = node_ids.len();
        let mut sim = vec![0.0_f64; n * n];

        for i in 0..n {
            sim[i * n + i] = 1.0;
        }

        for i in 0..n {
            let hist_i = match self.history.get(&node_ids[i]) {
                Some(h) => h,
                None => continue,
            };
            for j in (i + 1)..n {
                let hist_j = match self.history.get(&node_ids[j]) {
                    Some(h) => h,
                    None => continue,
                };
                let cos = cosine_similarity(hist_i, hist_j);
                sim[i * n + j] = cos;
                sim[j * n + i] = cos;
            }
        }

        sim
    }

    /// Compute FoolsGold weights: inversely proportional to max similarity.
    fn compute_weights(&self, node_ids: &[String]) -> Vec<f64> {
        let n = node_ids.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![1.0];
        }

        let sim = self.similarity_matrix(node_ids);

        let mut weights = vec![0.0_f64; n];
        for i in 0..n {
            let mut max_sim = 0.0_f64;
            for j in 0..n {
                if i != j {
                    max_sim = max_sim.max(sim[i * n + j]);
                }
            }
            let importance = 1.0 / (1.0 + max_sim);
            weights[i] = importance.powf(self.alpha);
        }

        // Normalize.
        let total: f64 = weights.iter().sum();
        if total > 1e-12 {
            for w in &mut weights {
                *w /= total;
            }
        }

        weights
    }

    /// Update history and compute Sybil-weighted aggregation.
    ///
    /// Each round, the gradient history is updated for every participating node,
    /// and weights are recomputed based on the full similarity matrix.
    pub fn aggregate(&mut self, gradients: &[Gradient]) -> Result<AggregationResult, FlError> {
        let dim = validate_gradients(gradients)?;

        // Update history for each node.
        for g in gradients {
            self.update_history(&g.node_id, &g.values);
        }

        // Compute weights.
        let node_ids: Vec<String> = gradients.iter().map(|g| g.node_id.clone()).collect();
        let weights = self.compute_weights(&node_ids);

        // Weighted aggregation.
        let mut agg = vec![0.0_f64; dim];
        for (g, &w) in gradients.iter().zip(weights.iter()) {
            for (a, &v) in agg.iter_mut().zip(g.values.iter()) {
                *a += w * v as f64;
            }
        }

        let scores: Vec<(String, f64)> = node_ids
            .iter()
            .zip(weights.iter())
            .map(|(id, &w)| (id.clone(), w))
            .collect();

        Ok(AggregationResult {
            gradient: agg.iter().map(|&v| v as f32).collect(),
            included_nodes: node_ids,
            excluded_nodes: Vec::new(),
            scores,
        })
    }

    /// Return the number of nodes currently tracked.
    pub fn num_tracked(&self) -> usize {
        self.history.len()
    }

    /// Clear all accumulated history.
    pub fn reset(&mut self) {
        self.history.clear();
        self.round_counts.clear();
    }
}

impl Default for FoolsGold {
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
    fn test_sybils_get_lower_weight() {
        let mut fg = FoolsGold::new();

        // Run multiple rounds so history accumulates.
        for _ in 0..5 {
            let gradients = vec![
                grad("honest_0", vec![1.0, 0.0, 0.0, 0.5]),
                grad("honest_1", vec![0.0, 1.0, 0.0, -0.3]),
                grad("honest_2", vec![0.0, 0.0, 1.0, 0.2]),
                grad("sybil_0", vec![0.5, 0.5, 0.0, 0.1]),
                grad("sybil_1", vec![0.5, 0.5, 0.0, 0.1]),
            ];
            let _ = fg.aggregate(&gradients).unwrap();
        }

        // Final round — check weights.
        let gradients = vec![
            grad("honest_0", vec![1.0, 0.0, 0.0, 0.5]),
            grad("honest_1", vec![0.0, 1.0, 0.0, -0.3]),
            grad("honest_2", vec![0.0, 0.0, 1.0, 0.2]),
            grad("sybil_0", vec![0.5, 0.5, 0.0, 0.1]),
            grad("sybil_1", vec![0.5, 0.5, 0.0, 0.1]),
        ];
        let result = fg.aggregate(&gradients).unwrap();

        let sybil_weight: f64 = result
            .scores
            .iter()
            .filter(|(id, _)| id.starts_with("sybil"))
            .map(|(_, w)| w)
            .sum();
        let honest_weight: f64 = result
            .scores
            .iter()
            .filter(|(id, _)| id.starts_with("honest"))
            .map(|(_, w)| w)
            .sum();

        assert!(
            sybil_weight < honest_weight,
            "Sybils ({sybil_weight:.4}) should have less total weight than honest ({honest_weight:.4})"
        );
    }

    #[test]
    fn test_empty_gradients_returns_error() {
        let mut fg = FoolsGold::new();
        assert!(fg.aggregate(&[]).is_err());
    }

    #[test]
    fn test_single_node_gets_full_weight() {
        let mut fg = FoolsGold::new();
        let gradients = vec![grad("only", vec![1.0, 2.0, 3.0])];
        let result = fg.aggregate(&gradients).unwrap();
        assert!((result.scores[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut fg = FoolsGold::new();
        let gradients = vec![
            grad("a", vec![1.0, 2.0]),
            grad("b", vec![1.0, 2.0, 3.0]),
        ];
        assert!(fg.aggregate(&gradients).is_err());
    }

    #[test]
    fn test_all_nodes_included() {
        let mut fg = FoolsGold::new();
        let gradients = vec![grad("a", vec![1.0, 2.0]), grad("b", vec![3.0, 4.0])];
        let result = fg.aggregate(&gradients).unwrap();
        assert_eq!(result.included_nodes.len(), 2);
        assert!(result.excluded_nodes.is_empty());
    }
}
