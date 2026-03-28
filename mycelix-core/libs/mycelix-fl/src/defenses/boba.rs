// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! BOBA: Label-Skew Aware Byzantine-Robust Aggregation.
//!
//! Handles non-IID federated learning by detecting label distribution
//! differences via Jensen-Shannon divergence. Nodes whose average JS
//! similarity to the group falls below a threshold are excluded from
//! aggregation. The remaining nodes are weighted by their similarity.
//!
//! Reference: Karimireddy et al., "Byzantine-Robust Learning on
//! Heterogeneous Datasets via Bucketing", ICML 2022.

use crate::error::FlError;
use crate::types::{AggregationResult, Gradient};

use super::validate_gradients;

/// BOBA label-skew aware aggregation.
///
/// Stateless — all information is derived from the gradients and their
/// accompanying label-distribution histograms in a single round.
pub struct Boba {
    /// Minimum average JS similarity for a node to be included.
    similarity_threshold: f64,
    /// Whether to normalize final weights to sum to 1.
    normalize_weights: bool,
}

impl Boba {
    /// Create a new BOBA instance with default parameters.
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.5,
            normalize_weights: true,
        }
    }

    /// Create a BOBA instance with custom threshold.
    pub fn with_threshold(similarity_threshold: f64) -> Self {
        Self {
            similarity_threshold,
            normalize_weights: true,
        }
    }

    /// Aggregate gradients using label-distribution-aware weighting.
    ///
    /// Each node must provide a label histogram (`class_histograms`) that
    /// describes the distribution of labels in its local dataset. Nodes
    /// whose average JS similarity to all others falls below
    /// `similarity_threshold` are excluded.
    ///
    /// If `class_histograms` is `None`, falls back to simple mean.
    pub fn aggregate(
        &self,
        gradients: &[Gradient],
        class_histograms: Option<&[Vec<f64>]>,
    ) -> Result<AggregationResult, FlError> {
        let dim = validate_gradients(gradients)?;

        let histograms = match class_histograms {
            Some(h) => h,
            None => return Ok(simple_mean(gradients, dim)),
        };

        if histograms.len() != gradients.len() {
            return Err(FlError::InvalidInput(format!(
                "histogram count ({}) does not match gradient count ({})",
                histograms.len(),
                gradients.len()
            )));
        }

        let n = gradients.len();

        // Compute pairwise JS similarity matrix.
        let mut sim_matrix = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                sim_matrix[i * n + j] = js_similarity(&histograms[i], &histograms[j]);
            }
        }

        // Average similarity per node, threshold to determine inclusion.
        let mut weights = vec![0.0_f64; n];
        for i in 0..n {
            let avg: f64 = sim_matrix[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64;
            weights[i] = if avg >= self.similarity_threshold { avg } else { 0.0 };
        }

        // If all rejected, fall back to simple mean.
        let total: f64 = weights.iter().sum();
        if total < 1e-12 {
            return Ok(simple_mean(gradients, dim));
        }
        if self.normalize_weights {
            for w in &mut weights {
                *w /= total;
            }
        }

        // Partition into included / excluded.
        let mut included = Vec::new();
        let mut excluded = Vec::new();
        for (g, &w) in gradients.iter().zip(weights.iter()) {
            if w > 0.0 {
                included.push(g.node_id.clone());
            } else {
                excluded.push(g.node_id.clone());
            }
        }

        // Weighted aggregation.
        let mut agg = vec![0.0_f64; dim];
        for (g, &w) in gradients.iter().zip(weights.iter()) {
            for (a, &v) in agg.iter_mut().zip(g.values.iter()) {
                *a += w * v as f64;
            }
        }

        let scores: Vec<(String, f64)> = gradients
            .iter()
            .zip(weights.iter())
            .map(|(g, &w)| (g.node_id.clone(), w))
            .collect();

        Ok(AggregationResult {
            gradient: agg.iter().map(|&v| v as f32).collect(),
            included_nodes: included,
            excluded_nodes: excluded,
            scores,
        })
    }
}

impl Default for Boba {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple mean aggregation (fallback when no histograms are available).
fn simple_mean(gradients: &[Gradient], dim: usize) -> AggregationResult {
    let n = gradients.len() as f64;
    let mut agg = vec![0.0_f64; dim];
    for g in gradients {
        for (a, &v) in agg.iter_mut().zip(g.values.iter()) {
            *a += v as f64;
        }
    }

    let included: Vec<String> = gradients.iter().map(|g| g.node_id.clone()).collect();
    let weight = 1.0 / n;
    let scores: Vec<(String, f64)> = gradients
        .iter()
        .map(|g| (g.node_id.clone(), weight))
        .collect();

    AggregationResult {
        gradient: agg.iter().map(|&v| (v / n) as f32).collect(),
        included_nodes: included,
        excluded_nodes: Vec::new(),
        scores,
    }
}

/// Jensen-Shannon similarity: `1.0 - JSD(p, q)`.
fn js_similarity(p: &[f64], q: &[f64]) -> f64 {
    1.0 - jensen_shannon_divergence(p, q)
}

/// Jensen-Shannon divergence between two probability distributions.
///
/// `JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)` where `M = 0.5 * (P + Q)`.
/// Uses base-2 logarithm so the result is bounded in [0, 1].
///
/// Input distributions are normalized internally (need not sum to 1).
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    let len = p.len().min(q.len());
    if len == 0 {
        return 0.0;
    }

    let p_norm = normalize_distribution(p);
    let q_norm = normalize_distribution(q);

    let m: Vec<f64> = p_norm
        .iter()
        .zip(q_norm.iter())
        .map(|(&pi, &qi)| 0.5 * (pi + qi))
        .collect();

    0.5 * kl_divergence(&p_norm, &m) + 0.5 * kl_divergence(&q_norm, &m)
}

/// KL divergence: `KL(P || Q) = sum(p_i * log2(p_i / q_i))`.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    const EPS: f64 = 1e-12;
    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > EPS {
            kl += pi * (pi / (qi + EPS)).log2();
        }
    }
    kl.max(0.0)
}

/// Normalize a slice to a probability distribution (sums to 1).
fn normalize_distribution(d: &[f64]) -> Vec<f64> {
    let sum: f64 = d.iter().sum();
    if sum < 1e-12 {
        let n = d.len() as f64;
        return vec![1.0 / n; d.len()];
    }
    d.iter().map(|&x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grad(id: &str, values: Vec<f32>) -> Gradient {
        Gradient { node_id: id.into(), values, round: 0 }
    }

    #[test]
    fn test_jsd_identical_distributions() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let jsd = jensen_shannon_divergence(&p, &p);
        assert!(jsd.abs() < 1e-10, "JSD of identical distributions should be 0");
    }

    #[test]
    fn test_jsd_disjoint_distributions() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let jsd = jensen_shannon_divergence(&p, &q);
        assert!(
            (jsd - 1.0).abs() < 1e-6,
            "JSD of disjoint distributions should be 1.0, got {jsd}"
        );
    }

    #[test]
    fn test_jsd_symmetry() {
        let p = vec![0.7, 0.2, 0.1];
        let q = vec![0.1, 0.3, 0.6];
        assert!((jensen_shannon_divergence(&p, &q) - jensen_shannon_divergence(&q, &p)).abs() < 1e-10);
    }

    #[test]
    fn test_boba_similar_distributions_all_included() {
        let boba = Boba::with_threshold(0.3);
        let gradients = vec![
            grad("a", vec![1.0, 2.0, 3.0]),
            grad("b", vec![1.1, 1.9, 3.1]),
            grad("c", vec![0.9, 2.1, 2.9]),
        ];
        let histograms = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.45, 0.35, 0.20],
            vec![0.48, 0.32, 0.20],
        ];
        let result = boba.aggregate(&gradients, Some(&histograms)).unwrap();
        assert_eq!(result.included_nodes.len(), 3);
        assert!(result.excluded_nodes.is_empty());
    }

    #[test]
    fn test_boba_outlier_distribution_downweighted() {
        let boba = Boba::with_threshold(0.7);
        let gradients = vec![
            grad("a", vec![1.0, 2.0]),
            grad("b", vec![1.1, 1.9]),
            grad("outlier", vec![5.0, -5.0]),
        ];
        let histograms = vec![
            vec![0.5, 0.5],
            vec![0.45, 0.55],
            vec![1.0, 0.0],
        ];
        let result = boba.aggregate(&gradients, Some(&histograms)).unwrap();

        let outlier_score = result.scores.iter().find(|(id, _)| id == "outlier").map(|(_, w)| *w).unwrap_or(1.0);
        let a_score = result.scores.iter().find(|(id, _)| id == "a").map(|(_, w)| *w).unwrap_or(0.0);
        assert!(outlier_score <= a_score, "outlier should have <= weight than similar node");
    }

    #[test]
    fn test_boba_no_histograms_falls_back_to_mean() {
        let boba = Boba::new();
        let gradients = vec![grad("a", vec![2.0, 4.0]), grad("b", vec![4.0, 6.0])];
        let result = boba.aggregate(&gradients, None).unwrap();
        assert!((result.gradient[0] - 3.0).abs() < 1e-6);
        assert!((result.gradient[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_boba_empty_gradients() {
        let boba = Boba::new();
        assert!(boba.aggregate(&[], None).is_err());
    }
}
