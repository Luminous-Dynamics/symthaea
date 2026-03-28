// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Core types for federated learning aggregation.

use serde::{Deserialize, Serialize};

/// A gradient submission from a single node in a federated learning round.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gradient {
    /// Unique identifier of the submitting node.
    pub node_id: String,
    /// The gradient vector (flattened model update).
    pub values: Vec<f32>,
    /// The FL round this gradient belongs to.
    pub round: u64,
}

impl Gradient {
    /// Create a new gradient from a node ID, values, and round number.
    pub fn new(node_id: impl Into<String>, values: Vec<f32>, round: u64) -> Self {
        Self {
            node_id: node_id.into(),
            values,
            round,
        }
    }

    /// Dimensionality of this gradient.
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// Result of an aggregation operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregationResult {
    /// The aggregated gradient vector.
    pub gradient: Vec<f32>,
    /// Node IDs that were included in the aggregation.
    pub included_nodes: Vec<String>,
    /// Node IDs that were excluded (filtered out by the defense).
    pub excluded_nodes: Vec<String>,
    /// Per-node quality scores (node_id, score).
    pub scores: Vec<(String, f64)>,
}

/// Configuration parameters shared across defenses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefenseConfig {
    /// Fraction to trim from each end for trimmed mean (default 0.1).
    pub trim_ratio: f32,
    /// Expected number of Byzantine nodes (for Krum-family algorithms).
    pub num_byzantine: usize,
    /// Maximum iterations for Weiszfeld algorithm in RFA (default 100).
    pub weiszfeld_max_iter: usize,
    /// Convergence tolerance for Weiszfeld algorithm (default 1e-6).
    pub weiszfeld_tol: f64,
    /// Maximum number of historical rounds to retain.
    pub history_window: usize,
    /// Cosine similarity threshold for Sybil/cluster detection.
    pub cosine_threshold: f64,
    /// False positive rate target for conformal methods.
    pub alpha: f64,
    /// Number of label classes (for label-distribution-aware defenses).
    pub num_classes: usize,
}

impl Default for DefenseConfig {
    fn default() -> Self {
        Self {
            trim_ratio: 0.1,
            num_byzantine: 0,
            weiszfeld_max_iter: 100,
            weiszfeld_tol: 1e-6,
            history_window: 10,
            cosine_threshold: 0.95,
            alpha: 0.10,
            num_classes: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

/// L2 (Euclidean) norm of a vector.
pub fn l2_norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

/// L2 (Euclidean) distance between two vectors.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "l2_distance: length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Element-wise mean of multiple vectors. Returns an empty vec if the input is empty.
pub fn elementwise_mean(vectors: &[&[f32]]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let n = vectors.len() as f64;
    let mut result = vec![0.0f64; dim];
    for v in vectors {
        for (i, &x) in v.iter().enumerate() {
            result[i] += x as f64;
        }
    }
    result.iter().map(|&x| (x / n) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = [0.0, 0.0];
        let b = [1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_l2_norm() {
        let v = [3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_l2_distance() {
        let a = [1.0, 0.0];
        let b = [0.0, 0.0];
        assert!((l2_distance(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_elementwise_mean() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let mean = elementwise_mean(&[&a[..], &b[..]]);
        assert!((mean[0] - 2.0).abs() < 1e-6);
        assert!((mean[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_mean_empty() {
        let mean = elementwise_mean(&[]);
        assert!(mean.is_empty());
    }

    #[test]
    fn test_defense_config_default() {
        let cfg = DefenseConfig::default();
        assert!((cfg.trim_ratio - 0.1).abs() < 1e-6);
        assert_eq!(cfg.num_byzantine, 0);
        assert_eq!(cfg.weiszfeld_max_iter, 100);
        assert!((cfg.weiszfeld_tol - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_serde_roundtrip() {
        let g = Gradient {
            values: vec![1.0, 2.0, 3.0],
            node_id: "node-a".into(),
            round: 42,
        };
        let json = serde_json::to_string(&g).unwrap();
        let g2: Gradient = serde_json::from_str(&json).unwrap();
        assert_eq!(g.values, g2.values);
        assert_eq!(g.node_id, g2.node_id);
        assert_eq!(g.round, g2.round);
    }
}
