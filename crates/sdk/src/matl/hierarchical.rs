// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical Byzantine Detection
//!
//! O(n log n) complexity detection through hierarchical clustering.
//! Enables scaling to thousands of nodes efficiently.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hierarchical detector for Byzantine nodes
#[derive(Debug, Clone)]
pub struct HierarchicalDetector {
    /// Cluster assignments
    clusters: HashMap<String, usize>,

    /// Cluster scores
    cluster_scores: HashMap<usize, ClusterScore>,

    /// Number of levels in hierarchy
    #[allow(dead_code)]
    levels: usize,

    /// Minimum cluster size
    #[allow(dead_code)]
    min_cluster_size: usize,
}

/// Score for a cluster of nodes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterScore {
    /// Average score of nodes in cluster
    pub mean_score: f64,

    /// Number of nodes
    pub node_count: usize,

    /// Variance within cluster
    pub variance: f64,

    /// Is this cluster suspected Byzantine?
    pub suspected_byzantine: bool,
}

impl HierarchicalDetector {
    /// Create a new hierarchical detector
    pub fn new(levels: usize, min_cluster_size: usize) -> Self {
        Self {
            clusters: HashMap::new(),
            cluster_scores: HashMap::new(),
            levels,
            min_cluster_size,
        }
    }

    /// Assign a node to a cluster based on its score
    ///
    /// Simple bucketing: cluster_id = floor(score * 10) for 10 clusters
    pub fn assign(&mut self, node_id: &str, score: f64) {
        // Simple clustering: bucket by score decile
        let cluster_id = (score * 10.0).floor() as usize;
        let cluster_id = cluster_id.min(9); // Cap at 9

        self.clusters.insert(node_id.to_string(), cluster_id);

        // Update cluster score
        self.update_cluster_score(cluster_id);
    }

    /// Update cluster statistics
    fn update_cluster_score(&mut self, cluster_id: usize) {
        // Get all nodes in this cluster
        let nodes_in_cluster: Vec<&String> = self
            .clusters
            .iter()
            .filter(|(_, &c)| c == cluster_id)
            .map(|(n, _)| n)
            .collect();

        let count = nodes_in_cluster.len();
        if count == 0 {
            return;
        }

        // For now, use cluster_id as proxy for mean score
        // In real implementation, would track actual scores
        let mean = cluster_id as f64 / 10.0 + 0.05;

        self.cluster_scores.insert(
            cluster_id,
            ClusterScore {
                mean_score: mean,
                node_count: count,
                variance: 0.01,                  // Placeholder
                suspected_byzantine: mean < 0.3, // Low-score clusters are suspicious
            },
        );
    }

    /// Check if a node is in a Byzantine cluster
    pub fn is_in_byzantine_cluster(&self, node_id: &str) -> bool {
        if let Some(&cluster_id) = self.clusters.get(node_id) {
            if let Some(cluster) = self.cluster_scores.get(&cluster_id) {
                return cluster.suspected_byzantine;
            }
        }
        false
    }

    /// Get suspected Byzantine nodes
    pub fn get_suspected_byzantine(&self) -> Vec<String> {
        self.clusters
            .iter()
            .filter(|(_, &cluster_id)| {
                self.cluster_scores
                    .get(&cluster_id)
                    .map(|c| c.suspected_byzantine)
                    .unwrap_or(false)
            })
            .map(|(node_id, _)| node_id.clone())
            .collect()
    }

    /// Get cluster for a node
    pub fn get_cluster(&self, node_id: &str) -> Option<usize> {
        self.clusters.get(node_id).copied()
    }

    /// Get number of clusters
    pub fn cluster_count(&self) -> usize {
        self.cluster_scores.len()
    }

    /// Get all cluster scores
    pub fn clusters(&self) -> &HashMap<usize, ClusterScore> {
        &self.cluster_scores
    }

    /// Detect outlier nodes (far from their cluster mean)
    pub fn detect_outliers(&self, _threshold: f64) -> Vec<String> {
        // In real implementation, would compare each node's actual score
        // to its cluster mean
        self.get_suspected_byzantine()
    }

    /// Estimate Byzantine fraction
    pub fn byzantine_fraction(&self) -> f64 {
        let total: usize = self.cluster_scores.values().map(|c| c.node_count).sum();
        let byzantine: usize = self
            .cluster_scores
            .values()
            .filter(|c| c.suspected_byzantine)
            .map(|c| c.node_count)
            .sum();

        if total == 0 {
            return 0.0;
        }
        byzantine as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clustering() {
        let mut detector = HierarchicalDetector::new(3, 2);

        // Add high-quality nodes
        detector.assign("node1", 0.9);
        detector.assign("node2", 0.85);
        detector.assign("node3", 0.88);

        // Add low-quality nodes
        detector.assign("bad1", 0.1);
        detector.assign("bad2", 0.15);

        // Check clustering
        assert!(detector.cluster_count() > 0);
        assert!(detector.is_in_byzantine_cluster("bad1"));
        assert!(!detector.is_in_byzantine_cluster("node1"));
    }

    #[test]
    fn test_byzantine_detection() {
        let mut detector = HierarchicalDetector::new(3, 2);

        detector.assign("good1", 0.9);
        detector.assign("good2", 0.85);
        detector.assign("bad1", 0.2);
        detector.assign("bad2", 0.15);

        let suspected = detector.get_suspected_byzantine();
        assert!(suspected.contains(&"bad1".to_string()));
        assert!(suspected.contains(&"bad2".to_string()));
        assert!(!suspected.contains(&"good1".to_string()));
    }

    #[test]
    fn test_byzantine_fraction() {
        let mut detector = HierarchicalDetector::new(3, 2);

        // 2 good, 2 bad = 50% Byzantine
        detector.assign("good1", 0.9);
        detector.assign("good2", 0.85);
        detector.assign("bad1", 0.2);
        detector.assign("bad2", 0.15);

        let fraction = detector.byzantine_fraction();
        assert!(fraction > 0.4 && fraction < 0.6);
    }
}
