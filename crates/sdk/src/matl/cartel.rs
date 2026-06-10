// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cartel Detection
//!
//! Detects coordinated groups of malicious actors attempting to
//! manipulate the federated learning system.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Cartel detector for identifying coordinated Byzantine actors
#[derive(Debug, Clone)]
pub struct CartelDetector {
    /// Similarity matrix between nodes
    similarities: HashMap<(String, String), f64>,

    /// Known cartels
    cartels: Vec<Cartel>,

    /// Similarity threshold for cartel membership
    threshold: f64,

    /// Minimum cartel size
    min_size: usize,
}

/// A detected cartel (group of coordinated actors)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cartel {
    /// Unique identifier
    pub id: String,

    /// Member node IDs
    pub members: HashSet<String>,

    /// Confidence score [0.0, 1.0]
    pub confidence: f64,

    /// Average similarity between members
    pub avg_similarity: f64,

    /// Detected timestamp
    pub detected_at: u64,
}

impl CartelDetector {
    /// Create a new cartel detector
    pub fn new(threshold: f64, min_size: usize) -> Self {
        Self {
            similarities: HashMap::new(),
            cartels: Vec::new(),
            threshold,
            min_size,
        }
    }

    /// Record similarity between two nodes
    ///
    /// Similarity is based on gradient cosine similarity or behavior patterns
    pub fn record_similarity(&mut self, node_a: &str, node_b: &str, similarity: f64) {
        // Store in canonical order (alphabetically smaller first)
        let key = if node_a < node_b {
            (node_a.to_string(), node_b.to_string())
        } else {
            (node_b.to_string(), node_a.to_string())
        };

        self.similarities.insert(key, similarity.clamp(-1.0, 1.0));
    }

    /// Get similarity between two nodes
    pub fn get_similarity(&self, node_a: &str, node_b: &str) -> Option<f64> {
        // Check canonical order without allocating
        if node_a < node_b {
            self.similarities
                .get(&(node_a.to_string(), node_b.to_string()))
                .copied()
        } else {
            self.similarities
                .get(&(node_b.to_string(), node_a.to_string()))
                .copied()
        }
    }

    /// Run cartel detection algorithm
    ///
    /// Uses agglomerative clustering to find highly similar groups
    pub fn detect(&mut self) -> Vec<Cartel> {
        // Collect unique nodes efficiently using iterators
        let mut nodes: HashSet<&str> = HashSet::with_capacity(self.similarities.len() * 2);
        for (a, b) in self.similarities.keys() {
            nodes.insert(a.as_str());
            nodes.insert(b.as_str());
        }

        // Build adjacency list of highly similar nodes using references
        let mut adjacency: HashMap<&str, HashSet<&str>> = HashMap::with_capacity(nodes.len());

        for ((a, b), sim) in &self.similarities {
            if *sim >= self.threshold {
                adjacency.entry(a.as_str()).or_default().insert(b.as_str());
                adjacency.entry(b.as_str()).or_default().insert(a.as_str());
            }
        }

        // Find connected components (potential cartels)
        let mut visited: HashSet<&str> = HashSet::with_capacity(nodes.len());
        let mut cartels = Vec::new();
        let mut stack: Vec<&str> = Vec::with_capacity(nodes.len());

        for node in nodes {
            if visited.contains(node) {
                continue;
            }

            let mut component: HashSet<String> = HashSet::new();
            stack.clear();
            stack.push(node);

            while let Some(current) = stack.pop() {
                if visited.contains(current) {
                    continue;
                }
                visited.insert(current);
                component.insert(current.to_string());

                if let Some(neighbors) = adjacency.get(current) {
                    for neighbor in neighbors {
                        if !visited.contains(*neighbor) {
                            stack.push(*neighbor);
                        }
                    }
                }
            }

            // Only consider as cartel if large enough
            if component.len() >= self.min_size {
                let avg_sim = self.average_similarity(&component);
                let confidence = self.calculate_confidence(&component, avg_sim);

                cartels.push(Cartel {
                    id: format!("cartel_{}", cartels.len()),
                    members: component,
                    confidence,
                    avg_similarity: avg_sim,
                    detected_at: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                });
            }
        }

        self.cartels = cartels;
        self.cartels.clone()
    }

    /// Calculate average similarity within a group
    fn average_similarity(&self, members: &HashSet<String>) -> f64 {
        let member_vec: Vec<&String> = members.iter().collect();
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..member_vec.len() {
            for j in (i + 1)..member_vec.len() {
                if let Some(sim) = self.get_similarity(member_vec[i], member_vec[j]) {
                    sum += sim;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Calculate confidence score for a potential cartel
    fn calculate_confidence(&self, members: &HashSet<String>, avg_sim: f64) -> f64 {
        // Confidence based on:
        // 1. Size (larger groups are more suspicious if coordinated)
        // 2. Average similarity (higher = more coordinated)
        let size_factor = (members.len() as f64 / 10.0).min(1.0);
        let sim_factor = avg_sim;

        (size_factor * 0.3 + sim_factor * 0.7).min(1.0)
    }

    /// Check if a node is in any detected cartel
    pub fn is_cartel_member(&self, node_id: &str) -> bool {
        self.cartels.iter().any(|c| c.members.contains(node_id))
    }

    /// Get cartel for a node
    pub fn get_cartel(&self, node_id: &str) -> Option<&Cartel> {
        self.cartels.iter().find(|c| c.members.contains(node_id))
    }

    /// Get all detected cartels
    pub fn get_cartels(&self) -> &[Cartel] {
        &self.cartels
    }

    /// Get all cartel members
    pub fn all_cartel_members(&self) -> HashSet<String> {
        self.cartels
            .iter()
            .flat_map(|c| c.members.iter().cloned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_similarity() {
        let mut detector = CartelDetector::new(0.8, 2);
        detector.record_similarity("a", "b", 0.95);

        assert_eq!(detector.get_similarity("a", "b"), Some(0.95));
        assert_eq!(detector.get_similarity("b", "a"), Some(0.95));
    }

    #[test]
    fn test_detect_cartel() {
        let mut detector = CartelDetector::new(0.8, 2);

        // Create a group of similar nodes
        detector.record_similarity("bad1", "bad2", 0.95);
        detector.record_similarity("bad2", "bad3", 0.92);
        detector.record_similarity("bad1", "bad3", 0.90);

        // Add some dissimilar nodes
        detector.record_similarity("good1", "good2", 0.3);
        detector.record_similarity("good1", "bad1", 0.1);

        let cartels = detector.detect();

        // Should detect one cartel with bad1, bad2, bad3
        assert!(!cartels.is_empty());
        assert!(detector.is_cartel_member("bad1"));
        assert!(detector.is_cartel_member("bad2"));
        assert!(!detector.is_cartel_member("good1"));
    }

    #[test]
    fn test_no_cartel_below_threshold() {
        let mut detector = CartelDetector::new(0.8, 2);

        // Only weak similarities
        detector.record_similarity("a", "b", 0.5);
        detector.record_similarity("b", "c", 0.4);

        let cartels = detector.detect();
        assert!(cartels.is_empty());
    }
}
