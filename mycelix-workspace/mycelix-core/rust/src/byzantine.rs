// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Byzantine Fault Detection Module
//!
//! Implements detection of malicious or faulty nodes in federated learning

use std::collections::HashMap;

/// Result of Byzantine analysis
#[derive(Debug, Clone)]
pub struct ByzantineAnalysis {
    pub is_byzantine: bool,
    pub deviation_score: f64,
    pub new_trust_score: f64,
}

/// Byzantine detector state
pub struct ByzantineDetector {
    threshold: f64,
    trust_scores: HashMap<String, f64>,
    history: HashMap<String, Vec<f64>>,
}

impl ByzantineDetector {
    /// Create a new Byzantine detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            trust_scores: HashMap::new(),
            history: HashMap::new(),
        }
    }

    /// Analyze if an update is potentially Byzantine
    pub fn analyze(
        &mut self,
        node_id: &str,
        gradients: &[f64],
        all_gradients: &[Vec<f64>],
    ) -> ByzantineAnalysis {
        if all_gradients.len() < 3 {
            return ByzantineAnalysis {
                is_byzantine: false,
                deviation_score: 0.0,
                new_trust_score: 1.0,
            };
        }

        // Calculate median gradient
        let dim = gradients.len();
        let mut median = vec![0.0; dim];
        for i in 0..dim {
            let mut values: Vec<f64> = all_gradients.iter().map(|g| g[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            median[i] = if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            };
        }

        // Calculate deviation from median
        let mut deviation_sq = 0.0;
        let mut median_norm_sq = 0.0;
        for i in 0..dim {
            deviation_sq += (gradients[i] - median[i]).powi(2);
            median_norm_sq += median[i].powi(2);
        }
        let deviation = deviation_sq.sqrt() / (median_norm_sq.sqrt() + 1e-8);

        // Update trust score
        let current_trust = self.trust_scores.get(node_id).copied().unwrap_or(1.0);
        let (new_trust, is_byzantine) = if deviation > 2.0 {
            let trust = (current_trust - 0.1).max(0.0);
            (trust, trust < self.threshold)
        } else {
            let trust = (current_trust + 0.01).min(1.0);
            (trust, false)
        };

        self.trust_scores.insert(node_id.to_string(), new_trust);

        // Track history
        self.history
            .entry(node_id.to_string())
            .or_insert_with(Vec::new)
            .push(deviation);

        ByzantineAnalysis {
            is_byzantine,
            deviation_score: deviation,
            new_trust_score: new_trust,
        }
    }

    /// Get trust score for a node
    pub fn get_trust_score(&self, node_id: &str) -> f64 {
        self.trust_scores.get(node_id).copied().unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byzantine_detector() {
        let mut detector = ByzantineDetector::new(0.33);

        let normal_gradients = vec![1.0, 2.0, 3.0];
        let all_gradients = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
        ];

        let result = detector.analyze("node-1", &normal_gradients, &all_gradients);
        assert!(!result.is_byzantine);
    }

    #[test]
    fn test_byzantine_detection() {
        let mut detector = ByzantineDetector::new(0.33);

        let malicious_gradients = vec![100.0, 200.0, 300.0];
        let all_gradients = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![100.0, 200.0, 300.0], // malicious
        ];

        // Run multiple times to reduce trust score
        for _ in 0..15 {
            detector.analyze("bad-node", &malicious_gradients, &all_gradients);
        }

        assert!(detector.get_trust_score("bad-node") < 0.33);
    }
}
