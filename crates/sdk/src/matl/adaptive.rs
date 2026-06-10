// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Adaptive Threshold System
//!
//! Revolutionary per-node adaptive thresholds that eliminate false positives
//! in non-IID (heterogeneous) data scenarios.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Adaptive threshold based on node's historical performance
///
/// Instead of a fixed threshold, each node has its own baseline
/// computed from historical observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThreshold {
    /// Node identifier
    pub node_id: String,

    /// Historical quality observations
    history: VecDeque<f64>,

    /// Maximum history size
    window_size: usize,

    /// Minimum threshold (floor)
    min_threshold: f64,

    /// Number of standard deviations for threshold
    sigma_multiplier: f64,
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold tracker
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    /// * `window_size` - How many observations to keep in history
    pub fn new(node_id: impl Into<String>, window_size: usize) -> Self {
        Self {
            node_id: node_id.into(),
            history: VecDeque::with_capacity(window_size),
            window_size,
            min_threshold: 0.1,
            sigma_multiplier: 2.0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        node_id: impl Into<String>,
        window_size: usize,
        min_threshold: f64,
        sigma_multiplier: f64,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            history: VecDeque::with_capacity(window_size),
            window_size,
            min_threshold,
            sigma_multiplier,
        }
    }

    /// Add a new observation
    pub fn observe(&mut self, quality: f64) {
        if self.history.len() >= self.window_size {
            self.history.pop_back();
        }
        self.history.push_front(quality.clamp(0.0, 1.0));
    }

    /// Compute the adaptive threshold
    ///
    /// Formula: max(min_threshold, mean - sigma_multiplier * std_dev)
    pub fn threshold(&self) -> f64 {
        if self.history.is_empty() {
            return super::DEFAULT_BYZANTINE_THRESHOLD;
        }

        let mean = self.mean();
        let std = self.std_dev();

        (mean - self.sigma_multiplier * std).max(self.min_threshold)
    }

    /// Check if a score is anomalous for this node
    pub fn is_anomalous(&self, score: f64) -> bool {
        score < self.threshold()
    }

    /// Get the mean of historical observations
    pub fn mean(&self) -> f64 {
        if self.history.is_empty() {
            return 0.5;
        }
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }

    /// Get the standard deviation
    pub fn std_dev(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.history.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / self.history.len() as f64;
        variance.sqrt()
    }

    /// Number of observations
    pub fn observation_count(&self) -> usize {
        self.history.len()
    }

    /// Check if we have enough data for reliable thresholds
    pub fn is_reliable(&self) -> bool {
        self.history.len() >= 10
    }

    /// Reset history (e.g., after detected attack)
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// Manager for multiple node thresholds
#[derive(Debug, Clone, Default)]
pub struct AdaptiveThresholdManager {
    thresholds: std::collections::HashMap<String, AdaptiveThreshold>,
    window_size: usize,
}

impl AdaptiveThresholdManager {
    /// Create a new manager
    pub fn new(window_size: usize) -> Self {
        Self {
            thresholds: std::collections::HashMap::new(),
            window_size,
        }
    }

    /// Get or create threshold for a node
    pub fn get_or_create(&mut self, node_id: &str) -> &mut AdaptiveThreshold {
        self.thresholds
            .entry(node_id.to_string())
            .or_insert_with(|| AdaptiveThreshold::new(node_id, self.window_size))
    }

    /// Record observation for a node
    pub fn observe(&mut self, node_id: &str, quality: f64) {
        self.get_or_create(node_id).observe(quality);
    }

    /// Get threshold for a node
    pub fn threshold(&mut self, node_id: &str) -> f64 {
        self.get_or_create(node_id).threshold()
    }

    /// Check if score is anomalous for a node
    pub fn is_anomalous(&mut self, node_id: &str, score: f64) -> bool {
        self.get_or_create(node_id).is_anomalous(score)
    }

    /// Get all tracked nodes
    pub fn nodes(&self) -> Vec<&str> {
        self.thresholds.keys().map(|s| s.as_str()).collect()
    }

    /// Number of tracked nodes
    pub fn node_count(&self) -> usize {
        self.thresholds.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_threshold() {
        let threshold = AdaptiveThreshold::new("node1", 100);
        assert_eq!(threshold.observation_count(), 0);
        assert!(!threshold.is_reliable());
    }

    #[test]
    fn test_observe() {
        let mut threshold = AdaptiveThreshold::new("node1", 100);
        threshold.observe(0.9);
        threshold.observe(0.85);
        threshold.observe(0.88);

        assert_eq!(threshold.observation_count(), 3);
        assert!((threshold.mean() - 0.8767).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut threshold = AdaptiveThreshold::new("node1", 100);

        // Simulate consistent high-quality node with slight variance (realistic)
        let values = [
            0.88, 0.91, 0.89, 0.92, 0.90, 0.87, 0.93, 0.88, 0.91, 0.89, 0.90, 0.88, 0.92, 0.89,
            0.91, 0.87, 0.90, 0.93, 0.88, 0.91,
        ];
        for &v in &values {
            threshold.observe(v);
        }

        // Threshold should be based on this node's performance
        let t = threshold.threshold();
        assert!(t > 0.8); // High-performing node has high threshold

        // A score of 0.5 would be anomalous for this high-performing node
        assert!(threshold.is_anomalous(0.5));
        // 0.88 is within normal range (mean ~0.9, std ~0.02)
        assert!(!threshold.is_anomalous(0.88));
    }

    #[test]
    fn test_manager() {
        let mut manager = AdaptiveThresholdManager::new(100);

        manager.observe("node1", 0.9);
        manager.observe("node2", 0.5);

        assert_eq!(manager.node_count(), 2);
    }
}
