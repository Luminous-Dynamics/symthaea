// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reputation System
//!
//! Tracks agent reputation across the Mycelix ecosystem.
//! Reputation is earned through positive interactions and lost through Byzantine behavior.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// A reputation score with metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReputationScore {
    /// The agent identifier (DID, pubkey hash, etc.)
    pub agent_id: String,

    /// Current reputation score [0.0, 1.0]
    pub score: f64,

    /// Total positive interactions
    pub positive_count: u64,

    /// Total negative interactions
    pub negative_count: u64,

    /// Last update timestamp
    pub last_updated: u64,

    /// Source system (e.g., "mail", "marketplace", "praxis")
    pub source: String,
}

impl ReputationScore {
    /// Create a new reputation score
    pub fn new(agent_id: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            score: 0.5, // Start neutral
            positive_count: 0,
            negative_count: 0,
            last_updated: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            source: source.into(),
        }
    }

    /// Record a positive interaction
    pub fn record_positive(&mut self) {
        self.positive_count += 1;
        self.recalculate();
    }

    /// Record a negative interaction
    pub fn record_negative(&mut self) {
        self.negative_count += 1;
        self.recalculate();
    }

    /// Recalculate score based on interaction counts
    fn recalculate(&mut self) {
        let total = self.positive_count + self.negative_count;
        if total == 0 {
            self.score = 0.5;
        } else {
            // Simple ratio with Bayesian smoothing (add 1 to each)
            self.score = (self.positive_count as f64 + 1.0) / (total as f64 + 2.0);
        }
        self.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Check if this is a new agent (few interactions)
    pub fn is_new(&self) -> bool {
        self.positive_count + self.negative_count < 10
    }

    /// Get interaction count
    pub fn total_interactions(&self) -> u64 {
        self.positive_count + self.negative_count
    }
}

/// Reputation history for time-series analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationHistory {
    /// Agent identifier
    pub agent_id: String,

    /// Historical scores (newest first)
    scores: VecDeque<(u64, f64)>, // (timestamp, score)

    /// Maximum history size
    max_size: usize,
}

/// Minimum allowed history window size (prevents division by zero in stats)
pub const MIN_HISTORY_SIZE: usize = 10;

/// Maximum allowed history window size (prevents memory exhaustion)
pub const MAX_HISTORY_SIZE: usize = 10_000;

impl ReputationHistory {
    /// Create new history tracker
    ///
    /// # Security Note (FIND-007 mitigation)
    ///
    /// The `max_size` parameter is clamped to a safe range [10, 10_000] to prevent:
    /// - Memory exhaustion attacks (very large values)
    /// - Division by zero in statistical calculations (very small values)
    ///
    /// # Arguments
    /// * `agent_id` - The agent identifier
    /// * `max_size` - Maximum history entries to keep (clamped to [10, 10_000])
    pub fn new(agent_id: impl Into<String>, max_size: usize) -> Self {
        // FIND-007: Clamp max_size to reasonable bounds
        let max_size = max_size.clamp(MIN_HISTORY_SIZE, MAX_HISTORY_SIZE);
        Self {
            agent_id: agent_id.into(),
            scores: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Get the configured maximum history size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Add a score observation
    pub fn observe(&mut self, score: f64) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if self.scores.len() >= self.max_size {
            self.scores.pop_back();
        }
        self.scores.push_front((timestamp, score));
    }

    /// Get current score (most recent)
    pub fn current(&self) -> Option<f64> {
        self.scores.front().map(|(_, s)| *s)
    }

    /// Calculate mean score
    pub fn mean(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.5;
        }
        self.scores.iter().map(|(_, s)| s).sum::<f64>() / self.scores.len() as f64
    }

    /// Calculate standard deviation
    pub fn std_dev(&self) -> f64 {
        if self.scores.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self
            .scores
            .iter()
            .map(|(_, s)| (s - mean).powi(2))
            .sum::<f64>()
            / self.scores.len() as f64;
        variance.sqrt()
    }

    /// Check if score is anomalous (outside 2 std devs)
    pub fn is_anomalous(&self, score: f64) -> bool {
        if self.scores.len() < 5 {
            return false; // Not enough data
        }
        let mean = self.mean();
        let std = self.std_dev();
        (score - mean).abs() > 2.0 * std
    }

    /// Get trend (positive = improving, negative = declining)
    pub fn trend(&self) -> f64 {
        if self.scores.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = self.scores.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, (_, score)) in self.scores.iter().enumerate() {
            let x = i as f64;
            let y = *score;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    }

    /// Number of observations
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_reputation() {
        let rep = ReputationScore::new("agent123", "mail");
        assert_eq!(rep.score, 0.5);
        assert!(rep.is_new());
    }

    #[test]
    fn test_positive_interaction() {
        let mut rep = ReputationScore::new("agent123", "mail");
        rep.record_positive();
        assert!(rep.score > 0.5);
    }

    #[test]
    fn test_negative_interaction() {
        let mut rep = ReputationScore::new("agent123", "mail");
        rep.record_negative();
        assert!(rep.score < 0.5);
    }

    #[test]
    fn test_history() {
        let mut history = ReputationHistory::new("agent123", 100);
        history.observe(0.8);
        history.observe(0.85);
        history.observe(0.82);

        assert_eq!(history.len(), 3);
        assert!((history.mean() - 0.8233).abs() < 0.01);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut history = ReputationHistory::new("agent123", 100);
        // Add slight variance for realistic behavior
        let values = [0.78, 0.82, 0.79, 0.83, 0.80, 0.77, 0.84, 0.81, 0.79, 0.82];
        for &v in &values {
            history.observe(v);
        }

        // 0.78 is within normal range (mean ~0.8, std ~0.02)
        assert!(!history.is_anomalous(0.78));
        // 0.2 is way below normal and anomalous
        assert!(history.is_anomalous(0.2));
    }

    // ==========================================================================
    // FIND-007 Mitigation Tests: Bounded history windows
    // ==========================================================================

    #[test]
    fn test_history_size_clamped_to_minimum() {
        // Zero should be clamped to MIN_HISTORY_SIZE
        let history = ReputationHistory::new("agent123", 0);
        assert_eq!(history.max_size(), MIN_HISTORY_SIZE);

        // 1 should be clamped to MIN_HISTORY_SIZE
        let history = ReputationHistory::new("agent123", 1);
        assert_eq!(history.max_size(), MIN_HISTORY_SIZE);

        // 5 should be clamped to MIN_HISTORY_SIZE
        let history = ReputationHistory::new("agent123", 5);
        assert_eq!(history.max_size(), MIN_HISTORY_SIZE);
    }

    #[test]
    fn test_history_size_clamped_to_maximum() {
        // Very large values should be clamped to MAX_HISTORY_SIZE
        let history = ReputationHistory::new("agent123", 100_000);
        assert_eq!(history.max_size(), MAX_HISTORY_SIZE);

        // usize::MAX should be clamped to MAX_HISTORY_SIZE
        let history = ReputationHistory::new("agent123", usize::MAX);
        assert_eq!(history.max_size(), MAX_HISTORY_SIZE);
    }

    #[test]
    fn test_history_size_valid_range() {
        // Values within valid range should be preserved
        let history = ReputationHistory::new("agent123", 100);
        assert_eq!(history.max_size(), 100);

        let history = ReputationHistory::new("agent123", 1000);
        assert_eq!(history.max_size(), 1000);

        // Boundary values
        let history = ReputationHistory::new("agent123", MIN_HISTORY_SIZE);
        assert_eq!(history.max_size(), MIN_HISTORY_SIZE);

        let history = ReputationHistory::new("agent123", MAX_HISTORY_SIZE);
        assert_eq!(history.max_size(), MAX_HISTORY_SIZE);
    }

    #[test]
    fn test_history_respects_max_size() {
        let mut history = ReputationHistory::new("agent123", 15); // Clamped to MIN_HISTORY_SIZE if < 10

        // Add more entries than max_size allows
        for i in 0..20 {
            history.observe(i as f64 / 100.0);
        }

        // Should not exceed max_size
        assert!(history.len() <= history.max_size());
    }
}
