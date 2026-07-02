// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Online learning system for improving intent classification
//!
//! Tracks user feedback and command outcomes to improve the HDC model over time.

use serde::{Deserialize, Serialize};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::intent::Intent;
use crate::paths;

/// Learning system for online improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSystem {
    /// Recorded interactions
    interactions: Vec<Interaction>,

    /// Statistics
    stats: LearningStats,
}

/// A recorded interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// Original query
    pub query: String,

    /// Classified intent
    pub intent: String,

    /// Outcome (1.0 = success, -0.5 = failure)
    pub outcome: f32,

    /// Timestamp
    pub timestamp: u64,
}

/// Learning statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningStats {
    /// Total number of interactions
    pub total_interactions: u64,

    /// Successful interactions
    pub successful: u64,

    /// Failed interactions
    pub failed: u64,

    /// Interactions by intent
    pub by_intent: std::collections::HashMap<String, IntentStats>,
}

/// Statistics for a specific intent
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntentStats {
    /// Total count
    pub total: u64,

    /// Successful count
    pub successful: u64,
}

impl LearningSystem {
    /// Create a new learning system
    pub fn new() -> Self {
        Self {
            interactions: Vec::new(),
            stats: LearningStats::default(),
        }
    }

    /// Load from disk
    pub fn load() -> Option<Self> {
        let path = paths::data_dir()?.join("learning.json");
        let content = fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Save to disk
    pub fn save(&self) -> anyhow::Result<()> {
        let data_dir = paths::data_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;

        fs::create_dir_all(&data_dir)?;

        let path = data_dir.join("learning.json");
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;

        Ok(())
    }

    /// Record an interaction outcome
    pub fn record(&mut self, query: &str, intent: &Intent, outcome: f32) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let intent_name = intent.action_name().to_string();

        let interaction = Interaction {
            query: query.to_string(),
            intent: intent_name.clone(),
            outcome,
            timestamp,
        };

        self.interactions.push(interaction);

        // Update stats
        self.stats.total_interactions += 1;

        if outcome > 0.0 {
            self.stats.successful += 1;
        } else {
            self.stats.failed += 1;
        }

        // Update per-intent stats
        let intent_stats = self.stats.by_intent.entry(intent_name).or_default();
        intent_stats.total += 1;
        if outcome > 0.0 {
            intent_stats.successful += 1;
        }

        // Prune old interactions (keep last 1000)
        if self.interactions.len() > 1000 {
            self.interactions = self.interactions.split_off(self.interactions.len() - 1000);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &LearningStats {
        &self.stats
    }

    /// Get recent interactions
    pub fn recent_interactions(&self, limit: usize) -> &[Interaction] {
        let start = self.interactions.len().saturating_sub(limit);
        &self.interactions[start..]
    }

    /// Get success rate for an intent
    pub fn success_rate(&self, intent_name: &str) -> Option<f64> {
        self.stats.by_intent.get(intent_name).map(|s| {
            if s.total == 0 {
                0.0
            } else {
                s.successful as f64 / s.total as f64
            }
        })
    }

    /// Export interactions for HDC retraining
    pub fn export_for_training(&self) -> Vec<(String, String, f32)> {
        self.interactions
            .iter()
            .map(|i| (i.query.clone(), i.intent.clone(), i.outcome))
            .collect()
    }
}

impl Default for LearningSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_interaction() {
        let mut learning = LearningSystem::new();

        learning.record(
            "search firefox",
            &Intent::Search {
                query: "firefox".to_string(),
            },
            1.0,
        );

        assert_eq!(learning.stats.total_interactions, 1);
        assert_eq!(learning.stats.successful, 1);
    }

    #[test]
    fn test_success_rate() {
        let mut learning = LearningSystem::new();

        for _ in 0..8 {
            learning.record(
                "search test",
                &Intent::Search {
                    query: "test".to_string(),
                },
                1.0,
            );
        }
        for _ in 0..2 {
            learning.record(
                "search fail",
                &Intent::Search {
                    query: "fail".to_string(),
                },
                -0.5,
            );
        }

        let rate = learning.success_rate("search").unwrap();
        assert!((rate - 0.8).abs() < 0.01);
    }
}
