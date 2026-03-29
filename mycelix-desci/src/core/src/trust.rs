// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MATL (Mycelix Adaptive Trust Layer) Integration
//!
//! Trust scoring and reputation management for DeSci participants

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trust score for a participant (0.0 - 1.0)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrustScore {
    /// Current trust score
    pub score: f64,

    /// Confidence in the score (based on history)
    pub confidence: f64,

    /// Number of interactions
    pub interaction_count: usize,
}

impl Default for TrustScore {
    fn default() -> Self {
        Self {
            score: 0.5, // Start neutral
            confidence: 0.1,
            interaction_count: 0,
        }
    }
}

/// Trust manager for tracking participant reputation
pub struct TrustManager {
    /// Trust scores by participant ID
    scores: HashMap<String, TrustScore>,

    /// Decay rate for old interactions (per day)
    decay_rate: f64,

    /// Minimum trust threshold for participation
    min_trust: f64,
}

impl TrustManager {
    /// Create a new trust manager
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            decay_rate: 0.01, // 1% per day
            min_trust: 0.3,
        }
    }

    /// Get trust score for a participant
    pub fn get_score(&self, participant_id: &str) -> TrustScore {
        self.scores
            .get(participant_id)
            .copied()
            .unwrap_or_default()
    }

    /// Update trust score based on behavior
    pub fn update_score(
        &mut self,
        participant_id: &str,
        positive: bool,
        weight: f64,
    ) -> Result<TrustScore> {
        let mut score = self.get_score(participant_id);

        // Update based on positive/negative behavior
        let delta = if positive { weight } else { -weight };
        score.score = (score.score + delta * 0.1).clamp(0.0, 1.0);

        // Increase confidence with more interactions
        score.interaction_count += 1;
        score.confidence = (score.interaction_count as f64 / 100.0).min(1.0);

        self.scores.insert(participant_id.to_string(), score);

        Ok(score)
    }

    /// Check if participant meets minimum trust threshold
    pub fn is_trusted(&self, participant_id: &str) -> bool {
        let score = self.get_score(participant_id);
        score.score >= self.min_trust
    }

    /// Apply time-based decay to all scores
    pub fn apply_decay(&mut self) {
        for score in self.scores.values_mut() {
            // Scores decay toward neutral (0.5)
            let decay = (score.score - 0.5) * self.decay_rate;
            score.score -= decay;
        }
    }
}

impl Default for TrustManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_trust_score() {
        let score = TrustScore::default();
        assert_eq!(score.score, 0.5);
        assert_eq!(score.interaction_count, 0);
    }

    #[test]
    fn test_trust_manager() {
        let mut manager = TrustManager::new();

        // Update with positive behavior
        manager
            .update_score("participant_1", true, 1.0)
            .unwrap();
        let score = manager.get_score("participant_1");

        assert!(score.score > 0.5);
        assert_eq!(score.interaction_count, 1);
    }

    #[test]
    fn test_trust_threshold() {
        let mut manager = TrustManager::new();
        manager.min_trust = 0.6;

        assert!(!manager.is_trusted("participant_1"));

        // Boost trust
        for _ in 0..5 {
            manager
                .update_score("participant_1", true, 1.0)
                .unwrap();
        }

        assert!(manager.is_trusted("participant_1"));
    }
}
