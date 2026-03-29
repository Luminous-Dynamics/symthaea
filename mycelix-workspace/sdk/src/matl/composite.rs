// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Composite Score Calculation
//!
//! Combines PoGQ, consistency, and reputation into a single trust score.

use super::{ProofOfGradientQuality, ReputationScore};
use serde::{Deserialize, Serialize};

/// Composite trust score combining multiple signals
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositeScore {
    /// The computed composite score [0.0, 1.0]
    pub score: f64,

    /// PoGQ component
    pub pogq_component: f64,

    /// Consistency component
    pub consistency_component: f64,

    /// Reputation component
    pub reputation_component: f64,

    /// Weights used
    pub weights: ScoreWeights,

    /// Timestamp of calculation
    pub calculated_at: u64,
}

/// Weights for composite score calculation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreWeights {
    /// Weight for gradient quality component
    pub quality: f64,
    /// Weight for temporal consistency component
    pub consistency: f64,
    /// Weight for reputation component
    pub reputation: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            quality: super::DEFAULT_QUALITY_WEIGHT,
            consistency: super::DEFAULT_CONSISTENCY_WEIGHT,
            reputation: super::DEFAULT_REPUTATION_WEIGHT,
        }
    }
}

impl CompositeScore {
    /// Calculate composite score from PoGQ and reputation
    pub fn calculate(pogq: &ProofOfGradientQuality, reputation: &ReputationScore) -> Self {
        Self::calculate_weighted(pogq, reputation, ScoreWeights::default())
    }

    /// Calculate with custom weights
    pub fn calculate_weighted(
        pogq: &ProofOfGradientQuality,
        reputation: &ReputationScore,
        weights: ScoreWeights,
    ) -> Self {
        let total_weight = weights.quality + weights.consistency + weights.reputation;

        let pogq_component = weights.quality * pogq.quality;
        let consistency_component = weights.consistency * pogq.consistency;
        let reputation_component = weights.reputation * reputation.score;

        let score = if total_weight > 0.0 {
            (pogq_component + consistency_component + reputation_component) / total_weight
        } else {
            0.0
        };

        Self {
            score,
            pogq_component: pogq.quality,
            consistency_component: pogq.consistency,
            reputation_component: reputation.score,
            weights,
            calculated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Check if score indicates trustworthy behavior
    pub fn is_trustworthy(&self, threshold: f64) -> bool {
        self.score >= threshold
    }

    /// Check if score indicates Byzantine behavior
    pub fn is_byzantine(&self, threshold: f64) -> bool {
        self.score < threshold
    }

    /// Get confidence level based on component agreement
    pub fn confidence(&self) -> f64 {
        // If all components agree (all high or all low), confidence is high
        let mean =
            (self.pogq_component + self.consistency_component + self.reputation_component) / 3.0;
        let variance = ((self.pogq_component - mean).powi(2)
            + (self.consistency_component - mean).powi(2)
            + (self.reputation_component - mean).powi(2))
            / 3.0;

        // Higher variance = lower confidence
        1.0 - variance.sqrt().min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate() {
        let pogq = ProofOfGradientQuality::new(0.9, 0.85, 0.1);
        let rep = ReputationScore::new("agent1", "test");

        let composite = CompositeScore::calculate(&pogq, &rep);
        assert!(composite.score > 0.0);
        assert!(composite.score <= 1.0);
    }

    #[test]
    fn test_trustworthy() {
        let pogq = ProofOfGradientQuality::new(0.9, 0.9, 0.1);
        let mut rep = ReputationScore::new("agent1", "test");
        for _ in 0..10 {
            rep.record_positive();
        }

        let composite = CompositeScore::calculate(&pogq, &rep);
        assert!(composite.is_trustworthy(0.5));
    }

    #[test]
    fn test_confidence() {
        // High agreement = high confidence
        let pogq = ProofOfGradientQuality::new(0.9, 0.9, 0.1);
        let mut rep = ReputationScore::new("agent1", "test");
        for _ in 0..100 {
            rep.record_positive();
        }

        let composite = CompositeScore::calculate(&pogq, &rep);
        assert!(composite.confidence() > 0.8);
    }
}
