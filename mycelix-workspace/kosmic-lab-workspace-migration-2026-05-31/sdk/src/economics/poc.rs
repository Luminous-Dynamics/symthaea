// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Proof of Contribution (PoC)
//!
//! Implementation of MIP-E-002 Article II: PoC Framework
//!
//! PoC replaces stake-weighted validation with contribution-based scoring:
//! - Behavioral Component (60%): Transaction consistency, validation quality, governance
//! - Physical Grounding (40%): See `pog` module for infrastructure contribution

use serde::{Deserialize, Serialize};

/// Proof of Contribution score combining behavioral and physical components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfContribution {
    /// Behavioral metrics component (0.0 - 1.0)
    pub behavioral_score: f64,
    /// Physical grounding component (0.0 - 1.0), see `pog` module
    pub pog_score: f64,
    /// Longevity multiplier (1.0 - 1.5)
    pub longevity_multiplier: f64,
    /// Active months for longevity calculation
    pub active_months: u32,
    /// Composite PoC score
    pub composite: f64,
}

/// Weights for PoC calculation (configurable via governance)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PoCWeights {
    /// Weight for behavioral component (default 0.60)
    pub behavioral: f64,
    /// Weight for physical grounding (default 0.40)
    pub pog: f64,
}

impl Default for PoCWeights {
    fn default() -> Self {
        Self {
            behavioral: 0.60,
            pog: 0.40,
        }
    }
}

/// Behavioral metrics for PoC calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralMetrics {
    /// Transaction consistency (90-day rolling entropy)
    pub transaction_consistency: f64,
    /// Validation quality (correct / total)
    pub validation_quality: f64,
    /// Governance participation score
    pub governance_participation: f64,
    /// Peer endorsement score (unique endorsers)
    pub peer_endorsements: f64,
    /// Dispute resolution participation
    pub dispute_resolution: f64,
    /// Network contribution (code, docs, etc.)
    pub network_contribution: f64,
}

impl BehavioralMetrics {
    /// Calculate weighted behavioral score
    pub fn calculate_score(&self) -> f64 {
        // Weights per MIP-E-002 Article II Section 2.1
        const TRANSACTION_WEIGHT: f64 = 0.25;
        const VALIDATION_WEIGHT: f64 = 0.20;
        const GOVERNANCE_WEIGHT: f64 = 0.20;
        const ENDORSEMENT_WEIGHT: f64 = 0.15;
        const DISPUTE_WEIGHT: f64 = 0.10;
        const CONTRIBUTION_WEIGHT: f64 = 0.10;

        let score = self.transaction_consistency * TRANSACTION_WEIGHT
            + self.validation_quality * VALIDATION_WEIGHT
            + self.governance_participation * GOVERNANCE_WEIGHT
            + self.peer_endorsements * ENDORSEMENT_WEIGHT
            + self.dispute_resolution * DISPUTE_WEIGHT
            + self.network_contribution * CONTRIBUTION_WEIGHT;

        score.clamp(0.0, 1.0)
    }
}

/// Contribution score result with detailed breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionScore {
    /// Final composite score (0.0 - 1.0)
    pub final_score: f64,
    /// Behavioral component contribution
    pub behavioral_contribution: f64,
    /// Physical grounding contribution
    pub pog_contribution: f64,
    /// Longevity bonus applied
    pub longevity_bonus: f64,
    /// Percentile rank among network participants
    pub percentile_rank: Option<f64>,
}

/// Calculate longevity multiplier from active months
pub fn calculate_longevity_multiplier(active_months: u32) -> f64 {
    // Per MIP-E-002: Longevity_Bonus = min(1.5, 1.0 + (active_months × 0.02))
    (1.0 + active_months as f64 * 0.02).min(1.5)
}

/// Calculate complete Proof of Contribution score
pub fn calculate_poc_score(
    behavioral: &BehavioralMetrics,
    pog_score: f64,
    active_months: u32,
    weights: &PoCWeights,
) -> ContributionScore {
    let behavioral_score = behavioral.calculate_score();
    let longevity = calculate_longevity_multiplier(active_months);

    // Weighted combination
    let behavioral_contribution = behavioral_score * weights.behavioral;
    let pog_contribution = pog_score * weights.pog;

    // Apply longevity multiplier to total
    let base_score = behavioral_contribution + pog_contribution;
    let final_score = (base_score * longevity).clamp(0.0, 1.0);

    ContributionScore {
        final_score,
        behavioral_contribution,
        pog_contribution,
        longevity_bonus: longevity - 1.0,
        percentile_rank: None, // Calculated by network statistics
    }
}

impl ProofOfContribution {
    /// Create new PoC from components
    pub fn new(behavioral: &BehavioralMetrics, pog_score: f64, active_months: u32) -> Self {
        let weights = PoCWeights::default();
        let longevity = calculate_longevity_multiplier(active_months);

        let behavioral_score = behavioral.calculate_score();
        let composite =
            ((behavioral_score * weights.behavioral) + (pog_score * weights.pog)) * longevity;

        Self {
            behavioral_score,
            pog_score,
            longevity_multiplier: longevity,
            active_months,
            composite: composite.clamp(0.0, 1.0),
        }
    }

    /// Check if eligible for validator status
    pub fn validator_eligible(&self) -> bool {
        // Minimum PoC threshold for validators
        self.composite >= 0.5
    }

    /// Check if eligible for steward status
    pub fn steward_eligible(&self) -> bool {
        self.composite >= 0.7
    }

    /// Check if eligible for guardian status
    pub fn guardian_eligible(&self) -> bool {
        self.composite >= 0.9
    }
}

/// Anti-gaming detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingDetection {
    /// Sybil cluster suspicion score
    pub sybil_score: f64,
    /// Burst activity detection
    pub burst_detected: bool,
    /// Wash trading suspicion
    pub wash_trade_score: f64,
    /// Recommendation
    pub recommendation: GamingRecommendation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Recommendation for handling suspected gaming behavior.
pub enum GamingRecommendation {
    /// No action needed
    Clear,
    /// Increased monitoring
    Watch,
    /// Manual review required
    Review,
    /// Score adjustment applied
    Penalized,
}

/// Detect potential gaming behavior in PoC metrics
pub fn detect_gaming(
    metrics: &BehavioralMetrics,
    historical: &[BehavioralMetrics],
) -> GamingDetection {
    // Simplified detection - full implementation would use MATL graph analysis
    let mut sybil_score = 0.0;
    let mut burst_detected = false;
    let mut wash_trade_score = 0.0;

    // Check for sudden spikes in participation
    if let Some(prev) = historical.last() {
        let participation_delta =
            (metrics.governance_participation - prev.governance_participation).abs();
        if participation_delta > 0.5 {
            burst_detected = true;
            sybil_score += 0.3;
        }
    }

    // Check for unusually high endorsement without transaction history
    if metrics.peer_endorsements > 0.8 && metrics.transaction_consistency < 0.3 {
        sybil_score += 0.4;
        wash_trade_score += 0.3;
    }

    let recommendation = if sybil_score > 0.6 {
        GamingRecommendation::Review
    } else if sybil_score > 0.3 || burst_detected {
        GamingRecommendation::Watch
    } else {
        GamingRecommendation::Clear
    };

    GamingDetection {
        sybil_score,
        burst_detected,
        wash_trade_score,
        recommendation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_behavioral() -> BehavioralMetrics {
        BehavioralMetrics {
            transaction_consistency: 0.8,
            validation_quality: 0.9,
            governance_participation: 0.6,
            peer_endorsements: 0.7,
            dispute_resolution: 0.5,
            network_contribution: 0.4,
        }
    }

    #[test]
    fn test_behavioral_score() {
        let metrics = sample_behavioral();
        let score = metrics.calculate_score();

        // 0.8*0.25 + 0.9*0.20 + 0.6*0.20 + 0.7*0.15 + 0.5*0.10 + 0.4*0.10
        // = 0.20 + 0.18 + 0.12 + 0.105 + 0.05 + 0.04 = 0.695
        assert!((score - 0.695).abs() < 0.01);
    }

    #[test]
    fn test_longevity_multiplier() {
        assert!((calculate_longevity_multiplier(0) - 1.0).abs() < 0.001);
        assert!((calculate_longevity_multiplier(12) - 1.24).abs() < 0.001);
        assert!((calculate_longevity_multiplier(25) - 1.5).abs() < 0.001);
        assert!((calculate_longevity_multiplier(100) - 1.5).abs() < 0.001); // Capped
    }

    #[test]
    fn test_poc_calculation() {
        let behavioral = sample_behavioral();
        let poc = ProofOfContribution::new(&behavioral, 0.6, 12);

        // Behavioral: 0.695 * 0.60 = 0.417
        // PoG: 0.6 * 0.40 = 0.24
        // Base: 0.657
        // With 1.24x longevity: 0.815
        assert!(poc.composite > 0.7 && poc.composite < 0.9);
        assert!(poc.steward_eligible());
        assert!(!poc.guardian_eligible());
    }

    #[test]
    fn test_gaming_detection() {
        let current = BehavioralMetrics {
            peer_endorsements: 0.9,
            transaction_consistency: 0.1, // Suspicious: high endorsements, low activity
            ..sample_behavioral()
        };

        let detection = detect_gaming(&current, &[]);
        assert!(detection.sybil_score > 0.3);
        assert!(matches!(
            detection.recommendation,
            GamingRecommendation::Watch | GamingRecommendation::Review
        ));
    }
}
