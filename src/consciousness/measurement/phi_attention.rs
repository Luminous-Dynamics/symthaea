// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ-Aware Attention Integration
//!
//! This module provides Φ (integrated information) integration with the
//! Global Workspace attention system. It enables consciousness metrics
//! to influence attention bidding and action confidence.
//!
//! ## Design Philosophy
//!
//! In biological systems, consciousness (as measured by Φ) affects:
//! - **Attention priority**: High Φ = confident processing, amplified bids
//! - **Decision thresholds**: Low Φ = higher barriers to action
//! - **Precision weighting**: Φ modulates how much we trust our predictions
//!
//! ## Integration with PrefrontalCortex
//!
//! Rather than modifying the existing prefrontal code directly (which is complex),
//! this module provides wrapper functions that can be used alongside it.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::phi_attention::{PhiAwareScoring, ConsciousnessThresholds};
//! use symthaea::brain::prefrontal::AttentionBid;
//!
//! let bid = AttentionBid::new("Thalamus", "User wants to install vim");
//! let phi = 0.65;  // Current consciousness level
//!
//! // Score with Φ awareness
//! let phi_score = PhiAwareScoring::score_bid(&bid, phi);
//!
//! // Check if action is allowed at this Φ
//! let thresholds = ConsciousnessThresholds::default();
//! if thresholds.allows_action(ActionType::StateModifying, phi) {
//!     // Proceed with action
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Types of actions that require different Φ thresholds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ActionType {
    /// Basic queries, information retrieval (Φ ≥ 0.2)
    #[default]
    BasicQuery,
    /// Commands that modify state (Φ ≥ 0.3)
    StateModifying,
    /// System-critical operations (Φ ≥ 0.4)
    SystemCritical,
    /// Irreversible operations (Φ ≥ 0.6)
    Irreversible,
}

/// Consciousness thresholds for different action types
///
/// These thresholds are inspired by IIT research suggesting that:
/// - Φ ≈ 0.2 represents minimal integration (simple responses)
/// - Φ ≈ 0.3-0.4 represents moderate integration (considered actions)
/// - Φ ≈ 0.5+ represents high integration (deliberate decisions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessThresholds {
    /// Threshold for basic queries (default: 0.2)
    pub basic: f32,
    /// Threshold for state-modifying actions (default: 0.3)
    pub state_modifying: f32,
    /// Threshold for system-critical operations (default: 0.4)
    pub system_critical: f32,
    /// Threshold for irreversible operations (default: 0.6)
    pub irreversible: f32,
}

impl Default for ConsciousnessThresholds {
    fn default() -> Self {
        Self {
            basic: 0.2,
            state_modifying: 0.3,
            system_critical: 0.4,
            irreversible: 0.6,
        }
    }
}

impl ConsciousnessThresholds {
    /// Get threshold for an action type
    pub fn threshold_for(&self, action_type: ActionType) -> f32 {
        match action_type {
            ActionType::BasicQuery => self.basic,
            ActionType::StateModifying => self.state_modifying,
            ActionType::SystemCritical => self.system_critical,
            ActionType::Irreversible => self.irreversible,
        }
    }

    /// Check if an action is allowed at the given Φ level
    pub fn allows_action(&self, action_type: ActionType, phi: f32) -> bool {
        phi >= self.threshold_for(action_type)
    }

    /// Get the required Φ deficit (how much more consciousness is needed)
    pub fn phi_deficit(&self, action_type: ActionType, phi: f32) -> f32 {
        let threshold = self.threshold_for(action_type);
        (threshold - phi).max(0.0)
    }

    /// Create thresholds tuned for high-risk environment
    pub fn high_security() -> Self {
        Self {
            basic: 0.3,
            state_modifying: 0.4,
            system_critical: 0.5,
            irreversible: 0.7,
        }
    }

    /// Create thresholds tuned for trusted environment
    pub fn low_security() -> Self {
        Self {
            basic: 0.1,
            state_modifying: 0.2,
            system_critical: 0.3,
            irreversible: 0.5,
        }
    }
}

/// Φ-aware scoring for attention bids
///
/// This provides methods to incorporate consciousness metrics
/// into the attention competition process.
pub struct PhiAwareScoring;

impl PhiAwareScoring {
    /// Score an attention bid with Φ as a confidence multiplier
    ///
    /// # Algorithm
    ///
    /// The base score (salience × urgency + emotional_boost) is modulated by Φ:
    /// - **High Φ (> 0.6)**: Amplifies high-salience bids (up to 1.2x)
    /// - **Medium Φ (0.3-0.6)**: Linear scaling (0.85-1.1x)
    /// - **Low Φ (< 0.3)**: Dampens all bids (0.5-0.65x)
    ///
    /// This reflects that low consciousness means less certainty in any bid,
    /// while high consciousness can amplify important signals.
    ///
    /// # Arguments
    ///
    /// * `salience` - How important the bid is (0.0-1.0)
    /// * `urgency` - How time-sensitive (0.0-1.0)
    /// * `emotional_weight` - Emotional boost (typically 0.0-0.2)
    /// * `phi` - Current integrated information (0.0-1.0)
    ///
    /// # Returns
    ///
    /// Φ-weighted score (typically 0.0-1.5)
    pub fn calculate_score(salience: f32, urgency: f32, emotional_weight: f32, phi: f32) -> f32 {
        let base_score = salience * urgency + emotional_weight;

        // Φ factor calculation - continuous scaling from 0.6x to 1.3x
        // - Φ = 0.0: factor = 0.6 (heavy dampening)
        // - Φ = 0.3: factor = 0.85
        // - Φ = 0.5: factor = 1.0 (neutral)
        // - Φ = 0.7: factor = 1.15
        // - Φ = 1.0: factor = 1.3 (maximum boost)
        let phi_factor = 0.6 + phi * 0.7;

        base_score * phi_factor
    }

    /// Calculate the confidence level for an action based on Φ
    ///
    /// Returns a descriptive confidence level and recommendation.
    pub fn confidence_level(phi: f32) -> ConfidenceLevel {
        if phi >= 0.6 {
            ConfidenceLevel::High {
                phi,
                recommendation: "Proceed with confidence".to_string(),
            }
        } else if phi >= 0.4 {
            ConfidenceLevel::Medium {
                phi,
                recommendation: "Proceed with standard caution".to_string(),
            }
        } else if phi >= 0.2 {
            ConfidenceLevel::Low {
                phi,
                recommendation: "Consider user confirmation".to_string(),
            }
        } else {
            ConfidenceLevel::VeryLow {
                phi,
                recommendation: "Require explicit confirmation".to_string(),
            }
        }
    }

    /// Calculate broadcast threshold with Φ adjustment
    ///
    /// In Global Workspace Theory, bids must exceed a threshold to be
    /// broadcast to all brain modules. This threshold is Φ-adaptive:
    /// - High Φ = lower threshold (more confident in selections)
    /// - Low Φ = higher threshold (more conservative)
    pub fn broadcast_threshold(base_threshold: f32, phi: f32) -> f32 {
        // Φ reduces the threshold (more permissive when conscious)
        let adjustment = phi * 0.3; // Up to 30% reduction
        (base_threshold - adjustment).max(0.1)
    }
}

/// Confidence level for actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Φ ≥ 0.6: High confidence
    High { phi: f32, recommendation: String },
    /// Φ ∈ [0.4, 0.6): Medium confidence
    Medium { phi: f32, recommendation: String },
    /// Φ ∈ [0.2, 0.4): Low confidence
    Low { phi: f32, recommendation: String },
    /// Φ < 0.2: Very low confidence
    VeryLow { phi: f32, recommendation: String },
}

impl ConfidenceLevel {
    /// Get the Φ value
    pub fn phi(&self) -> f32 {
        match self {
            Self::High { phi, .. } => *phi,
            Self::Medium { phi, .. } => *phi,
            Self::Low { phi, .. } => *phi,
            Self::VeryLow { phi, .. } => *phi,
        }
    }

    /// Get the recommendation
    pub fn recommendation(&self) -> &str {
        match self {
            Self::High { recommendation, .. } => recommendation,
            Self::Medium { recommendation, .. } => recommendation,
            Self::Low { recommendation, .. } => recommendation,
            Self::VeryLow { recommendation, .. } => recommendation,
        }
    }

    /// Check if this is sufficient for an action type
    pub fn allows(&self, action_type: ActionType) -> bool {
        let thresholds = ConsciousnessThresholds::default();
        self.phi() >= thresholds.threshold_for(action_type)
    }
}

/// Adaptive thresholds that learn from history
///
/// Adjusts consciousness thresholds based on:
/// - Rolling Φ average (stability indicator)
/// - Φ variance (uncertainty indicator)
/// - Success/failure history (learning)
#[derive(Debug, Clone)]
pub struct AdaptiveThresholds {
    /// Base thresholds
    base: ConsciousnessThresholds,
    /// Rolling Φ history
    phi_history: VecDeque<f32>,
    /// Maximum history size
    history_size: usize,
    /// Success count per action type
    successes: std::collections::HashMap<ActionType, usize>,
    /// Failure count per action type
    failures: std::collections::HashMap<ActionType, usize>,
}

impl Default for AdaptiveThresholds {
    fn default() -> Self {
        Self::new(100)
    }
}

impl AdaptiveThresholds {
    /// Create new adaptive thresholds with specified history size
    pub fn new(history_size: usize) -> Self {
        Self {
            base: ConsciousnessThresholds::default(),
            phi_history: VecDeque::with_capacity(history_size),
            history_size,
            successes: std::collections::HashMap::new(),
            failures: std::collections::HashMap::new(),
        }
    }

    /// Record a new Φ observation
    pub fn observe(&mut self, phi: f32) {
        if self.phi_history.len() >= self.history_size {
            self.phi_history.pop_front();
        }
        self.phi_history.push_back(phi);
    }

    /// Record an action outcome
    pub fn record_outcome(&mut self, action_type: ActionType, success: bool) {
        if success {
            *self.successes.entry(action_type).or_insert(0) += 1;
        } else {
            *self.failures.entry(action_type).or_insert(0) += 1;
        }
    }

    /// Get the rolling Φ average
    pub fn phi_average(&self) -> Option<f32> {
        if self.phi_history.is_empty() {
            return None;
        }
        Some(self.phi_history.iter().sum::<f32>() / self.phi_history.len() as f32)
    }

    /// Get the Φ variance (measure of stability)
    pub fn phi_variance(&self) -> Option<f32> {
        let avg = self.phi_average()?;
        if self.phi_history.len() < 2 {
            return None;
        }
        let variance = self
            .phi_history
            .iter()
            .map(|p| (p - avg).powi(2))
            .sum::<f32>()
            / (self.phi_history.len() - 1) as f32;
        Some(variance)
    }

    /// Get adaptive threshold for an action type
    ///
    /// Adjusts the base threshold based on:
    /// - Φ variance: High variance → raise threshold (uncertainty)
    /// - Success rate: Low success rate → raise threshold
    pub fn threshold_for(&self, action_type: ActionType) -> f32 {
        let base = self.base.threshold_for(action_type);

        // Variance adjustment
        let variance_adjustment = self
            .phi_variance()
            .map(|v| if v > 0.01 { v.sqrt() * 0.5 } else { -0.05 })
            .unwrap_or(0.0);

        // Success rate adjustment
        let successes = *self.successes.get(&action_type).unwrap_or(&0);
        let failures = *self.failures.get(&action_type).unwrap_or(&0);
        let total = successes + failures;

        let success_adjustment = if total > 10 {
            let rate = successes as f32 / total as f32;
            if rate < 0.5 {
                0.1 // Raise threshold if failing often
            } else if rate > 0.9 {
                -0.05 // Lower threshold if very successful
            } else {
                0.0
            }
        } else {
            0.0 // Not enough data
        };

        (base + variance_adjustment + success_adjustment).clamp(0.1, 0.9)
    }

    /// Check if an action is allowed at the given Φ level
    pub fn allows_action(&self, action_type: ActionType, phi: f32) -> bool {
        phi >= self.threshold_for(action_type)
    }

    /// Get statistics about the adaptive thresholds
    pub fn stats(&self) -> AdaptiveThresholdStats {
        AdaptiveThresholdStats {
            observations: self.phi_history.len(),
            phi_average: self.phi_average(),
            phi_variance: self.phi_variance(),
            current_thresholds: std::collections::HashMap::from([
                (
                    ActionType::BasicQuery,
                    self.threshold_for(ActionType::BasicQuery),
                ),
                (
                    ActionType::StateModifying,
                    self.threshold_for(ActionType::StateModifying),
                ),
                (
                    ActionType::SystemCritical,
                    self.threshold_for(ActionType::SystemCritical),
                ),
                (
                    ActionType::Irreversible,
                    self.threshold_for(ActionType::Irreversible),
                ),
            ]),
        }
    }
}

/// Statistics for adaptive thresholds
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdStats {
    pub observations: usize,
    pub phi_average: Option<f32>,
    pub phi_variance: Option<f32>,
    pub current_thresholds: std::collections::HashMap<ActionType, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_aware_scoring() {
        // High Φ should amplify scores
        let high_phi_score = PhiAwareScoring::calculate_score(0.8, 0.5, 0.1, 0.7);
        let base_score = PhiAwareScoring::calculate_score(0.8, 0.5, 0.1, 0.5);
        assert!(high_phi_score > base_score);

        // Low Φ should dampen scores
        let low_phi_score = PhiAwareScoring::calculate_score(0.8, 0.5, 0.1, 0.2);
        assert!(low_phi_score < base_score);
    }

    #[test]
    fn test_consciousness_thresholds() {
        let thresholds = ConsciousnessThresholds::default();

        // Basic queries allowed at low Φ
        assert!(thresholds.allows_action(ActionType::BasicQuery, 0.25));
        assert!(!thresholds.allows_action(ActionType::BasicQuery, 0.15));

        // Irreversible requires high Φ
        assert!(thresholds.allows_action(ActionType::Irreversible, 0.65));
        assert!(!thresholds.allows_action(ActionType::Irreversible, 0.55));
    }

    #[test]
    fn test_confidence_levels() {
        let high = PhiAwareScoring::confidence_level(0.7);
        assert!(matches!(high, ConfidenceLevel::High { .. }));
        assert!(high.allows(ActionType::Irreversible));

        let low = PhiAwareScoring::confidence_level(0.25);
        assert!(matches!(low, ConfidenceLevel::Low { .. }));
        assert!(low.allows(ActionType::BasicQuery));
        assert!(!low.allows(ActionType::SystemCritical));
    }

    #[test]
    fn test_adaptive_thresholds() {
        let mut adaptive = AdaptiveThresholds::new(10);

        // Observe some Φ values
        for phi in [0.45, 0.48, 0.52, 0.47, 0.50] {
            adaptive.observe(phi);
        }

        // Average should be around 0.48
        let avg = adaptive.phi_average().unwrap();
        assert!((avg - 0.484).abs() < 0.01);

        // Variance should be low (stable system)
        let variance = adaptive.phi_variance().unwrap();
        assert!(variance < 0.01);
    }

    #[test]
    fn test_adaptive_learning() {
        let mut adaptive = AdaptiveThresholds::new(10);

        // Record many failures for state-modifying
        for _ in 0..15 {
            adaptive.record_outcome(ActionType::StateModifying, false);
        }

        // Threshold should be raised
        let threshold = adaptive.threshold_for(ActionType::StateModifying);
        assert!(threshold > 0.3); // Base is 0.3
    }

    #[test]
    fn test_broadcast_threshold() {
        let base = 0.5;

        // High Φ lowers threshold
        let high_phi_threshold = PhiAwareScoring::broadcast_threshold(base, 0.7);
        assert!(high_phi_threshold < base);

        // Low Φ keeps threshold higher
        let low_phi_threshold = PhiAwareScoring::broadcast_threshold(base, 0.2);
        assert!(low_phi_threshold > high_phi_threshold);
    }
}
