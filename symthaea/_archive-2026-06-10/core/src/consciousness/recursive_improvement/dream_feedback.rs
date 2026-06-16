// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dream Feedback Bridge
//!
//! Connects the Counterfactual Dream Engine to the MAGI Loop's World Prediction system.
//!
//! ## The Insight
//!
//! Dreams generate counterfactual insights: "Action X in context C would have produced
//! higher Φ than what we actually did." This is wasted if we don't use it.
//!
//! ## The Bridge
//!
//! DreamFeedbackBridge converts dream insights into:
//! 1. **Action priors**: Bias future action selection toward dream-discovered patterns
//! 2. **Context tags**: Mark contexts where dreams found better alternatives
//! 3. **Confidence adjustments**: Reduce confidence in actions where dreams found failures
//!
//! ## Integration with MAGI Loop
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                         MAGI Loop                                │
//! │                                                                  │
//! │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
//! │  │ Predict │ -> │  Act    │ -> │ Observe │ -> │ Calibrate│      │
//! │  └────┬────┘    └─────────┘    └────┬────┘    └────┬────┘      │
//! │       │                             │               │          │
//! │       │  ┌─────────────────────────────────────────┘           │
//! │       │  │                                                      │
//! │       ▼  ▼                                                      │
//! │  ┌────────────────────┐                                        │
//! │  │  Dream Feedback    │ <- Dreams run during idle              │
//! │  │  Bridge            │                                        │
//! │  └────────┬───────────┘                                        │
//! │           │                                                     │
//! │           ▼                                                     │
//! │  - Update action priors                                        │
//! │  - Tag risky contexts                                          │
//! │  - Adjust confidence bounds                                    │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// A dream insight - a counterfactual that outperformed reality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamInsight {
    /// The context (state) where the insight applies
    pub context_hash: u64,

    /// Embedding of the original action
    pub original_action: Vec<f32>,

    /// Embedding of the better alternative action
    pub alternative_action: Vec<f32>,

    /// Φ improvement (alternative - original)
    pub phi_improvement: f64,

    /// How confident we are in this insight
    pub confidence: f64,

    /// When this insight was generated
    pub timestamp: u64,
}

impl DreamInsight {
    /// Create a new dream insight
    pub fn new(
        context_hash: u64,
        original_action: Vec<f32>,
        alternative_action: Vec<f32>,
        phi_improvement: f64,
    ) -> Self {
        Self {
            context_hash,
            original_action,
            alternative_action,
            phi_improvement,
            confidence: phi_improvement.min(1.0).max(0.0),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Action prior - bias toward certain actions in certain contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPrior {
    /// Context hash this prior applies to
    pub context_hash: u64,

    /// Direction in action space to prefer (normalized)
    pub preferred_direction: Vec<f32>,

    /// How strong this prior is (0-1)
    pub strength: f64,

    /// How many insights contributed to this prior
    pub evidence_count: usize,
}

/// Confidence adjustment for risky contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceAdjustment {
    /// Context hash
    pub context_hash: u64,

    /// Multiplicative adjustment (< 1.0 = reduce confidence)
    pub multiplier: f64,

    /// Why this adjustment exists
    pub reason: String,
}

/// Statistics for dream feedback effectiveness
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamFeedbackStats {
    /// Total insights processed
    pub total_insights: usize,

    /// Insights that led to prior updates
    pub priors_created: usize,

    /// Predictions made with dream priors
    pub dream_informed_predictions: usize,

    /// Predictions made without dream priors
    pub baseline_predictions: usize,

    /// Sum of Brier scores for dream-informed predictions
    pub dream_informed_brier_sum: f64,

    /// Sum of Brier scores for baseline predictions
    pub baseline_brier_sum: f64,
}

impl DreamFeedbackStats {
    /// Average Brier score for dream-informed predictions
    pub fn dream_informed_brier(&self) -> Option<f64> {
        if self.dream_informed_predictions > 0 {
            Some(self.dream_informed_brier_sum / self.dream_informed_predictions as f64)
        } else {
            None
        }
    }

    /// Average Brier score for baseline predictions
    pub fn baseline_brier(&self) -> Option<f64> {
        if self.baseline_predictions > 0 {
            Some(self.baseline_brier_sum / self.baseline_predictions as f64)
        } else {
            None
        }
    }

    /// Improvement from dream feedback (negative = better)
    pub fn brier_improvement(&self) -> Option<f64> {
        match (self.dream_informed_brier(), self.baseline_brier()) {
            (Some(dream), Some(baseline)) => Some(dream - baseline),
            _ => None,
        }
    }
}

/// The Dream Feedback Bridge
///
/// Connects dream insights to MAGI Loop predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamFeedbackBridge {
    /// Action priors learned from dreams
    action_priors: HashMap<u64, ActionPrior>,

    /// Confidence adjustments for risky contexts
    confidence_adjustments: HashMap<u64, ConfidenceAdjustment>,

    /// Recent insights (for analysis)
    recent_insights: VecDeque<DreamInsight>,

    /// Maximum insights to keep
    max_insights: usize,

    /// Minimum Φ improvement to create a prior
    min_phi_improvement: f64,

    /// Statistics
    stats: DreamFeedbackStats,
}

impl Default for DreamFeedbackBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl DreamFeedbackBridge {
    /// Create a new dream feedback bridge
    pub fn new() -> Self {
        Self {
            action_priors: HashMap::new(),
            confidence_adjustments: HashMap::new(),
            recent_insights: VecDeque::new(),
            max_insights: 1000,
            min_phi_improvement: 0.05,
            stats: DreamFeedbackStats::default(),
        }
    }

    /// Process a dream insight
    ///
    /// Returns true if the insight was significant enough to create/update a prior
    pub fn process_insight(&mut self, insight: DreamInsight) -> bool {
        self.stats.total_insights += 1;

        // Only process significant insights
        if insight.phi_improvement < self.min_phi_improvement {
            return false;
        }

        // Update or create action prior
        let context = insight.context_hash;
        let prior = self.action_priors.entry(context).or_insert(ActionPrior {
            context_hash: context,
            preferred_direction: insight.alternative_action.clone(),
            strength: 0.0,
            evidence_count: 0,
        });

        // Blend the new insight with existing prior
        prior.evidence_count += 1;
        let blend_weight = 1.0 / prior.evidence_count as f64;

        // Update preferred direction (exponential moving average)
        if prior.preferred_direction.len() == insight.alternative_action.len() {
            for (i, val) in insight.alternative_action.iter().enumerate() {
                prior.preferred_direction[i] = (1.0 - blend_weight as f32)
                    * prior.preferred_direction[i]
                    + blend_weight as f32 * val;
            }
        }

        // Update strength based on phi improvement
        prior.strength = (prior.strength * (1.0 - blend_weight)
            + insight.phi_improvement * blend_weight)
            .min(1.0);

        self.stats.priors_created += 1;

        // Store insight for analysis
        self.recent_insights.push_back(insight);
        if self.recent_insights.len() > self.max_insights {
            self.recent_insights.pop_front();
        }

        true
    }

    /// Get action prior for a context (if any)
    pub fn get_prior(&self, context_hash: u64) -> Option<&ActionPrior> {
        self.action_priors.get(&context_hash)
    }

    /// Get confidence adjustment for a context
    pub fn get_confidence_adjustment(&self, context_hash: u64) -> f64 {
        self.confidence_adjustments
            .get(&context_hash)
            .map(|adj| adj.multiplier)
            .unwrap_or(1.0)
    }

    /// Adjust a prediction confidence based on dream feedback
    ///
    /// Returns the adjusted confidence and whether dream priors were applied
    pub fn adjust_confidence(&self, base_confidence: f64, context_hash: u64) -> (f64, bool) {
        let adjustment = self.get_confidence_adjustment(context_hash);
        let has_prior = self.action_priors.contains_key(&context_hash);

        if has_prior {
            // If we have a prior, we're more confident (dreams taught us about this context)
            let boost = self
                .action_priors
                .get(&context_hash)
                .map(|p| p.strength * 0.1) // Up to 10% boost
                .unwrap_or(0.0);

            ((base_confidence + boost) * adjustment, true)
        } else {
            (base_confidence * adjustment, false)
        }
    }

    /// Record a prediction outcome for calibration tracking
    pub fn record_outcome(&mut self, context_hash: u64, brier_score: f64) {
        let was_dream_informed = self.action_priors.contains_key(&context_hash);

        if was_dream_informed {
            self.stats.dream_informed_predictions += 1;
            self.stats.dream_informed_brier_sum += brier_score;
        } else {
            self.stats.baseline_predictions += 1;
            self.stats.baseline_brier_sum += brier_score;
        }
    }

    /// Mark a context as risky (dreams found many failures here)
    pub fn mark_risky_context(&mut self, context_hash: u64, reason: String, severity: f64) {
        let multiplier = 1.0 - (severity * 0.5).min(0.5); // At most 50% reduction

        self.confidence_adjustments.insert(
            context_hash,
            ConfidenceAdjustment {
                context_hash,
                multiplier,
                reason,
            },
        );
    }

    /// Get statistics
    pub fn stats(&self) -> &DreamFeedbackStats {
        &self.stats
    }

    /// Get number of active priors
    pub fn num_priors(&self) -> usize {
        self.action_priors.len()
    }

    /// Get number of confidence adjustments
    pub fn num_adjustments(&self) -> usize {
        self.confidence_adjustments.len()
    }

    /// Clear old priors (decay over time)
    pub fn decay_priors(&mut self, decay_factor: f64) {
        self.action_priors.retain(|_, prior| {
            prior.strength *= decay_factor;
            prior.strength > 0.01
        });

        self.confidence_adjustments.retain(|_, adj| {
            adj.multiplier = 1.0 - (1.0 - adj.multiplier) * decay_factor;
            adj.multiplier < 0.99
        });
    }

    /// Generate a summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Dream Feedback Bridge Summary ===\n\n");

        report.push_str(&format!(
            "Total insights processed: {}\n",
            self.stats.total_insights
        ));
        report.push_str(&format!("Active priors: {}\n", self.num_priors()));
        report.push_str(&format!(
            "Confidence adjustments: {}\n",
            self.num_adjustments()
        ));

        report.push_str("\nCalibration:\n");
        if let Some(dream_brier) = self.stats.dream_informed_brier() {
            report.push_str(&format!(
                "  Dream-informed Brier: {:.4} (n={})\n",
                dream_brier, self.stats.dream_informed_predictions
            ));
        }
        if let Some(baseline_brier) = self.stats.baseline_brier() {
            report.push_str(&format!(
                "  Baseline Brier: {:.4} (n={})\n",
                baseline_brier, self.stats.baseline_predictions
            ));
        }
        if let Some(improvement) = self.stats.brier_improvement() {
            let direction = if improvement < 0.0 { "better" } else { "worse" };
            report.push_str(&format!(
                "  Dream feedback is {} by {:.4}\n",
                direction,
                improvement.abs()
            ));
        }

        report
    }
}

/// Compute a simple hash of a context vector
pub fn hash_context(context: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for &val in context {
        let bits = val.to_bits();
        hash ^= bits as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_feedback_bridge() {
        let mut bridge = DreamFeedbackBridge::new();

        // Create a dream insight
        let insight = DreamInsight::new(
            12345,
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            0.15, // 15% Φ improvement
        );

        // Process the insight
        assert!(bridge.process_insight(insight));

        // Check that a prior was created
        assert!(bridge.get_prior(12345).is_some());
        assert_eq!(bridge.num_priors(), 1);

        // Check confidence adjustment
        let (adjusted, was_informed) = bridge.adjust_confidence(0.7, 12345);
        assert!(was_informed);
        assert!(adjusted > 0.7); // Should be boosted
    }

    #[test]
    fn test_brier_tracking() {
        let mut bridge = DreamFeedbackBridge::new();

        // Add a prior for context 1
        let insight = DreamInsight::new(1, vec![0.1], vec![0.2], 0.1);
        bridge.process_insight(insight);

        // Record outcomes
        bridge.record_outcome(1, 0.1); // Dream-informed, good
        bridge.record_outcome(2, 0.3); // Baseline, worse

        assert_eq!(bridge.stats().dream_informed_predictions, 1);
        assert_eq!(bridge.stats().baseline_predictions, 1);

        let improvement = bridge.stats().brier_improvement().unwrap();
        assert!(improvement < 0.0); // Dream-informed is better
    }
}
