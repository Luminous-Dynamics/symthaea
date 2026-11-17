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

#[inline]
fn finite_confidence(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

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

    /// Canonical claim identity; empty for legacy/exploration-only insights.
    #[serde(default)]
    pub claim_identity: String,

    /// Canonical scope identity; empty for legacy/exploration-only insights.
    #[serde(default)]
    pub scope_identity: String,

    /// Declared dream source-family identity.
    #[serde(default)]
    pub source_family_identity: String,

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
        let confidence = if phi_improvement.is_finite() {
            phi_improvement.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            context_hash,
            original_action,
            alternative_action,
            phi_improvement,
            confidence,
            claim_identity: String::new(),
            scope_identity: String::new(),
            source_family_identity: String::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Bind this insight to its claim, scope and dream source family.
    /// Unbound legacy insights remain exploration-only.
    pub fn with_claim_scope_and_source(
        mut self,
        claim_identity: impl Into<String>,
        scope_identity: impl Into<String>,
        source_family_identity: impl Into<String>,
    ) -> Self {
        self.claim_identity = claim_identity.into();
        self.scope_identity = scope_identity.into();
        self.source_family_identity = source_family_identity.into();
        self
    }

    fn is_well_formed(&self) -> bool {
        self.phi_improvement.is_finite()
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && {
                let all_empty = self.claim_identity.is_empty()
                    && self.scope_identity.is_empty()
                    && self.source_family_identity.is_empty();
                let all_bound = !self.claim_identity.trim().is_empty()
                    && !self.scope_identity.trim().is_empty()
                    && !self.source_family_identity.trim().is_empty();
                all_empty || all_bound
            }
            && !self.original_action.is_empty()
            && !self.alternative_action.is_empty()
            && self
                .original_action
                .iter()
                .chain(self.alternative_action.iter())
                .all(|value| value.is_finite())
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

    /// Canonical claim identity; empty for legacy/exploration-only priors.
    #[serde(default)]
    pub claim_identity: String,

    /// Canonical scope identity; empty for legacy/exploration-only priors.
    #[serde(default)]
    pub scope_identity: String,

    /// Dream source family that generated this prior.
    #[serde(default)]
    pub source_family_identity: String,
}

fn is_valid_action_prior(prior: &ActionPrior) -> bool {
    !prior.preferred_direction.is_empty()
        && prior.preferred_direction.iter().all(|value| value.is_finite())
        && prior.strength.is_finite()
        && (0.0..=1.0).contains(&prior.strength)
        && prior.evidence_count > 0
        && {
            let all_empty = prior.claim_identity.is_empty()
                && prior.scope_identity.is_empty()
                && prior.source_family_identity.is_empty();
            let all_bound = !prior.claim_identity.trim().is_empty()
                && !prior.scope_identity.trim().is_empty()
                && !prior.source_family_identity.trim().is_empty();
            all_empty || all_bound
        }
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

        // Deserialized/generated inputs must be finite and structurally usable before
        // they can influence a future action prior.
        if !self.min_phi_improvement.is_finite()
            || self.min_phi_improvement < 0.0
            || !insight.is_well_formed()
            || insight.phi_improvement < self.min_phi_improvement
        {
            return false;
        }

        // Replace malformed legacy state rather than blending it into a new prior.
        let context = insight.context_hash;
        if self
            .action_priors
            .get(&context)
            .is_some_and(|prior| !is_valid_action_prior(prior))
        {
            self.action_priors.remove(&context);
        }

        // A fast-hash collision or changed action dimension must not blend unrelated
        // action spaces into one preferred direction.
        if self
            .action_priors
            .get(&context)
            .is_some_and(|prior| {
                prior.preferred_direction.len() != insight.alternative_action.len()
                    || prior.claim_identity != insight.claim_identity
                    || prior.scope_identity != insight.scope_identity
                    || prior.source_family_identity != insight.source_family_identity
            })
        {
            return false;
        }

        // Update or create action prior
        let prior = self.action_priors.entry(context).or_insert(ActionPrior {
            context_hash: context,
            preferred_direction: insight.alternative_action.clone(),
            strength: 0.0,
            evidence_count: 0,
            claim_identity: insight.claim_identity.clone(),
            scope_identity: insight.scope_identity.clone(),
            source_family_identity: insight.source_family_identity.clone(),
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
        self.action_priors
            .get(&context_hash)
            .filter(|prior| is_valid_action_prior(prior))
    }

    /// Get confidence adjustment for a context
    pub fn get_confidence_adjustment(&self, context_hash: u64) -> f64 {
        self.confidence_adjustments
            .get(&context_hash)
            .map(|adj| adj.multiplier)
            .unwrap_or(1.0)
    }

    /// Public confidence adjustment. Without an empirical authority capability,
    /// dream priors may preserve or reduce confidence but never increase it.
    pub fn adjust_confidence(&self, base_confidence: f64, context_hash: u64) -> (f64, bool) {
        let base = finite_confidence(base_confidence);
        let adjustment = finite_confidence(self.get_confidence_adjustment(context_hash));
        let informed = self.get_prior(context_hash).is_some();
        (finite_confidence(base * adjustment), informed)
    }

    /// Full dream-derived proposal. Restricted to the parent recursive-improvement
    /// module so only DreamConfidenceGate can use it to authorize a promotion.
    pub(super) fn proposed_confidence_adjustment(
        &self,
        base_confidence: f64,
        context_hash: u64,
        claim_identity: &str,
        scope_identity: &str,
        source_family_identity: &str,
    ) -> (f64, bool) {
        let base = finite_confidence(base_confidence);
        let adjustment = finite_confidence(self.get_confidence_adjustment(context_hash));
        let Some(prior) = self.get_prior(context_hash).filter(|prior| {
            prior.claim_identity == claim_identity
                && prior.scope_identity == scope_identity
                && prior.source_family_identity == source_family_identity
        }) else {
            return (finite_confidence(base * adjustment), false);
        };

        let strength = if prior.strength.is_finite() {
            prior.strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let boost = strength * 0.1;
        (finite_confidence((base + boost) * adjustment), true)
    }

    /// Record a prediction outcome for calibration tracking
    pub fn record_outcome(&mut self, context_hash: u64, brier_score: f64) {
        // Brier score is defined on [0, 1]. Invalid observations must not poison
        // the persisted calibration comparison.
        if !brier_score.is_finite() || !(0.0..=1.0).contains(&brier_score) {
            return;
        }
        let was_dream_informed = self.get_prior(context_hash).is_some();

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
        // Invalid severity is treated as maximum caution; valid values are bounded.
        let severity = if severity.is_finite() {
            severity.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let multiplier = 1.0 - severity * 0.5; // At most 50% reduction

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
        self.action_priors
            .values()
            .filter(|prior| is_valid_action_prior(prior))
            .count()
    }

    /// Get number of confidence adjustments
    pub fn num_adjustments(&self) -> usize {
        self.confidence_adjustments.len()
    }

    /// Clear old priors (decay over time)
    pub fn decay_priors(&mut self, decay_factor: f64) {
        // Decay may attenuate priors but may not amplify them. A non-finite
        // caller value is treated as a no-op rather than corrupting risk memory.
        let decay_factor = if decay_factor.is_finite() {
            decay_factor.clamp(0.0, 1.0)
        } else {
            1.0
        };
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
        assert!((adjusted - 0.7).abs() < 1e-12); // Dream priors do not promote confidence
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
    #[test]
    fn malformed_confidence_state_cannot_promote_or_poison_confidence() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.confidence_adjustments.insert(
            7,
            ConfidenceAdjustment {
                context_hash: 7,
                multiplier: 4.0,
                reason: "malformed persisted multiplier".into(),
            },
        );
        let (bounded, informed) = bridge.adjust_confidence(0.7, 7);
        assert!(informed);
        assert!((bounded - 0.7).abs() < 1e-12);

        bridge.confidence_adjustments.get_mut(&7).unwrap().multiplier = f64::NAN;
        let (invalid, _) = bridge.adjust_confidence(0.7, 7);
        assert_eq!(invalid, 0.0);
        let (invalid_base, _) = bridge.adjust_confidence(f64::NAN, 99);
        assert_eq!(invalid_base, 0.0);
    }

    #[test]
    fn risky_context_inputs_are_bounded_and_invalid_severity_fails_cautiously() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.mark_risky_context(1, "negative severity".into(), -3.0);
        assert_eq!(bridge.get_confidence_adjustment(1), 1.0);
        bridge.mark_risky_context(2, "invalid severity".into(), f64::NAN);
        assert_eq!(bridge.get_confidence_adjustment(2), 0.5);
    }

    #[test]
    fn invalid_decay_factor_does_not_erase_risk_adjustment() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.mark_risky_context(3, "risk".into(), 1.0);
        bridge.decay_priors(f64::NAN);
        assert_eq!(bridge.get_confidence_adjustment(3), 0.5);
    }

    #[test]
    fn non_finite_or_out_of_range_brier_scores_are_ignored() {
        let mut bridge = DreamFeedbackBridge::new();
        assert!(bridge.process_insight(DreamInsight::new(
            11,
            vec![0.1],
            vec![0.2],
            0.2,
        )));

        bridge.record_outcome(11, f64::NAN);
        bridge.record_outcome(11, f64::INFINITY);
        bridge.record_outcome(11, 1.1);
        assert_eq!(bridge.stats().dream_informed_predictions, 0);
        assert_eq!(bridge.stats().dream_informed_brier_sum, 0.0);

        bridge.record_outcome(11, 0.25);
        assert_eq!(bridge.stats().dream_informed_predictions, 1);
        assert_eq!(bridge.stats().dream_informed_brier_sum, 0.25);
    }

    #[test]
    fn malformed_persisted_minimum_phi_threshold_rejects_insights() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.min_phi_improvement = f64::NAN;
        let insight = DreamInsight::new(12, vec![0.1], vec![0.2], 0.2);

        assert!(!bridge.process_insight(insight));
        assert_eq!(bridge.num_priors(), 0);
    }

    #[test]
    fn malformed_deserialized_dream_insights_cannot_create_action_priors() {
        let mut bridge = DreamFeedbackBridge::new();
        let malformed = [
            DreamInsight {
                context_hash: 1,
                original_action: vec![0.1],
                alternative_action: vec![0.2],
                phi_improvement: f64::NAN,
                confidence: 0.5,
                claim_identity: String::new(),
                scope_identity: String::new(),
                source_family_identity: String::new(),
                timestamp: 1,
            },
            DreamInsight {
                context_hash: 2,
                original_action: vec![0.1],
                alternative_action: vec![f32::INFINITY],
                phi_improvement: 0.2,
                confidence: 0.2,
                claim_identity: String::new(),
                scope_identity: String::new(),
                source_family_identity: String::new(),
                timestamp: 2,
            },
            DreamInsight {
                context_hash: 3,
                original_action: vec![],
                alternative_action: vec![0.2],
                phi_improvement: 0.2,
                confidence: 0.2,
                claim_identity: String::new(),
                scope_identity: String::new(),
                source_family_identity: String::new(),
                timestamp: 3,
            },
        ];

        for insight in malformed {
            assert!(!bridge.process_insight(insight));
        }
        assert_eq!(bridge.num_priors(), 0);
    }

    #[test]
    fn unbound_legacy_prior_cannot_receive_claim_scoped_promotion() {
        let mut bridge = DreamFeedbackBridge::new();
        assert!(bridge.process_insight(DreamInsight::new(
            101,
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            0.2,
        )));
        let prior = bridge.get_prior(101).unwrap();
        assert!(prior.claim_identity.is_empty());
        let (proposed, informed) = bridge.proposed_confidence_adjustment(
            0.6,
            101,
            "claim:alpha",
            "scope:site-a",
            "family:dream",
        );
        assert_eq!(proposed, 0.6);
        assert!(!informed);
    }

    #[test]
    fn different_claim_scope_or_source_cannot_blend_into_a_prior() {
        let mut bridge = DreamFeedbackBridge::new();
        let first = DreamInsight::new(102, vec![0.0, 0.0], vec![1.0, 0.0], 0.2)
            .with_claim_scope_and_source("claim:alpha", "scope:site-a", "family:dream");
        let wrong_claim = DreamInsight::new(102, vec![0.0, 0.0], vec![1.0, 0.0], 0.3)
            .with_claim_scope_and_source("claim:beta", "scope:site-a", "family:dream");
        let wrong_scope = DreamInsight::new(102, vec![0.0, 0.0], vec![1.0, 0.0], 0.3)
            .with_claim_scope_and_source("claim:alpha", "scope:site-b", "family:dream");
        let wrong_source = DreamInsight::new(102, vec![0.0, 0.0], vec![1.0, 0.0], 0.3)
            .with_claim_scope_and_source("claim:alpha", "scope:site-a", "family:other-dream");

        assert!(bridge.process_insight(first));
        assert!(!bridge.process_insight(wrong_claim));
        assert!(!bridge.process_insight(wrong_scope));
        assert!(!bridge.process_insight(wrong_source));
        let prior = bridge.get_prior(102).unwrap();
        assert_eq!(prior.claim_identity, "claim:alpha");
        assert_eq!(prior.scope_identity, "scope:site-a");
        assert_eq!(prior.source_family_identity, "family:dream");
    }

    #[test]
    fn malformed_persisted_action_priors_are_not_retrievable() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.action_priors.insert(
            99,
            ActionPrior {
                context_hash: 99,
                preferred_direction: vec![f32::NAN],
                strength: 0.8,
                evidence_count: 1,
                claim_identity: String::new(),
                scope_identity: String::new(),
                source_family_identity: String::new(),
            },
        );

        assert!(bridge.get_prior(99).is_none());
        assert_eq!(bridge.num_priors(), 0);
        let (_, was_informed) = bridge.adjust_confidence(0.7, 99);
        assert!(!was_informed);
    }

    #[test]
    fn action_prior_context_collision_cannot_blend_different_dimensions() {
        let mut bridge = DreamFeedbackBridge::new();
        let first = DreamInsight::new(7, vec![0.1], vec![0.2, 0.3], 0.2);
        let second = DreamInsight::new(7, vec![0.1], vec![0.2, 0.3, 0.4], 0.2);

        assert!(bridge.process_insight(first));
        assert!(!bridge.process_insight(second));
        assert_eq!(
            bridge.get_prior(7).map(|prior| prior.preferred_direction.len()),
            Some(2)
        );
    }


}
