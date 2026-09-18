// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dream Feedback Bridge
//!
//! Connects counterfactual dream insights to future action selection without
//! allowing generated evidence to silently become epistemic authority.
//!
//! Dream-derived information may:
//! - propose action priors,
//! - identify risky contexts,
//! - reduce confidence as a caution signal.
//!
//! Dream-derived information may **not** increase prediction confidence through
//! this public bridge API. Confidence promotion is reserved for the empirical
//! `DreamConfidenceGate`, which records independent validation provenance.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// A dream insight - a counterfactual that outperformed reality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamInsight {
    /// The context (state) where the insight applies.
    pub context_hash: u64,
    /// Embedding of the original action.
    pub original_action: Vec<f32>,
    /// Embedding of the better alternative action.
    pub alternative_action: Vec<f32>,
    /// Φ improvement (alternative - original).
    pub phi_improvement: f64,
    /// Confidence assigned to the generated insight itself. This is not an
    /// authorization to increase waking prediction confidence.
    pub confidence: f64,
    /// When this insight was generated.
    pub timestamp: u64,
}

impl DreamInsight {
    /// Create a new dream insight.
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
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    fn is_finite(&self) -> bool {
        self.phi_improvement.is_finite()
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && self.original_action.iter().all(|value| value.is_finite())
            && self
                .alternative_action
                .iter()
                .all(|value| value.is_finite())
    }
}

/// Action prior - bias toward certain actions in certain contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPrior {
    /// Context hash this prior applies to.
    pub context_hash: u64,
    /// Direction in action space to prefer (normalized).
    pub preferred_direction: Vec<f32>,
    /// How strong this prior is (0-1).
    pub strength: f64,
    /// How many generated insights contributed to this prior.
    pub evidence_count: usize,
}

/// Confidence adjustment for risky contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceAdjustment {
    /// Context hash.
    pub context_hash: u64,
    /// Multiplicative adjustment in [0.5, 1.0].
    pub multiplier: f64,
    /// Why this adjustment exists.
    pub reason: String,
}

/// Statistics for dream feedback effectiveness.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamFeedbackStats {
    /// Total insights presented to the bridge, including rejected invalid inputs.
    pub total_insights: usize,
    /// Insights that led to prior updates.
    pub priors_created: usize,
    /// Predictions made with dream priors.
    pub dream_informed_predictions: usize,
    /// Predictions made without dream priors.
    pub baseline_predictions: usize,
    /// Sum of finite Brier scores for dream-informed predictions.
    pub dream_informed_brier_sum: f64,
    /// Sum of finite Brier scores for baseline predictions.
    pub baseline_brier_sum: f64,
}

impl DreamFeedbackStats {
    pub fn dream_informed_brier(&self) -> Option<f64> {
        if self.dream_informed_predictions > 0 {
            Some(self.dream_informed_brier_sum / self.dream_informed_predictions as f64)
        } else {
            None
        }
    }

    pub fn baseline_brier(&self) -> Option<f64> {
        if self.baseline_predictions > 0 {
            Some(self.baseline_brier_sum / self.baseline_predictions as f64)
        } else {
            None
        }
    }

    /// Improvement from dream feedback (negative = better).
    pub fn brier_improvement(&self) -> Option<f64> {
        match (self.dream_informed_brier(), self.baseline_brier()) {
            (Some(dream), Some(baseline)) => Some(dream - baseline),
            _ => None,
        }
    }
}

/// Connects dream insights to future action selection and caution signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamFeedbackBridge {
    action_priors: HashMap<u64, ActionPrior>,
    confidence_adjustments: HashMap<u64, ConfidenceAdjustment>,
    recent_insights: VecDeque<DreamInsight>,
    max_insights: usize,
    min_phi_improvement: f64,
    stats: DreamFeedbackStats,
}

impl Default for DreamFeedbackBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl DreamFeedbackBridge {
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

    /// Process a dream insight as a proposal for future action search.
    ///
    /// Invalid/non-finite generated data is rejected before it can poison priors.
    /// A successful prior update still does not authorize confidence promotion.
    pub fn process_insight(&mut self, insight: DreamInsight) -> bool {
        self.stats.total_insights += 1;
        if !insight.is_finite() || insight.phi_improvement < self.min_phi_improvement {
            return false;
        }

        let context = insight.context_hash;
        let prior = self.action_priors.entry(context).or_insert(ActionPrior {
            context_hash: context,
            preferred_direction: insight.alternative_action.clone(),
            strength: 0.0,
            evidence_count: 0,
        });

        prior.evidence_count += 1;
        let blend_weight = 1.0 / prior.evidence_count as f64;
        if prior.preferred_direction.len() == insight.alternative_action.len() {
            for (i, val) in insight.alternative_action.iter().enumerate() {
                prior.preferred_direction[i] = (1.0 - blend_weight as f32)
                    * prior.preferred_direction[i]
                    + blend_weight as f32 * val;
            }
        }
        prior.strength = (prior.strength * (1.0 - blend_weight)
            + insight.phi_improvement * blend_weight)
            .clamp(0.0, 1.0);

        self.stats.priors_created += 1;
        self.recent_insights.push_back(insight);
        if self.recent_insights.len() > self.max_insights {
            self.recent_insights.pop_front();
        }
        true
    }

    pub fn get_prior(&self, context_hash: u64) -> Option<&ActionPrior> {
        self.action_priors.get(&context_hash)
    }

    /// Get the caution multiplier for a context.
    pub fn get_confidence_adjustment(&self, context_hash: u64) -> f64 {
        self.confidence_adjustments
            .get(&context_hash)
            .map(|adj| adj.multiplier)
            .filter(|multiplier| multiplier.is_finite())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0)
    }

    /// Public confidence adjustment for unvalidated dream feedback.
    ///
    /// Generated evidence may preserve or reduce confidence, never increase it.
    pub fn adjust_confidence(&self, base_confidence: f64, context_hash: u64) -> (f64, bool) {
        let base_confidence = finite_confidence(base_confidence);
        let (proposed, dream_informed) =
            self.proposed_confidence_adjustment(base_confidence, context_hash);
        (
            finite_confidence(proposed).min(base_confidence),
            dream_informed,
        )
    }

    /// Full dream-derived confidence proposal, including the historical positive
    /// prior contribution. Restricted to the parent RSI module so only the
    /// empirical `DreamConfidenceGate` can use it for promotion.
    pub(super) fn proposed_confidence_adjustment(
        &self,
        base_confidence: f64,
        context_hash: u64,
    ) -> (f64, bool) {
        let base_confidence = finite_confidence(base_confidence);
        let adjustment = self.get_confidence_adjustment(context_hash);
        let Some(prior) = self.action_priors.get(&context_hash) else {
            return (finite_confidence(base_confidence * adjustment), false);
        };

        let strength = if prior.strength.is_finite() {
            prior.strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let boost = strength * 0.1;
        (
            finite_confidence((base_confidence + boost) * adjustment),
            true,
        )
    }

    /// Record a finite prediction outcome for calibration tracking.
    pub fn record_outcome(&mut self, context_hash: u64, brier_score: f64) {
        if !brier_score.is_finite() {
            return;
        }
        let was_dream_informed = self.action_priors.contains_key(&context_hash);
        if was_dream_informed {
            self.stats.dream_informed_predictions += 1;
            self.stats.dream_informed_brier_sum += brier_score;
        } else {
            self.stats.baseline_predictions += 1;
            self.stats.baseline_brier_sum += brier_score;
        }
    }

    /// Mark a context as risky. Counterfactual evidence may motivate caution, but
    /// it may never create a confidence multiplier above 1.0.
    pub fn mark_risky_context(&mut self, context_hash: u64, reason: String, severity: f64) {
        if !severity.is_finite() {
            return;
        }
        let severity = severity.clamp(0.0, 1.0);
        let multiplier = 1.0 - severity * 0.5;
        self.confidence_adjustments.insert(
            context_hash,
            ConfidenceAdjustment {
                context_hash,
                multiplier,
                reason,
            },
        );
    }

    pub fn stats(&self) -> &DreamFeedbackStats {
        &self.stats
    }

    pub fn num_priors(&self) -> usize {
        self.action_priors.len()
    }

    pub fn num_adjustments(&self) -> usize {
        self.confidence_adjustments.len()
    }

    /// Decay generated priors/caution toward neutrality without permitting an
    /// invalid factor to amplify either signal.
    pub fn decay_priors(&mut self, decay_factor: f64) {
        if !decay_factor.is_finite() {
            return;
        }
        let decay_factor = decay_factor.clamp(0.0, 1.0);
        self.action_priors.retain(|_, prior| {
            prior.strength = if prior.strength.is_finite() {
                (prior.strength * decay_factor).clamp(0.0, 1.0)
            } else {
                0.0
            };
            prior.strength > 0.01
        });

        self.confidence_adjustments.retain(|_, adj| {
            adj.multiplier = if adj.multiplier.is_finite() {
                1.0 - (1.0 - adj.multiplier.clamp(0.0, 1.0)) * decay_factor
            } else {
                1.0
            };
            adj.multiplier < 0.99
        });
    }

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

#[inline]
fn finite_confidence(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute a deterministic exact-context hash.
///
/// This remains an identity/provenance key, not a semantic-similarity key.
pub fn hash_context(context: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &val in context {
        let bits = val.to_bits();
        hash ^= bits as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unvalidated_dream_prior_cannot_raise_public_confidence() {
        let mut bridge = DreamFeedbackBridge::new();
        let insight = DreamInsight::new(
            12345,
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            0.15,
        );
        assert!(bridge.process_insight(insight));
        assert!(bridge.get_prior(12345).is_some());

        let (adjusted, was_informed) = bridge.adjust_confidence(0.7, 12345);
        assert!(was_informed);
        assert_eq!(adjusted, 0.7);

        let (proposal, _) = bridge.proposed_confidence_adjustment(0.7, 12345);
        assert!(proposal > 0.7);
    }

    #[test]
    fn unvalidated_risk_reduction_is_preserved() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.process_insight(DreamInsight::new(7, vec![0.1], vec![0.9], 0.5));
        bridge.mark_risky_context(7, "counterfactual failures".into(), 1.0);

        let (adjusted, informed) = bridge.adjust_confidence(0.8, 7);
        assert!(informed);
        assert!(adjusted < 0.8);
    }

    #[test]
    fn invalid_generated_numerics_fail_conservative() {
        let mut bridge = DreamFeedbackBridge::new();
        assert!(!bridge.process_insight(DreamInsight::new(
            1,
            vec![0.1],
            vec![0.2],
            f64::NAN,
        )));
        assert!(!bridge.process_insight(DreamInsight::new(
            2,
            vec![f32::INFINITY],
            vec![0.2],
            0.5,
        )));
        assert_eq!(bridge.num_priors(), 0);

        bridge.mark_risky_context(3, "invalid".into(), f64::NAN);
        assert_eq!(bridge.num_adjustments(), 0);

        bridge.mark_risky_context(4, "negative severity".into(), -10.0);
        assert_eq!(bridge.get_confidence_adjustment(4), 1.0);
        assert_eq!(bridge.adjust_confidence(f64::NAN, 4).0, 0.0);
    }

    #[test]
    fn test_brier_tracking() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.process_insight(DreamInsight::new(1, vec![0.1], vec![0.2], 0.1));
        bridge.record_outcome(1, 0.1);
        bridge.record_outcome(1, f64::NAN);
        bridge.record_outcome(2, 0.3);

        assert_eq!(bridge.stats().dream_informed_predictions, 1);
        assert_eq!(bridge.stats().baseline_predictions, 1);
        assert!(bridge.stats().brier_improvement().unwrap() < 0.0);
    }
}
