// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Grounded confidence gate for dream-derived feedback.
//!
//! The legacy `DreamFeedbackBridge` may use dream priors to alter confidence.
//! SYM-RSI routes those adjustments through this gate so generated evidence can
//! propose actions and reduce confidence, but cannot increase epistemic confidence
//! until an independent empirical validation has been recorded.

use super::dream_feedback::DreamFeedbackBridge;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct DreamConfidenceGate {
    /// Context -> digest of the recorded/replay evidence that validated it.
    validations: HashMap<u64, String>,
}

impl DreamConfidenceGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record independent empirical validation for a dream-informed context.
    /// Empty evidence digests are rejected so promotion stays provenance-bound.
    pub fn validate_context(&mut self, context_hash: u64, evidence_digest: impl Into<String>) -> bool {
        let digest = evidence_digest.into();
        if digest.trim().is_empty() {
            return false;
        }
        self.validations.insert(context_hash, digest);
        true
    }

    pub fn is_validated(&self, context_hash: u64) -> bool {
        self.validations.contains_key(&context_hash)
    }

    pub fn validation_digest(&self, context_hash: u64) -> Option<&str> {
        self.validations.get(&context_hash).map(String::as_str)
    }

    /// Apply legacy dream feedback while enforcing the epistemic promotion rule.
    ///
    /// - decreases are always preserved (dreams may motivate caution),
    /// - increases are clamped to the base confidence until empirically validated,
    /// - the boolean reports whether a dream prior participated in the adjustment.
    pub fn adjust_confidence(
        &self,
        bridge: &DreamFeedbackBridge,
        base_confidence: f64,
        context_hash: u64,
    ) -> (f64, bool) {
        let (dream_adjusted, dream_informed) =
            bridge.adjust_confidence(base_confidence, context_hash);

        if self.is_validated(context_hash) {
            (dream_adjusted, dream_informed)
        } else {
            (dream_adjusted.min(base_confidence), dream_informed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::dream_feedback::DreamInsight;

    #[test]
    fn unvalidated_dream_prior_cannot_raise_confidence() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.process_insight(DreamInsight::new(
            7,
            vec![0.1],
            vec![0.9],
            0.5,
        ));

        let gate = DreamConfidenceGate::new();
        let (adjusted, informed) = gate.adjust_confidence(&bridge, 0.6, 7);
        assert!(informed);
        assert_eq!(adjusted, 0.6);
    }

    #[test]
    fn risk_decrease_survives_without_validation() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.process_insight(DreamInsight::new(
            7,
            vec![0.1],
            vec![0.9],
            0.5,
        ));
        bridge.mark_risky_context(7, "counterfactual failures".into(), 1.0);

        let gate = DreamConfidenceGate::new();
        let (adjusted, _) = gate.adjust_confidence(&bridge, 0.8, 7);
        assert!(adjusted < 0.8);
    }

    #[test]
    fn empirical_validation_allows_promotion() {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.process_insight(DreamInsight::new(
            7,
            vec![0.1],
            vec![0.9],
            0.5,
        ));

        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_context(7, "sha256:recorded-evidence"));
        let (adjusted, informed) = gate.adjust_confidence(&bridge, 0.6, 7);
        assert!(informed);
        assert!(adjusted > 0.6);
    }

    #[test]
    fn empty_validation_digest_is_rejected() {
        let mut gate = DreamConfidenceGate::new();
        assert!(!gate.validate_context(7, "   "));
        assert!(!gate.is_validated(7));
    }
}
