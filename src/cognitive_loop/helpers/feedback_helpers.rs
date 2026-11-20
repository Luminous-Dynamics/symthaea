// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helper methods for attributed feedback variable mutations.
//!
//! Each helper records a proposal AND syncs the field to the running consensus
//! (all proposals integrated via averaged adds + geometric mean scales).
//! This makes field reads order-independent: the value always reflects the
//! consensus of ALL proposals so far, not the most recent mutation.
//!
//! The old double-write bug had each helper both immediately mutate (order-
//! dependent) AND record a proposal (order-independent consensus at cycle end).
//! Now both paths converge: the field IS the running consensus.

use super::super::feedback_state::FeedbackProposal;

// Generated dream sources use a reserved prefix so new dream pathways cannot
// bypass this boundary by choosing a different source label.
const DREAM_GENERATED_CONFIDENCE_SOURCE: &str = "dream_phi_insight";
const DREAM_GENERATED_CONFIDENCE_PREFIX: &str = "dream_";

#[inline]
fn is_generated_dream_confidence_source(source: &'static str) -> bool {
    source == "dream" || source.starts_with(DREAM_GENERATED_CONFIDENCE_PREFIX)
}

#[inline]
fn authorize_additive_confidence_delta(source: &'static str, delta: f32) -> f32 {
    if !delta.is_finite() {
        0.0
    } else if is_generated_dream_confidence_source(source) && delta > 0.0 {
        0.0
    } else {
        delta
    }
}

#[inline]
fn authorize_confidence_scale_factor(source: &'static str, factor: f32) -> f32 {
    if !factor.is_finite() {
        1.0
    } else if is_generated_dream_confidence_source(source) {
        factor.clamp(0.0, 1.0)
    } else {
        factor
    }
}

#[inline]
fn authorize_confidence_set_value(
    source: &'static str,
    value: f32,
    current_confidence: f64,
) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }

    let value = value.clamp(0.0, 1.0);
    if is_generated_dream_confidence_source(source) {
        let current = if current_confidence.is_finite() {
            current_confidence.clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        Some(value.min(current))
    } else {
        Some(value)
    }
}

#[inline]
fn bounded_confidence_weight(confidence: f32) -> f32 {
    if confidence.is_finite() {
        confidence.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

impl super::super::CognitiveLoopService {
    // ═══════════════════════════════════════════════════════════════════════
    // CONFIDENCE HELPERS (f64)
    // ═══════════════════════════════════════════════════════════════════════

    /// Record an additive delta proposal for prediction_confidence and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_confidence(
        &mut self,
        source: &'static str,
        delta: f32,
    ) {
        let delta = authorize_additive_confidence_delta(source, delta);
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Add(delta as f64));
        self.prediction_confidence = self.feedback_state.effective_confidence();
    }

    /// Record a multiplicative scale proposal for prediction_confidence and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn scale_confidence(
        &mut self,
        source: &'static str,
        factor: f32,
    ) {
        let factor = authorize_confidence_scale_factor(source, factor);
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Scale(factor as f64));
        self.prediction_confidence = self.feedback_state.effective_confidence();
    }

    /// Record a hard set proposal for prediction_confidence and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn set_confidence(&mut self, source: &'static str, value: f32) {
        let current = self.feedback_state.effective_confidence();
        if let Some(value) = authorize_confidence_set_value(source, value, current) {
            self.feedback_state
                .confidence
                .propose(source, FeedbackProposal::Set(value as f64));
        }
        self.prediction_confidence = self.feedback_state.effective_confidence();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LEARNING RATE HELPERS (f64)
    // ═══════════════════════════════════════════════════════════════════════

    /// Record an additive delta proposal for fep_lr_boost and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_lr(&mut self, source: &'static str, delta: f32) {
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Add(delta as f64));
        self.fep.lr_boost = self.feedback_state.effective_lr_boost();
    }

    /// Record a multiplicative scale proposal for fep_lr_boost and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn scale_lr(&mut self, source: &'static str, factor: f32) {
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Scale(factor as f64));
        self.fep.lr_boost = self.feedback_state.effective_lr_boost();
    }

    /// Record a hard set proposal for fep_lr_boost and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn set_lr(&mut self, source: &'static str, value: f32) {
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Set(value as f64));
        self.fep.lr_boost = self.feedback_state.effective_lr_boost();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EXPLORATION HELPERS (f64)
    // ═══════════════════════════════════════════════════════════════════════

    /// Record an additive delta proposal for exploration_urge and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_exploration(
        &mut self,
        source: &'static str,
        delta: f32,
    ) {
        self.feedback_state
            .exploration
            .propose(source, FeedbackProposal::Add(delta as f64));
        self.behavior.curiosity_drive.exploration_urge =
            self.feedback_state.effective_exploration();
    }

    /// Record a multiplicative scale proposal for exploration_urge and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn scale_exploration(
        &mut self,
        source: &'static str,
        factor: f32,
    ) {
        self.feedback_state
            .exploration
            .propose(source, FeedbackProposal::Scale(factor as f64));
        self.behavior.curiosity_drive.exploration_urge =
            self.feedback_state.effective_exploration();
    }

    /// Record a hard set proposal for exploration_urge and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn set_exploration(&mut self, source: &'static str, value: f32) {
        self.feedback_state
            .exploration
            .propose(source, FeedbackProposal::Set(value as f64));
        self.behavior.curiosity_drive.exploration_urge =
            self.feedback_state.effective_exploration();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // THRESHOLD HELPERS (f64)
    // ═══════════════════════════════════════════════════════════════════════

    /// Record a multiplicative scale proposal for adaptive_threshold_scale and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn scale_threshold(&mut self, source: &'static str, factor: f32) {
        self.feedback_state
            .threshold
            .propose(source, FeedbackProposal::Scale(factor as f64));
        self.carryover.learning.adaptive_threshold_scale =
            self.feedback_state.effective_threshold();
    }

    /// Record an additive delta proposal for adaptive_threshold_scale and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_threshold(&mut self, source: &'static str, delta: f32) {
        self.feedback_state
            .threshold
            .propose(source, FeedbackProposal::Add(delta as f64));
        self.carryover.learning.adaptive_threshold_scale =
            self.feedback_state.effective_threshold();
    }

    /// Record a hard set proposal for adaptive_threshold_scale and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn set_threshold(&mut self, source: &'static str, value: f32) {
        self.feedback_state
            .threshold
            .propose(source, FeedbackProposal::Set(value as f64));
        self.carryover.learning.adaptive_threshold_scale =
            self.feedback_state.effective_threshold();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONFIDENCE-WEIGHTED VARIANTS
    // ═══════════════════════════════════════════════════════════════════════
    // These use `propose_weighted()` to scale the priority weight by a
    // confidence score (0.0–1.0). A low-confidence Safety signal (conf=0.3)
    // gets weight 0.9, potentially less than a high-confidence Cognitive
    // signal (1.0). Use these when the subsystem has a natural confidence
    // metric (e.g., contradiction strength, moral score, honest confidence).

    /// Adjust confidence with confidence-scaled priority weight.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_confidence_weighted(
        &mut self,
        source: &'static str,
        delta: f32,
        priority: super::super::feedback_state::Priority,
        confidence: f32,
    ) {
        let delta = authorize_additive_confidence_delta(source, delta);
        let confidence = bounded_confidence_weight(confidence);
        self.feedback_state.confidence.propose_weighted(
            source,
            FeedbackProposal::Add(delta as f64),
            priority,
            confidence,
        );
        self.prediction_confidence = self.feedback_state.effective_confidence();
    }

    /// Scale confidence with confidence-scaled priority weight.
    #[inline]
    pub(in crate::cognitive_loop) fn scale_confidence_weighted(
        &mut self,
        source: &'static str,
        factor: f32,
        priority: super::super::feedback_state::Priority,
        confidence: f32,
    ) {
        let factor = authorize_confidence_scale_factor(source, factor);
        let confidence = bounded_confidence_weight(confidence);
        self.feedback_state.confidence.propose_weighted(
            source,
            FeedbackProposal::Scale(factor as f64),
            priority,
            confidence,
        );
        self.prediction_confidence = self.feedback_state.effective_confidence();
    }

    /// Adjust exploration with confidence-scaled priority weight.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_exploration_weighted(
        &mut self,
        source: &'static str,
        delta: f32,
        priority: super::super::feedback_state::Priority,
        confidence: f32,
    ) {
        self.feedback_state.exploration.propose_weighted(
            source,
            FeedbackProposal::Add(delta as f64),
            priority,
            confidence,
        );
        self.behavior.curiosity_drive.exploration_urge =
            self.feedback_state.effective_exploration();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MID-CYCLE EFFECTIVE VALUE ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Current effective prediction_confidence: cycle-start snapshot + accumulated proposals.
    #[inline]
    pub(in crate::cognitive_loop) fn current_confidence(&mut self) -> f64 {
        self.feedback_state.effective_confidence()
    }

    /// Current effective fep_lr_boost: cycle-start snapshot + accumulated proposals.
    #[inline]
    pub(in crate::cognitive_loop) fn current_lr_boost(&mut self) -> f64 {
        self.feedback_state.effective_lr_boost()
    }

    /// Current effective exploration_urge: cycle-start snapshot + accumulated proposals.
    #[inline]
    pub(in crate::cognitive_loop) fn current_exploration(&mut self) -> f64 {
        self.feedback_state.effective_exploration()
    }

    /// Current effective adaptive_threshold_scale: cycle-start snapshot + accumulated proposals.
    #[inline]
    pub(in crate::cognitive_loop) fn current_threshold(&mut self) -> f64 {
        self.feedback_state.effective_threshold()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRIORITY-AWARE VARIANTS
    // ═══════════════════════════════════════════════════════════════════════

    /// Adjust confidence with an explicit priority tier.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_confidence_pri(
        &mut self,
        source: &'static str,
        delta: f32,
        priority: super::super::feedback_state::Priority,
    ) {
        let delta = authorize_additive_confidence_delta(source, delta);
        self.feedback_state.confidence.propose_with_priority(
            source,
            FeedbackProposal::Add(delta as f64),
            priority,
        );
        self.prediction_confidence = self.feedback_state.effective_confidence();
    }

    /// Scale confidence with an explicit priority tier.
    #[inline]
    pub(in crate::cognitive_loop) fn scale_confidence_pri(
        &mut self,
        source: &'static str,
        factor: f32,
        priority: super::super::feedback_state::Priority,
    ) {
        let factor = authorize_confidence_scale_factor(source, factor);
        self.feedback_state.confidence.propose_with_priority(
            source,
            FeedbackProposal::Scale(factor as f64),
            priority,
        );
        self.prediction_confidence = self.feedback_state.effective_confidence();
    }

    /// Adjust LR with an explicit priority tier.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_lr_pri(
        &mut self,
        source: &'static str,
        delta: f32,
        priority: super::super::feedback_state::Priority,
    ) {
        self.feedback_state.learning_rate.propose_with_priority(
            source,
            FeedbackProposal::Add(delta as f64),
            priority,
        );
        self.fep.lr_boost = self.feedback_state.effective_lr_boost();
    }

    /// Scale LR with an explicit priority tier.
    #[inline]
    pub(in crate::cognitive_loop) fn scale_lr_pri(
        &mut self,
        source: &'static str,
        factor: f32,
        priority: super::super::feedback_state::Priority,
    ) {
        self.feedback_state.learning_rate.propose_with_priority(
            source,
            FeedbackProposal::Scale(factor as f64),
            priority,
        );
        self.fep.lr_boost = self.feedback_state.effective_lr_boost();
    }

    /// Adjust exploration with an explicit priority tier.
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_exploration_pri(
        &mut self,
        source: &'static str,
        delta: f32,
        priority: super::super::feedback_state::Priority,
    ) {
        self.feedback_state.exploration.propose_with_priority(
            source,
            FeedbackProposal::Add(delta as f64),
            priority,
        );
        self.behavior.curiosity_drive.exploration_urge =
            self.feedback_state.effective_exploration();
    }

    /// Scale exploration with an explicit priority tier.
    #[inline]
    pub(in crate::cognitive_loop) fn scale_exploration_pri(
        &mut self,
        source: &'static str,
        factor: f32,
        priority: super::super::feedback_state::Priority,
    ) {
        self.feedback_state.exploration.propose_with_priority(
            source,
            FeedbackProposal::Scale(factor as f64),
            priority,
        );
        self.behavior.curiosity_drive.exploration_urge =
            self.feedback_state.effective_exploration();
    }

    /// Set exploration with an explicit priority tier.
    #[inline]
    pub(in crate::cognitive_loop) fn set_exploration_pri(
        &mut self,
        source: &'static str,
        value: f32,
        priority: super::super::feedback_state::Priority,
    ) {
        self.feedback_state.exploration.propose_with_priority(
            source,
            FeedbackProposal::Set(value as f64),
            priority,
        );
        self.behavior.curiosity_drive.exploration_urge =
            self.feedback_state.effective_exploration();
    }
}


#[cfg(test)]
mod dream_confidence_authority_tests {
    use super::*;

    #[test]
    fn generated_dream_confidence_can_cause_caution_but_not_promotion() {
        assert_eq!(
            authorize_additive_confidence_delta(DREAM_GENERATED_CONFIDENCE_SOURCE, 0.05),
            0.0
        );
        assert_eq!(
            authorize_additive_confidence_delta(DREAM_GENERATED_CONFIDENCE_SOURCE, -0.05),
            -0.05
        );
    }

    #[test]
    fn malformed_additive_confidence_deltas_fail_closed() {
        assert_eq!(
            authorize_additive_confidence_delta("calibration", f32::NAN),
            0.0
        );
        assert_eq!(
            authorize_additive_confidence_delta("calibration", f32::INFINITY),
            0.0
        );
    }

    #[test]
    fn independent_confidence_sources_keep_valid_deltas() {
        assert_eq!(
            authorize_additive_confidence_delta("calibration", 0.05),
            0.05
        );
        assert_eq!(
            authorize_additive_confidence_delta("observation", -0.05),
            -0.05
        );
    }

    #[test]
    fn every_dream_source_namespace_is_non_authorizing() {
        assert_eq!(
            authorize_additive_confidence_delta("dream_counterfactual", 0.03),
            0.0
        );
        assert_eq!(
            authorize_confidence_scale_factor("dream_wisdom", 1.4),
            1.0
        );
        assert_eq!(
            authorize_confidence_set_value("dream_planning", 0.95, 0.7),
            Some(0.7)
        );
        assert_eq!(authorize_additive_confidence_delta("dream", 0.03), 0.0);
    }

    #[test]
    fn generated_dream_scale_can_reduce_but_never_raise_confidence() {
        assert_eq!(
            authorize_confidence_scale_factor(DREAM_GENERATED_CONFIDENCE_SOURCE, 1.2),
            1.0
        );
        assert_eq!(
            authorize_confidence_scale_factor(DREAM_GENERATED_CONFIDENCE_SOURCE, 0.8),
            0.8
        );
        assert_eq!(
            authorize_confidence_scale_factor(DREAM_GENERATED_CONFIDENCE_SOURCE, -0.2),
            0.0
        );
        assert_eq!(authorize_confidence_scale_factor("calibration", 1.2), 1.2);
    }

    #[test]
    fn generated_dream_hard_set_is_capped_at_current_confidence() {
        assert_eq!(
            authorize_confidence_set_value(DREAM_GENERATED_CONFIDENCE_SOURCE, 0.9, 0.7),
            Some(0.7)
        );
        assert_eq!(
            authorize_confidence_set_value(DREAM_GENERATED_CONFIDENCE_SOURCE, 0.4, 0.7),
            Some(0.4)
        );
        assert_eq!(
            authorize_confidence_set_value("calibration", 0.9, 0.7),
            Some(0.9)
        );
    }

    #[test]
    fn invalid_scale_set_and_weight_values_fail_closed() {
        assert_eq!(
            authorize_confidence_scale_factor(DREAM_GENERATED_CONFIDENCE_SOURCE, f32::NAN),
            1.0
        );
        assert_eq!(
            authorize_confidence_set_value("calibration", f32::NAN, 0.7),
            None
        );
        assert_eq!(bounded_confidence_weight(f32::NAN), 0.0);
        assert_eq!(bounded_confidence_weight(1.5), 1.0);
    }
}
