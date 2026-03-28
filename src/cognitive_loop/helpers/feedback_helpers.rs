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
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Scale(factor as f64));
        self.prediction_confidence = self.feedback_state.effective_confidence();
    }

    /// Record a hard set proposal for prediction_confidence and sync field.
    #[inline]
    pub(in crate::cognitive_loop) fn set_confidence(&mut self, source: &'static str, value: f32) {
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Set(value as f64));
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
