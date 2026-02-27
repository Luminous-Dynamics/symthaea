//! Helper methods for attributed feedback variable mutations.
//!
//! These methods both record proposals (for Phase 2.2 attribution tracking)
//! AND apply the direct mutation (preserving exact current behavior).
//!
//! When Phase 2.3 (staged computation) is ready, the direct mutations will
//! be removed and `FeedbackState::integrate()` will become the sole authority.

use super::super::feedback_state::FeedbackProposal;

impl super::super::CognitiveLoopService {
    // ═══════════════════════════════════════════════════════════════════════
    // CONFIDENCE HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    /// Record and apply an additive delta to prediction_confidence.
    ///
    /// Usage: `self.adjust_confidence("subsystem_name", delta);`
    /// Equivalent to: `self.prediction_confidence = (self.prediction_confidence + delta).clamp(0.0, 1.0);`
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_confidence(
        &mut self,
        source: &'static str,
        delta: f32,
    ) {
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Add(delta as f64));
        self.prediction_confidence = (self.prediction_confidence + delta).clamp(0.0, 1.0);
    }

    /// Record and apply a multiplicative scale factor to prediction_confidence.
    ///
    /// Usage: `self.scale_confidence("subsystem_name", 0.98);`
    /// Equivalent to: `self.prediction_confidence = (self.prediction_confidence * factor).clamp(0.0, 1.0);`
    #[inline]
    pub(in crate::cognitive_loop) fn scale_confidence(
        &mut self,
        source: &'static str,
        factor: f32,
    ) {
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Scale(factor as f64));
        self.prediction_confidence = (self.prediction_confidence * factor).clamp(0.0, 1.0);
    }

    /// Record and apply a hard set of prediction_confidence.
    ///
    /// Used sparingly (inference mode init, confidence reset).
    #[inline]
    pub(in crate::cognitive_loop) fn set_confidence(
        &mut self,
        source: &'static str,
        value: f32,
    ) {
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Set(value as f64));
        self.prediction_confidence = value.clamp(0.0, 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LEARNING RATE HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    /// Record and apply an additive delta to fep_lr_boost.
    ///
    /// Usage: `self.adjust_lr("subsystem_name", delta);`
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_lr(
        &mut self,
        source: &'static str,
        delta: f32,
    ) {
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Add(delta as f64));
        self.fep_lr_boost = (self.fep_lr_boost + delta).clamp(1.0, 3.0);
    }

    /// Record and apply a multiplicative scale factor to fep_lr_boost.
    ///
    /// Usage: `self.scale_lr("subsystem_name", 1.05);`
    #[inline]
    pub(in crate::cognitive_loop) fn scale_lr(
        &mut self,
        source: &'static str,
        factor: f32,
    ) {
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Scale(factor as f64));
        self.fep_lr_boost = (self.fep_lr_boost * factor).clamp(1.0, 3.0);
    }

    /// Record and apply a hard set of fep_lr_boost.
    ///
    /// Used sparingly (discontinuity reset, etc).
    #[inline]
    pub(in crate::cognitive_loop) fn set_lr(
        &mut self,
        source: &'static str,
        value: f32,
    ) {
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Set(value as f64));
        self.fep_lr_boost = value.clamp(1.0, 3.0);
    }
}
