//! Helper methods for attributed feedback variable mutations.
//!
//! These methods both record proposals (for attribution tracking)
//! AND apply the direct mutation. The proposal system provides
//! consensus-smoothed alternatives at cycle end.

use super::super::feedback_state::FeedbackProposal;

impl super::super::CognitiveLoopService {
    // ═══════════════════════════════════════════════════════════════════════
    // CONFIDENCE HELPERS (f64)
    // ═══════════════════════════════════════════════════════════════════════

    /// Record and apply an additive delta to prediction_confidence.
    ///
    /// Usage: `self.adjust_confidence("subsystem_name", delta);`
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_confidence(
        &mut self,
        source: &'static str,
        delta: f32,
    ) {
        let delta64 = delta as f64;
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Add(delta64));
        self.prediction_confidence = (self.prediction_confidence + delta64).clamp(0.01, 0.99);
    }

    /// Record and apply a multiplicative scale factor to prediction_confidence.
    ///
    /// Usage: `self.scale_confidence("subsystem_name", 0.98);`
    #[inline]
    pub(in crate::cognitive_loop) fn scale_confidence(
        &mut self,
        source: &'static str,
        factor: f32,
    ) {
        let factor64 = factor as f64;
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Scale(factor64));
        self.prediction_confidence = (self.prediction_confidence * factor64).clamp(0.01, 0.99);
    }

    /// Record and apply a hard set of prediction_confidence.
    ///
    /// Used sparingly (inference mode init, confidence reset).
    #[inline]
    pub(in crate::cognitive_loop) fn set_confidence(&mut self, source: &'static str, value: f32) {
        let value64 = value as f64;
        self.feedback_state
            .confidence
            .propose(source, FeedbackProposal::Set(value64));
        self.prediction_confidence = value64.clamp(0.01, 0.99);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LEARNING RATE HELPERS (f64)
    // ═══════════════════════════════════════════════════════════════════════

    /// Record and apply an additive delta to fep_lr_boost.
    ///
    /// Usage: `self.adjust_lr("subsystem_name", delta);`
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_lr(&mut self, source: &'static str, delta: f32) {
        let delta64 = delta as f64;
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Add(delta64));
        self.fep.lr_boost = (self.fep.lr_boost + delta64).clamp(1.0, 3.0);
    }

    /// Record and apply a multiplicative scale factor to fep_lr_boost.
    ///
    /// Usage: `self.scale_lr("subsystem_name", 1.05);`
    #[inline]
    pub(in crate::cognitive_loop) fn scale_lr(&mut self, source: &'static str, factor: f32) {
        let factor64 = factor as f64;
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Scale(factor64));
        self.fep.lr_boost = (self.fep.lr_boost * factor64).clamp(1.0, 3.0);
    }

    /// Record and apply a hard set of fep_lr_boost.
    ///
    /// Used sparingly (discontinuity reset, etc).
    #[inline]
    pub(in crate::cognitive_loop) fn set_lr(&mut self, source: &'static str, value: f32) {
        let value64 = value as f64;
        self.feedback_state
            .learning_rate
            .propose(source, FeedbackProposal::Set(value64));
        self.fep.lr_boost = value64.clamp(1.0, 3.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EXPLORATION HELPERS (f64)
    // ═══════════════════════════════════════════════════════════════════════

    /// Record and apply an additive delta to exploration_urge.
    ///
    /// Usage: `self.adjust_exploration("subsystem_name", delta);`
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_exploration(
        &mut self,
        source: &'static str,
        delta: f32,
    ) {
        let delta64 = delta as f64;
        self.feedback_state
            .exploration
            .propose(source, FeedbackProposal::Add(delta64));
        self.curiosity_drive.exploration_urge =
            (self.curiosity_drive.exploration_urge + delta64).clamp(0.0, 1.0);
    }

    /// Record and apply a multiplicative scale factor to exploration_urge.
    ///
    /// Usage: `self.scale_exploration("subsystem_name", 0.9);`
    #[inline]
    pub(in crate::cognitive_loop) fn scale_exploration(
        &mut self,
        source: &'static str,
        factor: f32,
    ) {
        let factor64 = factor as f64;
        self.feedback_state
            .exploration
            .propose(source, FeedbackProposal::Scale(factor64));
        self.curiosity_drive.exploration_urge =
            (self.curiosity_drive.exploration_urge * factor64).clamp(0.0, 1.0);
    }

    /// Record and apply a hard set of exploration_urge.
    ///
    /// Used sparingly (seizure protection freeze, etc).
    #[inline]
    pub(in crate::cognitive_loop) fn set_exploration(&mut self, source: &'static str, value: f32) {
        let value64 = value as f64;
        self.feedback_state
            .exploration
            .propose(source, FeedbackProposal::Set(value64));
        self.curiosity_drive.exploration_urge = value64.clamp(0.0, 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // THRESHOLD HELPERS (f64)
    // ═══════════════════════════════════════════════════════════════════════

    /// Record and apply a multiplicative scale factor to adaptive_threshold_scale.
    ///
    /// Usage: `self.scale_threshold("subsystem_name", 0.95);`
    #[inline]
    pub(in crate::cognitive_loop) fn scale_threshold(&mut self, source: &'static str, factor: f32) {
        let factor64 = factor as f64;
        self.feedback_state
            .threshold
            .propose(source, FeedbackProposal::Scale(factor64));
        self.carryover.learning.adaptive_threshold_scale =
            (self.carryover.learning.adaptive_threshold_scale * factor64).clamp(0.5, 2.0);
    }

    /// Record and apply an additive delta to adaptive_threshold_scale.
    ///
    /// Usage: `self.adjust_threshold("homeostasis_drift", 0.02);`
    #[inline]
    pub(in crate::cognitive_loop) fn adjust_threshold(&mut self, source: &'static str, delta: f32) {
        let delta64 = delta as f64;
        self.feedback_state
            .threshold
            .propose(source, FeedbackProposal::Add(delta64));
        self.carryover.learning.adaptive_threshold_scale =
            (self.carryover.learning.adaptive_threshold_scale + delta64).clamp(0.5, 2.0);
    }

    /// Record and apply a hard set of adaptive_threshold_scale.
    ///
    /// Used for consensus writeback at cycle start.
    #[inline]
    pub(in crate::cognitive_loop) fn set_threshold(&mut self, source: &'static str, value: f32) {
        let value64 = value as f64;
        self.feedback_state
            .threshold
            .propose(source, FeedbackProposal::Set(value64));
        self.carryover.learning.adaptive_threshold_scale = value64.clamp(0.5, 2.0);
    }
}
