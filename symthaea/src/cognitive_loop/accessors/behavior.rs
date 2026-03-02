//! Flow state, emotion, curiosity, self-reflection, adaptive behavior, voice,
//! learning loop, and unified architecture accessors.

use crate::cognitive_loop::CognitiveLoopService;
use crate::consciousness::consciousness_unification::EmotionalPattern;
use crate::dynamics::temporal_signatures::ConsciousnessPattern;
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::voice::cognitive_bridge::VoiceConsciousnessSignals;
use crate::voice::voice_feedback::VoiceOutputMetrics;

use super::super::{
    ActionHint, AdaptiveBehavior, CognitiveDepth, CouplingQuality, CuriosityDrive,
    CycleLearningResult, EmotionContagion, FlowState, Recommendation, ReflectionSummary,
    ReflectionThresholds, ResponseStrategy, SelfAssessment, SelfReflection,
};
use crate::voice::voice_feedback::VoiceQualitySummary;

#[allow(dead_code)]
impl CognitiveLoopService {
    cognitive_accessors! {
        // ═══════════════════════════════════════════════════════════════════
        // FLOW STATE
        // ═══════════════════════════════════════════════════════════════════

        /// Check if currently in flow state
        pub fn in_flow(&self) -> bool { self.flow_state.in_flow }

        /// Get flow state intensity (0.0 to 1.0)
        pub fn flow_intensity(&self) -> f32 { self.flow_state.intensity }

        /// Get flow state streak (consecutive flow-compatible cycles)
        pub fn flow_streak(&self) -> u32 { self.flow_state.streak }

        /// Get current flow state reference
        pub(crate) fn flow_state(&self) -> &FlowState { &self.flow_state }

        /// Get flow learning boost multiplier
        pub fn flow_learning_boost(&self) -> f32 { self.flow_state.learning_boost }

        // ═══════════════════════════════════════════════════════════════════
        // EMOTION CONTAGION
        // ═══════════════════════════════════════════════════════════════════

        /// Get current emotional valence from content analysis
        pub fn emotional_valence(&self) -> f32 { self.emotion_contagion.smoothed_valence() }

        /// Get current emotional arousal
        pub fn emotional_arousal(&self) -> f32 { self.emotion_contagion.smoothed_arousal() }

        /// Get emotion-based pattern nudge suggestion
        pub fn emotion_pattern_nudge(&self) -> (Option<ConsciousnessPattern>, f32) { self.emotion_contagion.pattern_nudge() }

        /// Get emotion contagion reference
        pub(crate) fn emotion_contagion(&self) -> &EmotionContagion { &self.emotion_contagion }

        // ═══════════════════════════════════════════════════════════════════
        // CURIOSITY DRIVE
        // ═══════════════════════════════════════════════════════════════════

        /// Get current boredom level (0.0 to 1.0)
        pub fn boredom(&self) -> f32 { self.curiosity_drive.boredom }

        /// Get curiosity level (0.0 to 1.0)
        pub fn curiosity(&self) -> f32 { self.curiosity_drive.curiosity }

        /// Check if curiosity-triggered exploration should occur
        pub fn curiosity_should_explore(&self) -> bool { self.curiosity_drive.should_explore() }

        /// Get curiosity drive reference
        pub(crate) fn curiosity_drive(&self) -> &CuriosityDrive { &self.curiosity_drive }

        /// Get novelty bonus for learning
        pub fn novelty_bonus(&self) -> f32 { self.curiosity_drive.novelty_bonus }

        /// Check if the system is bored (needs new stimuli)
        pub fn is_bored(&self) -> bool { self.curiosity_drive.boredom > 0.5 }

        // ═══════════════════════════════════════════════════════════════════
        // SELF-REFLECTION
        // ═══════════════════════════════════════════════════════════════════

        /// Get current self-assessment
        pub fn self_assessment(&self) -> SelfAssessment { self.self_model_tier.self_reflection.self_assessment }

        /// Get self-reflection summary
        pub fn reflection_summary(&self) -> ReflectionSummary { self.self_model_tier.self_reflection.summary() }

        /// Get adapted thresholds from self-reflection
        pub fn adapted_thresholds(&self) -> ReflectionThresholds { self.self_model_tier.self_reflection.get_thresholds() }

        /// Get current recommendations from self-reflection
        pub fn recommendations(&self) -> &[Recommendation] { &self.self_model_tier.self_reflection.recommendations }

        /// Get number of reflections performed
        pub fn reflection_count(&self) -> u64 { self.self_model_tier.self_reflection.reflection_count }

        /// Get learning effectiveness score
        pub fn learning_effectiveness(&self) -> f32 { self.self_model_tier.self_reflection.learning_effectiveness() }

        /// Check if system needs calibration
        pub fn needs_calibration(&self) -> bool { self.self_model_tier.self_reflection.self_assessment == SelfAssessment::NeedsCalibration }

        /// Check if system is performing optimally
        pub fn is_optimal(&self) -> bool { self.self_model_tier.self_reflection.self_assessment == SelfAssessment::Optimal }

        /// Get self-reflection reference
        pub(crate) fn self_reflection(&self) -> &SelfReflection { &self.self_model_tier.self_reflection }

        // ═══════════════════════════════════════════════════════════════════
        // VOICE FEEDBACK (simple delegators)
        // ═══════════════════════════════════════════════════════════════════

        /// Get voice quality summary for external systems
        pub fn voice_feedback_summary(&self) -> VoiceQualitySummary { self.voice_feedback_bridge.summary() }

        /// Check if voice indicates uncertainty
        pub fn voice_indicates_uncertainty(&self) -> bool { self.voice_feedback_bridge.is_uncertain() }

        // ═══════════════════════════════════════════════════════════════════
        // MEGA-UNIFIED ARCHITECTURE
        // ═══════════════════════════════════════════════════════════════════

        /// Get current cognitive depth from thalamic routing
        pub fn cognitive_depth(&self) -> CognitiveDepth { self.cognitive_depth }

        /// Get thalamic routing statistics (reflex_rate, cortical_rate, deep_rate)
        pub fn thalamic_stats(&self) -> (f32, f32, f32) { self.thalamic_router.routing_stats() }

        /// Get the emotional pattern (Stable/Escalating/Calming/Volatile)
        pub fn emotional_pattern(&self) -> EmotionalPattern { self.unification_engine.emotional.detect_pattern() }

        /// Get natural language description of current emotional state
        pub fn emotional_description(&self) -> String { self.unification_engine.emotional.state().describe() }

        /// Get the conversation coherence tracker reference
        pub(crate) fn coherence_tracker(&self) -> &ConversationCoherenceTracker { &self.coherence_tracker }

        /// Get the coupling quality assessment
        pub fn coupling_quality(&self) -> CouplingQuality { self.active_inference_bridge.coupling_quality() }

        // ═══════════════════════════════════════════════════════════════════
        // CLOSED LEARNING LOOP
        // ═══════════════════════════════════════════════════════════════════

        /// Get the current response strategy
        pub fn current_strategy(&self) -> ResponseStrategy { self.closed_learning_loop.current_strategy }

        /// Get the best strategy according to Q-learning
        pub fn best_strategy(&self) -> ResponseStrategy { self.closed_learning_loop.best_strategy() }

        /// Get average reward from the learning loop
        pub fn average_reward(&self) -> f32 { self.closed_learning_loop.average_reward() }

        /// Get Q-values for all strategies
        pub fn strategy_q_values(&self) -> &[f32; 5] { self.closed_learning_loop.q_values() }

        /// Get strategy usage counts
        pub fn strategy_usage_counts(&self) -> &[u64; 5] { self.closed_learning_loop.strategy_counts() }

        /// Get the last learning result
        pub fn last_learning_result(&self) -> Option<&CycleLearningResult> { self.closed_learning_loop.last_result.as_ref() }

        // ═══════════════════════════════════════════════════════════════════
        // ADAPTIVE BEHAVIOR
        // ═══════════════════════════════════════════════════════════════════

        /// Get current adaptive behavior
        pub(crate) fn adaptive_behavior(&self) -> &AdaptiveBehavior { &self.adaptive_behavior }

        /// Get current action hint
        pub fn action_hint(&self) -> ActionHint { self.adaptive_behavior.action_hint }

        /// Check if system should seek more input/clarification
        pub fn should_seek_input(&self) -> bool { self.adaptive_behavior.should_seek_input() }

        /// Check if system is in a confident state
        pub fn is_confident(&self) -> bool { self.adaptive_behavior.is_confident() }

        /// Get description of current adaptive state
        pub fn state_description(&self) -> &'static str { self.adaptive_behavior.description() }

        /// Get speech rate multiplier for voice synthesis
        pub fn speech_rate_multiplier(&self) -> f32 { self.adaptive_behavior.speech_rate_multiplier }

        /// Get pause duration multiplier for voice synthesis
        pub fn pause_multiplier(&self) -> f32 { self.adaptive_behavior.pause_multiplier }

        /// Get attention sensitivity for input processing
        pub fn attention_sensitivity(&self) -> f32 { self.adaptive_behavior.attention_sensitivity }

        /// Get exploration factor for decision making
        pub fn exploration_factor(&self) -> f32 { self.adaptive_behavior.exploration_factor }
    }

    /// Check if emotional content is significant
    pub fn has_emotional_content(&self) -> bool {
        self.emotion_contagion.smoothed_valence().abs() > 0.2
    }

    /// Force an immediate reflection cycle
    pub fn force_reflect(&mut self) -> Vec<Recommendation> {
        self.self_model_tier.self_reflection.reflect()
    }

    /// Update voice feedback with synthesis output metrics
    pub fn update_voice_feedback(&mut self, metrics: VoiceOutputMetrics) {
        self.voice_feedback_bridge.update(metrics);
    }

    /// Update listener prediction feedback
    pub fn update_listener_prediction(&mut self, success: f32) {
        self.voice_feedback_bridge
            .update_listener_prediction(success);
    }

    /// Get Phase 16 consciousness signals for voice prosody modulation.
    ///
    /// Returns a compact struct containing unified quality, epistemic gating,
    /// dissipative health, coherence velocity, and consciousness level —
    /// the signals needed by `CognitivePacing::from_cycle_metadata()`.
    pub fn voice_consciousness_signals(&self) -> VoiceConsciousnessSignals {
        let (_, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        let consciousness_level =
            super::super::snapshot::ConsciousnessSnapshot::compute_consciousness_level(
                self.prediction_confidence as f32,
                self.coherence_bridge.smoothed_coherence(),
                self.flow_state.intensity,
                pattern_confidence,
            );

        VoiceConsciousnessSignals {
            unified_quality: self.stats.avg_unified_quality,
            epistemic_confidence: self.carryover.quality.last_epistemic_confidence,
            dissipative_gated: self.stats.dissipative_health_gated_count > 0
                && self.stats.total_cycles > 0
                && (self.stats.dissipative_health_gated_count as f32
                    / self.stats.total_cycles as f32)
                    > 0.5,
            dissipative_factor: self.carryover.quality.last_dissipative_health as f32,
            coherence_velocity: self.carryover.quality.coherence_velocity,
            cross_module_agreement: self.stats.avg_cross_module_agreement,
            consciousness_level: consciousness_level as f64,
        }
    }

    /// Map consciousness signals to the 12-channel `VoiceCognitiveState` used
    /// by the vocal tract pipeline.
    ///
    /// This bridges the cognitive loop's consciousness metrics to the vocal tract
    /// encoder's input format. Includes Phi (integrated information) and EFE
    /// (expected free energy) for affective prosody modulation.
    /// Does NOT run the pipeline (too expensive for 50Hz).
    pub fn voice_cognitive_state(&self) -> crate::voice::vocal_tract_encoder::VoiceCognitiveState {
        let signals = self.voice_consciousness_signals();
        let emotional = self.unification_engine.emotional.state();

        crate::voice::vocal_tract_encoder::VoiceCognitiveState {
            prediction_error: self.stats.avg_prediction_error,
            emotional_valence: emotional.valence as f32,
            emotional_arousal: emotional.arousal as f32,
            unified_quality: signals.unified_quality,
            epistemic_confidence: signals.epistemic_confidence,
            coherence_velocity: signals.coherence_velocity,
            cross_agreement: signals.cross_module_agreement,
            consciousness_level: signals.consciousness_level as f32,
            articulation_quality: self.voice_feedback_bridge.smoothed_articulation(),
            rate_stability: self.voice_feedback_bridge.rate_stability(),
            integrated_phi: self
                .carryover
                .consciousness
                .last_spectral_mip_phi
                .unwrap_or(0.5) as f32,
            expected_free_energy: self
                .fep_agent
                .last_fe_components
                .as_ref()
                .map(|fe| fe.total as f32)
                .unwrap_or(1.0),
        }
    }

    /// Get combined phi contribution from all feedback sources
    pub fn combined_phi_contribution(&self) -> f32 {
        self.coherence_bridge.phi_contribution()
            + self.voice_feedback_bridge.compute_phi_adjustment()
    }

    /// Get the prediction-outcome coupling Modulation Index
    pub fn modulation_index(&self) -> Option<f64> {
        self.active_inference_bridge.modulation_index()
    }

    /// Process input through the unified dialogue pipeline
    pub fn process_unified(
        &mut self,
        input: &str,
    ) -> crate::consciousness::consciousness_unification::UnifiedConsciousnessResult {
        self.unification_engine.process(input)
    }

    /// Get the current FEP free energy (if available)
    pub fn fep_free_energy(&self) -> Option<f64> {
        self.fep_agent
            .last_fe_components
            .as_ref()
            .map(|fe| fe.total)
    }

    /// Get combined learning rate modifier
    pub fn combined_learning_rate(&self) -> f32 {
        let coherence_lr = self.coherence_bridge.effective_learning_rate();
        let voice_modifier = self.voice_feedback_bridge.learning_rate_modifier();
        coherence_lr * voice_modifier
    }

    /// Inject external reward signal for the next cycle.
    /// Blended with internal prediction-error-based reward at 50% weight.
    /// Resets to 0.0 after consumption in the next cycle.
    pub fn provide_reward(&mut self, reward: f32) {
        self.social.external_reward = reward.clamp(-1.0, 1.0);
    }

    /// Inject social signals from Mind module's SocialCoherence.
    /// Called by the Symthaea facade after Mind.tick() computes social stats.
    pub fn set_social_signals(&mut self, trust: f32, cooperation_rate: f32) {
        self.social.social_trust = trust.clamp(0.0, 1.0);
        self.social.social_cooperation_rate = cooperation_rate.clamp(0.0, 1.0);
    }

    /// Set the relational Psi from an external dyad computation.
    /// This is called by the Symthaea facade after computing Phi_dyad.
    pub fn set_relational_psi(&mut self, psi: f64) {
        self.social.relational_psi = psi;
    }
}
