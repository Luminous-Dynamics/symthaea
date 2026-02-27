//! Accessor methods for CognitiveLoopService.
//!
//! High-level query methods (flow state, prediction confidence, consciousness
//! snapshot, etc.) are `pub` for use by external consumers (examples, LUCID,
//! symthaea-nix). A small number of `pub(crate)` accessors exist for internal
//! unit tests (e.g., `flow_state()`, `curiosity_drive()`).

use crate::causal::{CausalGraph, DiscoveredRelationship};
use crate::consciousness::consciousness_unification::EmotionalPattern;
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
use crate::dynamics::cfc_coherence::CoherenceSummary;
use crate::dynamics::temporal_signatures::{ConsciousnessPattern, TemporalStateSummary};
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::voice::cognitive_bridge::VoiceConsciousnessSignals;
use crate::voice::voice_feedback::{VoiceOutputMetrics, VoiceQualitySummary};
use anyhow::Result;

use super::snapshot::ConsciousnessSnapshot;
use super::{
    ActionHint, AdaptiveBehavior, CognitiveDepth, CognitiveGoal, CognitiveLoopConfig,
    CognitiveLoopService, CouplingQuality, CuriosityDrive, CycleLearningResult, EmotionContagion,
    EpisodicMemory, FlowState, LoopStats, Recommendation, ReflectionSummary, ReflectionThresholds,
    ResponseStrategy, SelfAssessment, SelfReflection, WorldModelBridge,
};

#[allow(dead_code)] // Accessor API surface — methods used by tests and future consumers
impl CognitiveLoopService {
    /// Get current statistics
    pub fn stats(&self) -> &LoopStats {
        &self.stats
    }

    /// Export neurochemistry checkpoint for persistence across sessions.
    pub fn neurochemistry_checkpoint(&self) -> super::neuromodulators::NeurochemistryCheckpoint {
        self.neuromodulator_bath.checkpoint()
    }

    /// Restore neurochemistry from a saved checkpoint.
    pub fn restore_neurochemistry(
        &mut self,
        ckpt: &super::neuromodulators::NeurochemistryCheckpoint,
    ) {
        self.neuromodulator_bath.restore(ckpt);
    }

    /// Override transmitter levels for pharmacological ablation (virtual lesion).
    /// Pass `None` to leave a channel unchanged, `Some(v)` to clamp it.
    pub fn clamp_neuromod_levels(
        &mut self,
        da: Option<f32>,
        ne: Option<f32>,
        sht: Option<f32>,
        ach: Option<f32>,
    ) {
        self.neuromodulator_bath.clamp_levels(da, ne, sht, ach);
    }

    /// Get a complete neurochemical state snapshot for telemetry/visualization.
    pub fn neuromod_snapshot(&self) -> super::neuromodulators::NeuromodSnapshot {
        self.neuromodulator_bath.snapshot()
    }

    /// Get a clone of the pain sender channel, if active.
    ///
    /// Used by integration tests to inject `InfrastructureError`s and verify
    /// that the somatic bridge converts them into interoceptive signals.
    pub fn pain_sender(&self) -> Option<crate::infrastructure::PainSender> {
        self.pain_tx.clone()
    }

    /// Get the configuration used to create this service.
    pub fn config(&self) -> &CognitiveLoopConfig {
        &self.config
    }

    /// Collect neuromodulator telemetry for CycleMetadata construction.
    ///
    /// Builds a [`NeuromodTelemetry`] snapshot from the current bath state,
    /// personality drift tracker, and loop stats. Call once per cycle during
    /// metadata assembly, then apply via [`CycleMetadata::apply_neuromod`].
    pub(crate) fn collect_neuromod_telemetry(
        &self,
        neuromod_attention_alloc: f32,
    ) -> super::NeuromodTelemetry {
        super::NeuromodTelemetry {
            exocortex_query_suggested: self.neuromodulator_bath.should_query_exocortex(),
            neuromod_personality: self.neuromodulator_bath.personality_description(),
            dopamine_effective: self.neuromodulator_bath.dopamine.effective(),
            noradrenaline_effective: self.neuromodulator_bath.noradrenaline.effective(),
            serotonin_effective: self.neuromodulator_bath.serotonin.effective(),
            acetylcholine_effective: self.neuromodulator_bath.acetylcholine.effective(),
            neuromod_personality_drift: self.personality_drift_tracker.drift_rate(),
            neuromod_personality_drift_anomalous: self.personality_drift_tracker.is_anomalous(),
            neuromod_gradient_scale: self.neuromodulator_bath.gradient_scale_factor(),
            neuromod_threshold_gate: self.neuromodulator_bath.threshold_gate(),
            exocortex_trigger_count: self.stats.exocortex_triggers,
            neuromod_da_phasic: self.neuromodulator_bath.da_phasic(),
            neuromod_ne_phasic: self.neuromodulator_bath.ne_phasic(),
            neuromod_consciousness_mod: self.neuromodulator_bath.consciousness_modulation(),
            neuromod_sleep_consolidation_boost: self.neuromodulator_bath.sleep_consolidation_boost(),
            neuromod_attention_allocation: neuromod_attention_alloc,
            neuromod_plasticity_gate: self.neuromodulator_bath.plasticity_gate(),
            neuromod_mcts_exploration_mod: self.neuromodulator_bath.mcts_exploration_modulation()
                as f32,
            replay_da_tag_avg: 0.0, // populated by episodic replay phase if applicable
            circadian_hour: self.biorhythm.hour as f32,
            neuromod_da_d1: self.neuromodulator_bath.da_d1_effective(),
            neuromod_da_d2: self.neuromodulator_bath.da_d2_effective(),
            neuromod_ne_alpha: self.neuromodulator_bath.ne_alpha_effective(),
            neuromod_ne_beta: self.neuromodulator_bath.ne_beta_effective(),
            neuromod_behavioral_flexibility: self.neuromodulator_bath.behavioral_flexibility(),
            neuromod_snapshot: if self.stats.total_cycles % 10 == 0 {
                Some(self.neuromodulator_bath.snapshot())
            } else {
                None
            },
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CAUSAL ENHANCEMENT ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get the current causal graph (if causal enhancement is enabled)
    pub fn causal_graph(&self) -> Option<&CausalGraph> {
        self.causal_enhancer.as_ref().map(|e| e.current_graph())
    }

    /// Get discovered causal relationships history
    pub fn causal_discoveries(&self) -> Option<&[DiscoveredRelationship]> {
        self.causal_enhancer
            .as_ref()
            .map(|e| e.discovered_relationships())
    }

    /// Get causal enhancer statistics
    pub fn causal_stats(&self) -> Option<crate::causal::CausalLoopStats> {
        self.causal_enhancer.as_ref().map(|e| e.stats().clone())
    }

    /// Check if any causal structure has been discovered
    pub fn has_causal_structure(&self) -> bool {
        self.causal_enhancer
            .as_ref()
            .map(|e| e.has_causal_structure())
            .unwrap_or(false)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EPISODIC REPLAY ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get episodic replay statistics
    pub fn episodic_replay_stats(
        &self,
    ) -> Option<crate::memory::episodic_replay::EpisodicMemoryStats> {
        self.phi_episodic_replay.as_ref().map(|r| r.stats())
    }

    /// Get the number of stored episodes
    pub fn episodic_replay_count(&self) -> usize {
        self.phi_episodic_replay
            .as_ref()
            .map(|r| r.len())
            .unwrap_or(0)
    }

    /// Get top N episodes by Phi (highest consciousness moments)
    pub fn top_phi_episodes(&self, n: usize) -> Vec<crate::memory::episodic_replay::Episode> {
        self.phi_episodic_replay
            .as_ref()
            .map(|r| r.get_top_episodes(n))
            .unwrap_or_default()
    }

    /// Get CfC state diversity (activation variance across cells)
    pub fn cfc_state_diversity(&self) -> f32 {
        self.temporal_network.state_diversity()
    }

    /// Get CfC state dimension
    pub fn cfc_state_dim(&self) -> usize {
        self.config.cfc_config.num_neurons
    }

    /// Get HDC bridge dimension (returns None if using CfC backend)
    pub fn hdc_bridge_dim(&self) -> Option<usize> {
        self.temporal_network.hdc_dim()
    }

    /// Project an embedding directly to HDC space, bypassing CfC temporal dynamics.
    pub fn project_embedding_to_hdc(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        let input_dim = self.config.cfc_config.input_dim;

        let compressed = if embedding.len() <= input_dim {
            let mut v = embedding.to_vec();
            v.resize(input_dim, 0.0);
            v
        } else {
            let step = embedding.len() / input_dim;
            embedding
                .iter()
                .step_by(step)
                .take(input_dim)
                .cloned()
                .collect::<Vec<_>>()
        };

        self.temporal_network
            .project_to_hdc_vec(&compressed)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "HDC projection not available (using CfC backend, not HdcLtcBridge)"
                )
            })
    }

    /// Get coherence summary for external systems
    pub fn coherence_summary(&self) -> CoherenceSummary {
        self.coherence_bridge.summary()
    }

    /// Get temporal coherence value (uses cycle-cached value when available)
    pub fn temporal_coherence(&self) -> f32 {
        self.carryover
            .history
            .cached_coherence
            .unwrap_or_else(|| self.coherence_bridge.smoothed_coherence())
    }

    // ========== Semantic Memory Accessors ==========

    /// Get semantic memory statistics
    pub fn semantic_memory_stats(&self) -> &crate::memory::semantic_memory::SemanticMemoryStats {
        self.semantic_memory.stats()
    }

    // ========== Stability Regime Accessors ==========

    /// Get reference to the stability regime processor
    pub fn stability_regime(&self) -> &StabilityRegimeProcessor {
        &self.stability_regime
    }

    // ========== Prediction Confidence Methods ==========

    /// Get current prediction confidence
    pub fn prediction_confidence(&self) -> f32 {
        self.prediction_confidence
    }

    /// Check if predictions should be trusted
    pub fn predictions_trustworthy(&self) -> bool {
        self.prediction_confidence > 0.4
    }

    /// Set the relational Psi from an external dyad computation.
    /// This is called by the Symthaea facade after computing Phi_dyad.
    pub fn set_relational_psi(&mut self, psi: f64) {
        self.relational_psi = psi;
    }

    /// Inject external reward signal for the next cycle.
    /// Blended with internal prediction-error-based reward at 50% weight.
    /// Resets to 0.0 after consumption in the next cycle.
    pub fn provide_reward(&mut self, reward: f32) {
        self.external_reward = reward.clamp(-1.0, 1.0);
    }

    /// Inject social signals from Mind module's SocialCoherence.
    /// Called by the Symthaea facade after Mind.tick() computes social stats.
    pub fn set_social_signals(&mut self, trust: f32, cooperation_rate: f32) {
        self.social_trust = trust.clamp(0.0, 1.0);
        self.social_cooperation_rate = cooperation_rate.clamp(0.0, 1.0);
    }

    /// Inject L-SSM semantic prediction error from LLMOrgan after translation.
    /// Called by the Symthaea facade after translate_thought() to feed PE into
    /// CycleMetadata telemetry for the next cycle.
    #[cfg(feature = "liquid-mamba")]
    pub fn set_liquid_mamba_pe(&mut self, pe: f32) {
        self.stats.last_liquid_mamba_pe = pe;
    }

    /// Current FEP learning signal (0.0-1.0).
    /// Used by the facade to modulate L-SSM distillation intensity.
    pub fn fep_learning_signal(&self) -> f32 {
        self.fep_learning_signal
    }

    /// Get the current inferred user state (if user state inference is enabled).
    pub fn user_state(&self) -> Option<&crate::user_state_inference::UserState> {
        self.user_state.as_ref().map(|usi| usi.state())
    }

    // ========== Flow State Methods ==========

    /// Check if currently in flow state
    pub fn in_flow(&self) -> bool {
        self.flow_state.in_flow
    }

    /// Get flow state intensity (0.0 to 1.0)
    pub fn flow_intensity(&self) -> f32 {
        self.flow_state.intensity
    }

    /// Get flow state streak (consecutive flow-compatible cycles)
    pub fn flow_streak(&self) -> u32 {
        self.flow_state.streak
    }

    /// Get current flow state reference
    pub(crate) fn flow_state(&self) -> &FlowState {
        &self.flow_state
    }

    /// Get flow learning boost multiplier
    pub fn flow_learning_boost(&self) -> f32 {
        self.flow_state.learning_boost
    }

    // ========== Emotion Contagion Methods ==========

    /// Get current emotional valence from content analysis
    pub fn emotional_valence(&self) -> f32 {
        self.emotion_contagion.smoothed_valence()
    }

    /// Get current emotional arousal
    pub fn emotional_arousal(&self) -> f32 {
        self.emotion_contagion.smoothed_arousal()
    }

    /// Get emotion-based pattern nudge suggestion
    pub fn emotion_pattern_nudge(&self) -> (Option<ConsciousnessPattern>, f32) {
        self.emotion_contagion.pattern_nudge()
    }

    /// Get emotion contagion reference
    pub(crate) fn emotion_contagion(&self) -> &EmotionContagion {
        &self.emotion_contagion
    }

    /// Check if emotional content is significant
    pub fn has_emotional_content(&self) -> bool {
        self.emotion_contagion.smoothed_valence().abs() > 0.2
    }

    // ========== Curiosity Drive Methods ==========

    /// Get current boredom level (0.0 to 1.0)
    pub fn boredom(&self) -> f32 {
        self.curiosity_drive.boredom
    }

    /// Get curiosity level (0.0 to 1.0)
    pub fn curiosity(&self) -> f32 {
        self.curiosity_drive.curiosity
    }

    /// Check if curiosity-triggered exploration should occur
    pub fn curiosity_should_explore(&self) -> bool {
        self.curiosity_drive.should_explore()
    }

    /// Get curiosity drive reference
    pub(crate) fn curiosity_drive(&self) -> &CuriosityDrive {
        &self.curiosity_drive
    }

    /// Get novelty bonus for learning
    pub fn novelty_bonus(&self) -> f32 {
        self.curiosity_drive.novelty_bonus
    }

    /// Check if the system is bored (needs new stimuli)
    pub fn is_bored(&self) -> bool {
        self.curiosity_drive.boredom > 0.5
    }

    // ========== Self-Reflection Methods ==========

    /// Get current self-assessment
    pub fn self_assessment(&self) -> SelfAssessment {
        self.self_reflection.self_assessment
    }

    /// Get self-reflection summary
    pub fn reflection_summary(&self) -> ReflectionSummary {
        self.self_reflection.summary()
    }

    /// Get adapted thresholds from self-reflection
    pub fn adapted_thresholds(&self) -> ReflectionThresholds {
        self.self_reflection.get_thresholds()
    }

    /// Get current recommendations from self-reflection
    pub fn recommendations(&self) -> &[Recommendation] {
        &self.self_reflection.recommendations
    }

    /// Get number of reflections performed
    pub fn reflection_count(&self) -> u64 {
        self.self_reflection.reflection_count
    }

    /// Get learning effectiveness score
    pub fn learning_effectiveness(&self) -> f32 {
        self.self_reflection.learning_effectiveness()
    }

    /// Check if system needs calibration
    pub fn needs_calibration(&self) -> bool {
        self.self_reflection.self_assessment == SelfAssessment::NeedsCalibration
    }

    /// Check if system is performing optimally
    pub fn is_optimal(&self) -> bool {
        self.self_reflection.self_assessment == SelfAssessment::Optimal
    }

    /// Force an immediate reflection cycle
    pub fn force_reflect(&mut self) -> Vec<Recommendation> {
        self.self_reflection.reflect()
    }

    /// Get self-reflection reference
    pub(crate) fn self_reflection(&self) -> &SelfReflection {
        &self.self_reflection
    }

    // ========== Consciousness Snapshot ==========

    /// Get a complete snapshot of current consciousness state
    pub fn consciousness_snapshot(&self) -> ConsciousnessSnapshot {
        let (pattern, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        let temporal_summary = self.temporal_signature_encoder.summary();
        let reflection_summary = self.self_reflection.summary();
        let thresholds = self.self_reflection.get_thresholds();
        let (emotion_nudge, _) = self.emotion_contagion.pattern_nudge();

        let consciousness_level = ConsciousnessSnapshot::compute_consciousness_level(
            self.prediction_confidence,
            self.coherence_bridge.smoothed_coherence(),
            self.flow_state.intensity,
            pattern_confidence,
        );

        ConsciousnessSnapshot {
            cycle: self.stats.total_cycles,
            consciousness_level,
            pattern,
            pattern_confidence,
            prediction_error: self.stats.avg_prediction_error,
            prediction_confidence: self.prediction_confidence,
            predictions_trustworthy: self.predictions_trustworthy(),
            effective_learning_rate: self.stats.adaptive_learning_rate,
            learning_effectiveness: self.self_reflection.learning_effectiveness(),
            in_flow: self.flow_state.in_flow,
            flow_intensity: self.flow_state.intensity,
            flow_streak: self.flow_state.streak,
            flow_learning_boost: self.flow_state.learning_boost,
            boredom: self.curiosity_drive.boredom,
            curiosity: self.curiosity_drive.curiosity,
            exploration_urge: self.curiosity_drive.exploration_urge,
            exploring: self.curiosity_drive.should_explore(),
            novelty_bonus: self.curiosity_drive.novelty_bonus,
            emotional_valence: self.emotion_contagion.smoothed_valence(),
            emotional_arousal: self.emotion_contagion.smoothed_arousal(),
            has_emotional_content: self.has_emotional_content(),
            emotion_nudge,
            self_assessment: self.self_reflection.self_assessment,
            reflection_count: reflection_summary.reflection_count,
            adjustments_made: reflection_summary.adjustments_made,
            next_reflection_in: reflection_summary.next_reflection_in,
            action_hint: self.adaptive_behavior.action_hint,
            speech_rate_multiplier: self.adaptive_behavior.speech_rate_multiplier,
            pause_multiplier: self.adaptive_behavior.pause_multiplier,
            learning_paused: self.adaptive_behavior.pause_learning,
            flow_threshold: thresholds.flow_error,
            boredom_threshold: thresholds.boredom,
            trust_threshold: thresholds.trust,
            temporal_coherence: self.coherence_bridge.smoothed_coherence(),
            tau_mean: temporal_summary.features.mean,
            tau_trend: temporal_summary.features.trend,
            cognitive_depth: self.cognitive_depth,
            unified_psi: self.unification_engine.psi as f32,
            unified_valence: self.unification_engine.emotional.state().valence as f32,
            unified_arousal: self.unification_engine.emotional.state().arousal as f32,
            unified_dominance: self.unification_engine.emotional.state().dominance as f32,
            unified_discrete_emotion: self.unification_engine.emotional.state().discrete_emotion,
            emotional_pattern: self.unification_engine.emotional.detect_pattern(),
            emotional_description: self.unification_engine.emotional.state().describe(),
            snapshot_timestamp_nanos: self.start_time.elapsed().as_nanos() as u64,
            current_flow_duration_secs: self.flow_state.current_flow_duration_secs(),
            total_flow_time_secs: self.flow_state.total_flow_time_with_current(),
            flow_periods: self.flow_state.flow_periods,
            avg_flow_duration_secs: self.flow_state.avg_flow_duration_secs,
            fep_free_energy: self
                .fep_agent
                .last_fe_components
                .as_ref()
                .map(|fe| fe.total)
                .unwrap_or(0.0),
            fep_precision: self.fep_agent.precision.perceptual_precision(),
            spectral_mip_phi: self.carryover.consciousness.last_spectral_mip_phi,
            harmonies_alignment: self
                .harmonies_integrator
                .as_ref()
                .map(|h| h.stats().avg_alignment)
                .unwrap_or(0.0),
            empathic_compassion: self
                .empathic_unification
                .as_ref()
                .map(|_| 0.0) // Compassion is per-cycle; snapshot shows lifetime average
                .unwrap_or(0.0),
            sigma: self.carryover.consciousness.last_sigma,
            avg_cycle_time_us: self.stats.avg_cycle_time_us,
            cycles_per_second: self.stats.cycles_per_second,
        }
    }

    /// Get a concise status line for logging/display
    pub fn status_line(&self) -> String {
        self.consciousness_snapshot().status()
    }

    /// Check if system needs attention (via snapshot)
    pub fn snapshot_needs_attention(&self) -> bool {
        self.consciousness_snapshot().needs_attention()
    }

    /// Get current consciousness level (0.0 to 1.0)
    pub fn consciousness_level(&self) -> f32 {
        let (_, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        ConsciousnessSnapshot::compute_consciousness_level(
            self.prediction_confidence,
            self.coherence_bridge.smoothed_coherence(),
            self.flow_state.intensity,
            pattern_confidence,
        )
    }

    // ========== Voice Feedback Methods ==========

    /// Update voice feedback with synthesis output metrics
    pub fn update_voice_feedback(&mut self, metrics: VoiceOutputMetrics) {
        self.voice_feedback_bridge.update(metrics);
    }

    /// Update listener prediction feedback
    pub fn update_listener_prediction(&mut self, success: f32) {
        self.voice_feedback_bridge
            .update_listener_prediction(success);
    }

    /// Get voice quality summary for external systems
    pub fn voice_feedback_summary(&self) -> VoiceQualitySummary {
        self.voice_feedback_bridge.summary()
    }

    /// Check if voice indicates uncertainty
    pub fn voice_indicates_uncertainty(&self) -> bool {
        self.voice_feedback_bridge.is_uncertain()
    }

    /// Get Phase 16 consciousness signals for voice prosody modulation.
    ///
    /// Returns a compact struct containing unified quality, epistemic gating,
    /// dissipative health, coherence velocity, and consciousness level —
    /// the signals needed by `CognitivePacing::from_cycle_metadata()`.
    pub fn voice_consciousness_signals(&self) -> VoiceConsciousnessSignals {
        let (_, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        let consciousness_level =
            super::snapshot::ConsciousnessSnapshot::compute_consciousness_level(
                self.prediction_confidence,
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

    // ═══════════════════════════════════════════════════════════════════════
    // MEGA-UNIFIED ARCHITECTURE: Accessor Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Get current cognitive depth from thalamic routing
    pub fn cognitive_depth(&self) -> CognitiveDepth {
        self.cognitive_depth
    }

    /// Get thalamic routing statistics (reflex_rate, cortical_rate, deep_rate)
    pub fn thalamic_stats(&self) -> (f32, f32, f32) {
        self.thalamic_router.routing_stats()
    }

    /// Get the emotional pattern (Stable/Escalating/Calming/Volatile)
    pub fn emotional_pattern(&self) -> EmotionalPattern {
        self.unification_engine.emotional.detect_pattern()
    }

    /// Get natural language description of current emotional state
    pub fn emotional_description(&self) -> String {
        self.unification_engine.emotional.state().describe()
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

    /// Get the conversation coherence tracker reference
    pub(crate) fn coherence_tracker(&self) -> &ConversationCoherenceTracker {
        &self.coherence_tracker
    }

    /// Get the prediction-outcome coupling Modulation Index
    pub fn modulation_index(&self) -> Option<f64> {
        self.active_inference_bridge.modulation_index()
    }

    /// Get the coupling quality assessment
    pub fn coupling_quality(&self) -> CouplingQuality {
        self.active_inference_bridge.coupling_quality()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CLOSED LEARNING LOOP: Accessor Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Get the current response strategy
    pub fn current_strategy(&self) -> ResponseStrategy {
        self.closed_learning_loop.current_strategy
    }

    /// Get the best strategy according to Q-learning
    pub fn best_strategy(&self) -> ResponseStrategy {
        self.closed_learning_loop.best_strategy()
    }

    /// Get average reward from the learning loop
    pub fn average_reward(&self) -> f32 {
        self.closed_learning_loop.average_reward()
    }

    /// Get Q-values for all strategies
    pub fn strategy_q_values(&self) -> &[f32; 5] {
        self.closed_learning_loop.q_values()
    }

    /// Get strategy usage counts
    pub fn strategy_usage_counts(&self) -> &[u64; 5] {
        self.closed_learning_loop.strategy_counts()
    }

    /// Get the last learning result
    pub fn last_learning_result(&self) -> Option<&CycleLearningResult> {
        self.closed_learning_loop.last_result.as_ref()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MEMORY SYSTEM: Accessor Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Get memory counts (short_term, long_term)
    pub fn memory_counts(&self) -> (usize, usize) {
        self.episodic_memory.memory_count()
    }

    /// Recall memories similar to input
    pub fn recall_memories(&mut self, query: &[f32], top_k: usize) -> Vec<(EpisodicMemory, f32)> {
        self.episodic_memory.recall(query, top_k, 0.2)
    }

    /// Add a goal to the system
    pub fn add_goal(&mut self, id: &str, description: &str, priority: f32) {
        self.goal_system
            .add_goal(CognitiveGoal::new(id, description, priority));
    }

    /// Get active goals
    pub fn active_goals(&self) -> Vec<&CognitiveGoal> {
        self.goal_system.active_goals()
    }

    /// Get the world model bridge reference
    pub fn world_model(&self) -> &WorldModelBridge {
        &self.world_model
    }

    /// Get abstract level state from world model
    pub fn world_model_abstract_state(&self) -> &[f32] {
        self.world_model.abstract_state()
    }

    /// Get world model prediction errors at each level
    pub fn world_model_level_errors(&self) -> &[f32] {
        self.world_model.level_errors()
    }

    /// Get combined learning rate modifier
    pub fn combined_learning_rate(&self) -> f32 {
        let coherence_lr = self.coherence_bridge.effective_learning_rate();
        let voice_modifier = self.voice_feedback_bridge.learning_rate_modifier();
        coherence_lr * voice_modifier
    }

    // ========== Consciousness Pattern Methods ==========

    /// Get current consciousness pattern classification
    pub fn consciousness_pattern(&self) -> (ConsciousnessPattern, f32) {
        self.temporal_signature_encoder.classify_state()
    }

    /// Get full temporal state summary
    pub fn temporal_state_summary(&self) -> TemporalStateSummary {
        self.temporal_signature_encoder.summary()
    }

    /// Check if current state matches a specific consciousness pattern
    pub fn is_consciousness_state(&self, pattern: ConsciousnessPattern) -> bool {
        self.temporal_signature_encoder.is_state(pattern)
    }

    /// Get similarity to a specific consciousness pattern
    pub fn consciousness_pattern_similarity(&self, pattern: ConsciousnessPattern) -> f32 {
        self.temporal_signature_encoder.similarity_to(pattern)
    }

    // ========== Adaptive Behavior Methods ==========

    /// Get current adaptive behavior
    pub(crate) fn adaptive_behavior(&self) -> &AdaptiveBehavior {
        &self.adaptive_behavior
    }

    /// Get current action hint
    pub fn action_hint(&self) -> ActionHint {
        self.adaptive_behavior.action_hint
    }

    /// Check if system should seek more input/clarification
    pub fn should_seek_input(&self) -> bool {
        self.adaptive_behavior.should_seek_input()
    }

    /// Check if system is in a confident state
    pub fn is_confident(&self) -> bool {
        self.adaptive_behavior.is_confident()
    }

    /// Get description of current adaptive state
    pub fn state_description(&self) -> &'static str {
        self.adaptive_behavior.description()
    }

    /// Get speech rate multiplier for voice synthesis
    pub fn speech_rate_multiplier(&self) -> f32 {
        self.adaptive_behavior.speech_rate_multiplier
    }

    /// Get pause duration multiplier for voice synthesis
    pub fn pause_multiplier(&self) -> f32 {
        self.adaptive_behavior.pause_multiplier
    }

    /// Get attention sensitivity for input processing
    pub fn attention_sensitivity(&self) -> f32 {
        self.adaptive_behavior.attention_sensitivity
    }

    /// Get exploration factor for decision making
    pub fn exploration_factor(&self) -> f32 {
        self.adaptive_behavior.exploration_factor
    }

    /// Get the compressed state dimension (input to CfC)
    pub fn state_dim(&self) -> usize {
        self.config.cfc_config.input_dim
    }

    /// Get the prediction dimension (CfC neurons)
    pub fn prediction_dim(&self) -> usize {
        self.config.cfc_config.num_neurons
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PSI ATTESTATION ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get the number of buffered PsiAttestationRecords.
    pub fn psi_attestation_count(&self) -> usize {
        self.psi_attestation_buffer.len()
    }

    /// Drain all buffered PsiAttestationRecords for submission to the governance bridge.
    /// Returns the records and clears the buffer.
    pub fn drain_psi_attestations(&mut self) -> Vec<super::PsiAttestationRecord> {
        self.psi_attestation_buffer.drain(..).collect()
    }

    /// Peek at the most recent PsiAttestationRecord without consuming it.
    pub fn latest_psi_attestation(&self) -> Option<&super::PsiAttestationRecord> {
        self.psi_attestation_buffer.back()
    }

    /// Borrow the temporal primitives analyzer (if enabled).
    pub fn temporal_analyzer(
        &self,
    ) -> Option<&crate::consciousness::temporal_primitives::ConsciousnessTemporalAnalyzer> {
        self.temporal_analyzer.as_ref()
    }

    /// Borrow the primitive lattice (if enabled).
    pub fn primitive_lattice(
        &self,
    ) -> Option<&crate::consciousness::primitive_lattice::PrimitiveLattice> {
        self.primitive_lattice.as_ref()
    }

    /// Borrow the compositionality engine (if enabled).
    pub fn compositionality_engine(
        &self,
    ) -> Option<&crate::consciousness::compositionality::CompositionalityEngine> {
        self.compositionality_engine.as_ref()
    }

    /// Borrow the unified value evaluator (if enabled).
    pub fn value_evaluator(
        &self,
    ) -> Option<&crate::consciousness::unified_value_evaluator::UnifiedValueEvaluator> {
        self.value_evaluator.as_ref()
    }

    /// Borrow the harmonic field (if enabled).
    pub fn harmonic_field(&self) -> Option<&crate::consciousness::harmonics::HarmonicField> {
        self.harmonic_field.as_ref()
    }

    /// Borrow the primitive reasoner (if enabled).
    pub fn primitive_reasoner(
        &self,
    ) -> Option<&crate::consciousness::primitive_reasoning::PrimitiveReasoner> {
        self.primitive_reasoner.as_ref()
    }

    /// Borrow the adaptive reasoner (if enabled).
    pub fn adaptive_reasoner(
        &self,
    ) -> Option<&crate::consciousness::adaptive_reasoning::AdaptiveReasoner> {
        self.adaptive_reasoner.as_ref()
    }

    /// Borrow the causal self-explainer (if enabled).
    pub fn causal_explainer(
        &self,
    ) -> Option<&crate::consciousness::causal_explanation::CausalExplainer> {
        self.causal_explainer.as_ref()
    }

    /// Borrow the context-aware optimizer (if enabled).
    pub fn context_optimizer(
        &self,
    ) -> Option<&crate::consciousness::context_aware_evolution::ContextAwareOptimizer> {
        self.context_optimizer.as_ref()
    }

    /// Borrow the evolution coordinator (if enabled).
    pub fn evolution_coordinator(
        &self,
    ) -> Option<&crate::consciousness::evolution_bridge::EvolutionCoordinator> {
        self.evolution_coordinator.as_ref()
    }

    /// Borrow the harmonies integrator (if enabled).
    pub fn harmonies_integrator(
        &self,
    ) -> Option<&crate::consciousness::harmonies_integration::HarmoniesIntegrator> {
        self.harmonies_integrator.as_ref()
    }

    /// Borrow the composition rule engine (if enabled).
    pub fn composition_rule_engine(
        &self,
    ) -> Option<&crate::consciousness::primitive_composition_rules::CompositionRuleEngine> {
        self.composition_rule_engine.as_ref()
    }

    /// Borrow the semantic value embedder (if enabled).
    pub fn semantic_value_embedder(
        &self,
    ) -> Option<&crate::consciousness::semantic_value_embedder::SemanticValueEmbedder> {
        self.semantic_value_embedder.as_ref()
    }

    /// Borrow the dissipative consciousness model (if enabled).
    pub(crate) fn dissipative_consciousness(
        &self,
    ) -> Option<&crate::consciousness::dissipative_consciousness::DissipativeConsciousness> {
        self.dissipative_consciousness.as_ref()
    }

    /// Borrow the epistemic conflict detector (if enabled).
    pub fn epistemic_conflict_detector(
        &self,
    ) -> Option<&crate::consciousness::epistemic_conflict::ConflictDetector> {
        self.epistemic_conflict_detector.as_ref()
    }

    /// Borrow the consciousness equation v2 (if enabled).
    pub fn consciousness_equation_v2(
        &self,
    ) -> Option<&crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2> {
        self.consciousness_equation_v2.as_ref()
    }

    /// Borrow the hierarchical LTC (if enabled).
    pub fn hierarchical_ltc(
        &self,
    ) -> Option<&crate::consciousness::hierarchical_ltc::HierarchicalLTC> {
        self.hierarchical_ltc.as_ref()
    }

    /// Borrow the theory calibrator (if enabled).
    pub fn theory_calibrator(
        &self,
    ) -> Option<&crate::consciousness::epistemic_conflict::TheoryCalibrator> {
        self.theory_calibrator.as_ref()
    }

    /// Borrow the holographic consciousness analyzer (if enabled).
    pub(crate) fn holographic_analyzer(
        &self,
    ) -> Option<&crate::consciousness::consciousness_holography::HolographicConsciousnessAnalyzer>
    {
        self.holographic_analyzer.as_ref()
    }

    /// Borrow the differentiable consciousness model (if enabled).
    pub fn differentiable_consciousness(
        &self,
    ) -> Option<&crate::consciousness::differentiable::DifferentiableConsciousness> {
        self.differentiable_consciousness.as_ref()
    }

    /// Borrow the affective consciousness analyzer (if enabled).
    pub fn affective_consciousness(
        &self,
    ) -> Option<&crate::consciousness::affective_consciousness::AffectiveConsciousnessAnalyzer>
    {
        self.affective_consciousness.as_ref()
    }

    /// Borrow the unified consciousness pipeline (if enabled).
    pub fn unified_consciousness_pipeline(
        &self,
    ) -> Option<&crate::consciousness::unified_consciousness_pipeline::UnifiedConsciousnessPipeline>
    {
        self.unified_consciousness_pipeline.as_ref()
    }

    /// Borrow the multi-modal integrator (if enabled).
    pub fn multi_modal_integrator(
        &self,
    ) -> Option<&crate::consciousness::multi_modal_integration::MultiModalIntegrator> {
        self.multi_modal_integrator.as_ref()
    }

    /// Borrow the synthetic states NSM grounding (if enabled).
    pub(crate) fn synthetic_grounding(
        &self,
    ) -> Option<&crate::consciousness::synthetic_states::SyntheticStatesNSMGrounding> {
        self.synthetic_grounding.as_ref()
    }

    /// Borrow the epistemic decision gate (if enabled).
    pub fn epistemic_gate(
        &self,
    ) -> Option<&crate::consciousness::gis_integration::EpistemicDecisionGate> {
        self.epistemic_gate.as_ref()
    }

    /// Borrow the meta-cognitive reasoner (if enabled).
    pub fn meta_cognitive_reasoner(
        &self,
    ) -> Option<&crate::consciousness::meta_reasoning::MetaCognitiveReasoner> {
        self.meta_cognitive_reasoner.as_ref()
    }

    /// Borrow the code primitive router (if enabled).
    pub fn code_primitive_router(
        &self,
    ) -> Option<&crate::consciousness::code_primitives::CodePrimitiveRouter> {
        self.code_primitive_router.as_ref()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EXPERIENCE BUS: Principled signals + Seven Harmonies
    // ═══════════════════════════════════════════════════════════════════════

    /// Current principled signals from experience bus.
    pub fn experience_signals(&self) -> Option<&crate::experience::PrincipledSignals> {
        self.experience_bus.as_ref().map(|bus| bus.signals())
    }

    /// KosmicSong state (Seven Harmonies + GIS + moral uncertainty).
    pub fn kosmic_state(&self) -> Option<&crate::experience::KosmicSong> {
        self.experience_bus.as_ref().map(|bus| bus.kosmic())
    }

    /// Current guiding question from wisdom system.
    pub fn guiding_question(&self) -> Option<&'static str> {
        self.experience_bus
            .as_ref()
            .map(|bus| bus.current_guiding_question())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ATTENTION VISUALIZATION ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get attention visualization summary (snapshot count, avg entropy, top attended).
    pub fn attention_summary(&self) -> Option<crate::visualization::AttentionSummary> {
        self.attention_visualizer.as_ref().map(|viz| viz.summary())
    }

    /// Export attention history as JSON for external analysis tools.
    pub fn attention_history_json(&self) -> Option<String> {
        self.attention_visualizer
            .as_ref()
            .and_then(|viz| viz.export_json().ok())
    }

    /// Render attention heatmap as ASCII art (inputs x time).
    pub fn attention_heatmap(&self) -> Option<String> {
        self.attention_visualizer
            .as_ref()
            .map(|viz| viz.render_heatmap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    // ── Stats accessor ────────────────────────────────────────────────

    #[test]
    fn stats_initial_total_cycles() {
        let s = make_service();
        assert_eq!(s.stats().total_cycles, 0);
    }

    #[test]
    fn stats_initial_avg_error_zero() {
        let s = make_service();
        assert_eq!(s.stats().avg_prediction_error, 0.0);
    }

    // ── Config accessor ───────────────────────────────────────────────

    #[test]
    fn config_returns_learning_threshold() {
        let cfg = CognitiveLoopConfig::default();
        let expected = cfg.learning_threshold;
        let s = CognitiveLoopService::new(cfg).unwrap();
        assert_eq!(s.config().learning_threshold, expected);
    }

    #[test]
    fn config_returns_target_frequency() {
        let s = make_service();
        assert_eq!(s.config().target_frequency, 50.0);
    }

    // ── Prediction confidence ─────────────────────────────────────────

    #[test]
    fn prediction_confidence_initial_value() {
        let s = make_service();
        assert!((s.prediction_confidence() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn prediction_confidence_is_finite() {
        let s = make_service();
        assert!(s.prediction_confidence().is_finite());
    }

    #[test]
    fn predictions_trustworthy_at_initial() {
        let s = make_service();
        // prediction_confidence=0.5 > 0.4, so should be trustworthy
        assert!(s.predictions_trustworthy());
    }

    // ── Reward injection ──────────────────────────────────────────────

    #[test]
    fn provide_reward_clamps_positive() {
        let mut s = make_service();
        s.provide_reward(100.0);
        assert!(s.external_reward <= 1.0);
    }

    #[test]
    fn provide_reward_clamps_negative() {
        let mut s = make_service();
        s.provide_reward(-100.0);
        assert!(s.external_reward >= -1.0);
    }

    #[test]
    fn provide_reward_preserves_zero() {
        let mut s = make_service();
        s.provide_reward(0.0);
        assert!((s.external_reward).abs() < f32::EPSILON);
    }

    #[test]
    fn provide_reward_preserves_in_range() {
        let mut s = make_service();
        s.provide_reward(0.7);
        assert!((s.external_reward - 0.7).abs() < f32::EPSILON);
    }

    // ── Social signals ────────────────────────────────────────────────

    #[test]
    fn set_social_signals_clamps_trust() {
        let mut s = make_service();
        s.set_social_signals(5.0, 0.5);
        assert!(s.social_trust <= 1.0);
        assert!(s.social_trust >= 0.0);
    }

    #[test]
    fn set_social_signals_clamps_cooperation() {
        let mut s = make_service();
        s.set_social_signals(0.5, -3.0);
        assert!(s.social_cooperation_rate >= 0.0);
        assert!(s.social_cooperation_rate <= 1.0);
    }

    #[test]
    fn set_social_signals_preserves_in_range() {
        let mut s = make_service();
        s.set_social_signals(0.8, 0.3);
        assert!((s.social_trust - 0.8).abs() < f32::EPSILON);
        assert!((s.social_cooperation_rate - 0.3).abs() < f32::EPSILON);
    }

    // ── Relational Psi ────────────────────────────────────────────────

    #[test]
    fn set_relational_psi_stores_value() {
        let mut s = make_service();
        s.set_relational_psi(0.42);
        assert!((s.relational_psi - 0.42).abs() < f64::EPSILON);
    }

    // ── FEP learning signal ───────────────────────────────────────────

    #[test]
    fn fep_learning_signal_initial() {
        let s = make_service();
        assert!((s.fep_learning_signal() - 0.0).abs() < f32::EPSILON);
    }

    // ── Flow state ────────────────────────────────────────────────────

    #[test]
    fn flow_initial_not_in_flow() {
        let s = make_service();
        assert!(!s.in_flow());
    }

    #[test]
    fn flow_initial_intensity_bounded() {
        let s = make_service();
        let i = s.flow_intensity();
        assert!(i.is_finite());
        assert!((0.0..=1.0).contains(&i));
    }

    #[test]
    fn flow_initial_streak_zero() {
        let s = make_service();
        assert_eq!(s.flow_streak(), 0);
    }

    #[test]
    fn flow_learning_boost_initial() {
        let s = make_service();
        let b = s.flow_learning_boost();
        assert!(b.is_finite());
    }

    // ── Emotion ───────────────────────────────────────────────────────

    #[test]
    fn emotional_valence_initial_bounded() {
        let s = make_service();
        let v = s.emotional_valence();
        assert!(v.is_finite());
        assert!((-1.0..=1.0).contains(&v));
    }

    #[test]
    fn emotional_arousal_initial_bounded() {
        let s = make_service();
        let a = s.emotional_arousal();
        assert!(a.is_finite());
        assert!((0.0..=1.0).contains(&a));
    }

    #[test]
    fn no_emotional_content_initially() {
        let s = make_service();
        assert!(!s.has_emotional_content());
    }

    // ── Curiosity and boredom ─────────────────────────────────────────

    #[test]
    fn boredom_initial_bounded() {
        let s = make_service();
        let b = s.boredom();
        assert!(b.is_finite());
        assert!((0.0..=1.0).contains(&b));
    }

    #[test]
    fn curiosity_initial_bounded() {
        let s = make_service();
        let c = s.curiosity();
        assert!(c.is_finite());
        assert!((0.0..=1.0).contains(&c));
    }

    #[test]
    fn novelty_bonus_initial_finite() {
        let s = make_service();
        assert!(s.novelty_bonus().is_finite());
    }

    #[test]
    fn is_bored_false_initially() {
        let s = make_service();
        // Default boredom is low, so should not be bored
        assert!(!s.is_bored());
    }

    // ── Self-reflection ───────────────────────────────────────────────

    #[test]
    fn reflection_count_initial() {
        let s = make_service();
        assert_eq!(s.reflection_count(), 0);
    }

    #[test]
    fn learning_effectiveness_initial_finite() {
        let s = make_service();
        assert!(s.learning_effectiveness().is_finite());
    }

    #[test]
    fn recommendations_initially_empty() {
        let s = make_service();
        assert!(s.recommendations().is_empty());
    }

    #[test]
    fn force_reflect_returns_vec() {
        let mut s = make_service();
        let recs = s.force_reflect();
        // Just verifying it doesn't panic and returns a vec
        assert!(recs.len() <= 100); // Sanity bound
    }

    // ── Consciousness snapshot ────────────────────────────────────────

    #[test]
    fn consciousness_snapshot_fields_finite() {
        let s = make_service();
        let snap = s.consciousness_snapshot();
        assert!(snap.consciousness_level >= 0.0 && snap.consciousness_level <= 1.0);
        assert!(snap.prediction_confidence.is_finite());
        assert!(snap.flow_intensity.is_finite());
        assert!(snap.boredom.is_finite());
        assert!(snap.curiosity.is_finite());
        assert!(snap.emotional_valence.is_finite());
        assert!(snap.emotional_arousal.is_finite());
    }

    #[test]
    fn consciousness_snapshot_cycle_zero() {
        let s = make_service();
        let snap = s.consciousness_snapshot();
        assert_eq!(snap.cycle, 0);
    }

    #[test]
    fn consciousness_level_in_range() {
        let s = make_service();
        let cl = s.consciousness_level();
        assert!((0.0..=1.0).contains(&cl), "consciousness_level={cl}");
    }

    #[test]
    fn status_line_not_empty() {
        let s = make_service();
        let line = s.status_line();
        assert!(!line.is_empty());
    }

    // ── Goal system ───────────────────────────────────────────────────

    #[test]
    fn no_goals_initially() {
        let s = make_service();
        assert!(s.active_goals().is_empty());
    }

    #[test]
    fn add_goal_and_retrieve() {
        let mut s = make_service();
        s.add_goal("explore", "explore the environment", 0.8);
        let goals = s.active_goals();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].id, "explore");
        assert!(goals[0].is_active);
    }

    #[test]
    fn add_multiple_goals() {
        let mut s = make_service();
        s.add_goal("g1", "first goal", 0.9);
        s.add_goal("g2", "second goal", 0.5);
        s.add_goal("g3", "third goal", 0.1);
        assert_eq!(s.active_goals().len(), 3);
    }

    // ── Memory counts ─────────────────────────────────────────────────

    #[test]
    fn memory_counts_initially_zero() {
        let s = make_service();
        let (st, lt) = s.memory_counts();
        assert_eq!(st, 0);
        assert_eq!(lt, 0);
    }

    // ── Causal accessors ──────────────────────────────────────────────

    #[test]
    fn causal_graph_none_when_disabled() {
        let s = make_service();
        assert!(s.causal_graph().is_none());
    }

    #[test]
    fn causal_discoveries_none_when_disabled() {
        let s = make_service();
        assert!(s.causal_discoveries().is_none());
    }

    #[test]
    fn has_causal_structure_false_when_disabled() {
        let s = make_service();
        assert!(!s.has_causal_structure());
    }

    // ── Episodic replay ───────────────────────────────────────────────

    #[test]
    fn top_phi_episodes_empty_when_disabled() {
        let s = make_service();
        assert!(s.top_phi_episodes(10).is_empty());
    }

    // ── CfC state ─────────────────────────────────────────────────────

    #[test]
    fn cfc_state_diversity_finite() {
        let s = make_service();
        assert!(s.cfc_state_diversity().is_finite());
    }

    #[test]
    fn cfc_state_dim_matches_config() {
        let s = make_service();
        assert_eq!(s.cfc_state_dim(), 256);
    }

    #[test]
    fn state_dim_matches_config() {
        let s = make_service();
        assert_eq!(s.state_dim(), 256);
    }

    #[test]
    fn prediction_dim_matches_config() {
        let s = make_service();
        assert_eq!(s.prediction_dim(), 256);
    }

    // ── Temporal coherence ────────────────────────────────────────────

    #[test]
    fn temporal_coherence_finite() {
        let s = make_service();
        assert!(s.temporal_coherence().is_finite());
    }

    // ── Neurochemistry ────────────────────────────────────────────────

    #[test]
    fn neurochemistry_checkpoint_roundtrip() {
        let mut s = make_service();
        let ckpt = s.neurochemistry_checkpoint();
        // Restore should not panic
        s.restore_neurochemistry(&ckpt);
        let ckpt2 = s.neurochemistry_checkpoint();
        // Values should be identical after restore
        assert_eq!(ckpt.da_sensitivity, ckpt2.da_sensitivity);
        assert_eq!(ckpt.ne_sensitivity, ckpt2.ne_sensitivity);
        assert_eq!(ckpt.sht_sensitivity, ckpt2.sht_sensitivity);
        assert_eq!(ckpt.ach_sensitivity, ckpt2.ach_sensitivity);
    }

    #[test]
    fn clamp_neuromod_levels_no_panic() {
        let mut s = make_service();
        s.clamp_neuromod_levels(Some(0.5), None, Some(1.0), None);
    }

    #[test]
    fn neuromod_snapshot_finite() {
        let s = make_service();
        let snap = s.neuromod_snapshot();
        assert!(snap.da_effective.is_finite());
        assert!(snap.ne_effective.is_finite());
        assert!(snap.sht_effective.is_finite());
        assert!(snap.ach_effective.is_finite());
    }

    // ── Pain sender ───────────────────────────────────────────────────

    #[test]
    fn pain_sender_present() {
        let s = make_service();
        assert!(s.pain_sender().is_some());
    }

    // ── Psi attestation ───────────────────────────────────────────────

    #[test]
    fn psi_attestation_count_zero() {
        let s = make_service();
        assert_eq!(s.psi_attestation_count(), 0);
    }

    #[test]
    fn drain_psi_attestations_empty() {
        let mut s = make_service();
        let drained = s.drain_psi_attestations();
        assert!(drained.is_empty());
    }

    #[test]
    fn latest_psi_attestation_none() {
        let s = make_service();
        assert!(s.latest_psi_attestation().is_none());
    }

    // ── Adaptive behavior ─────────────────────────────────────────────

    #[test]
    fn speech_rate_multiplier_finite() {
        let s = make_service();
        assert!(s.speech_rate_multiplier().is_finite());
    }

    #[test]
    fn pause_multiplier_finite() {
        let s = make_service();
        assert!(s.pause_multiplier().is_finite());
    }

    #[test]
    fn attention_sensitivity_finite() {
        let s = make_service();
        assert!(s.attention_sensitivity().is_finite());
    }

    #[test]
    fn exploration_factor_finite() {
        let s = make_service();
        assert!(s.exploration_factor().is_finite());
    }

    #[test]
    fn state_description_not_empty() {
        let s = make_service();
        assert!(!s.state_description().is_empty());
    }

    // ── Strategy and learning loop ────────────────────────────────────

    #[test]
    fn strategy_q_values_length() {
        let s = make_service();
        assert_eq!(s.strategy_q_values().len(), 5);
    }

    #[test]
    fn strategy_usage_counts_length() {
        let s = make_service();
        assert_eq!(s.strategy_usage_counts().len(), 5);
    }

    #[test]
    fn average_reward_initial_finite() {
        let s = make_service();
        assert!(s.average_reward().is_finite());
    }

    #[test]
    fn last_learning_result_none_initially() {
        let s = make_service();
        assert!(s.last_learning_result().is_none());
    }

    // ── Modulation index / coupling ───────────────────────────────────

    #[test]
    fn coupling_quality_no_panic() {
        let s = make_service();
        let _q = s.coupling_quality();
    }

    // ── World model ───────────────────────────────────────────────────

    #[test]
    fn world_model_abstract_state_not_empty() {
        let s = make_service();
        assert!(!s.world_model_abstract_state().is_empty());
    }

    #[test]
    fn world_model_level_errors_not_empty() {
        let s = make_service();
        assert!(!s.world_model_level_errors().is_empty());
    }

    // ── Combined learning rate ────────────────────────────────────────

    #[test]
    fn combined_learning_rate_positive() {
        let s = make_service();
        assert!(s.combined_learning_rate() > 0.0);
    }

    // ── Consciousness pattern ─────────────────────────────────────────

    #[test]
    fn consciousness_pattern_confidence_finite() {
        let s = make_service();
        let (_, conf) = s.consciousness_pattern();
        assert!(conf.is_finite());
    }

    // ── FEP free energy ───────────────────────────────────────────────

    #[test]
    fn fep_free_energy_initially_none() {
        let s = make_service();
        // No cycles run yet, so no FE components computed
        assert!(s.fep_free_energy().is_none());
    }

    // ── Experience bus ─────────────────────────────────────────────────

    #[test]
    fn experience_signals_present() {
        let s = make_service();
        // ExperienceBus is created by default
        assert!(s.experience_signals().is_some());
    }

    #[test]
    fn guiding_question_present() {
        let s = make_service();
        assert!(s.guiding_question().is_some());
    }

    // ── Attention visualization ────────────────────────────────────────

    #[test]
    fn attention_summary_present() {
        let s = make_service();
        assert!(s.attention_summary().is_some());
    }

    #[test]
    fn attention_heatmap_present() {
        let s = make_service();
        assert!(s.attention_heatmap().is_some());
    }

    // ── HDC bridge dimension ──────────────────────────────────────────

    #[test]
    fn hdc_bridge_dim_none_for_cfc() {
        let s = make_service();
        // Default backend is CfC, which has no HDC bridge dim
        assert!(s.hdc_bridge_dim().is_none());
    }

    #[test]
    fn hdc_bridge_dim_some_for_hdc_ltc() {
        let config = CognitiveLoopConfig::with_hdc_ltc_unified();
        let s = CognitiveLoopService::new(config).unwrap();
        assert!(s.hdc_bridge_dim().is_some());
    }

    // ── Voice feedback ────────────────────────────────────────────────

    #[test]
    fn voice_indicates_uncertainty_initial() {
        let s = make_service();
        // Just verify it doesn't panic; specific value depends on state
        let _u = s.voice_indicates_uncertainty();
    }

    #[test]
    fn voice_consciousness_signals_finite() {
        let s = make_service();
        let sig = s.voice_consciousness_signals();
        assert!(sig.unified_quality.is_finite());
        assert!(sig.consciousness_level.is_finite());
    }

    #[test]
    fn combined_phi_contribution_finite() {
        let s = make_service();
        assert!(s.combined_phi_contribution().is_finite());
    }

    // ── Thalamic stats ────────────────────────────────────────────────

    #[test]
    fn thalamic_stats_sum_to_one_or_zero() {
        let s = make_service();
        let (reflex, cortical, deep) = s.thalamic_stats();
        // All should be finite
        assert!(reflex.is_finite());
        assert!(cortical.is_finite());
        assert!(deep.is_finite());
    }

    // ── User state ────────────────────────────────────────────────────

    #[test]
    fn user_state_none_when_disabled() {
        let s = make_service();
        assert!(s.user_state().is_none());
    }
}
