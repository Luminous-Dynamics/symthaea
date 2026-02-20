//! Accessor methods for CognitiveLoopService.
//!
//! High-level query methods (flow state, prediction confidence, consciousness
//! snapshot, etc.) are `pub` for use by external consumers (examples, LUCID,
//! symthaea-nix). Raw subsystem references (e.g., `thalamic_router()`,
//! `unification_engine_mut()`) are `pub(crate)` to prevent external coupling
//! to implementation details.

use crate::causal::{CausalGraph, DiscoveredRelationship};
use crate::consciousness::consciousness_unification::{
    ConsciousnessUnificationEngine, EmotionalPattern, UnifiedEmotion, UnifiedEmotionalState,
};
use crate::consciousness::fep_active_inference::ActiveInferenceAgent;
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
use crate::dynamics::cfc_coherence::CoherenceSummary;
use crate::dynamics::temporal_signatures::{ConsciousnessPattern, TemporalStateSummary};
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::voice::voice_feedback::{VoiceOutputMetrics, VoiceQualitySummary};
use anyhow::Result;

use super::snapshot::ConsciousnessSnapshot;
use super::temporal_network::TemporalNetwork;
use super::{
    ActionHint, ActiveInferenceBridge, AdaptiveBehavior, ClosedLearningLoop, CognitiveDepth,
    CognitiveGoal, CognitiveLoopConfig, CognitiveLoopService, CouplingQuality, CuriosityDrive,
    CycleLearningResult, EmotionContagion, EpisodicMemory, EpisodicMemoryBridge, FlowState,
    GoalSystemBridge, LoopStats, Recommendation, ReflectionSummary, ReflectionThresholds,
    ResponseStrategy, SelfAssessment, SelfReflection, ThalamicRouter, WorldModelBridge,
};

impl CognitiveLoopService {
    /// Get current statistics
    pub fn stats(&self) -> &LoopStats {
        &self.stats
    }

    /// Get the configuration used to create this service.
    pub fn config(&self) -> &CognitiveLoopConfig {
        &self.config
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CAUSAL ENHANCEMENT ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Check if causal enhancement is enabled
    pub fn causal_enhancement_enabled(&self) -> bool {
        self.causal_enhancer.is_some()
    }

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

    /// Force a causal discovery run (useful for testing)
    #[allow(dead_code)]
    pub(crate) fn force_causal_discovery(&mut self) -> Option<CausalGraph> {
        self.causal_enhancer.as_mut().map(|e| e.run_discovery())
    }

    /// Get causal attention weights for a target dimension
    #[allow(dead_code)]
    pub(crate) fn causal_attention_weights(&mut self, target_dim: usize) -> Option<Vec<f32>> {
        self.causal_enhancer
            .as_mut()
            .map(|e| e.causal_attention_weights(target_dim))
    }

    /// Suggest an intervention based on discovered causal structure
    #[allow(dead_code)]
    pub(crate) fn suggest_causal_intervention(&mut self) -> Option<(usize, f64)> {
        self.causal_enhancer
            .as_mut()
            .and_then(|e| e.suggest_intervention())
    }

    /// Get encoder statistics
    #[allow(dead_code)]
    pub(crate) fn encoder_stats(&self) -> &symthaea_core::hdc::predictive_encoder::EncoderStats {
        self.encoder.stats()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EPISODIC REPLAY ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Check if episodic replay is enabled
    pub fn episodic_replay_enabled(&self) -> bool {
        self.phi_episodic_replay.is_some()
    }

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

    /// Force an episodic replay session (useful for testing or manual consolidation)
    #[allow(dead_code)]
    pub(crate) fn force_episodic_replay(
        &mut self,
        learning_rate: f32,
    ) -> Option<crate::memory::episodic_replay::ReplaySessionResult> {
        if let Some(ref mut replay) = self.phi_episodic_replay {
            if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                // Temporarily bypass should_replay check by manually running replay
                let batch =
                    replay.sample_replay_batch(self.config.episodic_replay_config.batch_size);
                if batch.is_empty() {
                    return Some(crate::memory::episodic_replay::ReplaySessionResult {
                        episodes_replayed: 0,
                        average_loss: 0.0,
                        average_psi: 0.0,
                        skipped: true,
                    });
                }

                let mut total_loss = 0.0;
                let mut total_psi = 0.0;

                for episode in &batch {
                    let loss = replay.replay_training_step(
                        cfc,
                        episode,
                        learning_rate,
                        self.config.episodic_replay_config.replay_dt,
                    );
                    total_loss += loss;
                    total_psi += episode.psi;
                }

                let n = batch.len();
                return Some(crate::memory::episodic_replay::ReplaySessionResult {
                    episodes_replayed: n,
                    average_loss: total_loss / n as f32,
                    average_psi: total_psi / n as f64,
                    skipped: false,
                });
            }
        }
        None
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

    /// Check if loop is learning (error trend negative)
    pub fn is_learning(&self) -> bool {
        self.stats.error_trend < 0.0 && self.stats.learning_cycles > 0
    }

    /// Check if attention has emerged (variance > threshold)
    pub fn has_emerged_attention(&self) -> bool {
        self.stats.attention_variance > 0.01
    }

    /// Get coherence summary for external systems
    pub fn coherence_summary(&self) -> CoherenceSummary {
        self.coherence_bridge.summary()
    }

    /// Get temporal coherence value
    pub fn temporal_coherence(&self) -> f32 {
        self.coherence_bridge.smoothed_coherence()
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    /// Get exploration urge (0.0 to 1.0)
    pub fn exploration_urge(&self) -> f32 {
        self.curiosity_drive.exploration_urge
    }

    /// Check if curiosity-triggered exploration should occur
    pub fn curiosity_should_explore(&self) -> bool {
        self.curiosity_drive.should_explore()
    }

    /// Get curiosity drive reference
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    /// Get the thalamic router reference
    #[allow(dead_code)]
    pub(crate) fn thalamic_router(&self) -> &ThalamicRouter {
        &self.thalamic_router
    }

    /// Get thalamic routing statistics (reflex_rate, cortical_rate, deep_rate)
    pub fn thalamic_stats(&self) -> (f32, f32, f32) {
        self.thalamic_router.routing_stats()
    }

    /// Get the unified Phi from the ConsciousnessUnificationEngine
    pub fn unified_psi(&self) -> f64 {
        self.unification_engine.psi
    }

    /// Get the ConsciousnessUnificationEngine reference
    #[allow(dead_code)]
    pub(crate) fn unification_engine(&self) -> &ConsciousnessUnificationEngine {
        &self.unification_engine
    }

    /// Get mutable reference to the unification engine
    #[allow(dead_code)]
    pub(crate) fn unification_engine_mut(&mut self) -> &mut ConsciousnessUnificationEngine {
        &mut self.unification_engine
    }

    /// Get the unified emotional state (VAD-based)
    pub fn unified_emotional_state(&self) -> &UnifiedEmotionalState {
        self.unification_engine.emotional.state()
    }

    /// Get the emotional pattern (Stable/Escalating/Calming/Volatile)
    pub fn emotional_pattern(&self) -> EmotionalPattern {
        self.unification_engine.emotional.detect_pattern()
    }

    /// Get natural language description of current emotional state
    pub fn emotional_description(&self) -> String {
        self.unification_engine.emotional.state().describe()
    }

    /// Get the discrete unified emotion
    pub fn unified_emotion(&self) -> Option<UnifiedEmotion> {
        self.unification_engine.emotional.state().discrete_emotion
    }

    /// Process input through the unified dialogue pipeline
    pub fn process_unified(
        &mut self,
        input: &str,
    ) -> crate::consciousness::consciousness_unification::UnifiedConsciousnessResult {
        self.unification_engine.process(input)
    }

    /// Get a description of the current consciousness state
    pub fn unified_state_description(&self) -> String {
        self.unification_engine.describe_state()
    }

    /// Get the Active Inference Bridge reference
    #[allow(dead_code)]
    pub(crate) fn active_inference_bridge(&self) -> &ActiveInferenceBridge {
        &self.active_inference_bridge
    }

    /// Get the FEP Active Inference Agent reference
    #[allow(dead_code)]
    pub(crate) fn fep_agent(&self) -> &ActiveInferenceAgent {
        &self.fep_agent
    }

    /// Get the current FEP free energy (if available)
    pub fn fep_free_energy(&self) -> Option<f64> {
        self.fep_agent
            .last_fe_components
            .as_ref()
            .map(|fe| fe.total)
    }

    /// Get the conversation coherence tracker reference
    #[allow(dead_code)] // Used by fep_temporal_benchmark
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

    /// Check if prediction-outcome coupling is meaningful
    pub fn has_meaningful_coupling(&self) -> bool {
        self.active_inference_bridge
            .coupling_quality()
            .is_meaningful()
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

    /// Get the closed learning loop reference
    #[allow(dead_code)]
    pub(crate) fn closed_learning_loop(&self) -> &ClosedLearningLoop {
        &self.closed_learning_loop
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

    /// Get the episodic memory bridge reference
    #[allow(dead_code)]
    pub(crate) fn episodic_memory(&self) -> &EpisodicMemoryBridge {
        &self.episodic_memory
    }

    /// Get mutable reference to episodic memory
    #[allow(dead_code)]
    pub(crate) fn episodic_memory_mut(&mut self) -> &mut EpisodicMemoryBridge {
        &mut self.episodic_memory
    }

    /// Get memory counts (short_term, long_term)
    pub fn memory_counts(&self) -> (usize, usize) {
        self.episodic_memory.memory_count()
    }

    /// Recall memories similar to input
    pub fn recall_memories(&mut self, query: &[f32], top_k: usize) -> Vec<(EpisodicMemory, f32)> {
        self.episodic_memory.recall(query, top_k, 0.2)
    }

    /// Get the goal system bridge reference
    #[allow(dead_code)]
    pub(crate) fn goal_system(&self) -> &GoalSystemBridge {
        &self.goal_system
    }

    /// Get mutable reference to goal system
    #[allow(dead_code)]
    pub(crate) fn goal_system_mut(&mut self) -> &mut GoalSystemBridge {
        &mut self.goal_system
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
    #[allow(dead_code)]
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
    pub fn temporal_analyzer(&self) -> Option<&crate::consciousness::temporal_primitives::ConsciousnessTemporalAnalyzer> {
        self.temporal_analyzer.as_ref()
    }

    /// Borrow the primitive lattice (if enabled).
    pub fn primitive_lattice(&self) -> Option<&crate::consciousness::primitive_lattice::PrimitiveLattice> {
        self.primitive_lattice.as_ref()
    }

    /// Borrow the compositionality engine (if enabled).
    pub fn compositionality_engine(&self) -> Option<&crate::consciousness::compositionality::CompositionalityEngine> {
        self.compositionality_engine.as_ref()
    }

    /// Borrow the unified value evaluator (if enabled).
    pub fn value_evaluator(&self) -> Option<&crate::consciousness::unified_value_evaluator::UnifiedValueEvaluator> {
        self.value_evaluator.as_ref()
    }

    /// Borrow the harmonic field (if enabled).
    pub fn harmonic_field(&self) -> Option<&crate::consciousness::harmonics::HarmonicField> {
        self.harmonic_field.as_ref()
    }

    /// Borrow the primitive reasoner (if enabled).
    pub fn primitive_reasoner(&self) -> Option<&crate::consciousness::primitive_reasoning::PrimitiveReasoner> {
        self.primitive_reasoner.as_ref()
    }

    /// Borrow the adaptive reasoner (if enabled).
    pub fn adaptive_reasoner(&self) -> Option<&crate::consciousness::adaptive_reasoning::AdaptiveReasoner> {
        self.adaptive_reasoner.as_ref()
    }

    /// Borrow the causal self-explainer (if enabled).
    pub fn causal_explainer(&self) -> Option<&crate::consciousness::causal_explanation::CausalExplainer> {
        self.causal_explainer.as_ref()
    }

    /// Borrow the dissipative consciousness model (if enabled).
    pub fn dissipative_consciousness(&self) -> Option<&crate::consciousness::dissipative_consciousness::DissipativeConsciousness> {
        self.dissipative_consciousness.as_ref()
    }

    // epistemic_conflict_detector accessor removed — field lives in reasoning_engine feature gate.

    /// Borrow the consciousness equation v2 (if enabled).
    pub fn consciousness_equation_v2(&self) -> Option<&crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2> {
        self.consciousness_equation_v2.as_ref()
    }

    /// Borrow the hierarchical LTC (if enabled).
    pub fn hierarchical_ltc(&self) -> Option<&crate::consciousness::hierarchical_ltc::HierarchicalLTC> {
        self.hierarchical_ltc.as_ref()
    }

    // theory_calibrator accessor removed — field lives in reasoning_engine feature gate.

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
        self.experience_bus.as_ref().map(|bus| bus.current_guiding_question())
    }
}
