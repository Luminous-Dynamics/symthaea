// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flow state, emotion, curiosity, self-reflection, adaptive behavior, voice,
//! learning loop, and unified architecture accessors.

use crate::cognitive_loop::CognitiveLoopService;
use crate::consciousness::consciousness_unification::EmotionalPattern;
use crate::dynamics::temporal_signatures::ConsciousnessPattern;
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::voice::cognitive_bridge::VoiceConsciousnessSignals;
use crate::voice::voice_feedback::VoiceOutputMetrics;
use positioning::{GaussianEstimate3D, PeerEstimate3D, PublishableEstimate3D};

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
        pub fn in_flow(&self) -> bool { self.behavior.flow_state.in_flow }

        /// Get flow state intensity (0.0 to 1.0)
        pub fn flow_intensity(&self) -> f32 { self.behavior.flow_state.intensity }

        /// Get flow state streak (consecutive flow-compatible cycles)
        pub fn flow_streak(&self) -> u32 { self.behavior.flow_state.streak }

        /// Get current flow state reference
        pub(crate) fn flow_state(&self) -> &FlowState { &self.behavior.flow_state }

        /// Get flow learning boost multiplier
        pub fn flow_learning_boost(&self) -> f32 { self.behavior.flow_state.learning_boost }

        // ═══════════════════════════════════════════════════════════════════
        // EMOTION CONTAGION
        // ═══════════════════════════════════════════════════════════════════

        /// Get current emotional valence (from unified emotional state)
        pub fn emotional_valence(&self) -> f32 { self.unification_engine.emotional.state().valence as f32 }

        /// Get current emotional arousal (from unified emotional state)
        pub fn emotional_arousal(&self) -> f32 { self.unification_engine.emotional.state().arousal as f32 }

        /// Get emotion-based pattern nudge suggestion
        pub fn emotion_pattern_nudge(&self) -> (Option<ConsciousnessPattern>, f32) { self.behavior.emotion_contagion.pattern_nudge() }

        /// Get emotion contagion reference
        pub(crate) fn emotion_contagion(&self) -> &EmotionContagion { &self.behavior.emotion_contagion }

        // ═══════════════════════════════════════════════════════════════════
        // CURIOSITY DRIVE
        // ═══════════════════════════════════════════════════════════════════

        /// Get current boredom level (0.0 to 1.0)
        pub fn boredom(&self) -> f32 { self.behavior.curiosity_drive.boredom }

        /// Get curiosity level (0.0 to 1.0)
        pub fn curiosity(&self) -> f32 { self.behavior.curiosity_drive.curiosity }

        /// Check if curiosity-triggered exploration should occur
        pub fn curiosity_should_explore(&self) -> bool { self.behavior.curiosity_drive.should_explore() }

        /// Get curiosity drive reference
        pub(crate) fn curiosity_drive(&self) -> &CuriosityDrive { &self.behavior.curiosity_drive }

        /// Get current exploration urge (0.0 to 1.0)
        pub fn curiosity_drive_exploration_urge(&self) -> f64 { self.behavior.curiosity_drive.exploration_urge }

        /// Get novelty bonus for learning
        pub fn novelty_bonus(&self) -> f32 { self.behavior.curiosity_drive.novelty_bonus }

        /// Check if the system is bored (needs new stimuli)
        pub fn is_bored(&self) -> bool { self.behavior.curiosity_drive.boredom > 0.5 }

        // ═══════════════════════════════════════════════════════════════════
        // SELF-REFLECTION
        // ═══════════════════════════════════════════════════════════════════

        /// Get current self-assessment
        pub fn self_assessment(&self) -> SelfAssessment { self.consciousness.self_model_tier.self_reflection.self_assessment }

        /// Get self-reflection summary
        pub fn reflection_summary(&self) -> ReflectionSummary { self.consciousness.self_model_tier.self_reflection.summary() }

        /// Get adapted thresholds from self-reflection
        pub fn adapted_thresholds(&self) -> ReflectionThresholds { self.consciousness.self_model_tier.self_reflection.get_thresholds() }

        /// Get current recommendations from self-reflection
        pub fn recommendations(&self) -> &[Recommendation] { &self.consciousness.self_model_tier.self_reflection.recommendations }

        /// Get number of reflections performed
        pub fn reflection_count(&self) -> u64 { self.consciousness.self_model_tier.self_reflection.reflection_count }

        /// Get learning effectiveness score
        pub fn learning_effectiveness(&self) -> f32 { self.consciousness.self_model_tier.self_reflection.learning_effectiveness() }

        /// Check if system needs calibration
        pub fn needs_calibration(&self) -> bool { self.consciousness.self_model_tier.self_reflection.self_assessment == SelfAssessment::NeedsCalibration }

        /// Check if system is performing optimally
        pub fn is_optimal(&self) -> bool { self.consciousness.self_model_tier.self_reflection.self_assessment == SelfAssessment::Optimal }

        /// Get self-reflection reference
        pub(crate) fn self_reflection(&self) -> &SelfReflection { &self.consciousness.self_model_tier.self_reflection }

        // ═══════════════════════════════════════════════════════════════════
        // VOICE FEEDBACK (simple delegators)
        // ═══════════════════════════════════════════════════════════════════

        /// Get voice quality summary for external systems
        pub fn voice_feedback_summary(&self) -> VoiceQualitySummary { self.language_comm.voice_coherence.voice.summary() }

        /// Check if voice indicates uncertainty
        pub fn voice_indicates_uncertainty(&self) -> bool { self.language_comm.voice_coherence.voice.is_uncertain() }

        // ═══════════════════════════════════════════════════════════════════
        // MEGA-UNIFIED ARCHITECTURE
        // ═══════════════════════════════════════════════════════════════════

        /// Get current cognitive depth from thalamic routing
        pub fn cognitive_depth(&self) -> CognitiveDepth { self.cognitive_depth }

        /// Get thalamic routing statistics (reflex_rate, cortical_rate, deep_rate)
        pub fn thalamic_stats(&self) -> (f32, f32, f32) { self.behavior.thalamic_router.routing_stats() }

        /// Get the emotional pattern (Stable/Escalating/Calming/Volatile)
        pub fn emotional_pattern(&self) -> EmotionalPattern { self.unification_engine.emotional.detect_pattern() }

        /// Get natural language description of current emotional state
        pub fn emotional_description(&self) -> String { self.unification_engine.emotional.state().describe() }

        /// Get the conversation coherence tracker reference
        pub(crate) fn coherence_tracker(&self) -> &ConversationCoherenceTracker { &self.coherence_tracker }

        /// Get the coupling quality assessment
        pub fn coupling_quality(&self) -> CouplingQuality { self.fep.active_inference_bridge.coupling_quality() }

        // ═══════════════════════════════════════════════════════════════════
        // CLOSED LEARNING LOOP
        // ═══════════════════════════════════════════════════════════════════

        /// Get the current response strategy
        pub fn current_strategy(&self) -> ResponseStrategy { self.fep.closed_learning_loop.current_strategy }

        /// Get the best strategy according to Q-learning
        pub fn best_strategy(&self) -> ResponseStrategy { self.fep.closed_learning_loop.best_strategy() }

        /// Get average reward from the learning loop
        pub fn average_reward(&self) -> f32 { self.fep.closed_learning_loop.average_reward() }

        /// Get Q-values for all strategies
        pub fn strategy_q_values(&self) -> &[f32; 5] { self.fep.closed_learning_loop.q_values() }

        /// Get strategy usage counts
        pub fn strategy_usage_counts(&self) -> &[u64; 5] { self.fep.closed_learning_loop.strategy_counts() }

        /// Get the last learning result
        pub fn last_learning_result(&self) -> Option<&CycleLearningResult> { self.fep.closed_learning_loop.last_result.as_ref() }

        // ═══════════════════════════════════════════════════════════════════
        // ADAPTIVE BEHAVIOR
        // ═══════════════════════════════════════════════════════════════════

        /// Get current adaptive behavior
        pub(crate) fn adaptive_behavior(&self) -> &AdaptiveBehavior { &self.behavior.adaptive_behavior }

        /// Get current action hint
        pub fn action_hint(&self) -> ActionHint { self.behavior.adaptive_behavior.action_hint }

        /// Check if system should seek more input/clarification
        pub fn should_seek_input(&self) -> bool { self.behavior.adaptive_behavior.should_seek_input() }

        /// Check if system is in a confident state
        pub fn is_confident(&self) -> bool { self.behavior.adaptive_behavior.is_confident() }

        /// Get description of current adaptive state
        pub fn state_description(&self) -> &'static str { self.behavior.adaptive_behavior.description() }

        /// Get speech rate multiplier for voice synthesis
        pub fn speech_rate_multiplier(&self) -> f32 { self.behavior.adaptive_behavior.speech_rate_multiplier }

        /// Get pause duration multiplier for voice synthesis
        pub fn pause_multiplier(&self) -> f32 { self.behavior.adaptive_behavior.pause_multiplier }

        /// Get attention sensitivity for input processing
        pub fn attention_sensitivity(&self) -> f32 { self.behavior.adaptive_behavior.attention_sensitivity }

        /// Get exploration factor for decision making
        pub fn exploration_factor(&self) -> f32 { self.behavior.adaptive_behavior.exploration_factor }
    }

    /// Check if emotional content is significant
    pub fn has_emotional_content(&self) -> bool {
        (self.unification_engine.emotional.state().valence as f32).abs() > 0.2
    }

    /// Force an immediate reflection cycle
    pub fn force_reflect(&mut self) -> Vec<Recommendation> {
        self.consciousness.self_model_tier.self_reflection.reflect()
    }

    /// Update voice feedback with synthesis output metrics
    pub fn update_voice_feedback(&mut self, metrics: VoiceOutputMetrics) {
        self.language_comm.voice_coherence.voice.update(metrics);
    }

    /// Update listener prediction feedback
    pub fn update_listener_prediction(&mut self, success: f32) {
        self.language_comm
            .voice_coherence
            .voice
            .update_listener_prediction(success);
    }

    /// Get Phase 16 consciousness signals for voice prosody modulation.
    ///
    /// Returns a compact struct containing unified quality, epistemic gating,
    /// dissipative health, coherence velocity, and consciousness level —
    /// the signals needed by `CognitivePacing::from_cycle_metadata()`.
    pub fn voice_consciousness_signals(&self) -> VoiceConsciousnessSignals {
        let (_, pattern_confidence) = self.language_comm.voice_coherence.temporal.classify_state();
        let consciousness_level =
            super::super::snapshot::ConsciousnessSnapshot::compute_consciousness_level(
                self.prediction_confidence as f32,
                self.language_comm
                    .voice_coherence
                    .bridge
                    .smoothed_coherence(),
                self.behavior.flow_state.intensity,
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
            #[cfg(feature = "therapeutic")]
            client_distress: self.therapeutic_manager.client_distress(),
            #[cfg(feature = "therapeutic")]
            alliance_quality: self.therapeutic_manager.alliance_composite(),
            #[cfg(feature = "therapeutic")]
            therapeutic_intent: if self.therapeutic_manager.crisis_active {
                7.0
            } else {
                self.therapeutic_manager
                    .active_strategy()
                    .map(|s| match s {
                        symthaea_therapeutic::RegulationStrategy::Validation => 0.0,
                        symthaea_therapeutic::RegulationStrategy::Defusion => 2.0,
                        symthaea_therapeutic::RegulationStrategy::CognitiveReappraisal => 2.0,
                        symthaea_therapeutic::RegulationStrategy::Grounding => 4.0,
                        symthaea_therapeutic::RegulationStrategy::DistressTolerance => 4.0,
                        symthaea_therapeutic::RegulationStrategy::ExposurePrep => 5.0,
                        symthaea_therapeutic::RegulationStrategy::Containment => 6.0,
                    })
                    .unwrap_or(0.0)
            },
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
            articulation_quality: self
                .language_comm
                .voice_coherence
                .voice
                .smoothed_articulation(),
            rate_stability: self.language_comm.voice_coherence.voice.rate_stability(),
            integrated_phi: self
                .carryover
                .consciousness
                .last_spectral_mip_phi
                .unwrap_or(0.5) as f32,
            expected_free_energy: self
                .fep
                .agent
                .last_fe_components
                .as_ref()
                .map(|fe| fe.total as f32)
                .unwrap_or(1.0),
        }
    }

    /// Get combined phi contribution from all feedback sources
    pub fn combined_phi_contribution(&self) -> f32 {
        self.language_comm.voice_coherence.bridge.phi_contribution()
            + self
                .language_comm
                .voice_coherence
                .voice
                .compute_phi_adjustment()
    }

    /// Get the prediction-outcome coupling Modulation Index
    pub fn modulation_index(&self) -> Option<f64> {
        self.fep.active_inference_bridge.modulation_index()
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
        self.fep
            .agent
            .last_fe_components
            .as_ref()
            .map(|fe| fe.total)
    }

    /// Get combined learning rate modifier
    pub fn combined_learning_rate(&self) -> f32 {
        let coherence_lr = self
            .language_comm
            .voice_coherence
            .bridge
            .effective_learning_rate();
        let voice_modifier = self
            .language_comm
            .voice_coherence
            .voice
            .learning_rate_modifier();
        coherence_lr * voice_modifier
    }

    /// Inject external reward signal for the next cycle.
    /// Blended with internal prediction-error-based reward at 50% weight.
    /// Resets to 0.0 after consumption in the next cycle.
    pub fn provide_reward(&mut self, reward: f32) {
        self.behavior.social_mgr.social.external_reward = reward.clamp(-1.0, 1.0);
    }

    /// Inject social signals from Mind module's SocialCoherence.
    /// Called by the Symthaea facade after Mind.tick() computes social stats.
    ///
    /// `prediction_accuracy`: rolling ToM prediction accuracy (0.0–1.0).
    /// `models_count`: number of active mental models being tracked.
    /// `mean_trust`: mean trust across all tracked relationships.
    pub fn set_social_signals(
        &mut self,
        trust: f32,
        cooperation_rate: f32,
        prediction_accuracy: f32,
        models_count: usize,
        mean_trust: f32,
    ) {
        self.behavior.social_mgr.social.social_trust = trust.clamp(0.0, 1.0);
        self.behavior.social_mgr.social.social_cooperation_rate = cooperation_rate.clamp(0.0, 1.0);
        self.behavior.social_mgr.social.social_prediction_accuracy =
            prediction_accuracy.clamp(0.0, 1.0);
        self.behavior.social_mgr.social.social_models_count = models_count;
        self.behavior.social_mgr.social.social_mean_trust = mean_trust.clamp(0.0, 1.0);
    }

    /// Inject epistemic cube data from the facade's StructuredThought.
    ///
    /// Called by the Symthaea facade when it computes an EpistemicCube via
    /// domain context. The CLS uses this to enrich Broca generation with
    /// full 4D epistemic coordinates instead of deriving from confidence alone.
    pub fn inject_epistemic_cube(
        &mut self,
        e_tier: u8,
        n_tier: u8,
        m_tier: u8,
        h_value: f32,
        quality: f32,
    ) {
        self.carryover.quality.last_cube_e_tier = Some(e_tier.min(4));
        self.carryover.quality.last_cube_n_tier = Some(n_tier.min(3));
        self.carryover.quality.last_cube_m_tier = Some(m_tier.min(3));
        self.carryover.quality.last_cube_h_value = h_value.clamp(0.0, 1.0);
        self.carryover.quality.last_cube_quality = quality.clamp(0.0, 1.0);
    }

    /// Set the relational Psi from an external dyad computation.
    /// This is called by the Symthaea facade after computing Phi_dyad.
    pub fn set_relational_psi(&mut self, psi: f64) {
        self.behavior.social_mgr.social.relational_psi = psi;
    }

    /// Inject a governance event into the GovernanceManager for processing.
    ///
    /// Events are queued and drained during the next `process()` call (interval 37).
    /// Call this from the Mycelix bridge whenever governance activity occurs.
    #[cfg(feature = "mycelix")]
    pub fn inject_governance_event(
        &mut self,
        event: super::super::managers::governance_manager::GovernanceEvent,
    ) {
        // Bridge governance events to sentinel for threat detection
        #[cfg(feature = "sentinel")]
        if let Some(sentinel_event) =
            super::super::managers::sentinel_manager::bridge_governance_event(&event, "local")
        {
            self.sentinel_manager.inject_event(sentinel_event);
        }
        self.governance_mgr.inject_event(event);
    }

    /// Inject a governance outcome for learning feedback.
    ///
    /// Outcomes track whether our vote aligned with the result, enabling
    /// reward-based learning and confidence modulation.
    #[cfg(feature = "mycelix")]
    pub fn inject_governance_outcome(
        &mut self,
        outcome: super::super::managers::governance_manager::GovernanceOutcome,
    ) {
        self.governance_mgr.inject_outcome(outcome);
    }

    /// Poll a MycelixBridge for pending governance events and outcomes.
    ///
    /// Drains the bridge's internal event/outcome queues and injects them into
    /// the GovernanceManager. **External-facing**: call from the host application
    /// or Symthaea facade between cycles to keep governance synchronized with
    /// Holochain network activity. Not called internally by the cycle — the bridge
    /// is owned externally.
    #[cfg(feature = "mycelix")]
    pub fn poll_bridge_governance(
        &mut self,
        bridge: &mut crate::consciousness::mycelix_bridge::MycelixBridge,
    ) {
        let (events, outcomes) = bridge.drain_pending_governance();
        for event in events {
            // Bridge governance events to sentinel for threat detection
            #[cfg(feature = "sentinel")]
            if let Some(sentinel_event) =
                super::super::managers::sentinel_manager::bridge_governance_event(&event, "bridge")
            {
                self.sentinel_manager.inject_event(sentinel_event);
            }
            self.governance_mgr.inject_event(event);
        }
        for outcome in outcomes {
            self.governance_mgr.inject_outcome(outcome);
        }
    }

    /// Drain pending crisis events and forward them to the Mycelix civic bridge.
    ///
    /// Call from the host application between cycles to dispatch crisis events
    /// to the Holochain emergency-incidents zome. Each event is forwarded via
    /// `MycelixBridge::dispatch_crisis()`.
    #[cfg(all(feature = "mycelix", feature = "safety-agents"))]
    pub fn poll_bridge_crisis(
        &mut self,
        bridge: &mut crate::consciousness::mycelix_bridge::MycelixBridge,
    ) {
        let events: Vec<_> = self.pending_crisis_events.drain(..).collect();
        for event in &events {
            bridge.dispatch_crisis(event);
        }
    }

    /// Drain pending crisis events without a bridge (for testing or logging).
    #[cfg(feature = "safety-agents")]
    pub fn drain_pending_crisis_events(
        &mut self,
    ) -> Vec<super::super::civic_crisis_detector::CivicCrisisEvent> {
        self.pending_crisis_events.drain(..).collect()
    }

    /// Override the epistemic mesh with collective network data.
    ///
    /// Call from the host application after aggregating EpistemicSummary vectors
    /// from the Mycelix network. Once set, the external mesh is **persistent** —
    /// it will NOT be overwritten by the local single-agent fallback. To revert
    /// to local-only, call `clear_governance_external_mesh()`.
    #[cfg(feature = "mycelix")]
    pub fn set_governance_epistemic_mesh(
        &mut self,
        mesh: crate::mycelix::epistemic_mesh::EpistemicMesh,
    ) {
        self.governance_mgr.set_epistemic_mesh(mesh);
    }

    /// Override the community mode with collective identity data.
    ///
    /// Call from the host application after computing CollectiveKosmicSong from
    /// the network. Overrides the single-agent fallback.
    #[cfg(feature = "mycelix")]
    pub fn set_governance_community_mode(
        &mut self,
        mode: crate::mycelix::collective_identity::CommunityMode,
    ) {
        self.governance_mgr.set_community_mode(mode);
    }

    /// Current community mode (if known).
    #[cfg(feature = "mycelix")]
    pub fn governance_community_mode(
        &self,
    ) -> Option<crate::mycelix::collective_identity::CommunityMode> {
        self.governance_mgr.community_mode()
    }

    /// Extract the 6D consciousness vector for collective Phi computation
    /// from a CycleResult's metadata.
    ///
    /// Returns `(consciousness_level, meta_awareness, coherence, care_activation,
    ///           quality_score, epistemic_confidence)` — suitable for constructing
    /// an `AgentConsciousnessVector` for the collective_phi engine.
    ///
    /// Usage: call after `cycle()` and pass the result metadata.
    #[cfg(feature = "mycelix")]
    pub fn governance_consciousness_vector(
        metadata: &crate::cognitive_loop::CycleMetadata,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let lr = metadata.actual_effective_lr as f64;
        (
            metadata.consciousness.consciousness_level,
            lr,                                               // meta-awareness proxy
            metadata.consciousness.consciousness_level * 0.8, // coherence proxy
            metadata.consciousness.consciousness_level * 0.6, // care activation proxy
            metadata.consciousness.consciousness_level * 0.9, // quality proxy
            lr.min(1.0),                                      // epistemic confidence proxy
        )
    }

    /// Get the lagged consciousness level for governance gating.
    ///
    /// Uses a 50-cycle lag (~2.5s at 20Hz) to decorrelate the circular
    /// dependency: consciousness → governance → neuromod → consciousness.
    /// Returns the oldest value in the ring buffer, or 0.0 if empty.
    /// Science: Granger (1969) — temporal decorrelation breaks circular causation.
    pub fn governance_lagged_consciousness(&self) -> f64 {
        self.governance_consciousness_lag
            .front()
            .copied()
            .unwrap_or(0.0)
    }

    /// Record a predicted outcome alignment for a governance proposal.
    ///
    /// Call this when casting a vote to enable governance prediction error
    /// computation in Phase 2 learning.
    #[cfg(feature = "mycelix")]
    pub fn predict_governance_outcome(&mut self, proposal_id: String, predicted_alignment: f64) {
        self.governance_mgr
            .predict_outcome(proposal_id, predicted_alignment);
    }

    #[cfg(feature = "mycelix")]
    pub fn governance_reward_ema(&self) -> f64 {
        self.governance_mgr.reward_ema()
    }

    #[cfg(feature = "mycelix")]
    pub fn governance_pending_count(&self) -> usize {
        self.governance_mgr.pending_event_count()
    }

    /// Current financial health signals from the Mycelix finance cluster.
    #[cfg(feature = "mycelix")]
    pub fn finance_health(&self) -> &crate::consciousness::mycelix_bridge::FinanceHealthSignals {
        self.governance_mgr.finance_health()
    }

    /// Update cached financial health signals from the Mycelix finance cluster.
    #[cfg(feature = "mycelix")]
    pub fn update_finance_health(
        &mut self,
        signals: crate::consciousness::mycelix_bridge::FinanceHealthSignals,
    ) {
        self.governance_mgr.update_finance_health(signals);
    }

    /// Process governance learning signals: reward, harmonic deltas, episodic memory.
    ///
    /// Called after governance processing. Feeds reward to the FEP learning loop,
    /// applies harmonic deltas to the experience bus's KosmicSong, and drains
    /// completed outcomes for episodic recording.
    #[cfg(feature = "mycelix")]
    pub(crate) fn process_governance_learning(&mut self) {
        // 1. Feed governance reward to provide_reward()
        if let Some(reward) = self.governance_mgr.take_latest_reward() {
            self.behavior.social_mgr.social.external_reward = reward.clamp(-1.0, 1.0);
        }

        // 2. Update KosmicSong harmonic weights from governance feedback
        let deltas = self.governance_mgr.take_harmonic_deltas();
        let has_deltas = deltas.iter().any(|d| d.abs() > 1e-10);
        if has_deltas {
            if let Some(ref mut bus) = self.experience_bus {
                bus.kosmic_state.harmonies.apply_governance_deltas(&deltas);
            }
        }

        // 3. Drain completed outcomes and record as episodic memories
        let completed = self.governance_mgr.drain_completed();
        if !completed.is_empty() {
            if let Some(ref mut bus) = self.experience_bus {
                for (outcome, gov_pe) in &completed {
                    // Generate deterministic HDV from governance outcome content via BLAKE3 XOF.
                    // 256 floats in [-1,1] — compact but retrievable by similarity.
                    let gov_hdv = {
                        let content = format!(
                            "gov:{}:{}:{:.4}:{:.4}",
                            outcome.proposal_id,
                            outcome.passed,
                            outcome.value_alignment_score,
                            outcome.harmonic_resonance,
                        );
                        let mut hasher = blake3::Hasher::new();
                        hasher.update(content.as_bytes());
                        let mut xof = hasher.finalize_xof();
                        let mut embedding = vec![0f32; 256];
                        for val in &mut embedding {
                            let mut buf = [0u8; 4];
                            xof.fill(&mut buf);
                            *val = (i32::from_le_bytes(buf) as f32) / (i32::MAX as f32);
                        }
                        embedding
                    };
                    // Use governance prediction error for dream consolidation priority:
                    // high PE → surprising outcome → prioritized during sleep consolidation.
                    let pe_for_memory = *gov_pe as f32;
                    // Salience scales with PE: surprising outcomes are more memorable.
                    let salience =
                        if outcome.passed { 0.7 } else { 0.5 } + (pe_for_memory * 0.3).min(0.3);
                    let memory = crate::experience::memory::EpisodicMemory {
                        id: format!("gov-{}", outcome.proposal_id),
                        timestamp: self.stats.total_cycles as u64,
                        hdv_embedding: gov_hdv,
                        thought_primitives: vec!["governance_outcome".to_string()],
                        context_hash: outcome.proposal_id.clone(),
                        user_id: None,
                        prediction_error: pe_for_memory,
                        uncertainty: if outcome.my_vote_aligned.is_some() {
                            0.2
                        } else {
                            0.8
                        },
                        coherence: outcome.harmonic_resonance as f32,
                        confidence: if outcome.my_vote_aligned == Some(true) {
                            0.8
                        } else {
                            0.3
                        },
                        salience,
                        kosmic_snapshot: None,
                        outcome: None,
                        input_summary: format!(
                            "Governance: proposal {} {}",
                            outcome.proposal_id,
                            if outcome.passed { "passed" } else { "failed" }
                        ),
                        output_summary: format!(
                            "alignment={:.2}, resonance={:.2}, pe={:.3}",
                            outcome.value_alignment_score, outcome.harmonic_resonance, gov_pe,
                        ),
                    };
                    bus.record_experience(memory);
                }
            }
        }
    }

    /// Apply pending neuromodulatory effects from governance events.
    ///
    /// Drains the GovernanceManager's injection and baseline queues, applying
    /// each to the neuromodulator bath. Called after governance processing
    /// in the feedback phase.
    #[cfg(feature = "mycelix")]
    pub(crate) fn apply_governance_neuromod(&mut self) {
        use super::super::thresholds::{NEUROMOD_BASELINE_MAX, NEUROMOD_BASELINE_MIN};
        let injections = self.governance_mgr.drain_injections();
        for inj in injections {
            self.neuromod
                .bath
                .inject(inj.target, inj.dose, inj.half_life);
        }

        let baselines = self.governance_mgr.drain_baselines();
        for bl in baselines {
            match bl.target {
                "dopamine" => {
                    self.neuromod.bath.dopamine.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "noradrenaline" => {
                    self.neuromod.bath.noradrenaline.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "serotonin" => {
                    self.neuromod.bath.serotonin.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "oxytocin" => {
                    self.neuromod.bath.oxytocin.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "endocannabinoid" => {
                    self.neuromod.bath.endocannabinoid.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "acetylcholine" => {
                    self.neuromod.bath.acetylcholine.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "gaba" => {
                    self.neuromod.bath.gaba.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "glutamate" => {
                    self.neuromod.bath.glutamate.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "adenosine" => {
                    self.neuromod.bath.adenosine.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                _ => {
                    tracing::warn!(
                        target = bl.target,
                        "Unknown neuromod target in governance baseline"
                    );
                }
            }
        }
    }

    /// Drain fabrication manager's neuromod queues and apply each injection/baseline
    /// to the neuromodulator bath. Called after fabrication processing.
    #[cfg(feature = "advanced-manufacturing")]
    pub(crate) fn apply_fabrication_neuromod(&mut self) {
        use super::super::thresholds::{NEUROMOD_BASELINE_MAX, NEUROMOD_BASELINE_MIN};
        let injections = self.fabrication_manager.drain_injections();
        for inj in injections {
            self.neuromod
                .bath
                .inject(inj.target, inj.dose, inj.half_life);
        }

        let baselines = self.fabrication_manager.drain_baselines();
        for bl in baselines {
            match bl.target {
                "dopamine" => {
                    self.neuromod.bath.dopamine.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "norepinephrine" => {
                    self.neuromod.bath.noradrenaline.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "serotonin" => {
                    self.neuromod.bath.serotonin.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                "oxytocin" => {
                    self.neuromod.bath.oxytocin.adjust_baseline(
                        bl.nudge,
                        NEUROMOD_BASELINE_MIN,
                        NEUROMOD_BASELINE_MAX,
                    );
                }
                _ => {
                    tracing::warn!(
                        target = bl.target,
                        "Unknown neuromod target in fabrication baseline"
                    );
                }
            }
        }
    }

    /// Inject a Mycelix ConsciousnessProfile back into the cognitive loop,
    /// closing the bidirectional consciousness bridge.
    ///
    /// Forward path:  Symthaea C_unified → Mycelix engagement dimension
    /// Reverse path:  Mycelix 4D profile → Symthaea social state + neuromod
    ///
    /// Mapping:
    /// - `community` → `social_trust` (peer trust attestations → social trust)
    /// - `reputation` → `social_cooperation_rate` (cross-hApp rep → cooperation)
    /// - `combined_score` → `social_mean_trust` (overall profile → mean trust)
    /// - `identity` → `social_prediction_accuracy` (verification → confidence)
    ///
    /// When neuromodulators are available, `community` also modulates oxytocin
    /// (social bonding hormone scales with peer trust).
    pub fn inject_mycelix_profile(
        &mut self,
        identity: f64,
        reputation: f64,
        community: f64,
        engagement: f64,
    ) {
        let combined = identity * 0.25 + reputation * 0.25 + community * 0.30 + engagement * 0.20;

        self.behavior.social_mgr.social.social_trust = (community as f32).clamp(0.0, 1.0);
        self.behavior.social_mgr.social.social_cooperation_rate =
            (reputation as f32).clamp(0.0, 1.0);
        self.behavior.social_mgr.social.social_mean_trust = (combined as f32).clamp(0.0, 1.0);
        self.behavior.social_mgr.social.social_prediction_accuracy =
            (identity as f32).clamp(0.0, 1.0);

        // Modulate oxytocin from community dimension (social bonding).
        // Community trust maps to 0.0–0.3 oxytocin injection (conservative range).
        let oxy_dose = (community * 0.3).clamp(0.0, 0.3) as f32;
        if oxy_dose > 0.01 {
            self.neuromod.bath.inject("oxytocin", oxy_dose, 50);
        }
    }

    // ========================================================================
    // SWARM MANAGER ACCESSORS
    // ========================================================================

    /// Get a clone of the swarm event sender.
    ///
    /// The channel is created eagerly at CLS construction. Clone the returned
    /// sender and pass it to async components (NetworkServiceBridge, Hyperfeel,
    /// FederatedAggregator, ContinuousMind) so they can inject events into the
    /// cognitive loop. The receiver is drained non-blocking in Phase B.
    pub fn swarm_event_sender(
        &self,
    ) -> std::sync::mpsc::Sender<super::super::managers::swarm_manager::SwarmEvent> {
        self.swarm_event_tx.clone()
    }

    /// Take ownership of the safety alert receiver.
    ///
    /// The host application calls this once at startup and spawns a thread/task
    /// to drain alerts and forward them to desktop notifications, Slack, email,
    /// or any monitoring system. Returns `None` if already taken.
    pub fn take_safety_alert_receiver(
        &self,
    ) -> Option<std::sync::mpsc::Receiver<super::super::safety_alert::SafetyAlert>> {
        self.safety_alert_rx.lock().ok()?.take()
    }

    /// Enable network attestation for the cognitive loop.
    ///
    /// Creates a `NetworkService`, initializes Ed25519 attestation, and spawns
    /// an attestation-aware bridge that feeds verified swarm events into the
    /// cognitive loop's Phase B drain.
    ///
    /// This is the **production wiring point** for P2P consciousness sharing
    /// with cryptographic verification. Call once after CLS construction.
    ///
    /// # Requirements
    /// - `identity` feature must be enabled
    /// - `swarm` feature must be enabled for real networking (stub without)
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut cls = CognitiveLoopService::new(config)?;
    /// let rt = tokio::runtime::Handle::current();
    /// rt.block_on(cls.enable_network_attestation())?;
    /// ```
    #[cfg(feature = "identity")]
    pub async fn enable_network_attestation(&mut self) -> anyhow::Result<()> {
        use super::super::managers::network_service_bridge::NetworkServiceBridge;
        use crate::swarm::{NetworkService, SwarmConfig};

        let swarm_config = SwarmConfig::production();
        #[cfg(feature = "mesh")]
        self.spectrum_manager
            .set_domain_profile(swarm_config.domain.clone());
        let mut service = NetworkService::new(swarm_config)
            .await
            .map_err(|e| anyhow::anyhow!("NetworkService init failed: {e}"))?;

        let attestation_mgr = service
            .initialize_attestation()
            .map_err(|e| anyhow::anyhow!("Attestation init failed: {e}"))?;

        let service = std::sync::Arc::new(service);

        let tx = self.swarm_event_sender();
        let _bridge = NetworkServiceBridge::spawn_with_attestation(
            service.as_ref(),
            tx,
            Some(attestation_mgr.clone()),
        );

        self.network_service = Some(service);

        let pubkey = attestation_mgr.read().public_key_hex().to_string();
        tracing::info!(
            pubkey = %&pubkey[..16],
            "Network attestation enabled — verified CV exchange active"
        );

        Ok(())
    }

    /// Take the mesh outbound receiver (one-shot — returns None after first call).
    ///
    /// ContinuousMind calls this once during wiring to drain CLS-generated
    /// mesh packets (beacons, name responses, content announces) each tick.
    #[cfg(feature = "mesh")]
    pub fn take_mesh_outbound_rx(
        &self,
    ) -> Option<std::sync::mpsc::Receiver<crate::swarm::mesh::MeshOutbound>> {
        self.mesh_outbound_rx.lock().ok()?.take()
    }

    /// Backwards-compatible alias for `swarm_event_sender()`.
    pub fn create_swarm_event_channel(
        &self,
    ) -> std::sync::mpsc::Sender<super::super::managers::swarm_manager::SwarmEvent> {
        self.swarm_event_sender()
    }

    /// Inject a swarm event into the SwarmManager for processing.
    ///
    /// Events are queued and drained during the next `process()` call (interval 41).
    pub fn inject_swarm_event(&mut self, event: super::super::managers::swarm_manager::SwarmEvent) {
        self.swarm_manager.inject_event(event);
    }

    /// Get the current swarm telemetry snapshot.
    pub fn swarm_telemetry(&self) -> &super::super::managers::swarm_manager::SwarmTelemetry {
        self.swarm_manager.telemetry()
    }

    /// Set the expected peer count for connectivity ratio calculation.
    pub fn set_swarm_expected_peers(&mut self, n: usize) {
        self.swarm_manager.set_expected_peers(n);
    }

    /// Current number of connected swarm peers.
    pub fn swarm_connected_peers(&self) -> usize {
        self.swarm_manager.connected_peers()
    }

    /// Mean peer Φ across connected swarm peers.
    pub fn swarm_mean_peer_phi(&self) -> f64 {
        self.swarm_manager.mean_peer_phi()
    }

    /// Borrow the retained swarm network service, if one is active.
    pub fn network_service(&self) -> Option<&std::sync::Arc<crate::swarm::NetworkService>> {
        self.network_service.as_ref()
    }

    /// Publish a local navigation estimate into the retained swarm service.
    ///
    /// Returns the peer-shareable estimate if a live network service is active.
    pub fn publish_local_navigation_estimate<E: PublishableEstimate3D>(
        &self,
        estimate: &E,
        confidence: Option<f64>,
    ) -> Option<PeerEstimate3D> {
        self.network_service
            .as_ref()
            .map(|service| service.publish_local_navigation_estimate(estimate, confidence))
    }

    /// Receive a peer navigation estimate in conservative Gaussian form.
    pub fn receive_peer_navigation_estimate(
        &self,
        peer_id: &str,
        estimate: GaussianEstimate3D,
        confidence: Option<f64>,
    ) {
        if let Some(service) = &self.network_service {
            service.receive_publishable_navigation(peer_id, &estimate, confidence);
        }
    }

    // ========================================================================
    // HOLON RECEIVER (Soma↔Desktop bridge)
    // ========================================================================

    /// Enqueue a message from a connected Soma device.
    ///
    /// Called by the WebSocket handler when a Soma sends JSON over `/v1/ws/soma`.
    /// Messages are processed during the next cycle's Phase B.
    pub fn holon_enqueue_soma_message(
        &mut self,
        device_id: String,
        msg: crate::consciousness::holon_receiver::SomaMessage,
    ) {
        self.holon_receiver.enqueue_message(device_id, msg);
    }

    /// Number of connected Soma devices.
    pub fn holon_soma_peer_count(&self) -> usize {
        self.holon_receiver.peer_count()
    }

    /// Send a response to a specific Soma device.
    pub fn holon_send_to_soma(
        &mut self,
        device_id: &str,
        response: crate::consciousness::holon_receiver::HolonResponse,
    ) {
        self.holon_receiver.send_to_device(device_id, response);
    }

    /// Drain outbound responses for a specific Soma device (for WebSocket delivery).
    pub fn holon_drain_soma_outbound(
        &mut self,
        device_id: &str,
    ) -> Vec<crate::consciousness::holon_receiver::HolonResponse> {
        self.holon_receiver.drain_outbound(device_id)
    }

    /// Total messages processed by the HolonReceiver.
    pub fn holon_total_processed(&self) -> u64 {
        self.holon_receiver.total_processed()
    }

    /// Collective QOL summary across all connected Soma devices as JSON.
    ///
    /// Used by the Holon dashboard for quick polling without locking/serializing in the HTTP layer.
    pub fn holon_collective_qol_json(&self) -> String {
        self.holon_receiver.collective_qol_json()
    }

    /// Drain all pending Holon task requests across peers.
    ///
    /// Returns (device_id, task_type, payload) tuples.
    pub fn holon_drain_task_requests(&mut self) -> Vec<(String, String, String)> {
        self.holon_receiver.drain_task_requests()
    }

    /// Clone the Holon inbound sender for use by HTTP handlers.
    ///
    /// The channel is created eagerly at CLS construction. Clone the returned
    /// sender and pass it to `HolonHttpState::new(tx)` so HTTP handlers can
    /// inject SomaMessages into the cognitive loop. The receiver is drained
    /// non-blocking in Phase B before `holon_receiver.process_inbound()`.
    pub fn holon_inbound_sender(
        &self,
    ) -> std::sync::mpsc::Sender<(String, crate::consciousness::holon_receiver::SomaMessage)> {
        self.holon_inbound_tx.clone()
    }

    // ========================================================================
    // SWARM NEUROMODULATORY COUPLING
    // ========================================================================

    /// Apply swarm-derived neuromodulatory effects to the bath.
    ///
    /// Mapping (all doses use named constants from `thresholds.rs`):
    /// - Peer connectivity → oxytocin injection (social buffering, Zak 2012)
    /// - Network anomalies → NE baseline nudge (arousal/vigilance, Arnsten 2009)
    /// - High mean peer Φ → 5-HT baseline nudge (social confidence, Crockett 2009)
    /// - Strong affective contagion → DA baseline nudge (reward salience, Schultz 1997)
    pub(crate) fn apply_swarm_neuromod(&mut self) {
        use super::super::thresholds::{
            NEUROMOD_BASELINE_MAX, NEUROMOD_BASELINE_MIN, SWARM_ANOMALY_NE_CAP,
            SWARM_ANOMALY_NE_MULT, SWARM_CONTAGION_DA_CAP, SWARM_CONTAGION_DA_GAIN,
            SWARM_CONTAGION_DA_THRESHOLD, SWARM_OXY_CAP, SWARM_OXY_HALFLIFE,
            SWARM_OXY_PER_SQRT_PEER, SWARM_PHI_SHT_CAP, SWARM_PHI_SHT_GAIN,
        };
        let telem = self.swarm_manager.telemetry().clone();

        // Social buffering: peer count → oxytocin (Zak 2012)
        if telem.connected_peers > 0 {
            let oxy_dose = ((telem.connected_peers as f32).sqrt() * SWARM_OXY_PER_SQRT_PEER)
                .min(SWARM_OXY_CAP);
            if oxy_dose > super::super::thresholds::GOV_NEUROMOD_FLOOR {
                self.neuromod
                    .bath
                    .inject("oxytocin", oxy_dose, SWARM_OXY_HALFLIFE);
            }
        }

        // Anomaly vigilance: sudden peer loss → NE (Arnsten 2009)
        if telem.anomaly_count > 0 {
            let ne_nudge = (SWARM_ANOMALY_NE_MULT * telem.anomaly_count.min(3) as f32)
                .min(SWARM_ANOMALY_NE_CAP);
            self.neuromod.bath.noradrenaline.adjust_baseline(
                ne_nudge,
                NEUROMOD_BASELINE_MIN,
                NEUROMOD_BASELINE_MAX,
            );
        }

        // Collective coherence: high peer Φ → 5-HT (Crockett 2009)
        if telem.mean_peer_phi > 0.5 {
            let sht_nudge = ((telem.mean_peer_phi - 0.5) * SWARM_PHI_SHT_GAIN as f64)
                .min(SWARM_PHI_SHT_CAP as f64) as f32;
            if sht_nudge > super::super::thresholds::GOV_NEUROMOD_FLOOR {
                self.neuromod.bath.serotonin.adjust_baseline(
                    sht_nudge,
                    NEUROMOD_BASELINE_MIN,
                    NEUROMOD_BASELINE_MAX,
                );
            }
        }

        // Affective contagion: strong emotional sync → DA (Schultz 1997)
        if telem.affective_contagion > SWARM_CONTAGION_DA_THRESHOLD {
            let da_nudge = (telem.affective_contagion * SWARM_CONTAGION_DA_GAIN as f64)
                .min(SWARM_CONTAGION_DA_CAP as f64) as f32;
            self.neuromod.bath.dopamine.adjust_baseline(
                da_nudge,
                NEUROMOD_BASELINE_MIN,
                NEUROMOD_BASELINE_MAX,
            );
        }
    }

    /// Bidirectional coupling between SwarmManager and GovernanceManager.
    ///
    /// - High peer Φ → governance confidence boost (Woolley 2010: collective intelligence)
    /// - Network anomalies → governance confidence decrease (uncertainty signal)
    /// - Positive governance reward EMA → expected peer growth (success attracts)
    #[cfg(feature = "mycelix")]
    pub(crate) fn cross_couple_swarm_governance(&mut self) {
        let swarm_telem = self.swarm_manager.telemetry().clone();

        // Collective intelligence: high peer Φ → governance confidence (Woolley 2010)
        if swarm_telem.mean_peer_phi > 0.3 && swarm_telem.connected_peers > 1 {
            let phi_boost = ((swarm_telem.mean_peer_phi - 0.3) * 0.05).min(0.03);
            self.governance_mgr.nudge_confidence(phi_boost);
        }

        // Network instability: anomalies → lower governance confidence
        if swarm_telem.anomaly_count > 2 {
            self.governance_mgr.nudge_confidence(-0.02);
        }

        // Success attracts: positive governance rewards → grow expected peers
        let reward = self.governance_mgr.reward_ema();
        if reward > 0.05 {
            let current = self.swarm_manager.expected_peers();
            let growth = ((reward * 5.0) as usize).min(10);
            self.swarm_manager
                .set_expected_peers((current + growth).min(256));
        }
    }

    /// Cross-couple drive state → learning plasticity.
    ///
    /// - Boredom > 0.5 → boost plasticity by 10% (Berlyne 1960: boredom drives exploratory learning)
    /// - In flow → dampen LR modulation to 0.9 (Csikszentmihalyi 1990: don't disturb flow)
    pub(crate) fn cross_couple_drive_learning(&mut self) {
        let boredom = self.drive_manager.boredom();
        let in_flow = self.drive_manager.in_flow();

        // Boredom → plasticity boost: system is under-stimulated, open up to new learning
        if boredom > super::super::thresholds::DRIVE_BOREDOM_PLASTICITY_THRESHOLD {
            let boost = (boredom - super::super::thresholds::DRIVE_BOREDOM_PLASTICITY_THRESHOLD)
                * super::super::thresholds::DRIVE_BOREDOM_PLASTICITY_GAIN;
            // Direct plasticity nudge (within LearningManager's clamp range)
            let current = self.learning_manager.plasticity();
            let nudged = (current + boost).min(0.95);
            if nudged > current {
                // LearningManager doesn't expose set_plasticity — use the LR modulation
                // channel instead: boost carryover LR factor
                self.carryover.learning.subsystem_lr_factor *= 1.0 + boost;
            }
        }

        // Flow → LR dampening: stable parameters benefit flow state maintenance
        if in_flow {
            self.carryover.learning.subsystem_lr_factor *= 0.9;
        }
    }

    /// Cross-couple knowledge causal depth → moral reasoning confidence.
    ///
    /// Deep causal understanding (normalized depth > 0.6) → nudge prediction confidence
    /// by +0.01 per excess depth unit. This flows into the ethics engine evaluation,
    /// producing more confident moral verdicts when grounded in causal reasoning.
    ///
    /// Science: Pearl (2009) — deeper causal understanding → more confident moral reasoning.
    pub(crate) fn cross_couple_knowledge_ethics(&mut self) {
        if let Some(ref km) = self.memory.knowledge_manager {
            let depth = km.signals().causal_depth;
            if depth.is_finite()
                && depth > super::super::thresholds::KNOWLEDGE_ETHICS_CAUSAL_DEPTH_THRESHOLD
            {
                let nudge = (depth
                    - super::super::thresholds::KNOWLEDGE_ETHICS_CAUSAL_DEPTH_THRESHOLD)
                    * super::super::thresholds::KNOWLEDGE_ETHICS_CONFIDENCE_GAIN;
                let nudge = nudge.min(0.03); // cap to prevent runaway
                self.prediction_confidence = (self.prediction_confidence + nudge).clamp(0.0, 1.0);
            }
        }
    }

    /// Cross-couple memory consolidation state → learning plasticity.
    ///
    /// - High consolidation pressure (> 0.6) → boost LR factor (Born & Wilhelm 2012)
    /// - Low recall quality (< 0.3) → dampen LR factor (Tulving 2002)
    pub(crate) fn cross_couple_memory_learning(&mut self) {
        let pressure = self.memory_manager.consolidation_pressure();
        let recall = self.memory_manager.recall_quality();

        // High consolidation pressure → boost learning (primed for encoding)
        if pressure > super::super::thresholds::MEMORY_CONSOLIDATION_PLASTICITY_THRESHOLD {
            let boost = (pressure
                - super::super::thresholds::MEMORY_CONSOLIDATION_PLASTICITY_THRESHOLD)
                * super::super::thresholds::MEMORY_CONSOLIDATION_PLASTICITY_GAIN;
            self.carryover.learning.subsystem_lr_factor *= 1.0 + boost;
        }

        // Low recall quality → dampen learning (encoding unreliable)
        if recall < super::super::thresholds::MEMORY_RECALL_QUALITY_DAMPEN_THRESHOLD {
            let deficit = super::super::thresholds::MEMORY_RECALL_QUALITY_DAMPEN_THRESHOLD - recall;
            let dampening = deficit * super::super::thresholds::MEMORY_RECALL_QUALITY_DAMPEN_SCALE;
            self.carryover.learning.subsystem_lr_factor *= (1.0 - dampening).max(0.8);
        }
    }

    /// Cross-couple perception state → drive exploration.
    ///
    /// - Low perceptual coherence → boost exploration (Damasio 1994: orienting response)
    /// - High perceptual load → suppress exploration (Lavie 2005: load theory)
    pub(crate) fn cross_couple_perception_drive(&mut self) {
        let coherence = self.perception_manager.mean_coherence_score();
        let utilization = self.perception_manager.budget_utilization();

        // Low coherence → exploration boost (orienting reflex)
        if coherence < super::super::thresholds::PERCEPTION_LOW_COHERENCE_THRESHOLD {
            let deficit = super::super::thresholds::PERCEPTION_LOW_COHERENCE_THRESHOLD - coherence;
            let boost = deficit * super::super::thresholds::PERCEPTION_LOW_COHERENCE_EXPLORE_GAIN;
            // Nudge confidence down slightly to encourage exploration
            self.prediction_confidence =
                (self.prediction_confidence - boost as f64).clamp(0.0, 1.0);
        }

        // High perceptual load → suppress exploration (conserve resources)
        if utilization > super::super::thresholds::PERCEPTION_HIGH_LOAD_SUPPRESS_THRESHOLD {
            let excess =
                utilization - super::super::thresholds::PERCEPTION_HIGH_LOAD_SUPPRESS_THRESHOLD;
            let suppression =
                excess * super::super::thresholds::PERCEPTION_HIGH_LOAD_SUPPRESS_FACTOR;
            // Boost confidence slightly to discourage exploration
            self.prediction_confidence =
                (self.prediction_confidence + suppression as f64).clamp(0.0, 1.0);
        }
    }

    /// Cross-couple visual surprise → prediction confidence modulation.
    ///
    /// High visual prediction error (PE EMA > 0.4) → reduce prediction confidence
    /// to encourage exploratory attention toward novel visual input. Low surprise
    /// (< 0.1) with elevated confidence → gentle pull-back toward baseline.
    ///
    /// Science: Itti & Koch (2001) — bottom-up saliency captures attention.
    #[cfg(feature = "vision-manifold")]
    pub(crate) fn cross_couple_vision_attention(&mut self) {
        let visual_pe = self.vision_manager.visual_pe_ema();

        if visual_pe > 0.4 {
            // High visual surprise → reduce confidence (attention proxy for exploration)
            let nudge = ((visual_pe - 0.4) * 0.05) as f64;
            self.prediction_confidence = (self.prediction_confidence - nudge).clamp(0.0, 1.0);
        } else if visual_pe < 0.1 && self.prediction_confidence < 0.7 {
            // Low visual surprise → gentle confidence recovery
            self.prediction_confidence = (self.prediction_confidence + 0.002).min(1.0);
        }
    }

    /// Cross-couple language quality → prediction confidence & learning rate.
    ///
    /// Sustained high generation quality (quality_ema > 0.7 AND coherence > 0.5)
    /// → confidence boost. Low coherence (< 0.3) → LR dampening (language
    /// subsystem struggling, slow down to consolidate).
    ///
    /// Science: Clark (2013) — predictive processing confidence from generation success.
    #[cfg(feature = "ssm_language")]
    pub(crate) fn cross_couple_language_confidence(&mut self) {
        let quality = self.language_manager.quality_ema();
        let coherence = self.language_manager.coherence_ema();

        // High quality generation → confidence boost
        if quality > 0.7 && coherence > 0.5 {
            let boost = ((quality - 0.7) * 0.02) as f64;
            self.prediction_confidence = (self.prediction_confidence + boost).min(1.0);
        }

        // Low coherence → LR dampening (language subsystem struggling)
        if coherence < 0.3 {
            self.carryover.learning.subsystem_lr_factor *= 0.95;
        }
    }

    /// Cross-couple reasoning reliability → exploration modulation.
    ///
    /// Falling reasoning reliability (falling_streak > 3) → boost exploration
    /// via LR increase (reasoning failures should trigger novelty-seeking).
    /// High reliability (> 0.8) → dampen LR (exploit current reasoning strategy).
    ///
    /// Science: Carver & Scheier (1998) — discrepancy reducing vs. creating loops.
    #[cfg(feature = "reasoning_engine")]
    pub(crate) fn cross_couple_reasoning_exploration(&mut self) {
        let falling = self.reasoning_manager.falling_streak();
        let reliability = self.reasoning_manager.reliability_ema();

        // Falling reliability → exploration boost (break out of failing strategy)
        if falling > 3 {
            let boost = (falling as f64 - 3.0).min(5.0) * 0.01;
            self.carryover.learning.subsystem_lr_factor *= 1.0 + boost as f32;
        }

        // High reliability → LR dampening (conserve current strategy)
        if reliability > 0.8 {
            self.carryover.learning.subsystem_lr_factor *= 0.97;
        }
    }

    /// Access the resonant speech module (read-only).
    pub fn resonant_speech(&self) -> &crate::resonant_speech::ResonantSpeech {
        &self.resonant_speech
    }

    /// Access the resonant speech module (mutable).
    pub fn resonant_speech_mut(&mut self) -> &mut crate::resonant_speech::ResonantSpeech {
        &mut self.resonant_speech
    }

    /// Access the consciousness state manager.
    pub fn consciousness_state(
        &self,
    ) -> &super::super::consciousness_state_manager::ConsciousnessStateManager {
        &self.consciousness_state
    }

    /// Access the ethics and values manager.
    pub fn ethics_values(&self) -> &super::super::ethics_values_manager::EthicsAndValuesManager {
        &self.ethics_values
    }

    /// Access the streaming inference engine stats (if enabled).
    pub fn streaming_inference_stats(&self) -> Option<crate::inference::StreamingStats> {
        self.streaming_inference.as_ref().map(|si| si.stats())
    }

    /// Access the language and communication manager.
    pub fn language_comm(
        &self,
    ) -> &super::super::language_comm_manager::LanguageAndCommunicationManager {
        &self.language_comm
    }

    /// Access the vision and sensory manager.
    pub fn vision_sensory(&self) -> &super::super::vision_sensory_manager::VisionAndSensoryManager {
        &self.sensorimotor.vision_sensory
    }

    /// Get the current unified ethical verdict.
    ///
    /// Returns the override if set, otherwise the last verdict from the ethics engine.
    pub fn last_ethics_verdict(&self) -> &super::super::ethics_engine::EthicalVerdict {
        self.ethics_verdict_override
            .as_ref()
            .unwrap_or(&self.last_ethics_verdict)
    }

    /// Override the unified ethical verdict.
    ///
    /// When set, the override takes precedence over the ethics engine's output
    /// each cycle. The override persists until cleared via `clear_ethics_override()`.
    /// Used by external safety systems and integration tests to force a specific
    /// verdict that gates motor output and Broca generation.
    pub fn set_ethics_verdict(&mut self, verdict: super::super::ethics_engine::EthicalVerdict) {
        self.ethics_verdict_override = Some(verdict);
    }

    /// Clear the ethics verdict override, allowing the ethics engine to
    /// determine the verdict normally.
    pub fn clear_ethics_override(&mut self) {
        self.ethics_verdict_override = None;
    }

    /// Enable async voice synthesis: spawns a background thread that converts
    /// Broca/BrocaLite text output to audio without blocking the cognitive cycle.
    ///
    /// Call once after constructing the service. Audio is retrieved via
    /// `drain_voice_audio()` in subsequent cycles.
    pub fn enable_voice_synthesis(&mut self) {
        if self.voice_synthesis.is_none() {
            self.voice_synthesis =
                Some(super::super::voice_channel::VoiceSynthesisChannel::spawn());
        }
    }

    /// Drain completed voice audio from the background synthesis thread.
    ///
    /// Returns audio samples generated from previous cycles' text output.
    /// Non-blocking — returns empty vec if no audio is ready.
    pub fn drain_voice_audio(&self) -> Vec<super::super::voice_channel::VoiceResponse> {
        self.voice_synthesis
            .as_ref()
            .map(|vs| vs.drain_responses())
            .unwrap_or_default()
    }

    /// Enable async LLM language: spawns a background thread that sends
    /// consciousness state to Gemma 4 (via Ollama) for high-quality translation.
    ///
    /// When enabled, BrocaLite still fires instantly each cycle (structured grammar),
    /// but the LLM produces a higher-quality translation that replaces the output
    /// in subsequent cycles. Architecture:
    ///
    /// ```text
    /// Cycle N:   BrocaLite → immediate text + LLM request sent (async)
    /// Cycle N+K: LLM response → upgrades language_output
    /// ```
    ///
    /// Model defaults to `gemma4:e2b` via Ollama at localhost:11434.
    /// Override with `SYMTHAEA_LLM_MODEL` env var.
    pub fn enable_llm_language(&mut self) {
        if self.llm_language.is_none() {
            self.llm_language =
                Some(super::super::llm_language_channel::LlmLanguageChannel::spawn());
        }
    }

    /// Drain completed LLM language responses (non-blocking).
    pub fn drain_llm_language(
        &self,
    ) -> Vec<super::super::llm_language_channel::LlmLanguageResponse> {
        self.llm_language
            .as_ref()
            .map(|ch| ch.drain_responses())
            .unwrap_or_default()
    }
}
