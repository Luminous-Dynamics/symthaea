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

        /// Get current exploration urge (0.0 to 1.0)
        pub fn curiosity_drive_exploration_urge(&self) -> f64 { self.curiosity_drive.exploration_urge }

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
        pub fn voice_feedback_summary(&self) -> VoiceQualitySummary { self.language_comm.voice_coherence.voice.summary() }

        /// Check if voice indicates uncertainty
        pub fn voice_indicates_uncertainty(&self) -> bool { self.language_comm.voice_coherence.voice.is_uncertain() }

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
        self.social_mgr.social.external_reward = reward.clamp(-1.0, 1.0);
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
        self.social_mgr.social.social_trust = trust.clamp(0.0, 1.0);
        self.social_mgr.social.social_cooperation_rate = cooperation_rate.clamp(0.0, 1.0);
        self.social_mgr.social.social_prediction_accuracy = prediction_accuracy.clamp(0.0, 1.0);
        self.social_mgr.social.social_models_count = models_count;
        self.social_mgr.social.social_mean_trust = mean_trust.clamp(0.0, 1.0);
    }

    /// Set the relational Psi from an external dyad computation.
    /// This is called by the Symthaea facade after computing Phi_dyad.
    pub fn set_relational_psi(&mut self, psi: f64) {
        self.social_mgr.social.relational_psi = psi;
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
            self.governance_mgr.inject_event(event);
        }
        for outcome in outcomes {
            self.governance_mgr.inject_outcome(outcome);
        }
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
            metadata.consciousness_level,
            lr,                                 // meta-awareness proxy
            metadata.consciousness_level * 0.8, // coherence proxy
            metadata.consciousness_level * 0.6, // care activation proxy
            metadata.consciousness_level * 0.9, // quality proxy
            lr.min(1.0),                        // epistemic confidence proxy
        )
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

    /// Process governance learning signals: reward, harmonic deltas, episodic memory.
    ///
    /// Called after governance processing. Feeds reward to the FEP learning loop,
    /// applies harmonic deltas to the experience bus's KosmicSong, and drains
    /// completed outcomes for episodic recording.
    #[cfg(feature = "mycelix")]
    pub(crate) fn process_governance_learning(&mut self) {
        // 1. Feed governance reward to provide_reward()
        if let Some(reward) = self.governance_mgr.take_latest_reward() {
            self.social_mgr.social.external_reward = reward.clamp(-1.0, 1.0);
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
                    self.neuromod
                        .bath
                        .dopamine
                        .adjust_baseline(bl.nudge, 0.2, 0.8);
                }
                "noradrenaline" => {
                    self.neuromod
                        .bath
                        .noradrenaline
                        .adjust_baseline(bl.nudge, 0.2, 0.8);
                }
                "serotonin" => {
                    self.neuromod
                        .bath
                        .serotonin
                        .adjust_baseline(bl.nudge, 0.2, 0.8);
                }
                "oxytocin" => {
                    self.neuromod
                        .bath
                        .oxytocin
                        .adjust_baseline(bl.nudge, 0.2, 0.8);
                }
                "endocannabinoid" => {
                    self.neuromod
                        .bath
                        .endocannabinoid
                        .adjust_baseline(bl.nudge, 0.2, 0.8);
                }
                "acetylcholine" => {
                    self.neuromod
                        .bath
                        .acetylcholine
                        .adjust_baseline(bl.nudge, 0.2, 0.8);
                }
                "gaba" => {
                    self.neuromod.bath.gaba.adjust_baseline(bl.nudge, 0.2, 0.8);
                }
                "glutamate" => {
                    self.neuromod
                        .bath
                        .glutamate
                        .adjust_baseline(bl.nudge, 0.2, 0.8);
                }
                "adenosine" => {
                    self.neuromod
                        .bath
                        .adenosine
                        .adjust_baseline(bl.nudge, 0.2, 0.8);
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

        self.social_mgr.social.social_trust = (community as f32).clamp(0.0, 1.0);
        self.social_mgr.social.social_cooperation_rate = (reputation as f32).clamp(0.0, 1.0);
        self.social_mgr.social.social_mean_trust = (combined as f32).clamp(0.0, 1.0);
        self.social_mgr.social.social_prediction_accuracy = (identity as f32).clamp(0.0, 1.0);

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

    /// Apply swarm-driven neuromodulatory effects.
    ///
    /// Social bonding → oxytocin (Zak 2012), network anomaly → NE (Arnsten 2009),
    /// high collective Φ → serotonin (social satisfaction, Crockett 2009).
    pub(crate) fn apply_swarm_neuromod(&mut self) {
        let telem = self.swarm_manager.telemetry().clone();

        // Social bonding: connected peers → oxytocin (Zak 2012)
        // Sqrt for diminishing returns, capped at 0.08
        if telem.connected_peers > 0 {
            let oxy_dose = ((telem.connected_peers as f32).sqrt() * 0.02).min(0.08);
            if oxy_dose > 0.01 {
                self.neuromod.bath.inject("oxytocin", oxy_dose, 60);
            }
        }

        // Network anomaly → NE spike (Arnsten 2009)
        if telem.anomaly_count > 0 {
            let ne_nudge = (0.03 * telem.anomaly_count.min(3) as f32).min(0.09);
            self.neuromod
                .bath
                .noradrenaline
                .adjust_baseline(ne_nudge, 0.2, 0.8);
        }

        // High collective Φ → serotonin boost (social satisfaction)
        if telem.mean_peer_phi > 0.5 {
            let sht_nudge = ((telem.mean_peer_phi - 0.5) * 0.04).min(0.03) as f32;
            if sht_nudge > 0.01 {
                self.neuromod
                    .bath
                    .serotonin
                    .adjust_baseline(sht_nudge, 0.2, 0.8);
            }
        }

        // Affective contagion → DA modulation (shared positive affect)
        if telem.affective_contagion > 0.05 {
            let da_nudge = (telem.affective_contagion * 0.03).min(0.04) as f32;
            self.neuromod
                .bath
                .dopamine
                .adjust_baseline(da_nudge, 0.2, 0.8);
        }
    }

    /// Cross-couple SwarmManager and GovernanceManager signals.
    ///
    /// - Peer consciousness (mean_peer_phi) modulates governance confidence
    /// - Governance reward signal modulates swarm connectivity expectations
    ///
    /// Science: Woolley et al. (2010) collective intelligence factor
    #[cfg(feature = "mycelix")]
    pub(crate) fn cross_couple_swarm_governance(&mut self) {
        let swarm_telem = self.swarm_manager.telemetry().clone();

        // 1. High peer phi → governance confidence boost
        // When the swarm is collectively conscious, governance decisions
        // get a confidence uplift (Woolley 2010 collective intelligence)
        if swarm_telem.mean_peer_phi > 0.3 && swarm_telem.connected_peers > 1 {
            let phi_boost = ((swarm_telem.mean_peer_phi - 0.3) * 0.05).min(0.03);
            self.governance_mgr.nudge_confidence(phi_boost);
        }

        // 2. Network anomalies → reduce governance confidence
        // Unreliable network → less trust in collective decisions
        if swarm_telem.anomaly_count > 2 {
            self.governance_mgr.nudge_confidence(-0.02);
        }

        // 3. Governance reward → swarm expected peers adjustment
        // Positive governance outcomes → attract more peers (growth signal)
        let reward = self.governance_mgr.reward_ema();
        if reward > 0.05 {
            let current = self.swarm_manager.expected_peers();
            let growth = ((reward * 5.0) as usize).min(10);
            self.swarm_manager
                .set_expected_peers((current + growth).min(256));
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
        &self.vision_sensory
    }
}
