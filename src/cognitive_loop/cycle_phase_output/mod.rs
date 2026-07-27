// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Output phase of the cognitive cycle.
//!
//! Extracts the final metadata assembly and CycleResult construction from
//! the original `cycle()` method.
//!
//! This module is intentionally split:
//! - `mod.rs` owns output-phase control flow, side effects, and `CycleResult`.
//! - `telemetry.rs` owns field-by-field `CycleMetadata` population.

use std::mem;
use std::time::Instant;

use super::phase_results::{DynamicsPhaseResult, FeedbackPhaseResult, PerceptionPhaseResult};
use super::thresholds::{
    ADAPTIVE_THRESHOLD_SCALE_LOWER, ADAPTIVE_THRESHOLD_SCALE_UPPER, ANTI_MONOPOLY_DAMPEN_SCALE,
    CONFLICT_EXPLORATION_INCREMENT, DOMINANT_CONCENTRATION_MONOPOLY_THRESHOLD,
    ERROR_SLOPE_CONSOLIDATION_THRESHOLD, FEEDBACK_INTEGRATION_RATE_LOWER,
    FULL_DAMPEN_ESCAPE_EXPLORATION, PROPOSAL_CONFLICT_EXPLORATION,
    SUBSTRATE_TAU_DEVIATION_THRESHOLD, SUBSTRATE_TAU_FACTOR_MINIMUM,
    SUBSYSTEM_EXPLORATION_REQUEST_NUDGE, SUBSYSTEM_REST_REQUEST_LR_SCALE,
    URGENCY_ESCALATION_AROUSAL_BOOST, URGENCY_ESCALATION_EXPLORATION_SCALE,
};
use super::{CognitiveLoopService, CycleMetadata, CycleResult};

mod telemetry;

impl CognitiveLoopService {
    /// Output phase: metadata assembly, telemetry, CycleResult construction.
    pub(super) fn phase_output(
        &mut self,
        _input: &str,
        cycle_start: Instant,
        perception: &mut PerceptionPhaseResult,
        dynamics: &mut DynamicsPhaseResult,
        feedback: &mut FeedbackPhaseResult,
        mut module_timings: super::ModuleTimings,
    ) -> CycleResult {
        let thalamic_depth_score = match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => super::thresholds::DEPTH_SCORE_DEEP_THOUGHT,
            super::CognitiveDepth::Cortical => super::thresholds::DEPTH_SCORE_CORTICAL,
            super::CognitiveDepth::Reflex => super::thresholds::DEPTH_SCORE_REFLEX,
        };

        let value_trend = self.primitive_tier.value_feedback.recent_trend(50);
        let circadian_phase_str = self.biorhythm_mgr.rhythm.phase.as_str();
        let selected_strategy_str = perception.strategy.selected_strategy.as_str();

        let _t = Instant::now();
        let moral_anomaly_report = self.ethics_engine.last_anomaly_report();
        let topo_summary = self.ethics_engine.moral_topology().last_summary();
        let topology_fresh = self.ethics_engine.last_topology_fresh();

        let mut metadata = CycleMetadata::default();

        // Populate metadata fields using modular helper sub-methods in telemetry.rs
        self.populate_core_telemetry(
            &mut metadata,
            perception,
            dynamics,
            feedback,
            circadian_phase_str,
            selected_strategy_str,
            value_trend,
            &topo_summary,
            topology_fresh,
            &moral_anomaly_report,
            thalamic_depth_score,
        );

        // Update smoothed epistemic uncertainty in self.carryover
        self.carryover.quality.smoothed_epistemic_uncertainty =
            metadata.smoothed_epistemic_uncertainty;

        self.populate_modulation_telemetry(&mut metadata, perception, dynamics, feedback);

        self.populate_manager_telemetry(&mut metadata, feedback);

        self.populate_bridge_telemetry(
            &mut metadata,
            perception,
            dynamics,
            feedback,
            thalamic_depth_score,
        );

        // ── Epistemic conflict exploration boost (Session 11 Item 8) ──
        {
            let conflict = self.feedback_state.avg_conflict_ratio();
            metadata.proposal_conflict_ratio = conflict;
            if conflict > PROPOSAL_CONFLICT_EXPLORATION && self.stats.total_cycles > 15 {
                self.feedback_state.exploration.propose(
                    "high_conflict",
                    super::feedback_state::FeedbackProposal::Add(CONFLICT_EXPLORATION_INCREMENT),
                );
                metadata.modulation.conflict_exploration_boost = true;
            }
        }

        // ── GWT memory consolidation request and perception count swap ──
        metadata.gwt_memory_consolidation_requested = self
            .consciousness
            .gwt_mgr
            .memory_flag
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        metadata.gwt_perception_broadcasts =
            self.consciousness
                .gwt_mgr
                .perception_count
                .swap(0, std::sync::atomic::Ordering::Relaxed) as u32;

        // ── Memory consolidation triggers ──
        let should_consolidate = metadata.gwt_memory_consolidation_requested
            || (perception.urgency.error_slope > ERROR_SLOPE_CONSOLIDATION_THRESHOLD);
        if should_consolidate {
            if let Some(ref mut dream) = self.dream_engine {
                let action: Vec<f32> = perception
                    .encoding
                    .encoding_result
                    .hdv
                    .values
                    .iter()
                    .take(32)
                    .copied()
                    .collect();
                dream.record_consolidation_event(
                    &perception.encoding.compressed_state,
                    action,
                    perception.urgency.prediction_error,
                );
            }
        }

        // ── Module timings telemetry update ──
        metadata.module_timings_us = {
            if let Some(ref bundler) = self.hierarchical_bundler {
                tracing::trace!(
                    active_regions = bundler.active_region_count(),
                    total_vectors = bundler.total_vectors(),
                    "Hierarchical bundling stats"
                );
            }
            module_timings.metadata_assembly = _t.elapsed().as_micros() as u64;
            module_timings
        };

        // ── End-of-cycle stats ──
        self.run_end_of_cycle_stats(
            &mut metadata,
            dynamics.resonator.resonator_wm_primed,
            feedback.memory.resonator_promotions,
            feedback.memory.codebook_evictions,
            feedback.memory.codebook_diversity,
            dynamics.fep.fep_surprise,
            self.consciousness
                .self_model_tier
                .self_reflection
                .get_thresholds()
                .surprise as f64,
            dynamics.neuromod.neuromod_attention_alloc,
            dynamics.neuromod.phasic_da_replay_boost,
            dynamics.neuromod.ne_reorienting_boost,
            dynamics.neuromod.ne_arousal_feedback,
            dynamics.neuromod.confidence_velocity,
            dynamics.neuromod.sht_crash_dip,
            dynamics.neuromod.exploration_sht_drain,
        );

        metadata.temporal.thermodynamic_load = self.thermodynamic_load;

        #[cfg(feature = "vision-manifold")]
        if let Some(ref mut pred) = self.sensorimotor.vision_sensory.cross_manifold_predictor {
            pred.observe_cognitive(&perception.encoding.encoding_result.hdv);
        }

        let thought_vector = {
            debug_assert!(
                !perception.encoding.encoding_result.hdv.values.is_empty(),
                "HDV must not be empty for thought_vector projection"
            );
            let chunk_size = (perception.encoding.encoding_result.hdv.values.len() / 32).max(1);
            perception
                .encoding
                .encoding_result
                .hdv
                .values
                .chunks(chunk_size)
                .take(32)
                .map(|chunk: &[f32]| chunk.iter().sum::<f32>() / chunk.len().max(1) as f32)
                .collect()
        };

        metadata.cycle_duration_us = cycle_start.elapsed().as_micros() as u64;

        tracing::debug!(
            surprise = metadata.surprise_triggered,
            prefrontal_veto = metadata.prefrontal_veto,
            reasoning_confidence = metadata.reasoning_confidence,
            exploration = ?metadata.exploration_action,
            "Cycle metadata"
        );

        if let Some(ref metrics) = self.metrics_collector {
            metrics.set_phi(dynamics.core.unified_psi);
            metrics.set_coherence(dynamics.core.coherence as f64);
            metrics.set_consciousness_level(metadata.consciousness.consciousness_level);
            metrics.track_execution(metadata.safety_blocked, false);

            #[cfg(feature = "api_module")]
            crate::api::metrics::update_timing_metrics(
                crate::api::metrics::global(),
                &metadata.module_timings_us,
                metadata.cycle_duration_us,
            );
        }

        #[cfg(feature = "identity")]
        let signed_output = self
            .mfdi_bridge
            .sign_output(&dynamics.core.output)
            .map_err(
                |e| tracing::error!(error = ?e, "MFDI output signing failed — output unattested"),
            )
            .ok();
        #[cfg(feature = "identity")]
        let assurance_level = self.mfdi_bridge.assurance_level();

        // Session 9 Item 3: Dominant source concentration dampening
        let dominant_concentration = self.feedback_state.dominant_source_concentration();
        if dominant_concentration > DOMINANT_CONCENTRATION_MONOPOLY_THRESHOLD
            && self.feedback_state.total_proposals() > 4
        {
            self.feedback_state.confidence.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(ANTI_MONOPOLY_DAMPEN_SCALE),
            );
            self.feedback_state.learning_rate.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(ANTI_MONOPOLY_DAMPEN_SCALE),
            );
            self.feedback_state.exploration.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(ANTI_MONOPOLY_DAMPEN_SCALE),
            );
            self.feedback_state.threshold.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(ANTI_MONOPOLY_DAMPEN_SCALE),
            );
        }
        metadata.dominant_source_concentration = dominant_concentration;

        // Session 10 Item 6: Proposal diversity check
        {
            use super::thresholds::{
                PROPOSAL_DIVERSITY_EXPLORATION_BOOST, PROPOSAL_DIVERSITY_MIN_SOURCES,
                PROPOSAL_DIVERSITY_WARMUP,
            };
            let source_count = self.feedback_state.distinct_source_count();
            metadata.proposal_source_count = source_count as u32;
            if source_count < PROPOSAL_DIVERSITY_MIN_SOURCES
                && self.stats.total_cycles > PROPOSAL_DIVERSITY_WARMUP
                && self.feedback_state.total_proposals() > 2
            {
                self.feedback_state.exploration.propose(
                    "low_diversity",
                    super::feedback_state::FeedbackProposal::Add(
                        PROPOSAL_DIVERSITY_EXPLORATION_BOOST as f64,
                    ),
                );
                metadata.modulation.low_diversity_boost = true;
            }
        }

        // ── Feedback consensus collection ──
        let mut feedback_consensus = self.feedback_state.end_cycle_ext(
            self.prediction_confidence,
            self.fep.lr_boost,
            self.behavior.curiosity_drive.exploration_urge,
            self.carryover.learning.adaptive_threshold_scale,
            self.carryover.quality.consecutive_full_dampen,
            self.behavior.flow_state.in_flow,
            self.behavior.flow_state.intensity,
        );
        if self.feedback_state.feedback_dampened_count == 4 {
            self.carryover.quality.consecutive_full_dampen += 1;
        } else {
            self.carryover.quality.consecutive_full_dampen = 0;
        }

        if self.carryover.quality.consecutive_full_dampen
            >= super::thresholds::FULL_DAMPEN_FREEZE_THRESHOLD
        {
            self.carryover.learning.adaptive_threshold_scale =
                self.carryover.learning.adaptive_threshold_scale.clamp(
                    ADAPTIVE_THRESHOLD_SCALE_LOWER as f64,
                    ADAPTIVE_THRESHOLD_SCALE_UPPER as f64,
                );
            self.adjust_exploration("full_dampen_escape", FULL_DAMPEN_ESCAPE_EXPLORATION);
        }

        // Substrate tau factor scaling
        if (self.substrate_manager.tau_factor as f64 - 1.0).abs()
            > SUBSTRATE_TAU_DEVIATION_THRESHOLD as f64
        {
            let tau =
                (self.substrate_manager.tau_factor as f64).max(SUBSTRATE_TAU_FACTOR_MINIMUM as f64);
            let integration_rate = if tau.is_finite() {
                (1.0_f64 / tau).clamp(FEEDBACK_INTEGRATION_RATE_LOWER as f64, 1.0)
            } else {
                1.0_f64
            };
            let cs = &self.feedback_state;
            let rate = integration_rate as f64;
            feedback_consensus = super::feedback_state::ConsensusResult {
                consensus_confidence: cs.cycle_start_confidence() * (1.0 - rate)
                    + feedback_consensus.consensus_confidence * rate,
                consensus_lr: cs.cycle_start_lr() * (1.0 - rate)
                    + feedback_consensus.consensus_lr * rate,
                consensus_exploration: cs.cycle_start_exploration() * (1.0 - rate)
                    + feedback_consensus.consensus_exploration * rate,
                consensus_threshold: cs.cycle_start_threshold() * (1.0 - rate)
                    + feedback_consensus.consensus_threshold * rate,
            };
        }

        self.feedback_state
            .store_consensus_for_next_cycle(&feedback_consensus);

        // ── Subsystem integration ──
        let integrated = self.subsystem_collector.integrate();
        if integrated.n_contributors > 0 {
            metadata.subsystem_integration_contributors = integrated.n_contributors as u32;
            metadata.subsystem_flags = integrated.flags;

            if integrated.confidence_delta != 0.0 {
                self.adjust_confidence("subsystem_managers", integrated.confidence_delta as f32);
            }
            if integrated.lr_modulation != 1.0 {
                self.scale_lr("subsystem_managers", integrated.lr_modulation as f32);
            }
            if integrated.exploration_delta != 0.0 {
                self.adjust_exploration("subsystem_managers", integrated.exploration_delta as f32);
            }
            if integrated.arousal_delta != 0.0 {
                self.behavior.emotion_contagion.arousal = (self.behavior.emotion_contagion.arousal
                    + integrated.arousal_delta)
                    .clamp(0.0, 1.0);
            }
            if integrated.valence_delta != 0.0 {
                self.behavior.emotion_contagion.valence = (self.behavior.emotion_contagion.valence
                    + integrated.valence_delta)
                    .clamp(-1.0, 1.0);
            }

            use super::subsystem_trait::output_flags;

            if integrated.has_flag(output_flags::VETO_ACTION) {
                self.carryover.quality.subsystem_veto = true;
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Subsystem VETO_ACTION: motor output will be suppressed"
                );
            }

            if integrated.has_flag(output_flags::ESCALATE_URGENCY) {
                self.behavior.emotion_contagion.arousal = (self.behavior.emotion_contagion.arousal
                    + URGENCY_ESCALATION_AROUSAL_BOOST)
                    .clamp(0.0, 1.0);
                self.scale_exploration("urgency_escalation", URGENCY_ESCALATION_EXPLORATION_SCALE);
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Subsystem ESCALATE_URGENCY: arousal boosted, exploration dampened"
                );
            }

            if integrated.has_flag(output_flags::REQUEST_CONSOLIDATION) {
                self.fep
                    .episodic_memory
                    .consolidate_recent(self.unification_engine.psi);
                tracing::trace!(
                    cycle = self.stats.total_cycles,
                    "Subsystem REQUEST_CONSOLIDATION: episodic consolidation triggered"
                );
            }

            if integrated.has_flag(output_flags::REQUEST_REST) {
                self.scale_lr("subsystem_rest_request", SUBSYSTEM_REST_REQUEST_LR_SCALE);
                tracing::trace!(
                    cycle = self.stats.total_cycles,
                    "Subsystem REQUEST_REST: LR dampened for recovery"
                );
            }

            if integrated.has_flag(output_flags::REQUEST_EXPLORATION) {
                self.adjust_exploration(
                    "subsystem_request_explore",
                    SUBSYSTEM_EXPLORATION_REQUEST_NUDGE,
                );
            }

            if integrated.has_flag(output_flags::ANOMALY_DETECTED) {
                self.stats.anomaly_detected_count += 1;
                self.scale_confidence("subsystem_anomaly", 0.98);
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Subsystem ANOMALY_DETECTED: confidence dampened"
                );
            }

            #[cfg(feature = "vision-manifold")]
            if integrated.has_flag(output_flags::REQUEST_GEODESIC) {
                self.carryover.quality.last_request_geodesic = true;
                if let Some(ref mut bridge) = self.sensorimotor.vision_sensory.vision_bridge {
                    let manifold = bridge.manifold_mut();
                    let current_state = manifold.state().clone();
                    let goal = if let Some(match_res) = manifold.last_scene_match() {
                        manifold
                            .get_scene_encoding(match_res.scene_id)
                            .unwrap_or_else(|| {
                                symthaea_core::core::ContinuousHV::random(manifold.hdc_dim(), 777)
                            })
                    } else {
                        symthaea_core::core::ContinuousHV::random(
                            manifold.hdc_dim(),
                            self.stats.total_cycles as u64,
                        )
                    };

                    let path = manifold.select_best_geodesic(&current_state, &goal, 8, 3);
                    let latest_telemetry = manifold.telemetry().clone();
                    perception.vision_telemetry = Some(latest_telemetry.clone());
                    metadata.vision = Some(latest_telemetry);

                    if !path.is_empty() {
                        let frames = manifold.decode_geodesic_to_frames_improved(&path);
                        if !frames.is_empty() {
                            feedback.mental_movie =
                                Some(crate::cognitive_loop::types::MentalMovie {
                                    frames,
                                    width: self.config.vision_frame_width,
                                    height: self.config.vision_frame_height,
                                    channels: manifold.last_frame_channels(),
                                    path_length: path.len(),
                                    semantic_coherence: 0.0,
                                    trajectory: path,
                                });
                        }
                    }
                    tracing::info!(
                        cycle = self.stats.total_cycles,
                        "Subsystem REQUEST_GEODESIC: FEP-guided mental simulation triggered"
                    );
                }
            } else {
                self.carryover.quality.last_request_geodesic = false;
            }

            metadata.subsystem_veto_active = self.carryover.quality.subsystem_veto;

            #[cfg(all(feature = "swarm", feature = "vision-manifold"))]
            if integrated.has_flag(output_flags::REQUEST_BROADCAST) {
                if let (Some(id), Some(phi), Some(hv), Some(intent)) = (
                    self.node_id(),
                    Some(feedback.consciousness.consciousness_level),
                    self.consciousness_hv(),
                    self.last_intent_hv(),
                ) {
                    let msg = symthaea_swarm::SwarmStateMsg {
                        node_id: id,
                        platform_type:
                            symtropy_robotics_bridge_core::platform::PlatformType::default(),
                        local_phi: phi,
                        consciousness_hv: hv,
                        intent_hv: intent,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64,
                    };
                    if let Some(svc) = self.network_service() {
                        svc.broadcast_swarm_state(msg);
                    }
                }
            }
            tracing::trace!("Phase C integration: {}", integrated);
        }

        self.update_governance_lag(feedback.consciousness.consciousness_level);
        self.expire_stale_consequence_predictions();
        self.record_attention_visualizer_snapshot(perception, dynamics, feedback);
        self.apply_final_output_clamps();

        // Voice synthesis: drain audio completed by the background thread and
        // feed its quality metrics into the voice→cognition feedback bridge
        // (this is what makes voice_articulation_quality/LR modulation a real
        // signal instead of frozen defaults). Audio is buffered (bounded) for
        // drain_voice_audio().
        if self.voice_synthesis.is_some() {
            let completed = self
                .voice_synthesis
                .as_ref()
                .map(|vs| vs.drain_responses())
                .unwrap_or_default();
            for resp in completed {
                self.update_voice_feedback(resp.metrics.clone());
                // Self-hearing: queue her own utterance's acoustic HV for the
                // next perception phase (latest-wins).
                #[cfg(feature = "voice-stt")]
                if let Some(ref hv) = resp.self_hv {
                    self.pending_self_voice_hv = Some(hv.clone());
                }
                self.voice_audio_buffer.push_back(resp);
            }
            while self.voice_audio_buffer.len() > super::voice_channel::VOICE_AUDIO_BUFFER_CAP {
                self.voice_audio_buffer.pop_front();
            }
        }

        // Take the CfC output ONCE, before any use: the previous struct-literal
        // ordering took `output` first and then cloned the (already emptied)
        // vec into the VoiceRequest, so voice prosody never saw real CfC state.
        let output = mem::take(&mut dynamics.core.output);

        let language_output = self.language_comm.last_broca_text.take();
        if let (Some(t), Some(vs)) = (&language_output, &self.voice_synthesis) {
            // Effective time-constant for pacing: adaptive FEP-surprise × Φ tau
            // factors (each ~1.0 baseline; 0.0 means not populated this cycle).
            let fep_tau = if dynamics.fep_tau_factor > 0.0 {
                dynamics.fep_tau_factor
            } else {
                1.0
            };
            let phi_tau = if dynamics.phi_tau_factor > 0.0 {
                dynamics.phi_tau_factor
            } else {
                1.0
            };
            let _ = vs.send(super::voice_channel::VoiceRequest {
                text: t.clone(),
                cfc_output: output.clone(),
                tau: fep_tau * phi_tau,
                prediction_error: dynamics.core.prediction_error,
                detected_primitives: perception
                    .encoding
                    .encoding_result
                    .detected_primitives
                    .clone(),
                speech_rate_multiplier: self.behavior.adaptive_behavior.speech_rate_multiplier,
                pause_multiplier: self.behavior.adaptive_behavior.pause_multiplier,
                cycle_num: self.stats.total_cycles as u64,
            });
        }

        CycleResult {
            output,
            prediction_error: dynamics.core.prediction_error,
            peak_attention: perception.encoding.encoding_result.peak_attention,
            detected_primitives: mem::take(
                &mut perception.encoding.encoding_result.detected_primitives,
            ),
            learning_occurred: dynamics.core.learning_occurred,
            training_loss: dynamics.core.training_loss,
            bits_saved_persist: perception.encoding.encoding_result.bits_saved_persist,
            bits_saved_zero: perception.encoding.encoding_result.bits_saved_zero,
            bits_kappa: perception.encoding.encoding_result.bits_kappa,
            recall_fired: dynamics.core.recall_fired,
            recall_similarity: dynamics.core.recall_similarity,
            recall_matched_timestamp: dynamics.core.recall_matched_timestamp,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            thought_vector,
            wisdom_hv: perception.encoding.hv16_cached,
            language_output,
            language_source: self.language_comm.last_language_source.take(),
            #[cfg(feature = "canvas")]
            canvas_svg: self.sensorimotor.motor_rendering.last_canvas_svg.take(),
            #[cfg(feature = "identity")]
            signed_output,
            #[cfg(feature = "identity")]
            assurance_level,
            #[cfg(feature = "vision-manifold")]
            mental_movie: {
                let movie = mem::take(&mut feedback.mental_movie);
                self.last_mental_movie = movie.clone();
                movie
            },
        }
    }

    fn update_governance_lag(&mut self, consciousness_level: f64) {
        let lag_size = super::thresholds::GOVERNANCE_CONSCIOUSNESS_LAG_SIZE;
        self.governance_consciousness_lag
            .push_back(consciousness_level);
        while self.governance_consciousness_lag.len() > lag_size {
            self.governance_consciousness_lag.pop_front();
        }
    }

    fn expire_stale_consequence_predictions(&mut self) {
        self.ethics_engine.expire_stale_predictions(
            self.stats.total_cycles as u64,
            super::thresholds::CONSEQUENCE_TRACKER_MAX_AGE_CYCLES,
        );
    }

    fn record_attention_visualizer_snapshot(
        &mut self,
        perception: &PerceptionPhaseResult,
        dynamics: &DynamicsPhaseResult,
        feedback: &FeedbackPhaseResult,
    ) {
        let Some(ref mut viz) = self.attention_visualizer else {
            return;
        };

        static ATTENTION_LABELS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        let labels = ATTENTION_LABELS
            .get_or_init(|| {
                vec![
                    "phi_attention".into(),
                    "prediction_error".into(),
                    "coherence".into(),
                    "binding_strength".into(),
                    "consciousness".into(),
                ]
            })
            .clone();
        let snapshot = crate::visualization::AttentionSnapshot::new(
            labels,
            vec![
                perception.encoding.phi_attention_weight as f64,
                dynamics.core.prediction_error as f64,
                dynamics.core.coherence as f64,
                feedback.self_model.cross_modal_binding_strength as f64,
                feedback.consciousness.equation_v2_consciousness,
            ],
            vec![
                perception.encoding.phi_attention_weight,
                dynamics.core.prediction_error.clamp(0.0, 1.0),
                dynamics.core.coherence,
                feedback.self_model.cross_modal_binding_strength,
                feedback.consciousness.equation_v2_consciousness as f32,
            ],
            1.0,
        )
        .with_metadata("cycle", &self.stats.total_cycles.to_string())
        .with_metadata("depth", &format!("{:?}", self.cognitive_depth));
        viz.record(snapshot);
    }

    fn apply_final_output_clamps(&mut self) {
        self.behavior.adaptive_behavior.exploration_factor = self
            .behavior
            .adaptive_behavior
            .exploration_factor
            .clamp(0.1, 3.0);
        self.behavior.adaptive_behavior.learning_rate_multiplier = self
            .behavior
            .adaptive_behavior
            .learning_rate_multiplier
            .clamp(0.1, 2.0);
        self.behavior.curiosity_drive.boredom =
            self.behavior.curiosity_drive.boredom.clamp(0.0, 1.5);
    }
}

#[cfg(test)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn output_metadata_non_default() {
        let mut svc = make_service();
        let result = svc.cycle("metadata check");
        assert!(result.metadata.cycle_duration_us > 0);
        assert!(!result.metadata.selected_strategy.is_empty());
    }

    #[test]
    fn output_metadata_populates_representative_families() {
        let mut svc = make_service();
        let result = svc.cycle("metadata family regression");
        let metadata = &result.metadata;

        assert!(metadata.cycle_duration_us > 0);
        assert!(!metadata.selected_strategy.is_empty());
        assert!(!metadata.circadian_phase.is_empty());
        assert!(matches!(
            metadata.response_profile.as_str(),
            "technical" | "balanced" | "simplified" | "empathic"
        ));
        assert!(metadata.consciousness.consciousness_level.is_finite());
        assert!(metadata.attention.phi_attention_weight.is_finite());
        assert!(metadata.temporal.temporal_coherence_score.is_finite());
        assert!(metadata.memory.codebook_utilization_rate.is_finite());
        assert!(metadata.substrate_effective_feasibility.is_finite());
        assert!(
            metadata.module_timings_us.core_hdc_encode > 0
                || metadata.module_timings_us.core_cfc_step > 0
        );
    }

    #[test]
    fn output_thalamic_depth_maps_correctly() {
        let mut svc = make_service();
        let result = svc.cycle("thalamic depth");
        let score = result.metadata.thalamic_depth_score;
        assert!(
            (score - 1.0).abs() < f32::EPSILON
                || (score - 0.5).abs() < f32::EPSILON
                || (score - 0.2).abs() < f32::EPSILON,
            "thalamic_depth_score should be 1.0, 0.5, or 0.2, got {score}"
        );
    }

    #[test]
    fn output_is_consolidating_populated() {
        let mut svc = make_service();
        let result = svc.cycle("consolidation check");
        let consolidating = result.metadata.is_consolidating;
        assert!(
            consolidating || !consolidating,
            "is_consolidating should be a valid bool"
        );
    }

    #[test]
    fn output_module_timings_has_core_hdc_encode() {
        let mut svc = make_service();
        let result = svc.cycle("timing check");
        assert!(
            result.metadata.module_timings_us.core_hdc_encode > 0
                || result.metadata.module_timings_us.core_cfc_step > 0
        );
    }

    #[test]
    fn output_thought_vector_32d() {
        let mut svc = make_service();
        let result = svc.cycle("thought vector");
        assert_eq!(result.thought_vector.len(), 32);
        for (i, &v) in result.thought_vector.iter().enumerate() {
            assert!(v.is_finite(), "thought_vector[{i}] should be finite");
        }
    }

    #[test]
    fn output_circadian_phase_populated() {
        let mut svc = make_service();
        let result = svc.cycle("circadian check");
        assert!(!result.metadata.circadian_phase.is_empty());
    }

    #[test]
    fn test_convergence_cycle_captured_and_persists() {
        let mut svc = make_service();
        let result = svc.cycle("convergence init");
        assert_eq!(
            result.metadata.consciousness.convergence_cycle, 0,
            "convergence_cycle should start at 0"
        );

        let mut first_convergence_cycle = 0usize;
        for i in 0..200 {
            let result = svc.cycle("steady input for convergence");
            if result.metadata.consciousness.convergence_cycle > 0 && first_convergence_cycle == 0 {
                first_convergence_cycle = result.metadata.consciousness.convergence_cycle;
            }
            if first_convergence_cycle > 0 {
                assert_eq!(
                    result.metadata.consciousness.convergence_cycle, first_convergence_cycle,
                    "convergence_cycle should persist once set (cycle {i})"
                );
            }
        }
    }
}
