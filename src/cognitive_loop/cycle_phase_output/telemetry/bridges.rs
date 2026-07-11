// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use super::prelude::*;

impl CognitiveLoopService {
    pub(in crate::cognitive_loop::cycle_phase_output) fn populate_bridge_telemetry(
        &mut self,
        metadata: &mut CycleMetadata,
        perception: &mut PerceptionPhaseResult,
        dynamics: &mut DynamicsPhaseResult,
        feedback: &mut FeedbackPhaseResult,
        thalamic_depth_score: f32,
    ) {
        if self.stats.total_cycles % 47 == 0 && self.stats.total_cycles > 0 {
            if let Some(ref explainer) = self.primitive_tier.causal_explainer {
                let summary = explainer.summarize_understanding();
                if summary.total_causal_relations > 0 {
                    metadata.consciousness_causal_narrative = format!(
                        "{} causal relations ({} high-confidence), avg confidence {:.0}%, {} explanations generated",
                        summary.total_causal_relations,
                        summary.high_confidence_relations,
                        summary.average_confidence * 100.0,
                        summary.explanations_generated,
                    );
                }
            }
        }

        #[cfg(feature = "muse")]
        {
            metadata.muse = self.muse_manager.telemetry();
        }

        metadata.substrate = self.substrate_manager.telemetry(&self.config);
        metadata.substrate_transition = mem::take(&mut metadata.substrate.substrate_transition);
        metadata.substrate_feasibility_raw = metadata.substrate.substrate_feasibility_raw;
        metadata.substrate_honest_confidence = metadata.substrate.substrate_honest_confidence;
        metadata.substrate_effective_feasibility =
            metadata.substrate.substrate_effective_feasibility;
        metadata.substrate_tau_factor = metadata.substrate.substrate_tau_factor;
        metadata.substrate_scale_pressure = metadata.substrate.substrate_scale_pressure;

        #[cfg(feature = "jepa")]
        if let Some(ref jepa) = self.jepa_engine {
            let telem = jepa.telemetry();
            metadata.jepa_latent_pe = telem.latent_pe;
            metadata.jepa_total_energy = telem.total_energy;
            metadata.jepa_collapse_detected = telem.collapse_detected;
        }

        #[cfg(feature = "neural_validation")]
        {
            use symthaea_core::hdc::cortical_activation::{
                ActivationSource, CorticalActivationMap,
            };
            use symthaea_core::hdc::substrate_independence::CorticalRegion;

            let mut cam =
                CorticalActivationMap::zeros(ActivationSource::Simulated, self.stats.total_cycles);
            cam.set(
                CorticalRegion::Prefrontal,
                dynamics.reasoning.reasoning_confidence.clamp(0.0, 1.0),
            );
            cam.set(
                CorticalRegion::Visual,
                (metadata.grid_encoding_norm * 0.5 + metadata.grid_spatial_complexity * 0.5)
                    .clamp(0.0, 1.0),
            );
            cam.set(CorticalRegion::Auditory, metadata.voice_confidence);
            let lang_active = if metadata.reasoning_narrative.is_empty() {
                0.1
            } else {
                0.5
            };
            cam.set(CorticalRegion::Language, lang_active);
            cam.set(
                CorticalRegion::Memory,
                metadata.memory.codebook_utilization_rate,
            );
            let emotional = (feedback.consciousness.affect_cons_valence.abs()
                + feedback.consciousness.affect_cons_arousal)
                / 2.0;
            cam.set(CorticalRegion::Emotional, emotional.clamp(0.0, 1.0) as f32);
            cam.set(
                CorticalRegion::Motor,
                (dynamics.fep.fep_pragmatic_value as f32).clamp(0.0, 1.0),
            );
            cam.set(
                CorticalRegion::Social,
                (metadata.social_trust_current * 0.5 + metadata.social_prediction_accuracy * 0.5)
                    .clamp(0.0, 1.0),
            );
            cam.set(
                CorticalRegion::Executive,
                feedback.reasoning.epistemic_gate_confidence,
            );
            cam.set(
                CorticalRegion::Integration,
                (metadata.temporal_binding_strength * 0.5 + metadata.cross_module_agreement * 0.5)
                    .clamp(0.0, 1.0),
            );
            cam.set(
                CorticalRegion::Sensory,
                (thalamic_depth_score * 0.3).clamp(0.0, 1.0),
            );
            // Creative region: prefer a real signal from the creative/canvas
            // machinery when compiled in (see creative_region_activation).
            let creative = self.creative_region_activation().unwrap_or_else(|| {
                // Placeholder heuristic — NOT a real computation. Without the
                // `creative`/`canvas` features there is no creative machinery
                // to observe, so surprise-triggered exploration serves as a
                // coarse proxy for creative-cortex engagement.
                if perception.exploration.surprise_triggered {
                    0.7
                } else {
                    0.2
                }
            });
            cam.set(CorticalRegion::Creative, creative);

            if self.cortical_history.len() >= 1000 {
                self.cortical_history.pop_front();
            }
            self.cortical_history.push_back(cam.clone());
            metadata.cortical_activation = Some(cam);
        }

        {
            let thermal_signals = self.sensorimotor.thermal_bridge.signals();
            metadata.thermal = super::super::super::ThermalTelemetry {
                thermal_level: thermal_signals.level as u8,
                thermal_tau_factor: thermal_signals.tau_factor,
                should_reduce_profile: thermal_signals.should_reduce_profile,
                target_frequency_override: thermal_signals.target_frequency_override,
            };
        }

        #[cfg(feature = "integrity")]
        {
            let status = &self.integrity_manager.status;
            metadata.integrity = super::super::super::IntegrityTelemetry {
                attestation_passed: status.attestation_passed,
                temporal_passed: status.temporal_passed,
                canaries_passed: status.canaries_passed,
                anomaly_count: status.anomalies.len(),
                has_critical: self.integrity_manager.has_critical_anomaly(),
                last_check_cycle: status.last_check_cycle,
                integrity_confidence: self.integrity_manager.integrity_confidence,
                attestation_details: self
                    .integrity_manager
                    .attestation
                    .records()
                    .iter()
                    .map(|r| super::super::super::AttestationDetail {
                        name: r.name.to_string(),
                        passed: r
                            .last_verification
                            .as_ref()
                            .map(|v| v.passed)
                            .unwrap_or(true),
                        consecutive_failures: r.consecutive_failures,
                    })
                    .collect(),
                global_failure_streak: self.integrity_manager.global_failure_streak,
                confidence_history: self
                    .integrity_manager
                    .confidence_history()
                    .iter()
                    .copied()
                    .collect(),
            };
            let ic = self.integrity_manager.integrity_confidence;
            if ic < 1.0 {
                metadata.consciousness.consciousness_level *= ic as f64;
            }
        }

        #[cfg(feature = "physics-bridge")]
        {
            if let Some(ref mut physics) = self.feature_integ.physics_integration {
                let pt = physics.telemetry();
                let pareto = pt.pareto_context.as_ref();
                metadata.physics_bridge = Some(super::super::super::PhysicsBridgeTelemetry {
                    catalog_size: pt.catalog_size,
                    results_returned: pt.results_returned,
                    top_match: pt.top_match,
                    top_score: pt.top_score,
                    query_count: pt.query_count,
                    queried_this_cycle: pt.queried_this_cycle,
                    effective_interval: pt.effective_interval,
                    effective_blend_weight: pt.effective_blend_weight,
                    top_domain: pt.top_domain,
                    pareto_frontier_size: pareto.map(|p| p.frontier_size),
                    pareto_best_analogy: pareto.map(|p| p.best_analogy_score),
                });
            }
        }

        #[cfg(feature = "vision-manifold")]
        if let Some(ref tel) = perception.vision_telemetry {
            metadata.vision = Some(tel.clone());
        }

        #[cfg(feature = "foveation")]
        {
            if let Some(ref fov_mutex) = self.sensorimotor.vision_sensory.foveation_manager {
                if let Ok(fov) = fov_mutex.lock() {
                    let ft = fov.telemetry();
                    metadata.foveation = Some(super::super::super::FoveationBridgeTelemetry {
                        pending_count: ft.pending_count,
                        in_flight_count: ft.in_flight_count,
                        ready_count: ft.ready_count,
                        total_dispatched: ft.total_dispatched,
                        total_completed: ft.total_completed,
                        avg_processing_time_us: ft.avg_processing_time_us as u64,
                        last_confidence: ft.last_confidence,
                        effective_surprise_threshold: fov.effective_surprise_threshold(),
                        effective_max_concurrent: fov.effective_max_concurrent(),
                        recognition_count: perception.foveation_recognition_count,
                        top_recognition_confidence: perception.foveation_top_confidence,
                        hv_binding_applied: perception.foveation_recognition_count > 0,
                        dynamics_coupling_triggered: perception.foveation_recognition_count >= 2
                            && perception.foveation_top_confidence > 0.6,
                    });
                }
            }
        }

        #[cfg(feature = "ssm_language")]
        {
            metadata.broca = self
                .language_comm
                .broca_manager
                .as_ref()
                .map(|m| m.last_telemetry().clone());
        }

        #[cfg(feature = "mycelix")]
        {
            metadata.factcheck = Some(self.factcheck_bridge.telemetry());
        }

        metadata.consciousness.weight_convergence_state =
            mem::take(&mut feedback.consciousness.convergence_state);
        if metadata.consciousness.weight_convergence_state == "Converged"
            && self.convergence_cycle == 0
        {
            self.convergence_cycle = self.stats.total_cycles;
        }
        metadata.consciousness.convergence_cycle = self.convergence_cycle;

        let topo_summary = self.ethics_engine.moral_topology().last_summary();
        if topo_summary.beta_0 > 1 {
            tracing::warn!(
                target: "cognitive_loop::moral_topology",
                beta_0 = topo_summary.beta_0,
                unity = %format!("{:.3}", topo_summary.unity),
                scenario_count = topo_summary.scenario_count,
                cycle = self.stats.total_cycles,
                "Moral fragmentation: {} disjoint clusters",
                topo_summary.beta_0
            );
        }

        #[cfg(feature = "therapeutic")]
        {
            metadata.therapeutic.therapeutic_client_distress =
                self.therapeutic_manager.client_distress();
            metadata.therapeutic.therapeutic_alliance =
                self.therapeutic_manager.alliance_composite();
            metadata.therapeutic.therapeutic_crisis_active = self.therapeutic_manager.crisis_active;
            metadata.therapeutic.therapeutic_crisis_type = self
                .therapeutic_manager
                .last_crisis_type
                .clone()
                .unwrap_or_default();
            metadata.therapeutic.therapeutic_strategy = self
                .therapeutic_manager
                .active_strategy()
                .map(|s| s.as_str().to_string())
                .unwrap_or_default();
            metadata.therapeutic.therapeutic_narrative_coherence =
                self.therapeutic_manager.narrative_coherence();
            metadata.therapeutic.therapeutic_formulation_factors =
                self.therapeutic_manager.formulation.total_factors();
            metadata.therapeutic.therapeutic_resilience_ratio =
                self.therapeutic_manager.formulation_resilience_ratio();
            metadata.therapeutic.therapeutic_rupture_count =
                self.therapeutic_manager.alliance.rupture_count;
            metadata.therapeutic.therapeutic_repair_count =
                self.therapeutic_manager.alliance.repair_count;
            metadata.therapeutic.therapeutic_clinical_severity =
                self.therapeutic_manager.client_model.clinical_severity();
            metadata.therapeutic.therapeutic_narrative_fragments =
                self.therapeutic_manager.narrative.fragments.len();
            metadata.therapeutic.therapeutic_serotonin_debt =
                self.therapeutic_manager.regulation_engine.serotonin_debt();
            metadata.therapeutic.therapeutic_dopamine_debt =
                self.therapeutic_manager.regulation_engine.dopamine_debt();
            metadata.therapeutic.therapeutic_dream_accuracy =
                self.therapeutic_manager.dream_prediction_accuracy();

            if let Some(ref mut text) = self.language_comm.last_broca_text {
                if let Some(violation) = self.therapeutic_manager.scope_guard.check_response(text) {
                    tracing::warn!(
                        target: "therapeutic_manager::scope_guard",
                        violation = ?violation,
                        cycle = self.stats.total_cycles,
                        "Scope violation detected in Broca output — injecting disclaimer"
                    );
                    *text = self.therapeutic_manager.scope_guard.apply_disclaimers(text);
                    metadata.therapeutic.therapeutic_scope_violation = format!("{:?}", violation);
                }
            }

            metadata.therapeutic.therapeutic_last_rupture_type = self
                .therapeutic_manager
                .alliance
                .last_rupture_type()
                .map(|rt| rt.as_str().to_string())
                .unwrap_or_default();
            metadata.therapeutic.therapeutic_repair_rate =
                self.therapeutic_manager.alliance.repair_rate();
            metadata.therapeutic.therapeutic_withdrawal_count =
                self.therapeutic_manager.alliance.withdrawal_count();
            metadata.therapeutic.therapeutic_confrontation_count =
                self.therapeutic_manager.alliance.confrontation_count();

            let rdoc = &self.therapeutic_manager.client_model.rdoc_profile;
            metadata.therapeutic.therapeutic_rdoc_profile = [
                rdoc.score(symthaea_clinical::RDocDomain::NegativeValence),
                rdoc.score(symthaea_clinical::RDocDomain::PositiveValence),
                rdoc.score(symthaea_clinical::RDocDomain::CognitiveSystems),
                rdoc.score(symthaea_clinical::RDocDomain::SocialProcesses),
                rdoc.score(symthaea_clinical::RDocDomain::ArousalRegulatory),
                rdoc.score(symthaea_clinical::RDocDomain::Sensorimotor),
            ];
            metadata.therapeutic.therapeutic_perpetuating_factors = self
                .therapeutic_manager
                .formulation
                .perpetuating
                .iter()
                .map(|f| f.description.clone())
                .collect();
            metadata.therapeutic.therapeutic_protective_factors = self
                .therapeutic_manager
                .formulation
                .protective
                .iter()
                .map(|f| f.description.clone())
                .collect();
            metadata.therapeutic.therapeutic_strategy_effectiveness =
                symthaea_therapeutic::RegulationStrategy::ALL
                    .iter()
                    .filter_map(|s| {
                        self.therapeutic_manager
                            .regulation_engine
                            .effectiveness(s)
                            .filter(|eff| eff.applications > 0)
                            .map(|eff| {
                                (s.as_str().to_string(), eff.success_rate(), eff.applications)
                            })
                    })
                    .collect();
            metadata.therapeutic.therapeutic_temporal_coherence =
                self.therapeutic_manager.narrative.temporal_coherence();

            let st = &self.therapeutic_manager.last_shadow_telemetry;
            metadata.therapeutic.shadow_total_pressure = st.total_shadow_pressure;
            metadata.therapeutic.shadow_fragment_count = st.shadow_fragment_count;
            metadata.therapeutic.shadow_peak_pressure = st.peak_fragment_pressure;
            metadata.therapeutic.shadow_mean_prediction_error = st.shadow_mean_prediction_error;
            metadata.therapeutic.shadow_projection_detections = st.projection_detections;
            metadata.therapeutic.shadow_surfacing_indicated = st.surfacing_indicated;
            metadata.therapeutic.shadow_dream_queue_depth = st.dream_queue_depth;
            metadata.therapeutic.shadow_best_dream_phi = st.best_dream_phi_improvement;
            metadata.therapeutic.shadow_pressure_trend = st.pressure_trend;
            metadata.therapeutic.shadow_to_narrative_ratio = st.shadow_to_narrative_ratio;
        }

        #[cfg(feature = "nurture")]
        {
            if let Some(ref nurture) = self.nurture_attachment {
                metadata.attachment_style = Some(nurture.style().as_str().to_string());
                metadata.attachment_security = Some(nurture.security_score());
            }
        }

        if let Some(ref km) = self.memory.knowledge_manager {
            let telem = km.telemetry();
            let sigs = km.signals();
            metadata.knowledge_graph_size = telem.graph_size;
            metadata.knowledge_best_similarity = telem.best_search_similarity;
            metadata.knowledge_causal_edges = telem.causal_edge_count;
            metadata.knowledge_epistemic_surprise = sigs.epistemic_surprise;
            metadata.knowledge_calibration_ece = telem.calibration_ece;
            metadata.knowledge_contradictions = telem.contradictions_detected;
        }

        #[cfg(feature = "glyph_codex")]
        {
            metadata.glyph_dominant_modality =
                self.glyph_manager.dominant_modality().name().to_string();
            metadata.glyph_coherence = self.glyph_manager.last_coherence().value as f32;
            metadata.glyph_resonant_name = self
                .glyph_manager
                .resonant_glyph_name()
                .unwrap_or("")
                .to_string();
            metadata.glyph_spiral_position = self.glyph_manager.spiral_position();
        }

        metadata.reasoning_engine_enabled = cfg!(feature = "reasoning_engine");
        metadata.mesh_enabled = cfg!(feature = "mesh");
        metadata.ssm_language_enabled = cfg!(feature = "ssm_language");

        if let Some(svc) = self.network_service() {
            metadata.mesh.swarm_peer_count = svc.peer_count() as u32;
            metadata.mesh.network_mean_phi = svc.network_mean_phi();
            metadata.mesh.network_coherence = svc.network_coherence();
        }

        #[cfg(feature = "safety-agents")]
        {
            let level = self.safety_supervisor.agent.current_level();
            metadata.immune_safety_level = level.as_str_upper().to_string();
            let telem = self
                .safety_supervisor
                .guardian_state
                .telemetry(self.stats.total_cycles as usize);
            metadata.immune_guardian_posture = telem.posture;
            metadata.immune_patrol_active = telem.patrol_active;
            metadata.immune_emergency_cycles = telem.emergency_cycles;
        }
        #[cfg(feature = "sentinel")]
        {
            let st = self.sentinel_manager.telemetry();
            metadata.immune_active_threats = st.active_threats as u32;
            metadata.immune_max_severity = st.max_severity;
            metadata.immune_threat_level = st.threat_level;
            metadata.immune_quarantined_peers = st.quarantined_peers as u32;
            metadata.immune_threat_patterns = self.threat_memory.pattern_count() as u32;
            metadata.immune_response_active = self.collective_immune_state.immune_response_active;
        }
        #[cfg(feature = "neuroevolution")]
        {
            let nt = self.neuroevolution_manager.telemetry();
            metadata.neuroevo_generation = nt.generation;
            metadata.neuroevo_best_fitness = nt.best_fitness;
            metadata.neuroevo_diversity = nt.diversity;
            metadata.neuroevo_species_count = nt.species_count;
        }
        #[cfg(feature = "safety-agents")]
        {
            metadata.defense_actions_proposed = self.defense_actions_proposed;
            metadata.defense_actions_approved = self.defense_actions_approved;
            metadata.immune_motor_halt =
                self.carryover.quality.safety_motor_halt || self.carryover.quality.subsystem_veto;
        }

        metadata.multimodal = self.multimodal_manager.telemetry();
        #[cfg(feature = "vision-manifold")]
        {
            metadata.vision = perception.vision_telemetry.clone();
        }
    }

    /// Derive Creative cortical-region activation from live creative/canvas
    /// machinery, when compiled in.
    ///
    /// Returns `None` when neither the `creative` nor `canvas` feature is
    /// enabled (or their managers are absent) — the caller then falls back to
    /// the legacy surprise placeholder heuristic.
    ///
    /// Mapping (all constants in `thresholds/feedback.rs`):
    /// - `creative`: baseline + generation burst + aesthetic-EMA tonic
    ///   + refinement (generate-evaluate iterations), clamped to [0, 1].
    /// - `canvas` only: baseline + weaker frame burst + weaker aesthetic-EMA
    ///   tonic — the canvas passively renders state rather than making art.
    #[cfg(feature = "neural_validation")]
    fn creative_region_activation(&self) -> Option<f32> {
        #[cfg(feature = "creative")]
        if let Some(ref cm) = self.sensorimotor.motor_rendering.creative_manager {
            use super::super::super::thresholds::{
                CREATIVE_REGION_AESTHETIC_GAIN, CREATIVE_REGION_BASELINE,
                CREATIVE_REGION_GENERATION_BURST, CREATIVE_REGION_REFINEMENT_GAIN,
                CREATIVE_REGION_REFINEMENT_NORM,
            };
            let t = cm.last_telemetry();
            let burst = if t.generated {
                CREATIVE_REGION_GENERATION_BURST
            } else {
                0.0
            };
            let tonic = t.aesthetic_ema.clamp(0.0, 1.0) * CREATIVE_REGION_AESTHETIC_GAIN;
            let refinement = (t.iteration_count as f32 / CREATIVE_REGION_REFINEMENT_NORM).min(1.0)
                * CREATIVE_REGION_REFINEMENT_GAIN;
            return Some((CREATIVE_REGION_BASELINE + burst + tonic + refinement).clamp(0.0, 1.0));
        }

        #[cfg(feature = "canvas")]
        if let Some(ref mgr) = self.sensorimotor.motor_rendering.canvas_manager {
            use super::super::super::thresholds::{
                CANVAS_REGION_AESTHETIC_GAIN, CANVAS_REGION_GENERATION_BURST,
                CREATIVE_REGION_BASELINE,
            };
            let t = mgr.last_telemetry();
            let burst = if t.generated {
                CANVAS_REGION_GENERATION_BURST
            } else {
                0.0
            };
            let tonic = mgr.aesthetic_ema().clamp(0.0, 1.0) * CANVAS_REGION_AESTHETIC_GAIN;
            return Some((CREATIVE_REGION_BASELINE + burst + tonic).clamp(0.0, 1.0));
        }

        None
    }
}
