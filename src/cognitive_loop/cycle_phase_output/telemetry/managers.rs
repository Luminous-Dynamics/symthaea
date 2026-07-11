// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use super::prelude::*;

impl CognitiveLoopService {
    pub(in crate::cognitive_loop::cycle_phase_output) fn populate_manager_telemetry(
        &mut self,
        metadata: &mut CycleMetadata,
        feedback: &FeedbackPhaseResult,
    ) {
        {
            let voice_summary = self.language_comm.voice_coherence.voice.summary();
            metadata.voice_articulation_quality = self
                .language_comm
                .voice_coherence
                .voice
                .smoothed_articulation();
            metadata.voice_rate_stability =
                self.language_comm.voice_coherence.voice.rate_stability();
            metadata.voice_confidence = voice_summary.voice_confidence;
            metadata.voice_phi_adjustment = self
                .language_comm
                .voice_coherence
                .voice
                .compute_phi_adjustment();
        }

        metadata.perception_attention_sensitivity = self.perception_manager.attention_sensitivity();
        metadata.perception_budget_utilization = self.perception_manager.budget_utilization();
        metadata.perception_vigilant = self.perception_manager.is_vigilant();
        metadata.perception_mean_coherence = self.perception_manager.mean_coherence_score();

        metadata.drive_boredom = self.drive_manager.boredom();
        metadata.drive_flow_intensity = self.drive_manager.flow_intensity();
        metadata.drive_in_flow = self.drive_manager.in_flow();
        metadata.drive_exploration_threshold = self.drive_manager.exploration_threshold();

        metadata.learning_plasticity = self.learning_manager.plasticity();
        metadata.learning_in_dream_phase = self.learning_manager.in_dream_phase();
        metadata.learning_error_trend = self.learning_manager.error_trend();

        metadata.memory_consolidation_pressure = self.memory_manager.consolidation_pressure();
        metadata.memory_recall_quality = self.memory_manager.recall_quality();

        {
            let st = self.swarm_manager.telemetry();
            metadata.swarm_connected_peers = st.connected_peers;
            metadata.swarm_connectivity_ema = st.connectivity_ema as f32;
            metadata.swarm_mean_peer_phi = st.mean_peer_phi as f32;
            metadata.swarm_affective_contagion = st.affective_contagion as f32;
            metadata.swarm_federated_confidence = st.federated_confidence as f32;
            metadata.swarm_anomaly_count = st.anomaly_count;
        }

        #[cfg(feature = "mycelix")]
        {
            metadata.governance_reward_ema = self.governance_mgr.reward_ema() as f32;
            metadata.governance_pending_events = self.governance_mgr.pending_event_count();
            metadata.governance_pending_outcomes = self.governance_mgr.pending_outcome_count();
            metadata.governance_collective_phi = self.governance_mgr.last_collective_phi() as f32;
            metadata.governance_community_mode = self
                .governance_mgr
                .community_mode()
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            metadata.governance_blind_spot_count = self.governance_mgr.blind_spot_count();
            metadata.governance_max_blind_spot_severity =
                self.governance_mgr.max_blind_spot_severity() as f32;
            metadata.governance_epistemic_agents = self.governance_mgr.epistemic_agent_count();
            metadata.governance_harmonic_delta_max =
                self.governance_mgr.last_harmonic_delta_max() as f32;
            metadata.governance_lr_boost = self.governance_mgr.last_lr_boost() as f32;

            let fh = self.governance_mgr.finance_health();
            metadata.finance_active_positions = fh.active_positions;
            metadata.finance_stressed_positions = fh.stressed_positions;
            metadata.finance_critical_positions = fh.critical_positions;
            metadata.finance_avg_ltv = fh.avg_ltv;
            metadata.finance_sap_circulation = fh.sap_circulation;
            metadata.finance_compost_collected = fh.compost_collected;
            metadata.finance_active_covenants = fh.active_covenants;
            metadata.finance_open_breakers = fh.open_breakers;
            metadata.finance_oracle_confidence = fh.oracle_confidence;
            metadata.finance_stress_index = fh.stress_index;
        }

        #[cfg(feature = "cpg")]
        {
            let ct = self.cpg_manager.telemetry();
            metadata.cpg_sync_index = ct.sync_index as f32;
            metadata.cpg_mean_freq = ct.mean_freq as f32;
            metadata.cpg_motor_active = ct.motor_active;
            metadata.cpg_desync_alert = ct.desync_alert;
        }

        // Keep in sync with cycle.rs Phase 2.5's feature list.
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "subterranean",
            feature = "infrastructure",
            feature = "scavenger",
            feature = "agribot",
            feature = "biota",
            feature = "clime",
            feature = "phone"
        ))]
        {
            let et = &self.sensorimotor.embodiment_telemetry;
            metadata.embodiment_total_steps = et.total_steps;
            metadata.embodiment_control_effort = et.control_effort;
            metadata.embodiment_prediction_error = et.prediction_error;
            metadata.embodiment_platform = et.platform.clone();
            metadata.embodiment_num_actuators = et.num_actuators as u32;
        }

        #[cfg(feature = "advanced-manufacturing")]
        {
            let ft = self.fabrication_manager.telemetry();
            metadata.fabrication_manufacturing_fe = ft.manufacturing_free_energy;
            metadata.fabrication_design_loop_fe = ft.design_loop_free_energy;
            metadata.fabrication_safety_level = ft.safety_level;
            metadata.fabrication_anomaly_count = ft.anomaly_count;
            metadata.fabrication_anomaly_ema = ft.anomaly_ema;
            metadata.fabrication_pog_score_ema = ft.pog_score_ema;
            metadata.fabrication_active_jobs = ft.active_print_jobs;
            metadata.fabrication_reward_ema = ft.reward_ema;
            metadata.fabrication_prediction_coherence = ft.prediction_coherence;
            metadata.fabrication_mrp_planned_orders = ft.mrp_planned_orders;
            metadata.fabrication_mrp_feasible = ft.mrp_feasible;
            metadata.fabrication_mrp_shortages = ft.mrp_shortages_count;
            metadata.fabrication_mrp_work_orders = ft.mrp_work_order_count;
            metadata.fabrication_defect_prediction = ft.defect_prediction;
            metadata.fabrication_defect_confidence = ft.defect_confidence;
        }

        #[cfg(feature = "mesh")]
        {
            let rt = self.spectrum_manager.telemetry();
            metadata.spectrum_network_health = rt.network_health;
            metadata.spectrum_tier_available = {
                let mut bits: u8 = 0;
                if rt.tier_available[0] {
                    bits |= 1;
                }
                if rt.tier_available[1] {
                    bits |= 2;
                }
                if rt.tier_available[2] {
                    bits |= 4;
                }
                bits
            };
            metadata.spectrum_jamming_streak = rt.jamming_streak;
            metadata.spectrum_prediction_error = rt.spectrum_prediction_error as f32;
            metadata.spectrum_epistemic_discount = rt.epistemic_discount as f32;
            metadata.spectrum_degradation_streak = rt.degradation_streak;
            metadata.spectrum_known_peers = rt.known_peers;
            metadata.spectrum_encryption_sessions = rt.encryption_sessions;
        }

        #[cfg(feature = "mesh")]
        {
            let tt = self.time_manager.telemetry();
            metadata.sovereign_time_offset_us = tt.offset_us;
            metadata.sovereign_time_stratum = tt.stratum;
            metadata.sovereign_time_drift_ppm = tt.drift_ppm as f32;
            metadata.sovereign_time_peer_count = tt.peer_count;
            metadata.sovereign_time_quality = tt.quality;
        }
        #[cfg(feature = "mesh-trust")]
        {
            let tr = self.trust_manager.telemetry();
            metadata.sovereign_trust_avg = tr.avg_trust as f32;
            metadata.sovereign_trust_density = tr.graph_density as f32;
            metadata.sovereign_trust_anomalies = tr.anomaly_count;
            metadata.sovereign_trust_pq_fraction = tr.pq_fraction as f32;
        }
        #[cfg(feature = "social-fabric")]
        {
            let sf = self.social_fabric_manager.telemetry();
            metadata.sovereign_social_resonance_mean = sf.resonance_mean as f32;
            metadata.sovereign_social_diversity = sf.diversity as f32;
            metadata.sovereign_social_echo_risk = sf.echo_chamber_risk as f32;
            metadata.sovereign_social_peer_reach = sf.peer_reach;
        }
        #[cfg(feature = "survival")]
        {
            let sv = self.survival_manager.telemetry();
            metadata.sovereign_survival_water_pct = sv.water_pct as f32;
            metadata.sovereign_survival_power_kw = sv.power_kw as f32;
            metadata.sovereign_survival_emergency = sv.emergency_active;
            metadata.sovereign_survival_sensor_count = sv.sensor_count;
            metadata.sovereign_survival_alert_count = sv.alert_count;
        }

        #[cfg(feature = "mathematics")]
        {
            let mt = self.math_service.telemetry();
            metadata.math_problems_solved = mt.problems_solved;
            metadata.math_verification_rate = mt.verification_rate;
            metadata.math_avg_confidence = mt.average_confidence;
        }

        #[cfg(feature = "fhe-wisdom")]
        if self.config.fhe_wisdom_enabled {
            metadata.fhe_contributions_total = self.swarm_manager.fhe_contributions_total();
            metadata.fhe_aggregations_total = self.swarm_manager.fhe_aggregations_total();
            metadata.fhe_pool_count = self.swarm_manager.wisdom_pool_count();
            metadata.fhe_cycles_since_aggregation =
                self.swarm_manager.fhe_cycles_since_aggregation();
        }

        #[cfg(feature = "vision-manifold")]
        {
            metadata.vision_pe_ema = self.vision_manager.visual_pe_ema();
            metadata.vision_surprise_threshold = self.vision_manager.surprise_threshold();
            metadata.vision_low_surprise_streak = self.vision_manager.low_surprise_streak();
            metadata.vision_manifold_enabled = true;
        }

        #[cfg(feature = "ssm_language")]
        {
            metadata.language_quality_ema = self.language_manager.quality_ema();
            metadata.language_coherence_ema = self.language_manager.coherence_ema();
            metadata.language_low_coherence_streak = self.language_manager.low_coherence_streak();
        }

        #[cfg(feature = "reasoning_engine")]
        {
            metadata.reasoning_reliability_ema = self.reasoning_manager.reliability_ema();
            metadata.reasoning_cumulative_quality = self.reasoning_manager.cumulative_quality();
            metadata.reasoning_rising_streak = self.reasoning_manager.rising_streak();
            metadata.reasoning_falling_streak = self.reasoning_manager.falling_streak();
        }

        metadata.multi_obj_frontier_size = feedback.multi_obj_frontier_size;
    }
}
