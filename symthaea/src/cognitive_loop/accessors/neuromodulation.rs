// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neuromodulator bath, pharmacological ablation, and circadian accessors.

use crate::cognitive_loop::CognitiveLoopService;

#[allow(dead_code)]
impl CognitiveLoopService {
    cognitive_accessors! {
        // ═══════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH ACCESSORS
        // ═══════════════════════════════════════════════════════════════════

        /// Human-readable phase label for the current bath state.
        pub fn bath_phase_label(&self) -> &'static str { self.neuromod.bath.phase_label() }

        /// Borrow the bath phase tracker for trajectory/centroid/variance queries.
        pub fn bath_phase_tracker(&self) -> &super::super::neuromodulators::BathPhaseTracker { &self.neuromod.phase_tracker }

        /// 9-dimensional bath state vector [DA, NE, 5-HT, ACh, GABA, Oxy, Glut, Aden, ECB].
        pub fn bath_state_vector(&self) -> [f32; 9] { self.neuromod.bath.state_vector() }

        /// Borrow the bath phase transition detector.
        pub fn bath_phase_detector(&self) -> &super::super::neuromodulators::PhaseTransitionDetector { &self.neuromod.phase_detector }
    }

    /// Restore neurochemistry from a saved checkpoint.
    pub fn restore_neurochemistry(
        &mut self,
        ckpt: &super::super::neuromodulators::NeurochemistryCheckpoint,
    ) {
        self.neuromod.bath.restore(ckpt);
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
        self.neuromod.bath.clamp_levels(da, ne, sht, ach);
    }

    /// Collect neuromodulator telemetry for CycleMetadata construction.
    ///
    /// Builds a [`NeuromodTelemetry`] snapshot from the current bath state,
    /// personality drift tracker, and loop stats. Call once per cycle during
    /// metadata assembly, then assign directly to `metadata.neuromod`.
    pub(crate) fn collect_neuromod_telemetry(
        &self,
        neuromod_attention_alloc: f32,
    ) -> super::super::NeuromodTelemetry {
        use super::super::neuromodulators::NeuromodulatorBathExt;

        super::super::NeuromodTelemetry {
            exocortex_query_suggested: self.neuromod.bath.should_query_exocortex(),
            neuromod_personality: self.neuromod.bath.personality_description(),
            dopamine_effective: self.neuromod.bath.dopamine.effective(),
            noradrenaline_effective: self.neuromod.bath.noradrenaline.effective(),
            serotonin_effective: self.neuromod.bath.serotonin.effective(),
            acetylcholine_effective: self.neuromod.bath.acetylcholine.effective(),
            neuromod_personality_drift: self.neuromod.drift_tracker.drift_rate(),
            neuromod_personality_drift_anomalous: self.neuromod.drift_tracker.is_anomalous(),
            neuromod_gradient_scale: self.neuromod.bath.gradient_scale_factor(),
            neuromod_threshold_gate: self.neuromod.bath.threshold_gate(),
            exocortex_trigger_count: self.stats.exocortex_triggers,
            neuromod_da_phasic: self.neuromod.bath.da_phasic(),
            neuromod_ne_phasic: self.neuromod.bath.ne_phasic(),
            neuromod_consciousness_mod: self.neuromod.bath.consciousness_modulation(),
            neuromod_sleep_consolidation_boost: self.neuromod.bath.sleep_consolidation_boost(),
            neuromod_attention_allocation: neuromod_attention_alloc,
            neuromod_plasticity_gate: self.neuromod.bath.plasticity_gate(),
            neuromod_mcts_exploration_mod: self.neuromod.bath.mcts_exploration_modulation() as f32,
            replay_da_tag_avg: 0.0, // populated by episodic replay phase if applicable
            circadian_hour: self.biorhythm_mgr.rhythm.hour as f32,
            neuromod_da_d1: self.neuromod.bath.da_d1_effective(),
            neuromod_da_d2: self.neuromod.bath.da_d2_effective(),
            neuromod_ne_alpha: self.neuromod.bath.ne_alpha_effective(),
            neuromod_ne_beta: self.neuromod.bath.ne_beta_effective(),
            neuromod_behavioral_flexibility: self.neuromod.bath.behavioral_flexibility(),
            neuromod_snapshot: if self.stats.total_cycles % 10 == 0 {
                Some(self.neuromod.bath.snapshot())
            } else {
                None
            },
            // Phase 4: neuroendocrine control telemetry
            neuromod_derived_cortisol: self.neuromod.bath.to_hormone_state().cortisol as f32,
            ne_ach_suppression: {
                let ne_ph = self.neuromod.bath.ne_phasic();
                if ne_ph > 0.3 { ne_ph * 0.15 } else { 0.0 }
            },
            ach_ne_suppression: {
                let ach_eff = self.neuromod.bath.acetylcholine.effective();
                if ach_eff > 0.6 {
                    (ach_eff - 0.6) * 0.1
                } else {
                    0.0
                }
            },
            neuromod_gaba_effective: self.neuromod.bath.gaba.effective(),
            neuromod_global_inhibition: self.neuromod.bath.global_inhibition(),
            neuromod_oxytocin_effective: self.neuromod.bath.oxytocin.effective(),
            neuromod_social_coherence: self.neuromod.bath.social_coherence_factor(),
            neuromod_trust_factor: self.neuromod.bath.trust_factor(),
            neuromod_glutamate_effective: self.neuromod.bath.glutamate.effective(),
            neuromod_excitotoxicity_risk: self.neuromod.bath.excitotoxicity_risk(),
            neuromod_learning_fatigue: self.neuromod.bath.learning_fatigue_factor(),
            circadian_phase_offset: self.biorhythm_mgr.rhythm.phase_offset as f32,
            circadian_effective_hour: self.biorhythm_mgr.rhythm.effective_hour() as f32,
            circadian_timezone_offset: self.biorhythm_mgr.timezone_offset_hours as f32,
            // Phase 5: advanced neuroendocrine dynamics
            neuromod_adenosine_effective: self.neuromod.bath.adenosine.effective(),
            neuromod_sleep_pressure: self.neuromod.bath.sleep_pressure(),
            neuromod_allostatic_load: self.neuromod.bath.allostatic_load,
            neuromod_ei_ratio: self.neuromod.bath.ei_ratio(),
            neuromod_ei_seizure_events: self.neuromod.bath.ei_seizure_events,
            neuromod_bath_entropy: self.neuromod.phase_tracker.entropy(),
            neuromod_attractor_detected: self.neuromod.phase_tracker.detect_attractor().is_some(),
            active_injection_count: self.neuromod.bath.active_injections.len() as u8,
            // Phase 6: endocannabinoid + subtypes
            neuromod_endocannabinoid_effective: self.neuromod.bath.endocannabinoid.effective(),
            neuromod_sht_1a_signal: self.neuromod.bath.sht_1a_signal(),
            neuromod_sht_2a_signal: self.neuromod.bath.sht_2a_signal(),
            neuromod_gaba_a_signal: self.neuromod.bath.gaba_a_signal(),
            neuromod_gaba_b_signal: self.neuromod.bath.gaba_b_signal(),
            // Phase 7: self-assessment telemetry
            self_assessment_pe_ema: self.neuromod.self_assessment.pe_ema() as f32,
            self_assessment_coherence_ema: self.neuromod.self_assessment.coherence_ema() as f32,
            self_assessment_confidence_error_ema: self
                .neuromod
                .self_assessment
                .confidence_error_ema() as f32,
            self_assessment_attention_ema: self.neuromod.self_assessment.attention_utilization_ema()
                as f32,
            self_assessment_inhibition_error_ema: self
                .neuromod
                .self_assessment
                .inhibition_error_ema() as f32,
            self_assessment_observations: self.neuromod.self_assessment.observations_count(),
            self_assessment_cooldown: self.neuromod.self_assessment.cooldown_remaining(),
            self_assessment_calibration_fired: self.neuromod.self_assessment.last_triggered(),
            pending_calibration_waiting: self.neuromod.pending_calibration.is_some(),
            inhibition_errors_this_cycle: 0, // populated by output phase from metadata flags
        }
    }

    /// Inject a pharmacological agent into the neuromodulator bath.
    pub fn inject_pharmacological(&mut self, target: &str, dose: f32, half_life: u32) {
        self.neuromod.bath.inject(target, dose, half_life);
    }

    /// Clear all pharmacological injections.
    pub fn clear_pharmacological(&mut self) {
        self.neuromod.bath.clear_injections();
    }

    /// Couple this bath with a peer agent's state vector (oxytocin-mediated).
    pub fn couple_bath_with_peer(&mut self, peer_state: &[f32]) {
        self.neuromod.bath.couple_with_peer(peer_state);
    }

    /// Export the bath trajectory as a serializable timeline.
    pub fn bath_timeline(&self) -> super::super::neuromodulators::BathTimeline {
        self.neuromod
            .phase_tracker
            .to_timeline(self.neuromod.bath.phase_label())
    }

    /// Shift the circadian phase by the given number of hours (±12 max).
    ///
    /// Models jet lag / zeitgeber effects (Czeisler et al. 1999).
    /// The phase offset gradually returns to 0 via entrainment each cycle.
    pub fn shift_circadian_phase(&mut self, hours: f64) {
        self.biorhythm_mgr.rhythm.shift_phase(hours);
    }

    /// Set the timezone offset (hours from UTC), routing the delta through
    /// `shift_phase()` for gradual entrainment (jet lag model).
    pub fn set_timezone(&mut self, offset_hours: f64) {
        self.biorhythm_mgr.set_timezone(offset_hours);
    }

    /// Current timezone offset in hours from UTC.
    pub fn timezone_offset_hours(&self) -> f64 {
        self.biorhythm_mgr.timezone_offset_hours
    }

    /// Ingest psych-bench calibration data for deferred application.
    ///
    /// Accepts sign-corrected z-scores from a `NormativeReport` (positive = better).
    /// Each tuple is `(benchmark_name, key_metric, z_score)`.
    ///
    /// The calibration is stored and applied on the next sleep→wake transition,
    /// mirroring biological receptor sensitivity adjustment during sleep
    /// (Tononi & Cirelli 2006 — synaptic homeostasis hypothesis).
    ///
    /// # Example
    /// ```ignore
    /// let scores = vec![
    ///     ("Executive::Stroop", "stroop_effect", 1.5),
    ///     ("WorM::N-back", "nback_2::accuracy", -0.8),
    /// ];
    /// service.ingest_calibration(&scores);
    /// // Applied automatically on next sleep→wake transition
    /// ```
    pub fn ingest_calibration(&mut self, normative_z_scores: &[(&str, &str, f64)]) {
        let cal = super::super::calibration::NeuromodCalibration::from_normative_z_scores(
            normative_z_scores,
        );
        tracing::info!(
            coverage = cal.coverage,
            adjustments = cal.adjustments.len(),
            confidence_delta = cal.confidence_delta,
            "Psych-bench calibration ingested (pending sleep→wake)"
        );
        self.neuromod.pending_calibration = Some(cal);
        self.neuromod.pending_calibration_since_cycle = Some(self.stats.total_cycles as u64);
    }

    /// Ingest a pre-built calibration directly (e.g., from raw z-scores).
    pub fn ingest_calibration_raw(&mut self, cal: super::super::calibration::NeuromodCalibration) {
        self.neuromod.pending_calibration = Some(cal);
        self.neuromod.pending_calibration_since_cycle = Some(self.stats.total_cycles as u64);
    }

    /// Force-apply any pending calibration immediately (bypasses sleep→wake gate).
    ///
    /// Use sparingly — normal flow waits for sleep→wake transition.
    pub fn apply_pending_calibration(&mut self) {
        use super::super::thresholds::{NEUROMOD_BASELINE_MAX, NEUROMOD_BASELINE_MIN};
        if let Some(mut cal) = self.neuromod.pending_calibration.take() {
            self.neuromod.pending_calibration_since_cycle = None;
            // Record in history before applying (captures the intended adjustment)
            self.neuromod
                .calibration_history
                .record(&cal, self.stats.total_cycles as u64);

            // Check for systematic drift and auto-adjust baselines.
            // McEwen (1998): allostatic overload from persistent one-directional
            // correction → nudge the tonic baseline to reduce corrective load.
            let nudges = self.neuromod.calibration_history.compute_baseline_nudges();
            for (transmitter, nudge) in &nudges {
                let applied = match transmitter.as_str() {
                    "DA" => {
                        self.neuromod.bath.dopamine.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    "NE" | "NE-alpha" | "NE-beta" => {
                        self.neuromod.bath.noradrenaline.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    "5-HT" => {
                        self.neuromod.bath.serotonin.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    "ACh" => {
                        self.neuromod.bath.acetylcholine.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    "GABA" | "GABA-A" | "GABA-B" => {
                        self.neuromod.bath.gaba.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    "Oxytocin" => {
                        self.neuromod.bath.oxytocin.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    "Glutamate" => {
                        self.neuromod.bath.glutamate.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    "Adenosine" => {
                        self.neuromod.bath.adenosine.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    "Endocannabinoid" => {
                        self.neuromod.bath.endocannabinoid.adjust_baseline(
                            *nudge,
                            NEUROMOD_BASELINE_MIN,
                            NEUROMOD_BASELINE_MAX,
                        );
                        true
                    }
                    _ => false,
                };
                if applied {
                    tracing::info!(
                        %transmitter,
                        nudge = %format!("{nudge:+.4}"),
                        history_len = self.neuromod.calibration_history.len(),
                        "Auto-adjusted baseline from systematic drift"
                    );
                }
            }

            // Record pre-calibration metrics for closed-loop validation.
            // Powers & Cisek (2021): closed-loop neuromodulation requires outcome monitoring.
            self.neuromod.calibration_validator.record_pre_calibration(
                self.stats.avg_prediction_error as f64,
                self.language_comm
                    .voice_coherence
                    .bridge
                    .smoothed_coherence()
                    .into(),
                self.neuromod.self_assessment.confidence_error_ema(),
                self.stats.total_cycles as u64,
            );

            // Scale calibration adjustments by validator's damping multiplier.
            // If previous calibrations caused regressions, dampen future ones.
            let damping = self.neuromod.calibration_validator.adjustment_multiplier();
            if damping < 1.0 {
                for adj in &mut cal.adjustments {
                    // Move factor toward 1.0 (neutral) by damping proportion
                    adj.sensitivity_factor = 1.0 + (adj.sensitivity_factor - 1.0) * damping;
                }
                tracing::info!(damping, "Calibration adjustments dampened by validator");
            }

            cal.apply(&mut self.neuromod.bath);
            // Apply confidence delta (routed through feedback proposal system).
            // Note: adjust_confidence clamps to [0.01, 0.99], so the 0.01 floor
            // is inherently preserved — no separate .max(0.01) bypass needed.
            self.adjust_confidence("neuromod_calibration", cal.confidence_delta * damping);
            let summary = cal.summary();
            tracing::info!(%summary, "Neuromod calibration applied");
            self.neuromod.last_calibration_summary = Some(summary);
        }
    }

    /// Get the last applied calibration summary (if any).
    pub fn last_calibration_summary(&self) -> Option<&str> {
        self.neuromod.last_calibration_summary.as_deref()
    }

    /// Export pending calibration as a shareable profile for multi-agent coordination.
    ///
    /// Returns `None` if no pending calibration exists.
    pub fn export_calibration_profile(
        &self,
        agent_id: Option<String>,
    ) -> Option<super::super::calibration::SharedCalibrationProfile> {
        self.neuromod
            .pending_calibration
            .as_ref()
            .map(|cal| cal.to_shareable(agent_id))
    }

    /// Merge a peer agent's calibration profile into pending calibration.
    ///
    /// Weight formula: `base_weight + oxytocin.effective() * oxy_scale`
    /// - `base_weight` = 0.05 (minimum peer influence regardless of oxytocin)
    /// - `oxy_scale` = 0.15 (oxytocin modulates trust-based social learning)
    /// - Total weight capped at 0.35 by merge internals (0.5 hard cap in merge)
    ///
    /// Higher oxytocin → more trust → stronger peer influence.
    /// At typical oxy levels (~1.0), weight ≈ 0.20.
    /// At high oxy (~1.5), weight ≈ 0.275.
    ///
    /// If no pending calibration exists, this is a no-op — we don't create
    /// calibrations from peer data alone.
    ///
    /// Science: Zak (2012) — oxytocin modulates trust-based social learning;
    ///   Boyd & Richerson (1985) — conformist transmission in social learning.
    pub fn merge_peer_calibration(
        &mut self,
        peer: &super::super::calibration::SharedCalibrationProfile,
    ) {
        if let Some(ref mut cal) = self.neuromod.pending_calibration {
            const BASE_WEIGHT: f32 = 0.05;
            const OXY_SCALE: f32 = 0.15;
            let oxy_weight = BASE_WEIGHT + self.neuromod.bath.oxytocin.effective() * OXY_SCALE;
            // merge_peer_calibration clamps at 0.5, but we cap at 0.35 here
            let weight = oxy_weight.min(0.35);
            cal.merge_peer_calibration(peer, weight);
            tracing::info!(
                weight,
                oxy_effective = self.neuromod.bath.oxytocin.effective(),
                peer_agent = ?peer.agent_id,
                "Merged peer calibration profile"
            );
        }
    }

    /// Merge a batch of peer calibration profiles (from Mind's pending_peer_calibrations).
    ///
    /// Convenience method for external bridges: drains a Vec of profiles and merges
    /// each one using oxytocin-modulated weight.
    pub fn merge_peer_calibrations_batch(
        &mut self,
        profiles: Vec<super::super::calibration::SharedCalibrationProfile>,
    ) {
        for profile in &profiles {
            self.merge_peer_calibration(profile);
        }
    }

    /// Spawn calibration battery as async subprocess during Night phase onset.
    ///
    /// Non-blocking: spawns the child process and stores the handle. Call
    /// `poll_calibration_battery()` on subsequent cycles to check for completion.
    /// Only available when the `calibration_battery` binary exists on PATH.
    pub fn spawn_calibration_battery(&mut self, seed: u64) {
        // Don't spawn if one is already running
        if self.neuromod.calibration_battery_child.is_some() {
            return;
        }
        match std::process::Command::new("calibration_battery")
            .args(["--seed", &seed.to_string(), "--trials", "5"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(child) => {
                tracing::info!(pid = child.id(), "Spawned async calibration battery");
                self.neuromod.calibration_battery_child = Some(child);
                self.neuromod.calibration_battery_spawned_at = Some(std::time::Instant::now());
            }
            Err(e) => {
                tracing::debug!(?e, "calibration_battery not available on PATH");
            }
        }
    }

    /// Poll the running calibration battery for completion (non-blocking).
    ///
    /// Returns `true` if the battery finished and results were ingested.
    /// Kills the subprocess if it exceeds 30 seconds (hung process protection).
    /// Call this once per cycle during the neuromod phase.
    pub fn poll_calibration_battery(&mut self) -> bool {
        let child = match self.neuromod.calibration_battery_child.as_mut() {
            Some(c) => c,
            None => return false,
        };

        // Timeout protection: kill hung subprocess after 30 seconds
        const BATTERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
        if let Some(spawned_at) = self.neuromod.calibration_battery_spawned_at {
            if spawned_at.elapsed() > BATTERY_TIMEOUT {
                tracing::warn!(
                    elapsed_secs = spawned_at.elapsed().as_secs(),
                    "Calibration battery timed out — killing subprocess"
                );
                let _ = child.kill();
                let _ = child.wait(); // reap zombie
                self.neuromod.calibration_battery_child.take();
                self.neuromod.calibration_battery_spawned_at.take();
                return false;
            }
        }

        // Non-blocking check: try_wait returns Ok(Some(status)) if done
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process finished — take ownership to read stdout
                let Some(mut child) = self.neuromod.calibration_battery_child.take() else {
                    tracing::warn!("calibration_battery_child was None despite try_wait Ok(Some)");
                    self.neuromod.calibration_battery_spawned_at.take();
                    return false;
                };
                self.neuromod.calibration_battery_spawned_at.take();
                if status.success() {
                    use std::io::Read;
                    let mut stdout = String::new();
                    if let Some(ref mut pipe) = child.stdout {
                        let _ = pipe.read_to_string(&mut stdout);
                    }
                    match super::super::calibration::NeuromodCalibration::from_json_z_scores(
                        &stdout,
                    ) {
                        Ok(cal) => {
                            tracing::info!(
                                adjustments = cal.adjustments.len(),
                                coverage = cal.coverage,
                                "Async calibration battery completed"
                            );
                            self.neuromod.pending_calibration = Some(cal);
                            self.neuromod.pending_calibration_since_cycle =
                                Some(self.stats.total_cycles as u64);
                            return true;
                        }
                        Err(e) => {
                            tracing::warn!(?e, "Failed to parse calibration battery output");
                        }
                    }
                } else {
                    tracing::debug!(?status, "calibration_battery exited with error");
                }
                false
            }
            Ok(None) => false, // Still running
            Err(e) => {
                tracing::warn!(?e, "Error polling calibration battery");
                self.neuromod.calibration_battery_child.take();
                self.neuromod.calibration_battery_spawned_at.take();
                false
            }
        }
    }
}
