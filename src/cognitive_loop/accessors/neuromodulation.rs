//! Neuromodulator bath, pharmacological ablation, and circadian accessors.

use crate::cognitive_loop::CognitiveLoopService;

#[allow(dead_code)]
impl CognitiveLoopService {
    cognitive_accessors! {
        // ═══════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH ACCESSORS
        // ═══════════════════════════════════════════════════════════════════

        /// Human-readable phase label for the current bath state.
        pub fn bath_phase_label(&self) -> &'static str { self.neuromodulator_bath.phase_label() }

        /// Borrow the bath phase tracker for trajectory/centroid/variance queries.
        pub fn bath_phase_tracker(&self) -> &super::super::neuromodulators::BathPhaseTracker { &self.bath_phase_tracker }

        /// 9-dimensional bath state vector [DA, NE, 5-HT, ACh, GABA, Oxy, Glut, Aden, ECB].
        pub fn bath_state_vector(&self) -> [f32; 9] { self.neuromodulator_bath.state_vector() }

        /// Borrow the bath phase transition detector.
        pub fn bath_phase_detector(&self) -> &super::super::neuromodulators::PhaseTransitionDetector { &self.bath_phase_detector }
    }

    /// Restore neurochemistry from a saved checkpoint.
    pub fn restore_neurochemistry(
        &mut self,
        ckpt: &super::super::neuromodulators::NeurochemistryCheckpoint,
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
            neuromod_sleep_consolidation_boost: self
                .neuromodulator_bath
                .sleep_consolidation_boost(),
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
            // Phase 4: neuroendocrine control telemetry
            neuromod_derived_cortisol: self.neuromodulator_bath.to_hormone_state().cortisol as f32,
            ne_ach_suppression: {
                let ne_ph = self.neuromodulator_bath.ne_phasic();
                if ne_ph > 0.3 {
                    ne_ph * 0.15
                } else {
                    0.0
                }
            },
            ach_ne_suppression: {
                let ach_eff = self.neuromodulator_bath.acetylcholine.effective();
                if ach_eff > 0.6 {
                    (ach_eff - 0.6) * 0.1
                } else {
                    0.0
                }
            },
            neuromod_gaba_effective: self.neuromodulator_bath.gaba.effective(),
            neuromod_global_inhibition: self.neuromodulator_bath.global_inhibition(),
            neuromod_oxytocin_effective: self.neuromodulator_bath.oxytocin.effective(),
            neuromod_social_coherence: self.neuromodulator_bath.social_coherence_factor(),
            neuromod_trust_factor: self.neuromodulator_bath.trust_factor(),
            neuromod_glutamate_effective: self.neuromodulator_bath.glutamate.effective(),
            neuromod_excitotoxicity_risk: self.neuromodulator_bath.excitotoxicity_risk(),
            neuromod_learning_fatigue: self.neuromodulator_bath.learning_fatigue_factor(),
            circadian_phase_offset: self.biorhythm.phase_offset as f32,
            circadian_effective_hour: self.biorhythm.effective_hour() as f32,
            // Phase 5: advanced neuroendocrine dynamics
            neuromod_adenosine_effective: self.neuromodulator_bath.adenosine.effective(),
            neuromod_sleep_pressure: self.neuromodulator_bath.sleep_pressure(),
            neuromod_allostatic_load: self.neuromodulator_bath.allostatic_load,
            neuromod_ei_ratio: self.neuromodulator_bath.ei_ratio(),
            neuromod_ei_seizure_events: self.neuromodulator_bath.ei_seizure_events,
            neuromod_bath_entropy: self.bath_phase_tracker.entropy(),
            neuromod_attractor_detected: self.bath_phase_tracker.detect_attractor().is_some(),
            active_injection_count: self.neuromodulator_bath.active_injections.len() as u8,
            // Phase 6: endocannabinoid + subtypes
            neuromod_endocannabinoid_effective: self
                .neuromodulator_bath
                .endocannabinoid
                .effective(),
            neuromod_sht_1a_signal: self.neuromodulator_bath.sht_1a_signal(),
            neuromod_sht_2a_signal: self.neuromodulator_bath.sht_2a_signal(),
            neuromod_gaba_a_signal: self.neuromodulator_bath.gaba_a_signal(),
            neuromod_gaba_b_signal: self.neuromodulator_bath.gaba_b_signal(),
        }
    }

    /// Inject a pharmacological agent into the neuromodulator bath.
    pub fn inject_pharmacological(&mut self, target: &str, dose: f32, half_life: u32) {
        self.neuromodulator_bath.inject(target, dose, half_life);
    }

    /// Clear all pharmacological injections.
    pub fn clear_pharmacological(&mut self) {
        self.neuromodulator_bath.clear_injections();
    }

    /// Couple this bath with a peer agent's state vector (oxytocin-mediated).
    pub fn couple_bath_with_peer(&mut self, peer_state: &[f32]) {
        self.neuromodulator_bath.couple_with_peer(peer_state);
    }

    /// Export the bath trajectory as a serializable timeline.
    pub fn bath_timeline(&self) -> super::super::neuromodulators::BathTimeline {
        self.bath_phase_tracker
            .to_timeline(self.neuromodulator_bath.phase_label())
    }

    /// Shift the circadian phase by the given number of hours (±12 max).
    ///
    /// Models jet lag / zeitgeber effects (Czeisler et al. 1999).
    /// The phase offset gradually returns to 0 via entrainment each cycle.
    pub fn shift_circadian_phase(&mut self, hours: f64) {
        self.biorhythm.shift_phase(hours);
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
        let cal =
            super::super::calibration::NeuromodCalibration::from_normative_z_scores(normative_z_scores);
        tracing::info!(
            coverage = cal.coverage,
            adjustments = cal.adjustments.len(),
            confidence_delta = cal.confidence_delta,
            "Psych-bench calibration ingested (pending sleep→wake)"
        );
        self.pending_calibration = Some(cal);
    }

    /// Ingest a pre-built calibration directly (e.g., from raw z-scores).
    pub fn ingest_calibration_raw(&mut self, cal: super::super::calibration::NeuromodCalibration) {
        self.pending_calibration = Some(cal);
    }

    /// Force-apply any pending calibration immediately (bypasses sleep→wake gate).
    ///
    /// Use sparingly — normal flow waits for sleep→wake transition.
    pub fn apply_pending_calibration(&mut self) {
        if let Some(cal) = self.pending_calibration.take() {
            cal.apply(&mut self.neuromodulator_bath);
            // Apply confidence delta
            self.prediction_confidence =
                (self.prediction_confidence + cal.confidence_delta).clamp(0.01, 1.0);
            let summary = cal.summary();
            tracing::info!(%summary, "Neuromod calibration applied");
            self.last_calibration_summary = Some(summary);
        }
    }

    /// Get the last applied calibration summary (if any).
    pub fn last_calibration_summary(&self) -> Option<&str> {
        self.last_calibration_summary.as_deref()
    }
}
