// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cycle initialization, preprocessing, and end-of-cycle statistics.

use super::super::CognitiveLoopService;
use super::super::neuromodulators::NeuromodulatorBathExt;
use super::super::temporal_network::TemporalNetwork;
use super::cycle_phases::CycleInitResult;

impl CognitiveLoopService {
    // ═════════════════════════════════════════════════════════════════════════
    // Cycle init and preprocessing
    // Extracted from cycle.rs lines 100-216 (zero behavioral change).
    // ═════════════════════════════════════════════════════════════════════════

    /// Startup transient suppression, biorhythm refresh, nociception, and
    /// neuromodulator bath update. Run at the very start of each cycle.
    ///
    /// Mutates: `self.stats`, `self.behavior.curiosity_drive`, `self.carryover`,
    /// `self.feedback_state`, `self.subsystem_collector`, `self.biorhythm_mgr.rhythm`,
    /// `self.neuromod.bath`, `self.sensorimotor.somatic_bridge`, `self.behavior.emotion_contagion`,
    /// `self.thermodynamic_load`, `self.neuromod.phase_tracker`,
    /// `self.neuromod.drift_tracker`.
    pub(in crate::cognitive_loop) fn run_cycle_init(
        &mut self,
        module_timings: &mut super::super::ModuleTimings,
    ) -> CycleInitResult {
        // ── Phase 17: Startup transient suppression ─────────────────────────
        // Science: Hopfield (1982) — recurrent networks require settling time before
        // producing reliable dynamics. During warmup (cycles 0–50), suppress learning
        // rate and curiosity to prevent cementing transient noise as learned patterns.
        let startup_warmup_cycles = super::super::thresholds::STARTUP_WARMUP_CYCLES;
        let startup_suppressed = self.stats.total_cycles <= startup_warmup_cycles;
        let startup_warmup_progress = if startup_suppressed {
            self.stats.total_cycles as f32 / startup_warmup_cycles as f32
        } else {
            1.0
        };
        if startup_suppressed {
            self.stats.startup_suppressed_cycles += 1;
            // Ramp learning rate from 20% → 100% over warmup period
            let lr_scale = super::super::thresholds::STARTUP_LR_INITIAL_SCALE
                + super::super::thresholds::STARTUP_LR_RAMP_RANGE * startup_warmup_progress;
            self.stats.adaptive_learning_rate *= lr_scale;
            // Session 16 Item 3: Startup exploration ramp.
            // Instead of flat suppression, ramp from STARTUP_EXPLORATION_INITIAL to 1.0.
            // Early cycles get heavily constrained; later warmup cycles less so.
            // Science: Hopfield (1982) — settling time requires graded exploration.
            let explore_ramp = super::super::thresholds::STARTUP_EXPLORATION_INITIAL
                + (1.0 - super::super::thresholds::STARTUP_EXPLORATION_INITIAL)
                    * startup_warmup_progress;
            self.scale_exploration("startup_warmup", explore_ramp);
        }

        // Session 16 Item 6: Consciousness EMA → learning rate initialization bias.
        // High consciousness EMA → system is integrated → slight LR boost.
        // Low consciousness EMA → fragmented → slight LR dampening.
        // Science: Dehaene (2014) — integrated processing supports faster learning.
        if !startup_suppressed && self.stats.total_cycles > 30 {
            use super::super::thresholds::{
                CONSCIOUSNESS_EMA_HIGH_THRESHOLD, CONSCIOUSNESS_EMA_LOW_THRESHOLD,
                CONSCIOUSNESS_EMA_LR_BOOST, CONSCIOUSNESS_EMA_LR_DAMPEN,
            };
            let ema = self.carryover.history.consciousness_ema;
            if ema > CONSCIOUSNESS_EMA_HIGH_THRESHOLD {
                self.stats.adaptive_learning_rate *= CONSCIOUSNESS_EMA_LR_BOOST;
            } else if ema < CONSCIOUSNESS_EMA_LOW_THRESHOLD && ema > 0.0 {
                self.stats.adaptive_learning_rate *= CONSCIOUSNESS_EMA_LR_DAMPEN;
            }
            self.stats.adaptive_learning_rate = self.stats.adaptive_learning_rate.clamp(
                super::super::thresholds::ADAPTIVE_LR_MIN,
                super::super::thresholds::ADAPTIVE_LR_MAX,
            );
        }

        // Snapshot exploration_urge for end-of-cycle budget clamping (Task B)
        let exploration_urge_start = self.behavior.curiosity_drive.exploration_urge as f32;

        // Snapshot confidence for end-of-cycle drift clamping (Task G)
        self.carryover.learning.prediction_confidence = self.prediction_confidence;

        // ── Phase 2.2: Begin feedback proposal collection for this cycle ────
        self.feedback_state.begin_cycle();

        // Apply consensus overrides from the previous cycle as Set proposals,
        // syncing actual fields so both direct-mutation and proposal paths
        // start from the same base value.
        {
            let (consensus_conf, consensus_lr, consensus_explore, consensus_threshold) =
                self.feedback_state.apply_pending_consensus();
            if let Some(conf) = consensus_conf {
                self.set_confidence("consensus_writeback", conf as f32);
            }
            if let Some(lr) = consensus_lr {
                self.set_lr("consensus_writeback", lr as f32);
            }
            if let Some(explore) = consensus_explore {
                self.set_exploration("consensus_writeback", explore as f32);
            }
            if let Some(thresh) = consensus_threshold {
                self.set_threshold("consensus_writeback", thresh as f32);
            }
        }

        self.feedback_state.snapshot_cycle_start(
            self.prediction_confidence,
            self.fep.lr_boost,
            self.behavior.curiosity_drive.exploration_urge,
            self.carryover.learning.adaptive_threshold_scale,
        );
        // ── Phase 2.3: Clear subsystem output collector ────
        self.subsystem_collector.clear();
        self.carryover.quality.subsystem_veto = false;
        self.carryover.quality.safety_motor_halt = false;
        self.carryover.quality.safety_motor_readonly = false;

        // Chronobiology: refresh biorhythm every 97 cycles (co-prime amortization)
        self.biorhythm_mgr.refresh_counter += 1;
        if self.biorhythm_mgr.refresh_counter >= super::super::thresholds::BIORHYTHM_INTERVAL {
            self.biorhythm_mgr.refresh();
            // #14: Use effective_hour (with phase offset + timezone) for circadian modulation
            let effective_hour = self.biorhythm_mgr.rhythm.effective_hour();
            self.neuromod
                .bath
                .modulate_circadian_continuous(effective_hour);
            // #14: Entrain phase offset toward zero each refresh
            self.biorhythm_mgr.rhythm.entrain();
            // Record personality profile for drift detection
            let profile = self.neuromod.bath.personality_profile();
            self.neuromod.drift_tracker.record(&profile);
            // #4: Personality drift → anomaly recovery (Turrigiano 2008)
            if self.neuromod.drift_tracker.is_anomalous()
                && self.carryover.urgency.anomaly_drift_recovery == 0
            {
                self.neuromod.bath.engage_anomaly_recovery();
                self.carryover.urgency.anomaly_drift_recovery = 50;
            }
        }
        // #4: Countdown and disengage drift recovery
        if self.carryover.urgency.anomaly_drift_recovery > 0 {
            self.carryover.urgency.anomaly_drift_recovery -= 1;
            if self.carryover.urgency.anomaly_drift_recovery == 0 {
                self.neuromod.bath.disengage_anomaly_recovery();
            }
        }
        // ── Sleep→Wake transition: apply sleep recovery (Xie et al. 2013) ──
        {
            let is_sleep_now =
                self.biorhythm_mgr.rhythm.phase == crate::chronobiology::CircadianPhase::Night;
            if self.neuromod.was_sleeping && !is_sleep_now {
                let quality = (self.neuromod.bath.allostatic_recovery_cycles as f32
                    / super::super::thresholds::SLEEP_RECOVERY_QUALITY_SCALE)
                    .clamp(0.0, 1.0);
                self.neuromod.bath.apply_sleep_recovery(quality);

                // ── Psych-bench calibration: receptor sensitivity tuning ──
                // Apply pending calibration during sleep→wake with gradual confidence
                // scaling based on sleep quality. Longer sleep = stronger calibration.
                //
                // Science: Walker & Stickgold (2006) — sleep-dependent consolidation
                //   scales with duration. Tononi & Cirelli (2006) — synaptic homeostasis.
                const MIN_SLEEP_FOR_CALIBRATION: u32 = 10; // absolute minimum (~200ms at 50Hz)
                const OPTIMAL_SLEEP_FOR_CALIBRATION: u32 = 50; // full-strength (~1s at 50Hz)
                let recovery_cycles = self.neuromod.bath.allostatic_recovery_cycles;
                if self.neuromod.pending_calibration.is_some() {
                    if recovery_cycles >= MIN_SLEEP_FOR_CALIBRATION {
                        // Scale calibration strength by sleep quality
                        let sleep_quality = (recovery_cycles as f32
                            / OPTIMAL_SLEEP_FOR_CALIBRATION as f32)
                            .clamp(0.0, 1.0);
                        if sleep_quality < 1.0 {
                            tracing::info!(
                                recovery_cycles,
                                sleep_quality,
                                "Partial sleep — scaling calibration strength"
                            );
                        }
                        if let Some(ref mut cal) = self.neuromod.pending_calibration {
                            cal.scale_by_sleep_quality(sleep_quality);
                        }
                        self.apply_pending_calibration();
                        // Reset self-assessment cooldown: external calibration supersedes
                        self.neuromod.self_assessment.reset_after_calibration();
                    } else {
                        tracing::warn!(
                            recovery_cycles,
                            min = MIN_SLEEP_FOR_CALIBRATION,
                            "Sleep too short — deferring calibration to next sleep→wake"
                        );
                        // pending_calibration kept for next sleep→wake
                    }
                }
            }

            // ── Wake→Sleep transition: consolidation replay + calibration battery ──
            if !self.neuromod.was_sleeping && is_sleep_now {
                // Hippocampal replay: replay important experiences through CfC during sleep.
                // Wilson & McNaughton (1994): hippocampal replay strengthens memories offline.
                // Tononi & Cirelli (2006): sleep-dependent synaptic homeostasis.
                if self.config.enable_consolidation {
                    match self.consolidate() {
                        Ok(loss) if loss > 0.0 => {
                            tracing::info!(loss, "Sleep-onset consolidation replay completed");
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "Sleep consolidation replay failed");
                        }
                        _ => {}
                    }
                }

                // Spawn calibration battery subprocess at sleep onset so results
                // are ready by the next sleep→wake transition.
                if self.neuromod.pending_calibration.is_none() {
                    self.spawn_calibration_battery(self.stats.total_cycles as u64);
                }
            }

            self.neuromod.was_sleeping = is_sleep_now;
        }

        // ── Always-awake fallback: apply stale calibration ──
        // Systems that never enter sleep still need calibration maintenance.
        // McEwen (1998): allostatic load accumulates when corrections are deferred.
        if let (Some(_), Some(since)) = (
            &self.neuromod.pending_calibration,
            self.neuromod.pending_calibration_since_cycle,
        ) {
            let age = self.stats.total_cycles as u64 - since;
            if age >= super::super::thresholds::ALWAYS_AWAKE_STALE_CYCLES {
                tracing::info!(
                    age,
                    threshold = super::super::thresholds::ALWAYS_AWAKE_STALE_CYCLES,
                    "Always-awake fallback: applying stale calibration"
                );
                self.apply_pending_calibration();
                self.neuromod.self_assessment.reset_after_calibration();
            }
        }

        // Apply circadian plasticity to learning rate (Night=high plasticity, Day=low)
        // Halved: bath circadian baselines (Phase 2) provide the other 50%
        let plasticity_half = 1.0
            + (self.biorhythm_mgr.rhythm.plasticity_mod as f32 - 1.0)
                * super::super::thresholds::CIRCADIAN_PLASTICITY_SCALE;
        let circadian_lr = self.stats.adaptive_learning_rate * plasticity_half;
        self.stats.adaptive_learning_rate = circadian_lr.clamp(
            super::super::thresholds::ADAPTIVE_LR_MIN,
            super::super::thresholds::ADAPTIVE_LR_MAX,
        );

        // Circadian stillness: Night phase naturally elevates Sacred Stillness
        // Science: Tononi & Cirelli (2006) — synaptic homeostasis hypothesis;
        // rest is not absence of function but active consolidation.
        self.stats.circadian_stillness_boost = match self.biorhythm_mgr.rhythm.phase {
            crate::chronobiology::CircadianPhase::Night => {
                super::super::thresholds::CIRCADIAN_STILLNESS_NIGHT
            }
            crate::chronobiology::CircadianPhase::Dusk => {
                super::super::thresholds::CIRCADIAN_STILLNESS_DUSK
            }
            crate::chronobiology::CircadianPhase::Dawn => {
                super::super::thresholds::CIRCADIAN_STILLNESS_DAWN
            }
            _ => 0.0,
        };

        // ═══════════════════════════════════════════════════════════════════════
        // NOCICEPTION: Drain infrastructure errors and convert to felt signals
        // ═══════════════════════════════════════════════════════════════════════
        self.sensorimotor.somatic_bridge.update();
        let somatic_signals = self.sensorimotor.somatic_bridge.to_interoceptive_signals();
        // Apply somatic stress to thermodynamic load (additive)
        let old_thermo = self.thermodynamic_load;
        self.thermodynamic_load =
            (self.thermodynamic_load + somatic_signals.thermodynamic_load_delta).min(1.0);
        if self.thermodynamic_load != old_thermo {
            eprintln!(
                "DEBUG: run_cycle_init thermo change: {} -> {}",
                old_thermo, self.thermodynamic_load
            );
        }
        // Apply arousal spike from severe infrastructure damage
        if somatic_signals.arousal_spike > 0.0 {
            self.behavior.emotion_contagion.arousal =
                (self.behavior.emotion_contagion.arousal + somatic_signals.arousal_spike).min(1.0);
        }
        // #5: Forward somatic stress to neuromodulator bath (McEwen 2007)
        let somatic_stress_level = self.sensorimotor.somatic_bridge.systemic_stress() as f32;
        self.neuromod.bath.apply_stress(somatic_stress_level);

        // ═══════════════════════════════════════════════════════════════════════
        // THERMOCEPTION: Drain platform thermal reports and update tau modulation
        // Science: Angilletta (2009) thermal performance curves
        // ═══════════════════════════════════════════════════════════════════════
        self.sensorimotor.thermal_bridge.update();

        // ═══════════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH: Produce from previous cycle's signals (Phase A)
        // Science: Doya (2002) — DA/NE/5-HT/ACh unify metalearning modulation.
        // Uses carryover values (previous cycle) to avoid ordering dependencies.
        // ═══════════════════════════════════════════════════════════════════════
        {
            let neuromod_inputs = super::super::neuromodulators::NeuromodulatorInputs {
                prediction_error: self.stats.avg_prediction_error,
                surprise: self.stats.avg_prediction_error
                    > self.config.learning_threshold
                        * super::super::thresholds::SURPRISE_PE_MULTIPLIER,
                reward_signal: self.carryover.quality.last_value_score as f32,
                coherence: self
                    .carryover
                    .history
                    .cached_coherence
                    .unwrap_or(super::super::thresholds::COHERENCE_DEFAULT),
                arousal: self.behavior.emotion_contagion.arousal,
                binding_strength: self.carryover.quality.last_phenomenal_binding as f32,
                epistemic_confidence: self.carryover.quality.last_epistemic_confidence,
                flow_active: self.behavior.flow_state.in_flow,
                // Consciousness → neuromod baseline modulation (Dehaene et al. 2006)
                consciousness_level: self.carryover.consciousness.last_sigma.map(|s| s as f32),
                // Moral judgment → oxytocin/DA (Zak 2012)
                moral_signal: Some(self.carryover.quality.last_moral_score),
            };
            self.neuromod.bath.update(&neuromod_inputs);
        }

        // ── Phase 5: Post-update bath wiring ────────────────────────────────
        // Record bath state for phase space analysis
        self.neuromod
            .phase_tracker
            .record(self.neuromod.bath.state_vector());
        // Allostatic load accumulation (McEwen 1998)
        {
            let cortisol = self.neuromod.bath.to_hormone_state().cortisol as f32;
            let is_sleep =
                self.biorhythm_mgr.rhythm.phase == crate::chronobiology::CircadianPhase::Night;
            self.neuromod
                .bath
                .accumulate_allostatic_load(cortisol, is_sleep);
            // Adenosine clearance during sleep (Xie et al. 2013 — glymphatic)
            if is_sleep {
                self.neuromod.bath.clear_adenosine_sleep();
            }
        }

        // ── Phase transition detection (hysteresis-based, Kelso 1995) ──
        {
            let label = self.neuromod.bath.phase_label();
            self.neuromod.phase_detector.update(label);
        }

        // ── Bath metrics export (Prometheus gauges) ──
        #[cfg(feature = "api_module")]
        {
            let sv = self.neuromod.bath.state_vector();
            crate::api::metrics::update_bath_metrics(
                crate::api::metrics::global(),
                &sv,
                self.neuromod.bath.allostatic_load,
                self.neuromod.bath.ei_ratio(),
                self.neuromod.bath.sleep_pressure(),
                self.neuromod.bath.active_injections.len(),
            );
        }

        // ═══════════════════════════════════════════════════════════════════════
        // SELF-ASSESSMENT: Metacognitive performance monitoring
        // Tracks EMA of prediction error, coherence, confidence calibration,
        // attention utilization. Triggers self-calibration when drift > 1σ.
        // Science: Schmidhuber (2010) — formal theory of intrinsic motivation.
        // ═══════════════════════════════════════════════════════════════════════
        {
            let drift_anomalous = self.neuromod.drift_tracker.is_anomalous();
            // Use bath 5-HT effective directly as sustained attention proxy.
            // Previous approach (attention_sensitivity) was contaminated by ACh
            // multiplier accumulation across cycles. 5-HT effective is the clean
            // signal: low 5-HT → poor sustained attention → high "utilization".
            let sht_eff = self.neuromod.bath.serotonin.effective();
            // Inhibition error: fraction of gating signals (prefrontal veto) that fired.
            // Binary for now; extend to multi-signal average when more gates are tracked.
            let inhibition_error_rate = if self.carryover.quality.cached_prefrontal_veto {
                1.0
            } else {
                0.0
            };
            let sa_input = super::super::calibration::SelfAssessmentInput {
                prediction_error: self.stats.avg_prediction_error,
                coherence: self
                    .carryover
                    .history
                    .cached_coherence
                    .unwrap_or(super::super::thresholds::COHERENCE_DEFAULT),
                confidence_calibration_error: (self.prediction_confidence as f32
                    - (1.0 - self.stats.avg_prediction_error.min(1.0)))
                .abs(),
                // Invert 5-HT: low serotonin → high utilization (sustained attn deficit)
                attention_utilization: (1.0 - sht_eff).clamp(0.0, 1.0),
                inhibition_error_rate,
                drift_anomalous,
                // Phase 1F: 5 new proxy signals for expanded 9-transmitter calibration
                social_coherence: self.neuromod.bath.oxytocin.effective()
                    * super::super::thresholds::SOCIAL_COHERENCE_OXY_WEIGHT
                    + super::super::thresholds::SOCIAL_COHERENCE_OFFSET,
                ei_ratio: self.neuromod.bath.ei_ratio(),
                excitotoxicity_risk: self.neuromod.bath.excitotoxicity_risk(),
                sleep_pressure: self.neuromod.bath.adenosine.effective(),
                allostatic_load: self.neuromod.bath.allostatic_load,
            };
            self.neuromod.self_assessment.update(&sa_input);

            // Closed-loop calibration validation: check if previous calibration improved metrics.
            // Powers & Cisek (2021): outcome monitoring for closed-loop neuromodulation.
            if let Some(improved) = self.neuromod.calibration_validator.check_validation(
                self.stats.avg_prediction_error as f64,
                self.language_comm
                    .voice_coherence
                    .bridge
                    .smoothed_coherence()
                    .into(),
                self.neuromod.self_assessment.confidence_error_ema(),
                self.stats.total_cycles as u64,
            ) {
                let v = &self.neuromod.calibration_validator;
                tracing::info!(
                    improved,
                    total = v.total_validations(),
                    improvements = v.improvements,
                    regressions = v.regressions,
                    damping = v.regression_damping,
                    "Calibration validation completed"
                );

                // Item #3: Adaptive calibration cadence — feed validation outcomes
                // back to self-assessment trigger sensitivity.
                // Rescorla-Wagner (1972): learning rate adapts to prediction accuracy.
                let total = v.total_validations();
                if total >= 3 {
                    let improvement_ratio = v.improvements as f32 / total as f32;
                    self.neuromod
                        .self_assessment
                        .adapt_sensitivity(improvement_ratio);
                }
            }

            // Poll async calibration battery (non-blocking).
            self.poll_calibration_battery();

            // Check if self-assessment triggers calibration.
            // Guard: don't overwrite pending external (psych-bench) calibration —
            // external calibrations are higher quality than internal proxy z-scores.
            if self.neuromod.pending_calibration.is_none() {
                if let Some(cal) = self.neuromod.self_assessment.check_trigger(drift_anomalous) {
                    tracing::info!(
                        adjustments = cal.adjustments.len(),
                        confidence_delta = cal.confidence_delta,
                        drift_anomalous,
                        "Self-assessment triggered auto-calibration"
                    );
                    self.neuromod.pending_calibration = Some(cal);
                    self.neuromod.pending_calibration_since_cycle =
                        Some(self.stats.total_cycles as u64);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE -1: Ingest background-trained weights (non-blocking)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut trainer) = self.async_trainer {
            if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                trainer.apply_latest_weights(cfc);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // MORAL FREE ENERGY → EXPLORATION BOOST (FEP-principled)
        // High moral free energy = novel moral territory = boost exploration
        // to encounter scenarios in underrepresented harmony dimensions.
        //
        // Replaces raw topology completeness with continuous FEP signal:
        //   F = D_KL(q || p) + H(q)
        // where q = current harmony distribution, p = prior/expected.
        // High F → large KL divergence from moral prior → explore more.
        //
        // Science: Friston (2010) — active inference drives exploration to
        // minimize expected free energy; here applied to the moral manifold.
        // ═══════════════════════════════════════════════════════════════════════
        {
            // Copy values to avoid borrow overlap with adjust_exploration(&mut self)
            let free_energy = self.ethics_engine.last_moral_free_energy().free_energy;
            let gain = self.ethics_engine.moral_exploration_gain();
            let (scenario_count, completeness) = {
                let topo = self.ethics_engine.moral_topology().last_summary();
                (topo.scenario_count, topo.completeness)
            };

            // Bidirectional feedback: last cycle's PE modulates FE→exploration gain.
            // Uses avg_prediction_error as the outcome signal and the EMA-smoothed FE
            // as baseline — when exploration reduced PE below the smoothed FE expectation,
            // the coupling strengthens; when PE rose, it decays.
            if self.stats.total_cycles > 5 {
                let pe = self.stats.avg_prediction_error;
                let fe_ema = self.ethics_engine.moral_fe_ema() as f32;
                // Normalize: baseline PE ~0.3 for typical FE-driven exploration cycles
                let baseline_pe = super::super::thresholds::FEP_BASELINE_PE_BASE
                    + fe_ema * super::super::thresholds::FEP_BASELINE_PE_EMA_FACTOR;
                self.ethics_engine
                    .feedback_exploration_outcome(pe, baseline_pe);
            }

            // FEP-driven: continuous moral free energy signal with adaptive gain.
            // F > 0.5 → novel moral territory → explore (scaled by adaptive gain)
            // F < 0.5 → familiar moral ground → no exploration boost
            // Gain adapts via feedback_exploration_outcome() [0.05, 0.25].
            if free_energy > super::super::thresholds::MORAL_FE_EXPLORATION_THRESHOLD {
                let fe_boost = ((free_energy
                    - super::super::thresholds::MORAL_FE_EXPLORATION_THRESHOLD)
                    * gain as f64)
                    .min(super::super::thresholds::MORAL_FE_BOOST_CAP)
                    as f32;
                self.adjust_exploration("moral_free_energy", fe_boost);
            }

            // Topology completeness still provides structural signal:
            // When fewer than 3 of 8 harmonies explored, boost regardless of F.
            // This catches cold-start (prior is zero, F is undefined/zero).
            if scenario_count >= super::super::thresholds::MORAL_TOPOLOGY_MIN_SCENARIOS
                && completeness < super::super::thresholds::MORAL_TOPOLOGY_COMPLETENESS_THRESHOLD
            {
                let structural_boost =
                    (super::super::thresholds::MORAL_TOPOLOGY_COMPLETENESS_THRESHOLD
                        - completeness)
                        * super::super::thresholds::MORAL_TOPOLOGY_STRUCTURAL_BOOST_SCALE; // up to +0.09
                self.adjust_exploration("moral_topology_gap", structural_boost as f32);
            }
        }

        let _ = module_timings; // consumed by caller for timing
        CycleInitResult {
            exploration_urge_start,
            startup_suppressed,
            startup_warmup_progress,
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // End-of-cycle stats and telemetry
    // Extracted from cycle.rs post-metadata section (zero behavioral change).
    // ═════════════════════════════════════════════════════════════════════════

    /// Update cumulative stats, neuromod EMA, and populate remaining metadata fields.
    ///
    /// Called after the metadata struct literal is assembled.
    pub(in crate::cognitive_loop) fn run_end_of_cycle_stats(
        &mut self,
        metadata: &mut super::super::CycleMetadata,
        resonator_wm_primed: bool,
        resonator_promotions: usize,
        codebook_evictions: usize,
        codebook_diversity: f32,
        fep_surprise: f64,
        surprise_thresh: f64,
        neuromod_attention_alloc: f32,
        phasic_da_replay_boost: usize,
        ne_reorienting_boost: f32,
        ne_arousal_feedback: f32,
        confidence_velocity: f32,
        sht_crash_dip: f32,
        exploration_sht_drain: f32,
    ) {
        // Apply neuromodulator telemetry (replaces flat fields with nested struct)
        metadata.neuromod = self.collect_neuromod_telemetry(neuromod_attention_alloc);

        // Phase 4: local-variable telemetry fields (not bath-derived)
        metadata.neuromod_phasic_replay_boost = phasic_da_replay_boost;
        metadata.neuromod_ne_reorienting_boost = ne_reorienting_boost;
        metadata.neuromod_drift_recovery_remaining = self.carryover.urgency.anomaly_drift_recovery;

        // Populate inhibition error count from metadata flags (prefrontal veto,
        // reasoning gate block, safety block). Feeds back into self-assessment
        // NE proxy via SelfAssessmentInput::inhibition_error_rate next cycle.
        metadata.neuromod.inhibition_errors_this_cycle = metadata.prefrontal_veto as u8
            + metadata.reasoning_gate_blocked as u8
            + metadata.safety_blocked as u8;
        metadata.ne_arousal_feedback = ne_arousal_feedback;
        metadata.confidence_velocity = confidence_velocity;
        metadata.sht_crash_dip = sht_crash_dip > 0.0;
        metadata.exploration_sht_drain = exploration_sht_drain;

        // Update cumulative stats for resonator-memory loop diagnostics
        if resonator_wm_primed {
            self.stats.resonator_wm_primed_count += 1;
        }
        self.stats.resonator_promotions_total += resonator_promotions as u64;
        self.stats.codebook_evictions_total += codebook_evictions as u64;
        if codebook_diversity > 0.0 {
            self.stats.codebook_diversity = codebook_diversity;
        }
        if fep_surprise > surprise_thresh {
            self.stats.fep_surprise_replay_boosts += 1;
        }

        // Exocortex trigger counter
        if self.neuromod.bath.should_query_exocortex() {
            self.stats.exocortex_triggers += 1;
        }

        // Neuromodulator EMA stats (alpha=0.05)
        {
            let alpha = super::super::thresholds::NEUROMOD_EMA_ALPHA;
            let da = self.neuromod.bath.dopamine.effective();
            let ne = self.neuromod.bath.noradrenaline.effective();
            let sht = self.neuromod.bath.serotonin.effective();
            let ach = self.neuromod.bath.acetylcholine.effective();
            self.stats.avg_dopamine += alpha * (da - self.stats.avg_dopamine);
            self.stats.avg_noradrenaline += alpha * (ne - self.stats.avg_noradrenaline);
            self.stats.avg_serotonin += alpha * (sht - self.stats.avg_serotonin);
            self.stats.avg_acetylcholine += alpha * (ach - self.stats.avg_acetylcholine);
        }

        // Populate v0.8.0 Resonance Metadata
        metadata.temporal.thermodynamic_load = self.thermodynamic_load;
        metadata.embodied.somatic_stress = self.sensorimotor.somatic_bridge.systemic_stress();
        metadata.embodied.mood_temperature = self.mood_temperature;
        // Phase 2.2: feedback proposal attribution telemetry
        metadata.feedback.feedback_confidence_proposals =
            self.feedback_state.confidence.len() as u32;
        metadata.feedback.feedback_lr_proposals = self.feedback_state.learning_rate.len() as u32;
        metadata.feedback.feedback_exploration_proposals =
            self.feedback_state.exploration.len() as u32;
        metadata.feedback.feedback_threshold_proposals = self.feedback_state.threshold.len() as u32;
        // Consensus outcomes from last end_cycle() integration
        if let Some(ref consensus) = self.feedback_state.last_consensus {
            metadata.feedback.consensus_confidence = consensus.consensus_confidence;
            metadata.feedback.consensus_lr = consensus.consensus_lr;
            metadata.feedback.consensus_exploration = consensus.consensus_exploration;
            metadata.feedback.consensus_threshold = consensus.consensus_threshold;
        }
        if self.config.trace_feedback {
            metadata.feedback.feedback_trace_confidence = self
                .feedback_state
                .confidence
                .dump_proposals()
                .into_iter()
                .map(|(s, d)| (s.to_string(), d))
                .collect();
            metadata.feedback.feedback_trace_lr = self
                .feedback_state
                .learning_rate
                .dump_proposals()
                .into_iter()
                .map(|(s, d)| (s.to_string(), d))
                .collect();
            metadata.feedback.feedback_trace_exploration = self
                .feedback_state
                .exploration
                .dump_proposals()
                .into_iter()
                .map(|(s, d)| (s.to_string(), d))
                .collect();
            metadata.feedback.feedback_trace_threshold = self
                .feedback_state
                .threshold
                .dump_proposals()
                .into_iter()
                .map(|(s, d)| (s.to_string(), d))
                .collect();
        }
    }
}
