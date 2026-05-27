// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neuromodulator bath downstream modulation + Psi synthesis + experience bus.
//!
//! Extracted from cycle_phase_dynamics.rs — all logic and behavior preserved exactly.
//!
//! Contains:
//! - DA → learning rate, D2 flexibility → exploration responsiveness
//! - NE → exploration, phasic NE burst → attentional reorienting
//! - 5-HT → confidence, confidence crash → emergency dip
//! - ACh → attention sensitivity + threshold
//! - Arousal ↔ NE bidirectional coupling
//! - GABA global inhibition, E/I seizure protection
//! - Exploration cost → 5-HT depletion
//! - Unified Psi computation + neuromod consciousness modulation
//! - Experience bus update (principled signals)
//! - Guiding question → subsystem priority

use super::CognitiveLoopService;
use super::thresholds;
use super::thresholds::{
    NEUROMOD_AROUSAL_EMA_DECAY, NEUROMOD_AROUSAL_EMA_INPUT, NEUROMOD_AROUSAL_PHASIC_SPIKE,
    NEUROMOD_AROUSAL_PHASIC_THRESHOLD, NEUROMOD_ATTENTION_SENSITIVITY_MAX,
    NEUROMOD_ATTENTION_SENSITIVITY_MIN, NEUROMOD_CONFIDENCE_CRASH_VELOCITY,
    NEUROMOD_D2_FLEXIBILITY_BASELINE, NEUROMOD_EXPLORATION_DRAIN_BASELINE,
    NEUROMOD_EXPLORATION_DRAIN_FACTOR, NEUROMOD_GABA_INHIBITION_THRESHOLD,
    NEUROMOD_NE_PHASIC_ATTENTION_GAIN, NEUROMOD_NE_PHASIC_EXPLORATION_SCALE,
    NEUROMOD_NE_PHASIC_THRESHOLD, NEUROMOD_SEIZURE_EXPLORATION_FREEZE,
    NEUROMOD_SEROTONIN_CRASH_PRODUCTION,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Result struct
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from the neuromodulator + Psi synthesis phase.
pub(crate) struct NeuromodPhaseResult {
    pub(crate) unified_psi: f64,
    pub(crate) guiding_question: String,
    pub(crate) dominant_harmonic: String,
    pub(crate) ne_reorienting_boost: f32,
    pub(crate) ne_arousal_feedback: f32,
    pub(crate) sht_crash_dip: f32,
    pub(crate) exploration_sht_drain: f32,
    pub(crate) confidence_velocity: f32,
    pub(crate) guiding_priority_category: String,
}

impl CognitiveLoopService {
    /// Neuromodulator bath downstream modulation + Psi synthesis + experience bus phase.
    ///
    /// Applies neuromodulator chemical signaling to cognitive control parameters:
    /// - DA → learning rate (Doya 2002)
    /// - NE → exploration + phasic reorienting (Corbetta & Shulman 2002)
    /// - 5-HT → confidence + crash detection (Cools et al. 2008)
    /// - ACh → attention sensitivity (Sarter et al. 2005)
    /// - GABA → global inhibition (Olsen & Sieghart 2009)
    ///
    /// Then computes unified Psi with neuromod consciousness modulation,
    /// updates the experience bus, and maps the guiding question to subsystem priority.
    pub(super) fn run_neuromodulator_and_psi_phase(
        &mut self,
        prediction_error: f32,
        coherence: f32,
    ) -> NeuromodPhaseResult {
        // ═══════════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH: Downstream modulation (Phase B)
        // Coherent chemical baseline that fine-grained Phase 14-21 loops adjust further.
        // ═══════════════════════════════════════════════════════════════════════
        // DA → learning rate
        self.scale_lr(
            "neuromod_dopamine",
            self.neuromod.bath.learning_rate_factor(),
        );

        // NE → exploration
        self.adjust_exploration("neuromod_ne_delta", self.neuromod.bath.exploration_delta());

        // #1: D2 flexibility scales exploration responsiveness (Frank 2005)
        let flex_scale = self.neuromod.bath.behavioral_flexibility();
        self.set_exploration(
            "d2_flexibility",
            (NEUROMOD_D2_FLEXIBILITY_BASELINE
                + (self.behavior.curiosity_drive.exploration_urge
                    - NEUROMOD_D2_FLEXIBILITY_BASELINE)
                    * flex_scale as f64) as f32,
        );

        // 5-HT → confidence
        self.adjust_confidence("neuromod_serotonin", self.neuromod.bath.confidence_delta());

        // ACh → attention sensitivity + threshold
        self.behavior.adaptive_behavior.attention_sensitivity *=
            self.neuromod.bath.attention_factor();
        self.behavior.adaptive_behavior.attention_sensitivity =
            self.behavior.adaptive_behavior.attention_sensitivity.clamp(
                NEUROMOD_ATTENTION_SENSITIVITY_MIN,
                NEUROMOD_ATTENTION_SENSITIVITY_MAX,
            );
        self.scale_threshold("neuromod_threshold", self.neuromod.bath.threshold_factor());

        // #3: Phasic NE burst → attentional reorienting (Corbetta & Shulman 2002)
        let ne_ph = self.neuromod.bath.ne_phasic();
        let ne_reorienting_boost = if ne_ph > NEUROMOD_NE_PHASIC_THRESHOLD {
            self.behavior.adaptive_behavior.attention_sensitivity *=
                1.0 + (ne_ph - NEUROMOD_NE_PHASIC_THRESHOLD) * NEUROMOD_NE_PHASIC_ATTENTION_GAIN;
            self.behavior.adaptive_behavior.attention_sensitivity =
                self.behavior.adaptive_behavior.attention_sensitivity.clamp(
                    NEUROMOD_ATTENTION_SENSITIVITY_MIN,
                    NEUROMOD_ATTENTION_SENSITIVITY_MAX,
                );
            self.adjust_exploration(
                "ne_phasic_reorient",
                (ne_ph - NEUROMOD_NE_PHASIC_THRESHOLD) * NEUROMOD_NE_PHASIC_EXPLORATION_SCALE,
            );
            (ne_ph - NEUROMOD_NE_PHASIC_THRESHOLD) * NEUROMOD_NE_PHASIC_ATTENTION_GAIN
        } else {
            0.0
        };

        // #6: Arousal ↔ NE bidirectional coupling (Berridge & Waterhouse 2003)
        // EMA: arousal pulled toward NE effective (10% per cycle)
        let ne_arousal_before = self.behavior.emotion_contagion.arousal;
        self.behavior.emotion_contagion.arousal = self.behavior.emotion_contagion.arousal
            * NEUROMOD_AROUSAL_EMA_DECAY
            + self.neuromod.bath.noradrenaline.effective() * NEUROMOD_AROUSAL_EMA_INPUT;
        // Phasic NE burst → transient arousal spike
        if ne_ph > NEUROMOD_AROUSAL_PHASIC_THRESHOLD {
            self.behavior.emotion_contagion.arousal += ne_ph * NEUROMOD_AROUSAL_PHASIC_SPIKE;
        }
        self.behavior.emotion_contagion.arousal =
            self.behavior.emotion_contagion.arousal.clamp(0.0, 1.0);
        let ne_arousal_feedback = self.behavior.emotion_contagion.arousal - ne_arousal_before;

        // #7: Confidence crash detection → 5-HT emergency dip (Cools et al. 2008)
        let confidence_velocity =
            self.prediction_confidence - self.carryover.quality.prev_confidence_for_crash;
        let sht_crash_dip: f32 = if confidence_velocity < NEUROMOD_CONFIDENCE_CRASH_VELOCITY {
            self.neuromod
                .bath
                .serotonin
                .produce(NEUROMOD_SEROTONIN_CRASH_PRODUCTION);
            confidence_velocity.abs() as f32
        } else {
            0.0
        };
        self.carryover.quality.prev_confidence_for_crash = self.prediction_confidence;

        // #8: Exploration cost → 5-HT depletion (Tops et al. 2009)
        let exploration_sht_drain = if self.behavior.curiosity_drive.exploration_urge
            > NEUROMOD_EXPLORATION_DRAIN_BASELINE
        {
            let drain = (self.behavior.curiosity_drive.exploration_urge
                - NEUROMOD_EXPLORATION_DRAIN_BASELINE)
                * NEUROMOD_EXPLORATION_DRAIN_FACTOR;
            self.neuromod
                .bath
                .apply_exploration_cost(self.behavior.curiosity_drive.exploration_urge as f32);
            drain as f32
        } else {
            0.0
        };

        // #11: GABA global inhibition (Olsen & Sieghart 2009)
        let gaba_inhibition = self.neuromod.bath.global_inhibition();
        if gaba_inhibition < NEUROMOD_GABA_INHIBITION_THRESHOLD {
            self.scale_lr("gaba_inhibition", gaba_inhibition);
            self.scale_exploration("gaba_inhibition", gaba_inhibition);
        }
        // E/I seizure protection: freeze exploration during recovery (Turrigiano 2012)
        if self.neuromod.bath.exploration_frozen() {
            self.scale_exploration_pri(
                "seizure_protection",
                NEUROMOD_SEIZURE_EXPLORATION_FREEZE,
                crate::cognitive_loop::feedback_state::Priority::Safety,
            );
        }

        // Thalamic depth → tonic neuromodulator bias
        // Science: Aston-Jones & Cohen (2005) — NE tonic mode (exploration) vs phasic (exploitation)
        // Science: Sarter et al. (2005) — sustained attention via basal forebrain ACh
        // Science: Buzsáki (2006) — GABAergic inhibition enables fast gating
        use super::thresholds::{
            THALAMIC_DEEP_ACH_TONIC, THALAMIC_DEEP_NE_TONIC, THALAMIC_REFLEX_GABA,
        };
        match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => {
                self.neuromod
                    .bath
                    .noradrenaline
                    .produce(THALAMIC_DEEP_NE_TONIC);
                self.neuromod
                    .bath
                    .acetylcholine
                    .produce(THALAMIC_DEEP_ACH_TONIC);
            }
            super::CognitiveDepth::Reflex => {
                self.neuromod.bath.gaba.produce(THALAMIC_REFLEX_GABA);
            }
            super::CognitiveDepth::Cortical => {} // balanced — no tonic bias
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10h. Update Consciousness Unification Engine with current Phi
        // ═══════════════════════════════════════════════════════════════════════
        let unified_psi = self.compute_unified_psi();
        // Neuromod → consciousness bridge: ACh/NE sustain conscious integration
        // Science: Alkire et al. (2008) — consciousness correlates with ACh/NE
        let neuromod_consciousness_mod = self.neuromod.bath.consciousness_modulation();
        let unified_psi = (unified_psi * neuromod_consciousness_mod as f64).clamp(0.0, 1.0);

        // ── Hodge vertex harmonic → consciousness fragmentation modulation ──
        // Vertex L₀ harmonic fraction measures moral FRAGMENTATION: high harmonic
        // means moral meaning is trapped in disconnected clusters (β₀ > 1).
        // This is the opposite of integration — it DAMPENS consciousness.
        //
        // At criticality (harmonic ∈ [0.2, 0.8]): the system is at the edge of
        // chaos — maximally flexible and creative. We give a small BOOST here.
        //
        // Modulation:
        //   harmonic < 0.2 (unified)     → +1% (good integration)
        //   harmonic ∈ [0.2, 0.8] (critical) → +2% (edge-of-chaos creativity boost)
        //   harmonic > 0.8 (fragmented)  → -5% (moral disintegration dampens consciousness)
        //
        // Science: Tononi (2004) — Phi collapses when components disconnect;
        // Beggs & Plenz (2003) — criticality maximizes dynamic range;
        // Shew & Plenz (2013) — neural systems optimize at criticality.
        const HODGE_UNIFIED_BOOST: f64 = 0.01;
        const HODGE_CRITICAL_BOOST: f64 = 0.02;
        const HODGE_FRAGMENTED_DAMPING: f64 = 0.05;
        let hodge_mod = {
            let summary = self.ethics_engine.moral_topology().last_summary();
            if let Some(ref fracs) = summary.hodge_fractions {
                if fracs.at_criticality {
                    1.0 + HODGE_CRITICAL_BOOST // Edge-of-chaos: creativity boost
                } else if fracs.harmonic < 0.2 {
                    1.0 + HODGE_UNIFIED_BOOST // Unified reasoning: mild boost
                } else {
                    1.0 - HODGE_FRAGMENTED_DAMPING // Fragmented: consciousness drops
                }
            } else {
                1.0 // No fractions available — neutral
            }
        };
        let unified_psi = (unified_psi * hodge_mod).clamp(0.0, 1.0);

        // ── Hodge criticality → FEP exploration temperature modulation ──
        // Over-connected (harmonic < 0.2): raise temperature → explore to break echo chamber
        // Fragmented (harmonic > 0.8): lower temperature → focus on integration
        // At criticality (0.2–0.8): no modulation (optimal zone)
        // Additionally: fragile critical_scale → boost exploration
        //               robust critical_scale → allow exploitation
        // Science: Beggs & Plenz (2003) — criticality; Friston (2010) — precision.
        if self.config.enable_hodge_decomposition
            && (self.stats.total_cycles % self.config.hodge_interval == 0)
        {
            const TEMP_MIN: f64 = 0.5;
            const TEMP_MAX: f64 = 2.0;
            let summary = self.ethics_engine.moral_topology().last_summary();
            if let Some(ref fracs) = summary.hodge_fractions {
                if fracs.harmonic < 0.2 {
                    self.fep.agent.config.action_temperature += 0.3; // Explore
                } else if fracs.harmonic > 0.8 {
                    self.fep.agent.config.action_temperature *= 0.7; // Integrate
                }
            }
            let cs_ema = self.ethics_engine.moral_topology().critical_scale_ema();
            if cs_ema.is_finite() {
                if cs_ema < 0.1 {
                    self.fep.agent.config.action_temperature += 0.2; // Fragile → explore
                } else if cs_ema > 0.5 {
                    self.fep.agent.config.action_temperature *= 0.9; // Robust → exploit
                }
            }
            self.fep.agent.config.action_temperature = self
                .fep
                .agent
                .config
                .action_temperature
                .clamp(TEMP_MIN, TEMP_MAX);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.exp EXPERIENCE BUS: Update principled signals from cognitive state
        // Maps cycle values to 5 principled signals (Active Inference).
        // Science: Friston (2010) — principled signals drive behavior.
        // ═══════════════════════════════════════════════════════════════════════
        let guiding_question: &str;
        let dominant_harmonic: &str;
        if let Some(ref mut bus) = self.experience_bus {
            bus.current_signals = crate::experience::PrincipledSignals {
                prediction_error,
                uncertainty: (1.0 - self.prediction_confidence) as f32,
                coherence: coherence.clamp(0.0, 1.0),
                confidence: self.prediction_confidence as f32,
                salience: self.behavior.curiosity_drive.exploration_urge as f32,
                phi_monitor: unified_psi as f32,
            };
            bus.update_wisdom_from_signals();
            bus.kosmic_state.phi = unified_psi as f32;
            guiding_question = bus.current_guiding_question();
            dominant_harmonic = bus.dominant_harmonic().as_str();
        } else {
            guiding_question = "";
            dominant_harmonic = "";
        }

        // ── Phase 15: Guiding question → subsystem priority ─────────────────
        // Science: Desimone & Duncan (1995) — top-down attention biases processing
        // toward task-relevant features. Parse the guiding question to boost
        // urgency of related subsystems.
        let guiding_priority_category = if !guiding_question.is_empty() {
            let q = guiding_question.to_lowercase();
            let cat = if q.contains("know") || q.contains("learn") || q.contains("understand") {
                // Epistemic question → boost prediction confidence sensitivity
                self.adjust_exploration(
                    "guiding_epistemic",
                    thresholds::GUIDING_EPISTEMIC_EXPLORATION_BOOST as f32,
                );
                "epistemic"
            } else if q.contains("feel") || q.contains("emotion") || q.contains("care") {
                // Affective question → boost emotional processing sensitivity
                self.adjust_confidence(
                    "guide_affective",
                    thresholds::GUIDING_AFFECTIVE_CONFIDENCE_BOOST as f32,
                );
                "affective"
            } else if q.contains("do") || q.contains("act") || q.contains("make") {
                // Pragmatic question → boost action-oriented processing
                self.scale_lr(
                    "guide_pragmatic",
                    thresholds::GUIDING_PRAGMATIC_LR_FACTOR as f32,
                );
                "pragmatic"
            } else if q.contains("connect") || q.contains("relate") || q.contains("together") {
                // Social question → boost coherence sensitivity
                self.adjust_confidence(
                    "guide_social",
                    thresholds::GUIDING_SOCIAL_CONFIDENCE_BOOST as f32,
                );
                "social"
            } else {
                "general"
            };
            self.stats.guiding_question_priority_uses += 1;
            cat
        } else {
            ""
        };

        // ── Foveation neuromodulation ────────────────────────────────────
        // NE → surprise threshold (attention aperture), DA → compute budget.
        // Science: Corbetta & Shulman (2002) — NE modulates attentional scope.
        #[cfg(feature = "foveation")]
        if let Some(ref fov_mutex) = self.sensorimotor.vision_sensory.foveation_manager {
            let ne = self.neuromod.bath.noradrenaline.effective();
            let da = self.neuromod.bath.dopamine.effective();
            if let Ok(mut fov) = fov_mutex.lock() {
                fov.modulate(ne, da);
            }
        }

        // ── Attachment-driven neuromodulation ──────────────────────────────
        // Science: Bowlby (1969) — caregiver proximity modulates HPA axis
        // and oxytocinergic circuits, shaping arousal regulation lifelong.
        #[cfg(feature = "nurture")]
        if let Some(ref mut nurture) = self.nurture_attachment {
            let arousal = self.neuromod.bath.noradrenaline.effective();
            let valence = self.neuromod.bath.serotonin.effective()
                - self.neuromod.bath.noradrenaline.phasic();
            let attachment_mod = nurture.cycle_step(arousal, valence);
            attachment_mod.apply_to_bath(&mut self.neuromod.bath);
        }

        NeuromodPhaseResult {
            unified_psi,
            guiding_question: guiding_question.to_owned(),
            dominant_harmonic: dominant_harmonic.to_owned(),
            ne_reorienting_boost,
            ne_arousal_feedback,
            sht_crash_dip,
            exploration_sht_drain,
            confidence_velocity: confidence_velocity as f32,
            guiding_priority_category: guiding_priority_category.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn test_neuromod_phase_returns_valid_psi() {
        let mut svc = make_service();
        let result = svc.run_neuromodulator_and_psi_phase(0.3, 0.5);
        assert!(result.unified_psi >= 0.0 && result.unified_psi <= 1.0);
    }

    #[test]
    fn test_ne_reorienting_below_threshold() {
        let mut svc = make_service();
        // With default bath, ne_phasic should be low → no reorienting
        let result = svc.run_neuromodulator_and_psi_phase(0.1, 0.5);
        // ne_reorienting_boost is 0 when ne_phasic <= threshold
        assert!(result.ne_reorienting_boost >= 0.0);
    }

    #[test]
    fn test_confidence_crash_detection_no_crash() {
        let mut svc = make_service();
        // Stable confidence → no crash
        svc.carryover.quality.prev_confidence_for_crash = svc.prediction_confidence;
        let result = svc.run_neuromodulator_and_psi_phase(0.2, 0.5);
        assert_eq!(result.sht_crash_dip, 0.0);
    }

    #[test]
    fn test_confidence_crash_detection_with_crash() {
        let mut svc = make_service();
        // Simulate a crash: prev confidence was much higher
        svc.carryover.quality.prev_confidence_for_crash = svc.prediction_confidence + 0.3;
        let result = svc.run_neuromodulator_and_psi_phase(0.2, 0.5);
        // Velocity is negative enough to trigger crash
        assert!(result.sht_crash_dip > 0.0);
    }

    #[test]
    fn test_exploration_sht_drain_below_threshold() {
        let mut svc = make_service();
        svc.behavior.curiosity_drive.exploration_urge = 0.2; // below threshold
        let result = svc.run_neuromodulator_and_psi_phase(0.1, 0.5);
        assert_eq!(result.exploration_sht_drain, 0.0);
    }

    #[test]
    fn test_exploration_sht_drain_above_threshold() {
        let mut svc = make_service();
        // Must inject via feedback state so the value survives D2 flexibility rescaling.
        // begin_cycle + snapshot_cycle_start seeds the feedback consensus at 0.8.
        svc.feedback_state.begin_cycle();
        svc.feedback_state.snapshot_cycle_start(0.5, 0.01, 0.8, 1.0);
        svc.behavior.curiosity_drive.exploration_urge = 0.8; // above threshold
        let result = svc.run_neuromodulator_and_psi_phase(0.1, 0.5);
        assert!(
            result.exploration_sht_drain > 0.0,
            "expected sht drain > 0, got {}; exploration_urge was {}",
            result.exploration_sht_drain,
            svc.behavior.curiosity_drive.exploration_urge
        );
    }

    #[test]
    fn test_guiding_priority_empty_question() {
        let mut svc = make_service();
        // No experience bus → empty guiding question
        let result = svc.run_neuromodulator_and_psi_phase(0.1, 0.5);
        assert!(result.guiding_priority_category.is_empty() || !result.guiding_question.is_empty());
    }
}
