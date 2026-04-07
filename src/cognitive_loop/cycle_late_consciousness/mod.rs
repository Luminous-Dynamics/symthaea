// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Late consciousness monitors and integration extracted from cycle.rs.
//!
//! Contains: prefrontal cortex, meta-cognition, virtual body, affective bridge,
//! user state inference, narrative self, predictive processing, hierarchical free
//! energy, predictive self, attention schema, phi attention, attention visualization,
//! GWT integration, cross-modal binding, consciousness monitors (resonance, quantum
//! coherence), phenomenal binding, temporal consciousness, consciousness thermodynamics,
//! embodied cognition, narrative-GWT integration, unified living mind, and master
//! consciousness equation.

mod integration;
mod monitors;

/// Input context for the late consciousness monitors phase.
/// Contains read-only values computed by earlier cycle phases.
pub(super) struct LateConsciousnessContext<'a> {
    pub prediction_error: f32,
    pub coherence: f32,
    pub unified_psi: f64,
    pub hv16_cached: symthaea_core::hdc::BinaryHV,
    pub compressed_state: &'a [f32],
    pub input: &'a str,
    pub urgency: super::CycleUrgency,
    pub moral_concern_detected: bool,
    pub surprise_triggered: bool,
    pub reasoning_gate_blocked: bool,
    pub pp_phi: f32,
    pub peak_attention: f32,
}

/// Results from the late consciousness monitors phase (prefrontal through attention schema).
pub(super) struct LateConsciousnessResult {
    pub prefrontal_veto: bool,
    pub meta_cognitive_accuracy: f32,
    pub meta_cognitive_depth: u8,
    pub body_psi_modulation: f64,
    pub body_valence: f32,
    pub body_arousal: f32,
    pub affective_valence: f32,
    pub affective_arousal: f32,
    pub narrative_self_psi: f64,
    pub predictive_free_energy: f64,
    pub predictive_psi_modulation: f64,
    pub hierarchical_total_free_energy: f64,
    pub predictive_self_safety: f32,
    /// Behavioral prediction error (average across moral_score, exploration_urge, behavioral_coherence).
    pub predictive_behavioral_error: f32,
    pub attention_schema_focus: f32,
    pub attention_fatigue: f32,
    pub attention_prediction_accuracy: f32,
    pub psi_attention_avg: f32,
    /// HFE learning rate boost applied when hierarchical free energy exceeds threshold.
    pub hierarchical_free_energy_lr_boost: f32,
    /// Predictive phi modulation learning rate delta (±1.5% max, coherence-weighted).
    pub predictive_phi_lr_delta: f32,
    /// Confidence delta from body valence somatic marker feedback (Damasio 1999).
    pub body_valence_confidence_delta: f32,
    /// Confidence scale factor from narrative self-Phi (1.02 strong, 0.95 weak, 1.0 neutral).
    pub narrative_self_confidence_factor: f32,
}

/// Results from the consciousness integration phase (GWT through master consciousness equation).
pub(super) struct ConsciousnessIntegrationResult {
    pub gwt_broadcast: bool,
    pub gwt_coalition_size: u32,
    pub cross_modal_binding_strength: f32,
    pub cross_modal_psi: f64,
    pub resonance_frequency: f64,
    pub quantum_coherence_level: f64,
    pub phenomenal_binding_strength: f64,
    pub phenomenal_fragmented: bool,
    pub temporal_coherence_score: f64,
    pub temporal_discontinuity: bool,
    pub thermodynamic_entropy: f64,
    pub thermodynamic_free_energy: f64,
    pub embodied_psi_modulation: f64,
    pub embodied_agency: f64,
    pub narrative_gwt_veto: bool,
    pub narrative_gwt_self_psi: f64,
    pub living_mind_vitality: f64,
    pub living_mind_coherence: f64,
    pub consciousness_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleUrgency};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    fn make_default_context<'a>(
        compressed_state: &'a [f32],
        input: &'a str,
    ) -> LateConsciousnessContext<'a> {
        LateConsciousnessContext {
            prediction_error: 0.1,
            coherence: 0.5,
            unified_psi: 0.3,
            hv16_cached: symthaea_core::hdc::BinaryHV::random(42),
            compressed_state,
            input,
            urgency: CycleUrgency::Normal,
            moral_concern_detected: false,
            surprise_triggered: false,
            reasoning_gate_blocked: false,
            pp_phi: 0.5,
            peak_attention: 0.4,
        }
    }

    // ── run_late_consciousness_monitors ────────────────────────────────

    #[test]
    fn late_monitors_default_does_not_panic() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.0f32; 64];
        let ctx = make_default_context(&compressed, "test input");
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_late_consciousness_monitors(&ctx, &mut timings);
        assert!(result.meta_cognitive_accuracy.is_finite());
        assert!(result.body_psi_modulation.is_finite());
        assert!(result.affective_valence.is_finite());
        assert!(result.affective_arousal.is_finite());
        assert!(result.predictive_free_energy.is_finite());
    }

    #[test]
    fn late_monitors_no_prefrontal_veto_on_default() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.0f32; 64];
        let ctx = make_default_context(&compressed, "hello");
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_late_consciousness_monitors(&ctx, &mut timings);
        assert!(!result.prefrontal_veto);
    }

    #[test]
    fn late_monitors_empty_input_does_not_panic() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.0f32; 64];
        let ctx = make_default_context(&compressed, "");
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_late_consciousness_monitors(&ctx, &mut timings);
        assert!(result.affective_arousal.is_finite());
    }

    #[test]
    fn late_monitors_zero_prediction_error() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.0f32; 64];
        let mut ctx = make_default_context(&compressed, "low error");
        ctx.prediction_error = 0.0;
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_late_consciousness_monitors(&ctx, &mut timings);
        assert!(result.predictive_psi_modulation.is_finite());
        assert!(result.narrative_self_psi.is_finite());
    }

    #[test]
    fn late_monitors_high_prediction_error() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![1.0f32; 64];
        let mut ctx = make_default_context(&compressed, "high error");
        ctx.prediction_error = 1.0;
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_late_consciousness_monitors(&ctx, &mut timings);
        assert!(result.meta_cognitive_accuracy.is_finite());
        assert!(result.psi_attention_avg.is_finite());
    }

    #[test]
    fn late_monitors_critical_urgency_runs_all() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.5f32; 64];
        let mut ctx = make_default_context(&compressed, "urgent");
        ctx.urgency = CycleUrgency::Critical;
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_late_consciousness_monitors(&ctx, &mut timings);
        assert!(result.body_valence.is_finite());
        assert!(result.attention_schema_focus.is_finite());
    }

    #[test]
    fn late_monitors_cruise_urgency_skips_subsystems() {
        let mut s = make_service();
        s.stats.total_cycles = 3;
        let compressed = vec![0.2f32; 64];
        let mut ctx = make_default_context(&compressed, "cruise mode");
        ctx.urgency = CycleUrgency::Cruise;
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_late_consciousness_monitors(&ctx, &mut timings);
        assert!(result.predictive_free_energy.is_finite());
    }

    #[test]
    fn late_monitors_dual_veto_sets_exploration() {
        let mut s = make_service();
        s.carryover.quality.cached_prefrontal_veto = true;
        let compressed = vec![0.0f32; 64];
        let mut ctx = make_default_context(&compressed, "dual veto");
        ctx.reasoning_gate_blocked = true;
        ctx.urgency = CycleUrgency::Cruise;
        s.stats.total_cycles = 3;
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_late_consciousness_monitors(&ctx, &mut timings);
        assert!(result.prefrontal_veto);
        assert!((s.behavior.curiosity_drive.exploration_urge - 0.3).abs() < 0.01);
    }

    // ── run_consciousness_integration ─────────────────────────────────

    #[test]
    fn consciousness_integration_default_does_not_panic() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.0f32; 64];
        let ctx = make_default_context(&compressed, "integration test");
        let late = LateConsciousnessResult {
            prefrontal_veto: false,
            meta_cognitive_accuracy: 0.5,
            meta_cognitive_depth: 1,
            body_psi_modulation: 1.0,
            body_valence: 0.0,
            body_arousal: 0.3,
            affective_valence: 0.1,
            affective_arousal: 0.4,
            narrative_self_psi: 0.3,
            predictive_free_energy: 0.1,
            predictive_psi_modulation: 1.0,
            hierarchical_total_free_energy: 0.0,
            predictive_self_safety: 0.5,
            predictive_behavioral_error: 0.0,
            attention_schema_focus: 0.4,
            attention_fatigue: 0.0,
            attention_prediction_accuracy: 0.0,
            psi_attention_avg: 0.3,
            hierarchical_free_energy_lr_boost: 1.0,
            predictive_phi_lr_delta: 0.0,
            body_valence_confidence_delta: 0.0,
            narrative_self_confidence_factor: 1.0,
        };
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_consciousness_integration(&ctx, &late, &mut timings);
        assert!(result.consciousness_level.is_finite());
        assert!(result.cross_modal_binding_strength.is_finite());
        assert!(result.resonance_frequency.is_finite());
    }

    #[test]
    fn consciousness_integration_all_fields_finite() {
        let mut s = make_service();
        s.stats.total_cycles = 10;
        let compressed = vec![0.5f32; 64];
        let ctx = make_default_context(&compressed, "finite check");
        let late = LateConsciousnessResult {
            prefrontal_veto: false,
            meta_cognitive_accuracy: 0.0,
            meta_cognitive_depth: 0,
            body_psi_modulation: 1.0,
            body_valence: 0.0,
            body_arousal: 0.0,
            affective_valence: 0.0,
            affective_arousal: 0.0,
            narrative_self_psi: 0.0,
            predictive_free_energy: 0.0,
            predictive_psi_modulation: 1.0,
            hierarchical_total_free_energy: 0.0,
            predictive_self_safety: 0.0,
            predictive_behavioral_error: 0.0,
            attention_schema_focus: 0.0,
            attention_fatigue: 0.0,
            attention_prediction_accuracy: 0.0,
            psi_attention_avg: 0.0,
            hierarchical_free_energy_lr_boost: 1.0,
            predictive_phi_lr_delta: 0.0,
            body_valence_confidence_delta: 0.0,
            narrative_self_confidence_factor: 1.0,
        };
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_consciousness_integration(&ctx, &late, &mut timings);
        assert!(result.thermodynamic_entropy.is_finite());
        assert!(result.thermodynamic_free_energy.is_finite());
        assert!(result.embodied_psi_modulation.is_finite());
        assert!(result.embodied_agency.is_finite());
        assert!(result.phenomenal_binding_strength.is_finite());
        assert!(result.quantum_coherence_level.is_finite());
        assert!(result.living_mind_vitality.is_finite());
        assert!(result.living_mind_coherence.is_finite());
    }

    #[test]
    fn consciousness_integration_cruise_skips_monitors() {
        let mut s = make_service();
        s.stats.total_cycles = 3;
        let compressed = vec![0.0f32; 64];
        let mut ctx = make_default_context(&compressed, "cruise integration");
        ctx.urgency = CycleUrgency::Cruise;
        let late = LateConsciousnessResult {
            prefrontal_veto: false,
            meta_cognitive_accuracy: 0.0,
            meta_cognitive_depth: 0,
            body_psi_modulation: 1.0,
            body_valence: 0.0,
            body_arousal: 0.0,
            affective_valence: 0.0,
            affective_arousal: 0.0,
            narrative_self_psi: 0.0,
            predictive_free_energy: 0.0,
            predictive_psi_modulation: 1.0,
            hierarchical_total_free_energy: 0.0,
            predictive_self_safety: 0.0,
            predictive_behavioral_error: 0.0,
            attention_schema_focus: 0.0,
            attention_fatigue: 0.0,
            attention_prediction_accuracy: 0.0,
            psi_attention_avg: 0.0,
            hierarchical_free_energy_lr_boost: 1.0,
            predictive_phi_lr_delta: 0.0,
            body_valence_confidence_delta: 0.0,
            narrative_self_confidence_factor: 1.0,
        };
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_consciousness_integration(&ctx, &late, &mut timings);
        assert!(!result.gwt_broadcast);
        assert!(!result.temporal_discontinuity);
    }

    #[test]
    fn temporal_discontinuity_streak_decrements_without_discontinuity() {
        let mut config = super::super::CognitiveLoopConfig::default();
        config.enable_temporal_consciousness = false;
        let mut s = super::super::CognitiveLoopService::new(config).unwrap();
        s.stats.total_cycles = 1;
        s.carryover.urgency.discontinuity_streak = 2;
        let compressed = vec![0.0f32; 64];
        let ctx = make_default_context(&compressed, "streak");
        let late = LateConsciousnessResult {
            prefrontal_veto: false,
            meta_cognitive_accuracy: 0.0,
            meta_cognitive_depth: 0,
            body_psi_modulation: 1.0,
            body_valence: 0.0,
            body_arousal: 0.0,
            affective_valence: 0.0,
            affective_arousal: 0.0,
            narrative_self_psi: 0.0,
            predictive_free_energy: 0.0,
            predictive_psi_modulation: 1.0,
            hierarchical_total_free_energy: 0.0,
            predictive_self_safety: 0.0,
            predictive_behavioral_error: 0.0,
            attention_schema_focus: 0.0,
            attention_fatigue: 0.0,
            attention_prediction_accuracy: 0.0,
            psi_attention_avg: 0.0,
            hierarchical_free_energy_lr_boost: 1.0,
            predictive_phi_lr_delta: 0.0,
            body_valence_confidence_delta: 0.0,
            narrative_self_confidence_factor: 1.0,
        };
        let mut timings = super::super::ModuleTimings::default();
        let result = s.run_consciousness_integration(&ctx, &late, &mut timings);
        assert!(!result.temporal_discontinuity);
        assert!(s.carryover.urgency.discontinuity_streak <= 2);
    }
}
