#![allow(clippy::field_reassign_with_default)]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use super::*;

// ═══════════════════════════════════════════════════════════════════════
// CycleUrgency
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn urgency_default_is_normal() {
    assert_eq!(CycleUrgency::default(), CycleUrgency::Normal);
}

#[test]
fn urgency_from_arousal_high() {
    assert_eq!(CycleUrgency::from_arousal(0.9), CycleUrgency::Critical);
}

#[test]
fn urgency_from_arousal_mid() {
    assert_eq!(CycleUrgency::from_arousal(0.5), CycleUrgency::Normal);
}

#[test]
fn urgency_from_arousal_low() {
    assert_eq!(CycleUrgency::from_arousal(0.1), CycleUrgency::Cruise);
}

#[test]
fn urgency_from_state_surprise_triggers_critical() {
    let u = CycleUrgency::from_state(0.01, 0.05, true, 100);
    assert_eq!(u, CycleUrgency::Critical);
}

#[test]
fn urgency_from_state_high_error_triggers_critical() {
    let u = CycleUrgency::from_state(0.2, 0.05, false, 100);
    assert_eq!(u, CycleUrgency::Critical);
}

#[test]
fn urgency_from_state_moderate_error_is_normal() {
    let u = CycleUrgency::from_state(0.06, 0.05, false, 5);
    assert_eq!(u, CycleUrgency::Normal);
}

#[test]
fn urgency_from_state_low_error_long_streak_is_cruise() {
    let u = CycleUrgency::from_state(0.01, 0.05, false, 20);
    assert_eq!(u, CycleUrgency::Cruise);
}

#[test]
fn urgency_should_run_cycle_zero() {
    // All urgencies should run at cycle 0
    for urgency in [
        CycleUrgency::Critical,
        CycleUrgency::Normal,
        CycleUrgency::Cruise,
    ] {
        assert!(
            urgency.should_run(0, 3, 5, 20),
            "{:?} should run at cycle 0",
            urgency
        );
    }
}

#[test]
fn urgency_should_run_interval_modulo() {
    let u = CycleUrgency::Normal;
    assert!(u.should_run(10, 3, 5, 20)); // cycle 10 % 5 == 0
    assert!(!u.should_run(11, 3, 5, 20)); // cycle 11 % 5 != 0
}

#[test]
fn urgency_should_run_zero_interval_always_runs() {
    let u = CycleUrgency::Critical;
    assert!(u.should_run(42, 0, 5, 20)); // critical_interval=0 → always
}

#[test]
fn urgency_consciousness_monitors() {
    assert!(CycleUrgency::Critical.run_consciousness_monitors());
    assert!(CycleUrgency::Normal.run_consciousness_monitors());
    assert!(!CycleUrgency::Cruise.run_consciousness_monitors());
}

// ═══════════════════════════════════════════════════════════════════════
// Carryover struct defaults
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn consciousness_cache_default_modulations_neutral() {
    let cc = ConsciousnessCache::default();
    assert_eq!(cc.predictive_phi_modulation, 1.0);
    assert_eq!(cc.body_phi_modulation, 1.0);
    assert_eq!(cc.embodied_phi_modulation, 1.0);
    assert_eq!(cc.cross_modal_psi, 0.0);
    assert!(cc.last_sigma.is_none());
}

#[test]
fn urgency_state_default() {
    let us = UrgencyState::default();
    assert_eq!(us.urgency, CycleUrgency::Normal);
    assert_eq!(us.consecutive_low_error, 0);
    assert_eq!(us.arousal_trap_counter, 0);
    assert!(!us.anomaly_was_active);
    assert_eq!(us.mode_confidence, 1.0);
}

#[test]
fn learning_state_default() {
    let ls = LearningState::default();
    assert_eq!(ls.prediction_confidence, 0.5);
    assert_eq!(ls.mce_lr_boost, 0.0);
    assert_eq!(ls.adaptive_threshold_scale, 1.0);
    assert_eq!(ls.subsystem_lr_factor, 1.0);
}

#[test]
fn quality_metrics_default() {
    let qm = QualityMetrics::default();
    assert_eq!(qm.causal_chain_count, 0);
    assert_eq!(qm.temporal_continuity, 0.0);
    assert_eq!(qm.last_epistemic_confidence, 0.5);
    assert_eq!(qm.last_coherence, 0.5);
    assert!(!qm.narrative_veto_active);
}

// ═══════════════════════════════════════════════════════════════════════
// CycleMetadata
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn cycle_metadata_default_sensible() {
    let md = CycleMetadata::default();
    assert!(!md.surprise_triggered);
    assert!(md.consciousness.consciousness_level.is_finite());
}

#[test]
fn cycle_metadata_serde_roundtrip() {
    let mut md = CycleMetadata::default();
    md.consciousness.consciousness_level = 0.75;
    md.embodied.body_phi_modulation = 1.1;
    md.temporal.temporal_coherence_score = 0.8;
    md.structural.structural_micro_phi = 0.3;
    let json = serde_json::to_string(&md).expect("serialize CycleMetadata");
    let restored: CycleMetadata = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.surprise_triggered, md.surprise_triggered);
    // Verify sub-struct fields survived the roundtrip
    assert!((restored.consciousness.consciousness_level - 0.75).abs() < 1e-10);
    assert!((restored.embodied.body_phi_modulation - 1.1).abs() < 1e-10);
    assert!((restored.temporal.temporal_coherence_score - 0.8).abs() < 1e-10);
    assert!((restored.structural.structural_micro_phi - 0.3).abs() < 1e-10);
}

#[test]
fn cycle_metadata_flatten_produces_flat_json() {
    let mut md = CycleMetadata::default();
    md.consciousness.consciousness_level = 0.42;
    md.embodied.body_phi_modulation = 1.05;
    md.temporal.temporal_coherence_score = 0.6;
    md.structural.structural_micro_phi = 0.2;
    let json = serde_json::to_string(&md).expect("serialize");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse");
    let obj = value.as_object().expect("should be object");
    // Flattened fields appear as top-level keys (not nested)
    assert!(
        obj.contains_key("consciousness_level"),
        "consciousness_level should be flat key"
    );
    assert!(
        obj.contains_key("body_phi_modulation"),
        "body_phi_modulation should be flat key"
    );
    assert!(
        obj.contains_key("temporal_coherence_score"),
        "temporal_coherence_score should be flat key"
    );
    assert!(
        obj.contains_key("structural_micro_phi"),
        "structural_micro_phi should be flat key"
    );
    // No nested "consciousness", "embodied", "temporal", or "structural" sub-objects
    // (they're flattened, not nested)
    // Note: "consciousness" key should NOT exist as a nested object
    if let Some(v) = obj.get("consciousness") {
        assert!(
            !v.is_object(),
            "consciousness should not be a nested object"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MoralJudgmentSummary
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn moral_summary_default() {
    let mj = MoralJudgmentSummary::default();
    assert_eq!(mj.verdict, "Neutral");
    assert_eq!(mj.deontological_verdict, "Permissible");
    assert!(!mj.consent_violation);
    assert_eq!(mj.moral_score, 0.0);
    assert_eq!(mj.confidence, 0.0);
    assert!(mj.violations.is_empty());
    assert!(mj.satisfactions.is_empty());
}

#[test]
fn moral_summary_serde_roundtrip() {
    let mj = MoralJudgmentSummary {
        input: "test action".into(),
        verdict: "Good".into(),
        deontological_verdict: "Permissible".into(),
        violations: vec![],
        satisfactions: vec!["honesty".into()],
        consent_violation: false,
        moral_score: 0.7,
        confidence: 0.9,
    };
    let json = serde_json::to_string(&mj).unwrap();
    let restored: MoralJudgmentSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.verdict, "Good");
    assert_eq!(restored.moral_score, 0.7);
    assert_eq!(restored.satisfactions.len(), 1);
}

// ═══════════════════════════════════════════════════════════════════════
// AdaptiveBehavior
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adaptive_behavior_default_neutral() {
    let ab = AdaptiveBehavior::default();
    assert_eq!(ab.learning_rate_multiplier, 1.0);
    assert_eq!(ab.speech_rate_multiplier, 1.0);
    assert_eq!(ab.pause_multiplier, 1.0);
    assert_eq!(ab.action_hint, ActionHint::Continue);
    assert!(!ab.pause_learning);
}

#[test]
fn adaptive_behavior_from_focused_pattern() {
    use crate::dynamics::temporal_signatures::ConsciousnessPattern;
    let ab =
        AdaptiveBehavior::from_consciousness_state(ConsciousnessPattern::Focused, 0.8, 0.7, 0.6);
    assert!(ab.learning_rate_multiplier > 1.0, "focused should boost LR");
    assert!(
        ab.speech_rate_multiplier > 1.0,
        "focused should speed up speech"
    );
    assert_eq!(ab.action_hint, ActionHint::SpeedUp);
}

#[test]
fn adaptive_behavior_from_uncertain_pattern() {
    use crate::dynamics::temporal_signatures::ConsciousnessPattern;
    let ab =
        AdaptiveBehavior::from_consciousness_state(ConsciousnessPattern::Uncertain, 0.3, 0.3, 0.2);
    assert!(
        ab.learning_rate_multiplier < 1.0,
        "uncertain should reduce LR"
    );
    assert!(ab.confidence < 0.3, "uncertain should have low confidence");
    assert_eq!(ab.action_hint, ActionHint::SeekInput);
}

#[test]
fn adaptive_behavior_transitioning_pauses_learning() {
    use crate::dynamics::temporal_signatures::ConsciousnessPattern;
    let ab = AdaptiveBehavior::from_consciousness_state(
        ConsciousnessPattern::Transitioning,
        0.5,
        0.5,
        0.5,
    );
    assert!(ab.pause_learning);
    assert_eq!(ab.action_hint, ActionHint::Stabilize);
}

#[test]
fn adaptive_behavior_effective_lr_paused() {
    let mut ab = AdaptiveBehavior::default();
    ab.pause_learning = true;
    assert_eq!(ab.effective_learning_rate(0.01), 0.0);
}

#[test]
fn adaptive_behavior_effective_lr_normal() {
    let ab = AdaptiveBehavior::default();
    let lr = ab.effective_learning_rate(0.01);
    assert!((lr - 0.01).abs() < 1e-6, "default multiplier is 1.0");
}

#[test]
fn adaptive_behavior_should_seek_input() {
    let mut ab = AdaptiveBehavior::default();
    ab.confidence = 0.1; // Very low
    assert!(ab.should_seek_input());
}

#[test]
fn adaptive_behavior_is_confident() {
    let mut ab = AdaptiveBehavior::default();
    ab.confidence = 0.8;
    assert!(ab.is_confident());
    ab.pause_learning = true;
    assert!(!ab.is_confident()); // paused learning = not confident
}

#[test]
fn adaptive_behavior_description_coverage() {
    // Ensure all variants return non-empty strings
    for hint in [
        ActionHint::Continue,
        ActionHint::SlowDown,
        ActionHint::SpeedUp,
        ActionHint::Stabilize,
        ActionHint::Explore,
        ActionHint::SeekInput,
        ActionHint::Inhibit,
    ] {
        let mut ab = AdaptiveBehavior::default();
        ab.action_hint = hint;
        assert!(!ab.description().is_empty());
    }
}

#[test]
fn adaptive_behavior_confidence_bounded() {
    use crate::dynamics::temporal_signatures::ConsciousnessPattern;
    // All patterns should produce confidence in [0, 1]
    for pattern in [
        ConsciousnessPattern::Focused,
        ConsciousnessPattern::Contemplative,
        ConsciousnessPattern::Excited,
        ConsciousnessPattern::Exploratory,
        ConsciousnessPattern::Resting,
        ConsciousnessPattern::Transitioning,
        ConsciousnessPattern::Uncertain,
    ] {
        let ab = AdaptiveBehavior::from_consciousness_state(pattern, 1.0, 1.0, 1.0);
        assert!(
            ab.confidence >= 0.0 && ab.confidence <= 1.0,
            "{:?}: confidence {} out of bounds",
            pattern,
            ab.confidence
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ActionHint serde
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn action_hint_serde_roundtrip() {
    for hint in [
        ActionHint::Continue,
        ActionHint::SlowDown,
        ActionHint::SpeedUp,
        ActionHint::Stabilize,
        ActionHint::Explore,
        ActionHint::SeekInput,
        ActionHint::Inhibit,
    ] {
        let json = serde_json::to_string(&hint).unwrap();
        let restored: ActionHint = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, hint);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ModuleTimings
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn module_timings_default_zeros() {
    let mt = ModuleTimings::default();
    assert_eq!(mt.core_hdc_encode, 0);
    assert_eq!(mt.core_cfc_step, 0);
    assert_eq!(mt.core_predict, 0);
    assert_eq!(mt.core_training, 0);
}

#[test]
fn module_timings_serde_roundtrip() {
    let mt = ModuleTimings {
        core_hdc_encode: 100,
        core_cfc_step: 200,
        core_predict: 50,
        core_training: 1000,
        ..ModuleTimings::default()
    };
    let json = serde_json::to_string(&mt).unwrap();
    let restored: ModuleTimings = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.core_hdc_encode, 100);
    assert_eq!(restored.core_training, 1000);
}

// ═══════════════════════════════════════════════════════════════════
// Improvement 1+3: Structural Phi + Weight fields in CycleMetadata
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_structural_phi_in_cycle_metadata_defaults_zero() {
    let md = CycleMetadata::default();
    assert_eq!(md.structural.structural_micro_phi, 0.0);
    assert_eq!(md.structural.structural_meso_phi, 0.0);
    assert_eq!(md.structural.structural_macro_phi, 0.0);
    assert_eq!(md.structural.structural_bottleneck, 0.0);
    assert_eq!(md.structural.structural_emergence_ratio, 0.0);
    assert_eq!(md.structural.structural_num_clusters, 0);
    assert_eq!(md.consciousness.consciousness_weights, [0.0; 4]);
    assert_eq!(md.consciousness.consciousness_weight_variance, 0.0);
}
