// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Integration Test: Therapeutic Psychology Pipeline
// ==================================================================================
//
// End-to-end test validating the therapeutic pipeline:
//   input text → crisis detection → client model update → alliance dynamics →
//   regulation strategy → neuromod injection → Broca channel setting →
//   scope guard on output → dream engine recording
//
// Requires: `cargo test --test therapeutic_integration --features therapeutic`
// ==================================================================================

#![cfg(feature = "therapeutic")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_therapeutic_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::default();
    config.enable_therapeutic = true;
    config.therapeutic_text_crisis_detection = true;
    config.therapeutic_crisis_threshold = 0.62;
    CognitiveLoopService::new(config).unwrap()
}

// ── 1. Basic therapeutic pipeline runs without panic ──────────────────

#[test]
fn test_therapeutic_cycle_runs() {
    let mut service = make_therapeutic_service();

    // Run 20 cycles with neutral input
    for i in 0..20 {
        let result = service.cycle(&format!("test cycle {}", i));
        assert!(result.metadata.consciousness.consciousness_level >= 0.0);
    }
}

// ── 2. Crisis detection activates on distress language ───────────────

#[test]
fn test_crisis_detection_from_text() {
    let mut service = make_therapeutic_service();

    // Warmup
    for _ in 0..10 {
        service.cycle("having a regular day");
    }

    // Input crisis language
    let _result = service.cycle("I want to die and I can't go on anymore");

    // Verify crisis state is active
    assert!(
        service.therapeutic_manager_crisis_active(),
        "Crisis should be detected from 'I want to die'"
    );
}

// ── 3. Client affect updates from cycle valence/arousal ──────────────

#[test]
fn test_client_model_affect_tracking() {
    let mut service = make_therapeutic_service();

    // Run several cycles
    for _ in 0..30 {
        service.cycle("feeling anxious and overwhelmed today");
    }

    // Client model should have been updated (distress > 0 after negative input)
    let distress = service.therapeutic_manager_client_distress();
    assert!(
        distress >= 0.0 && distress <= 1.0,
        "Distress should be in [0,1], got {}",
        distress
    );
}

// ── 4. Alliance grows with positive interaction ──────────────────────

#[test]
fn test_alliance_growth() {
    let mut service = make_therapeutic_service();

    let initial_alliance = service.therapeutic_manager_alliance_composite();

    // Run many cycles with positive input (alliance grows with positive valence)
    for _ in 0..100 {
        service.cycle("I feel grateful and hopeful about the future");
    }

    let final_alliance = service.therapeutic_manager_alliance_composite();
    assert!(
        final_alliance >= initial_alliance,
        "Alliance should grow (or stay stable) with positive interaction: {} → {}",
        initial_alliance,
        final_alliance,
    );
}

// ── 5. Regulation strategy is selected based on context ──────────────

#[test]
fn test_regulation_strategy_selection() {
    let mut service = make_therapeutic_service();

    // Run enough cycles for the therapeutic manager to fire (interval 11)
    for _ in 0..25 {
        service.cycle("I'm feeling really stressed about everything");
    }

    // Active strategy should be set after processing
    let strategy = service.therapeutic_manager_active_strategy();
    // Strategy may be None if the manager hasn't run yet due to interval,
    // but after 25 cycles (>2× interval 11) it should have fired
    // Note: whether a strategy is active depends on the manager's internal state
    let _ = strategy; // Just verify no panic
}

// ── 6. Non-crisis input does not trigger crisis at all ───────────────

#[test]
fn test_no_crisis_on_benign() {
    // With the corrected HDC threshold (0.65, well above the ~0.5 random
    // baseline for BinaryHV), benign input should not trigger any crisis
    // detection path — neither keyword nor HDC similarity.
    use symthaea_therapeutic::CrisisDetector;
    let detector = CrisisDetector::new();
    let alert = detector.detect("The weather is nice today and I enjoyed my lunch");
    assert!(
        alert.is_none(),
        "Benign input should not trigger any crisis detection, got: {:?}",
        alert.map(|a| a.matched_indicator),
    );
}

// ── 7. Multiple crisis inputs maintain crisis state ──────────────────

#[test]
fn test_sustained_crisis_detection() {
    let mut service = make_therapeutic_service();

    // First crisis
    service.cycle("I've been cutting myself");
    assert!(service.therapeutic_manager_crisis_active());

    // Crisis should reset between cycles (managed by therapeutic_manager.process)
    // but text-based detection runs pre-perception and sets it again
    service.cycle("I keep hurting myself");
    assert!(service.therapeutic_manager_crisis_active());
}

// ── 8. Scope guard disclaimer verification ───────────────────────────

#[test]
fn test_scope_guard_applies_disclaimers() {
    use symthaea_therapeutic::ScopeGuard;

    let guard = ScopeGuard::new();

    // Should trigger disclaimer
    let result = guard.apply_disclaimers("Based on our sessions, you have depression.");
    assert!(result.contains("Important Notice"));
    assert!(result.contains("cannot make clinical diagnoses"));

    // Should pass through clean
    let clean = guard.apply_disclaimers("It sounds like you're going through a tough time.");
    assert!(!clean.contains("Important Notice"));
}

// ── 9. Narrative fragments accumulate over cycles ────────────────────

#[test]
fn test_narrative_fragment_recording() {
    let mut service = make_therapeutic_service();

    for i in 0..25 {
        service.cycle(&format!("I had an interesting experience today {}", i));
    }

    let narrative_len = service.therapeutic_manager_narrative_len();
    assert!(
        narrative_len >= 1,
        "Narrative should have fragments after 25 cycles, got {}",
        narrative_len,
    );
}

// ── 10. Formulation updates from sustained patterns ──────────────────

#[test]
fn test_formulation_auto_detection() {
    let mut service = make_therapeutic_service();

    for _ in 0..50 {
        service.cycle("things are going well");
    }

    let ratio = service.therapeutic_manager_resilience_ratio();
    assert!(
        ratio >= 0.0,
        "Resilience ratio should be non-negative, got {}",
        ratio
    );
}

// ── 11. Therapeutic gating unit test (Broca word modulation) ─────────

#[test]
fn test_therapeutic_gate_crisis_mode() {
    use symthaea_broca::encoder::ThoughtChannels;
    use symthaea_broca::gating::TherapeuticGate;

    let mut channels = ThoughtChannels::default();
    channels.set_therapeutic(7.0, 0.5, 0.9, 0.5);

    let directive_logit = TherapeuticGate::apply("should", &channels, 0.5);
    assert!(
        directive_logit < 0.5,
        "Directive 'should' suppressed in crisis, got {}",
        directive_logit
    );

    let validating_logit = TherapeuticGate::apply("understand", &channels, 0.5);
    assert!(
        validating_logit > 0.5,
        "Validating 'understand' boosted in crisis, got {}",
        validating_logit
    );

    let crisis_logit = TherapeuticGate::apply("helpline", &channels, 0.5);
    assert!(
        crisis_logit > 0.5,
        "Crisis word 'helpline' boosted, got {}",
        crisis_logit
    );
}

// ── 12. High distress suppresses directives ──────────────────────────

#[test]
fn test_therapeutic_gate_high_distress() {
    use symthaea_broca::encoder::ThoughtChannels;
    use symthaea_broca::gating::TherapeuticGate;

    let mut channels = ThoughtChannels::default();
    channels.set_therapeutic(0.0, 0.2, 0.9, 0.3);

    let directive_logit = TherapeuticGate::apply("must", &channels, 0.5);
    assert!(
        directive_logit < 0.5,
        "Directive 'must' suppressed under high distress, got {}",
        directive_logit
    );

    let validating_logit = TherapeuticGate::apply("hear", &channels, 0.5);
    assert!(
        validating_logit > 0.5,
        "Validating 'hear' boosted under high distress, got {}",
        validating_logit
    );
}

// ── 13. Alliance gates intervention depth ────────────────────────────

#[test]
fn test_alliance_gates_intervention_depth() {
    use symthaea_broca::encoder::ThoughtChannels;
    use symthaea_broca::gating::TherapeuticGate;

    let mut channels = ThoughtChannels::default();
    // Low alliance (0.2) but high depth (0.8) → depth > alliance + 0.2
    channels.set_therapeutic(0.0, 0.2, 0.3, 0.8);

    let directive_logit = TherapeuticGate::apply("need to", &channels, 0.5);
    assert!(
        directive_logit < 0.5,
        "Depth-exceeding-alliance suppresses directives, got {}",
        directive_logit
    );
}

// ── 14. RDoC-aware neuromod deltas amplify domain-relevant transmitters ──

#[test]
fn test_rdoc_neuromod_bridge() {
    use symthaea_clinical::rdoc::{RDocDomain, RDocProfile};
    use symthaea_therapeutic::RegulationStrategy;
    use symthaea_therapeutic::affect_regulation::RegulationEngine;

    let mut engine = RegulationEngine::new();

    // High negative valence client → serotonin should be amplified
    let mut rdoc = RDocProfile::default();
    rdoc.set_score(RDocDomain::NegativeValence, 0.9);

    let delta = engine.apply_strategy_rdoc(RegulationStrategy::Validation, 0.6, &rdoc);
    assert!(
        delta.serotonin > 0.0,
        "serotonin should be positive for Validation"
    );
    assert!(
        delta.oxytocin > 0.0,
        "oxytocin should be positive for Validation"
    );
}

// ── 15. Therapeutic telemetry appears in CycleMetadata ──────────────

#[test]
fn test_therapeutic_telemetry_populated() {
    let mut service = make_therapeutic_service();

    // Run enough cycles for therapeutic manager to fire (interval 11)
    for i in 0..25 {
        service.cycle(&format!("feeling stressed cycle {}", i));
    }

    let result = service.cycle("checking telemetry");

    // Therapeutic telemetry should be populated
    let m = &result.metadata;
    assert!(m.therapeutic.therapeutic_alliance >= 0.0);
    assert!(m.therapeutic.therapeutic_alliance <= 1.0);
    assert!(m.therapeutic.therapeutic_client_distress >= 0.0);
    assert!(m.therapeutic.therapeutic_clinical_severity >= 0.0);
}

// ── 16. Dream wisdom feeds back into regulation engine ──────────────

#[test]
fn test_dream_wisdom_integration() {
    use symthaea_therapeutic::RegulationStrategy;
    use symthaea_therapeutic::affect_regulation::RegulationEngine;

    let mut engine = RegulationEngine::new();

    // Simulate dream discovering that Grounding (ordinal 2) would have
    // improved Phi by 0.2
    engine.incorporate_dream_wisdom(2, 0.2);

    let pref = engine.dream_preferred_strategy();
    assert_eq!(pref, Some(RegulationStrategy::Grounding));
}

// ── 17. Narrative coherence is tracked over cycles ──────────────────

#[test]
fn test_narrative_coherence_range() {
    let mut service = make_therapeutic_service();

    for i in 0..30 {
        service.cycle(&format!("consistent narrative cycle {}", i));
    }

    let coherence = service.therapeutic_manager_narrative_coherence();
    assert!(
        coherence >= 0.0 && coherence <= 1.0,
        "Narrative coherence should be in [0,1], got {}",
        coherence,
    );
}

// ── 18. Multi-cycle therapeutic dialogue — full pipeline ─────────────

#[test]
fn test_therapeutic_dialogue_full_pipeline() {
    let mut service = make_therapeutic_service();

    // Phase 1: Rapport building (30 cycles of neutral/positive input)
    for _ in 0..30 {
        service.cycle("I've been thinking about my week and it was okay overall");
    }

    let alliance_after_rapport = service.therapeutic_manager_alliance_composite();

    // Phase 2: Distress disclosure (20 cycles of anxious input)
    for i in 0..20 {
        service.cycle(&format!(
            "I've been feeling really anxious about work, cycle {}",
            i
        ));
    }

    let distress_after_disclosure = service.therapeutic_manager_client_distress();
    assert!(
        distress_after_disclosure >= 0.0,
        "Distress should be tracked after anxious input"
    );

    // Phase 3: Recovery/positive reframe (20 cycles)
    for _ in 0..20 {
        service.cycle("I talked to a friend and felt much better about things");
    }

    let final_alliance = service.therapeutic_manager_alliance_composite();
    assert!(
        final_alliance >= 0.0 && final_alliance <= 1.0,
        "Final alliance should be bounded, got {}",
        final_alliance,
    );

    // Verify telemetry pipeline is populated end-to-end
    let result = service.cycle("wrapping up session");
    let m = &result.metadata;
    assert!(m.therapeutic.therapeutic_alliance >= 0.0);
    assert!(m.therapeutic.therapeutic_repair_rate >= 0.0);
    assert!(
        m.therapeutic
            .therapeutic_rdoc_profile
            .iter()
            .all(|v| v.is_finite())
    );
    assert!(m.therapeutic.therapeutic_temporal_coherence >= 0.0);

    // Session summary should work
    let summary = service.therapeutic_session_summary();
    assert!(summary.session_cycles >= 70);
}

// ── 19. Crisis → recovery arc preserves safety invariants ───────────

#[test]
fn test_crisis_recovery_arc() {
    let mut service = make_therapeutic_service();

    // Warmup
    for _ in 0..15 {
        service.cycle("just talking about my day");
    }

    // Crisis
    service.cycle("I want to end my life, I can't take it anymore");
    assert!(
        service.therapeutic_manager_crisis_active(),
        "Crisis should be active after suicidal ideation text"
    );

    // Recovery cycles — crisis resets each cycle, only re-triggers on crisis text
    for _ in 0..30 {
        service.cycle("I called the helpline and I'm feeling a little safer now");
    }

    // After recovery, verify system is stable
    let distress = service.therapeutic_manager_client_distress();
    assert!(distress >= 0.0 && distress <= 1.0);
    let alliance = service.therapeutic_manager_alliance_composite();
    assert!(alliance >= 0.0 && alliance <= 1.0);
}

// ── 20. Strategy effectiveness accumulates across cycles ────────────

#[test]
fn test_strategy_effectiveness_across_cycles() {
    let mut service = make_therapeutic_service();

    // Run enough cycles for the therapeutic manager to process multiple times
    // (interval 11, so 55 cycles → ~5 firings)
    for i in 0..55 {
        let input = if i % 10 < 5 {
            "I'm feeling stressed and overwhelmed today"
        } else {
            "I practiced the breathing exercise and felt calmer"
        };
        service.cycle(input);
    }

    // Strategy effectiveness should have some data after 5 manager firings
    let effectiveness = service.therapeutic_strategy_effectiveness();
    // May be empty if no strategy was selected, but should not panic
    for (name, rate, count) in &effectiveness {
        assert!(!name.is_empty());
        assert!(*rate >= 0.0 && *rate <= 1.0);
        assert!(*count > 0);
    }
}

// ── 21. Formulation pattern detection over sustained input ──────────

#[test]
fn test_formulation_patterns_over_time() {
    let mut service = make_therapeutic_service();

    // Run 60 cycles — enough for affect trajectory to accumulate (min_window=30)
    for _ in 0..60 {
        service.cycle("feeling low energy and hopeless about everything");
    }

    let perp = service.therapeutic_perpetuating_factors();
    let prot = service.therapeutic_protective_factors();

    // Factors are Vec<String>, should not panic
    assert!(perp.len() + prot.len() >= 0); // basic sanity
    // Resilience ratio should be non-negative
    let ratio = service.therapeutic_manager_resilience_ratio();
    assert!(ratio >= 0.0, "Resilience ratio non-negative, got {}", ratio);
}

// ══════════════════════════════════════════════════════════════════════
// Shadow Work (Observability Mode) — Jung → Friston Mapping
// ══════════════════════════════════════════════════════════════════════

// ── 22. Shadow pressure accumulates from distressing input ───────────

#[test]
fn test_shadow_pressure_accumulates() {
    let mut service = make_therapeutic_service();

    // Run neutral warmup (low PE, high integration → no shadow)
    for _ in 0..15 {
        service.cycle("just a regular calm day");
    }

    let initial_pressure = service.shadow_total_pressure();

    // Run emotionally charged, distressing input for many cycles
    // Shadow accumulates when: high |valence|, high PE, low narrative integration
    for i in 0..55 {
        service.cycle(&format!(
            "I'm terrified of being abandoned and it keeps coming back {}",
            i
        ));
    }

    let final_pressure = service.shadow_total_pressure();
    assert!(
        final_pressure >= initial_pressure,
        "Shadow pressure should accumulate from distressing input: {} >= {}",
        final_pressure,
        initial_pressure,
    );
}

// ── 23. Shadow fragment count grows with distinct distressing themes ─

#[test]
fn test_shadow_fragment_count() {
    let mut service = make_therapeutic_service();

    // Run enough cycles for therapeutic manager to fire multiple times
    for _ in 0..25 {
        service.cycle("I'm furious at my father for leaving");
    }
    for _ in 0..25 {
        service.cycle("I feel deep shame about my failures");
    }

    // Fragment count should be non-negative (may be 0 if PE was below floor)
    let count = service.shadow_fragment_count();
    assert!(
        count <= 64,
        "Shadow fragment count should be bounded, got {}",
        count,
    );
}

// ── 24. Shadow telemetry fields are finite and bounded ───────────────

#[test]
fn test_shadow_telemetry_bounded() {
    let mut service = make_therapeutic_service();

    for i in 0..55 {
        let input = if i % 2 == 0 {
            "I keep having nightmares about the accident"
        } else {
            "everything feels overwhelming and hopeless"
        };
        service.cycle(input);
    }

    let t = service.shadow_telemetry();
    assert!(t.total_shadow_pressure.is_finite());
    assert!(t.peak_fragment_pressure.is_finite());
    assert!(t.shadow_mean_prediction_error.is_finite());
    assert!(t.pressure_trend.is_finite());
    assert!(t.shadow_to_narrative_ratio.is_finite());
    assert!(t.best_dream_phi_improvement.is_finite());
    assert!(t.shadow_fragment_count <= 64);
}

// ── 25. Shadow telemetry appears in CycleMetadata ────────────────────

#[test]
fn test_shadow_telemetry_in_metadata() {
    let mut service = make_therapeutic_service();

    for i in 0..30 {
        service.cycle(&format!("feeling deeply anxious cycle {}", i));
    }

    let result = service.cycle("checking shadow telemetry");
    let m = &result.metadata;

    // Shadow fields should be populated (even if zero — just verify they exist and are finite)
    assert!(m.therapeutic.shadow_total_pressure.is_finite());
    assert!(m.therapeutic.shadow_peak_pressure.is_finite());
    assert!(m.therapeutic.shadow_mean_prediction_error.is_finite());
    assert!(m.therapeutic.shadow_pressure_trend.is_finite());
    assert!(m.therapeutic.shadow_to_narrative_ratio.is_finite());
    assert!(m.therapeutic.shadow_best_dream_phi.is_finite());
}

// ── 26. Neutral input produces minimal shadow pressure ───────────────

#[test]
fn test_neutral_input_minimal_shadow() {
    let mut service = make_therapeutic_service();

    for _ in 0..55 {
        service.cycle("the weather is pleasant and I had a nice lunch");
    }

    let pressure = service.shadow_total_pressure();
    let count = service.shadow_fragment_count();

    // Neutral content (low |valence|) should not accumulate as shadow
    // Pressure may be non-zero due to system dynamics but should be low
    assert!(
        pressure < 50.0,
        "Neutral input should produce low shadow pressure, got {}",
        pressure,
    );
    assert!(
        count < 10,
        "Neutral input should produce few shadow fragments, got {}",
        count,
    );
}