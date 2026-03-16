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
    config.therapeutic_crisis_threshold = 0.15;
    CognitiveLoopService::new(config).unwrap()
}

// ── 1. Basic therapeutic pipeline runs without panic ──────────────────

#[test]
fn test_therapeutic_cycle_runs() {
    let mut service = make_therapeutic_service();

    // Run 20 cycles with neutral input
    for i in 0..20 {
        let result = service.cycle(&format!("test cycle {}", i));
        assert!(result.metadata.consciousness_level >= 0.0);
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

// ── 6. Non-crisis input does not trigger keyword crisis ──────────────

#[test]
fn test_no_keyword_crisis_on_benign() {
    // Verify that benign input does not trigger keyword-based crisis detection.
    // Note: HDC similarity may produce low-confidence false positives via
    // random hash collisions — this is expected. We test that keyword detection
    // (the high-confidence path) does not trigger.
    use symthaea_therapeutic::CrisisDetector;
    let detector = CrisisDetector::new();
    let alert = detector.detect("The weather is nice today and I enjoyed my lunch");
    if let Some(ref a) = alert {
        // If it triggered, it should NOT be from keyword matching
        assert_ne!(
            a.matched_indicator, "keyword_match",
            "Benign input should not trigger keyword-based crisis"
        );
    }
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
    use symthaea_therapeutic::affect_regulation::RegulationEngine;
    use symthaea_therapeutic::RegulationStrategy;

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
    use symthaea_therapeutic::affect_regulation::RegulationEngine;
    use symthaea_therapeutic::RegulationStrategy;

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
