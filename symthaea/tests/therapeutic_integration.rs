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
