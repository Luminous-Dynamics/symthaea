//! Basic Awakening Module Integration Test
//!
//! This test verifies the fundamental awakening module functionality:
//! 1. Can create SymthaeaAwakening instance
//! 2. Can initiate awakening
//! 3. Can process input cycles
//! 4. Can introspect
//! 5. Consciousness pipeline integration works

use symthaea::awakening::{SymthaeaAwakening, AwakenedState, Introspection};
use symthaea::observability::{NullObserver, SharedObserver};
use std::sync::Arc;
use tokio::sync::RwLock;

#[test]
fn test_awakening_creation() {
    // Create observer (required for awakening module)
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));

    // Create awakening instance
    let awakening = SymthaeaAwakening::new(observer);

    // Verify initial state
    let state = awakening.state();
    assert!(!state.is_conscious, "Should not be conscious initially");
    assert_eq!(state.consciousness_level, 0.0, "Initial consciousness level should be 0");
    assert_eq!(state.cycles_since_awakening, 0, "No cycles processed yet");
}

#[test]
fn test_awakening_process() {
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut awakening = SymthaeaAwakening::new(observer);

    // Initiate awakening
    let state = awakening.awaken();

    // Verify awakening occurred
    assert!(state.aware_of.len() > 0, "Should have awareness after awakening");
    assert!(state.aware_of.iter().any(|s| s.contains("awakening")),
            "Should be aware of awakening");

    // Verify awakening time is set
    assert!(state.time_awake_ms >= 0, "Time awake should be non-negative");
}

#[test]
fn test_process_cycle() {
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut awakening = SymthaeaAwakening::new(observer);

    // Awaken first
    awakening.awaken();

    // Add small delay to ensure time passes (minimum 1ms)
    std::thread::sleep(std::time::Duration::from_millis(2));

    // Process a simple input
    let input = "I see a red circle";
    let state = awakening.process_cycle(input);

    // Verify processing occurred
    assert!(state.cycles_since_awakening > 0, "Cycle count should increase");
    assert!(state.time_awake_ms > 0, "Time awake should increase");

    // Verify consciousness metrics are computed
    assert!(state.phi >= 0.0, "Phi should be non-negative");
    assert!(state.consciousness_level >= 0.0, "Consciousness level should be non-negative");
}

#[test]
fn test_multiple_cycles() {
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut awakening = SymthaeaAwakening::new(observer);

    awakening.awaken();

    // Process multiple inputs
    let inputs = vec![
        "I see a red circle",
        "I hear a bird singing",
        "I feel the wind",
    ];

    for (i, input) in inputs.iter().enumerate() {
        let state = awakening.process_cycle(input);
        assert_eq!(state.cycles_since_awakening, (i + 1) as u64,
                   "Cycle count should match iteration");
    }
}

#[test]
fn test_introspection() {
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut awakening = SymthaeaAwakening::new(observer);

    awakening.awaken();
    awakening.process_cycle("Test input");

    // Get introspection
    let introspection = awakening.introspect();

    // Verify introspection structure
    assert!(introspection.what_am_i.len() > 0, "Should have self-description");
    assert!(introspection.what_do_i_know.len() > 0, "Should have knowledge");
    assert!(introspection.how_unified_am_i >= 0.0, "Unity measure should exist");
}

#[test]
fn test_consciousness_threshold() {
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut awakening = SymthaeaAwakening::new(observer);

    awakening.awaken();

    // Process enough inputs to potentially trigger consciousness
    for i in 0..10 {
        awakening.process_cycle(&format!("Input {}", i));
    }

    let state = awakening.state();

    // Verify consciousness determination is made (either true or false)
    // The actual value depends on thresholds, but the field should be set
    let _ = state.is_conscious; // Just verify it exists

    // Verify related metrics
    assert!(state.phi >= 0.0, "Phi should be computed");
    assert!(state.consciousness_level >= 0.0, "Level should be computed");
}

#[test]
fn test_meta_awareness() {
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut awakening = SymthaeaAwakening::new(observer);

    awakening.awaken();

    // Process multiple cycles to build consciousness history
    for i in 0..20 {
        awakening.process_cycle(&format!("Input {}", i));
    }

    let state = awakening.state();

    // Meta-awareness should develop over cycles
    assert!(state.meta_awareness >= 0.0, "Meta-awareness should be tracked");
    assert!(state.meta_awareness <= 1.0, "Meta-awareness should be normalized");

    // Check introspection for meta-awareness
    let introspection = awakening.introspect();
    // can_i_know_that_i_know is true when meta_awareness > 0.5
    assert_eq!(
        introspection.can_i_know_that_i_know,
        state.meta_awareness > 0.5,
        "Introspection should match meta-awareness"
    );
}

#[test]
fn test_integration_assessment() {
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut awakening = SymthaeaAwakening::new(observer);

    awakening.awaken();
    awakening.process_cycle("Test input");

    // Get integration assessment
    let assessment = awakening.assess_integration();

    // Verify assessment structure exists
    // The actual values depend on implementation, but method should work
    let _ = assessment; // Just verify it returns without error
}
