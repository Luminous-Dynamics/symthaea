// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the resonant_speech/ module.
//!
//! Tests cover:
//! - ResonantSpeech creation and response generation
//! - CognitiveLoad level boundaries
//! - ResponseProfile scaling with load
//! - UserState empathy and overwhelm detection
//! - Statistics accumulation

use symthaea::resonant_speech::{CognitiveLoad, ResonantSpeech, UserState};

// ============================================================================
// Speech Creation & Generation Tests
// ============================================================================

#[test]
fn test_speech_creation_and_generation() {
    let mut speech = ResonantSpeech::new();
    let response = speech.generate("Hello", "greeting");

    assert!(
        !response.is_empty(),
        "Generated response should be non-empty"
    );
    assert_eq!(
        speech.stats().total_responses,
        1,
        "Should have 1 total response"
    );
}

// ============================================================================
// CognitiveLoad Tests
// ============================================================================

#[test]
fn test_cognitive_load_boundaries() {
    assert_eq!(CognitiveLoad::from_level(0.0), CognitiveLoad::Low);
    assert_eq!(CognitiveLoad::from_level(0.25), CognitiveLoad::Medium);
    assert_eq!(CognitiveLoad::from_level(0.50), CognitiveLoad::High);
    assert_eq!(CognitiveLoad::from_level(0.75), CognitiveLoad::Overloaded);
}

#[test]
fn test_response_profile_scales_with_load() {
    let low_profile = CognitiveLoad::Low.response_profile();
    let overloaded_profile = CognitiveLoad::Overloaded.response_profile();

    assert!(
        low_profile.max_sentences > overloaded_profile.max_sentences,
        "Low load should allow more sentences ({}) than Overloaded ({})",
        low_profile.max_sentences,
        overloaded_profile.max_sentences
    );
    assert!(low_profile.use_examples, "Low load should allow examples");
    assert!(
        !overloaded_profile.use_examples,
        "Overloaded should not use examples"
    );
}

// ============================================================================
// UserState Detection Tests
// ============================================================================

#[test]
fn test_user_state_empathy_detection() {
    let state = UserState {
        frustration: 0.7,
        ..Default::default()
    };
    assert!(
        state.needs_empathy(),
        "frustration=0.7 should trigger needs_empathy()"
    );

    let calm_state = UserState {
        frustration: 0.2,
        confidence: 0.8,
        ..Default::default()
    };
    assert!(
        !calm_state.needs_empathy(),
        "Low frustration + high confidence should not need empathy"
    );
}

#[test]
fn test_user_state_overwhelm_detection() {
    let overwhelmed = UserState {
        cognitive_load: CognitiveLoad::Overloaded,
        frustration: 0.6,
        ..Default::default()
    };
    assert!(
        overwhelmed.is_overwhelmed(),
        "Overloaded + frustrated should be overwhelmed"
    );

    let high_but_calm = UserState {
        cognitive_load: CognitiveLoad::High,
        frustration: 0.3,
        ..Default::default()
    };
    assert!(
        !high_but_calm.is_overwhelmed(),
        "High load but calm (frustration=0.3) should not be overwhelmed"
    );
}

// ============================================================================
// Stats Accumulation Tests
// ============================================================================

#[test]
fn test_stats_accumulate() {
    let mut speech = ResonantSpeech::new();
    for i in 0..5 {
        speech.generate(&format!("Message number {i}"), "chat");
    }

    let stats = speech.stats();
    assert_eq!(stats.total_responses, 5, "Should have 5 total responses");
    assert!(
        stats.avg_response_length > 0.0,
        "Average response length should be positive, got {}",
        stats.avg_response_length
    );
}