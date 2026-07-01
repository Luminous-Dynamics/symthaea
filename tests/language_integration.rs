// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language Module Integration Tests
//!
//! Tests the public API surface of `symthaea::language::*` without requiring
//! network access or an LLM backend.  Covers:
//!
//! - `ConsciousnessLanguageCore` creation, understanding, state transitions,
//!   phi getter/setter with clamping, and the feedback loop.
//! - `NixErrorDiagnoser` pattern matching against real NixOS error messages.
//! - `SemanticIntentClassifier` HDC-based routing for NixOS, Programming,
//!   Math, SystemAdmin, and General categories.
//! - Enum defaults and construction helpers for `ConsciousnessStateLevel`
//!   and `ExecutionStrategy`.
//!
//! **Note**: HDC encodings are stochastic (character n-gram based). Assertions
//! test finiteness, valid ranges, and structural invariants rather than exact
//! numeric values.

use symthaea::language::*;

// ============================================================================
// ConsciousnessLanguageCore — creation and defaults
// ============================================================================

#[test]
fn core_default_starts_active_with_valid_phi() {
    let core = ConsciousnessLanguageCore::default();

    assert_eq!(core.consciousness_state(), ConsciousnessStateLevel::Active);
    assert!(core.phi().is_finite(), "phi must be finite");
    assert!(
        (0.0..=1.0).contains(&core.phi()),
        "phi must be in [0, 1], got {}",
        core.phi()
    );
}

#[test]
fn core_with_custom_config() {
    let config = ConsciousnessLanguageConfig {
        dimension: 256,
        emotional_enabled: false,
        metacognition_enabled: false,
        llm_config: LLMOrganConfig::default(),
        confidence_threshold: 0.8,
    };
    let core = ConsciousnessLanguageCore::new(config);

    // Custom config should still produce a valid core
    assert_eq!(core.consciousness_state(), ConsciousnessStateLevel::Active);
    assert!(core.phi().is_finite());
}

// ============================================================================
// ConsciousnessLanguageCore — understanding NixOS commands
// ============================================================================

#[test]
fn understand_install_command() {
    let mut core = ConsciousnessLanguageCore::default();
    let result = core.understand("install firefox");

    assert_eq!(result.nixos.intent, NixOSIntent::Install);
    assert!(result.confidence.is_finite());
    assert!(result.confidence > 0.0);
    assert!(!result.nixos.description.is_empty());
    // Entities should include "firefox"
    assert!(
        result.nixos.entities.iter().any(|e| e == "firefox"),
        "entities should contain 'firefox', got {:?}",
        result.nixos.entities
    );
}

#[test]
fn understand_configure_networking() {
    let mut core = ConsciousnessLanguageCore::default();
    let result = core.understand("configure networking");

    assert_eq!(result.nixos.intent, NixOSIntent::Configure);
    assert!(result.confidence.is_finite());
    assert!(!result.nixos.description.is_empty());
}

#[test]
fn understand_garbage_collect() {
    let mut core = ConsciousnessLanguageCore::default();
    let result = core.understand("garbage collect old generations");

    assert_eq!(result.nixos.intent, NixOSIntent::GarbageCollect);
    assert!(result.confidence > 0.0);
}

// ============================================================================
// ConsciousnessLanguageCore — non-NixOS / unknown input
// ============================================================================

#[test]
fn understand_non_nixos_input_yields_unknown_with_clarifying_questions() {
    let mut core = ConsciousnessLanguageCore::default();
    let result = core.understand("xyzzy plugh");

    assert_eq!(result.nixos.intent, NixOSIntent::Unknown);
    // Default confidence_threshold is 0.6; Unknown intent produces 0.3
    // so clarifying questions should be present.
    assert!(
        !result.clarifying_questions.is_empty(),
        "low-confidence understanding should generate clarifying questions"
    );
    // unified_free_energy should be finite
    assert!(result.unified_free_energy.is_finite());
}

// ============================================================================
// ConsciousnessLanguageCore — consciousness state transitions
// ============================================================================

#[test]
fn consciousness_state_transitions() {
    let mut core = ConsciousnessLanguageCore::default();
    assert_eq!(core.consciousness_state(), ConsciousnessStateLevel::Active);

    let all_states = [
        ConsciousnessStateLevel::Dormant,
        ConsciousnessStateLevel::Reactive,
        ConsciousnessStateLevel::Active,
        ConsciousnessStateLevel::Reflective,
        ConsciousnessStateLevel::Metacognitive,
    ];

    for &state in &all_states {
        core.set_consciousness_state(state);
        assert_eq!(core.consciousness_state(), state);
    }
}

// ============================================================================
// ConsciousnessLanguageCore — phi getter/setter with clamping
// ============================================================================

#[test]
fn phi_setter_clamps_to_unit_interval() {
    let mut core = ConsciousnessLanguageCore::default();

    core.set_phi(0.75);
    assert!((core.phi() - 0.75).abs() < f64::EPSILON);

    // Above 1.0 clamps to 1.0
    core.set_phi(999.0);
    assert!(
        (core.phi() - 1.0).abs() < f64::EPSILON,
        "phi should clamp to 1.0"
    );

    // Below 0.0 clamps to 0.0
    core.set_phi(-42.0);
    assert!(core.phi().abs() < f64::EPSILON, "phi should clamp to 0.0");

    // Exact boundaries
    core.set_phi(0.0);
    assert!(core.phi().abs() < f64::EPSILON);

    core.set_phi(1.0);
    assert!((core.phi() - 1.0).abs() < f64::EPSILON);
}

// ============================================================================
// NixErrorDiagnoser — real NixOS error messages
// ============================================================================

#[test]
fn diagnose_missing_attribute_error() {
    let diagnoser = NixErrorDiagnoser::new();
    let diagnosis =
        diagnoser.diagnose("error: attribute 'foo' missing at /etc/nixos/configuration.nix:42:5");

    assert_eq!(diagnosis.category, NixErrorCategory::Evaluation);
    assert!(diagnosis.confidence > 0.5);
    assert!(!diagnosis.explanation.is_empty());
    // Location extraction should capture the file path
    let loc = diagnosis.location.expect("location should be extracted");
    assert!(
        loc.contains("/etc/nixos/configuration.nix"),
        "location should contain the file path, got: {loc}"
    );
}

#[test]
fn diagnose_package_collision_as_unknown() {
    let diagnoser = NixErrorDiagnoser::new();
    // "collision between packages" does not match any built-in pattern
    let diagnosis =
        diagnoser.diagnose("collision between packages /nix/store/aaa and /nix/store/bbb");

    // This specific message is not in the pattern set, so it falls to Unknown
    assert_eq!(diagnosis.category, NixErrorCategory::Unknown);
    assert!(
        diagnosis.confidence < 0.5,
        "unrecognised errors should have low confidence"
    );
}

#[test]
fn diagnose_hash_mismatch() {
    let diagnoser = NixErrorDiagnoser::new();
    let diagnosis = diagnoser.diagnose("error: hash mismatch in fixed-output derivation");

    assert_eq!(diagnosis.category, NixErrorCategory::Build);
    assert!(diagnosis.confidence.is_finite());
    assert!(diagnosis.confidence > 0.0);
}

#[test]
fn diagnose_flake_error() {
    let diagnoser = NixErrorDiagnoser::new();
    let diagnosis = diagnoser.diagnose("error: flake 'github:user/repo' has no output 'packages'");

    assert_eq!(diagnosis.category, NixErrorCategory::Flake);
    assert!(!diagnosis.explanation.is_empty());
}

// ============================================================================
// SemanticIntentClassifier — HDC-based category routing
// ============================================================================

#[test]
fn classifier_routes_nixos_query() {
    let mut classifier = SemanticIntentClassifier::new();
    let result = classifier.classify("how to install firefox on nixos");

    assert_eq!(
        result.category,
        IntentCategory::NixOS,
        "Expected NixOS, got {:?} (scores: {:?})",
        result.category,
        result.scores
    );
    assert!(result.confidence.is_finite());
    // Scores should cover all 5 categories
    assert_eq!(result.scores.len(), 5);
}

#[test]
fn classifier_routes_math_query() {
    let mut classifier = SemanticIntentClassifier::new();
    let result = classifier.classify("solve this integral equation for x");

    assert_eq!(
        result.category,
        IntentCategory::Math,
        "Expected Math, got {:?} (scores: {:?})",
        result.category,
        result.scores
    );
}

#[test]
fn classifier_scores_sorted_descending() {
    let mut classifier = SemanticIntentClassifier::new();
    let result = classifier.classify("nixos-rebuild switch --flake .");

    for window in result.scores.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "scores must be sorted descending: {} >= {} failed",
            window[0].1,
            window[1].1
        );
    }
}

// ============================================================================
// Feedback loop: create_feedback_from_execution + receive_action_outcome
// ============================================================================

#[test]
fn feedback_loop_success_increases_phi() {
    let mut core = ConsciousnessLanguageCore::default();
    core.set_phi(0.5);
    let phi_before = core.phi();

    // Simulate understanding followed by a successful execution
    let _ = core.understand("install vim");
    let feedback = core
        .create_feedback_from_execution(true, 0.6, None, false)
        .expect("feedback should be created");

    assert!(feedback.action_succeeded);
    assert!(feedback.original_input.contains("install vim"));

    core.receive_action_outcome(feedback);
    // Successful action should nudge phi upward
    assert!(
        core.phi() >= phi_before,
        "phi should not decrease after success: before={phi_before}, after={}",
        core.phi()
    );
}

#[test]
fn feedback_loop_failure_decreases_phi() {
    let mut core = ConsciousnessLanguageCore::default();
    core.set_phi(0.5);
    let phi_before = core.phi();

    let _ = core.understand("build broken-package");
    let feedback = core
        .create_feedback_from_execution(false, 0.3, Some("build failed".to_string()), false)
        .expect("feedback should be created");

    assert!(!feedback.action_succeeded);
    assert_eq!(feedback.error_message.as_deref(), Some("build failed"));
    assert_eq!(feedback.errors, vec!["build failed".to_string()]);

    core.receive_action_outcome(feedback);
    // Failed action should nudge phi downward
    assert!(
        core.phi() <= phi_before,
        "phi should not increase after failure: before={phi_before}, after={}",
        core.phi()
    );
}

// ============================================================================
// ConsciousUnderstandingResult — structural field validation
// ============================================================================

#[test]
fn understanding_result_has_all_required_fields() {
    let mut core = ConsciousnessLanguageCore::default();
    let result = core.understand("upgrade the system");

    // Both legacy and new NixOS understanding aliases should match
    assert_eq!(result.nixos.intent, result.nix_understanding.intent);
    assert_eq!(result.nixos.confidence, result.nix_understanding.confidence);

    // Consciousness state should reflect core's state
    assert_eq!(result.consciousness_state, ConsciousnessStateLevel::Active);

    // Active quadrants list should be non-empty
    assert!(!result.active_quadrants.is_empty());

    // Execution strategies: recommended_strategy and execution_strategy
    // should both be present (structural check, not value equality since Clone)
    let _strategy_type: ExecutionStrategyType = (&result.execution_strategy).into();
    let _rec_type: ExecutionStrategyType = (&result.recommended_strategy).into();

    // Intent classification should be populated
    assert!(
        result.intent_classification.is_some(),
        "intent_classification should be populated"
    );

    // Primitive tiers grounding should include at least NSM (always present)
    assert!(
        result.primitive_tiers.contains(&"NSM".to_string()),
        "primitive_tiers should always include NSM, got {:?}",
        result.primitive_tiers
    );

    // Numeric fields should be finite
    assert!(result.consciousness_phi.is_finite());
    assert!(result.epistemic_confidence.is_finite());
    assert!(result.unified_free_energy.is_finite());
}

// ============================================================================
// ExecutionStrategy — enum construction helpers
// ============================================================================

#[test]
fn execution_strategy_constructors() {
    let confident = ExecutionStrategy::confident();
    assert!(matches!(confident, ExecutionStrategy::Confident { .. }));

    let autopilot = ExecutionStrategy::autopilot();
    assert!(matches!(autopilot, ExecutionStrategy::Autopilot { .. }));

    let lost = ExecutionStrategy::lost();
    assert!(matches!(lost, ExecutionStrategy::Lost { .. }));

    let safe = ExecutionStrategy::safe_default();
    assert!(matches!(safe, ExecutionStrategy::Curious { .. }));

    let default = ExecutionStrategy::default();
    assert!(matches!(default, ExecutionStrategy::Curious { .. }));

    // ExecutionStrategyType conversion
    assert_eq!(
        ExecutionStrategyType::from(&confident),
        ExecutionStrategyType::Confident
    );
    assert_eq!(
        ExecutionStrategyType::from(&autopilot),
        ExecutionStrategyType::Autopilot
    );
    assert_eq!(
        ExecutionStrategyType::from(&lost),
        ExecutionStrategyType::Lost
    );
    assert_eq!(
        ExecutionStrategyType::from(&safe),
        ExecutionStrategyType::Curious
    );
}