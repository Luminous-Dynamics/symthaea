// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the bidirectional Mycelix consciousness bridge.
//!
//! Tests `CognitiveLoopService::inject_mycelix_profile()` which maps Mycelix's
//! 4D consciousness profile (identity, reputation, community, engagement) back
//! into Symthaea's social state and neuromodulator bath.
//!
//! Forward path:  Symthaea C_unified → Mycelix engagement dimension
//! Reverse path:  Mycelix 4D profile → Symthaea social state + oxytocin

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default())
        .expect("CognitiveLoopService::new should succeed")
}

/// Inject a known Mycelix profile and verify the social metrics are mapped correctly.
///
/// Mapping under test:
/// - community (0.9) → social_trust
/// - reputation (0.6) → social_cooperation_rate
/// - identity (0.8) → social_prediction_accuracy
/// - combined_score = 0.25*0.8 + 0.25*0.6 + 0.30*0.9 + 0.20*0.5 = 0.72 → social_mean_trust
#[test]
fn test_inject_profile_updates_social_metrics() {
    let mut service = make_service();

    service.inject_mycelix_profile(0.8, 0.6, 0.9, 0.5);

    // Run one cycle so that the social state is captured in CycleMetadata.
    let result = service.cycle("mycelix profile test");
    let m = &result.metadata;

    let expected_combined: f32 = (0.25 * 0.8 + 0.25 * 0.6 + 0.30 * 0.9 + 0.20 * 0.5) as f32;

    assert!(
        (m.social_trust_current - 0.9).abs() < 1e-5,
        "social_trust should reflect community=0.9, got {}",
        m.social_trust_current,
    );
    assert!(
        (m.social_cooperation_current - 0.6).abs() < 1e-5,
        "social_cooperation_rate should reflect reputation=0.6, got {}",
        m.social_cooperation_current,
    );
    assert!(
        (m.social_prediction_accuracy - 0.8).abs() < 1e-5,
        "social_prediction_accuracy should reflect identity=0.8, got {}",
        m.social_prediction_accuracy,
    );
    assert!(
        (m.social_mean_trust - expected_combined).abs() < 1e-5,
        "social_mean_trust should reflect combined_score={}, got {}",
        expected_combined,
        m.social_mean_trust,
    );
}

/// Inject a profile with high community (0.9) and verify that oxytocin is
/// elevated above the baseline of a fresh service with no injection.
///
/// dose = (0.9 * 0.3).clamp(0.0, 0.3) = 0.27, which exceeds the 0.01 skip
/// threshold, so `inject("oxytocin", 0.27, 50)` should fire.
///
/// The pharmacological injection is queued and applied during the next
/// `NeuromodulatorBath::update()` call (i.e., during a cycle tick), so we
/// run one cycle after injection before sampling the snapshot.
#[test]
fn test_high_community_boosts_oxytocin() {
    // Baseline: run one cycle with no injection
    let mut baseline_service = make_service();
    let _ = baseline_service.cycle("baseline");
    let baseline_oxy = baseline_service.neuromod_snapshot().oxytocin_effective;

    // With injection: inject profile then run one cycle so the pharmacological
    // injection is processed by the bath's update() call.
    let mut injected_service = make_service();
    injected_service.inject_mycelix_profile(0.5, 0.5, 0.9, 0.5);
    let _ = injected_service.cycle("after injection");

    let injected_oxy = injected_service.neuromod_snapshot().oxytocin_effective;

    assert!(
        injected_oxy > baseline_oxy,
        "community=0.9 should boost oxytocin above baseline: injected={}, baseline={}",
        injected_oxy,
        baseline_oxy,
    );
}

/// Inject a profile with community=0.0. The oxytocin dose would be
/// (0.0 * 0.3) = 0.0, which is <= 0.01, so the injection is skipped.
/// Oxytocin should remain at the same level as a service with no injection,
/// after both run one cycle.
#[test]
fn test_zero_community_skips_oxytocin() {
    let mut baseline_service = make_service();
    let _ = baseline_service.cycle("baseline");
    let baseline_oxy = baseline_service.neuromod_snapshot().oxytocin_effective;

    let mut injected_service = make_service();
    injected_service.inject_mycelix_profile(0.5, 0.5, 0.0, 0.5);
    let _ = injected_service.cycle("after zero community injection");

    let injected_oxy = injected_service.neuromod_snapshot().oxytocin_effective;

    assert!(
        (injected_oxy - baseline_oxy).abs() < 1e-4,
        "community=0.0 should skip oxytocin injection: injected={}, baseline={}",
        injected_oxy,
        baseline_oxy,
    );
}

/// Inject wildly out-of-range values and verify all social fields are clamped
/// to [0, 1].
#[test]
fn test_out_of_range_values_clamp() {
    let mut service = make_service();

    service.inject_mycelix_profile(2.0, -1.0, 1.5, -0.5);

    // Run one cycle to capture telemetry.
    let result = service.cycle("clamping test");
    let m = &result.metadata;

    // community=1.5 → social_trust clamped to 1.0
    assert!(
        m.social_trust_current >= 0.0 && m.social_trust_current <= 1.0,
        "social_trust must be in [0,1], got {}",
        m.social_trust_current,
    );
    // reputation=-1.0 → social_cooperation_rate clamped to 0.0
    assert!(
        m.social_cooperation_current >= 0.0 && m.social_cooperation_current <= 1.0,
        "social_cooperation_rate must be in [0,1], got {}",
        m.social_cooperation_current,
    );
    // identity=2.0 → social_prediction_accuracy clamped to 1.0
    assert!(
        m.social_prediction_accuracy >= 0.0 && m.social_prediction_accuracy <= 1.0,
        "social_prediction_accuracy must be in [0,1], got {}",
        m.social_prediction_accuracy,
    );
    // combined_score with out-of-range inputs → clamped to [0,1]
    assert!(
        m.social_mean_trust >= 0.0 && m.social_mean_trust <= 1.0,
        "social_mean_trust must be in [0,1], got {}",
        m.social_mean_trust,
    );

    // Verify specific clamped values
    assert!(
        (m.social_trust_current - 1.0).abs() < 1e-5,
        "community=1.5 should clamp social_trust to 1.0, got {}",
        m.social_trust_current,
    );
    assert!(
        m.social_cooperation_current.abs() < 1e-5,
        "reputation=-1.0 should clamp social_cooperation_rate to 0.0, got {}",
        m.social_cooperation_current,
    );
    assert!(
        (m.social_prediction_accuracy - 1.0).abs() < 1e-5,
        "identity=2.0 should clamp social_prediction_accuracy to 1.0, got {}",
        m.social_prediction_accuracy,
    );
}

/// Inject a profile, run 5 cycles, and verify the social_trust value persists
/// (i.e., is not overwritten by the cycle machinery).
#[test]
fn test_profile_injection_persists_across_cycles() {
    let mut service = make_service();

    service.inject_mycelix_profile(0.7, 0.4, 0.85, 0.6);

    // Run 5 cycles and check that social_trust still reflects the injected community value.
    let mut last_trust = 0.0_f32;
    for i in 0..5 {
        let result = service.cycle(&format!("persistence cycle {}", i));
        last_trust = result.metadata.social_trust_current;
    }

    assert!(
        (last_trust - 0.85).abs() < 1e-5,
        "social_trust should persist as community=0.85 across 5 cycles, got {}",
        last_trust,
    );
}