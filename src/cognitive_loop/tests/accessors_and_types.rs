// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unit tests for cognitive loop types and accessors.
//!
//! Tests pure functions (CycleUrgency, defaults) and accessor methods
//! (stats, config, prediction, flow, emotion, curiosity, goals).

use super::super::*;

// ── CycleUrgency::from_state tests ──────────────────────────────────────

#[test]
fn test_cycle_urgency_from_state_critical_surprise() {
    // Surprise alone triggers Critical regardless of error magnitude.
    let u = CycleUrgency::from_state(0.0, 0.1, true, 100);
    assert_eq!(u, CycleUrgency::Critical);
}

#[test]
fn test_cycle_urgency_from_state_critical_high_error() {
    // Error > 3× threshold triggers Critical (no surprise needed).
    let u = CycleUrgency::from_state(0.5, 0.1, false, 0);
    assert_eq!(u, CycleUrgency::Critical);
}

#[test]
fn test_cycle_urgency_from_state_normal_above_threshold() {
    // Error above threshold but below 3× → Normal.
    let u = CycleUrgency::from_state(0.15, 0.1, false, 20);
    assert_eq!(u, CycleUrgency::Normal);
}

#[test]
fn test_cycle_urgency_from_state_normal_low_consecutive() {
    // Error below threshold but fewer than 10 consecutive low cycles → Normal.
    let u = CycleUrgency::from_state(0.01, 0.1, false, 5);
    assert_eq!(u, CycleUrgency::Normal);
}

#[test]
fn test_cycle_urgency_from_state_cruise() {
    // Error well below threshold and >=10 consecutive low-error cycles → Cruise.
    let u = CycleUrgency::from_state(0.01, 0.1, false, 15);
    assert_eq!(u, CycleUrgency::Cruise);
}

// ── CycleUrgency::should_run tests ──────────────────────────────────────

#[test]
fn test_should_run_critical_uses_critical_interval() {
    let u = CycleUrgency::Critical;
    // Critical interval = 2: runs on even cycles.
    assert!(u.should_run(0, 2, 5, 10));
    assert!(!u.should_run(1, 2, 5, 10));
    assert!(u.should_run(4, 2, 5, 10));
}

#[test]
fn test_should_run_normal_uses_normal_interval() {
    let u = CycleUrgency::Normal;
    // Normal interval = 3: cycle 0,3,6... run.
    assert!(u.should_run(0, 1, 3, 10));
    assert!(!u.should_run(1, 1, 3, 10));
    assert!(u.should_run(6, 1, 3, 10));
}

#[test]
fn test_should_run_cruise_uses_cruise_interval() {
    let u = CycleUrgency::Cruise;
    // Cruise interval = 4: runs on 0,4,8...
    assert!(u.should_run(0, 1, 2, 4));
    assert!(!u.should_run(1, 1, 2, 4));
    assert!(u.should_run(8, 1, 2, 4));
}

#[test]
fn test_should_run_zero_interval_always_runs() {
    // Interval 0 is a special case: always runs (0 || cycle % 0 would panic,
    // so the code short-circuits).
    assert!(CycleUrgency::Critical.should_run(7, 0, 99, 99));
    assert!(CycleUrgency::Normal.should_run(7, 99, 0, 99));
    assert!(CycleUrgency::Cruise.should_run(7, 99, 99, 0));
}

// ── CycleUrgency::run_consciousness_monitors ────────────────────────────

#[test]
fn test_run_consciousness_monitors() {
    assert!(CycleUrgency::Critical.run_consciousness_monitors());
    assert!(CycleUrgency::Normal.run_consciousness_monitors());
    assert!(!CycleUrgency::Cruise.run_consciousness_monitors());
}

// ── Default impls for carryover sub-structs ─────────────────────────────

#[test]
fn test_consciousness_cache_default() {
    let c = ConsciousnessCache::default();
    assert_eq!(c.predictive_phi_modulation, 1.0);
    assert_eq!(c.cross_modal_psi, 0.0);
    assert!(c.last_sigma.is_none());
    assert!(c.last_spectral_mip_phi.is_none());
    assert!(c.last_hierarchical_mip_phi.is_none());
}

#[test]
fn test_urgency_state_default() {
    let u = UrgencyState::default();
    assert_eq!(u.urgency, CycleUrgency::Normal);
    assert_eq!(u.consecutive_low_error, 0);
    assert_eq!(u.mode_confidence, 1.0);
    assert_eq!(u.prev_urgency, CycleUrgency::Normal);
}

#[test]
fn test_learning_state_default() {
    let l = LearningState::default();
    assert_eq!(l.prediction_confidence, 0.5);
    assert_eq!(l.mce_lr_boost, 0.0);
    assert_eq!(l.adaptive_threshold_scale, 1.0);
    assert_eq!(l.subsystem_lr_factor, 1.0);
    assert_eq!(l.self_model_accuracy, 0.5);
}

#[test]
fn test_cycle_carryover_default_composes_sub_defaults() {
    let c = CycleCarryover::default();
    // Spot-check that the derived Default delegates to sub-struct defaults.
    assert_eq!(c.urgency.urgency, CycleUrgency::Normal);
    assert_eq!(c.learning.prediction_confidence, 0.5);
    assert_eq!(c.consciousness.predictive_phi_modulation, 1.0);
    assert_eq!(c.quality.phi_spectral_weight, 0.6);
    assert_eq!(c.history.body_arousal, 0.5);
}

// ── CognitiveLoopService accessor tests ─────────────────────────────────

#[test]
fn test_stats_initial_cycle_count() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    assert_eq!(service.stats().total_cycles, 0);
}

#[test]
fn test_config_roundtrip() {
    let cfg = CognitiveLoopConfig::default();
    let service = CognitiveLoopService::new(cfg.clone()).unwrap();
    // The config accessor returns the same config we passed in.
    assert_eq!(service.config().learning_threshold, cfg.learning_threshold);
}

#[test]
fn test_prediction_confidence_initial() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let pc = service.prediction_confidence();
    assert!(pc.is_finite(), "prediction_confidence should be finite");
    assert!(
        (0.0..=1.0).contains(&pc),
        "prediction_confidence should be in [0,1]"
    );
}

#[test]
fn test_predictions_trustworthy_threshold() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Trustworthy iff prediction_confidence > 0.4.
    let trustworthy = service.predictions_trustworthy();
    if service.prediction_confidence() > 0.4 {
        assert!(trustworthy);
    } else {
        assert!(!trustworthy);
    }
}

#[test]
fn test_provide_reward_no_panic() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Extreme values should be clamped, not panic.
    service.provide_reward(100.0);
    service.provide_reward(-100.0);
    service.provide_reward(0.0);
    service.provide_reward(f32::NAN.clamp(-1.0, 1.0)); // NaN clamp → NaN, but provide_reward clamps internally
}

#[test]
fn test_set_social_signals_no_panic() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Out-of-range values should be clamped silently.
    service.set_social_signals(5.0, -3.0, 1.5, 0, -0.5);
    service.set_social_signals(0.5, 0.5, 0.5, 3, 0.5);
}

#[test]
fn test_flow_state_initial() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Fresh service should not be in flow.
    assert!(!service.in_flow());
    assert_eq!(service.flow_streak(), 0);
    let intensity = service.flow_intensity();
    assert!(intensity.is_finite());
    assert!((0.0..=1.0).contains(&intensity));
}

#[test]
fn test_emotion_initial() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let v = service.emotional_valence();
    assert!(v.is_finite(), "emotional_valence should be finite");
    assert!((-1.0..=1.0).contains(&v), "valence in [-1,1]");
    // No emotional content initially.
    assert!(!service.has_emotional_content());
}

#[test]
fn test_boredom_and_curiosity_initial() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let b = service.boredom();
    let c = service.curiosity();
    assert!(
        b.is_finite() && (0.0..=1.0).contains(&b),
        "boredom in [0,1]"
    );
    assert!(
        c.is_finite() && (0.0..=1.0).contains(&c),
        "curiosity in [0,1]"
    );
}

#[test]
fn test_consciousness_snapshot_fields_finite() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let snap = service.consciousness_snapshot();
    assert!(snap.consciousness_level >= 0.0 && snap.consciousness_level <= 1.0);
    assert!(snap.prediction_confidence.is_finite());
    assert!(snap.flow_intensity.is_finite());
    assert!(snap.boredom.is_finite());
    assert!(snap.curiosity.is_finite());
    assert!(snap.emotional_valence.is_finite());
    assert!(snap.emotional_arousal.is_finite());
}

#[test]
fn test_consciousness_level_in_range() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let cl = service.consciousness_level();
    assert!(
        (0.0..=1.0).contains(&cl),
        "consciousness_level should be in [0.0, 1.0], got {cl}"
    );
}

#[test]
fn test_add_goal_and_active_goals() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    assert!(service.active_goals().is_empty(), "no goals initially");

    service.add_goal("g1", "Explore environment", 0.8);
    service.add_goal("g2", "Learn patterns", 0.5);

    let goals = service.active_goals();
    assert_eq!(goals.len(), 2);
    assert_eq!(goals[0].id, "g1");
    assert_eq!(goals[1].id, "g2");
    assert!(goals[0].is_active);
    // Priority should be clamped to [0,1].
    assert!(goals[0].priority >= 0.0 && goals[0].priority <= 1.0);
}
