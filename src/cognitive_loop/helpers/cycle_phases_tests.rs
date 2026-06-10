// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for cycle phase result structs and extracted phase methods.

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Result struct construction and field verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_resonator_codebook_result_fields() {
    let result = ResonatorCodebookResult {
        resonator_promotions: 3,
        codebook_evictions: 1,
        codebook_diversity: 0.75,
        codebook_utilization_rate: 0.5,
    };
    assert_eq!(result.resonator_promotions, 3);
    assert_eq!(result.codebook_evictions, 1);
    assert!((result.codebook_diversity - 0.75).abs() < f32::EPSILON);
    assert!((result.codebook_utilization_rate - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_dream_phase_result_fields() {
    let result = DreamPhaseResult {
        dream_insights: 5,
        dream_phi_improvement: 0.12,
        dream_wisdom_count: 2,
    };
    assert_eq!(result.dream_insights, 5);
    assert!((result.dream_phi_improvement - 0.12).abs() < f32::EPSILON);
    assert_eq!(result.dream_wisdom_count, 2);
}

#[test]
fn test_episodic_replay_result_fields() {
    let result = EpisodicReplayResult {
        surprise_replay_batch_size: 16,
        phasic_da_replay_boost: 4,
        memory_db_flushed: false,
    };
    assert_eq!(result.surprise_replay_batch_size, 16);
    assert_eq!(result.phasic_da_replay_boost, 4);
}

#[test]
fn test_parameter_optimization_result_defaults() {
    let result = ParameterOptimizationResult {
        best_tau_scale: 1.0,
        phi_gain: 0.0,
        swap_occurred: false,
    };
    assert!((result.best_tau_scale - 1.0).abs() < f32::EPSILON);
    assert!((result.phi_gain - 0.0).abs() < f64::EPSILON);
    assert!(!result.swap_occurred);
}

#[test]
fn test_urgency_result_fields() {
    let result = UrgencyResult {
        urgency: super::super::super::CycleUrgency::Normal,
        error_pattern: "Stable",
        predicted_urgency: "Normal",
        prediction_coherence_urgency_bias: 0.0,
        error_slope: 0.0,
        oscillation_ratio: 0.0,
    };
    assert!(matches!(
        result.urgency,
        super::super::super::CycleUrgency::Normal
    ));
    assert_eq!(result.error_pattern, "Stable");
    assert_eq!(result.predicted_urgency, "Normal");
    assert!((result.prediction_coherence_urgency_bias).abs() < f32::EPSILON);
}

#[test]
fn test_cycle_init_result_fields() {
    let result = CycleInitResult {
        exploration_urge_start: 0.3,
        startup_suppressed: true,
        startup_warmup_progress: 0.5,
    };
    assert!((result.exploration_urge_start - 0.3).abs() < f32::EPSILON);
    assert!(result.startup_suppressed);
    assert!((result.startup_warmup_progress - 0.5).abs() < f32::EPSILON);
}

// ═══════════════════════════════════════════════════════════════════════════
// run_cycle_init tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_run_cycle_init_startup_suppressed_at_cycle_zero() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 0;
    let mut timings = super::super::super::ModuleTimings::default();
    let result = service.run_cycle_init(&mut timings);
    assert!(
        result.startup_suppressed,
        "Cycle 0 should be startup suppressed"
    );
    assert!(
        (result.startup_warmup_progress).abs() < f32::EPSILON,
        "Warmup progress at cycle 0 should be 0.0, got {}",
        result.startup_warmup_progress
    );
}

#[test]
fn test_run_cycle_init_startup_suppressed_midway() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 25; // half of 50
    let mut timings = super::super::super::ModuleTimings::default();
    let result = service.run_cycle_init(&mut timings);
    assert!(
        result.startup_suppressed,
        "Cycle 25 should be startup suppressed"
    );
    assert!(
        (result.startup_warmup_progress - 0.5).abs() < 0.01,
        "Warmup progress at cycle 25 should be ~0.5, got {}",
        result.startup_warmup_progress
    );
}

#[test]
fn test_run_cycle_init_not_suppressed_after_warmup() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 100; // well past 50
    let mut timings = super::super::super::ModuleTimings::default();
    let result = service.run_cycle_init(&mut timings);
    assert!(
        !result.startup_suppressed,
        "Cycle 100 should NOT be startup suppressed"
    );
    assert!(
        (result.startup_warmup_progress - 1.0).abs() < f32::EPSILON,
        "Warmup progress past warmup should be 1.0, got {}",
        result.startup_warmup_progress
    );
}

#[test]
fn test_run_cycle_init_lr_suppressed_during_warmup() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    let base_lr = service.stats.adaptive_learning_rate;
    service.stats.total_cycles = 10; // early warmup (10/50 = 20%)
    let mut timings = super::super::super::ModuleTimings::default();
    let _result = service.run_cycle_init(&mut timings);
    // During warmup, LR is scaled by 0.2 + 0.8 * progress, then clamped
    // and multiplied by circadian plasticity. It should be less than or
    // equal to the base learning rate.
    assert!(
        service.stats.adaptive_learning_rate <= base_lr + 0.001,
        "LR during warmup ({}) should not exceed base ({})",
        service.stats.adaptive_learning_rate,
        base_lr
    );
    assert!(
        service.stats.adaptive_learning_rate >= 0.0001,
        "LR should not go below minimum clamp"
    );
}

#[test]
fn test_run_cycle_init_exploration_urge_suppressed_during_warmup() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.behavior.curiosity_drive.exploration_urge = 0.8;
    service.stats.total_cycles = 10; // 10/50 = 0.2 progress
    let mut timings = super::super::super::ModuleTimings::default();
    let result = service.run_cycle_init(&mut timings);
    // exploration_urge_start should capture the value AFTER suppression
    // Original 0.8 * 0.2 = 0.16
    assert!(
        result.exploration_urge_start < 0.8,
        "Exploration urge should be suppressed during warmup, got {}",
        result.exploration_urge_start
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// compute_urgency_and_error_pattern tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_urgency_warmup_pattern_for_short_history() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 100;
    let threshold = service.config.learning_threshold;
    // With no error history, pattern should be "Warmup"
    let result = service.compute_urgency_and_error_pattern(0.01, false, threshold);
    assert_eq!(
        result.error_pattern, "Warmup",
        "Short error history should yield Warmup pattern"
    );
    assert_eq!(
        result.predicted_urgency, "Normal",
        "Warmup pattern should predict Normal urgency"
    );
}

#[test]
fn test_urgency_consecutive_low_error_tracking() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 100;
    let threshold = service.config.learning_threshold;
    // Reset consecutive counter
    service.carryover.urgency.consecutive_low_error = 0;
    // Low error below threshold should increment consecutive counter
    let low_error = threshold * 0.5;
    let _result = service.compute_urgency_and_error_pattern(low_error, false, threshold);
    assert!(
        service.carryover.urgency.consecutive_low_error > 0,
        "Low error should increment consecutive_low_error"
    );
}

#[test]
fn test_urgency_consecutive_low_error_resets_on_high_error() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 100;
    let threshold = service.config.learning_threshold;
    service.carryover.urgency.consecutive_low_error = 20;
    // High error above threshold should reset consecutive counter
    let high_error = threshold * 2.0;
    let _result = service.compute_urgency_and_error_pattern(high_error, false, threshold);
    assert_eq!(
        service.carryover.urgency.consecutive_low_error, 0,
        "High error should reset consecutive_low_error"
    );
}

#[test]
fn test_urgency_mode_transition_increments_stat() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 100;
    let threshold = service.config.learning_threshold;
    service.carryover.urgency.prev_urgency = super::super::super::CycleUrgency::Cruise;
    service.carryover.urgency.consecutive_low_error = 0;
    let transitions_before = service.stats.mode_transitions;
    // Trigger a Normal urgency (default cognitive depth = Cortical, high enough error)
    let _result = service.compute_urgency_and_error_pattern(threshold * 1.5, false, threshold);
    // Since prev_urgency was Cruise and the new one is likely Normal/Critical,
    // a mode transition should have been counted
    assert!(
        service.stats.mode_transitions > transitions_before,
        "Mode transition count should increment when urgency changes"
    );
}

#[test]
fn test_urgency_stable_pattern_from_constant_errors() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 100;
    let threshold = service.config.learning_threshold;
    // Push 5 identical low errors to create a stable pattern
    for _ in 0..5 {
        service.carryover.history.error_history.push_back(0.05);
    }
    let result = service.compute_urgency_and_error_pattern(0.05, false, threshold);
    // With constant errors, slope should be near-zero: pattern = "Stable"
    assert_eq!(
        result.error_pattern, "Stable",
        "Constant errors should yield Stable pattern, got {}",
        result.error_pattern
    );
}

#[test]
fn test_urgency_prediction_coherence_bias_low_coherence() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 100;
    let threshold = service.config.learning_threshold;
    // Set low prediction coherence to trigger the bias
    service.stats.avg_prediction_coherence = 0.2; // < 0.3 and > 0.0
    let result = service.compute_urgency_and_error_pattern(0.05, false, threshold);
    // coherence_mod = 0.85, bias = 0.85 - 1.0 = -0.15
    assert!(
        (result.prediction_coherence_urgency_bias - (-0.15)).abs() < 0.01,
        "Low coherence should produce negative bias, got {}",
        result.prediction_coherence_urgency_bias
    );
}

#[test]
fn test_urgency_prediction_coherence_bias_high_coherence() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 100;
    let threshold = service.config.learning_threshold;
    // Set high prediction coherence
    service.stats.avg_prediction_coherence = 0.8; // > 0.7
    let result = service.compute_urgency_and_error_pattern(0.05, false, threshold);
    // coherence_mod = 1.1, bias = 1.1 - 1.0 = 0.1
    assert!(
        (result.prediction_coherence_urgency_bias - 0.1).abs() < 0.01,
        "High coherence should produce positive bias, got {}",
        result.prediction_coherence_urgency_bias
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// run_parameter_optimization_phase tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_parameter_optimization_skips_non_500_cycles() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 42; // not divisible by 500
    let result = service.run_parameter_optimization_phase();
    assert!(
        (result.best_tau_scale - 1.0).abs() < f32::EPSILON,
        "Should return default tau scale on non-500 cycles"
    );
    assert!(
        (result.phi_gain).abs() < f64::EPSILON,
        "Should return zero phi gain on non-500 cycles"
    );
    assert!(
        !result.swap_occurred,
        "No swap should occur on non-500 cycles"
    );
}

#[test]
fn test_parameter_optimization_runs_on_500_cycles_no_episodes() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 500; // divisible by 500
    let result = service.run_parameter_optimization_phase();
    // With no phi_episodic_replay or empty episodes, should return defaults
    assert!(
        (result.best_tau_scale - 1.0).abs() < f32::EPSILON,
        "Should return default tau scale with no episodes"
    );
    assert!(!result.swap_occurred, "No swap with no episodes");
}

// ═══════════════════════════════════════════════════════════════════════════
// run_end_of_cycle_stats tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_end_of_cycle_stats_accumulates_promotions() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    let initial_promotions = service.stats.resonator_promotions_total;
    let initial_evictions = service.stats.codebook_evictions_total;
    let mut metadata = super::super::super::CycleMetadata::default();
    service.run_end_of_cycle_stats(
        &mut metadata,
        false, // resonator_wm_primed
        5,     // resonator_promotions
        2,     // codebook_evictions
        0.6,   // codebook_diversity
        0.5,   // fep_surprise
        0.3,   // surprise_thresh (surprise > thresh -> boost counted)
        0.0,   // neuromod_attention_alloc
        0,     // phasic_da_replay_boost
        0.0,   // ne_reorienting_boost
        0.0,   // ne_arousal_feedback
        0.0,   // confidence_velocity
        0.0,   // sht_crash_dip
        0.0,   // exploration_sht_drain
    );
    assert_eq!(
        service.stats.resonator_promotions_total,
        initial_promotions + 5,
        "Promotions should accumulate"
    );
    assert_eq!(
        service.stats.codebook_evictions_total,
        initial_evictions + 2,
        "Evictions should accumulate"
    );
}

#[test]
fn test_end_of_cycle_stats_codebook_diversity_updated() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    let mut metadata = super::super::super::CycleMetadata::default();
    service.run_end_of_cycle_stats(
        &mut metadata,
        false,
        0,
        0,
        0.85, // codebook_diversity > 0.0 -> should update
        0.0,
        0.3,
        0.0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    assert!(
        (service.stats.codebook_diversity - 0.85).abs() < f32::EPSILON,
        "Codebook diversity should be updated to 0.85, got {}",
        service.stats.codebook_diversity
    );
}

#[test]
fn test_end_of_cycle_stats_zero_diversity_not_stored() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    service.stats.codebook_diversity = 0.5; // pre-existing value
    let mut metadata = super::super::super::CycleMetadata::default();
    service.run_end_of_cycle_stats(
        &mut metadata,
        false,
        0,
        0,
        0.0, // codebook_diversity == 0.0 -> should NOT update
        0.0,
        0.3,
        0.0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    assert!(
        (service.stats.codebook_diversity - 0.5).abs() < f32::EPSILON,
        "Zero codebook diversity should not overwrite existing value"
    );
}

#[test]
fn test_end_of_cycle_stats_surprise_replay_boost_counted() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    let initial_boosts = service.stats.fep_surprise_replay_boosts;
    let mut metadata = super::super::super::CycleMetadata::default();
    service.run_end_of_cycle_stats(
        &mut metadata,
        false,
        0,
        0,
        0.0,
        0.8, // fep_surprise
        0.3, // surprise_thresh -- surprise > thresh -> count incremented
        0.0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    assert_eq!(
        service.stats.fep_surprise_replay_boosts,
        initial_boosts + 1,
        "FEP surprise replay boost should be counted when surprise > thresh"
    );
}

#[test]
fn test_end_of_cycle_stats_wm_primed_counted() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    let initial = service.stats.resonator_wm_primed_count;
    let mut metadata = super::super::super::CycleMetadata::default();
    service.run_end_of_cycle_stats(
        &mut metadata,
        true, // resonator_wm_primed
        0,
        0,
        0.0,
        0.0,
        0.3,
        0.0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    assert_eq!(
        service.stats.resonator_wm_primed_count,
        initial + 1,
        "WM primed count should increment"
    );
}

#[test]
fn test_end_of_cycle_stats_neuromod_ema_updates() {
    let mut service =
        CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
    let mut metadata = super::super::super::CycleMetadata::default();
    service.run_end_of_cycle_stats(
        &mut metadata,
        false,
        0,
        0,
        0.0,
        0.0,
        0.3,
        0.0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    // EMA with alpha=0.05 should produce finite values
    assert!(
        service.stats.avg_dopamine.is_finite(),
        "avg_dopamine should be finite after EMA update"
    );
    assert!(
        service.stats.avg_noradrenaline.is_finite(),
        "avg_noradrenaline should be finite after EMA update"
    );
    assert!(
        service.stats.avg_serotonin.is_finite(),
        "avg_serotonin should be finite after EMA update"
    );
    assert!(
        service.stats.avg_acetylcholine.is_finite(),
        "avg_acetylcholine should be finite after EMA update"
    );
}
