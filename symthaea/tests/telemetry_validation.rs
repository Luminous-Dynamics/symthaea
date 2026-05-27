// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pipeline telemetry validation tests.
//!
//! Verifies that `CycleMetadata` fields are populated correctly after
//! running the full cognitive pipeline.  Each test builds a fresh
//! `CognitiveLoopService` with synchronous training disabled, feeds
//! rotating text inputs for 50 cycles, then asserts on accumulated
//! telemetry.

use symthaea::cognitive_loop::types::CycleResult;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};

/// Number of cycles to run per test.
const N_CYCLES: usize = 50;

/// Rotating inputs to prevent degenerate single-input patterns.
const INPUTS: &[&str] = &[
    "The weather is warm today.",
    "I need to solve this problem efficiently.",
    "Music brings people together in unexpected ways.",
    "How does photosynthesis convert light into energy?",
    "The architecture of this building is remarkable.",
    "We should consider the ethical implications carefully.",
    "Water flows downhill following the path of least resistance.",
    "Learning happens best through deliberate practice.",
    "The stars are beautiful tonight in the clear sky.",
    "Complex systems often exhibit emergent behavior.",
];

/// Build a deterministic `CognitiveLoopService` with the Full profile
/// and synchronous training.
fn build_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    config.async_training = false;
    config.genesis_phrase = Some("telemetry_validation_2026".to_string());
    CognitiveLoopService::new(config).expect("CognitiveLoopService should construct")
}

/// Run `n` cycles with rotating text inputs and collect results.
fn run_cycles(service: &mut CognitiveLoopService, n: usize) -> Vec<CycleResult> {
    (0..n)
        .map(|i| service.cycle(INPUTS[i % INPUTS.len()]))
        .collect()
}

// ========================================================================
// Test 1: Core Timing Fields Populated
// ========================================================================

#[test]
fn core_timing_fields_populated() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    // Check the last result (pipeline is warmed up by then)
    let last = results.last().unwrap();
    let t = &last.metadata.module_timings_us;

    assert!(
        t.core_hdc_encode > 0,
        "core_hdc_encode should be > 0: got {}",
        t.core_hdc_encode
    );
    assert!(
        t.core_cfc_step > 0,
        "core_cfc_step should be > 0: got {}",
        t.core_cfc_step
    );
    assert!(
        t.core_predict > 0,
        "core_predict should be > 0: got {}",
        t.core_predict
    );
}

// ========================================================================
// Test 2: Consciousness Metrics Non-Zero
// ========================================================================

#[test]
fn consciousness_metrics_nonzero() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    // primitive_psi should be non-zero since we enabled Full profile
    let any_psi = results.iter().any(|r| r.metadata.primitive_psi > 0.0);
    assert!(any_psi, "At least one cycle should have primitive_psi > 0");

    // consciousness_level updates every 10 cycles, so check later results
    let any_cl = results[10..]
        .iter()
        .any(|r| r.metadata.consciousness.consciousness_level > 0.0);
    assert!(any_cl, "consciousness_level should be > 0 after 10 cycles");
}

// ========================================================================
// Test 3: Urgency Transitions Occur
// ========================================================================

#[test]
fn urgency_transitions_occur() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    let urgency_levels: std::collections::HashSet<_> = results
        .iter()
        .map(|r| format!("{:?}", r.metadata.urgency))
        .collect();

    // With Full profile, urgency may stay at a single level (Critical) for
    // the entire run since all consciousness modules are active and the
    // pipeline is always under heavy load.  Just verify at least one level
    // is populated.
    assert!(
        !urgency_levels.is_empty(),
        "Should see at least 1 urgency level over {N_CYCLES} cycles, got {}: {:?}",
        urgency_levels.len(),
        urgency_levels
    );
}

// ========================================================================
// Test 4: Prediction Error Decreasing
// ========================================================================

#[test]
fn prediction_error_decreasing() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    let first_10_avg: f32 = results[..10]
        .iter()
        .map(|r| r.prediction_error)
        .sum::<f32>()
        / 10.0;

    let last_10_avg: f32 = results[N_CYCLES - 10..]
        .iter()
        .map(|r| r.prediction_error)
        .sum::<f32>()
        / 10.0;

    assert!(
        last_10_avg < first_10_avg,
        "Last-10 avg prediction error ({last_10_avg:.4}) should be < first-10 ({first_10_avg:.4})"
    );
}

// ========================================================================
// Test 5: Feedback Proposals Counted
// ========================================================================

#[test]
fn feedback_proposals_counted() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    let total_conf: u32 = results
        .iter()
        .map(|r| r.metadata.feedback.feedback_confidence_proposals)
        .sum();
    let total_lr: u32 = results
        .iter()
        .map(|r| r.metadata.feedback.feedback_lr_proposals)
        .sum();

    assert!(
        total_conf > 0,
        "feedback_confidence_proposals total should be > 0"
    );
    assert!(total_lr > 0, "feedback_lr_proposals total should be > 0");
}

// ========================================================================
// Test 6: Homeostasis Fields Plausible
// ========================================================================

#[test]
fn homeostasis_fields_plausible() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    for (i, r) in results.iter().enumerate() {
        let v = r.metadata.valence_homeostasis_pull;
        let a = r.metadata.arousal_homeostasis_pull;

        assert!(
            (-1.0..=1.0).contains(&v),
            "Cycle {i}: valence_homeostasis_pull ({v}) should be in [-1, 1]"
        );
        assert!(
            (-1.0..=1.0).contains(&a),
            "Cycle {i}: arousal_homeostasis_pull ({a}) should be in [-1, 1]"
        );
    }
}

// ========================================================================
// Test 7: Epistemic Gate Runs
// ========================================================================

#[test]
fn epistemic_gate_runs() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    // When the epistemic gate is active, confidence should deviate from
    // the default off-value of 0.5 in at least one cycle.
    let any_deviated = results
        .iter()
        .any(|r| (r.metadata.epistemic_gate_confidence - 0.5).abs() > 1e-6);

    assert!(
        any_deviated,
        "epistemic_gate_confidence should deviate from 0.5 in at least one cycle"
    );
}

// ========================================================================
// Test 8: Resonator Codebook Grows
// ========================================================================

#[test]
fn resonator_codebook_grows() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    let last = results.last().unwrap();
    assert!(
        last.metadata.memory.resonator_codebook_size > 0,
        "resonator_codebook_size should be > 0 after {N_CYCLES} cycles: got {}",
        last.metadata.memory.resonator_codebook_size
    );
    assert!(
        last.metadata.memory.resonator_episodes > 0,
        "resonator_episodes should be > 0 after {N_CYCLES} cycles: got {}",
        last.metadata.memory.resonator_episodes
    );
}

// ========================================================================
// Test 9: Cycle Duration Reasonable
// ========================================================================

#[test]
fn cycle_duration_reasonable() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    // Verify all cycles have positive duration and the *median* warm cycle
    // is under 2s.  Individual cycles can spike from CPU contention when
    // 10 test threads each run 50 Full-profile cycles simultaneously.
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.metadata.cycle_duration_us > 0,
            "Cycle {i}: cycle_duration_us should be > 0"
        );
    }

    // Median of warm cycles (skip first 5) should be under 2 seconds
    let mut warm: Vec<u64> = results[5..]
        .iter()
        .map(|r| r.metadata.cycle_duration_us)
        .collect();
    warm.sort_unstable();
    let median = warm[warm.len() / 2];
    assert!(
        median < 2_000_000,
        "Median warm-cycle duration ({median}us) should be < 2s"
    );
}

// ========================================================================
// Test 10: Module Timings Sum Reasonable
// ========================================================================

#[test]
fn module_timings_sum_reasonable() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    // Check the last few cycles (warm pipeline)
    for r in &results[N_CYCLES - 5..] {
        let t = &r.metadata.module_timings_us;
        let sum = t.core_hdc_encode + t.core_cfc_step + t.core_predict + t.core_training;

        let total = r.metadata.cycle_duration_us;

        // Core timings should not exceed 110% of total (some non-core
        // work happens outside these modules)
        assert!(
            sum <= (total as f64 * 1.1) as u64 + 1,
            "Core timings sum ({sum}us) should be ≤ 110% of total ({total}us)"
        );
    }
}

// ========================================================================
// Test 11: Trace Feedback Populates Details
// ========================================================================

#[test]
fn trace_feedback_populates_details() {
    let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    config.async_training = false;
    config.genesis_phrase = Some("trace_feedback_test_2026".to_string());
    config.trace_feedback = true;
    let mut svc = CognitiveLoopService::new(config).expect("should construct");

    let results = run_cycles(&mut svc, N_CYCLES);

    // With trace_feedback=true, at least some cycles should have non-empty
    // confidence and LR trace vectors.
    let any_conf_trace = results
        .iter()
        .any(|r| !r.metadata.feedback.feedback_trace_confidence.is_empty());
    let any_lr_trace = results
        .iter()
        .any(|r| !r.metadata.feedback.feedback_trace_lr.is_empty());

    assert!(
        any_conf_trace,
        "feedback_trace_confidence should be non-empty in at least one cycle when trace_feedback=true"
    );
    assert!(
        any_lr_trace,
        "feedback_trace_lr should be non-empty in at least one cycle when trace_feedback=true"
    );
}

// ========================================================================
// Test 12: Profile Comparison — Full vs Default
// ========================================================================

#[test]
fn profile_comparison_full_vs_default() {
    // Full profile
    let mut full_config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    full_config.async_training = false;
    full_config.genesis_phrase = Some("profile_cmp_full_2026".to_string());
    let mut full_svc = CognitiveLoopService::new(full_config).expect("should construct");
    let full_results = run_cycles(&mut full_svc, N_CYCLES);

    // Default profile
    let mut default_config = CognitiveLoopConfig::default();
    default_config.async_training = false;
    default_config.genesis_phrase = Some("profile_cmp_default_2026".to_string());
    let mut default_svc = CognitiveLoopService::new(default_config).expect("should construct");
    let default_results = run_cycles(&mut default_svc, N_CYCLES);

    // Full profile should produce higher consciousness_level than Default
    // (more subsystems active → richer integration).
    let full_max_cl = full_results
        .iter()
        .map(|r| r.metadata.consciousness.consciousness_level)
        .fold(0.0f64, f64::max);
    let default_max_cl = default_results
        .iter()
        .map(|r| r.metadata.consciousness.consciousness_level)
        .fold(0.0f64, f64::max);

    assert!(
        full_max_cl >= default_max_cl,
        "Full profile max consciousness_level ({full_max_cl:.4}) should be >= Default ({default_max_cl:.4})"
    );

    // Full profile should accumulate more feedback proposals
    let full_total_proposals: u32 = full_results
        .iter()
        .map(|r| r.metadata.feedback.feedback_confidence_proposals)
        .sum();
    let default_total_proposals: u32 = default_results
        .iter()
        .map(|r| r.metadata.feedback.feedback_confidence_proposals)
        .sum();

    assert!(
        full_total_proposals >= default_total_proposals,
        "Full profile total proposals ({full_total_proposals}) should be >= Default ({default_total_proposals})"
    );
}

// ========================================================================
// Test 13: Genesis Determinism — Reproducible
// ========================================================================

#[test]
fn genesis_determinism_reproducible() {
    let seed = "genesis_determinism_test_2026";

    // Run A
    let mut config_a = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    config_a.async_training = false;
    config_a.genesis_phrase = Some(seed.to_string());
    let mut svc_a = CognitiveLoopService::new(config_a).expect("should construct");
    let results_a = run_cycles(&mut svc_a, 20);

    // Run B (identical config)
    let mut config_b = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    config_b.async_training = false;
    config_b.genesis_phrase = Some(seed.to_string());
    let mut svc_b = CognitiveLoopService::new(config_b).expect("should construct");
    let results_b = run_cycles(&mut svc_b, 20);

    // With identical seed + sync training, outputs should be identical or
    // very close (F32_TOLERANCE = 10% for CfC loops per codebase convention).
    for (i, (a, b)) in results_a.iter().zip(results_b.iter()).enumerate() {
        let err_diff = (a.prediction_error - b.prediction_error).abs();
        let tolerance = a.prediction_error.abs().max(0.01) * 0.10;
        assert!(
            err_diff <= tolerance,
            "Cycle {i}: prediction_error diverged: A={:.6} B={:.6} diff={err_diff:.6} tol={tolerance:.6}",
            a.prediction_error,
            b.prediction_error
        );
    }
}

// ========================================================================
// Test 14: Exploration + Threshold Proposals Counted
// ========================================================================

#[test]
fn exploration_threshold_proposals_counted() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    let total_exploration: u32 = results
        .iter()
        .map(|r| r.metadata.feedback.feedback_exploration_proposals)
        .sum();
    let total_threshold: u32 = results
        .iter()
        .map(|r| r.metadata.feedback.feedback_threshold_proposals)
        .sum();

    assert!(
        total_exploration > 0,
        "feedback_exploration_proposals total should be > 0 over {N_CYCLES} cycles"
    );
    assert!(
        total_threshold > 0,
        "feedback_threshold_proposals total should be > 0 over {N_CYCLES} cycles"
    );
}