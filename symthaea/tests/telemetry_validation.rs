//! Pipeline telemetry validation tests.
//!
//! Verifies that `CycleMetadata` fields are populated correctly after
//! running the full cognitive pipeline.  Each test builds a fresh
//! `CognitiveLoopService` with synchronous training disabled, feeds
//! rotating text inputs for 50 cycles, then asserts on accumulated
//! telemetry.

use symthaea::cognitive_loop::types::{CycleMetadata, CycleResult, CycleUrgency};
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
        .any(|r| r.metadata.consciousness_level > 0.0);
    assert!(
        any_cl,
        "consciousness_level should be > 0 after 10 cycles"
    );
}

// ========================================================================
// Test 3: Urgency Transitions Occur
// ========================================================================

#[test]
fn urgency_transitions_occur() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    let urgency_levels: std::collections::HashSet<_> =
        results.iter().map(|r| format!("{:?}", r.metadata.urgency)).collect();

    assert!(
        urgency_levels.len() >= 2,
        "Should see at least 2 urgency levels over {N_CYCLES} cycles, got {}: {:?}",
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
        .map(|r| r.metadata.feedback_confidence_proposals)
        .sum();
    let total_lr: u32 = results
        .iter()
        .map(|r| r.metadata.feedback_lr_proposals)
        .sum();

    assert!(
        total_conf > 0,
        "feedback_confidence_proposals total should be > 0"
    );
    assert!(
        total_lr > 0,
        "feedback_lr_proposals total should be > 0"
    );
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
        last.metadata.resonator_codebook_size > 0,
        "resonator_codebook_size should be > 0 after {N_CYCLES} cycles: got {}",
        last.metadata.resonator_codebook_size
    );
    assert!(
        last.metadata.resonator_episodes > 0,
        "resonator_episodes should be > 0 after {N_CYCLES} cycles: got {}",
        last.metadata.resonator_episodes
    );
}

// ========================================================================
// Test 9: Cycle Duration Reasonable
// ========================================================================

#[test]
fn cycle_duration_reasonable() {
    let mut svc = build_service();
    let results = run_cycles(&mut svc, N_CYCLES);

    for (i, r) in results.iter().enumerate() {
        assert!(
            r.metadata.cycle_duration_us > 0,
            "Cycle {i}: cycle_duration_us should be > 0"
        );
        // 500ms = 500_000 us — no single cycle should take this long
        assert!(
            r.metadata.cycle_duration_us < 500_000,
            "Cycle {i}: cycle_duration_us ({}) should be < 500ms",
            r.metadata.cycle_duration_us
        );
    }
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
        let sum = t.core_hdc_encode
            + t.core_cfc_step
            + t.core_predict
            + t.core_training;

        let total = r.metadata.cycle_duration_us;

        // Core timings should not exceed 110% of total (some non-core
        // work happens outside these modules)
        assert!(
            sum <= (total as f64 * 1.1) as u64 + 1,
            "Core timings sum ({sum}us) should be ≤ 110% of total ({total}us)"
        );
    }
}
