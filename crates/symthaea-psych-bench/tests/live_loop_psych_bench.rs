// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Integration Test: Live Cognitive Loop Psych Benchmarks
// ==================================================================================
//
// Runs psychological benchmarks through the full CognitiveLoopService pipeline.
// These are expensive tests that exercise the real loop — all marked #[ignore].
// Requires `symthaea-backend` feature.
// ==================================================================================

#![cfg(feature = "symthaea-backend")]

use symthaea_core::hdc::ContinuousHV;
use symthaea_psych_bench::harness::config::BenchmarkConfig;
use symthaea_psych_bench::harness::live_runner::{
    CognitiveLoopBenchmarkRunner, LoopDrivable, LoopStimulus,
};

fn bench_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 3,
        trial_trace: true,
        ..Default::default()
    }
}

/// Simple stub benchmark for testing the runner infrastructure.
struct StubBenchmark;

impl LoopDrivable for StubBenchmark {
    fn loop_name(&self) -> &str {
        "Stub::Test"
    }

    fn generate_stimuli(&self, config: &BenchmarkConfig) -> Vec<LoopStimulus> {
        let dim = config.dimension;
        let target = ContinuousHV::random(dim, 42);
        let distractor = ContinuousHV::random(dim, 999);

        (0..10)
            .map(|i| LoopStimulus {
                stimulus_hv: target.clone(),
                alternatives: vec![distractor.clone(), target.clone()],
                correct_idx: 1,
                condition: "stub".to_string(),
                trial_idx: i,
            })
            .collect()
    }
}

// ── Runner Creation ─────────────────────────────────────────────────

#[test]
#[ignore = "requires full cognitive loop infrastructure"]
fn test_loop_runner_creation() {
    let runner = CognitiveLoopBenchmarkRunner::new("psych-bench-test-genesis");
    assert!(
        runner.is_some(),
        "Runner should be creatable with Standard profile"
    );
}

// ── Benchmark Execution ─────────────────────────────────────────────

#[test]
#[ignore = "requires full cognitive loop infrastructure"]
fn test_loop_runner_executes_benchmark() {
    let mut runner =
        CognitiveLoopBenchmarkRunner::new("psych-bench-exec-genesis").expect("runner creation");
    let config = bench_config();
    let result = runner.run_benchmark(&StubBenchmark, &config);

    assert_eq!(result.benchmark, "Stub::Test");
    assert_eq!(result.trials.len(), 10);
    assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
    assert!(result.mean_prediction_error.is_finite());
    assert!(result.mean_da.is_finite());
    assert!(result.mean_ne.is_finite());
    assert!(result.mean_sht.is_finite());
    assert!(result.mean_ach.is_finite());
    assert!(result.elapsed_ms > 0);

    // Each trial should have valid neuromod readings
    for trial in &result.trials {
        assert!(trial.da.is_finite());
        assert!(trial.ne.is_finite());
        assert!(trial.sht.is_finite());
        assert!(trial.ach.is_finite());
        assert!(trial.prediction_error.is_finite());
    }
}

// ── Neuromod Clamping ───────────────────────────────────────────────

#[test]
#[ignore = "requires full cognitive loop infrastructure"]
fn test_loop_runner_neuromod_clamping() {
    let mut runner =
        CognitiveLoopBenchmarkRunner::new("psych-bench-clamp-genesis").expect("runner creation");

    // Clamp DA high
    runner
        .service_mut()
        .clamp_neuromod_levels(Some(0.9), None, None, None);

    let config = bench_config();
    let result = runner.run_benchmark(&StubBenchmark, &config);

    // DA should be elevated (clamped at 0.9)
    let mean_da = result.trials.iter().map(|t| t.da).sum::<f32>() / result.trials.len() as f32;
    assert!(
        mean_da > 0.5,
        "Clamped DA should be elevated, got {:.3}",
        mean_da
    );
}
