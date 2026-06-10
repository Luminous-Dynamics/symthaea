// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Integration Test: Cross-Benchmark Transfer & Interference
// ==================================================================================
//
// Data structure tests run without features.
// Live cognitive loop tests require `symthaea-backend` feature and are #[ignore].
// ==================================================================================

use symthaea_psych_bench::harness::transfer::{TransferDirection, TransferMatrix, TransferScore};

// ── Transfer Score Computation ──────────────────────────────────────

#[test]
fn test_transfer_score_classification() {
    // Positive transfer
    let pos = TransferScore::compute("A", "B", 0.60, 0.75);
    assert_eq!(pos.direction, TransferDirection::PositiveTransfer);
    assert!(pos.transfer_index > 0.05);

    // Negative interference
    let neg = TransferScore::compute("A", "B", 0.80, 0.60);
    assert_eq!(neg.direction, TransferDirection::NegativeInterference);
    assert!(neg.transfer_index < -0.05);

    // Neutral
    let neut = TransferScore::compute("A", "B", 0.80, 0.82);
    assert_eq!(neut.direction, TransferDirection::Neutral);
}

// ── Transfer Matrix ─────────────────────────────────────────────────

#[test]
fn test_transfer_matrix_aggregation() {
    let mut matrix = TransferMatrix::new();
    matrix.add(TransferScore::compute("Stroop", "WCST", 0.60, 0.75));
    matrix.add(TransferScore::compute("NBack", "DigitSpan", 0.85, 0.60));
    matrix.add(TransferScore::compute("GoNoGo", "StopSignal", 0.70, 0.71));

    assert_eq!(matrix.positive_transfers().len(), 1);
    assert_eq!(matrix.negative_interferences().len(), 1);

    let md = matrix.to_markdown();
    assert!(md.contains("Stroop"));
    assert!(md.contains("+"));
    assert!(md.contains("-"));
    assert!(md.contains("="));
}

// ── Default Transfer Pairs ──────────────────────────────────────────

#[test]
fn test_default_transfer_pairs() {
    let pairs = symthaea_psych_bench::harness::transfer::default_transfer_pairs();
    assert_eq!(pairs.len(), 10);

    // Verify a few known pairs
    assert!(pairs.contains(&("Executive::Stroop", "Executive::WCST")));
    assert!(pairs.contains(&("WorM::N-back", "WorM::DigitSpan")));
    assert!(pairs.contains(&("Motor::FittsLaw", "Motor::Bimanual")));
}

// ── Live Transfer Measurement (requires symthaea-backend) ───────────

#[cfg(feature = "symthaea-backend")]
mod live_tests {
    use super::*;
    use symthaea_core::hdc::ContinuousHV;
    use symthaea_psych_bench::harness::config::BenchmarkConfig;
    use symthaea_psych_bench::harness::live_runner::{
        CognitiveLoopBenchmarkRunner, LoopDrivable, LoopStimulus,
    };

    fn bench_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    struct TrainingBenchmark;

    impl LoopDrivable for TrainingBenchmark {
        fn loop_name(&self) -> &str {
            "Transfer::Training"
        }
        fn generate_stimuli(&self, config: &BenchmarkConfig) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let pattern = ContinuousHV::random(dim, 100);
            let distractor = ContinuousHV::random(dim, 200);
            (0..30)
                .map(|i| LoopStimulus {
                    stimulus_hv: pattern.clone(),
                    alternatives: vec![distractor.clone(), pattern.clone()],
                    correct_idx: 1,
                    condition: "training".to_string(),
                    trial_idx: i,
                })
                .collect()
        }
    }

    struct TestBenchmark;

    impl LoopDrivable for TestBenchmark {
        fn loop_name(&self) -> &str {
            "Transfer::Test"
        }
        fn generate_stimuli(&self, config: &BenchmarkConfig) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let probe = ContinuousHV::random(dim, 300);
            let distractor = ContinuousHV::random(dim, 400);
            (0..10)
                .map(|i| LoopStimulus {
                    stimulus_hv: probe.clone(),
                    alternatives: vec![distractor.clone(), probe.clone()],
                    correct_idx: 1,
                    condition: "test".to_string(),
                    trial_idx: i,
                })
                .collect()
        }
    }

    #[test]
    #[ignore = "requires full cognitive loop infrastructure"]
    fn test_live_transfer_measurement() {
        let config = bench_config();

        let mut runner_baseline =
            CognitiveLoopBenchmarkRunner::new("transfer-baseline").expect("runner");
        let baseline_result = runner_baseline.run_benchmark(&TestBenchmark, &config);

        let mut runner_transfer =
            CognitiveLoopBenchmarkRunner::new("transfer-trained").expect("runner");
        let _training_result = runner_transfer.run_benchmark(&TrainingBenchmark, &config);
        let post_result = runner_transfer.run_benchmark(&TestBenchmark, &config);

        let score = TransferScore::compute(
            "Transfer::Training",
            "Transfer::Test",
            baseline_result.accuracy,
            post_result.accuracy,
        );

        assert!(score.transfer_index.is_finite());
        assert!(score.baseline >= 0.0 && score.baseline <= 1.0);
        assert!(score.performance_after >= 0.0 && score.performance_after <= 1.0);
    }
}
