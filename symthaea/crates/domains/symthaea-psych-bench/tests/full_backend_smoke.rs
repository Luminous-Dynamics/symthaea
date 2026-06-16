// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Smoke tests for the full Symthaea backend.
//!
//! Runs all 22 benchmarks with the real ContinuousMind backend
//! and verifies no panics, all metrics finite.
#![cfg(feature = "symthaea-backend")]

use symthaea_psych_bench::benchmarks::butlin::ButlinIndicatorSuite;
use symthaea_psych_bench::benchmarks::cogbench::{
    BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
    ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, TemporalDiscountingBenchmark,
    TwoStepBenchmark,
};
use symthaea_psych_bench::benchmarks::memory_agent::{
    AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
    TestTimeLearningBenchmark,
};
use symthaea_psych_bench::benchmarks::tombench::{
    FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
    StrangeStoryBenchmark,
};
use symthaea_psych_bench::benchmarks::worm::{
    BindingBenchmark, ChangeDetectionBenchmark, NBackBenchmark, SerialRecallBenchmark,
    SpatialUpdatingBenchmark,
};
use symthaea_psych_bench::harness::report::BenchmarkResult;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

fn smoke_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        ..Default::default()
    }
}

fn assert_metrics_finite(result: &BenchmarkResult) {
    for (key, val) in &result.metrics {
        assert!(
            val.mean.is_finite(),
            "{}: metric '{}' mean is not finite: {}",
            result.benchmark,
            key,
            val.mean
        );
        assert!(
            val.std_dev.is_finite(),
            "{}: metric '{}' std_dev is not finite: {}",
            result.benchmark,
            key,
            val.std_dev
        );
    }
}

macro_rules! full_smoke_test {
    ($name:ident, $bench:expr) => {
        #[test]
        fn $name() {
            let config = smoke_config();
            let result = $bench.run(&config);
            assert!(
                !result.metrics.is_empty(),
                "{} produced no metrics",
                result.benchmark
            );
            assert_metrics_finite(&result);
        }
    };
}

// WorM
full_smoke_test!(full_nback, NBackBenchmark);
full_smoke_test!(full_change_detection, ChangeDetectionBenchmark);
full_smoke_test!(full_serial_recall, SerialRecallBenchmark);
full_smoke_test!(full_spatial_updating, SpatialUpdatingBenchmark);
full_smoke_test!(full_binding, BindingBenchmark);

// CogBench
full_smoke_test!(
    full_probabilistic_reasoning,
    ProbabilisticReasoningBenchmark
);
full_smoke_test!(full_horizon, HorizonBenchmark);
full_smoke_test!(full_restless_bandit, RestlessBanditBenchmark);
full_smoke_test!(full_instrumental, InstrumentalLearningBenchmark);
full_smoke_test!(full_two_step, TwoStepBenchmark);
full_smoke_test!(full_temporal_discounting, TemporalDiscountingBenchmark);
full_smoke_test!(full_bart, BartBenchmark);

// Butlin
full_smoke_test!(full_butlin, ButlinIndicatorSuite);

// ToMBench
full_smoke_test!(full_false_belief, FalseBeliefBenchmark);
full_smoke_test!(full_faux_pas, FauxPasBenchmark);
full_smoke_test!(full_persuasion, PersuasionBenchmark);
full_smoke_test!(full_strange_story, StrangeStoryBenchmark);
full_smoke_test!(full_hinting, HintingBenchmark);

// MemoryAgent
full_smoke_test!(full_accurate_retrieval, AccurateRetrievalBenchmark);
full_smoke_test!(full_test_time_learning, TestTimeLearningBenchmark);
full_smoke_test!(full_long_range, LongRangeBenchmark);
full_smoke_test!(full_conflict_resolution, ConflictResolutionBenchmark);
