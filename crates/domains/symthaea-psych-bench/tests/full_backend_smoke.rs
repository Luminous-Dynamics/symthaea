// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feature-enabled benchmark implementation smoke tests.
//!
//! Historical note: this target is named `full_backend_smoke`, but it does **not**
//! instantiate `CognitiveLoopBenchmarkRunner`, `ContinuousMind`, or another live
//! cognitive-loop backend. Its actual theorem is narrower:
//!
//! - compile the psych-bench crate with `symthaea-backend` enabled;
//! - call each listed benchmark's ordinary `PsychBenchmark::run()` implementation;
//! - require non-empty, finite metrics.
//!
//! A passing target therefore establishes feature-compatible benchmark execution,
//! not that 22 tasks were solved through the live Symthaea cognitive backend.
//! Live-backend evidence belongs to `harness::live_runner` / explicit execution
//! contracts and must be qualified separately.
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

macro_rules! feature_smoke_test {
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
feature_smoke_test!(feature_nback, NBackBenchmark);
feature_smoke_test!(feature_change_detection, ChangeDetectionBenchmark);
feature_smoke_test!(feature_serial_recall, SerialRecallBenchmark);
feature_smoke_test!(feature_spatial_updating, SpatialUpdatingBenchmark);
feature_smoke_test!(feature_binding, BindingBenchmark);

// CogBench
feature_smoke_test!(
    feature_probabilistic_reasoning,
    ProbabilisticReasoningBenchmark
);
feature_smoke_test!(feature_horizon, HorizonBenchmark);
feature_smoke_test!(feature_restless_bandit, RestlessBanditBenchmark);
feature_smoke_test!(feature_instrumental, InstrumentalLearningBenchmark);
feature_smoke_test!(feature_two_step, TwoStepBenchmark);
feature_smoke_test!(feature_temporal_discounting, TemporalDiscountingBenchmark);
feature_smoke_test!(feature_bart, BartBenchmark);

// Butlin
feature_smoke_test!(feature_butlin, ButlinIndicatorSuite);

// ToMBench
feature_smoke_test!(feature_false_belief, FalseBeliefBenchmark);
feature_smoke_test!(feature_faux_pas, FauxPasBenchmark);
feature_smoke_test!(feature_persuasion, PersuasionBenchmark);
feature_smoke_test!(feature_strange_story, StrangeStoryBenchmark);
feature_smoke_test!(feature_hinting, HintingBenchmark);

// MemoryAgent
feature_smoke_test!(feature_accurate_retrieval, AccurateRetrievalBenchmark);
feature_smoke_test!(feature_test_time_learning, TestTimeLearningBenchmark);
feature_smoke_test!(feature_long_range, LongRangeBenchmark);
feature_smoke_test!(feature_conflict_resolution, ConflictResolutionBenchmark);
