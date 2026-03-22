// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Smoke test: run each benchmark's simplest task with minimal trials.
//! Verifies no panics and all metrics are finite.

use symthaea_psych_bench::benchmarks::affect::{
    MoodCongruentRecallBenchmark, ValenceClassificationBenchmark,
};
use symthaea_psych_bench::benchmarks::attention::VisualSearchBenchmark;
use symthaea_psych_bench::benchmarks::butlin::ButlinIndicatorSuite;
use symthaea_psych_bench::benchmarks::cogbench::{
    BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
    ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, ReversalLearningBenchmark,
    TemporalDiscountingBenchmark, TwoStepBenchmark,
};
use symthaea_psych_bench::benchmarks::creativity::{
    AlternateUsesBenchmark, RemoteAssociatesBenchmark,
};
use symthaea_psych_bench::benchmarks::executive::{
    FlankerBenchmark, IowaGamblingBenchmark, RavensProgressiveMatricesBenchmark, StroopBenchmark,
    TowerOfLondonBenchmark, WisconsinCardSortingBenchmark,
};
use symthaea_psych_bench::benchmarks::inhibition::StopSignalBenchmark;
use symthaea_psych_bench::benchmarks::memory_agent::{
    AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
    TestTimeLearningBenchmark,
};
use symthaea_psych_bench::benchmarks::metacognition::{
    FeelingOfKnowingBenchmark, MetacognitiveCalibrationBenchmark,
};
use symthaea_psych_bench::benchmarks::tombench::{
    FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
    StrangeStoryBenchmark,
};
use symthaea_psych_bench::benchmarks::worm::{
    BindingBenchmark, ChangeDetectionBenchmark, DigitSpanBenchmark, NBackBenchmark,
    SerialRecallBenchmark, SpatialUpdatingBenchmark,
};
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

fn smoke_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        ..Default::default()
    }
}

fn assert_metrics_finite(result: &symthaea_psych_bench::harness::BenchmarkResult) {
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
            "{}: metric '{}' std_dev is not finite",
            result.benchmark,
            key,
        );
    }
}

#[test]
fn smoke_worm_nback() {
    let result = NBackBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_worm_change_detection() {
    let result = ChangeDetectionBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_worm_serial_recall() {
    let result = SerialRecallBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_worm_spatial_updating() {
    let result = SpatialUpdatingBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_worm_binding() {
    let result = BindingBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_cogbench_probabilistic() {
    let result = ProbabilisticReasoningBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_cogbench_horizon() {
    let result = HorizonBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_cogbench_restless_bandit() {
    let result = RestlessBanditBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_cogbench_instrumental() {
    let result = InstrumentalLearningBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_cogbench_two_step() {
    let result = TwoStepBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_cogbench_temporal_discounting() {
    let result = TemporalDiscountingBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_cogbench_bart() {
    let result = BartBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_butlin_indicators() {
    let result = ButlinIndicatorSuite.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
    // All 14 indicators should be evaluated
    let total = result.metrics["present_count"].mean
        + result.metrics["partial_count"].mean
        + result.metrics["absent_count"].mean;
    assert_eq!(total, 14.0);
}

#[test]
fn smoke_tombench_false_belief() {
    let result = FalseBeliefBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_tombench_faux_pas() {
    let result = FauxPasBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_tombench_persuasion() {
    let result = PersuasionBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_tombench_strange_story() {
    let result = StrangeStoryBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_tombench_hinting() {
    let result = HintingBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_memory_accurate_retrieval() {
    let result = AccurateRetrievalBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_memory_test_time_learning() {
    let result = TestTimeLearningBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_memory_long_range() {
    let result = LongRangeBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_memory_conflict_resolution() {
    let result = ConflictResolutionBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_worm_digit_span() {
    let result = DigitSpanBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_cogbench_reversal_learning() {
    let result = ReversalLearningBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_executive_wcst() {
    let result = WisconsinCardSortingBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_executive_igt() {
    let result = IowaGamblingBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_executive_ravens() {
    let result = RavensProgressiveMatricesBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_executive_stroop() {
    let result = StroopBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_executive_flanker() {
    let result = FlankerBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_executive_tower_of_london() {
    let result = TowerOfLondonBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_metacognitive_calibration() {
    let result = MetacognitiveCalibrationBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_affect_valence_classification() {
    let result = ValenceClassificationBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_affect_mood_congruent_recall() {
    let result = MoodCongruentRecallBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_creativity_remote_associates() {
    let result = RemoteAssociatesBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_creativity_alternate_uses() {
    let result = AlternateUsesBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_inhibition_stop_signal() {
    let result = StopSignalBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_attention_visual_search() {
    let result = VisualSearchBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}

#[test]
fn smoke_metacognition_feeling_of_knowing() {
    let result = FeelingOfKnowingBenchmark.run(&smoke_config());
    assert!(!result.metrics.is_empty());
    assert_metrics_finite(&result);
}
