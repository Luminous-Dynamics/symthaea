//! Full psychological benchmark battery with human baseline comparisons.
//!
//! Runs all 31 benchmarks across 7 suites and generates a report comparing
//! Symthaea's performance against published human norms.

use symthaea_psych_bench::benchmarks::{
    butlin::ButlinIndicatorSuite,
    cogbench::{
        BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
        ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, ReversalLearningBenchmark,
        TemporalDiscountingBenchmark, TwoStepBenchmark,
    },
    executive::{
        FlankerBenchmark, IowaGamblingBenchmark, RavensProgressiveMatricesBenchmark,
        StroopBenchmark, TowerOfLondonBenchmark, WisconsinCardSortingBenchmark,
    },
    memory_agent::{
        AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
        TestTimeLearningBenchmark,
    },
    metacognition::MetacognitiveCalibrationBenchmark,
    tombench::{
        FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
        StrangeStoryBenchmark,
    },
    worm::{
        BindingBenchmark, ChangeDetectionBenchmark, DigitSpanBenchmark, NBackBenchmark,
        SerialRecallBenchmark, SpatialUpdatingBenchmark,
    },
};
use symthaea_psych_bench::harness::{BenchmarkConfig, BenchmarkReport, PsychBenchmark};

fn battery_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        working_memory_capacity: 7,
        seed: 42,
        enable_social: true,
        enable_fep: true,
        planning_horizon: 3,
        action_temperature: 1.0,
        label: None,
    }
}

/// Run the complete battery and generate a human-baseline comparison report.
#[test]
fn full_battery_report() {
    let config = battery_config();
    let mut report = BenchmarkReport::new();

    // ── Working Memory (WorM) ──
    report.add(NBackBenchmark.run(&config));
    report.add(ChangeDetectionBenchmark.run(&config));
    report.add(SerialRecallBenchmark.run(&config));
    report.add(SpatialUpdatingBenchmark.run(&config));
    report.add(BindingBenchmark.run(&config));
    report.add(DigitSpanBenchmark.run(&config));

    // ── Executive Function ──
    report.add(StroopBenchmark.run(&config));
    report.add(FlankerBenchmark.run(&config));
    report.add(WisconsinCardSortingBenchmark.run(&config));
    report.add(IowaGamblingBenchmark.run(&config));
    report.add(RavensProgressiveMatricesBenchmark.run(&config));
    report.add(TowerOfLondonBenchmark.run(&config));

    // ── CogBench (Cognitive Psychology via FEP) ──
    report.add(ProbabilisticReasoningBenchmark.run(&config));
    report.add(HorizonBenchmark.run(&config));
    report.add(RestlessBanditBenchmark.run(&config));
    report.add(InstrumentalLearningBenchmark.run(&config));
    report.add(TwoStepBenchmark.run(&config));
    report.add(TemporalDiscountingBenchmark.run(&config));
    report.add(BartBenchmark.run(&config));
    report.add(ReversalLearningBenchmark.run(&config));

    // ── Theory of Mind ──
    report.add(FalseBeliefBenchmark.run(&config));
    report.add(FauxPasBenchmark.run(&config));
    report.add(HintingBenchmark.run(&config));
    report.add(PersuasionBenchmark.run(&config));
    report.add(StrangeStoryBenchmark.run(&config));

    // ── Memory Agent Pipeline ──
    report.add(AccurateRetrievalBenchmark.run(&config));
    report.add(TestTimeLearningBenchmark.run(&config));
    report.add(LongRangeBenchmark.run(&config));
    report.add(ConflictResolutionBenchmark.run(&config));

    // ── Metacognition ──
    report.add(MetacognitiveCalibrationBenchmark.run(&config));

    // ── Butlin Consciousness Indicators ──
    report.add(ButlinIndicatorSuite.run(&config));

    // Verify all 31 benchmarks produced results
    assert_eq!(
        report.results.len(),
        31,
        "Expected 31 benchmark results, got {}",
        report.results.len()
    );

    // All metrics must be finite
    for result in &report.results {
        for (key, val) in &result.metrics {
            assert!(
                val.mean.is_finite(),
                "{}: metric '{}' mean is not finite: {}",
                result.benchmark,
                key,
                val.mean
            );
        }
    }

    // Generate and print the full report with baseline comparisons
    let summary = report.summary();
    eprintln!("\n{}\n", summary);

    // Verify report contains baseline comparisons (at least some metrics matched)
    assert!(
        summary.contains("Baseline Comparisons"),
        "Report should contain at least one baseline comparison"
    );
}

/// Verify all baseline categories have at least one comparison.
#[test]
fn baseline_coverage_check() {
    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        ..Default::default()
    };

    let mut report = BenchmarkReport::new();

    // One representative from each baseline category
    report.add(NBackBenchmark.run(&config)); // worm baselines
    report.add(StroopBenchmark.run(&config)); // executive baselines
    report.add(BartBenchmark.run(&config)); // cogbench baselines
    report.add(FalseBeliefBenchmark.run(&config)); // tombench baselines
    report.add(AccurateRetrievalBenchmark.run(&config)); // memory baselines
    report.add(MetacognitiveCalibrationBenchmark.run(&config)); // metacognition baselines

    let summary = report.summary();
    eprintln!("\n{}\n", summary);

    // Count baseline comparison lines
    let comparison_count = summary.matches("% of human").count();
    assert!(
        comparison_count >= 5,
        "Expected at least 5 baseline comparisons, got {}",
        comparison_count
    );
}
