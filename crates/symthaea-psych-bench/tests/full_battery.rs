//! Full psychological benchmark battery with human baseline comparisons.
//!
//! Runs all 35 benchmarks across 9 suites and generates a report comparing
//! Symthaea's performance against published human norms.

use symthaea_psych_bench::benchmarks::{
    affect::{
        EmotionalStroopBenchmark, MoodCongruentRecallBenchmark, ValenceClassificationBenchmark,
    },
    attention::{AttentionalBlinkBenchmark, VisualSearchBenchmark},
    butlin::ButlinIndicatorSuite,
    cogbench::{
        BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
        ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, ReversalLearningBenchmark,
        TemporalDiscountingBenchmark, TwoStepBenchmark,
    },
    creativity::{AlternateUsesBenchmark, RemoteAssociatesBenchmark},
    executive::{
        DualTaskBenchmark, FlankerBenchmark, IowaGamblingBenchmark,
        RavensProgressiveMatricesBenchmark, StroopBenchmark, TowerOfLondonBenchmark,
        WisconsinCardSortingBenchmark,
    },
    inhibition::{GoNoGoBenchmark, StopSignalBenchmark},
    language::GardenPathBenchmark,
    memory_agent::{
        AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
        ProspectiveMemoryBenchmark, TestTimeLearningBenchmark,
    },
    metacognition::{FeelingOfKnowingBenchmark, MetacognitiveCalibrationBenchmark},
    motor::SrttBenchmark,
    reasoning::{
        ArcAbductiveBenchmark, ArcAlgebraBenchmark, ArcAnalogyBenchmark, ArcChainBenchmark,
        ArcCompositionalBenchmark, ArcFewShotBenchmark, ArcFluidBenchmark,
        ArcNoiseBenchmark, ArcRsaBenchmark, ArcScalingBenchmark, ArcStaircaseBenchmark,
    },
    social::RmeBenchmark,
    sustained_attention::SartBenchmark,
    tombench::{
        FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
        StrangeStoryBenchmark,
    },
    worm::{
        BindingBenchmark, ChangeDetectionBenchmark, DigitSpanBenchmark, NBackBenchmark,
        SerialRecallBenchmark, SpatialUpdatingBenchmark,
    },
};
use symthaea_psych_bench::harness::{
    snapshot::{RegressionReport, RegressionSnapshot},
    BenchmarkConfig, BenchmarkReport, PsychBenchmark,
};

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
        time_pressure: 0.0,
        ..Default::default()
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
    report.add(DualTaskBenchmark.run(&config));

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

    // ── Affect ──
    report.add(ValenceClassificationBenchmark.run(&config));
    report.add(MoodCongruentRecallBenchmark.run(&config));
    report.add(EmotionalStroopBenchmark.run(&config));

    // ── Creativity ──
    report.add(RemoteAssociatesBenchmark.run(&config));
    report.add(AlternateUsesBenchmark.run(&config));

    // ── Butlin Consciousness Indicators ──
    report.add(ButlinIndicatorSuite.run(&config));

    // ── Inhibition ──
    report.add(GoNoGoBenchmark.run(&config));
    report.add(StopSignalBenchmark.run(&config));

    // ── Attention ──
    report.add(AttentionalBlinkBenchmark.run(&config));
    report.add(VisualSearchBenchmark.run(&config));

    // ── Reasoning ──
    report.add(ArcFluidBenchmark.run(&config));
    report.add(ArcCompositionalBenchmark.run(&config));
    report.add(ArcAnalogyBenchmark.run(&config));
    report.add(ArcAbductiveBenchmark.run(&config));
    report.add(ArcChainBenchmark.run(&config));
    report.add(ArcNoiseBenchmark.run(&config));
    report.add(ArcFewShotBenchmark.run(&config));
    report.add(ArcScalingBenchmark.run(&config));
    report.add(ArcRsaBenchmark.run(&config));
    report.add(ArcAlgebraBenchmark.run(&config));
    report.add(ArcStaircaseBenchmark.run(&config));

    // ── Additional MemoryAgent ──
    report.add(ProspectiveMemoryBenchmark.run(&config));

    // ── Additional Metacognition ──
    report.add(FeelingOfKnowingBenchmark.run(&config));

    // ── Sustained Attention ──
    report.add(SartBenchmark.run(&config));

    // ── Motor ──
    report.add(SrttBenchmark.run(&config));

    // ── Language ──
    report.add(GardenPathBenchmark.run(&config));

    // ── Social ──
    report.add(RmeBenchmark.run(&config));

    // Verify all benchmarks produced results
    assert_eq!(
        report.results.len(),
        58,
        "Expected 58 benchmark results, got {}",
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

/// Regression guard: compare current results against committed baseline snapshot.
///
/// Fails on >10% degradation (critical) on any metric. Warns at >5%.
/// To regenerate the baseline after intentional changes:
///   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --snapshot baselines/v0.6.0.json
#[test]
fn regression_against_baseline() {
    let baseline_path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/baselines/v0.6.0.json"
    ));

    // Skip gracefully if no baseline snapshot exists yet
    if !baseline_path.exists() {
        eprintln!(
            "WARNING: No baseline snapshot at {}. Skipping regression test.",
            baseline_path.display()
        );
        return;
    }

    let baseline =
        RegressionSnapshot::load(baseline_path).expect("failed to load baseline snapshot");

    // Check staleness (warn if > 30 days old, but don't fail)
    if let Some(warning) = baseline.staleness_warning(30) {
        eprintln!("WARNING: {}", warning);
    }

    // Run the full battery with same config as baseline
    let config = battery_config();
    let mut report = BenchmarkReport::new();

    report.add(NBackBenchmark.run(&config));
    report.add(ChangeDetectionBenchmark.run(&config));
    report.add(SerialRecallBenchmark.run(&config));
    report.add(SpatialUpdatingBenchmark.run(&config));
    report.add(BindingBenchmark.run(&config));
    report.add(DigitSpanBenchmark.run(&config));
    report.add(StroopBenchmark.run(&config));
    report.add(FlankerBenchmark.run(&config));
    report.add(WisconsinCardSortingBenchmark.run(&config));
    report.add(IowaGamblingBenchmark.run(&config));
    report.add(RavensProgressiveMatricesBenchmark.run(&config));
    report.add(TowerOfLondonBenchmark.run(&config));
    report.add(DualTaskBenchmark.run(&config));
    report.add(ProbabilisticReasoningBenchmark.run(&config));
    report.add(HorizonBenchmark.run(&config));
    report.add(RestlessBanditBenchmark.run(&config));
    report.add(InstrumentalLearningBenchmark.run(&config));
    report.add(TwoStepBenchmark.run(&config));
    report.add(TemporalDiscountingBenchmark.run(&config));
    report.add(BartBenchmark.run(&config));
    report.add(ReversalLearningBenchmark.run(&config));
    report.add(FalseBeliefBenchmark.run(&config));
    report.add(FauxPasBenchmark.run(&config));
    report.add(HintingBenchmark.run(&config));
    report.add(PersuasionBenchmark.run(&config));
    report.add(StrangeStoryBenchmark.run(&config));
    report.add(AccurateRetrievalBenchmark.run(&config));
    report.add(TestTimeLearningBenchmark.run(&config));
    report.add(LongRangeBenchmark.run(&config));
    report.add(ConflictResolutionBenchmark.run(&config));
    report.add(MetacognitiveCalibrationBenchmark.run(&config));
    report.add(ValenceClassificationBenchmark.run(&config));
    report.add(MoodCongruentRecallBenchmark.run(&config));
    report.add(EmotionalStroopBenchmark.run(&config));
    report.add(RemoteAssociatesBenchmark.run(&config));
    report.add(AlternateUsesBenchmark.run(&config));
    report.add(ButlinIndicatorSuite.run(&config));
    report.add(GoNoGoBenchmark.run(&config));
    report.add(StopSignalBenchmark.run(&config));
    report.add(AttentionalBlinkBenchmark.run(&config));
    report.add(VisualSearchBenchmark.run(&config));
    report.add(ArcFluidBenchmark.run(&config));
    report.add(ArcCompositionalBenchmark.run(&config));
    report.add(ArcAnalogyBenchmark.run(&config));
    report.add(ArcAbductiveBenchmark.run(&config));
    report.add(ArcChainBenchmark.run(&config));
    report.add(ArcNoiseBenchmark.run(&config));
    report.add(ArcFewShotBenchmark.run(&config));
    report.add(ArcScalingBenchmark.run(&config));
    report.add(ArcRsaBenchmark.run(&config));
    report.add(ArcAlgebraBenchmark.run(&config));
    report.add(ArcStaircaseBenchmark.run(&config));
    report.add(ProspectiveMemoryBenchmark.run(&config));
    report.add(FeelingOfKnowingBenchmark.run(&config));
    report.add(SartBenchmark.run(&config));
    report.add(SrttBenchmark.run(&config));
    report.add(GardenPathBenchmark.run(&config));
    report.add(RmeBenchmark.run(&config));

    let current = RegressionSnapshot::from_report(&report, "current");
    let regression = RegressionReport::compare(&baseline, &current, 0.05, 0.10);

    eprintln!("\n{}\n", regression.format_summary());

    assert!(
        !regression.has_critical(),
        "Critical performance regression detected ({} critical)!\n{}",
        regression.summary.critical,
        regression.format_summary()
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
