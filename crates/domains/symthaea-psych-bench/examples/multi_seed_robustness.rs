// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-seed robustness analysis.
//!
//! Runs all benchmarks across multiple seeds and reports mean, SD, and CV
//! per benchmark key metric. Flags benchmarks with CV > 0.20 as UNSTABLE.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example multi_seed_robustness
//!   cargo run -p symthaea-psych-bench --example multi_seed_robustness -- --filter NBack

use symthaea_psych_bench::benchmarks::affect::{
    MoodCongruentRecallBenchmark, ValenceClassificationBenchmark,
};
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
use symthaea_psych_bench::benchmarks::memory_agent::{
    AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
    TestTimeLearningBenchmark,
};
use symthaea_psych_bench::benchmarks::metacognition::MetacognitiveCalibrationBenchmark;
use symthaea_psych_bench::benchmarks::tombench::{
    FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
    StrangeStoryBenchmark,
};
use symthaea_psych_bench::benchmarks::worm::{
    BindingBenchmark, ChangeDetectionBenchmark, DigitSpanBenchmark, NBackBenchmark,
    SerialRecallBenchmark, SpatialUpdatingBenchmark,
};
use symthaea_psych_bench::harness::report::key_metric_for_benchmark;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let filter: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--filter")
        .map(|w| w[1].to_lowercase());

    let seeds: &[u64] = &[42, 123, 456, 789, 1024];

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(NBackBenchmark),
        Box::new(ChangeDetectionBenchmark),
        Box::new(SerialRecallBenchmark),
        Box::new(SpatialUpdatingBenchmark),
        Box::new(BindingBenchmark),
        Box::new(DigitSpanBenchmark),
        Box::new(ProbabilisticReasoningBenchmark),
        Box::new(HorizonBenchmark),
        Box::new(RestlessBanditBenchmark),
        Box::new(InstrumentalLearningBenchmark),
        Box::new(TwoStepBenchmark),
        Box::new(TemporalDiscountingBenchmark),
        Box::new(BartBenchmark),
        Box::new(ReversalLearningBenchmark),
        Box::new(WisconsinCardSortingBenchmark),
        Box::new(IowaGamblingBenchmark),
        Box::new(RavensProgressiveMatricesBenchmark),
        Box::new(StroopBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(TowerOfLondonBenchmark),
        Box::new(MetacognitiveCalibrationBenchmark),
        Box::new(ButlinIndicatorSuite),
        Box::new(FalseBeliefBenchmark),
        Box::new(FauxPasBenchmark),
        Box::new(PersuasionBenchmark),
        Box::new(StrangeStoryBenchmark),
        Box::new(HintingBenchmark),
        Box::new(AccurateRetrievalBenchmark),
        Box::new(TestTimeLearningBenchmark),
        Box::new(LongRangeBenchmark),
        Box::new(ConflictResolutionBenchmark),
        Box::new(ValenceClassificationBenchmark),
        Box::new(MoodCongruentRecallBenchmark),
        Box::new(RemoteAssociatesBenchmark),
        Box::new(AlternateUsesBenchmark),
    ];

    eprintln!(
        "Multi-seed robustness: {} benchmarks x {} seeds",
        benchmarks.len(),
        seeds.len()
    );

    // Header
    println!(
        "| {:30} | {:12} | {:8} | {:8} | {:8} | {} |",
        "Benchmark", "Key Metric", "Mean", "SD", "CV", "Status"
    );
    println!(
        "|{:-<32}|{:-<14}|{:-<10}|{:-<10}|{:-<10}|{:-<10}|",
        "", "", "", "", "", ""
    );

    let mut unstable_count = 0;

    for bench in &benchmarks {
        if let Some(ref f) = filter {
            if !bench.name().to_lowercase().contains(f) {
                continue;
            }
        }

        eprint!("  {} ... ", bench.name());

        let key = key_metric_for_benchmark(bench.name());
        let mut values = Vec::new();

        for &seed in seeds {
            let config = BenchmarkConfig {
                dimension: 256,
                trials_per_condition: 5,
                seed,
                ..Default::default()
            };
            let result = bench.run(&config);
            if let Some(metric) = result.metrics.get(key) {
                values.push(metric.mean);
            }
        }

        if values.is_empty() {
            eprintln!("no metric '{}'", key);
            continue;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let sd = if values.len() > 1 {
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        let cv = if mean.abs() > 1e-15 {
            sd / mean.abs()
        } else {
            0.0
        };

        let status = if cv > 0.20 {
            unstable_count += 1;
            "UNSTABLE"
        } else {
            "stable"
        };

        eprintln!("mean={:.3}, cv={:.3}", mean, cv);

        let short_name = bench.name().split("::").last().unwrap_or(bench.name());
        let short_key = if key.len() > 12 { &key[..12] } else { key };

        println!(
            "| {:30} | {:12} | {:8.3} | {:8.3} | {:8.3} | {:8} |",
            short_name, short_key, mean, sd, cv, status
        );
    }

    if unstable_count > 0 {
        eprintln!(
            "\nWARNING: {} benchmark(s) flagged as UNSTABLE (CV > 0.20)",
            unstable_count
        );
    } else {
        eprintln!("\nAll benchmarks stable across seeds.");
    }
}
