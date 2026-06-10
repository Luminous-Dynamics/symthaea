// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full-backend comparison: runs all benchmarks with ContinuousMind WM
//! and compares against a baseline snapshot.
//!
//! Requires `--features symthaea-backend`.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --features symthaea-backend \
//!     --example backend_comparison -- baselines/v0.5.1.json

#[cfg(feature = "symthaea-backend")]
fn main() {
    use std::path::PathBuf;
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
        FlankerBenchmark, IowaGamblingBenchmark, RavensProgressiveMatricesBenchmark,
        StroopBenchmark, TowerOfLondonBenchmark, WisconsinCardSortingBenchmark,
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
    use symthaea_psych_bench::harness::report::domain_of;
    use symthaea_psych_bench::harness::{
        BenchmarkConfig, BenchmarkReport, PsychBenchmark, RegressionReport, RegressionSnapshot,
    };

    let args: Vec<String> = std::env::args().collect();
    let baseline_path: PathBuf = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("baselines/v0.5.1.json"));

    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        ..Default::default()
    };

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

    let mut report = BenchmarkReport::new();
    eprintln!(
        "Running {} benchmarks with full ContinuousMind backend...",
        benchmarks.len()
    );
    for bench in &benchmarks {
        eprint!("  {} ... ", bench.name());
        let result = bench.run(&config);
        eprintln!("{}ms", result.elapsed_ms);
        report.add(result);
    }

    // Load baseline and compare
    if baseline_path.exists() {
        let baseline = RegressionSnapshot::load(&baseline_path).expect("failed to load baseline");
        let current = RegressionSnapshot::from_report(&report, "full-backend");
        let regression = RegressionReport::compare(&baseline, &current, 0.05, 0.10);

        println!("\n{}", regression.format_summary());

        // Per-domain delta summary
        println!("\nPer-Domain Delta Summary");
        println!(
            "{:<16} {:>8} {:>8} {:>8}",
            "Domain", "Better", "Same", "Worse"
        );
        println!("{}", "-".repeat(44));

        let mut domain_counts: std::collections::BTreeMap<String, (usize, usize, usize)> =
            std::collections::BTreeMap::new();
        for r in &regression.results {
            let domain = domain_of(&r.benchmark).to_string();
            let entry = domain_counts.entry(domain).or_insert((0, 0, 0));
            if r.delta_pct > 0.02 {
                entry.0 += 1; // better
            } else if r.delta_pct < -0.02 {
                entry.2 += 1; // worse
            } else {
                entry.1 += 1; // same
            }
        }
        for (domain, (better, same, worse)) in &domain_counts {
            println!("{:<16} {:>8} {:>8} {:>8}", domain, better, same, worse);
        }
    } else {
        eprintln!(
            "Baseline not found at {}. Printing summary only.",
            baseline_path.display()
        );
        println!("\n{}", report.format_profile());
    }
}

#[cfg(not(feature = "symthaea-backend"))]
fn main() {
    eprintln!("Backend comparison requires --features symthaea-backend");
    eprintln!(
        "Usage: cargo run -p symthaea-psych-bench --features symthaea-backend --example backend_comparison"
    );
    std::process::exit(1);
}
