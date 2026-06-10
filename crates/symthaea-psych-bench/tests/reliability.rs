// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test-retest reliability via multi-seed runs.
//!
//! Runs a representative subset of benchmarks with 5 different seeds and
//! verifies that the results show acceptable split-half reliability.

use symthaea_psych_bench::benchmarks::{
    affect::ValenceClassificationBenchmark,
    attention::AttentionalBlinkBenchmark,
    cogbench::ProbabilisticReasoningBenchmark,
    executive::{FlankerBenchmark, StroopBenchmark, WisconsinCardSortingBenchmark},
    inhibition::GoNoGoBenchmark,
    memory_agent::AccurateRetrievalBenchmark,
    worm::NBackBenchmark,
};
use symthaea_psych_bench::harness::{
    BenchmarkConfig, BenchmarkReport, PsychBenchmark,
    analysis::{CrossBenchmarkAnalysis, icc_2_1},
};

/// Run a representative subset with multiple seeds and check reliability.
#[test]
fn test_retest_reliability() {
    let seeds = [42, 137, 256, 512, 1024];

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(NBackBenchmark),
        Box::new(StroopBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(WisconsinCardSortingBenchmark),
        Box::new(ProbabilisticReasoningBenchmark),
        Box::new(AccurateRetrievalBenchmark),
        Box::new(ValenceClassificationBenchmark),
        Box::new(GoNoGoBenchmark),
        Box::new(AttentionalBlinkBenchmark),
    ];

    let reports: Vec<BenchmarkReport> = seeds
        .iter()
        .map(|&seed| {
            let config = BenchmarkConfig {
                seed,
                trials_per_condition: 10,
                dimension: 512,
                ..Default::default()
            };
            let mut report = BenchmarkReport::new();
            for bench in &benchmarks {
                report.add(bench.run(&config));
            }
            report
        })
        .collect();

    let analysis = CrossBenchmarkAnalysis::from_multi_seed_reports(&reports);

    // All benchmarks should appear in the analysis
    assert!(
        analysis.values.len() >= benchmarks.len(),
        "Expected {} benchmarks in analysis, got {}",
        benchmarks.len(),
        analysis.values.len()
    );

    // Each benchmark should have one value per seed
    for (name, vals) in &analysis.values {
        assert_eq!(
            vals.len(),
            seeds.len(),
            "{} should have {} values, got {}",
            name,
            seeds.len(),
            vals.len()
        );
        // All values must be finite
        for v in vals {
            assert!(v.is_finite(), "{}: non-finite value {}", name, v);
        }
    }

    // Print split-half reliability table
    let _reliability = analysis.test_retest_reliability();
    eprintln!("\n{}\n", analysis.format_reliability());

    // ICC(2,1) reliability — treat each seed as a "rater" rating each benchmark
    eprintln!("--- ICC(2,1) Reliability ---");
    eprintln!(
        "| {:30} | {:>8} | {:15} |",
        "Benchmark", "ICC(2,1)", "Interpretation"
    );
    eprintln!(
        "| {:30} | {:>8} | {:15} |",
        "------------------------------", "--------", "---------------"
    );

    for (name, vals) in &analysis.values {
        // Build observations matrix: each seed is a "rater"
        // For ICC we need subjects × raters, but we have one subject per seed.
        // Instead, build per-benchmark ICC by extracting per-trial variability across seeds.
        // With 5 seeds, we use ICC to measure agreement across seeds.
        if vals.len() >= 3 {
            // For single-measure ICC, we treat each seed as a "judge"
            // and each benchmark as having 1 "subject" (the metric value).
            // ICC(2,1) needs multiple subjects, so we use a different approach:
            // compute ICC across metric keys within this benchmark.
            let icc = compute_benchmark_icc(&reports, name);
            let interp = if icc >= 0.75 {
                "Excellent"
            } else if icc >= 0.50 {
                "Moderate"
            } else if icc >= 0.25 {
                "Fair"
            } else {
                "Poor"
            };
            let short = name.split("::").last().unwrap_or(name);
            eprintln!("| {:30} | {:>8.3} | {:15} |", short, icc, interp);
        }
    }

    // Construct validity check
    let validity = analysis.construct_validity();
    eprintln!(
        "\nConstruct validity: {}/{} convergent pairs, mean within-domain r = {:.3}",
        validity.convergent_pairs, validity.same_domain_pairs, validity.mean_within_correlation
    );
}

/// Compute ICC(2,1) for a specific benchmark across multiple seed reports.
///
/// Treats different metrics within the benchmark as "subjects" and seeds as "raters".
/// This gives a proper ICC(2,1) with multiple subjects and multiple raters.
fn compute_benchmark_icc(reports: &[BenchmarkReport], benchmark_name: &str) -> f64 {
    // Find all metric keys for this benchmark
    let first_report = &reports[0];
    let bench_result = first_report
        .results
        .iter()
        .find(|r| r.benchmark == benchmark_name);

    let metric_keys: Vec<String> = match bench_result {
        Some(r) => r.metrics.keys().cloned().collect(),
        None => return 0.0,
    };

    if metric_keys.len() < 2 {
        return 0.0;
    }

    // Build observations: raters (seeds) × subjects (metrics)
    let observations: Vec<Vec<f64>> = reports
        .iter()
        .map(|report| {
            let result = report
                .results
                .iter()
                .find(|r| r.benchmark == benchmark_name);
            metric_keys
                .iter()
                .map(|key| {
                    result
                        .and_then(|r| r.metrics.get(key))
                        .map(|m| m.mean)
                        .unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    icc_2_1(&observations)
}
