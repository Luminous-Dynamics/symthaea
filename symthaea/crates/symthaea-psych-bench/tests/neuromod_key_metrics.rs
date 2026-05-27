// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: verify neuromod benchmarks produce metrics that match
//! key_metric_for_benchmark() routing and baseline comparisons.

use symthaea_psych_bench::benchmarks::neuromod::*;
use symthaea_psych_bench::harness::normative_comparison::NormativeReport;
use symthaea_psych_bench::harness::report::{BenchmarkReport, key_metric_for_benchmark};
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

#[test]
fn test_neuromod_key_metrics_exist() {
    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(RewardLearningBenchmark),
        Box::new(YerkesDodsonBenchmark),
        Box::new(AttentionNetworkBenchmark),
        Box::new(MoodInductionBenchmark),
        Box::new(PharmacologicalAblationBenchmark),
        Box::new(PharmacologicalChallengeBenchmark),
        Box::new(InjectionChallengeBenchmark),
        Box::new(AllostaticStressBenchmark),
    ];

    let mut failures = Vec::new();

    for bench in &benchmarks {
        let result = bench.run(&config);
        let key = key_metric_for_benchmark(bench.name());

        // Key metric must not be the fallback
        if key == "overall_accuracy" {
            failures.push(format!(
                "{}: key_metric_for_benchmark returned fallback 'overall_accuracy'",
                bench.name()
            ));
            continue;
        }

        // Key metric must exist in benchmark output
        if !result.metrics.contains_key(key) {
            failures.push(format!(
                "{}: key metric '{}' not found in output (available: {:?})",
                bench.name(),
                key,
                result.metrics.keys().collect::<Vec<_>>()
            ));
            continue;
        }

        // Key metric value must be finite
        let metric = &result.metrics[key];
        if !metric.mean.is_finite() {
            failures.push(format!(
                "{}: key metric '{}' has non-finite mean: {}",
                bench.name(),
                key,
                metric.mean
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Neuromod key metric failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn test_neuromod_normative_scores() {
    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };

    // Benchmarks with baselines that have SD (can produce z-scores)
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(RewardLearningBenchmark),
        Box::new(YerkesDodsonBenchmark),
        Box::new(AttentionNetworkBenchmark),
        Box::new(MoodInductionBenchmark),
        Box::new(PharmacologicalAblationBenchmark),
    ];

    let mut report = BenchmarkReport::new();
    for bench in &benchmarks {
        report.add(bench.run(&config));
    }

    let normative = NormativeReport::from_report(&report);

    // At least some neuromod benchmarks should produce normative scores
    let neuromod_scores: Vec<_> = normative
        .scores
        .iter()
        .filter(|s| s.benchmark.starts_with("Neuromod"))
        .collect();

    assert!(
        !neuromod_scores.is_empty(),
        "No normative scores produced for neuromod benchmarks. \
         Covered benchmarks: {:?}",
        normative
            .scores
            .iter()
            .map(|s| s.benchmark.as_str())
            .collect::<Vec<_>>()
    );

    // Verify z-scores and percentiles are valid
    for score in &neuromod_scores {
        assert!(
            score.z_score.is_finite(),
            "{}: z-score not finite: {}",
            score.benchmark,
            score.z_score
        );
        assert!(
            score.percentile >= 0.0 && score.percentile <= 100.0,
            "{}: percentile out of range: {}",
            score.benchmark,
            score.percentile
        );
    }
}
