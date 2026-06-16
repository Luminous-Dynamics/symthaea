// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mathematics benchmark battery.
//!
//! Runs all 10 mathematics benchmarks, computes NormativeReport,
//! and emits JSON z-scores to stdout.
//!
//! Usage:
//!   math_battery [--seed N] [--trials N]
//!
//! Output (stdout):
//!   [{"benchmark":"Mathematics::ArithmeticWordProblems","metric":"accuracy","z_score":1.23}, ...]

use symthaea_psych_bench::benchmarks::mathematics::{
    ArithmeticWordProblemsBenchmark, BayesianReasoningBenchmark, ConstraintPuzzlesBenchmark,
    DefiniteIntegralsBenchmark, LinearSystemSolvingBenchmark, LogicalDeductionBenchmark,
    MatrixOperationsBenchmark, PolynomialRootsBenchmark, ProofConstructionBenchmark,
    StatisticalInferenceBenchmark,
};
use symthaea_psych_bench::harness::{
    BenchmarkConfig, BenchmarkReport, NormativeReport, PsychBenchmark,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let seed = args
        .windows(2)
        .find(|w| w[0] == "--seed")
        .and_then(|w| w[1].parse::<u64>().ok())
        .unwrap_or(42);

    let trials = args
        .windows(2)
        .find(|w| w[0] == "--trials")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(5);

    let config = BenchmarkConfig {
        seed,
        trials_per_condition: trials,
        ..Default::default()
    };

    // All 10 mathematics benchmarks
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(ArithmeticWordProblemsBenchmark),
        Box::new(LinearSystemSolvingBenchmark),
        Box::new(PolynomialRootsBenchmark),
        Box::new(DefiniteIntegralsBenchmark),
        Box::new(MatrixOperationsBenchmark),
        Box::new(StatisticalInferenceBenchmark),
        Box::new(BayesianReasoningBenchmark),
        Box::new(LogicalDeductionBenchmark),
        Box::new(ConstraintPuzzlesBenchmark),
        Box::new(ProofConstructionBenchmark),
    ];

    eprintln!(
        "Running {} mathematics benchmarks (seed={}, trials={})...",
        benchmarks.len(),
        seed,
        trials
    );

    let mut report = BenchmarkReport::new();
    for bench in &benchmarks {
        let result = bench.run(&config);
        eprintln!("  {} — {} metrics", result.benchmark, result.metrics.len());
        report.add(result);
    }

    let normative = NormativeReport::from_report(&report);

    // Emit JSON z-scores
    #[derive(serde::Serialize)]
    struct ZScoreEntry {
        benchmark: String,
        metric: String,
        z_score: f64,
    }

    let entries: Vec<ZScoreEntry> = normative
        .scores
        .iter()
        .map(|s| ZScoreEntry {
            benchmark: s.benchmark.clone(),
            metric: s.metric.clone(),
            z_score: s.z_score,
        })
        .collect();

    // Domain composite: compute grand mean z-score
    eprintln!("\n=== Mathematics Domain Summary ===");

    // Grand mean
    if !entries.is_empty() {
        let grand_z: f64 = entries.iter().map(|e| e.z_score).sum::<f64>() / entries.len() as f64;
        eprintln!(
            "\nGrand mean z = {:+.3} across {} metrics",
            grand_z,
            entries.len()
        );
    }

    let json = serde_json::to_string_pretty(&entries).expect("JSON serialization should not fail");
    println!("{json}");
}
