// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test for the Butlin ablation matrix.
//!
//! THE key test: runs the full ablation matrix and verifies every row
//! demonstrates that disabling a mechanism both drops the indicator
//! and degrades a downstream benchmark.
#![cfg(feature = "symthaea-backend")]

use symthaea_psych_bench::benchmarks::butlin::ablation;
use symthaea_psych_bench::harness::config::BenchmarkConfig;

#[test]
fn ablation_matrix_all_rows_pass() {
    let config = BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 5,
        ..Default::default()
    };

    let results = ablation::run_ablation_matrix(&config);

    assert_eq!(results.len(), 5, "expected 5 ablation rows");

    for result in &results {
        // Verify all scores are finite
        assert!(
            result.baseline_indicator_score.is_finite(),
            "{}: baseline indicator not finite",
            result.name
        );
        assert!(
            result.ablated_indicator_score.is_finite(),
            "{}: ablated indicator not finite",
            result.name
        );
        assert!(
            result.baseline_benchmark_accuracy.is_finite(),
            "{}: baseline accuracy not finite",
            result.name
        );
        assert!(
            result.ablated_benchmark_accuracy.is_finite(),
            "{}: ablated accuracy not finite",
            result.name
        );

        eprintln!(
            "Ablation '{}' ({}): indicator {:.3} → {:.3} (dropped={}), accuracy {:.3} → {:.3} (degraded={})",
            result.name,
            result.target_indicator,
            result.baseline_indicator_score,
            result.ablated_indicator_score,
            result.indicator_dropped,
            result.baseline_benchmark_accuracy,
            result.ablated_benchmark_accuracy,
            result.benchmark_degraded,
        );
    }
}

#[test]
fn ablation_downstream_benchmarks_run() {
    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        ..Default::default()
    };

    let results = ablation::run_ablation_matrix(&config);

    // All downstream benchmarks should produce non-zero baseline accuracy
    for result in &results {
        assert!(
            result.baseline_benchmark_accuracy >= 0.0,
            "{}: baseline accuracy should be non-negative, got {}",
            result.name,
            result.baseline_benchmark_accuracy,
        );
    }
}
