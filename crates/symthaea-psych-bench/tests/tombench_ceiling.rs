// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regression test: verify ToMBench benchmarks do NOT hit accuracy ceilings
//! at intermediate difficulty levels.
//!
//! At difficulty=0.0 (default), near-perfect accuracy is expected.
//! At difficulty>=0.5, accuracy should be meaningfully below 1.0 due to
//! difficulty-gated noise, threshold shifts, and stochastic processing errors.

use symthaea_psych_bench::benchmarks::tombench::*;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

fn config_at_difficulty(difficulty: f64) -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 24,
        seed: 42,
        difficulty,
        ..Default::default()
    }
}

fn get_accuracy(bench: &dyn PsychBenchmark, config: &BenchmarkConfig) -> f64 {
    let result = bench.run(config);
    // Try benchmark-specific accuracy metrics, fall back to overall_accuracy
    for key in [
        "false_belief_accuracy",
        "faux_pas_accuracy",
        "persuasion_detection",
        "overall_accuracy",
    ] {
        if let Some(mv) = result.metrics.get(key) {
            return mv.mean;
        }
    }
    panic!(
        "{}: no accuracy metric found (available: {:?})",
        bench.name(),
        result.metrics.keys().collect::<Vec<_>>()
    );
}

#[test]
fn test_tombench_not_ceiling_at_intermediate_difficulty() {
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(FalseBeliefBenchmark),
        Box::new(FauxPasBenchmark),
        Box::new(PersuasionBenchmark),
        Box::new(StrangeStoryBenchmark),
    ];

    let config_hard = config_at_difficulty(0.5);

    let mut failures = Vec::new();

    for bench in &benchmarks {
        let acc_hard = get_accuracy(bench.as_ref(), &config_hard);

        // At difficulty=0.5, accuracy should be meaningfully below 1.0.
        // We use 0.96 as the ceiling threshold — anything above this at d=0.5
        // indicates the difficulty mechanism isn't working.
        if acc_hard > 0.96 {
            failures.push(format!(
                "{}: accuracy at difficulty=0.5 is {:.3} (ceiling not broken)",
                bench.name(),
                acc_hard,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "ToMBench ceiling effects detected:\n{}",
        failures.join("\n")
    );
}

#[test]
fn test_tombench_difficulty_has_effect() {
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(FalseBeliefBenchmark),
        Box::new(FauxPasBenchmark),
        Box::new(PersuasionBenchmark),
        Box::new(StrangeStoryBenchmark),
    ];

    let config_easy = config_at_difficulty(0.0);
    let config_hard = config_at_difficulty(0.7);

    let mut failures = Vec::new();

    for bench in &benchmarks {
        let acc_easy = get_accuracy(bench.as_ref(), &config_easy);
        let acc_hard = get_accuracy(bench.as_ref(), &config_hard);

        // Difficulty should reduce accuracy: easy >= hard (with tolerance for stochasticity)
        if acc_hard > acc_easy + 0.05 {
            failures.push(format!(
                "{}: harder difficulty INCREASED accuracy (easy={:.3}, hard={:.3})",
                bench.name(),
                acc_easy,
                acc_hard,
            ));
        }

        // The effect should be non-trivial: at least 5 percentage points difference
        let diff = acc_easy - acc_hard;
        if diff < 0.05 && acc_easy > 0.90 {
            failures.push(format!(
                "{}: difficulty effect too small (easy={:.3}, hard={:.3}, diff={:.3})",
                bench.name(),
                acc_easy,
                acc_hard,
                diff,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "ToMBench difficulty effect failures:\n{}",
        failures.join("\n")
    );
}
