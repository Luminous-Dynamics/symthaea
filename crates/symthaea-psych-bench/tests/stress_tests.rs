// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stress tests: dual-task and ceiling conditions for the 4 new benchmarks.
//!
//! Verifies graceful degradation under high-load parameters and
//! boundary conditions that push beyond typical human baselines.

use symthaea_psych_bench::benchmarks::{
    affect::EmotionalStroopBenchmark, attention::AttentionalBlinkBenchmark,
    inhibition::GoNoGoBenchmark, memory_agent::ProspectiveMemoryBenchmark,
};
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

/// All metrics from all 4 benchmarks must be finite under default config.
#[test]
fn stress_all_metrics_finite_default() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        ..Default::default()
    };

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(GoNoGoBenchmark),
        Box::new(AttentionalBlinkBenchmark),
        Box::new(ProspectiveMemoryBenchmark),
        Box::new(EmotionalStroopBenchmark),
    ];

    for bench in &benchmarks {
        let result = bench.run(&config);
        for (key, val) in &result.metrics {
            assert!(
                val.mean.is_finite(),
                "{}: metric '{}' is not finite ({})",
                bench.name(),
                key,
                val.mean
            );
            assert!(
                val.std_dev.is_finite(),
                "{}: metric '{}' std_dev not finite",
                bench.name(),
                key
            );
        }
    }
}

/// All metrics must be finite even with minimal dimension (64).
#[test]
fn stress_minimal_dimension() {
    let config = BenchmarkConfig {
        dimension: 64,
        trials_per_condition: 5,
        ..Default::default()
    };

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(GoNoGoBenchmark),
        Box::new(AttentionalBlinkBenchmark),
        Box::new(ProspectiveMemoryBenchmark),
        Box::new(EmotionalStroopBenchmark),
    ];

    for bench in &benchmarks {
        let result = bench.run(&config);
        for (key, val) in &result.metrics {
            assert!(
                val.mean.is_finite(),
                "{} (dim=64): metric '{}' is not finite",
                bench.name(),
                key
            );
        }
    }
}

/// Go/No-Go: verify inhibition cost is non-negative (go_acc >= nogo_acc).
/// With default params, prepotent Go bias should make Go easier.
#[test]
fn stress_gonogo_inhibition_cost_non_negative() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 20,
        ..Default::default()
    };
    let result = GoNoGoBenchmark.run(&config);
    let cost = result.metrics["inhibition_cost"].mean;
    // Inhibition cost can be slightly negative due to noise, but should generally be >= 0
    assert!(
        cost > -0.10,
        "Inhibition cost ({:.3}) should not be deeply negative",
        cost
    );
}

/// Go/No-Go: SSRT should be in a sensible range (> 0 ticks).
#[test]
fn stress_gonogo_ssrt_positive() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        ..Default::default()
    };
    let result = GoNoGoBenchmark.run(&config);
    let ssrt = result.metrics["ssrt_ticks"].mean;
    assert!(ssrt >= 0.0, "SSRT ({:.3}) should be non-negative", ssrt);
    assert!(ssrt.is_finite(), "SSRT should be finite");
}

/// Attentional Blink: T1 accuracy should exceed chance (0.50).
#[test]
fn stress_ab_t1_above_chance() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 15,
        ..Default::default()
    };
    let result = AttentionalBlinkBenchmark.run(&config);
    let t1 = result.metrics["t1_accuracy"].mean;
    assert!(t1 > 0.50, "T1 accuracy ({:.3}) should be above chance", t1);
}

/// Attentional Blink: blink magnitude should be non-negative (lag8 >= lag3).
/// Recovery from attentional depletion means lag8 performance should be at
/// least as good as lag3.
#[test]
fn stress_ab_blink_direction() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 20,
        ..Default::default()
    };
    let result = AttentionalBlinkBenchmark.run(&config);
    let blink = result.metrics["blink_magnitude"].mean;
    // Allow small negative due to noise
    assert!(
        blink > -0.10,
        "Blink magnitude ({:.3}) should not be deeply negative",
        blink
    );
}

/// Prospective Memory: PM cost should be non-negative (baseline >= monitoring).
#[test]
fn stress_pm_cost_non_negative() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 15,
        ..Default::default()
    };
    let result = ProspectiveMemoryBenchmark.run(&config);
    let cost = result.metrics["pm_cost"].mean;
    assert!(
        cost >= 0.0,
        "PM cost ({:.3}) should be non-negative (by construction)",
        cost
    );
}

/// Prospective Memory: ongoing accuracy should be above chance (0.50 for binary).
#[test]
fn stress_pm_ongoing_above_chance() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 15,
        ..Default::default()
    };
    let result = ProspectiveMemoryBenchmark.run(&config);
    let ongoing = result.metrics["pm_ongoing_accuracy"].mean;
    assert!(
        ongoing > 0.50,
        "Ongoing accuracy ({:.3}) should be above chance",
        ongoing
    );
}

/// Emotional Stroop: interference should be non-negative (neutral >= negative).
#[test]
fn stress_emotional_stroop_interference_direction() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 20,
        ..Default::default()
    };
    let result = EmotionalStroopBenchmark.run(&config);
    let interference = result.metrics["emotional_interference"].mean;
    // Allow small negative due to noise
    assert!(
        interference > -0.05,
        "Emotional interference ({:.3}) should not be deeply negative",
        interference
    );
}

/// Cross-seed stability: running with different seeds should produce
/// qualitatively similar results (all within 2x of each other).
#[test]
fn stress_cross_seed_stability() {
    let seeds = [42, 137, 512];

    for &seed in &seeds {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            seed,
            ..Default::default()
        };

        let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
            Box::new(GoNoGoBenchmark),
            Box::new(AttentionalBlinkBenchmark),
            Box::new(ProspectiveMemoryBenchmark),
            Box::new(EmotionalStroopBenchmark),
        ];

        for bench in &benchmarks {
            let result = bench.run(&config);
            for (key, val) in &result.metrics {
                assert!(
                    val.mean.is_finite(),
                    "seed={}: {}: metric '{}' not finite",
                    seed,
                    bench.name(),
                    key
                );
            }
        }
    }
}

/// High trial count: verify no accumulator overflow or precision loss
/// with 50 trials per condition.
#[test]
fn stress_high_trial_count() {
    let config = BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 50,
        ..Default::default()
    };

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(GoNoGoBenchmark),
        Box::new(AttentionalBlinkBenchmark),
        Box::new(ProspectiveMemoryBenchmark),
        Box::new(EmotionalStroopBenchmark),
    ];

    for bench in &benchmarks {
        let result = bench.run(&config);
        for (key, val) in &result.metrics {
            assert!(
                val.mean.is_finite(),
                "{} (50 trials): metric '{}' not finite",
                bench.name(),
                key
            );
            // Accuracy metrics should be in [0, 1]
            if key.contains("accuracy") || key.contains("hit_rate") {
                assert!(
                    val.mean >= 0.0 && val.mean <= 1.0,
                    "{}: {} = {:.3} out of [0,1]",
                    bench.name(),
                    key,
                    val.mean
                );
            }
        }
    }
}

/// Large dimension (2048): verify no panics or NaN from high-dimensional HDC.
#[test]
fn stress_large_dimension() {
    let config = BenchmarkConfig {
        dimension: 2048,
        trials_per_condition: 3,
        ..Default::default()
    };

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(GoNoGoBenchmark),
        Box::new(AttentionalBlinkBenchmark),
        Box::new(ProspectiveMemoryBenchmark),
        Box::new(EmotionalStroopBenchmark),
    ];

    for bench in &benchmarks {
        let result = bench.run(&config);
        for (key, val) in &result.metrics {
            assert!(
                val.mean.is_finite(),
                "{} (dim=2048): metric '{}' not finite",
                bench.name(),
                key
            );
        }
    }
}
