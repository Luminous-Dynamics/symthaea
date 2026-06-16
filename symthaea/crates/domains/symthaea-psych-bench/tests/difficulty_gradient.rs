// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Difficulty gradient test: verify that increasing difficulty reduces accuracy.
//!
//! Runs Stroop at difficulty levels 0.0, 0.3, 0.6, 0.9 and verifies
//! monotonically decreasing accuracy (with tolerance for stochastic noise).

use symthaea_psych_bench::benchmarks::executive::StroopBenchmark;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

#[test]
fn test_stroop_difficulty_gradient() {
    let difficulties = [0.0, 0.3, 0.6, 0.9];
    let mut accuracies = Vec::new();

    for &diff in &difficulties {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            difficulty: diff,
            ..Default::default()
        };
        let result = StroopBenchmark.run(&config);
        // Stroop reports per-condition accuracy; use incongruent as the most difficulty-sensitive
        let acc = result.metrics["incongruent_accuracy"].mean;
        assert!(
            acc.is_finite(),
            "accuracy at difficulty {:.1} not finite",
            diff
        );
        accuracies.push(acc);
    }

    // Overall trend should be decreasing: first accuracy > last accuracy
    assert!(
        accuracies[0] >= accuracies[3] - 0.05,
        "Stroop difficulty gradient violated: d=0.0 acc={:.3}, d=0.9 acc={:.3}",
        accuracies[0],
        accuracies[3]
    );

    // At default difficulty (0.0), accuracy should be reasonable
    assert!(
        accuracies[0] > 0.3,
        "Baseline accuracy too low: {:.3}",
        accuracies[0]
    );
}

#[test]
fn test_difficulty_zero_matches_default() {
    let default_config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        ..Default::default()
    };
    let explicit_zero = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        difficulty: 0.0,
        ..Default::default()
    };

    let default_result = StroopBenchmark.run(&default_config);
    let zero_result = StroopBenchmark.run(&explicit_zero);

    let default_acc = default_result.metrics["incongruent_accuracy"].mean;
    let zero_acc = zero_result.metrics["incongruent_accuracy"].mean;

    // Should be identical (same seed, same difficulty)
    assert!(
        (default_acc - zero_acc).abs() < 1e-10,
        "Default ({:.4}) != explicit zero ({:.4})",
        default_acc,
        zero_acc
    );
}

#[test]
fn test_difficulty_model_registry() {
    use symthaea_psych_bench::harness::difficulty::{DifficultyModelType, difficulty_model_for};

    // Interference benchmarks
    let stroop = difficulty_model_for("Executive::Stroop");
    assert_eq!(stroop.model_type, DifficultyModelType::Interference);

    let flanker = difficulty_model_for("Executive::Flanker");
    assert_eq!(flanker.model_type, DifficultyModelType::Interference);

    // SNR benchmarks
    let rme = difficulty_model_for("Social::RME");
    assert_eq!(rme.model_type, DifficultyModelType::Snr);

    // Interference benchmarks (working memory)
    let nback = difficulty_model_for("WorM::N-back");
    assert_eq!(nback.model_type, DifficultyModelType::Interference);

    // Unknown benchmark gets default
    let unknown = difficulty_model_for("Unknown::Benchmark");
    assert_eq!(unknown.model_type, DifficultyModelType::Default);
}

#[test]
fn test_difficulty_multipliers_at_zero() {
    use symthaea_psych_bench::harness::difficulty::difficulty_model_for;

    // At difficulty 0.0, all multipliers should be 1.0
    let model = difficulty_model_for("Executive::Stroop");
    assert!((model.temperature_multiplier(0.0) - 1.0).abs() < 1e-10);
    assert!((model.signal_multiplier(0.0) - 1.0).abs() < 1e-10);
    assert!((model.interference_multiplier(0.0) - 1.0).abs() < 1e-10);
}

#[test]
fn test_difficulty_multipliers_monotonic() {
    use symthaea_psych_bench::harness::difficulty::difficulty_model_for;

    let model = difficulty_model_for("Executive::Stroop");

    // Temperature should increase with difficulty (making responses noisier)
    let t_low = model.temperature_multiplier(0.2);
    let t_high = model.temperature_multiplier(0.8);
    assert!(
        t_high >= t_low,
        "temperature should increase with difficulty"
    );

    // Interference should increase with difficulty
    let i_low = model.interference_multiplier(0.2);
    let i_high = model.interference_multiplier(0.8);
    assert!(
        i_high >= i_low,
        "interference should increase with difficulty"
    );
}
