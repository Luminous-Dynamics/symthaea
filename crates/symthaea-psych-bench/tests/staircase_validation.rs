// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the adaptive staircase procedure.
//!
//! Verifies that `run_staircase()` converges to a valid threshold on
//! real benchmarks and that all output fields are well-formed.

use symthaea_psych_bench::benchmarks::{
    executive::{FlankerBenchmark, StroopBenchmark},
    worm::NBackBenchmark,
};
use symthaea_psych_bench::harness::{
    BenchmarkConfig, PsychBenchmark,
    staircase::{StaircaseConfig, StaircaseRule, run_staircase},
};

fn base_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 5,
        seed: 42,
        ..Default::default()
    }
}

/// Run staircase on 3 benchmarks and verify threshold is in [0, 1].
#[test]
fn staircase_threshold_in_range() {
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(StroopBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(NBackBenchmark),
    ];
    let staircase_cfg = StaircaseConfig {
        max_reversals: 4,
        max_steps: 15,
        ..Default::default()
    };
    let base = base_config();

    let mut failures = Vec::new();
    for bench in &benchmarks {
        let result = run_staircase(bench.as_ref(), &base, &staircase_cfg);
        if result.threshold < 0.0 || result.threshold > 1.0 {
            failures.push(format!(
                "{}: threshold {:.3} outside [0, 1]",
                result.benchmark, result.threshold
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Staircase threshold range violations:\n  {}",
        failures.join("\n  ")
    );
}

/// Verify all StaircaseResult fields are finite and well-formed.
#[test]
fn staircase_fields_finite() {
    let bench = StroopBenchmark;
    let staircase_cfg = StaircaseConfig {
        max_reversals: 4,
        max_steps: 15,
        ..Default::default()
    };
    let result = run_staircase(&bench, &base_config(), &staircase_cfg);

    assert!(result.threshold.is_finite(), "threshold not finite");
    assert!(result.threshold_se.is_finite(), "threshold_se not finite");
    assert!(
        result.target_accuracy.is_finite(),
        "target_accuracy not finite"
    );
    assert!(
        result.accuracy_at_threshold.is_finite(),
        "accuracy_at_threshold not finite"
    );
    assert!(!result.trajectory.is_empty(), "trajectory is empty");
    assert!(
        result.trajectory.len() <= 15,
        "trajectory exceeds max_steps"
    );

    for step in &result.trajectory {
        assert!(step.difficulty >= 0.0 && step.difficulty <= 1.0);
        assert!(step.accuracy.is_finite());
        assert!(step.direction >= -1 && step.direction <= 1);
    }
}

/// Verify to_markdown() produces expected sections.
#[test]
fn staircase_markdown_format() {
    let bench = FlankerBenchmark;
    let staircase_cfg = StaircaseConfig {
        max_reversals: 4,
        max_steps: 15,
        ..Default::default()
    };
    let result = run_staircase(&bench, &base_config(), &staircase_cfg);
    let md = result.to_markdown();

    assert!(
        md.contains("## Staircase:"),
        "markdown missing header: {}",
        md
    );
    assert!(
        md.contains("Threshold:"),
        "markdown missing threshold: {}",
        md
    );
    assert!(
        md.contains("Reversals:"),
        "markdown missing reversals: {}",
        md
    );
    assert!(
        md.contains("| Step |"),
        "markdown missing trajectory table: {}",
        md
    );
    assert!(
        md.contains("Target accuracy:"),
        "markdown missing target accuracy: {}",
        md
    );
}

/// Verify different staircase rules converge to different target accuracies.
#[test]
fn staircase_rules_produce_different_targets() {
    let bench = StroopBenchmark;
    let base = base_config();

    let rules = [
        (StaircaseRule::OneDownOneUp, 0.500),
        (StaircaseRule::TwoDownOneUp, 0.707),
        (StaircaseRule::ThreeDownOneUp, 0.794),
    ];

    for (rule, expected_target) in &rules {
        let staircase_cfg = StaircaseConfig {
            rule: *rule,
            max_reversals: 4,
            max_steps: 15,
            ..Default::default()
        };
        let result = run_staircase(&bench, &base, &staircase_cfg);
        assert!(
            (result.target_accuracy - expected_target).abs() < 0.01,
            "Rule {:?}: expected target {:.3}, got {:.3}",
            rule,
            expected_target,
            result.target_accuracy
        );
    }
}

/// Verify staircase trajectory shows at least one reversal when run long enough.
#[test]
fn staircase_produces_reversals() {
    let bench = StroopBenchmark;
    let staircase_cfg = StaircaseConfig {
        max_reversals: 4,
        max_steps: 40,
        start_difficulty: 0.7,
        ..Default::default()
    };
    let result = run_staircase(&bench, &base_config(), &staircase_cfg);

    // With 40 steps and start_difficulty=0.7, the staircase should
    // encounter at least 1 reversal (starting mid-range avoids ceiling)
    assert!(
        result.n_reversals >= 1,
        "{}: no reversals in {} steps (trajectory: {:?})",
        result.benchmark,
        result.trajectory.len(),
        result
            .trajectory
            .iter()
            .map(|s| (s.difficulty, s.accuracy, s.direction))
            .collect::<Vec<_>>()
    );
}
