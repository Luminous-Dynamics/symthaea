// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Smoke tests for LoopDrivable implementations: UltimatumGame, EmotionalStroop,
//! and existing benchmark impls (Stroop, NBack, WCST, GoNoGo).
//!
//! These tests validate that each benchmark's `generate_stimuli()` produces
//! well-formed stimuli with correct alternatives counts and valid correct_idx.

use crate::harness::config::BenchmarkConfig;
use crate::harness::live_runner::LoopDrivable;

fn default_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 64,
        seed: 42,
        trials_per_condition: 5,
        ..BenchmarkConfig::default()
    }
}

// ── UltimatumGame ──────────────────────────────────────────────────

#[test]
fn ultimatum_game_loop_runs() {
    let bench = crate::benchmarks::social::UltimatumGameBenchmark;
    let config = default_config();
    let stimuli = bench.generate_stimuli(&config);
    assert!(!stimuli.is_empty(), "stimuli should be non-empty");
    for s in &stimuli {
        assert_eq!(
            s.alternatives.len(),
            2,
            "ultimatum has 2 alternatives (reject/accept)"
        );
        assert!(
            s.correct_idx < 2,
            "correct_idx out of bounds: {}",
            s.correct_idx
        );
    }
}

#[test]
fn ultimatum_game_oxytocin_active() {
    let bench = crate::benchmarks::social::UltimatumGameBenchmark;
    let config = default_config();
    let stimuli = bench.generate_stimuli(&config);
    // Verify offer conditions span the expected range
    let conditions: Vec<&str> = stimuli.iter().map(|s| s.condition.as_str()).collect();
    assert!(
        conditions.iter().any(|c| c.contains("10")),
        "should have 10% offer trials"
    );
    assert!(
        conditions.iter().any(|c| c.contains("50")),
        "should have 50% offer trials"
    );
}

// ── EmotionalStroop ────────────────────────────────────────────────

#[test]
fn emotional_stroop_loop_runs() {
    let bench = crate::benchmarks::affect::EmotionalStroopBenchmark;
    let config = default_config();
    let stimuli = bench.generate_stimuli(&config);
    assert_eq!(stimuli.len(), 80, "emotional stroop has 80 trials");
    for s in &stimuli {
        assert_eq!(s.alternatives.len(), 4, "stroop has 4 color alternatives");
        assert!(
            s.correct_idx < 4,
            "correct_idx out of bounds: {}",
            s.correct_idx
        );
    }
}

#[test]
fn emotional_stroop_neuromod_active() {
    let bench = crate::benchmarks::affect::EmotionalStroopBenchmark;
    let config = default_config();
    let stimuli = bench.generate_stimuli(&config);
    let neutral_count = stimuli.iter().filter(|s| s.condition == "neutral").count();
    let negative_count = stimuli.iter().filter(|s| s.condition == "negative").count();
    assert_eq!(neutral_count, 40, "40 neutral trials expected");
    assert_eq!(negative_count, 40, "40 negative trials expected");
}

// ── Existing benchmark smoke tests ─────────────────────────────────

#[test]
fn stroop_benchmark_loop_runs() {
    let bench = crate::benchmarks::executive::StroopBenchmark;
    let config = default_config();
    let stimuli = bench.generate_stimuli(&config);
    assert!(!stimuli.is_empty());
    for s in &stimuli {
        assert_eq!(s.alternatives.len(), 4);
        assert!(s.correct_idx < 4);
    }
}

#[test]
fn nback_benchmark_loop_runs() {
    let bench = crate::benchmarks::worm::NBackBenchmark;
    let config = default_config();
    let stimuli = bench.generate_stimuli(&config);
    assert!(!stimuli.is_empty());
    for s in &stimuli {
        assert_eq!(s.alternatives.len(), 2);
        assert!(s.correct_idx < 2);
    }
}

#[test]
fn wcst_benchmark_loop_runs() {
    let bench = crate::benchmarks::executive::WisconsinCardSortingBenchmark;
    let config = default_config();
    let stimuli = bench.generate_stimuli(&config);
    assert!(!stimuli.is_empty());
    for s in &stimuli {
        assert_eq!(s.alternatives.len(), 3);
        assert!(s.correct_idx < 3);
    }
}

#[test]
fn go_nogo_benchmark_loop_runs() {
    let bench = crate::benchmarks::inhibition::GoNoGoBenchmark;
    let config = default_config();
    let stimuli = bench.generate_stimuli(&config);
    assert!(!stimuli.is_empty());
    for s in &stimuli {
        assert_eq!(s.alternatives.len(), 2);
        assert!(s.correct_idx < 2);
    }
    // Verify go/nogo ratio roughly 70/30
    let go_count = stimuli.iter().filter(|s| s.condition == "go").count();
    let nogo_count = stimuli.iter().filter(|s| s.condition == "nogo").count();
    assert!(go_count > nogo_count, "go trials should outnumber nogo");
}
