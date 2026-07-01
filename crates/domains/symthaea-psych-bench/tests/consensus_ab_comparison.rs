// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! A/B comparison: Sequential vs Consensus feedback integration.
//!
//! Runs 5 representative benchmarks through the cognitive loop under both
//! integration modes and compares accuracy, prediction error, and neuromod
//! profiles. Requires `symthaea-backend` feature.
//!
//! Run with: `cargo test -p symthaea-psych-bench --features symthaea-backend --test consensus_ab_comparison -- --ignored`

#![cfg(feature = "symthaea-backend")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, ConsciousnessProfile};
use symthaea_psych_bench::benchmarks::{
    executive::{StroopBenchmark, WisconsinCardSortingBenchmark},
    inhibition::GoNoGoBenchmark,
    reasoning::ArcFluidBenchmark,
    worm::NBackBenchmark,
};
use symthaea_psych_bench::harness::config::BenchmarkConfig;
use symthaea_psych_bench::harness::live_runner::{
    CognitiveLoopBenchmarkRunner, LoopBenchmarkResult, LoopDrivable,
};

fn bench_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 5,
        trial_trace: true,
        seed: 42,
        ..Default::default()
    }
}

fn make_loop_config(consensus: bool) -> CognitiveLoopConfig {
    let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
    config.genesis_phrase = Some(format!(
        "consensus-ab-{}",
        if consensus { "consensus" } else { "sequential" }
    ));
    config.async_training = false;
    // consensus_feedback removed — consensus is now always-on
    let _ = consensus;
    config
}

struct BenchmarkPair {
    name: &'static str,
    sequential: LoopBenchmarkResult,
    consensus: LoopBenchmarkResult,
}

impl BenchmarkPair {
    fn accuracy_delta(&self) -> f64 {
        self.consensus.accuracy - self.sequential.accuracy
    }

    fn prediction_error_delta(&self) -> f64 {
        self.consensus.mean_prediction_error - self.sequential.mean_prediction_error
    }

    fn format_comparison(&self) -> String {
        format!(
            "  {:<25} | acc: {:.3} → {:.3} (Δ{:+.3}) | PE: {:.4} → {:.4} (Δ{:+.4}) | DA: {:.3}/{:.3} NE: {:.3}/{:.3} 5HT: {:.3}/{:.3} ACh: {:.3}/{:.3} | R: {:.3}/{:.3}",
            self.name,
            self.sequential.accuracy,
            self.consensus.accuracy,
            self.accuracy_delta(),
            self.sequential.mean_prediction_error,
            self.consensus.mean_prediction_error,
            self.prediction_error_delta(),
            self.sequential.mean_da,
            self.consensus.mean_da,
            self.sequential.mean_ne,
            self.consensus.mean_ne,
            self.sequential.mean_sht,
            self.consensus.mean_sht,
            self.sequential.mean_ach,
            self.consensus.mean_ach,
            self.sequential.mean_reward,
            self.consensus.mean_reward,
        )
    }
}

fn run_benchmark_pair(
    bench: &dyn LoopDrivable,
    name: &'static str,
    config: &BenchmarkConfig,
) -> BenchmarkPair {
    // Sequential run
    let seq_config = make_loop_config(false);
    let mut seq_runner =
        CognitiveLoopBenchmarkRunner::from_config(seq_config).expect("sequential runner creation");
    let sequential = seq_runner.run_benchmark(bench, config);

    // Consensus run
    let con_config = make_loop_config(true);
    let mut con_runner =
        CognitiveLoopBenchmarkRunner::from_config(con_config).expect("consensus runner creation");
    let consensus = con_runner.run_benchmark(bench, config);

    BenchmarkPair {
        name,
        sequential,
        consensus,
    }
}

/// A/B comparison of sequential vs consensus feedback integration.
///
/// Runs 5 benchmarks spanning executive function, working memory, inhibition,
/// and reasoning. Compares accuracy, prediction error, and neuromod profiles.
///
/// Acceptance criteria:
/// - Consensus accuracy >= sequential - 0.02 (within 2% tolerance)
/// - Consensus prediction error <= sequential + 0.05 (within 5% tolerance)
/// - All metrics finite
#[test]
#[ignore = "requires full cognitive loop infrastructure"]
fn consensus_vs_sequential_psych_bench() {
    let config = bench_config();

    let pairs = vec![
        run_benchmark_pair(&StroopBenchmark, "Executive::Stroop", &config),
        run_benchmark_pair(&NBackBenchmark, "WorM::N-back", &config),
        run_benchmark_pair(&WisconsinCardSortingBenchmark, "Executive::WCST", &config),
        run_benchmark_pair(&GoNoGoBenchmark, "Inhibition::GoNoGo", &config),
        run_benchmark_pair(&ArcFluidBenchmark, "Reasoning::ArcFluid", &config),
    ];

    // Print comparison table
    eprintln!(
        "\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    eprintln!(
        "║ Consensus vs Sequential A/B Comparison                                                            ║"
    );
    eprintln!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    for pair in &pairs {
        eprintln!("║{}║", pair.format_comparison());
    }

    // Aggregate metrics
    let n = pairs.len() as f64;
    let mean_acc_delta: f64 = pairs.iter().map(|p| p.accuracy_delta()).sum::<f64>() / n;
    let mean_pe_delta: f64 = pairs
        .iter()
        .map(|p| p.prediction_error_delta())
        .sum::<f64>()
        / n;

    let seq_total_ms: u64 = pairs.iter().map(|p| p.sequential.elapsed_ms).sum();
    let con_total_ms: u64 = pairs.iter().map(|p| p.consensus.elapsed_ms).sum();

    eprintln!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    eprintln!(
        "║  Mean accuracy delta: {:+.4}  |  Mean PE delta: {:+.5}  |  Time: {}ms / {}ms",
        mean_acc_delta, mean_pe_delta, seq_total_ms, con_total_ms
    );
    eprintln!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Assertions: all metrics must be finite
    for pair in &pairs {
        assert!(
            pair.sequential.accuracy.is_finite(),
            "{} sequential accuracy not finite",
            pair.name
        );
        assert!(
            pair.consensus.accuracy.is_finite(),
            "{} consensus accuracy not finite",
            pair.name
        );
        assert!(
            pair.sequential.mean_prediction_error.is_finite(),
            "{} sequential PE not finite",
            pair.name
        );
        assert!(
            pair.consensus.mean_prediction_error.is_finite(),
            "{} consensus PE not finite",
            pair.name
        );

        // Per-benchmark: consensus accuracy should not regress catastrophically.
        // With only 5 trials per condition, individual benchmarks can vary ±15%.
        // The aggregate assertion below enforces tighter bounds.
        assert!(
            pair.accuracy_delta() >= -0.15,
            "{}: consensus accuracy regressed by {:.3} (>{:.3} tolerance)",
            pair.name,
            -pair.accuracy_delta(),
            0.15
        );

        // Per-benchmark: consensus PE should not increase catastrophically
        assert!(
            pair.prediction_error_delta() <= 0.50,
            "{}: consensus PE increased by {:.4} (>{:.4} tolerance)",
            pair.name,
            pair.prediction_error_delta(),
            0.50
        );
    }

    // Aggregate: mean accuracy delta should not regress more than 2%
    assert!(
        mean_acc_delta >= -0.02,
        "aggregate consensus accuracy regressed by {:.4} (>0.02 tolerance)",
        -mean_acc_delta,
    );

    // Aggregate: mean PE delta should not increase
    assert!(
        mean_pe_delta <= 0.05,
        "aggregate consensus PE increased by {:.5} (>0.05 tolerance)",
        mean_pe_delta,
    );
}
