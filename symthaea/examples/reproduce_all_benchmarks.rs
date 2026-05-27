// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paper Reproducibility Script
//!
//! Runs all benchmarks cited in the Hyperdimensional Active Inference paper
//! and outputs a structured table matching paper claims.
//!
//! Usage: `cargo run --example reproduce_all_benchmarks`

use symthaea::benchmarks::fep_temporal_benchmark::{
    FepTemporalBenchmark, FepTemporalBenchmarkConfig, TMazeConfig, run_t_maze,
};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Symthaea v0.5.0 — Paper Benchmark Reproduction Suite");
    println!("═══════════════════════════════════════════════════════════════\n");

    // 1. FEP Temporal Benchmarks
    println!("── FEP Temporal Prediction ──────────────────────────────────");
    run_fep_benchmarks();

    // 2. T-Maze Navigation
    println!("\n── T-Maze Active Inference ──────────────────────────────────");
    run_tmaze_benchmark();

    // 3. Phi Proxy Analysis (topology sweep)
    println!("\n── Phi Proxy Analysis ──────────────────────────────────────");
    run_phi_proxy_analysis();

    // 4. Cognitive Loop Convergence
    println!("\n── Cognitive Loop Convergence ──────────────────────────────");
    run_convergence_test();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Reproduction complete. Compare with paper Table 1-3.");
    println!("═══════════════════════════════════════════════════════════════");
}

fn run_fep_benchmarks() {
    let config = FepTemporalBenchmarkConfig {
        num_cycles: 200,
        warmup_cycles: 20,
        measurement_window: 10,
    };
    let bench = FepTemporalBenchmark::new(config);

    // Sine pattern
    let sine = bench.run_sine_pattern();
    println!(
        "  Sine:  init_err={:.4}  final_err={:.4}  reduction={:+.1}%  coherence={:.3}  {}",
        sine.initial_error,
        sine.final_error,
        sine.error_reduction_pct,
        sine.coherence_stability,
        if sine.passed { "PASS" } else { "FAIL" }
    );

    // Step function
    let config2 = FepTemporalBenchmarkConfig {
        num_cycles: 300,
        warmup_cycles: 20,
        measurement_window: 10,
    };
    let bench2 = FepTemporalBenchmark::new(config2);
    let step = bench2.run_step_function();
    println!(
        "  Step:  init_err={:.4}  final_err={:.4}  reduction={:+.1}%  coherence={:.3}  {}",
        step.initial_error,
        step.final_error,
        step.error_reduction_pct,
        step.coherence_stability,
        if step.passed { "PASS" } else { "FAIL" }
    );

    // Noisy pattern
    let noisy = bench.run_noisy_pattern();
    println!(
        "  Noisy: init_err={:.4}  final_err={:.4}  reduction={:+.1}%  coherence={:.3}  {}",
        noisy.initial_error,
        noisy.final_error,
        noisy.error_reduction_pct,
        noisy.coherence_stability,
        if noisy.passed { "PASS" } else { "FAIL" }
    );
}

fn run_tmaze_benchmark() {
    let config = TMazeConfig {
        num_episodes: 100,
        max_steps: 50,
        warmup_episodes: 10,
    };
    let result = run_t_maze(config);
    println!("  Accuracy:       {:.1}%", result.accuracy * 100.0);
    println!("  Early accuracy: {:.1}%", result.early_accuracy * 100.0);
    println!("  Late accuracy:  {:.1}%", result.late_accuracy * 100.0);
    println!("  Avg steps:      {:.1}", result.avg_steps);
    println!("  Avg pred error:  {:.4}", result.avg_prediction_error);
    println!(
        "  Learning signal: {}",
        if result.late_accuracy > result.early_accuracy {
            "YES"
        } else {
            "marginal"
        }
    );
    println!("  {}", if result.passed { "PASS" } else { "FAIL" });
}

fn run_phi_proxy_analysis() {
    // Run cognitive loop across different configurations and measure unified Phi
    let configs = [
        ("CfC_default", CognitiveLoopConfig::with_cfc()),
        (
            "HdcLtc_default",
            CognitiveLoopConfig::with_hdc_ltc_unified(),
        ),
        ("HdcLtc_fast", CognitiveLoopConfig::with_hdc_ltc_fast()),
    ];

    println!(
        "  {:<20} {:>8} {:>8} {:>8} {:>10}",
        "Config", "Phi_mean", "Phi_std", "Coh_mean", "Cycles/s"
    );

    for (name, config) in &configs {
        let mut service =
            CognitiveLoopService::new(config.clone()).expect("Failed to create service");

        let words = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"];
        let mut phi_values = Vec::new();
        let mut coherence_values = Vec::new();

        let start = std::time::Instant::now();
        for i in 0..100 {
            let _result = service.cycle(words[i % words.len()]);
            phi_values.push(service.unified_psi());
            coherence_values.push(service.temporal_coherence());
        }
        let elapsed = start.elapsed();

        let phi_mean: f64 = phi_values.iter().sum::<f64>() / phi_values.len() as f64;
        let phi_std: f64 = (phi_values
            .iter()
            .map(|v| (v - phi_mean).powi(2))
            .sum::<f64>()
            / phi_values.len() as f64)
            .sqrt();
        let coh_mean: f32 = coherence_values.iter().sum::<f32>() / coherence_values.len() as f32;
        let cycles_per_sec = 100.0 / elapsed.as_secs_f64();

        println!(
            "  {:<20} {:>8.4} {:>8.4} {:>8.4} {:>10.1}",
            name, phi_mean, phi_std, coh_mean, cycles_per_sec
        );
    }
}

fn run_convergence_test() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("Failed to create service");

    let words = ["cause", "effect", "reason", "result"];
    let mut errors = Vec::new();

    for i in 0..500 {
        let result = service.cycle(words[i % words.len()]);
        errors.push(result.prediction_error);
    }

    // Compute error in windows of 50
    let window = 50;
    let windows: Vec<f32> = errors
        .chunks(window)
        .map(|w| w.iter().sum::<f32>() / w.len() as f32)
        .collect();

    println!("  Error trajectory (50-cycle windows):");
    for (i, avg) in windows.iter().enumerate() {
        let bar_len = (*avg * 50.0).min(50.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  [{:>3}-{:>3}] {:.4} {}", i * 50, (i + 1) * 50, avg, bar);
    }

    let initial = windows.first().copied().unwrap_or(1.0);
    let final_val = windows.last().copied().unwrap_or(1.0);
    println!(
        "  Initial: {:.4}  Final: {:.4}  Reduction: {:.1}%",
        initial,
        final_val,
        if initial > 0.0 {
            ((initial - final_val) / initial) * 100.0
        } else {
            0.0
        }
    );
}
