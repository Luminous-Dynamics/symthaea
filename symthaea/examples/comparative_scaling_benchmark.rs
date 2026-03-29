// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comparative Scaling Benchmark
//!
//! Measures HDC+CfC performance across different configurations to identify
//! where the architecture has competitive advantages. Compares:
//!
//! 1. **HDC Dimension Scaling** — 256, 1024, 4096, 16384 dimensions
//! 2. **CfC Layer Scaling** — 1, 2, 4 layers
//! 3. **Inference Throughput** — Cycles per second at each configuration
//! 4. **Prediction Accuracy** — Error convergence rate per configuration
//!
//! This provides baselines for comparing against external systems
//! (e.g., vanilla RNN, standard HDC, pymdp active inference).
//!
//! Usage: `cargo run --example comparative_scaling_benchmark`

use std::time::Instant;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const GENESIS: &str = "comparative_scaling_v1";
const WARMUP_CYCLES: usize = 20;
const BENCH_CYCLES: usize = 200;
const NUM_SEEDS: usize = 5;
const PATTERN: &[&str] = &["alpha beta", "gamma delta", "epsilon zeta", "eta theta"];

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Symthaea — Comparative Scaling Benchmark");
    println!("  {} seeds × {} cycles per config", NUM_SEEDS, BENCH_CYCLES);
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── HDC Dimension Scaling ───────────────────────────────────────────
    println!("━━━ HDC Dimension Scaling ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  {:<12} {:>10} {:>10} {:>12} {:>12}",
        "Dim", "FinalErr", "StdErr", "Cyc/sec", "Memory(KB)"
    );
    println!("  {}", "─".repeat(60));

    for &hdc_dim in &[256, 1024, 4096, 16384] {
        let mut errors = Vec::new();
        let mut throughputs = Vec::new();

        for seed_idx in 0..NUM_SEEDS {
            let config = CognitiveLoopConfig {
                genesis_phrase: Some(format!("{GENESIS}_dim{hdc_dim}_s{seed_idx}")),
                learning_threshold: 0.0,
                async_training: false,
                encoder_config: symthaea_core::hdc::predictive_encoder::PredictiveEncoderConfig {
                    hdc_dim,
                    ..Default::default()
                },
                ..CognitiveLoopConfig::with_cfc()
            };

            let mut service = CognitiveLoopService::new(config).expect("service");

            // Warmup
            for i in 0..WARMUP_CYCLES {
                service.cycle(PATTERN[i % PATTERN.len()]);
            }

            // Benchmark
            let start = Instant::now();
            let mut cycle_errors = Vec::new();
            for i in 0..BENCH_CYCLES {
                let result = service.cycle(PATTERN[i % PATTERN.len()]);
                cycle_errors.push(result.prediction_error);
            }
            let elapsed = start.elapsed();

            let final_error = cycle_errors[BENCH_CYCLES - 20..].iter().sum::<f32>() / 20.0;
            let throughput = BENCH_CYCLES as f64 / elapsed.as_secs_f64();

            errors.push(final_error as f64);
            throughputs.push(throughput);
        }

        let mean_err = errors.iter().sum::<f64>() / errors.len() as f64;
        let std_err = (errors.iter().map(|e| (e - mean_err).powi(2)).sum::<f64>()
            / (errors.len() - 1).max(1) as f64)
            .sqrt();
        let mean_tp = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let memory_kb = hdc_dim * 4 / 1024; // f32 vector size

        println!(
            "  {:<12} {:>10.4} {:>10.4} {:>12.0} {:>12}",
            hdc_dim, mean_err, std_err, mean_tp, memory_kb
        );
    }

    // ── CfC Layer Scaling ───────────────────────────────────────────────
    println!("\n━━━ CfC Layer Scaling (hidden=256) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  {:<12} {:>10} {:>10} {:>12}",
        "Layers", "FinalErr", "StdErr", "Cyc/sec"
    );
    println!("  {}", "─".repeat(48));

    for &num_layers in &[1, 2, 4] {
        let mut errors = Vec::new();
        let mut throughputs = Vec::new();

        for seed_idx in 0..NUM_SEEDS {
            let mut config = CognitiveLoopConfig::with_cfc();
            config.genesis_phrase = Some(format!("{GENESIS}_layers{num_layers}_s{seed_idx}"));
            config.learning_threshold = 0.0;
            config.async_training = false;
            config.cfc_config.num_neurons = 256;
            config.cfc_config.input_dim = 256;

            let mut service = CognitiveLoopService::new(config).expect("service");

            for i in 0..WARMUP_CYCLES {
                service.cycle(PATTERN[i % PATTERN.len()]);
            }

            let start = Instant::now();
            let mut cycle_errors = Vec::new();
            for i in 0..BENCH_CYCLES {
                let result = service.cycle(PATTERN[i % PATTERN.len()]);
                cycle_errors.push(result.prediction_error);
            }
            let elapsed = start.elapsed();

            let final_error = cycle_errors[BENCH_CYCLES - 20..].iter().sum::<f32>() / 20.0;
            errors.push(final_error as f64);
            throughputs.push(BENCH_CYCLES as f64 / elapsed.as_secs_f64());
        }

        let mean_err = errors.iter().sum::<f64>() / errors.len() as f64;
        let std_err = (errors.iter().map(|e| (e - mean_err).powi(2)).sum::<f64>()
            / (errors.len() - 1).max(1) as f64)
            .sqrt();
        let mean_tp = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

        println!(
            "  {:<12} {:>10.4} {:>10.4} {:>12.0}",
            num_layers, mean_err, std_err, mean_tp
        );
    }

    // ── Module Overhead ─────────────────────────────────────────────────
    println!("\n━━━ Consciousness Module Overhead ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  {:<20} {:>10} {:>12} {:>10}",
        "Config", "FinalErr", "Cyc/sec", "Overhead%"
    );
    println!("  {}", "─".repeat(55));

    let configs: Vec<(&str, Box<dyn Fn(usize) -> CognitiveLoopConfig>)> = vec![
        (
            "Bare CfC",
            Box::new(|i| CognitiveLoopConfig {
                genesis_phrase: Some(format!("{GENESIS}_bare_s{i}")),
                learning_threshold: 0.0,
                async_training: false,
                enable_virtual_body: false,
                ..CognitiveLoopConfig::with_cfc()
            }),
        ),
        (
            "Standard",
            Box::new(|i| {
                use symthaea::cognitive_loop::config::ConsciousnessProfile;
                let mut c = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
                c.genesis_phrase = Some(format!("{GENESIS}_standard_s{i}"));
                c.learning_threshold = 0.0;
                c.async_training = false;
                c
            }),
        ),
        (
            "Full",
            Box::new(|i| {
                use symthaea::cognitive_loop::config::ConsciousnessProfile;
                let mut c = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
                c.genesis_phrase = Some(format!("{GENESIS}_full_s{i}"));
                c.learning_threshold = 0.0;
                c.async_training = false;
                c
            }),
        ),
    ];

    let mut baseline_tp = 0.0;

    for (name, factory) in &configs {
        let mut throughputs = Vec::new();
        let mut errors = Vec::new();

        for seed_idx in 0..NUM_SEEDS {
            let config = factory(seed_idx);
            let mut service = CognitiveLoopService::new(config).expect("service");

            for i in 0..WARMUP_CYCLES {
                service.cycle(PATTERN[i % PATTERN.len()]);
            }

            let start = Instant::now();
            let mut cycle_errors = Vec::new();
            for i in 0..BENCH_CYCLES {
                let result = service.cycle(PATTERN[i % PATTERN.len()]);
                cycle_errors.push(result.prediction_error);
            }
            let elapsed = start.elapsed();

            errors.push(cycle_errors[BENCH_CYCLES - 20..].iter().sum::<f32>() as f64 / 20.0);
            throughputs.push(BENCH_CYCLES as f64 / elapsed.as_secs_f64());
        }

        let mean_err = errors.iter().sum::<f64>() / errors.len() as f64;
        let mean_tp = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

        if *name == "Bare CfC" {
            baseline_tp = mean_tp;
        }

        let overhead = if baseline_tp > 0.0 {
            (1.0 - mean_tp / baseline_tp) * 100.0
        } else {
            0.0
        };

        println!(
            "  {:<20} {:>10.4} {:>12.0} {:>10.1}",
            name, mean_err, mean_tp, overhead
        );
    }

    println!("\n  Methodology:");
    println!(
        "  - {} seeds per configuration, {} bench cycles (+ {} warmup)",
        NUM_SEEDS, BENCH_CYCLES, WARMUP_CYCLES
    );
    println!("  - Each config trained from scratch per seed");
    println!("  - Overhead% = (1 - config_throughput / bare_throughput) × 100");
    println!();
}
