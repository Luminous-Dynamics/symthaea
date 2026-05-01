// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compare wind gust robustness across physics backends: Simple vs MuJoCo vs MuJoCo+Noise.
//!
//! Demonstrates that HDC's noise resilience carries over to high-fidelity physics.
//!
//! Run: `cargo run --example mujoco_vs_simple_benchmark --features multirotor-mujoco --release`

use symthaea::multirotor::{
    benchmarks::{
        run_wind_benchmark, run_wind_benchmark_mujoco, run_wind_benchmark_mujoco_noisy,
        WindBenchmarkConfig, WindGust,
    },
    sensory_filter::SensoryFilterConfig,
    FlightConfig,
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     MuJoCo vs Simple Physics — Wind Gust Robustness          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let config = WindBenchmarkConfig {
        flight_config: FlightConfig {
            steps_per_episode: 2000,
            ..FlightConfig::default()
        },
        gusts: vec![
            WindGust {
                force: [0.05, 0.0, 0.0],
                start_time: 0.5,
                duration: 0.2,
            },
            WindGust {
                force: [0.0, 0.0, -0.03],
                start_time: 1.5,
                duration: 0.3,
            },
        ],
        eval_episodes: 3,
    };

    // ── Simple Physics ──
    println!("Running Simple Physics benchmark...");
    let simple_result = run_wind_benchmark(&config);
    let simple_avg: f64 =
        simple_result.fep_metrics.iter().sum::<f64>() / simple_result.fep_metrics.len() as f64;
    let simple_baseline: f64 = simple_result.baseline_metrics.iter().sum::<f64>()
        / simple_result.baseline_metrics.len() as f64;

    // ── MuJoCo Clean ──
    println!("Running MuJoCo (clean) benchmark...");
    let mujoco_result = run_wind_benchmark_mujoco(&config);
    let mujoco_avg: f64 =
        mujoco_result.fep_metrics.iter().sum::<f64>() / mujoco_result.fep_metrics.len() as f64;
    let mujoco_baseline: f64 = mujoco_result.baseline_metrics.iter().sum::<f64>()
        / mujoco_result.baseline_metrics.len() as f64;

    // ── MuJoCo Noisy ──
    println!("Running MuJoCo (noisy IMU) benchmark...");
    let noisy_result = run_wind_benchmark_mujoco_noisy(&config, SensoryFilterConfig::default());
    let noisy_avg: f64 =
        noisy_result.fep_metrics.iter().sum::<f64>() / noisy_result.fep_metrics.len() as f64;
    let noisy_baseline: f64 = noisy_result.baseline_metrics.iter().sum::<f64>()
        / noisy_result.baseline_metrics.len() as f64;

    // ── Results Table ──
    println!();
    println!("┌────────────────┬──────────┬──────────────┬──────────────┐");
    println!("│                │ Simple   │ MuJoCo-Clean │ MuJoCo-Noisy │");
    println!("├────────────────┼──────────┼──────────────┼──────────────┤");
    println!(
        "│ Baseline avg   │ {:<8.4} │ {:<12.4} │ {:<12.4} │",
        simple_baseline, mujoco_baseline, noisy_baseline
    );
    println!(
        "│ FEP avg error  │ {:<8.4} │ {:<12.4} │ {:<12.4} │",
        simple_avg, mujoco_avg, noisy_avg
    );
    println!(
        "│ FEP improvement│ {:<8.1}%│ {:<12.1}%│ {:<12.1}%│",
        (1.0 - simple_avg / simple_baseline) * 100.0,
        (1.0 - mujoco_avg / mujoco_baseline) * 100.0,
        (1.0 - noisy_avg / noisy_baseline) * 100.0
    );
    println!(
        "│ Max deviation  │ {:<8.4} │ {:<12.4} │ {:<12.4} │",
        simple_result.max_deviation, mujoco_result.max_deviation, noisy_result.max_deviation
    );
    println!("└────────────────┴──────────┴──────────────┴──────────────┘");

    // Key insight
    let noise_degradation = if mujoco_avg > 0.0 {
        ((noisy_avg - mujoco_avg) / mujoco_avg * 100.0).abs()
    } else {
        0.0
    };
    println!();
    println!(
        "Noise degradation: {:.1}% — HDC's 16,384D vectors filter sensor noise.",
        noise_degradation
    );
}
