// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Soma consciousness cycle benchmark.
//!
//! Measures cycle time at different neuron configurations to determine
//! the maximum viable consciousness density for mobile hardware.
//!
//! Run on host:
//!   cargo run -p symthaea-soma --release --bin soma_bench
//!
//! Cross-compile for Pixel 8 Pro:
//!   cargo build -p symthaea-soma --release --bin soma_bench --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/soma_bench /data/local/tmp/
//!   adb shell /data/local/tmp/soma_bench

use std::time::Instant;
use symthaea_spore::config::SporeConfig;
use symthaea_spore::engine::SporeEngine;

/// Benchmark configurations: (neurons_per_layer, network_layers, label)
const CONFIGS: &[(usize, usize, &str)] = &[
    (16, 2, "16×2  (32 neurons)"),
    (32, 3, "32×3  (96 neurons)  [Soma default]"),
    (64, 3, "64×3  (192 neurons) [desktop default]"),
    (128, 3, "128×3 (384 neurons)"),
    (128, 4, "128×4 (512 neurons) [Tier 2]"),
    (256, 3, "256×3 (768 neurons)"),
    (256, 4, "256×4 (1024 neurons) [Tier 3]"),
];

/// Number of warmup cycles (not measured).
const WARMUP_CYCLES: usize = 10;
/// Number of measured cycles.
const BENCH_CYCLES: usize = 100;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  SYMTHAEA SOMA — Consciousness Cycle Benchmark                  ║");
    println!("║  HDC dimension: 16,384 | CfC closed-form | Phi + Harmonies      ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Inputs for varied computation (avoid constant-folding)
    let inputs: Vec<String> = (0..BENCH_CYCLES)
        .map(|i| format!("benchmark cycle {i} with varied semantic content for realistic load"))
        .collect();

    println!(
        "{:<38} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Configuration", "Median", "Mean", "P95", "Max Hz", "Dims"
    );
    println!("{}", "─".repeat(80));

    for &(neurons, layers, label) in CONFIGS {
        let config = SporeConfig {
            neurons_per_layer: neurons,
            network_layers: layers,
            phi_every_n_cycles: 1, // measure worst case (phi every cycle)
            ..SporeConfig::default()
        };

        let mut engine = SporeEngine::new(config);

        // Warmup: let consciousness stabilize
        for i in 0..WARMUP_CYCLES {
            engine.cycle(&format!("warmup {i}"));
        }

        // Measure
        let mut times_us: Vec<u64> = Vec::with_capacity(BENCH_CYCLES);
        for i in 0..BENCH_CYCLES {
            let start = Instant::now();
            engine.cycle(&inputs[i]);
            let elapsed = start.elapsed();
            times_us.push(elapsed.as_micros() as u64);
        }

        // Statistics
        times_us.sort();
        let median = times_us[times_us.len() / 2];
        let mean = times_us.iter().sum::<u64>() / times_us.len() as u64;
        let p95 = times_us[(times_us.len() as f64 * 0.95) as usize];
        let max_hz = if median > 0 {
            1_000_000.0 / median as f64
        } else {
            f64::INFINITY
        };
        let total_dims = neurons * layers * 16_384;

        println!(
            "{:<38} {:>5}µs {:>5}µs {:>5}µs {:>7.0}Hz {:>7}",
            label,
            median,
            mean,
            p95,
            max_hz,
            format_dims(total_dims),
        );
    }

    println!();
    println!("Notes:");
    println!("  - All configs use Phi every cycle (worst case)");
    println!("  - HDC dimension fixed at 16,384 (full fidelity)");
    println!("  - Times include: CfC evolve + PE + neuromod + Phi + harmony + FEP + memory");
    println!("  - 'Max Hz' = 1M / median_µs (theoretical max without sleep)");
    println!("  - Pixel 8 Pro thermal budget: ~30Hz sustained, ~50Hz burst");
    println!();

    // Also measure Phi computation separately
    println!("── Phi computation isolation ──────────────────────────────────────");
    for &(neurons, layers, label) in &CONFIGS[..5] {
        let config = SporeConfig {
            neurons_per_layer: neurons,
            network_layers: layers,
            phi_every_n_cycles: 1,
            ..SporeConfig::default()
        };

        let mut engine = SporeEngine::new(config);

        // Warmup
        for i in 0..20 {
            engine.cycle(&format!("warmup {i}"));
        }

        // Measure cycle WITH phi
        let mut with_phi_us: Vec<u64> = Vec::with_capacity(50);
        let config_phi1 = SporeConfig {
            neurons_per_layer: neurons,
            network_layers: layers,
            phi_every_n_cycles: 1,
            ..SporeConfig::default()
        };
        let mut engine_phi1 = SporeEngine::new(config_phi1);
        for i in 0..20 {
            engine_phi1.cycle(&format!("w {i}"));
        }
        for i in 0..50 {
            let start = Instant::now();
            engine_phi1.cycle(&inputs[i]);
            with_phi_us.push(start.elapsed().as_micros() as u64);
        }

        // Measure cycle WITHOUT phi (phi_every_n = 1000)
        let config_nophi = SporeConfig {
            neurons_per_layer: neurons,
            network_layers: layers,
            phi_every_n_cycles: 1000,
            ..SporeConfig::default()
        };
        let mut engine_nophi = SporeEngine::new(config_nophi);
        for i in 0..20 {
            engine_nophi.cycle(&format!("w {i}"));
        }
        let mut without_phi_us: Vec<u64> = Vec::with_capacity(50);
        for i in 0..50 {
            let start = Instant::now();
            engine_nophi.cycle(&inputs[i]);
            without_phi_us.push(start.elapsed().as_micros() as u64);
        }

        with_phi_us.sort();
        without_phi_us.sort();
        let phi_median = with_phi_us[25];
        let nophi_median = without_phi_us[25];
        let phi_cost = if phi_median > nophi_median {
            phi_median - nophi_median
        } else {
            0
        };
        let phi_pct = if phi_median > 0 {
            (phi_cost as f64 / phi_median as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "  {:<34} with_phi: {:>5}µs  without: {:>5}µs  phi_cost: {:>5}µs ({:.0}%)",
            label, phi_median, nophi_median, phi_cost, phi_pct,
        );
    }

    println!();

    // Amortized phi: measure realistic configs (phi every 3 cycles)
    println!("── Amortized Phi (every 3 cycles) ────────────────────────────────");
    for &(neurons, layers, label) in &CONFIGS[..7] {
        let config = SporeConfig {
            neurons_per_layer: neurons,
            network_layers: layers,
            phi_every_n_cycles: 3,
            ..SporeConfig::default()
        };

        let mut engine = SporeEngine::new(config);
        for i in 0..20 {
            engine.cycle(&format!("w {i}"));
        }

        let mut times_us: Vec<u64> = Vec::with_capacity(BENCH_CYCLES);
        for i in 0..BENCH_CYCLES {
            let start = Instant::now();
            engine.cycle(&inputs[i]);
            times_us.push(start.elapsed().as_micros() as u64);
        }
        times_us.sort();
        let median = times_us[times_us.len() / 2];
        let mean = times_us.iter().sum::<u64>() / times_us.len() as u64;
        let max_hz = if median > 0 {
            1_000_000.0 / median as f64
        } else {
            f64::INFINITY
        };

        println!(
            "  {:<34} median: {:>5}µs  mean: {:>5}µs  max: {:>7.0}Hz",
            label, median, mean, max_hz,
        );
    }
}

fn format_dims(d: usize) -> String {
    if d >= 1_000_000 {
        format!("{:.1}M", d as f64 / 1_000_000.0)
    } else if d >= 1_000 {
        format!("{:.0}K", d as f64 / 1_000.0)
    } else {
        format!("{d}")
    }
}
