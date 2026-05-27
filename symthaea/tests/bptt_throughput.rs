// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BPTT throughput benchmark for CfC networks.
//!
//! Marked `#[ignore]` so it doesn't run in CI.
//! Run with: cargo test -p symthaea --release --test bptt_throughput -- --ignored --nocapture
//!
//! This benchmark compares the original and optimized BPTT implementations.

use ndarray::Array1;
use std::time::Instant;
use symthaea::dynamics::cfc::{ActivationType, CfCConfig, CfCNetwork, CfCNetworkConfig};
use symthaea_core::genesis::GenesisSeed;

fn make_config() -> CfCNetworkConfig {
    let cell_config = CfCConfig {
        input_dim: 32,
        hidden_dim: 64,
        use_backbone: true,
        backbone_layers: 2,
        backbone_dim: 64,
        activation: ActivationType::SiLU,
        tau_range: (0.1, 10.0),
        dropout: 0.0,
        ..Default::default()
    };
    CfCNetworkConfig {
        input_dim: 32,
        hidden_dim: 64,
        num_layers: 2,
        output_dim: 16,
        cell_config,
        residual: true,
        bidirectional: false,
        ..Default::default()
    }
}

/// Generate random training samples (input, target, dt).
fn generate_samples(n: usize) -> (Vec<Array1<f32>>, Vec<Array1<f32>>, Vec<f32>) {
    let inputs: Vec<Array1<f32>> = (0..n)
        .map(|i| Array1::from_vec((0..32).map(|j| ((i * 7 + j) as f32 * 0.01).sin()).collect()))
        .collect();
    let targets: Vec<Array1<f32>> = (0..n)
        .map(|i| {
            Array1::from_vec(
                (0..16)
                    .map(|j| ((i * 3 + j + 1) as f32 * 0.02).cos())
                    .collect(),
            )
        })
        .collect();
    let dts: Vec<f32> = (0..n).map(|i| 0.01 + (i as f32 * 0.001)).collect();
    (inputs, targets, dts)
}

#[test]
#[ignore = "benchmark: ~10s CfC forward + BPTT training throughput"]
fn bptt_throughput_random_init() {
    let config = make_config();
    let mut net = CfCNetwork::new(config.clone());

    println!("\n========== BPTT Throughput Benchmark ==========");
    println!("Network: input=32, hidden=64, output=16, layers=2, backbone=yes");
    println!("Parameters: {}", net.num_parameters());

    let (inputs, targets, dts) = generate_samples(100);
    let n_steps = 100;

    // --- Forward-only benchmark ---
    net.reset();
    let t0 = Instant::now();
    for i in 0..n_steps {
        let _ = net.forward(&inputs[i % inputs.len()], dts[i % dts.len()]);
    }
    let fwd_elapsed = t0.elapsed();
    let fwd_us = fwd_elapsed.as_micros() as f64 / n_steps as f64;
    println!("\n[Random Init] Forward-only:");
    println!("  Total: {:.2?} for {} steps", fwd_elapsed, n_steps);
    println!("  Per step: {:.1} us", fwd_us);
    println!("  Steps/sec: {:.0}", 1_000_000.0 / fwd_us);

    // --- BPTT training benchmark ---
    net.reset();
    let t0 = Instant::now();
    let mut total_loss = 0.0f32;
    for i in 0..n_steps {
        let idx = i % inputs.len();
        let loss = net
            .train_step_bptt(
                &[inputs[idx].clone()],
                &[targets[idx].clone()],
                &[dts[idx]],
                0.001,
            )
            .unwrap();
        total_loss += loss;
    }
    let bptt_elapsed = t0.elapsed();
    let bptt_us = bptt_elapsed.as_micros() as f64 / n_steps as f64;
    println!("\n[Random Init] BPTT training:");
    println!("  Total: {:.2?} for {} steps", bptt_elapsed, n_steps);
    println!("  Per step: {:.1} us", bptt_us);
    println!("  Steps/sec: {:.0}", 1_000_000.0 / bptt_us);
    println!("  Avg loss: {:.6}", total_loss / n_steps as f32);
    println!(
        "  Real-time feasible (<1ms/step): {}",
        if bptt_us < 1000.0 { "YES" } else { "NO" }
    );

    println!("\n================================================");
}

#[test]
#[ignore = "benchmark: ~10s original vs optimized BPTT comparison"]
fn bptt_throughput_optimized_vs_original() {
    let config = make_config();
    let genesis = GenesisSeed::from_phrase("benchmark-comparison-2026");
    let mut net_orig = CfCNetwork::from_genesis(config.clone(), &genesis, "orig");
    let mut net_opt = CfCNetwork::from_genesis(config.clone(), &genesis, "opt");

    println!("\n======== BPTT Original vs Optimized Comparison ========");
    println!("Network: input=32, hidden=64, output=16, layers=2, backbone=yes");
    println!("Parameters: {}", net_orig.num_parameters());

    let (inputs, targets, dts) = generate_samples(100);
    let n_steps = 100;

    // --- Original BPTT benchmark ---
    net_orig.reset();
    let t0 = Instant::now();
    let mut total_loss_orig = 0.0f32;
    for i in 0..n_steps {
        let idx = i % inputs.len();
        let loss = net_orig
            .train_step_bptt(
                &[inputs[idx].clone()],
                &[targets[idx].clone()],
                &[dts[idx]],
                0.001,
            )
            .unwrap();
        total_loss_orig += loss;
    }
    let orig_elapsed = t0.elapsed();
    let orig_us = orig_elapsed.as_micros() as f64 / n_steps as f64;
    println!("\n[Original] BPTT training:");
    println!("  Total: {:.2?} for {} steps", orig_elapsed, n_steps);
    println!("  Per step: {:.1} us ({:.2} ms)", orig_us, orig_us / 1000.0);
    println!("  Steps/sec: {:.0}", 1_000_000.0 / orig_us);
    println!("  Avg loss: {:.6}", total_loss_orig / n_steps as f32);

    // --- Optimized BPTT benchmark ---
    net_opt.reset();
    let t0 = Instant::now();
    let mut total_loss_opt = 0.0f32;
    for i in 0..n_steps {
        let idx = i % inputs.len();
        let loss = net_opt
            .train_step_bptt_optimized(
                &[inputs[idx].clone()],
                &[targets[idx].clone()],
                &[dts[idx]],
                0.001,
            )
            .unwrap();
        total_loss_opt += loss;
    }
    let opt_elapsed = t0.elapsed();
    let opt_us = opt_elapsed.as_micros() as f64 / n_steps as f64;
    println!("\n[Optimized] BPTT training:");
    println!("  Total: {:.2?} for {} steps", opt_elapsed, n_steps);
    println!("  Per step: {:.1} us ({:.2} ms)", opt_us, opt_us / 1000.0);
    println!("  Steps/sec: {:.0}", 1_000_000.0 / opt_us);
    println!("  Avg loss: {:.6}", total_loss_opt / n_steps as f32);

    // --- Comparison ---
    let speedup = orig_us / opt_us;
    let savings_pct = (1.0 - opt_us / orig_us) * 100.0;
    println!("\n--- Comparison ---");
    println!("  Speedup: {:.2}x", speedup);
    println!("  Time savings: {:.1}%", savings_pct);
    println!(
        "  Original <1ms/step: {}",
        if orig_us < 1000.0 { "YES" } else { "NO" }
    );
    println!(
        "  Optimized <1ms/step: {}",
        if opt_us < 1000.0 { "YES" } else { "NO" }
    );

    // Loss should be similar (within tolerance)
    let loss_diff = (total_loss_orig - total_loss_opt).abs() / total_loss_orig;
    println!("  Loss difference: {:.4}%", loss_diff * 100.0);

    println!("\n=====================================================");
}

#[test]
#[ignore = "benchmark: ~10s CfC BPTT throughput with genesis seeding"]
fn bptt_throughput_genesis_init() {
    let config = make_config();
    let genesis = GenesisSeed::from_phrase("benchmark-seed-2026");
    let mut net = CfCNetwork::from_genesis(config.clone(), &genesis, "bench");

    println!("\n========= BPTT Throughput (Genesis Init) =========");
    println!("Network: input=32, hidden=64, output=16, layers=2, backbone=yes");
    println!("Parameters: {}", net.num_parameters());

    let (inputs, targets, dts) = generate_samples(100);
    let n_steps = 100;

    // --- Forward-only benchmark ---
    net.reset();
    let t0 = Instant::now();
    for i in 0..n_steps {
        let _ = net.forward(&inputs[i % inputs.len()], dts[i % dts.len()]);
    }
    let fwd_elapsed = t0.elapsed();
    let fwd_us = fwd_elapsed.as_micros() as f64 / n_steps as f64;
    println!("\n[Genesis Init] Forward-only:");
    println!("  Total: {:.2?} for {} steps", fwd_elapsed, n_steps);
    println!("  Per step: {:.1} us", fwd_us);
    println!("  Steps/sec: {:.0}", 1_000_000.0 / fwd_us);

    // --- BPTT training benchmark ---
    net.reset();
    let t0 = Instant::now();
    let mut total_loss = 0.0f32;
    for i in 0..n_steps {
        let idx = i % inputs.len();
        let loss = net
            .train_step_bptt(
                &[inputs[idx].clone()],
                &[targets[idx].clone()],
                &[dts[idx]],
                0.001,
            )
            .unwrap();
        total_loss += loss;
    }
    let bptt_elapsed = t0.elapsed();
    let bptt_us = bptt_elapsed.as_micros() as f64 / n_steps as f64;
    println!("\n[Genesis Init] BPTT training:");
    println!("  Total: {:.2?} for {} steps", bptt_elapsed, n_steps);
    println!("  Per step: {:.1} us", bptt_us);
    println!("  Steps/sec: {:.0}", 1_000_000.0 / bptt_us);
    println!("  Avg loss: {:.6}", total_loss / n_steps as f32);
    println!(
        "  Real-time feasible (<1ms/step): {}",
        if bptt_us < 1000.0 { "YES" } else { "NO" }
    );

    println!("\n==================================================");
}