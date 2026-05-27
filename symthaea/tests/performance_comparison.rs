// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Performance comparison: CfC vs HDC-LTC Bridge, with and without genesis seeding.
//!
//! Run with:
//!   cargo test --release --test performance_comparison -- --ignored --nocapture

use std::time::Instant;

use ndarray::Array1;
use symthaea::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};
use symthaea::hdc_ltc_bridge::{HdcLtcBridge, HdcLtcBridgeConfig};
use symthaea_core::genesis::GenesisSeed;

const CFC_STEPS: usize = 1000;
/// HDC-LTC uses 16,384-dim hypervectors so each step is ~1000x slower than CfC.
/// We use fewer steps to keep total runtime under 5 minutes.
const HDC_STEPS: usize = 50;
const INPUT_DIM: usize = 32;
const HIDDEN_DIM: usize = 64;
const OUTPUT_DIM: usize = 16;
const DT: f32 = 0.02;

fn cfc_config() -> CfCNetworkConfig {
    let cell_config = CfCConfig {
        input_dim: INPUT_DIM,
        hidden_dim: HIDDEN_DIM,
        use_backbone: false,
        ..Default::default()
    };
    CfCNetworkConfig {
        input_dim: INPUT_DIM,
        hidden_dim: HIDDEN_DIM,
        num_layers: 2,
        output_dim: OUTPUT_DIM,
        cell_config,
        residual: true,
        bidirectional: false,
        ..Default::default()
    }
}

fn hdc_config() -> HdcLtcBridgeConfig {
    HdcLtcBridgeConfig {
        input_dim: INPUT_DIM,
        output_dim: OUTPUT_DIM,
        layer_sizes: vec![4, 8, 4],
        ..Default::default()
    }
}

fn hdc_config_fast() -> HdcLtcBridgeConfig {
    HdcLtcBridgeConfig {
        input_dim: INPUT_DIM,
        output_dim: OUTPUT_DIM,
        layer_sizes: vec![4, 8, 4],
        hdc_dim: 2048,
        ..Default::default()
    }
}

/// Benchmark `steps` forward calls, returning total duration.
fn bench_forward<F: FnMut(&Array1<f32>)>(
    label: &str,
    steps: usize,
    warmup: usize,
    input: &Array1<f32>,
    mut step_fn: F,
) -> std::time::Duration {
    for _ in 0..warmup {
        step_fn(input);
    }
    let start = Instant::now();
    for _ in 0..steps {
        step_fn(input);
    }
    let elapsed = start.elapsed();
    let per_step_us = elapsed.as_micros() as f64 / steps as f64;
    println!(
        "  {:<40} {:>10.1} us/step  ({:>10.2} ms / {} steps)",
        label,
        per_step_us,
        elapsed.as_secs_f64() * 1000.0,
        steps,
    );
    elapsed
}

#[test]
#[ignore = "benchmark: ~60s CfC vs HDC-LTC forward-pass comparison, run in release mode"]
fn performance_comparison_cfc_vs_hdc_ltc() {
    let input = Array1::from_vec(vec![0.5f32; INPUT_DIM]);
    let genesis = GenesisSeed::from_phrase("benchmark-seed-2026");

    println!();
    println!("=== CfC vs HDC-LTC Performance Comparison ===");
    println!(
        "  Input dim: {}, Hidden dim: {}, Output dim: {}",
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM
    );
    println!("  CfC steps: {}, HDC-LTC steps: {}", CFC_STEPS, HDC_STEPS);
    println!();

    // ---- CfC ----
    let mut cfc_rand = CfCNetwork::new(cfc_config());
    let cfc_rand_dur = bench_forward("CfC (random init)", CFC_STEPS, 10, &input, |inp| {
        let _ = cfc_rand.forward(inp, DT);
    });

    let mut cfc_gen = CfCNetwork::from_genesis(cfc_config(), &genesis, "bench_cfc");
    let cfc_gen_dur = bench_forward("CfC (genesis init)", CFC_STEPS, 10, &input, |inp| {
        let _ = cfc_gen.forward(inp, DT);
    });

    // ---- HDC-LTC ----
    let mut hdc_rand = HdcLtcBridge::new(hdc_config());
    let hdc_rand_dur = bench_forward(
        "HDC-LTC Bridge (random init)",
        HDC_STEPS,
        2,
        &input,
        |inp| {
            let _ = hdc_rand.forward(inp, DT);
        },
    );

    let mut hdc_gen = HdcLtcBridge::from_genesis(hdc_config(), &genesis);
    let hdc_gen_dur = bench_forward(
        "HDC-LTC Bridge (genesis init)",
        HDC_STEPS,
        2,
        &input,
        |inp| {
            let _ = hdc_gen.forward(inp, DT);
        },
    );

    // ---- HDC-LTC with reduced dimension (2048) ----
    let hdc_fast_steps = 500;
    let mut hdc_fast = HdcLtcBridge::new(hdc_config_fast());
    let hdc_fast_dur = bench_forward(
        "HDC-LTC Bridge (dim=2048)",
        hdc_fast_steps,
        10,
        &input,
        |inp| {
            let _ = hdc_fast.forward(inp, DT);
        },
    );

    let mut hdc_fast_gen = HdcLtcBridge::from_genesis(hdc_config_fast(), &genesis);
    let hdc_fast_gen_dur = bench_forward(
        "HDC-LTC Bridge (dim=2048, genesis)",
        hdc_fast_steps,
        10,
        &input,
        |inp| {
            let _ = hdc_fast_gen.forward(inp, DT);
        },
    );

    // ---- Genesis initialization overhead ----
    println!();
    println!("--- Genesis Initialization Overhead ---");

    let cfc_n = 50;
    let hdc_n = 5; // HDC init is slow due to large projections

    let start = Instant::now();
    for _ in 0..cfc_n {
        let _ = CfCNetwork::new(cfc_config());
    }
    let cfc_rand_init = start.elapsed();

    let start = Instant::now();
    for _ in 0..cfc_n {
        let _ = CfCNetwork::from_genesis(cfc_config(), &genesis, "bench_cfc");
    }
    let cfc_gen_init = start.elapsed();

    let start = Instant::now();
    for _ in 0..hdc_n {
        let _ = HdcLtcBridge::new(hdc_config());
    }
    let hdc_rand_init = start.elapsed();

    let start = Instant::now();
    for _ in 0..hdc_n {
        let _ = HdcLtcBridge::from_genesis(hdc_config(), &genesis);
    }
    let hdc_gen_init = start.elapsed();

    let avg_us = |d: std::time::Duration, n: usize| d.as_micros() as f64 / n as f64;
    println!(
        "  {:<40} {:>10.1} us/init",
        "CfC::new()",
        avg_us(cfc_rand_init, cfc_n)
    );
    println!(
        "  {:<40} {:>10.1} us/init",
        "CfC::from_genesis()",
        avg_us(cfc_gen_init, cfc_n)
    );
    println!(
        "  {:<40} {:>10.1} us/init",
        "HdcLtcBridge::new()",
        avg_us(hdc_rand_init, hdc_n)
    );
    println!(
        "  {:<40} {:>10.1} us/init",
        "HdcLtcBridge::from_genesis()",
        avg_us(hdc_gen_init, hdc_n)
    );

    // ---- Summary table ----
    println!();
    println!("--- Forward-Pass Summary (per-step) ---");
    println!(
        "  {:<20} {:>14} {:>14} {:>12}",
        "Network", "Random us/step", "Genesis us/step", "Overhead %"
    );

    let cfc_rand_us = cfc_rand_dur.as_micros() as f64 / CFC_STEPS as f64;
    let cfc_gen_us = cfc_gen_dur.as_micros() as f64 / CFC_STEPS as f64;
    let hdc_rand_us = hdc_rand_dur.as_micros() as f64 / HDC_STEPS as f64;
    let hdc_gen_us = hdc_gen_dur.as_micros() as f64 / HDC_STEPS as f64;
    let hdc_fast_us = hdc_fast_dur.as_micros() as f64 / hdc_fast_steps as f64;
    let hdc_fast_gen_us = hdc_fast_gen_dur.as_micros() as f64 / hdc_fast_steps as f64;

    let cfc_overhead = overhead_pct(cfc_rand_dur, cfc_gen_dur);
    let hdc_overhead = overhead_pct(hdc_rand_dur, hdc_gen_dur);
    let hdc_fast_overhead = overhead_pct(hdc_fast_dur, hdc_fast_gen_dur);

    println!(
        "  {:<20} {:>14.1} {:>14.1} {:>12.1}",
        "CfC", cfc_rand_us, cfc_gen_us, cfc_overhead,
    );
    println!(
        "  {:<20} {:>14.1} {:>14.1} {:>12.1}",
        "HDC-LTC (16384)", hdc_rand_us, hdc_gen_us, hdc_overhead,
    );
    println!(
        "  {:<20} {:>14.1} {:>14.1} {:>12.1}",
        "HDC-LTC (2048)", hdc_fast_us, hdc_fast_gen_us, hdc_fast_overhead,
    );
    println!();
    println!(
        "  Speedup from dim reduction (16384 → 2048): {:.1}x",
        hdc_rand_us / hdc_fast_us,
    );
    println!(
        "  HDC-LTC (2048) vs CfC ratio: {:.1}x",
        hdc_fast_us / cfc_rand_us,
    );
    println!();

    // ---- Assertions ----
    // Genesis seeding only changes weight initialization, not the forward-pass
    // computation graph. Forward-pass time should be equivalent.
    // At sub-100μs per step (release builds), measurement noise dominates.
    // In debug builds overhead is <15%, but release optimizations shift the baseline.
    // Use abs() since genesis init can sometimes be _faster_ due to cache effects.
    let threshold = if cfc_rand_us < 100.0 { 100.0 } else { 50.0 };
    assert!(
        cfc_overhead.abs() < threshold,
        "CfC genesis forward-pass overhead {:.1}% exceeds {}% threshold",
        cfc_overhead,
        threshold,
    );
    assert!(
        hdc_overhead.abs() < threshold,
        "HDC-LTC genesis forward-pass overhead {:.1}% exceeds {}% threshold",
        hdc_overhead,
        threshold,
    );

    println!(
        "  PASS: Genesis forward-pass overhead < {}% for both networks.",
        threshold,
    );
}

fn overhead_pct(baseline: std::time::Duration, genesis: std::time::Duration) -> f64 {
    let b = baseline.as_secs_f64();
    if b == 0.0 {
        return 0.0;
    }
    ((genesis.as_secs_f64() - b) / b) * 100.0
}