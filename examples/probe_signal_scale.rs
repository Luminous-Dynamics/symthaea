// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Signal-scale diagnostic for the HDC-LTC bridge (PHI_SIGNAL_TRACE follow-up 1).
//!
//! `CycleResult.output` was measured at norm ~3e-8 (2026-07-16 persistence
//! probe) — downstream consumers of the "CfC output state" receive a
//! near-zero-magnitude signal. Hypothesis: systematic magnitude collapse
//! through the unified neuron's bind→bundle→activation chain (unit-normalized
//! 16,384-dim HVs have per-element scale ~1/128; element-wise bind products
//! are ~1/16K; tanh is ≈identity there), compounded by the output projection.
//! If confirmed, this also bounds the trainable gradient scale of the output
//! readout — a candidate root cause for "single-gradient-step training is too
//! weak" (keystone Phase 4).
//!
//! Read-only diagnosis: prints per-stage magnitudes. No fixes here.
//!
//! Run: cargo run --release --example probe_signal_scale

use ndarray::Array1;
use symthaea::hdc_ltc_bridge::{HdcLtcBridge, HdcLtcBridgeConfig};

fn stats(name: &str, v: &[f32]) {
    let n = v.len() as f32;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mean_abs = v.iter().map(|x| x.abs()).sum::<f32>() / n;
    let max_abs = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    println!(
        "  {name:28} norm={norm:.3e}  mean|x|={mean_abs:.3e}  max|x|={max_abs:.3e}  dim={}",
        v.len()
    );
}

fn main() {
    // Loop-realistic config (matches cognitive_loop defaults: 256-dim I/O,
    // full 16,384 HDC dim so the per-element scale story is the real one).
    let config = HdcLtcBridgeConfig::default();
    let mut bridge = HdcLtcBridge::new(config.clone());

    println!(
        "=== signal-scale probe (input_dim={}, hdc_dim={}) ===",
        config.input_dim, config.hdc_dim
    );

    // Loop-realistic input: compressed encodings have per-element scale ~1/sqrt(dim)
    // after unit-normalization upstream; use a unit-norm random-ish vector.
    let raw: Vec<f32> = (0..config.input_dim)
        .map(|i| ((i as f32 * 12.9898).sin() * 43758.547).fract() - 0.5)
        .collect();
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    let input = Array1::from_vec(raw.iter().map(|x| x / norm).collect());
    stats("input (unit-normalized)", input.as_slice().unwrap());

    for step in 1..=50 {
        bridge.step(&input, 0.02).unwrap();
        if step == 1 || step == 10 || step == 50 {
            println!("after step {step}:");
            let out = bridge.read_state().unwrap();
            stats("bridge output (read_state)", out.as_slice().unwrap());
        }
    }

    // Training-side scale: what loss and (implicitly) gradient magnitude does
    // the readout face against a unit-scale target?
    let target = input.clone();
    let loss = bridge.train_step(&input, &target, 0.02, 0.0).unwrap();
    println!("\nreadout MSE loss vs unit-norm target (lr=0): {loss:.6}");
    println!(
        "(loss ≈ mean(target²) = {:.6} would mean the readout contributes ~nothing)",
        target.iter().map(|x| x * x).sum::<f32>() / target.len() as f32
    );

    // Small-dim comparison: does the collapse scale with hdc_dim as predicted?
    for hdc_dim in [1024usize, 4096, 16384] {
        let cfg = HdcLtcBridgeConfig {
            hdc_dim,
            ..config.clone()
        };
        let mut b = HdcLtcBridge::new(cfg);
        for _ in 0..10 {
            b.step(&input, 0.02).unwrap();
        }
        let out = b.read_state().unwrap();
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("hdc_dim {hdc_dim:6}: output norm after 10 steps = {norm:.3e}");
    }
}
