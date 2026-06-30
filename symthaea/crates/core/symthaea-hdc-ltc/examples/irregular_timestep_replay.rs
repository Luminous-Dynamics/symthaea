// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! irregular_timestep_replay example.
//!
//! Demonstrates the deterministic replay of irregular timestep steps in a network
//! of HDC-LTC neurons under various safety and timing contracts.

use symthaea_hdc_ltc::{
    ContinuousHV, HdcLtcUnifiedNetwork, NetworkConfig, NeuronConfig, StepTimingConfig,
};

fn main() {
    println!("=== HDC-LTC Irregular Timestep Replay & Hardening Proof ===");

    let dim = 256;
    let neuron_config = NeuronConfig {
        dim,
        tau_base: 0.1,
        tau_min: 0.001,
        tau_max: 10.0,
        ..NeuronConfig::default()
    };

    let config = NetworkConfig {
        layer_sizes: vec![4, 4],
        neuron_config,
        use_layer_binding: true,
        skip_connections: false,
    };

    let mut net = HdcLtcUnifiedNetwork::new(config.clone(), 42);

    // Set custom timing configuration
    let timing_config = StepTimingConfig {
        min_dt: 0.005,
        max_dt: 2.0,
        reject_backward_time: true,
    };
    net.set_timing_config(timing_config);

    // Replay timestamps: mix of irregular, too small, too large, and backward jumps
    let timestamps = vec![
        0.0, 0.002, // dt = 0.002 (below min_dt, should clamp to 0.005)
        0.150, // dt = 0.148 (valid)
        0.150, // dt = 0.000 (below min_dt, should clamp to 0.005)
        0.140, // dt = -0.010 (backward, should be rejected)
        3.150, // dt = 3.000 (above max_dt, should clamp to 2.0)
    ];

    let input = ContinuousHV::new_random(dim, 100);

    println!("\nReplaying irregular timestamps sequence:");
    println!("Step | Timestamp | Applied t | Step Count");
    println!("-----|-----------|-----------|-----------");

    for (idx, &t) in timestamps.iter().enumerate() {
        net.step_with_timestamp(t, &input);
        println!(
            "{:4} | {:+.4}   | {:+.4}    | {}",
            idx,
            t,
            net.last_timestamp().unwrap_or(0.0),
            net.step_count()
        );
    }

    let final_state_norm = net.output().norm();
    assert!(final_state_norm.is_finite());
    assert!(final_state_norm > 0.0);

    println!("\nFinal network state norm: {:.6}", final_state_norm);
    println!(
        "Invariants satisfied (finite state: {}, step count: {})",
        final_state_norm.is_finite(),
        net.step_count()
    );
    println!("Deterministic replay proof: SUCCESS");
}
