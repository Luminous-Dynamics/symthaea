// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Demo: Measure structural coherence of a synthetic power grid.
//!
//! Simulates a 6-node power grid where nodes are coupled through shared
//! frequency dynamics. Tightly coupled clusters should show high integration
//! index; the MIP cut should separate weakly coupled regions.
//!
//! ```text
//!   [Gen1] --- [Bus1] --- [Bus2] --- [Gen2]
//!                |                     |
//!              [Load1]              [Load2]
//! ```
//!
//! Run: `cargo run -p symthaea-phi-oracle --example power_grid`

use symthaea_phi_oracle::{CoherenceTrend, IntegrationOracle, OracleConfig, TimeSeriesEncoder};

fn main() {
    let num_nodes = 6;
    let base_freq = 60.0; // Hz (nominal grid frequency)

    // Coupling matrix: how strongly each node influences each other.
    // Generators (0,3) strongly couple to their buses (1,4).
    // Buses weakly couple across the interconnect.
    let coupling = [
        //  G1    B1    B2    G2    L1    L2
        [0.0, 0.8, 0.0, 0.0, 0.0, 0.0], // Gen1
        [0.8, 0.0, 0.3, 0.0, 0.6, 0.0], // Bus1
        [0.0, 0.3, 0.0, 0.8, 0.0, 0.6], // Bus2
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0], // Gen2
        [0.0, 0.6, 0.0, 0.0, 0.0, 0.0], // Load1
        [0.0, 0.0, 0.6, 0.0, 0.0, 0.0], // Load2
    ];

    // Simulate frequency deviations over time
    let dt = 0.02; // 50 Hz sample rate
    let num_steps = 200;
    let mut state = vec![0.0f64; num_nodes]; // frequency deviations from 60Hz

    // Oracle setup
    let encoder = TimeSeriesEncoder::new(num_nodes, 256, 42).with_name("power-grid-6node");

    let config = OracleConfig {
        window_size: 50,
        temporal_probes: vec![0.01, 0.05, 0.1, 0.5, 1.0],
        ..Default::default()
    };

    let mut oracle =
        IntegrationOracle::new(Box::new(encoder), config).expect("oracle creation failed");

    println!("=== Integration Spectrometer: Power Grid Demo ===\n");
    println!("System: 6-node grid (2 generators, 2 buses, 2 loads)");
    println!("Nominal frequency: {base_freq} Hz");
    println!("Simulation: {num_steps} steps at dt={dt}s\n");

    let mut trend = CoherenceTrend::new(10);

    // Simple damped Kuramoto-like dynamics
    let mut rng_state: u64 = 12345;
    for step in 0..num_steps {
        // Compute coupling forces
        let mut forces = vec![0.0f64; num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                forces[i] += coupling[i][j] * (state[j] - state[i]);
            }
            // Damping
            forces[i] -= 0.1 * state[i];
            // Generator drive (small periodic perturbation)
            if i == 0 || i == 3 {
                forces[i] += 0.5 * (step as f64 * dt * 2.0 * std::f64::consts::PI * 0.1).sin();
            }
        }

        // Euler integration + noise
        for i in 0..num_nodes {
            // Simple LCG noise
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.02;
            state[i] += forces[i] * dt + noise;
        }

        // Feed observation: actual frequencies (base + deviation)
        let observation: Vec<f64> = state.iter().map(|&dev| base_freq + dev).collect();
        oracle.observe(&observation).expect("observe failed");

        // Take a mid-run measurement for trend tracking
        if step == 99 {
            if let Some(report) = oracle.measure() {
                trend.record(&report);
                println!("Mid-run (step {step}):");
                println!("  Integration index: {:.6}\n", report.integration_index);
            }
        }
    }

    // Final measurement
    match oracle.measure() {
        Some(report) => {
            trend.record(&report);
            println!("{report}");

            println!("--- Trend ---");
            println!("{trend}");
        }
        None => {
            println!("ERROR: Not enough data to compute integration report.");
        }
    }

    // Hierarchical measurement
    if let Some(hier) = oracle.measure_hierarchical() {
        println!("\n{hier}");
    }
}
