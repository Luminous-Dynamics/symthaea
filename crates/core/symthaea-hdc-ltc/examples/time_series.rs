// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sine wave prediction with an HDC-LTC network.
//!
//! Demonstrates: closed-form temporal evolution, Hebbian learning,
//! and the ability to track a smooth continuous signal.

use symthaea_hdc_ltc::{ContinuousHV, HdcLtcUnifiedNetwork, NetworkConfig, NeuronConfig};

fn main() {
    let dim = 512;
    let config = NetworkConfig {
        layer_sizes: vec![4, 4],
        neuron_config: NeuronConfig {
            dim,
            tau_base: 0.05,
            ..NeuronConfig::default()
        },
        use_layer_binding: true,
        skip_connections: false,
    };

    let mut net = HdcLtcUnifiedNetwork::new(config, 42);

    // Create basis vectors for encoding scalar values into HV space
    let pos_basis = ContinuousHV::new_random(dim, 1000);
    let neg_basis = ContinuousHV::new_random(dim, 2000);

    let dt = 0.01;
    let steps = 200;
    let mut total_error = 0.0f32;

    println!("Step | Actual   | Predicted | Error");
    println!("-----|----------|-----------|------");

    for step in 0..steps {
        let t = step as f32 * dt;
        let value = (t * 2.0 * std::f32::consts::PI).sin();

        // Encode the current sine value as a hypervector:
        // Blend between pos_basis and neg_basis based on value in [-1, 1]
        let alpha = (value + 1.0) * 0.5; // Map [-1,1] to [0,1]
        let input = pos_basis.scale(alpha).add(&neg_basis.scale(1.0 - alpha));

        // Step the network
        net.step(dt, &input);

        // Decode: project output onto basis vectors to recover a scalar
        let out = net.output();
        let pos_sim = out.similarity(&pos_basis);
        let neg_sim = out.similarity(&neg_basis);
        let predicted = pos_sim - neg_sim; // Range roughly [-1, 1]

        // Next value for comparison
        let next_t = (step + 1) as f32 * dt;
        let actual_next = (next_t * 2.0 * std::f32::consts::PI).sin();

        let error = (predicted - actual_next).abs();
        total_error += error;

        if step % 20 == 0 {
            println!(
                "{:4} | {:+.4}  | {:+.4}   | {:.4}",
                step, actual_next, predicted, error
            );
        }
    }

    let avg_error = total_error / steps as f32;
    println!("\nAverage absolute error: {:.4}", avg_error);
    println!(
        "Network: {} layers, {} neurons, {} dimensions",
        net.layer_count(),
        net.neuron_count(),
        dim
    );
}
