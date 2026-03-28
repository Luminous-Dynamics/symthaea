// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Simple proportional controller using an HDC-LTC neuron.
//!
//! Demonstrates: encoding scalar errors as hypervectors, evolving a neuron
//! to produce a control signal, and decoding the output back to a scalar.

use symthaea_hdc_ltc::{ContinuousHV, HdcLtcUnifiedNeuron, NeuronConfig};

fn main() {
    let dim = 512;
    let config = NeuronConfig {
        dim,
        tau_base: 0.05,
        ..NeuronConfig::default()
    };

    let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

    // Basis vectors for encoding error magnitude and sign
    let error_basis = ContinuousHV::new_random(dim, 100);
    let sign_pos = ContinuousHV::new_random(dim, 200);
    let sign_neg = ContinuousHV::new_random(dim, 300);
    let output_basis = ContinuousHV::new_random(dim, 400);

    let target = 1.0f32;
    let mut position = 0.0f32;
    let dt = 0.02;

    println!("Step | Position | Error    | Control");
    println!("-----|----------|----------|--------");

    for step in 0..100 {
        let error = target - position;

        // Encode error as HV: magnitude * error_basis bound with sign
        let magnitude = error.abs().min(2.0);
        let sign_hv = if error >= 0.0 { &sign_pos } else { &sign_neg };
        let input = error_basis.scale(magnitude).bind(sign_hv);

        // Evolve neuron
        neuron.evolve_closed_form(dt, &input);

        // Decode control signal: project state onto output basis
        let control = neuron.state().similarity(&output_basis) * 2.0;

        // Simple plant: position += control * dt
        position += control * dt;

        if step % 10 == 0 {
            println!(
                "{:4} | {:+.4}  | {:+.4}  | {:+.4}",
                step, position, error, control
            );
        }
    }

    let final_error = (target - position).abs();
    println!("\nFinal position: {:.4} (target: {:.4})", position, target);
    println!("Final error: {:.4}", final_error);
    println!("Total neuron time: {:.2} s", neuron.total_time());
}
