// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reproducible microbenchmark for the HDC-LTC/CfC neuron's O(D) evolution path.
//!
//! Run with:
//!   cargo run --release -p symthaea-hdc-ltc --example scaling_benchmark
//!
//! Emits CSV for multiple hypervector sizes and time increments. It does not claim a
//! speedup against another architecture; use the output to track dimension and dt scaling
//! on a fixed machine/toolchain, and compare against a matched baseline before drawing
//! performance conclusions.

use std::hint::black_box;
use std::time::Instant;
use symthaea_hdc_ltc::{ContinuousHV, HdcLtcUnifiedNeuron, NeuronConfig};

fn main() {
    println!(
        concat!(
            "dimension,dt_seconds,mode,iterations,elapsed_ns,",
            "ns_per_component_update,component_updates_per_second,final_state_norm"
        )
    );

    for dimension in [256_usize, 1_024, 4_096, 16_384] {
        // Keep the approximate amount of coordinate work per row comparable while
        // retaining at least 100 iterations for the largest vector.
        let iterations = (1_000_000_usize / dimension).max(100);
        let input = ContinuousHV::new_random(dimension, 0xCFC0_2026 ^ dimension as u64);

        for (dt_index, dt) in [1.0e-6_f32, 0.1, 100.0].into_iter().enumerate() {
            for checked_path in [false, true] {
                let mode = if checked_path { "checked" } else { "unchecked_fused" };
                let mut neuron = HdcLtcUnifiedNeuron::new(
                    NeuronConfig {
                        dim: dimension,
                        ..NeuronConfig::default()
                    },
                    0x4844_4300_u64
                        .wrapping_add(dimension as u64)
                        .wrapping_add(dt_index as u64),
                );

                // Warm up each path independently.
                for _ in 0..20 {
                    if checked_path {
                        neuron
                            .try_evolve_closed_form_fused(black_box(dt), black_box(&input))
                            .expect("benchmark input is valid");
                    } else {
                        neuron.evolve_closed_form_fused(black_box(dt), black_box(&input));
                    }
                }

                let start = Instant::now();
                for _ in 0..iterations {
                    if checked_path {
                        neuron
                            .try_evolve_closed_form_fused(black_box(dt), black_box(&input))
                            .expect("benchmark input is valid");
                    } else {
                        neuron.evolve_closed_form_fused(black_box(dt), black_box(&input));
                    }
                }
                let elapsed_ns = start.elapsed().as_nanos();
                let component_updates = (dimension * iterations) as f64;
                let ns_per_component_update = elapsed_ns as f64 / component_updates;
                let component_updates_per_second =
                    component_updates * 1_000_000_000.0 / elapsed_ns.max(1) as f64;
                let final_state_norm = black_box(neuron.state().norm());

                println!(
                    concat!(
                        "{dimension},{dt},{mode},{iterations},{elapsed_ns},",
                        "{ns_per_component_update:.6},{component_updates_per_second:.3},",
                        "{final_state_norm:.8}"
                    )
                );
            }
        }
    }
}
