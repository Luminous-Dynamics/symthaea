// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reproducible microbenchmark for the HDC-LTC/CfC neuron's O(D) evolution path.
//!
//! Run with:
//!   cargo run --release -p symthaea-hdc-ltc --example scaling_benchmark
//!
//! The default sweep reaches 1M dimensions. For the heavier 4M stress tier:
//!   SYTHAEA_HDC_LTC_BENCH_4M=1 cargo run --release -p symthaea-hdc-ltc --example scaling_benchmark
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
            "dimension,hypervector_bytes,neuron_vector_bytes,dt_seconds,mode,iterations,warmup_iterations,elapsed_ns,",
            "ns_per_component_update,component_updates_per_second,final_state_norm"
        )
    );

    const DIMENSIONS: &[usize] = &[
        256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576,
    ];
    const TARGET_COMPONENT_UPDATES: usize = 2_000_000;
    const MIN_ITERATIONS: usize = 50;
    const NEURON_VECTOR_COUNT: usize = 8;

    let mut dimensions = DIMENSIONS.to_vec();
    // Four-million-dimension trials allocate ~128 MiB for neuron vectors alone
    // and perform much more timed work; make this large-host tier explicit opt-in.
    let include_4m = std::env::var_os("SYTHAEA_HDC_LTC_BENCH_4M")
        .is_some_and(|value| value.to_string_lossy() == "1");
    if include_4m {
        dimensions.push(4_194_304);
        eprintln!("Including optional 4M dimension tier (SYTHAEA_HDC_LTC_BENCH_4M=1).");
    }

    for dimension in dimensions {
        // Keep coordinate work per row approximately comparable across dimensions.
        // Large dimensions use fewer warmups so the benchmark itself stays practical.
        let iterations = (TARGET_COMPONENT_UPDATES / dimension).max(MIN_ITERATIONS);
        let warmup_iterations = if dimension >= 262_144 { 3 } else { 20 };
        let hypervector_bytes = dimension * std::mem::size_of::<f32>();
        let neuron_vector_bytes = hypervector_bytes * NEURON_VECTOR_COUNT;
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
                for _ in 0..warmup_iterations {
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
                        "{dimension},{hypervector_bytes},{neuron_vector_bytes},{dt},{mode},{iterations},{warmup_iterations},{elapsed_ns},",
                        "{ns_per_component_update:.6},{component_updates_per_second:.3},",
                        "{final_state_norm:.8}"
                    )
                );
            }
        }
    }
}
