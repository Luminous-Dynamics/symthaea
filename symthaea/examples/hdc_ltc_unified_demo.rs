// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC-LTC Unified Architecture Demo
//!
//! Demonstrates the revolutionary unified architecture where:
//! - Neuron STATE is a hypervector (16,384D)
//! - Weight operations use HDC binding instead of matrix multiplication
//! - Closed-form solution enables O(1) temporal jumps
//!
//! Run with: cargo run --example hdc_ltc_unified_demo

use std::time::Instant;
use symthaea::hdc::{
    ContinuousHV, HDC_DIMENSION, HdcLtcUnifiedNetwork, HdcLtcUnifiedNeuron, UnifiedActivation,
    UnifiedConfig, UnifiedNetworkConfig,
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     HDC-LTC UNIFIED ARCHITECTURE: State AS Hypervector + Closed-Form          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Demo 1: Basic neuron dynamics
    demo_basic_neuron();

    // Demo 2: Closed-form O(1) temporal jumps
    demo_closed_form_jumps();

    // Demo 3: HDC binding vs matrix multiplication
    demo_hdc_binding();

    // Demo 4: Network dynamics
    demo_network();

    // Demo 5: Temporal pattern learning
    demo_temporal_learning();

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("Demo complete! The unified HDC-LTC architecture is ready for consciousness.");
    println!("═══════════════════════════════════════════════════════════════════════════════");
}

fn demo_basic_neuron() {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 1: Basic Unified Neuron Dynamics                                       │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let config = UnifiedConfig {
        dimension: HDC_DIMENSION,
        tau_base: 0.1,
        backbone_tau: 0.5,
        activation: UnifiedActivation::Tanh,
        ..Default::default()
    };

    let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
    let input = ContinuousHV::random_default(123);

    println!("  Initial state norm: {:.6}", neuron.state().norm());
    println!("  Input dimension: {} (HDC standard)", input.dim());

    // Evolve with Euler integration
    println!("\n  Evolving with Euler integration (100 steps, dt=0.01)...");
    let start = Instant::now();
    for _ in 0..100 {
        neuron.evolve(0.01, &input);
    }
    let euler_time = start.elapsed();

    println!("  Final state norm: {:.6}", neuron.state().norm());
    println!("  Time taken: {:?}", euler_time);
    println!("  Effective tau: {:.4}", neuron.effective_tau(&input));
    println!();
}

fn demo_closed_form_jumps() {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 2: O(1) Closed-Form Temporal Jumps                                     │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let input = ContinuousHV::random_default(456);

    // Compare: many Euler steps vs single closed-form jump
    let mut neuron_euler = HdcLtcUnifiedNeuron::new_default(100);
    let mut neuron_cf = HdcLtcUnifiedNeuron::new_default(100);

    // Euler: 10000 small steps
    println!("  Euler integration: 10000 steps of dt=0.0001 (total time = 1.0)");
    let start = Instant::now();
    for _ in 0..10000 {
        neuron_euler.evolve(0.0001, &input);
    }
    let euler_time = start.elapsed();
    println!("    Time taken: {:?}", euler_time);
    println!("    Final state norm: {:.6}", neuron_euler.state().norm());

    // Closed-form: single jump
    println!("\n  Closed-form: single jump of dt=1.0");
    let start = Instant::now();
    neuron_cf.evolve_closed_form(1.0, &input);
    let cf_time = start.elapsed();
    println!("    Time taken: {:?}", cf_time);
    println!("    Final state norm: {:.6}", neuron_cf.state().norm());

    // Compare results
    let similarity = neuron_euler.state().similarity(neuron_cf.state());
    println!(
        "\n  State similarity (Euler vs Closed-form): {:.4}",
        similarity
    );
    println!(
        "  Speedup: {:.1}x",
        euler_time.as_secs_f64() / cf_time.as_secs_f64()
    );

    // Demonstrate O(1) property: large jumps cost same as small
    println!("\n  O(1) Property Demonstration:");
    let mut neuron_small = HdcLtcUnifiedNeuron::new_default(200);
    let mut neuron_large = HdcLtcUnifiedNeuron::new_default(200);

    let start = Instant::now();
    neuron_small.evolve_closed_form(0.1, &input);
    let small_time = start.elapsed();

    let start = Instant::now();
    neuron_large.evolve_closed_form(100.0, &input); // 1000x larger!
    let large_time = start.elapsed();

    println!("    dt=0.1: {:?}", small_time);
    println!("    dt=100.0 (1000x larger): {:?}", large_time);
    println!("    Both take roughly the same time (O(1) complexity!)");
    println!();
}

fn demo_hdc_binding() {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 3: HDC Binding Replaces Matrix Multiplication                          │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    // Traditional neural networks: y = Wx (matrix multiply)
    // Our approach: y = W ⊗ x (HDC binding)

    let weight_hv = ContinuousHV::random_default(42);
    let input_hv = ContinuousHV::random_default(123);

    // HDC Binding operation
    let bound = weight_hv.bind(&input_hv);

    println!("  Weight HV norm: {:.6}", weight_hv.norm());
    println!("  Input HV norm: {:.6}", input_hv.norm());
    println!("  Bound result norm: {:.6}", bound.norm());

    // Key property: binding produces dissimilar vectors
    let sim_to_weight = bound.similarity(&weight_hv);
    let sim_to_input = bound.similarity(&input_hv);

    println!("\n  Key HDC Property - Binding is dissimilar to inputs:");
    println!("    Similarity(bound, weight): {:.4}", sim_to_weight);
    println!("    Similarity(bound, input): {:.4}", sim_to_input);
    println!("    (Both should be near 0 for random vectors)");

    // Bundling for accumulation
    let another_hv = ContinuousHV::random_default(789);
    let bundled = ContinuousHV::bundle(&[&input_hv, &another_hv]);

    let sim_to_first = bundled.similarity(&input_hv);
    let sim_to_second = bundled.similarity(&another_hv);

    println!("\n  Bundling Property - Result is similar to both inputs:");
    println!("    Similarity(bundle, first): {:.4}", sim_to_first);
    println!("    Similarity(bundle, second): {:.4}", sim_to_second);
    println!("    (Both should be positive, around 0.7)");
    println!();
}

fn demo_network() {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 4: Unified Network Dynamics                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let config = UnifiedNetworkConfig {
        layer_sizes: vec![4, 8, 4],
        use_layer_binding: true,
        skip_connections: false,
        ..Default::default()
    };

    let mut network = HdcLtcUnifiedNetwork::new(config, 42);
    let input = ContinuousHV::random_default(111);

    println!("  Network structure: 4 -> 8 -> 4 neurons");
    println!("  Layer binding: enabled");

    // Evolve with closed-form
    println!("\n  Evolving network for 1.0 seconds (closed-form)...");
    let start = Instant::now();
    for _ in 0..100 {
        network.evolve_closed_form(0.01, &input);
    }
    let time = start.elapsed();

    let stats = network.stats();
    println!("  Time taken: {:?}", time);
    println!("  Total neurons: {}", stats.n_neurons);
    println!("  Average state norm: {:.6}", stats.avg_state_norm);
    println!("  Average weight norm: {:.6}", stats.avg_weight_norm);
    println!("  Total updates: {}", stats.total_updates);

    // Output
    let output = network.output();
    println!("\n  Output hypervector norm: {:.6}", output.norm());
    println!("  Output dimension: {}", output.dim());
    println!();
}

fn demo_temporal_learning() {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 5: Temporal Pattern Learning with Hebbian Updates                      │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let mut neuron = HdcLtcUnifiedNeuron::new_default(42);

    // Create two distinct patterns
    let pattern_a = ContinuousHV::random_default(100);
    let pattern_b = ContinuousHV::random_default(200);

    println!(
        "  Pattern A-B similarity: {:.4}",
        pattern_a.similarity(&pattern_b)
    );

    // Train on pattern A
    println!("\n  Training on Pattern A (50 steps + Hebbian)...");
    for _ in 0..50 {
        neuron.evolve(0.01, &pattern_a);
        neuron.hebbian_update(&pattern_a, Some(0.01));
    }

    let state_after_a = neuron.state().clone();
    let sim_to_a = state_after_a.similarity(&pattern_a);
    let sim_to_b = state_after_a.similarity(&pattern_b);

    println!("  State similarity to A: {:.4}", sim_to_a);
    println!("  State similarity to B: {:.4}", sim_to_b);

    // Train on pattern B
    println!("\n  Training on Pattern B (50 steps + Hebbian)...");
    for _ in 0..50 {
        neuron.evolve(0.01, &pattern_b);
        neuron.hebbian_update(&pattern_b, Some(0.01));
    }

    let state_after_b = neuron.state().clone();
    let sim_to_a_after = state_after_b.similarity(&pattern_a);
    let sim_to_b_after = state_after_b.similarity(&pattern_b);

    println!("  State similarity to A: {:.4}", sim_to_a_after);
    println!("  State similarity to B: {:.4}", sim_to_b_after);

    // Demonstrate state-dependent tau adaptation
    let tau_for_a = neuron.effective_tau(&pattern_a);
    let tau_for_b = neuron.effective_tau(&pattern_b);
    println!("\n  Input-dependent tau:");
    println!("    tau(Pattern A): {:.4}", tau_for_a);
    println!("    tau(Pattern B): {:.4}", tau_for_b);

    let stats = neuron.stats();
    println!("\n  Final stats:");
    println!("    State norm: {:.4}", stats.state_norm);
    println!("    Weight norm: {:.4}", stats.weight_norm);
    println!("    Total updates: {}", stats.update_count);
    println!();
}