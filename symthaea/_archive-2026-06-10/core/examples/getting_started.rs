// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Getting Started with Symthaea
//!
//! This example demonstrates the core concepts of the Symthaea Holographic Liquid Brain:
//!
//! 1. **BinaryHV / ContinuousHV**: High-dimensional vector representations
//! 2. **Bind and Bundle Operations**: HDC algebra for associations and superpositions
//! 3. **LTC Neurons**: Liquid Time-Constant dynamics with closed-form evolution
//! 4. **Phi Measurement**: Integrated information as a consciousness metric
//! 5. **Cognitive Loop**: Full consciousness processing pipeline
//!
//! Run with: `cargo run --example getting_started`

use std::time::Instant;

// Import core HDC types from symthaea_core
use symthaea_core::hdc::unified_hv::{BinaryHV, ContinuousHV, HDC_DIMENSION};
use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

// Import LTC and cognitive components from symthaea
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::hdc::{HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig};

fn main() {
    println!("===============================================================================");
    println!("                    SYMTHAEA: GETTING STARTED GUIDE                           ");
    println!("===============================================================================");
    println!();

    // Step 1: Understanding Hypervectors
    demo_hypervectors();

    // Step 2: HDC Operations (Bind and Bundle)
    demo_hdc_operations();

    // Step 3: LTC Neuron Dynamics
    demo_ltc_neuron();

    // Step 4: Phi Measurement
    demo_phi_measurement();

    // Step 5: Cognitive Loop Integration
    demo_cognitive_loop();

    println!();
    println!("===============================================================================");
    println!("                         GETTING STARTED COMPLETE                             ");
    println!("===============================================================================");
    println!();
    println!("Next steps:");
    println!("  1. Explore examples/hdc_ltc_unified_demo.rs for advanced LTC features");
    println!("  2. See examples/fep_active_inference_demo.rs for motor command generation");
    println!("  3. Run the REPL: cargo run --bin symthaea-repl");
    println!();
}

/// Step 1: Understanding Hypervectors
///
/// Hypervectors are the foundation of Symthaea. They are high-dimensional vectors
/// (16,384 dimensions by default) with special properties:
/// - Random vectors are nearly orthogonal (similarity ~ 0)
/// - Operations preserve information through the holographic principle
fn demo_hypervectors() {
    println!("-------------------------------------------------------------------------------");
    println!("STEP 1: UNDERSTANDING HYPERVECTORS");
    println!("-------------------------------------------------------------------------------");
    println!();

    // Create random hypervectors with deterministic seeds
    let hv_apple = ContinuousHV::random(HDC_DIMENSION, 42);
    let hv_orange = ContinuousHV::random(HDC_DIMENSION, 43);
    let hv_fruit = ContinuousHV::random(HDC_DIMENSION, 44);

    println!("  Created three hypervectors:");
    println!(
        "    - hv_apple:  {} dimensions, norm = {:.4}",
        hv_apple.dim(),
        hv_apple.norm()
    );
    println!(
        "    - hv_orange: {} dimensions, norm = {:.4}",
        hv_orange.dim(),
        hv_orange.norm()
    );
    println!(
        "    - hv_fruit:  {} dimensions, norm = {:.4}",
        hv_fruit.dim(),
        hv_fruit.norm()
    );
    println!();

    // Key property: random vectors are nearly orthogonal
    let sim_apple_orange = hv_apple.similarity(&hv_orange);
    let sim_apple_fruit = hv_apple.similarity(&hv_fruit);

    println!("  Key Property - Random vectors are nearly orthogonal:");
    println!(
        "    similarity(apple, orange) = {:.4} (expected: ~0)",
        sim_apple_orange
    );
    println!(
        "    similarity(apple, fruit)  = {:.4} (expected: ~0)",
        sim_apple_fruit
    );
    println!();

    // Same seed = same vector (deterministic)
    let hv_apple_copy = ContinuousHV::random(HDC_DIMENSION, 42);
    let sim_identical = hv_apple.similarity(&hv_apple_copy);
    println!("  Determinism - Same seed produces identical vectors:");
    println!(
        "    similarity(apple, apple_copy) = {:.4} (expected: 1.0)",
        sim_identical
    );
    println!();

    // Binary hypervectors are more memory efficient
    let binary_a = BinaryHV::random(100);
    let binary_b = BinaryHV::random(101);

    println!("  Binary vs Continuous:");
    println!(
        "    ContinuousHV: {} bytes ({} x 4 bytes)",
        HDC_DIMENSION * 4,
        HDC_DIMENSION
    );
    println!(
        "    BinaryHV:     {} bytes ({} bits packed)",
        HDC_DIMENSION / 8,
        HDC_DIMENSION
    );
    println!(
        "    Similarity(binary_a, binary_b) = {:.4}",
        binary_a.similarity(&binary_b)
    );
    println!();
}

/// Step 2: HDC Operations (Bind and Bundle)
///
/// HDC provides two fundamental operations:
/// - **Bind (XOR for binary, element-wise multiply for continuous)**:
///   Creates associations that are dissimilar to both inputs
/// - **Bundle (majority vote for binary, element-wise add for continuous)**:
///   Creates superpositions that are similar to all inputs
fn demo_hdc_operations() {
    println!("-------------------------------------------------------------------------------");
    println!("STEP 2: HDC OPERATIONS - BIND AND BUNDLE");
    println!("-------------------------------------------------------------------------------");
    println!();

    let hv_red = ContinuousHV::random(HDC_DIMENSION, 100);
    let hv_color = ContinuousHV::random(HDC_DIMENSION, 101);
    let hv_shape = ContinuousHV::random(HDC_DIMENSION, 102);
    let hv_circle = ContinuousHV::random(HDC_DIMENSION, 103);

    // Binding: Create role-filler associations
    // "red" bound with "color" creates a unique representation
    println!("  BINDING - Creating Associations:");
    println!("  ─────────────────────────────────");

    let red_is_color = hv_red.bind(&hv_color);
    let circle_is_shape = hv_circle.bind(&hv_shape);

    println!("    red_is_color = red ⊗ color");
    println!("    circle_is_shape = circle ⊗ shape");
    println!();

    // Key property: bound result is dissimilar to both inputs
    println!("    Property: Binding produces vectors dissimilar to inputs:");
    println!(
        "      similarity(red_is_color, red)    = {:.4} (~0)",
        red_is_color.similarity(&hv_red)
    );
    println!(
        "      similarity(red_is_color, color)  = {:.4} (~0)",
        red_is_color.similarity(&hv_color)
    );
    println!();

    // Binding is invertible: A ⊗ B ⊗ B = A
    let recovered_red = red_is_color.bind(&hv_color);
    println!("    Property: Binding is invertible (A ⊗ B ⊗ B = A):");
    println!(
        "      similarity(recovered_red, red) = {:.4} (~1.0)",
        recovered_red.similarity(&hv_red)
    );
    println!();

    // Bundling: Create superpositions
    println!("  BUNDLING - Creating Superpositions:");
    println!("  ────────────────────────────────────");

    // A "red circle" is the superposition of red_is_color and circle_is_shape
    let red_circle = ContinuousHV::bundle(&[&red_is_color, &circle_is_shape]);

    println!("    red_circle = red_is_color ⊕ circle_is_shape");
    println!();

    // Key property: bundled result is similar to both inputs
    println!("    Property: Bundling produces vectors similar to all inputs:");
    println!(
        "      similarity(red_circle, red_is_color)    = {:.4} (~0.7)",
        red_circle.similarity(&red_is_color)
    );
    println!(
        "      similarity(red_circle, circle_is_shape) = {:.4} (~0.7)",
        red_circle.similarity(&circle_is_shape)
    );
    println!();

    // Query the bundled representation
    let query_color = red_circle.bind(&hv_color); // Unbind color role
    let query_shape = red_circle.bind(&hv_shape); // Unbind shape role

    println!("    Querying the bundled representation:");
    println!(
        "      Query 'what color?': similarity to red   = {:.4}",
        query_color.similarity(&hv_red)
    );
    println!(
        "      Query 'what shape?': similarity to circle = {:.4}",
        query_shape.similarity(&hv_circle)
    );
    println!();
}

/// Step 3: LTC Neuron Dynamics
///
/// LTC (Liquid Time-Constant) neurons have state-dependent time constants,
/// allowing them to adapt their response speed to the input. In Symthaea,
/// the neuron STATE is itself a hypervector.
fn demo_ltc_neuron() {
    println!("-------------------------------------------------------------------------------");
    println!("STEP 3: LTC NEURON DYNAMICS");
    println!("-------------------------------------------------------------------------------");
    println!();

    // Create an LTC neuron with default configuration
    let config = UnifiedConfig {
        dimension: HDC_DIMENSION,
        tau_base: 0.1,     // Base time constant
        backbone_tau: 0.5, // State-dependent modulation
        activation: UnifiedActivation::Tanh,
        ..Default::default()
    };

    let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
    let input = ContinuousHV::random_default(123);

    println!("  Created LTC neuron:");
    println!("    Dimension: {}", HDC_DIMENSION);
    println!("    Base tau: 0.1");
    println!("    Initial state norm: {:.6}", neuron.state().norm());
    println!();

    // Euler integration: small steps
    println!("  Euler Integration (100 steps of dt=0.01):");
    let start = Instant::now();
    for _ in 0..100 {
        neuron.evolve(0.01, &input);
    }
    let euler_time = start.elapsed();
    let euler_state = neuron.state().clone();
    println!("    Time: {:?}", euler_time);
    println!("    Final state norm: {:.6}", neuron.state().norm());
    println!();

    // Reset and use closed-form: O(1) jump
    let mut neuron_cf = HdcLtcUnifiedNeuron::new(config, 42);

    println!("  Closed-Form Evolution (single jump of dt=1.0):");
    let start = Instant::now();
    neuron_cf.evolve_closed_form(1.0, &input);
    let cf_time = start.elapsed();
    println!("    Time: {:?}", cf_time);
    println!("    Final state norm: {:.6}", neuron_cf.state().norm());
    println!();

    // Compare results
    let similarity = euler_state.similarity(neuron_cf.state());
    println!("  Comparison:");
    println!(
        "    State similarity (Euler vs Closed-form): {:.4}",
        similarity
    );
    println!(
        "    Speedup: {:.1}x",
        euler_time.as_secs_f64() / cf_time.as_secs_f64().max(1e-9)
    );
    println!();

    // State-dependent time constant
    let tau_for_input = neuron.effective_tau(&input);
    println!("  State-Dependent Dynamics:");
    println!("    Effective tau for current input: {:.4}", tau_for_input);
    println!("    (tau adapts based on input-state relationship)");
    println!();
}

/// Step 4: Phi Measurement
///
/// Phi (Φ) measures integrated information - a key metric from Integrated
/// Information Theory (IIT). Higher phi indicates more consciousness-like
/// information integration.
fn demo_phi_measurement() {
    println!("-------------------------------------------------------------------------------");
    println!("STEP 4: PHI MEASUREMENT");
    println!("-------------------------------------------------------------------------------");
    println!();

    // Create a small network of hypervectors to measure phi
    let nodes: Vec<ContinuousHV> = (0..8)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 * 100))
        .collect();

    println!(
        "  Created network of {} nodes for phi measurement",
        nodes.len()
    );
    println!();

    // Create phi engine with spectral connectivity method
    let engine = PhiEngine::new(PhiMethod::SpectralConnectivity);

    println!("  PhiEngine Configuration:");
    println!("    Method: SpectralConnectivity (algebraic connectivity / Fiedler value)");
    println!("    Complexity: O(n^3) for small networks");
    println!();

    // Measure phi using the compute method
    let start = Instant::now();
    let result = engine.compute(&nodes);
    let phi_time = start.elapsed();

    println!("  Phi Measurement Results:");
    println!("    Phi (Φ) = {:.4}", result.phi);
    println!("    Method: {}", result.method);
    println!("    Computation time: {:?}", phi_time);
    println!();

    // Interpretation
    println!("  Interpretation:");
    println!("    Φ measures how much information the system integrates");
    println!("    Higher Φ = more integrated = more 'conscious-like' processing");
    println!("    Random networks typically have Φ ~ 0.3-0.5");
    println!("    Highly structured networks can achieve Φ > 0.7");
    println!();
}

/// Step 5: Cognitive Loop Integration
///
/// The cognitive loop ties everything together: perception, temporal dynamics,
/// consciousness metrics, and action selection.
fn demo_cognitive_loop() {
    println!("-------------------------------------------------------------------------------");
    println!("STEP 5: COGNITIVE LOOP INTEGRATION");
    println!("-------------------------------------------------------------------------------");
    println!();

    // Create cognitive loop with CfC backend
    let config = CognitiveLoopConfig::with_cfc();

    println!("  Creating CognitiveLoopService with CfC backend...");
    let service = CognitiveLoopService::new(config);

    match service {
        Ok(mut service) => {
            println!("    Success! Cognitive loop initialized.");
            println!();

            // Process several inputs
            println!("  Processing inputs through cognitive loop:");
            println!();

            let inputs = [
                "Hello, I am learning about consciousness.",
                "What patterns emerge in neural processing?",
                "The system integrates information across time.",
            ];

            for (i, input) in inputs.iter().enumerate() {
                let result = service.cycle(input);

                println!("    Input {}: \"{}\"", i + 1, input);
                println!("      Prediction Error: {:.4}", result.prediction_error);
                println!(
                    "      Learning:         {}",
                    if result.learning_occurred {
                        "Yes"
                    } else {
                        "No"
                    }
                );
                println!("      Cycle Time:       {}us", result.cycle_time_us);
                println!();
            }

            // Get consciousness snapshot
            let snapshot = service.consciousness_snapshot();

            println!("  Consciousness Snapshot:");
            println!("    Unified Phi:        {:.4}", snapshot.unified_psi);
            println!("    Temporal Coherence: {:.4}", snapshot.temporal_coherence);
            println!("    Cognitive Depth:    {:?}", snapshot.cognitive_depth);
            println!("    In Flow State:      {}", snapshot.in_flow);
            println!("    Valence:            {:.4}", snapshot.unified_valence);
            println!("    Arousal:            {:.4}", snapshot.unified_arousal);
            println!();

            // Get statistics
            let stats = service.stats();
            println!("  Loop Statistics:");
            println!("    Total cycles:           {}", stats.total_cycles);
            println!("    Learning cycles:        {}", stats.learning_cycles);
            println!(
                "    Avg prediction error:   {:.4}",
                stats.avg_prediction_error
            );
            println!("    Cycles/second:          {:.1}", stats.cycles_per_second);
            println!();
        }
        Err(e) => {
            println!("    Note: Could not initialize cognitive loop: {}", e);
            println!("    This is expected if running without full features enabled.");
            println!();
        }
    }
}