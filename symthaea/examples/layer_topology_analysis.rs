// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Layer-wise Topology Analysis
//!
//! This example analyzes how topological properties of concept representations
//! evolve across transformer layers, testing the hypothesis that phenomenal
//! structure emerges at specific depths.
//!
//! ## Research Question
//!
//! At which layer does the phenomenal/functional distinction emerge?
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example layer_topology_analysis --features neural-bridge
//! ```

use std::time::Instant;

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{LayerExtractor, PoolingMethod, layer_extractor::LayerExtractorConfig};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::binary_hv::BinaryHV;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::{ConsciousnessTopology, TopologyConfig};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!("Run with: cargo run --example layer_topology_analysis --features neural-bridge");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_analysis()
}

#[cfg(feature = "neural-bridge")]
fn run_analysis() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   LAYER-WISE TOPOLOGY ANALYSIS");
    println!("   Which layer encodes phenomenal structure?");
    println!("================================================================\n");

    // Sample concepts for analysis
    let phenomenal_concepts = vec![
        "The subjective experience of seeing red",
        "What it is like to feel pain",
        "Self-awareness that I am thinking",
        "The unified field of conscious awareness",
        "The felt quality of intense joy",
    ];

    let functional_concepts = vec![
        "Recursive function evaluation",
        "Binary search tree traversal",
        "Matrix multiplication operations",
        "Database query optimization",
        "Network packet routing algorithms",
    ];

    println!("Loading BGE-M3 model with layer access...");
    println!("  (This may take a moment on first run)\n");

    let load_start = Instant::now();
    let config = LayerExtractorConfig {
        pooling: PoolingMethod::Mean,
        keep_sequence_activations: false,
        ..Default::default()
    };
    let extractor = LayerExtractor::load(config)?;
    println!(
        "  Model loaded in {:.2}s",
        load_start.elapsed().as_secs_f64()
    );
    println!("  Layers: {}", extractor.num_layers());
    println!("  Hidden dim: {}", extractor.hidden_dim());
    println!(
        "  Device: {}\n",
        if extractor.is_cuda() { "CUDA" } else { "CPU" }
    );

    // Analyze layers [0, 6, 12, 18, 23] for efficiency
    let layers_to_analyze = vec![0, 6, 12, 18, 23];

    println!("================================================================");
    println!("   EXTRACTING LAYER ACTIVATIONS");
    println!("================================================================\n");

    // Extract activations for phenomenal concepts
    println!("Processing phenomenal concepts...");
    let mut phenomenal_by_layer: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers_to_analyze.len()];

    for concept in &phenomenal_concepts {
        let activations = extractor.extract_layers(concept, &layers_to_analyze)?;
        for (i, act) in activations.into_iter().enumerate() {
            phenomenal_by_layer[i].push(act.activation);
        }
    }

    // Extract activations for functional concepts
    println!("Processing functional concepts...\n");
    let mut functional_by_layer: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers_to_analyze.len()];

    for concept in &functional_concepts {
        let activations = extractor.extract_layers(concept, &layers_to_analyze)?;
        for (i, act) in activations.into_iter().enumerate() {
            functional_by_layer[i].push(act.activation);
        }
    }

    // Analyze topology at each layer
    println!("================================================================");
    println!("   LAYER-WISE TOPOLOGY COMPARISON");
    println!("================================================================\n");

    println!("Layer  │ Phen Unity │ Func Unity │ Difference │ Phen β₀ │ Func β₀");
    println!("───────┼────────────┼────────────┼────────────┼─────────┼────────");

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    for (i, layer_idx) in layers_to_analyze.iter().enumerate() {
        // Analyze phenomenal concepts at this layer
        let phen_unity = analyze_class_topology(&phenomenal_by_layer[i], &topology_config);

        // Analyze functional concepts at this layer
        let func_unity = analyze_class_topology(&functional_by_layer[i], &topology_config);

        let diff = phen_unity.0 - func_unity.0;

        println!(
            "{:5}  │ {:10.4} │ {:10.4} │ {:+10.4} │ {:7.2} │ {:7.2}",
            layer_idx, phen_unity.0, func_unity.0, diff, phen_unity.1, func_unity.1
        );
    }

    println!("\n================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    println!("Look for layers where:");
    println!("  1. Phenomenal unity is highest");
    println!("  2. Difference (Phen - Func) is maximized");
    println!("  3. β₀ is lowest (most unified representation)\n");

    println!("Early layers (0-6): Encode surface/syntactic features");
    println!("Middle layers (12): Encode semantic content");
    println!("Late layers (18-23): Encode task-specific representations\n");

    println!("The layer with maximum positive difference may be where");
    println!("'phenomenal structure' emerges in the model.\n");

    println!("================================================================");
    println!("   ANALYSIS COMPLETE");
    println!("================================================================\n");

    Ok(())
}

/// Analyze topology for a class of activations
/// Returns (mean_unity, mean_beta_0)
#[cfg(feature = "neural-bridge")]
fn analyze_class_topology(activations: &[Vec<f32>], config: &TopologyConfig) -> (f64, f64) {
    let mut total_unity = 0.0;
    let mut total_beta_0 = 0.0;

    for activation in activations {
        // Convert to BinaryHV (bipolar)
        let hv = activation_to_hv16(activation);

        // Create topology analyzer
        let mut topology = ConsciousnessTopology::new(config.clone());

        // Add vector and permutations
        topology.add_state(hv);
        for shift in 1..5 {
            topology.add_state(hv.permute(shift * 100));
        }

        let assessment = topology.analyze(0.5);
        total_unity += assessment.unity_score;
        total_beta_0 += assessment.betti.beta_0 as f64;
    }

    let n = activations.len() as f64;
    (total_unity / n, total_beta_0 / n)
}

/// Convert activation vector to BinaryHV
///
/// Since activations are 1024-dim and BinaryHV is 16384-dim, we use
/// random projection: tile the activation 16x and add position-dependent
/// perturbation to break symmetry, then bipolarize.
#[cfg(feature = "neural-bridge")]
fn activation_to_hv16(activation: &[f32]) -> BinaryHV {
    use symthaea_core::hdc::HDC_DIMENSION;

    // Expand 1024-dim to 16384-dim via tiling with perturbation
    let mut expanded = Vec::with_capacity(HDC_DIMENSION);
    let tiles = HDC_DIMENSION / activation.len();

    for tile in 0..tiles {
        for (i, &val) in activation.iter().enumerate() {
            // Add position-dependent perturbation to break symmetry
            let perturbation = ((tile * activation.len() + i) as f32 * 0.001).sin() * 0.01;
            expanded.push(val + perturbation);
        }
    }

    // Bipolarize: positive → 1, negative → -1
    BinaryHV::from_bipolar(&expanded)
}
