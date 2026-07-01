// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binding vs Bundling: Betti Number Analysis
//!
//! Uses Betti numbers (β₀, β₁, β₂) instead of unity score to avoid saturation.
//! Tests whether HDC binding produces different topological signatures than bundling.
//!
//! Hypothesis: bind(a,b) creates more integrated representations (lower β₀)
//! than bundle(a,b), especially for phenomenal concept pairs.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example binding_betti_analysis --features neural-bridge --release
//! ```

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{LayerExtractor, PoolingMethod, layer_extractor::LayerExtractorConfig};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{HDC_DIMENSION, binary_hv::BinaryHV};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::{
    ConsciousnessTopology, TopologicalAssessment, TopologyConfig,
};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   BINDING vs BUNDLING: BETTI NUMBER ANALYSIS");
    println!("   Do bound pairs show different topology than bundled pairs?");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    // Create pairs
    let phen_pairs: Vec<(&str, &str)> = phenomenal
        .iter()
        .take(40)
        .collect::<Vec<_>>()
        .chunks(2)
        .map(|c| (c[0].as_str(), c[1].as_str()))
        .collect();

    let func_pairs: Vec<(&str, &str)> = functional
        .iter()
        .take(40)
        .collect::<Vec<_>>()
        .chunks(2)
        .map(|c| (c[0].as_str(), c[1].as_str()))
        .collect();

    println!(
        "Created {} phenomenal pairs, {} functional pairs\n",
        phen_pairs.len(),
        func_pairs.len()
    );

    // Load model
    println!("Loading BGE-M3...");
    let load_start = Instant::now();
    let config = LayerExtractorConfig {
        pooling: PoolingMethod::Mean,
        ..Default::default()
    };
    let extractor = LayerExtractor::load(config)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Use Layer 21 (best for phenomenal effect)
    let layer = 21;

    // Topology config with more detailed analysis
    let topology_config = TopologyConfig {
        min_persistence: 0.05, // Lower threshold to catch more features
        max_scale: 1.5,
        num_scales: 15,
        detect_cycles: true,
        detect_voids: true,
    };

    println!("================================================================");
    println!("   EXTRACTING LAYER {} ACTIVATIONS AND COMPOSING", layer);
    println!("================================================================\n");

    // Results storage
    #[derive(Default)]
    struct TopologyResults {
        beta_0: Vec<usize>,
        beta_1: Vec<usize>,
        beta_2: Vec<usize>,
        num_features: Vec<usize>,
        total_persistence: Vec<f64>,
    }

    let mut phen_bind = TopologyResults::default();
    let mut phen_bundle = TopologyResults::default();
    let mut func_bind = TopologyResults::default();
    let mut func_bundle = TopologyResults::default();

    // Process phenomenal pairs
    println!("Processing phenomenal pairs...");
    for (i, (a, b)) in phen_pairs.iter().enumerate() {
        if i % 5 == 0 {
            print!("  {}/{}\\r", i, phen_pairs.len());
        }

        let acts_a = extractor.extract_layers(a, &[layer])?;
        let acts_b = extractor.extract_layers(b, &[layer])?;

        let hv_a = activation_to_hv16(&acts_a[0].activation);
        let hv_b = activation_to_hv16(&acts_b[0].activation);

        // Bind and bundle
        let bound = hv_a.bind(&hv_b);
        let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

        // Analyze topology
        let bound_topo = analyze_topology(&bound, &topology_config);
        let bundle_topo = analyze_topology(&bundled, &topology_config);

        // Record Betti numbers
        phen_bind.beta_0.push(bound_topo.betti.beta_0);
        phen_bind.beta_1.push(bound_topo.betti.beta_1);
        phen_bind.beta_2.push(bound_topo.betti.beta_2);
        phen_bind.num_features.push(bound_topo.features.len());
        phen_bind
            .total_persistence
            .push(total_persistence(&bound_topo));

        phen_bundle.beta_0.push(bundle_topo.betti.beta_0);
        phen_bundle.beta_1.push(bundle_topo.betti.beta_1);
        phen_bundle.beta_2.push(bundle_topo.betti.beta_2);
        phen_bundle.num_features.push(bundle_topo.features.len());
        phen_bundle
            .total_persistence
            .push(total_persistence(&bundle_topo));
    }
    println!("  Done                    ");

    // Process functional pairs
    println!("Processing functional pairs...");
    for (i, (a, b)) in func_pairs.iter().enumerate() {
        if i % 5 == 0 {
            print!("  {}/{}\\r", i, func_pairs.len());
        }

        let acts_a = extractor.extract_layers(a, &[layer])?;
        let acts_b = extractor.extract_layers(b, &[layer])?;

        let hv_a = activation_to_hv16(&acts_a[0].activation);
        let hv_b = activation_to_hv16(&acts_b[0].activation);

        let bound = hv_a.bind(&hv_b);
        let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

        let bound_topo = analyze_topology(&bound, &topology_config);
        let bundle_topo = analyze_topology(&bundled, &topology_config);

        func_bind.beta_0.push(bound_topo.betti.beta_0);
        func_bind.beta_1.push(bound_topo.betti.beta_1);
        func_bind.beta_2.push(bound_topo.betti.beta_2);
        func_bind.num_features.push(bound_topo.features.len());
        func_bind
            .total_persistence
            .push(total_persistence(&bound_topo));

        func_bundle.beta_0.push(bundle_topo.betti.beta_0);
        func_bundle.beta_1.push(bundle_topo.betti.beta_1);
        func_bundle.beta_2.push(bundle_topo.betti.beta_2);
        func_bundle.num_features.push(bundle_topo.features.len());
        func_bundle
            .total_persistence
            .push(total_persistence(&bundle_topo));
    }
    println!("  Done\n");

    // Results
    println!("================================================================");
    println!("   RESULTS: BETTI NUMBERS BY OPERATION AND CONCEPT TYPE");
    println!("================================================================\n");

    println!("Metric      │ Phen Bind │ Phen Bundle │ Func Bind │ Func Bundle");
    println!("────────────┼───────────┼─────────────┼───────────┼────────────");

    println!(
        "β₀ (comps)  │ {:9.2} │ {:11.2} │ {:9.2} │ {:10.2}",
        mean_usize(&phen_bind.beta_0),
        mean_usize(&phen_bundle.beta_0),
        mean_usize(&func_bind.beta_0),
        mean_usize(&func_bundle.beta_0)
    );

    println!(
        "β₁ (cycles) │ {:9.2} │ {:11.2} │ {:9.2} │ {:10.2}",
        mean_usize(&phen_bind.beta_1),
        mean_usize(&phen_bundle.beta_1),
        mean_usize(&func_bind.beta_1),
        mean_usize(&func_bundle.beta_1)
    );

    println!(
        "β₂ (voids)  │ {:9.2} │ {:11.2} │ {:9.2} │ {:10.2}",
        mean_usize(&phen_bind.beta_2),
        mean_usize(&phen_bundle.beta_2),
        mean_usize(&func_bind.beta_2),
        mean_usize(&func_bundle.beta_2)
    );

    println!(
        "# Features  │ {:9.2} │ {:11.2} │ {:9.2} │ {:10.2}",
        mean_usize(&phen_bind.num_features),
        mean_usize(&phen_bundle.num_features),
        mean_usize(&func_bind.num_features),
        mean_usize(&func_bundle.num_features)
    );

    println!(
        "Tot Persist │ {:9.4} │ {:11.4} │ {:9.4} │ {:10.4}",
        mean_f64(&phen_bind.total_persistence),
        mean_f64(&phen_bundle.total_persistence),
        mean_f64(&func_bind.total_persistence),
        mean_f64(&func_bundle.total_persistence)
    );

    // Key comparisons
    println!("\n================================================================");
    println!("   KEY COMPARISONS");
    println!("================================================================\n");

    // 1. Binding vs Bundling effect on β₀ (integration)
    let bind_vs_bundle_phen_b0 = mean_usize(&phen_bind.beta_0) - mean_usize(&phen_bundle.beta_0);
    let bind_vs_bundle_func_b0 = mean_usize(&func_bind.beta_0) - mean_usize(&func_bundle.beta_0);

    println!("β₀ (Components) - Lower = More Integrated:");
    println!("  Phenomenal: Bind-Bundle = {:+.3}", bind_vs_bundle_phen_b0);
    println!("  Functional: Bind-Bundle = {:+.3}", bind_vs_bundle_func_b0);
    println!(
        "  Interaction: {:+.3}",
        bind_vs_bundle_phen_b0 - bind_vs_bundle_func_b0
    );

    // 2. Total persistence comparison
    let bind_vs_bundle_phen_pers =
        mean_f64(&phen_bind.total_persistence) - mean_f64(&phen_bundle.total_persistence);
    let bind_vs_bundle_func_pers =
        mean_f64(&func_bind.total_persistence) - mean_f64(&func_bundle.total_persistence);

    println!("\nTotal Persistence - Higher = More Stable Features:");
    println!(
        "  Phenomenal: Bind-Bundle = {:+.4}",
        bind_vs_bundle_phen_pers
    );
    println!(
        "  Functional: Bind-Bundle = {:+.4}",
        bind_vs_bundle_func_pers
    );
    println!(
        "  Interaction: {:+.4}",
        bind_vs_bundle_phen_pers - bind_vs_bundle_func_pers
    );

    // 3. Statistical tests
    println!("\n================================================================");
    println!("   STATISTICAL TESTS");
    println!("================================================================\n");

    // Test: Does binding produce different β₀ than bundling?
    let phen_bind_b0_f64: Vec<f64> = phen_bind.beta_0.iter().map(|&x| x as f64).collect();
    let phen_bundle_b0_f64: Vec<f64> = phen_bundle.beta_0.iter().map(|&x| x as f64).collect();
    let func_bind_b0_f64: Vec<f64> = func_bind.beta_0.iter().map(|&x| x as f64).collect();
    let func_bundle_b0_f64: Vec<f64> = func_bundle.beta_0.iter().map(|&x| x as f64).collect();

    let p_phen_b0 = permutation_test(&phen_bind_b0_f64, &phen_bundle_b0_f64, 5000);
    let p_func_b0 = permutation_test(&func_bind_b0_f64, &func_bundle_b0_f64, 5000);

    println!("Bind vs Bundle (β₀):");
    println!(
        "  Phenomenal pairs: p = {:.4} {}",
        p_phen_b0,
        if p_phen_b0 < 0.05 { "*" } else { "" }
    );
    println!(
        "  Functional pairs: p = {:.4} {}",
        p_func_b0,
        if p_func_b0 < 0.05 { "*" } else { "" }
    );

    // Test: Does phenomenal show different binding effect than functional?
    let phen_bind_effect: Vec<f64> = phen_bind
        .beta_0
        .iter()
        .zip(phen_bundle.beta_0.iter())
        .map(|(&b, &bu)| b as f64 - bu as f64)
        .collect();
    let func_bind_effect: Vec<f64> = func_bind
        .beta_0
        .iter()
        .zip(func_bundle.beta_0.iter())
        .map(|(&b, &bu)| b as f64 - bu as f64)
        .collect();

    let p_interaction = permutation_test(&phen_bind_effect, &func_bind_effect, 5000);
    println!("\nInteraction (Phen binding effect vs Func binding effect):");
    println!(
        "  p = {:.4} {}",
        p_interaction,
        if p_interaction < 0.05 { "*" } else { "" }
    );

    // Total persistence tests
    let p_phen_pers = permutation_test(
        &phen_bind.total_persistence,
        &phen_bundle.total_persistence,
        5000,
    );
    let p_func_pers = permutation_test(
        &func_bind.total_persistence,
        &func_bundle.total_persistence,
        5000,
    );

    println!("\nBind vs Bundle (Total Persistence):");
    println!(
        "  Phenomenal pairs: p = {:.4} {}",
        p_phen_pers,
        if p_phen_pers < 0.05 { "*" } else { "" }
    );
    println!(
        "  Functional pairs: p = {:.4} {}",
        p_func_pers,
        if p_func_pers < 0.05 { "*" } else { "" }
    );

    // Summary
    println!("\n================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    let binding_integrates = bind_vs_bundle_phen_b0 < 0.0 || bind_vs_bundle_func_b0 < 0.0;
    let phen_specific =
        (bind_vs_bundle_phen_b0 - bind_vs_bundle_func_b0).abs() > 0.1 && p_interaction < 0.1;

    if binding_integrates && phen_specific {
        println!("✓ BINDING CREATES INTEGRATION, ESPECIALLY FOR PHENOMENAL");
        println!("  Binding produces lower β₀ (more unified representations)");
        println!("  Effect is stronger for phenomenal concept pairs");
    } else if binding_integrates {
        println!("◐ BINDING CREATES INTEGRATION (NON-SPECIFIC)");
        println!("  Binding produces lower β₀ for both concept types");
        println!("  No phenomenal-specific enhancement detected");
    } else {
        println!("○ NO CLEAR BINDING INTEGRATION EFFECT");
        println!("  β₀ differences between bind and bundle are small");
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn load_corpus(path: &str) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let mut concepts = Vec::new();
    if let Some(items) = json["concepts"].as_array() {
        for item in items {
            if let Some(text) = item["text"].as_str() {
                concepts.push(text.to_string());
            }
        }
    }
    Ok(concepts)
}

#[cfg(feature = "neural-bridge")]
fn activation_to_hv16(activation: &[f32]) -> BinaryHV {
    let mut expanded = Vec::with_capacity(HDC_DIMENSION);
    let tiles = HDC_DIMENSION / activation.len();
    let remainder = HDC_DIMENSION % activation.len();

    for tile in 0..tiles {
        for (i, &val) in activation.iter().enumerate() {
            let perturbation = ((tile * activation.len() + i) as f32 * 0.001).sin() * 0.01;
            expanded.push(val + perturbation);
        }
    }

    for i in 0..remainder {
        expanded.push(activation[i]);
    }

    BinaryHV::from_bipolar(&expanded)
}

#[cfg(feature = "neural-bridge")]
fn analyze_topology(hv: &BinaryHV, config: &TopologyConfig) -> TopologicalAssessment {
    let mut topology = ConsciousnessTopology::new(config.clone());

    // Add the base state and several permutations to create a point cloud
    topology.add_state(*hv);
    for shift in 1..8 {
        topology.add_state(hv.permute(shift * 50));
    }

    topology.analyze(0.5)
}

#[cfg(feature = "neural-bridge")]
fn total_persistence(assessment: &TopologicalAssessment) -> f64 {
    assessment.features.iter().map(|f| f.persistence).sum()
}

#[cfg(feature = "neural-bridge")]
fn mean_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<usize>() as f64 / values.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn permutation_test(a: &[f64], b: &[f64], n_perm: usize) -> f64 {
    let observed = mean_f64(a) - mean_f64(b);
    let mut combined: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let n_a = a.len();

    let mut more_extreme = 0;
    let mut rng: u64 = 42;

    for _ in 0..n_perm {
        for i in (1..combined.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng as usize) % (i + 1);
            combined.swap(i, j);
        }

        let perm_a = combined[..n_a].iter().sum::<f64>() / n_a as f64;
        let perm_b = combined[n_a..].iter().sum::<f64>() / (combined.len() - n_a) as f64;

        if (perm_a - perm_b).abs() >= observed.abs() {
            more_extreme += 1;
        }
    }

    more_extreme as f64 / n_perm as f64
}
