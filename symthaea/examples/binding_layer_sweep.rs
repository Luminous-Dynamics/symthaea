// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binding Persistence Anomaly Investigation
//!
//! Investigates why binding reduces persistence for phenomenal but not functional concepts.
//! Tests across multiple layers to see if this effect is layer-specific.
//!
//! ## Hypothesis
//!
//! Binding may create "tighter" representations for phenomenal concepts because:
//! 1. Phenomenal concepts have higher intrinsic variance (more to compress)
//! 2. Binding's XOR operation eliminates redundant structure
//! 3. The effect may be strongest at integration layers (around 21)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example binding_layer_sweep --features neural-bridge --release
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
    println!("   BINDING PERSISTENCE ANOMALY: LAYER SWEEP");
    println!("   Why does binding reduce persistence for phenomenal concepts?");
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
        "Using {} phenomenal pairs, {} functional pairs\n",
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

    // Test across multiple layers
    let layers = vec![6, 12, 18, 21, 23];

    let topology_config = TopologyConfig {
        min_persistence: 0.05,
        max_scale: 1.5,
        num_scales: 15,
        detect_cycles: true,
        detect_voids: true,
    };

    println!("================================================================");
    println!("   LAYER-WISE BINDING EFFECT ON PERSISTENCE");
    println!("================================================================\n");

    #[derive(Default, Clone)]
    struct LayerResults {
        phen_bind_pers: Vec<f64>,
        phen_bundle_pers: Vec<f64>,
        func_bind_pers: Vec<f64>,
        func_bundle_pers: Vec<f64>,
    }

    let mut results_by_layer: Vec<(usize, LayerResults)> = Vec::new();

    for &layer in &layers {
        println!("Processing Layer {}...", layer);
        let mut results = LayerResults::default();

        // Process phenomenal pairs
        for (i, (a, b)) in phen_pairs.iter().enumerate() {
            if i % 5 == 0 {
                print!("  Phen {}/{}\\r", i, phen_pairs.len());
            }

            let acts_a = extractor.extract_layers(a, &[layer])?;
            let acts_b = extractor.extract_layers(b, &[layer])?;

            let hv_a = activation_to_hv16(&acts_a[0].activation);
            let hv_b = activation_to_hv16(&acts_b[0].activation);

            let bound = hv_a.bind(&hv_b);
            let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

            let bound_topo = analyze_topology(&bound, &topology_config);
            let bundle_topo = analyze_topology(&bundled, &topology_config);

            results.phen_bind_pers.push(total_persistence(&bound_topo));
            results
                .phen_bundle_pers
                .push(total_persistence(&bundle_topo));
        }

        // Process functional pairs
        for (i, (a, b)) in func_pairs.iter().enumerate() {
            if i % 5 == 0 {
                print!("  Func {}/{}\\r", i, func_pairs.len());
            }

            let acts_a = extractor.extract_layers(a, &[layer])?;
            let acts_b = extractor.extract_layers(b, &[layer])?;

            let hv_a = activation_to_hv16(&acts_a[0].activation);
            let hv_b = activation_to_hv16(&acts_b[0].activation);

            let bound = hv_a.bind(&hv_b);
            let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

            let bound_topo = analyze_topology(&bound, &topology_config);
            let bundle_topo = analyze_topology(&bundled, &topology_config);

            results.func_bind_pers.push(total_persistence(&bound_topo));
            results
                .func_bundle_pers
                .push(total_persistence(&bundle_topo));
        }
        println!("  Done                    ");

        results_by_layer.push((layer, results));
    }

    // Results table
    println!("\n================================================================");
    println!("   RESULTS: BINDING EFFECT (Bind - Bundle) ON PERSISTENCE");
    println!("================================================================\n");

    println!("Layer │ Phen Effect │ Func Effect │ Difference │ Phen p-val │ Interaction p");
    println!("──────┼─────────────┼─────────────┼────────────┼────────────┼──────────────");

    let mut strongest_layer = 0;
    let mut strongest_interaction = 0.0f64;

    for (layer, results) in &results_by_layer {
        let phen_effect = mean(&results.phen_bind_pers) - mean(&results.phen_bundle_pers);
        let func_effect = mean(&results.func_bind_pers) - mean(&results.func_bundle_pers);
        let interaction = phen_effect - func_effect;

        let p_phen = permutation_test(&results.phen_bind_pers, &results.phen_bundle_pers, 2000);

        // Interaction test
        let phen_diffs: Vec<f64> = results
            .phen_bind_pers
            .iter()
            .zip(results.phen_bundle_pers.iter())
            .map(|(b, bu)| b - bu)
            .collect();
        let func_diffs: Vec<f64> = results
            .func_bind_pers
            .iter()
            .zip(results.func_bundle_pers.iter())
            .map(|(b, bu)| b - bu)
            .collect();
        let p_interaction = permutation_test(&phen_diffs, &func_diffs, 2000);

        println!(
            "{:5} │ {:+11.4} │ {:+11.4} │ {:+10.4} │ {:10.4}{} │ {:12.4}{}",
            layer,
            phen_effect,
            func_effect,
            interaction,
            p_phen,
            if p_phen < 0.05 { "*" } else { " " },
            p_interaction,
            if p_interaction < 0.05 { "*" } else { " " }
        );

        if interaction.abs() > strongest_interaction.abs() {
            strongest_interaction = interaction;
            strongest_layer = *layer;
        }
    }

    // Analysis: Why does this happen?
    println!("\n================================================================");
    println!("   ANALYSIS: VARIANCE AND NORM COMPARISON");
    println!("================================================================\n");

    println!("Comparing intrinsic properties of bound vs bundled representations:\n");

    // Re-extract for detailed analysis at the strongest layer
    let analysis_layer = strongest_layer;
    println!(
        "Analyzing Layer {} (strongest interaction)...\n",
        analysis_layer
    );

    let mut phen_bind_norms = Vec::new();
    let mut phen_bundle_norms = Vec::new();
    let mut func_bind_norms = Vec::new();
    let mut func_bundle_norms = Vec::new();

    let mut phen_bind_vars = Vec::new();
    let mut phen_bundle_vars = Vec::new();
    let mut func_bind_vars = Vec::new();
    let mut func_bundle_vars = Vec::new();

    for (a, b) in phen_pairs.iter().take(10) {
        let acts_a = extractor.extract_layers(a, &[analysis_layer])?;
        let acts_b = extractor.extract_layers(b, &[analysis_layer])?;

        let hv_a = activation_to_hv16(&acts_a[0].activation);
        let hv_b = activation_to_hv16(&acts_b[0].activation);

        let bound = hv_a.bind(&hv_b);
        let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

        // Compute norms and variances
        phen_bind_norms.push(hv_norm(&bound));
        phen_bundle_norms.push(hv_norm(&bundled));
        phen_bind_vars.push(hv_variance(&bound));
        phen_bundle_vars.push(hv_variance(&bundled));
    }

    for (a, b) in func_pairs.iter().take(10) {
        let acts_a = extractor.extract_layers(a, &[analysis_layer])?;
        let acts_b = extractor.extract_layers(b, &[analysis_layer])?;

        let hv_a = activation_to_hv16(&acts_a[0].activation);
        let hv_b = activation_to_hv16(&acts_b[0].activation);

        let bound = hv_a.bind(&hv_b);
        let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

        func_bind_norms.push(hv_norm(&bound));
        func_bundle_norms.push(hv_norm(&bundled));
        func_bind_vars.push(hv_variance(&bound));
        func_bundle_vars.push(hv_variance(&bundled));
    }

    println!("Metric     │ Phen Bind │ Phen Bundle │ Func Bind │ Func Bundle");
    println!("───────────┼───────────┼─────────────┼───────────┼────────────");
    println!(
        "Mean Norm  │ {:9.2} │ {:11.2} │ {:9.2} │ {:10.2}",
        mean(&phen_bind_norms),
        mean(&phen_bundle_norms),
        mean(&func_bind_norms),
        mean(&func_bundle_norms)
    );
    println!(
        "Mean Var   │ {:9.4} │ {:11.4} │ {:9.4} │ {:10.4}",
        mean(&phen_bind_vars),
        mean(&phen_bundle_vars),
        mean(&func_bind_vars),
        mean(&func_bundle_vars)
    );

    // Interpretation
    println!("\n================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    let phen_effect_21 = results_by_layer
        .iter()
        .find(|(l, _)| *l == 21)
        .map(|(_, r)| mean(&r.phen_bind_pers) - mean(&r.phen_bundle_pers))
        .unwrap_or(0.0);

    if phen_effect_21 < -0.1 {
        println!("FINDING: Binding REDUCES persistence for phenomenal concepts\n");
        println!("Possible explanations:");
        println!("");
        println!("1. COMPRESSION HYPOTHESIS:");
        println!("   Phenomenal concepts have high intrinsic variance/redundancy.");
        println!("   Binding (XOR) removes redundant structure, creating tighter");
        println!("   representations with fewer persistent topological features.");
        println!("");
        println!("2. ORTHOGONALIZATION HYPOTHESIS:");
        println!("   Binding creates orthogonal representations. Phenomenal concepts");
        println!("   may have more correlated structure that gets orthogonalized,");
        println!("   reducing apparent topological complexity.");
        println!("");
        println!("3. INTEGRATION HYPOTHESIS:");
        println!("   Binding truly integrates phenomenal pairs into unified wholes,");
        println!("   which paradoxically have SIMPLER topology (fewer holes, cycles)");
        println!("   than superimposed (bundled) representations.");
    } else {
        println!("FINDING: No clear binding effect on phenomenal persistence");
    }

    println!(
        "\nStrongest layer-specific effect at Layer {}",
        strongest_layer
    );
    println!("Interaction magnitude: {:+.4}", strongest_interaction);

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
fn hv_norm(hv: &BinaryHV) -> f64 {
    // Use popcount as a measure of density
    hv.popcount() as f64
}

#[cfg(feature = "neural-bridge")]
fn hv_variance(hv: &BinaryHV) -> f64 {
    // Variance of the bipolar interpretation
    let bipolar = hv.to_bipolar();
    let mean_val: f64 = bipolar.iter().map(|&x| x as f64).sum::<f64>() / bipolar.len() as f64;
    let variance: f64 = bipolar
        .iter()
        .map(|&x| ((x as f64) - mean_val).powi(2))
        .sum::<f64>()
        / bipolar.len() as f64;
    variance
}

#[cfg(feature = "neural-bridge")]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn permutation_test(a: &[f64], b: &[f64], n_perm: usize) -> f64 {
    let observed = mean(a) - mean(b);
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
