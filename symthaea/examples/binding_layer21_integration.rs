// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binding + Layer 21 Integration Experiment
//!
//! Tests whether bound phenomenal concept pairs show stronger Layer 21 effects.
//!
//! Hypothesis: bind(qualia_a, qualia_b) will show higher topological unity at
//! Layer 21 than bundle(qualia_a, qualia_b), specifically for phenomenal pairs.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example binding_layer21_integration --features neural-bridge --release
//! ```

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{LayerExtractor, PoolingMethod, layer_extractor::LayerExtractorConfig};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{HDC_DIMENSION, binary_hv::BinaryHV};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::{ConsciousnessTopology, TopologyConfig};

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
    println!("   BINDING + LAYER 21 INTEGRATION");
    println!("   Do bound phenomenal pairs show stronger Layer 21 effects?");
    println!("================================================================\n");

    // Create concept pairs from expanded corpus
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    // Create pairs (first 25 pairs from each class)
    let phen_pairs: Vec<(&str, &str)> = phenomenal
        .iter()
        .take(50)
        .collect::<Vec<_>>()
        .chunks(2)
        .map(|c| (c[0].as_str(), c[1].as_str()))
        .collect();

    let func_pairs: Vec<(&str, &str)> = functional
        .iter()
        .take(50)
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
    println!("Loading BGE-M3 with layer access...");
    let load_start = Instant::now();
    let config = LayerExtractorConfig {
        pooling: PoolingMethod::Mean,
        ..Default::default()
    };
    let extractor = LayerExtractor::load(config)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    // Test at layers 18, 21, 23
    let test_layers = vec![18, 21, 23];

    println!("================================================================");
    println!("   EXTRACTING AND COMPOSING AT EACH LAYER");
    println!("================================================================\n");

    #[derive(Default)]
    struct LayerResults {
        phen_bind_unity: Vec<f64>,
        phen_bundle_unity: Vec<f64>,
        func_bind_unity: Vec<f64>,
        func_bundle_unity: Vec<f64>,
    }

    let mut results_by_layer: Vec<(usize, LayerResults)> = Vec::new();

    for &layer in &test_layers {
        println!("Processing Layer {}...", layer);
        let mut results = LayerResults::default();

        // Process phenomenal pairs
        for (i, (a, b)) in phen_pairs.iter().enumerate() {
            if i % 10 == 0 {
                print!("  Phen {}/{}\r", i, phen_pairs.len());
            }

            let acts_a = extractor.extract_layers(a, &[layer])?;
            let acts_b = extractor.extract_layers(b, &[layer])?;

            let hv_a = activation_to_hv16(&acts_a[0].activation);
            let hv_b = activation_to_hv16(&acts_b[0].activation);

            let bound = hv_a.bind(&hv_b);
            let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

            results
                .phen_bind_unity
                .push(compute_unity(&bound, &topology_config));
            results
                .phen_bundle_unity
                .push(compute_unity(&bundled, &topology_config));
        }

        // Process functional pairs
        for (i, (a, b)) in func_pairs.iter().enumerate() {
            if i % 10 == 0 {
                print!("  Func {}/{}\r", i, func_pairs.len());
            }

            let acts_a = extractor.extract_layers(a, &[layer])?;
            let acts_b = extractor.extract_layers(b, &[layer])?;

            let hv_a = activation_to_hv16(&acts_a[0].activation);
            let hv_b = activation_to_hv16(&acts_b[0].activation);

            let bound = hv_a.bind(&hv_b);
            let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

            results
                .func_bind_unity
                .push(compute_unity(&bound, &topology_config));
            results
                .func_bundle_unity
                .push(compute_unity(&bundled, &topology_config));
        }

        println!("  Done                    ");
        results_by_layer.push((layer, results));
    }

    // Analysis
    println!("\n================================================================");
    println!("   RESULTS: UNITY SCORES BY LAYER AND OPERATION");
    println!("================================================================\n");

    println!("Layer │ Phen Bind │ Phen Bundle │ Func Bind │ Func Bundle");
    println!("──────┼───────────┼─────────────┼───────────┼────────────");

    for (layer, results) in &results_by_layer {
        let phen_bind = mean(&results.phen_bind_unity);
        let phen_bundle = mean(&results.phen_bundle_unity);
        let func_bind = mean(&results.func_bind_unity);
        let func_bundle = mean(&results.func_bundle_unity);

        println!(
            "{:5} │ {:9.4} │ {:11.4} │ {:9.4} │ {:10.4}",
            layer, phen_bind, phen_bundle, func_bind, func_bundle
        );
    }

    // Key comparisons
    println!("\n================================================================");
    println!("   KEY COMPARISONS");
    println!("================================================================\n");

    for (layer, results) in &results_by_layer {
        let phen_bind = mean(&results.phen_bind_unity);
        let phen_bundle = mean(&results.phen_bundle_unity);
        let func_bind = mean(&results.func_bind_unity);
        let func_bundle = mean(&results.func_bundle_unity);

        println!("Layer {}:", layer);

        // Phenomenal advantage (bind)
        let phen_adv_bind = phen_bind - func_bind;
        println!("  Phenomenal advantage (BIND): {:+.4}", phen_adv_bind);

        // Phenomenal advantage (bundle)
        let phen_adv_bundle = phen_bundle - func_bundle;
        println!("  Phenomenal advantage (BUNDLE): {:+.4}", phen_adv_bundle);

        // Binding vs bundling for phenomenal
        let bind_adv_phen = phen_bind - phen_bundle;
        println!("  Binding advantage (phenomenal): {:+.4}", bind_adv_phen);

        // Interaction: Does binding specifically help phenomenal more than functional?
        let interaction = (phen_bind - phen_bundle) - (func_bind - func_bundle);
        println!("  Interaction (bind helps phen more?): {:+.4}", interaction);

        // Statistical test for interaction
        let p_interaction = permutation_test_interaction(
            &results.phen_bind_unity,
            &results.phen_bundle_unity,
            &results.func_bind_unity,
            &results.func_bundle_unity,
            5000,
        );
        println!(
            "  Interaction p-value: {:.4} {}\n",
            p_interaction,
            if p_interaction < 0.05 { "*" } else { "" }
        );
    }

    // Summary
    println!("================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    // Find best layer for interaction
    let mut best_interaction = (0, 0.0f64, 1.0f64);
    for (layer, results) in &results_by_layer {
        let phen_bind = mean(&results.phen_bind_unity);
        let phen_bundle = mean(&results.phen_bundle_unity);
        let func_bind = mean(&results.func_bind_unity);
        let func_bundle = mean(&results.func_bundle_unity);

        let interaction = (phen_bind - phen_bundle) - (func_bind - func_bundle);
        let p = permutation_test_interaction(
            &results.phen_bind_unity,
            &results.phen_bundle_unity,
            &results.func_bind_unity,
            &results.func_bundle_unity,
            5000,
        );

        if interaction > best_interaction.1 {
            best_interaction = (*layer, interaction, p);
        }
    }

    if best_interaction.1 > 0.0 && best_interaction.2 < 0.05 {
        println!("✓ BINDING SPECIFICALLY HELPS PHENOMENAL CONCEPTS");
        println!("  Best layer: {}", best_interaction.0);
        println!("  Interaction effect: {:+.4}", best_interaction.1);
        println!("  p-value: {:.4}", best_interaction.2);
    } else if best_interaction.1 < 0.0 {
        println!("✗ BUNDLING HELPS PHENOMENAL CONCEPTS MORE THAN BINDING");
    } else {
        println!("○ NO SIGNIFICANT BINDING × CONCEPT-TYPE INTERACTION");
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

    for tile in 0..tiles {
        for (i, &val) in activation.iter().enumerate() {
            let perturbation = ((tile * activation.len() + i) as f32 * 0.001).sin() * 0.01;
            expanded.push(val + perturbation);
        }
    }

    BinaryHV::from_bipolar(&expanded)
}

#[cfg(feature = "neural-bridge")]
fn compute_unity(hv: &BinaryHV, config: &TopologyConfig) -> f64 {
    let mut topology = ConsciousnessTopology::new(config.clone());

    topology.add_state(*hv);
    for shift in 1..5 {
        topology.add_state(hv.permute(shift * 100));
    }

    topology.analyze(0.5).unity_score
}

#[cfg(feature = "neural-bridge")]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn permutation_test_interaction(
    phen_bind: &[f64],
    phen_bundle: &[f64],
    func_bind: &[f64],
    func_bundle: &[f64],
    n_perm: usize,
) -> f64 {
    let observed = (mean(phen_bind) - mean(phen_bundle)) - (mean(func_bind) - mean(func_bundle));

    let mut data: Vec<(f64, usize, usize)> = Vec::new();
    for &v in phen_bind {
        data.push((v, 0, 0));
    }
    for &v in phen_bundle {
        data.push((v, 0, 1));
    }
    for &v in func_bind {
        data.push((v, 1, 0));
    }
    for &v in func_bundle {
        data.push((v, 1, 1));
    }

    let mut more_extreme = 0;
    let mut rng: u64 = 42;

    for _ in 0..n_perm {
        // Shuffle values
        for i in (1..data.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng as usize) % (i + 1);
            let tmp = data[i].0;
            data[i].0 = data[j].0;
            data[j].0 = tmp;
        }

        let pb: f64 = data
            .iter()
            .filter(|d| d.1 == 0 && d.2 == 0)
            .map(|d| d.0)
            .sum::<f64>()
            / data.iter().filter(|d| d.1 == 0 && d.2 == 0).count() as f64;
        let pbu: f64 = data
            .iter()
            .filter(|d| d.1 == 0 && d.2 == 1)
            .map(|d| d.0)
            .sum::<f64>()
            / data.iter().filter(|d| d.1 == 0 && d.2 == 1).count() as f64;
        let fb: f64 = data
            .iter()
            .filter(|d| d.1 == 1 && d.2 == 0)
            .map(|d| d.0)
            .sum::<f64>()
            / data.iter().filter(|d| d.1 == 1 && d.2 == 0).count() as f64;
        let fbu: f64 = data
            .iter()
            .filter(|d| d.1 == 1 && d.2 == 1)
            .map(|d| d.0)
            .sum::<f64>()
            / data.iter().filter(|d| d.1 == 1 && d.2 == 1).count() as f64;

        let perm_int = (pb - pbu) - (fb - fbu);
        if perm_int.abs() >= observed.abs() {
            more_extreme += 1;
        }
    }

    more_extreme as f64 / n_perm as f64
}
