// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Layer-wise Topology Analysis - Expanded Corpus (100+100)
//!
//! Validates the Layer 23 phenomenal effect with expanded 100-concept corpora.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example layer_topology_expanded --features neural-bridge --release
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
        println!("Run with: cargo run --example layer_topology_expanded --features neural-bridge");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_analysis()
}

#[cfg(feature = "neural-bridge")]
fn run_analysis() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   LAYER-WISE TOPOLOGY ANALYSIS - EXPANDED CORPUS");
    println!("   100 Phenomenal vs 100 Functional Concepts");
    println!("================================================================\n");

    // Load expanded concept corpora
    let phenomenal_corpus =
        load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional_corpus =
        load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    println!("Loaded expanded corpora:");
    println!("  Phenomenal concepts: {}", phenomenal_corpus.len());
    println!("  Functional concepts: {}\n", functional_corpus.len());

    println!("Loading BGE-M3 model with layer access...");
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
    println!(
        "  Device: {}\n",
        if extractor.is_cuda() { "CUDA" } else { "CPU" }
    );

    // Analyze all layers for fine-grained view
    let layers_to_analyze = vec![0, 3, 6, 9, 12, 15, 18, 21, 23];

    println!("================================================================");
    println!("   EXTRACTING LAYER ACTIVATIONS");
    println!("================================================================\n");

    // Extract activations for phenomenal concepts
    println!("Processing phenomenal concepts...");
    let phen_start = Instant::now();
    let mut phenomenal_by_layer: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers_to_analyze.len()];

    for (i, concept) in phenomenal_corpus.iter().enumerate() {
        if i % 20 == 0 {
            print!("  {}/{}\r", i, phenomenal_corpus.len());
        }
        let activations = extractor.extract_layers(concept, &layers_to_analyze)?;
        for (j, act) in activations.into_iter().enumerate() {
            phenomenal_by_layer[j].push(act.activation);
        }
    }
    println!(
        "  Completed {} concepts in {:.1}s",
        phenomenal_corpus.len(),
        phen_start.elapsed().as_secs_f64()
    );

    // Extract activations for functional concepts
    println!("Processing functional concepts...");
    let func_start = Instant::now();
    let mut functional_by_layer: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers_to_analyze.len()];

    for (i, concept) in functional_corpus.iter().enumerate() {
        if i % 20 == 0 {
            print!("  {}/{}\r", i, functional_corpus.len());
        }
        let activations = extractor.extract_layers(concept, &layers_to_analyze)?;
        for (j, act) in activations.into_iter().enumerate() {
            functional_by_layer[j].push(act.activation);
        }
    }
    println!(
        "  Completed {} concepts in {:.1}s\n",
        functional_corpus.len(),
        func_start.elapsed().as_secs_f64()
    );

    // Analyze topology at each layer
    println!("================================================================");
    println!(
        "   LAYER-WISE TOPOLOGY COMPARISON (n={} per class)",
        phenomenal_corpus.len()
    );
    println!("================================================================\n");

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    println!("Layer │ Phen Unity │ Func Unity │   Diff   │ Cohen's d │  p-value  │ Sig");
    println!("──────┼────────────┼────────────┼──────────┼───────────┼───────────┼────");

    let mut layer_results = Vec::new();

    for (i, layer_idx) in layers_to_analyze.iter().enumerate() {
        let phen_scores: Vec<f64> = phenomenal_by_layer[i]
            .iter()
            .map(|act| compute_unity(act, &topology_config))
            .collect();

        let func_scores: Vec<f64> = functional_by_layer[i]
            .iter()
            .map(|act| compute_unity(act, &topology_config))
            .collect();

        let phen_mean = mean(&phen_scores);
        let func_mean = mean(&func_scores);
        let phen_std = std_dev(&phen_scores);
        let func_std = std_dev(&func_scores);

        let diff = phen_mean - func_mean;
        let pooled_std = ((phen_std.powi(2) + func_std.powi(2)) / 2.0).sqrt();
        let cohens_d = if pooled_std > 0.0 {
            diff / pooled_std
        } else {
            0.0
        };

        let p_value = permutation_test(&phen_scores, &func_scores, 10000);
        let sig = if p_value < 0.01 {
            "**"
        } else if p_value < 0.05 {
            "*"
        } else {
            ""
        };

        println!(
            "{:5} │ {:10.4} │ {:10.4} │ {:+8.4} │ {:+9.4} │ {:9.4} │ {:>3}",
            layer_idx, phen_mean, func_mean, diff, cohens_d, p_value, sig
        );

        layer_results.push((*layer_idx, phen_mean, func_mean, diff, cohens_d, p_value));
    }

    // Summary statistics
    println!("\n================================================================");
    println!("   ANALYSIS SUMMARY");
    println!("================================================================\n");

    // Find significant layers
    let significant_layers: Vec<_> = layer_results
        .iter()
        .filter(|(_, _, _, _, _, p)| *p < 0.05)
        .collect();

    if significant_layers.is_empty() {
        println!("No statistically significant differences found at p < 0.05");
    } else {
        println!("Statistically significant layers (p < 0.05):");
        for (layer, phen, func, diff, d, p) in &significant_layers {
            println!(
                "  Layer {:2}: Phen={:.3}, Func={:.3}, Diff={:+.3}, d={:+.3}, p={:.4}",
                layer, phen, func, diff, d, p
            );
        }
    }

    // Best layer
    let best_layer = layer_results
        .iter()
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .unwrap();

    println!(
        "\nLayer with maximum phenomenal advantage: Layer {}",
        best_layer.0
    );
    println!("  Difference: {:+.4}", best_layer.3);
    println!(
        "  Cohen's d: {:+.4} ({})",
        best_layer.4,
        effect_size_label(best_layer.4)
    );
    println!("  p-value: {:.4}", best_layer.5);

    // Bonferroni correction
    let bonferroni_threshold = 0.05 / layers_to_analyze.len() as f64;
    let survives_bonferroni = best_layer.5 < bonferroni_threshold;

    println!("\nMultiple comparison correction:");
    println!("  Bonferroni threshold: {:.4}", bonferroni_threshold);
    println!(
        "  Survives correction: {}",
        if survives_bonferroni { "YES" } else { "NO" }
    );

    // Layer-wise pattern analysis
    println!("\n================================================================");
    println!("   LAYER-WISE PATTERN");
    println!("================================================================\n");

    let early_layers: Vec<_> = layer_results
        .iter()
        .filter(|(l, _, _, _, _, _)| *l <= 6)
        .collect();
    let middle_layers: Vec<_> = layer_results
        .iter()
        .filter(|(l, _, _, _, _, _)| *l > 6 && *l <= 15)
        .collect();
    let late_layers: Vec<_> = layer_results
        .iter()
        .filter(|(l, _, _, _, _, _)| *l > 15)
        .collect();

    let early_mean_diff = early_layers.iter().map(|x| x.3).sum::<f64>() / early_layers.len() as f64;
    let middle_mean_diff =
        middle_layers.iter().map(|x| x.3).sum::<f64>() / middle_layers.len() as f64;
    let late_mean_diff = late_layers.iter().map(|x| x.3).sum::<f64>() / late_layers.len() as f64;

    println!("Mean phenomenal advantage by layer group:");
    println!("  Early (0-6):   {:+.4}", early_mean_diff);
    println!("  Middle (9-15): {:+.4}", middle_mean_diff);
    println!("  Late (18-23):  {:+.4}", late_mean_diff);

    if late_mean_diff > early_mean_diff && late_mean_diff > middle_mean_diff {
        println!("\n  → Phenomenal advantage INCREASES toward late layers");
    } else if middle_mean_diff < 0.0 {
        println!("\n  → Middle layers show functional advantage (semantic encoding)");
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
fn compute_unity(activation: &[f32], config: &TopologyConfig) -> f64 {
    let hv = activation_to_hv16(activation);
    let mut topology = ConsciousnessTopology::new(config.clone());

    topology.add_state(hv);
    for shift in 1..5 {
        topology.add_state(hv.permute(shift * 100));
    }

    let assessment = topology.analyze(0.5);
    assessment.unity_score
}

#[cfg(feature = "neural-bridge")]
fn activation_to_hv16(activation: &[f32]) -> BinaryHV {
    use symthaea_core::hdc::HDC_DIMENSION;

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
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn std_dev(values: &[f64]) -> f64 {
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

#[cfg(feature = "neural-bridge")]
fn effect_size_label(d: f64) -> &'static str {
    match d.abs() {
        x if x < 0.2 => "negligible",
        x if x < 0.5 => "small",
        x if x < 0.8 => "medium",
        _ => "large",
    }
}

#[cfg(feature = "neural-bridge")]
fn permutation_test(group_a: &[f64], group_b: &[f64], n_permutations: usize) -> f64 {
    let observed_diff = mean(group_a) - mean(group_b);

    let mut combined: Vec<f64> = group_a.iter().chain(group_b.iter()).copied().collect();
    let n_a = group_a.len();

    let mut more_extreme = 0;
    let mut rng_state: u64 = 42;

    for _ in 0..n_permutations {
        for i in (1..combined.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            combined.swap(i, j);
        }

        let perm_a: f64 = combined[..n_a].iter().sum::<f64>() / n_a as f64;
        let perm_b: f64 = combined[n_a..].iter().sum::<f64>() / (combined.len() - n_a) as f64;
        let perm_diff = perm_a - perm_b;

        if perm_diff.abs() >= observed_diff.abs() {
            more_extreme += 1;
        }
    }

    more_extreme as f64 / n_permutations as f64
}
