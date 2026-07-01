// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Model Validation
//!
//! Tests whether the late-layer phenomenal effect replicates across different models.
//!
//! Compares BGE-M3 (24 layers, 1024 hidden) with XLM-RoBERTa-base (12 layers, 768 hidden).
//! If the effect appears in both, it's likely architecture-general.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example cross_model_validation --features neural-bridge --release
//! ```

use anyhow::Result;
use std::time::Instant;

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
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   CROSS-MODEL VALIDATION");
    println!("   Does the phenomenal effect generalize across model scales?");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    // Use subset for faster testing
    let phenomenal: Vec<_> = phenomenal.into_iter().take(30).collect();
    let functional: Vec<_> = functional.into_iter().take(30).collect();

    println!(
        "Using {} phenomenal, {} functional concepts\n",
        phenomenal.len(),
        functional.len()
    );

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    // Test Model 1: BGE-M3 (24 layers)
    println!("================================================================");
    println!("   MODEL 1: BGE-M3 (24 layers, 1024 hidden)");
    println!("================================================================\n");

    let bge_results = test_model(
        "BAAI/bge-m3",
        &phenomenal,
        &functional,
        &topology_config,
        vec![18, 21, 23], // Late layers
    )?;

    // Test Model 2: XLM-RoBERTa-base (12 layers)
    println!("\n================================================================");
    println!("   MODEL 2: XLM-RoBERTa-base (12 layers, 768 hidden)");
    println!("================================================================\n");

    let xlm_results = test_model(
        "xlm-roberta-base",
        &phenomenal,
        &functional,
        &topology_config,
        vec![8, 10, 11], // Equivalent late layers (2/3 through network)
    )?;

    // Cross-model comparison
    println!("\n================================================================");
    println!("   CROSS-MODEL COMPARISON");
    println!("================================================================\n");

    println!("Model         │ Best Layer │ Phen Unity │ Func Unity │ Diff   │ p-value");
    println!("──────────────┼────────────┼────────────┼────────────┼────────┼────────");

    // Find best layer for each model
    let bge_best = bge_results
        .iter()
        .max_by(|a, b| {
            (a.phen_mean - a.func_mean)
                .partial_cmp(&(b.phen_mean - b.func_mean))
                .unwrap()
        })
        .unwrap();

    let xlm_best = xlm_results
        .iter()
        .max_by(|a, b| {
            (a.phen_mean - a.func_mean)
                .partial_cmp(&(b.phen_mean - b.func_mean))
                .unwrap()
        })
        .unwrap();

    println!(
        "BGE-M3        │ {:10} │ {:10.4} │ {:10.4} │ {:+6.4} │ {:.4}{}",
        bge_best.layer,
        bge_best.phen_mean,
        bge_best.func_mean,
        bge_best.phen_mean - bge_best.func_mean,
        bge_best.p_value,
        if bge_best.p_value < 0.05 { "*" } else { "" }
    );

    println!(
        "XLM-RoBERTa   │ {:10} │ {:10.4} │ {:10.4} │ {:+6.4} │ {:.4}{}",
        xlm_best.layer,
        xlm_best.phen_mean,
        xlm_best.func_mean,
        xlm_best.phen_mean - xlm_best.func_mean,
        xlm_best.p_value,
        if xlm_best.p_value < 0.05 { "*" } else { "" }
    );

    // Interpretation
    println!("\n================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    let bge_effect = bge_best.phen_mean - bge_best.func_mean;
    let xlm_effect = xlm_best.phen_mean - xlm_best.func_mean;
    let bge_sig = bge_best.p_value < 0.05;
    let xlm_sig = xlm_best.p_value < 0.05;

    if bge_effect > 0.0 && xlm_effect > 0.0 && bge_sig && xlm_sig {
        println!("✓ EFFECT REPLICATES ACROSS MODELS");
        println!("  Both models show significant phenomenal advantage in late layers");
        println!("  This suggests the effect is scale-independent");
    } else if bge_effect > 0.0 && xlm_effect > 0.0 {
        println!("◐ CONSISTENT DIRECTION, VARIABLE SIGNIFICANCE");
        println!("  Both show phenomenal > functional in late layers");
        println!(
            "  BGE-M3: {} | XLM-RoBERTa: {}",
            if bge_sig {
                "significant"
            } else {
                "not significant"
            },
            if xlm_sig {
                "significant"
            } else {
                "not significant"
            }
        );
    } else if bge_effect > 0.0 && xlm_effect <= 0.0 {
        println!("○ EFFECT MAY BE MODEL-SPECIFIC");
        println!("  BGE-M3 shows phenomenal advantage");
        println!("  XLM-RoBERTa-base shows different pattern");
        println!("  Effect may depend on model scale or training");
    } else {
        println!("✗ NO CONSISTENT PHENOMENAL EFFECT");
    }

    // Layer depth analysis
    println!("\n----------------------------------------------------------------");
    println!("Layer Depth Analysis:");
    println!(
        "  BGE-M3 best layer: {} / 24 ({:.0}% depth)",
        bge_best.layer,
        (bge_best.layer as f64 / 24.0) * 100.0
    );
    println!(
        "  XLM-RoBERTa best layer: {} / 12 ({:.0}% depth)",
        xlm_best.layer,
        (xlm_best.layer as f64 / 12.0) * 100.0
    );

    let bge_depth = bge_best.layer as f64 / 24.0;
    let xlm_depth = xlm_best.layer as f64 / 12.0;
    if (bge_depth - xlm_depth).abs() < 0.15 {
        println!("\n  → Both models show effect at similar relative depth");
        println!("    This suggests a consistent architectural pattern");
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
struct LayerResult {
    layer: usize,
    phen_mean: f64,
    func_mean: f64,
    p_value: f64,
}

#[cfg(feature = "neural-bridge")]
fn test_model(
    model_id: &str,
    phenomenal: &[String],
    functional: &[String],
    topology_config: &TopologyConfig,
    layers: Vec<usize>,
) -> Result<Vec<LayerResult>> {
    println!("Loading {}...", model_id);
    let load_start = Instant::now();

    let config = LayerExtractorConfig {
        model_id: model_id.to_string(),
        pooling: PoolingMethod::Mean,
        ..Default::default()
    };

    let extractor = match LayerExtractor::load(config) {
        Ok(e) => e,
        Err(e) => {
            println!("  Failed to load model: {}", e);
            println!("  Skipping this model...\n");
            // Return empty results
            return Ok(layers
                .iter()
                .map(|&l| LayerResult {
                    layer: l,
                    phen_mean: 0.0,
                    func_mean: 0.0,
                    p_value: 1.0,
                })
                .collect());
        }
    };
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    let mut results = Vec::new();

    for &layer in &layers {
        println!("Processing layer {}...", layer);

        // Extract phenomenal
        let mut phen_unity = Vec::new();
        for (i, concept) in phenomenal.iter().enumerate() {
            if i % 10 == 0 {
                print!("  Phen {}/{}\\r", i, phenomenal.len());
            }
            match extractor.extract_layers(concept, &[layer]) {
                Ok(acts) => {
                    let hv = activation_to_hv16(&acts[0].activation);
                    phen_unity.push(compute_unity(&hv, topology_config));
                }
                Err(_) => continue,
            }
        }

        // Extract functional
        let mut func_unity = Vec::new();
        for (i, concept) in functional.iter().enumerate() {
            if i % 10 == 0 {
                print!("  Func {}/{}\\r", i, functional.len());
            }
            match extractor.extract_layers(concept, &[layer]) {
                Ok(acts) => {
                    let hv = activation_to_hv16(&acts[0].activation);
                    func_unity.push(compute_unity(&hv, topology_config));
                }
                Err(_) => continue,
            }
        }
        println!("  Done                    ");

        if phen_unity.is_empty() || func_unity.is_empty() {
            println!("  No valid activations, skipping...");
            continue;
        }

        let phen_mean = mean(&phen_unity);
        let func_mean = mean(&func_unity);
        let p_value = permutation_test(&phen_unity, &func_unity, 5000);

        println!(
            "  Layer {}: Phen={:.4}, Func={:.4}, Diff={:+.4}, p={:.4}{}",
            layer,
            phen_mean,
            func_mean,
            phen_mean - func_mean,
            p_value,
            if p_value < 0.05 { "*" } else { "" }
        );

        results.push(LayerResult {
            layer,
            phen_mean,
            func_mean,
            p_value,
        });
    }

    Ok(results)
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
    use symthaea_core::hdc::HDC_DIMENSION;

    let mut expanded = Vec::with_capacity(HDC_DIMENSION);
    let tiles = HDC_DIMENSION / activation.len();
    let remainder = HDC_DIMENSION % activation.len();

    for tile in 0..tiles {
        for (i, &val) in activation.iter().enumerate() {
            let perturbation = ((tile * activation.len() + i) as f32 * 0.001).sin() * 0.01;
            expanded.push(val + perturbation);
        }
    }

    // Handle remainder if dimensions don't divide evenly
    for i in 0..remainder {
        expanded.push(activation[i]);
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
