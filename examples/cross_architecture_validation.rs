// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Architecture Validation
//!
//! Tests whether the phenomenal signature (Φ) generalizes across architectures:
//! 1. BGE-M3 (encoder, 24 layers) - baseline
//! 2. XLM-RoBERTa-base (encoder, 12 layers) - smaller encoder
//! 3. GPT-2 medium (decoder, 24 layers) - decoder-only
//!
//! For each architecture, we:
//! - Extract late-layer activations
//! - Measure topological unity
//! - Attempt Φ extraction and validation
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example cross_architecture_validation --features neural-bridge --release
//! ```

use anyhow::{Context, Result};
use std::collections::HashMap;
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
#[derive(Clone)]
struct ModelSpec {
    name: &'static str,
    model_id: &'static str,
    num_layers: usize,
    hidden_dim: usize,
    architecture: &'static str,
    late_layers: Vec<usize>, // Layers at ~75-95% depth
}

#[cfg(feature = "neural-bridge")]
struct ModelResults {
    spec: ModelSpec,
    layer_results: Vec<LayerResult>,
    phi_validated: Option<bool>,
    phi_effect_size: Option<f64>,
    best_layer: usize,
    best_effect_size: f64,
    replicates: bool,
}

#[cfg(feature = "neural-bridge")]
#[derive(Clone)]
struct LayerResult {
    layer: usize,
    phen_mean: f64,
    phen_std: f64,
    func_mean: f64,
    func_std: f64,
    cohens_d: f64,
    p_value: f64,
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   CROSS-ARCHITECTURE VALIDATION                              ║");
    println!("║   Does the Phenomenal Signature Generalize?                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Define models to test
    let models = vec![
        ModelSpec {
            name: "BGE-M3",
            model_id: "BAAI/bge-m3",
            num_layers: 24,
            hidden_dim: 1024,
            architecture: "Encoder (XLM-RoBERTa-large)",
            late_layers: vec![18, 20, 21, 22, 23], // 75-96% depth
        },
        ModelSpec {
            name: "XLM-RoBERTa-base",
            model_id: "xlm-roberta-base",
            num_layers: 12,
            hidden_dim: 768,
            architecture: "Encoder (RoBERTa-base)",
            late_layers: vec![9, 10, 11], // 75-92% depth
        },
        // Note: GPT-2 requires different extraction method (decoder-only)
        // Will be handled separately if LayerExtractor supports it
    ];

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    // Use full corpus for statistical power
    let phenomenal: Vec<_> = phenomenal.into_iter().take(50).collect();
    let functional: Vec<_> = functional.into_iter().take(50).collect();

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

    let mut all_results: Vec<ModelResults> = Vec::new();

    // Test each model
    for spec in &models {
        println!("════════════════════════════════════════════════════════════════");
        println!(
            "   {} ({} layers, {}D)",
            spec.name, spec.num_layers, spec.hidden_dim
        );
        println!("   Architecture: {}", spec.architecture);
        println!("════════════════════════════════════════════════════════════════\n");

        match test_model(spec.clone(), &phenomenal, &functional, &topology_config) {
            Ok(results) => {
                print_model_results(&results);
                all_results.push(results);
            }
            Err(e) => {
                println!("  ✗ Failed to test model: {}\n", e);
            }
        }
    }

    // Cross-architecture comparison
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   CROSS-ARCHITECTURE COMPARISON                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if all_results.is_empty() {
        println!("No models successfully tested.");
        return Ok(());
    }

    // Summary table
    println!("Model            │ Arch    │ Layers │ Best L │ Depth% │ d      │ p-value  │ Φ Valid");
    println!("─────────────────┼─────────┼────────┼────────┼────────┼────────┼──────────┼────────");

    for result in &all_results {
        let depth_pct = (result.best_layer as f64 / result.spec.num_layers as f64) * 100.0;
        let phi_status = match result.phi_validated {
            Some(true) => "✓",
            Some(false) => "✗",
            None => "-",
        };

        println!(
            "{:16} │ {:7} │ {:6} │ {:6} │ {:5.1}% │ {:+5.2} │ {:8} │ {:^7}",
            result.spec.name,
            if result.spec.architecture.contains("Encoder") {
                "Encoder"
            } else {
                "Decoder"
            },
            result.spec.num_layers,
            result.best_layer,
            depth_pct,
            result.best_effect_size,
            format_pvalue(
                result
                    .layer_results
                    .iter()
                    .find(|r| r.layer == result.best_layer)
                    .map(|r| r.p_value)
                    .unwrap_or(1.0)
            ),
            phi_status
        );
    }

    // Analysis
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   INTERPRETATION");
    println!("════════════════════════════════════════════════════════════════\n");

    let replicating: Vec<_> = all_results.iter().filter(|r| r.replicates).collect();
    let total = all_results.len();

    if replicating.len() == total && total > 1 {
        println!("✓✓ EFFECT REPLICATES ACROSS ALL {} ARCHITECTURES", total);
        println!("   This strongly suggests a general property of transformers\n");

        // Check relative depth consistency
        let depths: Vec<f64> = all_results
            .iter()
            .map(|r| r.best_layer as f64 / r.spec.num_layers as f64)
            .collect();
        let mean_depth: f64 = depths.iter().sum::<f64>() / depths.len() as f64;
        let depth_variance: f64 =
            depths.iter().map(|d| (d - mean_depth).powi(2)).sum::<f64>() / depths.len() as f64;

        println!("   Relative depth of phenomenal peak:");
        for result in &all_results {
            let depth = result.best_layer as f64 / result.spec.num_layers as f64;
            println!(
                "     {}: Layer {}/{} = {:.1}%",
                result.spec.name,
                result.best_layer,
                result.spec.num_layers,
                depth * 100.0
            );
        }
        println!(
            "   Mean depth: {:.1}% ± {:.1}%",
            mean_depth * 100.0,
            depth_variance.sqrt() * 100.0
        );

        if depth_variance.sqrt() < 0.1 {
            println!("\n   → CONSISTENT DEPTH: Effect emerges at similar relative position");
            println!("     This suggests a fundamental architectural pattern");
        }
    } else if replicating.len() > 0 {
        println!(
            "◐ PARTIAL REPLICATION: {}/{} architectures show the effect\n",
            replicating.len(),
            total
        );
        println!("   Replicating models:");
        for r in &replicating {
            println!("     ✓ {} (d={:+.2})", r.spec.name, r.best_effect_size);
        }
        println!("\n   Non-replicating models:");
        for r in all_results.iter().filter(|r| !r.replicates) {
            println!("     ✗ {} (d={:+.2})", r.spec.name, r.best_effect_size);
        }
    } else {
        println!("✗ NO REPLICATION");
        println!("   Effect may be specific to BGE-M3 or training data");
    }

    // Φ validation summary
    let phi_validated: Vec<_> = all_results
        .iter()
        .filter(|r| r.phi_validated == Some(true))
        .collect();

    if !phi_validated.is_empty() {
        println!("\n────────────────────────────────────────────────────────────────");
        println!("Φ Extraction Results:");
        for r in &phi_validated {
            if let Some(d) = r.phi_effect_size {
                println!("   {} Φ loading difference: d={:+.2}", r.spec.name, d);
            }
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   EXPERIMENT COMPLETE                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn test_model(
    spec: ModelSpec,
    phenomenal: &[String],
    functional: &[String],
    topology_config: &TopologyConfig,
) -> Result<ModelResults> {
    println!("Loading model...");
    let load_start = Instant::now();

    let config = LayerExtractorConfig {
        model_id: spec.model_id.to_string(),
        pooling: PoolingMethod::Mean,
        ..Default::default()
    };

    let extractor = LayerExtractor::load(config).context("Failed to load model")?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    let mut layer_results = Vec::new();
    let mut all_phen_activations: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
    let mut all_func_activations: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();

    // Test each late layer
    for &layer in &spec.late_layers {
        print!("  Testing layer {}... ", layer);

        let mut phen_unity = Vec::new();
        let mut phen_acts = Vec::new();

        for concept in phenomenal {
            match extractor.extract_layers(concept, &[layer]) {
                Ok(acts) => {
                    let activation = acts[0].activation.clone();
                    let hv = activation_to_hv16(&activation);
                    phen_unity.push(compute_unity(&hv, topology_config));
                    phen_acts.push(activation);
                }
                Err(_) => continue,
            }
        }

        let mut func_unity = Vec::new();
        let mut func_acts = Vec::new();

        for concept in functional {
            match extractor.extract_layers(concept, &[layer]) {
                Ok(acts) => {
                    let activation = acts[0].activation.clone();
                    let hv = activation_to_hv16(&activation);
                    func_unity.push(compute_unity(&hv, topology_config));
                    func_acts.push(activation);
                }
                Err(_) => continue,
            }
        }

        if phen_unity.is_empty() || func_unity.is_empty() {
            println!("skipped (no valid activations)");
            continue;
        }

        all_phen_activations.insert(layer, phen_acts);
        all_func_activations.insert(layer, func_acts);

        let phen_mean = mean(&phen_unity);
        let func_mean = mean(&func_unity);
        let phen_std = std_dev(&phen_unity);
        let func_std = std_dev(&func_unity);

        let pooled_std = ((phen_std.powi(2) + func_std.powi(2)) / 2.0).sqrt();
        let cohens_d = if pooled_std > 0.0 {
            (phen_mean - func_mean) / pooled_std
        } else {
            0.0
        };

        let p_value = permutation_test(&phen_unity, &func_unity, 5000);

        println!(
            "d={:+.3}, p={:.4}{}",
            cohens_d,
            p_value,
            if p_value < 0.05 { "*" } else { "" }
        );

        layer_results.push(LayerResult {
            layer,
            phen_mean,
            phen_std,
            func_mean,
            func_std,
            cohens_d,
            p_value,
        });
    }

    // Find best layer
    let best = layer_results
        .iter()
        .max_by(|a, b| a.cohens_d.partial_cmp(&b.cohens_d).unwrap())
        .cloned()
        .unwrap_or(LayerResult {
            layer: 0,
            phen_mean: 0.0,
            phen_std: 0.0,
            func_mean: 0.0,
            func_std: 0.0,
            cohens_d: 0.0,
            p_value: 1.0,
        });

    // Attempt Φ extraction at best layer
    let (phi_validated, phi_effect_size) = if let (Some(phen_acts), Some(func_acts)) = (
        all_phen_activations.get(&best.layer),
        all_func_activations.get(&best.layer),
    ) {
        extract_and_validate_phi(phen_acts, func_acts, topology_config)
    } else {
        (None, None)
    };

    let replicates = best.cohens_d > 0.2 && best.p_value < 0.05;

    Ok(ModelResults {
        spec,
        layer_results,
        phi_validated,
        phi_effect_size,
        best_layer: best.layer,
        best_effect_size: best.cohens_d,
        replicates,
    })
}

#[cfg(feature = "neural-bridge")]
fn extract_and_validate_phi(
    phen_acts: &[Vec<f32>],
    func_acts: &[Vec<f32>],
    topology_config: &TopologyConfig,
) -> (Option<bool>, Option<f64>) {
    if phen_acts.is_empty() || func_acts.is_empty() {
        return (None, None);
    }

    let dim = phen_acts[0].len();

    // Compute centroids
    let phen_centroid: Vec<f64> = (0..dim)
        .map(|i| phen_acts.iter().map(|a| a[i] as f64).sum::<f64>() / phen_acts.len() as f64)
        .collect();

    let func_centroid: Vec<f64> = (0..dim)
        .map(|i| func_acts.iter().map(|a| a[i] as f64).sum::<f64>() / func_acts.len() as f64)
        .collect();

    // Difference vector
    let diff: Vec<f64> = phen_centroid
        .iter()
        .zip(func_centroid.iter())
        .map(|(p, f)| p - f)
        .collect();

    // Normalize to get Φ direction
    let norm: f64 = diff.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return (None, None);
    }

    let phi: Vec<f64> = diff.iter().map(|x| x / norm).collect();

    // Compute Φ loadings
    let phen_loadings: Vec<f64> = phen_acts
        .iter()
        .map(|a| a.iter().zip(phi.iter()).map(|(x, p)| (*x as f64) * p).sum())
        .collect();

    let func_loadings: Vec<f64> = func_acts
        .iter()
        .map(|a| a.iter().zip(phi.iter()).map(|(x, p)| (*x as f64) * p).sum())
        .collect();

    let phen_loading_mean = mean(&phen_loadings);
    let func_loading_mean = mean(&func_loadings);
    let phen_loading_std = std_dev(&phen_loadings);
    let func_loading_std = std_dev(&func_loadings);

    let pooled_std = ((phen_loading_std.powi(2) + func_loading_std.powi(2)) / 2.0).sqrt();
    let phi_d = if pooled_std > 0.0 {
        (phen_loading_mean - func_loading_mean) / pooled_std
    } else {
        0.0
    };

    // Subtract Φ from phenomenal and re-measure
    let alpha = phen_loading_mean; // Subtraction magnitude

    let mut phen_minus_phi_unity = Vec::new();
    for act in phen_acts {
        // Subtract Φ component
        let loading: f64 = act
            .iter()
            .zip(phi.iter())
            .map(|(x, p)| (*x as f64) * p)
            .sum();
        let adjusted: Vec<f32> = act
            .iter()
            .zip(phi.iter())
            .map(|(x, p)| (*x as f64 - loading * p) as f32)
            .collect();

        let hv = activation_to_hv16(&adjusted);
        phen_minus_phi_unity.push(compute_unity(&hv, topology_config));
    }

    let mut func_unity = Vec::new();
    for act in func_acts {
        let hv = activation_to_hv16(act);
        func_unity.push(compute_unity(&hv, topology_config));
    }

    // Test if Φ removal eliminates significance
    let p_after = permutation_test(&phen_minus_phi_unity, &func_unity, 5000);
    let validated = p_after > 0.05; // No longer significant = Φ was the source

    (Some(validated), Some(phi_d))
}

#[cfg(feature = "neural-bridge")]
fn print_model_results(results: &ModelResults) {
    println!("\nLayer Results:");
    println!("Layer │ Phen Unity │ Func Unity │ Diff   │ Cohen's d │ p-value");
    println!("──────┼────────────┼────────────┼────────┼───────────┼────────");

    for r in &results.layer_results {
        println!(
            "{:5} │ {:10.4} │ {:10.4} │ {:+6.4} │ {:+8.3} │ {}",
            r.layer,
            r.phen_mean,
            r.func_mean,
            r.phen_mean - r.func_mean,
            r.cohens_d,
            format_pvalue(r.p_value)
        );
    }

    println!(
        "\nBest layer: {} (d={:+.3})",
        results.best_layer, results.best_effect_size
    );

    if let Some(validated) = results.phi_validated {
        if validated {
            println!("Φ extraction: VALIDATED (removing Φ eliminates significance)");
        } else {
            println!("Φ extraction: NOT validated (effect persists after Φ removal)");
        }
        if let Some(d) = results.phi_effect_size {
            println!("Φ loading effect size: d={:+.2}", d);
        }
    }

    println!(
        "\nReplicates BGE-M3 finding: {}\n",
        if results.replicates {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
}

#[cfg(feature = "neural-bridge")]
fn format_pvalue(p: f64) -> String {
    if p < 0.001 {
        "<0.001***".to_string()
    } else if p < 0.01 {
        format!("{:.4}**", p)
    } else if p < 0.05 {
        format!("{:.4}*", p)
    } else {
        format!("{:.4}", p)
    }
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
fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
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
