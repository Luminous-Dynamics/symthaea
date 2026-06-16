// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fine-Grained Phenomenal Corridor Analysis
//!
//! Maps the phenomenal effect at each layer from 17-24 to identify
//! transition points and non-linear jumps in phenomenal structure emergence.
//!
//! ## Research Question
//!
//! How does phenomenal structure emerge across the late-layer corridor?
//! Is there a sharp transition or gradual development?
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example phenomenal_corridor_finegrained --features neural-bridge --release
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
    println!("   FINE-GRAINED PHENOMENAL CORRIDOR ANALYSIS");
    println!("   Mapping phenomenal structure emergence across layers 17-24");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    // Use full corpus for statistical power
    let phenomenal: Vec<_> = phenomenal.into_iter().take(80).collect();
    let functional: Vec<_> = functional.into_iter().take(80).collect();

    println!(
        "Using {} phenomenal, {} functional concepts\n",
        phenomenal.len(),
        functional.len()
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

    // Test ALL layers in the corridor (17-23, the last valid layer)
    let layers: Vec<usize> = (17..=23).collect();

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    println!("================================================================");
    println!("   LAYER-BY-LAYER ANALYSIS");
    println!("================================================================\n");

    #[derive(Default, Clone)]
    struct LayerMetrics {
        phen_unity: Vec<f64>,
        func_unity: Vec<f64>,
        phen_betti0: Vec<f64>,
        phen_betti1: Vec<f64>,
        func_betti0: Vec<f64>,
        func_betti1: Vec<f64>,
        phen_persistence: Vec<f64>,
        func_persistence: Vec<f64>,
    }

    let mut results: Vec<(usize, LayerMetrics)> = Vec::new();

    for &layer in &layers {
        print!("Processing Layer {}... ", layer);
        let mut metrics = LayerMetrics::default();

        // Process phenomenal concepts
        for concept in &phenomenal {
            let acts = extractor.extract_layers(concept, &[layer])?;
            let hv = activation_to_hv16(&acts[0].activation);
            let assessment = analyze_topology(&hv, &topology_config);

            metrics.phen_unity.push(assessment.unity_score);
            metrics.phen_betti0.push(assessment.betti.beta_0 as f64);
            metrics.phen_betti1.push(assessment.betti.beta_1 as f64);
            metrics
                .phen_persistence
                .push(total_persistence(&assessment));
        }

        // Process functional concepts
        for concept in &functional {
            let acts = extractor.extract_layers(concept, &[layer])?;
            let hv = activation_to_hv16(&acts[0].activation);
            let assessment = analyze_topology(&hv, &topology_config);

            metrics.func_unity.push(assessment.unity_score);
            metrics.func_betti0.push(assessment.betti.beta_0 as f64);
            metrics.func_betti1.push(assessment.betti.beta_1 as f64);
            metrics
                .func_persistence
                .push(total_persistence(&assessment));
        }

        println!("Done");
        results.push((layer, metrics));
    }

    // Results table: Unity scores
    println!("\n================================================================");
    println!("   UNITY SCORES BY LAYER");
    println!("================================================================\n");

    println!("Layer │ Phen Unity │ Func Unity │ Difference │ Cohen's d │ p-value");
    println!("──────┼────────────┼────────────┼────────────┼───────────┼─────────");

    let mut peak_layer = 0;
    let mut peak_effect = 0.0f64;

    for (layer, metrics) in &results {
        let phen_mean = mean(&metrics.phen_unity);
        let func_mean = mean(&metrics.func_unity);
        let diff = phen_mean - func_mean;
        let d = cohens_d(&metrics.phen_unity, &metrics.func_unity);
        let p = permutation_test(&metrics.phen_unity, &metrics.func_unity, 5000);

        let sig = if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "**"
        } else if p < 0.05 {
            "*"
        } else {
            ""
        };

        println!(
            "{:5} │ {:10.4} │ {:10.4} │ {:+10.4} │ {:+9.3} │ {:.4}{}",
            layer, phen_mean, func_mean, diff, d, p, sig
        );

        if diff > peak_effect {
            peak_effect = diff;
            peak_layer = *layer;
        }
    }

    // Betti numbers
    println!("\n================================================================");
    println!("   BETTI NUMBERS BY LAYER");
    println!("================================================================\n");

    println!("Layer │ Phen β₀ │ Func β₀ │ Phen β₁ │ Func β₁ │ β₀ p-val │ β₁ p-val");
    println!("──────┼─────────┼─────────┼─────────┼─────────┼──────────┼──────────");

    for (layer, metrics) in &results {
        let p_betti0 = permutation_test(&metrics.phen_betti0, &metrics.func_betti0, 2000);
        let p_betti1 = permutation_test(&metrics.phen_betti1, &metrics.func_betti1, 2000);

        let sig0 = if p_betti0 < 0.05 { "*" } else { "" };
        let sig1 = if p_betti1 < 0.05 { "*" } else { "" };

        println!(
            "{:5} │ {:7.2} │ {:7.2} │ {:7.2} │ {:7.2} │ {:8.4}{} │ {:8.4}{}",
            layer,
            mean(&metrics.phen_betti0),
            mean(&metrics.func_betti0),
            mean(&metrics.phen_betti1),
            mean(&metrics.func_betti1),
            p_betti0,
            sig0,
            p_betti1,
            sig1
        );
    }

    // Total persistence
    println!("\n================================================================");
    println!("   TOTAL PERSISTENCE BY LAYER");
    println!("================================================================\n");

    println!("Layer │ Phen Pers │ Func Pers │ Difference │ p-value");
    println!("──────┼───────────┼───────────┼────────────┼─────────");

    for (layer, metrics) in &results {
        let diff = mean(&metrics.phen_persistence) - mean(&metrics.func_persistence);
        let p = permutation_test(&metrics.phen_persistence, &metrics.func_persistence, 2000);
        let sig = if p < 0.05 { "*" } else { "" };

        println!(
            "{:5} │ {:9.4} │ {:9.4} │ {:+10.4} │ {:.4}{}",
            layer,
            mean(&metrics.phen_persistence),
            mean(&metrics.func_persistence),
            diff,
            p,
            sig
        );
    }

    // Transition analysis: detect non-linear jumps
    println!("\n================================================================");
    println!("   TRANSITION ANALYSIS");
    println!("================================================================\n");

    println!("Analyzing layer-to-layer changes in phenomenal advantage:\n");

    let mut prev_diff = 0.0;
    let mut transitions: Vec<(usize, usize, f64)> = Vec::new();

    for (layer, metrics) in &results {
        let diff = mean(&metrics.phen_unity) - mean(&metrics.func_unity);
        if *layer > 17 {
            let delta = diff - prev_diff;
            transitions.push((*layer - 1, *layer, delta));
        }
        prev_diff = diff;
    }

    println!("Transition │ Δ(Phen-Func) │ Interpretation");
    println!("───────────┼──────────────┼───────────────");

    for (from, to, delta) in &transitions {
        let interp = if *delta > 0.03 {
            "↑ Phenomenal emergence"
        } else if *delta < -0.03 {
            "↓ Phenomenal compression"
        } else {
            "→ Stable"
        };
        println!("  {} → {}   │ {:+12.4} │ {}", from, to, delta, interp);
    }

    // Find largest positive transition
    let max_transition = transitions
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    // Summary
    println!("\n================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    println!(
        "Peak phenomenal effect: Layer {} (diff = {:+.4})",
        peak_layer, peak_effect
    );
    println!(
        "Largest emergence transition: {} → {} (Δ = {:+.4})",
        max_transition.0, max_transition.1, max_transition.2
    );

    // Count significant layers
    let sig_layers: Vec<usize> = results
        .iter()
        .filter(|(_, m)| permutation_test(&m.phen_unity, &m.func_unity, 2000) < 0.05)
        .map(|(l, _)| *l)
        .collect();

    if sig_layers.is_empty() {
        println!("No layers show significant phenomenal advantage");
    } else {
        println!(
            "Significant phenomenal advantage at layers: {:?}",
            sig_layers
        );
    }

    // Corridor characterization
    let early_corridor: Vec<f64> = results
        .iter()
        .filter(|(l, _)| *l >= 17 && *l <= 19)
        .map(|(_, m)| mean(&m.phen_unity) - mean(&m.func_unity))
        .collect();

    let late_corridor: Vec<f64> = results
        .iter()
        .filter(|(l, _)| *l >= 20 && *l <= 22)
        .map(|(_, m)| mean(&m.phen_unity) - mean(&m.func_unity))
        .collect();

    let final_layer: f64 = results
        .iter()
        .find(|(l, _)| *l == 23)
        .map(|(_, m)| mean(&m.phen_unity) - mean(&m.func_unity))
        .unwrap_or(0.0);

    println!("\nCorridor characterization:");
    println!("  Early (17-19): mean diff = {:+.4}", mean(&early_corridor));
    println!("  Middle (20-22): mean diff = {:+.4}", mean(&late_corridor));
    println!("  Final (23): diff = {:+.4}", final_layer);

    if mean(&late_corridor) > mean(&early_corridor) {
        println!("\n✓ PHENOMENAL STRUCTURE PEAKS IN LATE LAYERS (20-23)");
        println!("  Gradual emergence across the phenomenal corridor");
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

    topology.add_state(*hv);
    for shift in 1..5 {
        topology.add_state(hv.permute(shift * 100));
    }

    topology.analyze(0.5)
}

#[cfg(feature = "neural-bridge")]
fn total_persistence(assessment: &TopologicalAssessment) -> f64 {
    assessment.features.iter().map(|f| f.persistence).sum()
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
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(feature = "neural-bridge")]
fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let mean_diff = mean(a) - mean(b);
    let pooled_var = ((a.len() - 1) as f64 * std_dev(a).powi(2)
        + (b.len() - 1) as f64 * std_dev(b).powi(2))
        / (a.len() + b.len() - 2) as f64;
    mean_diff / pooled_var.sqrt()
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
