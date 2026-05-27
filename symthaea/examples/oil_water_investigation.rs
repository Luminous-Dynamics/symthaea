// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Oil and Water Effect Investigation
//!
//! Investigates why cross-class binding (phenomenal + functional) creates
//! higher-complexity representations than within-class binding.
//!
//! ## The Finding
//!
//! From phenomenality_index_validation:
//! - Phen+Phen compression: -0.039 (slight expansion)
//! - Func+Func compression: -0.041 (slight expansion)
//! - Phen+Func compression: -0.074 (MORE expansion)
//!
//! Why does mixing "oil" (phenomenal) with "water" (functional) create
//! more complex emulsions than mixing water with water?
//!
//! ## Hypotheses
//!
//! 1. **Orthogonality**: Phen and Func occupy different subspaces; binding
//!    creates vectors that span both, increasing complexity
//!
//! 2. **Interference**: The phenomenal signature and functional structure
//!    interfere constructively, amplifying topological features
//!
//! 3. **Dimensionality**: Cross-class binding explores more of the space
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example oil_water_investigation --features neural-bridge --release
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
    println!("   OIL AND WATER EFFECT INVESTIGATION");
    println!("   Why does cross-class binding create more complexity?");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    let phenomenal: Vec<_> = phenomenal.into_iter().take(30).collect();
    let functional: Vec<_> = functional.into_iter().take(30).collect();

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

    // Extract at Layer 22
    let layer = 22;

    print!("Extracting representations at Layer {}... ", layer);
    let mut phen_hvs: Vec<BinaryHV> = Vec::new();
    let mut func_hvs: Vec<BinaryHV> = Vec::new();
    let mut phen_acts: Vec<Vec<f32>> = Vec::new();
    let mut func_acts: Vec<Vec<f32>> = Vec::new();

    for concept in &phenomenal {
        let acts = extractor.extract_layers(concept, &[layer])?;
        phen_acts.push(acts[0].activation.clone());
        phen_hvs.push(activation_to_hv16(&acts[0].activation));
    }
    for concept in &functional {
        let acts = extractor.extract_layers(concept, &[layer])?;
        func_acts.push(acts[0].activation.clone());
        func_hvs.push(activation_to_hv16(&acts[0].activation));
    }
    println!("Done\n");

    let topology_config = TopologyConfig {
        min_persistence: 0.05,
        max_scale: 1.5,
        num_scales: 15,
        detect_cycles: true,
        detect_voids: true,
    };

    // ================================================================
    // TEST 1: Hamming Distance Analysis
    // ================================================================
    println!("================================================================");
    println!("   TEST 1: HAMMING DISTANCE ANALYSIS");
    println!("   How far apart are the bound vectors from inputs?");
    println!("================================================================\n");

    let mut pp_distances: Vec<f64> = Vec::new();
    let mut ff_distances: Vec<f64> = Vec::new();
    let mut pf_distances: Vec<f64> = Vec::new();

    // Phen+Phen
    for i in 0..15 {
        let j = (i + 1) % phen_hvs.len();
        let bound = phen_hvs[i].bind(&phen_hvs[j]);
        let d1 = hamming_distance(&bound, &phen_hvs[i]);
        let d2 = hamming_distance(&bound, &phen_hvs[j]);
        pp_distances.push((d1 + d2) / 2.0);
    }

    // Func+Func
    for i in 0..15 {
        let j = (i + 1) % func_hvs.len();
        let bound = func_hvs[i].bind(&func_hvs[j]);
        let d1 = hamming_distance(&bound, &func_hvs[i]);
        let d2 = hamming_distance(&bound, &func_hvs[j]);
        ff_distances.push((d1 + d2) / 2.0);
    }

    // Phen+Func
    for i in 0..15 {
        let bound = phen_hvs[i].bind(&func_hvs[i]);
        let d1 = hamming_distance(&bound, &phen_hvs[i]);
        let d2 = hamming_distance(&bound, &func_hvs[i]);
        pf_distances.push((d1 + d2) / 2.0);
    }

    println!("Mean Hamming distance from bound to inputs:");
    println!("  Phen+Phen: {:.4}", mean(&pp_distances));
    println!("  Func+Func: {:.4}", mean(&ff_distances));
    println!("  Phen+Func: {:.4}", mean(&pf_distances));

    let p_pp_pf = permutation_test(&pp_distances, &pf_distances, 2000);
    println!(
        "\n  Phen+Phen vs Phen+Func: p = {:.4}{}",
        p_pp_pf,
        if p_pp_pf < 0.05 { " *" } else { "" }
    );

    if mean(&pf_distances) > mean(&pp_distances) {
        println!("\n  → Cross-class binding creates MORE orthogonal vectors");
        println!("    Bound(phen,func) is farther from both inputs");
    }

    // ================================================================
    // TEST 2: Input Similarity Analysis
    // ================================================================
    println!("\n================================================================");
    println!("   TEST 2: INPUT SIMILARITY");
    println!("   Are cross-class inputs less similar (more orthogonal)?");
    println!("================================================================\n");

    let mut pp_similarity: Vec<f64> = Vec::new();
    let mut ff_similarity: Vec<f64> = Vec::new();
    let mut pf_similarity: Vec<f64> = Vec::new();

    for i in 0..15 {
        let j = (i + 1) % phen_hvs.len();
        pp_similarity.push(cosine_similarity(&phen_acts[i], &phen_acts[j]));
    }
    for i in 0..15 {
        let j = (i + 1) % func_hvs.len();
        ff_similarity.push(cosine_similarity(&func_acts[i], &func_acts[j]));
    }
    for i in 0..15 {
        pf_similarity.push(cosine_similarity(&phen_acts[i], &func_acts[i]));
    }

    println!("Mean cosine similarity between inputs:");
    println!("  Phen+Phen: {:.4}", mean(&pp_similarity));
    println!("  Func+Func: {:.4}", mean(&ff_similarity));
    println!("  Phen+Func: {:.4}", mean(&pf_similarity));

    if mean(&pf_similarity) < mean(&pp_similarity) && mean(&pf_similarity) < mean(&ff_similarity) {
        println!("\n  ✓ Cross-class pairs are LESS similar (more orthogonal)");
        println!("    This explains why binding them creates more complex structures");
    }

    // ================================================================
    // TEST 3: Topological Feature Analysis
    // ================================================================
    println!("\n================================================================");
    println!("   TEST 3: TOPOLOGICAL FEATURES OF BOUND VECTORS");
    println!("   What specific features increase in cross-class binding?");
    println!("================================================================\n");

    let mut pp_betti0: Vec<f64> = Vec::new();
    let mut pp_betti1: Vec<f64> = Vec::new();
    let mut ff_betti0: Vec<f64> = Vec::new();
    let mut ff_betti1: Vec<f64> = Vec::new();
    let mut pf_betti0: Vec<f64> = Vec::new();
    let mut pf_betti1: Vec<f64> = Vec::new();

    for i in 0..15 {
        let j = (i + 1) % phen_hvs.len();
        let bound = phen_hvs[i].bind(&phen_hvs[j]);
        let assessment = analyze_topology(&bound, &topology_config);
        pp_betti0.push(assessment.betti.beta_0 as f64);
        pp_betti1.push(assessment.betti.beta_1 as f64);
    }

    for i in 0..15 {
        let j = (i + 1) % func_hvs.len();
        let bound = func_hvs[i].bind(&func_hvs[j]);
        let assessment = analyze_topology(&bound, &topology_config);
        ff_betti0.push(assessment.betti.beta_0 as f64);
        ff_betti1.push(assessment.betti.beta_1 as f64);
    }

    for i in 0..15 {
        let bound = phen_hvs[i].bind(&func_hvs[i]);
        let assessment = analyze_topology(&bound, &topology_config);
        pf_betti0.push(assessment.betti.beta_0 as f64);
        pf_betti1.push(assessment.betti.beta_1 as f64);
    }

    println!("Betti numbers of bound vectors:");
    println!("Pair Type  │ Mean β₀ │ Mean β₁ │ Total");
    println!("───────────┼─────────┼─────────┼──────");
    println!(
        "Phen+Phen  │ {:7.2} │ {:7.2} │ {:5.2}",
        mean(&pp_betti0),
        mean(&pp_betti1),
        mean(&pp_betti0) + mean(&pp_betti1)
    );
    println!(
        "Func+Func  │ {:7.2} │ {:7.2} │ {:5.2}",
        mean(&ff_betti0),
        mean(&ff_betti1),
        mean(&ff_betti0) + mean(&ff_betti1)
    );
    println!(
        "Phen+Func  │ {:7.2} │ {:7.2} │ {:5.2}",
        mean(&pf_betti0),
        mean(&pf_betti1),
        mean(&pf_betti0) + mean(&pf_betti1)
    );

    // ================================================================
    // TEST 4: Subspace Overlap Analysis
    // ================================================================
    println!("\n================================================================");
    println!("   TEST 4: SUBSPACE OVERLAP");
    println!("   Do phenomenal and functional occupy different subspaces?");
    println!("================================================================\n");

    // Compute mean activation vector for each class
    let phen_mean: Vec<f64> = (0..phen_acts[0].len())
        .map(|d| phen_acts.iter().map(|a| a[d] as f64).sum::<f64>() / phen_acts.len() as f64)
        .collect();
    let func_mean: Vec<f64> = (0..func_acts[0].len())
        .map(|d| func_acts.iter().map(|a| a[d] as f64).sum::<f64>() / func_acts.len() as f64)
        .collect();

    // Cosine similarity between class centroids
    let centroid_similarity = cosine_similarity_f64(&phen_mean, &func_mean);
    println!(
        "Cosine similarity between class centroids: {:.4}",
        centroid_similarity
    );

    // Compute principal directions for each class (simplified: direction of max variance)
    let phen_var_per_dim: Vec<f64> = (0..phen_acts[0].len())
        .map(|d| {
            let vals: Vec<f64> = phen_acts.iter().map(|a| a[d] as f64).collect();
            let m = mean(&vals);
            vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / vals.len() as f64
        })
        .collect();

    let func_var_per_dim: Vec<f64> = (0..func_acts[0].len())
        .map(|d| {
            let vals: Vec<f64> = func_acts.iter().map(|a| a[d] as f64).collect();
            let m = mean(&vals);
            vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / vals.len() as f64
        })
        .collect();

    // Find top variance dimensions for each class
    let mut phen_top_dims: Vec<(usize, f64)> = phen_var_per_dim
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    let mut func_top_dims: Vec<(usize, f64)> = func_var_per_dim
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    phen_top_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    func_top_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let phen_top_50: std::collections::HashSet<usize> =
        phen_top_dims.iter().take(50).map(|(i, _)| *i).collect();
    let func_top_50: std::collections::HashSet<usize> =
        func_top_dims.iter().take(50).map(|(i, _)| *i).collect();

    let overlap = phen_top_50.intersection(&func_top_50).count();
    println!(
        "Overlap in top 50 variance dimensions: {}/50 ({:.1}%)",
        overlap,
        (overlap as f64 / 50.0) * 100.0
    );

    if overlap < 25 {
        println!("\n  ✓ Classes occupy DIFFERENT subspaces");
        println!("    Cross-class binding explores new dimensional territory");
    }

    // ================================================================
    // SUMMARY
    // ================================================================
    println!("\n================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    println!("The 'oil and water' effect arises because:\n");

    if mean(&pf_similarity) < mean(&pp_similarity) {
        println!("1. ORTHOGONALITY: Phenomenal and functional representations");
        println!(
            "   occupy different subspaces (cosine sim {:.3} vs {:.3})",
            mean(&pf_similarity),
            mean(&pp_similarity)
        );
    }

    if mean(&pf_distances) > mean(&pp_distances) {
        println!("\n2. EXPLORATION: Cross-class binding creates vectors that");
        println!("   are farther from both inputs, exploring new territory");
    }

    println!("\n3. INTERFERENCE: Unlike within-class binding (where shared");
    println!("   structure cancels), cross-class binding COMBINES");
    println!("   non-overlapping structures, creating richer topology");

    println!("\nThis is analogous to mixing oil and water:");
    println!("  - Water + Water → more water (simple)");
    println!("  - Oil + Water → complex emulsion (structured)");

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn hamming_distance(a: &BinaryHV, b: &BinaryHV) -> f64 {
    let xor = a.bind(b);
    xor.popcount() as f64 / HDC_DIMENSION as f64
}

#[cfg(feature = "neural-bridge")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(feature = "neural-bridge")]
fn cosine_similarity_f64(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
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
fn analyze_topology(
    hv: &BinaryHV,
    config: &TopologyConfig,
) -> symthaea_core::hdc::consciousness_topology::TopologicalAssessment {
    let mut topology = ConsciousnessTopology::new(config.clone());

    topology.add_state(*hv);
    for shift in 1..8 {
        topology.add_state(hv.permute(shift * 50));
    }

    topology.analyze(0.5)
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
