// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phenomenality Index Validation
//!
//! Tests the XOR binding theory predictions:
//! 1. Cross-class binding (phen+func) should NOT compress
//! 2. Within-class phenomenal binding SHOULD compress
//! 3. The phenomenality index can detect phenomenal content
//!
//! ## Phenomenality Index
//!
//! ```
//! Phenomenality(C₁, C₂) = [Pers(bundle) - Pers(bind)] / Pers(bundle)
//! ```
//!
//! High phenomenality → large compression under binding → shared structure
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example phenomenality_index_validation --features neural-bridge --release
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
    println!("   PHENOMENALITY INDEX VALIDATION");
    println!("   Testing XOR binding theory predictions");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    let phenomenal: Vec<_> = phenomenal.into_iter().take(40).collect();
    let functional: Vec<_> = functional.into_iter().take(40).collect();

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

    // Use Layer 22 (peak phenomenal effect)
    let layer = 22;

    let topology_config = TopologyConfig {
        min_persistence: 0.05,
        max_scale: 1.5,
        num_scales: 15,
        detect_cycles: true,
        detect_voids: true,
    };

    // Extract all concept HVs at Layer 22
    println!("Extracting representations at Layer {}...", layer);
    let mut phen_hvs: Vec<BinaryHV> = Vec::new();
    let mut func_hvs: Vec<BinaryHV> = Vec::new();

    for concept in &phenomenal {
        let acts = extractor.extract_layers(concept, &[layer])?;
        phen_hvs.push(activation_to_hv16(&acts[0].activation));
    }

    for concept in &functional {
        let acts = extractor.extract_layers(concept, &[layer])?;
        func_hvs.push(activation_to_hv16(&acts[0].activation));
    }
    println!("  Done\n");

    // ================================================================
    // TEST 1: Cross-class binding prediction
    // ================================================================
    println!("================================================================");
    println!("   TEST 1: CROSS-CLASS BINDING");
    println!("   Prediction: Phen+Func binding should NOT compress");
    println!("================================================================\n");

    let mut phen_phen_compression: Vec<f64> = Vec::new();
    let mut func_func_compression: Vec<f64> = Vec::new();
    let mut phen_func_compression: Vec<f64> = Vec::new();

    // Phen+Phen pairs (first 20 pairs)
    print!("Computing Phen+Phen compression... ");
    for i in 0..20 {
        let j = (i + 1) % phen_hvs.len();
        let compression = compute_binding_compression(&phen_hvs[i], &phen_hvs[j], &topology_config);
        phen_phen_compression.push(compression);
    }
    println!("Done");

    // Func+Func pairs (first 20 pairs)
    print!("Computing Func+Func compression... ");
    for i in 0..20 {
        let j = (i + 1) % func_hvs.len();
        let compression = compute_binding_compression(&func_hvs[i], &func_hvs[j], &topology_config);
        func_func_compression.push(compression);
    }
    println!("Done");

    // Phen+Func pairs (first 20 pairs)
    print!("Computing Phen+Func compression... ");
    for i in 0..20 {
        let compression = compute_binding_compression(&phen_hvs[i], &func_hvs[i], &topology_config);
        phen_func_compression.push(compression);
    }
    println!("Done\n");

    // Results
    let mean_pp = mean(&phen_phen_compression);
    let mean_ff = mean(&func_func_compression);
    let mean_pf = mean(&phen_func_compression);

    println!("Binding Compression by Pair Type:");
    println!("────────────────────────────────────────");
    println!("  Phen+Phen: {:+.4} (mean)", mean_pp);
    println!("  Func+Func: {:+.4} (mean)", mean_ff);
    println!("  Phen+Func: {:+.4} (mean)", mean_pf);

    // Statistical tests
    let p_pp_vs_pf = permutation_test(&phen_phen_compression, &phen_func_compression, 5000);
    let p_pp_vs_ff = permutation_test(&phen_phen_compression, &func_func_compression, 5000);

    println!("\nStatistical Tests:");
    println!(
        "  Phen+Phen vs Phen+Func: p = {:.4}{}",
        p_pp_vs_pf,
        if p_pp_vs_pf < 0.05 { " *" } else { "" }
    );
    println!(
        "  Phen+Phen vs Func+Func: p = {:.4}{}",
        p_pp_vs_ff,
        if p_pp_vs_ff < 0.05 { " *" } else { "" }
    );

    // Interpretation
    println!("\nInterpretation:");
    if mean_pp > mean_pf && p_pp_vs_pf < 0.05 {
        println!("  ✓ PREDICTION CONFIRMED: Cross-class binding compresses LESS");
        println!(
            "    Phen+Phen compression ({:+.4}) > Phen+Func ({:+.4})",
            mean_pp, mean_pf
        );
        println!("    This supports the 'shared phenomenal signature' hypothesis");
    } else if mean_pp > mean_pf {
        println!("  ◐ TREND SUPPORTS PREDICTION: Cross-class binding compresses less");
        println!(
            "    But difference not statistically significant (p={:.4})",
            p_pp_vs_pf
        );
    } else {
        println!("  ✗ PREDICTION NOT CONFIRMED");
        println!("    Cross-class binding does not show expected pattern");
    }

    // ================================================================
    // TEST 2: Phenomenality Index
    // ================================================================
    println!("\n================================================================");
    println!("   TEST 2: PHENOMENALITY INDEX");
    println!("   Can we detect phenomenal content from binding compression?");
    println!("================================================================\n");

    // Compute phenomenality index for all pair types
    let phen_indices: Vec<f64> = phen_phen_compression.clone();
    let func_indices: Vec<f64> = func_func_compression.clone();
    let cross_indices: Vec<f64> = phen_func_compression.clone();

    println!("Phenomenality Index by Pair Type:");
    println!("────────────────────────────────────────");
    println!(
        "  Phen+Phen: {:.4} ± {:.4}",
        mean(&phen_indices),
        std_dev(&phen_indices)
    );
    println!(
        "  Func+Func: {:.4} ± {:.4}",
        mean(&func_indices),
        std_dev(&func_indices)
    );
    println!(
        "  Phen+Func: {:.4} ± {:.4}",
        mean(&cross_indices),
        std_dev(&cross_indices)
    );

    // Classification accuracy
    println!("\nClassification Test:");
    println!("  Can phenomenality index distinguish pair types?");

    // Use threshold at midpoint between phen and func means
    let threshold = (mean(&phen_indices) + mean(&func_indices)) / 2.0;
    println!("  Threshold: {:.4}", threshold);

    let phen_correct = phen_indices.iter().filter(|&&x| x > threshold).count();
    let func_correct = func_indices.iter().filter(|&&x| x <= threshold).count();
    let phen_accuracy = phen_correct as f64 / phen_indices.len() as f64;
    let func_accuracy = func_correct as f64 / func_indices.len() as f64;
    let overall_accuracy =
        (phen_correct + func_correct) as f64 / (phen_indices.len() + func_indices.len()) as f64;

    println!(
        "  Phen+Phen classification: {}/{} ({:.1}%)",
        phen_correct,
        phen_indices.len(),
        phen_accuracy * 100.0
    );
    println!(
        "  Func+Func classification: {}/{} ({:.1}%)",
        func_correct,
        func_indices.len(),
        func_accuracy * 100.0
    );
    println!("  Overall accuracy: {:.1}%", overall_accuracy * 100.0);

    // Effect size
    let d = cohens_d(&phen_indices, &func_indices);
    println!("\n  Cohen's d (Phen vs Func): {:.3}", d);

    // ================================================================
    // TEST 3: Individual Concept Phenomenality
    // ================================================================
    println!("\n================================================================");
    println!("   TEST 3: CONCEPT-LEVEL PHENOMENALITY SCORES");
    println!("   Average phenomenality when paired with same-class concepts");
    println!("================================================================\n");

    // Compute average phenomenality for each concept
    let mut phen_concept_scores: Vec<(usize, f64)> = Vec::new();
    let mut func_concept_scores: Vec<(usize, f64)> = Vec::new();

    for i in 0..phen_hvs.len().min(20) {
        let mut scores = Vec::new();
        for j in 0..phen_hvs.len().min(20) {
            if i != j {
                scores.push(compute_binding_compression(
                    &phen_hvs[i],
                    &phen_hvs[j],
                    &topology_config,
                ));
            }
        }
        phen_concept_scores.push((i, mean(&scores)));
    }

    for i in 0..func_hvs.len().min(20) {
        let mut scores = Vec::new();
        for j in 0..func_hvs.len().min(20) {
            if i != j {
                scores.push(compute_binding_compression(
                    &func_hvs[i],
                    &func_hvs[j],
                    &topology_config,
                ));
            }
        }
        func_concept_scores.push((i, mean(&scores)));
    }

    // Sort by phenomenality
    phen_concept_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    func_concept_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 Most 'Phenomenal' Phenomenal Concepts:");
    for (i, (idx, score)) in phen_concept_scores.iter().take(5).enumerate() {
        println!(
            "  {}. [{:.4}] {}",
            i + 1,
            score,
            truncate(&phenomenal[*idx], 50)
        );
    }

    println!("\nTop 5 Most 'Phenomenal' Functional Concepts:");
    for (i, (idx, score)) in func_concept_scores.iter().take(5).enumerate() {
        println!(
            "  {}. [{:.4}] {}",
            i + 1,
            score,
            truncate(&functional[*idx], 50)
        );
    }

    println!("\nBottom 5 Least 'Phenomenal' Phenomenal Concepts:");
    for (i, (idx, score)) in phen_concept_scores.iter().rev().take(5).enumerate() {
        println!(
            "  {}. [{:.4}] {}",
            i + 1,
            score,
            truncate(&phenomenal[*idx], 50)
        );
    }

    // ================================================================
    // SUMMARY
    // ================================================================
    println!("\n================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    println!("Cross-Class Binding Test:");
    if mean_pp > mean_pf && p_pp_vs_pf < 0.05 {
        println!("  ✓ CONFIRMED: Phen+Func compresses less than Phen+Phen");
    } else {
        println!("  ◐ INCONCLUSIVE: Trend present but not significant");
    }

    println!("\nPhenomenality Index Validity:");
    if overall_accuracy > 0.65 {
        println!(
            "  ✓ VALID: Can distinguish phenomenal from functional pairs ({:.1}%)",
            overall_accuracy * 100.0
        );
    } else {
        println!(
            "  ◐ WEAK: Classification accuracy below threshold ({:.1}%)",
            overall_accuracy * 100.0
        );
    }

    println!("\nEffect Size:");
    if d.abs() > 0.5 {
        println!("  ✓ MEDIUM-LARGE: Cohen's d = {:.3}", d);
    } else if d.abs() > 0.2 {
        println!("  ◐ SMALL-MEDIUM: Cohen's d = {:.3}", d);
    } else {
        println!("  ✗ NEGLIGIBLE: Cohen's d = {:.3}", d);
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn compute_binding_compression(hv1: &BinaryHV, hv2: &BinaryHV, config: &TopologyConfig) -> f64 {
    let bound = hv1.bind(hv2);
    let bundled = BinaryHV::bundle(&[*hv1, *hv2]);

    let bound_pers = total_persistence(&analyze_topology(&bound, config));
    let bundle_pers = total_persistence(&analyze_topology(&bundled, config));

    if bundle_pers > 0.0 {
        (bundle_pers - bound_pers) / bundle_pers
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
    if pooled_var > 0.0 {
        mean_diff / pooled_var.sqrt()
    } else {
        0.0
    }
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

#[cfg(feature = "neural-bridge")]
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
