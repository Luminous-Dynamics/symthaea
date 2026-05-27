// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Layer 22 Mechanistic Analysis
//!
//! Investigates WHAT specifically in Layer 22 encodes phenomenal structure.
//! Tests whether the effect comes from:
//! 1. Attention patterns (how tokens attend to each other)
//! 2. MLP activations (feedforward transformations)
//! 3. Specific dimensional subspaces
//!
//! ## Approach
//!
//! We decompose L22 representations and test which components
//! carry the phenomenal signal.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example l22_mechanistic_analysis --features neural-bridge --release
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
    println!("   LAYER 22 MECHANISTIC ANALYSIS");
    println!("   What encodes phenomenal structure?");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    let phenomenal: Vec<_> = phenomenal.into_iter().take(50).collect();
    let functional: Vec<_> = functional.into_iter().take(50).collect();

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

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    // ================================================================
    // ANALYSIS 1: Dimensional Subspace Analysis
    // ================================================================
    println!("================================================================");
    println!("   ANALYSIS 1: DIMENSIONAL SUBSPACES");
    println!("   Which dimensions carry the phenomenal signal?");
    println!("================================================================\n");

    // Extract full L22 activations
    let mut phen_activations: Vec<Vec<f32>> = Vec::new();
    let mut func_activations: Vec<Vec<f32>> = Vec::new();

    print!("Extracting L22 activations... ");
    for concept in &phenomenal {
        let acts = extractor.extract_layers(concept, &[22])?;
        phen_activations.push(acts[0].activation.clone());
    }
    for concept in &functional {
        let acts = extractor.extract_layers(concept, &[22])?;
        func_activations.push(acts[0].activation.clone());
    }
    println!("Done\n");

    let dim = phen_activations[0].len();
    println!("Activation dimensionality: {}\n", dim);

    // Compute per-dimension statistics
    let mut dim_discriminability: Vec<(usize, f64)> = Vec::new();

    for d in 0..dim {
        let phen_vals: Vec<f64> = phen_activations.iter().map(|a| a[d] as f64).collect();
        let func_vals: Vec<f64> = func_activations.iter().map(|a| a[d] as f64).collect();

        // Cohen's d for this dimension
        let d_score = cohens_d(&phen_vals, &func_vals);
        dim_discriminability.push((d, d_score));
    }

    // Sort by absolute discriminability
    dim_discriminability.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Top 20 Most Discriminative Dimensions:");
    println!("Dim   │ Cohen's d │ Direction");
    println!("──────┼───────────┼──────────");
    for (d, score) in dim_discriminability.iter().take(20) {
        let dir = if *score > 0.0 {
            "Phen > Func"
        } else {
            "Func > Phen"
        };
        println!("{:5} │ {:+9.4} │ {}", d, score, dir);
    }

    // ================================================================
    // ANALYSIS 2: Subspace Ablation
    // ================================================================
    println!("\n================================================================");
    println!("   ANALYSIS 2: SUBSPACE ABLATION");
    println!("   Does ablating top discriminative dims eliminate effect?");
    println!("================================================================\n");

    let top_dims: Vec<usize> = dim_discriminability
        .iter()
        .take(100)
        .map(|(d, _)| *d)
        .collect();

    // Test ablating different numbers of top dimensions
    let ablation_counts = vec![10, 25, 50, 100];

    println!("Dims Ablated │ Phen Unity │ Func Unity │ Difference │ p-value");
    println!("─────────────┼────────────┼────────────┼────────────┼─────────");

    // Baseline (no ablation)
    let (phen_base, func_base) =
        compute_unity_scores(&phen_activations, &func_activations, &[], &topology_config);
    let base_diff = mean(&phen_base) - mean(&func_base);
    let base_p = permutation_test(&phen_base, &func_base, 2000);
    println!(
        "{:12} │ {:10.4} │ {:10.4} │ {:+10.4} │ {:.4}{}",
        "0 (baseline)",
        mean(&phen_base),
        mean(&func_base),
        base_diff,
        base_p,
        if base_p < 0.05 { "*" } else { "" }
    );

    for &n in &ablation_counts {
        let dims_to_ablate: Vec<usize> = top_dims.iter().take(n).copied().collect();
        let (phen_unity, func_unity) = compute_unity_scores(
            &phen_activations,
            &func_activations,
            &dims_to_ablate,
            &topology_config,
        );

        let diff = mean(&phen_unity) - mean(&func_unity);
        let p = permutation_test(&phen_unity, &func_unity, 2000);
        let effect_remaining = if base_diff != 0.0 {
            (diff / base_diff) * 100.0
        } else {
            0.0
        };

        println!(
            "{:12} │ {:10.4} │ {:10.4} │ {:+10.4} │ {:.4}{} ({:.0}% remaining)",
            n,
            mean(&phen_unity),
            mean(&func_unity),
            diff,
            p,
            if p < 0.05 { "*" } else { "" },
            effect_remaining
        );
    }

    // ================================================================
    // ANALYSIS 3: Activation Statistics
    // ================================================================
    println!("\n================================================================");
    println!("   ANALYSIS 3: ACTIVATION STATISTICS");
    println!("   Do phenomenal concepts have different activation patterns?");
    println!("================================================================\n");

    // Compute activation statistics for each class
    let phen_means: Vec<f64> = phen_activations
        .iter()
        .map(|a| a.iter().map(|&x| x as f64).sum::<f64>() / a.len() as f64)
        .collect();
    let func_means: Vec<f64> = func_activations
        .iter()
        .map(|a| a.iter().map(|&x| x as f64).sum::<f64>() / a.len() as f64)
        .collect();

    let phen_vars: Vec<f64> = phen_activations
        .iter()
        .map(|a| {
            let m = a.iter().map(|&x| x as f64).sum::<f64>() / a.len() as f64;
            a.iter().map(|&x| ((x as f64) - m).powi(2)).sum::<f64>() / a.len() as f64
        })
        .collect();
    let func_vars: Vec<f64> = func_activations
        .iter()
        .map(|a| {
            let m = a.iter().map(|&x| x as f64).sum::<f64>() / a.len() as f64;
            a.iter().map(|&x| ((x as f64) - m).powi(2)).sum::<f64>() / a.len() as f64
        })
        .collect();

    let phen_l2: Vec<f64> = phen_activations
        .iter()
        .map(|a| (a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()).sqrt())
        .collect();
    let func_l2: Vec<f64> = func_activations
        .iter()
        .map(|a| (a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()).sqrt())
        .collect();

    let phen_sparsity: Vec<f64> = phen_activations
        .iter()
        .map(|a| a.iter().filter(|&&x| x.abs() < 0.01).count() as f64 / a.len() as f64)
        .collect();
    let func_sparsity: Vec<f64> = func_activations
        .iter()
        .map(|a| a.iter().filter(|&&x| x.abs() < 0.01).count() as f64 / a.len() as f64)
        .collect();

    println!("Statistic     │ Phenomenal │ Functional │ Cohen's d │ p-value");
    println!("──────────────┼────────────┼────────────┼───────────┼─────────");

    let stats = vec![
        ("Mean activation", &phen_means, &func_means),
        ("Variance", &phen_vars, &func_vars),
        ("L2 norm", &phen_l2, &func_l2),
        ("Sparsity", &phen_sparsity, &func_sparsity),
    ];

    for (name, phen, func) in stats {
        let d = cohens_d(phen, func);
        let p = permutation_test(phen, func, 2000);
        println!(
            "{:13} │ {:10.4} │ {:10.4} │ {:+9.4} │ {:.4}{}",
            name,
            mean(phen),
            mean(func),
            d,
            p,
            if p < 0.05 { "*" } else { "" }
        );
    }

    // ================================================================
    // ANALYSIS 4: Correlation Structure
    // ================================================================
    println!("\n================================================================");
    println!("   ANALYSIS 4: CORRELATION STRUCTURE");
    println!("   Do phenomenal concepts cluster differently?");
    println!("================================================================\n");

    // Compute pairwise correlations within each class
    let phen_correlations = compute_pairwise_correlations(&phen_activations);
    let func_correlations = compute_pairwise_correlations(&func_activations);

    println!("Within-class pairwise correlations:");
    println!(
        "  Phenomenal: mean={:.4}, std={:.4}",
        mean(&phen_correlations),
        std_dev(&phen_correlations)
    );
    println!(
        "  Functional: mean={:.4}, std={:.4}",
        mean(&func_correlations),
        std_dev(&func_correlations)
    );

    let corr_d = cohens_d(&phen_correlations, &func_correlations);
    let corr_p = permutation_test(&phen_correlations, &func_correlations, 2000);
    println!(
        "  Cohen's d: {:+.4}, p={:.4}{}",
        corr_d,
        corr_p,
        if corr_p < 0.05 { " *" } else { "" }
    );

    if mean(&phen_correlations) > mean(&func_correlations) {
        println!("\n  → Phenomenal concepts are MORE correlated with each other");
        println!("    This supports the 'phenomenal signature' hypothesis");
    } else {
        println!("\n  → Functional concepts are more correlated with each other");
    }

    // ================================================================
    // SUMMARY
    // ================================================================
    println!("\n================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    let top_dim_count = dim_discriminability
        .iter()
        .filter(|(_, d)| d.abs() > 0.3)
        .count();
    println!(
        "Discriminative dimensions (|d| > 0.3): {}/{}",
        top_dim_count, dim
    );

    if corr_p < 0.05 && mean(&phen_correlations) > mean(&func_correlations) {
        println!("\n✓ PHENOMENAL SIGNATURE CONFIRMED");
        println!("  Phenomenal concepts share more structure with each other");
        println!("  than functional concepts do.");
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn compute_unity_scores(
    phen_acts: &[Vec<f32>],
    func_acts: &[Vec<f32>],
    dims_to_zero: &[usize],
    config: &TopologyConfig,
) -> (Vec<f64>, Vec<f64>) {
    let mut phen_unity = Vec::new();
    let mut func_unity = Vec::new();

    for act in phen_acts {
        let mut modified = act.clone();
        for &d in dims_to_zero {
            if d < modified.len() {
                modified[d] = 0.0;
            }
        }
        let hv = activation_to_hv16(&modified);
        let assessment = analyze_topology(&hv, config);
        phen_unity.push(assessment.unity_score);
    }

    for act in func_acts {
        let mut modified = act.clone();
        for &d in dims_to_zero {
            if d < modified.len() {
                modified[d] = 0.0;
            }
        }
        let hv = activation_to_hv16(&modified);
        let assessment = analyze_topology(&hv, config);
        func_unity.push(assessment.unity_score);
    }

    (phen_unity, func_unity)
}

#[cfg(feature = "neural-bridge")]
fn compute_pairwise_correlations(activations: &[Vec<f32>]) -> Vec<f64> {
    let mut correlations = Vec::new();
    let n = activations.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let corr = pearson_correlation(&activations[i], &activations[j]);
            correlations.push(corr);
        }
    }

    correlations
}

#[cfg(feature = "neural-bridge")]
fn pearson_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..a.len() {
        let da = (a[i] as f64) - mean_a;
        let db = (b[i] as f64) - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a > 0.0 && var_b > 0.0 {
        cov / (var_a.sqrt() * var_b.sqrt())
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
    for shift in 1..5 {
        topology.add_state(hv.permute(shift * 100));
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
