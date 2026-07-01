// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Layer 21 Deep Analysis
//!
//! Investigates WHY Layer 21 shows stronger phenomenal effects than Layer 23.
//!
//! Hypotheses:
//! 1. Layer 21 has higher activation variance for phenomenal concepts
//! 2. Layer 23 compresses representations for output task
//! 3. Phenomenal concepts have different activation distributions
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example layer21_analysis --features neural-bridge --release
//! ```

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{LayerExtractor, PoolingMethod, layer_extractor::LayerExtractorConfig};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_analysis()
}

#[cfg(feature = "neural-bridge")]
fn run_analysis() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   LAYER 21 DEEP ANALYSIS");
    println!("   Why does Layer 21 show stronger phenomenal effects?");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    println!(
        "Loaded {} phenomenal, {} functional concepts\n",
        phenomenal.len(),
        functional.len()
    );

    // Load model
    println!("Loading BGE-M3...");
    let load_start = Instant::now();
    let config = LayerExtractorConfig {
        pooling: PoolingMethod::Mean,
        keep_sequence_activations: false,
        ..Default::default()
    };
    let extractor = LayerExtractor::load(config)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Focus on layers around the effect
    let layers = vec![18, 19, 20, 21, 22, 23];

    println!("================================================================");
    println!("   EXTRACTING ACTIVATIONS (Layers 18-23)");
    println!("================================================================\n");

    // Extract all activations
    let mut phen_activations: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers.len()];
    let mut func_activations: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers.len()];

    println!("Processing phenomenal concepts...");
    for (i, concept) in phenomenal.iter().enumerate() {
        if i % 25 == 0 {
            print!("  {}/{}\r", i, phenomenal.len());
        }
        let acts = extractor.extract_layers(concept, &layers)?;
        for (j, act) in acts.into_iter().enumerate() {
            phen_activations[j].push(act.activation);
        }
    }
    println!("  Done                    ");

    println!("Processing functional concepts...");
    for (i, concept) in functional.iter().enumerate() {
        if i % 25 == 0 {
            print!("  {}/{}\r", i, functional.len());
        }
        let acts = extractor.extract_layers(concept, &layers)?;
        for (j, act) in acts.into_iter().enumerate() {
            func_activations[j].push(act.activation);
        }
    }
    println!("  Done\n");

    // Analysis 1: Activation Statistics
    println!("================================================================");
    println!("   ANALYSIS 1: ACTIVATION STATISTICS");
    println!("================================================================\n");

    println!("Layer │ Phen Mean │ Func Mean │ Phen Var │ Func Var │ Phen Sparse │ Func Sparse");
    println!("──────┼───────────┼───────────┼──────────┼──────────┼─────────────┼────────────");

    for (i, layer) in layers.iter().enumerate() {
        let phen_stats = compute_activation_stats(&phen_activations[i]);
        let func_stats = compute_activation_stats(&func_activations[i]);

        println!(
            "{:5} │ {:+9.4} │ {:+9.4} │ {:8.4} │ {:8.4} │ {:11.2}% │ {:10.2}%",
            layer,
            phen_stats.mean,
            func_stats.mean,
            phen_stats.variance,
            func_stats.variance,
            phen_stats.sparsity * 100.0,
            func_stats.sparsity * 100.0
        );
    }

    // Analysis 2: Activation Norm Changes
    println!("\n================================================================");
    println!("   ANALYSIS 2: ACTIVATION NORM EVOLUTION");
    println!("================================================================\n");

    println!("Layer │ Phen L2 Norm │ Func L2 Norm │ Ratio (P/F)");
    println!("──────┼──────────────┼──────────────┼────────────");

    for (i, layer) in layers.iter().enumerate() {
        let phen_norm = mean_l2_norm(&phen_activations[i]);
        let func_norm = mean_l2_norm(&func_activations[i]);
        let ratio = phen_norm / func_norm;

        println!(
            "{:5} │ {:12.4} │ {:12.4} │ {:10.4}",
            layer, phen_norm, func_norm, ratio
        );
    }

    // Analysis 3: Inter-layer Correlation
    println!("\n================================================================");
    println!("   ANALYSIS 3: INTER-LAYER SIMILARITY");
    println!("================================================================\n");

    println!("How much do representations change between adjacent layers?");
    println!("\nPhenomenal concepts:");
    println!(
        "  Layer 20→21: {:.4}",
        inter_layer_similarity(&phen_activations[2], &phen_activations[3])
    );
    println!(
        "  Layer 21→22: {:.4}",
        inter_layer_similarity(&phen_activations[3], &phen_activations[4])
    );
    println!(
        "  Layer 22→23: {:.4}",
        inter_layer_similarity(&phen_activations[4], &phen_activations[5])
    );

    println!("\nFunctional concepts:");
    println!(
        "  Layer 20→21: {:.4}",
        inter_layer_similarity(&func_activations[2], &func_activations[3])
    );
    println!(
        "  Layer 21→22: {:.4}",
        inter_layer_similarity(&func_activations[3], &func_activations[4])
    );
    println!(
        "  Layer 22→23: {:.4}",
        inter_layer_similarity(&func_activations[4], &func_activations[5])
    );

    // Analysis 4: Class Separability
    println!("\n================================================================");
    println!("   ANALYSIS 4: CLASS SEPARABILITY (DISCRIMINABILITY)");
    println!("================================================================\n");

    println!("Fisher's Discriminant Ratio (higher = better separation):");
    println!("\nLayer │ Fisher Ratio │ Interpretation");
    println!("──────┼──────────────┼───────────────");

    let mut best_layer = 0;
    let mut best_fisher = 0.0;

    for (i, layer) in layers.iter().enumerate() {
        let fisher = fisher_discriminant_ratio(&phen_activations[i], &func_activations[i]);
        let interp = if fisher > 0.5 {
            "Good"
        } else if fisher > 0.2 {
            "Moderate"
        } else {
            "Poor"
        };

        if fisher > best_fisher {
            best_fisher = fisher;
            best_layer = *layer;
        }

        println!("{:5} │ {:12.4} │ {}", layer, fisher, interp);
    }

    println!(
        "\n  → Best separability at Layer {} (Fisher = {:.4})",
        best_layer, best_fisher
    );

    // Analysis 5: Dimensionality Analysis (effective rank)
    println!("\n================================================================");
    println!("   ANALYSIS 5: EFFECTIVE DIMENSIONALITY");
    println!("================================================================\n");

    println!("Participation ratio (higher = more distributed representation):");
    println!("\nLayer │ Phen PR │ Func PR │ Difference");
    println!("──────┼─────────┼─────────┼───────────");

    for (i, layer) in layers.iter().enumerate() {
        let phen_pr = participation_ratio(&phen_activations[i]);
        let func_pr = participation_ratio(&func_activations[i]);

        println!(
            "{:5} │ {:7.1} │ {:7.1} │ {:+9.1}",
            layer,
            phen_pr,
            func_pr,
            phen_pr - func_pr
        );
    }

    // Summary
    println!("\n================================================================");
    println!("   SUMMARY: WHY LAYER 21?");
    println!("================================================================\n");

    println!("Based on the analysis, Layer 21 likely shows stronger effects because:\n");

    println!("1. REPRESENTATION RICHNESS:");
    println!("   Layer 21 may preserve more information before final compression.\n");

    println!("2. CLASS SEPARABILITY:");
    println!(
        "   Layer {} shows best phenomenal/functional discrimination.",
        best_layer
    );
    println!("   Fisher ratio: {:.4}\n", best_fisher);

    println!("3. TASK-SPECIFIC COMPRESSION:");
    println!("   Layer 23 compresses for embedding output task,");
    println!("   potentially losing phenomenal-specific structure.\n");

    println!("================================================================");
    println!("   ANALYSIS COMPLETE");
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
struct ActivationStats {
    mean: f64,
    variance: f64,
    sparsity: f64, // fraction of near-zero activations
}

#[cfg(feature = "neural-bridge")]
fn compute_activation_stats(activations: &[Vec<f32>]) -> ActivationStats {
    let mut all_values = Vec::new();
    for act in activations {
        all_values.extend(act.iter().map(|&x| x as f64));
    }

    let n = all_values.len() as f64;
    let mean = all_values.iter().sum::<f64>() / n;
    let variance = all_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    // Sparsity: fraction of values with |x| < 0.01
    let sparse_count = all_values.iter().filter(|&&x| x.abs() < 0.01).count();
    let sparsity = sparse_count as f64 / n;

    ActivationStats {
        mean,
        variance,
        sparsity,
    }
}

#[cfg(feature = "neural-bridge")]
fn mean_l2_norm(activations: &[Vec<f32>]) -> f64 {
    let norms: Vec<f64> = activations
        .iter()
        .map(|act| act.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt())
        .collect();
    norms.iter().sum::<f64>() / norms.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn inter_layer_similarity(layer_a: &[Vec<f32>], layer_b: &[Vec<f32>]) -> f64 {
    // Average cosine similarity between same-concept activations across layers
    let mut total_sim = 0.0;
    for (a, b) in layer_a.iter().zip(layer_b.iter()) {
        total_sim += cosine_similarity(a, b);
    }
    total_sim / layer_a.len() as f64
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
fn fisher_discriminant_ratio(class_a: &[Vec<f32>], class_b: &[Vec<f32>]) -> f64 {
    // Fisher's criterion: (μ1 - μ2)² / (σ1² + σ2²)
    // Averaged across dimensions

    let dim = class_a[0].len();
    let mut total_fisher = 0.0;

    for d in 0..dim {
        let a_values: Vec<f64> = class_a.iter().map(|v| v[d] as f64).collect();
        let b_values: Vec<f64> = class_b.iter().map(|v| v[d] as f64).collect();

        let mean_a = a_values.iter().sum::<f64>() / a_values.len() as f64;
        let mean_b = b_values.iter().sum::<f64>() / b_values.len() as f64;

        let var_a =
            a_values.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / a_values.len() as f64;
        let var_b =
            b_values.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / b_values.len() as f64;

        if var_a + var_b > 1e-10 {
            total_fisher += (mean_a - mean_b).powi(2) / (var_a + var_b);
        }
    }

    total_fisher / dim as f64
}

#[cfg(feature = "neural-bridge")]
fn participation_ratio(activations: &[Vec<f32>]) -> f64 {
    // Participation ratio measures effective dimensionality
    // PR = (Σλi)² / Σλi² where λi are eigenvalues of covariance matrix
    // Approximated using variance of each dimension

    let dim = activations[0].len();
    let n = activations.len() as f64;

    let mut variances = Vec::with_capacity(dim);
    for d in 0..dim {
        let values: Vec<f64> = activations.iter().map(|v| v[d] as f64).collect();
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        variances.push(var);
    }

    let sum_var: f64 = variances.iter().sum();
    let sum_var_sq: f64 = variances.iter().map(|v| v.powi(2)).sum();

    if sum_var_sq > 1e-10 {
        sum_var.powi(2) / sum_var_sq
    } else {
        0.0
    }
}
