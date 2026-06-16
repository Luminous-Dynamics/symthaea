// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ (Phenomenal Signature) Extraction and Validation
//!
//! The ultimate test: Can we extract the phenomenal signature Φ directly,
//! and does removing it from phenomenal concepts eliminate their special properties?
//!
//! ## Method
//!
//! 1. Extract L22 activations for all concepts
//! 2. Compute Φ as the component unique to phenomenal concepts:
//!    - Find principal components of phenomenal activations
//!    - Find principal components of functional activations
//!    - Φ = phenomenal direction that is ORTHOGONAL to functional subspace
//! 3. Subtract Φ from phenomenal concepts
//! 4. Re-measure unity and correlations
//!
//! ## Prediction
//!
//! If Φ is real and causal:
//! - Phen_minus_Φ should have LOWER unity (closer to functional)
//! - Phen_minus_Φ should have LOWER pairwise correlation
//! - The phenomenal advantage should disappear
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example phi_extraction_validation --features neural-bridge --release
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
    println!("   Φ EXTRACTION AND VALIDATION");
    println!("   Can we isolate and remove the phenomenal signature?");
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
    // PHASE 1: Extract all activations
    // ================================================================
    println!("================================================================");
    println!("   PHASE 1: EXTRACT ACTIVATIONS");
    println!("================================================================\n");

    let layer = 22;
    let mut phen_acts: Vec<Vec<f64>> = Vec::new();
    let mut func_acts: Vec<Vec<f64>> = Vec::new();

    print!("Extracting L22 activations... ");
    for concept in &phenomenal {
        let acts = extractor.extract_layers(concept, &[layer])?;
        phen_acts.push(acts[0].activation.iter().map(|&x| x as f64).collect());
    }
    for concept in &functional {
        let acts = extractor.extract_layers(concept, &[layer])?;
        func_acts.push(acts[0].activation.iter().map(|&x| x as f64).collect());
    }
    println!("Done\n");

    let dim = phen_acts[0].len();
    println!("Dimensionality: {}\n", dim);

    // ================================================================
    // PHASE 2: Compute Φ (phenomenal signature)
    // ================================================================
    println!("================================================================");
    println!("   PHASE 2: EXTRACT Φ (PHENOMENAL SIGNATURE)");
    println!("================================================================\n");

    // Method: Φ = mean(phenomenal) - projection onto functional subspace

    // Step 1: Compute class centroids
    let phen_centroid = compute_centroid(&phen_acts);
    let func_centroid = compute_centroid(&func_acts);

    println!("Class centroids computed.");
    println!("  Phen centroid L2 norm: {:.4}", l2_norm(&phen_centroid));
    println!("  Func centroid L2 norm: {:.4}", l2_norm(&func_centroid));

    // Step 2: Difference vector (raw)
    let diff_vector: Vec<f64> = phen_centroid
        .iter()
        .zip(func_centroid.iter())
        .map(|(p, f)| p - f)
        .collect();

    println!("  Difference vector L2 norm: {:.4}", l2_norm(&diff_vector));

    // Step 3: Compute functional subspace (top k principal components)
    println!("\nComputing functional subspace (PCA)...");
    let func_pcs = compute_principal_components(&func_acts, 10);
    println!("  Extracted {} principal components", func_pcs.len());

    // Step 4: Project difference onto functional subspace and subtract
    // Φ = diff - projection(diff, func_subspace)
    let diff_proj_onto_func = project_onto_subspace(&diff_vector, &func_pcs);
    let phi: Vec<f64> = diff_vector
        .iter()
        .zip(diff_proj_onto_func.iter())
        .map(|(d, p)| d - p)
        .collect();

    // Normalize Φ
    let phi_norm = l2_norm(&phi);
    let phi_normalized: Vec<f64> = phi.iter().map(|x| x / phi_norm).collect();

    println!("\nΦ extracted:");
    println!("  Raw Φ L2 norm: {:.4}", phi_norm);
    println!("  Φ normalized to unit vector");

    // Step 5: Compute Φ loading for each concept
    let phen_phi_loadings: Vec<f64> = phen_acts
        .iter()
        .map(|a| dot_product(a, &phi_normalized))
        .collect();
    let func_phi_loadings: Vec<f64> = func_acts
        .iter()
        .map(|a| dot_product(a, &phi_normalized))
        .collect();

    println!("\nΦ loadings:");
    println!(
        "  Phenomenal mean: {:.4} ± {:.4}",
        mean(&phen_phi_loadings),
        std_dev(&phen_phi_loadings)
    );
    println!(
        "  Functional mean: {:.4} ± {:.4}",
        mean(&func_phi_loadings),
        std_dev(&func_phi_loadings)
    );

    let phi_d = cohens_d(&phen_phi_loadings, &func_phi_loadings);
    let phi_p = permutation_test(&phen_phi_loadings, &func_phi_loadings, 5000);
    println!(
        "  Cohen's d: {:+.4}, p = {:.4}{}",
        phi_d,
        phi_p,
        if phi_p < 0.001 {
            "***"
        } else if phi_p < 0.01 {
            "**"
        } else if phi_p < 0.05 {
            "*"
        } else {
            ""
        }
    );

    // ================================================================
    // PHASE 3: Subtract Φ from phenomenal concepts
    // ================================================================
    println!("\n================================================================");
    println!("   PHASE 3: SUBTRACT Φ FROM PHENOMENAL CONCEPTS");
    println!("================================================================\n");

    // Subtract Φ component from each phenomenal activation
    // phen_minus_phi[i] = phen[i] - (phen[i] · Φ) * Φ
    let phen_minus_phi: Vec<Vec<f64>> = phen_acts
        .iter()
        .map(|a| {
            let loading = dot_product(a, &phi_normalized);
            a.iter()
                .zip(phi_normalized.iter())
                .map(|(x, phi_i)| x - loading * phi_i)
                .collect()
        })
        .collect();

    println!(
        "Φ subtracted from {} phenomenal concepts",
        phen_minus_phi.len()
    );

    // Verify: Φ loadings should now be ~0 for modified phenomenal
    let phen_minus_phi_loadings: Vec<f64> = phen_minus_phi
        .iter()
        .map(|a| dot_product(a, &phi_normalized))
        .collect();
    println!("Verification - Φ loadings after subtraction:");
    println!(
        "  Mean: {:.6} (should be ~0)",
        mean(&phen_minus_phi_loadings)
    );

    // ================================================================
    // PHASE 4: Re-measure unity scores
    // ================================================================
    println!("\n================================================================");
    println!("   PHASE 4: COMPARE UNITY SCORES");
    println!("================================================================\n");

    // Compute unity for original phenomenal, modified phenomenal, and functional
    print!("Computing unity scores... ");

    let phen_orig_unity: Vec<f64> = phen_acts
        .iter()
        .map(|a| compute_unity_from_activation(a, &topology_config))
        .collect();

    let phen_minus_phi_unity: Vec<f64> = phen_minus_phi
        .iter()
        .map(|a| compute_unity_from_activation(a, &topology_config))
        .collect();

    let func_unity: Vec<f64> = func_acts
        .iter()
        .map(|a| compute_unity_from_activation(a, &topology_config))
        .collect();

    println!("Done\n");

    println!("Unity Scores:");
    println!("────────────────────────────────────────────────────────────");
    println!(
        "  Phenomenal (original):  {:.4} ± {:.4}",
        mean(&phen_orig_unity),
        std_dev(&phen_orig_unity)
    );
    println!(
        "  Phenomenal (minus Φ):   {:.4} ± {:.4}",
        mean(&phen_minus_phi_unity),
        std_dev(&phen_minus_phi_unity)
    );
    println!(
        "  Functional:             {:.4} ± {:.4}",
        mean(&func_unity),
        std_dev(&func_unity)
    );

    let orig_diff = mean(&phen_orig_unity) - mean(&func_unity);
    let new_diff = mean(&phen_minus_phi_unity) - mean(&func_unity);
    let reduction = if orig_diff != 0.0 {
        ((orig_diff - new_diff) / orig_diff) * 100.0
    } else {
        0.0
    };

    println!("\nPhenomenal Advantage:");
    println!("  Original: {:+.4}", orig_diff);
    println!("  After Φ removal: {:+.4}", new_diff);
    println!("  Reduction: {:.1}%", reduction);

    // Statistical tests
    let p_orig = permutation_test(&phen_orig_unity, &func_unity, 5000);
    let p_new = permutation_test(&phen_minus_phi_unity, &func_unity, 5000);

    println!("\nStatistical Significance:");
    println!(
        "  Original vs Functional: p = {:.4}{}",
        p_orig,
        if p_orig < 0.05 { " *" } else { "" }
    );
    println!(
        "  Minus-Φ vs Functional:  p = {:.4}{}",
        p_new,
        if p_new < 0.05 { " *" } else { "" }
    );

    // ================================================================
    // PHASE 5: Re-measure pairwise correlations
    // ================================================================
    println!("\n================================================================");
    println!("   PHASE 5: COMPARE PAIRWISE CORRELATIONS");
    println!("================================================================\n");

    let phen_orig_corr = compute_pairwise_correlations(&phen_acts);
    let phen_minus_phi_corr = compute_pairwise_correlations(&phen_minus_phi);
    let func_corr = compute_pairwise_correlations(&func_acts);

    println!("Pairwise Correlations:");
    println!("────────────────────────────────────────────────────────────");
    println!(
        "  Phenomenal (original):  {:.4} ± {:.4}",
        mean(&phen_orig_corr),
        std_dev(&phen_orig_corr)
    );
    println!(
        "  Phenomenal (minus Φ):   {:.4} ± {:.4}",
        mean(&phen_minus_phi_corr),
        std_dev(&phen_minus_phi_corr)
    );
    println!(
        "  Functional:             {:.4} ± {:.4}",
        mean(&func_corr),
        std_dev(&func_corr)
    );

    let corr_reduction = mean(&phen_orig_corr) - mean(&phen_minus_phi_corr);
    let target_reduction = mean(&phen_orig_corr) - mean(&func_corr);
    let corr_pct = if target_reduction != 0.0 {
        (corr_reduction / target_reduction) * 100.0
    } else {
        0.0
    };

    println!("\nCorrelation Change:");
    println!("  Original phen-phen: {:.4}", mean(&phen_orig_corr));
    println!("  Target (func-func): {:.4}", mean(&func_corr));
    println!("  After Φ removal:    {:.4}", mean(&phen_minus_phi_corr));
    println!("  Progress toward target: {:.1}%", corr_pct);

    // ================================================================
    // PHASE 6: Interpretation
    // ================================================================
    println!("\n================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    let unity_success = reduction > 30.0;
    let corr_success = corr_pct > 30.0;
    let sig_eliminated = p_new > 0.05;

    if unity_success && corr_success {
        println!("★ Φ EXTRACTION VALIDATED");
        println!("");
        println!("  Removing Φ from phenomenal concepts:");
        println!("    - Reduced unity advantage by {:.1}%", reduction);
        println!(
            "    - Reduced correlation by {:.1}% toward functional baseline",
            corr_pct
        );
        if sig_eliminated {
            println!(
                "    - ELIMINATED statistical significance (p = {:.4})",
                p_new
            );
            println!("");
            println!("  This is CAUSAL PROOF that Φ is the phenomenal signature.");
        } else {
            println!(
                "    - Significance reduced but not eliminated (p = {:.4})",
                p_new
            );
            println!("");
            println!("  Φ captures MOST but not all phenomenal structure.");
        }
    } else if unity_success || corr_success {
        println!("◐ Φ PARTIALLY VALIDATED");
        println!("");
        println!("  Φ removal had partial effect:");
        if unity_success {
            println!("    ✓ Unity reduced by {:.1}%", reduction);
        } else {
            println!("    ✗ Unity only reduced by {:.1}%", reduction);
        }
        if corr_success {
            println!("    ✓ Correlation reduced by {:.1}%", corr_pct);
        } else {
            println!("    ✗ Correlation only reduced by {:.1}%", corr_pct);
        }
        println!("");
        println!("  Φ captures some but not all phenomenal structure.");
        println!("  May need more sophisticated extraction method.");
    } else {
        println!("✗ Φ EXTRACTION INSUFFICIENT");
        println!("");
        println!("  Removing extracted Φ did not substantially affect:");
        println!("    - Unity (reduced {:.1}%)", reduction);
        println!("    - Correlation (reduced {:.1}%)", corr_pct);
        println!("");
        println!("  Possible explanations:");
        println!("    1. Φ is distributed across many dimensions (not low-rank)");
        println!("    2. Simple centroid difference doesn't capture Φ");
        println!("    3. Need nonlinear extraction method");
    }

    // ================================================================
    // PHASE 7: Visualize Φ
    // ================================================================
    println!("\n================================================================");
    println!("   Φ CHARACTERISTICS");
    println!("================================================================\n");

    // Find dimensions with highest Φ loading
    let mut phi_dims: Vec<(usize, f64)> = phi_normalized
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    phi_dims.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Top 10 dimensions in Φ:");
    println!("Dim   │ Φ weight │ Direction");
    println!("──────┼──────────┼──────────");
    for (dim, weight) in phi_dims.iter().take(10) {
        let dir = if *weight > 0.0 { "+" } else { "-" };
        println!("{:5} │ {:+8.4} │ {}", dim, weight, dir);
    }

    // Sparsity of Φ
    let phi_sparsity =
        phi_normalized.iter().filter(|&&x| x.abs() < 0.01).count() as f64 / dim as f64;
    println!("\nΦ sparsity (|w| < 0.01): {:.1}%", phi_sparsity * 100.0);

    let phi_concentrated = phi_dims
        .iter()
        .take(50)
        .map(|(_, w)| w.powi(2))
        .sum::<f64>();
    println!("Variance in top 50 dims: {:.1}%", phi_concentrated * 100.0);

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn compute_centroid(activations: &[Vec<f64>]) -> Vec<f64> {
    let dim = activations[0].len();
    let n = activations.len() as f64;

    (0..dim)
        .map(|d| activations.iter().map(|a| a[d]).sum::<f64>() / n)
        .collect()
}

#[cfg(feature = "neural-bridge")]
fn compute_principal_components(activations: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    // Simplified PCA using power iteration
    let dim = activations[0].len();
    let centroid = compute_centroid(activations);

    // Center the data
    let centered: Vec<Vec<f64>> = activations
        .iter()
        .map(|a| a.iter().zip(centroid.iter()).map(|(x, c)| x - c).collect())
        .collect();

    let mut pcs: Vec<Vec<f64>> = Vec::new();
    let mut residual = centered.clone();

    for _ in 0..k {
        // Power iteration to find top eigenvector
        let mut v: Vec<f64> = (0..dim).map(|i| ((i * 7 + 13) as f64).sin()).collect();
        let v_norm = l2_norm(&v);
        v = v.iter().map(|x| x / v_norm).collect();

        for _ in 0..20 {
            // 20 iterations usually enough
            // v = A^T A v (where A is the data matrix)
            let mut new_v = vec![0.0; dim];
            for row in &residual {
                let dot = dot_product(row, &v);
                for (j, &r) in row.iter().enumerate() {
                    new_v[j] += dot * r;
                }
            }

            let norm = l2_norm(&new_v);
            if norm > 1e-10 {
                v = new_v.iter().map(|x| x / norm).collect();
            }
        }

        // Deflate: remove this component from residual
        for row in residual.iter_mut() {
            let proj = dot_product(row, &v);
            for (j, r) in row.iter_mut().enumerate() {
                *r -= proj * v[j];
            }
        }

        pcs.push(v);
    }

    pcs
}

#[cfg(feature = "neural-bridge")]
fn project_onto_subspace(v: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut projection = vec![0.0; v.len()];

    for b in basis {
        let coef = dot_product(v, b);
        for (i, &bi) in b.iter().enumerate() {
            projection[i] += coef * bi;
        }
    }

    projection
}

#[cfg(feature = "neural-bridge")]
fn compute_unity_from_activation(activation: &[f64], config: &TopologyConfig) -> f64 {
    let act_f32: Vec<f32> = activation.iter().map(|&x| x as f32).collect();
    let hv = activation_to_hv16(&act_f32);

    let mut topology = ConsciousnessTopology::new(config.clone());
    topology.add_state(hv);
    for shift in 1..5 {
        topology.add_state(hv.permute(shift * 100));
    }

    topology.analyze(0.5).unity_score
}

#[cfg(feature = "neural-bridge")]
fn compute_pairwise_correlations(activations: &[Vec<f64>]) -> Vec<f64> {
    let mut correlations = Vec::new();
    let n = activations.len();

    for i in 0..n {
        for j in (i + 1)..n {
            correlations.push(pearson_correlation(&activations[i], &activations[j]));
        }
    }

    correlations
}

#[cfg(feature = "neural-bridge")]
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = mean(a);
    let mean_b = mean(b);

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
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
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(feature = "neural-bridge")]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
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
