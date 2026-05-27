// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binding Compositionality Experiment
//!
//! Tests whether HDC binding better captures compositional semantics than bundling.
//!
//! Hypothesis: For unified pairs like (red, apple) → "red apple":
//!   similarity(bind(red, apple), "red apple") > similarity(bundle(red, apple), "red apple")
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example binding_compositionality --features neural-bridge --release
//! ```

use std::time::Instant;

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::bge_m3::BgeM3;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{HDC_DIMENSION, binary_hv::BinaryHV};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!("Run with: cargo run --example binding_compositionality --features neural-bridge");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   BINDING COMPOSITIONALITY EXPERIMENT");
    println!("   Does binding capture compositional semantics?");
    println!("================================================================\n");

    // Load concept pairs from JSON
    let pairs = load_concept_pairs("data/binding_study/concept_pairs.json")?;

    let unified: Vec<_> = pairs.iter().filter(|p| p.pair_type == "unified").collect();
    let separate: Vec<_> = pairs.iter().filter(|p| p.pair_type == "separate").collect();

    println!("Loaded concept pairs:");
    println!(
        "  Unified pairs: {} (e.g., red+apple→'red apple')",
        unified.len()
    );
    println!(
        "  Separate pairs: {} (e.g., red+mailbox→'red mailbox')\n",
        separate.len()
    );

    // Load BGE-M3 encoder
    println!("Loading BGE-M3 encoder...");
    let load_start = Instant::now();
    let encoder = BgeM3::load_from_hub("BAAI/bge-m3")?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    println!("================================================================");
    println!("   ENCODING AND ANALYZING PAIRS");
    println!("================================================================\n");

    // Process unified pairs
    println!("Processing unified pairs...");
    let unified_start = Instant::now();
    let mut unified_results = Vec::new();

    for (i, pair) in unified.iter().enumerate() {
        if i % 10 == 0 {
            print!("  {}/{}\r", i, unified.len());
        }

        let result = analyze_pair(&encoder, pair)?;
        unified_results.push(result);
    }
    println!(
        "  Completed in {:.1}s",
        unified_start.elapsed().as_secs_f64()
    );

    // Process separate pairs
    println!("Processing separate pairs...");
    let separate_start = Instant::now();
    let mut separate_results = Vec::new();

    for (i, pair) in separate.iter().enumerate() {
        if i % 10 == 0 {
            print!("  {}/{}\r", i, separate.len());
        }

        let result = analyze_pair(&encoder, pair)?;
        separate_results.push(result);
    }
    println!(
        "  Completed in {:.1}s\n",
        separate_start.elapsed().as_secs_f64()
    );

    // Compute statistics
    println!("================================================================");
    println!("   RESULTS: Similarity to Combined Phrase");
    println!("================================================================\n");

    // Extract similarity metrics
    let unified_bind_sim: Vec<f64> = unified_results.iter().map(|r| r.bind_similarity).collect();
    let unified_bundle_sim: Vec<f64> = unified_results
        .iter()
        .map(|r| r.bundle_similarity)
        .collect();
    let separate_bind_sim: Vec<f64> = separate_results.iter().map(|r| r.bind_similarity).collect();
    let separate_bundle_sim: Vec<f64> = separate_results
        .iter()
        .map(|r| r.bundle_similarity)
        .collect();

    let mean_unified_bind = mean(&unified_bind_sim);
    let mean_unified_bundle = mean(&unified_bundle_sim);
    let mean_separate_bind = mean(&separate_bind_sim);
    let mean_separate_bundle = mean(&separate_bundle_sim);

    println!("Mean Similarity to Combined Phrase:");
    println!(
        "                    Unified (n={})    Separate (n={})",
        unified.len(),
        separate.len()
    );
    println!(
        "  Bind:             {:.4}              {:.4}",
        mean_unified_bind, mean_separate_bind
    );
    println!(
        "  Bundle:           {:.4}              {:.4}\n",
        mean_unified_bundle, mean_separate_bundle
    );

    // Binding advantage (bind - bundle similarity)
    let unified_advantage: Vec<f64> = unified_results
        .iter()
        .map(|r| r.bind_similarity - r.bundle_similarity)
        .collect();
    let separate_advantage: Vec<f64> = separate_results
        .iter()
        .map(|r| r.bind_similarity - r.bundle_similarity)
        .collect();

    let mean_unified_adv = mean(&unified_advantage);
    let mean_separate_adv = mean(&separate_advantage);

    println!("Binding Advantage (bind_sim - bundle_sim):");
    println!(
        "  Unified pairs: {:+.4} (std: {:.4})",
        mean_unified_adv,
        std_dev(&unified_advantage)
    );
    println!(
        "  Separate pairs: {:+.4} (std: {:.4})\n",
        mean_separate_adv,
        std_dev(&separate_advantage)
    );

    // Interaction effect
    let interaction = mean_unified_adv - mean_separate_adv;
    println!("Interaction Effect:");
    println!("  (Bind advantage for unified) - (Bind advantage for separate)");
    println!(
        "  = {:+.4} - {:+.4} = {:+.4}\n",
        mean_unified_adv, mean_separate_adv, interaction
    );

    // Statistical tests
    let p_unified = permutation_test(&unified_bind_sim, &unified_bundle_sim, 10000);
    let p_separate = permutation_test(&separate_bind_sim, &separate_bundle_sim, 10000);
    let p_interaction = permutation_test(&unified_advantage, &separate_advantage, 10000);

    println!("Statistical Significance (Permutation Tests, n=10000):");
    println!(
        "  Binding advantage (unified): p = {:.4} {}",
        p_unified,
        if p_unified < 0.05 { "*" } else { "" }
    );
    println!(
        "  Binding advantage (separate): p = {:.4} {}",
        p_separate,
        if p_separate < 0.05 { "*" } else { "" }
    );
    println!(
        "  Interaction effect: p = {:.4} {}\n",
        p_interaction,
        if p_interaction < 0.05 { "*" } else { "" }
    );

    // Effect sizes
    let d_unified = cohens_d(&unified_bind_sim, &unified_bundle_sim);
    let d_separate = cohens_d(&separate_bind_sim, &separate_bundle_sim);
    let d_interaction = cohens_d(&unified_advantage, &separate_advantage);

    println!("Effect Sizes (Cohen's d):");
    println!("  Binding advantage (unified): {:+.4}", d_unified);
    println!("  Binding advantage (separate): {:+.4}", d_separate);
    println!("  Interaction: {:+.4}\n", d_interaction);

    // Interpretation
    println!("================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if mean_unified_adv > 0.0 && p_unified < 0.05 {
        println!("✓ BINDING CAPTURES COMPOSITIONALITY FOR UNIFIED PAIRS");
        println!("  bind(a,b) is more similar to 'a b' than bundle(a,b)");
        println!("  Effect size: d = {:+.3}", d_unified);
    } else if mean_unified_adv < 0.0 && p_unified < 0.05 {
        println!("✗ BUNDLING BETTER CAPTURES UNIFIED SEMANTICS");
        println!("  bundle(a,b) is more similar to 'a b' than bind(a,b)");
    } else {
        println!("○ NO SIGNIFICANT BINDING ADVANTAGE FOR UNIFIED PAIRS");
    }

    if interaction > 0.0 && p_interaction < 0.05 {
        println!("\n✓ BINDING SPECIFICALLY HELPS UNIFIED PAIRS");
        println!("  The binding advantage is greater for unified than separate pairs");
    } else if interaction < 0.0 && p_interaction < 0.05 {
        println!("\n✗ BINDING SPECIFICALLY HELPS SEPARATE PAIRS (unexpected)");
    }

    // Sample results
    println!("\n================================================================");
    println!("   SAMPLE RESULTS");
    println!("================================================================\n");

    // Sort by binding advantage
    let mut all_results: Vec<_> = unified_results
        .iter()
        .chain(separate_results.iter())
        .collect();
    all_results.sort_by(|a, b| {
        let adv_a = a.bind_similarity - a.bundle_similarity;
        let adv_b = b.bind_similarity - b.bundle_similarity;
        adv_b.partial_cmp(&adv_a).unwrap()
    });

    println!("Top 5 pairs where BINDING better captures meaning:");
    for r in all_results.iter().take(5) {
        let adv = r.bind_similarity - r.bundle_similarity;
        println!(
            "  '{}' ({}): bind={:.3}, bundle={:.3}, adv={:+.3}",
            r.combined, r.pair_type, r.bind_similarity, r.bundle_similarity, adv
        );
    }

    println!("\nTop 5 pairs where BUNDLING better captures meaning:");
    for r in all_results.iter().rev().take(5) {
        let adv = r.bind_similarity - r.bundle_similarity;
        println!(
            "  '{}' ({}): bind={:.3}, bundle={:.3}, adv={:+.3}",
            r.combined, r.pair_type, r.bind_similarity, r.bundle_similarity, adv
        );
    }

    // Also show raw embedding similarity between parts
    println!("\n================================================================");
    println!("   ADDITIONAL: Part Similarity Analysis");
    println!("================================================================\n");

    let unified_part_sim: Vec<f64> = unified_results.iter().map(|r| r.parts_similarity).collect();
    let separate_part_sim: Vec<f64> = separate_results
        .iter()
        .map(|r| r.parts_similarity)
        .collect();

    println!("Mean similarity between concept parts (embed(a) · embed(b)):");
    println!("  Unified pairs: {:.4}", mean(&unified_part_sim));
    println!("  Separate pairs: {:.4}", mean(&separate_part_sim));

    let p_parts = permutation_test(&unified_part_sim, &separate_part_sim, 10000);
    println!(
        "  p-value: {:.4} {}",
        p_parts,
        if p_parts < 0.05 { "*" } else { "" }
    );

    if mean(&unified_part_sim) > mean(&separate_part_sim) && p_parts < 0.05 {
        println!("\n  → Unified pairs have more semantically related parts (as expected)");
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
#[derive(Debug)]
struct ConceptPair {
    id: String,
    concept_a: String,
    concept_b: String,
    combined: String,
    pair_type: String,
}

#[cfg(feature = "neural-bridge")]
#[derive(Debug)]
struct CompositionResult {
    pair_id: String,
    pair_type: String,
    combined: String,
    bind_similarity: f64,
    bundle_similarity: f64,
    parts_similarity: f64,
}

#[cfg(feature = "neural-bridge")]
fn load_concept_pairs(path: &str) -> Result<Vec<ConceptPair>> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let mut pairs = Vec::new();

    let pair_keys = [("unified_pairs", "unified"), ("separate_pairs", "separate")];

    for (json_key, pair_type) in pair_keys {
        if let Some(items) = json[json_key].as_array() {
            for item in items {
                let concept_a = item["concept_a"].as_str().unwrap_or("").to_string();
                let concept_b = item["concept_b"].as_str().unwrap_or("").to_string();
                let combined = item["combined"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{} {}", concept_a, concept_b));

                pairs.push(ConceptPair {
                    id: item["id"].as_str().unwrap_or("unknown").to_string(),
                    concept_a,
                    concept_b,
                    combined,
                    pair_type: pair_type.to_string(),
                });
            }
        }
    }

    Ok(pairs)
}

#[cfg(feature = "neural-bridge")]
fn analyze_pair(encoder: &BgeM3, pair: &ConceptPair) -> Result<CompositionResult> {
    // Encode individual concepts
    let emb_a = encoder.encode(&pair.concept_a)?;
    let emb_b = encoder.encode(&pair.concept_b)?;
    let emb_combined = encoder.encode(&pair.combined)?;

    // Convert to HDC
    let hv_a = embedding_to_hv16(&emb_a);
    let hv_b = embedding_to_hv16(&emb_b);
    let hv_combined = embedding_to_hv16(&emb_combined);

    // Compose via binding and bundling
    let bound = hv_a.bind(&hv_b);
    let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

    // Compute similarities to combined phrase
    let bind_similarity = hv_similarity(&bound, &hv_combined);
    let bundle_similarity = hv_similarity(&bundled, &hv_combined);

    // Also compute similarity between parts (in embedding space)
    let parts_similarity = cosine_similarity(&emb_a, &emb_b);

    Ok(CompositionResult {
        pair_id: pair.id.clone(),
        pair_type: pair.pair_type.clone(),
        combined: pair.combined.clone(),
        bind_similarity,
        bundle_similarity,
        parts_similarity,
    })
}

#[cfg(feature = "neural-bridge")]
fn embedding_to_hv16(embedding: &[f32]) -> BinaryHV {
    let mut expanded = Vec::with_capacity(HDC_DIMENSION);
    let tiles = HDC_DIMENSION / embedding.len();

    for tile in 0..tiles {
        for (i, &val) in embedding.iter().enumerate() {
            let perturbation = ((tile * embedding.len() + i) as f32 * 0.001).sin() * 0.01;
            expanded.push(val + perturbation);
        }
    }

    BinaryHV::from_bipolar(&expanded)
}

#[cfg(feature = "neural-bridge")]
fn hv_similarity(a: &BinaryHV, b: &BinaryHV) -> f64 {
    // Hamming similarity: 1 - (hamming_distance / dimension)
    let hamming = a.hamming_distance(b);
    1.0 - (hamming as f64 / HDC_DIMENSION as f64)
}

#[cfg(feature = "neural-bridge")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
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
fn cohens_d(group_a: &[f64], group_b: &[f64]) -> f64 {
    let mean_diff = mean(group_a) - mean(group_b);
    let pooled_std = ((std_dev(group_a).powi(2) + std_dev(group_b).powi(2)) / 2.0).sqrt();
    if pooled_std > 0.0 {
        mean_diff / pooled_std
    } else {
        0.0
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
