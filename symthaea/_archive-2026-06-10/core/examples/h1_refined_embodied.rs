// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H1 Refined: Sensory/Embodied vs Abstract/Procedural
//!
//! Tests the refined hypothesis that the topological distinction is between:
//! - Sensory/embodied language (qualia, self_awareness, aesthetic)
//! - Abstract/procedural language (computation, algorithms, engineering)
//!
//! Excludes confounding categories:
//! - consciousness_unity (phenomenal but low unity - abstract)
//! - machine_learning (functional but high unity - bridges both domains)

#[cfg(feature = "neural-bridge")]
use std::fs::File;
#[cfg(feature = "neural-bridge")]
use std::io::BufReader;
#[cfg(feature = "neural-bridge")]
use std::path::Path;
#[cfg(feature = "neural-bridge")]
use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "neural-bridge")]
use serde::Deserialize;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::ConsciousnessProbeV2;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::{ConsciousnessTopology, TopologyConfig};

#[cfg(feature = "neural-bridge")]
#[derive(Deserialize)]
struct ConceptCorpus {
    concepts: Vec<Concept>,
}

#[cfg(feature = "neural-bridge")]
#[derive(Deserialize, Clone)]
struct Concept {
    id: String,
    text: String,
    category: String,
    subcategory: String,
}

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_refined_analysis()
}

#[cfg(feature = "neural-bridge")]
fn run_refined_analysis() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   H1 REFINED: SENSORY/EMBODIED vs ABSTRACT/PROCEDURAL");
    println!("   Testing the refined hypothesis");
    println!("================================================================\n");

    // Load corpora
    let phenomenal_corpus: ConceptCorpus = serde_json::from_reader(BufReader::new(File::open(
        "data/consciousness_probe/phenomenal_concepts_expanded.json",
    )?))?;
    let functional_corpus: ConceptCorpus = serde_json::from_reader(BufReader::new(File::open(
        "data/consciousness_probe/functional_concepts_expanded.json",
    )?))?;

    // Filter to sensory/embodied categories (high-unity phenomenal)
    let sensory_categories = vec!["qualia", "self_awareness", "aesthetic", "emotion"];
    let sensory_concepts: Vec<_> = phenomenal_corpus
        .concepts
        .iter()
        .filter(|c| sensory_categories.contains(&c.category.as_str()))
        .collect();

    // Filter to abstract/procedural categories (low-unity functional)
    let procedural_categories = vec!["computation", "engineering", "systems"];
    let procedural_concepts: Vec<_> = functional_corpus
        .concepts
        .iter()
        .filter(|c| procedural_categories.contains(&c.category.as_str()))
        .collect();

    println!(
        "Sensory/embodied concepts: {} (categories: {:?})",
        sensory_concepts.len(),
        sensory_categories
    );
    println!(
        "Abstract/procedural concepts: {} (categories: {:?})\n",
        procedural_concepts.len(),
        procedural_categories
    );

    // Load probe
    let probe_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    println!("Loading BGE-M3 model...");
    let load_start = Instant::now();
    let mut probe = ConsciousnessProbeV2::load_with_probe(probe_path)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    let topo_config = TopologyConfig {
        min_persistence: 0.05,
        max_scale: 1.0,
        num_scales: 20,
        detect_cycles: true,
        detect_voids: true,
    };

    let analyze_concept = |probe: &mut ConsciousnessProbeV2, text: &str| -> Result<f64> {
        let hv = probe.concept_to_hv(text)?;
        let mut topology = ConsciousnessTopology::new(topo_config.clone());
        topology.add_state(hv);
        for shift in 1..5 {
            let permuted = hv.permute(shift * 100);
            topology.add_state(permuted);
        }
        Ok(topology.analyze(0.5).unity_score)
    };

    // Analyze sensory concepts
    println!("Analyzing sensory/embodied concepts...");
    let start = Instant::now();
    let mut sensory_scores = Vec::new();
    for concept in &sensory_concepts {
        if let Ok(score) = analyze_concept(&mut probe, &concept.text) {
            sensory_scores.push(score);
        }
    }
    println!(
        "  Completed {} concepts in {:.2}s\n",
        sensory_scores.len(),
        start.elapsed().as_secs_f64()
    );

    // Analyze procedural concepts
    println!("Analyzing abstract/procedural concepts...");
    let start = Instant::now();
    let mut procedural_scores = Vec::new();
    for concept in &procedural_concepts {
        if let Ok(score) = analyze_concept(&mut probe, &concept.text) {
            procedural_scores.push(score);
        }
    }
    println!(
        "  Completed {} concepts in {:.2}s\n",
        procedural_scores.len(),
        start.elapsed().as_secs_f64()
    );

    // Statistics
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };

    let sensory_mean = mean(&sensory_scores);
    let sensory_std = std(&sensory_scores);
    let procedural_mean = mean(&procedural_scores);
    let procedural_std = std(&procedural_scores);

    println!("================================================================");
    println!("   RESULTS");
    println!("================================================================\n");

    println!("Sensory/Embodied (n={}):", sensory_scores.len());
    println!("  Mean: {:.4}", sensory_mean);
    println!("  SD:   {:.4}", sensory_std);

    println!("\nAbstract/Procedural (n={}):", procedural_scores.len());
    println!("  Mean: {:.4}", procedural_mean);
    println!("  SD:   {:.4}", procedural_std);

    let diff = sensory_mean - procedural_mean;
    let pooled_std = ((sensory_std.powi(2) + procedural_std.powi(2)) / 2.0).sqrt();
    let cohens_d = if pooled_std > 0.0 {
        diff / pooled_std
    } else {
        0.0
    };

    println!("\nEffect:");
    println!("  Difference: {:.4}", diff);
    println!("  Cohen's d:  {:.4}", cohens_d);

    // Permutation test
    println!("\nRunning permutation test (n=10000)...");
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let all_scores: Vec<f64> = sensory_scores
        .iter()
        .chain(procedural_scores.iter())
        .copied()
        .collect();
    let n_sensory = sensory_scores.len();
    let n_perms = 10000;
    let mut extreme_count = 0;

    for _ in 0..n_perms {
        let mut indices: Vec<usize> = (0..all_scores.len()).collect();
        indices.shuffle(&mut rng);
        let perm_sensory: Vec<f64> = indices[..n_sensory]
            .iter()
            .map(|&i| all_scores[i])
            .collect();
        let perm_proc: Vec<f64> = indices[n_sensory..]
            .iter()
            .map(|&i| all_scores[i])
            .collect();
        let perm_diff = mean(&perm_sensory) - mean(&perm_proc);
        if perm_diff.abs() >= diff.abs() {
            extreme_count += 1;
        }
    }

    let p_value = extreme_count as f64 / n_perms as f64;
    println!("  p-value: {:.4}", p_value);
    println!("  Significant (p < 0.05): {}", p_value < 0.05);
    println!("  Significant (p < 0.01): {}", p_value < 0.01);

    // Bootstrap CI
    println!("\nBootstrap 95% CI (n=10000)...");
    let mut bootstrap_diffs = Vec::with_capacity(10000);
    for _ in 0..10000 {
        let boot_sensory: Vec<f64> = (0..sensory_scores.len())
            .map(|_| *sensory_scores.choose(&mut rng).unwrap())
            .collect();
        let boot_proc: Vec<f64> = (0..procedural_scores.len())
            .map(|_| *procedural_scores.choose(&mut rng).unwrap())
            .collect();
        bootstrap_diffs.push(mean(&boot_sensory) - mean(&boot_proc));
    }
    bootstrap_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lower = bootstrap_diffs[250];
    let ci_upper = bootstrap_diffs[9749];
    println!("  95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
    println!("  CI excludes zero: {}", ci_lower > 0.0 || ci_upper < 0.0);

    let effect_size = if cohens_d.abs() < 0.2 {
        "negligible"
    } else if cohens_d.abs() < 0.5 {
        "small"
    } else if cohens_d.abs() < 0.8 {
        "medium"
    } else {
        "large"
    };

    println!("\n================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    println!("H1 REFINED (Sensory vs Procedural):");
    println!("  Sensory mean:    {:.4}", sensory_mean);
    println!("  Procedural mean: {:.4}", procedural_mean);
    println!("  Difference:      {:.4}", diff);
    println!("  Cohen's d:       {:.4} ({})", cohens_d, effect_size);
    println!("  p-value:         {:.4}", p_value);
    println!("  95% CI:          [{:.4}, {:.4}]", ci_lower, ci_upper);

    if p_value < 0.05 && diff > 0.0 {
        println!("\nRESULT: H1-REFINED SUPPORTED");
        println!("Sensory/embodied language shows higher topological unity");
        println!("than abstract/procedural language.");
    } else {
        println!("\nRESULT: NOT SIGNIFICANT");
    }

    println!("\n================================================================");
    println!("   H1 REFINED ANALYSIS COMPLETE");
    println!("================================================================\n");

    Ok(())
}