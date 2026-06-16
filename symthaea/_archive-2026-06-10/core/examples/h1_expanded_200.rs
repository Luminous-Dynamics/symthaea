// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H1 Expanded: 200 Concept Analysis
//!
//! Tests the phenomenal-computational distinction with the full expanded corpora:
//! - 100 phenomenal concepts (qualia, self_awareness, consciousness_unity, emotion, philosophical, altered_states, aesthetic)
//! - 100 functional concepts (computation, mathematics, systems, science, engineering, AI/ML, finance, linguistics)
//!
//! This provides maximum statistical power and subcategory analysis.

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
    metadata: Metadata,
    concepts: Vec<Concept>,
}

#[cfg(feature = "neural-bridge")]
#[derive(Deserialize)]
struct Metadata {
    description: String,
    count: usize,
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
        println!(
            "Run with: cargo run --example h1_expanded_200 --features neural-bridge --release"
        );
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_expanded_analysis()
}

#[cfg(feature = "neural-bridge")]
fn run_expanded_analysis() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   H1 EXPANDED: 200 CONCEPT ANALYSIS");
    println!("   Maximum statistical power with full corpora");
    println!("================================================================\n");

    // Load expanded corpora
    let phenomenal_path = Path::new("data/consciousness_probe/phenomenal_concepts_expanded.json");
    let functional_path = Path::new("data/consciousness_probe/functional_concepts_expanded.json");

    if !phenomenal_path.exists() || !functional_path.exists() {
        println!("ERROR: Expanded concept corpora not found");
        return Ok(());
    }

    let phenomenal_corpus: ConceptCorpus =
        serde_json::from_reader(BufReader::new(File::open(phenomenal_path)?))?;
    let functional_corpus: ConceptCorpus =
        serde_json::from_reader(BufReader::new(File::open(functional_path)?))?;

    println!(
        "Loaded {} phenomenal concepts",
        phenomenal_corpus.concepts.len()
    );
    println!(
        "Loaded {} functional concepts\n",
        functional_corpus.concepts.len()
    );

    // Load probe
    let probe_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !probe_path.exists() {
        println!("ERROR: Probe weights not found");
        return Ok(());
    }

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
        let assessment = topology.analyze(0.5);
        Ok(assessment.unity_score)
    };

    // Analyze phenomenal concepts
    println!("================================================================");
    println!("   ANALYZING PHENOMENAL CONCEPTS");
    println!("================================================================\n");

    let start = Instant::now();
    let mut phenomenal_results: Vec<(Concept, f64)> = Vec::new();
    for (i, concept) in phenomenal_corpus.concepts.iter().enumerate() {
        match analyze_concept(&mut probe, &concept.text) {
            Ok(score) => phenomenal_results.push((concept.clone(), score)),
            Err(e) => println!("  Error on concept {}: {}", i, e),
        }
        if (i + 1) % 20 == 0 {
            println!(
                "  Processed {}/{} phenomenal concepts...",
                i + 1,
                phenomenal_corpus.concepts.len()
            );
        }
    }
    println!("  Completed in {:.2}s\n", start.elapsed().as_secs_f64());

    // Analyze functional concepts
    println!("================================================================");
    println!("   ANALYZING FUNCTIONAL CONCEPTS");
    println!("================================================================\n");

    let start = Instant::now();
    let mut functional_results: Vec<(Concept, f64)> = Vec::new();
    for (i, concept) in functional_corpus.concepts.iter().enumerate() {
        match analyze_concept(&mut probe, &concept.text) {
            Ok(score) => functional_results.push((concept.clone(), score)),
            Err(e) => println!("  Error on concept {}: {}", i, e),
        }
        if (i + 1) % 20 == 0 {
            println!(
                "  Processed {}/{} functional concepts...",
                i + 1,
                functional_corpus.concepts.len()
            );
        }
    }
    println!("  Completed in {:.2}s\n", start.elapsed().as_secs_f64());

    // Statistics helpers
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };

    let phenomenal_scores: Vec<f64> = phenomenal_results.iter().map(|(_, s)| *s).collect();
    let functional_scores: Vec<f64> = functional_results.iter().map(|(_, s)| *s).collect();

    // Overall results
    println!("================================================================");
    println!(
        "   OVERALL RESULTS (N={})",
        phenomenal_scores.len() + functional_scores.len()
    );
    println!("================================================================\n");

    let phen_mean = mean(&phenomenal_scores);
    let phen_std = std(&phenomenal_scores);
    let func_mean = mean(&functional_scores);
    let func_std = std(&functional_scores);

    println!("Phenomenal (n={}):", phenomenal_scores.len());
    println!("  Mean: {:.4}", phen_mean);
    println!("  SD:   {:.4}", phen_std);

    println!("\nFunctional (n={}):", functional_scores.len());
    println!("  Mean: {:.4}", func_mean);
    println!("  SD:   {:.4}", func_std);

    let diff = phen_mean - func_mean;
    let pooled_std = ((phen_std.powi(2) + func_std.powi(2)) / 2.0).sqrt();
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
    let all_scores: Vec<f64> = phenomenal_scores
        .iter()
        .chain(functional_scores.iter())
        .copied()
        .collect();
    let n_phen = phenomenal_scores.len();
    let n_perms = 10000;
    let mut extreme_count = 0;

    for _ in 0..n_perms {
        let mut indices: Vec<usize> = (0..all_scores.len()).collect();
        indices.shuffle(&mut rng);
        let perm_phen: Vec<f64> = indices[..n_phen].iter().map(|&i| all_scores[i]).collect();
        let perm_func: Vec<f64> = indices[n_phen..].iter().map(|&i| all_scores[i]).collect();
        let perm_diff = mean(&perm_phen) - mean(&perm_func);
        if perm_diff.abs() >= diff.abs() {
            extreme_count += 1;
        }
    }

    let p_value = extreme_count as f64 / n_perms as f64;
    println!("  p-value: {:.4}", p_value);
    println!("  Significant (p < 0.05): {}", p_value < 0.05);
    println!("  Significant (p < 0.01): {}", p_value < 0.01);
    println!("  Significant (p < 0.001): {}", p_value < 0.001);

    // Subcategory analysis
    println!("\n================================================================");
    println!("   PHENOMENAL SUBCATEGORY ANALYSIS");
    println!("================================================================\n");

    let mut phen_by_category: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();
    for (concept, score) in &phenomenal_results {
        phen_by_category
            .entry(concept.category.clone())
            .or_default()
            .push(*score);
    }

    let mut phen_categories: Vec<_> = phen_by_category.iter().collect();
    phen_categories.sort_by(|a, b| mean(b.1).partial_cmp(&mean(a.1)).unwrap());

    for (cat, scores) in &phen_categories {
        println!(
            "{:20} (n={:2}): {:.4} (+/- {:.4})",
            cat,
            scores.len(),
            mean(scores),
            std(scores)
        );
    }

    println!("\n================================================================");
    println!("   FUNCTIONAL SUBCATEGORY ANALYSIS");
    println!("================================================================\n");

    let mut func_by_category: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();
    for (concept, score) in &functional_results {
        func_by_category
            .entry(concept.category.clone())
            .or_default()
            .push(*score);
    }

    let mut func_categories: Vec<_> = func_by_category.iter().collect();
    func_categories.sort_by(|a, b| mean(b.1).partial_cmp(&mean(a.1)).unwrap());

    for (cat, scores) in &func_categories {
        println!(
            "{:20} (n={:2}): {:.4} (+/- {:.4})",
            cat,
            scores.len(),
            mean(scores),
            std(scores)
        );
    }

    // Save detailed CSV
    let csv_path = "data/consciousness_probe/h1_expanded_200_results.csv";
    let mut csv_file = File::create(csv_path)?;
    writeln!(csv_file, "id,text,type,category,subcategory,unity_score")?;

    for (concept, score) in &phenomenal_results {
        writeln!(
            csv_file,
            "\"{}\",\"{}\",phenomenal,{},{},{:.4}",
            concept.id,
            concept.text.replace("\"", "\\\""),
            concept.category,
            concept.subcategory,
            score
        )?;
    }
    for (concept, score) in &functional_results {
        writeln!(
            csv_file,
            "\"{}\",\"{}\",functional,{},{},{:.4}",
            concept.id,
            concept.text.replace("\"", "\\\""),
            concept.category,
            concept.subcategory,
            score
        )?;
    }

    println!("\nSaved detailed results to: {}", csv_path);

    // Effect size interpretation
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

    println!("H1 EXPANDED ANALYSIS (N=200):");
    println!("  Phenomenal mean: {:.4}", phen_mean);
    println!("  Functional mean: {:.4}", func_mean);
    println!("  Difference:      {:.4}", diff);
    println!("  Cohen's d:       {:.4} ({})", cohens_d, effect_size);
    println!("  p-value:         {:.4}", p_value);

    if p_value < 0.05 && diff > 0.0 {
        println!("\nRESULT: H1 STRONGLY SUPPORTED");
        println!("Phenomenal concepts show significantly higher topological unity");
        println!("than functional concepts across 200 diverse concepts.");
    } else if p_value < 0.05 {
        println!("\nRESULT: OPPOSITE TO H1");
    } else {
        println!("\nRESULT: NOT SIGNIFICANT");
    }

    println!("\n================================================================");
    println!("   H1 EXPANDED ANALYSIS COMPLETE");
    println!("================================================================\n");

    Ok(())
}