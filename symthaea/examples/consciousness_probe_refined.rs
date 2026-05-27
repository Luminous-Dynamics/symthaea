// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Refined H1 Test: Pure Qualia vs Pure Computation
//!
//! This example tests the refined hypothesis using only:
//! - Pure qualia: sensory experiences (seeing, hearing, tasting, etc.)
//! - Pure computation: algorithms and data structures
//!
//! This removes philosophical/abstract concepts that may muddy the signal.

#[cfg(feature = "neural-bridge")]
use std::path::Path;
#[cfg(feature = "neural-bridge")]
use std::time::Instant;

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{ConceptCorpus, ConsciousnessProbeV2, ProbeConfig};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::TopologyConfig;

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!(
            "Run with: cargo run --example consciousness_probe_refined --features neural-bridge --release"
        );
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   H1 REFINED: PURE QUALIA vs PURE COMPUTATION");
    println!("   Testing cleaner signal without philosophical abstractions");
    println!("================================================================\n");

    let probe_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    let qualia_path = Path::new("data/consciousness_probe/qualia_only.json");
    let computation_path = Path::new("data/consciousness_probe/computation_only.json");

    if !probe_path.exists() {
        println!("ERROR: Probe weights not found at {}", probe_path.display());
        return Ok(());
    }

    // Load filtered corpora
    println!("Loading filtered concept corpora...");

    let qualia_corpus = if qualia_path.exists() {
        ConceptCorpus::load(qualia_path)?
    } else {
        println!("ERROR: qualia_only.json not found");
        return Ok(());
    };

    let computation_corpus = if computation_path.exists() {
        ConceptCorpus::load(computation_path)?
    } else {
        println!("ERROR: computation_only.json not found");
        return Ok(());
    };

    println!("  Pure qualia concepts: {}", qualia_corpus.len());
    println!(
        "  Pure computation concepts: {}\n",
        computation_corpus.len()
    );

    // Initialize probe
    println!("Loading BGE-M3 model and probe weights...");
    let load_start = Instant::now();
    let mut probe = ConsciousnessProbeV2::load_with_probe(probe_path)?;
    println!(
        "  Model loaded in {:.2}s\n",
        load_start.elapsed().as_secs_f64()
    );

    // More sensitive topology config for refined analysis
    let config = ProbeConfig {
        topology_config: TopologyConfig {
            min_persistence: 0.05, // Lower threshold to catch subtle features
            max_scale: 1.0,
            num_scales: 20, // More scales for finer resolution
            detect_cycles: true,
            detect_voids: true,
        },
        n_permutations: 10_000,
        analysis_scale: 0.5,
        min_states: 5,
    };
    probe = probe.with_config(config);

    // Run experiment
    println!("================================================================");
    println!("   PROBING PURE CONCEPTS");
    println!("================================================================\n");

    println!("Probing pure qualia concepts...");
    let q_start = Instant::now();
    let qualia_results = probe.probe_corpus_texts(&qualia_corpus)?;
    println!(
        "  Completed in {:.2}s ({} concepts)\n",
        q_start.elapsed().as_secs_f64(),
        qualia_results.len()
    );

    println!("Probing pure computation concepts...");
    let c_start = Instant::now();
    let computation_results = probe.probe_corpus_texts(&computation_corpus)?;
    println!(
        "  Completed in {:.2}s ({} concepts)\n",
        c_start.elapsed().as_secs_f64(),
        computation_results.len()
    );

    // Compare
    println!("================================================================");
    println!("   STATISTICAL COMPARISON (REFINED)");
    println!("================================================================\n");

    let comparison = probe.compare_classes(&qualia_results, &computation_results);

    println!("Pure Qualia (n={}):", comparison.phenomenal_stats.n);
    println!(
        "  Mean Unity Score: {:.4} (+/- {:.4})",
        comparison.phenomenal_stats.mean_unity, comparison.phenomenal_stats.std_unity
    );
    println!(
        "  Mean beta_0 (components): {:.2}",
        comparison.phenomenal_stats.mean_beta_0
    );
    println!(
        "  Mean beta_1 (cycles): {:.2}\n",
        comparison.phenomenal_stats.mean_beta_1
    );

    println!("Pure Computation (n={}):", comparison.functional_stats.n);
    println!(
        "  Mean Unity Score: {:.4} (+/- {:.4})",
        comparison.functional_stats.mean_unity, comparison.functional_stats.std_unity
    );
    println!(
        "  Mean beta_0 (components): {:.2}",
        comparison.functional_stats.mean_beta_0
    );
    println!(
        "  Mean beta_1 (cycles): {:.2}\n",
        comparison.functional_stats.mean_beta_1
    );

    println!("Statistical Tests:");
    println!(
        "  Observed Difference: {:.4}",
        comparison.observed_difference
    );
    println!("  Cohen's d: {:.4}", comparison.cohens_d);
    println!(
        "  p-value: {:.4} (n={} permutations)",
        comparison.p_value, comparison.n_permutations
    );
    println!("  Significant (p < 0.05): {}\n", comparison.is_significant);

    // Effect size interpretation
    let effect_size = if comparison.cohens_d.abs() < 0.2 {
        "negligible"
    } else if comparison.cohens_d.abs() < 0.5 {
        "small"
    } else if comparison.cohens_d.abs() < 0.8 {
        "medium"
    } else {
        "large"
    };

    println!("================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if comparison.is_significant {
        if comparison.observed_difference > 0.0 {
            println!("RESULT: SIGNIFICANT - Qualia > Computation");
            println!("H1 SUPPORTED with refined concepts!");
        } else {
            println!("RESULT: SIGNIFICANT - Computation > Qualia");
            println!("Opposite to H1 prediction.");
        }
    } else {
        println!("RESULT: NOT SIGNIFICANT");
        println!("Refined test also shows no clear difference.");
    }
    println!(
        "\nEffect size: {} (|d| = {:.2})",
        effect_size,
        comparison.cohens_d.abs()
    );

    // Show all individual results for detailed analysis
    println!("\n================================================================");
    println!("   ALL INDIVIDUAL RESULTS");
    println!("================================================================\n");

    println!("All Qualia Results:");
    for r in &qualia_results {
        println!(
            "  [{:.4}] {} - \"{}\"",
            r.unity_score,
            r.concept.subcategory,
            truncate(&r.concept.text, 50)
        );
    }

    println!("\nAll Computation Results:");
    for r in &computation_results {
        println!(
            "  [{:.4}] {} - \"{}\"",
            r.unity_score,
            r.concept.subcategory,
            truncate(&r.concept.text, 50)
        );
    }

    println!("\n================================================================");
    println!("   REFINED H1 EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}