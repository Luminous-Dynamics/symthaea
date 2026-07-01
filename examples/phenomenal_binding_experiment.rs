// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phenomenal Binding Experiment
//!
//! Tests Research Direction 2: HDC binding vs bundling for phenomenal unity.
//!
//! ## Hypothesis (H2)
//!
//! HDC binding (XOR) produces representations with higher topological integration
//! (lower component count, higher unity score) than bundling (majority vote),
//! specifically for concept pairs humans report as phenomenally unified.
//!
//! ## 2x2 Analysis Design
//!
//! - Factor A: Operation (Bind vs Bundle)
//! - Factor B: Pair Type (Unified vs Separate)
//!
//! Key test: Interaction effect - does binding specifically help unified pairs?
//!
//! ## Usage
//!
//! ```bash
//! # Run with default settings
//! cargo run --example phenomenal_binding_experiment --release
//!
//! # Run with verbose output
//! cargo run --example phenomenal_binding_experiment --release -- --verbose
//!
//! # Run quick mode (fewer iterations)
//! cargo run --example phenomenal_binding_experiment --release -- --quick
//! ```

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::consciousness_topology::{ConsciousnessTopology, TopologyConfig};
use symthaea_core::hdc::phenomenal_binding_study::PairType;

/// JSON structure for concept pairs file
#[derive(Debug, Deserialize)]
struct ConceptPairsFile {
    unified_pairs: Vec<ConceptPairJson>,
    separate_pairs: Vec<ConceptPairJson>,
}

/// JSON structure for a single concept pair
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ConceptPairJson {
    id: String,
    concept_a: String,
    concept_b: String,
    combined: String,
}

/// Concept pair with HDC vectors
struct ConceptPairWithVectors {
    id: String,
    pair_type: PairType,
    hv_a: BinaryHV,
    hv_b: BinaryHV,
}

/// Result of comparing bind vs bundle
struct ComparisonResult {
    pair_id: String,
    pair_type: PairType,
    bind_unity: f64,
    bundle_unity: f64,
    unity_advantage: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let quick = args.iter().any(|a| a == "--quick" || a == "-q");

    println!("\n================================================================");
    println!("   PHENOMENAL BINDING EXPERIMENT");
    println!("   Research Direction 2: HDC Binding vs Bundling");
    println!("================================================================\n");

    // Load or create pairs
    let pairs = load_or_create_pairs("data/binding_study/concept_pairs.json")?;
    let n_unified = pairs
        .iter()
        .filter(|p| p.pair_type == PairType::Unified)
        .count();
    let n_separate = pairs
        .iter()
        .filter(|p| p.pair_type == PairType::Separate)
        .count();

    println!(
        "Pairs: {} unified, {} separate, {} total\n",
        n_unified,
        n_separate,
        pairs.len()
    );

    // Run experiment
    let results: Vec<ComparisonResult> = pairs
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            if verbose && (i + 1) % 20 == 0 {
                println!("  Processing {}/{}...", i + 1, pairs.len());
            }
            compare_operations(pair)
        })
        .collect();

    // Compute cell means
    let unified: Vec<_> = results
        .iter()
        .filter(|r| r.pair_type == PairType::Unified)
        .collect();
    let separate: Vec<_> = results
        .iter()
        .filter(|r| r.pair_type == PairType::Separate)
        .collect();

    let bind_unified = mean(unified.iter().map(|r| r.bind_unity));
    let bundle_unified = mean(unified.iter().map(|r| r.bundle_unity));
    let bind_separate = mean(separate.iter().map(|r| r.bind_unity));
    let bundle_separate = mean(separate.iter().map(|r| r.bundle_unity));

    // Display results
    println!("================================================================");
    println!("   CELL MEANS (Unity Score)");
    println!("================================================================\n");
    println!(
        "                 Unified (n={})    Separate (n={})",
        unified.len(),
        separate.len()
    );
    println!(
        "  Bind:          {:.4}              {:.4}",
        bind_unified, bind_separate
    );
    println!(
        "  Bundle:        {:.4}              {:.4}\n",
        bundle_unified, bundle_separate
    );

    // Main effects
    let main_op = ((bind_unified + bind_separate) - (bundle_unified + bundle_separate)) / 2.0;
    let main_type = ((bind_unified + bundle_unified) - (bind_separate + bundle_separate)) / 2.0;

    // Interaction
    let adv_unified = bind_unified - bundle_unified;
    let adv_separate = bind_separate - bundle_separate;
    let interaction = adv_unified - adv_separate;

    println!("================================================================");
    println!("   EFFECTS");
    println!("================================================================\n");
    println!("Main Effects:");
    println!("  Operation (Bind - Bundle): {:+.4}", main_op);
    println!("  Pair Type (Unified - Separate): {:+.4}\n", main_type);

    println!("Binding Advantage:");
    println!("  Unified pairs: {:+.4}", adv_unified);
    println!("  Separate pairs: {:+.4}\n", adv_separate);

    println!("Interaction: {:+.4}", interaction);

    // Effect sizes
    let all_unity: Vec<f64> = results
        .iter()
        .flat_map(|r| vec![r.bind_unity, r.bundle_unity])
        .collect();
    let pooled_std = std_dev(&all_unity).max(0.001);

    let d_unified = adv_unified / pooled_std;
    let d_separate = adv_separate / pooled_std;
    let d_interaction = interaction / pooled_std;

    println!("\nEffect Sizes (Cohen's d):");
    println!("  Unified binding advantage: {:+.4}", d_unified);
    println!("  Separate binding advantage: {:+.4}", d_separate);
    println!("  Interaction: {:+.4}", d_interaction);

    // Permutation test
    let n_perm = if quick { 500 } else { 5000 };
    let p_value = permutation_test_interaction(&unified, &separate, n_perm);

    println!("\nStatistical Test (permutation, n={}):", n_perm);
    println!(
        "  Interaction p-value: {:.4} {}",
        p_value,
        if p_value < 0.05 { "*" } else { "" }
    );

    // Interpretation
    println!("\n================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if interaction > 0.0 && p_value < 0.05 {
        println!("H2 SUPPORTED: Binding produces higher integration for unified pairs.");
        println!("Effect size: d = {:+.3}", d_interaction);
    } else if main_op > 0.0 && d_unified.abs() > 0.2 {
        println!("PARTIAL SUPPORT: Binding improves integration, but not pair-type specific.");
    } else {
        println!("H2 NOT SUPPORTED: No clear binding advantage detected.");
    }

    // Sample results
    if verbose {
        println!("\n================================================================");
        println!("   SAMPLE RESULTS");
        println!("================================================================\n");

        let mut sorted: Vec<_> = results.iter().collect();
        sorted.sort_by(|a, b| b.unity_advantage.partial_cmp(&a.unity_advantage).unwrap());

        println!("Top 5 by binding advantage:");
        for r in sorted.iter().take(5) {
            println!(
                "  {} ({}): adv={:+.4}",
                r.pair_id,
                r.pair_type.as_str(),
                r.unity_advantage
            );
        }
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

/// Load pairs from JSON or create synthetic ones
fn load_or_create_pairs(path: &str) -> Result<Vec<ConceptPairWithVectors>> {
    if Path::new(path).exists() {
        println!("Loading pairs from: {}", path);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data: ConceptPairsFile = serde_json::from_reader(reader)?;

        let mut pairs = Vec::new();
        for (i, p) in data.unified_pairs.iter().enumerate() {
            pairs.push(ConceptPairWithVectors {
                id: p.id.clone(),
                pair_type: PairType::Unified,
                hv_a: BinaryHV::random(1000 + i as u64 * 2),
                hv_b: BinaryHV::random(1001 + i as u64 * 2),
            });
        }
        for (i, p) in data.separate_pairs.iter().enumerate() {
            pairs.push(ConceptPairWithVectors {
                id: p.id.clone(),
                pair_type: PairType::Separate,
                hv_a: BinaryHV::random(2000 + i as u64 * 2),
                hv_b: BinaryHV::random(2001 + i as u64 * 2),
            });
        }
        Ok(pairs)
    } else {
        println!("Using synthetic pairs (file not found)\n");
        Ok(create_synthetic_pairs())
    }
}

/// Create synthetic pairs
fn create_synthetic_pairs() -> Vec<ConceptPairWithVectors> {
    let mut pairs = Vec::new();

    // 20 unified pairs
    for i in 0..20 {
        pairs.push(ConceptPairWithVectors {
            id: format!("unified_{:03}", i + 1),
            pair_type: PairType::Unified,
            hv_a: BinaryHV::random(1000 + i * 2),
            hv_b: BinaryHV::random(1001 + i * 2),
        });
    }

    // 20 separate pairs
    for i in 0..20 {
        pairs.push(ConceptPairWithVectors {
            id: format!("separate_{:03}", i + 1),
            pair_type: PairType::Separate,
            hv_a: BinaryHV::random(2000 + i * 2),
            hv_b: BinaryHV::random(2001 + i * 2),
        });
    }

    pairs
}

/// Compare bind vs bundle for a pair
fn compare_operations(pair: &ConceptPairWithVectors) -> ComparisonResult {
    let bound = pair.hv_a.bind(&pair.hv_b);
    let bundled = BinaryHV::bundle(&[pair.hv_a, pair.hv_b]);

    let bind_unity = analyze_topology(&bound);
    let bundle_unity = analyze_topology(&bundled);

    ComparisonResult {
        pair_id: pair.id.clone(),
        pair_type: pair.pair_type,
        bind_unity,
        bundle_unity,
        unity_advantage: bind_unity - bundle_unity,
    }
}

/// Analyze topology for an HDC vector
fn analyze_topology(hv: &BinaryHV) -> f64 {
    let config = TopologyConfig::default();
    let mut topology = ConsciousnessTopology::new(config);

    topology.add_state(*hv);
    for shift in 1..5 {
        topology.add_state(hv.permute(shift * 100));
    }

    topology.analyze(0.5).unity_score
}

/// Permutation test for interaction effect
fn permutation_test_interaction(
    unified: &[&ComparisonResult],
    separate: &[&ComparisonResult],
    n_perm: usize,
) -> f64 {
    let unified_advs: Vec<f64> = unified.iter().map(|r| r.unity_advantage).collect();
    let separate_advs: Vec<f64> = separate.iter().map(|r| r.unity_advantage).collect();
    let observed = mean(unified_advs.iter().copied()) - mean(separate_advs.iter().copied());

    let mut combined: Vec<f64> = unified_advs.iter().chain(&separate_advs).copied().collect();
    let n_u = unified_advs.len();
    let mut more_extreme = 0;
    let mut rng: u64 = 42;

    for _ in 0..n_perm {
        for i in (1..combined.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            combined.swap(i, (rng as usize) % (i + 1));
        }
        let perm = mean(combined[..n_u].iter().copied()) - mean(combined[n_u..].iter().copied());
        if perm.abs() >= observed.abs() {
            more_extreme += 1;
        }
    }

    more_extreme as f64 / n_perm as f64
}

fn mean<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let v: Vec<f64> = iter.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
}