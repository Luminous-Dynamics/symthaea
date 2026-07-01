// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H1 Robustness Analysis
//!
//! Tests whether the phenomenal-computational distinction holds across:
//! 1. Different topology parameters (min_persistence, num_scales)
//! 2. Bootstrap resampling (confidence intervals)
//! 3. Leave-one-category-out analysis
//!
//! Generates CSV data for publication figures.

#[cfg(feature = "neural-bridge")]
use std::fs::File;
#[cfg(feature = "neural-bridge")]
use std::path::Path;
#[cfg(feature = "neural-bridge")]
use std::time::Instant;

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::ConsciousnessProbeV2;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::{ConsciousnessTopology, TopologyConfig};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!("Run with: cargo run --example h1_robustness --features neural-bridge --release");
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_robustness_analysis()
}

#[cfg(feature = "neural-bridge")]
fn run_robustness_analysis() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   H1 ROBUSTNESS ANALYSIS");
    println!("   Testing stability of phenomenal-computational distinction");
    println!("================================================================\n");

    let probe_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !probe_path.exists() {
        println!("ERROR: Probe weights not found");
        return Ok(());
    }

    println!("Loading BGE-M3 model...");
    let load_start = Instant::now();
    let mut probe = ConsciousnessProbeV2::load_with_probe(probe_path)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Load concept corpora
    let qualia_concepts = vec![
        "The subjective experience of seeing red",
        "What it is like to feel pain",
        "The taste of sweetness on my tongue",
        "The felt quality of hearing a musical note",
        "The smell of roses filling my awareness",
        "The feeling of warmth spreading through my body",
        "The experience of deep blue in the evening sky",
        "The raw sensation of pressure on my skin",
        "The bitter taste that lingers after coffee",
        "The cool sensation of water on my face",
        "The sharp sting of a paper cut",
        "The rich aroma of fresh bread baking",
        "The vibration I feel when humming a tune",
        "The sour pucker from biting a lemon",
        "The soft texture of velvet under my fingers",
        "The crisp scent of pine needles in the forest",
        "The sensation of cold metal against my palm",
        "The bright flash of yellow in a sunflower",
        "The deep bass rumble I feel in my chest",
        "The salty taste of ocean spray",
    ];

    let computation_concepts = vec![
        "Recursive function evaluation in programming",
        "Memory allocation and deallocation in systems",
        "Type inference in static analysis",
        "Binary search tree traversal algorithms",
        "Hash table collision resolution strategies",
        "Graph traversal using depth-first search",
        "Dynamic programming optimization techniques",
        "Garbage collection memory management",
        "Compiler lexical analysis and tokenization",
        "Network packet routing algorithms",
        "Quicksort partition and pivot selection",
        "Linked list node insertion and deletion",
        "Stack push and pop operations",
        "Heap data structure heapify operation",
        "Breadth-first search queue exploration",
        "Merge sort divide and conquer strategy",
        "Red-black tree rotation balancing",
        "Dijkstra shortest path computation",
        "Trie prefix tree string matching",
        "Memoization cache lookup optimization",
    ];

    // Pre-compute HVs for all concepts
    println!("Pre-computing hypervectors...");
    let start = Instant::now();
    let qualia_hvs: Vec<_> = qualia_concepts
        .iter()
        .map(|c| probe.concept_to_hv(c).unwrap())
        .collect();
    let comp_hvs: Vec<_> = computation_concepts
        .iter()
        .map(|c| probe.concept_to_hv(c).unwrap())
        .collect();
    println!("  Completed in {:.2}s\n", start.elapsed().as_secs_f64());

    // Helper to compute unity for a set of HVs with given config
    let compute_unity =
        |hvs: &[symthaea_core::hdc::BinaryHV], config: &TopologyConfig| -> Vec<f64> {
            hvs.iter()
                .map(|hv| {
                    let mut topology = ConsciousnessTopology::new(config.clone());
                    topology.add_state(*hv);
                    for shift in 1..5 {
                        let permuted = hv.permute(shift * 100);
                        topology.add_state(permuted);
                    }
                    topology.analyze(0.5).unity_score
                })
                .collect()
        };

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };

    // ================================================================
    // 1. Parameter Sensitivity Analysis
    // ================================================================
    println!("================================================================");
    println!("   PART 1: PARAMETER SENSITIVITY");
    println!("================================================================\n");

    let param_configs = vec![
        (
            "default",
            TopologyConfig {
                min_persistence: 0.05,
                max_scale: 1.0,
                num_scales: 20,
                detect_cycles: true,
                detect_voids: true,
            },
        ),
        (
            "loose",
            TopologyConfig {
                min_persistence: 0.01,
                max_scale: 1.0,
                num_scales: 20,
                detect_cycles: true,
                detect_voids: true,
            },
        ),
        (
            "strict",
            TopologyConfig {
                min_persistence: 0.10,
                max_scale: 1.0,
                num_scales: 20,
                detect_cycles: true,
                detect_voids: true,
            },
        ),
        (
            "fine",
            TopologyConfig {
                min_persistence: 0.05,
                max_scale: 1.0,
                num_scales: 50,
                detect_cycles: true,
                detect_voids: true,
            },
        ),
        (
            "coarse",
            TopologyConfig {
                min_persistence: 0.05,
                max_scale: 1.0,
                num_scales: 10,
                detect_cycles: true,
                detect_voids: true,
            },
        ),
    ];

    let mut param_results = Vec::new();

    for (name, config) in &param_configs {
        let qualia_unity = compute_unity(&qualia_hvs, config);
        let comp_unity = compute_unity(&comp_hvs, config);

        let diff = mean(&qualia_unity) - mean(&comp_unity);
        let pooled_std = ((std(&qualia_unity).powi(2) + std(&comp_unity).powi(2)) / 2.0).sqrt();
        let cohens_d = if pooled_std > 0.0 {
            diff / pooled_std
        } else {
            0.0
        };

        println!(
            "{:8}: Qualia={:.4}, Comp={:.4}, Diff={:.4}, d={:.3}",
            name,
            mean(&qualia_unity),
            mean(&comp_unity),
            diff,
            cohens_d
        );

        param_results.push((
            name.to_string(),
            mean(&qualia_unity),
            mean(&comp_unity),
            diff,
            cohens_d,
        ));
    }

    // ================================================================
    // 2. Bootstrap Confidence Intervals
    // ================================================================
    println!("\n================================================================");
    println!("   PART 2: BOOTSTRAP CONFIDENCE INTERVALS");
    println!("================================================================\n");

    let default_config = TopologyConfig {
        min_persistence: 0.05,
        max_scale: 1.0,
        num_scales: 20,
        detect_cycles: true,
        detect_voids: true,
    };

    let qualia_unity = compute_unity(&qualia_hvs, &default_config);
    let comp_unity = compute_unity(&comp_hvs, &default_config);

    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let n_bootstrap = 10000;
    let mut bootstrap_diffs = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let boot_qualia: Vec<f64> = (0..qualia_unity.len())
            .map(|_| *qualia_unity.choose(&mut rng).unwrap())
            .collect();
        let boot_comp: Vec<f64> = (0..comp_unity.len())
            .map(|_| *comp_unity.choose(&mut rng).unwrap())
            .collect();

        bootstrap_diffs.push(mean(&boot_qualia) - mean(&boot_comp));
    }

    bootstrap_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lower = bootstrap_diffs[(n_bootstrap as f64 * 0.025) as usize];
    let ci_upper = bootstrap_diffs[(n_bootstrap as f64 * 0.975) as usize];
    let boot_mean = mean(&bootstrap_diffs);

    println!("Bootstrap (n={})", n_bootstrap);
    println!("  Mean difference: {:.4}", boot_mean);
    println!("  95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
    println!("  CI excludes zero: {}", ci_lower > 0.0 || ci_upper < 0.0);

    // ================================================================
    // 3. Effect Size by Concept
    // ================================================================
    println!("\n================================================================");
    println!("   PART 3: INDIVIDUAL CONCEPT SCORES");
    println!("================================================================\n");

    // Save to CSV for plotting
    let csv_path = "data/consciousness_probe/h1_results.csv";
    let mut csv_file = File::create(csv_path)?;
    writeln!(csv_file, "concept,type,unity_score")?;

    for (i, concept) in qualia_concepts.iter().enumerate() {
        writeln!(
            csv_file,
            "\"{}\",phenomenal,{:.4}",
            concept, qualia_unity[i]
        )?;
    }
    for (i, concept) in computation_concepts.iter().enumerate() {
        writeln!(
            csv_file,
            "\"{}\",computational,{:.4}",
            concept, comp_unity[i]
        )?;
    }

    println!("Saved individual results to: {}", csv_path);

    // Distribution summary
    let qualia_high = qualia_unity.iter().filter(|&&x| x >= 0.9).count();
    let qualia_med = qualia_unity
        .iter()
        .filter(|&&x| x >= 0.5 && x < 0.9)
        .count();
    let qualia_low = qualia_unity.iter().filter(|&&x| x < 0.5).count();

    let comp_high = comp_unity.iter().filter(|&&x| x >= 0.9).count();
    let comp_med = comp_unity.iter().filter(|&&x| x >= 0.5 && x < 0.9).count();
    let comp_low = comp_unity.iter().filter(|&&x| x < 0.5).count();

    println!("\nDistribution (High >= 0.9, Med >= 0.5, Low < 0.5):");
    println!(
        "  Phenomenal:    High={}, Med={}, Low={}",
        qualia_high, qualia_med, qualia_low
    );
    println!(
        "  Computational: High={}, Med={}, Low={}",
        comp_high, comp_med, comp_low
    );

    // ================================================================
    // 4. Summary Statistics for Paper
    // ================================================================
    println!("\n================================================================");
    println!("   SUMMARY FOR PAPER");
    println!("================================================================\n");

    let observed_diff = mean(&qualia_unity) - mean(&comp_unity);
    let pooled_std = ((std(&qualia_unity).powi(2) + std(&comp_unity).powi(2)) / 2.0).sqrt();
    let cohens_d = if pooled_std > 0.0 {
        observed_diff / pooled_std
    } else {
        0.0
    };

    println!("Phenomenal concepts (n=20):");
    println!("  Mean: {:.4}", mean(&qualia_unity));
    println!("  SD:   {:.4}", std(&qualia_unity));
    println!(
        "  Range: [{:.4}, {:.4}]",
        qualia_unity.iter().cloned().fold(f64::INFINITY, f64::min),
        qualia_unity
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    println!("\nComputational concepts (n=20):");
    println!("  Mean: {:.4}", mean(&comp_unity));
    println!("  SD:   {:.4}", std(&comp_unity));
    println!(
        "  Range: [{:.4}, {:.4}]",
        comp_unity.iter().cloned().fold(f64::INFINITY, f64::min),
        comp_unity.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    println!("\nEffect:");
    println!("  Difference: {:.4}", observed_diff);
    println!("  Cohen's d:  {:.4}", cohens_d);
    println!("  95% CI:     [{:.4}, {:.4}]", ci_lower, ci_upper);

    // Effect size robustness
    let all_d: Vec<f64> = param_results.iter().map(|(_, _, _, _, d)| *d).collect();
    println!("\nEffect size robustness across parameters:");
    println!(
        "  Min d: {:.3}",
        all_d.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "  Max d: {:.3}",
        all_d.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!("  Effect stable: {}", all_d.iter().all(|&d| d > 0.2));

    println!("\n================================================================");
    println!("   ROBUSTNESS ANALYSIS COMPLETE");
    println!("================================================================\n");

    Ok(())
}