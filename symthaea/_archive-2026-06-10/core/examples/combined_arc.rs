// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Combined Arc: Bound Qualia vs Bound Computation
//!
//! Tests whether bound qualia pairs show different structure than bound
//! computation pairs. Combines H1 (phenomenal distinction) with H2 (binding).
//!
//! ## Hypothesis
//!
//! If binding captures phenomenal unity, then:
//! - Bound qualia pairs should show more integration than bound computation pairs
//! - The binding operation should amplify phenomenal differences

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::ConsciousnessProbeV2;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{
    BinaryHV,
    consciousness_topology::{ConsciousnessTopology, TopologyConfig},
};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!("Run with: cargo run --example combined_arc --features neural-bridge --release");
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_combined_arc()
}

#[cfg(feature = "neural-bridge")]
fn run_combined_arc() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   COMBINED ARC: BOUND QUALIA vs BOUND COMPUTATION");
    println!("   Does binding amplify phenomenal differences?");
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

    // Qualia pairs (sensory quality + object)
    let qualia_pairs = vec![
        // Color + object
        ("red", "seeing"),
        ("blue", "perceiving"),
        ("green", "experiencing"),
        ("yellow", "awareness"),
        ("orange", "sensation"),
        // Sound + quality
        ("loud", "hearing"),
        ("soft", "listening"),
        ("sharp", "feeling"),
        ("deep", "sensing"),
        ("high", "noticing"),
        // Touch + quality
        ("warm", "touching"),
        ("cold", "feeling"),
        ("smooth", "sensing"),
        ("rough", "experiencing"),
        ("soft", "noticing"),
        // Taste + quality
        ("sweet", "tasting"),
        ("sour", "savoring"),
        ("bitter", "experiencing"),
        ("salty", "sensing"),
        ("spicy", "feeling"),
    ];

    // Computation pairs (algorithm + operation)
    let computation_pairs = vec![
        // Sort + operation
        ("quicksort", "partitioning"),
        ("mergesort", "dividing"),
        ("heapsort", "extracting"),
        ("bubblesort", "swapping"),
        ("insertsort", "placing"),
        // Search + operation
        ("binary", "searching"),
        ("linear", "scanning"),
        ("hash", "probing"),
        ("depth", "traversing"),
        ("breadth", "exploring"),
        // Data structure + operation
        ("stack", "pushing"),
        ("queue", "enqueueing"),
        ("heap", "heapifying"),
        ("tree", "balancing"),
        ("graph", "connecting"),
        // Memory + operation
        ("allocate", "reserving"),
        ("deallocate", "freeing"),
        ("cache", "storing"),
        ("garbage", "collecting"),
        ("pointer", "dereferencing"),
    ];

    println!(
        "Testing {} qualia pairs and {} computation pairs\n",
        qualia_pairs.len(),
        computation_pairs.len()
    );

    // Topology config
    let topo_config = TopologyConfig {
        min_persistence: 0.05,
        max_scale: 1.0,
        num_scales: 20,
        detect_cycles: true,
        detect_voids: true,
    };

    // Helper to analyze a bound pair
    let analyze_bound_pair =
        |probe: &mut ConsciousnessProbeV2, a: &str, b: &str| -> Result<(f64, f64, f64)> {
            let hv_a = probe.concept_to_hv(a)?;
            let hv_b = probe.concept_to_hv(b)?;
            let bound = hv_a.bind(&hv_b);

            // Topology analysis
            let mut topology = ConsciousnessTopology::new(topo_config.clone());
            topology.add_state(bound);
            for shift in 1..10 {
                let permuted = bound.permute(shift * 50);
                topology.add_state(permuted);
            }
            let assessment = topology.analyze(0.5);

            // Vector metrics (using built-in hamming_distance)
            let hamming_a = bound.hamming_distance(&hv_a) as f64 / 16384.0;
            let hamming_b = bound.hamming_distance(&hv_b) as f64 / 16384.0;
            let equidistance = (hamming_a - hamming_b).abs();

            Ok((
                assessment.unity_score,
                hamming_a.min(hamming_b),
                equidistance,
            ))
        };

    // Process qualia pairs
    println!("Processing bound qualia pairs...");
    let start = Instant::now();
    let mut qualia_unity = Vec::new();
    let mut qualia_novelty = Vec::new();
    let mut qualia_equidist = Vec::new();
    for (a, b) in &qualia_pairs {
        let (unity, novelty, equidist) = analyze_bound_pair(&mut probe, a, b)?;
        qualia_unity.push(unity);
        qualia_novelty.push(novelty);
        qualia_equidist.push(equidist);
    }
    println!("  Completed in {:.2}s\n", start.elapsed().as_secs_f64());

    // Process computation pairs
    println!("Processing bound computation pairs...");
    let start = Instant::now();
    let mut comp_unity = Vec::new();
    let mut comp_novelty = Vec::new();
    let mut comp_equidist = Vec::new();
    for (a, b) in &computation_pairs {
        let (unity, novelty, equidist) = analyze_bound_pair(&mut probe, a, b)?;
        comp_unity.push(unity);
        comp_novelty.push(novelty);
        comp_equidist.push(equidist);
    }
    println!("  Completed in {:.2}s\n", start.elapsed().as_secs_f64());

    // Statistics
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };

    println!("================================================================");
    println!("   RESULTS: BOUND PAIR COMPARISON");
    println!("================================================================\n");

    println!("METRIC               QUALIA PAIRS         COMPUTATION PAIRS");
    println!("-----------------------------------------------------------------");
    println!(
        "Unity Score          {:.4} (+/-{:.4})      {:.4} (+/-{:.4})",
        mean(&qualia_unity),
        std(&qualia_unity),
        mean(&comp_unity),
        std(&comp_unity)
    );
    println!(
        "Novelty (min dist)   {:.4} (+/-{:.4})      {:.4} (+/-{:.4})",
        mean(&qualia_novelty),
        std(&qualia_novelty),
        mean(&comp_novelty),
        std(&comp_novelty)
    );
    println!(
        "Equidistance         {:.4} (+/-{:.4})      {:.4} (+/-{:.4})\n",
        mean(&qualia_equidist),
        std(&qualia_equidist),
        mean(&comp_equidist),
        std(&comp_equidist)
    );

    // Statistical tests
    println!("================================================================");
    println!("   STATISTICAL ANALYSIS");
    println!("================================================================\n");

    // Unity score comparison
    let unity_diff = mean(&qualia_unity) - mean(&comp_unity);
    let pooled_std = ((std(&qualia_unity).powi(2) + std(&comp_unity).powi(2)) / 2.0).sqrt();
    let cohens_d_unity = if pooled_std > 0.0 {
        unity_diff / pooled_std
    } else {
        0.0
    };

    println!("Unity Score Comparison:");
    println!("  Difference (Qualia - Computation): {:.4}", unity_diff);
    println!("  Cohen's d: {:.4}", cohens_d_unity);

    // Permutation test for unity
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let n_permutations = 10000;

    let all_unity: Vec<f64> = qualia_unity
        .iter()
        .chain(comp_unity.iter())
        .copied()
        .collect();
    let n_qualia = qualia_pairs.len();
    let mut extreme_count = 0;

    for _ in 0..n_permutations {
        let mut indices: Vec<usize> = (0..all_unity.len()).collect();
        indices.shuffle(&mut rng);
        let perm_qualia: Vec<f64> = indices[..n_qualia].iter().map(|&i| all_unity[i]).collect();
        let perm_comp: Vec<f64> = indices[n_qualia..].iter().map(|&i| all_unity[i]).collect();
        let perm_diff = mean(&perm_qualia) - mean(&perm_comp);
        if perm_diff.abs() >= unity_diff.abs() {
            extreme_count += 1;
        }
    }

    let p_value = extreme_count as f64 / n_permutations as f64;
    println!(
        "  p-value: {:.4} (n={} permutations)",
        p_value, n_permutations
    );
    println!("  Significant (p < 0.05): {}\n", p_value < 0.05);

    // Effect size interpretation
    let effect_size = if cohens_d_unity.abs() < 0.2 {
        "negligible"
    } else if cohens_d_unity.abs() < 0.5 {
        "small"
    } else if cohens_d_unity.abs() < 0.8 {
        "medium"
    } else {
        "large"
    };

    println!("================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if p_value < 0.05 {
        if unity_diff > 0.0 {
            println!("RESULT: COMBINED HYPOTHESIS SUPPORTED");
            println!("Bound qualia pairs show HIGHER unity than bound computation pairs.");
            println!("Binding AMPLIFIES phenomenal differences.");
        } else {
            println!("RESULT: OPPOSITE TO HYPOTHESIS");
            println!("Bound computation pairs show higher unity.");
        }
    } else {
        println!("RESULT: NOT SIGNIFICANT");
        println!("No clear difference between bound qualia and computation pairs.");
    }
    println!(
        "\nEffect size: {} (|d| = {:.2})",
        effect_size,
        cohens_d_unity.abs()
    );

    // Detailed results
    println!("\n================================================================");
    println!("   SAMPLE BOUND PAIR DETAILS");
    println!("================================================================\n");

    println!("Qualia Pairs (first 5):");
    for i in 0..5.min(qualia_pairs.len()) {
        let (a, b) = qualia_pairs[i];
        println!(
            "  bind({}, {}) -> Unity: {:.4}, Novelty: {:.4}",
            a, b, qualia_unity[i], qualia_novelty[i]
        );
    }

    println!("\nComputation Pairs (first 5):");
    for i in 0..5.min(computation_pairs.len()) {
        let (a, b) = computation_pairs[i];
        println!(
            "  bind({}, {}) -> Unity: {:.4}, Novelty: {:.4}",
            a, b, comp_unity[i], comp_novelty[i]
        );
    }

    println!("\n================================================================");
    println!("   COMBINED ARC EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}