// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H2: Phenomenal Binding Hypothesis with Real Embeddings
//!
//! Tests whether HDC binding (XOR) produces higher topological unity than
//! bundling (majority vote) specifically for phenomenally-unified concept pairs.
//!
//! ## Hypothesis
//!
//! H2: Binding creates genuine phenomenal unity (measurable by topology),
//! while bundling creates mere superposition.
//!
//! ## Test Design
//!
//! - Unified pairs: (red, apple), (loud, crash) - humans report as single experience
//! - Separate pairs: (red, mailbox), (loud, background) - two distinct experiences
//! - 2x2 ANOVA: Operation (Bind/Bundle) x Pair Type (Unified/Separate)
//! - Key test: Interaction effect (binding specifically helps unified pairs)

#[cfg(feature = "neural-bridge")]
use std::path::Path;
#[cfg(feature = "neural-bridge")]
use std::time::Instant;

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
        println!(
            "Run with: cargo run --example phenomenal_binding_real --features neural-bridge --release"
        );
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_h2_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_h2_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   H2: PHENOMENAL BINDING HYPOTHESIS");
    println!("   Does binding (XOR) create more unity than bundling?");
    println!("================================================================\n");

    let probe_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !probe_path.exists() {
        println!("ERROR: Probe weights not found");
        return Ok(());
    }

    // Load probe for concept -> HDC projection
    println!("Loading BGE-M3 model...");
    let load_start = Instant::now();
    let mut probe = ConsciousnessProbeV2::load_with_probe(probe_path)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Unified phenomenal pairs (single unified experience)
    let unified_pairs = vec![
        // Visual-object unity
        ("red", "apple"),
        ("blue", "sky"),
        ("green", "leaf"),
        ("golden", "sunset"),
        ("white", "snow"),
        // Sound-event unity
        ("loud", "crash"),
        ("soft", "whisper"),
        ("deep", "rumble"),
        ("crackling", "fire"),
        ("rushing", "waterfall"),
        // Touch-sensation unity
        ("warm", "sunlight"),
        ("soft", "fur"),
        ("cool", "breeze"),
        ("smooth", "silk"),
        ("cold", "ice"),
        // Taste-substance unity
        ("sweet", "honey"),
        ("sour", "lemon"),
        ("bitter", "coffee"),
        ("spicy", "chili"),
        ("rich", "chocolate"),
    ];

    // Separate pairs (two distinct experiences)
    let separate_pairs = vec![
        // Color and container
        ("red", "mailbox"),
        ("blue", "building"),
        ("green", "fence"),
        ("yellow", "sign"),
        ("white", "wall"),
        // Sound and background
        ("loud", "background"),
        ("soft", "environment"),
        ("deep", "context"),
        ("quiet", "office"),
        ("noisy", "street"),
        // Touch and space
        ("warm", "room"),
        ("soft", "carpet"),
        ("cool", "basement"),
        ("smooth", "floor"),
        ("cold", "garage"),
        // Taste and container
        ("sweet", "jar"),
        ("sour", "bottle"),
        ("bitter", "cup"),
        ("spicy", "bowl"),
        ("bland", "dish"),
    ];

    println!(
        "Testing {} unified pairs and {} separate pairs\n",
        unified_pairs.len(),
        separate_pairs.len()
    );

    // Topology configuration
    let topo_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: true,
    };

    // Helper to analyze a single HV's topology
    let analyze_hv = |hv: &BinaryHV| -> f64 {
        let mut topology = ConsciousnessTopology::new(topo_config.clone());
        topology.add_state(*hv);
        // Add permuted variations to create point cloud
        for shift in 1..5 {
            let permuted = hv.permute(shift * 100);
            topology.add_state(permuted);
        }
        let assessment = topology.analyze(0.5);
        assessment.unity_score
    };

    // Results
    let mut unified_bind_unity = Vec::new();
    let mut unified_bundle_unity = Vec::new();
    let mut separate_bind_unity = Vec::new();
    let mut separate_bundle_unity = Vec::new();

    // Process unified pairs
    println!("Processing unified pairs...");
    let unified_start = Instant::now();
    for (a, b) in &unified_pairs {
        let hv_a = probe.concept_to_hv(a)?;
        let hv_b = probe.concept_to_hv(b)?;

        // Binding (XOR) - creates new association
        let bound = hv_a.bind(&hv_b);
        unified_bind_unity.push(analyze_hv(&bound));

        // Bundling (majority vote) - creates superposition
        let bundled = BinaryHV::bundle(&[hv_a.clone(), hv_b.clone()]);
        unified_bundle_unity.push(analyze_hv(&bundled));
    }
    println!(
        "  Completed in {:.2}s\n",
        unified_start.elapsed().as_secs_f64()
    );

    // Process separate pairs
    println!("Processing separate pairs...");
    let separate_start = Instant::now();
    for (a, b) in &separate_pairs {
        let hv_a = probe.concept_to_hv(a)?;
        let hv_b = probe.concept_to_hv(b)?;

        let bound = hv_a.bind(&hv_b);
        separate_bind_unity.push(analyze_hv(&bound));

        let bundled = BinaryHV::bundle(&[hv_a.clone(), hv_b.clone()]);
        separate_bundle_unity.push(analyze_hv(&bundled));
    }
    println!(
        "  Completed in {:.2}s\n",
        separate_start.elapsed().as_secs_f64()
    );

    // Calculate statistics
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };

    let unified_bind_mean = mean(&unified_bind_unity);
    let unified_bundle_mean = mean(&unified_bundle_unity);
    let separate_bind_mean = mean(&separate_bind_unity);
    let separate_bundle_mean = mean(&separate_bundle_unity);

    println!("================================================================");
    println!("   RESULTS: 2x2 ANALYSIS");
    println!("================================================================\n");

    println!("                      BIND (XOR)       BUNDLE (MAJ)");
    println!("                    -------------     -------------");
    println!(
        "UNIFIED PAIRS        {:.4} (+/-{:.4})   {:.4} (+/-{:.4})",
        unified_bind_mean,
        std(&unified_bind_unity),
        unified_bundle_mean,
        std(&unified_bundle_unity)
    );
    println!(
        "SEPARATE PAIRS       {:.4} (+/-{:.4})   {:.4} (+/-{:.4})\n",
        separate_bind_mean,
        std(&separate_bind_unity),
        separate_bundle_mean,
        std(&separate_bundle_unity)
    );

    // Key metrics
    let bind_advantage_unified = unified_bind_mean - unified_bundle_mean;
    let bind_advantage_separate = separate_bind_mean - separate_bundle_mean;
    let interaction_effect = bind_advantage_unified - bind_advantage_separate;

    println!("KEY METRICS:");
    println!("  Bind advantage (unified):  {:.4}", bind_advantage_unified);
    println!(
        "  Bind advantage (separate): {:.4}",
        bind_advantage_separate
    );
    println!("  INTERACTION EFFECT:        {:.4}\n", interaction_effect);

    // Permutation test for interaction effect
    println!("Running permutation test for interaction effect...");
    let n_permutations = 10000;
    let observed_interaction = interaction_effect;

    // Combine all data for permutation
    let all_bind: Vec<f64> = unified_bind_unity
        .iter()
        .chain(separate_bind_unity.iter())
        .copied()
        .collect();
    let all_bundle: Vec<f64> = unified_bundle_unity
        .iter()
        .chain(separate_bundle_unity.iter())
        .copied()
        .collect();
    let n_unified = unified_pairs.len();

    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut extreme_count = 0;

    for _ in 0..n_permutations {
        // Shuffle labels
        let mut indices: Vec<usize> = (0..all_bind.len()).collect();
        indices.shuffle(&mut rng);

        // Split into permuted unified/separate
        let perm_unified_bind: Vec<f64> =
            indices[..n_unified].iter().map(|&i| all_bind[i]).collect();
        let perm_unified_bundle: Vec<f64> = indices[..n_unified]
            .iter()
            .map(|&i| all_bundle[i])
            .collect();
        let perm_separate_bind: Vec<f64> =
            indices[n_unified..].iter().map(|&i| all_bind[i]).collect();
        let perm_separate_bundle: Vec<f64> = indices[n_unified..]
            .iter()
            .map(|&i| all_bundle[i])
            .collect();

        // Calculate permuted interaction
        let perm_bind_adv_unified = mean(&perm_unified_bind) - mean(&perm_unified_bundle);
        let perm_bind_adv_separate = mean(&perm_separate_bind) - mean(&perm_separate_bundle);
        let perm_interaction = perm_bind_adv_unified - perm_bind_adv_separate;

        if perm_interaction.abs() >= observed_interaction.abs() {
            extreme_count += 1;
        }
    }

    let p_value = extreme_count as f64 / n_permutations as f64;
    let is_significant = p_value < 0.05;

    println!("  Permutations: {}", n_permutations);
    println!("  p-value: {:.4}", p_value);
    println!("  Significant (p < 0.05): {}\n", is_significant);

    // Interpretation
    println!("================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if is_significant {
        if interaction_effect > 0.0 {
            println!("RESULT: H2 SUPPORTED");
            println!("Binding produces MORE unity than bundling,");
            println!("SPECIFICALLY for phenomenally-unified concept pairs.");
            println!("\nThis suggests binding captures something about");
            println!("phenomenal unity that bundling does not.");
        } else {
            println!("RESULT: OPPOSITE TO H2");
            println!("Binding advantage is greater for SEPARATE pairs.");
        }
    } else {
        println!("RESULT: H2 NOT SUPPORTED");
        println!("No significant interaction effect.");
        println!("Binding advantage (if any) is similar for both pair types.");
    }

    // Effect size (Cohen's d for interaction)
    let pooled_std = ((std(&unified_bind_unity).powi(2)
        + std(&unified_bundle_unity).powi(2)
        + std(&separate_bind_unity).powi(2)
        + std(&separate_bundle_unity).powi(2))
        / 4.0)
        .sqrt();
    let cohens_d = if pooled_std > 0.0 {
        interaction_effect / pooled_std
    } else {
        0.0
    };

    let effect_size = if cohens_d.abs() < 0.2 {
        "negligible"
    } else if cohens_d.abs() < 0.5 {
        "small"
    } else if cohens_d.abs() < 0.8 {
        "medium"
    } else {
        "large"
    };

    println!("\nEffect size: {} (d = {:.3})", effect_size, cohens_d);

    // Sample detailed results
    println!("\n================================================================");
    println!("   SAMPLE DETAILED RESULTS");
    println!("================================================================\n");

    println!("Unified Pairs (first 5):");
    for i in 0..5.min(unified_pairs.len()) {
        let (a, b) = unified_pairs[i];
        println!(
            "  ({}, {}) - Bind: {:.4}, Bundle: {:.4}, Delta: {:.4}",
            a,
            b,
            unified_bind_unity[i],
            unified_bundle_unity[i],
            unified_bind_unity[i] - unified_bundle_unity[i]
        );
    }

    println!("\nSeparate Pairs (first 5):");
    for i in 0..5.min(separate_pairs.len()) {
        let (a, b) = separate_pairs[i];
        println!(
            "  ({}, {}) - Bind: {:.4}, Bundle: {:.4}, Delta: {:.4}",
            a,
            b,
            separate_bind_unity[i],
            separate_bundle_unity[i],
            separate_bind_unity[i] - separate_bundle_unity[i]
        );
    }

    println!("\n================================================================");
    println!("   H2 EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}