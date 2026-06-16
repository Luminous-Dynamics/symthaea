// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H2 Revised: Phenomenal Binding via Vector Metrics
//!
//! Tests whether HDC binding (XOR) creates different vector structure than
//! bundling (majority vote) using direct metrics instead of topology.
//!
//! ## Metrics
//!
//! 1. Equidistance: Is bind(a,b) equidistant from a and b?
//! 2. Hamming distance: Bit differences from components
//! 3. Novelty: How different is the result from both inputs?
//!
//! ## Hypothesis
//!
//! H2 Revised: Binding creates vectors equidistant from inputs (true integration),
//! while bundling creates vectors closer to one input (superposition).

#[cfg(feature = "neural-bridge")]
use std::path::Path;
#[cfg(feature = "neural-bridge")]
use std::time::Instant;

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::ConsciousnessProbeV2;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::BinaryHV;

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!(
            "Run with: cargo run --example phenomenal_binding_vectors --features neural-bridge --release"
        );
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_h2_vectors()
}

#[cfg(feature = "neural-bridge")]
fn run_h2_vectors() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   H2 REVISED: BINDING VIA VECTOR METRICS");
    println!("   Testing equidistance and novelty of bind vs bundle");
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

    // Unified phenomenal pairs
    let unified_pairs = vec![
        ("red", "apple"),
        ("blue", "sky"),
        ("green", "leaf"),
        ("golden", "sunset"),
        ("white", "snow"),
        ("loud", "crash"),
        ("soft", "whisper"),
        ("deep", "rumble"),
        ("crackling", "fire"),
        ("rushing", "waterfall"),
        ("warm", "sunlight"),
        ("soft", "fur"),
        ("cool", "breeze"),
        ("smooth", "silk"),
        ("cold", "ice"),
        ("sweet", "honey"),
        ("sour", "lemon"),
        ("bitter", "coffee"),
        ("spicy", "chili"),
        ("rich", "chocolate"),
    ];

    // Separate pairs
    let separate_pairs = vec![
        ("red", "mailbox"),
        ("blue", "building"),
        ("green", "fence"),
        ("yellow", "sign"),
        ("white", "wall"),
        ("loud", "background"),
        ("soft", "environment"),
        ("deep", "context"),
        ("quiet", "office"),
        ("noisy", "street"),
        ("warm", "room"),
        ("soft", "carpet"),
        ("cool", "basement"),
        ("smooth", "floor"),
        ("cold", "garage"),
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

    // Metrics storage
    #[derive(Debug, Clone)]
    struct PairMetrics {
        // Binding metrics
        bind_dist_to_a: f64,    // Hamming distance from a
        bind_dist_to_b: f64,    // Hamming distance from b
        bind_equidistance: f64, // |dist_a - dist_b| (lower = more equidistant)
        bind_novelty: f64,      // min(dist_a, dist_b) / HV_DIM (higher = more novel)

        // Bundling metrics
        bundle_dist_to_a: f64,
        bundle_dist_to_b: f64,
        bundle_equidistance: f64,
        bundle_novelty: f64,
    }

    let compute_metrics = |hv_a: &BinaryHV, hv_b: &BinaryHV| -> PairMetrics {
        let bound = hv_a.bind(hv_b);
        let bundled = BinaryHV::bundle(&[hv_a.clone(), hv_b.clone()]);

        let hv_dim = 16384.0;

        // Binding distances (using built-in hamming_distance)
        let bind_dist_a = bound.hamming_distance(hv_a) as f64;
        let bind_dist_b = bound.hamming_distance(hv_b) as f64;

        // Bundling distances
        let bundle_dist_a = bundled.hamming_distance(hv_a) as f64;
        let bundle_dist_b = bundled.hamming_distance(hv_b) as f64;

        PairMetrics {
            bind_dist_to_a: bind_dist_a / hv_dim,
            bind_dist_to_b: bind_dist_b / hv_dim,
            bind_equidistance: (bind_dist_a - bind_dist_b).abs() / hv_dim,
            bind_novelty: bind_dist_a.min(bind_dist_b) / hv_dim,

            bundle_dist_to_a: bundle_dist_a / hv_dim,
            bundle_dist_to_b: bundle_dist_b / hv_dim,
            bundle_equidistance: (bundle_dist_a - bundle_dist_b).abs() / hv_dim,
            bundle_novelty: bundle_dist_a.min(bundle_dist_b) / hv_dim,
        }
    };

    // Process pairs
    println!("Processing unified pairs...");
    let start = Instant::now();
    let mut unified_metrics = Vec::new();
    for (a, b) in &unified_pairs {
        let hv_a = probe.concept_to_hv(a)?;
        let hv_b = probe.concept_to_hv(b)?;
        unified_metrics.push(compute_metrics(&hv_a, &hv_b));
    }
    println!("  Completed in {:.2}s\n", start.elapsed().as_secs_f64());

    println!("Processing separate pairs...");
    let start = Instant::now();
    let mut separate_metrics = Vec::new();
    for (a, b) in &separate_pairs {
        let hv_a = probe.concept_to_hv(a)?;
        let hv_b = probe.concept_to_hv(b)?;
        separate_metrics.push(compute_metrics(&hv_a, &hv_b));
    }
    println!("  Completed in {:.2}s\n", start.elapsed().as_secs_f64());

    // Statistics helpers
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };

    // Extract metrics
    let unified_bind_equidist: Vec<f64> = unified_metrics
        .iter()
        .map(|m| m.bind_equidistance)
        .collect();
    let unified_bundle_equidist: Vec<f64> = unified_metrics
        .iter()
        .map(|m| m.bundle_equidistance)
        .collect();
    let unified_bind_novelty: Vec<f64> = unified_metrics.iter().map(|m| m.bind_novelty).collect();
    let unified_bundle_novelty: Vec<f64> =
        unified_metrics.iter().map(|m| m.bundle_novelty).collect();

    let separate_bind_equidist: Vec<f64> = separate_metrics
        .iter()
        .map(|m| m.bind_equidistance)
        .collect();
    let separate_bundle_equidist: Vec<f64> = separate_metrics
        .iter()
        .map(|m| m.bundle_equidistance)
        .collect();
    let separate_bind_novelty: Vec<f64> = separate_metrics.iter().map(|m| m.bind_novelty).collect();
    let separate_bundle_novelty: Vec<f64> =
        separate_metrics.iter().map(|m| m.bundle_novelty).collect();

    println!("================================================================");
    println!("   RESULTS: EQUIDISTANCE ANALYSIS");
    println!("   (Lower = more equidistant from both inputs)");
    println!("================================================================\n");

    println!("                      BIND (XOR)       BUNDLE (MAJ)");
    println!("                    -------------     -------------");
    println!(
        "UNIFIED PAIRS        {:.4} (+/-{:.4})   {:.4} (+/-{:.4})",
        mean(&unified_bind_equidist),
        std(&unified_bind_equidist),
        mean(&unified_bundle_equidist),
        std(&unified_bundle_equidist)
    );
    println!(
        "SEPARATE PAIRS       {:.4} (+/-{:.4})   {:.4} (+/-{:.4})\n",
        mean(&separate_bind_equidist),
        std(&separate_bind_equidist),
        mean(&separate_bundle_equidist),
        std(&separate_bundle_equidist)
    );

    println!("================================================================");
    println!("   RESULTS: NOVELTY ANALYSIS");
    println!("   (Higher = more different from inputs)");
    println!("================================================================\n");

    println!("                      BIND (XOR)       BUNDLE (MAJ)");
    println!("                    -------------     -------------");
    println!(
        "UNIFIED PAIRS        {:.4} (+/-{:.4})   {:.4} (+/-{:.4})",
        mean(&unified_bind_novelty),
        std(&unified_bind_novelty),
        mean(&unified_bundle_novelty),
        std(&unified_bundle_novelty)
    );
    println!(
        "SEPARATE PAIRS       {:.4} (+/-{:.4})   {:.4} (+/-{:.4})\n",
        mean(&separate_bind_novelty),
        std(&separate_bind_novelty),
        mean(&separate_bundle_novelty),
        std(&separate_bundle_novelty)
    );

    // Key comparisons
    println!("================================================================");
    println!("   KEY COMPARISONS");
    println!("================================================================\n");

    let bind_more_equidistant = mean(&unified_bind_equidist) < mean(&unified_bundle_equidist);
    let bind_more_novel = mean(&unified_bind_novelty) > mean(&unified_bundle_novelty);

    println!("1. Is binding more equidistant than bundling?");
    println!(
        "   Unified:  Bind={:.4} vs Bundle={:.4} -> {}",
        mean(&unified_bind_equidist),
        mean(&unified_bundle_equidist),
        if bind_more_equidistant {
            "YES - Binding more equidistant"
        } else {
            "NO"
        }
    );

    println!("\n2. Is binding more novel (different from inputs)?");
    println!(
        "   Unified:  Bind={:.4} vs Bundle={:.4} -> {}",
        mean(&unified_bind_novelty),
        mean(&unified_bundle_novelty),
        if bind_more_novel {
            "YES - Binding creates more novel vectors"
        } else {
            "NO"
        }
    );

    // Interaction effect for equidistance
    let bind_equidist_diff = mean(&unified_bind_equidist) - mean(&separate_bind_equidist);
    let bundle_equidist_diff = mean(&unified_bundle_equidist) - mean(&separate_bundle_equidist);
    let interaction_equidist = bind_equidist_diff - bundle_equidist_diff;

    println!("\n3. Interaction effect (equidistance):");
    println!("   Does binding specifically help unified pairs?");
    println!("   Interaction = {:.4}", interaction_equidist);

    // Permutation test
    println!("\nRunning permutation test for interaction...");
    let n_permutations = 10000;
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let all_bind_equidist: Vec<f64> = unified_bind_equidist
        .iter()
        .chain(separate_bind_equidist.iter())
        .copied()
        .collect();
    let all_bundle_equidist: Vec<f64> = unified_bundle_equidist
        .iter()
        .chain(separate_bundle_equidist.iter())
        .copied()
        .collect();
    let n_unified = unified_pairs.len();

    let mut extreme_count = 0;
    for _ in 0..n_permutations {
        let mut indices: Vec<usize> = (0..all_bind_equidist.len()).collect();
        indices.shuffle(&mut rng);

        let perm_unified_bind: Vec<f64> = indices[..n_unified]
            .iter()
            .map(|&i| all_bind_equidist[i])
            .collect();
        let perm_unified_bundle: Vec<f64> = indices[..n_unified]
            .iter()
            .map(|&i| all_bundle_equidist[i])
            .collect();
        let perm_separate_bind: Vec<f64> = indices[n_unified..]
            .iter()
            .map(|&i| all_bind_equidist[i])
            .collect();
        let perm_separate_bundle: Vec<f64> = indices[n_unified..]
            .iter()
            .map(|&i| all_bundle_equidist[i])
            .collect();

        let perm_bind_diff = mean(&perm_unified_bind) - mean(&perm_separate_bind);
        let perm_bundle_diff = mean(&perm_unified_bundle) - mean(&perm_separate_bundle);
        let perm_interaction = perm_bind_diff - perm_bundle_diff;

        if perm_interaction.abs() >= interaction_equidist.abs() {
            extreme_count += 1;
        }
    }

    let p_value = extreme_count as f64 / n_permutations as f64;
    println!("  p-value: {:.4}", p_value);
    println!("  Significant (p < 0.05): {}", p_value < 0.05);

    // Sample detailed results
    println!("\n================================================================");
    println!("   SAMPLE PAIR DETAILS");
    println!("================================================================\n");

    println!("Unified Pairs (first 5):");
    println!(
        "  {:20} | Bind equidist | Bundle equidist | Bind novelty | Bundle novelty",
        "Pair"
    );
    println!(
        "  {:20} | ------------- | --------------- | ------------ | --------------",
        "----"
    );
    for i in 0..5.min(unified_pairs.len()) {
        let (a, b) = unified_pairs[i];
        let m = &unified_metrics[i];
        println!(
            "  {:20} | {:.4}        | {:.4}          | {:.4}       | {:.4}",
            format!("({}, {})", a, b),
            m.bind_equidistance,
            m.bundle_equidistance,
            m.bind_novelty,
            m.bundle_novelty
        );
    }

    println!("\nSeparate Pairs (first 5):");
    println!(
        "  {:20} | Bind equidist | Bundle equidist | Bind novelty | Bundle novelty",
        "Pair"
    );
    println!(
        "  {:20} | ------------- | --------------- | ------------ | --------------",
        "----"
    );
    for i in 0..5.min(separate_pairs.len()) {
        let (a, b) = separate_pairs[i];
        let m = &separate_metrics[i];
        println!(
            "  {:20} | {:.4}        | {:.4}          | {:.4}       | {:.4}",
            format!("({}, {})", a, b),
            m.bind_equidistance,
            m.bundle_equidistance,
            m.bind_novelty,
            m.bundle_novelty
        );
    }

    println!("\n================================================================");
    println!("   H2 VECTOR METRICS EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}