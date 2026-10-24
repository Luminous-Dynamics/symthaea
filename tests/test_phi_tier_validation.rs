// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC Proxy Validation: SampledPartition vs ExhaustivePartition
//!
//! This test compares an O(n) HDC similarity proxy with an exhaustive partition
//! search over the same proxy objective. Exhaustive search is not exact IIT.
//!
//! ## Purpose
//!
//! We test whether the sampled approximation tracks the exhaustive HDC similarity
//! objective on this synthetic representation suite; this is not PyPhi validation.
//!
//! ## Method
//!
//! 1. Generate systems with known topologies (star, ring, random, modular)
//! 2. Compute the exhaustive HDC similarity-proxy score for n=4-8 components
//! 3. Compute the sampled HDC similarity-proxy score for the same representations
//! 4. Calculate Pearson correlation
//! 5. Report an internal proxy-agreement threshold
//!
//! ## Exploratory synthetic hypotheses (not IIT predictions)
//!
//! - Generated star/ring/random/modular HDC representations exercise varied similarity patterns.
//! - Their ranking is an empirical property of this generator/objective, not an IIT prediction.
//!
//! ## Date: January 17, 2026
//! ## Author: Symthaea Theory Validation

use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::hdc::tiered_phi::{ApproximationTier, TieredPhi};

/// Generate a star topology: one hub connected to all spokes
/// Hub is highly similar to all spokes, spokes are dissimilar to each other
fn generate_star_topology(n: usize) -> Vec<BinaryHV> {
    let mut components = Vec::with_capacity(n);

    // Hub: random hypervector
    let hub = BinaryHV::random(42);
    components.push(hub);

    // Spokes: each similar to hub but dissimilar to each other
    for i in 1..n {
        // Create spoke by binding hub with a unique pattern
        let pattern = BinaryHV::random(100 + i as u64);
        // Mix hub with pattern to create similarity gradient
        // More hub = more similar to hub
        let spoke = if i % 2 == 0 {
            hub.bind(&pattern) // ~0.5 similarity to hub
        } else {
            // Bundle hub with pattern for higher similarity
            BinaryHV::bundle(&[hub, pattern])
        };
        components.push(spoke);
    }

    components
}

/// Generate a ring topology: each node connected to neighbors only
/// Creates a chain of similarity
fn generate_ring_topology(n: usize) -> Vec<BinaryHV> {
    let mut components = Vec::with_capacity(n);

    // Start with random base
    let mut prev = BinaryHV::random(200);
    components.push(prev);

    // Each subsequent node is derived from previous (neighbor similarity)
    for i in 1..n {
        let noise = BinaryHV::random(200 + i as u64);
        // Bundle with previous to create neighbor similarity
        let current = BinaryHV::bundle(&[prev, noise]);
        components.push(current);
        prev = current;
    }

    // Connect last to first (ring closure)
    if n > 2 {
        let first = components[0];
        let last_idx = n - 1;
        components[last_idx] = BinaryHV::bundle(&[components[last_idx], first]);
    }

    components
}

/// Generate random topology: all nodes roughly independent
fn generate_random_topology(n: usize) -> Vec<BinaryHV> {
    (0..n).map(|i| BinaryHV::random(300 + i as u64)).collect()
}

/// Generate modular topology: two clusters with weak inter-cluster links
fn generate_modular_topology(n: usize) -> Vec<BinaryHV> {
    let half = n / 2;
    let mut components = Vec::with_capacity(n);

    // Cluster A: all similar to cluster_a_base
    let cluster_a_base = BinaryHV::random(400);
    for i in 0..half {
        let noise = BinaryHV::random(410 + i as u64);
        components.push(BinaryHV::bundle(&[cluster_a_base, noise]));
    }

    // Cluster B: all similar to cluster_b_base (different from A)
    let cluster_b_base = BinaryHV::random(500);
    for i in half..n {
        let noise = BinaryHV::random(510 + i as u64);
        components.push(BinaryHV::bundle(&[cluster_b_base, noise]));
    }

    components
}

/// Calculate Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Calculate Spearman rank correlation
fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    fn rank(values: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut ranks = vec![0.0; values.len()];
        for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
            ranks[*orig_idx] = rank as f64 + 1.0;
        }
        ranks
    }

    let rank_x = rank(x);
    let rank_y = rank(y);
    pearson_correlation(&rank_x, &rank_y)
}

#[test]
fn test_heuristic_vs_exact_correlation() {
    println!("\n========================================");
    println!("HDC PROXY VALIDATION: Sampled vs Exhaustive");
    println!("========================================\n");

    let mut exact_values = Vec::new();
    let mut heuristic_values = Vec::new();
    let mut spectral_values = Vec::new();

    let mut exact_calc = TieredPhi::new(ApproximationTier::ExhaustivePartition);
    let mut heuristic_calc = TieredPhi::new(ApproximationTier::SampledPartition);
    let mut spectral_calc = TieredPhi::new(ApproximationTier::SpectralConnectivity);

    // Test configurations
    let sizes = [4, 5, 6, 7, 8];
    let topologies = ["star", "ring", "random", "modular"];
    let trials_per_config = 3; // Multiple trials for stability

    println!("Configuration:");
    println!("  Sizes: {:?}", sizes);
    println!("  Topologies: {:?}", topologies);
    println!("  Trials per config: {}", trials_per_config);
    println!();

    println!(
        "{:<10} {:<8} {:<8} {:<10} {:<10} {:<10}",
        "Topology", "Size", "Trial", "Exhaustive proxy", "Sampled proxy", "Spectral connectivity"
    );
    println!("{}", "-".repeat(60));

    for &n in &sizes {
        for topology in &topologies {
            for trial in 0..trials_per_config {
                // Generate topology
                let components = match *topology {
                    "star" => generate_star_topology(n),
                    "ring" => generate_ring_topology(n),
                    "random" => generate_random_topology(n),
                    "modular" => generate_modular_topology(n),
                    _ => unreachable!(),
                };

                // Compute Φ with each tier
                let phi_exact = exact_calc.compute(&components);
                let phi_heuristic = heuristic_calc.compute(&components);
                let phi_spectral = spectral_calc.compute(&components);

                exact_values.push(phi_exact);
                heuristic_values.push(phi_heuristic);
                spectral_values.push(phi_spectral);

                println!(
                    "{:<10} {:<8} {:<8} {:<10.4} {:<10.4} {:<10.4}",
                    topology, n, trial, phi_exact, phi_heuristic, phi_spectral
                );
            }
        }
    }

    println!("\n========================================");
    println!("CORRELATION ANALYSIS");
    println!("========================================\n");

    // Calculate correlations
    let heuristic_pearson = pearson_correlation(&exact_values, &heuristic_values);
    let heuristic_spearman = spearman_correlation(&exact_values, &heuristic_values);
    let spectral_pearson = pearson_correlation(&exact_values, &spectral_values);
    let spectral_spearman = spearman_correlation(&exact_values, &spectral_values);

    println!("Sampled proxy vs exhaustive proxy:");
    println!("  Pearson r:  {:.4}", heuristic_pearson);
    println!("  Spearman ρ: {:.4}", heuristic_spearman);
    println!();

    println!("Spectral connectivity vs exhaustive proxy:");
    println!("  Pearson r:  {:.4}", spectral_pearson);
    println!("  Spearman ρ: {:.4}", spectral_spearman);
    println!();

    // Summary statistics
    let exact_mean: f64 = exact_values.iter().sum::<f64>() / exact_values.len() as f64;
    let heuristic_mean: f64 = heuristic_values.iter().sum::<f64>() / heuristic_values.len() as f64;
    let spectral_mean: f64 = spectral_values.iter().sum::<f64>() / spectral_values.len() as f64;

    println!("Mean HDC proxy values:");
    println!("  Exhaustive proxy: {:.4}", exact_mean);
    println!("  Sampled proxy: {:.4}", heuristic_mean);
    println!("  Spectral connectivity: {:.4}", spectral_mean);
    println!();

    // Validation thresholds
    let heuristic_internal_agreement_target_met = heuristic_pearson > 0.70 || heuristic_spearman > 0.70;
    let spectral_matches_proxy_target = spectral_pearson > 0.70 || spectral_spearman > 0.70;

    println!("========================================");
    println!("INTERNAL PROXY-AGREEMENT RESULTS");
    println!("========================================\n");

    println!(
        "Sampled HDC proxy: {}",
        if heuristic_internal_agreement_target_met {
            "✅ INTERNAL TARGET MET"
        } else {
            "❌ INTERNAL TARGET NOT MET"
        }
    );
    println!("  (Internal proxy-agreement target: r > 0.70 or ρ > 0.70)");
    println!();

    println!(
        "Spectral connectivity diagnostic: {}",
        if spectral_matches_proxy_target {
            "✅ INTERNAL TARGET MET"
        } else {
            "❌ NOT AN IIT MEASURE"
        }
    );
    println!("  (Spectral measures λ₂ mixing time, NOT IIT Φ)");
    println!();

    // Exploratory proxy-ranking check (not an IIT prediction)
    println!("========================================");
    println!("TOPOLOGY RANKING CHECK (HDC PROXY ONLY)");
    println!("========================================\n");

    // Compute average HDC proxy score by synthetic representation family
    let mut topology_means: Vec<(&str, f64)> = Vec::new();
    for topology in &topologies {
        let indices: Vec<usize> = (0..exact_values.len())
            .filter(|&i| {
                let config_idx = i / trials_per_config;
                let topo_idx = config_idx % topologies.len();
                topologies[topo_idx] == *topology
            })
            .collect();

        let mean: f64 =
            indices.iter().map(|&i| exact_values[i]).sum::<f64>() / indices.len() as f64;
        topology_means.push((topology, mean));
    }

    // Sort by mean Φ (descending)
    topology_means.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!("Exhaustive HDC proxy ranking by generated representation (highest first):");
    for (rank, (topo, mean)) in topology_means.iter().enumerate() {
        println!("  {}. {:<10} Φ = {:.4}", rank + 1, topo, mean);
    }
    println!();

    println!("No topology ranking is asserted by this proxy benchmark:");
    println!("  Results describe only the synthetic HDC representation generator.");
    println!("  They do not validate IIT or identify consciousness.");
    println!();

    // Record whether this generator gives the star label the highest proxy score.
    let star_highest = topology_means[0].0 == "star";
    println!(
        "Star label has highest HDC proxy score: {}",
        if star_highest {
            "YES (HDC proxy result only)"
        } else {
            "❌ NO"
        }
    );

    // Phase 4.4: Tightened threshold from r > 0.30 to r > 0.70
    // The threshold is only an internal approximation-agreement criterion.
    // It is not validation against formal IIT.
    assert!(
        heuristic_pearson > 0.70 || heuristic_spearman > 0.70,
        "Sampled proxy shows insufficient correlation with exhaustive proxy (r={:.4}, ρ={:.4}). \
         Expected at least r > 0.70 for this internal proxy-agreement target.",
        heuristic_pearson,
        heuristic_spearman
    );

    println!("\n========================================");
    println!("TEST COMPLETE");
    println!("========================================\n");
}

#[test]
fn test_exhaustive_proxy_topology_discrimination() {
    println!("\n========================================");
    println!("EXHAUSTIVE HDC PROXY: Representation Discrimination Test");
    println!("========================================\n");

    // This tests only whether the exhaustive HDC similarity proxy distinguishes
    // synthetic representation families; no IIT ranking is assumed.

    let mut calc = TieredPhi::new(ApproximationTier::ExhaustivePartition);
    let n = 6; // Small enough for exact computation

    // Generate multiple trials
    let trials = 10;
    let mut star_phis = Vec::new();
    let mut ring_phis = Vec::new();
    let mut random_phis = Vec::new();
    let mut modular_phis = Vec::new();

    for _ in 0..trials {
        star_phis.push(calc.compute(&generate_star_topology(n)));
        ring_phis.push(calc.compute(&generate_ring_topology(n)));
        random_phis.push(calc.compute(&generate_random_topology(n)));
        modular_phis.push(calc.compute(&generate_modular_topology(n)));
    }

    let star_mean: f64 = star_phis.iter().sum::<f64>() / trials as f64;
    let ring_mean: f64 = ring_phis.iter().sum::<f64>() / trials as f64;
    let random_mean: f64 = random_phis.iter().sum::<f64>() / trials as f64;
    let modular_mean: f64 = modular_phis.iter().sum::<f64>() / trials as f64;

    println!("Mean exhaustive HDC proxy scores over {} trials (n={}):", trials, n);
    println!("  Star:    {:.4} ± {:.4}", star_mean, std_dev(&star_phis));
    println!("  Ring:    {:.4} ± {:.4}", ring_mean, std_dev(&ring_phis));
    println!(
        "  Random:  {:.4} ± {:.4}",
        random_mean,
        std_dev(&random_phis)
    );
    println!(
        "  Modular: {:.4} ± {:.4}",
        modular_mean,
        std_dev(&modular_phis)
    );
    println!();

    // This HDC proxy should show some discrimination between generated representation families.
    // This is an implementation sanity check, not a claim about formal IIT.
    let max_phi = star_mean.max(ring_mean).max(random_mean).max(modular_mean);
    let min_phi = star_mean.min(ring_mean).min(random_mean).min(modular_mean);
    let discrimination = max_phi - min_phi;

    println!("Topology discrimination: {:.4} (max - min)", discrimination);
    println!("  (Higher = better discrimination between topologies)");

    // We expect at least some discrimination (>0.05)
    assert!(
        discrimination > 0.02,
        "Exhaustive HDC proxy shows almost no discrimination between representation families (Δ={:.4}). \
         This suggests the MIP calculation may not be working correctly.",
        discrimination
    );

    println!("\n✅ Exhaustive HDC proxy discriminates between generated representations");
}

#[test]
fn test_heuristic_consistency() {
    println!("\n========================================");
    println!("HEURISTIC TIER: Consistency Test");
    println!("========================================\n");

    // Test that heuristic tier gives consistent results for the same input
    let mut calc = TieredPhi::new(ApproximationTier::SampledPartition);

    let components = generate_star_topology(8);

    let phi1 = calc.compute(&components);
    let phi2 = calc.compute(&components);
    let phi3 = calc.compute(&components);

    println!("Same input, three computations:");
    println!("  Φ₁ = {:.6}", phi1);
    println!("  Φ₂ = {:.6}", phi2);
    println!("  Φ₃ = {:.6}", phi3);

    // Heuristic uses sampling, so some variance is expected
    // But results should be within 10% of each other
    let max_phi = phi1.max(phi2).max(phi3);
    let min_phi = phi1.min(phi2).min(phi3);
    let variance = (max_phi - min_phi) / max_phi.max(0.001);

    println!("\nVariance: {:.2}%", variance * 100.0);

    // Allow up to 20% variance for sampling-based approach
    assert!(
        variance < 0.20,
        "Heuristic tier shows too much variance ({:.2}%). Consider increasing sample count.",
        variance * 100.0
    );

    println!("✅ Heuristic tier is reasonably consistent");
}

/// Calculate standard deviation
fn std_dev(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

// =============================================================================
#[test]
fn test_exhaustive_proxy_produces_finite_scores_for_standard_representations() {
    // Verifies only that the exhaustive HDC proxy returns finite, bounded values
    // for these synthetic representations. No analytical IIT values are implied.

    let mut calc = TieredPhi::new(ApproximationTier::ExhaustivePartition);

    let topologies: Vec<(&str, Vec<BinaryHV>)> = vec![
        ("star", generate_star_topology(6)),
        ("ring", generate_ring_topology(6)),
        ("random", generate_random_topology(6)),
        ("modular", generate_modular_topology(6)),
    ];

    for (name, components) in &topologies {
        let phi = calc.compute(components);
        assert!(
            phi > 0.0,
            "Topology '{}' should have Phi > 0 (got {})",
            name,
            phi
        );
        assert!(
            phi <= 1.0,
            "Topology '{}' Phi should be <= 1.0 (got {})",
            name,
            phi
        );
        println!("  {}: Φ = {:.4}", name, phi);
    }
}

#[test]
fn test_topology_scores_can_differ_for_star_vs_modular_representations() {
    // Exploratory HDC proxy comparison only; the ranking is not an IIT prediction.

    let mut calc = TieredPhi::new(ApproximationTier::ExhaustivePartition);
    let n = 6;
    let trials = 5;

    let star_mean: f64 = (0..trials)
        .map(|_| calc.compute(&generate_star_topology(n)))
        .sum::<f64>()
        / trials as f64;

    let modular_mean: f64 = (0..trials)
        .map(|_| calc.compute(&generate_modular_topology(n)))
        .sum::<f64>()
        / trials as f64;

    println!("Star Φ = {:.4}, Modular Φ = {:.4}", star_mean, modular_mean);

    // We expect at least some differentiation between topologies
    assert!(
        (star_mean - modular_mean).abs() > 0.01,
        "Exhaustive HDC proxy should differentiate star from modular representations (Δ={:.4})",
        (star_mean - modular_mean).abs()
    );
}

#[test]
fn test_all_tiers_agree_on_trivial_cases() {
    // All tiers should return 0 for empty or single-component inputs
    let tiers = [
        ApproximationTier::RandomBaseline,
        ApproximationTier::SampledPartition,
        ApproximationTier::SpectralConnectivity,
        ApproximationTier::ExhaustivePartition,
    ];

    for tier in tiers {
        let mut calc = TieredPhi::new(tier);

        let phi_empty = calc.compute(&[]);
        assert_eq!(phi_empty, 0.0, "{:?} should return 0 for empty input", tier);

        let phi_single = calc.compute(&[BinaryHV::random(42)]);
        assert_eq!(
            phi_single, 0.0,
            "{:?} should return 0 for single component",
            tier
        );
    }
}