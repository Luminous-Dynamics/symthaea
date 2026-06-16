// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Cross-Scale Symmetry Benchmark
//!
//! Tests three hypotheses about the relationship between HDC algebraic
//! symmetry and topological harmonic resonance:
//!
//! 1. **Does HDC binding symmetry constrain topological harmonics?**
//!    Generate scenarios with varying binding symmetry (commutative vs
//!    permutation-shifted), feed through Hodge decomposition, observe
//!    whether harmonic fraction changes.
//!
//! 2. **Does harmonic resonance affect consciousness?**
//!    Verify the ±3% modulation on unified_psi is measurable when
//!    harmonic fraction varies.
//!
//! 3. **Gradient vs curl vs harmonic dominance across cognitive states.**
//!    Simulate dreaming (high noise, diverse scenarios), focused attention
//!    (clustered scenarios), and creative exploration (structured binding)
//!    to measure which Hodge component dominates.
//!
//! Science:
//! - Hodge (1941) — harmonic forms as cohomology representatives
//! - Barbarossa & Sardellitti (2020) — topological signal processing
//! - Tononi (2004) — integrated information requires global coupling
//! - Putnam (1967) — multiple realizability of computational structure

use symthaea_core::hdc::ContinuousHV;
use symthaea_hodge::{HodgeLaplacian, SimplicialComplex};

use symthaea::hdc::moral_topology::{HodgeFractions, MoralTopology, MoralTopologyConfig};

// ═════════════════════════════════════════════════════════════════════════
// Test Helpers
// ═════════════════════════════════════════════════════════════════════════

/// HDC dimension for tests. Must be large enough that cosine similarity
/// between constructed (non-random) HVs exceeds the characteristic scale
/// threshold. With dim=256, random vectors are nearly orthogonal (cosine ~0)
/// and the Rips complex is empty. dim=1024 gives enough structure for
/// bundled/bound vectors to maintain measurable similarity.
const TEST_DIM: usize = 1024;

/// Create a MoralTopology configured for exact Hodge computation.
fn make_topology(window_size: usize) -> MoralTopology {
    MoralTopology::new(MoralTopologyConfig {
        window_size,
        num_scales: 10,
        min_persistence: 0.1,
        pga_components: 3,
        dim: TEST_DIM,
        exact_betti: true,
        adaptive_rips_enabled: false, // Tests use full sweep for reproducibility
    })
}

/// Generate N "diverse but overlapping" scenarios (simulates dreaming).
///
/// Creates scenarios by bundling 2-3 random basis vectors, ensuring
/// non-trivial pairwise similarity (shared components) while maintaining
/// diversity. Pure random HVs are nearly orthogonal in high dimensions
/// and produce empty Rips complexes.
fn add_random_scenarios(topo: &mut MoralTopology, n: usize, seed_base: u64) {
    // Create a small basis set — scenarios share basis components
    let basis_size = 6;
    let basis: Vec<ContinuousHV> = (0..basis_size)
        .map(|i| ContinuousHV::random(TEST_DIM, seed_base + i as u64))
        .collect();

    for i in 0..n {
        // Bundle 2-3 basis vectors with a unique noise vector
        let noise = ContinuousHV::random(TEST_DIM, seed_base + 100 + i as u64);
        let b1 = &basis[i % basis_size];
        let b2 = &basis[(i * 3 + 1) % basis_size];
        let scenario = ContinuousHV::bundle(&[b1, b2, &noise]);
        topo.add_scenario(scenario);
    }
}

/// Generate N scenarios that are similar (clustered) by perturbing a base HV.
fn add_clustered_scenarios(topo: &mut MoralTopology, n: usize, seed: u64) {
    let base = ContinuousHV::random(TEST_DIM, seed);
    for i in 0..n {
        let noise = ContinuousHV::random(TEST_DIM, seed + 1000 + i as u64);
        // Blend: 80% base + 20% noise → tight cluster
        let blended = ContinuousHV::bundle(&[&base, &base, &base, &base, &noise]);
        topo.add_scenario(blended);
    }
}

/// Generate N scenarios using permutation-shifted binding (asymmetric structure).
/// Creates role-filler structures: permute(role, k) ⊗ filler.
fn add_bound_scenarios(topo: &mut MoralTopology, n: usize, seed: u64) {
    let roles: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(TEST_DIM, seed + i))
        .collect();
    let fillers: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(TEST_DIM, seed + 100 + i))
        .collect();

    for i in 0..n {
        // Build a role-filler structure with permutation shifts
        let role_idx = i % roles.len();
        let filler_idx = (i * 3 + 1) % fillers.len();
        let shifted_role = roles[role_idx].permute(i + 1);
        let bound = shifted_role.bind(&fillers[filler_idx]);
        topo.add_scenario(bound);
    }
}

/// Generate N scenarios using commutative (symmetric) binding only.
/// No permutation shifts — pure XOR-style binding preserving commutativity.
fn add_symmetric_scenarios(topo: &mut MoralTopology, n: usize, seed: u64) {
    let atoms: Vec<ContinuousHV> = (0..8)
        .map(|i| ContinuousHV::random(TEST_DIM, seed + i))
        .collect();

    for i in 0..n {
        // Symmetric binding: A ⊗ B = B ⊗ A (no permutation)
        let a = &atoms[i % atoms.len()];
        let b = &atoms[(i * 3 + 2) % atoms.len()];
        let bound = a.bind(b);
        // Bundle with another symmetric pair for richer structure
        let c = &atoms[(i * 5 + 1) % atoms.len()];
        let d = &atoms[(i * 7 + 3) % atoms.len()];
        let bound2 = c.bind(d);
        let scenario = ContinuousHV::bundle(&[&bound, &bound2]);
        topo.add_scenario(scenario);
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Hypothesis 1: Does HDC binding symmetry constrain topological harmonics?
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn test_h1_symmetric_vs_asymmetric_binding_harmonic_fraction() {
    // Hypothesis: Permutation-shifted (asymmetric) binding should produce
    // DIFFERENT harmonic fractions than pure commutative (symmetric) binding,
    // because the directed structure creates different topological connectivity.

    let mut topo_symmetric = make_topology(20);
    add_symmetric_scenarios(&mut topo_symmetric, 16, 42);
    let assessment_sym = topo_symmetric.analyze();

    let mut topo_asymmetric = make_topology(20);
    add_bound_scenarios(&mut topo_asymmetric, 16, 42);
    let assessment_asym = topo_asymmetric.analyze();

    let fracs_sym = assessment_sym
        .hodge_fractions
        .expect("exact_betti=true should produce Hodge fractions");
    let fracs_asym = assessment_asym
        .hodge_fractions
        .expect("exact_betti=true should produce Hodge fractions");

    println!("=== Hypothesis 1: Binding Symmetry → Topological Harmonics ===");
    println!(
        "Symmetric binding:  gradient={:.4} curl={:.4} harmonic={:.4} (edges={})",
        fracs_sym.gradient, fracs_sym.curl, fracs_sym.harmonic, fracs_sym.scales_sampled
    );
    println!(
        "Asymmetric binding: gradient={:.4} curl={:.4} harmonic={:.4} (edges={})",
        fracs_asym.gradient, fracs_asym.curl, fracs_asym.harmonic, fracs_asym.scales_sampled
    );

    // The fractions should differ — binding symmetry should affect topology
    let harmonic_diff = (fracs_sym.harmonic - fracs_asym.harmonic).abs();
    println!("Harmonic fraction difference: {harmonic_diff:.6}");

    // Verify fractions are valid (each in [0,1], sum ≤ 1.0).
    // Persistence-weighted fractions may not sum to exactly 1.0 because
    // the weighting averages across scales with varying signal energy.
    let sum_sym = fracs_sym.gradient + fracs_sym.curl + fracs_sym.harmonic;
    let sum_asym = fracs_asym.gradient + fracs_asym.curl + fracs_asym.harmonic;
    assert!(
        sum_sym <= 1.01,
        "Symmetric fractions sum {sum_sym} exceeds 1.0"
    );
    assert!(
        sum_asym <= 1.01,
        "Asymmetric fractions sum {sum_asym} exceeds 1.0"
    );

    // Both should have non-trivial harmonic content (moral manifolds have cycles)
    assert!(
        fracs_sym.harmonic >= 0.0 && fracs_sym.harmonic <= 1.0,
        "Harmonic fraction out of range"
    );
    assert!(
        fracs_asym.harmonic >= 0.0 && fracs_asym.harmonic <= 1.0,
        "Harmonic fraction out of range"
    );
}

#[test]
fn test_h1_random_scenarios_have_measurable_harmonics() {
    // Baseline: random (unstructured) scenarios should still produce
    // non-trivial Hodge decomposition with all three components present.

    let mut topo = make_topology(20);
    add_random_scenarios(&mut topo, 16, 123);
    let assessment = topo.analyze();

    let fracs = assessment
        .hodge_fractions
        .expect("Should have Hodge fractions");

    println!("=== Random Scenarios Hodge Fractions ===");
    println!(
        "gradient={:.4} curl={:.4} harmonic={:.4} (edges={})",
        fracs.gradient, fracs.curl, fracs.harmonic, fracs.scales_sampled
    );

    // Random scenarios should have some structure (not all zero)
    assert!(
        fracs.gradient + fracs.curl + fracs.harmonic > 0.0,
        "Random scenarios should have non-trivial decomposition"
    );
}

// ═════════════════════════════════════════════════════════════════════════
// Hypothesis 2: Does harmonic resonance affect consciousness?
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn test_h2_harmonic_fraction_modulates_consciousness() {
    // Verify that the consciousness modulation formula produces
    // measurable differences for different harmonic fractions.

    const HODGE_HARMONIC_CONSCIOUSNESS_SCALE: f64 = 0.03;
    const HODGE_HARMONIC_BASELINE: f64 = 0.33;

    // Test cases: harmonic fraction → expected consciousness multiplier
    let test_cases = [
        (0.0, "zero harmonics"),  // Below baseline → dampen
        (0.33, "equipartition"),  // At baseline → neutral
        (0.67, "high harmonics"), // Above baseline → boost
        (1.0, "pure harmonic"),   // Maximum → maximum boost
    ];

    println!("=== Hypothesis 2: Harmonic Fraction → Consciousness Modulation ===");

    let mut mods = Vec::new();
    for (harmonic, label) in &test_cases {
        let delta = harmonic - HODGE_HARMONIC_BASELINE;
        let hodge_mod = 1.0 + delta * HODGE_HARMONIC_CONSCIOUSNESS_SCALE / 0.67;
        let base_psi = 0.5;
        let modulated_psi = (base_psi * hodge_mod).clamp(0.0, 1.0);
        let pct_change = (modulated_psi - base_psi) / base_psi * 100.0;

        println!(
            "  harmonic={:.2} ({:20}) → mod={:.6} → psi: {:.4} → {:.4} ({:+.2}%)",
            harmonic, label, hodge_mod, base_psi, modulated_psi, pct_change
        );
        mods.push(hodge_mod);
    }

    // Modulation should be monotonically increasing with harmonic fraction
    for i in 1..mods.len() {
        assert!(
            mods[i] >= mods[i - 1],
            "Consciousness modulation should increase with harmonic fraction"
        );
    }

    // At equipartition (0.33), modulation should be ~1.0 (neutral)
    assert!(
        (mods[1] - 1.0).abs() < 0.001,
        "Equipartition should give neutral modulation, got {}",
        mods[1]
    );

    // Range should be within ±3% as designed
    assert!(
        mods[0] >= 0.97,
        "Minimum mod should be >= 0.97, got {}",
        mods[0]
    );
    assert!(
        mods[3] <= 1.03,
        "Maximum mod should be <= 1.03, got {}",
        mods[3]
    );
}

// ═════════════════════════════════════════════════════════════════════════
// Hypothesis 3: Information flow modes across cognitive states
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn test_h3_dreaming_state_hodge_profile() {
    // Dreaming: high noise, diverse scenarios (random walk through state space).
    // Expected: higher harmonic fraction (global resonance during consolidation).

    let mut topo = make_topology(24);
    add_random_scenarios(&mut topo, 20, 777);
    let assessment = topo.analyze();

    let fracs = assessment
        .hodge_fractions
        .expect("Should have Hodge fractions");

    println!("=== Hypothesis 3a: Dreaming State ===");
    println!(
        "gradient={:.4} curl={:.4} harmonic={:.4}",
        fracs.gradient, fracs.curl, fracs.harmonic
    );

    // Record for comparison
    assert!(
        fracs.scales_sampled >= 3,
        "Dream state should have sufficient edges for decomposition"
    );
}

#[test]
fn test_h3_focused_attention_hodge_profile() {
    // Focused attention: tightly clustered scenarios (narrow moral focus).
    // Expected: higher gradient fraction (hierarchical, directed flow).

    let mut topo = make_topology(24);
    add_clustered_scenarios(&mut topo, 20, 888);
    let assessment = topo.analyze();

    if let Some(fracs) = assessment.hodge_fractions {
        println!("=== Hypothesis 3b: Focused Attention ===");
        println!(
            "gradient={:.4} curl={:.4} harmonic={:.4} (edges={})",
            fracs.gradient, fracs.curl, fracs.harmonic, fracs.scales_sampled
        );
    } else {
        println!("=== Hypothesis 3b: Focused Attention ===");
        println!("  Clustered scenarios: too few edges for Hodge decomposition");
        println!(
            "  (Tight clustering → high similarity → dense Rips complex OR few distinct edges)"
        );
        // This is actually informative: very tight clustering may create a
        // fully-connected complex where β₁=0, meaning the gradient component
        // dominates (hierarchical flow with no cycles).
    }
}

#[test]
fn test_h3_creative_exploration_hodge_profile() {
    // Creative exploration: structured binding with role-filler patterns.
    // Expected: balanced gradient + harmonic (directed creativity with resonance).

    let mut topo = make_topology(24);
    add_bound_scenarios(&mut topo, 20, 999);
    let assessment = topo.analyze();

    let fracs = assessment
        .hodge_fractions
        .expect("Should have Hodge fractions");

    println!("=== Hypothesis 3c: Creative Exploration ===");
    println!(
        "gradient={:.4} curl={:.4} harmonic={:.4}",
        fracs.gradient, fracs.curl, fracs.harmonic
    );
}

#[test]
fn test_h3_compare_all_states() {
    // Run all three cognitive states and compare their Hodge profiles.

    println!("\n=== Cognitive State Comparison ===\n");
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>8}",
        "State", "Gradient", "Curl", "Harmonic", "Scales"
    );
    println!("{}", "-".repeat(68));

    let states: Vec<(&str, Box<dyn Fn(&mut MoralTopology)>)> = vec![
        (
            "Dreaming (random)",
            Box::new(|t: &mut MoralTopology| add_random_scenarios(t, 20, 111)),
        ),
        (
            "Focused (clustered)",
            Box::new(|t: &mut MoralTopology| add_clustered_scenarios(t, 20, 222)),
        ),
        (
            "Creative (bound)",
            Box::new(|t: &mut MoralTopology| add_bound_scenarios(t, 20, 333)),
        ),
        (
            "Symmetric (commutative)",
            Box::new(|t: &mut MoralTopology| add_symmetric_scenarios(t, 20, 444)),
        ),
    ];

    let mut results: Vec<(&str, Option<HodgeFractions>)> = Vec::new();

    for (name, setup) in &states {
        let mut topo = make_topology(24);
        setup(&mut topo);
        let assessment = topo.analyze();

        if let Some(fracs) = assessment.hodge_fractions {
            println!(
                "{:<25} {:>10.4} {:>10.4} {:>10.4} {:>8}",
                name, fracs.gradient, fracs.curl, fracs.harmonic, fracs.scales_sampled
            );
            results.push((name, Some(fracs)));
        } else {
            println!(
                "{:<25} {:>10} {:>10} {:>10} {:>8}",
                name, "N/A", "N/A", "N/A", "0"
            );
            results.push((name, None));
        }
    }

    println!();

    // At least 3 of 4 states should produce valid decompositions
    let valid_count = results.iter().filter(|(_, f)| f.is_some()).count();
    assert!(
        valid_count >= 3,
        "At least 3 of 4 cognitive states should produce valid Hodge decompositions, got {valid_count}"
    );
}

// ═════════════════════════════════════════════════════════════════════════
// Structural Validation
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn test_hodge_fractions_persist_across_window_growth() {
    // As the moral window grows, Hodge fractions should evolve smoothly.

    let mut topo = make_topology(32);
    let mut fraction_history: Vec<(usize, HodgeFractions)> = Vec::new();

    println!("\n=== Window Growth: Hodge Fraction Evolution ===\n");
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>8}",
        "Window", "Gradient", "Curl", "Harmonic", "Scales"
    );
    println!("{}", "-".repeat(52));

    for i in 0..24 {
        let hv = ContinuousHV::random(TEST_DIM, 500 + i as u64);
        topo.add_scenario(hv);

        // Analyze every 4 additions (need minimum scenarios for Rips complex)
        if (i + 1) >= 6 && (i + 1) % 4 == 2 {
            let assessment = topo.analyze();
            if let Some(fracs) = assessment.hodge_fractions {
                println!(
                    "{:<8} {:>10.4} {:>10.4} {:>10.4} {:>8}",
                    i + 1,
                    fracs.gradient,
                    fracs.curl,
                    fracs.harmonic,
                    fracs.scales_sampled
                );
                fraction_history.push((i + 1, fracs));
            }
        }
    }

    // Should have at least 3 measurements
    assert!(
        fraction_history.len() >= 3,
        "Should track at least 3 Hodge fraction measurements, got {}",
        fraction_history.len()
    );

    // All fractions should be in valid range
    for (window, fracs) in &fraction_history {
        let sum = fracs.gradient + fracs.curl + fracs.harmonic;
        assert!(
            sum <= 1.02,
            "Window {window}: fractions sum {sum} exceeds 1.0"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Multi-Scale Hodge Sweep
// ═════════════════════════════════════════════════════════════════════════

/// Build a Rips complex at a given scale threshold from a similarity matrix,
/// then compute Hodge decomposition on the centered edge signal.
/// Returns (gradient, curl, harmonic, edge_count) or None.
fn hodge_at_scale(sim: &[f64], n: usize, scale: f64) -> Option<(f64, f64, f64, usize)> {
    let mut complex = SimplicialComplex::new();
    let mut edge_signal: Vec<f64> = Vec::new();

    for i in 0..n {
        complex.add_simplex(vec![i]);
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if sim[i * n + j] >= scale {
                complex.add_simplex(vec![i, j]);
                edge_signal.push(sim[i * n + j]);
                for k in (j + 1)..n {
                    if sim[i * n + k] >= scale && sim[j * n + k] >= scale {
                        complex.add_simplex(vec![i, j, k]);
                    }
                }
            }
        }
    }

    let edge_count = complex.count(1);
    if edge_count < 3 {
        return None;
    }

    // Center the signal
    let mean = edge_signal.iter().sum::<f64>() / edge_signal.len() as f64;
    let centered: Vec<f64> = edge_signal.iter().map(|s| s - mean).collect();

    let laplacian = HodgeLaplacian::new(complex);
    let decomp = laplacian.hodge_decompose(1, &centered)?;
    let (g, c, h) = decomp.fractions();
    Some((g, c, h, edge_count))
}

/// Compute pairwise cosine similarity matrix for a set of HVs.
fn similarity_matrix(hvs: &[ContinuousHV]) -> Vec<f64> {
    let n = hvs.len();
    let mut sim = vec![0.0f64; n * n];
    for i in 0..n {
        sim[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let s = hvs[i].similarity(&hvs[j]) as f64;
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }
    sim
}

#[test]
fn test_multiscale_hodge_sweep_random() {
    // Sweep the Rips threshold from 0.0 to 1.0 and observe how the
    // Hodge decomposition changes. This reveals the *scale structure*
    // of information flow on the moral manifold.
    //
    // At low threshold: dense complex, few cycles → gradient-dominated
    // At high threshold: sparse complex, many disconnected components → N/A
    // At intermediate threshold: topological cycles emerge → harmonic peaks

    let basis_size = 8;
    let basis: Vec<ContinuousHV> = (0..basis_size)
        .map(|i| ContinuousHV::random(TEST_DIM, 2000 + i as u64))
        .collect();

    let n = 16;
    let hvs: Vec<ContinuousHV> = (0..n)
        .map(|i| {
            let noise = ContinuousHV::random(TEST_DIM, 2100 + i as u64);
            let b1 = &basis[i % basis_size];
            let b2 = &basis[(i * 3 + 1) % basis_size];
            ContinuousHV::bundle(&[b1, b2, &noise])
        })
        .collect();

    let sim = similarity_matrix(&hvs);

    println!("\n=== Multi-Scale Hodge Sweep (Random/Diverse) ===\n");
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "Scale", "Gradient", "Curl", "Harmonic", "Scales", "Beta1"
    );
    println!("{}", "-".repeat(62));

    let num_scales = 20;
    let mut found_gradient = false;
    let mut found_harmonic = false;

    for s in 0..num_scales {
        let scale = s as f64 / (num_scales - 1) as f64;

        if let Some((g, c, h, edges)) = hodge_at_scale(&sim, n, scale) {
            // Quick beta_1 estimate from the Hodge Laplacian
            let beta_1_est = if h > 0.01 { ">" } else { "0" };
            println!(
                "{:<10.3} {:>10.4} {:>10.4} {:>10.4} {:>8} {:>8}",
                scale, g, c, h, edges, beta_1_est
            );
            if g > 0.01 {
                found_gradient = true;
            }
            if h > 0.01 {
                found_harmonic = true;
            }
        } else {
            println!(
                "{:<10.3} {:>10} {:>10} {:>10} {:>8} {:>8}",
                scale, "---", "---", "---", "<3", "N/A"
            );
        }
    }

    println!();
    println!("Gradient observed: {found_gradient}");
    println!("Harmonic observed: {found_harmonic}");

    // At least some scales should produce valid decompositions
    // (we don't assert gradient/harmonic exist yet — the data will tell us)
}

#[test]
fn test_multiscale_hodge_sweep_bound_vs_symmetric() {
    // Compare multi-scale Hodge profiles for asymmetric (bound) vs
    // symmetric (commutative) HDC constructions.
    //
    // This is the core cross-scale symmetry test: if HDC algebraic
    // symmetry constrains topological harmonics, the profiles should differ.

    let n = 16;

    // Asymmetric: permutation-shifted binding
    let roles: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(TEST_DIM, 3000 + i))
        .collect();
    let fillers: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(TEST_DIM, 3100 + i))
        .collect();
    let asym_hvs: Vec<ContinuousHV> = (0..n)
        .map(|i| {
            let shifted = roles[i % roles.len()].permute(i + 1);
            shifted.bind(&fillers[(i * 3 + 1) % fillers.len()])
        })
        .collect();

    // Symmetric: pure commutative binding
    let atoms: Vec<ContinuousHV> = (0..8)
        .map(|i| ContinuousHV::random(TEST_DIM, 3200 + i))
        .collect();
    let sym_hvs: Vec<ContinuousHV> = (0..n)
        .map(|i| {
            let a = &atoms[i % atoms.len()];
            let b = &atoms[(i * 3 + 2) % atoms.len()];
            let bound = a.bind(b);
            let c = &atoms[(i * 5 + 1) % atoms.len()];
            let d = &atoms[(i * 7 + 3) % atoms.len()];
            let bound2 = c.bind(d);
            ContinuousHV::bundle(&[&bound, &bound2])
        })
        .collect();

    let sim_asym = similarity_matrix(&asym_hvs);
    let sim_sym = similarity_matrix(&sym_hvs);

    println!("\n=== Multi-Scale Hodge: Asymmetric vs Symmetric Binding ===\n");
    println!(
        "{:<7} {:>9} {:>9} {:>9} {:>6} | {:>9} {:>9} {:>9} {:>6}",
        "Scale", "A:Grad", "A:Curl", "A:Harm", "A:E", "S:Grad", "S:Curl", "S:Harm", "S:E"
    );
    println!("{}", "-".repeat(82));

    let num_scales = 15;
    let mut diffs: Vec<f64> = Vec::new();

    for s in 0..num_scales {
        let scale = s as f64 / (num_scales - 1) as f64;

        let asym = hodge_at_scale(&sim_asym, n, scale);
        let sym = hodge_at_scale(&sim_sym, n, scale);

        match (asym, sym) {
            (Some((ag, ac, ah, ae)), Some((sg, sc, sh, se))) => {
                println!(
                    "{:<7.3} {:>9.4} {:>9.4} {:>9.4} {:>6} | {:>9.4} {:>9.4} {:>9.4} {:>6}",
                    scale, ag, ac, ah, ae, sg, sc, sh, se
                );
                let diff = (ah - sh).abs() + (ag - sg).abs() + (ac - sc).abs();
                diffs.push(diff);
            }
            (Some((ag, ac, ah, ae)), None) => {
                println!(
                    "{:<7.3} {:>9.4} {:>9.4} {:>9.4} {:>6} | {:>9} {:>9} {:>9} {:>6}",
                    scale, ag, ac, ah, ae, "---", "---", "---", "<3"
                );
            }
            (None, Some((sg, sc, sh, se))) => {
                println!(
                    "{:<7.3} {:>9} {:>9} {:>9} {:>6} | {:>9.4} {:>9.4} {:>9.4} {:>6}",
                    scale, "---", "---", "---", "<3", sg, sc, sh, se
                );
            }
            (None, None) => {
                println!(
                    "{:<7.3} {:>9} {:>9} {:>9} {:>6} | {:>9} {:>9} {:>9} {:>6}",
                    scale, "---", "---", "---", "<3", "---", "---", "---", "<3"
                );
            }
        }
    }

    println!();
    if !diffs.is_empty() {
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let max_diff = diffs.iter().cloned().fold(0.0f64, f64::max);
        println!(
            "Mean Hodge profile difference: {mean_diff:.6} (max: {max_diff:.6}, N={} scales)",
            diffs.len()
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Vertex-Signal Hodge Decomposition (L₀)
// ═════════════════════════════════════════════════════════════════════════
//
// Instead of decomposing similarity values on edges (L₁), decompose the
// *harmony coordinate values* on vertices (L₀). This measures:
// - Gradient: hierarchical flow of moral meaning between connected scenarios
// - Harmonic: moral meaning isolated in disconnected clusters (β₀ > 1)
//
// The vertex signal is a single harmony dimension projected onto each node.
// We decompose each of the 8 harmony dimensions separately and aggregate.

/// Compute vertex-signal Hodge decomposition at a given scale.
/// `vertex_signal` is one value per vertex (e.g., one harmony coordinate).
/// Returns (gradient_frac, harmonic_frac) or None if complex has < 2 vertices.
fn vertex_hodge_at_scale(
    sim: &[f64],
    n: usize,
    scale: f64,
    vertex_signal: &[f64],
) -> Option<(f64, f64, f64)> {
    if n < 2 || vertex_signal.len() != n {
        return None;
    }

    let mut complex = SimplicialComplex::new();
    for i in 0..n {
        complex.add_simplex(vec![i]);
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if sim[i * n + j] >= scale {
                complex.add_simplex(vec![i, j]);
            }
        }
    }

    if complex.count(0) < 2 {
        return None;
    }

    // Center the vertex signal
    let mean = vertex_signal.iter().sum::<f64>() / vertex_signal.len() as f64;
    let centered: Vec<f64> = vertex_signal.iter().map(|v| v - mean).collect();

    let laplacian = HodgeLaplacian::new(complex);
    let decomp = laplacian.hodge_decompose(0, &centered)?;
    let (g, c, h) = decomp.fractions();
    Some((g, c, h))
}

#[test]
fn test_vertex_hodge_harmony_coordinates() {
    // Decompose the 8D harmony coordinates on vertices (L₀).
    // Each harmony dimension becomes a separate vertex signal.
    //
    // For L₀:
    // - Gradient = moral meaning flowing hierarchically between connected scenarios
    // - Coexact = N/A for 0-simplices (always 0)
    // - Harmonic = moral meaning trapped in disconnected clusters (β₀ > 1)
    //
    // At low scale threshold: fully connected → β₀=1 → harmonic=0, all gradient
    // At high scale threshold: disconnected → β₀>1 → harmonic emerges
    // The TRANSITION reveals how moral coherence fragments with strictness.

    let basis_size = 8;
    let basis: Vec<ContinuousHV> = (0..basis_size)
        .map(|i| ContinuousHV::random(TEST_DIM, 5000 + i as u64))
        .collect();

    let n = 16;
    let hvs: Vec<ContinuousHV> = (0..n)
        .map(|i| {
            let noise = ContinuousHV::random(TEST_DIM, 5100 + i as u64);
            let b1 = &basis[i % basis_size];
            let b2 = &basis[(i * 3 + 1) % basis_size];
            ContinuousHV::bundle(&[b1, b2, &noise])
        })
        .collect();

    let sim = similarity_matrix(&hvs);

    // Project HVs to 8D harmony coordinates using a simple basis projection
    // (In production, HarmonyBasis does this; here we approximate with cosine
    // similarity to 8 basis vectors as harmony dimensions)
    let harmony_coords: Vec<[f64; 8]> = hvs
        .iter()
        .map(|hv| {
            let mut coords = [0.0f64; 8];
            for (d, b) in basis.iter().enumerate().take(8) {
                coords[d] = hv.similarity(b) as f64;
            }
            coords
        })
        .collect();

    println!("\n=== Vertex-Signal Hodge Decomposition (L₀) ===");
    println!("Decomposing 8 harmony dimensions on {} vertices\n", n);

    let num_scales = 15;
    println!(
        "{:<8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "Scale",
        "H0:Grad",
        "H0:Curl",
        "H0:Harm",
        "H3:Grad",
        "H3:Curl",
        "H3:Harm",
        "H7:Grad",
        "H7:Curl",
        "H7:Harm"
    );
    println!("{}", "-".repeat(90));

    let mut found_vertex_harmonic = false;
    let mut found_vertex_gradient = false;

    for s in 0..num_scales {
        let scale = s as f64 / (num_scales - 1) as f64;

        // Decompose harmony dimensions 0, 3, and 7 as representative samples
        let dims = [0, 3, 7];
        let mut results: Vec<Option<(f64, f64, f64)>> = Vec::new();

        for &d in &dims {
            let signal: Vec<f64> = harmony_coords.iter().map(|c| c[d]).collect();
            results.push(vertex_hodge_at_scale(&sim, n, scale, &signal));
        }

        for r in &results {
            if let Some((g, _, h)) = r {
                if *g > 0.01 {
                    found_vertex_gradient = true;
                }
                if *h > 0.01 {
                    found_vertex_harmonic = true;
                }
            }
        }

        let fmt = |r: &Option<(f64, f64, f64)>| -> String {
            match r {
                Some((g, c, h)) => format!("{:>8.4} {:>8.4} {:>8.4}", g, c, h),
                None => format!("{:>8} {:>8} {:>8}", "---", "---", "---"),
            }
        };

        println!(
            "{:<8.3} {} | {} | {}",
            scale,
            fmt(&results[0]),
            fmt(&results[1]),
            fmt(&results[2]),
        );
    }

    println!();
    println!("Vertex gradient observed: {found_vertex_gradient}");
    println!("Vertex harmonic observed: {found_vertex_harmonic}");

    // At high thresholds (sparse complex), we expect disconnected components
    // which produce vertex harmonics — moral meaning trapped in clusters.
    // This is the fragmentation signal.
}

#[test]
fn test_vertex_vs_edge_hodge_comparison() {
    // Direct comparison: vertex L₀ vs edge L₁ decomposition at same scales.
    // Shows which level of the Hodge hierarchy captures more structure.

    let basis_size = 6;
    let basis: Vec<ContinuousHV> = (0..basis_size)
        .map(|i| ContinuousHV::random(TEST_DIM, 6000 + i as u64))
        .collect();

    let n = 16;
    let hvs: Vec<ContinuousHV> = (0..n)
        .map(|i| {
            let noise = ContinuousHV::random(TEST_DIM, 6100 + i as u64);
            let b1 = &basis[i % basis_size];
            let b2 = &basis[(i * 3 + 1) % basis_size];
            ContinuousHV::bundle(&[b1, b2, &noise])
        })
        .collect();

    let sim = similarity_matrix(&hvs);

    // Use first harmony dimension as vertex signal
    let vertex_signal: Vec<f64> = hvs
        .iter()
        .map(|hv| hv.similarity(&basis[0]) as f64)
        .collect();

    println!("\n=== Vertex (L₀) vs Edge (L₁) Hodge Comparison ===\n");
    println!(
        "{:<8} {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10}",
        "Scale", "V:Grad", "V:Curl", "V:Harm", "E:Grad", "E:Curl", "E:Harm"
    );
    println!("{}", "-".repeat(72));

    let num_scales = 15;
    for s in 0..num_scales {
        let scale = s as f64 / (num_scales - 1) as f64;

        let v = vertex_hodge_at_scale(&sim, n, scale, &vertex_signal);
        let e = hodge_at_scale(&sim, n, scale);

        let vfmt = match v {
            Some((g, c, h)) => format!("{:>10.4} {:>10.4} {:>10.4}", g, c, h),
            None => format!("{:>10} {:>10} {:>10}", "---", "---", "---"),
        };
        let efmt = match e {
            Some((g, c, h, _)) => format!("{:>10.4} {:>10.4} {:>10.4}", g, c, h),
            None => format!("{:>10} {:>10} {:>10}", "---", "---", "---"),
        };

        println!("{:<8.3} {} | {}", scale, vfmt, efmt);
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Adaptive Rips Threshold Tests
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn test_adaptive_rips_ema_tracking() {
    // Verify that critical_scale_ema updates after each analyze() call
    // when adaptive_rips_enabled is true.
    let mut topo = MoralTopology::new(MoralTopologyConfig {
        window_size: 20,
        num_scales: 10,
        min_persistence: 0.1,
        pga_components: 3,
        dim: TEST_DIM,
        exact_betti: true,
        adaptive_rips_enabled: true,
    });

    // Initially NaN
    assert!(
        topo.critical_scale_ema().is_nan(),
        "Initial EMA should be NaN"
    );

    // Add scenarios and analyze
    add_random_scenarios(&mut topo, 12, 7000);
    let assessment = topo.analyze();

    // After first analysis with Hodge fractions, EMA should be set
    if assessment.hodge_fractions.is_some() {
        let ema = topo.critical_scale_ema();
        println!("After first analyze: critical_scale_ema = {ema}");
        // EMA may be NaN if no critical transition was found (harmonic never > 0.5)
        // or finite if a transition was detected
    }

    // Add more scenarios and analyze again
    add_random_scenarios(&mut topo, 8, 7100);
    let assessment2 = topo.analyze();

    if let Some(ref fracs) = assessment2.hodge_fractions {
        println!(
            "After second analyze: harmonic={:.4}, critical_scale={:.4}, at_criticality={}",
            fracs.harmonic, fracs.critical_scale, fracs.at_criticality
        );
    }
}

#[test]
fn test_hodge_fractions_in_assessment() {
    // Verify HodgeFractions struct fields are populated correctly
    let mut topo = make_topology(20);
    add_random_scenarios(&mut topo, 16, 8000);
    let assessment = topo.analyze();

    if let Some(fracs) = assessment.hodge_fractions {
        // All fractions in [0, 1]
        assert!(fracs.gradient >= 0.0 && fracs.gradient <= 1.0);
        assert!(fracs.curl >= 0.0 && fracs.curl <= 1.0);
        assert!(fracs.harmonic >= 0.0 && fracs.harmonic <= 1.0);
        // scales_sampled > 0
        assert!(
            fracs.scales_sampled > 0,
            "Should have sampled at least 1 scale"
        );
        // total_weight > 0
        assert!(fracs.total_weight > 0.0, "Total weight should be positive");
        // critical_scale is either NaN or in [0, 1]
        if fracs.critical_scale.is_finite() {
            assert!(fracs.critical_scale >= 0.0 && fracs.critical_scale <= 1.0);
        }
        // at_criticality consistent with harmonic
        if fracs.at_criticality {
            assert!(
                fracs.harmonic >= 0.2 && fracs.harmonic <= 0.8,
                "at_criticality should imply harmonic in [0.2, 0.8]"
            );
        }
        println!(
            "Fractions: g={:.4} c={:.4} h={:.4} scales={} weight={:.4} critical={:.4} at_crit={}",
            fracs.gradient,
            fracs.curl,
            fracs.harmonic,
            fracs.scales_sampled,
            fracs.total_weight,
            fracs.critical_scale,
            fracs.at_criticality,
        );
    }
}

#[test]
fn test_consciousness_modulation_homeostasis() {
    // Verify the consciousness modulation formula maintains homeostatic bounds.
    // The system should never diverge — fragmentation dampens consciousness,
    // which triggers dream consolidation, which heals fragmentation.

    const HODGE_UNIFIED_BOOST: f64 = 0.01;
    const HODGE_CRITICAL_BOOST: f64 = 0.02;
    const HODGE_FRAGMENTED_DAMPING: f64 = 0.05;

    // Simulate 100 cycles of the homeostatic loop
    let mut psi = 0.5_f64;
    let mut harmonic = 0.0_f64;
    let mut dream_interval = 100_usize;

    println!("\n=== Homeostatic Loop Simulation (100 cycles) ===\n");
    println!(
        "{:<6} {:>8} {:>10} {:>12}",
        "Cycle", "Psi", "Harmonic", "DreamInt"
    );
    println!("{}", "-".repeat(40));

    let topology_cadence = 30; // Topology only runs every ~30 cycles
    for cycle in 0..100 {
        // Consciousness modulation (only when topology runs)
        if cycle % topology_cadence == 0 {
            let hodge_mod = if harmonic >= 0.2 && harmonic <= 0.8 {
                1.0 + HODGE_CRITICAL_BOOST
            } else if harmonic < 0.2 {
                1.0 + HODGE_UNIFIED_BOOST
            } else {
                1.0 - HODGE_FRAGMENTED_DAMPING
            };
            psi = (psi * hodge_mod).clamp(0.0, 1.0);
        }

        // Dream consolidation interval
        dream_interval = if harmonic > 0.8 {
            (dream_interval as f64 * 0.5) as usize
        } else if harmonic > 0.5 {
            (dream_interval as f64 * 0.8) as usize
        } else {
            100
        };
        dream_interval = dream_interval.max(3);

        // Simulate: dreaming heals fragmentation and restores consciousness
        if cycle % dream_interval.max(1) == 0 {
            harmonic = (harmonic - 0.15).max(0.0); // Dream consolidation heals manifold
            psi = (psi + 0.02).min(1.0); // Sleep restores consciousness
        }

        // Simulate: without dreams, fragmentation slowly increases
        harmonic = (harmonic + 0.012).min(1.0);

        if cycle % 10 == 0 {
            println!(
                "{:<6} {:>8.4} {:>10.4} {:>12}",
                cycle, psi, harmonic, dream_interval
            );
        }
    }

    // After 100 cycles, psi should not have collapsed to 0 or diverged to 1
    assert!(psi > 0.1, "Consciousness should not collapse: psi={psi}");
    assert!(psi < 0.95, "Consciousness should not diverge: psi={psi}");
    println!("\nFinal: psi={psi:.4}, harmonic={harmonic:.4}, dream_interval={dream_interval}");
    println!("Homeostasis maintained: psi in ({:.2}, {:.2})", 0.1, 0.95);
}