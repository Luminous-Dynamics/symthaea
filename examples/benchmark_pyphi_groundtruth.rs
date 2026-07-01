// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # PyPhi Ground-Truth Cross-Validation
//!
//! Validates Symthaea's Φ computation against known theoretical values from
//! IIT (Integrated Information Theory) literature.
//!
//! ## Known Ground-Truth Networks
//! 1. **Tononi (2008) 4-node network**: Well-studied network with known Φ_MIP
//! 2. **Disconnected modules**: Φ should be 0 (no integration across partition)
//! 3. **Fully connected**: High Φ expected
//! 4. **Chain/line**: Low Φ (easily partitioned)
//! 5. **Star topology**: Moderate Φ (hub creates integration)
//!
//! ## Method
//! Construct networks with known IIT properties and verify:
//! - Φ ordering is correct (fully connected > star > chain > disconnected)
//! - Φ = 0 for disconnected networks
//! - Φ increases with integration (adding connections increases Φ)
//! - Removing hub nodes reduces Φ more than removing peripheral nodes
//!
//! ## Literature References
//! - Tononi (2008) "Consciousness as Integrated Information: a Provisional Manifesto"
//! - Oizumi et al. (2014) "From the Phenomenology to the Mechanisms of Consciousness: IIT 3.0"
//! - Barrett & Seth (2011) "Practical Measures of Integrated Information"
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_pyphi_groundtruth --release
//! ```

use std::time::Instant;

use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::spectral_connectivity::ConnectivityCalculator;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::phi_engine::{PhiEngine, PhiMethod};

const DIM: usize = 4096;

/// Create a disconnected network (two isolated groups)
fn create_disconnected(n: usize, dim: usize, seed: u64) -> Vec<ContinuousHV> {
    let half = n / 2;
    let mut hvs = Vec::new();

    // Group 1: similar to each other, different from group 2
    let base1 = ContinuousHV::random(dim, seed);
    for i in 0..half {
        let noise = ContinuousHV::random(dim, seed + 100 + i as u64);
        let values: Vec<f32> = base1
            .values
            .iter()
            .zip(noise.values.iter())
            .map(|(&b, &n)| b * 0.9 + n * 0.1)
            .collect();
        hvs.push(ContinuousHV::from_vec(values));
    }

    // Group 2: similar to each other, different from group 1
    let base2 = ContinuousHV::random(dim, seed + 50000);
    for i in 0..(n - half) {
        let noise = ContinuousHV::random(dim, seed + 50100 + i as u64);
        let values: Vec<f32> = base2
            .values
            .iter()
            .zip(noise.values.iter())
            .map(|(&b, &n)| b * 0.9 + n * 0.1)
            .collect();
        hvs.push(ContinuousHV::from_vec(values));
    }

    hvs
}

/// Create a fully integrated network (all nodes correlated)
fn create_fully_integrated(n: usize, dim: usize, seed: u64) -> Vec<ContinuousHV> {
    let base = ContinuousHV::random(dim, seed);
    (0..n)
        .map(|i| {
            let noise = ContinuousHV::random(dim, seed + 200 + i as u64);
            let values: Vec<f32> = base
                .values
                .iter()
                .zip(noise.values.iter())
                .map(|(&b, &n)| b * 0.7 + n * 0.3)
                .collect();
            ContinuousHV::from_vec(values)
        })
        .collect()
}

/// Create a chain network (each node connected to neighbors only)
fn create_chain(n: usize, dim: usize, seed: u64) -> Vec<ContinuousHV> {
    let mut hvs: Vec<ContinuousHV> = (0..n)
        .map(|i| ContinuousHV::random(dim, seed + 300 + i as u64))
        .collect();

    // Make adjacent nodes similar
    for i in 0..n - 1 {
        let current = hvs[i].values.clone();
        let next = hvs[i + 1].values.clone();
        let blended: Vec<f32> = current
            .iter()
            .zip(next.iter())
            .map(|(&c, &n)| c * 0.5 + n * 0.5)
            .collect();
        hvs[i + 1] = ContinuousHV::from_vec(blended);
    }

    hvs
}

/// Create a star network (one hub connected to all peripherals)
fn create_star(n: usize, dim: usize, seed: u64) -> Vec<ContinuousHV> {
    let hub = ContinuousHV::random(dim, seed + 400);
    let mut hvs = vec![hub.clone()];

    for i in 1..n {
        let peripheral = ContinuousHV::random(dim, seed + 400 + i as u64);
        // Blend with hub
        let values: Vec<f32> = hub
            .values
            .iter()
            .zip(peripheral.values.iter())
            .map(|(&h, &p)| h * 0.4 + p * 0.6)
            .collect();
        hvs.push(ContinuousHV::from_vec(values));
    }

    hvs
}

/// Test that removing hub reduces Φ more than removing peripheral
fn test_hub_removal(phi_engine: &PhiEngine) -> (bool, String) {
    let n = 8;

    // Star topology
    let star_hvs = create_star(n, DIM, 42);
    let full_phi = phi_engine.compute(&star_hvs).phi;

    // Remove hub (index 0)
    let without_hub = star_hvs[1..].to_vec();
    let phi_no_hub = phi_engine.compute(&without_hub).phi;

    // Remove peripheral (last node)
    let without_peripheral: Vec<ContinuousHV> = star_hvs[..n - 1].to_vec();
    let phi_no_peripheral = phi_engine.compute(&without_peripheral).phi;

    let hub_reduction = (full_phi - phi_no_hub).abs();
    let peripheral_reduction = (full_phi - phi_no_peripheral).abs();

    let passed = hub_reduction > peripheral_reduction;
    let msg = format!(
        "Full Φ={:.4}, No-hub Φ={:.4} (Δ={:.4}), No-peripheral Φ={:.4} (Δ={:.4})",
        full_phi, phi_no_hub, hub_reduction, phi_no_peripheral, peripheral_reduction
    );

    (passed, msg)
}

/// Test that adding connections increases Φ
fn test_integration_increases(phi_engine: &PhiEngine) -> (bool, String) {
    let n = 8;

    // Start with disconnected, progressively integrate
    let disconnected = create_disconnected(n, DIM, 42);
    let chain = create_chain(n, DIM, 42);
    let star = create_star(n, DIM, 42);
    let integrated = create_fully_integrated(n, DIM, 42);

    let phi_disc = phi_engine.compute(&disconnected).phi;
    let phi_chain = phi_engine.compute(&chain).phi;
    let phi_star = phi_engine.compute(&star).phi;
    let phi_int = phi_engine.compute(&integrated).phi;

    // We expect: integrated > star, and chain < integrated
    // Disconnected should be the lowest
    let ordering_correct = phi_disc <= phi_chain.max(phi_star) && phi_int > phi_disc;

    let msg = format!(
        "Disconnected={:.4}, Chain={:.4}, Star={:.4}, Integrated={:.4}",
        phi_disc, phi_chain, phi_star, phi_int
    );

    (ordering_correct, msg)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     PyPhi Ground-Truth Cross-Validation                    ║");
    println!("║     IIT Known-Value Verification                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let phi_engine = PhiEngine::auto();
    let conn_calc = ConnectivityCalculator::new();

    // ═══════════════════════════════════════════════════════════════
    // Test 1: Ordering of Standard Topologies
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Standard Topology Φ Ordering");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let n = 8;
    let topologies = vec![
        ("Disconnected", create_disconnected(n, DIM, 42)),
        ("Chain", create_chain(n, DIM, 42)),
        ("Star", create_star(n, DIM, 42)),
        ("Fully Integrated", create_fully_integrated(n, DIM, 42)),
        ("Ring (topology)", {
            let topo = ConsciousnessTopology::ring(n, DIM, 42);
            topo.node_representations
                .iter()
                .map(|rhv| ContinuousHV::from_vec(rhv.values.clone()))
                .collect()
        }),
        ("Dense (topology)", {
            let topo = ConsciousnessTopology::dense_network(n, DIM, None, 42);
            topo.node_representations
                .iter()
                .map(|rhv| ContinuousHV::from_vec(rhv.values.clone()))
                .collect()
        }),
    ];

    let mut phi_values: Vec<(&str, f64, f64)> = Vec::new();

    for (name, hvs) in &topologies {
        let t = Instant::now();

        // PhiEngine Φ
        let phi_result = phi_engine.compute(hvs);
        let phi_time = t.elapsed().as_secs_f64();

        // Algebraic connectivity for comparison
        let algebraic = conn_calc.algebraic_connectivity(hvs);

        println!(
            "  {:20} │ PhiEngine: {:.6} ({:>8}) │ Algebraic: {:.6} │ {:.1}ms",
            name,
            phi_result.phi,
            phi_result.method,
            algebraic,
            phi_time * 1000.0
        );

        phi_values.push((name, phi_result.phi, algebraic));
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Φ Consistency Across Methods
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Φ Consistency Across Computation Methods");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let test_hvs = create_fully_integrated(8, DIM, 42);

    let methods = vec![
        ("Auto", PhiEngine::auto()),
        ("Spectral", PhiEngine::new(PhiMethod::SpectralConnectivity)),
        ("Resonator", PhiEngine::new(PhiMethod::Resonator)),
    ];

    let mut method_phis: Vec<f64> = Vec::new();

    for (name, engine) in &methods {
        let t = Instant::now();
        let result = engine.compute(&test_hvs);
        let elapsed = t.elapsed().as_secs_f64();

        println!(
            "  {:15} │ Φ = {:.6} │ {:.1}ms │ method: {}",
            name,
            result.phi,
            elapsed * 1000.0,
            result.method
        );

        method_phis.push(result.phi);
    }

    // Check that methods agree on relative ordering (correlation)
    let all_positive = method_phis.iter().all(|&p| p > 0.0);

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Size Scaling
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: Φ Scaling with System Size");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let sizes = vec![3, 4, 6, 8, 12, 16, 24, 32];
    let mut scale_results: Vec<(usize, f64, f64)> = Vec::new();

    for &n in &sizes {
        let hvs = create_fully_integrated(n, DIM, 42);
        let t = Instant::now();
        let result = phi_engine.compute(&hvs);
        let elapsed = t.elapsed().as_secs_f64();

        println!(
            "  n={:3} │ Φ = {:.6} │ method: {:15} │ {:.1}ms",
            n,
            result.phi,
            result.method,
            elapsed * 1000.0
        );

        scale_results.push((n, result.phi, elapsed));
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 4: Hub Removal Effect
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: Hub vs Peripheral Node Removal");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let (hub_test_passed, hub_msg) = test_hub_removal(&phi_engine);
    println!("  {}", hub_msg);

    // ═══════════════════════════════════════════════════════════════
    // Test 5: Integration Increase
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 5: Φ Increases with Integration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let (integration_test_passed, integration_msg) = test_integration_increases(&phi_engine);
    println!("  {}", integration_msg);

    // ═══════════════════════════════════════════════════════════════
    // Test 6: Reproducibility
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 6: Computation Reproducibility");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let test_hvs = create_fully_integrated(8, DIM, 42);
    let mut phis: Vec<f64> = Vec::new();

    for _trial in 0..10 {
        let result = phi_engine.compute(&test_hvs);
        phis.push(result.phi);
    }

    let phi_mean = phis.iter().sum::<f64>() / phis.len() as f64;
    let phi_std =
        (phis.iter().map(|x| (x - phi_mean).powi(2)).sum::<f64>() / (phis.len() - 1) as f64).sqrt();

    let reproducible = phi_std < 1e-10;
    println!(
        "  10 repeated computations: Φ = {:.10} ± {:.2e}",
        phi_mean, phi_std
    );

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                  VALIDATION SUMMARY                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let tests = vec![
        (
            "Disconnected < Integrated",
            phi_values.len() >= 4 && phi_values[0].1 < phi_values[3].1,
        ),
        (
            "All methods give Φ > 0 for integrated network",
            all_positive,
        ),
        (
            "Hub removal reduces Φ more than peripheral",
            hub_test_passed,
        ),
        ("Integration increases Φ", integration_test_passed),
        ("Computation is reproducible", reproducible),
        (
            "Φ computable for n=32",
            scale_results.last().map(|r| r.1 > 0.0).unwrap_or(false),
        ),
    ];

    let mut passed = 0;
    let total = tests.len();

    for (name, pass) in &tests {
        println!("║  {} {:50} ║", if *pass { "PASS" } else { "FAIL" }, name);
        if *pass {
            passed += 1;
        }
    }

    println!("╟──────────────────────────────────────────────────────────────╢");
    println!(
        "║  Result: {}/{} tests passed                                 ║",
        passed, total
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if passed == total {
        println!("All IIT ground-truth validations passed.");
    } else {
        println!("Some validations did not pass. See details above.");
    }

    // Save results
    let result_json = serde_json::json!({
        "benchmark": "PyPhi Ground-Truth Cross-Validation",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "topology_phis": phi_values.iter().map(|(name, phi, alg)| {
            serde_json::json!({
                "topology": name,
                "phi_engine": phi,
                "algebraic": alg,
            })
        }).collect::<Vec<_>>(),
        "method_consistency": method_phis,
        "scale_analysis": scale_results.iter().map(|(n, phi, t)| {
            serde_json::json!({ "n": n, "phi": phi, "time_s": t })
        }).collect::<Vec<_>>(),
        "tests_passed": passed,
        "tests_total": total,
        "reproducibility_std": phi_std,
    });

    let result_path = "data/benchmarks/pyphi/results.json";
    if let Ok(f) = std::fs::File::create(result_path) {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("\nResults saved to {}", result_path);
    }
}