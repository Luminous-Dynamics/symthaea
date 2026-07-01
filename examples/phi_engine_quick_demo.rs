// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PhiEngine Quick Demo
//!
//! Minimal example showing how to use the PhiEngine facade on
//! small consciousness topologies. This is intentionally small
//! and deterministic, suitable for quick sanity checks.

use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::phi_engine::{PhiEngine, PhiMethod};

fn main() {
    // Small topology for a fast, reproducible demo
    let n_nodes = 8;
    let dim = 512;
    let seed = 42_u64;

    println!("\nPhiEngine Quick Demo (n_nodes = {n_nodes}, dim = {dim})");
    println!("=====================================================\n");

    // Build three simple topologies
    let star = ConsciousnessTopology::star(n_nodes, dim, seed);
    let ring = ConsciousnessTopology::ring(n_nodes, dim, seed);
    let random = ConsciousnessTopology::random(n_nodes, dim, seed + 1);

    // Helper: convert topology nodes to ContinuousHV for PhiEngine
    fn to_continuous(topology: &ConsciousnessTopology) -> Vec<ContinuousHV> {
        topology
            .node_representations
            .iter()
            .map(|rhv| ContinuousHV::from_vec(rhv.values.clone()))
            .collect()
    }

    let star_hvs = to_continuous(&star);
    let ring_hvs = to_continuous(&ring);
    let random_hvs = to_continuous(&random);

    // 1. Auto method (will choose Continuous for n <= 64)
    let engine_auto = PhiEngine::auto();
    println!(
        "Method suggestion for n={n_nodes}: {:?}",
        PhiEngine::suggest_method(n_nodes)
    );

    let star_auto = engine_auto.compute(&star_hvs);
    let ring_auto = engine_auto.compute(&ring_hvs);
    let random_auto = engine_auto.compute(&random_hvs);

    println!("\nAuto method ({}):", star_auto.method);
    println!("  Star   Φ = {:.4}", star_auto.phi);
    println!("  Ring   Φ = {:.4}", ring_auto.phi);
    println!("  Random Φ = {:.4}", random_auto.phi);

    // 2. Explicit Tiered method (Mock tier for speed/determinism)
    use symthaea::phi_engine::ApproximationTier;
    let engine_tiered = PhiEngine::new(PhiMethod::Tiered(ApproximationTier::RandomBaseline));

    let star_tiered = engine_tiered.compute(&star_hvs);
    let ring_tiered = engine_tiered.compute(&ring_hvs);
    let random_tiered = engine_tiered.compute(&random_hvs);

    println!("\nTiered method ({} / RandomBaseline):", star_tiered.method);
    println!("  Star   Φ ≈ {:.4}", star_tiered.phi);
    println!("  Ring   Φ ≈ {:.4}", ring_tiered.phi);
    println!("  Random Φ ≈ {:.4}", random_tiered.phi);

    // 3. Resonator method (fast config)
    let engine_res = PhiEngine::new(PhiMethod::Resonator);

    let star_res = engine_res.compute(&star_hvs);
    let ring_res = engine_res.compute(&ring_hvs);
    let random_res = engine_res.compute(&random_hvs);

    println!("\nResonator method ({} / fast):", star_res.method);
    println!("  Star   Φ ≈ {:.4}", star_res.phi);
    println!("  Ring   Φ ≈ {:.4}", ring_res.phi);
    println!("  Random Φ ≈ {:.4}", random_res.phi);

    println!("\nDone.\n");
}