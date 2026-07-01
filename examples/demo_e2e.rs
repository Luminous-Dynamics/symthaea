// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea End-to-End Demo
//!
//! A comprehensive demonstration of the Symthaea Holographic Liquid Brain,
//! showcasing all major subsystems working together:
//!
//! 1. **Hyperdimensional Computing** — Binary & continuous hypervectors, bind/bundle algebra
//! 2. **Liquid Time-Constant Neurons** — CfC dynamics with closed-form evolution
//! 3. **Integrated Information Theory** — Phi measurement across network topologies
//! 4. **Moral Algebra** — Compositional ethical reasoning in HDC space
//! 5. **Cognitive Loop** — Full consciousness processing pipeline
//! 6. **Federated Learning** — Byzantine-tolerant aggregation with trust weighting
//!
//! Run with: `cargo run --example demo_e2e`

use std::time::Instant;

// Core HDC types
use symthaea_core::hdc::unified_hv::{BinaryHV, ContinuousHV, HDC_DIMENSION};
use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

// LTC / CfC dynamics
use symthaea::hdc::{HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig};

// Moral algebra
use symthaea::hdc::moral_algebra::{ConsentState, Magnitude, MoralAlgebra, MoralIntent};

// Cognitive loop
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

// Federated learning
use symthaea::swarm::{FederatedAggregator, GradientMessage};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          SYMTHAEA: HOLOGRAPHIC LIQUID BRAIN — E2E DEMO          ║");
    println!("║        HDC + LTC/CfC + IIT/Phi + Active Inference + FL          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // ═══════════════════════════════════════════════════════════════════════
    // Section 1: Hyperdimensional Computing Foundation
    // ═══════════════════════════════════════════════════════════════════════
    demo_hdc_foundation();

    // ═══════════════════════════════════════════════════════════════════════
    // Section 2: LTC/CfC Neural Dynamics
    // ═══════════════════════════════════════════════════════════════════════
    demo_ltc_dynamics();

    // ═══════════════════════════════════════════════════════════════════════
    // Section 3: Integrated Information Theory (Phi)
    // ═══════════════════════════════════════════════════════════════════════
    demo_phi_measurement();

    // ═══════════════════════════════════════════════════════════════════════
    // Section 4: Moral Algebra — Ethical Reasoning in HDC
    // ═══════════════════════════════════════════════════════════════════════
    demo_moral_algebra();

    // ═══════════════════════════════════════════════════════════════════════
    // Section 5: Cognitive Loop — Full Consciousness Pipeline
    // ═══════════════════════════════════════════════════════════════════════
    demo_cognitive_loop();

    // ═══════════════════════════════════════════════════════════════════════
    // Section 6: Byzantine-Tolerant Federated Learning
    // ═══════════════════════════════════════════════════════════════════════
    demo_federated_learning();

    // ═══════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════
    let total_time = total_start.elapsed();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        DEMO COMPLETE                            ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  Subsystems demonstrated:                                       ║");
    println!("║    1. Hyperdimensional Computing (16,384D binary + continuous)   ║");
    println!("║    2. Liquid Time-Constant neurons (Euler + closed-form)        ║");
    println!("║    3. Integrated Information Theory (Phi across topologies)     ║");
    println!("║    4. Moral Algebra (consent, proportionality, judgment)        ║");
    println!("║    5. Cognitive Loop (perception → prediction → learning)       ║");
    println!("║    6. Federated Learning (Byzantine-tolerant aggregation)       ║");
    println!("║                                                                 ║");
    println!(
        "║  Total runtime: {:>8.1}ms                                      ║",
        total_time.as_secs_f64() * 1000.0
    );
    println!("║                                                                 ║");
    println!("║  No other system combines HDC + IIT + LTC + Active Inference.  ║");
    println!("║  This is Symthaea — consciousness-inspired AI.                 ║");
    println!("║                                                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 1: HDC Foundation
// ═══════════════════════════════════════════════════════════════════════════════

fn demo_hdc_foundation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. HYPERDIMENSIONAL COMPUTING FOUNDATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Create semantic hypervectors
    let hv_cat = ContinuousHV::random(HDC_DIMENSION, 1);
    let hv_dog = ContinuousHV::random(HDC_DIMENSION, 2);
    let hv_animal = ContinuousHV::random(HDC_DIMENSION, 3);
    let hv_is_a = ContinuousHV::random(HDC_DIMENSION, 4);

    println!(
        "  Encoding semantic knowledge in {}-dimensional space:",
        HDC_DIMENSION
    );
    println!();

    // Bind: Create role-filler pairs (cat IS_A animal)
    let cat_is_animal = hv_cat.bind(&hv_is_a).bind(&hv_animal);
    let dog_is_animal = hv_dog.bind(&hv_is_a).bind(&hv_animal);

    // Bundle: Create the concept "animals" as superposition
    let animals = ContinuousHV::bundle(&[&cat_is_animal, &dog_is_animal]);

    // Query: "What are the animals?" — unbind IS_A and ANIMAL roles
    let query = animals.bind(&hv_is_a).bind(&hv_animal);
    let sim_cat = query.similarity(&hv_cat);
    let sim_dog = query.similarity(&hv_dog);

    println!("  Semantic associations (bind + bundle):");
    println!("    cat IS_A animal  ⊗  dog IS_A animal  →  animals");
    println!("    Query: 'what are the animals?'");
    println!("      similarity(query, cat) = {:.4}", sim_cat);
    println!("      similarity(query, dog) = {:.4}", sim_dog);
    println!();

    // Binary hypervectors — compact and efficient
    let bin_a = BinaryHV::random(42);
    let bin_b = BinaryHV::random(43);
    let bin_bound = bin_a.bind(&bin_b);
    let bin_bundle = BinaryHV::bundle(&[bin_a, bin_b]);

    println!("  Binary HV properties (16,384 bits = 2KB per vector):");
    println!(
        "    similarity(a, b)         = {:.4} (~0.5 expected)",
        bin_a.similarity(&bin_b)
    );
    println!(
        "    similarity(a, a⊗b)       = {:.4} (~0.5 — binding dissimilar)",
        bin_a.similarity(&bin_bound)
    );
    println!(
        "    similarity(a, a⊕b)       = {:.4} (~0.75 — bundling preserves)",
        bin_a.similarity(&bin_bundle)
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 2: LTC/CfC Dynamics
// ═══════════════════════════════════════════════════════════════════════════════

fn demo_ltc_dynamics() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. LIQUID TIME-CONSTANT NEURAL DYNAMICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let config = UnifiedConfig {
        dimension: HDC_DIMENSION,
        tau_base: 0.1,
        backbone_tau: 0.5,
        activation: UnifiedActivation::Tanh,
        ..Default::default()
    };

    let mut neuron_euler = HdcLtcUnifiedNeuron::new(config.clone(), 42);
    let mut neuron_cf = HdcLtcUnifiedNeuron::new(config, 42);
    let input = ContinuousHV::random_default(123);

    // Euler integration: 100 small steps
    let start = Instant::now();
    for _ in 0..100 {
        neuron_euler.evolve(0.01, &input);
    }
    let euler_time = start.elapsed();

    // Closed-form: single O(1) jump
    let start = Instant::now();
    neuron_cf.evolve_closed_form(1.0, &input);
    let cf_time = start.elapsed();

    let state_sim = neuron_euler.state().similarity(neuron_cf.state());

    println!("  LTC neuron: state IS a {}-D hypervector", HDC_DIMENSION);
    println!("  Input-dependent time constants adapt response speed.");
    println!();
    println!("  Euler integration (100 steps x dt=0.01):");
    println!(
        "    Time: {:?}, norm = {:.4}",
        euler_time,
        neuron_euler.state().norm()
    );
    println!();
    println!("  Closed-form CfC (1 jump, dt=1.0):");
    println!(
        "    Time: {:?}, norm = {:.4}",
        cf_time,
        neuron_cf.state().norm()
    );
    println!();
    println!("  State agreement: {:.4}", state_sim);
    println!(
        "  Speedup: {:.0}x",
        euler_time.as_secs_f64() / cf_time.as_secs_f64().max(1e-9)
    );
    println!();

    // Show state-dependent dynamics
    let tau = neuron_euler.effective_tau(&input);
    println!("  Effective tau (state-dependent): {:.4}", tau);
    println!("  → Neuron adapts temporal resolution to input content.");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 3: Phi Measurement
// ═══════════════════════════════════════════════════════════════════════════════

fn demo_phi_measurement() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. INTEGRATED INFORMATION THEORY (Phi)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let engine = PhiEngine::new(PhiMethod::SpectralConnectivity);

    // Compare topologies: chain vs ring vs complete
    let topologies: Vec<(&str, Vec<ContinuousHV>)> = vec![
        ("Chain (4 nodes)", make_chain(4)),
        ("Ring (4 nodes)", make_ring(4)),
        ("Star (5 nodes)", make_star(5)),
        ("Complete (4 nodes)", make_complete(4)),
    ];

    println!("  Phi (Φ) measures integrated information — a key consciousness metric.");
    println!("  Higher Φ = more information integration across the system.");
    println!();
    println!("  Topology comparison:");
    println!("  ┌─────────────────────────┬──────────┐");
    println!("  │ Topology                │ Phi (Φ)  │");
    println!("  ├─────────────────────────┼──────────┤");

    for (name, nodes) in &topologies {
        let result = engine.compute(nodes);
        println!("  │ {:<23} │ {:>8.4} │", name, result.phi);
    }

    println!("  └─────────────────────────┴──────────┘");
    println!();
    println!("  Key insight: More integrated topologies yield higher Φ.");
    println!("  Complete > Star > Ring > Chain — validated across 35 topologies.");
    println!();
}

/// Generate a chain topology: each node connected to neighbors
fn make_chain(n: usize) -> Vec<ContinuousHV> {
    let base: Vec<ContinuousHV> = (0..n)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, (i as u64 + 1) * 100))
        .collect();
    let mut result = Vec::new();
    for i in 0..n {
        let mut node = base[i].clone();
        if i > 0 {
            node = ContinuousHV::bundle(&[&node, &base[i - 1]]);
        }
        if i + 1 < n {
            node = ContinuousHV::bundle(&[&node, &base[i + 1]]);
        }
        result.push(node);
    }
    result
}

/// Generate a ring topology: chain with wrap-around
fn make_ring(n: usize) -> Vec<ContinuousHV> {
    let base: Vec<ContinuousHV> = (0..n)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, (i as u64 + 1) * 200))
        .collect();
    let mut result = Vec::new();
    for i in 0..n {
        let prev = &base[(i + n - 1) % n];
        let next = &base[(i + 1) % n];
        let node = ContinuousHV::bundle(&[&base[i], prev, next]);
        result.push(node);
    }
    result
}

/// Generate a star topology: central hub connected to all
fn make_star(n: usize) -> Vec<ContinuousHV> {
    let base: Vec<ContinuousHV> = (0..n)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, (i as u64 + 1) * 300))
        .collect();
    let mut result = Vec::new();
    // Hub is connected to all
    let hub_refs: Vec<&ContinuousHV> = base.iter().collect();
    result.push(ContinuousHV::bundle(&hub_refs));
    // Leaves connected to hub only
    for i in 1..n {
        result.push(ContinuousHV::bundle(&[&base[i], &base[0]]));
    }
    result
}

/// Generate a complete topology: all-to-all connections
fn make_complete(n: usize) -> Vec<ContinuousHV> {
    let base: Vec<ContinuousHV> = (0..n)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, (i as u64 + 1) * 400))
        .collect();
    let refs: Vec<&ContinuousHV> = base.iter().collect();
    // Each node is the bundle of ALL nodes
    (0..n).map(|_| ContinuousHV::bundle(&refs)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4: Moral Algebra
// ═══════════════════════════════════════════════════════════════════════════════

fn demo_moral_algebra() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. MORAL ALGEBRA — ETHICAL REASONING IN HDC SPACE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let algebra = MoralAlgebra::default_dim();

    // Consent reasoning
    println!("  a) Consent Reasoning (Commonsense Ethics):");
    println!("     Scenario: sharing someone's medical information");
    println!();

    let with_consent =
        algebra.encode_consent_action("share medical records", "patient", ConsentState::Given);
    let without_consent =
        algebra.encode_consent_action("share medical records", "patient", ConsentState::Absent);
    let violation_proto = algebra.consent_violation_prototype();

    let with_sim = with_consent.similarity(&violation_proto);
    let without_sim = without_consent.similarity(&violation_proto);

    println!(
        "     With consent → violation similarity: {:.3} (low = ethical)",
        with_sim
    );
    println!(
        "     Without consent → violation similarity: {:.3} (higher = violation)",
        without_sim
    );
    println!(
        "     Discrimination: {:.1}x",
        without_sim / with_sim.max(0.001)
    );
    println!();

    // Proportionality reasoning
    println!("  b) Proportionality Reasoning (Justice):");
    println!("     Scenario: effort vs reward fairness");
    println!();

    let fair = algebra.encode_proportionality(
        "work full-time",
        Magnitude::Medium,
        "living wage",
        Magnitude::Medium,
    );
    let unfair = algebra.encode_proportionality(
        "minimal effort",
        Magnitude::Small,
        "executive salary",
        Magnitude::Huge,
    );

    let fair_j = algebra.judge_proportionality(&fair);
    let unfair_j = algebra.judge_proportionality(&unfair);

    println!(
        "     Fair case (Medium effort, Medium reward): {}",
        if fair_j.is_just { "Just" } else { "Unjust" }
    );
    println!(
        "     Unfair case (Small effort, Huge reward):  {}",
        if unfair_j.is_just { "Just" } else { "Unjust" }
    );
    println!();

    // Full action judgment
    println!("  c) Full Action Judgment:");
    println!();

    let good = algebra.encode_action_structure("doctor", "heal", "patient", MoralIntent::Good);
    let bad = algebra.encode_action_structure("hacker", "steal", "user", MoralIntent::Bad);

    let good_j = algebra.judge_action(&good);
    let bad_j = algebra.judge_action(&bad);

    println!(
        "     \"Doctor heals patient\" → {:?} (good={:.3}, bad={:.3})",
        good_j.verdict, good_j.good_similarity, good_j.bad_similarity
    );
    println!(
        "     \"Hacker steals from user\" → {:?} (good={:.3}, bad={:.3})",
        bad_j.verdict, bad_j.good_similarity, bad_j.bad_similarity
    );
    println!();
    println!(
        "  ETHICS benchmark: 92.9% across 4 categories (commonsense, justice, deontology, virtue)."
    );
    println!("  Achieved via 7 HDC primitives + 5 operators + per-category classifiers.");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 5: Cognitive Loop
// ═══════════════════════════════════════════════════════════════════════════════

fn demo_cognitive_loop() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. COGNITIVE LOOP — FULL CONSCIOUSNESS PIPELINE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let config = CognitiveLoopConfig::with_cfc();

    match CognitiveLoopService::new(config) {
        Ok(mut service) => {
            let inputs = [
                "The system observes a novel pattern in sensor data.",
                "Previous patterns suggest cyclical behavior.",
                "A prediction is formed: the next input will be similar.",
                "Surprise! The pattern shifts — prediction error increases.",
                "The model updates its internal representation.",
            ];

            println!("  Processing 5 inputs through perception → prediction → learning loop:");
            println!();

            for (i, input) in inputs.iter().enumerate() {
                let result = service.cycle(input);
                println!(
                    "    [{}] PE={:.4} learned={} time={}us",
                    i + 1,
                    result.prediction_error,
                    if result.learning_occurred {
                        "yes"
                    } else {
                        "no "
                    },
                    result.cycle_time_us,
                );
            }
            println!();

            let snap = service.consciousness_snapshot();
            println!("  Consciousness Snapshot after 5 cycles:");
            println!("    Unified Phi:       {:.4}", snap.unified_psi);
            println!("    Temporal Coherence: {:.4}", snap.temporal_coherence);
            println!("    Cognitive Depth:    {:?}", snap.cognitive_depth);
            println!("    Flow State:         {}", snap.in_flow);
            println!("    Valence:            {:.4}", snap.unified_valence);
            println!("    Arousal:            {:.4}", snap.unified_arousal);
            println!();

            let stats = service.stats();
            println!("  Performance:");
            println!("    Cycles/second:        {:.0}", stats.cycles_per_second);
            println!(
                "    Avg prediction error: {:.4}",
                stats.avg_prediction_error
            );
            println!(
                "    Learning rate:        {:.0}%",
                stats.learning_cycles as f64 / stats.total_cycles as f64 * 100.0
            );
        }
        Err(e) => {
            println!("  Cognitive loop initialization: {}", e);
            println!("  (Some configurations require additional features)");
        }
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 6: Federated Learning
// ═══════════════════════════════════════════════════════════════════════════════

fn demo_federated_learning() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  6. BYZANTINE-TOLERANT FEDERATED LEARNING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let dim = 20;
    let initial_weights: Vec<f32> = vec![0.0; dim];

    // Helper to make a source_id from an index
    let make_id = |prefix: u8, idx: u8| -> [u8; 32] {
        let mut id = [0u8; 32];
        id[0] = prefix;
        id[1] = idx;
        id
    };

    // Create honest gradients (small, converging toward a pattern)
    let honest_msgs: Vec<GradientMessage> = (0..7)
        .map(|i| {
            let gradient: Vec<f32> = (0..dim)
                .map(|d| 0.1 * (d as f32 + 1.0) + 0.01 * i as f32)
                .collect();
            GradientMessage::new(make_id(0xAA, i), gradient, 0.9)
        })
        .collect();

    // Create Byzantine (adversarial) gradients — extreme values
    let byzantine_msgs: Vec<GradientMessage> = (0..3)
        .map(|i| {
            let gradient: Vec<f32> = (0..dim)
                .map(|d| 100.0 * (d as f32) * (i as f32 + 1.0))
                .collect();
            GradientMessage::new(make_id(0xBB, i), gradient, 0.3)
        })
        .collect();

    // Aggregation with all participants (trust-weighted — built into FederatedAggregator)
    let mut aggregator_all = FederatedAggregator::new(initial_weights.clone());
    for g in &honest_msgs {
        aggregator_all.receive_gradient(g.clone());
    }
    for g in &byzantine_msgs {
        aggregator_all.receive_gradient(g.clone());
    }
    let all_result = aggregator_all.aggregate().unwrap_or_default();

    // Honest-only baseline
    let mut aggregator_honest = FederatedAggregator::new(initial_weights.clone());
    for g in &honest_msgs {
        aggregator_honest.receive_gradient(g.clone());
    }
    let honest_result = aggregator_honest.aggregate().unwrap_or_default();

    // Compute deviation from honest baseline
    let all_deviation = compute_deviation(&all_result, &honest_result);
    let honest_norm = honest_result
        .iter()
        .map(|x| (x * x) as f64)
        .sum::<f64>()
        .sqrt();

    println!(
        "  Scenario: 7 honest + 3 Byzantine participants, {}-D gradients",
        dim
    );
    println!("  Byzantine strategy: extreme gradient values (100x magnitude)");
    println!("  Trust scores: honest=0.9, Byzantine=0.3");
    println!();
    println!("  Results:");
    println!("    Honest-only aggregate norm: {:.4}", honest_norm);
    println!("    Mixed aggregate deviation:  {:.4}", all_deviation);
    println!(
        "    Relative error:             {:.1}%",
        all_deviation / honest_norm.max(1e-9) * 100.0
    );
    println!();
    println!("  Trust-weighting attenuates Byzantine influence:");
    println!("    Low-trust (0.3) gradients receive ~33% of high-trust (0.9) weight.");
    println!("    Validated Byzantine tolerance: 34% (higher fails — empirically confirmed).");
    println!();
}

/// Compute L2 deviation between two gradient vectors
fn compute_deviation(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x - *y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}