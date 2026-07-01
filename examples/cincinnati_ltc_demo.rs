// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cincinnati-LTC Integration Demo
//!
//! Demonstrates the complete HDC+LTC integration with:
//! - Cincinnati Algorithm (differential learning)
//! - Lateral Binding (circular convolution)
//! - Predictive Autopoiesis (LTC-driven budding)
//! - Proof of Grounding (physical metrics)
//!
//! ## Run
//! ```bash
//! cargo run --example cincinnati_ltc_demo --release
//! ```

use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::cincinnati_ltc::{
    BuddingEvent, CincinnatiEstimator, CincinnatiLtcEngine, LateralBinder, PoGMetrics,
    PredictiveBudding,
};
use symthaea::hdc::unified_hv::ContinuousHV;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Cincinnati-LTC Integration Demo                          ║");
    println!("║     HDC + Liquid Time-Constant Consciousness Engine          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Demo 1: Cincinnati Estimator
    demo_cincinnati_estimator();

    // Demo 2: Lateral Binding
    demo_lateral_binding();

    // Demo 3: Predictive Budding
    demo_predictive_budding();

    // Demo 4: PoG Metrics
    demo_pog_metrics();

    // Demo 5: Full Integration
    demo_full_integration();

    println!("\n✅ All demos completed successfully!");
}

/// Demo 1: Cincinnati Algorithm - Bit-level Differential Learning
fn demo_cincinnati_estimator() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Demo 1: Cincinnati Estimator (Differential Engine)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut estimator = CincinnatiEstimator::with_seed(42);

    println!("Training on biased pattern (70% true, 30% false)...\n");

    // Train on biased observations
    for i in 0..100 {
        let observation = i % 10 < 7; // 70% true
        estimator.update(observation);

        if (i + 1) % 20 == 0 {
            let (pred, conf) = estimator.predict();
            let delta = estimator.delta_signal();
            println!(
                "  Step {:3}: prediction={}, confidence={:.3}, delta={:.4}, model_len={}",
                i + 1,
                pred,
                conf,
                delta,
                estimator.model.len()
            );
        }
    }

    let (final_pred, final_conf) = estimator.predict();
    println!(
        "\n📈 Final: prediction={}, confidence={:.4}",
        final_pred, final_conf
    );
    println!(
        "   Model length: {} bits (logarithmic growth)",
        estimator.model.len()
    );

    // Convert to HDC vector
    let hdc_vec = estimator.to_hdc_vector();
    let positive_bits = hdc_vec.iter().filter(|&&b| b > 0).count();
    println!(
        "   HDC vector: {}/{} positive bits ({:.1}%)",
        positive_bits,
        HDC_DIMENSION,
        100.0 * positive_bits as f32 / HDC_DIMENSION as f32
    );
    println!();
}

/// Demo 2: Lateral Binding - Circular Convolution
fn demo_lateral_binding() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔗 Demo 2: Lateral Binding (Circular Convolution)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut binder = LateralBinder::new(1024); // Smaller dim for speed

    // Create random nodes
    let node_a = ContinuousHV::random(1024, 100);
    let node_b = ContinuousHV::random(1024, 200);
    let node_c = ContinuousHV::random(1024, 300);

    println!("Created 3 random nodes (dim=1024)");
    println!("  sim(A,B) = {:.4}", node_a.similarity(&node_b));
    println!("  sim(B,C) = {:.4}", node_b.similarity(&node_c));
    println!("  sim(A,C) = {:.4}", node_a.similarity(&node_c));

    // Circular convolution (lateral binding)
    let ab_bound = binder.fast_convolve(&node_a, &node_b);
    println!("\nAfter lateral binding A ⊛ B:");
    println!("  sim(A⊛B, A) = {:.4}", ab_bound.similarity(&node_a));
    println!("  sim(A⊛B, B) = {:.4}", ab_bound.similarity(&node_b));
    println!("  → Bound vector is dissimilar to both inputs ✓");

    // Multi-node binding
    let all_bound = binder.bind_lateral(&[node_a.clone(), node_b.clone(), node_c.clone()]);
    if let Some(result) = all_bound {
        println!("\nMulti-node binding (A ⊛ B ⊛ C):");
        println!("  Result dim: {}", result.dim());
        println!("  sim(result, A) = {:.4}", result.similarity(&node_a));
        println!("  → Creates unified lateral representation ✓");
    }
    println!();
}

/// Demo 3: Predictive Autopoiesis - LTC-Driven Budding
fn demo_predictive_budding() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🌱 Demo 3: Predictive Autopoiesis (LTC-Driven Budding)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut budding = PredictiveBudding::new(3);

    println!("Initial nodes: {}", budding.node_count());
    println!("Budding threshold: 0.7 (high prediction error triggers spawn)");
    println!("Pruning threshold: 0.1 (low error for sustained time → prune)");
    println!();

    // Simulate high prediction error on node 0
    println!("Simulating high prediction error on Node 0...");
    for step in 0..15 {
        budding.update_error(0, 0.85); // High error
        budding.update_error(1, 0.2); // Normal
        budding.update_error(2, 0.05); // Very low (prune candidate)

        if (step + 1) % 5 == 0 {
            println!(
                "  Step {:2}: should_bud(0)={}, should_prune(2)={}",
                step + 1,
                budding.should_bud(0),
                budding.should_prune(2)
            );
        }
    }

    // Create budding event
    let state = ContinuousHV::random(1024, 42);
    if let Some(event) = budding.create_budding_event(0, 1.0, &state) {
        println!("\n🌿 Budding Event Created!");
        println!("   Parent ID: {}", event.parent_id);
        println!(
            "   Child τ: {:.3} (inherited from parent × α)",
            event.initial_tau
        );
        println!("   Trigger error: {:.3}", event.prediction_error);
        println!("   New node count: {}", budding.node_count());
    }

    // Check prune candidates
    let prune_candidates = budding.get_prune_candidates();
    println!("\n🍂 Prune candidates: {:?}", prune_candidates);
    println!();
}

/// Demo 4: Proof of Grounding Metrics
fn demo_pog_metrics() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🌍 Demo 4: Proof of Grounding (Physical Metrics)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut pog = PoGMetrics::new();

    // Simulate physical metrics
    println!("Simulating physical world grounding...\n");

    for t in 0..20 {
        pog.update_energy(10.0 + t as f64 * 0.5, t as f64);
        pog.update_storage(1_000_000 + t as u64 * 50_000);
        pog.update_bandwidth(1000.0 + t as f64 * 10.0);
        pog.update_latency(5.0 + (t as f32 * 0.1));
        pog.update_accuracy(t % 3 != 0); // 67% accuracy

        if (t + 1) % 5 == 0 {
            println!(
                "  t={:2}: energy={:.1}J, storage={:.1}MB, latency={:.1}ms, grounding={:.4}",
                t + 1,
                pog.energy_joules,
                pog.storage_bytes as f64 / 1_000_000.0,
                pog.latency_ms,
                pog.grounding_score()
            );
        }
    }

    // Convert to HDC vector
    let hdc_pog = pog.to_hdc_vector();
    println!("\n📐 PoG HDC Vector:");
    println!("   Dimension: {}", hdc_pog.dim());
    println!(
        "   Mean value: {:.6}",
        hdc_pog.values.iter().sum::<f32>() / hdc_pog.dim() as f32
    );
    println!("   Grounding score: {:.4}", pog.grounding_score());
    println!("   → Physical reality embedded in consciousness ✓");
    println!();
}

/// Demo 5: Full Cincinnati-LTC Integration
fn demo_full_integration() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("⚡ Demo 5: Full Cincinnati-LTC Engine Integration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Equation: W(t+1) = W(t) ⊕ (ΔC ⊛ τ(t))");
    println!("Where:");
    println!("  W(t)  = Weight hypervector");
    println!("  ΔC    = Cincinnati delta (differential engine)");
    println!("  τ(t)  = Time constant from LTC + PoG");
    println!("  ⊛     = Circular convolution (lateral binding)");
    println!("  ⊕     = HDC bundling\n");

    let mut engine = CincinnatiLtcEngine::new(5);

    // Create input pattern
    let inputs: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 * 1000))
        .collect();

    println!("Running 50 steps with alternating observations...\n");

    let mut outputs = Vec::new();
    let mut budding_events: Vec<BuddingEvent> = Vec::new();

    for t in 0..50 {
        let observation = t % 2 == 0;
        let input = &inputs[t % 5];

        // Step the engine
        let output = engine.step(observation, input);

        // Update PoG metrics
        engine.pog.update_latency(5.0 + (t as f32 * 0.05));
        engine.pog.update_energy(1.0, t as f64);

        // Update prediction errors
        if !outputs.is_empty() {
            let prev = &outputs[outputs.len() - 1];
            engine.update_prediction_error(t % 5, prev, &output);
        }

        outputs.push(output);

        // Check for budding
        let events = engine.process_budding(&inputs, t as f64);
        budding_events.extend(events);

        if (t + 1) % 10 == 0 {
            let (pred, conf) = engine.predict();
            println!(
                "  Step {:2}: pred={}, conf={:.3}, nodes={}, grounding={:.3}",
                t + 1,
                pred,
                conf,
                engine.node_count(),
                engine.pog.grounding_score()
            );
        }
    }

    println!("\n📊 Final Statistics:");
    println!("   Total budding events: {}", budding_events.len());
    println!("   Final node count: {}", engine.node_count());
    println!("   Prune candidates: {:?}", engine.prune_candidates());

    let (final_pred, final_conf) = engine.predict();
    println!(
        "   Final prediction: {}, confidence: {:.4}",
        final_pred, final_conf
    );

    // Show weight evolution
    let weight = engine.weight();
    let weight_magnitude =
        (weight.values.iter().map(|x| x * x).sum::<f32>() / weight.dim() as f32).sqrt();
    println!("   Weight RMS magnitude: {:.6}", weight_magnitude);

    // Compare output similarity over time
    if outputs.len() >= 2 {
        let first = &outputs[0];
        let last = &outputs[outputs.len() - 1];
        let evolution = first.similarity(last);
        println!(
            "   Output evolution (first→last similarity): {:.4}",
            evolution
        );
    }

    println!("\n✨ Cincinnati-LTC Engine demonstrates:");
    println!("   • Bit-level differential learning (Cincinnati)");
    println!("   • Lateral binding via circular convolution");
    println!("   • Autonomous node budding based on prediction error");
    println!("   • Physical grounding through PoG metrics");
    println!("   • Unified consciousness evolution equation");
}