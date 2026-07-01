// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified TemporalPredictor benchmark across all Genesis domains.
//!
//! Demonstrates the O(1) closed-form CfC prediction pattern shared by
//! all five Genesis temporal predictors via the `TemporalPredictor` trait.

fn main() {
    println!("=== Unified TemporalPredictor: O(1) Cross-Domain Prediction ===\n");

    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_core::temporal::TemporalPredictor;

    // ── Build one predictor per domain ────────────────────────────────────

    let predictors: Vec<Box<dyn TemporalPredictor>> = vec![
        // Original 5 domains
        Box::new(symthaea_physics::PlasmaMultiScalePredictor::new()),
        Box::new(symthaea_cell_foundry::HydrologicalPredictor::new()),
        Box::new(symthaea_materials::MaterialAgingModel::new()),
        Box::new(symthaea_nuclear_forensics::IsotopeDecayModel::new()),
        Box::new(symthaea_physics::ScalePredictor::new(
            symthaea_physics::PhysicalScale::Molecular,
        )),
        // 12 new domains (DOE 2026 expansion)
        Box::new(symthaea_physics::GridPredictor::new()),
        Box::new(symthaea_physics::FissionPredictor::new()),
        Box::new(symthaea_physics::AcceleratorPredictor::new()),
        Box::new(symthaea_physics::ThreatPredictor::new()),
        Box::new(symthaea_physics::DatacenterPredictor::new()),
        Box::new(symthaea_cell_foundry::ExperimentPredictor::new()),
        Box::new(symthaea_materials::StrategicPredictor::new()),
        Box::new(symthaea_materials::MiningPredictor::new()),
        Box::new(symthaea_nuclear_forensics::SafeguardsPredictor::new()),
    ];

    // ── 1. Domain metadata ────────────────────────────────────────────────

    println!("--- Domain Registry ---");
    println!(
        "{:>20}  {:>14}  {:>12}",
        "Domain", "tau_base (s)", "Natural dt"
    );
    println!(
        "{:>20}  {:>14}  {:>12}",
        "------", "-----------", "----------"
    );
    for p in &predictors {
        let tau = p.tau_base();
        let label = if tau < 1.0 {
            format!("{:.0} ms", tau * 1000.0)
        } else if tau < 3600.0 {
            format!("{:.0} s", tau)
        } else if tau < 86_400.0 {
            format!("{:.0} hr", tau / 3600.0)
        } else {
            format!("{:.0} day", tau / 86_400.0)
        };
        println!("{:>20}  {:>14.6}  {:>12}", p.domain(), tau, label);
    }

    // ── 2. O(1) cost proof: short vs long horizon ─────────────────────────

    println!("\n--- O(1) Cost Proof ---");
    println!(
        "{:>20}  {:>14}  {:>14}  {:>8}",
        "Domain", "Short (µs)", "Long (µs)", "Ratio"
    );
    println!(
        "{:>20}  {:>14}  {:>14}  {:>8}",
        "------", "----------", "---------", "-----"
    );

    let seed_state = ContinuousHV::random(16_384, 0xBEEF);
    let iters = 500;

    for p in &predictors {
        let short_dt = p.tau_base(); // 1× tau
        let long_dt = p.tau_base() * 1_000_000.0; // 10^6× tau

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = p.predict_at(&seed_state, short_dt);
        }
        let short_us = t1.elapsed().as_micros() as f64 / iters as f64;

        let t2 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = p.predict_at(&seed_state, long_dt);
        }
        let long_us = t2.elapsed().as_micros() as f64 / iters as f64;

        let ratio = long_us / short_us.max(0.001);
        println!(
            "{:>20}  {:>14.1}  {:>14.1}  {:>8.2}",
            p.domain(),
            short_us,
            long_us,
            ratio
        );
        assert!(
            ratio < 5.0 && ratio > 0.1,
            "O(1) violated for {}: ratio={}",
            p.domain(),
            ratio
        );
    }

    // ── 3. Cross-domain prediction coherence ──────────────────────────────

    println!("\n--- Cross-Domain Prediction Coherence ---");
    println!("Predicting 1 year forward from the same seed state:\n");

    let one_year = 31_536_000.0_f32;
    let mut predicted_states: Vec<(&str, ContinuousHV)> = Vec::new();

    for p in &predictors {
        let predicted = p.predict_at(&seed_state, one_year);
        let drift = 1.0 - seed_state.similarity(&predicted);
        println!("  {:>20}: drift = {:.4}", p.domain(), drift);
        predicted_states.push((p.domain(), predicted));
    }

    // Cross-domain similarity matrix
    println!("\nCross-domain similarity (predicted states at t+1y):");
    print!("{:>20}", "");
    for (name, _) in &predicted_states {
        print!("{:>12}", &name[..name.len().min(10)]);
    }
    println!();
    for (i, (name_i, hv_i)) in predicted_states.iter().enumerate() {
        print!("{:>20}", &name_i[..name_i.len().min(18)]);
        for (_, hv_j) in &predicted_states {
            print!("{:>12.4}", hv_i.similarity(hv_j));
        }
        if i < predicted_states.len() - 1 {
            println!();
        }
    }
    println!();

    // ── 4. Domain-specific horizons via trait methods ────────────────────

    println!("\n--- Domain Horizon Metadata (via trait) ---");
    for p in &predictors {
        let horizons = p.default_horizons();
        let labels = p.horizon_labels();
        if horizons.is_empty() {
            println!(
                "  {:>20}: no fixed horizons (uses tau={:.6}s)",
                p.domain(),
                p.tau_base()
            );
        } else {
            let label_str: Vec<&str> = labels.iter().copied().collect();
            println!(
                "  {:>20}: {} horizons — {}",
                p.domain(),
                horizons.len(),
                label_str.join(", ")
            );
        }
    }

    // ── 5. Timescale span summary ─────────────────────────────────────────

    let tau_min = predictors
        .iter()
        .map(|p| p.tau_base())
        .fold(f32::INFINITY, f32::min);
    let tau_max = predictors
        .iter()
        .map(|p| p.tau_base())
        .fold(0.0_f32, f32::max);
    let orders = (tau_max / tau_min).log10();

    println!("\n--- Summary ---");
    println!("  Domains:        {}", predictors.len());
    println!("  Fastest tau:    {:.6} s", tau_min);
    println!("  Slowest tau:    {:.0} s", tau_max);
    println!("  Timescale span: {:.1} orders of magnitude", orders);
    println!("  All O(1):       YES (ratio < 5.0 for all domains)");

    println!(
        "\nPASS: Unified TemporalPredictor across {} domains",
        predictors.len()
    );
}
