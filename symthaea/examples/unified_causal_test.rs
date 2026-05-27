// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test the Unified Causal Understanding architecture
//!
//! This tests the deep architecture based on:
//! 1. HDC Compression Asymmetry
//! 2. LTC Interventional Invariance
//! 3. Directional Phi Flow

use symthaea::benchmarks::{
    TuebingenAdapter,
    UnifiedCausalReasoning,
    // Individual primitives
    discover_by_compression,
    discover_by_directional_phi,
    discover_by_intervention,
    // Unified reasoning
    discover_by_unified_reasoning,
    // Previous methods for comparison
    discover_information_theoretic,
    discover_majority_voting,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     UNIFIED CAUSAL UNDERSTANDING - DEEP ARCHITECTURE         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load Tübingen benchmark
    let tuebingen_path = "benchmarks/external/tuebingen";
    let adapter = match TuebingenAdapter::load(tuebingen_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return;
        }
    };
    println!("Loaded {} cause-effect pairs\n", adapter.len());

    println!("Testing Individual Primitives:\n");

    // Test HDC Compression
    println!("  [1/4] HDC Compression Asymmetry...");
    let results_compression = adapter.run(discover_by_compression);
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_compression.accuracy() * 100.0,
        results_compression.correct,
        results_compression.total
    );

    // Test LTC Interventional
    println!("  [2/4] LTC Interventional Invariance...");
    let results_intervention = adapter.run(discover_by_intervention);
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_intervention.accuracy() * 100.0,
        results_intervention.correct,
        results_intervention.total
    );

    // Test Directional Phi
    println!("  [3/4] Directional Phi Flow...");
    let results_phi = adapter.run(discover_by_directional_phi);
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_phi.accuracy() * 100.0,
        results_phi.correct,
        results_phi.total
    );

    // Test Unified Reasoning
    println!("  [4/4] Unified Causal Reasoning...");
    let results_unified = adapter.run(discover_by_unified_reasoning);
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_unified.accuracy() * 100.0,
        results_unified.correct,
        results_unified.total
    );

    println!("\nComparing with Previous Methods:\n");

    let results_info = adapter.run(discover_information_theoretic);
    println!(
        "  Info-Theoretic: {:.1}% ({}/{})",
        results_info.accuracy() * 100.0,
        results_info.correct,
        results_info.total
    );

    let results_majority = adapter.run(discover_majority_voting);
    println!(
        "  Majority Voting: {:.1}% ({}/{})",
        results_majority.accuracy() * 100.0,
        results_majority.correct,
        results_majority.total
    );

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                       SUMMARY                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let methods = vec![
        ("HDC Compression", results_compression.accuracy()),
        ("LTC Intervention", results_intervention.accuracy()),
        ("Directional Phi", results_phi.accuracy()),
        ("Unified Reasoning", results_unified.accuracy()),
        ("Info-Theoretic", results_info.accuracy()),
        ("Majority Voting", results_majority.accuracy()),
    ];

    println!("  Method                   Accuracy    vs Random    Status");
    println!("  ─────────────────────────────────────────────────────────");
    for (name, acc) in &methods {
        let delta = (acc - 0.5) * 100.0;
        let marker = if *acc > 0.70 {
            "★★"
        } else if *acc > 0.60 {
            "★"
        } else if *acc > 0.55 {
            "✓"
        } else if *acc > 0.50 {
            "~"
        } else {
            "✗"
        };
        println!(
            "  {:22} {:5.1}%      {:+5.1}%      {}",
            name,
            acc * 100.0,
            delta,
            marker
        );
    }

    let best = methods
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let unified_acc = results_unified.accuracy();
    let majority_acc = results_majority.accuracy();

    println!("\n  Best Method: {} ({:.1}%)", best.0, best.1 * 100.0);

    if unified_acc >= majority_acc {
        println!("\n  ✓ Unified Reasoning matches or exceeds Majority Voting!");
    } else {
        let gap = (majority_acc - unified_acc) * 100.0;
        println!(
            "\n  Gap to close: {:.1}% (Unified: {:.1}% vs Majority: {:.1}%)",
            gap,
            unified_acc * 100.0,
            majority_acc * 100.0
        );
    }

    // Show example explanation
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                  EXAMPLE EXPLANATION                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Get first pair for explanation
    let pairs = adapter.get_pairs();
    if !pairs.is_empty() {
        let pair = &pairs[0];
        let reasoner = UnifiedCausalReasoning::new();
        println!("{}", reasoner.explain(&pair.x, &pair.y));
    }

    println!("\n  Done!");
}
