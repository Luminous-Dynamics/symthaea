// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binding Reversibility Experiment - H2 Revised
//!
//! This experiment tests the revised H2 hypothesis:
//!
//! **H2-revised**: HDC binding (XOR) preserves component information better than
//! bundling (majority vote), measurable via reversibility.
//!
//! ## Background
//!
//! The original H2 hypothesis failed because unity measures (Betti numbers, etc.)
//! measure topological self-similarity of permutations, not binding quality.
//!
//! The key insight is that binding and bundling differ fundamentally in
//! **reversibility**:
//!
//! - **Binding (XOR)**: A can be perfectly recovered from (A XOR B) XOR B = A
//!   This is the mathematical self-inverse property of XOR.
//!
//! - **Bundling (majority vote)**: Components are mixed and cannot be cleanly
//!   separated. There is no inverse operation.
//!
//! ## What This Experiment Tests
//!
//! 1. **Reversibility**: Can we recover A from the combined representation?
//!    - For binding: similarity(A, (A XOR B) XOR B) should be 1.0
//!    - For bundling: no true inverse exists
//!
//! 2. **Component Preservation**: How much of A survives in the result?
//!    - For binding: similarity(A, A XOR B) is ~0.5 (orthogonal)
//!    - For bundling: similarity(A, bundle(A,B)) is ~0.67 (superposition)
//!
//! 3. **Category Comparison**: Does this differ for unified vs separate pairs?
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example binding_reversibility_experiment --release
//! ```

use std::time::Instant;

use anyhow::Result;

use symthaea::hdc::binding_reversibility_study::{ReversibilityConfig, ReversibilityStudy};
use symthaea_core::hdc::binary_hv::BinaryHV;

fn main() -> Result<()> {
    println!();
    println!("================================================================");
    println!("   BINDING REVERSIBILITY EXPERIMENT - H2 REVISED");
    println!("   Testing HDC binding via reversibility");
    println!("================================================================");
    println!();

    // First, demonstrate the core mathematical property
    demonstrate_reversibility_property();

    println!();
    println!("================================================================");
    println!("   RUNNING FULL STUDY ON CONCEPT PAIRS");
    println!("================================================================");
    println!();

    // Run the full study
    let config = ReversibilityConfig {
        verbose: true,
        n_permutations: 5000,
        ..Default::default()
    };

    let mut study = ReversibilityStudy::new(config);

    let pairs_path = "data/binding_study/concept_pairs.json";
    println!("Loading pairs from: {}", pairs_path);

    let start = Instant::now();
    let results = study.run(pairs_path)?;
    let duration = start.elapsed();

    println!();
    println!("Study completed in {:.2}s", duration.as_secs_f64());
    println!();

    // Print the full report
    println!("{}", results.summary_report());

    // Print sample results
    println!("================================================================");
    println!("   SAMPLE PAIR RESULTS");
    println!("================================================================");
    println!();

    println!("Top pairs by reversibility advantage (bind - bundle):");
    println!();
    println!(
        "{:<25} {:>8} {:>8} {:>8} {:>8}",
        "Pair", "Bind Rev", "Bund Rev", "Bind Pres", "Bund Pres"
    );
    println!("{}", "-".repeat(65));

    for result in results.sample_results(10) {
        let pair_name = format!("{}+{}", result.pair.concept_a, result.pair.concept_b);
        let truncated = if pair_name.len() > 24 {
            format!("{}...", &pair_name[..21])
        } else {
            pair_name
        };
        println!(
            "{:<25} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            truncated,
            result.mean_bind_reversibility(),
            result.mean_bundle_reversibility(),
            result.mean_bind_preservation(),
            result.mean_bundle_preservation(),
        );
    }

    println!();

    // Print category breakdown
    println!("================================================================");
    println!("   CATEGORY BREAKDOWN");
    println!("================================================================");
    println!();

    for (name, stats) in &results.category_stats {
        println!("{} pairs (n={}):", name, stats.count);
        println!("  Reversibility:");
        println!(
            "    Bind:   {:.4} +/- {:.4}",
            stats.mean_bind_reversibility, stats.std_bind_reversibility
        );
        println!(
            "    Bundle: {:.4} +/- {:.4}",
            stats.mean_bundle_reversibility, stats.std_bundle_reversibility
        );
        println!(
            "    Advantage: {:+.4} +/- {:.4}",
            stats.mean_reversibility_advantage, stats.std_reversibility_advantage
        );
        println!();
        println!("  Component Preservation:");
        println!("    Bind:   {:.4}", stats.mean_bind_preservation);
        println!("    Bundle: {:.4}", stats.mean_bundle_preservation);
        println!();
    }

    // Final interpretation
    println!("================================================================");
    println!("   FINAL INTERPRETATION");
    println!("================================================================");
    println!();

    // Find key results
    let unified_stats = results.category_stats.get("unified");
    let separate_stats = results.category_stats.get("separate");

    if let Some(unified) = unified_stats {
        if unified.mean_bind_reversibility > 0.99 {
            println!(
                "KEY FINDING: Binding achieves near-perfect reversibility ({:.4})",
                unified.mean_bind_reversibility
            );
            println!();
            println!("This confirms the mathematical property of XOR:");
            println!("  (A XOR B) XOR B = A");
            println!();
            println!("For phenomenal binding, this means:");
            println!("  - The unified percept (bound representation) is DIFFERENT from");
            println!("    its components (~0.5 similarity)");
            println!("  - Yet the components are PERFECTLY RECOVERABLE");
            println!("  - This models how 'red apple' is a unified experience that");
            println!("    still contains accessible information about 'red' and 'apple'");
        }
    }

    if let (Some(unified), Some(separate)) = (unified_stats, separate_stats) {
        let diff = unified.mean_reversibility_advantage - separate.mean_reversibility_advantage;
        if diff.abs() < 0.001 {
            println!();
            println!("Note: Reversibility advantage is similar for unified and separate pairs.");
            println!("This is expected - the XOR property is mathematical, not semantic.");
            println!("The KEY difference is that binding creates appropriate structure");
            println!("for unified percepts, not that different pairs bind differently.");
        }
    }

    println!();
    println!("================================================================");
    println!("   COMPARISON: Binding vs Bundling for Phenomenal Unity");
    println!("================================================================");
    println!();
    println!("BINDING (XOR):");
    println!("  + Creates NEW orthogonal representation");
    println!("  + Components PERFECTLY RECOVERABLE via unbinding");
    println!("  + Models 'emergent unity with accessible parts'");
    println!("  - Result looks different from inputs (~0.5 similarity)");
    println!();
    println!("BUNDLING (Majority Vote):");
    println!("  + Result similar to both components (~0.67 similarity)");
    println!("  + Intuitive 'superposition' of features");
    println!("  - Components IRREVERSIBLY MIXED");
    println!("  - Cannot cleanly extract original information");
    println!();
    println!("CONCLUSION:");
    println!("  Binding is the appropriate operation for phenomenal unity");
    println!("  because unified percepts should:");
    println!("    1. Be distinct from their parts (binding achieves this)");
    println!("    2. Preserve information about parts (binding achieves this)");
    println!("  Bundling fails criterion 2 - information is lost.");
    println!();
    println!("================================================================");

    Ok(())
}

/// Demonstrate the core reversibility property
fn demonstrate_reversibility_property() {
    println!("================================================================");
    println!("   DEMONSTRATING REVERSIBILITY PROPERTY");
    println!("================================================================");
    println!();
    println!("Core mathematical property of XOR (binding):");
    println!("  (A XOR B) XOR B = A");
    println!();

    // Create test vectors
    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);

    println!("Creating test vectors:");
    println!("  A = BinaryHV::random(42)");
    println!("  B = BinaryHV::random(43)");
    println!();

    // Baseline similarity
    let ab_sim = a.similarity(&b);
    println!("Baseline similarity(A, B) = {:.4}", ab_sim);
    println!("  (Random vectors have ~0.5 similarity)");
    println!();

    // Binding
    let bound = a.bind(&b);
    println!("BINDING (XOR):");
    println!("  bound = A XOR B");
    println!("  similarity(A, bound) = {:.4}", a.similarity(&bound));
    println!("  similarity(B, bound) = {:.4}", b.similarity(&bound));
    println!("  Note: Bound result is orthogonal to both inputs (~0.5)");
    println!();

    // Unbinding - perfect recovery
    let recovered_a = bound.bind(&b);
    let recovered_b = bound.bind(&a);
    println!("UNBINDING (reversibility test):");
    println!("  recovered_A = bound XOR B");
    println!("  recovered_B = bound XOR A");
    println!();
    println!(
        "  similarity(A, recovered_A) = {:.4}",
        a.similarity(&recovered_a)
    );
    println!(
        "  similarity(B, recovered_B) = {:.4}",
        b.similarity(&recovered_b)
    );
    println!("  PERFECT RECOVERY (similarity = 1.0)!");
    println!();

    // Bundling
    let bundled = BinaryHV::bundle(&[a, b]);
    println!("BUNDLING (Majority Vote):");
    println!("  bundled = bundle([A, B])");
    println!("  similarity(A, bundled) = {:.4}", a.similarity(&bundled));
    println!("  similarity(B, bundled) = {:.4}", b.similarity(&bundled));
    println!("  Note: Bundled result is similar to both inputs (~0.67)");
    println!();

    // Attempt to "unbundle" (no true inverse exists)
    let attempt_a = bundled.bind(&b); // XOR is the best heuristic
    let attempt_b = bundled.bind(&a);
    println!("UNBUNDLE ATTEMPT (best heuristic: XOR):");
    println!("  attempt_A = bundled XOR B");
    println!("  attempt_B = bundled XOR A");
    println!();
    println!(
        "  similarity(A, attempt_A) = {:.4}",
        a.similarity(&attempt_a)
    );
    println!(
        "  similarity(B, attempt_B) = {:.4}",
        b.similarity(&attempt_b)
    );
    println!("  IMPERFECT RECOVERY (similarity < 1.0)");
    println!();

    // Summary
    println!("SUMMARY:");
    println!("  Binding reversibility:  1.0000 (perfect)");
    println!(
        "  Bundling reversibility: {:.4} (imperfect)",
        (a.similarity(&attempt_a) + b.similarity(&attempt_b)) / 2.0
    );
    println!();
    println!("Binding preserves information via perfect reversibility,");
    println!("while bundling irreversibly mixes components.");
    println!();
}