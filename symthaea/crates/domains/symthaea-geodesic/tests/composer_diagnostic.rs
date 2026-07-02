// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Composer Diagnostic — can the composer find the right sub-trees?

use symthaea_core::hdc::program_algebra::ProgramNode;
use symthaea_geodesic::periodic_table::encode_structural;
use symthaea_geodesic::program_memory::ProgramMemory;

#[test]
fn diagnostic_subtree_matching() {
    let memory = ProgramMemory::basic();

    // Target: second_largest's iterate (the structure we want to compose)
    let target_iterate = ProgramNode::iterate(
        ProgramNode::atom("i = 2"),
        ProgramNode::branch(
            ProgramNode::apply(
                ProgramNode::op("GT"),
                vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("max1")],
            ),
            ProgramNode::atom("update maxes"),
            ProgramNode::atom("skip"),
        ),
        ProgramNode::apply(
            ProgramNode::op("LT"),
            vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
        ),
    );

    let target_enc = encode_structural(&target_iterate);

    println!("\n=== DIAGNOSTIC: Sub-tree matching for second_largest ===\n");
    println!("{:<25} {:>10} {:>10}", "Pattern", "Struct", "Name");
    println!("{}", "-".repeat(48));

    // Check top-5 structural matches
    let mut results: Vec<(String, f32, f32)> = memory
        .nearest_k(&target_iterate.encode(), 30)
        .iter()
        .map(|(entry, _)| {
            let structural = encode_structural(&entry.node).similarity(&target_enc);
            let name_based = entry.encoding.similarity(&target_iterate.encode());
            (entry.name.clone(), structural, name_based)
        })
        .collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));

    for (name, structural, name_based) in results.iter().take(10) {
        let marker = if *structural > 0.7 { " ← GOOD" } else { "" };
        println!(
            "{:<25} {:>10.4} {:>10.4}{}",
            name, structural, name_based, marker
        );
    }

    // Key question: does find_max's iterate match our target iterate?
    let find_max_iterate = ProgramNode::iterate(
        ProgramNode::atom("i = 1"),
        ProgramNode::branch(
            ProgramNode::apply(
                ProgramNode::op("GT"),
                vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("max")],
            ),
            ProgramNode::atom("max = arr[i]"),
            ProgramNode::atom("/* no-op */"),
        ),
        ProgramNode::apply(
            ProgramNode::op("LT"),
            vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
        ),
    );

    let structural_sim = encode_structural(&find_max_iterate).similarity(&target_enc);
    let name_sim = find_max_iterate
        .encode()
        .similarity(&target_iterate.encode());

    println!("\n=== KEY COMPARISON ===");
    println!("find_max iterate vs second_largest iterate:");
    println!("  Structural (periodic table): {:.4}", structural_sim);
    println!("  Name-based (old encoding):   {:.4}", name_sim);
    println!(
        "  Improvement:                 {:+.4}",
        structural_sim - name_sim
    );

    if structural_sim > 0.8 {
        println!("\n  VERDICT: Periodic table enables matching ✓");
        println!("  The composer CAN find find_max's iterate as a sub-tree for second_largest");
    } else if structural_sim > 0.6 {
        println!("\n  VERDICT: Partial improvement — encoding helps but not enough");
        println!("  The structural similarity is moderate; deeper structural encoding needed");
    } else {
        println!("\n  VERDICT: Structural encoding doesn't help here");
        println!("  The iterate structures are too different even with property-based encoding");
    }

    // Also test: are ALL atoms treated as identical slots?
    let loop_a = ProgramNode::iterate(
        ProgramNode::atom("x"),
        ProgramNode::atom("y"),
        ProgramNode::atom("z"),
    );
    let loop_b = ProgramNode::iterate(
        ProgramNode::atom("p"),
        ProgramNode::atom("q"),
        ProgramNode::atom("r"),
    );
    let slot_sim = encode_structural(&loop_a).similarity(&encode_structural(&loop_b));
    println!(
        "\n  Iterate(x,y,z) vs Iterate(p,q,r) structural: {:.4}",
        slot_sim
    );
    println!("  (Should be 1.0 if atoms are truly name-independent)");
}
