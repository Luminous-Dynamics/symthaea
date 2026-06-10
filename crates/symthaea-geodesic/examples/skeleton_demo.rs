// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Skeleton Synthesis Demo — watch topology-first code generation in action.
//!
//! Run: `cargo run -p symthaea-geodesic --example skeleton_demo`

use symthaea_geodesic::manifold::ProgramManifold;
use symthaea_geodesic::pdg::ProgramDependenceGraph;
use symthaea_geodesic::skeleton_synthesis::*;
use symthaea_geodesic::topology::BettiNumbers;
use symthaea_geodesic::topology::TopologicalFingerprint;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     GEODESIC CODE SYNTHESIS — Skeleton Demo                  ║");
    println!("║     Topology-first code generation                           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // === Demo 1: Linear function (no loops, no branches) ===
    demo(
        "Simple addition (β₁=0, no loops)",
        &BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        },
        &["add two numbers"],
    );

    // === Demo 2: Single loop (sorting) ===
    demo(
        "Bubble sort (β₁=2, nested loops)",
        &BettiNumbers {
            beta_0: 1,
            beta_1: 2,
            beta_2: 0,
        },
        &["sort", "compare and swap"],
    );

    // === Demo 3: Binary search (loop + branch) ===
    demo(
        "Binary search (β₁=1, loop with branch)",
        &BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        },
        &["binary search", "find element"],
    );

    // === Demo 4: Recursive fibonacci ===
    demo(
        "Fibonacci (β₁=1, recursive)",
        &BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        },
        &["recursive", "fibonacci"],
    );

    // === Demo 5: Map-filter chain ===
    demo(
        "Filter even numbers (β₁=1, iterator)",
        &BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        },
        &["filter", "keep even numbers"],
    );

    // === Demo 6: Reduce/fold ===
    demo(
        "Sum with fold (β₁=1, accumulator)",
        &BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        },
        &["reduce", "sum", "accumulate"],
    );

    // === Demo 7: Branch only (no loops) ===
    demo(
        "Classify value (β₁=0, branch only)",
        &BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        },
        &["if condition", "classify", "branch"],
    );

    // === Demo 8: PDG analysis of real code ===
    println!("\n{}", "=".repeat(60));
    println!("=== PDG + Topology on Real Code ===\n");

    let source = r#"
fn binary_search(arr: &[i32], target: i32) -> Option<usize> {
    let mut low = 0;
    let mut high = arr.len();
    while low < high {
        let mid = low + (high - low) / 2;
        if arr[mid] == target {
            return Some(mid);
        } else if arr[mid] < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    None
}
"#;

    println!("Source:\n{}", source.trim());
    let pdg = ProgramDependenceGraph::from_rust_source(source, "binary_search");
    println!(
        "\nPDG: {} nodes, {} edges",
        pdg.nodes.len(),
        pdg.edges.len()
    );
    println!("  Loops: {}", pdg.loop_count());
    println!("  Branches: {}", pdg.branch_count());

    let complex = pdg.to_simplicial_complex();
    let fingerprint = TopologicalFingerprint::from_complex(&complex);
    println!("\nTopological Fingerprint:");
    println!("  β₀ = {} (connected components)", fingerprint.betti.beta_0);
    println!("  β₁ = {} (cycles/loops)", fingerprint.betti.beta_1);
    println!("  β₂ = {} (voids/dead code)", fingerprint.betti.beta_2);
    println!(
        "  χ  = {} (Euler characteristic)",
        fingerprint.euler_characteristic
    );

    // === Demo 9: Active Inference synthesis ===
    println!("\n{}", "=".repeat(60));
    println!("=== Active Inference Synthesis ===\n");

    let manifold = ProgramManifold::new();
    let betti = BettiNumbers {
        beta_0: 1,
        beta_1: 1,
        beta_2: 0,
    };
    let result =
        active_inference_synthesize(&betti, &["iterate", "accumulate sum"], &manifold, None, 5);

    println!("Topology satisfied: {}", result.topology_satisfied);
    println!("Topology signature: β₁={}", result.topology.delta_beta_1);
    println!("Manifold fills: {}", result.manifold_fills);
    println!("Refinements: {}", result.refinements);
    if let Some(ref code) = result.emitted_code {
        println!("\nEmitted code (unfilled slots shown as todo!):");
        println!("{}", code);
    } else {
        println!("\nNo code emitted (unfilled slots remain)");
        println!("Unfilled slots: {}", result.skeleton.unfilled_count());
    }

    // === Demo 10: Fill slots manually and emit ===
    println!("\n{}", "=".repeat(60));
    println!("=== Manual Slot Filling + Emit ===\n");

    let betti = BettiNumbers {
        beta_0: 1,
        beta_1: 1,
        beta_2: 0,
    };
    let mut skeleton = build_skeleton_from_topology(&betti, &["sum", "accumulate"]);
    println!(
        "Skeleton before filling ({} unfilled slots):",
        skeleton.unfilled_count()
    );

    // Fill the slots with real expressions
    fn fill_all(c: &mut SkeletonCombinator, fills: &mut std::collections::VecDeque<&str>) {
        match c {
            SkeletonCombinator::Sequence(steps) => {
                for s in steps.iter_mut() {
                    fill_all(s, fills);
                }
            }
            SkeletonCombinator::Iterate {
                init,
                condition,
                body,
            } => {
                if !init.is_filled() {
                    if let Some(f) = fills.pop_front() {
                        init.fill(f);
                    }
                }
                if !condition.is_filled() {
                    if let Some(f) = fills.pop_front() {
                        condition.fill(f);
                    }
                }
                fill_all(body, fills);
            }
            SkeletonCombinator::Reduce {
                operation,
                initial,
                collection,
            } => {
                if !operation.is_filled() {
                    if let Some(f) = fills.pop_front() {
                        operation.fill(f);
                    }
                }
                if !initial.is_filled() {
                    if let Some(f) = fills.pop_front() {
                        initial.fill(f);
                    }
                }
                if !collection.is_filled() {
                    if let Some(f) = fills.pop_front() {
                        collection.fill(f);
                    }
                }
            }
            SkeletonCombinator::Leaf(slot) => {
                if !slot.is_filled() {
                    if let Some(f) = fills.pop_front() {
                        slot.fill(f);
                    }
                }
            }
            _ => {}
        }
    }

    let mut fills: std::collections::VecDeque<&str> = [
        "let mut total = 0i64;", // init
        "|acc, x| acc + x",      // operation (for reduce) or...
        "0i64",                  // initial
        "numbers",               // collection
        "total",                 // return
    ]
    .iter()
    .copied()
    .collect();

    fill_all(&mut skeleton, &mut fills);
    println!(
        "After filling ({} unfilled slots):",
        skeleton.unfilled_count()
    );

    if let Some(code) = skeleton.emit_rust(1) {
        println!("\n    fn sum(numbers: &[i64]) -> i64 {{");
        println!("{}", code);
        println!("    }}");
    } else {
        println!(
            "Could not emit — {} slots still unfilled",
            skeleton.unfilled_count()
        );
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Demo complete — topology-first code generation works!     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

fn demo(title: &str, betti: &BettiNumbers, hints: &[&str]) {
    println!("{}", "=".repeat(60));
    println!("=== {} ===", title);
    println!(
        "Target: β₀={}, β₁={}, β₂={}",
        betti.beta_0, betti.beta_1, betti.beta_2
    );
    println!("Hints: {:?}\n", hints);

    let skeleton = build_skeleton_from_topology(betti, hints);
    let sig = skeleton.topological_signature();

    println!("Generated skeleton:");
    println!(
        "  Topology: Δβ₁={} (loops), nodes={}, edges={}",
        sig.delta_beta_1, sig.node_count, sig.edge_count
    );
    println!("  Unfilled slots: {}", skeleton.unfilled_count());
    println!(
        "  Topology matches target: {}\n",
        sig.delta_beta_1 == betti.beta_1 as i32
    );
}
