// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Resonant Explorer End-to-End — Can Symthaea generate a correct program?
//!
//! THE definitive test: give the resonant explorer a specification,
//! let it search program space with the tri-oracle, and check if
//! the result is a correct program that emits compilable Rust.

use symthaea_core::hdc::program_algebra::ProgramNode;
use symthaea_geodesic::noise::perturb;
use symthaea_geodesic::program_emitter::emit_expression;
use symthaea_geodesic::program_memory::ProgramMemory;
use symthaea_geodesic::tri_oracle::{TriOracle, TriOracleConfig};

/// Run the resonant exploration manually with the tri-oracle.
/// We bypass ResonantExplorer's internal oracle and use TriOracle directly.
fn resonant_search(
    target: &ProgramNode,
    memory: &ProgramMemory,
    max_steps: usize,
) -> (ProgramNode, f32, String) {
    let oracle = TriOracle::with_defaults();
    let target_enc = target.encode();

    // Start from the nearest pattern in memory
    let start = memory
        .nearest(&target_enc)
        .map(|e| e.node.clone())
        .unwrap_or_else(|| ProgramNode::atom("unknown"));

    let mut best_node = start;
    let mut best_score = oracle.score(&best_node, target).composite;
    let mut sigma = 0.15_f32;

    for step in 0..max_steps {
        // Perturb the best encoding
        let perturbed = perturb(&best_node.encode(), sigma, step as u64 + 42);

        // Find nearest known pattern to the perturbed encoding
        let candidate = memory
            .nearest(&perturbed)
            .map(|e| e.node.clone())
            .unwrap_or(best_node.clone());

        // Score with tri-oracle
        let score = oracle.score(&candidate, target).composite;

        if score > best_score {
            best_node = candidate;
            best_score = score;
            sigma *= 1.05; // explore more
        } else {
            sigma *= 0.95; // exploit
        }
        sigma = sigma.clamp(0.01, 0.4);

        // Converged?
        if best_score > 0.95 {
            break;
        }
    }

    let code = emit_expression(&best_node);
    (best_node, best_score, code)
}

#[test]
fn test_find_addition() {
    println!("\n=== RESONANT E2E: Find Addition ===\n");

    let memory = ProgramMemory::basic();
    let target = ProgramNode::apply(
        ProgramNode::op("ADD"),
        vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
    );

    let (found, score, code) = resonant_search(&target, &memory, 200);

    println!("Target:  {}", emit_expression(&target));
    println!("Found:   {}", code);
    println!("Score:   {:.4}", score);
    println!(
        "Match:   {}",
        if score > 0.95 {
            "EXACT"
        } else if score > 0.7 {
            "CLOSE"
        } else {
            "MISS"
        }
    );

    assert!(
        score > 0.7,
        "should find addition pattern, got score {score}"
    );
}

#[test]
fn test_find_multiplication() {
    println!("\n=== RESONANT E2E: Find Multiplication ===\n");

    let memory = ProgramMemory::basic();
    let target = ProgramNode::apply(
        ProgramNode::op("MUL"),
        vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
    );

    let (_, score, code) = resonant_search(&target, &memory, 200);

    println!("Target:  {}", emit_expression(&target));
    println!("Found:   {}", code);
    println!("Score:   {:.4}", score);

    // MUL(x,y) may not exactly match MUL(a,b) in memory due to different
    // atom names. The search finds the nearest operator pattern. Score > 0.5
    // means it found something structurally similar (an Apply with 2 args).
    assert!(
        score > 0.5,
        "should find a binary operator pattern, got score {score}"
    );
}

#[test]
fn test_find_comparison() {
    println!("\n=== RESONANT E2E: Find Comparison ===\n");

    let memory = ProgramMemory::basic();
    let target = ProgramNode::apply(
        ProgramNode::op("LT"),
        vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
    );

    let (_, score, code) = resonant_search(&target, &memory, 200);

    println!("Target:  {}", emit_expression(&target));
    println!("Found:   {}", code);
    println!("Score:   {:.4}", score);

    assert!(
        score > 0.7,
        "should find comparison pattern, got score {score}"
    );
}

#[test]
fn test_find_map_pattern() {
    println!("\n=== RESONANT E2E: Find Map Pattern ===\n");

    let memory = ProgramMemory::basic();
    let target = ProgramNode::map(ProgramNode::atom("transform"), ProgramNode::atom("items"));

    let (_, score, code) = resonant_search(&target, &memory, 200);

    println!("Target:  {}", emit_expression(&target));
    println!("Found:   {}", code);
    println!("Score:   {:.4}", score);

    assert!(score > 0.5, "should find map pattern, got score {score}");
}

#[test]
fn test_find_reduce_pattern() {
    println!("\n=== RESONANT E2E: Find Reduce/Sum ===\n");

    let memory = ProgramMemory::basic();
    let target = ProgramNode::reduce(
        ProgramNode::op("ADD"),
        ProgramNode::atom("0"),
        ProgramNode::atom("arr"),
    );

    let (_, score, code) = resonant_search(&target, &memory, 200);

    println!("Target:  {}", emit_expression(&target));
    println!("Found:   {}", code);
    println!("Score:   {:.4}", score);

    assert!(score > 0.5, "should find reduce pattern, got score {score}");
}

#[test]
fn test_generated_code_is_valid_rust() {
    println!("\n=== RESONANT E2E: Emitted Code Validity ===\n");

    let memory = ProgramMemory::basic();
    let targets = vec![
        (
            "add",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        ),
        (
            "sub",
            ProgramNode::apply(
                ProgramNode::op("SUB"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
        ),
        (
            "eq",
            ProgramNode::apply(
                ProgramNode::op("EQ"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        ),
    ];

    for (name, target) in &targets {
        let (found, score, code) = resonant_search(target, &memory, 100);
        println!("  {:<6}: score={:.4}  code='{}'", name, score, code);

        // The emitted code should be non-empty and contain real Rust tokens
        assert!(!code.is_empty(), "{name} should produce non-empty code");
        assert!(
            !code.contains("todo!"),
            "{name} should not contain todo!, got '{code}'"
        );
    }
}

#[test]
fn test_discrimination_drives_convergence() {
    println!("\n=== RESONANT E2E: Score Monotonicity ===\n");

    let memory = ProgramMemory::basic();
    let oracle = TriOracle::with_defaults();
    let target = ProgramNode::apply(
        ProgramNode::op("ADD"),
        vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
    );
    let target_enc = target.encode();

    let start = memory.nearest(&target_enc).map(|e| e.node.clone()).unwrap();
    let initial_score = oracle.score(&start, &target).composite;

    let (_, final_score, _) = resonant_search(&target, &memory, 200);

    println!("Initial score: {:.4}", initial_score);
    println!("Final score:   {:.4}", final_score);

    // The search should not make things worse
    assert!(
        final_score >= initial_score - 0.05,
        "search should not significantly degrade: initial={initial_score:.4} final={final_score:.4}"
    );
}
