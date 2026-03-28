// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hard Generation Tests — can Symthaea generate multi-step programs?
//!
//! These test cases go beyond simple binary expressions (a+b) to test:
//! 1. Multi-step sequences (init → loop → return)
//! 2. Loops with branches inside
//! 3. Recursive functions
//! 4. Higher-order compositions (filter → map)
//! 5. Programs with variable name independence

use symthaea_core::hdc::program_algebra::ProgramNode;
use symthaea_geodesic::noise::perturb;
use symthaea_geodesic::program_emitter::emit_expression;
use symthaea_geodesic::program_memory::ProgramMemory;
use symthaea_geodesic::tri_oracle::TriOracle;

/// Run resonant search with tri-oracle scoring.
fn search(target: &ProgramNode, memory: &ProgramMemory, max_steps: usize) -> (ProgramNode, f32, String) {
    let oracle = TriOracle::with_defaults();
    let target_enc = target.encode();

    let start = memory.nearest(&target_enc)
        .map(|e| e.node.clone())
        .unwrap_or_else(|| ProgramNode::atom("unknown"));

    let mut best = start;
    let mut best_score = oracle.score(&best, target).composite;
    let mut sigma = 0.12_f32;

    for step in 0..max_steps {
        let perturbed = perturb(&best.encode(), sigma, step as u64 + 7);
        let candidate = memory.nearest(&perturbed)
            .map(|e| e.node.clone())
            .unwrap_or(best.clone());

        let score = oracle.score(&candidate, target).composite;
        if score > best_score {
            best = candidate;
            best_score = score;
            sigma = (sigma * 1.05).min(0.3);
        } else {
            sigma = (sigma * 0.97).max(0.02);
        }

        if best_score > 0.95 { break; }
    }

    (best.clone(), best_score, emit_expression(&best))
}

#[test]
fn test_hard_mul_with_xy_names() {
    println!("\n=== HARD: MUL(x,y) — variable name independence ===\n");
    let memory = ProgramMemory::basic();
    let target = ProgramNode::apply(
        ProgramNode::op("MUL"),
        vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
    );
    let (_, score, code) = search(&target, &memory, 300);
    println!("Target: x * y");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    // With xy variants in memory, should now find exact match
    assert!(score > 0.9, "should find MUL(x,y) now that it's in memory, got {score}");
}

#[test]
fn test_hard_sum_loop() {
    println!("\n=== HARD: Sum loop (init → iterate → return) ===\n");
    let memory = ProgramMemory::basic();
    let target = ProgramNode::Sequence(vec![
        ProgramNode::typed("sum", "INT"),
        ProgramNode::iterate(
            ProgramNode::atom("sum = 0"),
            ProgramNode::apply(ProgramNode::op("ADD"),
                vec![ProgramNode::atom("sum"), ProgramNode::atom("arr[i]")]),
            ProgramNode::apply(ProgramNode::op("LT"),
                vec![ProgramNode::atom("i"), ProgramNode::atom("len")]),
        ),
        ProgramNode::atom("sum"),
    ]);
    let (_, score, code) = search(&target, &memory, 300);
    println!("Target: sum_loop pattern");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    // The memory has "sum_loop" — should find it or something close
    assert!(score > 0.5, "should find a loop pattern, got {score}");
}

#[test]
fn test_hard_find_max() {
    println!("\n=== HARD: Find max (loop + branch) ===\n");
    let memory = ProgramMemory::basic();
    let target = ProgramNode::Sequence(vec![
        ProgramNode::atom("max = arr[0]"),
        ProgramNode::iterate(
            ProgramNode::atom("i = 1"),
            ProgramNode::branch(
                ProgramNode::apply(ProgramNode::op("GT"),
                    vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("max")]),
                ProgramNode::atom("max = arr[i]"),
                ProgramNode::atom("/* skip */"),
            ),
            ProgramNode::apply(ProgramNode::op("LT"),
                vec![ProgramNode::atom("i"), ProgramNode::atom("len")]),
        ),
        ProgramNode::atom("max"),
    ]);
    let (_, score, code) = search(&target, &memory, 300);
    println!("Target: find_max (loop+branch)");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    assert!(score > 0.5, "should find loop+branch pattern, got {score}");
}

#[test]
fn test_hard_fibonacci() {
    println!("\n=== HARD: Fibonacci (recursive) ===\n");
    let memory = ProgramMemory::basic();
    let target = ProgramNode::recurse(
        ProgramNode::branch(
            ProgramNode::apply(ProgramNode::op("LT"),
                vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")]),
            ProgramNode::atom("n"),
            ProgramNode::apply(ProgramNode::op("ADD"), vec![
                ProgramNode::apply(ProgramNode::atom("fib"),
                    vec![ProgramNode::apply(ProgramNode::op("SUB"),
                        vec![ProgramNode::atom("n"), ProgramNode::typed("1", "INT")])]),
                ProgramNode::apply(ProgramNode::atom("fib"),
                    vec![ProgramNode::apply(ProgramNode::op("SUB"),
                        vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")])]),
            ]),
        ),
        ProgramNode::atom("fib"),
    );
    let (_, score, code) = search(&target, &memory, 300);
    println!("Target: fibonacci (recursive)");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    assert!(score > 0.5, "should find recursive pattern, got {score}");
}

#[test]
fn test_hard_filter_map_chain() {
    println!("\n=== HARD: Filter + Map chain ===\n");
    let memory = ProgramMemory::basic();
    let target = ProgramNode::Sequence(vec![
        ProgramNode::filter(ProgramNode::atom("predicate"), ProgramNode::atom("items")),
        ProgramNode::map(ProgramNode::atom("transform"), ProgramNode::atom("filtered")),
    ]);
    let (_, score, code) = search(&target, &memory, 300);
    println!("Target: filter → map chain");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    assert!(score > 0.5, "should find filter+map chain, got {score}");
}

#[test]
fn test_hard_contains_search() {
    println!("\n=== HARD: Contains (loop + branch + early return) ===\n");
    let memory = ProgramMemory::basic();
    let target = ProgramNode::Sequence(vec![
        ProgramNode::iterate(
            ProgramNode::atom("i = 0"),
            ProgramNode::branch(
                ProgramNode::apply(ProgramNode::op("EQ"),
                    vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("target")]),
                ProgramNode::atom("return true"),
                ProgramNode::atom("i += 1"),
            ),
            ProgramNode::apply(ProgramNode::op("LT"),
                vec![ProgramNode::atom("i"), ProgramNode::atom("len")]),
        ),
        ProgramNode::atom("false"),
    ]);
    let (_, score, code) = search(&target, &memory, 300);
    println!("Target: contains (loop+branch+early return)");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    assert!(score > 0.5, "should find contains pattern, got {score}");
}

#[test]
fn test_hard_product_reduce() {
    println!("\n=== HARD: Product reduce (MUL accumulator) ===\n");
    let memory = ProgramMemory::basic();
    let target = ProgramNode::reduce(
        ProgramNode::op("MUL"),
        ProgramNode::typed("1", "INT"),
        ProgramNode::atom("arr"),
    );
    let (_, score, code) = search(&target, &memory, 300);
    println!("Target: product (fold with MUL)");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    // product_reduce is in memory — should find exact match
    assert!(score > 0.9, "should find product reduce, got {score}");
}

#[test]
fn test_summary() {
    println!("\n=== GENERATION SUMMARY ===\n");
    let memory = ProgramMemory::basic();
    println!("Memory size: {} patterns", memory.len());

    let tests: Vec<(&str, ProgramNode)> = vec![
        ("a + b", ProgramNode::apply(ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")])),
        ("x * y", ProgramNode::apply(ProgramNode::op("MUL"),
            vec![ProgramNode::atom("x"), ProgramNode::atom("y")])),
        ("a == b", ProgramNode::apply(ProgramNode::op("EQ"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")])),
        ("map(f,c)", ProgramNode::map(ProgramNode::atom("transform"), ProgramNode::atom("items"))),
        ("product", ProgramNode::reduce(ProgramNode::op("MUL"),
            ProgramNode::typed("1", "INT"), ProgramNode::atom("arr"))),
    ];

    println!("{:<15} {:>8} {:<30}", "Test", "Score", "Generated Code");
    println!("{}", "-".repeat(60));

    let mut exact = 0;
    let mut close = 0;
    let mut miss = 0;

    for (name, target) in &tests {
        let (_, score, code) = search(target, &memory, 200);
        let short_code: String = code.chars().take(28).collect();
        let status = if score > 0.95 { exact += 1; "EXACT" }
            else if score > 0.7 { close += 1; "CLOSE" }
            else { miss += 1; "MISS" };
        println!("{:<15} {:>8.4} {:<30} {}", name, score, short_code, status);
    }

    println!("{}", "-".repeat(60));
    println!("EXACT: {}  CLOSE: {}  MISS: {}  Total: {}", exact, close, miss, tests.len());
}
