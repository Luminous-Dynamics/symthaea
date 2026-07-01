// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Novel Generation Test — THE honest test.
//!
//! Can Symthaea generate a program that ISN'T in its memory?
//! If yes: genuine generation. If no: sophisticated retrieval.

use symthaea_core::hdc::program_algebra::ProgramNode;
use symthaea_geodesic::composer::compose_from_patterns;
use symthaea_geodesic::noise::perturb;
use symthaea_geodesic::program_emitter::emit_expression;
use symthaea_geodesic::program_memory::ProgramMemory;
use symthaea_geodesic::tri_oracle::TriOracle;

fn search(
    target: &ProgramNode,
    memory: &ProgramMemory,
    max_steps: usize,
) -> (ProgramNode, f32, String) {
    let oracle = TriOracle::with_defaults();
    let target_enc = target.encode();
    let start = memory
        .nearest(&target_enc)
        .map(|e| e.node.clone())
        .unwrap_or_else(|| ProgramNode::atom("unknown"));

    let mut best = start;
    let mut best_score = oracle.score(&best, target).composite;
    let mut sigma = 0.12_f32;

    // Try compositional recombination first — it may beat any single pattern
    if let Some((composed, comp_score)) = compose_from_patterns(target, memory, 5) {
        if comp_score > best_score {
            best = composed;
            best_score = comp_score;
        }
    }

    // Then run resonant search to refine further
    for step in 0..max_steps {
        let perturbed = perturb(&best.encode(), sigma, step as u64 + 7);
        let candidate = memory
            .nearest(&perturbed)
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
        if best_score > 0.95 {
            break;
        }
    }

    (best.clone(), best_score, emit_expression(&best))
}

#[test]
fn test_novel_second_largest() {
    println!("\n=== NOVEL: Second largest element (NOT in memory) ===\n");

    let memory = ProgramMemory::basic();

    // This program is NOT in the 30-pattern memory.
    // It requires: two tracking variables, a loop, and comparison logic.
    let target = ProgramNode::Sequence(vec![
        ProgramNode::atom("max1 = arr[0]"),
        ProgramNode::atom("max2 = arr[1]"),
        ProgramNode::iterate(
            ProgramNode::atom("i = 2"),
            ProgramNode::branch(
                ProgramNode::apply(
                    ProgramNode::op("GT"),
                    vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("max1")],
                ),
                ProgramNode::Sequence(vec![
                    ProgramNode::atom("max2 = max1"),
                    ProgramNode::atom("max1 = arr[i]"),
                ]),
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("GT"),
                        vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("max2")],
                    ),
                    ProgramNode::atom("max2 = arr[i]"),
                    ProgramNode::atom("/* skip */"),
                ),
            ),
            ProgramNode::apply(
                ProgramNode::op("LT"),
                vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
            ),
        ),
        ProgramNode::atom("max2"),
    ]);

    let (found, score, code) = search(&target, &memory, 500);

    println!("Target: second_largest (loop + nested branches + 2 vars)");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    println!();

    // What did it actually find?
    let target_code = emit_expression(&target);
    println!("Target code:\n{}\n", target_code);
    println!("Found code:\n{}\n", code);

    let is_exact = score > 0.95;
    let is_close = score > 0.7;
    let is_partial = score > 0.5;

    if is_exact {
        println!("RESULT: EXACT match — but this shouldn't happen (not in memory!)");
    } else if is_close {
        println!("RESULT: CLOSE — found a similar pattern, adapted partially");
    } else if is_partial {
        println!("RESULT: PARTIAL — found something structurally related");
    } else {
        println!("RESULT: MISS — returned nearest pattern, no composition");
    }

    // We don't assert a specific score — this test is observational.
    // The result tells us whether the system generates or retrieves.
    println!(
        "\nVerdict: score={:.4} → {}",
        score,
        if is_exact {
            "GENERATION (exact)"
        } else if is_close {
            "GENERATION (approximate)"
        } else if is_partial {
            "RETRIEVAL (nearest pattern)"
        } else {
            "RETRIEVAL (random nearest)"
        }
    );
}

#[test]
fn test_novel_reverse_array() {
    println!("\n=== NOVEL: Reverse array (NOT in memory) ===\n");

    let memory = ProgramMemory::basic();

    // Reverse: swap from ends toward middle. Not in memory.
    let target = ProgramNode::Sequence(vec![
        ProgramNode::atom("left = 0"),
        ProgramNode::apply(
            ProgramNode::op("SUB"),
            vec![
                ProgramNode::atom("right = len"),
                ProgramNode::typed("1", "INT"),
            ],
        ),
        ProgramNode::iterate(
            ProgramNode::atom("/* init */"),
            ProgramNode::Sequence(vec![
                ProgramNode::atom("swap(arr[left], arr[right])"),
                ProgramNode::atom("left += 1"),
                ProgramNode::atom("right -= 1"),
            ]),
            ProgramNode::apply(
                ProgramNode::op("LT"),
                vec![ProgramNode::atom("left"), ProgramNode::atom("right")],
            ),
        ),
    ]);

    let (_, score, code) = search(&target, &memory, 500);
    println!("Target: reverse_array (loop + swap + two pointers)");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    println!(
        "Verdict: {}",
        if score > 0.7 {
            "GENERATION"
        } else {
            "RETRIEVAL"
        }
    );
}

#[test]
fn test_novel_binary_search() {
    println!("\n=== NOVEL: Binary search (NOT in memory as complete program) ===\n");

    let memory = ProgramMemory::basic();

    let target = ProgramNode::Sequence(vec![
        ProgramNode::atom("low = 0"),
        ProgramNode::atom("high = len"),
        ProgramNode::iterate(
            ProgramNode::atom("/* init */"),
            ProgramNode::Sequence(vec![
                ProgramNode::apply(
                    ProgramNode::op("DIV"),
                    vec![
                        ProgramNode::apply(
                            ProgramNode::op("ADD"),
                            vec![ProgramNode::atom("low"), ProgramNode::atom("high")],
                        ),
                        ProgramNode::typed("2", "INT"),
                    ],
                ),
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("EQ"),
                        vec![ProgramNode::atom("arr[mid]"), ProgramNode::atom("target")],
                    ),
                    ProgramNode::atom("return mid"),
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("LT"),
                            vec![ProgramNode::atom("arr[mid]"), ProgramNode::atom("target")],
                        ),
                        ProgramNode::atom("low = mid + 1"),
                        ProgramNode::atom("high = mid"),
                    ),
                ),
            ]),
            ProgramNode::apply(
                ProgramNode::op("LT"),
                vec![ProgramNode::atom("low"), ProgramNode::atom("high")],
            ),
        ),
        ProgramNode::atom("None"),
    ]);

    let (_, score, code) = search(&target, &memory, 500);
    println!("Target: binary_search (loop + nested branches + mid calc)");
    println!("Found:  {}", code);
    println!("Score:  {:.4}", score);
    println!(
        "Verdict: {}",
        if score > 0.7 {
            "GENERATION"
        } else {
            "RETRIEVAL"
        }
    );
}

#[test]
fn test_novel_vs_known_comparison() {
    println!("\n=== NOVEL vs KNOWN — Side by Side ===\n");

    let memory = ProgramMemory::basic();

    let known = vec![
        (
            "add (KNOWN)",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        ),
        (
            "fibonacci (KNOWN)",
            ProgramNode::recurse(
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")],
                    ),
                    ProgramNode::atom("n"),
                    ProgramNode::apply(
                        ProgramNode::op("ADD"),
                        vec![
                            ProgramNode::apply(
                                ProgramNode::atom("fib"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("1", "INT")],
                                )],
                            ),
                            ProgramNode::apply(
                                ProgramNode::atom("fib"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")],
                                )],
                            ),
                        ],
                    ),
                ),
                ProgramNode::atom("fib"),
            ),
        ),
    ];

    let novel = vec![
        (
            "min element (NOVEL)",
            ProgramNode::Sequence(vec![
                ProgramNode::atom("min = arr[0]"),
                ProgramNode::iterate(
                    ProgramNode::atom("i = 1"),
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("LT"),
                            vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("min")],
                        ),
                        ProgramNode::atom("min = arr[i]"),
                        ProgramNode::atom("/* skip */"),
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                    ),
                ),
                ProgramNode::atom("min"),
            ]),
        ),
        (
            "power (NOVEL)",
            ProgramNode::Sequence(vec![
                ProgramNode::atom("result = 1"),
                ProgramNode::iterate(
                    ProgramNode::atom("i = 0"),
                    ProgramNode::apply(
                        ProgramNode::op("MUL"),
                        vec![ProgramNode::atom("result"), ProgramNode::atom("base")],
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("exp")],
                    ),
                ),
                ProgramNode::atom("result"),
            ]),
        ),
    ];

    println!("{:<25} {:>8} {:<30}", "Test", "Score", "Generated");
    println!("{}", "-".repeat(70));

    for (name, target) in known.iter().chain(novel.iter()) {
        let (_, score, code) = search(target, &memory, 300);
        let short: String = code.chars().take(28).collect();
        println!("{:<25} {:>8.4} {:<30}", name, score, short);
    }
}
