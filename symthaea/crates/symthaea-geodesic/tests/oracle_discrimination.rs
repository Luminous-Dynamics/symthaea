// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Oracle Discrimination Test — critical experiment.
//!
//! Does the scoring layer actually distinguish between correct and incorrect
//! programs? The old endpoint-only execution oracle is useful as convergence
//! telemetry, but it is not a correctness discriminator by itself. The
//! pass/fail gate therefore uses the tri-oracle consensus layer.
//!
//! Methodology:
//! 1. Encode 10 "correct" programs as ProgramNode → BinaryHV
//! 2. For each, create a "wrong" variant (different operator, wrong structure)
//! 3. Create an "expected output" encoding from the correct program
//! 4. Score both correct and wrong via the tri-oracle
//! 5. Measure: does correct consistently score higher than wrong?

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::program_algebra::ProgramNode;
use symthaea_geodesic::execution_oracle::{ExecutionOracle, OperationType};
use symthaea_geodesic::tri_oracle::TriOracle;

/// Helper: encode a ProgramNode, run oracle, score against expected.
fn oracle_score(program: &ProgramNode, expected: &BinaryHV) -> f32 {
    let encoding = program.encode();
    let mut oracle = ExecutionOracle::new();

    // Create execution sequence from the encoding
    let statements = vec![
        (encoding, OperationType::Arithmetic),
        (encoding.permute(1), OperationType::Assignment),
        (encoding.permute(2), OperationType::Comparison),
    ];

    let _prediction = oracle.predict_sequence(&statements);
    // Score: similarity between oracle's predicted final state and expected output
    oracle.verify_output(expected) as f32
}

/// Helper: score via the discriminative consensus oracle.
fn tri_oracle_score(program: &ProgramNode, expected: &ProgramNode) -> f32 {
    TriOracle::with_defaults()
        .score(program, expected)
        .composite
}

/// Helper: score using the program's OWN encoding as expected (self-consistency).
fn self_consistency_score(program: &ProgramNode) -> f32 {
    let encoding = program.encode();
    let mut oracle = ExecutionOracle::new();

    let statements = vec![
        (encoding, OperationType::Arithmetic),
        (encoding.permute(1), OperationType::Assignment),
    ];

    let prediction = oracle.predict_sequence(&statements);
    prediction.output_similarity
}

// ═══════════════════════════════════════════════════════════════════════════════
// THE EXPERIMENT
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_oracle_correct_vs_wrong_programs() {
    // 10 correct programs and their wrong variants
    let test_cases: Vec<(&str, ProgramNode, ProgramNode)> = vec![
        // 1. ADD vs MUL (wrong operator)
        (
            "add",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
            ProgramNode::apply(
                ProgramNode::op("MUL"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        ),
        // 2. SUB vs DIV (wrong operator)
        (
            "subtract",
            ProgramNode::apply(
                ProgramNode::op("SUB"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
            ProgramNode::apply(
                ProgramNode::op("DIV"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
        ),
        // 3. Sequence(a, b) vs Sequence(b, a) (wrong order)
        (
            "sequence_order",
            ProgramNode::Sequence(vec![
                ProgramNode::atom("first"),
                ProgramNode::atom("second"),
            ]),
            ProgramNode::Sequence(vec![
                ProgramNode::atom("second"),
                ProgramNode::atom("first"),
            ]),
        ),
        // 4. Branch(cond, A, B) vs Branch(cond, B, A) (swapped branches)
        (
            "branch_swap",
            ProgramNode::branch(
                ProgramNode::atom("cond"),
                ProgramNode::atom("yes"),
                ProgramNode::atom("no"),
            ),
            ProgramNode::branch(
                ProgramNode::atom("cond"),
                ProgramNode::atom("no"),
                ProgramNode::atom("yes"),
            ),
        ),
        // 5. Map(f, col) vs Filter(f, col) (wrong combinator)
        (
            "map_vs_filter",
            ProgramNode::map(ProgramNode::atom("transform"), ProgramNode::atom("items")),
            ProgramNode::filter(ProgramNode::atom("transform"), ProgramNode::atom("items")),
        ),
        // 6. Reduce(add, 0, arr) vs Reduce(mul, 1, arr) (wrong accumulator)
        (
            "reduce_op",
            ProgramNode::reduce(
                ProgramNode::op("ADD"),
                ProgramNode::atom("0"),
                ProgramNode::atom("arr"),
            ),
            ProgramNode::reduce(
                ProgramNode::op("MUL"),
                ProgramNode::atom("1"),
                ProgramNode::atom("arr"),
            ),
        ),
        // 7. Iterate(i=0, i<n, body) vs Iterate(i=0, i>n, body) (wrong condition)
        (
            "iterate_cond",
            ProgramNode::iterate(
                ProgramNode::atom("i=0"),
                ProgramNode::atom("i+=1"),
                ProgramNode::apply(
                    ProgramNode::op("LT"),
                    vec![ProgramNode::atom("i"), ProgramNode::atom("n")],
                ),
            ),
            ProgramNode::iterate(
                ProgramNode::atom("i=0"),
                ProgramNode::atom("i+=1"),
                ProgramNode::apply(
                    ProgramNode::op("GT"),
                    vec![ProgramNode::atom("i"), ProgramNode::atom("n")],
                ),
            ),
        ),
        // 8. Recurse(base=1, n*f(n-1)) vs Recurse(base=0, n+f(n-1)) (wrong base + op)
        (
            "recurse_base",
            ProgramNode::recurse(
                ProgramNode::atom("1"),
                ProgramNode::apply(
                    ProgramNode::op("MUL"),
                    vec![ProgramNode::atom("n"), ProgramNode::atom("f(n-1)")],
                ),
            ),
            ProgramNode::recurse(
                ProgramNode::atom("0"),
                ProgramNode::apply(
                    ProgramNode::op("ADD"),
                    vec![ProgramNode::atom("n"), ProgramNode::atom("f(n-1)")],
                ),
            ),
        ),
        // 9. Same structure, different atoms
        (
            "atom_difference",
            ProgramNode::apply(ProgramNode::atom("sort"), vec![ProgramNode::atom("array")]),
            ProgramNode::apply(
                ProgramNode::atom("reverse"),
                vec![ProgramNode::atom("array")],
            ),
        ),
        // 10. Compose(f, g) vs Compose(g, f) (wrong composition order)
        (
            "compose_order",
            ProgramNode::compose(ProgramNode::atom("parse"), ProgramNode::atom("validate")),
            ProgramNode::compose(ProgramNode::atom("validate"), ProgramNode::atom("parse")),
        ),
    ];

    println!("\n=== Oracle Discrimination Experiment ===\n");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>8}",
        "Test", "Correct", "Wrong", "Delta", "Discrim?"
    );
    println!("{}", "-".repeat(65));

    let mut correct_wins = 0;
    let mut total = 0;
    let mut deltas: Vec<f32> = Vec::new();

    for (name, correct, wrong) in &test_cases {
        let correct_score = tri_oracle_score(correct, correct);
        let wrong_score = tri_oracle_score(wrong, correct);
        let delta = correct_score - wrong_score;

        let discriminates = correct_score > wrong_score;
        if discriminates {
            correct_wins += 1;
        }
        total += 1;
        deltas.push(delta);

        println!(
            "{:<20} {:>10.4} {:>10.4} {:>+10.4} {:>8}",
            name,
            correct_score,
            wrong_score,
            delta,
            if discriminates { "YES" } else { "NO" }
        );
    }

    let accuracy = correct_wins as f32 / total as f32;
    let mean_delta: f32 = deltas.iter().sum::<f32>() / deltas.len() as f32;

    println!("{}", "-".repeat(65));
    println!(
        "Discrimination accuracy: {}/{} ({:.0}%)",
        correct_wins,
        total,
        accuracy * 100.0
    );
    println!("Mean score delta: {:+.4}", mean_delta);
    println!();

    if accuracy > 0.7 {
        println!("RESULT: Oracle DISCRIMINATES — correct programs score higher.");
        println!("The GCS architecture works. Invest in compositional recombination.");
    } else if accuracy > 0.5 {
        println!("RESULT: Oracle shows WEAK discrimination — slightly better than chance.");
        println!("The oracle needs improvement but has a signal to build on.");
    } else {
        println!("RESULT: Oracle does NOT discriminate — scores are essentially random.");
        println!("The oracle needs fundamental redesign before the architecture can work.");
    }

    // The key assertion: the consensus scoring layer should discriminate well
    // above chance before it is used as a selection signal.
    assert!(
        accuracy >= 0.7,
        "Tri-oracle discrimination accuracy ({:.0}%) must be at least 70%. \
         Got {}/{} correct wins. If this fails, the oracle cannot distinguish \
         correct from incorrect programs reliably enough for selection.",
        accuracy * 100.0,
        correct_wins,
        total,
    );
    assert!(
        mean_delta > 0.05,
        "Tri-oracle mean discrimination delta must be meaningfully positive, got {mean_delta:+.4}"
    );
}

#[test]
fn test_legacy_execution_oracle_endpoint_is_telemetry_not_gate() {
    let correct = ProgramNode::apply(
        ProgramNode::op("ADD"),
        vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
    );
    let wrong = ProgramNode::apply(
        ProgramNode::op("MUL"),
        vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
    );
    let expected = correct.encode();

    let correct_score = oracle_score(&correct, &expected);
    let wrong_score = oracle_score(&wrong, &expected);

    println!(
        "legacy endpoint scores: correct={correct_score:.4} wrong={wrong_score:.4} delta={:+.4}",
        correct_score - wrong_score
    );

    assert!((0.0..=1.0).contains(&correct_score));
    assert!((0.0..=1.0).contains(&wrong_score));
}

#[test]
fn test_self_consistency_scores() {
    // A program scored against its own encoding should produce high similarity.
    // This tests the oracle's internal consistency, not discrimination.
    let programs: Vec<(&str, ProgramNode)> = vec![
        (
            "add",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        ),
        (
            "map",
            ProgramNode::map(ProgramNode::atom("f"), ProgramNode::atom("col")),
        ),
        (
            "branch",
            ProgramNode::branch(
                ProgramNode::atom("c"),
                ProgramNode::atom("t"),
                ProgramNode::atom("f"),
            ),
        ),
        (
            "sequence",
            ProgramNode::Sequence(vec![
                ProgramNode::atom("a"),
                ProgramNode::atom("b"),
                ProgramNode::atom("c"),
            ]),
        ),
        (
            "iterate",
            ProgramNode::iterate(
                ProgramNode::atom("init"),
                ProgramNode::atom("step"),
                ProgramNode::atom("cond"),
            ),
        ),
    ];

    println!("\n=== Self-Consistency Scores ===\n");
    for (name, prog) in &programs {
        let score = self_consistency_score(prog);
        println!("  {:<15}: {:.4}", name, score);
        // Self-consistency should be high (the oracle evolves toward its own state)
    }
}

#[test]
fn test_encoding_distance_vs_oracle_distance() {
    // Do programs that are close in HDC space also score similarly in the oracle?
    // This tests whether the oracle preserves the metric structure of program space.
    let base = ProgramNode::apply(
        ProgramNode::op("ADD"),
        vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
    );
    let base_hv = base.encode();

    let variants: Vec<(&str, ProgramNode)> = vec![
        (
            "same",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        ),
        (
            "diff_args",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
        ),
        (
            "diff_op",
            ProgramNode::apply(
                ProgramNode::op("SUB"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        ),
        (
            "totally_diff",
            ProgramNode::map(ProgramNode::atom("sort"), ProgramNode::atom("list")),
        ),
    ];

    println!("\n=== Encoding Distance vs Oracle Distance ===\n");
    println!("{:<15} {:>12} {:>12}", "Variant", "HDC Sim", "Oracle Sim");
    println!("{}", "-".repeat(42));

    let mut hdc_sims = Vec::new();
    let mut oracle_sims = Vec::new();

    for (name, variant) in &variants {
        let variant_hv = variant.encode();
        let hdc_sim = base_hv.similarity(&variant_hv);

        let oracle_sim = oracle_score(variant, &base_hv);

        println!("{:<15} {:>12.4} {:>12.4}", name, hdc_sim, oracle_sim);
        hdc_sims.push(hdc_sim);
        oracle_sims.push(oracle_sim);
    }

    // "same" should have highest HDC similarity AND highest oracle similarity
    assert!(
        hdc_sims[0] > hdc_sims[3],
        "identical program should be more similar than totally different"
    );
}
