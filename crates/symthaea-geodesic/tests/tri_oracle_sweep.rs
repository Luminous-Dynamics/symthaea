// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tri-Oracle Weight Sweep — find optimal oracle configuration.
//!
//! Tests ALL 10 correct/wrong pairs across 27 weight configurations
//! to find the combination that maximizes discrimination accuracy.

use symthaea_core::hdc::program_algebra::ProgramNode;
use symthaea_geodesic::tri_oracle::{TriOracle, TriOracleConfig};

fn test_cases() -> Vec<(&'static str, ProgramNode, ProgramNode)> {
    vec![
        (
            "add_vs_mul",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
            ProgramNode::apply(
                ProgramNode::op("MUL"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        ),
        (
            "sub_vs_div",
            ProgramNode::apply(
                ProgramNode::op("SUB"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
            ProgramNode::apply(
                ProgramNode::op("DIV"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
        ),
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
        (
            "map_vs_filter",
            ProgramNode::map(ProgramNode::atom("transform"), ProgramNode::atom("items")),
            ProgramNode::filter(ProgramNode::atom("transform"), ProgramNode::atom("items")),
        ),
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
        (
            "atom_difference",
            ProgramNode::apply(ProgramNode::atom("sort"), vec![ProgramNode::atom("array")]),
            ProgramNode::apply(
                ProgramNode::atom("reverse"),
                vec![ProgramNode::atom("array")],
            ),
        ),
        (
            "compose_order",
            ProgramNode::compose(ProgramNode::atom("parse"), ProgramNode::atom("validate")),
            ProgramNode::compose(ProgramNode::atom("validate"), ProgramNode::atom("parse")),
        ),
    ]
}

fn run_with_config(config: &TriOracleConfig) -> (usize, usize, f32) {
    let oracle = TriOracle::new(config.clone());
    let cases = test_cases();
    let mut wins = 0;
    let mut total = 0;
    let mut total_delta = 0.0_f32;

    for (_, correct, wrong) in &cases {
        let correct_score = oracle.score(correct, correct).composite;
        let wrong_score = oracle.score(wrong, correct).composite;
        if correct_score > wrong_score {
            wins += 1;
        }
        total += 1;
        total_delta += correct_score - wrong_score;
    }

    (wins, total, total_delta / total as f32)
}

#[test]
fn test_weight_sweep() {
    println!("\n{}", "=".repeat(80));
    println!("=== TRI-ORACLE WEIGHT SWEEP ===");
    println!("10 test cases x 27 weight configurations\n");

    let structural_weights = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0];
    let topological_weights = [0.0, 0.1, 0.2, 0.3];
    let trajectory_weights = [0.0, 0.1, 0.2, 0.3, 0.5];
    let trajectory_steps = [1, 5, 10, 20];

    println!(
        "{:<8} {:<8} {:<8} {:<6} {:>6} {:>8} {:>10}",
        "Struct", "Topo", "Traj", "Steps", "Wins", "Acc%", "MeanDelta"
    );
    println!("{}", "-".repeat(65));

    let mut best_accuracy = 0;
    let mut best_config = TriOracleConfig::default();
    let mut best_delta = 0.0_f32;
    let mut configs_tested = 0;

    for &sw in &structural_weights {
        for &tw in &topological_weights {
            for &trw in &trajectory_weights {
                // Normalize weights to sum to 1.0
                let sum = sw + tw + trw;
                if sum < 0.01 {
                    continue;
                }
                let sw_n = sw / sum;
                let tw_n = tw / sum;
                let trw_n = trw / sum;

                for &steps in &trajectory_steps {
                    // Skip redundant configs (if trajectory weight is 0, steps don't matter)
                    if trw < 0.01 && steps > 1 {
                        continue;
                    }

                    let config = TriOracleConfig {
                        structural_weight: sw_n,
                        topological_weight: tw_n,
                        trajectory_weight: trw_n,
                        trajectory_steps: steps,
                    };

                    let (wins, total, mean_delta) = run_with_config(&config);
                    configs_tested += 1;

                    if wins > best_accuracy || (wins == best_accuracy && mean_delta > best_delta) {
                        best_accuracy = wins;
                        best_config = config.clone();
                        best_delta = mean_delta;

                        println!(
                            "{:<8.2} {:<8.2} {:<8.2} {:<6} {:>4}/{} {:>6.0}% {:>+10.4} ← NEW BEST",
                            sw_n,
                            tw_n,
                            trw_n,
                            steps,
                            wins,
                            total,
                            wins as f32 / total as f32 * 100.0,
                            mean_delta
                        );
                    }
                }
            }
        }
    }

    println!("{}", "-".repeat(65));
    println!("\nConfigs tested: {}", configs_tested);
    println!("\n=== BEST CONFIGURATION ===");
    println!("Structural weight: {:.2}", best_config.structural_weight);
    println!("Topological weight: {:.2}", best_config.topological_weight);
    println!("Trajectory weight: {:.2}", best_config.trajectory_weight);
    println!("Trajectory steps: {}", best_config.trajectory_steps);
    println!(
        "Accuracy: {}/10 ({:.0}%)",
        best_accuracy,
        best_accuracy as f32 / 10.0 * 100.0
    );
    println!("Mean delta: {:+.4}", best_delta);

    // Print per-case breakdown with best config
    println!("\n=== PER-CASE BREAKDOWN (best config) ===\n");
    let oracle = TriOracle::new(best_config.clone());
    let cases = test_cases();
    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "Case", "C.struct", "C.topo", "C.traj", "W.struct", "W.topo", "W.traj", "Win?"
    );
    println!("{}", "-".repeat(85));

    for (name, correct, wrong) in &cases {
        let c = oracle.score(correct, correct);
        let w = oracle.score(wrong, correct);
        println!(
            "{:<20} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>6}",
            name,
            c.structural,
            c.topological,
            c.trajectory,
            w.structural,
            w.topological,
            w.trajectory,
            if c.composite > w.composite {
                "YES"
            } else {
                "NO"
            }
        );
    }

    // The sweep must achieve at least 60% accuracy
    assert!(
        best_accuracy >= 6,
        "Best configuration should achieve at least 60% accuracy, got {}/10",
        best_accuracy
    );
}

#[test]
fn test_individual_oracle_accuracy() {
    println!("\n=== INDIVIDUAL ORACLE ACCURACY ===\n");

    let cases = test_cases();

    // Test each oracle individually (weight=1.0 for one, 0.0 for others)
    let configs = [
        (
            "Structural only",
            TriOracleConfig {
                structural_weight: 1.0,
                topological_weight: 0.0,
                trajectory_weight: 0.0,
                trajectory_steps: 1,
            },
        ),
        (
            "Topological only",
            TriOracleConfig {
                structural_weight: 0.0,
                topological_weight: 1.0,
                trajectory_weight: 0.0,
                trajectory_steps: 1,
            },
        ),
        (
            "Trajectory only (5 steps)",
            TriOracleConfig {
                structural_weight: 0.0,
                topological_weight: 0.0,
                trajectory_weight: 1.0,
                trajectory_steps: 5,
            },
        ),
        (
            "Trajectory only (20 steps)",
            TriOracleConfig {
                structural_weight: 0.0,
                topological_weight: 0.0,
                trajectory_weight: 1.0,
                trajectory_steps: 20,
            },
        ),
        ("Default (0.5/0.2/0.3)", TriOracleConfig::default()),
    ];

    for (name, config) in &configs {
        let (wins, total, delta) = run_with_config(config);
        println!(
            "{:<30} {}/{}  ({:.0}%)  mean_delta={:+.4}",
            name,
            wins,
            total,
            wins as f32 / total as f32 * 100.0,
            delta
        );
    }
}
