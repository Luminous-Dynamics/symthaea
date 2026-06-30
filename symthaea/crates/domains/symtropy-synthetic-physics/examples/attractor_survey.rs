// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Attractor Survey
//!
//! Tests all 5 update rules against 5 deterministic seeds for 500 ticks each,
//! using `GraphSafetyGuards::default()`, and prints:
//!
//! 1. A full result table (Rule | Seed | Ticks Completed | Attractor Class | Halted Early | Final Dim | Final Betti0)
//! 2. A summary: which rule/seed combos produced `UsefulEmergentManifold` and
//!    which rules are most stable overall.
//!
//! Run with:
//! ```
//! cargo run --example attractor_survey -p symtropy-synthetic-physics
//! ```

use symtropy_synthetic_physics::{AttractorClass, GraphSafetyGuards, UpdateRule, run_experiment};

const TICKS: usize = 500;
const SEEDS: &[u64] = &[42, 123, 777, 1337, 9999];

fn rules() -> Vec<UpdateRule> {
    vec![
        UpdateRule::NearestNeighborAttachment { nodes_per_tick: 1 },
        UpdateRule::TriangulationPressure { probability: 0.1 },
        UpdateRule::DegreeBalancingRemoval {
            degree_threshold: 4,
            probability: 0.3,
        },
        UpdateRule::CurvatureFlow { fraction: 0.2 },
        UpdateRule::FreeEnergyMinimization {
            candidates_per_tick: 10,
        },
    ]
}

/// Pretty-print an [`AttractorClass`] for the table column.
fn class_label(c: AttractorClass) -> &'static str {
    match c {
        AttractorClass::Unknown => "Unknown",
        AttractorClass::StableManifold => "StableManifold",
        AttractorClass::OscillatoryAttractor => "OscillatoryAttractor",
        AttractorClass::StrangeAttractorRisk => "StrangeAttractorRisk",
        AttractorClass::HairballExplosion => "HairballExplosion",
        AttractorClass::StringCollapse => "StringCollapse",
        AttractorClass::Fragmentation => "Fragmentation",
        AttractorClass::UsefulEmergentManifold => "UsefulEmergentManifold 🎯",
    }
}

/// Row of results for one (rule, seed) experiment.
struct Row {
    rule_name: &'static str,
    seed: u64,
    ticks_completed: usize,
    attractor_class: AttractorClass,
    halted_early: bool,
    final_dim: f64,
    final_betti0: usize,
}

fn main() {
    let guards = GraphSafetyGuards::default();
    let mut rows: Vec<Row> = Vec::new();

    // ── Run all experiments ──────────────────────────────────────────────────
    for rule in rules() {
        let rule_name = rule.name();
        for &seed in SEEDS {
            let result = run_experiment(rule.clone(), TICKS, seed, guards.clone());

            // Pull the last metrics snapshot from the ring buffer.
            let history_slice = result.history.as_slice();
            let (final_dim, final_betti0) = history_slice
                .last()
                .map(|m| (m.estimated_dimension, m.betti_0))
                .unwrap_or((0.0, 0));

            rows.push(Row {
                rule_name,
                seed,
                ticks_completed: result.ticks_completed,
                attractor_class: result.attractor_class,
                halted_early: result.halted_early,
                final_dim,
                final_betti0,
            });
        }
    }

    // ── Print table ──────────────────────────────────────────────────────────
    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                              ATTRACTOR SURVEY  –  5 rules × 5 seeds × 500 ticks                        ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ {:<34} | {:>6} | {:>15} | {:<26} | {:>12} | {:>9} | {:>11} ║",
        "Rule",
        "Seed",
        "TicksCompleted",
        "AttractorClass",
        "HaltedEarly",
        "FinalDim",
        "FinalBetti0"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    for row in &rows {
        println!(
            "║ {:<34} | {:>6} | {:>15} | {:<26} | {:>12} | {:>9.3} | {:>11} ║",
            row.rule_name,
            row.seed,
            row.ticks_completed,
            class_label(row.attractor_class),
            if row.halted_early { "YES" } else { "no" },
            row.final_dim,
            row.final_betti0,
        );
    }

    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );

    // ── Summary ──────────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");

    // 1. UsefulEmergentManifold hits
    let useful: Vec<&Row> = rows
        .iter()
        .filter(|r| r.attractor_class == AttractorClass::UsefulEmergentManifold)
        .collect();

    if useful.is_empty() {
        println!("\n  🔍  No runs produced UsefulEmergentManifold in this survey.");
    } else {
        println!("\n  🎯  UsefulEmergentManifold produced by:");
        for r in &useful {
            println!("        rule={:<34}  seed={}", r.rule_name, r.seed);
        }
    }

    // 2. Per-rule stability (% non-halted runs)
    println!("\n  📊  Rule stability (% runs that did NOT halt early):");
    for rule in rules() {
        let name = rule.name();
        let rule_rows: Vec<&Row> = rows.iter().filter(|r| r.rule_name == name).collect();
        let total = rule_rows.len();
        let completed = rule_rows.iter().filter(|r| !r.halted_early).count();
        let pct = 100.0 * completed as f64 / total as f64;
        println!(
            "        {:<34}  {}/{} = {:.0}%",
            name, completed, total, pct
        );
    }

    // 3. Most common attractor class per rule
    println!("\n  📋  Most common attractor class per rule:");
    for rule in rules() {
        let name = rule.name();
        let rule_rows: Vec<&Row> = rows.iter().filter(|r| r.rule_name == name).collect();

        // Count occurrences of each class
        let mut counts: std::collections::HashMap<&'static str, usize> =
            std::collections::HashMap::new();
        for r in &rule_rows {
            *counts.entry(class_label(r.attractor_class)).or_insert(0) += 1;
        }
        let most_common = counts
            .into_iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, _)| k)
            .unwrap_or("—");

        println!("        {:<34}  → {}", name, most_common);
    }

    // 4. Total halt rate
    let halted_count = rows.iter().filter(|r| r.halted_early).count();
    let total = rows.len();
    println!(
        "\n  ⚠️   Overall early-halt rate: {}/{} ({:.0}%)",
        halted_count,
        total,
        100.0 * halted_count as f64 / total as f64
    );

    println!();
    println!("═══════════════════════════════════════════════════════════════");
}
