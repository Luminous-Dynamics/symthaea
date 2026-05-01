// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Scenario EFE Evaluation: Test 7 geometry variants.
//!
//! Evaluates EFE-based decision making across different threat geometries:
//! Default, CloseBeam, FarBeam, ReversedGeometry, NoHuman, LowDanger, StaticThreat.
//!
//! No MuJoCo required — purely analytical.
//!
//! Usage: `cargo run -p symthaea-multirotor --example multi_scenario --release`

use symthaea_multirotor::benchmarks::{evaluate_scenario_variants, ScenarioVariant};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Multi-Scenario EFE Evaluation: 7 Geometry Variants");
    println!("═══════════════════════════════════════════════════════════════\n");

    let results = evaluate_scenario_variants();

    println!(
        "{:<20} {:>14} {:>14} {:>10} {:>10} {:>7} {:>8}",
        "Variant", "EFE(mission)", "EFE(intercept)", "Decision", "Expected", "Match?", "RV frac"
    );
    println!("{}", "─".repeat(87));

    let mut all_match = true;
    for r in &results {
        let decision = if r.chose_override {
            "INTERCEPT"
        } else {
            "MISSION"
        };
        let match_str = if r.matches_expected { "YES" } else { "NO" };
        if !r.matches_expected {
            all_match = false;
        }
        let rv_str = match r.best_rendezvous_frac {
            Some(f) => format!("{:.0}%", f * 100.0),
            None => "---".to_string(),
        };
        println!(
            "{:<20} {:>14.4} {:>14.4} {:>10} {:>10} {:>7} {:>8}",
            r.variant_name,
            r.efe_mission,
            r.efe_intercept,
            decision,
            r.expected_decision,
            match_str,
            rv_str
        );
    }

    println!();
    if all_match {
        println!("All {} scenarios match expected decisions.", results.len());
    } else {
        println!("WARNING: Some scenarios did not match expected decisions!");
    }

    // Rendezvous analysis
    println!("\nRendezvous Time Analysis:");
    for r in &results {
        if let Some(frac) = r.best_rendezvous_frac {
            println!(
                "  {:<20} Best intercept at {:.0}% of impact time",
                r.variant_name,
                frac * 100.0
            );
        }
    }

    // Print scenario descriptions
    println!("\nScenario Descriptions:");
    for variant in ScenarioVariant::all() {
        let (_, _, env) = variant.build();
        println!(
            "  {:<20} danger={:.2}, threat={:?}, entity={:?}",
            variant.name(),
            env.human_danger,
            env.threat_pos,
            env.entity_pos
        );
    }

    // Write CSV
    let csv_path = "multi_scenario.csv";
    let mut csv = String::from(
        "variant,efe_mission,efe_intercept,chose_override,expected,matches,best_rv_frac\n",
    );
    for r in &results {
        let rv = r
            .best_rendezvous_frac
            .map_or("".to_string(), |f| format!("{:.2}", f));
        csv.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            r.variant_name,
            r.efe_mission,
            r.efe_intercept,
            r.chose_override as u8,
            r.expected_decision,
            r.matches_expected as u8,
            rv
        ));
    }
    std::fs::write(csv_path, &csv).expect("Failed to write CSV");
    println!("\nResults written to {csv_path}");
}
