//! Multi-Scenario EFE Evaluation: Test 6 geometry variants.
//!
//! Evaluates EFE-based decision making across different threat geometries:
//! Default, CloseBeam, FarBeam, ReversedGeometry, NoHuman, LowDanger.
//!
//! No MuJoCo required — purely analytical.
//!
//! Usage: `cargo run -p symthaea-flight --example multi_scenario --release`

use symthaea_flight::benchmarks::{evaluate_scenario_variants, ScenarioVariant};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Multi-Scenario EFE Evaluation: 6 Geometry Variants");
    println!("═══════════════════════════════════════════════════════════════\n");

    let results = evaluate_scenario_variants();

    println!(
        "{:<20} {:>14} {:>14} {:>10} {:>10} {:>7}",
        "Variant", "EFE(mission)", "EFE(intercept)", "Decision", "Expected", "Match?"
    );
    println!("{}", "─".repeat(79));

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
        println!(
            "{:<20} {:>14.4} {:>14.4} {:>10} {:>10} {:>7}",
            r.variant_name, r.efe_mission, r.efe_intercept, decision, r.expected_decision, match_str
        );
    }

    println!();
    if all_match {
        println!("All {} scenarios match expected decisions.", results.len());
    } else {
        println!("WARNING: Some scenarios did not match expected decisions!");
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
    let mut csv =
        String::from("variant,efe_mission,efe_intercept,chose_override,expected,matches\n");
    for r in &results {
        csv.push_str(&format!(
            "{},{},{},{},{},{}\n",
            r.variant_name,
            r.efe_mission,
            r.efe_intercept,
            r.chose_override as u8,
            r.expected_decision,
            r.matches_expected as u8
        ));
    }
    std::fs::write(csv_path, &csv).expect("Failed to write CSV");
    println!("\nResults written to {csv_path}");
}
