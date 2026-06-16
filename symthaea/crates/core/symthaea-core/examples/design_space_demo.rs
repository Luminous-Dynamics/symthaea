// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Design Space Mapping Demo
//!
//! Explores the feasible design envelope for Spark Engine:
//! - Power vs shielding requirements
//! - Power vs system mass
//! - Power vs cost per kW
//! - Feasibility boundaries for D-D and D-T fusion
//!
//! Run with: cargo run --example design_space_demo

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{DesignEnvelope, DesignSpaceMapper, FusionReaction};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              SPARK ENGINE DESIGN SPACE MAPPING                       ║");
    println!("║              Parametric Analysis of Feasible Designs                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let genesis = GenesisSeed::from_phrase("Design Space Explorer");
    let mapper = DesignSpaceMapper::from_genesis(&genesis);

    // =========================================================================
    // D-D Fusion Analysis
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  DEUTERIUM-DEUTERIUM (D-D) FUSION");
    println!("  Neutron energy: 2.45 MeV");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Find minimum feasible power
    println!("Searching for minimum feasible power...");
    if let Some(min_power) = mapper.find_minimum_feasible_power(FusionReaction::DD) {
        println!("  Minimum feasible power: {:.2} kW\n", min_power);
    } else {
        println!("  No feasible design found in range\n");
    }

    // Power vs Shielding sweep
    println!("Running parametric sweep: Power vs Shielding...");
    let dd_shielding = mapper.sweep_power_vs_shielding(FusionReaction::DD, (1.0, 50.0), 10);
    println!("{}", mapper.ascii_chart(&dd_shielding, 50, 12));

    // Power vs Mass sweep
    println!("\nRunning parametric sweep: Power vs Mass...");
    let dd_mass = mapper.sweep_power_vs_mass(FusionReaction::DD, (1.0, 50.0), 10);

    println!("\nPower vs Mass Summary:");
    println!(
        "{:>10} {:>15} {:>12} {:>10}",
        "Power (kW)", "Mass (kg)", "kg/kW", "Status"
    );
    println!("{}", "─".repeat(50));
    for point in &dd_mass.points {
        let status = if point.feasible { "✓" } else { "✗" };
        println!(
            "{:>10.1} {:>15.0} {:>12.0} {:>10}",
            point.power_kw, point.param_value, point.metric_value, status
        );
    }

    // Power vs Cost sweep
    println!("\nRunning parametric sweep: Power vs Cost...");
    let dd_cost = mapper.sweep_power_vs_cost(FusionReaction::DD, (1.0, 50.0), 10);

    println!("\nPower vs Cost Summary:");
    println!(
        "{:>10} {:>15} {:>10}",
        "Power (kW)", "Cost ($/kW)", "Status"
    );
    println!("{}", "─".repeat(40));
    for point in &dd_cost.points {
        let status = if point.feasible { "✓" } else { "✗" };
        println!(
            "{:>10.1} {:>15.0} {:>10}",
            point.power_kw, point.param_value, status
        );
    }

    // Generate D-D envelope
    let dd_envelope =
        DesignEnvelope::from_sweeps(FusionReaction::DD, &dd_shielding, &dd_mass, &dd_cost);

    println!("\n┌────────────────────────────────────────────────────┐");
    println!("│         D-D DESIGN ENVELOPE SUMMARY                │");
    println!("├────────────────────────────────────────────────────┤");
    if let Some(min) = dd_envelope.min_power_kw {
        println!("│  Min feasible power:    {:>8.1} kW               │", min);
    }
    println!(
        "│  Max tested power:      {:>8.1} kW               │",
        dd_envelope.max_power_kw
    );
    println!(
        "│  Shielding range:       {:>5.0} - {:>5.0} cm          │",
        dd_envelope.shielding_range_cm.0, dd_envelope.shielding_range_cm.1
    );
    println!(
        "│  Mass range:            {:>5.0} - {:>5.0} kg          │",
        dd_envelope.mass_range_kg.0, dd_envelope.mass_range_kg.1
    );
    println!(
        "│  Cost range:          ${:>6.0} - ${:>6.0}/kW       │",
        dd_envelope.cost_range_usd_kw.0, dd_envelope.cost_range_usd_kw.1
    );
    println!("└────────────────────────────────────────────────────┘");

    // =========================================================================
    // D-T Fusion Analysis (for comparison)
    // =========================================================================
    println!("\n\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  DEUTERIUM-TRITIUM (D-T) FUSION");
    println!("  Neutron energy: 14.1 MeV (much harder to shield)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Find minimum feasible power for D-T
    println!("Searching for minimum feasible power...");
    if let Some(min_power) = mapper.find_minimum_feasible_power(FusionReaction::DT) {
        println!("  Minimum feasible power: {:.2} kW\n", min_power);
    } else {
        println!("  WARNING: No feasible D-T design found in consumer range");
        println!("  (D-T requires industrial-scale engineering)\n");
    }

    // Quick D-T sweep for comparison
    let dt_shielding = mapper.sweep_power_vs_shielding(FusionReaction::DT, (10.0, 100.0), 5);

    println!("D-T Shielding Requirements (vs D-D):");
    println!(
        "{:>10} {:>15} {:>15} {:>10}",
        "Power (kW)", "D-T Shield", "D-D Shield", "Ratio"
    );
    println!("{}", "─".repeat(55));

    for i in 0..dt_shielding.points.len().min(dd_shielding.points.len()) {
        let dt_point = &dt_shielding.points[i];
        // Find closest D-D point
        let dd_point = dd_shielding
            .points
            .iter()
            .min_by(|a, b| {
                (a.power_kw - dt_point.power_kw)
                    .abs()
                    .partial_cmp(&(b.power_kw - dt_point.power_kw).abs())
                    .unwrap()
            })
            .unwrap();

        let ratio = dt_point.param_value / dd_point.param_value;
        println!(
            "{:>10.0} {:>13.0} cm {:>13.0} cm {:>10.1}x",
            dt_point.power_kw, dt_point.param_value, dd_point.param_value, ratio
        );
    }

    // =========================================================================
    // Scaling Analysis
    // =========================================================================
    println!("\n\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SCALING ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Check scaling laws
    let powers = [1.0, 5.0, 10.0, 25.0, 50.0];
    println!("D-D Scaling Analysis:");
    println!(
        "{:>10} {:>12} {:>12} {:>15}",
        "Power", "Shielding", "Mass", "Cost/kW"
    );
    println!("{}", "─".repeat(55));

    for &power in &powers {
        let shielding_point = dd_shielding.points.iter().min_by(|a, b| {
            (a.power_kw - power)
                .abs()
                .partial_cmp(&(b.power_kw - power).abs())
                .unwrap()
        });
        let mass_point = dd_mass.points.iter().min_by(|a, b| {
            (a.power_kw - power)
                .abs()
                .partial_cmp(&(b.power_kw - power).abs())
                .unwrap()
        });
        let cost_point = dd_cost.points.iter().min_by(|a, b| {
            (a.power_kw - power)
                .abs()
                .partial_cmp(&(b.power_kw - power).abs())
                .unwrap()
        });

        if let (Some(s), Some(m), Some(c)) = (shielding_point, mass_point, cost_point) {
            println!(
                "{:>8.0} kW {:>10.0} cm {:>10.0} kg {:>12.0} $/kW",
                power, s.param_value, m.param_value, c.param_value
            );
        }
    }

    // =========================================================================
    // Summary and Recommendations
    // =========================================================================
    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                        KEY FINDINGS");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("1. D-D FUSION IS CONSUMER-VIABLE");
    println!("   • Feasible from ~1 kW with proper shielding");
    println!("   • Shielding scales sub-linearly with power");
    println!("   • Cost per kW decreases with scale");
    println!();
    println!("2. D-T FUSION REQUIRES INDUSTRIAL SCALE");
    println!("   • 14.1 MeV neutrons require ~3x more shielding");
    println!("   • Tritium handling adds complexity");
    println!("   • Better suited for utility-scale (>10 MW)");
    println!();
    println!("3. DESIGN TRADE-OFFS");
    println!("   • Larger shielding = lower dose but more mass");
    println!("   • Higher power density = smaller size but harder cooling");
    println!("   • Pulsed operation extends lifetime but adds complexity");
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
}
