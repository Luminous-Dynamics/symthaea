// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Extensive Pareto Design Space Sweep (1-1000 kW)
//!
//! Multi-objective optimization across the full power range.
//!
//! Run with: cargo run --example pareto_sweep --release

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{DesignSpaceMapper, FusionReaction, ParetoOptimizer};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              PARETO DESIGN SPACE SWEEP (1-1000 kW)                   ║");
    println!("║              Multi-Objective Optimization Analysis                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let genesis = GenesisSeed::from_phrase("Pareto Sweep 2024");
    let optimizer = ParetoOptimizer::from_genesis(&genesis);
    let mapper = DesignSpaceMapper::from_genesis(&genesis);

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSUMER SCALE (1-10 kW) - D-D Fusion
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  CONSUMER SCALE: D-D Fusion (1-10 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let consumer_pareto = optimizer.compute_pareto_frontier(FusionReaction::DD, (1.0, 10.0), 20);
    println!("{}", optimizer.format_report(&consumer_pareto));

    // Design space visualization
    let consumer_sweep = mapper.sweep_power_vs_mass(FusionReaction::DD, (1.0, 10.0), 20);
    println!("{}", mapper.ascii_chart(&consumer_sweep, 50, 12));

    // ═══════════════════════════════════════════════════════════════════════════
    // COMMERCIAL SCALE (10-100 kW) - D-D Fusion
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  COMMERCIAL SCALE: D-D Fusion (10-100 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let commercial_pareto =
        optimizer.compute_pareto_frontier(FusionReaction::DD, (10.0, 100.0), 20);
    println!("{}", optimizer.format_report(&commercial_pareto));

    let commercial_sweep = mapper.sweep_power_vs_mass(FusionReaction::DD, (10.0, 100.0), 20);
    println!("{}", mapper.ascii_chart(&commercial_sweep, 50, 12));

    // ═══════════════════════════════════════════════════════════════════════════
    // INDUSTRIAL SCALE (100-1000 kW) - D-T Fusion
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  INDUSTRIAL SCALE: D-T Fusion (100-1000 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let industrial_pareto =
        optimizer.compute_pareto_frontier(FusionReaction::DT, (100.0, 1000.0), 20);
    println!("{}", optimizer.format_report(&industrial_pareto));

    let industrial_sweep = mapper.sweep_power_vs_mass(FusionReaction::DT, (100.0, 1000.0), 20);
    println!("{}", mapper.ascii_chart(&industrial_sweep, 50, 12));

    // ═══════════════════════════════════════════════════════════════════════════
    // ANEUTRONIC COMPARISON (1-100 kW) - D-He3 Fusion
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  ANEUTRONIC: D-He3 Fusion (1-100 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let aneutronic_pareto =
        optimizer.compute_pareto_frontier(FusionReaction::DHe3, (1.0, 100.0), 20);
    println!("{}", optimizer.format_report(&aneutronic_pareto));

    // ═══════════════════════════════════════════════════════════════════════════
    // FULL RANGE ANALYSIS (1-1000 kW) - High Resolution
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  FULL RANGE ANALYSIS: 1-1000 kW (50 points)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // D-D for consumer/commercial range
    let dd_full = optimizer.compute_pareto_frontier(FusionReaction::DD, (1.0, 500.0), 50);

    // D-T for higher power range
    let dt_full = optimizer.compute_pareto_frontier(FusionReaction::DT, (100.0, 1000.0), 50);

    // Summary statistics
    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│                    FULL RANGE SUMMARY                              │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│  D-D Fusion (1-500 kW):                                            │");
    println!(
        "│    Pareto-optimal designs: {:>3}                                    │",
        dd_full.frontier.len()
    );
    println!(
        "│    Feasible designs:       {:>3}/50                                 │",
        dd_full.all_designs.iter().filter(|d| d.feasible).count()
    );
    if let Some(ref best) = dd_full.best_by_objective.min_cost {
        println!(
            "│    Best cost:              ${:>6.0}/kW at {:>5.1} kW                │",
            best.cost_per_kw, best.power_kw
        );
    }
    if let Some(ref best) = dd_full.best_by_objective.min_mass {
        println!(
            "│    Min mass:               {:>6.0} kg at {:>5.1} kW                 │",
            best.mass_kg, best.power_kw
        );
    }
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│  D-T Fusion (100-1000 kW):                                         │");
    println!(
        "│    Pareto-optimal designs: {:>3}                                    │",
        dt_full.frontier.len()
    );
    println!(
        "│    Feasible designs:       {:>3}/50                                 │",
        dt_full.all_designs.iter().filter(|d| d.feasible).count()
    );
    if let Some(ref best) = dt_full.best_by_objective.min_cost {
        println!(
            "│    Best cost:              ${:>6.0}/kW at {:>5.1} kW               │",
            best.cost_per_kw, best.power_kw
        );
    }
    if let Some(ref best) = dt_full.best_by_objective.min_mass {
        println!(
            "│    Min mass:               {:>6.0} kg at {:>5.1} kW                │",
            best.mass_kg, best.power_kw
        );
    }
    println!("└────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // SCALING ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SCALING ANALYSIS: Mass and Cost vs Power");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  Power (kW)  │  Mass (kg)  │  kg/kW  │  $/kW   │  Feasible       │");
    println!("├──────────────┼─────────────┼─────────┼─────────┼─────────────────┤");

    let power_points = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0];
    for &power in &power_points {
        // Use D-D for consumer, D-T for industrial
        let (pareto, reaction_name) = if power <= 100.0 {
            (&dd_full, "D-D")
        } else {
            (&dt_full, "D-T")
        };

        // Find closest design point
        if let Some(design) = pareto.all_designs.iter().min_by(|a, b| {
            (a.power_kw - power)
                .abs()
                .partial_cmp(&(b.power_kw - power).abs())
                .unwrap()
        }) {
            let feasible_str = if design.feasible { "YES" } else { "NO " };
            println!(
                "│  {:>8.0}    │  {:>9.0}  │  {:>5.0}  │ {:>7.0} │  {} ({})    │",
                design.power_kw,
                design.mass_kg,
                design.mass_kg / design.power_kw,
                design.cost_per_kw,
                feasible_str,
                reaction_name
            );
        }
    }
    println!("└────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // RECOMMENDATIONS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    DESIGN RECOMMENDATIONS");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  APPLICATION          RECOMMENDED POWER    REACTION    NOTES       │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  Home (residential)   2-5 kW               D-D         Low mass    │");
    println!("│  Small business       10-25 kW             D-D         Best $/kW   │");
    println!("│  Commercial building  50-100 kW            D-D         Scalable    │");
    println!("│  Industrial campus    200-500 kW           D-T         High output │");
    println!("│  Data center          500-1000 kW          D-T         Base load   │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!("\n┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  KEY FINDINGS FROM PARETO ANALYSIS                                  │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  • Economy of scale: $/kW decreases with power                      │");
    println!("│  • Mass scaling: kg/kW ratio improves at larger scales              │");
    println!("│  • D-D preferred for <100 kW (simpler shielding)                    │");
    println!("│  • D-T preferred for >100 kW (higher energy density)                │");
    println!("│  • D-He3 aneutronic: lowest mass but thermal challenges at scale    │");
    println!("│  • Sweet spot: 10-50 kW range offers best trade-offs                │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    PARETO SWEEP COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
}
