// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the 150-year civilization simulation and print the full report.

use mycelix_multiworld_sim::{MultiWorldSimulator, config::SimulationConfig};

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);

    eprintln!("=== Running 150-year civilization simulation (seed: {}) ===", seed);
    eprintln!("1,800 ticks (1 tick = 1 month)...\n");

    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    // Print the built-in summary
    println!("{}", report.summary());

    // Print epoch snapshots
    println!("\n=== EPOCH SNAPSHOTS (every 5 years) ===");
    for snap in &report.epoch_snapshots {
        println!(
            "  Yr {:>5.1} | Pop {:>5} (off {:>4}) | Phi {:.3} | Love {:.3} | CVS {:.3} | Load {:.3} | Eng {:.3} | Gen {:.3} | Tech {:.2}",
            snap.tick as f64 / 12.0, snap.total_population, snap.off_earth_population,
            snap.mean_phi, snap.mean_love_coherence,
            snap.civilization_viability_score,
            snap.mean_allostatic_load, snap.mean_engagement,
            snap.genetic_diversity, snap.mean_tech_level,
        );
    }

    // Print checkpoint results
    println!("\n=== CHECKPOINT RESULTS ===");
    for (tick, desc, passed, failures) in &report.checkpoint_results {
        let status = if *passed { "PASS" } else { "FAIL" };
        println!("  [{status}] Tick {tick:>5}: {desc}");
        for f in failures {
            println!("           -> {f}");
        }
    }

    // Print key milestones
    println!("\n=== KEY MILESTONES ===");
    if let Some(t) = report.first_birth_tick {
        println!("  First birth:             tick {} (year {:.1})", t, t as f64 / 12.0);
    }
    if let Some(t) = report.first_constitution_tick {
        println!("  Constitution ratified:   tick {} (year {:.1})", t, t as f64 / 12.0);
    }
    if let Some(t) = report.first_trade_tick {
        println!("  First inter-world trade: tick {} (year {:.1})", t, t as f64 / 12.0);
    }

    // Final verdict
    println!("\n{}", "=".repeat(60));
    if report.survived {
        println!("  CIVILIZATION SURVIVED 150 YEARS (CVS: {:.4})", report.final_cvs);
    } else {
        println!("  CIVILIZATION COLLAPSED (CVS: {:.4})", report.final_cvs);
    }
    println!("{}", "=".repeat(60));

    // Export epoch data as CSV for plotting
    println!("\n=== CSV DATA (copy to file for visualization) ===");
    println!("tick,year,population,off_earth,worlds,phi,love_coherence,cvs,tech_level,genetic_diversity,self_sufficiency,allostatic_load,engagement");
    for snap in &report.epoch_snapshots {
        println!(
            "{},{:.1},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            snap.tick, snap.tick as f64 / 12.0,
            snap.total_population, snap.off_earth_population, snap.world_count,
            snap.mean_phi, snap.mean_love_coherence,
            snap.civilization_viability_score, snap.mean_tech_level,
            snap.genetic_diversity, snap.mean_self_sufficiency,
            snap.mean_allostatic_load, snap.mean_engagement,
        );
    }
}
