// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run a civilization simulation. Usage: run_simulation [DURATION_YEARS] [SEED]

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let duration = args
        .get(1)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(150);
    let seed = args
        .get(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);

    let mut config = match duration {
        d if d >= 1000 => SimulationConfig::default_1000_year(),
        d if d >= 777 => SimulationConfig::default_777_year(),
        d if d >= 300 => SimulationConfig::default_300_year(),
        _ => SimulationConfig::default_150_year(),
    };
    config.seed = seed;

    eprintln!(
        "=== Running {}-year civilization simulation (seed: {}) ===",
        duration, seed
    );
    eprintln!(
        "{} ticks (1 tick = 1 month)...
",
        config.total_ticks
    );

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    if report.duration_years >= 300.0 {
        println!("{}", report.summary_extended());
    } else {
        println!("{}", report.summary());
    }

    println!(
        "
=== EPOCH SNAPSHOTS ==="
    );
    for snap in &report.epoch_snapshots {
        println!(
            "  Yr {:>5.1} | Pop {:>6} | Phi {:.3} | CVS {:.3} | Fac {} | Trauma {:.3}",
            snap.tick as f64 / 12.0,
            snap.total_population,
            snap.mean_phi,
            snap.civilization_viability_score,
            snap.faction_count,
            snap.trauma_level
        );
        // Per-world breakdown (show only at 50-year intervals to avoid flooding)
        if snap.tick % 600 == 0 && !snap.worlds.is_empty() {
            for ws in &snap.worlds {
                println!("    {:>12} ({:>6}) | Pop {:>5} | Phi {:.3} | Load {:.3} | Conatus {:+.3} | Moral {:+.3} | SS {:.2}",
                    ws.name, ws.location, ws.population, ws.phi,
                    ws.allostatic_load, ws.net_conatus, ws.moral_balance,
                    ws.self_sufficiency);
            }
        }
    }

    // Print narrative chronicle (connected prose, not just events)
    if !sim.narrative_engine.events.is_empty() {
        println!("\n{}", sim.narrative_engine.format_chronicle());
    }

    let years = report.total_ticks / 12;
    println!(
        "
{}",
        "=".repeat(60)
    );
    if report.survived {
        println!(
            "  CIVILIZATION SURVIVED {} YEARS (CVS: {:.4})",
            years, report.final_cvs
        );
    } else {
        println!("  CIVILIZATION COLLAPSED (CVS: {:.4})", report.final_cvs);
    }
    println!("{}", "=".repeat(60));
    println!("  Tech milestones: {}", report.tech_milestones_achieved);
    println!("  Narrative events: {}", sim.narrative_engine.events.len());
    println!("{}", "=".repeat(60));
}
