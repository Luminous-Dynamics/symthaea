// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interstellar simulation: 1000-year civilization + generation ship to Proxima.
//!
//! Run: cargo run --release --bin interstellar_sim

use mycelix_multiworld_sim::generation_ship::*;
use mycelix_multiworld_sim::stochastic::StochasticEngine;

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);

    println!("=== INTERSTELLAR SIMULATION (seed: {}) ===\n", seed);

    // Create generation ship: 500 colonists to Proxima at 0.05c
    let mut ship = GenerationShip::new(
        99, // world_id
        InterstellarDestination::ProximaCentauri,
        0.05,
        6000, // launch tick (year 500)
        500,
    );

    let mut rng = StochasticEngine::new(seed);
    let mut pop = 500_usize;
    let mut total_disasters = 0;
    let mut year_10_report = false;
    let mut year_25_report = false;
    let mut year_50_report = false;
    let mut year_75_report = false;

    println!(
        "Ship: {} → {} ({:.2} ly at {:.1}%c)",
        "Sol",
        ship.destination.name(),
        ship.distance_ly,
        ship.cruise_velocity_c * 100.0
    );
    println!(
        "Transit: {:.1} years ({} ticks)",
        ship.transit_ticks as f64 / 12.0,
        ship.transit_ticks
    );
    println!("Passengers: {}\n", pop);

    // Run transit
    loop {
        let disasters = ship.tick(&mut rng);

        // Apply disaster effects to population
        for d in &disasters {
            match d {
                InterstellarDisaster::CosmicRayBurst { severity } => {
                    let deaths = (pop as f64 * severity * 0.01) as usize;
                    pop = pop.saturating_sub(deaths);
                }
                InterstellarDisaster::MicrometeoiteImpact { hull_damage } => {
                    if *hull_damage > 0.005 {
                        let deaths = (pop as f64 * hull_damage * 0.5) as usize;
                        pop = pop.saturating_sub(deaths);
                    }
                }
                InterstellarDisaster::KnowledgeAttrition { .. } => {
                    // Knowledge loss doesn't kill directly but weakens resilience
                }
                InterstellarDisaster::SocialFracture { .. } => {
                    // Social fracture doesn't kill directly
                }
                InterstellarDisaster::NavigationDrift { .. } => {
                    // Fuel cost, no direct deaths
                }
            }
            total_disasters += 1;
        }

        // Simple population dynamics: births vs deaths
        if pop > 10 {
            let birth_rate = 0.002; // ~2.4% annual
            let death_rate = 0.0015; // ~1.8% annual (higher cosmic ray mortality)
            let stress_modifier = if ship.hull_integrity < 0.8 { 1.5 } else { 1.0 };
            let births = (pop as f64 * birth_rate) as usize;
            let deaths = (pop as f64 * death_rate * stress_modifier) as usize;
            pop = pop + births - deaths.min(pop);
        }

        // Milestone reports
        let years = ship.elapsed_ticks as f64 / 12.0;
        if years >= 10.0 && !year_10_report {
            println!(
                "  Year {:.0}: Pop {} | Hull {:.1}% | Fuel {:.1}% | Phase {:?} | Disasters {}",
                years,
                pop,
                ship.hull_integrity * 100.0,
                ship.fuel_fraction * 100.0,
                ship.phase,
                total_disasters
            );
            year_10_report = true;
        }
        if years >= 25.0 && !year_25_report {
            println!(
                "  Year {:.0}: Pop {} | Hull {:.1}% | Fuel {:.1}% | Phase {:?} | Disasters {}",
                years,
                pop,
                ship.hull_integrity * 100.0,
                ship.fuel_fraction * 100.0,
                ship.phase,
                total_disasters
            );
            year_25_report = true;
        }
        if years >= 50.0 && !year_50_report {
            println!("  Year {:.0}: Pop {} | Hull {:.1}% | Comms RT {:.1}yr | Speciated: {} | Disasters {}",
                years, pop, ship.hull_integrity * 100.0, ship.comms_roundtrip_years(), ship.culturally_speciated, total_disasters);
            year_50_report = true;
        }
        if years >= 75.0 && !year_75_report {
            println!("  Year {:.0}: Pop {} | Hull {:.1}% | Comms RT {:.1}yr | Henrich risk: {} | Disasters {}",
                years, pop, ship.hull_integrity * 100.0, ship.comms_roundtrip_years(), ship.henrich_risk(pop), total_disasters);
            year_75_report = true;
        }

        if ship.phase == ShipPhase::Arrived {
            break;
        }

        // Extinction check
        if pop == 0 {
            println!("\n  ██ EXTINCTION at year {:.1} — all crew perished", years);
            break;
        }
        if ship.hull_integrity <= 0.0 {
            println!(
                "\n  ██ HULL FAILURE at year {:.1} — catastrophic breach",
                years
            );
            pop = 0;
            break;
        }
    }

    println!("\n{}", ship.status_report(pop));

    if ship.phase == ShipPhase::Arrived && pop > 0 {
        println!(
            "\n✓ ARRIVAL at {} with {} colonists",
            ship.destination.name(),
            pop
        );
        println!(
            "  Population change: {} → {} ({:+.1}%)",
            ship.launch_population,
            pop,
            (pop as f64 / ship.launch_population as f64 - 1.0) * 100.0
        );
        if ship.below_minimum_viable(pop) {
            println!("  ⚠ BELOW MINIMUM VIABLE POPULATION ({} < {})", pop, 98);
        }
        if ship.henrich_risk(pop) {
            println!("  ⚠ HENRICH RISK: population too small for cultural transmission");
        }
        if ship.culturally_speciated {
            println!("  ◆ Cultural speciation occurred — the colony is a new human culture");
        }
    } else if pop == 0 {
        println!("\n✗ MISSION FAILURE — crew did not survive transit");
    }
}
