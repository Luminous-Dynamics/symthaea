// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integrated interstellar simulation: 1000-year civilization + generation ship.
//!
//! Runs the full civilization simulator for 1000 years with all disaster types
//! and resilience mechanisms, PLUS launches a generation ship at year 500.
//! The ship transits to Proxima Centauri while the solar system civilization
//! continues developing.
//!
//! Run: cargo run --release --bin interstellar_integrated

use mycelix_multiworld_sim::config::SimulationConfig;
use mycelix_multiworld_sim::generation_ship::ShipPhase;
use mycelix_multiworld_sim::MultiWorldSimulator;

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);

    println!(
        "=== INTEGRATED INTERSTELLAR SIMULATION (seed: {}) ===",
        seed
    );
    println!("1000-year civilization + generation ship to Proxima Centauri\n");

    // Use the standard 1000-year config
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = 12_000; // 1000 years
    config.seed = seed;

    let mut sim = MultiWorldSimulator::new(config);

    // Enable generation ship: launch at tick 6000 (year 500)
    sim.generation_ship_launch_tick = 6000;

    let report = sim.run();

    // Print civilization results
    println!("=== CIVILIZATION RESULTS ===");
    println!(
        "Duration: {:.0} years ({} ticks)",
        report.total_ticks as f64 / 12.0,
        report.total_ticks
    );
    println!(
        "RESULT: {} (CVS: {:.3})",
        if report.survived {
            "SURVIVED"
        } else {
            "COLLAPSED"
        },
        report.final_cvs
    );
    println!("Population: {}", report.final_population);
    println!("Breakthroughs: {}", report.breakthroughs);
    println!();

    // Print generation ship results
    if let Some(ref ship) = sim.generation_ship {
        println!("=== GENERATION SHIP RESULTS ===");
        println!(
            "Destination: {} ({:.2} ly)",
            ship.destination.name(),
            ship.distance_ly
        );
        println!("Phase: {:?}", ship.phase);
        println!("Transit progress: {:.1}%", ship.transit_progress() * 100.0);
        println!("Hull integrity: {:.1}%", ship.hull_integrity * 100.0);
        println!("Fuel remaining: {:.1}%", ship.fuel_fraction * 100.0);
        println!("Cosmic ray dose: {:.3} Sv", ship.cosmic_ray_dose);
        println!(
            "Comms delay: {:.1} years (round trip)",
            ship.comms_roundtrip_years()
        );
        println!(
            "Cultural speciation: {}",
            if ship.culturally_speciated {
                "YES"
            } else {
                "No"
            }
        );
        println!("Interstellar disasters: {}", ship.disasters_survived.len());

        if ship.phase == ShipPhase::Arrived {
            println!("\n★ GENERATION SHIP ARRIVED at {}", ship.destination.name());
            println!("  Launched at year 500 with 500 colonists");
            println!("  Transit: {:.1} years", ship.transit_ticks as f64 / 12.0);
            println!("  Humanity is now an interstellar civilization.");
        } else {
            let years_remaining = ship.years_remaining();
            let arrival_year = 500.0 + (ship.elapsed_ticks as f64 / 12.0) + years_remaining;
            println!(
                "\nShip in transit — arrives at year {:.0} ({:.1} years remaining)",
                arrival_year, years_remaining
            );
        }
    } else {
        println!(
            "No generation ship was launched (launch tick {} not reached?)",
            sim.generation_ship_launch_tick
        );
    }
}
